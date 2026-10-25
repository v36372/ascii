// SPDX-License-Identifier: Apache-2.0
//! Native terminal backend. Only one live session may own the process terminal.
//!
//! The panic hook is deliberately opt-in; libraries must not replace their host's
//! hook. Normal errors and unwinding are handled by the session's RAII guard.

use crate::graphics::{Color, Pixel, Rect, Screen};
use crossterm::{
    cursor::{Hide, Show},
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    style::ResetColor,
    terminal::{
        self, Clear, ClearType, DisableLineWrap, EnableLineWrap, EnterAlternateScreen,
        LeaveAlternateScreen,
    },
};
use std::{
    io::{self, IsTerminal, Write},
    sync::{
        atomic::{AtomicBool, AtomicU8, Ordering},
        Once,
    },
    time::Duration,
};
use unicode_width::UnicodeWidthChar;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    Up,
    Down,
    Left,
    Right,
    Char(char),
    Ctrl(char),
    Alt(char),
    Enter,
    Escape,
    Space,
    Tab,
    Backspace,
    Delete,
    Insert,
    Home,
    End,
    PageUp,
    PageDown,
    /// Function key in `1..=12`.
    F(u8),
}

impl Key {
    pub fn printable(self) -> Option<char> {
        match self {
            Self::Char(ch) | Self::Ctrl(ch) | Self::Alt(ch) => Some(ch),
            Self::Space => Some(' '),
            _ => None,
        }
    }

    pub fn label(self) -> String {
        match self {
            Self::Char(ch) => ch.to_string(),
            Self::Ctrl(ch) => format!("Ctrl+{ch}"),
            Self::Alt(ch) => format!("Alt+{ch}"),
            Self::F(n) => format!("F{n}"),
            Self::Up => "Up".into(),
            Self::Down => "Down".into(),
            Self::Left => "Left".into(),
            Self::Right => "Right".into(),
            Self::Enter => "Enter".into(),
            Self::Escape => "Escape".into(),
            Self::Space => "Space".into(),
            Self::Tab => "Tab".into(),
            Self::Backspace => "Backspace".into(),
            Self::Delete => "Delete".into(),
            Self::Insert => "Insert".into(),
            Self::Home => "Home".into(),
            Self::End => "End".into(),
            Self::PageUp => "PageUp".into(),
            Self::PageDown => "PageDown".into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputEvent {
    Key(Key),
    Resize(u16, u16),
    Quit,
}

// Bound *raw* events, not just recognized keys: mouse/release floods must not
// starve the game loop either.
const MAX_INPUT_EVENTS: usize = 64;
static CLAIMED: AtomicBool = AtomicBool::new(false);
static SESSION: SessionState = SessionState::new();
static PANIC_HOOK: Once = Once::new();

pub struct Terminal {
    owns_session: bool,
    renderer: Renderer,
}

impl Terminal {
    pub fn new() -> io::Result<Self> {
        validate_tty(io::stdin().is_terminal(), io::stdout().is_terminal())?;
        CLAIMED
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .map_err(|_| io::Error::new(io::ErrorKind::AlreadyExists, "terminal already owned"))?;
        // Construct the guard before doing anything that can change the terminal.
        let terminal = Self {
            owns_session: true,
            renderer: Renderer::default(),
        };
        SESSION.initialize(&mut NativeControl)?;
        Ok(terminal)
    }

    pub fn size(&self) -> io::Result<(u16, u16)> {
        terminal::size()
    }

    /// Wait for the first event, then drain at most 64 immediately ready events.
    pub fn poll_events(&mut self, timeout: Duration) -> io::Result<Vec<InputEvent>> {
        self.require_active()?;
        let events = collect_events(&mut NativeInput, timeout)?;
        if events.iter().any(|e| matches!(e, InputEvent::Resize(..))) {
            self.renderer.invalidate();
        }
        Ok(events)
    }

    /// Intersect the camera with the screen, crop its bottom/right if necessary,
    /// and center the resulting viewport in a larger terminal. Borders use the
    /// terminal's default background. A viewport change always causes a redraw.
    pub fn render(&mut self, screen: &Screen, camera: Rect, force: bool) -> io::Result<()> {
        self.require_active()?;
        let size = self.size()?;
        self.renderer
            .render_to(&mut io::stdout(), screen, camera, size, force)
    }

    /// Best-effort all restoration steps, returning the first failure. Failed
    /// steps remain pending so a later call (including Drop) can retry them.
    pub fn restore(&mut self) -> io::Result<()> {
        if !self.owns_session {
            return Ok(());
        }
        self.renderer.invalidate();
        SESSION.restore(&mut NativeControl)?;
        self.owns_session = false;
        CLAIMED.store(false, Ordering::SeqCst);
        Ok(())
    }

    fn require_active(&self) -> io::Result<()> {
        if self.owns_session && SESSION.flags.load(Ordering::SeqCst) == ALL_FLAGS {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "terminal session has been restored or interrupted",
            ))
        }
    }
}

impl Drop for Terminal {
    fn drop(&mut self) {
        if self.owns_session {
            let _ = SESSION.restore(&mut NativeControl);
            // A new owner may retry any cleanup that failed during this Drop.
            CLAIMED.store(false, Ordering::SeqCst);
        }
    }
}

/// Install once in an interactive binary, before creating its Terminal. Restore
/// terminal modes before invoking the previously installed diagnostic hook.
/// Like Drop, this is best effort and cannot recover from SIGKILL/process abort.
pub fn install_panic_hook() {
    PANIC_HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if CLAIMED.load(Ordering::SeqCst) {
                let _ = SESSION.restore(&mut NativeControl);
            }
            previous(info);
        }));
    });
}

fn validate_tty(stdin: bool, stdout: bool) -> io::Result<()> {
    if stdin && stdout {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::NotConnected,
            "interactive terminal requires TTY stdin and stdout",
        ))
    }
}

const RAW: u8 = 1;
const ALTERNATE: u8 = 2;
const CURSOR: u8 = 4;
const WRAP: u8 = 8;
const COLORS: u8 = 16;
const ALL_FLAGS: u8 = RAW | ALTERNATE | CURSOR | WRAP | COLORS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Operation {
    EnableRaw,
    EnterAlternate,
    HideCursor,
    DisableWrap,
    ResetColors,
    Clear,
    EnableWrap,
    ShowCursor,
    LeaveAlternate,
    DisableRaw,
}

trait Control {
    fn run(&mut self, operation: Operation) -> io::Result<()>;
}

struct NativeControl;

impl Control for NativeControl {
    fn run(&mut self, operation: Operation) -> io::Result<()> {
        match operation {
            Operation::EnableRaw => terminal::enable_raw_mode(),
            Operation::DisableRaw => terminal::disable_raw_mode(),
            Operation::EnterAlternate => execute!(io::stdout(), EnterAlternateScreen),
            Operation::LeaveAlternate => execute!(io::stdout(), LeaveAlternateScreen),
            Operation::HideCursor => execute!(io::stdout(), Hide),
            Operation::ShowCursor => execute!(io::stdout(), Show),
            Operation::DisableWrap => execute!(io::stdout(), DisableLineWrap),
            Operation::EnableWrap => execute!(io::stdout(), EnableLineWrap),
            Operation::ResetColors => execute!(io::stdout(), ResetColor),
            Operation::Clear => execute!(io::stdout(), Clear(ClearType::All)),
        }
    }
}

// Atomic flags let the panic hook restore without acquiring an application
// mutex that could already be held by the panicking thread.
struct SessionState {
    flags: AtomicU8,
}

impl SessionState {
    const fn new() -> Self {
        Self {
            flags: AtomicU8::new(0),
        }
    }

    fn initialize(&self, control: &mut impl Control) -> io::Result<()> {
        // A previous owner's unsuccessful Drop must not leak into this session.
        self.restore(control)?;
        let result = (|| {
            for (flag, operation) in [
                (RAW, Operation::EnableRaw),
                (ALTERNATE, Operation::EnterAlternate),
                (CURSOR, Operation::HideCursor),
                (WRAP, Operation::DisableWrap),
                (COLORS, Operation::ResetColors),
            ] {
                // A failed write/flush may already have changed terminal state.
                self.flags.fetch_or(flag, Ordering::SeqCst);
                control.run(operation)?;
            }
            control.run(Operation::Clear)
        })();
        if result.is_err() {
            let _ = self.restore(control);
        }
        result
    }

    fn restore(&self, control: &mut impl Control) -> io::Result<()> {
        let mut first_error = None;
        for (flag, operation) in [
            (COLORS, Operation::ResetColors),
            (WRAP, Operation::EnableWrap),
            (CURSOR, Operation::ShowCursor),
            (ALTERNATE, Operation::LeaveAlternate),
            (RAW, Operation::DisableRaw),
        ] {
            if self.flags.load(Ordering::SeqCst) & flag != 0 {
                match control.run(operation) {
                    Ok(()) => {
                        self.flags.fetch_and(!flag, Ordering::SeqCst);
                    }
                    Err(error) => {
                        if first_error.is_none() {
                            first_error = Some(error);
                        }
                    }
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

trait InputSource {
    fn poll(&mut self, timeout: Duration) -> io::Result<bool>;
    fn read(&mut self) -> io::Result<Event>;
}

struct NativeInput;

impl InputSource for NativeInput {
    fn poll(&mut self, timeout: Duration) -> io::Result<bool> {
        event::poll(timeout)
    }

    fn read(&mut self) -> io::Result<Event> {
        event::read()
    }
}

fn collect_events(source: &mut impl InputSource, timeout: Duration) -> io::Result<Vec<InputEvent>> {
    let mut result = Vec::new();
    for index in 0..MAX_INPUT_EVENTS {
        if !source.poll(if index == 0 { timeout } else { Duration::ZERO })? {
            break;
        }
        if let Some(event) = map_event(source.read()?) {
            result.push(event);
            if event == InputEvent::Quit {
                break;
            }
        }
    }
    Ok(result)
}

fn map_event(event: Event) -> Option<InputEvent> {
    match event {
        Event::Resize(width, height) => Some(InputEvent::Resize(width, height)),
        Event::Key(key) if key.kind != KeyEventKind::Release => {
            if key.modifiers.contains(KeyModifiers::CONTROL)
                && matches!(key.code, KeyCode::Char('c' | 'C'))
            {
                return Some(InputEvent::Quit);
            }
            let key = match key.code {
                KeyCode::Up => Key::Up,
                KeyCode::Down => Key::Down,
                KeyCode::Left => Key::Left,
                KeyCode::Right => Key::Right,
                KeyCode::Char(' ') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    Key::Ctrl(' ')
                }
                KeyCode::Char(ch) if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    Key::Ctrl(ch.to_ascii_lowercase())
                }
                KeyCode::Char(ch) if key.modifiers.contains(KeyModifiers::ALT) => Key::Alt(ch),
                KeyCode::Char(' ') => Key::Space,
                KeyCode::Char(ch) => Key::Char(ch),
                KeyCode::Enter => Key::Enter,
                KeyCode::Esc => Key::Escape,
                KeyCode::Tab | KeyCode::BackTab => Key::Tab,
                KeyCode::Backspace => Key::Backspace,
                KeyCode::Delete => Key::Delete,
                KeyCode::Insert => Key::Insert,
                KeyCode::Home => Key::Home,
                KeyCode::End => Key::End,
                KeyCode::PageUp => Key::PageUp,
                KeyCode::PageDown => Key::PageDown,
                KeyCode::F(n @ 1..=12) => Key::F(n),
                _ => return None,
            };
            Some(InputEvent::Key(key))
        }
        _ => None,
    }
}

fn blank_pixel() -> Pixel {
    Pixel {
        ch: ' ',
        fg: Color::WHITE,
        bg: None,
    }
}

fn safe_pixel(mut pixel: Pixel) -> Pixel {
    if pixel.ch == '\0' {
        return blank_pixel();
    }
    if pixel.ch.is_control()
        || matches!(pixel.ch, '\u{2028}' | '\u{2029}')
        || pixel.ch.width() != Some(1)
    {
        pixel.ch = '?';
    }
    pixel
}

struct Frame {
    size: (u16, u16),
    screen_size: (u16, u16),
    camera: Rect,
    cells: Vec<Pixel>,
}

impl Frame {
    fn capture(screen: &Screen, camera: Rect, size: (u16, u16)) -> io::Result<Self> {
        let len = usize::from(size.0) * usize::from(size.1);
        let mut cells = Vec::new();
        cells.try_reserve_exact(len).map_err(io::Error::other)?;
        cells.resize(len, blank_pixel());
        if let Some(crop) = screen.bounds().intersection(camera) {
            let width = crop.width.min(size.0);
            let height = crop.height.min(size.1);
            let left = (size.0 - width) / 2;
            let top = (size.1 - height) / 2;
            for y in 0..height {
                for x in 0..width {
                    // The intersection lies inside Screen's nonnegative bounds.
                    if let Some(cell) = screen.get(crop.x + i32::from(x), crop.y + i32::from(y)) {
                        let index =
                            usize::from(top + y) * usize::from(size.0) + usize::from(left + x);
                        cells[index] = safe_pixel(cell.pixel);
                    }
                }
            }
        }
        Ok(Self {
            size,
            screen_size: (screen.width(), screen.height()),
            camera,
            cells,
        })
    }

    fn same_viewport(&self, other: &Self) -> bool {
        self.size == other.size
            && self.screen_size == other.screen_size
            && self.camera == other.camera
    }
}

#[derive(Default)]
struct Renderer {
    last: Option<Frame>,
}

impl Renderer {
    fn invalidate(&mut self) {
        self.last = None;
    }

    fn render_to(
        &mut self,
        writer: &mut impl Write,
        screen: &Screen,
        camera: Rect,
        size: (u16, u16),
        force: bool,
    ) -> io::Result<()> {
        let next = Frame::capture(screen, camera, size)?;
        let previous = self
            .last
            .as_ref()
            .filter(|last| !force && last.same_viewport(&next));
        let bytes = encode_frame(&next, previous)?;
        if !bytes.is_empty() {
            if let Err(error) = writer.write_all(&bytes).and_then(|()| writer.flush()) {
                // Neither a partial write nor a failed flush commits a frame.
                // The terminal may contain arbitrary fragments: retry in full.
                self.invalidate();
                return Err(error);
            }
        }
        self.last = Some(next);
        Ok(())
    }
}

// Pure ANSI encoding, even on Windows: do not queue crossterm commands into a
// Vec (legacy Windows commands can bypass the writer). Native setup uses
// crossterm to enable VT support; rendering targets ANSI-capable terminals.
fn encode_frame(next: &Frame, previous: Option<&Frame>) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    if previous.is_none() {
        // Reset background *before* clearing exposed borders.
        bytes.extend_from_slice(b"\x1b[0m\x1b[2J");
    }
    let mut foreground = None;
    let mut background = None;
    for (index, pixel) in next.cells.iter().enumerate() {
        if previous.is_some_and(|last| last.cells[index] == *pixel) {
            continue;
        }
        let width = usize::from(next.size.0);
        write!(bytes, "\x1b[{};{}H", index / width + 1, index % width + 1)?;
        if foreground != Some(pixel.fg) {
            let Color { r, g, b } = pixel.fg;
            write!(bytes, "\x1b[38;2;{r};{g};{b}m")?;
            foreground = Some(pixel.fg);
        }
        if background != Some(pixel.bg) {
            match pixel.bg {
                Some(Color { r, g, b }) => write!(bytes, "\x1b[48;2;{r};{g};{b}m")?,
                None => bytes.extend_from_slice(b"\x1b[49m"),
            }
            background = Some(pixel.bg);
        }
        let mut buffer = [0; 4];
        bytes.extend_from_slice(pixel.ch.encode_utf8(&mut buffer).as_bytes());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyEvent;
    use std::collections::VecDeque;

    fn pixel(ch: char) -> Pixel {
        Pixel::new(ch, Color::WHITE, None)
    }

    fn screen(text: &str, width: u16, height: u16) -> Screen {
        let mut screen = Screen::new(width, height, pixel('.'));
        for (index, ch) in text.chars().enumerate() {
            screen.canvas().draw_pixel(
                (index % usize::from(width)) as i32,
                (index / usize::from(width)) as i32,
                0,
                pixel(ch),
            );
        }
        screen
    }

    fn glyphs(frame: &Frame) -> String {
        frame.cells.iter().map(|pixel| pixel.ch).collect()
    }

    #[test]
    fn requires_both_terminal_streams() {
        assert!(validate_tty(true, true).is_ok());
        for (input, output) in [(false, false), (false, true), (true, false)] {
            assert_eq!(
                validate_tty(input, output).unwrap_err().kind(),
                io::ErrorKind::NotConnected
            );
        }
    }

    #[test]
    fn non_tty_new_fails_without_entering_modes() {
        const CHILD: &str = "COSPLAY_TERMINAL_NON_TTY_TEST_CHILD";
        if std::env::var_os(CHILD).is_some() {
            assert_eq!(
                Terminal::new().err().unwrap().kind(),
                io::ErrorKind::NotConnected
            );
            assert_eq!(SESSION.flags.load(Ordering::SeqCst), 0);
            assert!(!CLAIMED.load(Ordering::SeqCst));
            return;
        }
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "terminal::tests::non_tty_new_fails_without_entering_modes",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .stdin(std::process::Stdio::null())
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        assert!(!output.stdout.contains(&0x1b));
    }

    #[derive(Default)]
    struct FakeControl {
        operations: Vec<Operation>,
        fail_once: Option<Operation>,
    }

    impl Control for FakeControl {
        fn run(&mut self, operation: Operation) -> io::Result<()> {
            self.operations.push(operation);
            if self.fail_once == Some(operation) {
                self.fail_once = None;
                Err(io::Error::other("injected control failure"))
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn lifecycle_restores_all_modes_and_is_idempotent() {
        let state = SessionState::new();
        let mut control = FakeControl::default();
        state.initialize(&mut control).unwrap();
        assert_eq!(state.flags.load(Ordering::SeqCst), ALL_FLAGS);
        assert_eq!(
            control.operations,
            [
                Operation::EnableRaw,
                Operation::EnterAlternate,
                Operation::HideCursor,
                Operation::DisableWrap,
                Operation::ResetColors,
                Operation::Clear,
            ]
        );
        control.operations.clear();
        state.restore(&mut control).unwrap();
        assert_eq!(
            control.operations,
            [
                Operation::ResetColors,
                Operation::EnableWrap,
                Operation::ShowCursor,
                Operation::LeaveAlternate,
                Operation::DisableRaw,
            ]
        );
        assert_eq!(state.flags.load(Ordering::SeqCst), 0);
        control.operations.clear();
        state.restore(&mut control).unwrap();
        assert!(control.operations.is_empty());
    }

    #[test]
    fn every_partial_initialization_failure_rolls_back() {
        for (failed, rollback) in [
            (Operation::EnableRaw, Operation::DisableRaw),
            (Operation::EnterAlternate, Operation::LeaveAlternate),
            (Operation::HideCursor, Operation::ShowCursor),
            (Operation::DisableWrap, Operation::EnableWrap),
            (Operation::ResetColors, Operation::ResetColors),
            (Operation::Clear, Operation::ResetColors),
        ] {
            let state = SessionState::new();
            let mut control = FakeControl {
                fail_once: Some(failed),
                ..FakeControl::default()
            };
            assert!(state.initialize(&mut control).is_err());
            let failure_index = control
                .operations
                .iter()
                .position(|op| *op == failed)
                .unwrap();
            assert!(control.operations[failure_index + 1..].contains(&rollback));
            assert_eq!(control.operations.last(), Some(&Operation::DisableRaw));
            assert_eq!(state.flags.load(Ordering::SeqCst), 0);
        }
    }

    #[test]
    fn failed_cleanup_does_not_skip_raw_restoration_and_can_retry() {
        for failed in [
            Operation::ResetColors,
            Operation::EnableWrap,
            Operation::ShowCursor,
            Operation::LeaveAlternate,
            Operation::DisableRaw,
        ] {
            let state = SessionState::new();
            let mut control = FakeControl::default();
            state.initialize(&mut control).unwrap();
            control.operations.clear();
            control.fail_once = Some(failed);
            assert!(state.restore(&mut control).is_err());
            assert_eq!(control.operations.len(), 5);
            assert_eq!(control.operations.last(), Some(&Operation::DisableRaw));
            assert_ne!(state.flags.load(Ordering::SeqCst), 0);
            control.operations.clear();
            state.restore(&mut control).unwrap();
            assert_eq!(control.operations, [failed]);
            assert_eq!(state.flags.load(Ordering::SeqCst), 0);
        }
    }

    fn key_event(code: KeyCode, modifiers: KeyModifiers, kind: KeyEventKind) -> Event {
        Event::Key(KeyEvent::new_with_kind(code, modifiers, kind))
    }

    #[test]
    fn maps_keys_repeats_resize_and_ctrl_c_but_not_releases() {
        for (code, expected) in [
            (KeyCode::Up, Key::Up),
            (KeyCode::Down, Key::Down),
            (KeyCode::Left, Key::Left),
            (KeyCode::Right, Key::Right),
            (KeyCode::Char('x'), Key::Char('x')),
            (KeyCode::Char(' '), Key::Space),
            (KeyCode::Enter, Key::Enter),
            (KeyCode::Esc, Key::Escape),
        ] {
            for kind in [KeyEventKind::Press, KeyEventKind::Repeat] {
                assert_eq!(
                    map_event(key_event(code, KeyModifiers::NONE, kind)),
                    Some(InputEvent::Key(expected))
                );
            }
            assert_eq!(
                map_event(key_event(code, KeyModifiers::NONE, KeyEventKind::Release)),
                None
            );
        }
        for ch in ['c', 'C'] {
            assert_eq!(
                map_event(key_event(
                    KeyCode::Char(ch),
                    KeyModifiers::CONTROL,
                    KeyEventKind::Press
                )),
                Some(InputEvent::Quit)
            );
            assert_eq!(
                map_event(key_event(
                    KeyCode::Char(ch),
                    KeyModifiers::CONTROL,
                    KeyEventKind::Release
                )),
                None
            );
        }
        assert_eq!(
            map_event(Event::Resize(80, 24)),
            Some(InputEvent::Resize(80, 24))
        );
        assert_eq!(map_event(Event::FocusGained), None);
        assert_eq!(
            map_event(key_event(
                KeyCode::F(1),
                KeyModifiers::NONE,
                KeyEventKind::Press
            )),
            Some(InputEvent::Key(Key::F(1)))
        );
        for (code, expected) in [
            (KeyCode::Tab, Key::Tab),
            (KeyCode::Backspace, Key::Backspace),
            (KeyCode::Delete, Key::Delete),
            (KeyCode::Insert, Key::Insert),
            (KeyCode::Home, Key::Home),
            (KeyCode::End, Key::End),
            (KeyCode::PageUp, Key::PageUp),
            (KeyCode::PageDown, Key::PageDown),
            (KeyCode::F(12), Key::F(12)),
        ] {
            assert_eq!(
                map_event(key_event(code, KeyModifiers::NONE, KeyEventKind::Press)),
                Some(InputEvent::Key(expected))
            );
        }
        assert_eq!(
            map_event(key_event(
                KeyCode::Char('a'),
                KeyModifiers::CONTROL,
                KeyEventKind::Press
            )),
            Some(InputEvent::Key(Key::Ctrl('a')))
        );
        assert_eq!(
            map_event(key_event(
                KeyCode::Char('x'),
                KeyModifiers::ALT,
                KeyEventKind::Press
            )),
            Some(InputEvent::Key(Key::Alt('x')))
        );
    }

    #[derive(Default)]
    struct FakeInput {
        events: VecDeque<Event>,
        waits: Vec<Duration>,
        fail_poll: bool,
        fail_read: bool,
    }

    impl InputSource for FakeInput {
        fn poll(&mut self, timeout: Duration) -> io::Result<bool> {
            self.waits.push(timeout);
            if self.fail_poll {
                Err(io::Error::other("poll failed"))
            } else {
                Ok(!self.events.is_empty())
            }
        }

        fn read(&mut self) -> io::Result<Event> {
            if self.fail_read {
                Err(io::Error::other("read failed"))
            } else {
                Ok(self.events.pop_front().expect("poll must precede read"))
            }
        }
    }

    #[test]
    fn input_waits_once_and_bounds_even_unrecognized_event_floods() {
        let timeout = Duration::from_millis(25);
        let mut input = FakeInput {
            events: vec![Event::FocusGained; MAX_INPUT_EVENTS + 10].into(),
            ..FakeInput::default()
        };
        assert!(collect_events(&mut input, timeout).unwrap().is_empty());
        assert_eq!(input.events.len(), 10);
        assert_eq!(input.waits.len(), MAX_INPUT_EVENTS);
        assert_eq!(input.waits[0], timeout);
        assert!(input.waits[1..].iter().all(Duration::is_zero));
        input.events.clear();
        input.waits.clear();
        assert!(collect_events(&mut input, timeout).unwrap().is_empty());
        assert_eq!(input.waits, [timeout]);
    }

    #[test]
    fn input_preserves_order_and_stops_at_quit() {
        let mut input = FakeInput {
            events: [
                Event::Resize(10, 20),
                key_event(KeyCode::Up, KeyModifiers::NONE, KeyEventKind::Press),
                key_event(
                    KeyCode::Char('c'),
                    KeyModifiers::CONTROL,
                    KeyEventKind::Press,
                ),
                Event::Resize(30, 40),
            ]
            .into(),
            ..FakeInput::default()
        };
        assert_eq!(
            collect_events(&mut input, Duration::ZERO).unwrap(),
            [
                InputEvent::Resize(10, 20),
                InputEvent::Key(Key::Up),
                InputEvent::Quit
            ]
        );
        assert_eq!(input.events.len(), 1);
    }

    #[test]
    fn input_errors_are_propagated() {
        for fail_poll in [true, false] {
            let mut input = FakeInput {
                events: [Event::FocusGained].into(),
                fail_poll,
                fail_read: !fail_poll,
                ..FakeInput::default()
            };
            assert!(collect_events(&mut input, Duration::ZERO).is_err());
        }
    }

    #[test]
    fn camera_offsets_centering_and_top_left_crop() {
        let screen = screen("abcdefghijkl", 4, 3);
        let frame = Frame::capture(&screen, Rect::new(1, 1, 2, 2), (6, 4)).unwrap();
        assert_eq!(glyphs(&frame), "        fg    jk        ");
        let frame = Frame::capture(&screen, Rect::new(1, 0, 3, 3), (2, 2)).unwrap();
        assert_eq!(glyphs(&frame), "bcfg");
    }

    #[test]
    fn camera_is_clipped_safely_and_empty_viewports_are_blank() {
        let screen = screen("abcdef", 3, 2);
        let frame = Frame::capture(&screen, Rect::new(-1, -1, 3, 3), (2, 2)).unwrap();
        assert_eq!(glyphs(&frame), "abde");
        for camera in [
            Rect::new(0, 0, 0, 2),
            Rect::new(0, 0, 2, 0),
            Rect::new(i32::MAX, i32::MAX, u16::MAX, u16::MAX),
            Rect::new(i32::MIN, i32::MIN, u16::MAX, u16::MAX),
        ] {
            assert_eq!(
                glyphs(&Frame::capture(&screen, camera, (3, 2)).unwrap()),
                "      "
            );
        }
        for size in [(0, 2), (3, 0), (0, 0)] {
            let frame = Frame::capture(&screen, screen.bounds(), size).unwrap();
            assert!(frame.cells.is_empty());
            assert_eq!(encode_frame(&frame, None).unwrap(), b"\x1b[0m\x1b[2J");
        }
    }

    #[test]
    fn sanitizes_public_pixels_including_escape_wide_and_zero_width() {
        for ch in [
            '\x1b', '\n', '\r', '\t', '\x7f', '\u{85}', '界', '\u{301}', '\u{200d}', '\u{2028}',
            '\u{2029}',
        ] {
            let unsafe_pixel = Pixel {
                ch,
                fg: Color::WHITE,
                bg: None,
            };
            assert_eq!(safe_pixel(unsafe_pixel).ch, '?', "{ch:?}");
        }
        assert_eq!(
            safe_pixel(Pixel {
                ch: '\0',
                fg: Color::BLACK,
                bg: Some(Color::WHITE)
            }),
            blank_pixel()
        );
        assert_eq!(safe_pixel(pixel('é')).ch, 'é');
    }

    #[test]
    fn unchanged_frames_write_nothing_and_one_change_is_a_single_cell_diff() {
        let mut screen = screen("ab", 2, 1);
        let mut renderer = Renderer::default();
        let mut bytes = Vec::new();
        renderer
            .render_to(&mut bytes, &screen, screen.bounds(), (2, 1), false)
            .unwrap();
        assert!(bytes.starts_with(b"\x1b[0m\x1b[2J"));
        bytes.clear();
        renderer
            .render_to(&mut bytes, &screen, screen.bounds(), (2, 1), false)
            .unwrap();
        assert!(bytes.is_empty());
        screen.canvas().draw_pixel(1, 0, 1, pixel('z'));
        renderer
            .render_to(&mut bytes, &screen, screen.bounds(), (2, 1), false)
            .unwrap();
        assert_eq!(bytes, b"\x1b[1;2H\x1b[38;2;255;255;255m\x1b[49mz");
    }

    #[test]
    fn background_is_reset_when_changing_to_default() {
        let colored = Screen::new(1, 1, Pixel::new('x', Color::WHITE, Some(Color::BLACK)));
        let default = Screen::new(1, 1, pixel('x'));
        let mut renderer = Renderer::default();
        let mut bytes = Vec::new();
        renderer
            .render_to(&mut bytes, &colored, colored.bounds(), (1, 1), false)
            .unwrap();
        assert!(String::from_utf8(bytes.clone())
            .unwrap()
            .contains("\x1b[48;2;0;0;0m"));
        bytes.clear();
        renderer
            .render_to(&mut bytes, &default, default.bounds(), (1, 1), false)
            .unwrap();
        assert_eq!(bytes, b"\x1b[1;1H\x1b[38;2;255;255;255m\x1b[49mx");
    }

    #[test]
    fn color_only_changes_are_not_lost_and_transparency_erases_old_cells() {
        let mut renderer = Renderer::default();
        let first = Screen::new(1, 1, pixel('a'));
        renderer
            .render_to(&mut Vec::new(), &first, first.bounds(), (1, 1), false)
            .unwrap();
        for (fg, bg, expected) in [
            (Color::BLACK, None, "\x1b[1;1H\x1b[38;2;0;0;0m\x1b[49ma"),
            (
                Color::BLACK,
                Some(Color::rgb(1, 2, 3)),
                "\x1b[1;1H\x1b[38;2;0;0;0m\x1b[48;2;1;2;3ma",
            ),
        ] {
            let screen = Screen::new(1, 1, Pixel::new('a', fg, bg));
            let mut bytes = Vec::new();
            renderer
                .render_to(&mut bytes, &screen, screen.bounds(), (1, 1), false)
                .unwrap();
            assert_eq!(bytes, expected.as_bytes());
        }
        let transparent = Screen::new(1, 1, Pixel::transparent());
        let mut bytes = Vec::new();
        renderer
            .render_to(
                &mut bytes,
                &transparent,
                transparent.bounds(),
                (1, 1),
                false,
            )
            .unwrap();
        assert_eq!(bytes, b"\x1b[1;1H\x1b[38;2;255;255;255m\x1b[49m ");
    }

    #[test]
    fn direct_pixel_fields_cannot_inject_controls_into_output() {
        for ch in ['\x1b', '\n', '界', '\u{301}', '\u{2028}', '\u{2029}'] {
            let screen = Screen::new(
                1,
                1,
                Pixel {
                    ch,
                    fg: Color::WHITE,
                    bg: None,
                },
            );
            let mut bytes = Vec::new();
            Renderer::default()
                .render_to(&mut bytes, &screen, screen.bounds(), (1, 1), false)
                .unwrap();
            assert_eq!(
                bytes,
                b"\x1b[0m\x1b[2J\x1b[1;1H\x1b[38;2;255;255;255m\x1b[49m?"
            );
        }
    }

    #[test]
    fn resize_camera_force_and_screen_size_invalidate_and_clear_borders() {
        let screen = screen("abcd", 4, 1);
        let mut renderer = Renderer::default();
        let mut bytes = Vec::new();
        let first_camera = Rect::new(0, 0, 2, 1);
        renderer
            .render_to(&mut bytes, &screen, first_camera, (2, 1), false)
            .unwrap();
        for (camera, size, force) in [
            (first_camera, (4, 3), false),
            (first_camera, (2, 1), false),
            (Rect::new(2, 0, 2, 1), (2, 1), false),
            (Rect::new(2, 0, 1, 1), (2, 1), false),
            (Rect::new(2, 0, 1, 1), (2, 1), true),
        ] {
            bytes.clear();
            renderer
                .render_to(&mut bytes, &screen, camera, size, force)
                .unwrap();
            assert!(bytes.starts_with(b"\x1b[0m\x1b[2J"));
        }
        assert_eq!(glyphs(renderer.last.as_ref().unwrap()), "c ");
        let larger = Screen::new(8, 2, pixel('c'));
        bytes.clear();
        renderer
            .render_to(&mut bytes, &larger, Rect::new(2, 0, 1, 1), (2, 1), false)
            .unwrap();
        assert!(bytes.starts_with(b"\x1b[0m\x1b[2J"));
        bytes.clear();
        renderer.invalidate();
        renderer
            .render_to(&mut bytes, &larger, Rect::new(2, 0, 1, 1), (2, 1), false)
            .unwrap();
        assert!(bytes.starts_with(b"\x1b[0m\x1b[2J"));
    }

    struct FailingWriter {
        remaining: usize,
        fail_flush: bool,
        bytes: Vec<u8>,
    }

    impl Write for FailingWriter {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            if self.remaining == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "injected write failure",
                ));
            }
            let len = buffer.len().min(self.remaining);
            self.bytes.extend_from_slice(&buffer[..len]);
            self.remaining -= len;
            Ok(len)
        }

        fn flush(&mut self) -> io::Result<()> {
            if self.fail_flush {
                Err(io::Error::other("injected flush failure"))
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn partial_write_and_flush_failures_force_a_complete_retry() {
        for (remaining, fail_flush) in [(0, false), (5, false), (usize::MAX, true)] {
            let mut screen = screen("ab", 2, 1);
            let mut renderer = Renderer::default();
            renderer
                .render_to(&mut Vec::new(), &screen, screen.bounds(), (2, 1), false)
                .unwrap();
            screen.canvas().draw_pixel(1, 0, 1, pixel('z'));
            let mut writer = FailingWriter {
                remaining,
                fail_flush,
                bytes: Vec::new(),
            };
            assert!(renderer
                .render_to(&mut writer, &screen, screen.bounds(), (2, 1), false)
                .is_err());
            assert!(renderer.last.is_none());
            let mut retry = Vec::new();
            renderer
                .render_to(&mut retry, &screen, screen.bounds(), (2, 1), false)
                .unwrap();
            let mut full = Vec::new();
            Renderer::default()
                .render_to(&mut full, &screen, screen.bounds(), (2, 1), false)
                .unwrap();
            assert_eq!(retry, full);
            retry.clear();
            renderer
                .render_to(&mut retry, &screen, screen.bounds(), (2, 1), false)
                .unwrap();
            assert!(retry.is_empty());
        }
    }

    #[test]
    fn initial_output_failure_also_retries_the_full_frame() {
        let screen = screen("a", 1, 1);
        let mut renderer = Renderer::default();
        let mut writer = FailingWriter {
            remaining: 3,
            fail_flush: false,
            bytes: Vec::new(),
        };
        assert!(renderer
            .render_to(&mut writer, &screen, screen.bounds(), (1, 1), false)
            .is_err());
        assert!(renderer.last.is_none());
        let mut bytes = Vec::new();
        renderer
            .render_to(&mut bytes, &screen, screen.bounds(), (1, 1), false)
            .unwrap();
        assert_eq!(
            bytes,
            b"\x1b[0m\x1b[2J\x1b[1;1H\x1b[38;2;255;255;255m\x1b[49ma"
        );
    }

    #[test]
    fn successful_short_writes_are_completed_before_committing() {
        #[derive(Default)]
        struct ShortWriter {
            bytes: Vec<u8>,
            flushes: usize,
        }
        impl Write for ShortWriter {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                let len = bytes.len().min(2);
                self.bytes.extend_from_slice(&bytes[..len]);
                Ok(len)
            }
            fn flush(&mut self) -> io::Result<()> {
                self.flushes += 1;
                Ok(())
            }
        }
        let screen = screen("é", 1, 1);
        let mut renderer = Renderer::default();
        let mut writer = ShortWriter::default();
        renderer
            .render_to(&mut writer, &screen, screen.bounds(), (1, 1), false)
            .unwrap();
        assert_eq!(
            writer.bytes,
            "\x1b[0m\x1b[2J\x1b[1;1H\x1b[38;2;255;255;255m\x1b[49mé".as_bytes()
        );
        assert_eq!(writer.flushes, 1);
        assert!(renderer.last.is_some());
    }

    #[test]
    fn unchanged_frame_does_not_even_flush() {
        let screen = screen("a", 1, 1);
        let mut renderer = Renderer::default();
        renderer
            .render_to(&mut Vec::new(), &screen, screen.bounds(), (1, 1), false)
            .unwrap();
        let mut writer = FailingWriter {
            remaining: 0,
            fail_flush: true,
            bytes: Vec::new(),
        };
        renderer
            .render_to(&mut writer, &screen, screen.bounds(), (1, 1), false)
            .unwrap();
    }
}
