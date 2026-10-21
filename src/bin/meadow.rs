// SPDX-License-Identifier: Apache-2.0
//! Native 2D glyph meadow with a live animated 3D knight.
#![forbid(unsafe_code)]

use cosplay::meadow::{renderer::GpuRenderer, world::World, Controls, GlyphInstance, MeadowFrame};
use std::{
    collections::HashSet,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const HELP: &str = "CosPlay Meadow - native 2D glyph world, live 3D character\n\nUsage: meadow [options]\n\nWASD / arrows move | Shift run | +/- or wheel zoom\nSpace pause | R reset | F1 toggle auto-walk | Esc quit\n\n--headless       Capture a GPU-rendered PNG without opening a window\n--frames N       Headless simulation steps (default 120); window frame limit if supplied\n--output PATH    Headless PNG destination (default meadow.png)\n--size WxH       Pixels (default 1280x800; 64..4096 by 64..2160)\n--seed N         Deterministic world seed (default 2024)\n--density X      Grass/flower density (0.25..2; default 1)\n--zoom X         Zoom (0.5..3; default 1)\n--auto           Start in auto-walk mode\n--licenses       Print embedded font license\n--help           Show this help\n\nHeadless capture still requires a supported GPU/graphics adapter.\n";
const FRAME: Duration = Duration::from_nanos(1_000_000_000 / 60);

#[derive(Clone, Debug, PartialEq)]
struct Options {
    headless: bool,
    frames: Option<u32>,
    output: PathBuf,
    size: (u32, u32),
    seed: u32,
    density: f32,
    zoom: f32,
    autoplay: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            headless: false,
            frames: None,
            output: "meadow.png".into(),
            size: (1280, 800),
            seed: 2024,
            density: 1.0,
            zoom: 1.0,
            autoplay: false,
        }
    }
}

#[derive(Debug, PartialEq)]
enum Invocation {
    Run(Options),
    Help,
    Licenses,
}

fn parse(args: impl Iterator<Item = String>) -> Result<Invocation, String> {
    let mut args = args;
    let mut opts = Options::default();
    let mut output_given = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => return Ok(Invocation::Help),
            "--licenses" => return Ok(Invocation::Licenses),
            "--headless" => opts.headless = true,
            "--auto" => opts.autoplay = true,
            "--frames" => {
                let value: u32 = args
                    .next()
                    .ok_or("--frames requires a count")?
                    .parse()
                    .map_err(|_| "invalid frame count")?;
                if value > 100_000 {
                    return Err("frame count must be in 0..100000".into());
                }
                opts.frames = Some(value);
            }
            "--output" => {
                output_given = true;
                let path = args.next().ok_or("--output requires a path")?;
                if path.is_empty() {
                    return Err("output path must not be empty".into());
                }
                opts.output = path.into();
            }
            "--size" => {
                let text = args.next().ok_or("--size requires WIDTHxHEIGHT")?;
                let (w, h) = text.split_once('x').ok_or("size must be WIDTHxHEIGHT")?;
                opts.size = (
                    w.parse().map_err(|_| "invalid width")?,
                    h.parse().map_err(|_| "invalid height")?,
                );
                if !(64..=4096).contains(&opts.size.0) || !(64..=2160).contains(&opts.size.1) {
                    return Err("size must be 64..4096 by 64..2160".into());
                }
            }
            "--seed" => {
                opts.seed = args
                    .next()
                    .ok_or("--seed requires an unsigned integer")?
                    .parse()
                    .map_err(|_| "invalid seed")?
            }
            "--density" | "--zoom" => {
                let value: f32 = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a number"))?
                    .parse()
                    .map_err(|_| format!("invalid {arg}"))?;
                let range = if arg == "--density" {
                    0.25..=2.0
                } else {
                    0.5..=3.0
                };
                if !value.is_finite() || !range.contains(&value) {
                    return Err(format!("{arg} must be in {range:?}"));
                }
                if arg == "--density" {
                    opts.density = value;
                } else {
                    opts.zoom = value;
                }
            }
            _ => return Err(format!("unknown option: {arg}")),
        }
    }
    if output_given && !opts.headless {
        return Err("--output requires --headless".into());
    }
    if opts.frames == Some(0) && !opts.headless {
        return Err("window frame limit must be positive".into());
    }
    Ok(Invocation::Run(opts))
}

#[derive(Default)]
struct InputState {
    held: HashSet<KeyCode>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Action {
    Quit,
    Pause,
    Reset,
    Auto,
    Zoom(f32),
}

impl InputState {
    /// Repeats and synthetic focus synchronization update held state, but must
    /// never fire one-shot actions such as reset, pause, or quit.
    fn event(
        &mut self,
        code: KeyCode,
        state: ElementState,
        suppress_actions: bool,
    ) -> Option<Action> {
        if state == ElementState::Released {
            self.held.remove(&code);
            return None;
        }
        let first = self.held.insert(code);
        if suppress_actions || !first {
            return None;
        }
        match code {
            KeyCode::Escape => Some(Action::Quit),
            KeyCode::Space => Some(Action::Pause),
            KeyCode::KeyR => Some(Action::Reset),
            KeyCode::F1 => Some(Action::Auto),
            KeyCode::Equal | KeyCode::NumpadAdd => Some(Action::Zoom(1.15)),
            KeyCode::Minus | KeyCode::NumpadSubtract => Some(Action::Zoom(1.0 / 1.15)),
            _ => None,
        }
    }

    fn clear(&mut self) {
        self.held.clear();
    }

    fn controls(&self, autoplay: bool) -> Controls {
        let held = |a, b| self.held.contains(&a) || self.held.contains(&b);
        Controls {
            up: held(KeyCode::KeyW, KeyCode::ArrowUp),
            down: held(KeyCode::KeyS, KeyCode::ArrowDown),
            left: held(KeyCode::KeyA, KeyCode::ArrowLeft),
            right: held(KeyCode::KeyD, KeyCode::ArrowRight),
            run: held(KeyCode::ShiftLeft, KeyCode::ShiftRight),
            autoplay,
        }
    }
}

/// Screen-space HUD expressed through the same independent glyph renderer.
fn hud(frame: &mut MeadowFrame, width: u32, height: u32, paused: bool, autoplay: bool) {
    let unit = frame.view_height / height as f32;
    let left = frame.camera[0] - frame.view_height * width as f32 / height as f32 / 2.0;
    let top = frame.camera[1] + frame.view_height / 2.0;
    frame.instances.push(GlyphInstance {
        position: [left + 177.0 * unit, top - 46.0 * unit],
        size: [334.0 * unit, 72.0 * unit],
        color: [0.003, 0.008, 0.009, 1.0],
        rotation: 0.0,
        depth: 0.002,
        glyph: 32,
        kind: 1,
    });
    let status = if paused {
        "PAUSED - SPACE TO RESUME"
    } else if autoplay {
        "AUTO-WALK - F1 FOR MANUAL"
    } else {
        "WASD / ARROWS MOVE - SHIFT RUN"
    };
    for (row, line) in [
        "COSPLAY / MEADOW",
        status,
        "WHEEL +/- ZOOM  R RESET  ESC QUIT",
    ]
    .iter()
    .enumerate()
    {
        for (column, ch) in line.chars().enumerate() {
            frame.instances.push(GlyphInstance {
                position: [
                    left + (20.0 + column as f32 * 9.8) * unit,
                    top - (26.0 + row as f32 * 20.0) * unit,
                ],
                size: [10.0 * unit, 16.0 * unit],
                color: if row == 0 {
                    [0.5, 0.9, 0.7, 1.0]
                } else {
                    [0.5, 0.65, 0.6, 1.0]
                },
                rotation: 0.0,
                depth: 0.001,
                glyph: ch as u32,
                kind: 0,
            });
        }
    }
}

fn headless(opts: Options) -> Result<(), String> {
    let mut world = World::new(opts.seed, opts.density);
    let steps = opts.frames.unwrap_or(120);
    for _ in 0..steps {
        world.step(
            1.0 / 30.0,
            Controls {
                autoplay: opts.autoplay,
                ..Default::default()
            },
        );
    }
    let (width, height) = opts.size;
    let mut frame = world.frame(width as f32 / height as f32, opts.zoom);
    hud(&mut frame, width, height, false, opts.autoplay);
    let instance = wgpu::Instance::default();
    let mut renderer = pollster::block_on(GpuRenderer::new(&instance, None))?;
    let info = renderer.adapter.get_info();
    renderer.capture_png(width, height, &frame, &opts.output)?;
    println!(
        "meadow: adapter={} backend={:?} frames={steps} seed={} glyphs={} player={:?} output={}",
        info.name,
        info.backend,
        opts.seed,
        frame.instances.len(),
        world.player_position(),
        opts.output.display()
    );
    Ok(())
}

struct WindowState {
    renderer: GpuRenderer,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    window: Arc<Window>,
    size: PhysicalSize<u32>,
    _instance: wgpu::Instance,
}

impl WindowState {
    fn new(event_loop: &ActiveEventLoop, opts: &Options) -> Result<Self, String> {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("CosPlay Meadow - 2D world / live 3D glyph knight")
                        .with_inner_size(PhysicalSize::new(opts.size.0, opts.size.1)),
                )
                .map_err(|e| format!("create window: {e}"))?,
        );
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| format!("create GPU surface: {e}"))?;
        let renderer = pollster::block_on(GpuRenderer::new(&instance, Some(&surface)))?;
        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&renderer.adapter, size.width.max(1), size.height.max(1))
            .ok_or("no supported surface configuration")?;
        config.format = renderer.format;
        config.present_mode = wgpu::PresentMode::Fifo;
        surface.configure(&renderer.device, &config);
        let info = renderer.adapter.get_info();
        println!("meadow: {} ({:?}) - WASD/arrows move, Shift run, wheel zoom, Space pause, F1 auto-walk, Esc quit",info.name,info.backend);
        Ok(Self {
            renderer,
            surface,
            config,
            window,
            size,
            _instance: instance,
        })
    }

    fn resize(&mut self, size: PhysicalSize<u32>) -> Result<(), String> {
        self.size = size;
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }
        let max = self.renderer.device.limits().max_texture_dimension_2d;
        if size.width > max || size.height > max {
            return Err(format!("window exceeds GPU texture dimension limit {max}"));
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.renderer.device, &self.config);
        Ok(())
    }

    fn draw(&mut self, frame: &MeadowFrame) -> Result<bool, String> {
        if self.size.width == 0 || self.size.height == 0 {
            return Ok(false);
        }
        let output = match self.surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.renderer.device, &self.config);
                return Ok(false);
            }
            Err(wgpu::SurfaceError::Timeout) => return Ok(false),
            Err(error) => return Err(format!("acquire surface: {error}")),
        };
        let view = output.texture.create_view(&Default::default());
        self.renderer
            .render(&view, self.size.width, self.size.height, frame)?;
        self.window.pre_present_notify();
        output.present();
        Ok(true)
    }
}

struct App {
    opts: Options,
    world: World,
    input: InputState,
    state: Option<WindowState>,
    paused: bool,
    error: Option<String>,
    last_tick: Instant,
    next_redraw: Instant,
    stats_at: Instant,
    stats_frames: u32,
    frames: u32,
}

impl App {
    fn new(opts: Options) -> Self {
        let now = Instant::now();
        Self {
            world: World::new(opts.seed, opts.density),
            opts,
            input: InputState::default(),
            state: None,
            paused: false,
            error: None,
            last_tick: now,
            next_redraw: now,
            stats_at: now,
            stats_frames: 0,
            frames: 0,
        }
    }
    fn fail(&mut self, event_loop: &ActiveEventLoop, error: String) {
        self.error = Some(error);
        event_loop.exit();
    }
    fn action(&mut self, event_loop: &ActiveEventLoop, action: Action) {
        match action {
            Action::Quit => event_loop.exit(),
            Action::Pause => self.paused = !self.paused,
            Action::Auto => self.opts.autoplay = !self.opts.autoplay,
            Action::Reset => {
                self.world = World::new(self.opts.seed, self.opts.density);
                self.paused = false;
            }
            Action::Zoom(factor) => self.opts.zoom = (self.opts.zoom * factor).clamp(0.5, 3.0),
        }
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_tick).as_secs_f32();
        self.last_tick = now;
        self.next_redraw = now + FRAME;
        let Some(state) = &mut self.state else {
            return;
        };
        if state.size.width == 0 || state.size.height == 0 {
            return;
        }
        if !self.paused {
            self.world.step(dt, self.input.controls(self.opts.autoplay));
        }
        let mut frame = self.world.frame(
            state.size.width as f32 / state.size.height as f32,
            self.opts.zoom,
        );
        hud(
            &mut frame,
            state.size.width,
            state.size.height,
            self.paused,
            self.opts.autoplay,
        );
        match state.draw(&frame) {
            Ok(true) => {
                self.frames += 1;
                self.stats_frames += 1;
            }
            Ok(false) => (),
            Err(error) => {
                self.fail(event_loop, error);
                return;
            }
        }
        let elapsed = now.duration_since(self.stats_at).as_secs_f32();
        if elapsed >= 0.5 {
            state.window.set_title(&format!(
                "CosPlay Meadow | {:.0} FPS | {} glyphs | 2D world + live 3D knight",
                self.stats_frames as f32 / elapsed,
                frame.instances.len()
            ));
            self.stats_frames = 0;
            self.stats_at = now;
        }
        if self.opts.frames.is_some_and(|limit| self.frames >= limit) {
            event_loop.exit();
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            match WindowState::new(event_loop, &self.opts) {
                Ok(state) => {
                    self.state = Some(state);
                    self.last_tick = Instant::now();
                    self.next_redraw = self.last_tick;
                    self.stats_at = self.last_tick;
                }
                Err(error) => self.fail(event_loop, error),
            }
        }
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if self.state.as_ref().is_none_or(|s| s.window.id() != id) {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Err(error) = self.state.as_mut().unwrap().resize(size) {
                    self.fail(event_loop, error);
                }
                self.last_tick = Instant::now();
                self.next_redraw = self.last_tick;
            }
            WindowEvent::Focused(false) => self.input.clear(),
            WindowEvent::KeyboardInput {
                event,
                is_synthetic,
                ..
            } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if let Some(action) =
                        self.input
                            .event(code, event.state, event.repeat || is_synthetic)
                    {
                        self.action(event_loop, action);
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => (p.y / 60.0) as f32,
                };
                if amount.is_finite() {
                    self.action(
                        event_loop,
                        Action::Zoom(1.1f32.powf(amount.clamp(-5.0, 5.0))),
                    );
                }
            }
            WindowEvent::RedrawRequested => self.redraw(event_loop),
            _ => (),
        }
    }
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(state) = &self.state else {
            event_loop.set_control_flow(ControlFlow::Wait);
            return;
        };
        if state.size.width == 0 || state.size.height == 0 {
            event_loop.set_control_flow(ControlFlow::Wait);
            return;
        }
        let now = Instant::now();
        if now >= self.next_redraw {
            state.window.request_redraw();
        }
        event_loop.set_control_flow(redraw_flow(now, self.next_redraw));
    }
    fn suspended(&mut self, _: &ActiveEventLoop) {
        self.input.clear();
        self.state = None;
    }
}

fn redraw_flow(now: Instant, deadline: Instant) -> ControlFlow {
    // A redraw is already requested when due. Wait for its delivery instead of
    // repeatedly waking on an expired deadline while the compositor delays it.
    if now >= deadline {
        ControlFlow::Wait
    } else {
        ControlFlow::WaitUntil(deadline)
    }
}

fn run(opts: Options) -> Result<(), String> {
    if opts.headless {
        return headless(opts);
    }
    let event_loop = EventLoop::new().map_err(|e| format!("create event loop: {e}"))?;
    let mut app = App::new(opts);
    event_loop
        .run_app(&mut app)
        .map_err(|e| format!("event loop: {e}"))?;
    match app.error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn main() {
    let result = match parse(std::env::args().skip(1)) {
        Ok(Invocation::Help) => {
            print!("{HELP}");
            return;
        }
        Ok(Invocation::Licenses) => {
            print!(
                "{}",
                include_str!("../../assets/SOURCE-CODE-PRO-LICENSE.txt")
            );
            return;
        }
        Ok(Invocation::Run(opts)) => run(opts),
        Err(error) => Err(error),
    };
    if let Err(error) = result {
        eprintln!("meadow: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn opts(args: &[&str]) -> Result<Invocation, String> {
        parse(args.iter().map(|s| s.to_string()))
    }
    #[test]
    fn cli_validates_before_any_gpu_or_window_is_created() {
        assert_eq!(opts(&["--help"]), Ok(Invocation::Help));
        assert_eq!(opts(&["--licenses"]), Ok(Invocation::Licenses));
        assert!(matches!(
            opts(&["--headless", "--frames", "0", "--size", "640x400", "--auto"]),
            Ok(Invocation::Run(_))
        ));
        for args in [
            vec!["--wat"],
            vec!["--frames"],
            vec!["--frames", "0"],
            vec!["--output", "a.png"],
            vec!["--size", "0x400"],
            vec!["--size", "400x99999"],
            vec!["--zoom", "NaN"],
            vec!["--density", "inf"],
            vec!["--density", "0"],
            vec!["--seed", "-1"],
        ] {
            assert!(opts(&args).is_err(), "{args:?}");
        }
    }
    #[test]
    fn held_keys_release_independently_and_focus_loss_clears_them() {
        let mut input = InputState::default();
        input.event(KeyCode::KeyW, ElementState::Pressed, false);
        input.event(KeyCode::ArrowUp, ElementState::Pressed, false);
        input.event(KeyCode::ShiftLeft, ElementState::Pressed, false);
        assert!(input.controls(false).up && input.controls(false).run);
        input.event(KeyCode::KeyW, ElementState::Released, false);
        assert!(input.controls(false).up);
        input.clear();
        assert!(!input.controls(false).up);
        assert!(!input.controls(false).run);
    }
    #[test]
    fn repeated_keydown_does_not_repeat_toggles() {
        let mut input = InputState::default();
        assert_eq!(
            input.event(KeyCode::Space, ElementState::Pressed, false),
            Some(Action::Pause)
        );
        assert_eq!(
            input.event(KeyCode::Space, ElementState::Pressed, true),
            None
        );
        assert_eq!(
            input.event(KeyCode::Space, ElementState::Pressed, false),
            None
        );
        input.event(KeyCode::Space, ElementState::Released, false);
        assert_eq!(
            input.event(KeyCode::Space, ElementState::Pressed, false),
            Some(Action::Pause)
        );
    }
    #[test]
    fn synthetic_focus_keys_update_movement_without_firing_actions() {
        let mut input = InputState::default();
        for key in [KeyCode::Escape, KeyCode::Space, KeyCode::KeyR, KeyCode::F1] {
            assert_eq!(input.event(key, ElementState::Pressed, true), None);
        }
        input.event(KeyCode::KeyW, ElementState::Pressed, true);
        assert!(input.controls(false).up);
        input.event(KeyCode::KeyW, ElementState::Released, true);
        assert!(!input.controls(false).up);
    }
    #[test]
    fn delayed_redraw_waits_instead_of_spinning_on_an_expired_deadline() {
        let now = Instant::now();
        assert_eq!(redraw_flow(now, now), ControlFlow::Wait);
        assert_eq!(redraw_flow(now + FRAME, now), ControlFlow::Wait);
        assert_eq!(
            redraw_flow(now, now + FRAME),
            ControlFlow::WaitUntil(now + FRAME)
        );
    }
    #[test]
    fn hud_is_frontmost_and_uses_screen_space() {
        let mut frame = MeadowFrame {
            camera: [100.0, 200.0],
            view_height: 640.0,
            time: 0.0,
            instances: vec![],
            hero: Default::default(),
        };
        hud(&mut frame, 1280, 800, false, false);
        assert_eq!(frame.instances[0].kind, 1);
        assert!(frame.instances.iter().all(|i| i.depth < 0.01));
        assert!(frame
            .instances
            .iter()
            .all(|i| i.position.iter().all(|v| v.is_finite())));
    }
}
