// SPDX-License-Identifier: Apache-2.0
use cosplay::engine::Engine;
use cosplay::pong;
use cosplay::terminal::{install_panic_hook, InputEvent, Key, Terminal};
use std::error::Error;
use std::time::{Duration, Instant};

const FRAME: Duration = Duration::from_nanos(1_000_000_000 / 30);
const HELP: &str = "CosPlay Rust Pong\n\nUsage: pong [--headless [--frames N] [--size WIDTHxHEIGHT]]\n\nControls: W/S or arrows move; Space/P pause; R restart; Q/Esc/Ctrl-C quit.\nFirst to 7 wins. Minimum terminal size: 40x16.\n\n--headless  Run deterministic autoplay without a terminal (default: 300 frames)\n--frames N  Number of headless frames (1..1000000)\n--size WxH  Headless dimensions (1..240 by 1..100; default: 80x24)\n--help      Show this help\n";

#[derive(Debug, PartialEq)]
struct Options {
    headless: bool,
    frames: u64,
    size: (u16, u16),
}

fn parse(args: impl Iterator<Item = String>) -> Result<Option<Options>, String> {
    let mut args = args.peekable();
    let mut opts = Options {
        headless: false,
        frames: 300,
        size: (80, 24),
    };
    let mut headless_option = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => return Ok(None),
            "--headless" => opts.headless = true,
            "--frames" => {
                headless_option = true;
                opts.frames = args
                    .next()
                    .ok_or("--frames requires a number")?
                    .parse()
                    .map_err(|_| "invalid frame count")?;
                if !(1..=1_000_000).contains(&opts.frames) {
                    return Err("frames must be in 1..1000000".into());
                }
            }
            "--size" => {
                headless_option = true;
                let value = args.next().ok_or("--size requires WIDTHxHEIGHT")?;
                let (w, h) = value.split_once('x').ok_or("size must be WIDTHxHEIGHT")?;
                opts.size = (
                    w.parse().map_err(|_| "invalid width")?,
                    h.parse().map_err(|_| "invalid height")?,
                );
                if !(1..=240).contains(&opts.size.0) || !(1..=100).contains(&opts.size.1) {
                    return Err("size must be in 1..240 by 1..100".into());
                }
            }
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }
    if headless_option && !opts.headless {
        return Err("--frames and --size require --headless".into());
    }
    Ok(Some(opts))
}

fn game(size: (u16, u16), autoplay: bool) -> Result<Engine, Box<dyn Error>> {
    let mut engine = Engine::new(size.0, size.1);
    engine.add_scene(pong::scene(autoplay))?;
    engine.start("pong")?;
    Ok(engine)
}

/// A deadline clock separated from wall-clock access for deterministic tests.
#[derive(Default)]
struct FrameClock {
    next: Duration,
    last: Duration,
}

impl FrameClock {
    fn wait(&self, now: Duration) -> Duration {
        self.next.saturating_sub(now)
    }

    fn tick(&mut self, now: Duration) -> Option<Duration> {
        if now < self.next {
            return None;
        }
        let delta = now.saturating_sub(self.last);
        self.last = now;
        // Schedule from this tick, not a stale deadline: never replay missed
        // frames in a burst. Rendering time is included in the next delta.
        self.next = now.saturating_add(FRAME);
        Some(delta)
    }
}

fn run(opts: Options) -> Result<(), Box<dyn Error>> {
    if opts.headless {
        let mut engine = game(opts.size, true)?;
        for _ in 0..opts.frames {
            engine.tick(FRAME, &[])?;
        }
        let snapshot = engine.screen().snapshot();
        let hash = snapshot.bytes().fold(0xcbf29ce484222325u64, |hash, b| {
            (hash ^ u64::from(b)).wrapping_mul(0x100000001b3)
        });
        println!(
            "headless: frames={} size={}x{} snapshot={hash:016x}",
            engine.frame_count(),
            opts.size.0,
            opts.size.1
        );
        println!("{snapshot}");
        return Ok(());
    }
    install_panic_hook();
    let mut terminal = Terminal::new()?;
    let mut engine = game(terminal.size()?, false)?;
    let epoch = Instant::now();
    let mut clock = FrameClock::default();
    let mut keys = Vec::new();
    let mut redraw = true;
    while engine.is_running() {
        let wait = clock.wait(epoch.elapsed());
        for event in terminal.poll_events(wait)? {
            match event {
                InputEvent::Quit => {
                    engine.shutdown();
                    break;
                }
                InputEvent::Key(key) => {
                    // Never lose quit behind a burst of keyboard repeat events.
                    if matches!(key, Key::Escape | Key::Char('q' | 'Q')) {
                        engine.shutdown();
                        break;
                    }
                    if keys.len() < 128 {
                        keys.push(key);
                    }
                }
                InputEvent::Resize(width, height) => {
                    engine.resize(width, height);
                    redraw = true;
                }
            }
        }
        if !engine.is_running() {
            break;
        }
        let Some(delta) = clock.tick(epoch.elapsed()) else {
            continue;
        };
        engine.tick(delta, &keys)?;
        keys.clear();
        terminal.render(engine.screen(), engine.camera(), redraw)?;
        redraw = false;
    }
    terminal.restore()?;
    Ok(())
}

fn main() {
    let result = match parse(std::env::args().skip(1)) {
        Ok(None) => {
            print!("{HELP}");
            return;
        }
        Ok(Some(opts)) => run(opts),
        Err(error) => Err(error.into()),
    };
    if let Err(error) = result {
        eprintln!("pong: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn options(args: &[&str]) -> Result<Option<Options>, String> {
        parse(args.iter().map(|s| s.to_string()))
    }
    #[test]
    fn frame_clock_accounts_for_render_latency_without_catch_up_bursts() {
        let mut clock = FrameClock::default();
        assert_eq!(clock.tick(Duration::ZERO), Some(Duration::ZERO));
        let render_latency = Duration::from_millis(10);
        assert_eq!(clock.wait(render_latency), FRAME - render_latency);
        assert_eq!(clock.tick(FRAME), Some(FRAME));
        let stalled_at = FRAME + Duration::from_millis(150);
        assert_eq!(clock.wait(stalled_at), Duration::ZERO);
        assert_eq!(clock.tick(stalled_at), Some(Duration::from_millis(150)));
        assert_eq!(clock.tick(stalled_at), None);
        assert_eq!(clock.wait(stalled_at), FRAME);
    }
    #[test]
    fn validates_arguments_without_opening_a_terminal() {
        assert_eq!(options(&["--help"]), Ok(None));
        assert_eq!(
            options(&["--headless", "--frames", "12", "--size", "40x16"]).unwrap(),
            Some(Options {
                headless: true,
                frames: 12,
                size: (40, 16)
            })
        );
        for bad in [
            vec!["--unknown"],
            vec!["--frames"],
            vec!["--frames", "2"],
            vec!["--headless", "--frames", "0"],
            vec!["--headless", "--size", "0x5"],
            vec!["--headless", "--size", "65535x65535"],
        ] {
            assert!(options(&bad).is_err(), "{bad:?}");
        }
    }
}
