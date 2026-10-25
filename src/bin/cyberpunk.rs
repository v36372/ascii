// SPDX-License-Identifier: Apache-2.0
//! Native Neon Ward demo, using the existing spatial glyph renderer.
#![forbid(unsafe_code)]

use cosplay::{
    cyberpunk::{hud, City},
    spatial::{
        app::{self, AppOptions, Application, View},
        Controls, SceneFrame,
    },
};
use std::path::PathBuf;

const HELP: &str = "CosPlay / NEON WARD - native cyberpunk glyph city\n\nUsage: cyberpunk [options]\n\nWASD / arrows move | Shift run | +/- or wheel zoom\nSpace pause | R reset | F1 toggle auto-walk | Esc quit\n\n--headless       Capture a GPU-rendered PNG without opening a window\n--frames N       Headless simulation steps (default 120); window frame limit if supplied\n--output PATH    Headless PNG destination (default cyberpunk.png)\n--size WxH       Pixels (default 1280x800; 64..4096 by 64..2160)\n--seed N         Deterministic world seed (default 2024)\n--zoom X         Zoom (0.5..3; default 1)\n--auto           Start in auto-walk mode\n--licenses       Print embedded font license\n--help           Show this help\n\nHeadless capture still requires a supported GPU/graphics adapter.\n";

#[derive(Clone, Debug, PartialEq)]
struct Options {
    headless: bool,
    frames: Option<u32>,
    output: PathBuf,
    size: (u32, u32),
    seed: u32,
    zoom: f32,
    autoplay: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            headless: false,
            frames: None,
            output: "cyberpunk.png".into(),
            size: (1280, 800),
            seed: 2024,
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
            "--zoom" => {
                let value: f32 = args
                    .next()
                    .ok_or("--zoom requires a number")?
                    .parse()
                    .map_err(|_| "invalid zoom")?;
                if !value.is_finite() || !(0.5..=3.0).contains(&value) {
                    return Err("zoom must be in 0.5..3".into());
                }
                opts.zoom = value;
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

struct CityApp {
    world: City,
    seed: u32,
}
impl Application for CityApp {
    type Frame = SceneFrame;
    fn update(&mut self, dt: f32, controls: Controls) {
        self.world.step(dt, controls);
    }
    fn reset(&mut self) -> Result<(), String> {
        self.world = City::new(self.seed).map_err(|e| e.to_string())?;
        Ok(())
    }
    fn frame(&self, view: View) -> Result<Self::Frame, String> {
        let mut frame = self
            .world
            .frame(view.aspect(), view.zoom)
            .map_err(|e| e.to_string())?;
        hud(
            &mut frame,
            view.width,
            view.height,
            view.paused,
            view.autoplay,
        )
        .map_err(|e| e.to_string())?;
        Ok(frame)
    }
}

fn run(opts: Options) -> Result<(), String> {
    let mut game = CityApp {
        world: City::new(opts.seed).map_err(|e| e.to_string())?,
        seed: opts.seed,
    };
    let options = AppOptions {
        title: "CosPlay / NEON WARD".into(),
        size: opts.size,
        frames: opts.frames,
        zoom: opts.zoom,
        autoplay: opts.autoplay,
    };
    if opts.headless {
        let steps = opts.frames.unwrap_or(120);
        let report = app::capture(&mut game, &options, steps, &opts.output)?;
        println!(
            "cyberpunk: adapter={} backend={:?} frames={steps} seed={} glyphs={} player={:?} output={}",
            report.adapter.name, report.adapter.backend, opts.seed, report.glyphs,
            game.world.player_position(), opts.output.display()
        );
        Ok(())
    } else {
        app::run(game, options)
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
        eprintln!("cyberpunk: {error}");
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
    fn hud_is_frontmost_and_uses_screen_space() {
        let mut frame = SceneFrame::default();
        frame.camera.center = [100.0, 200.0];
        frame.camera.view_height = 640.0;
        hud(&mut frame, 1280, 800, false, false).unwrap();
        assert_eq!(frame.instances()[0].kind, 1);
        assert!(frame.instances().iter().all(|i| i.depth < 0.01));
        assert!(frame
            .instances()
            .iter()
            .all(|i| i.position.iter().all(|v| v.is_finite())));
    }
}
