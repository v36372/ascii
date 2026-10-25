// SPDX-License-Identifier: Apache-2.0
//! Native 2D glyph meadow with a live animated 3D knight.
#![forbid(unsafe_code)]

use cosplay::{
    meadow::{world::World, GlyphInstance, MeadowFrame},
    spatial::{
        app::{self, AppOptions, Application, View},
        Controls,
    },
};
use std::path::PathBuf;

const HELP: &str = "CosPlay Meadow - native 2D glyph world, live 3D character\n\nUsage: meadow [options]\n\nWASD / arrows move | Shift run | +/- or wheel zoom\nSpace pause | R reset | F1 toggle auto-walk | Esc quit\n\n--headless       Capture a GPU-rendered PNG without opening a window\n--frames N       Headless simulation steps (default 120); window frame limit if supplied\n--output PATH    Headless PNG destination (default meadow.png)\n--size WxH       Pixels (default 1280x800; 64..4096 by 64..2160)\n--seed N         Deterministic world seed (default 2024)\n--density X      Grass/flower density (0.25..2; default 1)\n--zoom X         Zoom (0.5..3; default 1)\n--auto           Start in auto-walk mode\n--licenses       Print embedded font license\n--help           Show this help\n\nHeadless capture still requires a supported GPU/graphics adapter.\n";

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

struct MeadowApp {
    world: World,
    seed: u32,
    density: f32,
}
impl Application for MeadowApp {
    type Frame = MeadowFrame;
    fn update(&mut self, dt: f32, controls: Controls) {
        self.world.step(dt, controls);
    }
    fn reset(&mut self) -> Result<(), String> {
        self.world = World::new(self.seed, self.density);
        Ok(())
    }
    fn frame(&self, view: View) -> Result<Self::Frame, String> {
        let mut frame = self.world.frame(view.aspect(), view.zoom);
        hud(
            &mut frame,
            view.width,
            view.height,
            view.paused,
            view.autoplay,
        );
        Ok(frame)
    }
}

fn run(opts: Options) -> Result<(), String> {
    let mut game = MeadowApp {
        world: World::new(opts.seed, opts.density),
        seed: opts.seed,
        density: opts.density,
    };
    let options = AppOptions {
        title: "CosPlay Meadow - 2D world / live 3D glyph knight".into(),
        size: opts.size,
        frames: opts.frames,
        zoom: opts.zoom,
        autoplay: opts.autoplay,
    };
    if opts.headless {
        let steps = opts.frames.unwrap_or(120);
        let report = app::capture(&mut game, &options, steps, &opts.output)?;
        println!(
            "meadow: adapter={} backend={:?} frames={steps} seed={} glyphs={} player={:?} output={}",
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
