// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "meadow")]

use std::process::Command;

#[test]
fn help_licenses_and_invalid_options_do_not_initialize_graphics() {
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_meadow"))
            .args(args)
            .output()
            .unwrap()
    };
    let help = run(&["--help"]);
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("live 3D character"));
    let license = run(&["--licenses"]);
    assert!(license.status.success());
    assert!(String::from_utf8_lossy(&license.stdout).contains("SIL OPEN FONT LICENSE"));
    for args in [
        vec!["--density", "NaN"],
        vec!["--zoom", "0"],
        vec!["--size", "99999x99999"],
        vec!["--output", "x.png"],
        vec!["--frames", "-1"],
    ] {
        let result = run(&args);
        assert!(!result.status.success(), "{args:?}");
        assert!(String::from_utf8_lossy(&result.stderr).starts_with("meadow:"));
    }
}

/// Opt-in because a native graphics adapter is required even without a window.
#[test]
#[ignore = "requires a graphics adapter; run with --ignored --nocapture"]
fn gpu_png_is_nonblank_deterministic_and_animates() {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };
    let dir = std::env::temp_dir().join(format!(
        "cosplay-meadow-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&dir).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(dir.clone());
    let capture = |name: &str, frames: &str| {
        let path = dir.join(name);
        let result = Command::new(env!("CARGO_BIN_EXE_meadow"))
            .args([
                "--headless",
                "--auto",
                "--frames",
                frames,
                "--size",
                "640x400",
                "--output",
            ])
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        println!("{}", String::from_utf8_lossy(&result.stdout));
        let decoder = png::Decoder::new(fs::File::open(path).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let mut bytes = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut bytes).unwrap();
        assert_eq!((info.width, info.height), (640, 400));
        assert_eq!(info.color_type, png::ColorType::Rgba);
        bytes.truncate(info.buffer_size());
        bytes
    };
    let initial = capture("initial.png", "0");
    let repeat = capture("repeat.png", "0");
    assert_eq!(
        initial, repeat,
        "fixed seed/frame must reproduce on the same adapter"
    );
    let advanced = capture("advanced.png", "60");
    assert_ne!(
        initial, advanced,
        "movement/wind/character animation must change the image"
    );
    // Check the world below the HUD, not just the overlay text.
    let world_pixels = &initial[640 * 100 * 4..];
    let lit = world_pixels
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|p| p[0].max(p[1]).max(p[2]) > 45)
        .count();
    assert!(
        lit > 5000,
        "world is blank or almost blank: {lit} lit pixels"
    );
    let colors: std::collections::HashSet<_> = world_pixels
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| [p[0], p[1], p[2]])
        .collect();
    assert!(
        colors.len() > 100,
        "expected shaded colored meadow, got {} colors",
        colors.len()
    );
}
