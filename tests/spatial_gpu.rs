// SPDX-License-Identifier: Apache-2.0
//! Opt-in GPU contract tests through public APIs (no meadow World).
#![cfg(feature = "spatial")]
use cosplay::spatial::{
    AsciiGlyph, Camera2d, Depth, GlyphGroup, GpuRenderer, KnightPose, LinearRgba, Quad, SceneFrame,
    Transform2d,
};
use std::{fs::File, path::Path};

fn pixels(path: &Path) -> Vec<u8> {
    let mut reader = png::Decoder::new(File::open(path).unwrap())
        .read_info()
        .unwrap();
    let mut bytes = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut bytes).unwrap();
    bytes.truncate(info.buffer_size());
    bytes
}

#[test]
#[ignore = "requires native GPU; run with --ignored --nocapture"]
fn generic_scene_empty_optional_knight_group_rotation_and_occlusion() {
    let instance = wgpu::Instance::default();
    let mut renderer = pollster::block_on(GpuRenderer::new(&instance, None)).unwrap();
    eprintln!("spatial capture adapter: {:?}", renderer.adapter.get_info());
    let directory = std::env::temp_dir().join(format!("cosplay-spatial-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("scene.png");
    let mut frame = SceneFrame::new(
        Camera2d {
            center: [0.0; 2],
            view_height: 100.0,
        },
        LinearRgba::from_srgb8(30, 60, 90),
    );
    renderer.capture_scene_png(101, 100, &frame, &path).unwrap();
    let blank = pixels(&path);
    for pixel in blank.as_chunks::<4>().0 {
        for (actual, expected) in pixel.iter().zip([30u8, 60, 90, 255]) {
            assert!(
                actual.abs_diff(expected) <= 1,
                "empty frame must contain only configured background"
            );
        }
    }
    let mut plate = GlyphGroup::new();
    plate
        .push(Quad::solid([20.0; 2], LinearRgba::from_srgb8(0, 255, 0)))
        .unwrap();
    plate
        .push(
            Quad::glyph(AsciiGlyph::new(' ').unwrap(), [20.0; 2], LinearRgba::WHITE).with_order(1),
        )
        .unwrap();
    let mut farther = GlyphGroup::new();
    farther
        .push(Quad::solid([20.0; 2], LinearRgba::from_srgb8(255, 0, 0)))
        .unwrap();
    frame
        .add(&plate, Transform2d::default(), Depth::new(0.2).unwrap())
        .unwrap();
    frame
        .add(&farther, Transform2d::default(), Depth::new(0.8).unwrap())
        .unwrap();
    let mut beam = GlyphGroup::new();
    beam.push(Quad::solid(
        [30.0, 4.0],
        LinearRgba::from_srgb8(255, 255, 0),
    ))
    .unwrap();
    frame
        .add(
            &beam,
            Transform2d {
                rotation: std::f32::consts::FRAC_PI_2,
                ..Transform2d::at(-30.0, 0.0)
            },
            Depth::new(0.2).unwrap(),
        )
        .unwrap();
    let text = GlyphGroup::from_text(
        "#",
        [20.0, 30.0],
        LinearRgba::WHITE,
        Some(LinearRgba::BLACK),
    )
    .unwrap();
    frame
        .add(&text, Transform2d::at(30.0, 0.0), Depth::new(0.2).unwrap())
        .unwrap();
    renderer.capture_scene_png(101, 100, &frame, &path).unwrap();
    let composed = pixels(&path);
    let pixel = |x: usize, y: usize| &composed[(y * 101 + x) * 4..(y * 101 + x + 1) * 4];
    assert_eq!(
        pixel(50, 50),
        [0, 255, 0, 255],
        "farther red must not erase green; space is transparent"
    );
    assert_eq!(pixel(20, 40), [255, 255, 0, 255]);
    assert_eq!(
        pixel(30, 50),
        pixel(0, 0),
        "group rotation must turn the beam vertical"
    );
    let ink = (35..65)
        .flat_map(|y| (70..91).map(move |x| (x, y)))
        .filter(|&(x, y)| pixel(x, y)[0] > 240)
        .count();
    assert!(ink > 10, "text ink must be in front of its backing");
    renderer.capture_scene_png(101, 100, &frame, &path).unwrap();
    assert_eq!(pixels(&path), composed);

    // Toggle the specialized character on and off on the SAME renderer. There
    // must be no implicit character or stale texture/instance after removing it.
    frame.clear();
    frame.camera = Camera2d {
        center: [0.0, 70.0],
        view_height: 230.0,
    };
    frame.knight = Some(KnightPose::default());
    renderer.capture_scene_png(241, 240, &frame, &path).unwrap();
    let knight = pixels(&path);
    assert!(
        knight
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|p| p[0] > 100 && p[1] > 100 && p[2] > 100)
            .count()
            > 40
    );
    frame.knight = None;
    frame.camera = Camera2d {
        center: [0.0; 2],
        view_height: 100.0,
    };
    renderer.capture_scene_png(101, 100, &frame, &path).unwrap();
    assert_eq!(pixels(&path), blank);
    assert!(renderer.capture_scene_png(0, 100, &frame, &path).is_err());
    frame.camera.view_height = f32::NAN;
    assert!(renderer.capture_scene_png(101, 100, &frame, &path).is_err());
    assert_eq!(
        pixels(&path),
        blank,
        "invalid input must not overwrite the last capture"
    );
    std::fs::remove_dir_all(directory).unwrap();
}
