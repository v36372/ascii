// SPDX-License-Identifier: Apache-2.0
//! Composed terminal showcase: FIGlet, images, animation, particles, shaders,
//! widgets, layout, cache, timers, messages and audio-independent rendering.
//!
//! Run with `cargo run --example terminal_showcase`.

use cosplay::{
    animation::Animation,
    engine::{Engine, Scene},
    font::{FigletFont, SystemFont},
    graphics::{Color, Pixel},
    image::Image,
    layout::{Layout, LayoutSpec},
    particles::{ConfettiConfig, ConfettiEmitter, ParticleEmitter, ParticleSystem},
    shader::{ShadedObject, ShimmerShader},
    sprites::{AnimationSprite, ImageSprite, LabelSprite, ParticleSprite},
    ui::{LayoutSprite, ListBox, TextInput, TitlePanel},
};
use std::{sync::Arc, time::Duration};

fn spinner_frames() -> Vec<Arc<Image>> {
    ["-", "\\", "|", "/"]
        .into_iter()
        .map(|ch| {
            Arc::new(Image::new(
                3,
                2,
                Pixel::new(ch.chars().next().unwrap(), Color::rgb(110, 240, 180), None),
            ))
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const WIDTH: u16 = 80;
    const HEIGHT: u16 = 26;
    let background = Pixel::new(' ', Color::WHITE, Some(Color::rgb(8, 14, 13)));
    let mut engine = Engine::new(WIDTH, HEIGHT);
    let mut scene = Scene::new("showcase", background);

    // A panel and linked layout rules demonstrate the UI layer.
    let mut layout = Layout::new();
    layout.insert("panel", LayoutSpec::at(1, 1));
    layout.insert("menu", LayoutSpec::inside_right("panel").with_offset(-27));
    layout.insert("input", LayoutSpec::at(2, 21));
    let mut chrome = LayoutSprite::new(WIDTH, HEIGHT, layout).with_z(1);
    chrome.add(
        TitlePanel::new("panel", 78, 24, "COSPLAY RUST - FUNCTIONAL RENDER SHOWCASE").with_colors(
            Pixel::new(' ', Color::WHITE, Some(Color::rgb(10, 22, 20))),
            Color::rgb(90, 210, 170),
        ),
    );
    let mut menu = ListBox::new("menu", 24, 6);
    for (index, item) in [
        "Glyph canvas",
        "Images + XP",
        "FIGlet fonts",
        "Animation",
        "Particles",
        "Shaders",
    ]
    .iter()
    .enumerate()
    {
        menu.add_item(*item, index);
    }
    menu.set_selection(0);
    chrome.add(menu);
    chrome.add(TextInput::new("input", 48).with_placeholder("type here..."));
    scene.add_object(chrome);
    scene.add_object(LabelSprite::new(
        Arc::new(SystemFont::new()),
        "WASD move | Enter submit | Esc cancel",
        Color::rgb(110, 150, 140),
        None,
        3,
        HEIGHT as i32 - 4,
        8,
    )?);

    // FIGlet rendering uses the bundled 277-font collection.
    let font = FigletFont::from_flf(
        cosplay::font::builtin_font("Small").ok_or("bundled FIGlet font is missing")?,
    )?;
    let banner = font.render("GLYPHS", Color::rgb(255, 190, 90), None)?;
    scene.add_object(ImageSprite::new(banner, 5, 3, 4));

    // Image + shimmer shader composition.
    let card = Image::from_text(
        "Canvas: RGB + Z + clip\nImage: crop/split/REXPaint\nFont: system + FIGlet",
        Color::rgb(160, 230, 210),
        None,
        false,
    );
    let card = ShadedObject::with_shader(
        ImageSprite::new(card, 5, 11, 4),
        ShimmerShader::new(
            2024,
            240,
            vec![Color::rgb(255, 220, 120), Color::rgb(120, 240, 200)],
        ),
    );
    scene.add_object(card);

    // Animation and particles run through the same scene loop.
    let animation = Animation::film_strip(
        "spinner",
        Duration::from_millis(100),
        true,
        false,
        spinner_frames(),
    )?;
    scene.add_object(AnimationSprite::single(animation, 48, 9, 5));

    let mut particles = ParticleSystem::new();
    let mut emitter = ConfettiEmitter::fixed_with(
        "confetti",
        ConfettiConfig {
            count_per_burst: 14,
            burst_interval_ms: 120,
            seed: 7,
            z: 5,
            ..ConfettiConfig::default()
        },
        46.0,
        14.0,
    );
    emitter.resume(true);
    particles.add_emitter(Box::new(emitter));
    scene.add_object(ParticleSprite::new(particles, 0, 0, 5));

    // Runtime services are usable from ordinary scene objects.
    scene.add_object(cosplay::sprites::OffScreenSprite::new(|ctx| {
        if ctx.frame.frame == 0 {
            ctx.cache_put("showcase.started", true);
            ctx.run_later(Duration::from_millis(20), || {});
        }
    }));

    engine.add_scene(scene)?;
    engine.start("showcase")?;
    for _ in 0..90 {
        engine.tick(Duration::from_millis(33), &[])?;
    }

    println!("{}", engine.screen().snapshot());
    assert!(engine.game_cache().contains("showcase.started"));
    Ok(())
}
