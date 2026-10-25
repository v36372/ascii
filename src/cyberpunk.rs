// SPDX-License-Identifier: Apache-2.0
//! Neon Ward: a deterministic street scene authored entirely with `spatial`.
//! No renderer changes, external art, terminal canvas, or implicit meadow knight.
use crate::spatial::{
    AsciiGlyph, Camera2d, Controls, Depth, GlyphGroup, LinearRgba, Quad, SceneError, SceneFrame,
    Transform2d,
};
use glam::Vec2;
use std::f32::consts::TAU;

type Result<T> = std::result::Result<T, SceneError>;
const LIMIT: f32 = 1500.0;
const CYAN: [u8; 3] = [80, 242, 255];
const PINK: [u8; 3] = [255, 65, 177];
const GOLD: [u8; 3] = [255, 191, 93];
const SIGNS: [&str; 10] = [
    "BYTE BAR",
    "NOODLE 24",
    "CYBERNET",
    "HOTEL // 09",
    "RAMEN",
    "AFTERLIFE",
    "SYNTH LAB",
    "NIGHT SHIFT",
    "ARCADE",
    "REPLICANT",
];

fn rgb(c: [u8; 3]) -> LinearRgba {
    LinearRgba::from_srgb8(c[0], c[1], c[2])
}
fn dim(c: [u8; 3], factor: f32) -> [u8; 3] {
    c.map(|v| (f32::from(v) * factor) as u8)
}
fn rect(g: &mut GlyphGroup, p: [f32; 2], size: [f32; 2], c: [u8; 3]) -> Result<()> {
    g.push(Quad::solid(size, rgb(c)).at(p[0], p[1]))
}
fn ink(g: &mut GlyphGroup, p: [f32; 2], size: [f32; 2], ch: char, c: [u8; 3]) -> Result<()> {
    g.push(Quad::glyph(AsciiGlyph::new(ch)?, size, rgb(c)).at(p[0], p[1]))
}
fn text(g: &mut GlyphGroup, p: [f32; 2], size: f32, s: &str, c: [u8; 3]) -> Result<()> {
    let art = GlyphGroup::from_text(s, [size * 0.62, size], rgb(c), None)?;
    g.extend(&art, Transform2d::at(p[0], p[1]))
}
fn line(g: &mut GlyphGroup, a: [f32; 2], b: [f32; 2], width: f32, c: [u8; 3]) -> Result<()> {
    let a = Vec2::from_array(a);
    let b = Vec2::from_array(b);
    let d = b - a;
    let mut quad =
        Quad::solid([d.length(), width], rgb(c)).at((a.x + b.x) * 0.5, (a.y + b.y) * 0.5);
    quad.rotation = d.y.atan2(d.x);
    g.push(quad)
}
// Opaque, nested dark-to-bright strokes give neon a halo without depending on
// order-independent transparency (the engine deliberately depth-writes alpha).
fn neon(g: &mut GlyphGroup, a: [f32; 2], b: [f32; 2], c: [u8; 3]) -> Result<()> {
    line(g, a, b, 7.0, dim(c, 0.12))?;
    line(g, a, b, 3.0, dim(c, 0.48))?;
    line(g, a, b, 1.0, c)
}
fn noise(seed: u32, n: u32) -> f32 {
    let mut v = seed.wrapping_add(n.wrapping_mul(0x9e3779b9));
    v = (v ^ (v >> 16)).wrapping_mul(0x7feb352d);
    v = (v ^ (v >> 15)).wrapping_mul(0x846ca68b);
    (v ^ (v >> 16)) as f32 / u32::MAX as f32
}

struct Block {
    x: f32,
    art: GlyphGroup,
    sign: GlyphGroup,
    color: [u8; 3],
}

/// Game-owned collision, movement, camera, and cached city assets.
pub struct City {
    seed: u32,
    time: f32,
    player: Vec2,
    velocity: Vec2,
    camera: Vec2,
    phase: f32,
    facing: f32,
    ground: GlyphGroup,
    skyline: [GlyphGroup; 2],
    blocks: Vec<Block>,
    lamp: GlyphGroup,
    drone: GlyphGroup,
}
impl City {
    pub fn new(seed: u32) -> Result<Self> {
        let ground = street(seed)?;
        let skyline = [skyline(seed, 0)?, skyline(seed, 1)?];
        let mut blocks = Vec::new();
        for (i, label) in SIGNS.iter().enumerate() {
            let color = [CYAN, PINK, GOLD][i % 3];
            let x = -1350.0 + i as f32 * 300.0;
            let (art, sign) = storefront(i, label, color)?;
            blocks.push(Block {
                x,
                art,
                sign,
                color,
            });
        }
        Ok(Self {
            seed,
            time: 0.0,
            player: Vec2::new(0.0, -100.0),
            velocity: Vec2::ZERO,
            camera: Vec2::new(0.0, 112.0),
            phase: 0.0,
            facing: 1.0,
            ground,
            skyline,
            blocks,
            lamp: lamp()?,
            drone: drone()?,
        })
    }
    pub fn player_position(&self) -> [f32; 2] {
        self.player.to_array()
    }
    /// Bounded steps, normalized diagonals, smooth acceleration and axis sliding.
    pub fn step(&mut self, dt: f32, controls: Controls) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let dt = dt.min(0.05);
        self.time += dt;
        let mut axis = Vec2::new(
            i32::from(controls.right) as f32 - i32::from(controls.left) as f32,
            i32::from(controls.up) as f32 - i32::from(controls.down) as f32,
        );
        if controls.autoplay && axis == Vec2::ZERO {
            axis = Vec2::new((self.time * 0.18).cos(), (self.time * 0.7).sin() * 0.35);
        }
        let speed = if controls.run { 155.0 } else { 88.0 };
        let target = axis.normalize_or_zero() * speed;
        self.velocity = self.velocity.lerp(target, 1.0 - (-14.0 * dt).exp());
        // Substeps keep a sprint from tunneling through posts or storefronts.
        for _ in 0..4 {
            for axis in 0..2 {
                let mut next = self.player;
                next[axis] += self.velocity[axis] * dt / 4.0;
                next.x = next.x.clamp(-LIMIT, LIMIT);
                next.y = next.y.clamp(-232.0, 112.0);
                if !self.blocked(next) {
                    self.player = next;
                } else {
                    self.velocity[axis] = 0.0;
                }
            }
        }
        if self.velocity.x.abs() > 2.0 {
            self.facing = self.velocity.x.signum();
        }
        self.phase = (self.phase + self.velocity.length() * dt * 0.12).rem_euclid(TAU);
        let target = Vec2::new(self.player.x, 112.0 + (self.player.y + 100.0) * 0.22);
        self.camera = self.camera.lerp(target, 1.0 - (-4.0 * dt).exp());
    }
    fn blocked(&self, p: Vec2) -> bool {
        self.blocks
            .iter()
            .any(|b| (p.x - b.x).abs() < 121.0 && p.y > 47.0)
            || (-5..=5).any(|i| {
                let x = i as f32 * 300.0 + 22.0;
                [30.0, -188.0]
                    .iter()
                    .any(|y| ((p - Vec2::new(x, *y)) * Vec2::new(1.0, 1.5)).length() < 12.0)
            })
    }

    /// Same CPU frame construction is used by native presentation and PNG capture.
    pub fn frame(&self, aspect: f32, zoom: f32) -> Result<SceneFrame> {
        let mut frame = SceneFrame::new(
            Camera2d {
                center: self.camera.to_array(),
                view_height: 700.0 / zoom,
            },
            rgb([7, 9, 24]),
        );
        let bounds = frame.camera.bounds(aspect)?;
        // A horizontal atmospheric gradient is geometry, not a post-process.
        let mut sky = GlyphGroup::new();
        for i in 0..30 {
            let t = i as f32 / 29.0;
            rect(
                &mut sky,
                [self.camera.x, 150.0 + i as f32 * 16.0],
                [frame.camera.view_height * aspect + 32.0, 16.5],
                [
                    (23.0 - t * 16.0) as u8,
                    (19.0 - t * 11.0) as u8,
                    (45.0 - t * 22.0) as u8,
                ],
            )?;
        }
        frame.add(&sky, Transform2d::default(), Depth::new(0.995)?)?;
        for (i, layer) in self.skyline.iter().enumerate() {
            frame.add(
                layer,
                Transform2d::at(self.camera.x * (0.78 - i as f32 * 0.17), 0.0),
                Depth::new(0.94 - i as f32 * 0.04)?,
            )?;
        }
        frame.add(
            &holograms(self.time)?,
            Transform2d::at(self.camera.x * 0.45, 0.0),
            Depth::new(0.875)?,
        )?;
        // Elevated maglev rail, scrolling train, and aerial traffic.
        let mut rail = GlyphGroup::new();
        rect(
            &mut rail,
            [self.camera.x, 220.0],
            [frame.camera.view_height * aspect + 50.0, 7.0],
            [29, 35, 57],
        )?;
        neon(
            &mut rail,
            [bounds.min[0], 225.0],
            [bounds.max[0], 225.0],
            dim(CYAN, 0.4),
        )?;
        let train_x = (self.time * 75.0).rem_euclid(2100.0) - 1050.0 + self.camera.x * 0.7;
        for i in 0..4 {
            let x = train_x + i as f32 * 68.0;
            rect(&mut rail, [x, 239.0], [65.0, 22.0], [27, 32, 56])?;
            for j in 0..6 {
                rect(
                    &mut rail,
                    [x - 25.0 + j as f32 * 10.0, 242.0],
                    [6.0, 6.0],
                    dim(GOLD, 0.65),
                )?;
            }
            line(&mut rail, [x - 31.0, 230.0], [x + 31.0, 230.0], 1.0, PINK)?;
        }
        frame.add(&rail, Transform2d::default(), Depth::new(0.86)?)?;
        frame.add(&self.ground, Transform2d::default(), Depth::new(0.97)?)?;
        for (i, block) in self.blocks.iter().enumerate() {
            if block.x + 150.0 < bounds.min[0] || block.x - 150.0 > bounds.max[0] {
                continue;
            }
            frame.add(
                &block.art,
                Transform2d::at(block.x, 65.0),
                Depth::ground(65.0)?,
            )?;
            // An occasional broken ballast, rather than strobing the whole scene.
            if i != 4 || (self.time * 8.0).sin() > -0.92 {
                frame.add(
                    &block.sign,
                    Transform2d::at(block.x, 65.0),
                    Depth::new(0.5029)?,
                )?;
            }
            let mut steam = GlyphGroup::new();
            for j in 0..10 {
                let age = (self.time * 0.22 + j as f32 * 0.1).fract();
                let x = block.x - 63.0 + (age * 7.0 + j as f32).sin() * 6.0;
                ink(
                    &mut steam,
                    [x, 65.0 + 43.0 + age * 48.0],
                    [8.0 + age * 9.0, 5.0 + age * 5.0],
                    '~',
                    dim([107, 129, 152], (1.0 - age) * 0.6),
                )?;
            }
            frame.add(&steam, Transform2d::default(), Depth::new(0.5028)?)?;
            let mut reflection = GlyphGroup::new();
            for j in 0..36 {
                let t = j as f32 / 36.0;
                let jitter = (self.time * 1.6 + j as f32 * 2.3).sin();
                let x = block.x + jitter * (3.0 + t * 5.0);
                let width = (85.0 * (1.0 - t) + jitter * 10.0).max(2.0);
                line(
                    &mut reflection,
                    [x - width * 0.5, 23.0 - t * 222.0],
                    [x + width * 0.5, 23.0 - t * 222.0],
                    1.0,
                    dim(block.color, (1.0 - t) * 0.22 + 0.035),
                )?;
            }
            frame.add(&reflection, Transform2d::default(), Depth::new(0.955)?)?;
        }
        let mut cables = GlyphGroup::new();
        for i in -5..5 {
            let x = i as f32 * 300.0;
            for j in 0..24 {
                let a = j as f32 / 24.0;
                let b = (j + 1) as f32 / 24.0;
                line(
                    &mut cables,
                    [
                        x + a * 300.0,
                        251.0 - (a * std::f32::consts::PI).sin() * 19.0,
                    ],
                    [
                        x + b * 300.0,
                        251.0 - (b * std::f32::consts::PI).sin() * 19.0,
                    ],
                    1.0,
                    [8, 12, 21],
                )?;
            }
        }
        frame.add(&cables, Transform2d::default(), Depth::new(0.502)?)?;
        for i in -5..=5 {
            let x = i as f32 * 300.0 + 22.0;
            if x < bounds.min[0] - 80.0 || x > bounds.max[0] + 80.0 {
                continue;
            }
            for y in [30.0, -188.0] {
                frame.add(&self.lamp, Transform2d::at(x, y), Depth::ground(y)?)?;
            }
        }
        // Distant people are grounded and occlude exactly like the player.
        for i in 0..7 {
            let x = (i as f32 * 421.0 + self.time * (12.0 + i as f32)).rem_euclid(2800.0) - 1400.0;
            let y = if i % 2 == 0 { 22.0 } else { -166.0 };
            let art = runner(self.time * 3.0 + i as f32, 1.0, true, [108, 103, 152])?;
            frame.add(
                &art,
                Transform2d {
                    scale: 0.82,
                    ..Transform2d::at(x, y)
                },
                Depth::ground(y)?,
            )?;
        }
        let mut player = runner(self.phase, self.facing, self.velocity.length() > 4.0, CYAN)?;
        // Keep the controllable runner identifiable among passing pedestrians.
        ink(&mut player, [0.0, 49.0], [7.0, 7.0], 'v', CYAN)?;
        frame.add(
            &player,
            Transform2d::at(self.player.x, self.player.y),
            Depth::ground(self.player.y)?,
        )?;
        for i in 0..3 {
            let x = (self.time * (25.0 + i as f32 * 8.0) + i as f32 * 417.0).rem_euclid(1800.0)
                - 900.0
                + self.camera.x * 0.6;
            let y = 310.0 + i as f32 * 63.0 + (self.time + i as f32).sin() * 7.0;
            frame.add(&self.drone, Transform2d::at(x, y), Depth::new(0.80)?)?;
        }
        let mut rain = GlyphGroup::new();
        let width = bounds.max[0] - bounds.min[0];
        let height = frame.camera.view_height;
        for i in 0..360 {
            let x = bounds.min[0]
                + (noise(self.seed, i * 3) * width - self.time * 35.0).rem_euclid(width);
            let y = bounds.min[1]
                + (noise(self.seed, i * 3 + 1) * height
                    - self.time * (300.0 + noise(self.seed, i * 3 + 2) * 130.0))
                    .rem_euclid(height);
            let mut q = Quad::glyph(AsciiGlyph::new('/')?, [3.0, 11.0], rgb([42, 62, 83])).at(x, y);
            q.rotation = -0.25;
            rain.push(q)?;
        }
        // Ripples on the asphalt, animated independently of the falling streaks.
        for i in 0..44 {
            let age = (self.time * 1.3 + noise(self.seed.wrapping_add(1), i)).fract();
            let x = noise(self.seed.wrapping_add(2), i) * 3000.0 - 1500.0;
            let y = noise(self.seed.wrapping_add(3), i) * 250.0 - 228.0;
            ink(
                &mut rain,
                [x, y],
                [3.0 + age * 11.0, 2.0 + age * 4.0],
                'o',
                [33, 52, 68],
            )?;
        }
        frame.add(&rain, Transform2d::default(), Depth::new(0.08)?)?;
        Ok(frame)
    }
}

fn holograms(time: f32) -> Result<GlyphGroup> {
    let mut g = GlyphGroup::new();
    for x in [-1140.0, -340.0, 460.0, 1260.0] {
        // Rooftop advertising rig with a sweeping, cutout scan line.
        rect(&mut g, [x, 330.0], [128.0, 96.0], [16, 13, 32])?;
        for y in [282.0, 378.0] {
            neon(&mut g, [x - 64.0, y], [x + 64.0, y], dim(PINK, 0.7))?;
        }
        neon(&mut g, [x - 64.0, 282.0], [x - 64.0, 378.0], dim(PINK, 0.7))?;
        text(
            &mut g,
            [x - 51.0, 361.0],
            9.0,
            "[ OMNI / CORP ]",
            [134, 92, 149],
        )?;
        text(&mut g, [x - 49.0, 338.0], 18.0, "UPGRADE", PINK)?;
        text(&mut g, [x - 49.0, 316.0], 18.0, "REALITY", PINK)?;
        text(
            &mut g,
            [x - 50.0, 296.0],
            8.0,
            "A BETTER YOU. V2.09",
            [153, 128, 171],
        )?;
        let scan = 284.0 + (time * 19.0).rem_euclid(92.0);
        line(
            &mut g,
            [x - 60.0, scan],
            [x + 60.0, scan],
            0.6,
            [92, 44, 91],
        )?;
        line(
            &mut g,
            [x - 40.0, 282.0],
            [x - 40.0, 243.0],
            3.0,
            [33, 30, 48],
        )?;
        line(
            &mut g,
            [x + 40.0, 282.0],
            [x + 40.0, 243.0],
            3.0,
            [33, 30, 48],
        )?;
        // A wireframe orbital logo, drawn as individually transformed glyphs.
        let center = Vec2::new(x + 330.0, 352.0);
        for i in 0..80 {
            let a = i as f32 / 80.0 * TAU;
            let p = center + Vec2::new(a.cos(), a.sin()) * 49.0;
            ink(&mut g, p.to_array(), [5.0, 6.0], '+', dim(CYAN, 0.65))?;
            let p = center + Vec2::new(a.cos() * 43.0, a.sin() * 14.0);
            ink(&mut g, p.to_array(), [4.0, 5.0], '.', dim(CYAN, 0.45))?;
            let p = center + Vec2::new(a.cos() * (time * 0.4).cos() * 18.0, a.sin() * 43.0);
            ink(&mut g, p.to_array(), [4.0, 5.0], ':', dim(CYAN, 0.45))?;
        }
        text(
            &mut g,
            [center.x - 37.0, center.y + 2.0],
            13.0,
            "OFF-WORLD",
            CYAN,
        )?;
        text(
            &mut g,
            [center.x - 37.0, center.y - 65.0],
            8.0,
            "YOUR NEXT LIFE AWAITS",
            [75, 119, 145],
        )?;
    }
    Ok(g)
}

fn skyline(seed: u32, layer: u32) -> Result<GlyphGroup> {
    let mut g = GlyphGroup::new();
    for i in 0..72 {
        let n = i * 17 + layer * 107;
        let x = i as f32 * 65.0 - 2340.0;
        let width = 38.0 + noise(seed, n) * 40.0;
        let height = 75.0 + noise(seed, n + 1) * if layer == 0 { 290.0 } else { 200.0 };
        let base = if layer == 0 { 190.0 } else { 145.0 };
        let color = if layer == 0 {
            [20, 23, 44]
        } else {
            [17, 25, 39]
        };
        rect(&mut g, [x, base + height * 0.5], [width, height], color)?;
        line(
            &mut g,
            [x - width * 0.5, base + height],
            [x - width * 0.5, base],
            1.0,
            [39, 42, 64],
        )?;
        if i % 3 == 0 {
            rect(
                &mut g,
                [x + 5.0, base + height + 8.0],
                [width * 0.4, 16.0],
                color,
            )?;
            line(
                &mut g,
                [x + 5.0, base + height + 16.0],
                [x + 5.0, base + height + 40.0],
                1.0,
                [51, 43, 66],
            )?;
            ink(
                &mut g,
                [x + 5.0, base + height + 41.0],
                [5.0, 5.0],
                '*',
                dim(PINK, 0.7),
            )?;
        }
        for row in 0..(height / 11.0) as u32 {
            for col in 0..(width / 9.0) as u32 {
                if noise(seed, n * 1000 + row * 31 + col) > 0.52 {
                    let c = if (col + i) % 5 == 0 { GOLD } else { CYAN };
                    ink(
                        &mut g,
                        [
                            x - width * 0.5 + 6.0 + col as f32 * 9.0,
                            base + 8.0 + row as f32 * 11.0,
                        ],
                        [4.0, 6.0],
                        if row % 4 == 0 { '=' } else { ':' },
                        dim(c, if layer == 0 { 0.25 } else { 0.37 }),
                    )?;
                }
            }
        }
        if i % 8 == 3 {
            neon(
                &mut g,
                [x + width * 0.3, base + 30.0],
                [x + width * 0.3, base + height - 10.0],
                dim(PINK, 0.6),
            )?;
        }
    }
    Ok(g)
}

fn street(seed: u32) -> Result<GlyphGroup> {
    let mut g = GlyphGroup::new();
    rect(&mut g, [0.0, -100.0], [3400.0, 330.0], [11, 18, 28])?;
    rect(&mut g, [0.0, 33.0], [3400.0, 55.0], [25, 32, 44])?;
    rect(&mut g, [0.0, -218.0], [3400.0, 65.0], [22, 28, 40])?;
    for y in [4.0, -184.0] {
        line(&mut g, [-1700.0, y], [1700.0, y], 3.0, [53, 67, 81])?;
        line(
            &mut g,
            [-1700.0, y - 3.0],
            [1700.0, y - 3.0],
            1.0,
            [8, 11, 19],
        )?;
    }
    for i in -50..=50 {
        let x = i as f32 * 34.0;
        line(&mut g, [x, 6.0], [x, 60.0], 0.7, [40, 46, 60])?;
        line(&mut g, [x, -245.0], [x, -187.0], 0.7, [35, 42, 56])?;
        if i % 3 == 0 {
            rect(&mut g, [x, -95.0], [35.0, 1.5], [98, 86, 58])?;
        }
    }
    for x in [-1200.0, -600.0, 0.0, 600.0, 1200.0] {
        for row in 0..9 {
            rect(
                &mut g,
                [x, -167.0 + row as f32 * 18.0],
                [42.0, 7.0],
                [64, 75, 85],
            )?;
        }
        text(&mut g, [x + 56.0, -135.0], 9.0, "SLOW", [59, 72, 88])?;
        for y in [-22.0, -160.0] {
            rect(&mut g, [x + 95.0, y], [28.0, 10.0], [7, 12, 19])?;
            for j in 0..7 {
                line(
                    &mut g,
                    [x + 84.0 + j as f32 * 3.5, y - 4.0],
                    [x + 84.0 + j as f32 * 3.5, y + 4.0],
                    1.0,
                    [38, 51, 65],
                )?;
            }
        }
    }
    for i in 0..2600 {
        let x = noise(seed, i * 3) * 3400.0 - 1700.0;
        let y = noise(seed, i * 3 + 1) * 300.0 - 240.0;
        ink(
            &mut g,
            [x, y],
            [3.0 + noise(seed, i * 3 + 2) * 6.0, 2.0],
            '-',
            [24, 35, 48],
        )?;
    }
    Ok(g)
}

fn storefront(i: usize, label: &str, c: [u8; 3]) -> Result<(GlyphGroup, GlyphGroup)> {
    let mut g = GlyphGroup::new();
    let h = 132.0 + (i % 3) as f32 * 24.0;
    // A solid silhouette hides skyline ink, with a raised side face and roof.
    rect(&mut g, [0.0, h * 0.5], [224.0, h], [19, 22, 34])?;
    rect(&mut g, [104.0, h * 0.5], [16.0, h], [12, 17, 28])?;
    rect(&mut g, [0.0, h], [236.0, 9.0], [42, 43, 60])?;
    for row in 0..(h / 9.0) as i32 {
        for col in 0..22 {
            ink(
                &mut g,
                [-106.0 + col as f32 * 10.0, 5.0 + row as f32 * 9.0],
                [6.0, 7.0],
                if (row + col) % 4 == 0 { '=' } else { '.' },
                [36, 39, 54],
            )?;
        }
    }
    // Upper windows, gutters, ducts, and rooftop equipment.
    for x in [-75.0, -25.0, 25.0, 75.0] {
        rect(&mut g, [x, h - 29.0], [32.0, 32.0], [7, 13, 23])?;
        for j in 0..4 {
            line(
                &mut g,
                [x - 13.0, h - 41.0 + j as f32 * 7.0],
                [x + 13.0, h - 41.0 + j as f32 * 7.0],
                2.0,
                dim(c, 0.25),
            )?;
        }
        line(&mut g, [x, h - 45.0], [x, h - 13.0], 2.0, [42, 48, 63])?;
    }
    line(&mut g, [99.0, 0.0], [99.0, h - 7.0], 3.0, [61, 60, 77])?;
    rect(&mut g, [-62.0, h + 12.0], [45.0, 18.0], [44, 47, 64])?;
    text(&mut g, [-78.0, h + 12.0], 9.0, "[###]", [91, 95, 115])?;
    line(&mut g, [63.0, h + 4.0], [63.0, h + 50.0], 1.0, [67, 64, 82])?;
    line(
        &mut g,
        [43.0, h + 35.0],
        [82.0, h + 35.0],
        1.0,
        [67, 64, 82],
    )?;
    // Lit shopfront behind a striped canopy.
    rect(&mut g, [0.0, 34.0], [199.0, 62.0], [7, 12, 23])?;
    for x in [-64.0, 6.0] {
        rect(&mut g, [x, 32.0], [58.0, 46.0], dim(c, 0.11))?;
        for j in 0..6 {
            line(
                &mut g,
                [x - 26.0, 14.0 + j as f32 * 7.0],
                [x + 26.0, 14.0 + j as f32 * 7.0],
                1.0,
                dim(c, 0.28),
            )?;
        }
        text(
            &mut g,
            [x - 20.0, 33.0],
            9.0,
            if i.is_multiple_of(2) {
                "[o_o]"
            } else {
                "//24h"
            },
            dim(c, 0.8),
        )?;
    }
    rect(&mut g, [72.0, 25.0], [32.0, 50.0], [15, 21, 34])?;
    neon(&mut g, [57.0, 2.0], [57.0, 49.0], dim(c, 0.6))?;
    ink(&mut g, [80.0, 23.0], [4.0, 5.0], '+', GOLD)?;
    for j in 0..20 {
        rect(
            &mut g,
            [-100.0 + j as f32 * 10.5, 66.0],
            [10.0, 13.0],
            if j % 2 == 0 {
                dim(c, 0.3)
            } else {
                [24, 25, 42]
            },
        )?;
    }
    line(&mut g, [-108.0, 59.0], [108.0, 59.0], 2.0, dim(c, 0.7))?;
    text(
        &mut g,
        [-92.0, 6.0],
        6.0,
        "NO SIGNAL / CASH ONLY",
        [90, 96, 115],
    )?;
    // Hanging vertical sign deliberately breaks the silhouette into the alley.
    rect(&mut g, [120.0, 110.0], [24.0, 88.0], [10, 15, 28])?;
    neon(&mut g, [107.0, 68.0], [107.0, 152.0], c)?;
    text(
        &mut g,
        [114.0, 140.0],
        14.0,
        if i.is_multiple_of(2) {
            "O\nP\nE\nN"
        } else {
            "2\n4\nH\nR"
        },
        c,
    )?;
    // Utility boxes and alley litter.
    rect(&mut g, [-104.0, 9.0], [16.0, 18.0], [41, 46, 57])?;
    text(&mut g, [-108.0, 11.0], 7.0, "!", GOLD)?;
    let mut sign = GlyphGroup::new();
    rect(&mut sign, [-7.0, 88.0], [192.0, 27.0], [9, 11, 23])?;
    neon(&mut sign, [-103.0, 103.0], [88.0, 103.0], c)?;
    neon(&mut sign, [-103.0, 73.0], [88.0, 73.0], c)?;
    let font = (172.0 / (label.len() as f32 * 0.62)).min(23.0);
    text(
        &mut sign,
        [-7.0 - (label.len() - 1) as f32 * font * 0.31, 88.0],
        font,
        label,
        c,
    )?;
    Ok((g, sign))
}

fn lamp() -> Result<GlyphGroup> {
    let mut g = GlyphGroup::new();
    rect(&mut g, [0.0, 2.0], [12.0, 5.0], [8, 14, 23])?;
    rect(&mut g, [0.0, 45.0], [3.0, 86.0], [49, 64, 82])?;
    line(&mut g, [0.0, 87.0], [30.0, 91.0], 3.0, [58, 66, 86])?;
    neon(&mut g, [15.0, 89.0], [35.0, 89.0], GOLD)?;
    rect(&mut g, [0.0, 52.0], [10.0, 15.0], [21, 30, 46])?;
    text(&mut g, [-2.0, 54.0], 8.0, "+", CYAN)?;
    Ok(g)
}
fn drone() -> Result<GlyphGroup> {
    let mut g = GlyphGroup::new();
    text(&mut g, [-18.0, 0.0], 10.0, "--[==]--", [93, 112, 137])?;
    line(&mut g, [-13.0, -5.0], [-5.0, -5.0], 1.0, PINK)?;
    line(&mut g, [13.0, -5.0], [21.0, -5.0], 1.0, CYAN)?;
    Ok(g)
}
fn runner(phase: f32, facing: f32, moving: bool, c: [u8; 3]) -> Result<GlyphGroup> {
    let mut g = GlyphGroup::new();
    let stride = if moving { phase.sin() * 4.0 } else { 0.0 };
    let bob = if moving {
        (phase * 2.0).cos().abs() * 1.5
    } else {
        0.0
    };
    // Ink-based, articulated silhouette; the coat and helmet carry neon edging.
    ink(&mut g, [0.0, 0.0], [24.0, 6.0], '=', [5, 9, 16])?;
    for (x, step) in [(-4.0, stride), (4.0, -stride)] {
        line(&mut g, [x, 13.0 + bob], [x + step, 3.0], 4.0, [43, 47, 67])?;
        rect(
            &mut g,
            [x + step + facing * 1.5, 2.0],
            [7.0, 3.0],
            [117, 128, 153],
        )?;
    }
    rect(&mut g, [0.0, 20.0 + bob], [15.0, 18.0], [15, 24, 40])?;
    text(&mut g, [-6.0, 24.0 + bob], 10.0, "##", [72, 80, 107])?;
    line(
        &mut g,
        [-7.0, 28.0 + bob],
        [-9.0 - stride * 0.4, 15.0 + bob],
        3.0,
        [86, 82, 114],
    )?;
    line(
        &mut g,
        [7.0, 28.0 + bob],
        [9.0 + stride * 0.4, 15.0 + bob],
        3.0,
        [86, 82, 114],
    )?;
    line(&mut g, [-7.0, 12.0 + bob], [7.0, 12.0 + bob], 1.0, c)?;
    line(
        &mut g,
        [facing * 5.0, 27.0 + bob],
        [facing * 5.0, 14.0 + bob],
        1.0,
        dim(c, 0.7),
    )?;
    rect(&mut g, [0.0, 34.0 + bob], [11.0, 10.0], [34, 43, 63])?;
    text(&mut g, [-5.0, 37.0 + bob], 10.0, "_", [126, 139, 165])?;
    neon(
        &mut g,
        [-3.0 + facing, 34.0 + bob],
        [4.0 + facing, 34.0 + bob],
        c,
    )?;
    // Wind-blown scarf.
    line(
        &mut g,
        [-facing * 5.0, 29.0 + bob],
        [-facing * (15.0 + stride * 0.5), 26.0 + bob],
        2.0,
        PINK,
    )?;
    Ok(g)
}

/// Pixel-anchored HUD composed through the same public scene API.
pub fn hud(
    frame: &mut SceneFrame,
    width: u32,
    height: u32,
    paused: bool,
    autoplay: bool,
) -> Result<()> {
    let u = frame.camera.view_height / height as f32;
    let left =
        frame.camera.center[0] - frame.camera.view_height * width as f32 / height as f32 * 0.5;
    let top = frame.camera.center[1] + frame.camera.view_height * 0.5;
    let mut g = GlyphGroup::new();
    rect(&mut g, [164.0, -51.0], [292.0, 68.0], [7, 12, 25])?;
    rect(&mut g, [19.0, -51.0], [2.0, 68.0], CYAN)?;
    text(
        &mut g,
        [33.0, -30.0],
        10.0,
        "COSPLAY  /  SECTOR 07",
        [104, 137, 165],
    )?;
    text(&mut g, [32.0, -53.0], 25.0, "NEON WARD", CYAN)?;
    text(
        &mut g,
        [33.0, -73.0],
        10.0,
        "02:17 AM   //   RAIN PROTOCOL",
        [169, 111, 160],
    )?;
    let bottom = -(height as f32) + 27.0;
    rect(
        &mut g,
        [width as f32 * 0.5, bottom],
        [width as f32, 36.0],
        [7, 12, 24],
    )?;
    let status = if paused {
        "PAUSED // SPACE TO RESUME"
    } else if autoplay {
        "AUTO-WALK // F1 FOR MANUAL"
    } else {
        "WASD / ARROWS : MOVE   SHIFT : SPRINT"
    };
    text(&mut g, [22.0, bottom + 1.0], 12.0, status, CYAN)?;
    if width >= 950 {
        text(
            &mut g,
            [width as f32 - 479.0, bottom + 1.0],
            11.0,
            "WHEEL +/- ZOOM  SPACE PAUSE  R RESET  ESC EXIT",
            [123, 142, 167],
        )?;
    }
    frame.add(
        &g,
        Transform2d {
            translation: [left, top],
            scale: u,
            rotation: 0.0,
        },
        Depth::new(0.002)?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn advance(city: &mut City, n: u32, controls: Controls) {
        for _ in 0..n {
            city.step(1.0 / 60.0, controls);
        }
    }
    #[test]
    fn deterministic_scene_animates_without_an_implicit_knight() {
        let mut a = City::new(7).unwrap();
        let mut b = City::new(7).unwrap();
        let first = a.frame(1.6, 1.0).unwrap();
        assert_eq!(first, b.frame(1.6, 1.0).unwrap());
        assert!(first.knight.is_none());
        assert!(first.instances().len() > 10000);
        let controls = Controls {
            right: true,
            up: true,
            ..Default::default()
        };
        advance(&mut a, 60, controls);
        advance(&mut b, 60, controls);
        assert_eq!(a.frame(1.6, 1.0).unwrap(), b.frame(1.6, 1.0).unwrap());
        assert_ne!(first, a.frame(1.6, 1.0).unwrap());
        assert!(a.player.x > 40.0 && a.player.y > -60.0);
        assert_ne!(first, City::new(8).unwrap().frame(1.6, 1.0).unwrap());
    }
    #[test]
    fn movement_is_normalized_running_is_faster_and_stopping_is_smooth() {
        let mut straight = City::new(1).unwrap();
        let mut diagonal = City::new(1).unwrap();
        let start = straight.player;
        let right = Controls {
            right: true,
            ..Default::default()
        };
        advance(&mut straight, 60, right);
        advance(
            &mut diagonal,
            60,
            Controls {
                down: true,
                ..right
            },
        );
        assert!(
            ((straight.player - start).length() - (diagonal.player - start).length()).abs() < 0.01
        );
        let mut running = City::new(1).unwrap();
        advance(&mut running, 60, Controls { run: true, ..right });
        assert!(running.player.x > straight.player.x * 1.6);
        advance(&mut running, 120, Controls::default());
        assert!(running.velocity.length() < 0.01);
    }
    #[test]
    fn buildings_posts_and_world_edges_block_but_alleys_are_open() {
        let mut c = City::new(1).unwrap();
        c.player = Vec2::new(150.0, 20.0);
        let up = Controls {
            up: true,
            run: true,
            ..Default::default()
        };
        advance(&mut c, 180, up);
        assert!(c.player.y <= 47.0);
        c.player = Vec2::new(-15.0, 20.0);
        c.velocity = Vec2::ZERO;
        advance(&mut c, 180, up);
        assert_eq!(c.player.y, 112.0);
        c.player = Vec2::new(22.0, -220.0);
        c.velocity = Vec2::ZERO;
        advance(&mut c, 60, up);
        assert!(c.player.y < -188.0);
        c.player = Vec2::new(LIMIT - 1.0, -100.0);
        advance(
            &mut c,
            120,
            Controls {
                right: true,
                run: true,
                ..Default::default()
            },
        );
        assert_eq!(c.player.x, LIMIT);
    }
    #[test]
    fn invalid_time_cannot_poison_state_and_camera_hud_are_checked() {
        let mut c = City::new(1).unwrap();
        let before = c.frame(1.6, 1.0).unwrap();
        for dt in [f32::NAN, f32::INFINITY, -1.0, 0.0] {
            c.step(dt, Controls::default());
        }
        assert_eq!(before, c.frame(1.6, 1.0).unwrap());
        assert!(c.frame(f32::NAN, 1.0).is_err());
        assert!(c.frame(1.6, 0.0).is_err());
        assert!(City::new(u32::MAX).unwrap().frame(1.6, 1.0).is_ok());
        let mut frame = c.frame(1.6, 1.0).unwrap();
        hud(&mut frame, 1280, 800, false, false).unwrap();
        assert!(frame.instances().len() > before.instances().len());
    }
}
