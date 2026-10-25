// SPDX-License-Identifier: Apache-2.0
//! Small reusable visual generators. No player, camera, collision, or global
//! lighting is implied. Grass bend and prop wind are explicit caller inputs.
//! Organic surface sampling and grass segmentation are shared with the meadow.

use super::{AsciiGlyph, Depth, GlyphGroup, LinearRgba, Quad, SceneError, SceneFrame, Transform2d};
use glam::{Vec2, Vec3};
use std::f32::consts::TAU;

pub(crate) const LETTERS: &[u8] = b".,ircvzxeoasunwqpdbhkgmQ8B";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BladeStyle {
    #[default]
    Straight,
    Slash,
    Backslash,
    CurledTip,
}

/// A segmented grass blade rooted at local [0, 0], growing toward +Y.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GrassBlade {
    pub segments: u8,
    pub segment_length: f32,
    pub lean: f32,
    pub curl: f32,
    pub style: BladeStyle,
}
impl Default for GrassBlade {
    fn default() -> Self {
        Self {
            segments: 3,
            segment_length: 7.0,
            lean: 0.0,
            curl: 0.15,
            style: BladeStyle::Straight,
        }
    }
}
impl GrassBlade {
    /// Bend is an angle-like input in radians; callers may combine wind,
    /// interaction, or animation before passing it in. No hidden clock/seed.
    pub fn group(self, bend: f32, color: LinearRgba) -> Result<GlyphGroup, SceneError> {
        if !(1..=32).contains(&self.segments)
            || !self.segment_length.is_finite()
            || self.segment_length <= 0.0
            || !self.lean.is_finite()
            || !self.curl.is_finite()
            || !bend.is_finite()
        {
            return Err(SceneError(
                "grass needs 1..=32 segments, positive length and finite angles",
            ));
        }
        let mut group = GlyphGroup::new();
        for segment in self.geometry(Vec2::ZERO, bend) {
            let [r, g, b, a] = color.channels();
            let shade = segment.fraction * 0.5 + 0.5;
            let mut quad = Quad::glyph(
                AsciiGlyph::new(char::from(segment.glyph))?,
                segment.size.to_array(),
                LinearRgba::new(r * shade, g * shade, b * shade, a)?,
            );
            quad.position = segment.position.to_array();
            quad.rotation = segment.rotation;
            group.push(quad)?;
        }
        Ok(group)
    }

    pub(crate) fn geometry(self, root: Vec2, bend: f32) -> impl Iterator<Item = BladeSegment> {
        let mut cursor = root;
        (0..self.segments).map(move |segment| {
            let fraction = (f32::from(segment) + 1.0) / f32::from(self.segments);
            let middle = (f32::from(segment) + 0.5) / f32::from(self.segments);
            let angle = self.lean + self.curl * middle.powf(1.6) + bend * fraction.powf(1.8);
            let vector = Vec2::new(angle.sin(), angle.cos()) * self.segment_length;
            let position = cursor + vector * 0.5;
            cursor += vector;
            let (glyph, inherent) = match self.style {
                BladeStyle::Straight => (b'|', 0.0),
                BladeStyle::Slash => (b'/', 0.44),
                BladeStyle::Backslash => (b'\\', -0.44),
                BladeStyle::CurledTip if segment + 1 == self.segments => {
                    (if self.curl > 0.0 { b')' } else { b'(' }, 0.0)
                }
                BladeStyle::CurledTip => (b'|', 0.0),
            };
            let height = self.segment_length * 1.18;
            BladeSegment {
                position,
                size: Vec2::new(height * 0.625 * 1.25, height),
                rotation: inherent - angle,
                fraction,
                glyph,
            }
        })
    }
}

/// Seeded placement in a rectangle centered on local [0, 0].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GrassPatchConfig {
    pub size: [f32; 2],
    pub count: usize,
    pub seed: u32,
    pub color: LinearRgba,
}
impl Default for GrassPatchConfig {
    fn default() -> Self {
        Self {
            size: [120.0, 60.0],
            count: 100,
            seed: 2024,
            color: LinearRgba::from_srgb8(110, 180, 145),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GrassPatch {
    color: LinearRgba,
    blades: Vec<([f32; 2], GrassBlade, f32)>,
}
impl GrassPatch {
    pub fn new(config: GrassPatchConfig) -> Result<Self, SceneError> {
        if config.count > 10_000 || !config.size.iter().all(|v| v.is_finite() && *v > 0.0) {
            return Err(SceneError(
                "grass patch needs positive finite size and at most 10000 blades",
            ));
        }
        let mut rng = Random::new(config.seed);
        let blades = (0..config.count)
            .map(|_| {
                let root = [
                    rng.signed() * config.size[0] * 0.5,
                    rng.signed() * config.size[1] * 0.5,
                ];
                let blade = GrassBlade {
                    segment_length: 5.0 + rng.next() * 6.0,
                    lean: rng.signed() * 0.3,
                    curl: rng.signed() * 0.4,
                    style: if rng.next() < 0.2 {
                        BladeStyle::CurledTip
                    } else {
                        BladeStyle::Straight
                    },
                    ..GrassBlade::default()
                };
                (root, blade, rng.next() * TAU)
            })
            .collect();
        Ok(Self {
            color: config.color,
            blades,
        })
    }

    /// Atomically append the patch, sorting each blade by its transformed root
    /// (not by its tip or the patch center). Time is seconds, wind is bend.
    pub fn add_to(
        &self,
        frame: &mut SceneFrame,
        transform: Transform2d,
        time: f32,
        wind: f32,
    ) -> Result<(), SceneError> {
        transform.validate()?;
        if !time.is_finite() || !wind.is_finite() {
            return Err(SceneError("grass time and wind must be finite"));
        }
        let mut addition = SceneFrame::default();
        for (root, blade, phase) in &self.blades {
            let position = transform.point(*root);
            let group = blade.group(wind + (time * 1.7 + phase).sin() * 0.12, self.color)?;
            addition.add(
                &group,
                Transform2d {
                    translation: position,
                    ..transform
                },
                Depth::ground(position[1])?,
            )?;
        }
        frame.instances.extend(addition.instances);
        Ok(())
    }
}

pub(crate) struct BladeSegment {
    pub position: Vec2,
    pub size: Vec2,
    pub rotation: f32,
    pub fraction: f32,
    pub glyph: u8,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OrganicKind {
    #[default]
    Tree,
    Rock,
}

/// Bounded procedural geometry. Construct once, then reuse the cached surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrganicConfig {
    pub kind: OrganicKind,
    pub radius: f32,
    pub height: f32,
    pub seed: u32,
    pub dark: LinearRgba,
    pub lit: LinearRgba,
}
impl Default for OrganicConfig {
    fn default() -> Self {
        Self {
            kind: OrganicKind::Tree,
            radius: 45.0,
            height: 130.0,
            seed: 2024,
            dark: LinearRgba::from_srgb8(25, 65, 55),
            lit: LinearRgba::from_srgb8(140, 210, 140),
        }
    }
}

/// A tree or boulder, independently placeable through a GlyphGroup transform.
/// Bounds come from the emitted group, so wind and canopy overhang are included.
#[derive(Clone, Debug, PartialEq)]
pub struct OrganicProp {
    config: OrganicConfig,
    cells: Vec<OrganicCell>,
}
impl OrganicProp {
    pub fn new(config: OrganicConfig) -> Result<Self, SceneError> {
        if !(1.0..=256.0).contains(&config.radius) || !(1.0..=1024.0).contains(&config.height) {
            return Err(SceneError(
                "organic radius must be 1..=256 and height 1..=1024",
            ));
        }
        let cells = organic_cells(
            config.radius,
            config.height,
            config.kind == OrganicKind::Rock,
            &mut Random::new(config.seed),
        );
        Ok(Self { config, cells })
    }

    /// Pure visual sampling at a time in seconds. Wind is supplied by the game;
    /// rocks ignore both inputs. Does not generate shadows or collision bodies.
    pub fn group(&self, time: f32, wind: f32) -> Result<GlyphGroup, SceneError> {
        if !time.is_finite() || !wind.is_finite() {
            return Err(SceneError("prop time and wind must be finite"));
        }
        let rock = self.config.kind == OrganicKind::Rock;
        let mut group = GlyphGroup::new();
        for cell in &self.cells {
            let fraction = (cell.local.y / self.config.height).clamp(0.0, 1.0);
            let dx = if rock {
                0.0
            } else {
                wind * 26.0 * fraction * fraction
                    + if cell.canopy {
                        (time * 1.9 + cell.local.x * 0.045 + cell.local.y * 0.03).sin()
                            * 1.4
                            * fraction
                    } else {
                        0.0
                    }
            };
            let position = cell.local + Vec2::new(dx, 0.0);
            let rustle = if cell.canopy && !rock {
                (time * 1.3 + cell.seed * 40.0).sin() * 0.05
            } else {
                0.0
            };
            let brightness = (cell.brightness + rustle).clamp(0.0, 0.999);
            let index = (brightness * LETTERS.len() as f32 + (cell.seed - 0.5) * 2.2)
                .floor()
                .clamp(0.0, (LETTERS.len() - 1) as f32) as usize;
            let (dark, lit) = if cell.canopy {
                (self.config.dark.channels(), self.config.lit.channels())
            } else {
                ([0.12, 0.07, 0.045, 1.0], [0.62, 0.42, 0.26, 1.0])
            };
            let mix = if cell.canopy {
                brightness.powf(0.9)
            } else {
                brightness
            };
            let color: [f32; 4] = std::array::from_fn(|i| dark[i] + (lit[i] - dark[i]) * mix);
            group.push(
                Quad::solid(
                    [5.1, 8.16],
                    LinearRgba::new(dark[0] * 0.45, dark[1] * 0.45, dark[2] * 0.45, dark[3])?,
                )
                .at(position.x, position.y)
                .with_order(1),
            )?;
            group.push(
                Quad::glyph(
                    AsciiGlyph::new(char::from(LETTERS[index]))?,
                    [5.6, 8.96],
                    LinearRgba::new(color[0], color[1], color[2], color[3])?,
                )
                .at(position.x, position.y)
                .with_order(2),
            )?;
        }
        Ok(group)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct OrganicCell {
    pub local: Vec2,
    pub brightness: f32,
    pub seed: f32,
    pub canopy: bool,
}
struct Blob {
    center: Vec2,
    radius: f32,
}

pub(crate) fn organic_cells(
    radius: f32,
    height: f32,
    rock: bool,
    rng: &mut Random,
) -> Vec<OrganicCell> {
    let cy = if rock {
        radius * 0.75
    } else {
        height - radius * 0.95
    };
    let mut blobs = vec![Blob {
        center: Vec2::new(0.0, cy),
        radius: radius * if rock { 0.85 } else { 0.75 },
    }];
    let count = if rock {
        3
    } else {
        7 + (rng.next() * 4.0) as usize
    };
    for _ in 0..count {
        let angle = rng.next() * TAU;
        let distance = radius * (0.35 + rng.next() * 0.45);
        blobs.push(Blob {
            center: Vec2::new(
                angle.cos() * distance * 1.15,
                cy + angle.sin() * distance * if rock { 0.4 } else { 0.8 },
            ),
            radius: radius * (0.35 + rng.next() * 0.3),
        });
    }
    let surface = |point: Vec2| {
        let height = blobs
            .iter()
            .map(|blob| {
                let squared = point.distance_squared(blob.center);
                if squared < blob.radius * blob.radius {
                    (blob.radius * blob.radius - squared).sqrt()
                } else {
                    blob.radius - squared.sqrt()
                }
            })
            .fold(f32::NEG_INFINITY, f32::max);
        let noise = (point.x * 0.11 + point.y * 0.07).sin()
            * (point.y * 0.13 - point.x * 0.05).sin()
            + 0.5 * (point.x * 0.31 - point.y * 0.27).sin();
        height + noise * radius * 0.09
    };
    let light = Vec3::new(-0.55, 0.62, 0.56).normalize();
    let trunk_width = radius * 0.11 + 4.0;
    let mut cells = Vec::new();
    for iy in 0..=((height + radius * 0.2) / 8.0) as usize {
        let y = iy as f32 * 8.0;
        for ix in 0..=(radius * 3.8 / 5.0) as usize {
            let x = -radius * 1.9 + ix as f32 * 5.0;
            let local = Vec2::new(x, y);
            let canopy = surface(local) > 0.0;
            let half_trunk = if rock || y > cy {
                0.0
            } else {
                trunk_width * (1.0 + 1.1 * (-y / 10.0).exp()) * (1.0 - 0.3 * y / cy.max(1.0))
            };
            let branch_y = cy * 0.55;
            let branch_k = (y - branch_y) / (cy - branch_y + 1.0);
            let branch = !rock
                && y >= branch_y
                && y <= cy + radius * 0.2
                && [-1.0, 1.0].iter().any(|sign| {
                    (x - sign * branch_k * radius * 0.45).abs()
                        < trunk_width * 0.6 * (1.0 - branch_k * 0.4)
                });
            if !canopy && x.abs() >= half_trunk && !branch {
                continue;
            }
            let brightness = if canopy {
                let nx = -(surface(local + Vec2::new(2.0, 0.0))
                    - surface(local - Vec2::new(2.0, 0.0)))
                    / 4.0;
                let ny = -(surface(local + Vec2::new(0.0, 2.0))
                    - surface(local - Vec2::new(0.0, 2.0)))
                    / 4.0;
                let ao = if rock {
                    1.0
                } else {
                    0.55 + 0.45 * ((y - (cy - radius)) / (radius * 1.4)).clamp(0.0, 1.0)
                };
                Vec3::new(nx, ny, 1.0)
                    .normalize()
                    .dot(light)
                    .max(0.0)
                    .powf(1.6)
                    * ao
            } else {
                let nx = (x / half_trunk.max(1.0)).clamp(-1.0, 1.0);
                (0.25 + 0.45 * (-nx * 0.8 + 0.3).max(0.0)) * if y > cy * 0.7 { 0.6 } else { 1.0 }
            };
            cells.push(OrganicCell {
                local,
                brightness: brightness.clamp(0.0, 0.999),
                seed: rng.next(),
                canopy,
            });
        }
    }
    cells
}

/// Mulberry32, matching the reference's wrapping integer arithmetic. Take 24
/// bits for f32 so rounding can never produce 1.0 and overrun palette indices.
pub(crate) struct Random(u32);
impl Random {
    pub(crate) fn new(seed: u32) -> Self {
        Self(seed)
    }
    pub(crate) fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x6d2b79f5);
        let mut t = (self.0 ^ (self.0 >> 15)).wrapping_mul(1 | self.0);
        t = t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t)) ^ t;
        ((t ^ (t >> 14)) >> 8) as f32 / 16_777_216.0
    }
    pub(crate) fn signed(&mut self) -> f32 {
        self.next() * 2.0 - 1.0
    }
}
