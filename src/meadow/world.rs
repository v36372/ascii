// SPDX-License-Identifier: Apache-2.0
//! Deterministic, device-independent meadow simulation and glyph generation.
//! Grass/flowers repeat in spatially indexed tiles; trees cache their expensive
//! height-field sampling. All occlusion uses root/ground y, never glyph-tip y.

use super::{depth, Controls, GlyphInstance, HeroPose, MeadowFrame};
use crate::spatial::prefabs::{
    organic_cells, BladeStyle, GrassBlade, OrganicCell, Random, LETTERS,
};
use glam::{Vec2, Vec3};
use std::f32::consts::{PI, TAU};

const TILE: Vec2 = Vec2::new(1800.0, 1100.0);
const BUCKET_SIZE: f32 = 100.0;
const COLUMNS: i32 = 18;
const ROWS: i32 = 11;
const GRASS_MARGIN: f32 = 85.0;
const WORLD_LIMIT: f32 = 2800.0;
const WALK_SPEED: f32 = 115.0;
const RUN_SPEED: f32 = 230.0;
const PALETTES: [([f32; 3], [f32; 3]); 4] = [
    ([0.02, 0.08, 0.06], [0.34, 0.62, 0.27]),
    ([0.16, 0.025, 0.035], [0.82, 0.22, 0.17]),
    ([0.02, 0.07, 0.07], [0.20, 0.55, 0.45]),
    ([0.05, 0.05, 0.07], [0.58, 0.60, 0.68]),
];

pub struct World {
    buckets: Vec<Bucket>,
    trees: Vec<Tree>,
    player: Vec2,
    velocity: Vec2,
    camera: Vec2,
    yaw: f32,
    phase: f32,
    movement: f32,
    time: f32,
}

impl World {
    /// Density scales the number of clumps and flowers, not glyph size. Zero
    /// disables ground vegetation; nonfinite density uses the rich default.
    /// The upper bound prevents accidental unbounded procedural allocations.
    pub fn new(seed: u32, density: f32) -> Self {
        let density = finite_or(density, 1.0).clamp(0.0, 3.0);
        let mut buckets: Vec<_> = (0..COLUMNS * ROWS).map(|_| Bucket::default()).collect();
        populate_grass(&mut buckets, seed ^ 2024 ^ 4242, density);
        populate_flowers(&mut buckets, seed ^ 2024 ^ 777, density);
        let mut rng = Random::new(seed ^ 2024 ^ 31337);
        let trees = layout_trees(seed)
            .into_iter()
            .map(|spot| Tree::new(spot, &mut rng))
            .collect();
        Self {
            buckets,
            trees,
            player: Vec2::ZERO,
            velocity: Vec2::ZERO,
            camera: Vec2::new(0.0, 40.0),
            yaw: 0.0,
            phase: 0.0,
            movement: 0.0,
            time: 0.0,
        }
    }

    pub fn player_position(&self) -> [f32; 2] {
        self.player.to_array()
    }

    /// Integrate bounded real time. Autoplay only supplies direction when there
    /// is no manual direction; manual run remains available in either mode.
    pub fn step(&mut self, delta_seconds: f32, input: Controls) {
        let dt = finite_or(delta_seconds, 0.0).clamp(0.0, 0.05);
        if dt == 0.0 {
            return;
        }
        self.time += dt;
        let mut direction = Vec2::new(
            f32::from(input.right) - f32::from(input.left),
            f32::from(input.up) - f32::from(input.down),
        );
        if direction == Vec2::ZERO && input.autoplay {
            direction = Vec2::new((self.time * 0.4).cos(), (self.time * 0.55).sin() * 0.7);
        }
        let speed = if input.run { RUN_SPEED } else { WALK_SPEED };
        let desired = direction.normalize_or_zero() * speed;
        self.velocity = self.velocity.lerp(desired, 1.0 - (-dt * 10.0).exp());
        let previous = self.player;
        // Small steps avoid jumping across narrow trunks. Project velocity onto
        // the contact tangent so holding a diagonal slides along obstacles.
        let steps = (self.velocity.length() * dt / 3.0).ceil().max(1.0) as usize;
        for _ in 0..steps {
            self.player += self.velocity * (dt / steps as f32);
            for tree in &self.trees {
                resolve_collision(&mut self.player, &mut self.velocity, tree.spot);
            }
            let bounded = self
                .player
                .clamp(Vec2::splat(-WORLD_LIMIT), Vec2::splat(WORLD_LIMIT));
            if bounded.x != self.player.x {
                self.velocity.x = 0.0;
            }
            if bounded.y != self.player.y {
                self.velocity.y = 0.0;
            }
            self.player = bounded;
        }
        let travelled = self.player.distance(previous);
        self.movement = (travelled / dt / WALK_SPEED).min(1.3);
        self.phase = (self.phase + travelled * 0.075).rem_euclid(TAU);
        if self.velocity.length() > 8.0 {
            // Model +z faces the viewer, which is screen/world -y.
            let target = self.velocity.x.atan2(-self.velocity.y);
            let difference = wrap_angle(target - self.yaw);
            self.yaw = wrap_angle(self.yaw + difference * (1.0 - (-dt * 12.0).exp()));
        }
        self.camera = self
            .camera
            .lerp(self.player + Vec2::new(0.0, 40.0), 1.0 - (-dt * 4.0).exp());
    }

    /// Generate only visible glyphs. Aspect/zoom are bounded defensively for
    /// minimized windows and malformed external callers (base height = 640).
    /// Multiple tile copies cover even ultrawide/zoomed-out views without seams.
    pub fn frame(&self, aspect: f32, zoom: f32) -> MeadowFrame {
        // Cover the app's entire legal 64..4096 by 64..2160 size range.
        // Culling with a narrower aspect than the GPU projection leaves blank
        // side bands on very wide windows, even though the tiles are endless.
        let aspect = finite_or(aspect, 1.6).clamp(1.0 / 64.0, 64.0);
        let zoom = finite_or(zoom, 1.0).clamp(0.25, 4.0);
        let view_height = 640.0 / zoom;
        let half = Vec2::new(view_height * aspect, view_height) * 0.5;
        let view = Bounds {
            min: self.camera - half,
            max: self.camera + half,
        };
        let roots = view.expanded(GRASS_MARGIN);
        let shadows: Vec<_> = self
            .trees
            .iter()
            .filter(|tree| tree.spot.shadow_bounds().intersects(roots))
            .map(|tree| tree.spot)
            .collect();
        let mut instances = Vec::new();
        let first_x = (roots.min.x / BUCKET_SIZE).floor() as i32;
        let last_x = (roots.max.x / BUCKET_SIZE).floor() as i32;
        let first_y = (roots.min.y / BUCKET_SIZE).floor() as i32;
        let last_y = (roots.max.y / BUCKET_SIZE).floor() as i32;
        for by in first_y..=last_y {
            for bx in first_x..=last_x {
                let (index, offset) = tile_bucket(bx, by);
                let bucket = &self.buckets[index];
                for blade in &bucket.blades {
                    let root = blade.root + offset;
                    if roots.contains(root) {
                        self.draw_blade(blade, root, view, &shadows, &mut instances);
                    }
                }
                for flower in &bucket.flowers {
                    let root = flower.root + offset;
                    if roots.contains(root) {
                        self.draw_flower(flower, root, view, &shadows, &mut instances);
                    }
                }
            }
        }
        for tree in &self.trees {
            if tree.bounds.intersects(view) {
                self.draw_tree(tree, view, &mut instances);
            }
        }
        MeadowFrame {
            camera: self.camera.to_array(),
            view_height,
            time: self.time,
            instances,
            hero: HeroPose {
                position: self.player.to_array(),
                yaw: self.yaw,
                phase: self.phase,
                movement: self.movement,
                time: self.time,
            },
        }
    }

    fn draw_blade(
        &self,
        blade: &Blade,
        root: Vec2,
        view: Bounds,
        shadows: &[Spot],
        out: &mut Vec<GlyphInstance>,
    ) {
        let gust = gust_band(root, self.time);
        let wind =
            wind_at(root, self.time, gust) + (self.time * 2.7 + blade.seed * 31.0).sin() * 0.06;
        let bend = wind * blade.flex + player_push(root, self.player, blade.seed);
        let patch = 1.0 + noise(root * 0.0042) * 0.28;
        let hue = noise(root * 0.0023 + Vec2::new(17.0, 3.0)) * 0.5 + 0.5;
        let tint = Vec3::new(0.85, 1.0, 1.1).lerp(Vec3::new(1.08, 1.04, 0.85), hue);
        let shade = 1.0 - shadow_at(root, self.player, shadows) * 0.82;
        let color = blade.color
            * tint
            * patch
            * (gust * 0.6 + 0.82)
            * shade
            * light_pool(root, self.player);
        let shape = GrassBlade {
            segments: blade.segments,
            segment_length: blade.length,
            lean: blade.lean,
            curl: blade.curl,
            style: match blade.style {
                0..=6 => BladeStyle::Straight,
                7 => BladeStyle::Slash,
                8 => BladeStyle::Backslash,
                _ => BladeStyle::CurledTip,
            },
        };
        for segment in shape.geometry(root, bend) {
            if view.expanded(segment.size.y).contains(segment.position) {
                out.push(glyph_instance(
                    segment.position,
                    segment.size,
                    color * (segment.fraction * 0.5 + 0.5),
                    segment.rotation,
                    depth(root.y),
                    segment.glyph,
                ));
            }
        }
    }

    fn draw_flower(
        &self,
        flower: &Flower,
        root: Vec2,
        view: Bounds,
        shadows: &[Spot],
        out: &mut Vec<GlyphInstance>,
    ) {
        let wind = wind_at(root, self.time, gust_band(root, self.time))
            + player_push(root, self.player, 0.5) * 0.65;
        let position = root + Vec2::new(wind.sin(), wind.cos()) * flower.height;
        if !view.expanded(flower.size).contains(position) {
            return;
        }
        let color = flower.color
            * (1.0 - shadow_at(root, self.player, shadows) * 0.7)
            * (light_pool(root, self.player) * 0.5 + 0.5);
        out.push(glyph_instance(
            position,
            Vec2::new(flower.size * 0.625, flower.size),
            color,
            0.0,
            depth(root.y) - 0.000008,
            flower.glyph,
        ));
    }

    fn draw_tree(&self, tree: &Tree, view: Bounds, out: &mut Vec<GlyphInstance>) {
        let spot = tree.spot;
        let gust = gust_band(spot.position, self.time);
        let wind = wind_at(spot.position, self.time, gust);
        for cell in &tree.cells {
            let fraction = (cell.local.y / spot.height).clamp(0.0, 1.0);
            let ripple = (self.time * 1.9 + cell.local.x * 0.045 + cell.local.y * 0.03).sin()
                * 1.4
                * fraction;
            let dx = if spot.kind == 3 {
                0.0
            } else {
                wind * 26.0 * fraction * fraction + if cell.canopy { ripple } else { 0.0 }
            };
            let position = spot.position + cell.local + Vec2::new(dx, 0.0);
            if !view.expanded(9.0).contains(position) {
                continue;
            }
            let rustle = if cell.canopy && spot.kind != 3 {
                (self.time * 1.3 + cell.seed * 40.0).sin() * 0.05 + gust * 0.08
            } else {
                0.0
            };
            let brightness = (cell.brightness + rustle).clamp(0.0, 0.999);
            let index = (brightness * LETTERS.len() as f32 + (cell.seed - 0.5) * 2.2)
                .floor()
                .clamp(0.0, (LETTERS.len() - 1) as f32) as usize;
            let (dark, lit) = if cell.canopy {
                PALETTES[spot.kind as usize]
            } else {
                ([0.12, 0.07, 0.045], [0.62, 0.42, 0.26])
            };
            let dark = Vec3::from_array(dark);
            let color = dark.lerp(
                Vec3::from_array(lit),
                if cell.canopy {
                    brightness.powf(0.9)
                } else {
                    brightness
                },
            ) * light_pool(spot.position + cell.local, self.player).max(0.7);
            let mut backing = glyph_instance(
                position,
                Vec2::new(5.1, 8.16),
                dark * 0.45,
                0.0,
                depth(spot.position.y) - 0.000001,
                b' ',
            );
            backing.kind = 1;
            out.push(backing);
            out.push(glyph_instance(
                position,
                Vec2::new(5.6, 8.96),
                color,
                0.0,
                depth(spot.position.y) - 0.000002,
                LETTERS[index],
            ));
        }
    }
}

fn glyph_instance(
    position: Vec2,
    size: Vec2,
    color: Vec3,
    rotation: f32,
    depth: f32,
    glyph: u8,
) -> GlyphInstance {
    let color = color.clamp(Vec3::ZERO, Vec3::ONE);
    GlyphInstance {
        position: position.to_array(),
        size: size.to_array(),
        color: [color.x, color.y, color.z, 1.0],
        rotation,
        depth,
        glyph: u32::from(glyph),
        kind: 0,
    }
}

#[derive(Clone, Copy)]
struct Bounds {
    min: Vec2,
    max: Vec2,
}
impl Bounds {
    fn contains(self, point: Vec2) -> bool {
        point.cmpge(self.min).all() && point.cmple(self.max).all()
    }
    fn intersects(self, other: Self) -> bool {
        self.min.cmple(other.max).all() && other.min.cmple(self.max).all()
    }
    fn expanded(self, margin: f32) -> Self {
        Self {
            min: self.min - margin,
            max: self.max + margin,
        }
    }
}

#[derive(Default)]
struct Bucket {
    blades: Vec<Blade>,
    flowers: Vec<Flower>,
}
struct Blade {
    root: Vec2,
    color: Vec3,
    segments: u8,
    lean: f32,
    curl: f32,
    length: f32,
    flex: f32,
    seed: f32,
    style: u8,
}
struct Flower {
    root: Vec2,
    color: Vec3,
    height: f32,
    size: f32,
    glyph: u8,
}

fn wrap_root(root: Vec2) -> Vec2 {
    // Floating-point rem_euclid can round a tiny negative input up to the
    // modulus itself. Fold that endpoint to zero before indexing a bucket.
    let x = root.x.rem_euclid(TILE.x);
    let y = root.y.rem_euclid(TILE.y);
    Vec2::new(
        if x < TILE.x { x } else { 0.0 },
        if y < TILE.y { y } else { 0.0 },
    )
}
fn root_bucket(root: Vec2) -> usize {
    ((root.y / BUCKET_SIZE) as i32 * COLUMNS + (root.x / BUCKET_SIZE) as i32) as usize
}
fn tile_bucket(x: i32, y: i32) -> (usize, Vec2) {
    (
        (y.rem_euclid(ROWS) * COLUMNS + x.rem_euclid(COLUMNS)) as usize,
        Vec2::new(
            x.div_euclid(COLUMNS) as f32 * TILE.x,
            y.div_euclid(ROWS) as f32 * TILE.y,
        ),
    )
}

fn populate_grass(buckets: &mut [Bucket], seed: u32, density: f32) {
    if density == 0.0 {
        return;
    }
    let mut rng = Random::new(seed);
    let step = 13.5 / density.sqrt();
    let palette = [
        Vec3::new(0.13, 0.44, 0.37),
        Vec3::new(0.15, 0.48, 0.28),
        Vec3::new(0.09, 0.32, 0.27),
        Vec3::new(0.17, 0.42, 0.32),
    ];
    for gx in 0..(TILE.x / step).ceil() as u32 {
        for gy in 0..(TILE.y / step).ceil() as u32 {
            let center = Vec2::new(
                (gx as f32 + rng.next()) * step,
                (gy as f32 + rng.next()) * step,
            );
            let count = 1 + (rng.next() * rng.next() * 8.0) as usize;
            let color = palette[(rng.next() * 4.0) as usize];
            let lean = (rng.next() - 0.5) * 0.35;
            for _ in 0..count {
                let offset = Vec2::new((rng.next() - 0.5) * 9.0, (rng.next() - 0.5) * 4.0);
                let root = wrap_root(center + offset);
                let wisp = rng.next() < 0.07;
                let segments = if wisp {
                    4 + (rng.next() * 2.0) as u8
                } else {
                    2 + (rng.next() * rng.next() * 2.6) as u8
                };
                let lean =
                    lean + if count > 1 {
                        offset.x / 4.5 * 0.28
                    } else {
                        0.0
                    } + (rng.next() - 0.5) * 0.36;
                let curl = lean.signum() * (0.04 + rng.next() * 0.3) * if wisp { 1.8 } else { 1.0 };
                let length = if wisp {
                    10.0 + rng.next() * 2.5
                } else {
                    7.5 + rng.next() * 4.0
                };
                let style = (rng.next() * 10.0) as u8;
                let jitter = 0.82 + rng.next() * 0.3;
                let color = if wisp {
                    Vec3::new(0.26, 0.52, 0.44)
                } else {
                    color * Vec3::new(jitter, 0.85 + rng.next() * 0.3, jitter)
                };
                let flex = if wisp { 1.3 } else { 0.8 + rng.next() * 0.5 };
                buckets[root_bucket(root)].blades.push(Blade {
                    root,
                    color,
                    segments,
                    lean,
                    curl,
                    length,
                    flex,
                    seed: rng.next(),
                    style,
                });
            }
        }
    }
}

fn populate_flowers(buckets: &mut [Bucket], seed: u32, density: f32) {
    let mut rng = Random::new(seed);
    for _ in 0..(TILE.x * TILE.y / 600.0 * density) as usize {
        let choice = rng.next();
        let mut size = 9.0 + rng.next() * 4.0;
        let (glyph, color) = if choice < 0.6 {
            (b'o', Vec3::new(0.9, 0.93, 0.92))
        } else if choice < 0.75 {
            // ASCII-only atlas: a small raised o substitutes for degree sign.
            size *= 0.8;
            (b'o', Vec3::new(0.95, 0.97, 0.95))
        } else if choice < 0.84 {
            (b'*', Vec3::new(0.88, 0.9, 1.0))
        } else if choice < 0.95 {
            (b'.', Vec3::new(0.95, 0.95, 0.9))
        } else {
            size = 10.0 + rng.next() * 3.0;
            (
                if rng.next() < 0.6 { b'0' } else { b'o' },
                Vec3::new(0.55, 0.2, 0.13),
            )
        };
        let root = Vec2::new(rng.next() * TILE.x, rng.next() * TILE.y);
        buckets[root_bucket(root)].flowers.push(Flower {
            root,
            color,
            height: 6.0 + rng.next() * 26.0,
            size,
            glyph,
        });
    }
}

#[derive(Clone, Copy)]
struct Spot {
    position: Vec2,
    radius: f32,
    height: f32,
    kind: u8,
}
impl Spot {
    fn new(x: f32, y: f32, radius: f32, height: f32, kind: u8) -> Self {
        Self {
            position: Vec2::new(x, y),
            radius,
            height,
            kind,
        }
    }
    fn collision_radius(self) -> f32 {
        if self.kind == 3 {
            self.radius * 0.9
        } else {
            18.0
        }
    }
    fn shadow_bounds(self) -> Bounds {
        let center = self.position + Vec2::new(self.radius * 0.45, -self.radius * 0.05);
        let half = Vec2::new(1.45, 0.55) * (self.radius * 1.1);
        Bounds {
            min: center - half,
            max: center + half,
        }
    }
}

fn layout_trees(seed: u32) -> Vec<Spot> {
    let mut spots = vec![
        Spot::new(-330.0, 140.0, 135.0, 390.0, 1),
        Spot::new(360.0, 250.0, 115.0, 330.0, 0),
        Spot::new(120.0, -330.0, 95.0, 280.0, 2),
        Spot::new(-150.0, -170.0, 34.0, 50.0, 3),
        Spot::new(250.0, -40.0, 26.0, 40.0, 3),
    ];
    let mut rng = Random::new(seed);
    for _ in 0..5000 {
        if spots.len() == 60 {
            break;
        }
        let position = Vec2::new(rng.signed(), rng.signed()) * 3000.0;
        if position.length() < 450.0 {
            continue;
        }
        let rock = rng.next() < 0.3;
        let radius = if rock {
            22.0 + rng.next() * 26.0
        } else {
            90.0 + rng.next() * 60.0
        };
        if spots
            .iter()
            .any(|other| position.distance(other.position) < if rock { 120.0 } else { 330.0 })
        {
            continue;
        }
        let height = if rock {
            radius * 1.4
        } else {
            radius * (2.5 + rng.next() * 0.6)
        };
        let kind = if rock {
            3
        } else if rng.next() < 0.3 {
            1
        } else if rng.next() < 0.5 {
            2
        } else {
            0
        };
        spots.push(Spot {
            position,
            radius,
            height,
            kind,
        });
    }
    spots
}

struct Tree {
    spot: Spot,
    cells: Vec<OrganicCell>,
    bounds: Bounds,
}
impl Tree {
    fn new(spot: Spot, rng: &mut Random) -> Self {
        let rock = spot.kind == 3;
        let cells = organic_cells(spot.radius, spot.height, rock, rng);
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        for cell in &cells {
            min = min.min(cell.local);
            max = max.max(cell.local);
        }
        let margin = Vec2::new(if rock { 3.0 } else { 36.0 }, 5.0);
        let bounds = Bounds {
            min: spot.position + min - margin,
            max: spot.position + max + margin,
        };
        Self {
            spot,
            cells,
            bounds,
        }
    }
}

fn resolve_collision(position: &mut Vec2, velocity: &mut Vec2, spot: Spot) {
    let radius = spot.collision_radius();
    let relative = (*position - spot.position) * Vec2::new(1.0, 2.0);
    if relative.length_squared() >= radius * radius {
        return;
    }
    let direction = if relative.length_squared() > 0.000001 {
        relative.normalize()
    } else if velocity.length_squared() > 0.000001 {
        (-*velocity * Vec2::new(1.0, 2.0)).normalize()
    } else {
        Vec2::X
    };
    *position = spot.position + direction * (radius + 0.001) * Vec2::new(1.0, 0.5);
    let normal = (direction * Vec2::new(1.0, 2.0)).normalize();
    let into = velocity.dot(normal);
    if into < 0.0 {
        *velocity -= normal * into;
    }
}

fn player_push(root: Vec2, player: Vec2, seed: f32) -> f32 {
    let relative = root - player;
    let near = smoothstep(32.0, 3.0, (relative * Vec2::new(1.0, 2.2)).length());
    let side = if relative.x.abs() > 0.5 {
        relative.x.signum()
    } else if seed < 0.5 {
        -1.0
    } else {
        1.0
    };
    side * near * 1.15
}
fn light_pool(point: Vec2, player: Vec2) -> f32 {
    smoothstep(
        800.0,
        100.0,
        ((point - player) * Vec2::new(1.0, 1.25)).length(),
    ) * 0.5
        + 0.45
}
fn shadow_at(point: Vec2, player: Vec2, spots: &[Spot]) -> f32 {
    let hero = ((point - player - Vec2::new(12.0, -3.0)) / Vec2::new(46.0, 15.0)).length();
    let mut shadow = smoothstep(1.0, 0.3, hero) * 0.85;
    for spot in spots {
        let center = spot.position + Vec2::new(spot.radius * 0.45, -spot.radius * 0.05);
        let distance = ((point - center) / (Vec2::new(1.45, 0.55) * spot.radius * 1.1)).length();
        let opacity = if distance < 0.6 {
            0.95 - smoothstep(0.1, 0.6, distance) * 0.2
        } else {
            smoothstep(1.0, 0.6, distance) * 0.75
        };
        shadow = shadow.max(opacity);
    }
    shadow
}
fn gust_band(point: Vec2, time: f32) -> f32 {
    let band = (point.dot(Vec2::new(0.0058, 0.0021)) - time * 1.15).sin() * 0.5 + 0.5;
    let modulation = noise(point * 0.0035 + Vec2::new(time * 0.05, 0.0)) * 0.5 + 0.5;
    band.powf(2.5) * (modulation * 0.8 + 0.4)
}
fn wind_at(point: Vec2, time: f32, gust: f32) -> f32 {
    gust * 0.55 + noise(point * 0.0016 + Vec2::new(-time * 0.09, time * 0.03)) * 0.3 + 0.03
}
fn smoothstep(low: f32, high: f32, value: f32) -> f32 {
    let t = ((value - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}
fn wrap_angle(angle: f32) -> f32 {
    (angle + PI).rem_euclid(TAU) - PI
}
fn noise(point: Vec2) -> f32 {
    let cell = point.floor();
    let fraction = point - cell;
    let u = fraction * fraction * (Vec2::splat(3.0) - fraction * 2.0);
    let x = cell.x as i32;
    let y = cell.y as i32;
    let a = hash_noise(x, y);
    let b = hash_noise(x + 1, y);
    let c = hash_noise(x, y + 1);
    let d = hash_noise(x + 1, y + 1);
    let lower = a + (b - a) * u.x;
    let upper = c + (d - c) * u.x;
    lower + (upper - lower) * u.y
}
fn hash_noise(x: i32, y: i32) -> f32 {
    let mut hash = (x as u32).wrapping_mul(0x9e3779b9) ^ (y as u32).wrapping_mul(0x85ebca6b);
    hash ^= hash >> 16;
    hash = hash.wrapping_mul(0x7feb352d);
    hash ^= hash >> 15;
    (hash >> 8) as f32 / 16_777_215.0 * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_world() -> World {
        World {
            buckets: (0..COLUMNS * ROWS).map(|_| Bucket::default()).collect(),
            trees: Vec::new(),
            player: Vec2::ZERO,
            velocity: Vec2::ZERO,
            camera: Vec2::new(0.0, 40.0),
            yaw: 0.0,
            phase: 0.0,
            movement: 0.0,
            time: 0.0,
        }
    }
    fn advance(world: &mut World, frames: usize, controls: Controls) {
        for _ in 0..frames {
            world.step(1.0 / 60.0, controls);
        }
    }
    fn assert_finite(frame: &MeadowFrame) {
        assert!(frame.camera.iter().all(|v| v.is_finite()));
        assert!(frame.view_height.is_finite() && frame.view_height > 0.0);
        for instance in &frame.instances {
            assert!(instance
                .position
                .iter()
                .chain(&instance.size)
                .chain(&instance.color)
                .all(|v| v.is_finite()));
            assert!(instance.rotation.is_finite());
            assert!((0.0..1.0).contains(&instance.depth));
            assert!(instance.size.iter().all(|v| *v > 0.0));
            assert!(instance.color.iter().all(|v| (0.0..=1.0).contains(v)));
            assert!(instance.glyph < 128);
            assert!(instance.kind <= 1, "world must never generate knight cells");
        }
    }

    #[test]
    fn seeded_construction_frames_and_autoplay_are_deterministic() {
        let mut first = World::new(2024, 0.2);
        let mut second = World::new(2024, 0.2);
        assert_eq!(first.frame(1.6, 1.0), second.frame(1.6, 1.0));
        let initial = first.player_position();
        let controls = Controls {
            autoplay: true,
            ..Controls::default()
        };
        advance(&mut first, 120, controls);
        advance(&mut second, 120, controls);
        assert_ne!(first.player_position(), initial);
        assert_eq!(first.frame(1.6, 1.0), second.frame(1.6, 1.0));
        assert_ne!(
            first.frame(1.6, 1.0).instances,
            World::new(2025, 0.2).frame(1.6, 1.0).instances
        );
    }

    #[test]
    fn layout_preserves_reference_landmarks_and_all_palettes() {
        let spots = layout_trees(2024);
        assert_eq!(spots.len(), 60);
        assert_eq!(spots[0].position, Vec2::new(-330.0, 140.0));
        assert_eq!(spots[1].position, Vec2::new(360.0, 250.0));
        assert_eq!(spots[2].position, Vec2::new(120.0, -330.0));
        for kind in 0..=3 {
            assert!(spots.iter().any(|spot| spot.kind == kind));
        }
    }

    #[test]
    fn diagonal_is_normalized_run_is_faster_and_motion_is_smoothed() {
        let mut walk = empty_world();
        let mut diagonal = empty_world();
        let mut run = empty_world();
        let right = Controls {
            right: true,
            ..Controls::default()
        };
        walk.step(1.0 / 60.0, right);
        assert!(walk.velocity.x > 0.0 && walk.velocity.x < WALK_SPEED);
        advance(&mut walk, 59, right);
        advance(&mut diagonal, 60, Controls { up: true, ..right });
        advance(&mut run, 60, Controls { run: true, ..right });
        assert!((walk.player.length() - diagonal.player.length()).abs() < 0.001);
        assert!((run.player.length() / walk.player.length() - 2.0).abs() < 0.001);
        let before = walk.player;
        walk.step(1.0 / 60.0, Controls::default());
        assert!(walk.player.x > before.x, "deceleration should be smooth");
        advance(&mut walk, 120, Controls::default());
        assert!(walk.velocity.length() < 0.001);
    }

    #[test]
    fn manual_input_overrides_autoplay_and_opposites_cancel() {
        let mut first = empty_world();
        let mut second = empty_world();
        advance(
            &mut first,
            30,
            Controls {
                left: true,
                autoplay: true,
                ..Controls::default()
            },
        );
        advance(
            &mut second,
            30,
            Controls {
                left: true,
                ..Controls::default()
            },
        );
        assert_eq!(first.player, second.player);
        let mut idle = empty_world();
        advance(
            &mut idle,
            30,
            Controls {
                left: true,
                right: true,
                up: true,
                down: true,
                ..Controls::default()
            },
        );
        assert_eq!(idle.player, Vec2::ZERO);
    }

    #[test]
    fn camera_lags_then_converges_and_model_faces_motion() {
        let mut world = empty_world();
        advance(
            &mut world,
            60,
            Controls {
                right: true,
                ..Controls::default()
            },
        );
        assert!(world.camera.x > 0.0 && world.camera.x < world.player.x);
        assert!((world.yaw - PI / 2.0).abs() < 0.01);
        advance(&mut world, 300, Controls::default());
        assert!(world.camera.distance(world.player + Vec2::new(0.0, 40.0)) < 0.01);
        assert!((0.0..TAU).contains(&world.phase));
    }

    #[test]
    fn dt_is_bounded_and_bad_inputs_cannot_poison_state() {
        let mut first = empty_world();
        let mut second = empty_world();
        let controls = Controls {
            up: true,
            ..Controls::default()
        };
        first.step(100.0, controls);
        second.step(0.05, controls);
        assert_eq!(first.frame(1.6, 1.0), second.frame(1.6, 1.0));
        for dt in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            first.step(dt, controls);
        }
        assert_eq!(first.frame(1.6, 1.0), second.frame(1.6, 1.0));
        for (aspect, zoom) in [
            (f32::NAN, f32::INFINITY),
            (-1.0, 0.0),
            (f32::MAX, f32::MIN_POSITIVE),
        ] {
            assert_finite(&first.frame(aspect, zoom));
        }
        first.player = Vec2::splat(WORLD_LIMIT - 0.1);
        advance(
            &mut first,
            60,
            Controls {
                right: true,
                up: true,
                run: true,
                ..Controls::default()
            },
        );
        assert_eq!(first.player, Vec2::splat(WORLD_LIMIT));
    }

    #[test]
    fn collision_resolves_centers_and_removes_only_inward_velocity() {
        for kind in [0, 3] {
            let spot = Spot::new(10.0, -5.0, 34.0, 50.0, kind);
            let mut position = spot.position;
            let mut velocity = Vec2::ZERO;
            resolve_collision(&mut position, &mut velocity, spot);
            assert!(position.is_finite());
            assert!(
                ((position - spot.position) * Vec2::new(1.0, 2.0)).length()
                    >= spot.collision_radius()
            );
            position = spot.position + Vec2::new(-1.0, 0.0);
            velocity = Vec2::new(100.0, 25.0);
            resolve_collision(&mut position, &mut velocity, spot);
            assert_eq!(velocity, Vec2::new(0.0, 25.0));
        }
    }

    #[test]
    fn running_does_not_cross_a_trunk_or_boulder() {
        for kind in [0, 3] {
            let mut world = empty_world();
            let spot = Spot::new(35.0, 0.0, 30.0, 60.0, kind);
            world.trees.push(Tree::new(spot, &mut Random::new(1)));
            advance(
                &mut world,
                240,
                Controls {
                    right: true,
                    run: true,
                    ..Controls::default()
                },
            );
            assert!(world.player.x < spot.position.x - spot.collision_radius() + 0.01);
            assert!(world.player.is_finite());
        }
    }

    #[test]
    fn tiling_wraps_negative_coordinates_without_duplicates_or_gaps() {
        assert_eq!(wrap_root(Vec2::new(-1.0, -1.0)), TILE - Vec2::ONE);
        assert_eq!(wrap_root(TILE), Vec2::ZERO);
        assert_eq!(wrap_root(Vec2::splat(-f32::EPSILON)), Vec2::ZERO);
        assert_eq!(tile_bucket(-1, -1), ((COLUMNS * ROWS - 1) as usize, -TILE));
        let mut world = empty_world();
        let root = Vec2::new(20.0, 20.0);
        world.buckets[root_bucket(root)].flowers.push(Flower {
            root,
            color: Vec3::ONE,
            height: 10.0,
            size: 10.0,
            glyph: b'*',
        });
        world.camera = root;
        let first = world.frame(1.0, 4.0);
        world.camera += TILE;
        world.player += TILE;
        let second = world.frame(1.0, 4.0);
        assert_eq!(first.instances.len(), 1);
        assert_eq!(second.instances.len(), 1);
        assert!((Vec2::from_array(second.instances[0].position) - TILE - root).length() < 11.0);
        world.camera = TILE * 0.5;
        let wide = world.frame(2.0, 0.25);
        assert!(
            wide.instances.len() > 1,
            "zoomed out views need multiple tile copies"
        );
        world.camera = Vec2::ZERO;
        let ultrawide = world.frame(16.0, 1.0);
        assert!(
            ultrawide
                .instances
                .iter()
                .any(|cell| cell.position[0] > 3000.0),
            "culling must cover the full GPU projection on wide windows"
        );
    }

    #[test]
    fn grass_and_flowers_use_root_depth_not_their_elevated_tips() {
        let mut world = empty_world();
        let root = Vec2::new(0.0, -10.0);
        let blade = Blade {
            root,
            color: Vec3::ONE,
            segments: 4,
            lean: 0.0,
            curl: 0.1,
            length: 10.0,
            flex: 1.0,
            seed: 0.2,
            style: 0,
        };
        let view = Bounds {
            min: Vec2::splat(-100.0),
            max: Vec2::splat(100.0),
        };
        let mut cells = Vec::new();
        world.draw_blade(&blade, root, view, &[], &mut cells);
        assert_eq!(cells.len(), 4);
        assert!(cells.iter().all(|cell| cell.depth == depth(root.y)));
        assert!(cells.last().unwrap().position[1] > 0.0);
        world.draw_flower(
            &Flower {
                root,
                color: Vec3::ONE,
                height: 20.0,
                size: 10.0,
                glyph: b'o',
            },
            root,
            view,
            &[],
            &mut cells,
        );
        assert!(cells.last().unwrap().depth < cells[0].depth);
        let before = cells[0].position;
        world.player = Vec2::new(100.0, 100.0);
        cells.clear();
        world.draw_blade(&blade, root, view, &[], &mut cells);
        assert_ne!(
            before, cells[0].position,
            "player should bend nearby blades"
        );
    }

    #[test]
    fn shadows_light_pool_and_player_push_have_reference_direction() {
        assert!(player_push(Vec2::new(-5.0, 0.0), Vec2::ZERO, 0.5) < 0.0);
        assert!(player_push(Vec2::new(5.0, 0.0), Vec2::ZERO, 0.5) > 0.0);
        assert_eq!(player_push(Vec2::splat(100.0), Vec2::ZERO, 0.5), 0.0);
        let spot = Spot::new(0.0, 0.0, 100.0, 300.0, 0);
        let player = Vec2::splat(1000.0);
        assert!(shadow_at(Vec2::new(45.0, -5.0), player, &[spot]) > 0.9);
        assert_eq!(shadow_at(Vec2::splat(500.0), player, &[spot]), 0.0);
        assert!(light_pool(Vec2::ZERO, Vec2::ZERO) > light_pool(Vec2::splat(1000.0), Vec2::ZERO));
    }

    #[test]
    fn cached_tree_backings_are_behind_letters_but_share_ground_depth() {
        let mut world = empty_world();
        world.trees.push(Tree::new(
            Spot::new(0.0, 0.0, 30.0, 90.0, 0),
            &mut Random::new(1),
        ));
        let cells_ptr = world.trees[0].cells.as_ptr();
        let first = world.frame(1.6, 1.0);
        assert!(!first.instances.is_empty());
        for pair in first.instances.as_chunks::<2>().0 {
            assert_eq!((pair[0].kind, pair[1].kind), (1, 0));
            assert_eq!(pair[0].position, pair[1].position);
            assert!(pair[0].depth > pair[1].depth);
            assert_eq!(pair[1].depth, depth(0.0) - 0.000002);
        }
        world.step(0.05, Controls::default());
        assert_eq!(cells_ptr, world.trees[0].cells.as_ptr());
        assert_ne!(first.instances, world.frame(1.6, 1.0).instances);
        world.camera = Vec2::splat(2000.0);
        assert!(world.frame(1.6, 1.0).instances.is_empty());
    }

    #[test]
    fn canopy_is_visible_even_when_its_ground_root_is_offscreen() {
        let mut world = empty_world();
        let spot = Spot::new(0.0, -500.0, 150.0, 650.0, 1);
        world.trees.push(Tree::new(spot, &mut Random::new(17)));
        let frame = world.frame(1.6, 1.0);
        assert!(spot.position.y < world.camera.y - frame.view_height * 0.5);
        assert!(!frame.instances.is_empty());
        assert!(frame.instances.iter().all(|cell| cell.depth < depth(0.0)));
        assert_finite(&frame);
    }

    #[test]
    fn rocks_remain_static_while_wind_moves_grass_and_trees() {
        let mut world = empty_world();
        world.trees.push(Tree::new(
            Spot::new(0.0, 0.0, 40.0, 56.0, 3),
            &mut Random::new(11),
        ));
        let before = world.frame(1.6, 1.0).instances;
        world.time = 3.0;
        assert_eq!(before, world.frame(1.6, 1.0).instances);
        let mut buckets: Vec<_> = (0..COLUMNS * ROWS).map(|_| Bucket::default()).collect();
        populate_grass(&mut buckets, 1, 0.0);
        populate_flowers(&mut buckets, 1, 0.0);
        assert!(buckets
            .iter()
            .all(|bucket| bucket.blades.is_empty() && bucket.flowers.is_empty()));
        populate_grass(&mut buckets, 1, 0.1);
        world.buckets = buckets;
        let before = world.frame(1.6, 1.0).instances;
        world.time = 4.0;
        assert_ne!(before, world.frame(1.6, 1.0).instances);
    }

    #[test]
    fn default_is_dense_finite_culled_and_contains_no_knight_cells() {
        let world = World::new(2024, 1.0);
        let frame = world.frame(1.6, 1.0);
        assert!(
            frame.instances.len() > 15_000,
            "default should be a dense meadow"
        );
        assert_finite(&frame);
        let raw_segments: usize = world
            .buckets
            .iter()
            .flat_map(|bucket| &bucket.blades)
            .map(|blade| usize::from(blade.segments))
            .sum();
        let raw_tree_cells: usize = world.trees.iter().map(|tree| tree.cells.len() * 2).sum();
        assert!(frame.instances.len() < raw_segments + raw_tree_cells);
        assert!(world.frame(1.6, 3.0).instances.len() < frame.instances.len());
        let half = Vec2::new(frame.view_height * 1.6, frame.view_height) * 0.5;
        for cell in &frame.instances {
            let relative = Vec2::from_array(cell.position) - world.camera;
            assert!(relative.x.abs() <= half.x + 16.0 && relative.y.abs() <= half.y + 16.0);
        }
        let whites = frame
            .instances
            .iter()
            .filter(|cell| cell.glyph == u32::from(b'o') && cell.color[0] > 0.7)
            .count();
        assert!(whites > 50, "white blossoms should sparkle over the grass");
    }

    #[test]
    fn random_range_and_noise_are_bounded() {
        let mut rng = Random::new(u32::MAX);
        for _ in 0..10000 {
            let value = rng.next();
            assert!((0.0..1.0).contains(&value));
            let n = noise(Vec2::new(value * 100.0 - 50.0, rng.signed() * 100.0));
            assert!((-1.0..=1.0).contains(&n));
        }
    }
}
