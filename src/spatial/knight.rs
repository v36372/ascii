// SPDX-License-Identifier: Apache-2.0
//! Live low-poly knight geometry. All vertices are posed in 3D each frame, before
//! the GPU depth-tests and lights them in the small orthographic render target.

use super::super::{depth, GlyphInstance, KnightPose};
use glam::{Mat4, Vec3};
use std::f32::consts::{FRAC_PI_2, PI, TAU};

pub(super) const COLS: u32 = 29;
pub(super) const ROWS: u32 = 28;
pub(super) const WIDTH: u32 = COLS * 2;
pub(super) const HEIGHT: u32 = ROWS * 2;
const FRAME_WIDTH: f32 = 1.5;
const FRAME_HEIGHT: f32 = 2.3;
const SCALE: f32 = 76.0;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

pub(super) fn camera() -> Mat4 {
    let target = Vec3::new(0.0, 0.98, 0.0);
    let elevation = PI / 6.0;
    Mat4::orthographic_rh(
        -FRAME_WIDTH / 2.0,
        FRAME_WIDTH / 2.0,
        -FRAME_HEIGHT / 2.0,
        FRAME_HEIGHT / 2.0,
        0.1,
        20.0,
    ) * Mat4::look_at_rh(
        target + Vec3::new(0.0, elevation.sin(), elevation.cos()) * 6.0,
        target,
        Vec3::Y,
    )
}

pub(super) fn cells(pose: KnightPose) -> impl Iterator<Item = GlyphInstance> {
    let feet_y = (camera().project_point3(Vec3::ZERO).y + 1.0) * 0.5;
    (0..COLS * ROWS).map(move |index| {
        let x = (index % COLS) as f32 + 0.5;
        let y = (index / COLS) as f32 + 0.5;
        GlyphInstance {
            position: [
                pose.position[0] + (x / COLS as f32 - 0.5) * FRAME_WIDTH * SCALE,
                pose.position[1] + (y / ROWS as f32 - feet_y) * FRAME_HEIGHT * SCALE,
            ],
            size: [
                FRAME_WIDTH * SCALE / COLS as f32,
                FRAME_HEIGHT * SCALE / ROWS as f32,
            ],
            color: [1.0; 4],
            rotation: 0.0,
            // All cells share the feet's ground depth: foreground grass can
            // cover legs, but grass rooted behind the hero cannot cover armour.
            depth: depth(pose.position[1]) - 0.00002,
            glyph: index,
            kind: 2,
        }
    })
}

fn material(rgb: u32) -> Vec3 {
    let channel = |shift: u32| {
        let value = ((rgb >> shift) & 255u32) as f32 / 255.0;
        if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    };
    Vec3::new(channel(16), channel(8), channel(0))
}

#[derive(Default)]
struct Mesh(Vec<Vertex>);

impl Mesh {
    fn triangle(&mut self, transform: Mat4, color: Vec3, a: Vec3, b: Vec3, c: Vec3) {
        let a = transform.transform_point3(a);
        let b = transform.transform_point3(b);
        let c = transform.transform_point3(c);
        let normal = (b - a).cross(c - a).normalize_or_zero();
        if normal == Vec3::ZERO {
            return;
        }
        for position in [a, b, c] {
            self.0.push(Vertex {
                position: position.to_array(),
                normal: normal.to_array(),
                color: color.to_array(),
            });
        }
    }

    fn quad(&mut self, transform: Mat4, color: Vec3, corners: [Vec3; 4]) {
        let [a, b, c, d] = corners;
        self.triangle(transform, color, a, b, c);
        self.triangle(transform, color, a, c, d);
    }

    fn cuboid(&mut self, transform: Mat4, size: Vec3, color: Vec3) {
        let h = size * 0.5;
        // Counterclockwise when viewed from outside; each face has flat normals.
        for (n, u, v) in [
            (Vec3::X, Vec3::Y, Vec3::Z),
            (-Vec3::X, Vec3::Z, Vec3::Y),
            (Vec3::Y, Vec3::Z, Vec3::X),
            (-Vec3::Y, Vec3::X, Vec3::Z),
            (Vec3::Z, Vec3::X, Vec3::Y),
            (-Vec3::Z, Vec3::Y, Vec3::X),
        ] {
            self.quad(
                transform,
                color,
                [
                    (n - u - v) * h,
                    (n + u - v) * h,
                    (n + u + v) * h,
                    (n - u + v) * h,
                ],
            );
        }
    }

    fn cylinder(
        &mut self,
        transform: Mat4,
        top: f32,
        bottom: f32,
        height: f32,
        sides: u32,
        color: Vec3,
    ) {
        for i in 0..sides {
            let angle = i as f32 * TAU / sides as f32;
            let next = (i + 1) as f32 * TAU / sides as f32;
            let a = Vec3::new(angle.cos() * bottom, -height / 2.0, angle.sin() * bottom);
            let b = Vec3::new(angle.cos() * top, height / 2.0, angle.sin() * top);
            let c = Vec3::new(next.cos() * top, height / 2.0, next.sin() * top);
            let d = Vec3::new(next.cos() * bottom, -height / 2.0, next.sin() * bottom);
            self.quad(transform, color, [a, b, c, d]);
            self.triangle(transform, color, Vec3::Y * (height / 2.0), c, b);
            self.triangle(transform, color, -Vec3::Y * (height / 2.0), a, d);
        }
    }

    fn sphere(&mut self, transform: Mat4, radius: f32, hemisphere: bool, color: Vec3) {
        let rows = if hemisphere { 3 } else { 6 };
        let point = |row: u32, col: u32| {
            // Exact poles avoid tiny inverted triangles from sin(PI) rounding.
            if row == 0 {
                return Vec3::Y * radius;
            }
            if row == 6 {
                return -Vec3::Y * radius;
            }
            let latitude = row as f32 * PI / 6.0;
            let longitude = col as f32 * TAU / 8.0;
            Vec3::new(
                latitude.sin() * longitude.cos(),
                latitude.cos(),
                latitude.sin() * longitude.sin(),
            ) * radius
        };
        for row in 0..rows {
            for col in 0..8 {
                self.quad(
                    transform,
                    color,
                    [
                        point(row, col),
                        point(row, col + 1),
                        point(row + 1, col + 1),
                        point(row + 1, col),
                    ],
                );
            }
        }
    }
}

fn at(x: f32, y: f32, z: f32) -> Mat4 {
    Mat4::from_translation(Vec3::new(x, y, z))
}

pub(super) fn mesh(pose: KnightPose) -> Vec<Vertex> {
    let steel = material(0xc4ccd8);
    let dark = material(0x3a3f4c);
    let leather = material(0x6b4528);
    let red = material(0x8a1f1f);
    let blade = material(0xe8eef8);
    let movement = pose.movement.clamp(0.0, 1.0);
    let swing = pose.phase.sin() * movement;
    let bounce = pose.phase.cos().abs() * 0.05 * movement + (pose.time * 2.0).sin() * 0.008;
    let body = Mat4::from_rotation_y(pose.yaw) * at(0.0, bounce, 0.0);
    let mut mesh = Mesh(Vec::with_capacity(3600));
    mesh.cylinder(body * at(0.0, 1.2, 0.0), 0.23, 0.18, 0.55, 8, steel);
    mesh.cuboid(
        body * at(0.0, 1.28, 0.16),
        Vec3::new(0.3, 0.22, 0.08),
        steel,
    );
    mesh.cylinder(body * at(0.0, 0.95, 0.0), 0.2, 0.2, 0.08, 8, leather);
    mesh.cylinder(body * at(0.0, 0.83, 0.0), 0.2, 0.27, 0.22, 8, dark);
    mesh.cylinder(body * at(0.0, 1.64, 0.0), 0.15, 0.16, 0.27, 8, steel);
    mesh.sphere(body * at(0.0, 1.775, 0.0), 0.15, true, steel);
    mesh.cuboid(
        body * at(0.0, 1.66, 0.15),
        Vec3::new(0.2, 0.035, 0.03),
        dark,
    );
    mesh.cuboid(body * at(0.0, 1.9, -0.05), Vec3::new(0.045, 0.14, 0.3), red);
    let cape_angle = 0.12 + 0.35 * movement + (pose.time * 2.3).sin() * 0.04;
    mesh.cuboid(
        body * at(0.0, 1.05, -0.21) * Mat4::from_rotation_x(cape_angle),
        Vec3::new(0.44, 0.78, 0.03),
        red,
    );
    for side in [-1.0, 1.0] {
        let leg = body * at(side * 0.1, 0.88, 0.0) * Mat4::from_rotation_x(-side * swing * 0.7);
        mesh.cuboid(leg * at(0.0, -0.21, 0.0), Vec3::new(0.15, 0.42, 0.17), dark);
        mesh.cuboid(leg * at(0.0, -0.6, 0.0), Vec3::new(0.13, 0.4, 0.15), steel);
        mesh.cuboid(
            leg * at(0.0, -0.84, 0.04),
            Vec3::new(0.15, 0.09, 0.26),
            leather,
        );
        mesh.sphere(body * at(side * 0.27, 1.44, 0.0), 0.12, false, steel);
        let angle = if side < 0.0 {
            -swing * 0.45 - 0.25
        } else {
            swing * 0.5 - 0.15
        };
        let arm = body * at(side * 0.29, 1.4, 0.0) * Mat4::from_rotation_x(angle);
        mesh.cuboid(arm * at(0.0, -0.16, 0.0), Vec3::new(0.11, 0.32, 0.12), dark);
        mesh.cuboid(arm * at(0.0, -0.46, 0.0), Vec3::new(0.11, 0.3, 0.12), steel);
        if side > 0.0 {
            let sword = arm * at(0.0, -0.62, 0.05) * Mat4::from_rotation_x(0.5);
            mesh.cuboid(
                sword * at(0.0, -0.33, 0.0),
                Vec3::new(0.04, 0.75, 0.012),
                blade,
            );
            mesh.cuboid(sword, Vec3::new(0.18, 0.03, 0.05), leather);
        } else {
            let shield = arm * at(-0.08, -0.38, 0.06) * Mat4::from_rotation_z(FRAC_PI_2);
            mesh.cylinder(shield, 0.25, 0.25, 0.05, 10, red);
            mesh.sphere(arm * at(-0.12, -0.38, 0.06), 0.06, false, steel);
        }
    }
    mesh.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knight_is_finite_flat_shaded_triangle_geometry() {
        let vertices = mesh(KnightPose::default());
        assert!(vertices.len() > 1000);
        assert_eq!(vertices.len() % 3, 0);
        for tri in vertices.as_chunks::<3>().0 {
            for v in tri {
                assert!(v
                    .position
                    .iter()
                    .chain(&v.normal)
                    .chain(&v.color)
                    .all(|v| v.is_finite()));
                assert!((Vec3::from_array(v.normal).length() - 1.0).abs() < 0.0001);
                assert_eq!(v.normal, tri[0].normal);
            }
        }
        assert!(vertices.iter().any(|v| v.position[1] > 1.9));
        assert!(vertices.iter().any(|v| v.position[1] < 0.1));
    }

    #[test]
    fn pose_changes_geometry_but_world_translation_does_not() {
        let base = mesh(KnightPose::default());
        assert_eq!(base, mesh(KnightPose::default()));
        assert_eq!(
            base,
            mesh(KnightPose {
                position: [100.0, -200.0],
                ..KnightPose::default()
            })
        );
        for pose in [
            KnightPose {
                yaw: FRAC_PI_2,
                ..KnightPose::default()
            },
            KnightPose {
                phase: FRAC_PI_2,
                movement: 1.0,
                ..KnightPose::default()
            },
            KnightPose {
                time: 1.0,
                ..KnightPose::default()
            },
        ] {
            assert_ne!(base, mesh(pose));
        }
    }

    #[test]
    fn cells_have_correct_grid_foot_anchor_and_ground_depth() {
        let pose = KnightPose {
            position: [13.0, -70.0],
            ..KnightPose::default()
        };
        let cells: Vec<_> = cells(pose).collect();
        assert_eq!(cells.len(), 812);
        let foot_y = (camera().project_point3(Vec3::ZERO).y + 1.0) * 0.5;
        assert!(
            (cells[0].position[1] - cells[0].size[1] * 0.5 + foot_y * FRAME_HEIGHT * SCALE
                - pose.position[1])
                .abs()
                < 0.001
        );
        for (i, cell) in cells.iter().enumerate() {
            assert_eq!(cell.kind, 2);
            assert_eq!(cell.glyph, i as u32);
            assert_eq!(cell.depth, depth(pose.position[1]) - 0.00002);
        }
        assert!(cells[COLS as usize].position[1] > cells[0].position[1]);
    }

    #[test]
    fn primitive_normals_face_outward() {
        let mut mesh = Mesh::default();
        mesh.cuboid(Mat4::IDENTITY, Vec3::ONE, Vec3::ONE);
        mesh.cylinder(Mat4::IDENTITY, 0.5, 0.5, 1.0, 8, Vec3::ONE);
        mesh.sphere(Mat4::IDENTITY, 0.5, false, Vec3::ONE);
        for tri in mesh.0.as_chunks::<3>().0 {
            let center = tri
                .iter()
                .map(|v| Vec3::from_array(v.position))
                .sum::<Vec3>()
                / 3.0;
            assert!(center.dot(Vec3::from_array(tri[0].normal)) > 0.0);
        }
    }
}
