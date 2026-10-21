// SPDX-License-Identifier: Apache-2.0
//! Native glyph-sprite rendering and a 2D meadow with a live 3D knight.
//!
//! Unlike the terminal framebuffer, independent glyph sprites have floating-point
//! positions, arbitrary rotation/scale, and depth. Simulation remains independent
//! of the window and graphics device for deterministic headless captures.

pub mod renderer;
pub mod world;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlyphInstance {
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub color: [f32; 4],
    pub rotation: f32,
    pub depth: f32,
    /// ASCII atlas code for kind0; character-cell index for kind2.
    pub glyph: u32,
    /// 0 = atlas glyph, 1 = solid backing, 2 = live knight texture cell.
    pub kind: u32,
}

/// Ground-based depth: objects lower on the screen occlude objects behind them.
pub fn depth(ground_y: f32) -> f32 {
    (0.5 + ground_y * 0.00005).clamp(0.01, 0.99)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Controls {
    pub up: bool,
    pub down: bool,
    pub left: bool,
    pub right: bool,
    pub run: bool,
    pub autoplay: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HeroPose {
    pub position: [f32; 2],
    pub yaw: f32,
    pub phase: f32,
    pub movement: f32,
    pub time: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeadowFrame {
    pub camera: [f32; 2],
    pub view_height: f32,
    pub time: f32,
    pub instances: Vec<GlyphInstance>,
    pub hero: HeroPose,
}
