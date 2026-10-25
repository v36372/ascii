// SPDX-License-Identifier: Apache-2.0
//! Native glyph-sprite rendering and a 2D meadow with a live 3D knight.
//!
//! Unlike the terminal framebuffer, independent glyph sprites have floating-point
//! positions, arbitrary rotation/scale, and depth. Simulation remains independent
//! of the window and graphics device for deterministic headless captures.

pub mod renderer;
pub mod world;

// Compatibility exports. Shared engine data is owned by spatial, not this demo.
pub use crate::spatial::{depth, Controls, GlyphInstance, KnightPose as HeroPose};

#[derive(Clone, Debug, PartialEq)]
pub struct MeadowFrame {
    pub camera: [f32; 2],
    pub view_height: f32,
    pub time: f32,
    pub instances: Vec<GlyphInstance>,
    pub hero: HeroPose,
}

impl crate::spatial::FrameSource for MeadowFrame {
    fn frame_view(&self) -> crate::spatial::FrameView<'_> {
        crate::spatial::FrameView {
            camera: self.camera,
            view_height: self.view_height,
            time: self.time,
            background: [0.0015, 0.0045, 0.0055, 1.0],
            instances: &self.instances,
            knight: Some(self.hero),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::FrameSource;

    #[test]
    fn legacy_frame_adapts_without_copying_or_changing_defaults() {
        let frame = MeadowFrame {
            camera: [0.0; 2],
            view_height: 640.0,
            time: 0.0,
            instances: vec![],
            hero: HeroPose::default(),
        };
        let view = frame.frame_view();
        assert_eq!(view.knight, Some(frame.hero));
        assert_eq!(view.background, [0.0015, 0.0045, 0.0055, 1.0]);
        assert!(std::ptr::eq(view.instances, frame.instances.as_slice()));
    }
}
