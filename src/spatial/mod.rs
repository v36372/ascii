// SPDX-License-Identifier: Apache-2.0
//! Composable, device-independent authoring for the native glyph renderer.
//!
//! Coordinates are world units: +X right, +Y up. Quad positions are centers,
//! angles are counterclockwise radians, and colors are linear RGB. Lower depth
//! is nearer. Groups use one explicit depth anchor, never each glyph's height.
//! Uniform group scaling deliberately avoids shear, which glyph quads cannot
//! represent. See `docs/spatial.md` and `examples/spatial_courtyard.rs`.
//!
//! ```
//! use cosplay::spatial::{Depth, GlyphGroup, LinearRgba, SceneFrame, Transform2d};
//! let art = GlyphGroup::from_text("[o]", [8.0, 12.0], LinearRgba::WHITE, None)?;
//! let mut frame = SceneFrame::default(); // No implicit character.
//! frame.add(&art, Transform2d::at(20.0, 0.0), Depth::ground(0.0)?)?;
//! assert_eq!(frame.instances().len(), 3);
//! # Ok::<(), cosplay::spatial::SceneError>(())
//! ```

pub mod app;
pub mod prefabs;
pub mod renderer;
mod types;

use glam::Vec2;
pub use renderer::{FrameSource, FrameView, GpuRenderer};
use std::fmt;
pub use types::{depth, Controls, GlyphInstance, KnightPose};

/// Invalid authoring input. Errors are returned before modifying a frame/group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SceneError(pub(crate) &'static str);
impl fmt::Display for SceneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for SceneError {}

/// A printable character in the renderer's built-in ASCII atlas (space..=~).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AsciiGlyph(u8);
impl AsciiGlyph {
    pub fn new(ch: char) -> Result<Self, SceneError> {
        if (' '..='~').contains(&ch) {
            Ok(Self(ch as u8))
        } else {
            Err(SceneError("glyph must be printable ASCII"))
        }
    }
    pub fn character(self) -> char {
        char::from(self.0)
    }
}

/// Finite linear RGBA channels in 0..=1 (not sRGB byte values).
///
/// The current renderer depth-writes translucent quads too; this is NOT an
/// order-independent transparency API. Atlas ink below alpha 0.5 is discarded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearRgba([f32; 4]);
impl LinearRgba {
    pub const WHITE: Self = Self([1.0; 4]);
    pub const BLACK: Self = Self([0.0, 0.0, 0.0, 1.0]);
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Result<Self, SceneError> {
        let channels = [r, g, b, a];
        if channels.iter().all(|v| (0.0..=1.0).contains(v)) {
            Ok(Self(channels))
        } else {
            Err(SceneError(
                "linear color channels must be finite and in 0..=1",
            ))
        }
    }
    pub fn from_srgb8(r: u8, g: u8, b: u8) -> Self {
        let linear = |v: u8| {
            let v = f32::from(v) / 255.0;
            if v <= 0.04045 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        };
        Self([linear(r), linear(g), linear(b), 1.0])
    }
    pub fn channels(self) -> [f32; 4] {
        self.0
    }
}

/// Normalized GPU depth; smaller values draw in front of larger ones.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Depth(f32);
impl Depth {
    pub fn new(value: f32) -> Result<Self, SceneError> {
        if (0.0..=1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err(SceneError("depth must be finite and in 0..=1"))
        }
    }
    /// Meadow-compatible ground sorting. Lower world Y is closer to the viewer.
    /// Extreme positions saturate at 0.01/0.99; use explicit depth for other views.
    pub fn ground(y: f32) -> Result<Self, SceneError> {
        if !y.is_finite() {
            return Err(SceneError("ground Y must be finite"));
        }
        Self::new(depth(y))
    }
    pub fn value(self) -> f32 {
        self.0
    }
}

/// Local -> world: uniform scale, then rotation, then translation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform2d {
    pub translation: [f32; 2],
    pub rotation: f32,
    /// Must be finite and strictly positive. Reflection/shear are not supported.
    pub scale: f32,
}
impl Default for Transform2d {
    fn default() -> Self {
        Self {
            translation: [0.0; 2],
            rotation: 0.0,
            scale: 1.0,
        }
    }
}
impl Transform2d {
    pub fn at(x: f32, y: f32) -> Self {
        Self {
            translation: [x, y],
            ..Self::default()
        }
    }
    pub(crate) fn validate(self) -> Result<(), SceneError> {
        if self.translation.iter().all(|v| v.is_finite())
            && self.rotation.is_finite()
            && self.scale.is_finite()
            && self.scale > 0.0
        {
            Ok(())
        } else {
            Err(SceneError(
                "transform must be finite with positive uniform scale",
            ))
        }
    }
    fn point(self, point: [f32; 2]) -> [f32; 2] {
        let point = Vec2::from_array(point) * self.scale;
        let (s, c) = self.rotation.sin_cos();
        (Vec2::new(c * point.x - s * point.y, s * point.x + c * point.y)
            + Vec2::from_array(self.translation))
        .to_array()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Primitive {
    Glyph(AsciiGlyph),
    Solid,
}

/// One local-space quad. Public geometry is validated when composed/emitted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quad {
    pub primitive: Primitive,
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub rotation: f32,
    pub color: LinearRgba,
    /// Local layering: +1 is 0.000001 nearer than the group's depth anchor.
    /// Keep this small; it is a visual offset, not an independent world layer.
    pub order: i16,
}
impl Quad {
    pub fn glyph(glyph: AsciiGlyph, size: [f32; 2], color: LinearRgba) -> Self {
        Self {
            primitive: Primitive::Glyph(glyph),
            ..Self::solid(size, color)
        }
    }
    pub fn solid(size: [f32; 2], color: LinearRgba) -> Self {
        Self {
            primitive: Primitive::Solid,
            position: [0.0; 2],
            size,
            rotation: 0.0,
            color,
            order: 0,
        }
    }
    pub fn at(mut self, x: f32, y: f32) -> Self {
        self.position = [x, y];
        self
    }
    pub fn with_order(mut self, order: i16) -> Self {
        self.order = order;
        self
    }
    fn validate(self) -> Result<(), SceneError> {
        if self.position.iter().all(|v| v.is_finite())
            && self.rotation.is_finite()
            && self.size.iter().all(|v| v.is_finite() && *v > 0.0)
        {
            Ok(())
        } else {
            Err(SceneError(
                "quad geometry must be finite with positive dimensions",
            ))
        }
    }
    fn transformed(mut self, transform: Transform2d) -> Result<Self, SceneError> {
        self.validate()?;
        self.position = transform.point(self.position);
        self.size = self.size.map(|v| v * transform.scale);
        self.rotation += transform.rotation;
        self.validate()?; // Also catches overflow in otherwise finite operands.
        Ok(self)
    }
}

/// Axis-aligned bounds of the complete transformed quads, including rotation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds2d {
    pub min: [f32; 2],
    pub max: [f32; 2],
}
impl Bounds2d {
    pub fn intersects(self, other: Self) -> bool {
        (0..2).all(|axis| self.min[axis] <= other.max[axis] && self.max[axis] >= other.min[axis])
    }
}

/// Reusable visual asset in local coordinates. No simulation or collision state.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GlyphGroup {
    quads: Vec<Quad>,
}
impl GlyphGroup {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn quads(&self) -> &[Quad] {
        &self.quads
    }
    pub fn push(&mut self, quad: Quad) -> Result<(), SceneError> {
        quad.validate()?;
        self.quads.push(quad);
        Ok(())
    }
    /// Flatten another asset into this one; the source remains unchanged.
    pub fn extend(&mut self, source: &Self, transform: Transform2d) -> Result<(), SceneError> {
        let quads = source.transformed(transform)?;
        self.quads.extend(quads);
        Ok(())
    }
    fn transformed(&self, transform: Transform2d) -> Result<Vec<Quad>, SceneError> {
        transform.validate()?;
        self.quads
            .iter()
            .map(|quad| quad.transformed(transform))
            .collect()
    }
    pub fn bounds(&self, transform: Transform2d) -> Result<Option<Bounds2d>, SceneError> {
        let quads = self.transformed(transform)?;
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        for quad in &quads {
            let (s, c) = quad.rotation.sin_cos();
            let [w, h] = quad.size;
            let half = Vec2::new(c.abs() * w + s.abs() * h, s.abs() * w + c.abs() * h) * 0.5;
            let center = Vec2::from_array(quad.position);
            min = min.min(center - half);
            max = max.max(center + half);
        }
        if quads.is_empty() {
            return Ok(None);
        }
        if !min.is_finite() || !max.is_finite() {
            return Err(SceneError("bounds overflow"));
        }
        Ok(Some(Bounds2d {
            min: min.to_array(),
            max: max.to_array(),
        }))
    }
    /// Multiline ASCII art. Origin is the first cell's center; rows go down (-Y).
    /// Spaces are holes without a backing, filled cells with one. Newline is the
    /// only accepted control character; normalize CRLF before passing text.
    pub fn from_text(
        text: &str,
        cell: [f32; 2],
        ink: LinearRgba,
        backing: Option<LinearRgba>,
    ) -> Result<Self, SceneError> {
        Quad::solid(cell, ink).validate()?;
        let mut group = Self::new();
        for (row, line) in text.split('\n').enumerate() {
            for (col, ch) in line.chars().enumerate() {
                let glyph = AsciiGlyph::new(ch)?;
                let (x, y) = (col as f32 * cell[0], -(row as f32) * cell[1]);
                if let Some(color) = backing {
                    group.push(Quad::solid(cell, color).at(x, y))?;
                }
                if ch != ' ' {
                    group.push(Quad::glyph(glyph, cell, ink).at(x, y).with_order(1))?;
                }
            }
        }
        Ok(group)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera2d {
    pub center: [f32; 2],
    /// Visible world height; visible width is height * render-target aspect.
    pub view_height: f32,
}
impl Default for Camera2d {
    fn default() -> Self {
        Self {
            center: [0.0; 2],
            view_height: 640.0,
        }
    }
}
impl Camera2d {
    pub fn bounds(self, aspect: f32) -> Result<Bounds2d, SceneError> {
        if !self.center.iter().all(|v| v.is_finite())
            || !self.view_height.is_finite()
            || self.view_height <= 0.0
            || !aspect.is_finite()
            || aspect <= 0.0
        {
            return Err(SceneError(
                "camera and aspect must be finite with positive view size",
            ));
        }
        let half = Vec2::new(self.view_height * aspect, self.view_height) * 0.5;
        let center = Vec2::from_array(self.center);
        let min = center - half;
        let max = center + half;
        if !min.is_finite() || !max.is_finite() {
            return Err(SceneError("camera bounds overflow"));
        }
        Ok(Bounds2d {
            min: min.to_array(),
            max: max.to_array(),
        })
    }
}

/// A generic frame: empty by default, no implicit world or mandatory character.
/// Build afresh each tick (or clear instances), keeping reusable groups cached.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneFrame {
    pub camera: Camera2d,
    pub background: LinearRgba,
    /// Optional specialized live 3D knight. This is not a general model loader.
    pub knight: Option<KnightPose>,
    instances: Vec<GlyphInstance>,
}
impl Default for SceneFrame {
    fn default() -> Self {
        Self {
            camera: Camera2d::default(),
            background: LinearRgba::BLACK,
            knight: None,
            instances: Vec::new(),
        }
    }
}
impl SceneFrame {
    pub fn new(camera: Camera2d, background: LinearRgba) -> Self {
        Self {
            camera,
            background,
            ..Self::default()
        }
    }
    pub fn instances(&self) -> &[GlyphInstance] {
        &self.instances
    }
    pub fn clear(&mut self) {
        self.instances.clear();
    }
    /// Append one placement atomically. All quads share the supplied depth plus
    /// local order offsets. Geometry transforms do not implicitly change depth.
    pub fn add(
        &mut self,
        group: &GlyphGroup,
        transform: Transform2d,
        depth: Depth,
    ) -> Result<(), SceneError> {
        let quads = group.transformed(transform)?;
        let instances = quads
            .into_iter()
            .map(|quad| {
                let z = Depth::new(depth.0 - f32::from(quad.order) * 0.000001)?;
                let (kind, glyph) = match quad.primitive {
                    Primitive::Glyph(glyph) => (0, u32::from(glyph.0)),
                    Primitive::Solid => (1, 0),
                };
                Ok(GlyphInstance {
                    position: quad.position,
                    size: quad.size,
                    color: quad.color.0,
                    rotation: quad.rotation,
                    depth: z.0,
                    glyph,
                    kind,
                })
            })
            .collect::<Result<Vec<_>, SceneError>>()?;
        self.instances.extend(instances);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
