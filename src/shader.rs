// SPDX-License-Identifier: Apache-2.0
//! Reusable post-processing effects for character-cell canvases.
//!
//! Shaders run after an object has rendered and mutate the already composited
//! cells in place. They preserve each cell's depth, so a shader never lifts a
//! glyph above another object that was drawn later. Most effects are attached to
//! an object through [`ShadedObject`], but every shader can also be invoked
//! directly with [`Shader::render`] and a [`ShaderContext`].
//!
//! Effects use [`FrameInfo::elapsed`] rather than the host wall clock. The first
//! render of a timed effect starts its clock, which keeps playback deterministic
//! in tests and headless simulations. All pseudo-random choices are derived from
//! explicit seeds and integer hashing; no global RNG is used.
//!
//! # Example
//!
//! ```
//! use cosplay::{
//!     engine::{Engine, FrameInfo, Scene, SceneObject},
//!     graphics::{Canvas, Color, Pixel, Rect},
//!     shader::{FadeInShader, ShadedObject},
//! };
//! use std::time::Duration;
//!
//! struct Dot;
//! impl SceneObject for Dot {
//!     fn rect(&self) -> Rect { Rect::new(1, 0, 1, 1) }
//!     fn render(&self, canvas: &mut Canvas<'_>, _: &FrameInfo) {
//!         canvas.draw_pixel(1, 0, 0, Pixel::new('*', Color::WHITE, None));
//!     }
//! }
//!
//! let background = Pixel::new(' ', Color::BLACK, Some(Color::BLACK));
//! let mut scene = Scene::new("fade", background);
//! let mut object = ShadedObject::new(Dot);
//! object.add_shader(FadeInShader::new(100, background));
//! scene.add_object(object);
//!
//! let mut engine = Engine::new(4, 1);
//! engine.add_scene(scene)?;
//! engine.start("fade")?;
//! engine.tick(Duration::ZERO, &[])?;
//! assert_eq!(engine.screen().snapshot(), "    ");
//! # Ok::<(), cosplay::engine::EngineError>(())
//! ```
//!
//! A shader whose [`ShaderContext::object`] is empty falls back to
//! [`FrameInfo::camera`]. [`ShaderTarget`] makes that choice explicit for the
//! effects that support full-camera processing.
//!
//! [`ShadedObject`] applies its shaders from [`SceneObject::render`], which the
//! engine only calls for visible, on-camera objects. For a full-screen
//! post-process that must also run while its owner is off camera, invoke
//! [`apply_shaders`] from a custom [`SceneObject::shader`] implementation.

use crate::audio::BeatEnvelope;
use crate::engine::{FrameInfo, SceneObject};
use crate::graphics::{Canvas, Cell, Color, Pixel, Rect};
use std::cell::RefCell;
use std::time::Duration;

/// The largest per-cell shimmer palette that is cached. Larger targets fall
/// back to deterministic per-cell hashing so a shader cannot allocate without
/// bound.
const MAX_SHIMMER_CACHE_CELLS: usize = 262_144;

/// A hard cap on stars in one streak layer.
const MAX_STARS_PER_LAYER: usize = 1_024;

/// Fallback colors used by effects that accept a palette.
const DEFAULT_COLORS: [Color; 3] = [
    Color::WHITE,
    Color::rgb(0x9a, 0xd8, 0xff),
    Color::rgb(0xff, 0xe0, 0x9a),
];

/// Fallback characters used by randomized effects.
const DEFAULT_CHARS: [char; 4] = ['#', '*', '+', '.'];

/// A deterministic easing curve for timed shaders.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Easing {
    /// Constant speed.
    #[default]
    Linear,
    /// Smoothstep (`t * t * (3 - 2t)`).
    Smooth,
    /// Accelerating quadratic curve (`t * t`).
    Quadratic,
}

impl Easing {
    /// Applies the curve to a clamped progress value.
    pub fn apply(self, progress: f32) -> f32 {
        let t = if progress.is_nan() {
            0.0
        } else {
            progress.clamp(0.0, 1.0)
        };
        match self {
            Self::Linear => t,
            Self::Smooth => t * t * (3.0 - 2.0 * t),
            Self::Quadratic => t * t,
        }
    }
}

/// Selects the rectangle an effect processes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShaderTarget {
    /// The owning object's rectangle, falling back to the camera when empty.
    #[default]
    Object,
    /// The current camera viewport, falling back to the object when the camera
    /// has no area.
    Camera,
}

impl ShaderTarget {
    /// Resolves this target against a context and the canvas's active clip.
    pub fn rect(self, ctx: &ShaderContext<'_>, canvas: &Canvas<'_>) -> Rect {
        let base = match self {
            Self::Object => ctx.object_or_camera(),
            Self::Camera if !ctx.frame.camera.is_empty() => ctx.frame.camera,
            Self::Camera => ctx.object_or_camera(),
        };
        visible_rect(canvas, base)
    }

    fn should_render(self, ctx: &ShaderContext<'_>) -> bool {
        match self {
            Self::Object => ctx.in_camera || ctx.object.is_empty(),
            Self::Camera => ctx.in_camera || !ctx.frame.camera.is_empty(),
        }
    }
}

/// Per-callback information supplied to a [`Shader`].
#[derive(Clone, Copy, Debug)]
pub struct ShaderContext<'a> {
    /// The frame being shaded.
    pub frame: &'a FrameInfo,
    /// The owning object's rectangle. An empty rectangle means "the camera".
    pub object: Rect,
    /// Whether the owning object intersects the camera viewport.
    pub in_camera: bool,
}

impl<'a> ShaderContext<'a> {
    pub const fn new(frame: &'a FrameInfo, object: Rect, in_camera: bool) -> Self {
        Self {
            frame,
            object,
            in_camera,
        }
    }

    /// The object rectangle, or the camera when the object rectangle is empty.
    pub fn object_or_camera(&self) -> Rect {
        if !self.object.is_empty() {
            self.object
        } else if !self.frame.camera.is_empty() {
            self.frame.camera
        } else {
            Rect::new(0, 0, self.frame.width, self.frame.height)
        }
    }

    /// The context rectangle intersected with the canvas bounds and clip.
    pub fn target_rect(&self, canvas: &Canvas<'_>) -> Rect {
        visible_rect(canvas, self.object_or_camera())
    }
}

/// A reusable canvas post-processing effect.
pub trait Shader {
    /// Applies the effect. Implementations must preserve the existing cell
    /// depth and must not draw outside the canvas's current clip.
    fn render(&mut self, canvas: &mut Canvas<'_>, ctx: &ShaderContext<'_>);

    /// Whether this shader should currently run.
    fn is_active(&self) -> bool {
        true
    }

    /// Restores the shader to its initial state. The default is a no-op.
    fn reset(&mut self) {}
}

impl Shader for Box<dyn Shader> {
    fn render(&mut self, canvas: &mut Canvas<'_>, ctx: &ShaderContext<'_>) {
        (**self).render(canvas, ctx);
    }

    fn is_active(&self) -> bool {
        (**self).is_active()
    }

    fn reset(&mut self) {
        (**self).reset();
    }
}

/// Runs every active shader in order. Inactive shaders are skipped.
pub fn apply_shaders(
    shaders: &mut [Box<dyn Shader>],
    canvas: &mut Canvas<'_>,
    ctx: &ShaderContext<'_>,
) {
    for shader in shaders {
        if shader.is_active() {
            shader.render(canvas, ctx);
        }
    }
}

#[derive(Clone, Debug)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: splitmix64(seed),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        splitmix64(self.state)
    }

    fn next_f32(&mut self) -> f32 {
        // Keep 24 bits so every value is exactly representable in f32.
        (self.next_u64() >> 40) as f32 / 16_777_216.0
    }

    fn range_f32(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn hash_cell(x: i32, y: i32, seed: u64) -> u64 {
    let mut value = seed;
    value ^= (i64::from(x) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    value ^= (i64::from(y) as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    splitmix64(value)
}

fn hash_fraction(x: i32, y: i32, seed: u64) -> f32 {
    let bits = hash_cell(x, y, seed) >> 40;
    bits as f32 / 16_777_216.0
}

fn pick_color(colors: &[Color], index: u64) -> Color {
    if colors.is_empty() {
        DEFAULT_COLORS[index as usize % DEFAULT_COLORS.len()]
    } else {
        colors[index as usize % colors.len()]
    }
}

fn pick_char(chars: &[char], index: u64) -> char {
    if chars.is_empty() {
        DEFAULT_CHARS[index as usize % DEFAULT_CHARS.len()]
    } else {
        chars[index as usize % chars.len()]
    }
}

fn elapsed_ms(elapsed: Duration) -> u64 {
    elapsed
        .as_secs()
        .saturating_mul(1_000)
        .saturating_add(u64::from(elapsed.subsec_millis()))
}

fn visible_rect(canvas: &Canvas<'_>, rect: Rect) -> Rect {
    rect.intersection(canvas.bounds())
        .and_then(|rect| rect.intersection(canvas.current_clip()))
        .unwrap_or(Rect::new(0, 0, 0, 0))
}

fn for_each_visible_cell(
    canvas: &mut Canvas<'_>,
    rect: Rect,
    mut apply: impl FnMut(&mut Canvas<'_>, i32, i32, Cell),
) {
    let visible = visible_rect(canvas, rect);
    if visible.is_empty() {
        return;
    }
    for y in i64::from(visible.y)..visible.bottom() {
        for x in i64::from(visible.x)..visible.right() {
            let (x, y) = (x as i32, y as i32);
            if let Some(cell) = canvas.get(x, y) {
                apply(canvas, x, y, cell);
            }
        }
    }
}

fn write_pixel(canvas: &mut Canvas<'_>, x: i32, y: i32, cell: Cell, pixel: Pixel) {
    if !pixel.is_transparent() {
        canvas.draw_cell(x, y, cell.z, Cell { pixel, z: cell.z });
    }
}

fn hidden_pixel(background: Pixel, original: Pixel) -> Pixel {
    let bg = background.bg.or(original.bg);
    let ch = if background.ch == '\0' {
        ' '
    } else {
        background.ch
    };
    Pixel::new(ch, background.fg, bg)
}

fn visible_pixel(pixel: Pixel, fallback: Pixel) -> Pixel {
    if pixel.is_transparent() {
        hidden_pixel(pixel, fallback)
    } else {
        pixel
    }
}

fn hide_cell(canvas: &mut Canvas<'_>, x: i32, y: i32, cell: Cell, background: Pixel) {
    write_pixel(canvas, x, y, cell, hidden_pixel(background, cell.pixel));
}

fn mix_background(from: Option<Color>, to: Option<Color>, progress: f32) -> Option<Color> {
    match (from, to) {
        (Some(from), Some(to)) => Some(from.mixture(to, progress)),
        (Some(from), None) if progress < 1.0 => Some(from),
        (None, Some(to)) if progress > 0.0 => Some(to),
        _ => None,
    }
}

/// Interpolates one cell from `background` (progress zero) to its rendered
/// appearance (progress one), or the reverse for a fade-out.
fn fade_pixel(original: Pixel, background: Pixel, progress: f32, fade_out: bool) -> Pixel {
    let progress = if progress.is_nan() {
        0.0
    } else {
        progress.clamp(0.0, 1.0)
    };
    let (from, to) = if fade_out {
        (original, background)
    } else {
        (background, original)
    };
    if progress <= 0.0 {
        return visible_pixel(from, original);
    }
    if progress >= 1.0 {
        return visible_pixel(to, original);
    }
    Pixel::new(
        original.ch,
        from.fg.mixture(to.fg, progress),
        mix_background(from.bg, to.bg, progress),
    )
}

#[derive(Clone, Debug)]
struct TimedState {
    active: bool,
    started_at: Option<Duration>,
    duration: Duration,
    easing: Easing,
    progress: f32,
    done: bool,
}

impl TimedState {
    fn new(duration_ms: u64, easing: Easing) -> Self {
        Self {
            active: true,
            started_at: None,
            duration: Duration::from_millis(duration_ms),
            easing,
            progress: 0.0,
            done: false,
        }
    }

    fn progress_at(&mut self, now: Duration) -> f32 {
        if self.duration.is_zero() {
            self.progress = 1.0;
            self.done = true;
            return 1.0;
        }
        let start = *self.started_at.get_or_insert(now);
        let elapsed = now.saturating_sub(start);
        let raw = (elapsed.as_secs_f64() / self.duration.as_secs_f64()).clamp(0.0, 1.0) as f32;
        self.progress = self.easing.apply(raw);
        if raw >= 1.0 {
            self.done = true;
        }
        self.progress
    }

    fn start(&mut self) {
        self.active = true;
        self.started_at = None;
        self.progress = 0.0;
        self.done = false;
    }

    fn stop(&mut self) {
        self.active = false;
    }

    fn set_active(&mut self, active: bool) {
        if active {
            if !self.active {
                self.start();
            }
        } else {
            self.active = false;
        }
    }

    fn reset(&mut self) {
        self.started_at = None;
        self.progress = 0.0;
        self.done = false;
    }

    fn set_duration_ms(&mut self, duration_ms: u64) {
        self.duration = Duration::from_millis(duration_ms);
    }
}

/// Fades a rendered object in from a background pixel.
///
/// At progress zero the `background` pixel is shown; at progress one the original
/// cells are left untouched. In between, the foreground and background colors are
/// blended while the rendered glyph is preserved.
#[derive(Clone, Debug)]
pub struct FadeInShader {
    state: TimedState,
    background: Pixel,
    target: ShaderTarget,
}

impl FadeInShader {
    /// Creates a fade-in over `duration_ms`. `background` is the pixel shown at
    /// the start of the effect, normally the scene background.
    pub fn new(duration_ms: u64, background: Pixel) -> Self {
        Self {
            state: TimedState::new(duration_ms, Easing::Linear),
            background,
            target: ShaderTarget::Object,
        }
    }

    /// Alias emphasizing that `background` is the starting pixel.
    pub fn from_background(duration_ms: u64, background: Pixel) -> Self {
        Self::new(duration_ms, background)
    }

    /// Creates a fade-in from a solid background color.
    pub fn from_color(duration_ms: u64, color: Color) -> Self {
        Self::new(duration_ms, Pixel::new(' ', color, Some(color)))
    }

    pub fn with_background(mut self, background: Pixel) -> Self {
        self.background = background;
        self
    }

    pub fn with_easing(mut self, easing: Easing) -> Self {
        self.state.easing = easing;
        self
    }

    pub fn with_duration_ms(mut self, duration_ms: u64) -> Self {
        self.state.set_duration_ms(duration_ms);
        self
    }

    pub fn with_target(mut self, target: ShaderTarget) -> Self {
        self.target = target;
        self
    }

    pub fn background(&self) -> Pixel {
        self.background
    }

    pub fn set_background(&mut self, background: Pixel) {
        self.background = background;
    }

    pub fn duration_ms(&self) -> u64 {
        self.state.duration.as_millis().min(u128::from(u64::MAX)) as u64
    }

    pub fn set_duration_ms(&mut self, duration_ms: u64) {
        self.state.set_duration_ms(duration_ms);
    }

    pub fn easing(&self) -> Easing {
        self.state.easing
    }

    pub fn set_easing(&mut self, easing: Easing) {
        self.state.easing = easing;
    }

    pub fn target(&self) -> ShaderTarget {
        self.target
    }

    pub fn set_target(&mut self, target: ShaderTarget) {
        self.target = target;
    }

    pub fn start(&mut self) {
        self.state.start();
    }

    pub fn stop(&mut self) {
        self.state.stop();
    }

    pub fn set_active(&mut self, active: bool) {
        self.state.set_active(active);
    }

    pub fn is_active(&self) -> bool {
        self.state.active
    }

    pub fn reset(&mut self) {
        self.state.reset();
    }

    pub fn is_done(&self) -> bool {
        self.state.done
    }

    pub fn progress(&self) -> f32 {
        self.state.progress
    }
}

impl Shader for FadeInShader {
    fn render(&mut self, canvas: &mut Canvas<'_>, ctx: &ShaderContext<'_>) {
        if !self.state.active || !self.target.should_render(ctx) {
            return;
        }
        let progress = self.state.progress_at(ctx.frame.elapsed);
        if progress >= 1.0 {
            return;
        }
        let rect = self.target.rect(ctx, canvas);
        let background = self.background;
        for_each_visible_cell(canvas, rect, |canvas, x, y, cell| {
            write_pixel(
                canvas,
                x,
                y,
                cell,
                fade_pixel(cell.pixel, background, progress, false),
            );
        });
    }

    fn is_active(&self) -> bool {
        self.state.active
    }

    fn reset(&mut self) {
        self.state.reset();
    }
}

/// Fades a rendered object out toward a background pixel.
///
/// At progress zero the original cells are left untouched; at progress one the
/// `background` pixel is shown. In between, the colors are blended while the
/// rendered glyph is preserved.
#[derive(Clone, Debug)]
pub struct FadeOutShader {
    state: TimedState,
    background: Pixel,
    target: ShaderTarget,
}

impl FadeOutShader {
    /// Creates a fade-out over `duration_ms`. `background` is the pixel shown at
    /// the end of the effect, normally the scene background.
    pub fn new(duration_ms: u64, background: Pixel) -> Self {
        Self {
            state: TimedState::new(duration_ms, Easing::Linear),
            background,
            target: ShaderTarget::Object,
        }
    }

    /// Alias emphasizing that `background` is the destination pixel.
    pub fn to_background(duration_ms: u64, background: Pixel) -> Self {
        Self::new(duration_ms, background)
    }

    /// Creates a fade-out toward a solid background color.
    pub fn to_color(duration_ms: u64, color: Color) -> Self {
        Self::new(duration_ms, Pixel::new(' ', color, Some(color)))
    }

    pub fn with_background(mut self, background: Pixel) -> Self {
        self.background = background;
        self
    }

    pub fn with_easing(mut self, easing: Easing) -> Self {
        self.state.easing = easing;
        self
    }

    pub fn with_duration_ms(mut self, duration_ms: u64) -> Self {
        self.state.set_duration_ms(duration_ms);
        self
    }

    pub fn with_target(mut self, target: ShaderTarget) -> Self {
        self.target = target;
        self
    }

    pub fn background(&self) -> Pixel {
        self.background
    }

    pub fn set_background(&mut self, background: Pixel) {
        self.background = background;
    }

    pub fn duration_ms(&self) -> u64 {
        self.state.duration.as_millis().min(u128::from(u64::MAX)) as u64
    }

    pub fn set_duration_ms(&mut self, duration_ms: u64) {
        self.state.set_duration_ms(duration_ms);
    }

    pub fn easing(&self) -> Easing {
        self.state.easing
    }

    pub fn set_easing(&mut self, easing: Easing) {
        self.state.easing = easing;
    }

    pub fn target(&self) -> ShaderTarget {
        self.target
    }

    pub fn set_target(&mut self, target: ShaderTarget) {
        self.target = target;
    }

    pub fn start(&mut self) {
        self.state.start();
    }

    pub fn stop(&mut self) {
        self.state.stop();
    }

    pub fn set_active(&mut self, active: bool) {
        self.state.set_active(active);
    }

    pub fn is_active(&self) -> bool {
        self.state.active
    }

    pub fn reset(&mut self) {
        self.state.reset();
    }

    pub fn is_done(&self) -> bool {
        self.state.done
    }

    pub fn progress(&self) -> f32 {
        self.state.progress
    }
}

impl Shader for FadeOutShader {
    fn render(&mut self, canvas: &mut Canvas<'_>, ctx: &ShaderContext<'_>) {
        if !self.state.active || !self.target.should_render(ctx) {
            return;
        }
        let progress = self.state.progress_at(ctx.frame.elapsed);
        if progress <= 0.0 {
            return;
        }
        let rect = self.target.rect(ctx, canvas);
        let background = self.background;
        for_each_visible_cell(canvas, rect, |canvas, x, y, cell| {
            write_pixel(
                canvas,
                x,
                y,
                cell,
                fade_pixel(cell.pixel, background, progress, true),
            );
        });
    }

    fn is_active(&self) -> bool {
        self.state.active
    }

    fn reset(&mut self) {
        self.state.reset();
    }
}

/// A fade-in that also scrambles glyphs deterministically.
///
/// The substitution salt is refreshed every `refresh_frames` engine frames,
/// giving the effect its characteristic flicker. Two shaders with the same seed
/// and frame sequence produce identical output.
#[derive(Clone, Debug)]
pub struct RandomFadeInShader {
    state: TimedState,
    background: Pixel,
    chars: Vec<char>,
    refresh_frames: u64,
    seed: u64,
    rng: Rng,
    salt: u64,
    epoch: Option<u64>,
    target: ShaderTarget,
}

impl RandomFadeInShader {
    /// Creates a randomized fade-in using `chars` for substituted glyphs.
    pub fn new(duration_ms: u64, background: Pixel, chars: Vec<char>) -> Self {
        Self {
            state: TimedState::new(duration_ms, Easing::Linear),
            background,
            chars,
            refresh_frames: 2,
            seed: 0,
            rng: Rng::new(0),
            salt: 0,
            epoch: None,
            target: ShaderTarget::Object,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.rng = Rng::new(seed);
        self.salt = 0;
        self.epoch = None;
        self
    }

    pub fn with_refresh_frames(mut self, refresh_frames: u64) -> Self {
        self.refresh_frames = refresh_frames.max(1);
        self
    }

    pub fn with_easing(mut self, easing: Easing) -> Self {
        self.state.easing = easing;
        self
    }

    pub fn with_duration_ms(mut self, duration_ms: u64) -> Self {
        self.state.set_duration_ms(duration_ms);
        self
    }

    pub fn with_target(mut self, target: ShaderTarget) -> Self {
        self.target = target;
        self
    }

    pub fn duration_ms(&self) -> u64 {
        self.state.duration.as_millis().min(u128::from(u64::MAX)) as u64
    }

    pub fn set_duration_ms(&mut self, duration_ms: u64) {
        self.state.set_duration_ms(duration_ms);
    }

    pub fn easing(&self) -> Easing {
        self.state.easing
    }

    pub fn set_easing(&mut self, easing: Easing) {
        self.state.easing = easing;
    }

    pub fn target(&self) -> ShaderTarget {
        self.target
    }

    pub fn set_target(&mut self, target: ShaderTarget) {
        self.target = target;
    }

    pub fn chars(&self) -> &[char] {
        &self.chars
    }

    pub fn set_chars(&mut self, chars: Vec<char>) {
        self.chars = chars;
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.reset();
    }

    pub fn refresh_frames(&self) -> u64 {
        self.refresh_frames
    }

    pub fn set_refresh_frames(&mut self, refresh_frames: u64) {
        self.refresh_frames = refresh_frames.max(1);
        self.epoch = None;
    }

    pub fn with_background(mut self, background: Pixel) -> Self {
        self.background = background;
        self
    }

    pub fn background(&self) -> Pixel {
        self.background
    }

    pub fn set_background(&mut self, background: Pixel) {
        self.background = background;
    }

    pub fn start(&mut self) {
        self.state.start();
    }

    pub fn stop(&mut self) {
        self.state.stop();
    }

    pub fn set_active(&mut self, active: bool) {
        self.state.set_active(active);
    }

    pub fn is_active(&self) -> bool {
        self.state.active
    }

    pub fn reset(&mut self) {
        self.state.reset();
        self.rng = Rng::new(self.seed);
        self.salt = 0;
        self.epoch = None;
    }

    pub fn is_done(&self) -> bool {
        self.state.done
    }

    pub fn progress(&self) -> f32 {
        self.state.progress
    }
}

impl Shader for RandomFadeInShader {
    fn render(&mut self, canvas: &mut Canvas<'_>, ctx: &ShaderContext<'_>) {
        if !self.state.active || !self.target.should_render(ctx) {
            return;
        }
        let progress = self.state.progress_at(ctx.frame.elapsed);
        if progress >= 1.0 {
            return;
        }
        let refresh = self.refresh_frames.max(1);
        let epoch = ctx.frame.frame / refresh;
        if self.epoch != Some(epoch) {
            self.epoch = Some(epoch);
            self.salt = self.rng.next_u64();
        }
        let rect = self.target.rect(ctx, canvas);
        let background = self.background;
        let chars = &self.chars;
        let seed = self.seed ^ self.salt;
        for_each_visible_cell(canvas, rect, |canvas, x, y, cell| {
            if progress <= 0.0 {
                hide_cell(canvas, x, y, cell, background);
                return;
            }
            let mut pixel = fade_pixel(cell.pixel, background, progress, false);
            if !cell.pixel.ch.is_whitespace() && hash_fraction(x, y, seed) > progress {
                let index = hash_cell(x, y, seed) >> 32;
                pixel.ch = pick_char(chars, index);
            }
            write_pixel(canvas, x, y, cell, pixel);
        });
    }

    fn is_active(&self) -> bool {
        self.state.active
    }

    fn reset(&mut self) {
        RandomFadeInShader::reset(self);
    }
}

/// The reveal direction used by [`SlideInShader`] and [`SlideOutShader`].
///
/// The directional modes map cell coordinates to a threshold in `0.0..1.0`.
/// A slide-in reveals cells whose threshold is below the current progress; a
/// slide-out hides cells whose threshold is at or below it. For a slide-in, the
/// collapse modes reveal from the edges inward and the expand modes from the
/// center outward; a slide-out reverses those orders.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SlideDirection {
    #[default]
    Left,
    Right,
    Top,
    Bottom,
    HorizontalCollapse,
    HorizontalExpand,
    VerticalCollapse,
    VerticalExpand,
    Centrifugal,
    Centripetal,
    RandomPixels,
    RandomVerticalLines,
    RandomHorizontalLines,
}

impl SlideDirection {
    /// All thirteen modes in declaration order.
    pub const ALL: [Self; 13] = [
        Self::Left,
        Self::Right,
        Self::Top,
        Self::Bottom,
        Self::HorizontalCollapse,
        Self::HorizontalExpand,
        Self::VerticalCollapse,
        Self::VerticalExpand,
        Self::Centrifugal,
        Self::Centripetal,
        Self::RandomPixels,
        Self::RandomVerticalLines,
        Self::RandomHorizontalLines,
    ];

    /// Upstream alias for [`Self::VerticalCollapse`]: horizontal lines collapse
    /// from the top and bottom.
    pub const HOR_COLLAPSE: Self = Self::VerticalCollapse;
    /// Upstream alias for [`Self::VerticalExpand`]: horizontal lines expand from
    /// the center toward the top and bottom.
    pub const HOR_EXPAND: Self = Self::VerticalExpand;
    /// Upstream alias for [`Self::HorizontalCollapse`]: vertical lines collapse
    /// from the left and right.
    pub const VER_COLLAPSE: Self = Self::HorizontalCollapse;
    /// Upstream alias for [`Self::HorizontalExpand`]: vertical lines expand from
    /// the center toward the left and right.
    pub const VER_EXPAND: Self = Self::HorizontalExpand;
    /// Alias for [`Self::Left`].
    pub const LEFT_TO_RIGHT: Self = Self::Left;
    /// Alias for [`Self::Right`].
    pub const RIGHT_TO_LEFT: Self = Self::Right;
    /// Alias for [`Self::Top`].
    pub const TOP_TO_BOTTOM: Self = Self::Top;
    /// Alias for [`Self::Bottom`].
    pub const BOTTOM_TO_TOP: Self = Self::Bottom;
    /// Alias for [`Self::Centrifugal`].
    pub const CENTRIFUGAL: Self = Self::Centrifugal;
    /// Alias for [`Self::Centripetal`].
    pub const CENTRIPETAL: Self = Self::Centripetal;
    /// Alias for [`Self::RandomPixels`].
    pub const RANDOM: Self = Self::RandomPixels;
    /// Alias for [`Self::RandomVerticalLines`].
    pub const RANDOM_VERT_LINE: Self = Self::RandomVerticalLines;
    /// Alias for [`Self::RandomHorizontalLines`].
    pub const RANDOM_HOR_LINE: Self = Self::RandomHorizontalLines;
}

fn slide_threshold(
    direction: SlideDirection,
    local_x: i32,
    local_y: i32,
    width: u16,
    height: u16,
    seed: u64,
) -> f32 {
    let w = f32::from(width.max(1));
    let h = f32::from(height.max(1));
    let nx = (local_x as f32 + 0.5) / w;
    let ny = (local_y as f32 + 0.5) / h;
    let centered_x = (nx * 2.0 - 1.0).abs();
    let centered_y = (ny * 2.0 - 1.0).abs();
    let threshold = match direction {
        SlideDirection::Left => nx,
        SlideDirection::Right => 1.0 - nx,
        SlideDirection::Top => ny,
        SlideDirection::Bottom => 1.0 - ny,
        SlideDirection::HorizontalCollapse => 1.0 - centered_x,
        SlideDirection::HorizontalExpand => centered_x,
        SlideDirection::VerticalCollapse => 1.0 - centered_y,
        SlideDirection::VerticalExpand => centered_y,
        SlideDirection::Centrifugal => {
            ((centered_x * centered_x + centered_y * centered_y) / 2.0).sqrt()
        }
        SlideDirection::Centripetal => {
            1.0 - ((centered_x * centered_x + centered_y * centered_y) / 2.0).sqrt()
        }
        SlideDirection::RandomPixels => hash_fraction(local_x, local_y, seed),
        SlideDirection::RandomVerticalLines => hash_fraction(local_x, 0, seed),
        SlideDirection::RandomHorizontalLines => hash_fraction(0, local_y, seed),
    };
    threshold.clamp(0.0, 0.999_999)
}

/// Slides a rendered object into view according to a [`SlideDirection`].
///
/// Hidden cells are replaced by the configured background pixel before their
/// reveal threshold is reached. Cell depth is preserved.
#[derive(Clone, Debug)]
pub struct SlideInShader {
    state: TimedState,
    direction: SlideDirection,
    background: Pixel,
    seed: u64,
    target: ShaderTarget,
}

impl SlideInShader {
    pub fn new(duration_ms: u64, direction: SlideDirection) -> Self {
        Self {
            state: TimedState::new(duration_ms, Easing::Linear),
            direction,
            background: Pixel::transparent(),
            seed: 0,
            target: ShaderTarget::Object,
        }
    }

    pub fn with_background(mut self, background: Pixel) -> Self {
        self.background = background;
        self
    }

    pub fn with_easing(mut self, easing: Easing) -> Self {
        self.state.easing = easing;
        self
    }

    pub fn with_duration_ms(mut self, duration_ms: u64) -> Self {
        self.state.set_duration_ms(duration_ms);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_target(mut self, target: ShaderTarget) -> Self {
        self.target = target;
        self
    }

    pub fn direction(&self) -> SlideDirection {
        self.direction
    }

    pub fn set_direction(&mut self, direction: SlideDirection) {
        self.direction = direction;
    }

    pub fn duration_ms(&self) -> u64 {
        self.state.duration.as_millis().min(u128::from(u64::MAX)) as u64
    }

    pub fn set_duration_ms(&mut self, duration_ms: u64) {
        self.state.set_duration_ms(duration_ms);
    }

    pub fn easing(&self) -> Easing {
        self.state.easing
    }

    pub fn set_easing(&mut self, easing: Easing) {
        self.state.easing = easing;
    }

    pub fn background(&self) -> Pixel {
        self.background
    }

    pub fn set_background(&mut self, background: Pixel) {
        self.background = background;
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    pub fn target(&self) -> ShaderTarget {
        self.target
    }

    pub fn set_target(&mut self, target: ShaderTarget) {
        self.target = target;
    }

    pub fn start(&mut self) {
        self.state.start();
    }

    pub fn stop(&mut self) {
        self.state.stop();
    }

    pub fn set_active(&mut self, active: bool) {
        self.state.set_active(active);
    }

    pub fn is_active(&self) -> bool {
        self.state.active
    }

    pub fn reset(&mut self) {
        self.state.reset();
    }

    pub fn is_done(&self) -> bool {
        self.state.done
    }

    pub fn progress(&self) -> f32 {
        self.state.progress
    }
}

impl Shader for SlideInShader {
    fn render(&mut self, canvas: &mut Canvas<'_>, ctx: &ShaderContext<'_>) {
        if !self.state.active || !self.target.should_render(ctx) {
            return;
        }
        let progress = self.state.progress_at(ctx.frame.elapsed);
        if progress >= 1.0 {
            return;
        }
        let rect = self.target.rect(ctx, canvas);
        let background = self.background;
        let direction = self.direction;
        let seed = self.seed;
        for_each_visible_cell(canvas, rect, |canvas, x, y, cell| {
            let local_x = x.wrapping_sub(rect.x);
            let local_y = y.wrapping_sub(rect.y);
            let threshold =
                slide_threshold(direction, local_x, local_y, rect.width, rect.height, seed);
            if progress <= 0.0 || progress <= threshold {
                hide_cell(canvas, x, y, cell, background);
            }
        });
    }

    fn is_active(&self) -> bool {
        self.state.active
    }

    fn reset(&mut self) {
        self.state.reset();
    }
}

/// Slides a rendered object out of view according to a [`SlideDirection`].
///
/// Cells are hidden once their reveal threshold is passed. Hidden cells use the
/// configured background pixel, and cell depth is preserved.
#[derive(Clone, Debug)]
pub struct SlideOutShader {
    state: TimedState,
    direction: SlideDirection,
    background: Pixel,
    seed: u64,
    target: ShaderTarget,
}

impl SlideOutShader {
    pub fn new(duration_ms: u64, direction: SlideDirection) -> Self {
        Self {
            state: TimedState::new(duration_ms, Easing::Linear),
            direction,
            background: Pixel::transparent(),
            seed: 0,
            target: ShaderTarget::Object,
        }
    }

    pub fn with_background(mut self, background: Pixel) -> Self {
        self.background = background;
        self
    }

    pub fn with_easing(mut self, easing: Easing) -> Self {
        self.state.easing = easing;
        self
    }

    pub fn with_duration_ms(mut self, duration_ms: u64) -> Self {
        self.state.set_duration_ms(duration_ms);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_target(mut self, target: ShaderTarget) -> Self {
        self.target = target;
        self
    }

    pub fn direction(&self) -> SlideDirection {
        self.direction
    }

    pub fn set_direction(&mut self, direction: SlideDirection) {
        self.direction = direction;
    }

    pub fn duration_ms(&self) -> u64 {
        self.state.duration.as_millis().min(u128::from(u64::MAX)) as u64
    }

    pub fn set_duration_ms(&mut self, duration_ms: u64) {
        self.state.set_duration_ms(duration_ms);
    }

    pub fn easing(&self) -> Easing {
        self.state.easing
    }

    pub fn set_easing(&mut self, easing: Easing) {
        self.state.easing = easing;
    }

    pub fn background(&self) -> Pixel {
        self.background
    }

    pub fn set_background(&mut self, background: Pixel) {
        self.background = background;
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    pub fn target(&self) -> ShaderTarget {
        self.target
    }

    pub fn set_target(&mut self, target: ShaderTarget) {
        self.target = target;
    }

    pub fn start(&mut self) {
        self.state.start();
    }

    pub fn stop(&mut self) {
        self.state.stop();
    }

    pub fn set_active(&mut self, active: bool) {
        self.state.set_active(active);
    }

    pub fn is_active(&self) -> bool {
        self.state.active
    }

    pub fn reset(&mut self) {
        self.state.reset();
    }

    pub fn is_done(&self) -> bool {
        self.state.done
    }

    pub fn progress(&self) -> f32 {
        self.state.progress
    }
}

impl Shader for SlideOutShader {
    fn render(&mut self, canvas: &mut Canvas<'_>, ctx: &ShaderContext<'_>) {
        if !self.state.active || !self.target.should_render(ctx) {
            return;
        }
        let progress = self.state.progress_at(ctx.frame.elapsed);
        if progress <= 0.0 {
            return;
        }
        let rect = self.target.rect(ctx, canvas);
        let background = self.background;
        let direction = self.direction;
        let seed = self.seed;
        for_each_visible_cell(canvas, rect, |canvas, x, y, cell| {
            let local_x = x.wrapping_sub(rect.x);
            let local_y = y.wrapping_sub(rect.y);
            let threshold =
                slide_threshold(direction, local_x, local_y, rect.width, rect.height, seed);
            if progress >= threshold {
                hide_cell(canvas, x, y, cell, background);
            }
        });
    }

    fn is_active(&self) -> bool {
        self.state.active
    }

    fn reset(&mut self) {
        self.state.reset();
    }
}

/// Randomly recolors non-space glyphs, refreshing the palette on a fixed
/// keyframe interval.
///
/// The per-cell colors for a keyframe are cached when the target is small
/// enough. Larger targets use the same seeded hash directly, so results remain
/// deterministic without an unbounded allocation.
#[derive(Clone, Debug)]
pub struct ShimmerShader {
    seed: u64,
    keyframe_ms: u64,
    colors: Vec<Color>,
    target: ShaderTarget,
    active: bool,
    rng: Rng,
    cache_key: Option<u64>,
    cache_rect: Rect,
    cache_colors: Vec<Color>,
}

impl ShimmerShader {
    pub fn new(seed: u64, keyframe_ms: u64, colors: Vec<Color>) -> Self {
        Self {
            seed,
            keyframe_ms: keyframe_ms.max(1),
            colors,
            target: ShaderTarget::Object,
            active: true,
            rng: Rng::new(seed),
            cache_key: None,
            cache_rect: Rect::new(0, 0, 0, 0),
            cache_colors: Vec::new(),
        }
    }

    /// Builds a shimmer for an explicit target in one call.
    pub fn for_target(
        seed: u64,
        keyframe_ms: u64,
        colors: Vec<Color>,
        target: ShaderTarget,
    ) -> Self {
        Self::new(seed, keyframe_ms, colors).with_target(target)
    }

    pub fn with_target(mut self, target: ShaderTarget) -> Self {
        self.target = target;
        self
    }

    pub fn with_keyframe_ms(mut self, keyframe_ms: u64) -> Self {
        self.keyframe_ms = keyframe_ms.max(1);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.rng = Rng::new(seed);
        self.cache_key = None;
        self
    }

    pub fn with_colors(mut self, colors: Vec<Color>) -> Self {
        self.colors = colors;
        self.cache_key = None;
        self
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.reset();
    }

    pub fn keyframe_ms(&self) -> u64 {
        self.keyframe_ms
    }

    pub fn set_keyframe_ms(&mut self, keyframe_ms: u64) {
        self.keyframe_ms = keyframe_ms.max(1);
        self.cache_key = None;
    }

    pub fn colors(&self) -> &[Color] {
        &self.colors
    }

    pub fn set_colors(&mut self, colors: Vec<Color>) {
        self.colors = colors;
        self.cache_key = None;
    }

    pub fn target(&self) -> ShaderTarget {
        self.target
    }

    pub fn set_target(&mut self, target: ShaderTarget) {
        self.target = target;
        self.cache_key = None;
    }

    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn reset(&mut self) {
        self.rng = Rng::new(self.seed);
        self.cache_key = None;
        self.cache_rect = Rect::new(0, 0, 0, 0);
        self.cache_colors.clear();
    }

    fn refresh_cache(&mut self, rect: Rect, key: u64) {
        let area = usize::from(rect.width).saturating_mul(usize::from(rect.height));
        self.cache_colors.clear();
        if area > MAX_SHIMMER_CACHE_CELLS {
            self.cache_key = None;
            return;
        }
        self.cache_colors.reserve(area);
        for _ in 0..area {
            let index = self.rng.next_u64();
            self.cache_colors.push(pick_color(&self.colors, index));
        }
        self.cache_key = Some(key);
        self.cache_rect = rect;
    }

    fn color_for(&self, rect: Rect, key: u64, x: i32, y: i32) -> Color {
        let width = usize::from(rect.width);
        let local_x = x.wrapping_sub(rect.x) as usize;
        let local_y = y.wrapping_sub(rect.y) as usize;
        let area = width.saturating_mul(usize::from(rect.height));
        if self.cache_key == Some(key)
            && self.cache_rect == rect
            && area <= MAX_SHIMMER_CACHE_CELLS
            && self.cache_colors.len() == area
        {
            let index = local_y * width + local_x;
            self.cache_colors[index]
        } else {
            pick_color(&self.colors, hash_cell(x, y, self.seed ^ key))
        }
    }
}

impl Shader for ShimmerShader {
    fn render(&mut self, canvas: &mut Canvas<'_>, ctx: &ShaderContext<'_>) {
        if !self.active || !self.target.should_render(ctx) {
            return;
        }
        let rect = self.target.rect(ctx, canvas);
        if rect.is_empty() {
            return;
        }
        let key = elapsed_ms(ctx.frame.elapsed) / self.keyframe_ms.max(1);
        let area = usize::from(rect.width).saturating_mul(usize::from(rect.height));
        let cache_valid = self.cache_key == Some(key)
            && self.cache_rect == rect
            && area <= MAX_SHIMMER_CACHE_CELLS;
        if !cache_valid {
            self.refresh_cache(rect, key);
        }
        let use_cache = self.cache_key == Some(key)
            && self.cache_rect == rect
            && area <= MAX_SHIMMER_CACHE_CELLS;
        let colors = &self.colors;
        let seed = self.seed;
        for_each_visible_cell(canvas, rect, |canvas, x, y, cell| {
            if cell.pixel.is_transparent() || cell.pixel.ch.is_whitespace() {
                return;
            }
            let color = if use_cache {
                self.color_for(rect, key, x, y)
            } else {
                pick_color(colors, hash_cell(x, y, seed ^ key))
            };
            let pixel = Pixel {
                fg: color,
                ..cell.pixel
            };
            write_pixel(canvas, x, y, cell, pixel);
        });
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn reset(&mut self) {
        ShimmerShader::reset(self);
    }
}

/// Adds deterministic twinkling highlights to non-space glyphs.
///
/// Each candidate cell has an independent phase derived from the seed, so the
/// sparkle pattern remains stable across runs. `density` is the fraction of
/// non-space cells that participate, clamped to `0.0..=1.0`.
#[derive(Clone, Debug)]
pub struct SparkleShader {
    seed: u64,
    density: f32,
    lifetime_ms: u64,
    colors: Vec<Color>,
    target: ShaderTarget,
    active: bool,
    started_at: Option<Duration>,
}

impl SparkleShader {
    pub fn new(seed: u64, density: f32, colors: Vec<Color>) -> Self {
        Self {
            seed,
            density: if density.is_nan() {
                0.0
            } else {
                density.clamp(0.0, 1.0)
            },
            lifetime_ms: 1_000,
            colors,
            target: ShaderTarget::Object,
            active: true,
            started_at: None,
        }
    }

    /// Alias with the palette before the density.
    pub fn from_colors(seed: u64, colors: Vec<Color>, density: f32) -> Self {
        Self::new(seed, density, colors)
    }

    pub fn with_lifetime_ms(mut self, lifetime_ms: u64) -> Self {
        self.lifetime_ms = lifetime_ms.max(1);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.started_at = None;
        self
    }

    pub fn with_density(mut self, density: f32) -> Self {
        self.set_density(density);
        self
    }

    pub fn with_colors(mut self, colors: Vec<Color>) -> Self {
        self.colors = colors;
        self
    }

    pub fn with_target(mut self, target: ShaderTarget) -> Self {
        self.target = target;
        self
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.started_at = None;
    }

    pub fn density(&self) -> f32 {
        self.density
    }

    pub fn set_density(&mut self, density: f32) {
        self.density = if density.is_nan() {
            0.0
        } else {
            density.clamp(0.0, 1.0)
        };
    }

    pub fn lifetime_ms(&self) -> u64 {
        self.lifetime_ms
    }

    pub fn set_lifetime_ms(&mut self, lifetime_ms: u64) {
        self.lifetime_ms = lifetime_ms.max(1);
    }

    pub fn colors(&self) -> &[Color] {
        &self.colors
    }

    pub fn set_colors(&mut self, colors: Vec<Color>) {
        self.colors = colors;
    }

    pub fn target(&self) -> ShaderTarget {
        self.target
    }

    pub fn set_target(&mut self, target: ShaderTarget) {
        self.target = target;
    }

    pub fn start(&mut self) {
        self.active = true;
        self.started_at = None;
    }

    pub fn stop(&mut self) {
        self.active = false;
    }

    pub fn set_active(&mut self, active: bool) {
        if active && !self.active {
            self.started_at = None;
        }
        self.active = active;
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn reset(&mut self) {
        self.started_at = None;
    }
}

impl Shader for SparkleShader {
    fn render(&mut self, canvas: &mut Canvas<'_>, ctx: &ShaderContext<'_>) {
        if !self.active || !self.target.should_render(ctx) {
            return;
        }
        let start = *self.started_at.get_or_insert(ctx.frame.elapsed);
        let elapsed = elapsed_ms(ctx.frame.elapsed.saturating_sub(start)) as f32;
        let lifetime = self.lifetime_ms.max(1) as f32;
        let rect = self.target.rect(ctx, canvas);
        let density = self.density;
        let seed = self.seed;
        let colors = &self.colors;
        for_each_visible_cell(canvas, rect, |canvas, x, y, cell| {
            if cell.pixel.is_transparent() || cell.pixel.ch.is_whitespace() {
                return;
            }
            let hash = hash_cell(x, y, seed);
            let candidate = (hash >> 40) as f32 / 16_777_216.0;
            if candidate >= density {
                return;
            }
            let phase = (hash & 0xFFFF) as f32 / 65_536.0;
            let t = ((elapsed / lifetime) + phase).rem_euclid(1.0);
            let brightness = 1.0 - (2.0 * t - 1.0).abs();
            if brightness <= 0.0 {
                return;
            }
            let color = pick_color(colors, hash >> 16);
            let pixel = Pixel {
                fg: cell.pixel.fg.mixture(color, brightness),
                ..cell.pixel
            };
            write_pixel(canvas, x, y, cell, pixel);
        });
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn reset(&mut self) {
        SparkleShader::reset(self);
    }
}

/// Shades cells according to their distance from the target's nearest border.
///
/// `aspect` is the cell height divided by its width and defaults to `2.0`, which
/// makes vertical and horizontal distances comparable on a typical terminal.
/// At the border the outer color dominates; the tint fades toward the inner
/// color and then to no effect at `max_distance`.
#[derive(Clone, Debug)]
pub struct BorderShader {
    border: Color,
    inner: Color,
    max_distance: f32,
    aspect: f32,
    target: ShaderTarget,
    active: bool,
}

impl BorderShader {
    pub fn new(border: Color, inner: Color, max_distance: f32) -> Self {
        Self {
            border,
            inner,
            max_distance: if max_distance.is_nan() {
                1.0
            } else {
                max_distance.max(0.0)
            },
            aspect: 2.0,
            target: ShaderTarget::Object,
            active: true,
        }
    }

    /// Creates a border shader from an `[outer, inner]` color pair.
    pub fn from_colors(colors: [Color; 2], max_distance: f32) -> Self {
        Self::new(colors[0], colors[1], max_distance)
    }

    pub fn with_aspect(mut self, aspect: f32) -> Self {
        self.set_aspect(aspect);
        self
    }

    pub fn with_colors(mut self, border: Color, inner: Color) -> Self {
        self.set_colors(border, inner);
        self
    }

    pub fn with_max_distance(mut self, max_distance: f32) -> Self {
        self.set_max_distance(max_distance);
        self
    }

    pub fn with_target(mut self, target: ShaderTarget) -> Self {
        self.target = target;
        self
    }

    pub fn border_color(&self) -> Color {
        self.border
    }

    pub fn inner_color(&self) -> Color {
        self.inner
    }

    pub fn set_colors(&mut self, border: Color, inner: Color) {
        self.border = border;
        self.inner = inner;
    }

    pub fn max_distance(&self) -> f32 {
        self.max_distance
    }

    pub fn set_max_distance(&mut self, max_distance: f32) {
        self.max_distance = if max_distance.is_nan() {
            1.0
        } else {
            max_distance.max(0.0)
        };
    }

    pub fn aspect(&self) -> f32 {
        self.aspect
    }

    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = if aspect.is_finite() && aspect > 0.0 {
            aspect
        } else {
            2.0
        };
    }

    pub fn target(&self) -> ShaderTarget {
        self.target
    }

    pub fn set_target(&mut self, target: ShaderTarget) {
        self.target = target;
    }

    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn reset(&mut self) {}
}

impl Shader for BorderShader {
    fn render(&mut self, canvas: &mut Canvas<'_>, ctx: &ShaderContext<'_>) {
        if !self.active || !self.target.should_render(ctx) {
            return;
        }
        let rect = self.target.rect(ctx, canvas);
        if rect.is_empty() {
            return;
        }
        let border = self.border;
        let inner = self.inner;
        let max_distance = self.max_distance;
        let aspect = self.aspect;
        for_each_visible_cell(canvas, rect, |canvas, x, y, cell| {
            if cell.pixel.is_transparent() {
                return;
            }
            let local_x = i64::from(x) - i64::from(rect.x);
            let local_y = i64::from(y) - i64::from(rect.y);
            let right = i64::from(rect.width) - 1;
            let bottom = i64::from(rect.height) - 1;
            let dx = local_x.min(right - local_x) as f32;
            let dy = (local_y.min(bottom - local_y) as f32) * aspect;
            let distance = dx.min(dy);
            if distance > max_distance {
                return;
            }
            let t = if max_distance > 0.0 {
                (distance / max_distance).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let weight = 1.0 - t;
            if weight <= 0.0 {
                return;
            }
            let shade = border.mixture(inner, t);
            let background = cell.pixel.bg.unwrap_or(Color::BLACK).mixture(shade, weight);
            let pixel = Pixel {
                fg: cell.pixel.fg.mixture(shade, weight),
                bg: Some(background),
                ..cell.pixel
            };
            write_pixel(canvas, x, y, cell, pixel);
        });
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn reset(&mut self) {
        BorderShader::reset(self);
    }
}

/// A radial light that brightens non-space glyphs near a focus point.
///
/// With no explicit focus the center of the target rectangle is used. Focus
/// coordinates are absolute canvas coordinates. `aspect` corrects vertical
/// distances in the same way as [`BorderShader`].
#[derive(Clone, Debug)]
pub struct FlashlightShader {
    color: Color,
    outer_color: Color,
    inner_radius: f32,
    outer_radius: f32,
    aspect: f32,
    focus: Option<(f32, f32)>,
    target: ShaderTarget,
    active: bool,
}

impl FlashlightShader {
    pub fn new(color: Color, inner_radius: f32, outer_radius: f32) -> Self {
        let inner_radius = if inner_radius.is_nan() {
            0.0
        } else {
            inner_radius.max(0.0)
        };
        let outer_radius = if outer_radius.is_nan() || outer_radius <= inner_radius {
            inner_radius + 1.0
        } else {
            outer_radius
        };
        Self {
            color,
            outer_color: color,
            inner_radius,
            outer_radius,
            aspect: 2.0,
            focus: None,
            target: ShaderTarget::Object,
            active: true,
        }
    }

    /// Creates a light that blends from `inner_color` at the focus toward
    /// `outer_color` at the edge of the falloff.
    pub fn from_colors(
        inner_color: Color,
        outer_color: Color,
        inner_radius: f32,
        outer_radius: f32,
    ) -> Self {
        let mut shader = Self::new(inner_color, inner_radius, outer_radius);
        shader.outer_color = outer_color;
        shader
    }

    pub fn with_colors(mut self, inner_color: Color, outer_color: Color) -> Self {
        self.color = inner_color;
        self.outer_color = outer_color;
        self
    }

    pub fn with_radii(mut self, inner_radius: f32, outer_radius: f32) -> Self {
        self.set_radii(inner_radius, outer_radius);
        self
    }

    pub fn with_focus(mut self, x: f32, y: f32) -> Self {
        self.focus = Some((x, y));
        self
    }

    /// Alias for [`Self::with_focus`].
    pub fn with_center(self, x: f32, y: f32) -> Self {
        self.with_focus(x, y)
    }

    pub fn with_aspect(mut self, aspect: f32) -> Self {
        self.set_aspect(aspect);
        self
    }

    pub fn with_target(mut self, target: ShaderTarget) -> Self {
        self.target = target;
        self
    }

    pub fn color(&self) -> Color {
        self.color
    }

    pub fn set_color(&mut self, color: Color) {
        self.color = color;
        self.outer_color = color;
    }

    pub fn colors(&self) -> (Color, Color) {
        (self.color, self.outer_color)
    }

    pub fn set_colors(&mut self, inner_color: Color, outer_color: Color) {
        self.color = inner_color;
        self.outer_color = outer_color;
    }

    pub fn radii(&self) -> (f32, f32) {
        (self.inner_radius, self.outer_radius)
    }

    pub fn set_radii(&mut self, inner_radius: f32, outer_radius: f32) {
        let inner = if inner_radius.is_nan() {
            0.0
        } else {
            inner_radius.max(0.0)
        };
        self.inner_radius = inner;
        self.outer_radius = if outer_radius.is_nan() || outer_radius <= inner {
            inner + 1.0
        } else {
            outer_radius
        };
    }

    pub fn aspect(&self) -> f32 {
        self.aspect
    }

    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = if aspect.is_finite() && aspect > 0.0 {
            aspect
        } else {
            2.0
        };
    }

    pub fn focus(&self) -> Option<(f32, f32)> {
        self.focus
    }

    pub fn set_focus(&mut self, focus: Option<(f32, f32)>) {
        self.focus = focus;
    }

    pub fn clear_focus(&mut self) {
        self.focus = None;
    }

    pub fn target(&self) -> ShaderTarget {
        self.target
    }

    pub fn set_target(&mut self, target: ShaderTarget) {
        self.target = target;
    }

    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn reset(&mut self) {}
}

impl Shader for FlashlightShader {
    fn render(&mut self, canvas: &mut Canvas<'_>, ctx: &ShaderContext<'_>) {
        if !self.active || !self.target.should_render(ctx) {
            return;
        }
        let rect = self.target.rect(ctx, canvas);
        if rect.is_empty() {
            return;
        }
        let (focus_x, focus_y) = self.focus.unwrap_or_else(|| {
            let (x, y) = rect.center();
            (x as f32 + 0.5, y as f32 + 0.5)
        });
        let color = self.color;
        let outer_color = self.outer_color;
        let inner = self.inner_radius;
        let outer = self.outer_radius;
        let aspect = self.aspect;
        for_each_visible_cell(canvas, rect, |canvas, x, y, cell| {
            if cell.pixel.is_transparent() || cell.pixel.ch.is_whitespace() {
                return;
            }
            let dx = x as f32 + 0.5 - focus_x;
            let dy = (y as f32 + 0.5 - focus_y) * aspect;
            let distance = (dx * dx + dy * dy).sqrt();
            let intensity = if distance <= inner {
                1.0
            } else if distance >= outer {
                0.0
            } else {
                let t = (outer - distance) / (outer - inner);
                t * t * (3.0 - 2.0 * t)
            };
            if intensity <= 0.0 {
                return;
            }
            let shade = color.mixture(outer_color, 1.0 - intensity);
            let pixel = Pixel {
                fg: cell.pixel.fg.mixture(shade, intensity),
                ..cell.pixel
            };
            write_pixel(canvas, x, y, cell, pixel);
        });
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn reset(&mut self) {
        FlashlightShader::reset(self);
    }
}

/// One layer of moving star streaks.
///
/// `density` is the fraction of the target area occupied by stars, `speed` is in
/// cells per second, and `length` is capped at 256 cells per streak.
#[derive(Clone, Debug)]
pub struct StarStreakLayer {
    pub density: f32,
    pub speed: f32,
    pub lifetime_ms: u64,
    pub length: u16,
    pub chars: Vec<char>,
    pub colors: Vec<Color>,
    pub z: i32,
}

impl StarStreakLayer {
    pub fn new(density: f32, speed: f32, chars: Vec<char>, colors: Vec<Color>) -> Self {
        Self {
            density: if density.is_nan() {
                0.0
            } else {
                density.clamp(0.0, 1.0)
            },
            speed: if speed.is_nan() { 0.0 } else { speed.max(0.0) },
            lifetime_ms: 1_200,
            length: 3,
            chars,
            colors,
            z: 0,
        }
    }

    pub fn with_lifetime_ms(mut self, lifetime_ms: u64) -> Self {
        self.lifetime_ms = lifetime_ms.max(1);
        self
    }

    pub fn with_length(mut self, length: u16) -> Self {
        self.length = length.clamp(1, 256);
        self
    }

    pub fn with_z(mut self, z: i32) -> Self {
        self.z = z;
        self
    }
}

#[derive(Clone, Debug)]
struct Star {
    x: f32,
    y: f32,
    speed: f32,
    life: f32,
    max_life: f32,
    ch: char,
    color: Color,
    length: u16,
}

/// Draws seeded, moving star streaks over the target.
///
/// Stars are initialized lazily from the target rectangle and wrap from the top
/// back to the bottom. Rendering is frame-aware: calling the shader more than
/// once for the same [`FrameInfo::frame`] redraws the same positions instead of
/// advancing the simulation twice.
#[derive(Clone, Debug)]
pub struct StarStreakShader {
    seed: u64,
    layers: Vec<StarStreakLayer>,
    stars: Vec<Vec<Star>>,
    target: ShaderTarget,
    active: bool,
    initialized: bool,
    last_rect: Rect,
    last_frame: Option<u64>,
    rng: Rng,
}

impl StarStreakShader {
    pub fn new(seed: u64, layers: Vec<StarStreakLayer>) -> Self {
        let stars = vec![Vec::new(); layers.len()];
        Self {
            seed,
            layers,
            stars,
            target: ShaderTarget::Object,
            active: true,
            initialized: false,
            last_rect: Rect::new(0, 0, 0, 0),
            last_frame: None,
            rng: Rng::new(seed),
        }
    }

    pub fn from_layers(layers: Vec<StarStreakLayer>) -> Self {
        Self::new(0, layers)
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.rng = Rng::new(seed);
        self.initialized = false;
        self
    }

    pub fn with_target(mut self, target: ShaderTarget) -> Self {
        self.target = target;
        self
    }

    pub fn with_layer(mut self, layer: StarStreakLayer) -> Self {
        self.layers.push(layer);
        self.stars.push(Vec::new());
        self.initialized = false;
        self
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.reset();
    }

    pub fn layers(&self) -> &[StarStreakLayer] {
        &self.layers
    }

    pub fn set_layers(&mut self, layers: Vec<StarStreakLayer>) {
        self.layers = layers;
        self.stars = vec![Vec::new(); self.layers.len()];
        self.initialized = false;
    }

    pub fn add_layer(&mut self, layer: StarStreakLayer) {
        self.layers.push(layer);
        self.stars.push(Vec::new());
        self.initialized = false;
    }

    pub fn star_count(&self) -> usize {
        self.stars.iter().map(Vec::len).sum()
    }

    pub fn target(&self) -> ShaderTarget {
        self.target
    }

    pub fn set_target(&mut self, target: ShaderTarget) {
        self.target = target;
        self.initialized = false;
    }

    pub fn start(&mut self) {
        self.active = true;
        self.reset();
    }

    pub fn stop(&mut self) {
        self.active = false;
    }

    pub fn set_active(&mut self, active: bool) {
        if active && !self.active {
            self.reset();
        }
        self.active = active;
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn reset(&mut self) {
        self.rng = Rng::new(self.seed);
        self.stars = vec![Vec::new(); self.layers.len()];
        self.initialized = false;
        self.last_rect = Rect::new(0, 0, 0, 0);
        self.last_frame = None;
    }

    fn ensure_initialized(&mut self, rect: Rect) {
        if self.initialized && self.last_rect == rect {
            return;
        }
        self.initialized = true;
        self.last_rect = rect;
        self.stars = vec![Vec::new(); self.layers.len()];
        let area = rect.area() as f32;
        for (index, layer) in self.layers.iter().enumerate() {
            let density = layer.density.clamp(0.0, 1.0);
            let count = ((area * density).ceil() as usize).min(MAX_STARS_PER_LAYER);
            self.stars[index].reserve(count);
            for _ in 0..count {
                let star = make_star(&mut self.rng, layer, rect, true);
                self.stars[index].push(star);
            }
        }
    }

    fn advance(&mut self, frame: &FrameInfo, rect: Rect) {
        let same_frame = self.last_frame == Some(frame.frame);
        self.last_frame = Some(frame.frame);
        if same_frame {
            return;
        }
        let dt = frame.delta.as_secs_f32();
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        for (index, layer) in self.layers.iter().enumerate() {
            let top = rect.y as f32;
            for star in &mut self.stars[index] {
                star.y -= star.speed * dt;
                star.life -= dt;
                if star.life <= 0.0 || star.y + f32::from(star.length) < top {
                    *star = make_star(&mut self.rng, layer, rect, false);
                }
            }
        }
    }
}

fn make_star(rng: &mut Rng, layer: &StarStreakLayer, rect: Rect, anywhere: bool) -> Star {
    let width = f32::from(rect.width.max(1));
    let height = f32::from(rect.height.max(1));
    let x = rect.x as f32 + rng.range_f32(0.0, width);
    let y = if anywhere {
        rect.y as f32 + rng.range_f32(0.0, height)
    } else {
        rect.y as f32 + height + rng.range_f32(0.0, height * 0.25)
    };
    let max_life = (layer.lifetime_ms.max(1) as f32 / 1_000.0) * rng.range_f32(0.5, 1.0);
    Star {
        x,
        y,
        speed: layer.speed.max(0.0) * rng.range_f32(0.5, 1.5),
        life: max_life,
        max_life,
        ch: pick_char(&layer.chars, rng.next_u64()),
        color: pick_color(&layer.colors, rng.next_u64()),
        length: layer.length.clamp(1, 256),
    }
}

impl Shader for StarStreakShader {
    fn render(&mut self, canvas: &mut Canvas<'_>, ctx: &ShaderContext<'_>) {
        if !self.active || !self.target.should_render(ctx) {
            return;
        }
        let rect = self.target.rect(ctx, canvas);
        if rect.is_empty() {
            return;
        }
        self.ensure_initialized(rect);
        self.advance(ctx.frame, rect);
        for (layer, stars) in self.layers.iter().zip(self.stars.iter()) {
            let length = layer.length.clamp(1, 256);
            for star in stars {
                let x = star.x.floor() as i32;
                let base_y = star.y.floor() as i32;
                let life_ratio = (star.life / star.max_life).clamp(0.0, 1.0);
                for trail in 0..length {
                    let y = base_y + i32::from(trail);
                    if !rect.contains(x, y) {
                        continue;
                    }
                    let trail_fraction = f32::from(trail) / f32::from(length);
                    let color = star
                        .color
                        .darker(trail_fraction * 0.7 + (1.0 - life_ratio) * 0.3);
                    canvas.draw_pixel(x, y, layer.z, Pixel::new(star.ch, color, None));
                }
            }
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn reset(&mut self) {
        StarStreakShader::reset(self);
    }
}

/// Brightens non-space glyphs in response to a beat envelope.
///
/// The shader can either follow [`FrameInfo::elapsed`] automatically, sample a
/// supplied playback time with [`Self::set_time`], or use a manually supplied
/// level with [`Self::set_level`]. The last two options let an application pass
/// [`crate::audio::Sound::current_time`] without coupling the shader to a
/// particular audio backend.
#[derive(Clone, Debug)]
pub struct BeatShader {
    envelope: Option<BeatEnvelope>,
    offset: Duration,
    color: Color,
    level: f32,
    threshold: f32,
    intensity: f32,
    target: ShaderTarget,
    active: bool,
    use_frame_time: bool,
    time: Option<Duration>,
}

impl BeatShader {
    /// Creates a beat shader for `envelope`, adding `source_offset` to sampled
    /// times and tinting toward `color`.
    pub fn new(envelope: BeatEnvelope, source_offset: Duration, color: Color) -> Self {
        Self {
            envelope: Some(envelope),
            offset: source_offset,
            color,
            level: 0.0,
            threshold: 0.0,
            intensity: 1.0,
            target: ShaderTarget::Object,
            active: true,
            use_frame_time: false,
            time: None,
        }
    }

    /// Creates a shader that follows the engine frame clock.
    pub fn following_frame(envelope: BeatEnvelope, source_offset: Duration, color: Color) -> Self {
        Self::new(envelope, source_offset, color).with_frame_time(true)
    }

    /// Creates a shader already sampled at `elapsed`.
    pub fn from_beat_envelope(
        envelope: BeatEnvelope,
        source_offset: Duration,
        color: Color,
        elapsed: Duration,
    ) -> Self {
        let mut shader = Self::new(envelope, source_offset, color);
        shader.set_time(elapsed);
        shader
    }

    pub fn with_beat_envelope(mut self, envelope: BeatEnvelope, elapsed: Duration) -> Self {
        self.envelope = Some(envelope);
        self.set_time(elapsed);
        self
    }

    pub fn with_frame_time(mut self, enabled: bool) -> Self {
        self.use_frame_time = enabled;
        self
    }

    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    pub fn with_source_offset(mut self, source_offset: Duration) -> Self {
        self.offset = source_offset;
        self
    }

    pub fn with_level(mut self, level: f32) -> Self {
        self.set_level(level);
        self
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.set_threshold(threshold);
        self
    }

    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.set_intensity(intensity);
        self
    }

    pub fn with_target(mut self, target: ShaderTarget) -> Self {
        self.target = target;
        self
    }

    pub fn envelope(&self) -> Option<&BeatEnvelope> {
        self.envelope.as_ref()
    }

    pub fn set_envelope(&mut self, envelope: BeatEnvelope) {
        self.envelope = Some(envelope);
        self.time = None;
        self.level = 0.0;
    }

    pub fn source_offset(&self) -> Duration {
        self.offset
    }

    pub fn set_source_offset(&mut self, source_offset: Duration) {
        self.offset = source_offset;
    }

    pub fn color(&self) -> Color {
        self.color
    }

    pub fn set_color(&mut self, color: Color) {
        self.color = color;
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = if threshold.is_nan() {
            0.0
        } else {
            threshold.clamp(0.0, 1.0)
        };
    }

    pub fn intensity(&self) -> f32 {
        self.intensity
    }

    pub fn set_intensity(&mut self, intensity: f32) {
        self.intensity = if intensity.is_nan() {
            0.0
        } else {
            intensity.clamp(0.0, 1.0)
        };
    }

    /// Sets a manual brightness in `0.0..=1.0` and disables sampled time.
    pub fn set_level(&mut self, level: f32) {
        self.level = if level.is_nan() {
            0.0
        } else {
            level.clamp(0.0, 1.0)
        };
        self.use_frame_time = false;
        self.time = None;
    }

    pub fn level(&self) -> f32 {
        self.level
    }

    /// Samples the envelope at `elapsed` (plus the source offset) and stores the
    /// result for subsequent renders.
    pub fn set_time(&mut self, elapsed: Duration) {
        self.use_frame_time = false;
        self.time = Some(elapsed);
        self.level = self.brightness_at(elapsed);
    }

    /// Alias for [`Self::set_time`] emphasizing playback-time sampling.
    pub fn render_at(&mut self, elapsed: Duration) {
        self.set_time(elapsed);
    }

    /// Convenience for an application polling a playing sound.
    pub fn set_sound_time(&mut self, sound: &crate::audio::Sound) {
        self.set_time(sound.current_time());
    }

    /// Brightness at a playback time without changing stored state.
    pub fn brightness_at(&self, elapsed: Duration) -> f32 {
        match &self.envelope {
            Some(envelope) => envelope.brightness_at(elapsed.saturating_add(self.offset)),
            None => self.level,
        }
    }

    pub fn target(&self) -> ShaderTarget {
        self.target
    }

    pub fn set_target(&mut self, target: ShaderTarget) {
        self.target = target;
    }

    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn reset(&mut self) {
        self.level = 0.0;
        self.time = None;
    }

    fn current_level(&mut self, frame: &FrameInfo) -> f32 {
        if self.use_frame_time {
            self.level = self.brightness_at(frame.elapsed);
        } else if let Some(time) = self.time {
            self.level = self.brightness_at(time);
        }
        self.level
    }
}

impl Shader for BeatShader {
    fn render(&mut self, canvas: &mut Canvas<'_>, ctx: &ShaderContext<'_>) {
        if !self.active || !self.target.should_render(ctx) {
            return;
        }
        let level = self.current_level(ctx.frame);
        if level <= self.threshold {
            return;
        }
        let strength = if self.threshold >= 1.0 {
            1.0
        } else {
            ((level - self.threshold) / (1.0 - self.threshold)).clamp(0.0, 1.0)
        } * self.intensity;
        if strength <= 0.0 {
            return;
        }
        let rect = self.target.rect(ctx, canvas);
        let color = self.color;
        for_each_visible_cell(canvas, rect, |canvas, x, y, cell| {
            if cell.pixel.is_transparent() || cell.pixel.ch.is_whitespace() {
                return;
            }
            let pixel = Pixel {
                fg: cell.pixel.fg.mixture(color, strength),
                ..cell.pixel
            };
            write_pixel(canvas, x, y, cell, pixel);
        });
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn reset(&mut self) {
        BeatShader::reset(self);
    }
}

/// A [`SceneObject`] wrapper that applies shaders after the wrapped object's
/// render callback.
///
/// Shaders are stored behind [`RefCell`] because [`SceneObject::render`] takes
/// `&self` while shader state is mutable. This type is therefore not `Sync`; the
/// engine is single-threaded and does not require it.
///
/// The engine invokes `render` only for visible objects that intersect the
/// camera. If an effect must keep running off camera, implement
/// [`SceneObject::shader`] on a custom wrapper and call [`apply_shaders`] there
/// instead.
pub struct ShadedObject<T: SceneObject> {
    base: T,
    shaders: RefCell<Vec<Box<dyn Shader>>>,
}

impl<T: SceneObject> ShadedObject<T> {
    pub fn new(base: T) -> Self {
        Self {
            base,
            shaders: RefCell::new(Vec::new()),
        }
    }

    pub fn with_shader(base: T, shader: impl Shader + 'static) -> Self {
        let object = Self::new(base);
        object.shaders.borrow_mut().push(Box::new(shader));
        object
    }

    pub fn with_shaders(base: T, shaders: Vec<Box<dyn Shader>>) -> Self {
        Self {
            base,
            shaders: RefCell::new(shaders),
        }
    }

    pub fn add_shader(&mut self, shader: impl Shader + 'static) -> &mut Self {
        self.shaders.get_mut().push(Box::new(shader));
        self
    }

    pub fn shader_count(&self) -> usize {
        self.shaders.borrow().len()
    }

    /// Resets every attached shader.
    pub fn reset_shaders(&self) {
        for shader in self.shaders.borrow_mut().iter_mut() {
            shader.reset();
        }
    }

    /// Applies the attached shaders immediately. Useful from a custom
    /// [`SceneObject::shader`] implementation.
    pub fn apply(&self, canvas: &mut Canvas<'_>, ctx: &ShaderContext<'_>) {
        apply_shaders(&mut self.shaders.borrow_mut(), canvas, ctx);
    }

    pub fn base(&self) -> &T {
        &self.base
    }

    pub fn base_mut(&mut self) -> &mut T {
        &mut self.base
    }

    pub fn into_base(self) -> T {
        self.base
    }
}

impl<T: SceneObject> SceneObject for ShadedObject<T> {
    fn rect(&self) -> Rect {
        self.base.rect()
    }

    fn z(&self) -> i32 {
        self.base.z()
    }

    fn visible(&self) -> bool {
        self.base.visible()
    }

    fn on_start(&mut self) {
        self.base.on_start();
        self.reset_shaders();
    }

    fn on_activate(&mut self) {
        self.base.on_activate();
    }

    fn on_deactivate(&mut self) {
        self.base.on_deactivate();
    }

    fn on_stop(&mut self) {
        self.base.on_stop();
    }

    fn update(&mut self, ctx: &mut crate::engine::FrameContext<'_>) {
        self.base.update(ctx);
    }

    fn monitor(&mut self, ctx: &mut crate::engine::FrameContext<'_>) {
        self.base.monitor(ctx);
    }

    fn render(&self, canvas: &mut Canvas<'_>, frame: &FrameInfo) {
        self.base.render(canvas, frame);
        let object = self.base.rect();
        let in_camera = object.is_empty() || frame.camera.intersects(object);
        let ctx = ShaderContext::new(frame, object, in_camera);
        self.apply(canvas, &ctx);
    }

    fn shader(&mut self, canvas: &mut Canvas<'_>, frame: &FrameInfo) {
        self.base.shader(canvas, frame);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::AudioClip;
    use crate::engine::FrameInfo;
    use crate::graphics::Screen;
    use std::cell::RefCell;
    use std::rc::Rc;

    const RED: Color = Color::rgb(255, 0, 0);
    const BLUE: Color = Color::rgb(0, 0, 255);

    fn blank() -> Pixel {
        Pixel::new(' ', Color::BLACK, Some(Color::BLACK))
    }

    fn screen(width: u16, height: u16) -> Screen {
        Screen::new(
            width,
            height,
            Pixel::new('.', Color::WHITE, Some(Color::BLACK)),
        )
    }

    fn frame(width: u16, height: u16, elapsed_ms: u64, delta_ms: u64, number: u64) -> FrameInfo {
        FrameInfo {
            frame: number,
            scene_frame: number,
            elapsed: Duration::from_millis(elapsed_ms),
            delta: Duration::from_millis(delta_ms),
            width,
            height,
            camera: Rect::new(0, 0, width, height),
            keys: Vec::new(),
        }
    }

    fn draw(screen: &mut Screen, x: i32, y: i32, z: i32, text: &str) {
        screen.canvas().draw_text(x, y, z, text, Color::WHITE, None);
    }

    fn render(shader: &mut dyn Shader, screen: &mut Screen, frame: &FrameInfo, object: Rect) {
        let ctx = ShaderContext::new(frame, object, true);
        shader.render(&mut screen.canvas(), &ctx);
    }

    #[test]
    fn easing_curves_are_clamped_and_monotonic() {
        for easing in [Easing::Linear, Easing::Smooth, Easing::Quadratic] {
            assert_eq!(easing.apply(-1.0), 0.0);
            assert_eq!(easing.apply(0.0), 0.0);
            assert_eq!(easing.apply(1.0), 1.0);
            assert_eq!(easing.apply(2.0), 1.0);
            let mut previous = 0.0;
            for step in 0..=20 {
                let value = easing.apply(step as f32 / 20.0);
                assert!(value >= previous);
                previous = value;
            }
        }
        assert_eq!(Easing::Quadratic.apply(0.5), 0.25);
        assert_eq!(Easing::Smooth.apply(0.5), 0.5);
    }

    #[test]
    fn fade_in_reaches_the_original_and_preserves_depth() {
        let mut screen = screen(4, 1);
        draw(&mut screen, 0, 0, 7, "AB");
        let mut shader = FadeInShader::new(100, blank());

        let f0 = frame(4, 1, 0, 0, 0);
        render(&mut shader, &mut screen, &f0, Rect::new(0, 0, 2, 1));
        assert_eq!(screen.snapshot(), "  ..");
        assert_eq!(screen.get(0, 0).unwrap().z, 7);
        assert!(!shader.is_done());

        draw(&mut screen, 0, 0, 7, "AB");
        let f1 = frame(4, 1, 50, 50, 1);
        render(&mut shader, &mut screen, &f1, Rect::new(0, 0, 2, 1));
        let halfway = screen.get(0, 0).unwrap().pixel.fg;
        assert!(halfway.r > 0 && halfway.r < 255);
        assert!(!shader.is_done());

        draw(&mut screen, 0, 0, 7, "AB");
        let f2 = frame(4, 1, 100, 50, 2);
        render(&mut shader, &mut screen, &f2, Rect::new(0, 0, 2, 1));
        assert_eq!(screen.snapshot(), "AB..");
        assert_eq!(screen.get(0, 0).unwrap().pixel.fg, Color::WHITE);
        assert!(shader.is_done());
        assert_eq!(shader.progress(), 1.0);
    }

    #[test]
    fn fade_out_reaches_the_background() {
        let mut screen = screen(4, 1);
        draw(&mut screen, 0, 0, 0, "AB");
        let mut shader = FadeOutShader::new(100, blank());

        let f0 = frame(4, 1, 0, 0, 0);
        render(&mut shader, &mut screen, &f0, Rect::new(0, 0, 2, 1));
        assert_eq!(screen.snapshot(), "AB..");

        draw(&mut screen, 0, 0, 0, "AB");
        let f1 = frame(4, 1, 100, 100, 1);
        render(&mut shader, &mut screen, &f1, Rect::new(0, 0, 2, 1));
        assert_eq!(screen.snapshot(), "  ..");
        assert!(shader.is_done());
    }

    #[test]
    fn reset_and_inactive_state_control_fades() {
        let mut screen = screen(4, 1);
        draw(&mut screen, 0, 0, 0, "AB");
        let mut shader = FadeInShader::new(100, blank());
        let f0 = frame(4, 1, 0, 0, 0);
        render(&mut shader, &mut screen, &f0, Rect::new(0, 0, 2, 1));
        assert_eq!(screen.snapshot(), "  ..");

        draw(&mut screen, 0, 0, 0, "AB");
        shader.reset();
        assert!(!shader.is_done());
        let f1 = frame(4, 1, 100, 100, 1);
        render(&mut shader, &mut screen, &f1, Rect::new(0, 0, 2, 1));
        assert_eq!(screen.snapshot(), "  ..");

        draw(&mut screen, 0, 0, 0, "AB");
        shader.set_active(false);
        assert!(!shader.is_active());
        let f2 = frame(4, 1, 200, 100, 2);
        render(&mut shader, &mut screen, &f2, Rect::new(0, 0, 2, 1));
        assert_eq!(screen.snapshot(), "AB..");

        shader.start();
        draw(&mut screen, 0, 0, 0, "AB");
        let f3 = frame(4, 1, 300, 100, 3);
        render(&mut shader, &mut screen, &f3, Rect::new(0, 0, 2, 1));
        assert_eq!(screen.snapshot(), "  ..");
    }

    #[test]
    fn empty_object_rect_falls_back_to_the_camera() {
        let mut screen = screen(4, 2);
        draw(&mut screen, 0, 0, 0, "AB");
        let mut shader = FadeInShader::new(100, blank());
        let mut f = frame(4, 2, 0, 0, 0);
        f.camera = Rect::new(0, 0, 2, 1);
        render(&mut shader, &mut screen, &f, Rect::new(0, 0, 0, 0));
        assert_eq!(screen.snapshot(), "  ..\n....");
    }

    #[test]
    fn empty_object_falls_back_to_the_camera_when_off_camera() {
        let mut screen = screen(4, 2);
        draw(&mut screen, 0, 0, 0, "AB");
        let mut shader = FadeInShader::new(100, blank());
        let mut f = frame(4, 2, 0, 0, 0);
        f.camera = Rect::new(0, 0, 2, 1);
        let ctx = ShaderContext::new(&f, Rect::new(0, 0, 0, 0), false);
        shader.render(&mut screen.canvas(), &ctx);
        assert_eq!(screen.snapshot(), "  ..\n....");
    }

    #[test]
    fn fade_can_target_the_camera_explicitly() {
        let mut screen = screen(4, 2);
        draw(&mut screen, 0, 0, 0, "AB");
        let mut shader = FadeInShader::new(100, blank()).with_target(ShaderTarget::Camera);
        assert_eq!(shader.target(), ShaderTarget::Camera);
        let mut f = frame(4, 2, 0, 0, 0);
        f.camera = Rect::new(0, 0, 2, 1);
        render(&mut shader, &mut screen, &f, Rect::new(0, 0, 4, 2));
        assert_eq!(screen.snapshot(), "  ..\n....");
    }

    #[test]
    fn random_fade_is_deterministic_and_restores_at_the_end() {
        let mut first = screen(4, 1);
        let mut second = screen(4, 1);
        draw(&mut first, 0, 0, 0, "AB");
        draw(&mut second, 0, 0, 0, "AB");
        let mut a = RandomFadeInShader::new(100, blank(), vec!['#']).with_seed(9);
        let mut b = RandomFadeInShader::new(100, blank(), vec!['#']).with_seed(9);

        let f0 = frame(4, 1, 0, 0, 0);
        render(&mut a, &mut first, &f0, Rect::new(0, 0, 2, 1));
        render(&mut b, &mut second, &f0, Rect::new(0, 0, 2, 1));

        draw(&mut first, 0, 0, 0, "AB");
        draw(&mut second, 0, 0, 0, "AB");
        let f1 = frame(4, 1, 50, 50, 1);
        render(&mut a, &mut first, &f1, Rect::new(0, 0, 2, 1));
        render(&mut b, &mut second, &f1, Rect::new(0, 0, 2, 1));
        assert_eq!(first.snapshot(), second.snapshot());
        assert!(first
            .snapshot()
            .chars()
            .all(|ch| ch == 'A' || ch == 'B' || ch == '#' || ch == '.'));

        draw(&mut first, 0, 0, 0, "AB");
        let f2 = frame(4, 1, 100, 50, 2);
        render(&mut a, &mut first, &f2, Rect::new(0, 0, 2, 1));
        assert_eq!(first.snapshot(), "AB..");
        assert!(a.is_done());
    }

    #[test]
    fn slide_in_hides_before_revealing_and_reveals_by_progress() {
        let mut screen = screen(4, 1);
        draw(&mut screen, 0, 0, 5, "ABCD");
        let mut shader = SlideInShader::new(100, SlideDirection::Left);

        let f0 = frame(4, 1, 0, 0, 0);
        render(&mut shader, &mut screen, &f0, Rect::new(0, 0, 4, 1));
        assert_eq!(screen.snapshot(), "    ");
        assert_eq!(screen.get(0, 0).unwrap().z, 5);

        draw(&mut screen, 0, 0, 5, "ABCD");
        let f1 = frame(4, 1, 50, 50, 1);
        render(&mut shader, &mut screen, &f1, Rect::new(0, 0, 4, 1));
        assert_eq!(screen.snapshot(), "AB  ");

        draw(&mut screen, 0, 0, 5, "ABCD");
        let f2 = frame(4, 1, 100, 50, 2);
        render(&mut shader, &mut screen, &f2, Rect::new(0, 0, 4, 1));
        assert_eq!(screen.snapshot(), "ABCD");
        assert!(shader.is_done());
    }

    #[test]
    fn slide_out_hides_after_the_threshold() {
        let mut screen = screen(4, 1);
        draw(&mut screen, 0, 0, 0, "ABCD");
        let mut shader = SlideOutShader::new(100, SlideDirection::Left);
        let f0 = frame(4, 1, 0, 0, 0);
        render(&mut shader, &mut screen, &f0, Rect::new(0, 0, 4, 1));
        assert_eq!(screen.snapshot(), "ABCD");

        draw(&mut screen, 0, 0, 0, "ABCD");
        let f1 = frame(4, 1, 50, 50, 1);
        render(&mut shader, &mut screen, &f1, Rect::new(0, 0, 4, 1));
        assert_eq!(screen.snapshot(), "  CD");

        draw(&mut screen, 0, 0, 0, "ABCD");
        let f2 = frame(4, 1, 100, 50, 2);
        render(&mut shader, &mut screen, &f2, Rect::new(0, 0, 4, 1));
        assert_eq!(screen.snapshot(), "    ");
        assert!(shader.is_done());
    }

    #[test]
    fn every_slide_direction_produces_a_bounded_threshold() {
        for direction in SlideDirection::ALL {
            for (x, y) in [(0, 0), (1, 2), (3, 3)] {
                let threshold = slide_threshold(direction, x, y, 4, 4, 17);
                assert!((0.0..1.0).contains(&threshold), "{direction:?}");
            }
        }
    }

    #[test]
    fn shimmer_changes_only_non_space_cells_inside_the_target() {
        let mut first = screen(6, 1);
        let mut second = screen(6, 1);
        draw(&mut first, 0, 0, 3, "AB C");
        draw(&mut second, 0, 0, 3, "AB C");
        let mut a = ShimmerShader::new(7, 100, vec![RED]);
        let mut b = ShimmerShader::new(7, 100, vec![RED]);

        let f = frame(6, 1, 0, 0, 0);
        render(&mut a, &mut first, &f, Rect::new(0, 0, 4, 1));
        render(&mut b, &mut second, &f, Rect::new(0, 0, 4, 1));
        assert_eq!(first.snapshot(), second.snapshot());
        assert_eq!(first.get(0, 0).unwrap().pixel.fg, RED);
        assert_eq!(first.get(1, 0).unwrap().pixel.fg, RED);
        assert_eq!(first.get(2, 0).unwrap().pixel.fg, Color::WHITE);
        assert_eq!(first.get(3, 0).unwrap().pixel.fg, RED);
        assert_eq!(first.get(4, 0).unwrap().pixel.fg, Color::WHITE);
        assert_eq!(first.get(0, 0).unwrap().z, 3);

        draw(&mut first, 0, 0, 3, "AB C");
        a.set_active(false);
        render(&mut a, &mut first, &f, Rect::new(0, 0, 4, 1));
        assert_eq!(first.get(0, 0).unwrap().pixel.fg, Color::WHITE);
    }

    #[test]
    fn shimmer_cache_refreshes_on_keyframes() {
        let mut screen = screen(2, 1);
        draw(&mut screen, 0, 0, 0, "AB");
        let mut shader = ShimmerShader::new(3, 100, vec![RED]);
        let f0 = frame(2, 1, 0, 0, 0);
        render(&mut shader, &mut screen, &f0, Rect::new(0, 0, 2, 1));
        assert_eq!(shader.cache_key, Some(0));
        let cached = shader.cache_colors.clone();

        let f1 = frame(2, 1, 100, 100, 1);
        draw(&mut screen, 0, 0, 0, "AB");
        render(&mut shader, &mut screen, &f1, Rect::new(0, 0, 2, 1));
        assert_eq!(shader.cache_key, Some(1));
        assert_eq!(shader.cache_colors.len(), cached.len());
    }

    #[test]
    fn sparkle_is_deterministic_and_skips_spaces() {
        let mut first = screen(5, 1);
        let mut second = screen(5, 1);
        draw(&mut first, 0, 0, 0, "A C");
        draw(&mut second, 0, 0, 0, "A C");
        let mut a = SparkleShader::new(11, 1.0, vec![RED]).with_lifetime_ms(1_000);
        let mut b = SparkleShader::new(11, 1.0, vec![RED]).with_lifetime_ms(1_000);

        let f = frame(5, 1, 0, 0, 0);
        render(&mut a, &mut first, &f, Rect::new(0, 0, 3, 1));
        render(&mut b, &mut second, &f, Rect::new(0, 0, 3, 1));
        assert_eq!(first.snapshot(), second.snapshot());
        assert_eq!(first.get(1, 0).unwrap().pixel.fg, Color::WHITE);
        assert!(
            first.get(0, 0).unwrap().pixel.fg != Color::WHITE
                || first.get(2, 0).unwrap().pixel.fg != Color::WHITE
        );

        draw(&mut first, 0, 0, 0, "A C");
        a.set_density(0.0);
        let f1 = frame(5, 1, 50, 50, 1);
        render(&mut a, &mut first, &f1, Rect::new(0, 0, 3, 1));
        assert_eq!(first.snapshot(), "A C..");
    }

    #[test]
    fn border_shades_edges_more_than_the_center() {
        let mut edge_screen = screen(5, 3);
        for y in 0..3 {
            draw(&mut edge_screen, 0, y, 2, "XXXXX");
        }
        let mut shader = BorderShader::new(RED, BLUE, 1.0).with_aspect(1.0);
        let f = frame(5, 3, 0, 0, 0);
        render(&mut shader, &mut edge_screen, &f, Rect::new(0, 0, 5, 3));
        assert_eq!(edge_screen.get(0, 0).unwrap().pixel.fg, RED);
        assert_eq!(edge_screen.get(2, 1).unwrap().pixel.fg, Color::WHITE);
        assert_eq!(edge_screen.get(0, 0).unwrap().z, 2);

        let mut shader = BorderShader::new(RED, BLUE, 1.5);
        shader.set_aspect(1.0);
        let mut tall = screen(5, 5);
        for y in 0..5 {
            draw(&mut tall, 0, y, 0, "XXXXX");
        }
        let f = frame(5, 5, 0, 0, 0);
        render(&mut shader, &mut tall, &f, Rect::new(0, 0, 5, 5));
        let aspect_one = tall.get(2, 1).unwrap().pixel.fg;

        let mut shader = BorderShader::new(RED, BLUE, 1.5);
        shader.set_aspect(2.0);
        let mut tall = screen(5, 5);
        for y in 0..5 {
            draw(&mut tall, 0, y, 0, "XXXXX");
        }
        render(&mut shader, &mut tall, &f, Rect::new(0, 0, 5, 5));
        let aspect_two = tall.get(2, 1).unwrap().pixel.fg;
        assert!(aspect_one != aspect_two);
    }

    #[test]
    fn flashlight_focus_moves_the_lit_cell() {
        let mut screen = screen(5, 1);
        draw(&mut screen, 0, 0, 0, "ABCDE");
        let mut left = FlashlightShader::new(RED, 0.0, 2.0)
            .with_aspect(1.0)
            .with_focus(0.5, 0.5);
        let f = frame(5, 1, 0, 0, 0);
        render(&mut left, &mut screen, &f, Rect::new(0, 0, 5, 1));
        assert_eq!(screen.get(0, 0).unwrap().pixel.fg, RED);
        assert_eq!(screen.get(4, 0).unwrap().pixel.fg, Color::WHITE);

        draw(&mut screen, 0, 0, 0, "ABCDE");
        let mut right = FlashlightShader::new(RED, 0.0, 2.0)
            .with_aspect(1.0)
            .with_focus(4.5, 0.5);
        render(&mut right, &mut screen, &f, Rect::new(0, 0, 5, 1));
        assert_eq!(screen.get(4, 0).unwrap().pixel.fg, RED);
        assert_eq!(screen.get(0, 0).unwrap().pixel.fg, Color::WHITE);
    }

    fn star_layer() -> StarStreakLayer {
        StarStreakLayer::new(1.0, 20.0, vec!['*'], vec![Color::WHITE])
            .with_lifetime_ms(1_000)
            .with_length(2)
            .with_z(5)
    }

    #[test]
    fn star_streaks_are_deterministic_and_move_once_per_frame() {
        let mut first = screen(8, 6);
        let mut second = screen(8, 6);
        let mut a = StarStreakShader::new(42, vec![star_layer()]);
        let mut b = StarStreakShader::new(42, vec![star_layer()]);

        let f0 = frame(8, 6, 0, 0, 0);
        render(&mut a, &mut first, &f0, Rect::new(0, 0, 8, 6));
        render(&mut b, &mut second, &f0, Rect::new(0, 0, 8, 6));
        assert_eq!(first.snapshot(), second.snapshot());
        assert!(a.star_count() > 0);
        let first_frame = first.snapshot();
        assert!(first_frame.contains('*'));
        for y in 0..6 {
            for x in 0..8 {
                let cell = first.get(x, y).unwrap();
                if cell.pixel.ch == '*' {
                    assert_eq!(cell.z, 5);
                }
            }
        }

        // A second render in the same frame redraws, but does not advance.
        render(&mut a, &mut first, &f0, Rect::new(0, 0, 8, 6));
        assert_eq!(first.snapshot(), first_frame);

        first.clear();
        second.clear();
        let f1 = frame(8, 6, 100, 100, 1);
        render(&mut a, &mut first, &f1, Rect::new(0, 0, 8, 6));
        render(&mut b, &mut second, &f1, Rect::new(0, 0, 8, 6));
        assert_eq!(first.snapshot(), second.snapshot());
        assert_ne!(first.snapshot(), first_frame);
    }

    #[test]
    fn beat_shader_follows_the_envelope_or_a_manual_level() {
        let clip = AudioClip {
            samples: vec![0.5; 4_410],
            sample_rate: 44_100,
            channels: 1,
            origin: None,
        };
        let envelope = BeatEnvelope::new(&clip);
        let mut screen = screen(2, 1);
        draw(&mut screen, 0, 0, 0, "AB");
        let mut shader = BeatShader::following_frame(envelope, Duration::ZERO, RED);
        let f = frame(2, 1, 0, 0, 0);
        render(&mut shader, &mut screen, &f, Rect::new(0, 0, 2, 1));
        assert_eq!(screen.get(0, 0).unwrap().pixel.fg, RED);

        let empty = BeatEnvelope::new(&AudioClip {
            samples: Vec::new(),
            sample_rate: 44_100,
            channels: 1,
            origin: None,
        });
        let mut manual = BeatShader::new(empty, Duration::ZERO, BLUE);
        draw(&mut screen, 0, 0, 0, "AB");
        manual.set_level(0.0);
        render(&mut manual, &mut screen, &f, Rect::new(0, 0, 2, 1));
        assert_eq!(screen.get(0, 0).unwrap().pixel.fg, Color::WHITE);

        draw(&mut screen, 0, 0, 0, "AB");
        manual.set_level(1.0);
        render(&mut manual, &mut screen, &f, Rect::new(0, 0, 2, 1));
        assert_eq!(screen.get(0, 0).unwrap().pixel.fg, BLUE);
    }

    struct Base {
        rect: Rect,
        z: i32,
        visible: bool,
        starts: u32,
        activations: u32,
        deactivations: u32,
        stops: u32,
    }

    impl Base {
        fn new(rect: Rect, z: i32) -> Self {
            Self {
                rect,
                z,
                visible: true,
                starts: 0,
                activations: 0,
                deactivations: 0,
                stops: 0,
            }
        }
    }

    impl SceneObject for Base {
        fn rect(&self) -> Rect {
            self.rect
        }

        fn z(&self) -> i32 {
            self.z
        }

        fn visible(&self) -> bool {
            self.visible
        }

        fn on_start(&mut self) {
            self.starts += 1;
        }

        fn on_activate(&mut self) {
            self.activations += 1;
        }

        fn on_deactivate(&mut self) {
            self.deactivations += 1;
        }

        fn on_stop(&mut self) {
            self.stops += 1;
        }

        fn render(&self, canvas: &mut Canvas<'_>, _: &FrameInfo) {
            canvas.draw_text(self.rect.x, self.rect.y, self.z, "AB", Color::WHITE, None);
        }
    }

    struct RecolorShader {
        active: bool,
        log: Rc<RefCell<Vec<&'static str>>>,
    }

    impl Shader for RecolorShader {
        fn render(&mut self, canvas: &mut Canvas<'_>, ctx: &ShaderContext<'_>) {
            if !ctx.in_camera {
                return;
            }
            self.log.borrow_mut().push("render");
            let rect = ctx.target_rect(canvas);
            for_each_visible_cell(canvas, rect, |canvas, x, y, cell| {
                if cell.pixel.ch.is_whitespace() {
                    return;
                }
                let pixel = Pixel {
                    fg: RED,
                    ..cell.pixel
                };
                write_pixel(canvas, x, y, cell, pixel);
            });
        }

        fn is_active(&self) -> bool {
            self.active
        }

        fn reset(&mut self) {
            self.log.borrow_mut().push("reset");
        }
    }

    #[test]
    fn shaded_object_applies_shaders_after_the_base_render() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut object = ShadedObject::with_shader(
            Base::new(Rect::new(0, 0, 2, 1), 4),
            RecolorShader {
                active: true,
                log: log.clone(),
            },
        );
        let boxed: Box<dyn Shader> = Box::new(RecolorShader {
            active: true,
            log: log.clone(),
        });
        object.add_shader(boxed);
        assert_eq!(object.shader_count(), 2);
        assert_eq!(object.rect(), Rect::new(0, 0, 2, 1));
        assert_eq!(object.z(), 4);
        assert!(object.visible());

        let mut screen = screen(4, 1);
        let f = frame(4, 1, 0, 0, 0);
        object.render(&mut screen.canvas(), &f);
        assert_eq!(screen.snapshot(), "AB..");
        assert_eq!(screen.get(0, 0).unwrap().pixel.fg, RED);
        assert_eq!(screen.get(0, 0).unwrap().z, 4);
        assert_eq!(*log.borrow(), ["render", "render"]);

        object.on_start();
        object.on_activate();
        object.on_deactivate();
        object.on_stop();
        assert_eq!(object.base().starts, 1);
        assert_eq!(object.base().activations, 1);
        assert_eq!(object.base().deactivations, 1);
        assert_eq!(object.base().stops, 1);
        assert!(log.borrow().contains(&"reset"));
    }

    #[test]
    fn shaded_object_skips_inactive_and_off_camera_shaders() {
        let object = ShadedObject::with_shader(
            Base::new(Rect::new(0, 0, 2, 1), 0),
            RecolorShader {
                active: false,
                log: Rc::new(RefCell::new(Vec::new())),
            },
        );
        let mut screen = screen(4, 1);
        let f = frame(4, 1, 0, 0, 0);
        object.render(&mut screen.canvas(), &f);
        assert_eq!(screen.get(0, 0).unwrap().pixel.fg, Color::WHITE);

        let log = Rc::new(RefCell::new(Vec::new()));
        let object = ShadedObject::with_shader(
            Base::new(Rect::new(0, 0, 2, 1), 0),
            RecolorShader {
                active: true,
                log: log.clone(),
            },
        );
        let mut off_camera = frame(4, 1, 0, 0, 0);
        off_camera.camera = Rect::new(10, 10, 2, 2);
        object.render(&mut screen.canvas(), &off_camera);
        assert!(log.borrow().is_empty());
        assert_eq!(screen.get(0, 0).unwrap().pixel.fg, Color::WHITE);
    }

    #[test]
    fn helper_rng_and_hashes_are_stable() {
        let mut a = Rng::new(123);
        let mut b = Rng::new(123);
        for _ in 0..16 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        assert_eq!(hash_cell(3, -4, 99), hash_cell(3, -4, 99));
        assert_ne!(hash_cell(3, -4, 99), hash_cell(3, -4, 100));
        for x in -2..2 {
            for y in -2..2 {
                let fraction = hash_fraction(x, y, 1);
                assert!((0.0..1.0).contains(&fraction));
            }
        }
    }
}
