// SPDX-License-Identifier: Apache-2.0
//! Deterministic, single-pixel particle effects inspired by CosPlay's
//! `CPParticle`, `CPParticleEmitter`, and `CPParticleSprite`.
//!
//! The module deliberately has no dependency on wall-clock time: every update
//! receives the same [`FrameInfo`] used by the rest of the engine. Randomness is
//! supplied by the small [`Rand`] Mulberry32 generator, so two systems created
//! with the same seed produce exactly the same particles.
//!
//! The core pieces mirror CosPlay:
//!
//! - [`Particle`] is one character-cell particle with its own lifetime.
//! - [`ParticleEmitter`] creates particles and starts paused, like
//!   `CPParticleEmitter`.
//! - [`ParticleSystem`] is the Rust equivalent of `CPParticleSprite`: it owns
//!   emitters, purges dead particles, emits new ones, and renders live particles
//!   at their own z-index.
//!
//! ```no_run
//! use cosplay::engine::FrameInfo;
//! use cosplay::graphics::{Canvas, Color, Rect};
//! use cosplay::particles::{ConfettiConfig, ConfettiEmitter, ParticleSystem};
//!
//! let config = ConfettiConfig {
//!     count_per_burst: 8,
//!     colors: vec![Color::WHITE],
//!     ..ConfettiConfig::default()
//! };
//! let emitter = ConfettiEmitter::new("confetti", config, |frame| {
//!     (f32::from(frame.width) / 2.0, f32::from(frame.height) / 2.0)
//! });
//! let mut system = ParticleSystem::new().with_emitter(Box::new(emitter));
//! system.resume_all(false);
//! // Call `system.update(&frame)` once per tick and `system.render(&mut canvas)`
//! // after updating.
//! # let _ = (|_: &FrameInfo, _: &mut Canvas<'_>| {});
//! ```

use crate::engine::FrameInfo;
use crate::graphics::{Canvas, Color, Pixel, Rect};
#[cfg(test)]
use std::collections::VecDeque;
use std::f32::consts::TAU;
use std::time::Duration;

/// A lightweight borrowed frame wrapper.
///
/// The traits in this module accept `&FrameInfo` directly because that is the
/// value the engine already provides. This wrapper exists for callers that
/// prefer a named type at their API boundary.
#[derive(Clone, Copy, Debug)]
pub struct ParticleFrame<'a> {
    /// The engine frame being processed.
    pub info: &'a FrameInfo,
}

impl<'a> ParticleFrame<'a> {
    /// Wraps a borrowed engine frame.
    pub const fn new(info: &'a FrameInfo) -> Self {
        Self { info }
    }

    /// Returns the wrapped frame.
    pub const fn info(&self) -> &'a FrameInfo {
        self.info
    }
}

impl<'a> From<&'a FrameInfo> for ParticleFrame<'a> {
    fn from(info: &'a FrameInfo) -> Self {
        Self::new(info)
    }
}

/// A deterministic 32-bit Mulberry32 pseudo-random generator.
///
/// Mulberry32 is intentionally tiny and not cryptographically secure. It is a
/// good fit for particle effects because it is fast, dependency-free, and
/// reproducible across platforms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rand {
    state: u32,
}

impl Rand {
    /// Creates a generator from a 64-bit seed.
    ///
    /// Both halves of the seed are folded into the 32-bit Mulberry32 state, so
    /// seeds such as `1` and `1 << 32 | 1` do not produce the same sequence.
    pub fn new(seed: u64) -> Self {
        let folded = (seed as u32) ^ ((seed >> 32) as u32);
        Self {
            state: folded ^ 0x9E37_79B9,
        }
    }

    /// Returns the next raw `u32` value.
    pub fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_add(0x6D2B_79F5);
        let mut value = self.state;
        value = (value ^ (value >> 15)).wrapping_mul(value | 1);
        value ^= value.wrapping_add((value ^ (value >> 7)).wrapping_mul(value | 61));
        value ^ (value >> 14)
    }

    /// Returns a value in `0.0..1.0`.
    pub fn next_f32(&mut self) -> f32 {
        // Taking the high 24 bits maps exactly onto f32's significand and avoids
        // ever returning 1.0.
        (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0)
    }

    /// Returns a value in the inclusive-exclusive range `min..max`.
    ///
    /// Reversed bounds are accepted and swapped. Non-finite bounds fall back to
    /// the lower bound when it is finite, or zero otherwise, which keeps results
    /// finite.
    pub fn range_f32(&mut self, min: f32, max: f32) -> f32 {
        let (low, high) = if min <= max { (min, max) } else { (max, min) };
        if !low.is_finite() || !high.is_finite() {
            return if low.is_finite() { low } else { 0.0 };
        }
        low + (high - low) * self.next_f32()
    }

    /// Returns an index in `0..bound`, or zero when `bound` is zero.
    pub fn index(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            (self.next_u32() as usize) % bound
        }
    }

    /// Picks a random element from a slice.
    pub fn choose<'a, T>(&mut self, values: &'a [T]) -> Option<&'a T> {
        values.get(self.index(values.len()))
    }
}

impl Default for Rand {
    fn default() -> Self {
        Self::new(0)
    }
}

/// A single character-cell particle.
pub trait Particle {
    /// Advances this particle by one engine frame.
    fn update(&mut self, frame: &FrameInfo);

    /// Current rounded x-coordinate.
    fn x(&self) -> i32;

    /// Current rounded y-coordinate.
    fn y(&self) -> i32;

    /// Z-index used when this particle is rendered.
    fn z(&self) -> i32;

    /// The cell drawn for this particle on the current frame.
    fn pixel(&self) -> Pixel;

    /// Whether this particle should remain in the system after an update.
    fn is_alive(&self) -> bool;
}

/// A source of particles.
///
/// Emitters start paused, matching CosPlay's `CPParticleEmitter`. A system only
/// calls [`ParticleEmitter::emit`] while the emitter is not paused.
pub trait ParticleEmitter {
    /// Stable identifier used for diagnostics and lookup.
    fn id(&self) -> &str;

    /// Restores this emitter to its initial state without changing pause state.
    fn reset(&mut self);

    /// Stops this emitter from producing new particles.
    fn pause(&mut self);

    /// Resumes emission, optionally resetting first.
    fn resume(&mut self, reset: bool);

    /// Whether this emitter is currently paused.
    fn is_paused(&self) -> bool;

    /// Flips the pause state without resetting.
    fn toggle(&mut self);

    /// Produces particles for one frame.
    ///
    /// Implementations should return an empty vector while paused. The system
    /// checks [`ParticleEmitter::is_paused`] before calling this method, but
    /// keeping the check inside the implementation makes direct use safe too.
    fn emit(&mut self, frame: &FrameInfo) -> Vec<Box<dyn Particle>>;
}

/// Owns a set of emitters and the particles they have produced.
///
/// Updates follow CosPlay's particle-sprite order: purge dead particles, emit
/// from active emitters, then update every particle (including particles emitted
/// on this frame). Dead particles are purged again after updating so
/// [`ParticleSystem::particles`] only exposes live particles.
pub struct ParticleSystem {
    emitters: Vec<Box<dyn ParticleEmitter>>,
    particles: Vec<Box<dyn Particle>>,
    on_start: Option<Box<dyn FnMut()>>,
    on_finish: Option<Box<dyn FnMut()>>,
    effect_active: bool,
}

impl ParticleSystem {
    /// Creates an empty system.
    pub fn new() -> Self {
        Self {
            emitters: Vec::new(),
            particles: Vec::new(),
            on_start: None,
            on_finish: None,
            effect_active: false,
        }
    }

    /// Adds an emitter and returns the system for builder-style construction.
    pub fn with_emitter(mut self, emitter: Box<dyn ParticleEmitter>) -> Self {
        self.add_emitter(emitter);
        self
    }

    /// Adds an emitter.
    pub fn add_emitter(&mut self, emitter: Box<dyn ParticleEmitter>) {
        self.emitters.push(emitter);
    }

    /// Returns the emitters in insertion order.
    pub fn emitters(&self) -> &[Box<dyn ParticleEmitter>] {
        &self.emitters
    }

    /// Returns mutable access to the emitters in insertion order.
    pub fn emitters_mut(&mut self) -> &mut [Box<dyn ParticleEmitter>] {
        &mut self.emitters
    }

    /// Returns the currently live particles in emission order.
    pub fn particles(&self) -> &[Box<dyn Particle>] {
        &self.particles
    }

    /// Returns mutable access to the live particles.
    pub fn particles_mut(&mut self) -> &mut [Box<dyn Particle>] {
        &mut self.particles
    }

    /// Number of live particles (not the number of emitters).
    pub fn len(&self) -> usize {
        self.particles.len()
    }

    /// Whether there are no live particles.
    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }

    /// Whether at least one live particle remains.
    pub fn has_live_particles(&self) -> bool {
        !self.is_empty()
    }

    /// Advances the effect by one frame.
    ///
    /// The order is intentional and deterministic:
    ///
    /// 1. purge particles that died on the previous frame;
    /// 2. ask every non-paused emitter for new particles;
    /// 3. update every particle, including newly emitted ones;
    /// 4. purge particles that died during this update.
    ///
    /// Start and finish callbacks are evaluated after step 4.
    pub fn update(&mut self, frame: &FrameInfo) {
        self.particles.retain(|particle| particle.is_alive());

        let mut emitted = Vec::new();
        for emitter in &mut self.emitters {
            if !emitter.is_paused() {
                emitted.append(&mut emitter.emit(frame));
            }
        }
        self.particles.append(&mut emitted);
        // Remember whether emission produced anything before the update pass.
        // A particle can die immediately when its lifetime is shorter than the
        // frame delta, but the effect still started on this frame.
        let emitted_particles = !self.particles.is_empty();

        for particle in &mut self.particles {
            particle.update(frame);
        }
        self.particles.retain(|particle| particle.is_alive());

        self.refresh_callbacks(emitted_particles);
    }

    /// Draws every live particle at its own z-index.
    pub fn render(&self, canvas: &mut Canvas<'_>) {
        for particle in &self.particles {
            if particle.is_alive() {
                canvas.draw_pixel(particle.x(), particle.y(), particle.z(), particle.pixel());
            }
        }
    }

    /// Returns the inclusive bounding box of all live particles.
    ///
    /// For particles at x-coordinates 2 and 5, the returned rectangle starts at
    /// 2 and has width 4 (`2..=5`). This deliberately fixes CosPlay's
    /// `CPParticleSprite`, which stored `max - min` and therefore reported a
    /// half-open rectangle that excluded the rightmost and bottommost cells.
    ///
    /// Coordinates that span more than `u16::MAX` cells are saturated to the
    /// largest representable rectangle because [`Rect`] stores `u16` dimensions.
    pub fn bounds(&self) -> Option<Rect> {
        let mut min_x = i32::MAX;
        let mut min_y = i32::MAX;
        let mut max_x = i32::MIN;
        let mut max_y = i32::MIN;
        let mut found = false;

        for particle in &self.particles {
            if !particle.is_alive() {
                continue;
            }
            found = true;
            min_x = min_x.min(particle.x());
            min_y = min_y.min(particle.y());
            max_x = max_x.max(particle.x());
            max_y = max_y.max(particle.y());
        }

        found.then(|| {
            let width =
                (i64::from(max_x) - i64::from(min_x) + 1).clamp(1, i64::from(u16::MAX)) as u16;
            let height =
                (i64::from(max_y) - i64::from(min_y) + 1).clamp(1, i64::from(u16::MAX)) as u16;
            Rect::new(min_x, min_y, width, height)
        })
    }

    /// Pauses every emitter.
    pub fn pause_all(&mut self) {
        for emitter in &mut self.emitters {
            emitter.pause();
        }
    }

    /// Resumes every emitter, optionally resetting each one first.
    pub fn resume_all(&mut self, reset: bool) {
        for emitter in &mut self.emitters {
            emitter.resume(reset);
        }
    }

    /// Toggles the pause state of every emitter.
    pub fn toggle_all(&mut self) {
        for emitter in &mut self.emitters {
            emitter.toggle();
        }
    }

    /// Resets every emitter and removes all live particles.
    ///
    /// Pause state is not changed by an emitter's reset, so a paused system
    /// remains paused. Callback bookkeeping is cleared, allowing a later start
    /// to fire [`ParticleSystem::set_on_start`] again.
    pub fn reset(&mut self) {
        for emitter in &mut self.emitters {
            emitter.reset();
        }
        self.particles.clear();
        self.effect_active = false;
    }

    /// Removes all live particles without resetting emitters.
    ///
    /// No finish callback is invoked by `clear`; callbacks are evaluated by
    /// [`ParticleSystem::update`].
    pub fn clear(&mut self) {
        self.particles.clear();
        self.effect_active = false;
    }

    /// Returns `true` when every emitter is paused.
    ///
    /// An empty system is paused. This is the all-paused predicate that
    /// CosPlay's `CPParticleSprite.isPaused` actually implements
    /// (`!emitters.exists(_.isPaused)`). Its Scaladoc was ambiguous about
    /// whether it meant "all paused" or "any paused"; this port documents the
    /// fixed, unambiguous behavior.
    pub fn is_paused(&self) -> bool {
        self.emitters.iter().all(|emitter| emitter.is_paused())
    }

    /// Returns `true` when all emitters are paused and no particles remain.
    ///
    /// An empty system is finished. This is a stricter condition than
    /// [`ParticleSystem::is_paused`] because live particles may outlive a pause.
    pub fn finished(&self) -> bool {
        self.is_empty() && self.is_paused()
    }

    /// Sets or removes the callback invoked when an effect cycle starts.
    ///
    /// The callback fires once when the first particle becomes live after the
    /// system was inactive. Replacing the callback while an effect is already
    /// active does not invoke the new callback retroactively.
    pub fn set_on_start(&mut self, callback: Option<Box<dyn FnMut()>>) {
        self.on_start = callback;
    }

    /// Sets or removes the callback invoked when the system becomes finished.
    ///
    /// The callback fires once on the transition from active to
    /// [`ParticleSystem::finished`]. It is not called repeatedly while the
    /// system remains finished.
    pub fn set_on_finish(&mut self, callback: Option<Box<dyn FnMut()>>) {
        self.on_finish = callback;
    }

    fn refresh_callbacks(&mut self, emitted_particles: bool) {
        if !self.effect_active && emitted_particles {
            self.effect_active = true;
            if let Some(callback) = self.on_start.as_mut() {
                callback();
            }
        } else if self.effect_active && self.finished() {
            self.effect_active = false;
            if let Some(callback) = self.on_finish.as_mut() {
                callback();
            }
        }
    }
}

impl Default for ParticleSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for [`ConfettiEmitter`].
///
/// Velocities and lifetimes are expressed in seconds and character cells.
/// Gravity is in cells per second squared; drag is an exponential damping
/// coefficient per second. This is a deliberately simpler model than CosPlay's
/// Lagrange speed curves, but it keeps the motion frame-rate independent and
/// easy to reproduce.
#[derive(Clone, Debug, PartialEq)]
pub struct ConfettiConfig {
    /// Number of particles produced by each burst.
    pub count_per_burst: usize,
    /// Minimum time between bursts, in milliseconds. Zero emits every update.
    pub burst_interval_ms: u64,
    /// Minimum and maximum initial speed, in cells per second.
    pub speed_range: (f32, f32),
    /// Minimum and maximum particle lifetime, in seconds.
    pub lifetime_range: (f32, f32),
    /// Downward acceleration in cells per second squared.
    pub gravity: f32,
    /// Exponential velocity damping coefficient per second.
    pub drag: f32,
    /// Characters randomly assigned to particles.
    pub chars: Vec<char>,
    /// Colors randomly assigned as each particle's starting color.
    pub colors: Vec<Color>,
    /// Z-index used for this emitter's particles.
    pub z: i32,
    /// Seed for the deterministic generator.
    pub seed: u64,
}

impl Default for ConfettiConfig {
    fn default() -> Self {
        Self {
            count_per_burst: 10,
            burst_interval_ms: 100,
            speed_range: (1.0, 8.0),
            lifetime_range: (0.5, 1.5),
            gravity: 12.0,
            drag: 0.5,
            chars: vec!['*', '+', '.', 'o', 'x'],
            colors: vec![
                Color::WHITE,
                Color::rgb(255, 80, 80),
                Color::rgb(255, 210, 80),
                Color::rgb(80, 220, 120),
                Color::rgb(90, 180, 255),
            ],
            z: 0,
            seed: 0,
        }
    }
}

/// A single confetti particle with drag, gravity, and a linear color fade.
#[derive(Clone, Debug, PartialEq)]
pub struct ConfettiParticle {
    /// Horizontal position in cells, before rounding for rendering.
    pub x: f32,
    /// Vertical position in cells, before rounding for rendering.
    pub y: f32,
    /// Horizontal velocity in cells per second.
    pub vx: f32,
    /// Vertical velocity in cells per second.
    pub vy: f32,
    /// Age in seconds.
    pub age: f32,
    /// Lifetime in seconds. The particle is alive while `age < max_age`.
    pub max_age: f32,
    /// Character drawn for this particle.
    pub ch: char,
    /// Color at birth.
    pub color_start: Color,
    /// Color approached as the particle reaches its lifetime.
    pub color_end: Color,
    /// Z-index used for rendering.
    pub z: i32,
    /// Downward acceleration in cells per second squared.
    pub gravity: f32,
    /// Exponential velocity damping coefficient per second.
    pub drag: f32,
}

impl ConfettiParticle {
    /// Creates a particle without gravity or drag.
    ///
    /// Use [`ConfettiParticle::with_physics`] to set the motion parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        x: f32,
        y: f32,
        vx: f32,
        vy: f32,
        max_age: f32,
        ch: char,
        color_start: Color,
        color_end: Color,
        z: i32,
    ) -> Self {
        Self {
            x,
            y,
            vx,
            vy,
            age: 0.0,
            max_age,
            ch,
            color_start,
            color_end,
            z,
            gravity: 0.0,
            drag: 0.0,
        }
    }

    /// Sets gravity and drag, returning the particle for chaining.
    pub fn with_physics(mut self, gravity: f32, drag: f32) -> Self {
        self.gravity = gravity;
        self.drag = drag;
        self
    }

    /// The current interpolated color.
    ///
    /// The blend factor is `age / max_age`, clamped to `0..=1`. A non-positive
    /// or NaN lifetime reports the end color; a positive infinite lifetime
    /// reports the start color because the particle never ages proportionally.
    pub fn color(&self) -> Color {
        let blend = if self.max_age > 0.0 && self.max_age.is_finite() {
            (self.age / self.max_age).clamp(0.0, 1.0)
        } else if self.max_age > 0.0 {
            0.0
        } else {
            1.0
        };
        self.color_start.mixture(self.color_end, blend)
    }
}

impl Default for ConfettiParticle {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0, 1.0, '*', Color::WHITE, Color::BLACK, 0)
    }
}

impl Particle for ConfettiParticle {
    fn update(&mut self, frame: &FrameInfo) {
        let delta = frame.delta.as_secs_f32();
        if !delta.is_finite() {
            return;
        }

        self.age += delta;

        let drag = if self.drag.is_finite() {
            self.drag.max(0.0)
        } else {
            0.0
        };
        if drag > 0.0 {
            let damping = (-drag * delta).exp();
            self.vx *= damping;
            self.vy *= damping;
        }
        if self.gravity.is_finite() {
            self.vy += self.gravity * delta;
        }

        self.x += self.vx * delta;
        self.y += self.vy * delta;
    }

    fn x(&self) -> i32 {
        self.x.round() as i32
    }

    fn y(&self) -> i32 {
        self.y.round() as i32
    }

    fn z(&self) -> i32 {
        self.z
    }

    fn pixel(&self) -> Pixel {
        Pixel::new(self.ch, self.color(), None)
    }

    fn is_alive(&self) -> bool {
        self.age < self.max_age
    }
}

/// The color confetti fades toward. It matches CosPlay's default dark gray
/// (`C_GRAY1`) rather than pure black so particles remain visible on a dark
/// terminal background.
pub const CONFETTI_FADE_COLOR: Color = Color::GRAY1;

type CenterFn = Box<dyn Fn(&FrameInfo) -> (f32, f32)>;

/// A deterministic confetti emitter with timed bursts.
pub struct ConfettiEmitter {
    id: String,
    config: ConfettiConfig,
    rng: Rand,
    last_burst_ms: Option<u64>,
    paused: bool,
    center: CenterFn,
}

impl ConfettiEmitter {
    /// Creates a confetti emitter that starts paused.
    pub fn new(
        id: impl Into<String>,
        config: ConfettiConfig,
        center: impl Fn(&FrameInfo) -> (f32, f32) + 'static,
    ) -> Self {
        let rng = Rand::new(config.seed);
        Self {
            id: id.into(),
            config,
            rng,
            last_burst_ms: None,
            paused: true,
            center: Box::new(center),
        }
    }

    /// Creates an emitter with the default configuration at a fixed center.
    ///
    /// Use [`ConfettiEmitter::new`] or [`ConfettiEmitter::fixed_with`] when a
    /// custom configuration or ID is needed.
    pub fn fixed(x: f32, y: f32) -> Self {
        Self::new("confetti", ConfettiConfig::default(), move |_| (x, y))
    }

    /// Creates an emitter with a custom ID and configuration at a fixed center.
    pub fn fixed_with(id: impl Into<String>, config: ConfettiConfig, x: f32, y: f32) -> Self {
        Self::new(id, config, move |_| (x, y))
    }

    /// This emitter's configuration.
    pub fn config(&self) -> &ConfettiConfig {
        &self.config
    }

    /// Mutable access to the configuration.
    ///
    /// Changing the seed through this reference does not rewind the generator;
    /// call [`ParticleEmitter::reset`] or [`ConfettiEmitter::set_config`] to
    /// restart the sequence.
    pub fn config_mut(&mut self) -> &mut ConfettiConfig {
        &mut self.config
    }

    /// Replaces the configuration and restarts timing and randomness.
    ///
    /// Pause state is preserved.
    pub fn set_config(&mut self, config: ConfettiConfig) {
        self.rng = Rand::new(config.seed);
        self.config = config;
        self.last_burst_ms = None;
    }

    /// Replaces the center function.
    pub fn set_center(&mut self, center: impl Fn(&FrameInfo) -> (f32, f32) + 'static) {
        self.center = Box::new(center);
    }

    /// The last burst time in elapsed milliseconds, if a burst has happened.
    pub fn last_burst_ms(&self) -> Option<u64> {
        self.last_burst_ms
    }

    /// Produces one confetti burst if the interval has elapsed.
    ///
    /// The first call after construction or reset emits immediately. Subsequent
    /// bursts are spaced by [`ConfettiConfig::burst_interval_ms`] using
    /// [`FrameInfo::elapsed`].
    fn burst_due(&mut self, frame: &FrameInfo) -> bool {
        let now = duration_millis(frame.elapsed);
        let due = match self.last_burst_ms {
            None => true,
            Some(last) => now.saturating_sub(last) >= self.config.burst_interval_ms,
        };
        if due {
            self.last_burst_ms = Some(now);
        }
        due
    }

    fn spawn(&mut self, frame: &FrameInfo) -> Vec<Box<dyn Particle>> {
        let (center_x, center_y) = (self.center)(frame);
        let mut particles = Vec::with_capacity(self.config.count_per_burst);
        let chars = &self.config.chars;
        let colors = &self.config.colors;

        for _ in 0..self.config.count_per_burst {
            let angle = self.rng.next_f32() * TAU;
            let speed = self
                .rng
                .range_f32(self.config.speed_range.0, self.config.speed_range.1);
            let lifetime = self
                .rng
                .range_f32(self.config.lifetime_range.0, self.config.lifetime_range.1);
            let ch = self.rng.choose(chars).copied().unwrap_or('*');
            let color_start = self.rng.choose(colors).copied().unwrap_or(Color::WHITE);

            let particle = ConfettiParticle::new(
                center_x,
                center_y,
                angle.cos() * speed,
                angle.sin() * speed,
                lifetime,
                ch,
                color_start,
                CONFETTI_FADE_COLOR,
                self.config.z,
            )
            .with_physics(self.config.gravity, self.config.drag);

            particles.push(Box::new(particle) as Box<dyn Particle>);
        }

        particles
    }
}

impl ParticleEmitter for ConfettiEmitter {
    fn id(&self) -> &str {
        &self.id
    }

    fn reset(&mut self) {
        self.rng = Rand::new(self.config.seed);
        self.last_burst_ms = None;
    }

    fn pause(&mut self) {
        self.paused = true;
    }

    fn resume(&mut self, reset: bool) {
        if reset {
            self.reset();
        }
        self.paused = false;
    }

    fn is_paused(&self) -> bool {
        self.paused
    }

    fn toggle(&mut self) {
        self.paused = !self.paused;
    }

    fn emit(&mut self, frame: &FrameInfo) -> Vec<Box<dyn Particle>> {
        if self.paused || !self.burst_due(frame) {
            return Vec::new();
        }
        self.spawn(frame)
    }
}

fn duration_millis(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

/// A small scripted emitter used by the module tests.
#[cfg(test)]
struct ScriptedEmitter {
    id: &'static str,
    paused: bool,
    script: VecDeque<Vec<Box<dyn Particle>>>,
    pause_when_empty: bool,
}

#[cfg(test)]
impl ScriptedEmitter {
    fn new(id: &'static str, paused: bool) -> Self {
        Self {
            id,
            paused,
            script: VecDeque::new(),
            pause_when_empty: false,
        }
    }

    fn push(&mut self, particles: Vec<Box<dyn Particle>>) {
        self.script.push_back(particles);
    }
}

#[cfg(test)]
impl ParticleEmitter for ScriptedEmitter {
    fn id(&self) -> &str {
        self.id
    }

    fn reset(&mut self) {
        self.script.clear();
        self.pause_when_empty = false;
    }

    fn pause(&mut self) {
        self.paused = true;
    }

    fn resume(&mut self, reset: bool) {
        if reset {
            self.reset();
        }
        self.paused = false;
    }

    fn is_paused(&self) -> bool {
        self.paused
    }

    fn toggle(&mut self) {
        self.paused = !self.paused;
    }

    fn emit(&mut self, _frame: &FrameInfo) -> Vec<Box<dyn Particle>> {
        if self.paused {
            return Vec::new();
        }
        let particles = self.script.pop_front().unwrap_or_default();
        if self.script.is_empty() && self.pause_when_empty {
            self.paused = true;
        }
        particles
    }
}

#[cfg(test)]
struct TestParticle {
    x: i32,
    y: i32,
    z: i32,
    ch: char,
    color: Color,
    alive: bool,
    /// Number of updates before the particle dies.
    updates_left: u32,
}

#[cfg(test)]
impl TestParticle {
    fn new(x: i32, y: i32, z: i32, ch: char) -> Self {
        Self {
            x,
            y,
            z,
            ch,
            color: Color::WHITE,
            alive: true,
            updates_left: u32::MAX,
        }
    }

    fn expiring(x: i32, y: i32, updates_left: u32) -> Self {
        Self {
            updates_left,
            ..Self::new(x, y, 0, '*')
        }
    }

    fn colored(mut self, color: Color) -> Self {
        self.color = color;
        self
    }
}

#[cfg(test)]
impl Particle for TestParticle {
    fn update(&mut self, _frame: &FrameInfo) {
        if self.updates_left != u32::MAX {
            self.updates_left = self.updates_left.saturating_sub(1);
            if self.updates_left == 0 {
                self.alive = false;
            }
        }
    }

    fn x(&self) -> i32 {
        self.x
    }

    fn y(&self) -> i32 {
        self.y
    }

    fn z(&self) -> i32 {
        self.z
    }

    fn pixel(&self) -> Pixel {
        Pixel::new(self.ch, self.color, None)
    }

    fn is_alive(&self) -> bool {
        self.alive
    }
}

#[cfg(test)]
fn frame(elapsed_ms: u64, delta_ms: u64) -> FrameInfo {
    FrameInfo {
        frame: elapsed_ms / 16,
        scene_frame: elapsed_ms / 16,
        elapsed: Duration::from_millis(elapsed_ms),
        delta: Duration::from_millis(delta_ms),
        width: 20,
        height: 10,
        camera: Rect::new(0, 0, 20, 10),
        keys: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphics::Screen;
    use std::cell::Cell;
    use std::rc::Rc;

    fn confetti_config() -> ConfettiConfig {
        ConfettiConfig {
            count_per_burst: 4,
            burst_interval_ms: 100,
            speed_range: (0.0, 0.0),
            lifetime_range: (10.0, 10.0),
            gravity: 0.0,
            drag: 0.0,
            chars: vec!['*'],
            colors: vec![Color::WHITE],
            z: 3,
            seed: 42,
        }
    }

    #[test]
    fn emitter_starts_paused_and_bursts_on_interval() {
        let mut emitter = ConfettiEmitter::new("burst", confetti_config(), |_| (4.0, 5.0));

        assert!(emitter.is_paused());
        assert_eq!(emitter.id(), "burst");
        assert!(emitter.emit(&frame(0, 16)).is_empty());

        emitter.resume(false);
        assert!(!emitter.is_paused());
        let first = emitter.emit(&frame(0, 16));
        assert_eq!(first.len(), 4);
        assert_eq!(first[0].x(), 4);
        assert_eq!(first[0].y(), 5);
        assert_eq!(first[0].z(), 3);

        assert!(emitter.emit(&frame(50, 16)).is_empty());
        assert_eq!(emitter.emit(&frame(100, 16)).len(), 4);
        assert_eq!(emitter.last_burst_ms(), Some(100));
    }

    #[test]
    fn confetti_moves_fades_and_expires() {
        let mut particle =
            ConfettiParticle::new(0.0, 0.0, 2.0, 0.0, 1.0, '*', Color::WHITE, Color::BLACK, 0);
        assert_eq!(particle.pixel().fg, Color::WHITE);

        particle.update(&frame(0, 500));
        assert_eq!(particle.x(), 1);
        assert_eq!(particle.y(), 0);
        assert_eq!(particle.pixel().fg, Color::rgb(128, 128, 128));
        assert!(particle.is_alive());

        particle.update(&frame(500, 500));
        assert!(!particle.is_alive());
        assert_eq!(particle.pixel().fg, Color::BLACK);
    }

    #[test]
    fn gravity_and_rounding_use_delta_seconds() {
        let mut particle =
            ConfettiParticle::new(0.0, 0.0, 1.6, 0.0, 10.0, '*', Color::WHITE, Color::BLACK, 0)
                .with_physics(10.0, 0.0);

        particle.update(&frame(0, 1000));
        assert_eq!(particle.x(), 2);
        assert_eq!(particle.y(), 10);
        assert_eq!(particle.vy, 10.0);
    }

    #[test]
    fn system_bounds_are_inclusive_and_ignore_dead_particles() {
        let mut emitter = ScriptedEmitter::new("bounds", false);
        emitter.push(vec![
            Box::new(TestParticle::new(2, 3, 0, 'a')),
            Box::new(TestParticle::new(5, 7, 0, 'b')),
            Box::new(TestParticle::new(2, 7, 0, 'c')),
        ]);
        let mut dead = TestParticle::new(100, 100, 0, 'd');
        dead.alive = false;
        emitter.push(vec![Box::new(dead)]);

        let mut system = ParticleSystem::new().with_emitter(Box::new(emitter));
        assert_eq!(system.bounds(), None);
        system.update(&frame(0, 16));
        assert_eq!(system.bounds(), Some(Rect::new(2, 3, 4, 5)));
        assert_eq!(system.len(), 3);

        // The second scripted emission contains only a dead particle. It must
        // be purged after the update and must not affect the bounds.
        system.update(&frame(16, 16));
        assert_eq!(system.bounds(), Some(Rect::new(2, 3, 4, 5)));
        assert_eq!(system.len(), 3);
    }

    #[test]
    fn render_draws_each_particle_at_its_own_z() {
        let mut emitter = ScriptedEmitter::new("render", false);
        emitter.push(vec![
            Box::new(TestParticle::new(1, 1, 0, 'a').colored(Color::rgb(255, 0, 0))),
            Box::new(TestParticle::new(1, 1, 5, 'b').colored(Color::rgb(0, 255, 0))),
            Box::new(TestParticle::new(2, 1, 0, 'c')),
        ]);
        let mut system = ParticleSystem::new().with_emitter(Box::new(emitter));
        system.update(&frame(0, 16));

        let mut screen = Screen::new(5, 3, Pixel::new('.', Color::WHITE, None));
        system.render(&mut screen.canvas());

        let cell = screen.get(1, 1).unwrap();
        assert_eq!(cell.pixel.ch, 'b');
        assert_eq!(cell.pixel.fg, Color::rgb(0, 255, 0));
        assert_eq!(cell.z, 5);
        assert_eq!(screen.get(2, 1).unwrap().pixel.ch, 'c');
    }

    #[test]
    fn start_and_finish_callbacks_fire_once_per_cycle() {
        let starts = Rc::new(Cell::new(0));
        let finishes = Rc::new(Cell::new(0));
        let mut system = ParticleSystem::new();
        {
            let starts = starts.clone();
            system.set_on_start(Some(Box::new(move || starts.set(starts.get() + 1))));
        }
        {
            let finishes = finishes.clone();
            system.set_on_finish(Some(Box::new(move || finishes.set(finishes.get() + 1))));
        }

        let mut emitter = ScriptedEmitter::new("once", false);
        emitter.push(vec![Box::new(TestParticle::expiring(1, 1, 2))]);
        emitter.pause_when_empty = true;
        system.add_emitter(Box::new(emitter));

        system.update(&frame(0, 16));
        assert_eq!(starts.get(), 1);
        assert_eq!(finishes.get(), 0);
        assert!(!system.finished());

        system.update(&frame(16, 16));
        assert_eq!(starts.get(), 1);
        assert_eq!(finishes.get(), 1);
        assert!(system.finished());

        system.update(&frame(32, 16));
        assert_eq!(starts.get(), 1);
        assert_eq!(finishes.get(), 1);
    }

    #[test]
    fn all_paused_semantics_include_empty_systems() {
        let mut system = ParticleSystem::new();
        assert!(system.is_paused());
        assert!(system.finished());
        assert!(system.is_empty());
        assert_eq!(system.len(), 0);

        system.add_emitter(Box::new(ScriptedEmitter::new("paused", true)));
        system.add_emitter(Box::new(ScriptedEmitter::new("active", false)));
        assert!(!system.is_paused());
        assert!(!system.finished());

        system.pause_all();
        assert!(system.is_paused());
        assert!(system.finished());

        system.toggle_all();
        assert!(!system.is_paused());
        system.resume_all(false);
        assert!(!system.is_paused());
    }

    #[test]
    fn same_seed_produces_identical_particles() {
        let config = ConfettiConfig {
            count_per_burst: 12,
            burst_interval_ms: 0,
            speed_range: (1.0, 6.0),
            lifetime_range: (5.0, 5.0),
            gravity: 3.0,
            drag: 0.25,
            chars: vec!['*', '+', '.'],
            colors: vec![Color::WHITE, Color::rgb(255, 0, 0)],
            z: 1,
            seed: 0xDEAD_BEEF,
        };
        let mut first = ParticleSystem::new().with_emitter(Box::new(ConfettiEmitter::new(
            "first",
            config.clone(),
            |_| (5.0, 5.0),
        )));
        let mut second = ParticleSystem::new().with_emitter(Box::new(ConfettiEmitter::new(
            "second",
            config,
            |_| (5.0, 5.0),
        )));
        first.resume_all(false);
        second.resume_all(false);

        for step in 0..4 {
            let frame = frame(step * 50, 50);
            first.update(&frame);
            second.update(&frame);
        }

        assert_eq!(first.len(), second.len());
        assert!(!first.is_empty());
        for (a, b) in first.particles().iter().zip(second.particles()) {
            assert_eq!(
                (a.x(), a.y(), a.z(), a.pixel()),
                (b.x(), b.y(), b.z(), b.pixel())
            );
        }
    }

    #[test]
    fn reset_and_clear_have_distinct_behavior() {
        let mut emitter = ConfettiEmitter::new("reset", confetti_config(), |_| (1.0, 1.0));
        emitter.resume(false);
        assert_eq!(emitter.emit(&frame(0, 16)).len(), 4);
        emitter.reset();
        assert!(!emitter.is_paused());
        assert_eq!(emitter.last_burst_ms(), None);
        assert_eq!(emitter.emit(&frame(0, 16)).len(), 4);

        let mut system = ParticleSystem::new().with_emitter(Box::new(emitter));
        system.update(&frame(100, 16));
        assert_eq!(system.len(), 4);
        system.clear();
        assert!(system.is_empty());
        assert_eq!(system.emitters().len(), 1);

        system.update(&frame(200, 16));
        assert_eq!(system.len(), 4);
        system.reset();
        assert!(system.is_empty());
        assert_eq!(system.emitters().len(), 1);
    }

    #[test]
    fn fixed_constructor_uses_defaults() {
        let emitter = ConfettiEmitter::fixed(7.0, 8.0);
        assert!(emitter.is_paused());
        assert_eq!(emitter.id(), "confetti");
        assert_eq!(emitter.config(), &ConfettiConfig::default());
    }

    #[test]
    fn rand_is_seed_deterministic_and_in_range() {
        let mut a = Rand::new(99);
        let mut b = Rand::new(99);
        for _ in 0..32 {
            assert_eq!(a.next_u32(), b.next_u32());
        }

        let mut rand = Rand::new(1);
        for _ in 0..100 {
            let value = rand.next_f32();
            assert!((0.0..1.0).contains(&value));
            let ranged = rand.range_f32(-2.0, 3.0);
            assert!((-2.0..3.0).contains(&ranged));
        }

        assert_eq!(rand.range_f32(3.0, f32::INFINITY), 3.0);
        assert_eq!(rand.range_f32(f32::NEG_INFINITY, 1.0), 0.0);
        assert!(rand.range_f32(f32::NAN, 1.0).is_finite());
    }
}
