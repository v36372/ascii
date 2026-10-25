// SPDX-License-Identifier: Apache-2.0
//! Shared native application lifecycle, input, and deterministic capture.
//! Games implement [Application]; neither this module nor the renderer imports
//! any demo. CLI parsing and scene-specific HUD/content stay in the application.
//!
//! ```no_run
//! use cosplay::spatial::{app::{self, Application, AppOptions, View},
//!     Controls, Depth, GlyphGroup, LinearRgba, SceneFrame, Transform2d};
//!
//! struct Game { art: GlyphGroup, x: f32 }
//! impl Application for Game {
//!     type Frame = SceneFrame;
//!     fn update(&mut self, dt: f32, controls: Controls) {
//!         self.x += (i32::from(controls.right) - i32::from(controls.left)) as f32 * dt * 60.0;
//!     }
//!     fn reset(&mut self) -> Result<(), String> { self.x = 0.0; Ok(()) }
//!     fn frame(&self, view: View) -> Result<SceneFrame, String> {
//!         let mut frame = SceneFrame::default();
//!         frame.camera.view_height = 320.0 / view.zoom;
//!         let depth = Depth::ground(0.0).map_err(|e| e.to_string())?;
//!         frame.add(&self.art, Transform2d::at(self.x, 0.0), depth)
//!             .map_err(|e| e.to_string())?;
//!         Ok(frame)
//!     }
//! }
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let art = GlyphGroup::from_text("[o]", [8.0, 12.0], LinearRgba::WHITE, None)?;
//! app::run(Game { art, x: 0.0 }, AppOptions::default())?;
//! # Ok(()) }
//! ```
use super::{Controls, FrameSource, GpuRenderer};
use std::{
    collections::HashSet,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const FRAME: Duration = Duration::from_nanos(1_000_000_000 / 60);

/// Shared presentation settings. Scene seeds and other game settings belong to
/// the Application implementation, not to the runner.
#[derive(Clone, Debug)]
pub struct AppOptions {
    pub title: String,
    pub size: (u32, u32),
    /// Interactive presented-frame limit. None runs until quit.
    pub frames: Option<u32>,
    pub zoom: f32,
    pub autoplay: bool,
}
impl Default for AppOptions {
    fn default() -> Self {
        Self {
            title: "Glyph scene".into(),
            size: (1280, 800),
            frames: None,
            zoom: 1.0,
            autoplay: false,
        }
    }
}
impl AppOptions {
    fn validate(&self) -> Result<(), String> {
        if !(1..=4096).contains(&self.size.0) || !(1..=2160).contains(&self.size.1) {
            return Err("application size must be 1..4096 by 1..2160".into());
        }
        if !self.zoom.is_finite() || !(0.5..=3.0).contains(&self.zoom) {
            return Err("application zoom must be in 0.5..=3".into());
        }
        Ok(())
    }
}

/// Presentation state passed identically to interactive and headless frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    pub width: u32,
    pub height: u32,
    pub zoom: f32,
    pub paused: bool,
    pub autoplay: bool,
}
impl View {
    pub fn aspect(self) -> f32 {
        self.width as f32 / self.height as f32
    }
}

/// Minimal scene contract. Simulation, collision, seeds, reset policy, assets,
/// and HUD construction belong to the game. The runner owns the OS/GPU loop.
pub trait Application {
    type Frame: FrameSource;
    fn update(&mut self, delta_seconds: f32, controls: Controls);
    fn reset(&mut self) -> Result<(), String>;
    fn frame(&self, view: View) -> Result<Self::Frame, String>;
}

/// Advance by N fixed 1/30-second updates and build a frame without a GPU.
/// Zero steps captures the initial state. Does not reset the application.
pub fn prepare_capture<A: Application>(
    app: &mut A,
    options: &AppOptions,
    steps: u32,
) -> Result<A::Frame, String> {
    options.validate()?;
    if steps > 100_000 {
        return Err("capture steps must be in 0..=100000".into());
    }
    for _ in 0..steps {
        app.update(
            1.0 / 30.0,
            Controls {
                autoplay: options.autoplay,
                ..Controls::default()
            },
        );
    }
    app.frame(View {
        width: options.size.0,
        height: options.size.1,
        zoom: options.zoom,
        paused: false,
        autoplay: options.autoplay,
    })
}

pub struct CaptureReport {
    pub adapter: wgpu::AdapterInfo,
    pub glyphs: usize,
}

/// Use the same frame contract and renderer as the interactive runner.
/// A native graphics adapter is required; errors never substitute a fake image.
pub fn capture<A: Application>(
    app: &mut A,
    options: &AppOptions,
    steps: u32,
    path: &Path,
) -> Result<CaptureReport, String> {
    let frame = prepare_capture(app, options, steps)?;
    let instance = wgpu::Instance::default();
    let mut renderer = pollster::block_on(GpuRenderer::new(&instance, None))?;
    renderer.capture_png(options.size.0, options.size.1, &frame, path)?;
    Ok(CaptureReport {
        adapter: renderer.adapter.get_info(),
        glyphs: frame.frame_view().instances.len(),
    })
}

/// Run held-key input, pause/reset/autoplay/zoom, camera-sized frames, resize,
/// surface recovery, presentation, and orderly window/GPU teardown.
pub fn run<A: Application>(app: A, options: AppOptions) -> Result<(), String> {
    options.validate()?;
    if options.frames == Some(0) {
        return Err("window frame limit must be positive".into());
    }
    let event_loop = EventLoop::new().map_err(|e| format!("create event loop: {e}"))?;
    let mut runner = Runner::new(options, app);
    event_loop
        .run_app(&mut runner)
        .map_err(|e| format!("event loop: {e}"))?;
    match runner.error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

#[derive(Default)]
struct InputState {
    held: HashSet<KeyCode>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Action {
    Quit,
    Pause,
    Reset,
    Auto,
    Zoom(f32),
}

impl InputState {
    /// Repeats and synthetic focus synchronization update held state, but must
    /// never fire one-shot actions such as reset, pause, or quit.
    fn event(
        &mut self,
        code: KeyCode,
        state: ElementState,
        suppress_actions: bool,
    ) -> Option<Action> {
        if state == ElementState::Released {
            self.held.remove(&code);
            return None;
        }
        let first = self.held.insert(code);
        if suppress_actions || !first {
            return None;
        }
        match code {
            KeyCode::Escape => Some(Action::Quit),
            KeyCode::Space => Some(Action::Pause),
            KeyCode::KeyR => Some(Action::Reset),
            KeyCode::F1 => Some(Action::Auto),
            KeyCode::Equal | KeyCode::NumpadAdd => Some(Action::Zoom(1.15)),
            KeyCode::Minus | KeyCode::NumpadSubtract => Some(Action::Zoom(1.0 / 1.15)),
            _ => None,
        }
    }

    fn clear(&mut self) {
        self.held.clear();
    }

    fn controls(&self, autoplay: bool) -> Controls {
        let held = |a, b| self.held.contains(&a) || self.held.contains(&b);
        Controls {
            up: held(KeyCode::KeyW, KeyCode::ArrowUp),
            down: held(KeyCode::KeyS, KeyCode::ArrowDown),
            left: held(KeyCode::KeyA, KeyCode::ArrowLeft),
            right: held(KeyCode::KeyD, KeyCode::ArrowRight),
            run: held(KeyCode::ShiftLeft, KeyCode::ShiftRight),
            autoplay,
        }
    }
}

struct WindowState {
    renderer: GpuRenderer,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    window: Arc<Window>,
    size: PhysicalSize<u32>,
    _instance: wgpu::Instance,
}

impl WindowState {
    fn new(event_loop: &ActiveEventLoop, opts: &AppOptions) -> Result<Self, String> {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title(&opts.title)
                        .with_inner_size(PhysicalSize::new(opts.size.0, opts.size.1)),
                )
                .map_err(|e| format!("create window: {e}"))?,
        );
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| format!("create GPU surface: {e}"))?;
        let renderer = pollster::block_on(GpuRenderer::new(&instance, Some(&surface)))?;
        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&renderer.adapter, size.width.max(1), size.height.max(1))
            .ok_or("no supported surface configuration")?;
        config.format = renderer.format;
        config.present_mode = wgpu::PresentMode::Fifo;
        surface.configure(&renderer.device, &config);
        let info = renderer.adapter.get_info();
        println!("{}: {} ({:?})", opts.title, info.name, info.backend);
        Ok(Self {
            renderer,
            surface,
            config,
            window,
            size,
            _instance: instance,
        })
    }

    fn resize(&mut self, size: PhysicalSize<u32>) -> Result<(), String> {
        self.size = size;
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }
        let max = self.renderer.device.limits().max_texture_dimension_2d;
        if size.width > max || size.height > max {
            return Err(format!("window exceeds GPU texture dimension limit {max}"));
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.renderer.device, &self.config);
        Ok(())
    }

    fn draw(&mut self, frame: &impl FrameSource) -> Result<bool, String> {
        if self.size.width == 0 || self.size.height == 0 {
            return Ok(false);
        }
        let output = match self.surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.renderer.device, &self.config);
                return Ok(false);
            }
            Err(wgpu::SurfaceError::Timeout) => return Ok(false),
            Err(error) => return Err(format!("acquire surface: {error}")),
        };
        let view = output.texture.create_view(&Default::default());
        self.renderer
            .render(&view, self.size.width, self.size.height, frame)?;
        self.window.pre_present_notify();
        output.present();
        Ok(true)
    }
}

struct Runner<A: Application> {
    opts: AppOptions,
    world: A,
    input: InputState,
    state: Option<WindowState>,
    paused: bool,
    error: Option<String>,
    last_tick: Instant,
    next_redraw: Instant,
    stats_at: Instant,
    stats_frames: u32,
    frames: u32,
}

impl<A: Application> Runner<A> {
    fn new(opts: AppOptions, world: A) -> Self {
        let now = Instant::now();
        Self {
            world,
            opts,
            input: InputState::default(),
            state: None,
            paused: false,
            error: None,
            last_tick: now,
            next_redraw: now,
            stats_at: now,
            stats_frames: 0,
            frames: 0,
        }
    }
    fn fail(&mut self, event_loop: &ActiveEventLoop, error: String) {
        self.error = Some(error);
        event_loop.exit();
    }
    fn action(&mut self, event_loop: &ActiveEventLoop, action: Action) {
        match self.apply_action(action) {
            Ok(true) => event_loop.exit(),
            Ok(false) => (),
            Err(error) => self.fail(event_loop, error),
        }
    }

    /// Return true only for quit; no OS handles are needed to apply game actions.
    fn apply_action(&mut self, action: Action) -> Result<bool, String> {
        match action {
            Action::Quit => return Ok(true),
            Action::Pause => self.paused = !self.paused,
            Action::Auto => self.opts.autoplay = !self.opts.autoplay,
            Action::Reset => {
                self.world.reset()?;
                self.paused = false;
            }
            Action::Zoom(factor) => self.opts.zoom = (self.opts.zoom * factor).clamp(0.5, 3.0),
        }
        Ok(false)
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_tick).as_secs_f32();
        self.last_tick = now;
        self.next_redraw = now + FRAME;
        let Some(state) = &mut self.state else {
            return;
        };
        if state.size.width == 0 || state.size.height == 0 {
            return;
        }
        if !self.paused {
            self.world
                .update(dt, self.input.controls(self.opts.autoplay));
        }
        let result = self.world.frame(View {
            width: state.size.width,
            height: state.size.height,
            zoom: self.opts.zoom,
            paused: self.paused,
            autoplay: self.opts.autoplay,
        });
        let frame = match result {
            Ok(frame) => frame,
            Err(error) => {
                self.fail(event_loop, error.to_string());
                return;
            }
        };
        match state.draw(&frame) {
            Ok(true) => {
                self.frames += 1;
                self.stats_frames += 1;
            }
            Ok(false) => (),
            Err(error) => {
                self.fail(event_loop, error);
                return;
            }
        }
        let elapsed = now.duration_since(self.stats_at).as_secs_f32();
        if elapsed >= 0.5 {
            state.window.set_title(&format!(
                "{} | {:.0} FPS | {} glyphs",
                self.opts.title,
                self.stats_frames as f32 / elapsed,
                frame.frame_view().instances.len()
            ));
            self.stats_frames = 0;
            self.stats_at = now;
        }
        if self.opts.frames.is_some_and(|limit| self.frames >= limit) {
            event_loop.exit();
        }
    }
}

impl<A: Application> ApplicationHandler for Runner<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            match WindowState::new(event_loop, &self.opts) {
                Ok(state) => {
                    self.state = Some(state);
                    self.last_tick = Instant::now();
                    self.next_redraw = self.last_tick;
                    self.stats_at = self.last_tick;
                }
                Err(error) => self.fail(event_loop, error),
            }
        }
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if self.state.as_ref().is_none_or(|s| s.window.id() != id) {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Err(error) = self.state.as_mut().unwrap().resize(size) {
                    self.fail(event_loop, error);
                }
                self.last_tick = Instant::now();
                self.next_redraw = self.last_tick;
            }
            WindowEvent::Focused(false) => self.input.clear(),
            WindowEvent::KeyboardInput {
                event,
                is_synthetic,
                ..
            } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if let Some(action) =
                        self.input
                            .event(code, event.state, event.repeat || is_synthetic)
                    {
                        self.action(event_loop, action);
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => (p.y / 60.0) as f32,
                };
                if amount.is_finite() {
                    self.action(
                        event_loop,
                        Action::Zoom(1.1f32.powf(amount.clamp(-5.0, 5.0))),
                    );
                }
            }
            WindowEvent::RedrawRequested => self.redraw(event_loop),
            _ => (),
        }
    }
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(state) = &self.state else {
            event_loop.set_control_flow(ControlFlow::Wait);
            return;
        };
        if state.size.width == 0 || state.size.height == 0 {
            event_loop.set_control_flow(ControlFlow::Wait);
            return;
        }
        let now = Instant::now();
        if now >= self.next_redraw {
            state.window.request_redraw();
        }
        event_loop.set_control_flow(redraw_flow(now, self.next_redraw));
    }
    fn suspended(&mut self, _: &ActiveEventLoop) {
        self.input.clear();
        self.state = None;
    }
    fn exiting(&mut self, _: &ActiveEventLoop) {
        // run_app consumes and drops the event loop before the App itself.
        // Destroy the window, surface, and GPU backends while Wayland/X11
        // connections are still alive.
        self.state = None;
    }
}

fn redraw_flow(now: Instant, deadline: Instant) -> ControlFlow {
    // A redraw is already requested when due. Wait for its delivery instead of
    // repeatedly waking on an expired deadline while the compositor delays it.
    if now >= deadline {
        ControlFlow::Wait
    } else {
        ControlFlow::WaitUntil(deadline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::SceneFrame;

    #[derive(Default)]
    struct Probe {
        updates: Vec<(f32, Controls)>,
        resets: usize,
        fail_reset: bool,
        fail_frame: bool,
        view: std::cell::Cell<Option<View>>,
    }
    impl Application for Probe {
        type Frame = SceneFrame;
        fn update(&mut self, dt: f32, controls: Controls) {
            self.updates.push((dt, controls));
        }
        fn reset(&mut self) -> Result<(), String> {
            if self.fail_reset {
                return Err("reset failed".into());
            }
            self.resets += 1;
            self.updates.clear();
            Ok(())
        }
        fn frame(&self, view: View) -> Result<SceneFrame, String> {
            if self.fail_frame {
                return Err("frame failed".into());
            }
            self.view.set(Some(view));
            Ok(SceneFrame::default())
        }
    }

    #[test]
    fn capture_uses_fixed_steps_and_the_shared_frame_contract_without_a_gpu() {
        let mut app = Probe::default();
        let options = AppOptions {
            size: (800, 400),
            zoom: 2.0,
            autoplay: true,
            ..AppOptions::default()
        };
        prepare_capture(&mut app, &options, 0).unwrap();
        assert!(app.updates.is_empty());
        prepare_capture(&mut app, &options, 3).unwrap();
        assert_eq!(app.updates.len(), 3);
        assert!(app
            .updates
            .iter()
            .all(|(dt, c)| *dt == 1.0 / 30.0 && c.autoplay && !c.up && !c.run));
        let view = app.view.get().unwrap();
        assert_eq!(
            view,
            View {
                width: 800,
                height: 400,
                zoom: 2.0,
                paused: false,
                autoplay: true
            }
        );
        assert_eq!(view.aspect(), 2.0);
        assert_eq!(
            app.resets, 0,
            "capture does not secretly reset an existing simulation"
        );
    }

    #[test]
    fn invalid_capture_settings_fail_before_simulating_and_frame_errors_propagate() {
        let mut app = Probe::default();
        for options in [
            AppOptions {
                size: (0, 100),
                ..AppOptions::default()
            },
            AppOptions {
                size: (5000, 100),
                ..AppOptions::default()
            },
            AppOptions {
                zoom: f32::NAN,
                ..AppOptions::default()
            },
        ] {
            assert!(prepare_capture(&mut app, &options, 1).is_err());
        }
        assert!(prepare_capture(&mut app, &AppOptions::default(), 100_001).is_err());
        assert!(app.updates.is_empty());
        app.fail_frame = true;
        assert_eq!(
            prepare_capture(&mut app, &AppOptions::default(), 0).unwrap_err(),
            "frame failed"
        );
        assert!(run(
            Probe::default(),
            AppOptions {
                frames: Some(0),
                ..AppOptions::default()
            }
        )
        .is_err());
    }

    #[test]
    fn common_actions_reset_pause_zoom_autoplay_and_surface_reset_errors() {
        let mut runner = Runner::new(AppOptions::default(), Probe::default());
        assert!(!runner.apply_action(Action::Pause).unwrap());
        assert!(runner.paused);
        runner.apply_action(Action::Auto).unwrap();
        assert!(runner.opts.autoplay);
        runner.apply_action(Action::Zoom(100.0)).unwrap();
        assert_eq!(runner.opts.zoom, 3.0);
        runner.apply_action(Action::Zoom(0.001)).unwrap();
        assert_eq!(runner.opts.zoom, 0.5);
        runner.world.fail_reset = true;
        assert_eq!(
            runner.apply_action(Action::Reset),
            Err("reset failed".into())
        );
        assert!(runner.paused);
        assert_eq!(runner.world.resets, 0);
        runner.world.fail_reset = false;
        runner.apply_action(Action::Reset).unwrap();
        assert_eq!(runner.world.resets, 1);
        assert!(!runner.paused);
        assert!(runner.apply_action(Action::Quit).unwrap());
    }

    #[test]
    fn held_keys_release_independently_and_focus_loss_clears_them() {
        let mut input = InputState::default();
        input.event(KeyCode::KeyW, ElementState::Pressed, false);
        input.event(KeyCode::ArrowUp, ElementState::Pressed, false);
        input.event(KeyCode::ShiftLeft, ElementState::Pressed, false);
        assert!(input.controls(false).up && input.controls(false).run);
        input.event(KeyCode::KeyW, ElementState::Released, false);
        assert!(input.controls(false).up);
        input.clear();
        assert!(!input.controls(false).up);
        assert!(!input.controls(false).run);
    }
    #[test]
    fn repeated_keydown_does_not_repeat_toggles() {
        let mut input = InputState::default();
        assert_eq!(
            input.event(KeyCode::Space, ElementState::Pressed, false),
            Some(Action::Pause)
        );
        assert_eq!(
            input.event(KeyCode::Space, ElementState::Pressed, true),
            None
        );
        assert_eq!(
            input.event(KeyCode::Space, ElementState::Pressed, false),
            None
        );
        input.event(KeyCode::Space, ElementState::Released, false);
        assert_eq!(
            input.event(KeyCode::Space, ElementState::Pressed, false),
            Some(Action::Pause)
        );
    }
    #[test]
    fn synthetic_focus_keys_update_movement_without_firing_actions() {
        let mut input = InputState::default();
        for key in [KeyCode::Escape, KeyCode::Space, KeyCode::KeyR, KeyCode::F1] {
            assert_eq!(input.event(key, ElementState::Pressed, true), None);
        }
        input.event(KeyCode::KeyW, ElementState::Pressed, true);
        assert!(input.controls(false).up);
        input.event(KeyCode::KeyW, ElementState::Released, true);
        assert!(!input.controls(false).up);
    }
    #[test]
    fn delayed_redraw_waits_instead_of_spinning_on_an_expired_deadline() {
        let now = Instant::now();
        assert_eq!(redraw_flow(now, now), ControlFlow::Wait);
        assert_eq!(redraw_flow(now + FRAME, now), ControlFlow::Wait);
        assert_eq!(
            redraw_flow(now, now + FRAME),
            ControlFlow::WaitUntil(now + FRAME)
        );
    }
}
