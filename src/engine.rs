// SPDX-License-Identifier: Apache-2.0
//! Owned, single-threaded scene runtime with injectable time and input.
//!
//! Update and monitor callbacks use the pre-update `(z, id)` order. Update
//! queries see the pre-update snapshot; monitor queries see the completed update
//! pass. Rendering uses a new post-monitor snapshot in `(z, id)` order, including
//! its bounds and visibility for camera culling. Shader callbacks run exactly
//! once per object, visible objects before invisible ones, with membership and
//! `(z, id)` order fixed by that same post-monitor snapshot. Changes made during
//! rendering or shading do not reschedule callbacks in the current frame.
//!
//! Each render/shader callback receives a fresh canvas with full-screen clipping;
//! clipping changes are local to that callback, while drawn cells remain shared.
//! Structural commands take effect after rendering, so newly spawned objects are
//! first activated and updated on the next tick. IDs are never reused in a scene.

use crate::graphics::{Canvas, Pixel, Rect, Screen};
use crate::terminal::Key;
use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObjectId(u64);

#[derive(Clone, Debug)]
pub struct ObjectInfo {
    pub id: ObjectId,
    pub rect: Rect,
    pub z: i32,
    pub visible: bool,
}

#[derive(Clone, Debug)]
pub struct FrameInfo {
    pub frame: u64,
    pub scene_frame: u64,
    pub elapsed: Duration,
    pub delta: Duration,
    pub width: u16,
    pub height: u16,
    pub camera: Rect,
    pub keys: Vec<Key>,
}

impl FrameInfo {
    pub fn pressed(&self, key: Key) -> bool {
        self.keys.contains(&key)
    }
}

/// Scene mutations are queued, never performed while iterating objects.
pub enum Command {
    Spawn(Box<dyn SceneObject>),
    Despawn(ObjectId),
    SwitchScene(String),
    Quit,
}

pub struct FrameContext<'a> {
    pub id: ObjectId,
    pub frame: &'a FrameInfo,
    objects: &'a [ObjectInfo],
    commands: &'a mut Vec<Command>,
}

impl FrameContext<'_> {
    pub fn objects(&self) -> &[ObjectInfo] {
        self.objects
    }

    pub fn object(&self, id: ObjectId) -> Option<&ObjectInfo> {
        self.objects.iter().find(|obj| obj.id == id)
    }

    /// Visible bounding-box intersections, excluding the caller.
    pub fn collisions(&self, rect: Rect) -> impl Iterator<Item = &ObjectInfo> {
        self.objects
            .iter()
            .filter(move |obj| obj.id != self.id && obj.visible && obj.rect.intersects(rect))
    }

    pub fn command(&mut self, command: Command) {
        self.commands.push(command);
    }

    pub fn spawn(&mut self, object: impl SceneObject + 'static) {
        self.command(Command::Spawn(Box::new(object)));
    }

    pub fn despawn(&mut self, id: ObjectId) {
        self.command(Command::Despawn(id));
    }

    pub fn switch_scene(&mut self, id: impl Into<String>) {
        self.command(Command::SwitchScene(id.into()));
    }

    pub fn quit(&mut self) {
        self.command(Command::Quit);
    }
}

/// Game state belongs to its object, not to a global engine singleton.
pub trait SceneObject {
    fn rect(&self) -> Rect;
    fn z(&self) -> i32 {
        0
    }
    fn visible(&self) -> bool {
        true
    }
    fn on_start(&mut self) {}
    fn on_activate(&mut self) {}
    fn on_deactivate(&mut self) {}
    fn on_stop(&mut self) {}
    fn update(&mut self, _ctx: &mut FrameContext<'_>) {}
    fn monitor(&mut self, _ctx: &mut FrameContext<'_>) {}
    fn render(&self, _canvas: &mut Canvas<'_>, _frame: &FrameInfo) {}
    /// CPU post-processing, once per frame even for off-camera objects.
    /// Post-monitor visibility determines the visible-then-invisible passes;
    /// changes during rendering or shading affect scheduling next frame only.
    fn shader(&mut self, _canvas: &mut Canvas<'_>, _frame: &FrameInfo) {}
}

struct Entry {
    object: Box<dyn SceneObject>,
    started: bool,
    active: bool,
}

impl Entry {
    fn activate(&mut self) {
        if !self.started {
            self.object.on_start();
            self.started = true;
        }
        if !self.active {
            self.object.on_activate();
            self.active = true;
        }
    }

    fn deactivate(&mut self) {
        if self.active {
            self.object.on_deactivate();
            self.active = false;
        }
    }

    fn stop(&mut self) {
        self.deactivate();
        if self.started {
            self.object.on_stop();
            self.started = false;
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Camera {
    pub x: i32,
    pub y: i32,
    pub follow: Option<ObjectId>,
}

impl Camera {
    fn viewport(&self, size: (u16, u16), terminal: (u16, u16), objects: &[ObjectInfo]) -> Rect {
        let width = size.0.min(terminal.0);
        let height = size.1.min(terminal.1);
        let (x, y) = self
            .follow
            .and_then(|id| objects.iter().find(|o| o.id == id))
            .map(|o| {
                (
                    i64::from(o.rect.x) + i64::from(o.rect.width) / 2 - i64::from(width) / 2,
                    i64::from(o.rect.y) + i64::from(o.rect.height) / 2 - i64::from(height) / 2,
                )
            })
            .unwrap_or((i64::from(self.x), i64::from(self.y)));
        Rect::new(
            x.clamp(0, i64::from(size.0 - width)) as i32,
            y.clamp(0, i64::from(size.1 - height)) as i32,
            width,
            height,
        )
    }
}

pub struct Scene {
    id: String,
    background: Pixel,
    size: Option<(u16, u16)>,
    objects: BTreeMap<ObjectId, Entry>,
    next_id: u64,
    pub camera: Camera,
}

impl Scene {
    /// Adaptive scenes use the current terminal dimensions.
    pub fn new(id: impl Into<String>, background: Pixel) -> Self {
        Self {
            id: id.into(),
            background,
            size: None,
            objects: BTreeMap::new(),
            next_id: 0,
            camera: Camera::default(),
        }
    }

    /// A fixed-size world can be larger than the terminal and viewed by camera.
    pub fn with_size(mut self, width: u16, height: u16) -> Self {
        self.size = Some((width, height));
        self
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn add_object(&mut self, object: impl SceneObject + 'static) -> ObjectId {
        self.add_boxed(Box::new(object))
    }

    fn add_boxed(&mut self, object: Box<dyn SceneObject>) -> ObjectId {
        let id = ObjectId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .expect("scene object ID exhausted");
        self.objects.insert(
            id,
            Entry {
                object,
                started: false,
                active: false,
            },
        );
        id
    }

    fn snapshot(&self) -> Vec<ObjectInfo> {
        let mut objects: Vec<_> = self
            .objects
            .iter()
            .map(|(&id, entry)| ObjectInfo {
                id,
                rect: entry.object.rect(),
                z: entry.object.z(),
                visible: entry.object.visible(),
            })
            .collect();
        objects.sort_by_key(|o| (o.z, o.id));
        objects
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum EngineError {
    DuplicateScene(String),
    UnknownScene(String),
    NoActiveScene,
    Stopped,
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateScene(id) => write!(f, "duplicate scene: {id}"),
            Self::UnknownScene(id) => write!(f, "unknown scene: {id}"),
            Self::NoActiveScene => write!(f, "no active scene; call Engine::start first"),
            Self::Stopped => write!(f, "engine has stopped"),
        }
    }
}
impl std::error::Error for EngineError {}

pub struct Engine {
    scenes: BTreeMap<String, Scene>,
    current: Option<String>,
    terminal_size: (u16, u16),
    screen: Screen,
    camera: Rect,
    frame: u64,
    scene_frame: u64,
    elapsed: Duration,
    running: bool,
    stopped: bool,
}

impl Engine {
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            scenes: BTreeMap::new(),
            current: None,
            terminal_size: (width, height),
            screen: Screen::new(
                width,
                height,
                Pixel::new(
                    ' ',
                    crate::graphics::Color::WHITE,
                    Some(crate::graphics::Color::BLACK),
                ),
            ),
            camera: Rect::new(0, 0, width, height),
            frame: 0,
            scene_frame: 0,
            elapsed: Duration::ZERO,
            running: false,
            stopped: false,
        }
    }

    pub fn add_scene(&mut self, scene: Scene) -> Result<(), EngineError> {
        if self.stopped {
            return Err(EngineError::Stopped);
        }
        if self.scenes.contains_key(&scene.id) {
            return Err(EngineError::DuplicateScene(scene.id));
        }
        self.scenes.insert(scene.id.clone(), scene);
        Ok(())
    }

    pub fn start(&mut self, id: &str) -> Result<(), EngineError> {
        if self.stopped {
            return Err(EngineError::Stopped);
        }
        if !self.scenes.contains_key(id) {
            return Err(EngineError::UnknownScene(id.into()));
        }
        if let Some(previous) = &self.current {
            for entry in self.scenes.get_mut(previous).unwrap().objects.values_mut() {
                entry.deactivate();
            }
        }
        self.current = Some(id.into());
        self.scene_frame = 0;
        self.running = true;
        // Activation is deferred until tick, including on a scene switch.
        Ok(())
    }

    pub fn is_running(&self) -> bool {
        self.running
    }
    pub fn current_scene(&self) -> Option<&str> {
        self.current.as_deref()
    }
    pub fn screen(&self) -> &Screen {
        &self.screen
    }
    /// Camera for the last rendered frame (including when a switch was queued).
    pub fn camera(&self) -> Rect {
        self.camera
    }
    pub fn frame_count(&self) -> u64 {
        self.frame
    }

    pub fn resize(&mut self, width: u16, height: u16) {
        self.terminal_size = (width, height);
    }

    pub fn tick(&mut self, delta: Duration, keys: &[Key]) -> Result<&Screen, EngineError> {
        if self.stopped || (self.current.is_some() && !self.running) {
            return Err(EngineError::Stopped);
        }
        let id = self.current.clone().ok_or(EngineError::NoActiveScene)?;
        let scene = self.scenes.get_mut(&id).unwrap();
        for entry in scene.objects.values_mut() {
            entry.activate();
        }
        let size = scene.size.unwrap_or(self.terminal_size);
        if self.screen.width() != size.0 || self.screen.height() != size.1 || self.scene_frame == 0
        {
            self.screen = Screen::new(size.0, size.1, scene.background);
        } else {
            self.screen.clear();
        }
        self.elapsed = self.elapsed.saturating_add(delta);
        let before = scene.snapshot();
        let order: Vec<_> = before.iter().map(|o| o.id).collect();
        let mut info = FrameInfo {
            frame: self.frame,
            scene_frame: self.scene_frame,
            elapsed: self.elapsed,
            delta,
            width: size.0,
            height: size.1,
            keys: keys.to_vec(),
            camera: scene.camera.viewport(size, self.terminal_size, &before),
        };
        let mut commands = Vec::new();
        for &object_id in &order {
            let mut ctx = FrameContext {
                id: object_id,
                frame: &info,
                objects: &before,
                commands: &mut commands,
            };
            scene
                .objects
                .get_mut(&object_id)
                .unwrap()
                .object
                .update(&mut ctx);
        }
        let updated = scene.snapshot();
        for &object_id in &order {
            let mut ctx = FrameContext {
                id: object_id,
                frame: &info,
                objects: &updated,
                commands: &mut commands,
            };
            scene
                .objects
                .get_mut(&object_id)
                .unwrap()
                .object
                .monitor(&mut ctx);
        }
        let rendered = scene.snapshot();
        info.camera = scene.camera.viewport(size, self.terminal_size, &rendered);
        self.camera = info.camera;
        for metadata in &rendered {
            if metadata.visible && metadata.rect.intersects(info.camera) {
                let object = &scene.objects[&metadata.id].object;
                object.render(&mut self.screen.canvas(), &info);
            }
        }
        for visible in [true, false] {
            for metadata in &rendered {
                if metadata.visible == visible {
                    let object = &mut scene.objects.get_mut(&metadata.id).unwrap().object;
                    object.shader(&mut self.screen.canvas(), &info);
                }
            }
        }
        // Validate switches before applying any structural mutations. All spawn/
        // despawn commands target the originating scene; the last switch wins.
        for command in &commands {
            if let Command::SwitchScene(target) = command {
                if !self.scenes.contains_key(target) {
                    return Err(EngineError::UnknownScene(target.clone()));
                }
            }
        }
        let mut next_scene = None;
        let mut quit = false;
        let scene = self.scenes.get_mut(&id).unwrap();
        for command in commands {
            match command {
                Command::Spawn(object) => {
                    scene.add_boxed(object);
                }
                Command::Despawn(object_id) => {
                    if let Some(mut entry) = scene.objects.remove(&object_id) {
                        entry.stop();
                    }
                }
                Command::SwitchScene(target) => next_scene = Some(target),
                Command::Quit => quit = true,
            }
        }
        self.frame = self.frame.saturating_add(1);
        self.scene_frame = self.scene_frame.saturating_add(1);
        if quit {
            self.shutdown();
        } else if let Some(target) = next_scene {
            self.start(&target)?;
        }
        Ok(&self.screen)
    }

    /// Stop all activated objects exactly once. Also called by Drop.
    pub fn shutdown(&mut self) {
        if self.stopped {
            return;
        }
        for scene in self.scenes.values_mut() {
            for entry in scene.objects.values_mut() {
                entry.stop();
            }
        }
        self.running = false;
        self.stopped = true;
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphics::Color;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn bg() -> Pixel {
        Pixel::new('.', Color::WHITE, Some(Color::BLACK))
    }
    type Log = Rc<RefCell<Vec<String>>>;
    struct Probe {
        name: &'static str,
        log: Log,
        x: i32,
        visible: bool,
        action: Option<Command>,
    }
    impl Probe {
        fn new(name: &'static str, log: &Log) -> Self {
            Self {
                name,
                log: log.clone(),
                x: 0,
                visible: true,
                action: None,
            }
        }
        fn log(&self, event: &str) {
            self.log.borrow_mut().push(format!("{}:{event}", self.name));
        }
    }
    impl SceneObject for Probe {
        fn rect(&self) -> Rect {
            Rect::new(self.x, 0, 1, 1)
        }
        fn visible(&self) -> bool {
            self.visible
        }
        fn on_start(&mut self) {
            self.log("start");
        }
        fn on_activate(&mut self) {
            self.log("activate");
        }
        fn on_deactivate(&mut self) {
            self.log("deactivate");
        }
        fn on_stop(&mut self) {
            self.log("stop");
        }
        fn update(&mut self, ctx: &mut FrameContext<'_>) {
            self.log("update");
            self.x += 1;
            if let Some(command) = self.action.take() {
                ctx.command(command);
            }
        }
        fn monitor(&mut self, ctx: &mut FrameContext<'_>) {
            assert_eq!(ctx.object(ctx.id).unwrap().rect.x, self.x);
            self.log("monitor");
        }
        fn render(&self, canvas: &mut Canvas<'_>, _: &FrameInfo) {
            self.log("render");
            canvas.draw_pixel(self.x, 0, 0, Pixel::new('X', Color::WHITE, None));
        }
        fn shader(&mut self, _: &mut Canvas<'_>, _: &FrameInfo) {
            self.log("shader");
        }
    }
    fn engine(scene: Scene) -> Engine {
        let id = scene.id.clone();
        let mut e = Engine::new(8, 2);
        e.add_scene(scene).unwrap();
        e.start(&id).unwrap();
        e
    }

    #[test]
    fn lifecycle_passes_and_clear_are_deterministic() {
        let log = Log::default();
        let mut scene = Scene::new("a", bg());
        scene.add_object(Probe::new("a", &log));
        let mut e = engine(scene);
        e.tick(Duration::from_millis(33), &[]).unwrap();
        assert_eq!(e.screen().snapshot(), ".X......\n........");
        assert_eq!(
            *log.borrow(),
            [
                "a:start",
                "a:activate",
                "a:update",
                "a:monitor",
                "a:render",
                "a:shader"
            ]
        );
        e.tick(Duration::from_millis(33), &[]).unwrap();
        assert_eq!(e.screen().snapshot(), "..X.....\n........");
        e.shutdown();
        e.shutdown();
        drop(e);
        assert_eq!(
            log.borrow().iter().filter(|s| s.ends_with(":stop")).count(),
            1
        );
    }

    #[test]
    fn spawn_and_despawn_are_deferred() {
        let log = Log::default();
        let mut scene = Scene::new("a", bg());
        let mut parent = Probe::new("parent", &log);
        parent.action = Some(Command::Spawn(Box::new(Probe::new("child", &log))));
        let parent_id = scene.add_object(parent);
        let mut remover = Probe::new("remover", &log);
        remover.action = Some(Command::Despawn(parent_id));
        scene.add_object(remover);
        let mut e = engine(scene);
        e.tick(Duration::ZERO, &[]).unwrap();
        assert!(!log.borrow().iter().any(|s| s.starts_with("child:")));
        assert!(log.borrow().contains(&"parent:render".into()));
        assert!(log.borrow().contains(&"parent:stop".into()));
        log.borrow_mut().clear();
        e.tick(Duration::ZERO, &[]).unwrap();
        assert!(log.borrow().contains(&"child:start".into()));
        assert!(!log.borrow().iter().any(|s| s.starts_with("parent:")));
    }

    #[test]
    fn switch_preserves_last_frame_and_reactivates_without_restarting() {
        let log = Log::default();
        let mut a = Scene::new("a", bg());
        let mut p = Probe::new("a", &log);
        p.action = Some(Command::SwitchScene("b".into()));
        a.add_object(p);
        let mut b = Scene::new("b", bg());
        b.add_object(Probe::new("b", &log));
        let mut e = engine(a);
        e.add_scene(b).unwrap();
        e.tick(Duration::ZERO, &[]).unwrap();
        assert_eq!(e.current_scene(), Some("b"));
        assert_eq!(e.screen().snapshot(), ".X......\n........");
        assert!(log.borrow().contains(&"a:deactivate".into()));
        assert!(!log.borrow().contains(&"b:start".into()));
        e.tick(Duration::ZERO, &[]).unwrap();
        e.start("a").unwrap();
        e.tick(Duration::ZERO, &[]).unwrap();
        assert_eq!(log.borrow().iter().filter(|s| *s == "a:start").count(), 1);
        assert_eq!(
            log.borrow().iter().filter(|s| *s == "a:activate").count(),
            2
        );
    }

    #[test]
    fn invisible_objects_update_and_shader_after_visible_objects() {
        let log = Log::default();
        let mut scene = Scene::new("a", bg());
        let mut hidden = Probe::new("hidden", &log);
        hidden.visible = false;
        scene.add_object(hidden);
        scene.add_object(Probe::new("visible", &log));
        let mut e = engine(scene);
        e.tick(Duration::ZERO, &[]).unwrap();
        let l = log.borrow();
        assert!(l.contains(&"hidden:update".into()));
        assert!(!l.contains(&"hidden:render".into()));
        assert_eq!(&l[l.len() - 2..], ["visible:shader", "hidden:shader"]);
    }

    #[test]
    fn render_and_shader_callbacks_have_independent_clips() {
        struct Clipper {
            x: i32,
            visible: bool,
        }
        impl SceneObject for Clipper {
            fn rect(&self) -> Rect {
                Rect::new(self.x, 0, 1, 1)
            }
            fn visible(&self) -> bool {
                self.visible
            }
            fn render(&self, canvas: &mut Canvas<'_>, _: &FrameInfo) {
                canvas.draw_pixel(self.x, 0, 0, Pixel::new('R', Color::WHITE, None));
                canvas.set_clip(Rect::new(0, 0, 0, 0));
            }
            fn shader(&mut self, canvas: &mut Canvas<'_>, _: &FrameInfo) {
                canvas.draw_pixel(self.x, 1, 0, Pixel::new('S', Color::WHITE, None));
                canvas.set_clip(Rect::new(0, 0, 0, 0));
            }
        }
        let mut scene = Scene::new("clips", bg());
        for (x, visible) in [(0, true), (3, true), (6, false)] {
            scene.add_object(Clipper { x, visible });
        }
        let mut e = engine(scene);
        for _ in 0..2 {
            e.tick(Duration::ZERO, &[]).unwrap();
            // Covers render->render, render->shader, shader->shader and the
            // visible->invisible shader boundary without losing prior writes.
            assert_eq!(e.screen().snapshot(), "R..R....\nS..S..S.");
        }
    }

    #[test]
    fn shader_membership_is_frozen_after_monitor() {
        struct VisibilityProbe {
            name: &'static str,
            visible: std::cell::Cell<bool>,
            monitor_visible: Option<bool>,
            hide_on_render: bool,
            x: i32,
            log: Log,
        }
        impl SceneObject for VisibilityProbe {
            fn rect(&self) -> Rect {
                Rect::new(self.x, 0, 1, 1)
            }
            fn visible(&self) -> bool {
                self.visible.get()
            }
            fn monitor(&mut self, _: &mut FrameContext<'_>) {
                if let Some(visible) = self.monitor_visible {
                    self.visible.set(visible);
                }
            }
            fn render(&self, _: &mut Canvas<'_>, _: &FrameInfo) {
                self.log.borrow_mut().push(format!("{}:render", self.name));
                if self.hide_on_render {
                    self.visible.set(false);
                }
            }
            fn shader(&mut self, _: &mut Canvas<'_>, _: &FrameInfo) {
                self.log.borrow_mut().push(format!("{}:shader", self.name));
                self.visible.set(!self.visible.get());
            }
        }
        for hide_on_render in [false, true] {
            let log = Log::default();
            let mut scene = Scene::new("visibility", bg());
            for (name, visible, monitor_visible, x) in [
                ("hidden", true, Some(false), 0),
                ("visible", false, Some(true), 0),
                ("offscreen", true, None, 100),
            ] {
                scene.add_object(VisibilityProbe {
                    name,
                    visible: std::cell::Cell::new(visible),
                    monitor_visible,
                    hide_on_render,
                    x,
                    log: log.clone(),
                });
            }
            let mut e = engine(scene);
            e.tick(Duration::ZERO, &[]).unwrap();
            // Monitor changes take effect immediately; render/shader changes
            // cannot move an object between passes or execute it twice.
            assert_eq!(
                *log.borrow(),
                [
                    "visible:render",
                    "visible:shader",
                    "offscreen:shader",
                    "hidden:shader"
                ],
                "hide_on_render={hide_on_render}"
            );
        }
    }

    #[test]
    fn render_and_shader_order_use_post_monitor_z_with_stable_ties() {
        struct Layer {
            ch: char,
            z: i32,
            new_z: i32,
            change_in_monitor: bool,
            background: Option<Color>,
            log: Log,
        }
        impl SceneObject for Layer {
            fn rect(&self) -> Rect {
                Rect::new(0, 0, 1, 1)
            }
            fn z(&self) -> i32 {
                self.z
            }
            fn update(&mut self, _: &mut FrameContext<'_>) {
                if !self.change_in_monitor {
                    self.z = self.new_z;
                }
            }
            fn monitor(&mut self, _: &mut FrameContext<'_>) {
                if self.change_in_monitor {
                    self.z = self.new_z;
                }
            }
            fn render(&self, canvas: &mut Canvas<'_>, _: &FrameInfo) {
                self.log.borrow_mut().push(format!("{}:render", self.ch));
                canvas.draw_pixel(
                    0,
                    0,
                    self.z,
                    Pixel::new(self.ch, Color::WHITE, self.background),
                );
            }
            fn shader(&mut self, _: &mut Canvas<'_>, _: &FrameInfo) {
                self.log.borrow_mut().push(format!("{}:shader", self.ch));
            }
        }
        let red = Color::rgb(255, 0, 0);
        for change_in_monitor in [false, true] {
            let log = Log::default();
            let mut scene = Scene::new("layers", bg());
            for (ch, z, new_z, background) in
                [('A', 0, 2, None), ('B', 1, 1, Some(red)), ('C', 2, 2, None)]
            {
                scene.add_object(Layer {
                    ch,
                    z,
                    new_z,
                    change_in_monitor,
                    background,
                    log: log.clone(),
                });
            }
            let mut e = engine(scene);
            for _ in 0..2 {
                log.borrow_mut().clear();
                e.tick(Duration::ZERO, &[]).unwrap();
                assert_eq!(
                    *log.borrow(),
                    ["B:render", "A:render", "C:render", "B:shader", "A:shader", "C:shader"],
                    "change_in_monitor={change_in_monitor}"
                );
                let cell = e.screen().get(0, 0).unwrap();
                assert_eq!(cell.pixel, Pixel::new('C', Color::WHITE, Some(red)));
                assert_eq!(cell.z, 2);
            }
        }
    }

    #[test]
    fn camera_tracks_and_clamps_fixed_world_on_resize() {
        let log = Log::default();
        let mut scene = Scene::new("a", bg()).with_size(100, 50);
        let mut p = Probe::new("a", &log);
        p.x = 90;
        scene.camera.follow = Some(scene.add_object(p));
        let mut e = engine(scene);
        e.tick(Duration::ZERO, &[]).unwrap();
        assert_eq!(e.camera(), Rect::new(87, 0, 8, 2));
        e.resize(200, 100);
        e.tick(Duration::ZERO, &[]).unwrap();
        assert_eq!(e.camera(), Rect::new(0, 0, 100, 50));
    }

    #[test]
    fn invalid_transitions_and_quit_are_explicit() {
        let mut e = Engine::new(2, 2);
        assert!(matches!(
            e.tick(Duration::ZERO, &[]),
            Err(EngineError::NoActiveScene)
        ));
        e.add_scene(Scene::new("a", bg())).unwrap();
        assert!(matches!(
            e.add_scene(Scene::new("a", bg())),
            Err(EngineError::DuplicateScene(_))
        ));
        assert!(matches!(
            e.start("missing"),
            Err(EngineError::UnknownScene(_))
        ));
        let log = Log::default();
        let mut b = Scene::new("b", bg());
        let mut p = Probe::new("b", &log);
        p.action = Some(Command::Quit);
        b.add_object(p);
        e.add_scene(b).unwrap();
        e.start("b").unwrap();
        e.tick(Duration::ZERO, &[]).unwrap();
        assert!(!e.is_running());
        assert!(matches!(
            e.tick(Duration::ZERO, &[]),
            Err(EngineError::Stopped)
        ));
    }
}
