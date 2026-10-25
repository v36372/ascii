// SPDX-License-Identifier: Apache-2.0
//! Reusable scene objects: images, animations, video, particles and helpers.

use crate::animation::{Animation, VideoEvent, VideoPlayer};
use crate::engine::{FrameContext, FrameInfo, SceneObject};
use crate::font::Font;
use crate::graphics::{Canvas, Pixel, Rect};
use crate::image::Image;
use crate::particles::ParticleSystem;
use crate::runtime::SingletonScope;
use crate::terminal::Key;
use std::cell::RefCell;
use std::sync::Arc;
use std::time::Duration;

fn image_rect(x: i32, y: i32, image: &Image) -> Rect {
    Rect::new(x, y, image.width(), image.height())
}

/// Base movement and visibility state shared by dynamic sprites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynamicSprite {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub visible: bool,
}

impl DynamicSprite {
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self {
            x,
            y,
            z,
            visible: true,
        }
    }

    pub const fn position(self) -> (i32, i32) {
        (self.x, self.y)
    }

    pub fn set_position(&mut self, x: i32, y: i32) {
        self.x = x;
        self.y = y;
    }
}

/// A movable image sprite.
pub struct ImageSprite {
    pub state: DynamicSprite,
    image: Arc<Image>,
}

impl ImageSprite {
    pub fn new(image: impl Into<Arc<Image>>, x: i32, y: i32, z: i32) -> Self {
        Self {
            state: DynamicSprite::new(x, y, z),
            image: image.into(),
        }
    }

    pub fn image(&self) -> &Arc<Image> {
        &self.image
    }

    pub fn set_image(&mut self, image: impl Into<Arc<Image>>) {
        self.image = image.into();
    }

    pub fn set_position(&mut self, x: i32, y: i32) {
        self.state.set_position(x, y);
    }

    pub fn set_z(&mut self, z: i32) {
        self.state.z = z;
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.state.visible = visible;
    }
}

impl SceneObject for ImageSprite {
    fn rect(&self) -> Rect {
        image_rect(self.state.x, self.state.y, &self.image)
    }

    fn z(&self) -> i32 {
        self.state.z
    }

    fn visible(&self) -> bool {
        self.state.visible
    }

    fn render(&self, canvas: &mut Canvas<'_>, _: &FrameInfo) {
        canvas.draw_image(self.state.x, self.state.y, self.state.z, &self.image);
    }
}

/// An image whose position is fixed after construction.
pub struct StaticImageSprite(ImageSprite);

impl StaticImageSprite {
    pub fn new(image: impl Into<Arc<Image>>, x: i32, y: i32, z: i32) -> Self {
        Self(ImageSprite::new(image, x, y, z))
    }

    pub fn image(&self) -> &Arc<Image> {
        self.0.image()
    }

    pub fn set_image(&mut self, image: impl Into<Arc<Image>>) {
        self.0.set_image(image);
    }
}

impl SceneObject for StaticImageSprite {
    fn rect(&self) -> Rect {
        self.0.rect()
    }
    fn z(&self) -> i32 {
        self.0.z()
    }
    fn visible(&self) -> bool {
        self.0.visible()
    }
    fn render(&self, canvas: &mut Canvas<'_>, frame: &FrameInfo) {
        self.0.render(canvas, frame)
    }
}

/// A font-rendered text label.
pub struct LabelSprite {
    pub state: DynamicSprite,
    font: Arc<dyn Font>,
    text: String,
    fg: crate::graphics::Color,
    bg: Option<crate::graphics::Color>,
    image: Arc<Image>,
}

impl LabelSprite {
    pub fn new(
        font: Arc<dyn Font>,
        text: impl Into<String>,
        fg: crate::graphics::Color,
        bg: Option<crate::graphics::Color>,
        x: i32,
        y: i32,
        z: i32,
    ) -> Result<Self, crate::font::FontError> {
        let text = text.into();
        let image = Arc::new(font.render(&text, fg, bg)?);
        Ok(Self {
            state: DynamicSprite::new(x, y, z),
            font,
            text,
            fg,
            bg,
            image,
        })
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn set_text(&mut self, text: impl Into<String>) -> Result<(), crate::font::FontError> {
        self.text = text.into();
        self.image = Arc::new(self.font.render(&self.text, self.fg, self.bg)?);
        Ok(())
    }

    pub fn set_colors(
        &mut self,
        fg: crate::graphics::Color,
        bg: Option<crate::graphics::Color>,
    ) -> Result<(), crate::font::FontError> {
        self.fg = fg;
        self.bg = bg;
        self.image = Arc::new(self.font.render(&self.text, fg, bg)?);
        Ok(())
    }

    pub fn set_position(&mut self, x: i32, y: i32) {
        self.state.set_position(x, y);
    }
}

impl SceneObject for LabelSprite {
    fn rect(&self) -> Rect {
        image_rect(self.state.x, self.state.y, &self.image)
    }
    fn z(&self) -> i32 {
        self.state.z
    }
    fn visible(&self) -> bool {
        self.state.visible
    }
    fn render(&self, canvas: &mut Canvas<'_>, _: &FrameInfo) {
        canvas.draw_image(self.state.x, self.state.y, self.state.z, &self.image);
    }
}

/// A sprite driven by one or more named [`Animation`] assets.
pub struct AnimationSprite {
    pub state: DynamicSprite,
    animations: Vec<Animation>,
    current: usize,
    pending: Option<usize>,
    image: Option<Arc<Image>>,
    paused: bool,
}

impl AnimationSprite {
    pub fn single(animation: Animation, x: i32, y: i32, z: i32) -> Self {
        Self {
            state: DynamicSprite::new(x, y, z),
            animations: vec![animation],
            current: 0,
            pending: None,
            image: None,
            paused: false,
        }
    }

    pub fn with_animations(
        animations: Vec<Animation>,
        initial_id: &str,
        x: i32,
        y: i32,
        z: i32,
    ) -> Result<Self, String> {
        let current = animations
            .iter()
            .position(|a| a.id() == initial_id)
            .ok_or_else(|| format!("unknown animation: {initial_id}"))?;
        if animations.is_empty() {
            return Err("animation list is empty".into());
        }
        Ok(Self {
            state: DynamicSprite::new(x, y, z),
            animations,
            current,
            pending: None,
            image: None,
            paused: false,
        })
    }

    pub fn current_id(&self) -> &str {
        self.animations[self.current].id()
    }

    pub fn current_image(&self) -> Option<&Arc<Image>> {
        self.image.as_ref()
    }

    pub fn animations(&self) -> &[Animation] {
        &self.animations
    }

    /// Changes animation. With `immediate == false`, the change happens after
    /// the current animation completes or hides.
    pub fn change(&mut self, id: &str, immediate: bool) -> Result<(), String> {
        let index = self
            .animations
            .iter()
            .position(|a| a.id() == id)
            .ok_or_else(|| format!("unknown animation: {id}"))?;
        if immediate {
            self.current = index;
            self.pending = None;
            self.animations[self.current].reset();
            self.image = None;
        } else {
            self.pending = Some(index);
        }
        Ok(())
    }

    pub fn rewind(&mut self) {
        self.animations[self.current].reset();
        self.image = None;
    }

    pub fn pause(&mut self) {
        self.paused = true;
    }

    pub fn resume(&mut self) {
        self.paused = false;
    }

    pub fn set_position(&mut self, x: i32, y: i32) {
        self.state.set_position(x, y);
    }
}

impl SceneObject for AnimationSprite {
    fn rect(&self) -> Rect {
        self.image
            .as_ref()
            .map(|image| image_rect(self.state.x, self.state.y, image))
            .unwrap_or(Rect::new(self.state.x, self.state.y, 0, 0))
    }

    fn z(&self) -> i32 {
        self.state.z
    }

    fn visible(&self) -> bool {
        self.state.visible && self.image.is_some()
    }

    fn update(&mut self, ctx: &mut FrameContext<'_>) {
        if self.paused {
            return;
        }
        if let Some(frame) = self.animations[self.current].key_frame(ctx.frame.elapsed) {
            self.image = Some(frame.image);
        } else {
            self.image = None;
            if let Some(next) = self.pending.take() {
                self.current = next;
                self.animations[self.current].reset();
            }
        }
    }

    fn render(&self, canvas: &mut Canvas<'_>, _: &FrameInfo) {
        if let Some(image) = &self.image {
            canvas.draw_image(self.state.x, self.state.y, self.state.z, image);
        }
    }
}

/// A sprite that renders the current frame of an ASCII video.
pub struct VideoSprite {
    pub state: DynamicSprite,
    player: VideoPlayer,
    image: Option<Arc<Image>>,
}

impl VideoSprite {
    pub fn new(player: VideoPlayer, x: i32, y: i32, z: i32) -> Self {
        let image = player.current_event().map(|event| event.image);
        Self {
            state: DynamicSprite::new(x, y, z),
            player,
            image,
        }
    }

    pub fn player(&self) -> &VideoPlayer {
        &self.player
    }

    pub fn player_mut(&mut self) -> &mut VideoPlayer {
        &mut self.player
    }

    pub fn current_image(&self) -> Option<&Arc<Image>> {
        self.image.as_ref()
    }

    pub fn play(&mut self) {
        self.player.play();
    }
    pub fn stop(&mut self) {
        self.player.stop();
    }
    pub fn toggle(&mut self) {
        self.player.toggle();
    }
    pub fn rewind(&mut self) {
        self.player.rewind();
    }
    pub fn seek_frame(&mut self, index: usize) -> bool {
        self.player.seek_frame(index)
    }
}

impl SceneObject for VideoSprite {
    fn rect(&self) -> Rect {
        self.image
            .as_ref()
            .map(|image| image_rect(self.state.x, self.state.y, image))
            .unwrap_or(Rect::new(self.state.x, self.state.y, 0, 0))
    }
    fn z(&self) -> i32 {
        self.state.z
    }
    fn visible(&self) -> bool {
        self.state.visible && self.image.is_some()
    }
    fn update(&mut self, ctx: &mut FrameContext<'_>) {
        if let Some(VideoEvent { image, .. }) = self.player.update(ctx.frame.elapsed) {
            self.image = Some(image);
        }
    }
    fn render(&self, canvas: &mut Canvas<'_>, _: &FrameInfo) {
        if let Some(image) = &self.image {
            canvas.draw_image(self.state.x, self.state.y, self.state.z, image);
        }
    }
}

/// A sprite that renders a [`ParticleSystem`] at an optional screen offset.
pub struct ParticleSprite {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub visible: bool,
    system: ParticleSystem,
}

impl ParticleSprite {
    pub fn new(system: ParticleSystem, x: i32, y: i32, z: i32) -> Self {
        Self {
            x,
            y,
            z,
            visible: true,
            system,
        }
    }

    pub fn system(&self) -> &ParticleSystem {
        &self.system
    }

    pub fn system_mut(&mut self) -> &mut ParticleSystem {
        &mut self.system
    }

    pub fn pause(&mut self) {
        self.system.pause_all();
    }
    pub fn resume(&mut self, reset: bool) {
        self.system.resume_all(reset);
    }
    pub fn toggle(&mut self) {
        self.system.toggle_all();
    }
}

impl SceneObject for ParticleSprite {
    fn rect(&self) -> Rect {
        self.system
            .bounds()
            .map(|rect| {
                Rect::new(
                    rect.x.saturating_add(self.x),
                    rect.y.saturating_add(self.y),
                    rect.width,
                    rect.height,
                )
            })
            .unwrap_or(Rect::new(self.x, self.y, 0, 0))
    }

    fn z(&self) -> i32 {
        self.z
    }

    fn visible(&self) -> bool {
        self.visible && !self.system.is_empty()
    }

    fn update(&mut self, ctx: &mut FrameContext<'_>) {
        self.system.update(ctx.frame);
    }

    fn render(&self, canvas: &mut Canvas<'_>, _: &FrameInfo) {
        for particle in self.system.particles() {
            canvas.draw_pixel(
                particle.x().saturating_add(self.x),
                particle.y().saturating_add(self.y),
                particle.z().saturating_add(self.z),
                particle.pixel(),
            );
        }
    }
}

type UpdateCallback = Box<dyn FnMut(&mut FrameContext<'_>)>;
type DrawCallback = Box<dyn FnMut(&mut Canvas<'_>, &FrameInfo)>;

/// Invisible object used for update logic and shaders.
pub struct OffScreenSprite {
    update: UpdateCallback,
    draw: Option<DrawCallback>,
    monitor: bool,
}

impl OffScreenSprite {
    pub fn new(update: impl FnMut(&mut FrameContext<'_>) + 'static) -> Self {
        Self {
            update: Box::new(update),
            draw: None,
            monitor: false,
        }
    }

    pub fn with_draw(mut self, draw: impl FnMut(&mut Canvas<'_>, &FrameInfo) + 'static) -> Self {
        self.draw = Some(Box::new(draw));
        self
    }

    pub fn in_monitor(mut self) -> Self {
        self.monitor = true;
        self
    }

    pub fn set_update(&mut self, update: impl FnMut(&mut FrameContext<'_>) + 'static) {
        self.update = Box::new(update);
    }
}

impl SceneObject for OffScreenSprite {
    fn rect(&self) -> Rect {
        Rect::new(0, 0, 0, 0)
    }
    fn visible(&self) -> bool {
        false
    }
    fn update(&mut self, ctx: &mut FrameContext<'_>) {
        if !self.monitor {
            (self.update)(ctx);
        }
    }
    fn monitor(&mut self, ctx: &mut FrameContext<'_>) {
        if self.monitor {
            (self.update)(ctx);
        }
    }
    fn shader(&mut self, canvas: &mut Canvas<'_>, frame: &FrameInfo) {
        if let Some(draw) = &mut self.draw {
            draw(canvas, frame);
        }
    }
}

/// Executes an action when any configured key is pressed.
pub struct KeyboardSprite {
    keys: Vec<Key>,
    handler: Box<dyn FnMut(&mut FrameContext<'_>)>,
    repeat: bool,
    fired: bool,
}

impl KeyboardSprite {
    pub fn new(
        keys: impl IntoIterator<Item = Key>,
        handler: impl FnMut(&mut FrameContext<'_>) + 'static,
    ) -> Self {
        Self {
            keys: keys.into_iter().collect(),
            handler: Box::new(handler),
            repeat: false,
            fired: false,
        }
    }

    pub fn on_key(keys: Key, handler: impl FnMut(&mut FrameContext<'_>) + 'static) -> Self {
        Self::new([keys], handler)
    }

    pub fn with_repeat(mut self, repeat: bool) -> Self {
        self.repeat = repeat;
        self
    }
}

impl SceneObject for KeyboardSprite {
    fn rect(&self) -> Rect {
        Rect::new(0, 0, 0, 0)
    }
    fn visible(&self) -> bool {
        false
    }
    fn update(&mut self, ctx: &mut FrameContext<'_>) {
        let pressed = self.keys.iter().any(|key| ctx.frame.pressed(*key));
        if pressed && (self.repeat || !self.fired) {
            (self.handler)(ctx);
            self.fired = true;
        } else if !pressed {
            self.fired = false;
        }
    }
}

/// Arbitrary canvas drawing object spanning a logical frame area.
pub struct CanvasSprite {
    width: u16,
    height: u16,
    z: i32,
    draw: RefCell<DrawCallback>,
}

impl CanvasSprite {
    pub fn new(
        width: u16,
        height: u16,
        z: i32,
        draw: impl FnMut(&mut Canvas<'_>, &FrameInfo) + 'static,
    ) -> Self {
        Self {
            width,
            height,
            z,
            draw: RefCell::new(Box::new(draw)),
        }
    }
}

impl SceneObject for CanvasSprite {
    fn rect(&self) -> Rect {
        Rect::new(0, 0, self.width, self.height)
    }
    fn z(&self) -> i32 {
        self.z
    }
    fn render(&self, canvas: &mut Canvas<'_>, frame: &FrameInfo) {
        (self.draw.borrow_mut())(canvas, frame);
    }
}

/// Runs a callback once according to a [`SingletonScope`].
pub struct SingletonSprite {
    scope: SingletonScope,
    callback: Box<dyn FnMut(&mut FrameContext<'_>)>,
    monitor: bool,
    touched: bool,
}

impl SingletonSprite {
    pub fn new(
        scope: SingletonScope,
        callback: impl FnMut(&mut FrameContext<'_>) + 'static,
    ) -> Self {
        Self {
            scope,
            callback: Box::new(callback),
            monitor: false,
            touched: false,
        }
    }

    pub fn in_monitor(mut self) -> Self {
        self.monitor = true;
        self
    }
}

impl SceneObject for SingletonSprite {
    fn rect(&self) -> Rect {
        Rect::new(0, 0, 0, 0)
    }
    fn visible(&self) -> bool {
        false
    }
    fn update(&mut self, ctx: &mut FrameContext<'_>) {
        if !self.monitor
            && self
                .scope
                .should_run(ctx.frame.frame, ctx.frame.scene_frame, self.touched)
        {
            (self.callback)(ctx);
        }
        self.touched = true;
    }
    fn monitor(&mut self, ctx: &mut FrameContext<'_>) {
        if self.monitor
            && self
                .scope
                .should_run(ctx.frame.frame, ctx.frame.scene_frame, self.touched)
        {
            (self.callback)(ctx);
        }
        if self.monitor {
            self.touched = true;
        }
    }
}

/// A moving image that removes itself after its lifetime expires.
pub struct BubbleSprite {
    pub state: DynamicSprite,
    image: Arc<Image>,
    vx: f32,
    vy: f32,
    lifetime: Duration,
    age: Duration,
    auto_delete: bool,
}

impl BubbleSprite {
    pub fn new(
        image: impl Into<Arc<Image>>,
        x: f32,
        y: f32,
        vx: f32,
        vy: f32,
        lifetime: Duration,
        z: i32,
    ) -> Self {
        Self {
            state: DynamicSprite::new(x.round() as i32, y.round() as i32, z),
            image: image.into(),
            vx,
            vy,
            lifetime,
            age: Duration::ZERO,
            auto_delete: true,
        }
    }

    pub fn with_auto_delete(mut self, auto_delete: bool) -> Self {
        self.auto_delete = auto_delete;
        self
    }

    pub fn age(&self) -> Duration {
        self.age
    }

    pub fn set_velocity(&mut self, vx: f32, vy: f32) {
        self.vx = vx;
        self.vy = vy;
    }
}

impl SceneObject for BubbleSprite {
    fn rect(&self) -> Rect {
        image_rect(self.state.x, self.state.y, &self.image)
    }
    fn z(&self) -> i32 {
        self.state.z
    }
    fn visible(&self) -> bool {
        self.age < self.lifetime
    }
    fn update(&mut self, ctx: &mut FrameContext<'_>) {
        self.age = self.age.saturating_add(ctx.frame.delta);
        let seconds = ctx.frame.delta.as_secs_f32();
        self.state.x = self
            .state
            .x
            .saturating_add((self.vx * seconds).round() as i32);
        self.state.y = self
            .state
            .y
            .saturating_add((self.vy * seconds).round() as i32);
        if self.auto_delete && self.age >= self.lifetime {
            ctx.despawn(ctx.id);
        }
    }
    fn render(&self, canvas: &mut Canvas<'_>, _: &FrameInfo) {
        canvas.draw_image(self.state.x, self.state.y, self.state.z, &self.image);
    }
}

/// Re-centers an image on each axis every frame.
pub struct CenteredImageSprite {
    image: Arc<Image>,
    horizontal: bool,
    vertical: bool,
    offset_x: i32,
    offset_y: i32,
    z: i32,
}

impl CenteredImageSprite {
    pub fn new(image: impl Into<Arc<Image>>, horizontal: bool, vertical: bool, z: i32) -> Self {
        Self {
            image: image.into(),
            horizontal,
            vertical,
            offset_x: 0,
            offset_y: 0,
            z,
        }
    }

    pub fn with_offset(mut self, x: i32, y: i32) -> Self {
        self.offset_x = x;
        self.offset_y = y;
        self
    }

    fn position(&self, frame: &FrameInfo) -> (i32, i32) {
        let x = if self.horizontal {
            (i32::from(frame.width) - i32::from(self.image.width())) / 2 + self.offset_x
        } else {
            self.offset_x
        };
        let y = if self.vertical {
            (i32::from(frame.height) - i32::from(self.image.height())) / 2 + self.offset_y
        } else {
            self.offset_y
        };
        (x, y)
    }
}

impl SceneObject for CenteredImageSprite {
    fn rect(&self) -> Rect {
        // Centering depends on the terminal size, which is only available at
        // render time; the full-screen rect ensures camera culling never hides it.
        Rect::new(0, 0, u16::MAX, u16::MAX)
    }
    fn z(&self) -> i32 {
        self.z
    }
    fn render(&self, canvas: &mut Canvas<'_>, frame: &FrameInfo) {
        let (x, y) = self.position(frame);
        canvas.draw_image(x, y, self.z, &self.image);
    }
}

/// Splits a source image into fixed cells and places selected tiles.
pub struct TileMapper {
    pub image: Arc<Image>,
    pub tile_width: u16,
    pub tile_height: u16,
}

impl TileMapper {
    pub fn new(image: impl Into<Arc<Image>>, tile_width: u16, tile_height: u16) -> Self {
        Self {
            image: image.into(),
            tile_width,
            tile_height,
        }
    }

    /// Maps non-transparent tile indices to image sprites in row-major order.
    /// A `None` entry leaves a gap.
    pub fn map(
        &self,
        map: &[Vec<Option<usize>>],
        origin_x: i32,
        origin_y: i32,
        z: i32,
    ) -> Result<Vec<ImageSprite>, String> {
        if self.tile_width == 0 || self.tile_height == 0 {
            return Err("tile dimensions must be non-zero".into());
        }
        let cols = (self.image.width() / self.tile_width) as usize;
        let rows = (self.image.height() / self.tile_height) as usize;
        let count = cols.checked_mul(rows).ok_or("tile count overflow")?;
        let mut sprites = Vec::new();
        for (row, tiles) in map.iter().enumerate() {
            for (col, tile) in tiles.iter().enumerate() {
                let Some(index) = tile else { continue };
                if *index >= count {
                    return Err(format!("tile index {index} is out of range 0..{count}"));
                }
                let sx = ((index % cols) * self.tile_width as usize) as u16;
                let sy = ((index / cols) * self.tile_height as usize) as u16;
                let crop = self.image.crop(Rect::new(
                    i32::from(sx),
                    i32::from(sy),
                    self.tile_width,
                    self.tile_height,
                ));
                let x = origin_x.saturating_add((col * self.tile_width as usize) as i32);
                let y = origin_y.saturating_add((row * self.tile_height as usize) as i32);
                sprites.push(ImageSprite::new(crop, x, y, z));
            }
        }
        Ok(sprites)
    }
}

/// Convenience raster image used by tests and simple backgrounds.
pub fn solid_image(width: u16, height: u16, pixel: Pixel) -> Arc<Image> {
    Arc::new(Image::new(width, height, pixel))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::{Animation, VideoClip};
    use crate::engine::{Engine, Scene};
    use crate::graphics::Color;
    use crate::particles::{ConfettiConfig, ConfettiEmitter, ParticleEmitter};
    use std::cell::Cell;
    use std::rc::Rc;

    fn white_image(width: u16, height: u16, ch: char) -> Arc<Image> {
        Arc::new(Image::new(
            width,
            height,
            Pixel::new(ch, Color::WHITE, None),
        ))
    }

    #[test]
    fn image_label_and_static_sprites_render() {
        let mut engine = Engine::new(8, 2);
        let mut scene = Scene::new("scene", Pixel::new('.', Color::WHITE, None));
        scene.add_object(ImageSprite::new(white_image(2, 1, 'A'), 1, 0, 1));
        scene.add_object(
            LabelSprite::new(
                Arc::new(crate::font::SystemFont::new()),
                "hi",
                Color::WHITE,
                None,
                4,
                0,
                2,
            )
            .unwrap(),
        );
        engine.add_scene(scene).unwrap();
        engine.start("scene").unwrap();
        engine.tick(Duration::from_millis(10), &[]).unwrap();
        assert_eq!(engine.screen().snapshot(), ".AA.hi..\n........");
    }

    #[test]
    fn animation_sprite_advances_and_changes() {
        let first = Animation::film_strip(
            "first",
            Duration::from_millis(1),
            true,
            false,
            vec![white_image(1, 1, '1'), white_image(1, 1, '2')],
        )
        .unwrap();
        let second = Animation::film_strip(
            "second",
            Duration::from_millis(1),
            true,
            false,
            vec![white_image(1, 1, 'B')],
        )
        .unwrap();
        let mut sprite =
            AnimationSprite::with_animations(vec![first, second], "first", 0, 0, 0).unwrap();
        let mut engine = Engine::new(2, 1);
        let mut scene = Scene::new("s", Pixel::new('.', Color::WHITE, None));
        scene.add_object(sprite);
        engine.add_scene(scene).unwrap();
        engine.start("s").unwrap();
        engine.tick(Duration::ZERO, &[]).unwrap();
        assert_eq!(engine.screen().snapshot(), "1.");
        engine.tick(Duration::from_millis(5), &[]).unwrap();
        engine.tick(Duration::from_millis(5), &[]).unwrap();
        let mut engine2 = Engine::new(2, 1);
        let first = Animation::film_strip(
            "first",
            Duration::from_millis(1),
            true,
            false,
            vec![white_image(1, 1, '1')],
        )
        .unwrap();
        let second = Animation::film_strip(
            "second",
            Duration::from_millis(1),
            true,
            false,
            vec![white_image(1, 1, 'B')],
        )
        .unwrap();
        sprite = AnimationSprite::with_animations(vec![first, second], "first", 0, 0, 0).unwrap();
        sprite.change("second", true).unwrap();
        let mut scene = Scene::new("s", Pixel::new('.', Color::WHITE, None));
        scene.add_object(sprite);
        engine2.add_scene(scene).unwrap();
        engine2.start("s").unwrap();
        engine2.tick(Duration::ZERO, &[]).unwrap();
        assert_eq!(engine2.screen().snapshot(), "B.");
    }

    #[test]
    fn video_sprite_advances_across_frames() {
        let clip = VideoClip::new(
            "v",
            vec![white_image(1, 1, 'X'), white_image(1, 1, 'Y')],
            1000.0,
        )
        .unwrap();
        let player = VideoPlayer::new(Arc::new(clip), true, true);
        let mut engine = Engine::new(2, 1);
        let mut scene = Scene::new("s", Pixel::new('.', Color::WHITE, None));
        scene.add_object(VideoSprite::new(player, 0, 0, 0));
        engine.add_scene(scene).unwrap();
        engine.start("s").unwrap();
        engine.tick(Duration::ZERO, &[]).unwrap();
        assert_eq!(engine.screen().snapshot(), "X.");
        engine.tick(Duration::from_millis(1100), &[]).unwrap();
        assert_eq!(engine.screen().snapshot(), "Y.");
    }

    #[test]
    fn particle_sprite_emits_particles_and_reports_bounds() {
        let mut system = ParticleSystem::new();
        let mut emitter = ConfettiEmitter::fixed_with(
            "c",
            ConfettiConfig {
                count_per_burst: 2,
                burst_interval_ms: 1,
                ..ConfettiConfig::default()
            },
            4.0,
            3.0,
        );
        emitter.resume(true);
        system.add_emitter(Box::new(emitter));
        let mut engine = Engine::new(10, 6);
        let mut scene = Scene::new("s", Pixel::new('.', Color::WHITE, None));
        scene.add_object(ParticleSprite::new(system, 0, 0, 1));
        engine.add_scene(scene).unwrap();
        engine.start("s").unwrap();
        engine.tick(Duration::from_millis(5), &[]).unwrap();
        assert!(engine
            .screen()
            .snapshot()
            .chars()
            .any(|c| c != '.' && c != ' '));
    }

    #[test]
    fn keyboard_callbacks_are_one_shot_until_released() {
        let count = Rc::new(Cell::new(0));
        let count2 = count.clone();
        let mut engine = Engine::new(1, 1);
        let mut scene = Scene::new("s", Pixel::new('.', Color::WHITE, None));
        scene.add_object(KeyboardSprite::on_key(Key::Space, move |_| {
            count2.set(count2.get() + 1)
        }));
        engine.add_scene(scene).unwrap();
        engine.start("s").unwrap();
        engine.tick(Duration::ZERO, &[Key::Space]).unwrap();
        engine.tick(Duration::ZERO, &[Key::Space]).unwrap();
        engine.tick(Duration::ZERO, &[]).unwrap();
        engine.tick(Duration::ZERO, &[Key::Space]).unwrap();
        assert_eq!(count.get(), 2);
    }

    #[test]
    fn offscreen_singleton_and_bubble_callbacks_work() {
        let updates = Rc::new(Cell::new(0));
        let updates2 = updates.clone();
        let mut engine = Engine::new(4, 1);
        let mut scene = Scene::new("s", Pixel::new('.', Color::WHITE, None));
        scene.add_object(OffScreenSprite::new(move |_| {
            updates2.set(updates2.get() + 1)
        }));
        scene.add_object(BubbleSprite::new(
            white_image(1, 1, 'B'),
            0.0,
            0.0,
            0.0,
            0.0,
            Duration::from_millis(25),
            0,
        ));
        engine.add_scene(scene).unwrap();
        engine.start("s").unwrap();
        engine.tick(Duration::from_millis(10), &[]).unwrap();
        engine.tick(Duration::from_millis(10), &[]).unwrap();
        engine.tick(Duration::from_millis(10), &[]).unwrap();
        assert_eq!(updates.get(), 3);
        assert!(!engine.screen().snapshot().contains('B'));
    }

    #[test]
    fn centered_sprite_and_tile_mapper_place_content() {
        let mut engine = Engine::new(5, 3);
        let mut scene = Scene::new("s", Pixel::new('.', Color::WHITE, None));
        scene.add_object(CenteredImageSprite::new(
            white_image(1, 1, 'C'),
            true,
            true,
            0,
        ));
        let atlas = white_image(2, 1, 'T');
        let mapper = TileMapper::new(atlas, 1, 1);
        let tiles = mapper
            .map(&[vec![Some(0), None, Some(1)]], 0, 0, 0)
            .unwrap();
        assert_eq!(tiles.len(), 2);
        assert_eq!(tiles[1].rect().x, 2);
        assert!(mapper.map(&[vec![Some(2)]], 0, 0, 0).is_err());
        scene.add_object(ImageSprite::new(white_image(1, 1, 'Z'), 0, 0, 0));
        engine.add_scene(scene).unwrap();
        engine.start("s").unwrap();
        engine.tick(Duration::ZERO, &[]).unwrap();
        assert!(engine.screen().snapshot().contains('C'));
        assert!(engine.screen().snapshot().contains('Z'));
    }
}
