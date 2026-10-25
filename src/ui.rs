// SPDX-License-Identifier: Apache-2.0
//! Terminal widgets and dependency-driven layout composition.

use crate::engine::{FrameContext, FrameInfo, SceneObject};
use crate::graphics::{Canvas, Color, Pixel, Rect};
use crate::layout::{Layout, LayoutError};
use crate::terminal::Key;
use crate::text::Align;

/// A scene object that can be positioned by [`LayoutSprite`].
pub trait Layoutable: SceneObject {
    fn layout_id(&self) -> &str;
    fn size(&self) -> (u16, u16);
    fn position(&self) -> (i32, i32);
    fn set_position(&mut self, x: i32, y: i32);
}

fn fit(text: &str, width: u16) -> String {
    let mut result: String = text.chars().take(usize::from(width)).collect();
    let len = result.chars().count();
    result.extend(std::iter::repeat_n(' ', usize::from(width) - len));
    result
}

fn aligned(text: &str, width: u16, align: Align) -> String {
    let text = fit(text, width);
    let count = text.chars().count();
    let spare = usize::from(width).saturating_sub(count);
    let (left, right) = match align {
        Align::Left => (0, spare),
        Align::Center => (spare / 2, spare - spare / 2),
        Align::Right => (spare, 0),
    };
    format!("{}{}{}", " ".repeat(left), text, " ".repeat(right))
}

/// A non-rendering, sized placeholder for layout spacing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spacer {
    id: String,
    x: i32,
    y: i32,
    width: u16,
    height: u16,
    z: i32,
}

impl Spacer {
    pub fn new(id: impl Into<String>, width: u16, height: u16) -> Self {
        Self {
            id: id.into(),
            x: 0,
            y: 0,
            width,
            height,
            z: 0,
        }
    }

    pub fn at(mut self, x: i32, y: i32) -> Self {
        self.x = x;
        self.y = y;
        self
    }

    pub fn set_size(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
    }
}

impl SceneObject for Spacer {
    fn rect(&self) -> Rect {
        Rect::new(self.x, self.y, self.width, self.height)
    }
    fn z(&self) -> i32 {
        self.z
    }
    fn visible(&self) -> bool {
        false
    }
}

impl Layoutable for Spacer {
    fn layout_id(&self) -> &str {
        &self.id
    }
    fn size(&self) -> (u16, u16) {
        (self.width, self.height)
    }
    fn position(&self) -> (i32, i32) {
        (self.x, self.y)
    }
    fn set_position(&mut self, x: i32, y: i32) {
        self.x = x;
        self.y = y;
    }
}

/// A filled panel with a configurable border and centered title.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TitlePanel {
    id: String,
    x: i32,
    y: i32,
    z: i32,
    width: u16,
    height: u16,
    title: String,
    title_align: Align,
    fill: Pixel,
    border: Pixel,
    corners: [char; 4],
    horizontal: char,
    vertical: char,
    border_color: Color,
}

impl TitlePanel {
    pub fn new(id: impl Into<String>, width: u16, height: u16, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            x: 0,
            y: 0,
            z: 0,
            width,
            height,
            title: title.into(),
            title_align: Align::Center,
            fill: Pixel::new(' ', Color::WHITE, Some(Color::BLACK)),
            border: Pixel::new(' ', Color::WHITE, None),
            corners: ['+', '+', '+', '+'],
            horizontal: '-',
            vertical: '|',
            border_color: Color::WHITE,
        }
    }

    pub fn at(mut self, x: i32, y: i32, z: i32) -> Self {
        self.x = x;
        self.y = y;
        self.z = z;
        self
    }

    pub fn with_colors(mut self, fill: Pixel, border: Color) -> Self {
        self.fill = fill;
        self.border_color = border;
        self.border = Pixel::new(self.border.ch, border, None);
        self
    }

    pub fn with_border_chars(
        mut self,
        corners: [char; 4],
        horizontal: char,
        vertical: char,
    ) -> Self {
        self.corners = corners;
        self.horizontal = horizontal;
        self.vertical = vertical;
        self
    }

    pub fn with_alignment(mut self, align: Align) -> Self {
        self.title_align = align;
        self
    }

    pub fn set_title(&mut self, title: impl Into<String>) {
        self.title = title.into();
    }

    pub fn title(&self) -> &str {
        &self.title
    }

    pub fn set_size(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
    }
}

impl SceneObject for TitlePanel {
    fn rect(&self) -> Rect {
        Rect::new(self.x, self.y, self.width, self.height)
    }
    fn z(&self) -> i32 {
        self.z
    }
    fn render(&self, canvas: &mut Canvas<'_>, _: &FrameInfo) {
        if self.width == 0 || self.height == 0 {
            return;
        }
        let rect = self.rect();
        canvas.fill_rect(rect, self.z, self.fill);
        let border = Pixel::new(' ', self.border_color, self.border.bg);
        canvas.draw_rect(rect, self.z + 1, border);
        if self.width < 2 || self.height < 2 {
            return;
        }
        let (x0, y0) = (self.x, self.y);
        let (x1, y1) = (
            self.x + i32::from(self.width) - 1,
            self.y + i32::from(self.height) - 1,
        );
        canvas.draw_pixel(
            x0,
            y0,
            self.z + 2,
            Pixel::new(self.corners[0], self.border_color, self.fill.bg),
        );
        canvas.draw_pixel(
            x1,
            y0,
            self.z + 2,
            Pixel::new(self.corners[1], self.border_color, self.fill.bg),
        );
        canvas.draw_pixel(
            x1,
            y1,
            self.z + 2,
            Pixel::new(self.corners[2], self.border_color, self.fill.bg),
        );
        canvas.draw_pixel(
            x0,
            y1,
            self.z + 2,
            Pixel::new(self.corners[3], self.border_color, self.fill.bg),
        );
        if self.width > 2 {
            for x in (x0 + 1)..x1 {
                canvas.draw_pixel(
                    x,
                    y0,
                    self.z + 2,
                    Pixel::new(self.horizontal, self.border_color, self.fill.bg),
                );
                canvas.draw_pixel(
                    x,
                    y1,
                    self.z + 2,
                    Pixel::new(self.horizontal, self.border_color, self.fill.bg),
                );
            }
        }
        if self.height > 2 {
            for y in (y0 + 1)..y1 {
                canvas.draw_pixel(
                    x0,
                    y,
                    self.z + 2,
                    Pixel::new(self.vertical, self.border_color, self.fill.bg),
                );
                canvas.draw_pixel(
                    x1,
                    y,
                    self.z + 2,
                    Pixel::new(self.vertical, self.border_color, self.fill.bg),
                );
            }
        }
        if !self.title.is_empty() && self.width > 2 {
            let row = aligned(&self.title, self.width - 2, self.title_align);
            canvas.draw_text(x0 + 1, y0, self.z + 3, &row, self.fill.fg, self.fill.bg);
        }
    }
}

impl Layoutable for TitlePanel {
    fn layout_id(&self) -> &str {
        &self.id
    }
    fn size(&self) -> (u16, u16) {
        (self.width, self.height)
    }
    fn position(&self) -> (i32, i32) {
        (self.x, self.y)
    }
    fn set_position(&mut self, x: i32, y: i32) {
        self.x = x;
        self.y = y;
    }
}

/// One model item in a [`ListBox`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ListBoxElement<T> {
    pub value: T,
    pub text: String,
    pub enabled: bool,
}

impl<T> ListBoxElement<T> {
    pub fn new(text: impl Into<String>, value: T) -> Self {
        Self {
            value,
            text: text.into(),
            enabled: true,
        }
    }

    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }
}

/// A vertically scrolling, selection-aware list.
pub struct ListBox<T> {
    id: String,
    x: i32,
    y: i32,
    z: i32,
    width: u16,
    height: u16,
    elements: Vec<ListBoxElement<T>>,
    selected: usize,
    scroll: usize,
    scroll_x: usize,
    normal: Pixel,
    selected_skin: Pixel,
    disabled: Pixel,
    last_activated: Option<T>,
}

impl<T: Clone> ListBox<T> {
    pub fn new(id: impl Into<String>, width: u16, height: u16) -> Self {
        Self {
            id: id.into(),
            x: 0,
            y: 0,
            z: 0,
            width,
            height,
            elements: Vec::new(),
            selected: 0,
            scroll: 0,
            scroll_x: 0,
            normal: Pixel::new(' ', Color::WHITE, None),
            selected_skin: Pixel::new(' ', Color::BLACK, Some(Color::WHITE)),
            disabled: Pixel::new(' ', Color::rgb(100, 100, 100), None),
            last_activated: None,
        }
    }

    pub fn at(mut self, x: i32, y: i32, z: i32) -> Self {
        self.x = x;
        self.y = y;
        self.z = z;
        self
    }

    pub fn with_colors(mut self, normal: Pixel, selected: Pixel, disabled: Pixel) -> Self {
        self.normal = normal;
        self.selected_skin = selected;
        self.disabled = disabled;
        self
    }

    pub fn add_item(&mut self, text: impl Into<String>, value: T) {
        self.elements.push(ListBoxElement::new(text, value));
    }

    pub fn extend(&mut self, elements: impl IntoIterator<Item = ListBoxElement<T>>) {
        self.elements.extend(elements);
    }

    pub fn elements(&self) -> &[ListBoxElement<T>] {
        &self.elements
    }

    pub fn selected_index(&self) -> usize {
        self.selected
    }

    pub fn selected(&self) -> Option<&T> {
        self.elements.get(self.selected).map(|e| &e.value)
    }

    pub fn take_activated(&mut self) -> Option<T> {
        self.last_activated.take()
    }

    pub fn set_selection(&mut self, index: usize) {
        if !self.elements.is_empty() {
            self.selected = index.min(self.elements.len() - 1);
            self.ensure_visible();
        }
    }

    pub fn move_selection(&mut self, delta: i64) {
        if self.elements.is_empty() {
            return;
        }
        self.selected =
            (self.selected as i64 + delta).clamp(0, self.elements.len() as i64 - 1) as usize;
        self.ensure_visible();
    }

    fn ensure_visible(&mut self) {
        let rows = usize::from(self.height.max(1));
        if self.selected < self.scroll {
            self.scroll = self.selected;
        } else if self.selected >= self.scroll + rows {
            self.scroll = self.selected + 1 - rows;
        }
    }

    pub fn handle_key(&mut self, key: Key) -> Option<T> {
        match key {
            Key::Up => self.move_selection(-1),
            Key::Down => self.move_selection(1),
            Key::PageUp => self.move_selection(-(i64::from(self.height.max(1)))),
            Key::PageDown => self.move_selection(i64::from(self.height.max(1))),
            Key::Home => self.set_selection(0),
            Key::End => self.set_selection(usize::MAX),
            Key::Enter | Key::Space => {
                if let Some(element) = self.elements.get(self.selected) {
                    if element.enabled {
                        self.last_activated = Some(element.value.clone());
                    }
                }
            }
            _ => return None,
        }
        None
    }
}

impl<T: Clone + 'static> SceneObject for ListBox<T> {
    fn rect(&self) -> Rect {
        Rect::new(self.x, self.y, self.width, self.height)
    }
    fn z(&self) -> i32 {
        self.z
    }
    fn update(&mut self, ctx: &mut FrameContext<'_>) {
        for key in ctx.frame.keys.clone() {
            self.handle_key(key);
        }
    }
    fn render(&self, canvas: &mut Canvas<'_>, _: &FrameInfo) {
        if self.width == 0 {
            return;
        }
        let rows = usize::from(self.height);
        for row in 0..rows {
            let Some(element) = self.elements.get(self.scroll + row) else {
                break;
            };
            let selected = self.scroll + row == self.selected;
            let skin = if !element.enabled {
                self.disabled
            } else if selected {
                self.selected_skin
            } else {
                self.normal
            };
            let text: String = element.text.chars().skip(self.scroll_x).collect();
            canvas.draw_text(
                self.x,
                self.y + row as i32,
                self.z,
                &fit(&text, self.width),
                skin.fg,
                skin.bg,
            );
        }
    }
}

impl<T: Clone + 'static> Layoutable for ListBox<T> {
    fn layout_id(&self) -> &str {
        &self.id
    }
    fn size(&self) -> (u16, u16) {
        (self.width, self.height)
    }
    fn position(&self) -> (i32, i32) {
        (self.x, self.y)
    }
    fn set_position(&mut self, x: i32, y: i32) {
        self.x = x;
        self.y = y;
    }
}

/// A one-row text editing control.
pub struct TextInput {
    id: String,
    x: i32,
    y: i32,
    z: i32,
    width: u16,
    buffer: Vec<char>,
    cursor: usize,
    scroll: usize,
    active: bool,
    ready: bool,
    result: Option<String>,
    filter: Box<dyn Fn(char) -> bool>,
    active_skin: Pixel,
    inactive_skin: Pixel,
    placeholder: Option<String>,
    password: bool,
}

impl TextInput {
    pub fn new(id: impl Into<String>, width: u16) -> Self {
        Self {
            id: id.into(),
            x: 0,
            y: 0,
            z: 0,
            width,
            buffer: Vec::new(),
            cursor: 0,
            scroll: 0,
            active: true,
            ready: false,
            result: None,
            filter: Box::new(|_| true),
            active_skin: Pixel::new(' ', Color::WHITE, Some(Color::BLACK)),
            inactive_skin: Pixel::new(' ', Color::rgb(120, 120, 120), None),
            placeholder: None,
            password: false,
        }
    }

    pub fn at(mut self, x: i32, y: i32, z: i32) -> Self {
        self.x = x;
        self.y = y;
        self.z = z;
        self
    }

    pub fn with_filter(mut self, filter: impl Fn(char) -> bool + 'static) -> Self {
        self.filter = Box::new(filter);
        self
    }

    pub fn with_placeholder(mut self, text: impl Into<String>) -> Self {
        self.placeholder = Some(text.into());
        self
    }

    pub fn password(mut self, password: bool) -> Self {
        self.password = password;
        self
    }

    pub fn with_skin(mut self, active: Pixel, inactive: Pixel) -> Self {
        self.active_skin = active;
        self.inactive_skin = inactive;
        self
    }

    pub fn text(&self) -> String {
        self.buffer.iter().collect()
    }

    pub fn set_text(&mut self, text: &str) {
        self.buffer = text.chars().filter(|c| (self.filter)(*c)).collect();
        self.cursor = self.buffer.len();
        self.ready = false;
        self.result = None;
        self.ensure_visible();
    }

    pub fn activate(&mut self) {
        self.active = true;
    }
    pub fn deactivate(&mut self) {
        self.active = false;
    }
    pub fn focus(&mut self, active: bool) {
        self.active = active;
    }
    pub fn is_ready(&self) -> bool {
        self.ready
    }
    pub fn is_active(&self) -> bool {
        self.active
    }
    pub fn take_result(&mut self) -> Option<String> {
        self.result.take()
    }

    fn ensure_visible(&mut self) {
        let width = usize::from(self.width.max(1));
        if self.cursor < self.scroll {
            self.scroll = self.cursor;
        } else if self.cursor >= self.scroll + width {
            self.scroll = self.cursor + 1 - width;
        }
    }

    fn submit(&mut self) {
        self.ready = true;
        self.result = Some(self.text());
        self.active = false;
        self.cursor = self.buffer.len();
        self.ensure_visible();
    }

    pub fn handle_key(&mut self, key: Key) {
        if !self.active {
            return;
        }
        match key {
            Key::Char(c) => {
                if (self.filter)(c) && !c.is_control() {
                    self.buffer.insert(self.cursor, c);
                    self.cursor += 1;
                    self.ensure_visible();
                }
            }
            Key::Backspace => {
                if self.cursor > 0 {
                    self.cursor -= 1;
                    self.buffer.remove(self.cursor);
                    self.ensure_visible();
                }
            }
            Key::Delete => {
                if self.cursor < self.buffer.len() {
                    self.buffer.remove(self.cursor);
                }
            }
            Key::Left => {
                self.cursor = self.cursor.saturating_sub(1);
                self.ensure_visible();
            }
            Key::Right => {
                self.cursor = (self.cursor + 1).min(self.buffer.len());
                self.ensure_visible();
            }
            Key::Home => {
                self.cursor = 0;
                self.ensure_visible();
            }
            Key::End => {
                self.cursor = self.buffer.len();
                self.ensure_visible();
            }
            Key::Enter => self.submit(),
            Key::Escape => {
                self.buffer.clear();
                self.cursor = 0;
                self.ready = false;
                self.result = None;
                self.ensure_visible();
            }
            _ => {}
        }
    }
}

impl SceneObject for TextInput {
    fn rect(&self) -> Rect {
        Rect::new(self.x, self.y, self.width, 1)
    }
    fn z(&self) -> i32 {
        self.z
    }
    fn update(&mut self, ctx: &mut FrameContext<'_>) {
        for key in ctx.frame.keys.clone() {
            self.handle_key(key);
        }
    }
    fn render(&self, canvas: &mut Canvas<'_>, _: &FrameInfo) {
        let skin = if self.active {
            self.active_skin
        } else {
            self.inactive_skin
        };
        let mut line = String::new();
        for (index, ch) in self.buffer.iter().enumerate() {
            if !self.password && index == self.cursor && self.active {
                // The cursor is represented by an inverse space inserted only
                // when it can fit without consuming an input character.
                if line.chars().count() < usize::from(self.width) {
                    line.push(' ');
                }
            }
            line.push(if self.password { '*' } else { *ch });
        }
        if self.buffer.is_empty() {
            if let Some(placeholder) = &self.placeholder {
                line = placeholder.clone();
            }
        }
        if self.active
            && self.cursor >= self.buffer.len()
            && line.chars().count() < usize::from(self.width)
        {
            line.push(' ');
        }
        let visible: String = line.chars().skip(self.scroll).collect();
        canvas.draw_text(
            self.x,
            self.y,
            self.z,
            &fit(&visible, self.width),
            skin.fg,
            skin.bg,
        );
    }
}

impl Layoutable for TextInput {
    fn layout_id(&self) -> &str {
        &self.id
    }
    fn size(&self) -> (u16, u16) {
        (self.width, 1)
    }
    fn position(&self) -> (i32, i32) {
        (self.x, self.y)
    }
    fn set_position(&mut self, x: i32, y: i32) {
        self.x = x;
        self.y = y;
    }
}

/// Composes [`Layoutable`] children under one scene object.
pub struct LayoutSprite {
    children: Vec<Box<dyn Layoutable>>,
    layout: Layout,
    width: u16,
    height: u16,
    z: i32,
    last_error: Option<LayoutError>,
}

impl LayoutSprite {
    pub fn new(width: u16, height: u16, layout: Layout) -> Self {
        Self {
            children: Vec::new(),
            layout,
            width,
            height,
            z: 0,
            last_error: None,
        }
    }

    pub fn with_z(mut self, z: i32) -> Self {
        self.z = z;
        self
    }

    pub fn add(&mut self, child: impl Layoutable + 'static) {
        self.children.push(Box::new(child));
    }

    pub fn add_boxed(&mut self, child: Box<dyn Layoutable>) {
        self.children.push(child);
    }

    pub fn children(&self) -> &[Box<dyn Layoutable>] {
        &self.children
    }

    pub fn children_mut(&mut self) -> &mut [Box<dyn Layoutable>] {
        &mut self.children
    }

    pub fn last_error(&self) -> Option<&LayoutError> {
        self.last_error.as_ref()
    }

    pub fn take_error(&mut self) -> Option<LayoutError> {
        self.last_error.take()
    }

    fn resolve(&mut self) -> Result<(), LayoutError> {
        let canvas = Rect::new(0, 0, self.width, self.height);
        // Layout resolves one shared own-size for all objects. Resolve each
        // child against its own size by grouping children by size.
        let mut grouped: std::collections::BTreeMap<(u16, u16), Layout> =
            std::collections::BTreeMap::new();
        for child in &self.children {
            if let Some(spec) = self.layout.get(child.layout_id()) {
                grouped
                    .entry(child.size())
                    .or_default()
                    .insert(child.layout_id(), spec.clone());
            }
        }
        let mut positions = std::collections::BTreeMap::new();
        // Cross-size dependencies are resolved iteratively: each pass publishes
        // already-known positions as external anchors for the next size group.
        for _ in 0..=self.children.len() {
            let external: std::collections::BTreeMap<_, _> = self
                .children
                .iter()
                .map(|child| {
                    let (x, y) = positions
                        .get(child.layout_id())
                        .copied()
                        .unwrap_or_else(|| child.position());
                    let (width, height) = child.size();
                    (
                        child.layout_id().to_string(),
                        Rect::new(x, y, width, height),
                    )
                })
                .collect();
            let before = positions.clone();
            for ((width, height), layout) in &grouped {
                positions.extend(layout.resolve(&external, canvas, (*width, *height))?);
            }
            if positions == before {
                break;
            }
        }
        for child in &mut self.children {
            if let Some(&(x, y)) = positions.get(child.layout_id()) {
                child.set_position(x, y);
            }
        }
        Ok(())
    }
}

impl SceneObject for LayoutSprite {
    fn rect(&self) -> Rect {
        Rect::new(0, 0, self.width, self.height)
    }
    fn z(&self) -> i32 {
        self.z
    }
    fn on_start(&mut self) {
        for child in &mut self.children {
            child.on_start();
        }
    }
    fn on_stop(&mut self) {
        for child in &mut self.children {
            child.on_stop();
        }
    }
    fn on_activate(&mut self) {
        for child in &mut self.children {
            child.on_activate();
        }
    }
    fn on_deactivate(&mut self) {
        for child in &mut self.children {
            child.on_deactivate();
        }
    }
    fn update(&mut self, ctx: &mut FrameContext<'_>) {
        for child in &mut self.children {
            child.update(ctx);
        }
    }
    fn monitor(&mut self, ctx: &mut FrameContext<'_>) {
        if let Err(error) = self.resolve() {
            self.last_error = Some(error);
        }
        for child in &mut self.children {
            child.monitor(ctx);
        }
    }
    fn render(&self, canvas: &mut Canvas<'_>, frame: &FrameInfo) {
        let mut children: Vec<_> = self.children.iter().collect();
        children.sort_by_key(|child| child.z());
        for child in children {
            child.render(canvas, frame);
        }
    }
}

impl LayoutSprite {
    /// Read access to a child by layout ID (useful in tests and controllers).
    pub fn child(&self, id: &str) -> Option<&dyn Layoutable> {
        self.children
            .iter()
            .find(|child| child.layout_id() == id)
            .map(Box::as_ref)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Engine;
    use crate::layout::LayoutSpec;

    fn scene(pixel: char) -> crate::engine::Scene {
        crate::engine::Scene::new("s", Pixel::new(pixel, Color::WHITE, None))
    }

    #[test]
    fn title_panel_draws_border_title_and_fill() {
        let mut engine = Engine::new(9, 4);
        let mut s = scene('.');
        s.add_object(
            TitlePanel::new("p", 7, 3, "Hi")
                .at(1, 0, 1)
                .with_alignment(Align::Left),
        );
        engine.add_scene(s).unwrap();
        engine.start("s").unwrap();
        engine.tick(std::time::Duration::ZERO, &[]).unwrap();
        let frame = engine.screen().snapshot();
        assert_eq!(frame.lines().next().unwrap(), ".+Hi   +.");
        assert!(frame.contains("+-----+"));
    }

    #[test]
    fn list_box_navigates_scrolls_and_activates() {
        let mut list = ListBox::new("list", 4, 2);
        list.extend((0..4).map(|i| ListBoxElement::new(format!("item{i}"), i)));
        list.handle_key(Key::End);
        assert_eq!(list.selected_index(), 3);
        list.handle_key(Key::Down);
        assert_eq!(list.selected_index(), 3);
        list.handle_key(Key::Up);
        assert_eq!(list.selected_index(), 2);
        list.handle_key(Key::Enter);
        assert_eq!(list.take_activated(), Some(2));
        let mut engine = Engine::new(4, 2);
        let mut s = scene('.');
        s.add_object(list);
        engine.add_scene(s).unwrap();
        engine.start("s").unwrap();
        engine.tick(std::time::Duration::ZERO, &[]).unwrap();
        assert!(engine.screen().snapshot().contains("item"));
    }

    #[test]
    fn text_input_edits_filters_and_submits() {
        let mut input = TextInput::new("input", 5)
            .with_filter(|c| c.is_ascii_digit())
            .with_placeholder("12345");
        for ch in "12a3".chars() {
            input.handle_key(Key::Char(ch));
        }
        assert_eq!(input.text(), "123");
        input.handle_key(Key::Left);
        input.handle_key(Key::Backspace);
        assert_eq!(input.text(), "13");
        input.handle_key(Key::Home);
        input.handle_key(Key::Delete);
        assert_eq!(input.text(), "3");
        input.handle_key(Key::Enter);
        assert!(input.is_ready());
        assert_eq!(input.take_result().as_deref(), Some("3"));
        input.activate();
        input.handle_key(Key::Char('9'));
        assert_eq!(input.text(), "39");
        input.handle_key(Key::Escape);
        assert!(input.text().is_empty());
    }

    #[test]
    fn password_input_and_render() {
        let mut engine = Engine::new(5, 1);
        let mut input = TextInput::new("password", 5).password(true);
        input.set_text("secret");
        let mut s = scene('.');
        s.add_object(input);
        engine.add_scene(s).unwrap();
        engine.start("s").unwrap();
        engine.tick(std::time::Duration::ZERO, &[]).unwrap();
        assert_eq!(engine.screen().snapshot(), "**** ");
    }

    #[test]
    fn layout_sprite_positions_children_by_their_own_size() {
        let mut layout = Layout::new();
        layout.insert("a", LayoutSpec::at(1, 1));
        layout.insert("b", LayoutSpec::after("a").with_offset(2));
        let mut group = LayoutSprite::new(20, 6, layout);
        group.add(Spacer::new("a", 3, 2));
        group.add(TitlePanel::new("b", 4, 2, "X"));
        let mut engine = Engine::new(20, 6);
        let mut s = scene('.');
        s.add_object(group);
        engine.add_scene(s).unwrap();
        engine.start("s").unwrap();
        engine.tick(std::time::Duration::ZERO, &[]).unwrap();
        let frame = engine.screen().snapshot();
        let lines: Vec<_> = frame.lines().collect();
        // `a` is placed at x=1 with width 3; `b` follows it after a two-cell gap.
        assert_eq!(&lines[1][6..10], "+X +");
    }
}
