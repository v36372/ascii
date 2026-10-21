// SPDX-License-Identifier: Apache-2.0
//! Character-cell graphics with clipping and immediate-mode z compositing.

use unicode_width::UnicodeWidthChar;

/// An RGB color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    pub const BLACK: Self = Self::rgb(0, 0, 0);
    pub const WHITE: Self = Self::rgb(255, 255, 255);

    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }
}

/// One character cell. NUL denotes a transparent, non-drawing pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pixel {
    pub ch: char,
    pub fg: Color,
    pub bg: Option<Color>,
}

impl Pixel {
    pub fn new(ch: char, fg: Color, bg: Option<Color>) -> Self {
        Self {
            ch: sanitize_glyph(ch),
            fg,
            bg,
        }
    }

    pub const fn transparent() -> Self {
        Self {
            ch: '\0',
            fg: Color::BLACK,
            bg: None,
        }
    }

    pub const fn is_transparent(&self) -> bool {
        self.ch == '\0'
    }
}

fn sanitize_glyph(ch: char) -> char {
    if ch == '\0' {
        ch
    } else if ch.is_control() || matches!(ch, '\u{2028}' | '\u{2029}') || ch.width() != Some(1) {
        '?'
    } else {
        ch
    }
}

/// A half-open rectangle. Its exclusive far edges may exceed `i32::MAX`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u16,
    pub height: u16,
}

impl Rect {
    pub const fn new(x: i32, y: i32, width: u16, height: u16) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn contains(&self, x: i32, y: i32) -> bool {
        i64::from(x) >= i64::from(self.x)
            && i64::from(y) >= i64::from(self.y)
            && i64::from(x) < self.right()
            && i64::from(y) < self.bottom()
    }

    pub fn intersects(&self, other: Rect) -> bool {
        self.intersection(other).is_some()
    }

    pub fn intersection(&self, other: Rect) -> Option<Rect> {
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let right = self.right().min(other.right());
        let bottom = self.bottom().min(other.bottom());
        if right <= i64::from(x) || bottom <= i64::from(y) {
            None
        } else {
            // An intersection cannot be wider/taller than either input.
            Some(Self::new(
                x,
                y,
                (right - i64::from(x)) as u16,
                (bottom - i64::from(y)) as u16,
            ))
        }
    }

    fn right(self) -> i64 {
        i64::from(self.x) + i64::from(self.width)
    }

    fn bottom(self) -> i64 {
        i64::from(self.y) + i64::from(self.height)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cell {
    pub pixel: Pixel,
    pub z: i32,
}

/// An owned row-major framebuffer. Clearing restores the original background
/// and `i32::MIN` depth, so even a draw at the minimum depth can replace it.
#[derive(Debug)]
pub struct Screen {
    width: u16,
    height: u16,
    background: Pixel,
    cells: Vec<Cell>,
}

impl Screen {
    pub fn new(width: u16, height: u16, background: Pixel) -> Self {
        // Public Pixel fields allow callers to bypass its constructor.
        let background = Pixel::new(background.ch, background.fg, background.bg);
        Self {
            width,
            height,
            background,
            cells: vec![
                Cell {
                    pixel: background,
                    z: i32::MIN,
                };
                usize::from(width) * usize::from(height)
            ],
        }
    }

    pub fn width(&self) -> u16 {
        self.width
    }

    pub fn height(&self) -> u16 {
        self.height
    }

    pub fn bounds(&self) -> Rect {
        Rect::new(0, 0, self.width, self.height)
    }

    pub fn get(&self, x: i32, y: i32) -> Option<&Cell> {
        if self.bounds().contains(x, y) {
            Some(&self.cells[y as usize * usize::from(self.width) + x as usize])
        } else {
            None
        }
    }

    pub fn clear(&mut self) {
        self.cells.fill(Cell {
            pixel: self.background,
            z: i32::MIN,
        });
    }

    pub fn canvas(&mut self) -> Canvas<'_> {
        let clip = self.bounds();
        Canvas { screen: self, clip }
    }

    /// Glyph rows separated by newlines, without a trailing row separator.
    /// Transparent background cells appear as spaces, never as NUL controls.
    pub fn snapshot(&self) -> String {
        let mut result = String::new();
        for y in 0..usize::from(self.height) {
            if y != 0 {
                result.push('\n');
            }
            let start = y * usize::from(self.width);
            for cell in &self.cells[start..start + usize::from(self.width)] {
                result.push(if cell.pixel.is_transparent() {
                    ' '
                } else {
                    cell.pixel.ch
                });
            }
        }
        result
    }
}

/// A drawing view of a screen with its own replaceable clip rectangle.
pub struct Canvas<'a> {
    screen: &'a mut Screen,
    clip: Rect,
}

impl Canvas<'_> {
    /// The full screen bounds, independent of the current clip.
    pub fn bounds(&self) -> Rect {
        self.screen.bounds()
    }

    /// Replaces the clip (rather than intersecting the previous clip).
    pub fn set_clip(&mut self, rect: Rect) {
        self.clip = self
            .bounds()
            .intersection(rect)
            .unwrap_or(Rect::new(0, 0, 0, 0));
    }

    pub fn draw_pixel(&mut self, x: i32, y: i32, z: i32, pixel: Pixel) {
        if pixel.is_transparent() || !self.clip.contains(x, y) {
            return;
        }
        let index = y as usize * usize::from(self.screen.width) + x as usize;
        let cell = &mut self.screen.cells[index];
        if z >= cell.z {
            *cell = Cell {
                pixel: Pixel::new(pixel.ch, pixel.fg, pixel.bg.or(cell.pixel.bg)),
                z,
            };
        }
    }

    /// Draws one cell per Unicode scalar value on a single row. Controls,
    /// including newlines and tabs, become '?'; NUL leaves a transparent gap.
    pub fn draw_text(&mut self, x: i32, y: i32, z: i32, text: &str, fg: Color, bg: Option<Color>) {
        if self.clip.width == 0
            || y < self.clip.y
            || i64::from(y) >= self.clip.bottom()
            || i64::from(x) >= self.clip.right()
        {
            return;
        }
        let start = i64::from(x).max(i64::from(self.clip.x));
        let skip = (start - i64::from(x)) as usize;
        let count = (self.clip.right() - start) as usize;
        // Skip existing text, not offscreen coordinates; stop at the right edge.
        for (offset, ch) in text.chars().skip(skip).take(count).enumerate() {
            self.draw_pixel((start + offset as i64) as i32, y, z, Pixel::new(ch, fg, bg));
        }
    }

    pub fn fill_rect(&mut self, rect: Rect, z: i32, pixel: Pixel) {
        if pixel.is_transparent() {
            return;
        }
        if let Some(visible) = rect.intersection(self.clip) {
            for y in i64::from(visible.y)..visible.bottom() {
                for x in i64::from(visible.x)..visible.right() {
                    self.draw_pixel(x as i32, y as i32, z, pixel);
                }
            }
        }
    }

    /// Draws the original perimeter only; clipping never creates new edges.
    pub fn draw_rect(&mut self, rect: Rect, z: i32, pixel: Pixel) {
        if pixel.is_transparent() {
            return;
        }
        let Some(visible) = rect.intersection(self.clip) else {
            return;
        };
        let top = i64::from(rect.y);
        let bottom = rect.bottom() - 1;
        let left = i64::from(rect.x);
        let right = rect.right() - 1;
        for y in [top, bottom] {
            if y >= i64::from(visible.y) && y < visible.bottom() {
                for x in i64::from(visible.x)..visible.right() {
                    self.draw_pixel(x as i32, y as i32, z, pixel);
                }
            }
        }
        for x in [left, right] {
            if x >= i64::from(visible.x) && x < visible.right() {
                for y in i64::from(visible.y)..visible.bottom() {
                    self.draw_pixel(x as i32, y as i32, z, pixel);
                }
            }
        }
    }

    /// An inclusive integer line, with Bresenham's nearest-cell rasterization.
    /// Halfway ties advance toward the end point. Work is bounded by the clip's
    /// major-axis extent, even for endpoints at opposite i32 extremes.
    pub fn draw_line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, z: i32, pixel: Pixel) {
        if pixel.is_transparent() || self.clip.width == 0 || self.clip.height == 0 {
            return;
        }
        let dx = (i64::from(x1) - i64::from(x0)).abs();
        let dy = (i64::from(y1) - i64::from(y0)).abs();
        if dx == 0 && dy == 0 {
            self.draw_pixel(x0, y0, z, pixel);
            return;
        }
        let steep = dy > dx;
        let (a0, b0, a1, b1, low, high) = if steep {
            (
                y0,
                x0,
                y1,
                x1,
                i64::from(self.clip.y),
                self.clip.bottom() - 1,
            )
        } else {
            (
                x0,
                y0,
                x1,
                y1,
                i64::from(self.clip.x),
                self.clip.right() - 1,
            )
        };
        let a0 = i64::from(a0);
        let b0 = i64::from(b0);
        let da = i64::from(a1) - a0;
        let db = i64::from(b1) - b0;
        let major = da.abs();
        let minor = db.abs();
        let (first, last) = if da > 0 {
            ((low - a0).max(0), (high - a0).min(major))
        } else {
            ((a0 - high).max(0), (a0 - low).min(major))
        };
        // Jump directly to visible major-axis steps without changing the line's
        // original error phase. i128 avoids overflow in the extreme products.
        for step in first..=last {
            let advance = ((i128::from(step) * i128::from(minor) + i128::from(major / 2))
                / i128::from(major)) as i64;
            let a = (a0 + da.signum() * step) as i32;
            let b = (b0 + db.signum() * advance) as i32;
            let (x, y) = if steep { (b, a) } else { (a, b) };
            self.draw_pixel(x, y, z, pixel);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RED: Color = Color::rgb(255, 0, 0);
    const BLUE: Color = Color::rgb(0, 0, 255);

    fn px(ch: char) -> Pixel {
        Pixel::new(ch, Color::WHITE, None)
    }

    fn screen(width: u16, height: u16) -> Screen {
        Screen::new(width, height, px('.'))
    }

    #[test]
    fn colors_and_pixels_are_copy_and_equal() {
        assert_eq!(Color::BLACK, Color::rgb(0, 0, 0));
        assert_eq!(
            Color::WHITE,
            Color {
                r: 255,
                g: 255,
                b: 255
            }
        );
        let pixel = Pixel::new('x', RED, Some(BLUE));
        let copy = pixel;
        assert_eq!(pixel, copy);
        assert_eq!(copy.fg, RED);
        assert_eq!(copy.bg, Some(BLUE));
        assert!(!pixel.is_transparent());
        assert!(Pixel::transparent().is_transparent());
        assert!(Pixel::new('\0', RED, Some(BLUE)).is_transparent());
    }

    #[test]
    fn glyph_sanitizing_preserves_only_single_column_characters() {
        for ch in ['a', ' ', 'é', '─', '█'] {
            assert_eq!(px(ch).ch, ch);
        }
        for ch in [
            '\n', '\r', '\t', '\x1b', '\x7f', '\u{85}', '界', '🦀', '\u{301}', '\u{200d}',
            '\u{fe0f}', '\u{2028}', '\u{2029}',
        ] {
            assert_eq!(px(ch).ch, '?', "glyph {ch:?}");
        }
    }

    #[test]
    fn rectangle_coordinates_are_half_open() {
        let rect = Rect::new(-2, 3, 4, 2);
        for (x, y) in [(-2, 3), (1, 4)] {
            assert!(rect.contains(x, y));
        }
        for (x, y) in [(-3, 3), (2, 3), (0, 2), (0, 5)] {
            assert!(!rect.contains(x, y));
        }
        let overlap = Rect::new(0, 4, 5, 3);
        assert_eq!(rect.intersection(overlap), Some(Rect::new(0, 4, 2, 1)));
        assert_eq!(rect.intersection(overlap), overlap.intersection(rect));
        assert!(rect.intersects(overlap));
        for other in [
            Rect::new(2, 3, 1, 2),
            Rect::new(-2, 5, 4, 1),
            Rect::new(0, 3, 0, 1),
            Rect::new(-2, 3, 4, 0),
        ] {
            assert!(!rect.intersects(other));
            assert!(!other.intersects(rect));
            assert_eq!(rect.intersection(other), None);
        }
        assert!(!Rect::new(1, 1, 0, 1).contains(1, 1));
        assert!(!Rect::new(1, 1, 1, 0).contains(1, 1));
    }

    #[test]
    fn rectangles_do_not_overflow_at_integer_extremes() {
        let high = Rect::new(i32::MAX, i32::MAX, u16::MAX, u16::MAX);
        let low = Rect::new(i32::MIN, i32::MIN, u16::MAX, u16::MAX);
        assert!(high.contains(i32::MAX, i32::MAX));
        assert!(low.contains(i32::MIN, i32::MIN));
        assert_eq!(high.intersection(high), Some(high));
        assert_eq!(low.intersection(low), Some(low));
        assert!(!high.intersects(low));
        assert_eq!(
            high.intersection(Rect::new(i32::MAX - 1, i32::MAX - 1, 2, 2)),
            Some(Rect::new(i32::MAX, i32::MAX, 1, 1))
        );
        assert!(!Rect::new(i32::MIN, i32::MAX, 0, 0).contains(i32::MIN, i32::MAX));
    }

    #[test]
    fn screen_bounds_lookup_and_clear() {
        let bg = Pixel::new('.', RED, Some(BLUE));
        let mut screen = Screen::new(3, 2, bg);
        assert_eq!((screen.width(), screen.height()), (3, 2));
        assert_eq!(screen.bounds(), Rect::new(0, 0, 3, 2));
        assert_eq!(
            screen.get(2, 1),
            Some(&Cell {
                pixel: bg,
                z: i32::MIN
            })
        );
        for (x, y) in [(-1, 0), (0, -1), (3, 0), (0, 2), (i32::MIN, i32::MAX)] {
            assert_eq!(screen.get(x, y), None);
        }
        screen.canvas().draw_pixel(1, 1, i32::MAX, px('X'));
        assert_eq!(screen.snapshot(), "...\n.X.");
        screen.clear();
        assert_eq!(screen.snapshot(), "...\n...");
        assert_eq!(
            screen.get(1, 1),
            Some(&Cell {
                pixel: bg,
                z: i32::MIN
            })
        );
        screen.canvas().draw_pixel(1, 1, i32::MIN, px('x'));
        assert_eq!(screen.snapshot(), "...\n.x.");
    }

    #[test]
    fn z_order_transparency_and_background_inheritance() {
        let mut screen = Screen::new(1, 1, Pixel::new('.', Color::WHITE, Some(BLUE)));
        screen.canvas().draw_pixel(0, 0, -10, px('a'));
        assert_eq!(screen.get(0, 0).unwrap().pixel.bg, Some(BLUE));
        screen
            .canvas()
            .draw_pixel(0, 0, -11, Pixel::new('b', RED, Some(RED)));
        assert_eq!(screen.snapshot(), "a");
        assert_eq!(screen.get(0, 0).unwrap().pixel.bg, Some(BLUE));
        screen
            .canvas()
            .draw_pixel(0, 0, -10, Pixel::new('c', RED, Some(RED)));
        screen.canvas().draw_pixel(0, 0, 1, px('d'));
        assert_eq!(
            screen.get(0, 0),
            Some(&Cell {
                pixel: Pixel::new('d', Color::WHITE, Some(RED)),
                z: 1
            })
        );
        screen
            .canvas()
            .draw_pixel(0, 0, i32::MAX, Pixel::new('\0', BLUE, Some(BLUE)));
        assert_eq!(screen.get(0, 0).unwrap().z, 1);
        screen.canvas().draw_pixel(0, 0, 2, px('e'));
        assert_eq!(screen.snapshot(), "e");
        assert_eq!(screen.get(0, 0).unwrap().pixel.bg, Some(RED));
        screen.canvas().draw_pixel(0, 0, i32::MAX, px('f'));
        screen.canvas().draw_pixel(0, 0, i32::MAX, px('g'));
        assert_eq!(screen.snapshot(), "g");
    }

    #[test]
    fn absent_background_remains_absent() {
        let mut screen = screen(1, 1);
        screen.canvas().draw_pixel(0, 0, 0, px('x'));
        assert_eq!(screen.get(0, 0).unwrap().pixel.bg, None);
    }

    #[test]
    fn public_pixel_fields_cannot_bypass_sanitizing() {
        let invalid = Pixel {
            ch: '\x1b',
            fg: RED,
            bg: Some(BLUE),
        };
        let mut screen = Screen::new(2, 1, invalid);
        assert_eq!(screen.snapshot(), "??");
        screen.canvas().draw_pixel(
            0,
            0,
            1,
            Pixel {
                ch: '界',
                ..invalid
            },
        );
        assert_eq!(
            screen.get(0, 0).unwrap().pixel,
            Pixel::new('?', RED, Some(BLUE))
        );
        screen.clear();
        assert_eq!(screen.snapshot(), "??");
    }

    #[test]
    fn transparent_background_snapshots_as_spaces() {
        let mut screen = Screen::new(3, 1, Pixel::transparent());
        assert!(screen.get(0, 0).unwrap().pixel.is_transparent());
        assert_eq!(screen.snapshot(), "   ");
        screen.canvas().draw_pixel(1, 0, i32::MIN, px('é'));
        assert_eq!(screen.snapshot(), " é ");
        screen.clear();
        assert_eq!(screen.snapshot(), "   ");
    }

    #[test]
    fn clip_is_intersected_with_screen_and_is_replaceable() {
        let mut screen = screen(5, 3);
        {
            let mut canvas = screen.canvas();
            canvas.set_clip(Rect::new(-2, -2, 4, 4));
            assert_eq!(canvas.bounds(), Rect::new(0, 0, 5, 3));
            canvas.fill_rect(Rect::new(-100, -100, 200, 200), 0, px('a'));
            canvas.set_clip(Rect::new(i32::MAX, i32::MIN, 10, 10));
            canvas.fill_rect(canvas.bounds(), 1, px('!'));
            canvas.set_clip(Rect::new(3, 1, 20, 20));
            canvas.fill_rect(canvas.bounds(), 0, px('b'));
        }
        assert_eq!(screen.snapshot(), "aa...\naa.bb\n...bb");
        screen.canvas().draw_pixel(2, 2, 1, px('c'));
        assert_eq!(screen.snapshot(), "aa...\naa.bb\n..cbb");
    }

    #[test]
    fn text_uses_scalar_cells_and_clips_without_wrapping() {
        let mut screen = screen(7, 3);
        let mut canvas = screen.canvas();
        canvas.draw_text(-2, 0, 0, "abé界\u{301}Z\n!", RED, Some(BLUE));
        canvas.draw_text(1, 1, 0, "a\0b\tX", Color::WHITE, None);
        canvas.set_clip(Rect::new(2, 2, 3, 1));
        canvas.draw_text(0, 2, 0, "0123456789", RED, None);
        canvas.draw_text(i32::MIN, 2, 1, "short", RED, None);
        canvas.draw_text(i32::MAX, 2, 1, "offscreen", RED, None);
        canvas.draw_text(0, i32::MAX, 1, "offscreen", RED, None);
        assert_eq!(screen.snapshot(), "é??Z?!.\n.a.b?X.\n..234..");
        assert_eq!(screen.get(0, 0).unwrap().pixel.bg, Some(BLUE));
    }

    #[test]
    fn rectangle_snapshots_and_degenerate_dimensions() {
        let mut screen = screen(9, 6);
        let mut canvas = screen.canvas();
        canvas.fill_rect(Rect::new(1, 1, 6, 4), 0, px('+'));
        canvas.draw_rect(Rect::new(1, 1, 6, 4), 1, px('#'));
        canvas.draw_rect(Rect::new(3, 2, 1, 2), 2, px('|'));
        canvas.draw_rect(Rect::new(5, 2, 3, 1), 2, px('-'));
        canvas.draw_rect(Rect::new(8, 5, 1, 1), 2, px('X'));
        canvas.draw_rect(Rect::new(0, 0, 0, 6), 3, px('!'));
        canvas.fill_rect(Rect::new(0, 0, 9, 0), 3, px('!'));
        assert_eq!(
            screen.snapshot(),
            ".........\n.######..\n.#+|+---.\n.#+|++#..\n.######..\n........X"
        );
    }

    #[test]
    fn clipping_rectangles_does_not_invent_perimeters() {
        let mut screen = screen(5, 4);
        let mut canvas = screen.canvas();
        canvas.draw_rect(Rect::new(-1, -1, 7, 6), 0, px('!'));
        canvas.set_clip(Rect::new(1, 1, 3, 2));
        canvas.draw_rect(Rect::new(0, 0, 4, 3), 1, px('#'));
        assert_eq!(screen.snapshot(), ".....\n...#.\n.###.\n.....");
    }

    #[test]
    fn extreme_rectangles_are_clipped_before_iteration() {
        let mut screen = screen(4, 3);
        let mut canvas = screen.canvas();
        for origin in [i32::MIN, i32::MAX] {
            let rect = Rect::new(origin, origin, u16::MAX, u16::MAX);
            canvas.fill_rect(rect, 1, px('!'));
            canvas.draw_rect(rect, 1, px('!'));
        }
        canvas.fill_rect(Rect::new(-1, -1, u16::MAX, u16::MAX), 0, px('+'));
        canvas.draw_rect(Rect::new(-1, -1, u16::MAX, u16::MAX), 1, px('!'));
        assert_eq!(screen.snapshot(), "++++\n++++\n++++");
    }

    #[test]
    fn line_snapshot_includes_both_endpoints() {
        let mut screen = screen(9, 5);
        let mut canvas = screen.canvas();
        canvas.draw_line(0, 0, 8, 4, 0, px('x'));
        canvas.draw_line(0, 4, 8, 0, 0, px('o'));
        canvas.draw_line(4, 0, 4, 4, 1, px('|'));
        canvas.draw_line(0, 2, 8, 2, 2, px('-'));
        canvas.draw_line(4, 2, 4, 2, 3, px('+'));
        assert_eq!(
            screen.snapshot(),
            "x...|..oo\n.xx.|oo..\n----+----\n.oo.|xx..\no...|..xx"
        );
    }

    // A deliberately unoptimized reference verifies the jump-ahead formula,
    // including tie-breaking, every octant and clipping of the original raster.
    fn reference_line(screen: &mut Screen, clip: Rect, start: (i32, i32), end: (i32, i32)) {
        let (mut x, mut y) = start;
        let (x1, y1) = end;
        let dx = (x1 - x).abs();
        let dy = -(y1 - y).abs();
        let sx = if x < x1 { 1 } else { -1 };
        let sy = if y < y1 { 1 } else { -1 };
        let mut error = dx + dy;
        let mut canvas = screen.canvas();
        canvas.set_clip(clip);
        loop {
            canvas.draw_pixel(x, y, 0, px('#'));
            if x == x1 && y == y1 {
                break;
            }
            let twice = 2 * error;
            if twice >= dy {
                error += dy;
                x += sx;
            }
            if twice <= dx {
                error += dx;
                y += sy;
            }
        }
    }

    #[test]
    fn clipped_lines_match_bresenham_in_every_octant() {
        for clip in [Rect::new(0, 0, 5, 5), Rect::new(1, 2, 3, 2)] {
            for x0 in -2..=6 {
                for y0 in -2..=6 {
                    for x1 in -2..=6 {
                        for y1 in -2..=6 {
                            let mut actual = screen(5, 5);
                            let mut expected = screen(5, 5);
                            let mut canvas = actual.canvas();
                            canvas.set_clip(clip);
                            canvas.draw_line(x0, y0, x1, y1, 0, px('#'));
                            reference_line(&mut expected, clip, (x0, y0), (x1, y1));
                            assert_eq!(
                                actual.cells, expected.cells,
                                "({x0},{y0})->({x1},{y1}), clip {clip:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn enormous_lines_are_bounded_and_do_not_overflow() {
        let mut screen = screen(5, 5);
        let mut canvas = screen.canvas();
        canvas.draw_line(i32::MIN, i32::MIN, i32::MAX, i32::MAX, 0, px('x'));
        canvas.draw_line(i32::MAX, 2, i32::MIN, 2, 1, px('-'));
        canvas.draw_line(2, i32::MIN, 2, i32::MAX, 2, px('|'));
        canvas.draw_line(i32::MIN, i32::MAX, i32::MAX, i32::MIN, 3, px('!'));
        canvas.draw_line(i32::MIN, i32::MIN, i32::MIN, i32::MAX, 3, px('!'));
        canvas.draw_line(i32::MIN, i32::MAX, i32::MAX, i32::MAX, 3, px('!'));
        assert_eq!(screen.snapshot(), "x.|..\n.x|..\n--|--\n..|x.\n..|.x");
    }

    #[test]
    fn all_primitives_respect_transparency_and_empty_clips() {
        let mut screen = screen(3, 3);
        let mut canvas = screen.canvas();
        for pixel in [Pixel::transparent(), px('!')] {
            if !pixel.is_transparent() {
                canvas.set_clip(Rect::new(0, 0, 0, 3));
            }
            canvas.draw_pixel(1, 1, i32::MAX, pixel);
            canvas.draw_line(i32::MIN, i32::MIN, i32::MAX, i32::MAX, i32::MAX, pixel);
            canvas.fill_rect(canvas.bounds(), i32::MAX, pixel);
            canvas.draw_rect(canvas.bounds(), i32::MAX, pixel);
        }
        canvas.draw_text(0, 0, 0, "oops", RED, None);
        assert_eq!(screen.snapshot(), "...\n...\n...");
        assert!(screen.cells.iter().all(|cell| cell.z == i32::MIN));
    }

    #[test]
    fn empty_screens_are_safe_for_every_primitive() {
        for (width, height, snapshot) in [(0, 0, ""), (0, 3, "\n\n"), (3, 0, "")] {
            let mut screen = screen(width, height);
            assert_eq!(screen.get(0, 0), None);
            let mut canvas = screen.canvas();
            canvas.set_clip(Rect::new(0, 0, u16::MAX, u16::MAX));
            canvas.draw_pixel(0, 0, 0, px('!'));
            canvas.draw_text(0, 0, 0, "oops", RED, None);
            canvas.fill_rect(Rect::new(0, 0, u16::MAX, u16::MAX), 0, px('!'));
            canvas.draw_rect(Rect::new(0, 0, u16::MAX, u16::MAX), 0, px('!'));
            canvas.draw_line(i32::MIN, i32::MIN, i32::MAX, i32::MAX, 0, px('!'));
            screen.clear();
            assert_eq!(screen.snapshot(), snapshot);
        }
    }
}
