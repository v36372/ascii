// SPDX-License-Identifier: Apache-2.0
//! Character-cell graphics with clipping and immediate-mode z compositing.

use crate::image::Image;
use std::collections::HashSet;
use std::fmt;
use unicode_width::UnicodeWidthChar;

/// The maximum number of cells a single [`Canvas::flood_fill`] call will visit.
///
/// The limit keeps flood fills on very large canvases from allocating without
/// bound. A fill that reaches this many cells stops early.
pub const MAX_FLOOD_FILL_CELLS: usize = 262_144;

/// Radii beyond this are drawn by a visible-row scanline fallback.
const MIDPOINT_CIRCLE_LIMIT: i64 = 4_096;

/// An error returned when a string cannot be parsed as an RGB color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorParseError {
    /// The string was empty after removing any prefix and surrounding spaces.
    Empty,
    /// The string did not contain exactly three or six hexadecimal digits.
    InvalidLength,
    /// A character was not a hexadecimal digit.
    InvalidDigit(char),
}

impl fmt::Display for ColorParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("empty color string"),
            Self::InvalidLength => f.write_str("expected 3 or 6 hexadecimal digits"),
            Self::InvalidDigit(ch) => write!(f, "invalid hexadecimal digit {ch:?}"),
        }
    }
}

impl std::error::Error for ColorParseError {}

/// An RGB color.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    pub const BLACK: Self = Self::rgb(0, 0, 0);
    pub const WHITE: Self = Self::rgb(255, 255, 255);

    /// Basic dark grays matching CosPlay's `C_GRAY1` through `C_GRAY4`.
    pub const GRAY1: Self = Self::rgb(0x11, 0x11, 0x11);
    pub const GRAY2: Self = Self::rgb(0x22, 0x22, 0x22);
    pub const GRAY3: Self = Self::rgb(0x33, 0x33, 0x33);
    pub const GRAY4: Self = Self::rgb(0x44, 0x44, 0x44);

    /// CosPlay's default dark background color (`C_DFLT_BG`).
    pub const DFLT_BG: Self = Self::rgb(0x11, 0x11, 0x11);

    // Lower-case aliases make the CosPlay names convenient to spell while the
    // upper-case constants above remain the idiomatic Rust names.
    #[allow(non_upper_case_globals)]
    pub const gray1: Self = Self::GRAY1;
    #[allow(non_upper_case_globals)]
    pub const gray2: Self = Self::GRAY2;
    #[allow(non_upper_case_globals)]
    pub const gray3: Self = Self::GRAY3;
    #[allow(non_upper_case_globals)]
    pub const gray4: Self = Self::GRAY4;
    #[allow(non_upper_case_globals)]
    pub const dflt_bg: Self = Self::DFLT_BG;

    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Parses `#rrggbb`, `0xrrggbb`, `rrggbb`, or the three-digit CSS shorthand
    /// `#rgb`. Surrounding whitespace and hexadecimal letter case are ignored.
    pub fn from_hex(value: &str) -> Result<Self, ColorParseError> {
        let value = value.trim();
        if value.is_empty() {
            return Err(ColorParseError::Empty);
        }
        let bytes = value.as_bytes();
        let digits = if bytes.first() == Some(&b'#') {
            &value[1..]
        } else if bytes.len() >= 2 && bytes[0] == b'0' && matches!(bytes[1], b'x' | b'X') {
            &value[2..]
        } else {
            value
        };
        if digits.is_empty() {
            return Err(ColorParseError::Empty);
        }
        // Taking seven scalars both rejects long strings and keeps allocation
        // bounded even for hostile input containing multi-byte characters.
        let chars: Vec<char> = digits.chars().take(7).collect();
        let mut nibbles = [0_u8; 6];
        match chars.len() {
            3 => {
                for (index, ch) in chars.iter().enumerate() {
                    let digit = ch.to_digit(16).ok_or(ColorParseError::InvalidDigit(*ch))?;
                    nibbles[index] = (digit * 17) as u8;
                }
                Ok(Self::rgb(nibbles[0], nibbles[1], nibbles[2]))
            }
            6 => {
                for (index, pair) in chars.as_chunks::<2>().0.iter().enumerate() {
                    let high = pair[0]
                        .to_digit(16)
                        .ok_or(ColorParseError::InvalidDigit(pair[0]))?;
                    let low = pair[1]
                        .to_digit(16)
                        .ok_or(ColorParseError::InvalidDigit(pair[1]))?;
                    nibbles[index] = ((high << 4) | low) as u8;
                }
                Ok(Self::rgb(nibbles[0], nibbles[1], nibbles[2]))
            }
            _ => Err(ColorParseError::InvalidLength),
        }
    }

    /// Lower-case CSS-style hexadecimal, e.g. `#ff0080`.
    pub fn hex(&self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }

    /// Alias for [`Color::hex`], the form accepted by CSS.
    pub fn css_hex(&self) -> String {
        self.hex()
    }

    /// ITU-R BT.709 luma rounded to an 8-bit channel value.
    pub fn luma(&self) -> u8 {
        (0.2126 * f32::from(self.r) + 0.7152 * f32::from(self.g) + 0.0722 * f32::from(self.b))
            .round()
            .clamp(0.0, 255.0) as u8
    }

    /// Returns a copy interpolated toward white by `factor`.
    ///
    /// Non-finite factors are treated as zero and finite factors are clamped to
    /// `[0, 1]`.
    pub fn lighter(&self, factor: f32) -> Self {
        let factor = unit_factor(factor);
        Self::rgb(
            channel(f32::from(self.r) + f32::from(255 - self.r) * factor),
            channel(f32::from(self.g) + f32::from(255 - self.g) * factor),
            channel(f32::from(self.b) + f32::from(255 - self.b) * factor),
        )
    }

    /// Returns a copy interpolated toward black by `factor`.
    ///
    /// Non-finite factors are treated as zero and finite factors are clamped to
    /// `[0, 1]`.
    pub fn darker(&self, factor: f32) -> Self {
        let remaining = 1.0 - unit_factor(factor);
        Self::rgb(
            channel(f32::from(self.r) * remaining),
            channel(f32::from(self.g) * remaining),
            channel(f32::from(self.b) * remaining),
        )
    }

    /// Mixes two colors. A `balance` of zero returns `self` and one returns
    /// `other`; non-finite values are treated as zero and finite values clamp.
    pub fn mixture(&self, other: Self, balance: f32) -> Self {
        let balance = unit_factor(balance);
        Self::rgb(
            channel(f32::from(self.r) + (f32::from(other.r) - f32::from(self.r)) * balance),
            channel(f32::from(self.g) + (f32::from(other.g) - f32::from(self.g)) * balance),
            channel(f32::from(self.b) + (f32::from(other.b) - f32::from(self.b)) * balance),
        )
    }

    pub const fn with_r(self, r: u8) -> Self {
        Self { r, ..self }
    }

    pub const fn with_g(self, g: u8) -> Self {
        Self { g, ..self }
    }

    pub const fn with_b(self, b: u8) -> Self {
        Self { b, ..self }
    }

    /// Returns the photographic negative of this color.
    pub const fn invert(self) -> Self {
        Self::rgb(255 - self.r, 255 - self.g, 255 - self.b)
    }

    /// Converts RGB to hue, saturation, and brightness.
    ///
    /// All three components are normalized to `[0, 1]`, matching Java's
    /// `Color.RGBtoHSB` and CosPlay. Grays have a hue of zero.
    pub fn rgb_to_hsb(self) -> (f32, f32, f32) {
        let r = f32::from(self.r) / 255.0;
        let g = f32::from(self.g) / 255.0;
        let b = f32::from(self.b) / 255.0;
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;
        let brightness = max;
        let saturation = if max == 0.0 { 0.0 } else { delta / max };
        let hue = if delta == 0.0 {
            0.0
        } else if max == r {
            ((g - b) / delta).rem_euclid(6.0)
        } else if max == g {
            (b - r) / delta + 2.0
        } else {
            (r - g) / delta + 4.0
        } / 6.0;
        (
            hue.clamp(0.0, 1.0),
            saturation.clamp(0.0, 1.0),
            brightness.clamp(0.0, 1.0),
        )
    }

    /// Creates a color from normalized hue, saturation, and brightness.
    ///
    /// Saturation and brightness are clamped to `[0, 1]`. A finite hue wraps
    /// modulo one; a non-finite hue is treated as zero. This mirrors Java's
    /// `Color.HSBtoRGB` convention used by CosPlay.
    pub fn from_hsb(hue: f32, saturation: f32, brightness: f32) -> Self {
        let hue = if hue.is_finite() {
            hue.rem_euclid(1.0)
        } else {
            0.0
        };
        let saturation = unit_factor(saturation);
        let brightness = unit_factor(brightness);
        if saturation == 0.0 {
            let gray = channel(brightness * 255.0);
            return Self::rgb(gray, gray, gray);
        }
        let sector = hue * 6.0;
        let index = sector.floor();
        let fraction = sector - index;
        let p = brightness * (1.0 - saturation);
        let q = brightness * (1.0 - saturation * fraction);
        let t = brightness * (1.0 - saturation * (1.0 - fraction));
        let (r, g, b) = match index as i32 {
            0 => (brightness, t, p),
            1 => (q, brightness, p),
            2 => (p, brightness, t),
            3 => (p, q, brightness),
            4 => (t, p, brightness),
            _ => (brightness, p, q),
        };
        Self::rgb(channel(r * 255.0), channel(g * 255.0), channel(b * 255.0))
    }

    /// Returns `steps` colors from just after `start` through `end`, inclusive.
    ///
    /// This is CosPlay's `gradientSeq` behavior: `start` itself is not part of
    /// the result and the final element is exactly `end`. Zero steps produces an
    /// empty vector.
    pub fn gradient(start: Self, end: Self, steps: usize) -> Vec<Self> {
        let mut colors = Vec::with_capacity(steps);
        for index in 1..=steps {
            colors.push(start.mixture(end, index as f32 / steps as f32));
        }
        colors
    }

    /// Approximates this color with the nearest xterm-256 palette entry.
    ///
    /// The adaptive color-cube and grayscale comparison follows CosPlay's
    /// `toXterm` implementation, so results are in the `16..=255` range.
    pub fn nearest_xterm(self) -> u8 {
        const LEVELS: [i32; 6] = [0x00, 0x5f, 0x87, 0xaf, 0xd7, 0xff];

        fn cube_index(value: i32) -> i32 {
            if value < 48 {
                0
            } else if value < 115 {
                1
            } else {
                (value - 35) / 40
            }
        }
        fn distance(r1: i32, g1: i32, b1: i32, r2: i32, g2: i32, b2: i32) -> i32 {
            (r1 - r2) * (r1 - r2) + (g1 - g2) * (g1 - g2) + (b1 - b2) * (b1 - b2)
        }

        let r = i32::from(self.r);
        let g = i32::from(self.g);
        let b = i32::from(self.b);
        let ir = cube_index(r);
        let ig = cube_index(g);
        let ib = cube_index(b);
        let cube = 36 * ir + 6 * ig + ib;
        let average = (r + g + b) / 3;
        let gray = if average > 238 {
            23
        } else {
            (average - 3) / 10
        };
        let cube_r = LEVELS[ir as usize];
        let cube_g = LEVELS[ig as usize];
        let cube_b = LEVELS[ib as usize];
        let gray_value = 8 + 10 * gray;
        let cube_error = distance(cube_r, cube_g, cube_b, r, g, b);
        let gray_error = distance(gray_value, gray_value, gray_value, r, g, b);
        if cube_error <= gray_error {
            (16 + cube) as u8
        } else {
            (232 + gray) as u8
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }
}

impl std::str::FromStr for Color {
    type Err = ColorParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::from_hex(value)
    }
}

fn unit_factor(factor: f32) -> f32 {
    if factor.is_nan() {
        0.0
    } else {
        factor.clamp(0.0, 1.0)
    }
}

fn channel(value: f32) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
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

    /// Alias for [`Rect::contains`] that names the point form explicitly.
    pub fn contains_point(&self, x: i32, y: i32) -> bool {
        self.contains(x, y)
    }

    /// Whether this rectangle fully contains `other`. Empty rectangles are not
    /// considered contained.
    pub fn contains_rect(&self, other: Rect) -> bool {
        !self.is_empty()
            && !other.is_empty()
            && i64::from(self.x) <= i64::from(other.x)
            && i64::from(self.y) <= i64::from(other.y)
            && self.right() >= other.right()
            && self.bottom() >= other.bottom()
    }

    pub fn intersects(&self, other: Rect) -> bool {
        self.intersection(other).is_some()
    }

    /// Alias for [`Rect::intersects`].
    pub fn overlaps(&self, other: Rect) -> bool {
        self.intersects(other)
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

    /// Exclusive right edge as an `i64`, so it cannot overflow.
    pub fn right(self) -> i64 {
        i64::from(self.x) + i64::from(self.width)
    }

    /// Exclusive bottom edge as an `i64`, so it cannot overflow.
    pub fn bottom(self) -> i64 {
        i64::from(self.y) + i64::from(self.height)
    }

    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Area in cells, widened to `u64`.
    pub fn area(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// The center cell. Even dimensions are biased toward the top-left cell;
    /// a zero-length axis reports its origin on that axis.
    pub fn center(&self) -> (i32, i32) {
        (
            axis_center(self.x, self.width),
            axis_center(self.y, self.height),
        )
    }

    /// A normalized bounding rectangle containing both inputs. Empty inputs are
    /// ignored. The result's width and height saturate at `u16::MAX`, the most
    /// this representation can express.
    pub fn union(&self, other: Rect) -> Rect {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return *self;
        }
        Self::from_edges(
            i64::from(self.x).min(i64::from(other.x)),
            i64::from(self.y).min(i64::from(other.y)),
            self.right().max(other.right()),
            self.bottom().max(other.bottom()),
        )
    }

    /// Shrinks each side by the corresponding positive inset. Negative values
    /// grow that side. If opposite insets cross, an empty rectangle is returned.
    pub fn inset(&self, left: i32, top: i32, right: i32, bottom: i32) -> Rect {
        Self::from_edges(
            i64::from(self.x) + i64::from(left),
            i64::from(self.y) + i64::from(top),
            self.right() - i64::from(right),
            self.bottom() - i64::from(bottom),
        )
    }

    /// Grows by `dx` cells on both horizontal sides and `dy` on both vertical
    /// sides. Negative values shrink the corresponding sides.
    pub fn grow(&self, dx: i32, dy: i32) -> Rect {
        Self::from_edges(
            i64::from(self.x) - i64::from(dx),
            i64::from(self.y) - i64::from(dy),
            self.right() + i64::from(dx),
            self.bottom() + i64::from(dy),
        )
    }

    /// Builds a rectangle from half-open edges, clamping to representable
    /// coordinates and dimensions without panicking. Each axis collapses to a
    /// zero length independently when its far edge is not past its near edge.
    fn from_edges(left: i64, top: i64, right: i64, bottom: i64) -> Self {
        let left = left.clamp(i64::from(i32::MIN), i64::from(i32::MAX));
        let top = top.clamp(i64::from(i32::MIN), i64::from(i32::MAX));
        let right = right.clamp(i64::from(i32::MIN), i64::from(i32::MAX));
        let bottom = bottom.clamp(i64::from(i32::MIN), i64::from(i32::MAX));
        let width = if right <= left {
            0
        } else {
            (right - left).min(i64::from(u16::MAX)) as u16
        };
        let height = if bottom <= top {
            0
        } else {
            (bottom - top).min(i64::from(u16::MAX)) as u16
        };
        Self::new(left as i32, top as i32, width, height)
    }
}

fn axis_center(origin: i32, length: u16) -> i32 {
    if length == 0 {
        origin
    } else {
        (i64::from(origin) + (i64::from(length) - 1) / 2)
            .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
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
        Canvas {
            screen: self,
            clip,
            clip_stack: Vec::new(),
        }
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
    clip_stack: Vec<Rect>,
}

impl Canvas<'_> {
    /// The full screen bounds, independent of the current clip.
    pub fn bounds(&self) -> Rect {
        self.screen.bounds()
    }

    /// The composited cell at a screen coordinate, independent of the current
    /// clip. Returns a copy so callers do not borrow the canvas.
    pub fn get(&self, x: i32, y: i32) -> Option<Cell> {
        self.screen.get(x, y).copied()
    }

    /// The current clip rectangle.
    pub fn current_clip(&self) -> Rect {
        self.clip
    }

    /// Replaces the clip (rather than intersecting the previous clip).
    pub fn set_clip(&mut self, rect: Rect) {
        self.clip = self
            .bounds()
            .intersection(rect)
            .unwrap_or(Rect::new(0, 0, 0, 0));
    }

    /// Saves the current clip and intersects it with `rect`. The new clip can
    /// never extend beyond the clip that was active when pushed.
    pub fn push_clip(&mut self, rect: Rect) {
        self.clip_stack.push(self.clip);
        self.clip = self
            .clip
            .intersection(rect)
            .unwrap_or(Rect::new(0, 0, 0, 0));
    }

    /// Restores the most recently pushed clip. Returns false when the stack was
    /// empty and the clip was left unchanged.
    pub fn pop_clip(&mut self) -> bool {
        if let Some(previous) = self.clip_stack.pop() {
            self.clip = previous;
            true
        } else {
            false
        }
    }

    /// Restores the full-screen clip and discards every saved clip.
    pub fn reset_clip(&mut self) {
        self.clip = self.bounds();
        self.clip_stack.clear();
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

    /// Draws a cell's pixel at the explicitly supplied depth. The cell's own
    /// `z` is ignored; `z` follows the same last-write-wins rule as
    /// [`Canvas::draw_pixel`], and its pixel is sanitized the same way.
    pub fn draw_cell(&mut self, x: i32, y: i32, z: i32, cell: Cell) {
        self.draw_pixel(x, y, z, cell.pixel);
    }

    fn draw_pixel_i64(&mut self, x: i64, y: i64, z: i32, pixel: Pixel) {
        if self.clip.width == 0
            || self.clip.height == 0
            || x < i64::from(self.clip.x)
            || y < i64::from(self.clip.y)
            || x >= self.clip.right()
            || y >= self.clip.bottom()
        {
            return;
        }
        self.draw_pixel(x as i32, y as i32, z, pixel);
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

    /// Alias for [`Canvas::fill_rect`].
    pub fn fill(&mut self, rect: Rect, z: i32, pixel: Pixel) {
        self.fill_rect(rect, z, pixel);
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

    /// Draws a midpoint-rasterized circle outline. A radius of zero draws one
    /// center cell; negative radii draw nothing. Large radii use a clipped
    /// scanline fallback so work stays bounded by the visible clip.
    pub fn draw_circle(&mut self, cx: i32, cy: i32, radius: i32, z: i32, pixel: Pixel) {
        if radius < 0 || pixel.is_transparent() || self.clip.is_empty() {
            return;
        }
        let radius = i64::from(radius);
        if radius > MIDPOINT_CIRCLE_LIMIT {
            if !self.circle_may_intersect(cx, cy, radius) {
                return;
            }
            self.draw_circle_scanline(cx, cy, radius, z, pixel);
        } else {
            self.draw_circle_midpoint(cx, cy, radius, z, pixel);
        }
    }

    fn draw_circle_midpoint(&mut self, cx: i32, cy: i32, radius: i64, z: i32, pixel: Pixel) {
        let (cx, cy) = (i64::from(cx), i64::from(cy));
        let mut x = radius;
        let mut y = 0_i64;
        let mut error = 0_i64;
        while x >= y {
            for (px, py) in [
                (cx + x, cy + y),
                (cx + y, cy + x),
                (cx - y, cy + x),
                (cx - x, cy + y),
                (cx - x, cy - y),
                (cx - y, cy - x),
                (cx + y, cy - x),
                (cx + x, cy - y),
            ] {
                self.draw_pixel_i64(px, py, z, pixel);
            }
            if error <= 0 {
                y += 1;
                error += 2 * y + 1;
            }
            if error > 0 {
                x -= 1;
                error -= 2 * x + 1;
            }
        }
    }

    /// Fallback for enormous radii: only visible rows are considered, so work is
    /// proportional to the clip height rather than to the radius.
    fn draw_circle_scanline(&mut self, cx: i32, cy: i32, radius: i64, z: i32, pixel: Pixel) {
        let (cx_i, cy_i) = (i64::from(cx), i64::from(cy));
        let radius_squared = i128::from(radius) * i128::from(radius);
        for y in i64::from(self.clip.y)..self.clip.bottom() {
            let dy = y - cy_i;
            let remainder = radius_squared - i128::from(dy) * i128::from(dy);
            if remainder < 0 {
                continue;
            }
            let dx = isqrt_i128(remainder);
            self.draw_pixel_i64(cx_i - dx, y, z, pixel);
            if dx != 0 {
                self.draw_pixel_i64(cx_i + dx, y, z, pixel);
            }
        }
    }

    fn circle_may_intersect(&self, cx: i32, cy: i32, radius: i64) -> bool {
        if self.clip.is_empty() {
            return false;
        }
        let cx = i64::from(cx);
        let cy = i64::from(cy);
        let left = i64::from(self.clip.x);
        let right = self.clip.right() - 1;
        let top = i64::from(self.clip.y);
        let bottom = self.clip.bottom() - 1;
        let nearest_x = if cx < left {
            left - cx
        } else if cx > right {
            cx - right
        } else {
            0
        };
        let nearest_y = if cy < top {
            top - cy
        } else if cy > bottom {
            cy - bottom
        } else {
            0
        };
        let farthest_x = (cx - left).abs().max((cx - right).abs());
        let farthest_y = (cy - top).abs().max((cy - bottom).abs());
        let radius_squared = i128::from(radius) * i128::from(radius);
        let nearest = i128::from(nearest_x) * i128::from(nearest_x)
            + i128::from(nearest_y) * i128::from(nearest_y);
        let farthest = i128::from(farthest_x) * i128::from(farthest_x)
            + i128::from(farthest_y) * i128::from(farthest_y);
        // One cell of rasterization tolerance at each side of the ideal circle.
        let tolerance = i128::from(radius) * 2 + 1;
        radius_squared + tolerance >= nearest && radius_squared <= farthest + tolerance
    }

    /// Draws each consecutive pair of points as a line. A single point is drawn
    /// as one cell; an empty slice draws nothing.
    pub fn draw_polyline(&mut self, points: &[(i32, i32)], z: i32, pixel: Pixel) {
        match points {
            [] => {}
            [point] => self.draw_pixel(point.0, point.1, z, pixel),
            _ => {
                for pair in points.windows(2) {
                    self.draw_line(pair[0].0, pair[0].1, pair[1].0, pair[1].1, z, pixel);
                }
            }
        }
    }

    /// Draws an inclusive DDA vector. `pixel_along` receives normalized progress
    /// from zero at the start through one at the end. Work is clipped to the
    /// canvas so extreme endpoints cannot cause unbounded iteration.
    pub fn draw_vector(
        &mut self,
        x0: i32,
        y0: i32,
        x1: i32,
        y1: i32,
        z: i32,
        mut pixel_along: impl FnMut(f32) -> Pixel,
    ) {
        if self.clip.is_empty() {
            return;
        }
        let x0f = f64::from(x0);
        let y0f = f64::from(y0);
        let x1f = f64::from(x1);
        let y1f = f64::from(y1);
        let steps = (i64::from(x1) - i64::from(x0))
            .abs()
            .max((i64::from(y1) - i64::from(y0)).abs());
        if steps == 0 {
            if self.clip.contains(x0, y0) {
                self.draw_pixel_i64(i64::from(x0), i64::from(y0), z, pixel_along(0.0));
            }
            return;
        }
        let Some((enter, exit)) = line_clip_range(x0f, y0f, x1f, y1f, self.clip) else {
            return;
        };
        let steps_f = steps as f64;
        let padding = 2.0 / steps_f;
        let first = (((enter - padding).max(0.0)) * steps_f).floor() as i64;
        let last = (((exit + padding).min(1.0)) * steps_f).ceil() as i64;
        for step in first.clamp(0, steps)..=last.clamp(0, steps) {
            let progress = step as f64 / steps_f;
            let x = dda_round(x0f + (x1f - x0f) * progress);
            let y = dda_round(y0f + (y1f - y0f) * progress);
            if x < i64::from(self.clip.x)
                || y < i64::from(self.clip.y)
                || x >= self.clip.right()
                || y >= self.clip.bottom()
            {
                continue;
            }
            let pixel = pixel_along(progress as f32);
            self.draw_pixel_i64(x, y, z, pixel);
        }
    }

    /// Fills the contiguous region reachable from `(x, y)` inside the current
    /// clip. With `match_color`, only cells whose selected color equals that
    /// value are filled; otherwise the seed cell's color is used. `use_bg`
    /// selects the background channel, including "no background", instead of
    /// the foreground. Work is capped at [`MAX_FLOOD_FILL_CELLS`].
    pub fn flood_fill(
        &mut self,
        x: i32,
        y: i32,
        z: i32,
        pixel: Pixel,
        match_color: Option<Color>,
        use_bg: bool,
    ) {
        if pixel.is_transparent() || self.clip.is_empty() || !self.clip.contains(x, y) {
            return;
        }
        let Some(seed) = self.get(x, y) else {
            return;
        };
        let target = match_color.or(if use_bg {
            seed.pixel.bg
        } else {
            Some(seed.pixel.fg)
        });
        if selected_color(&seed, use_bg) != target {
            return;
        }
        let mut visited = HashSet::new();
        let mut stack = Vec::new();
        visited.insert((x, y));
        stack.push((x, y));
        let mut filled = 0;
        while let Some((cx, cy)) = stack.pop() {
            if filled >= MAX_FLOOD_FILL_CELLS {
                break;
            }
            let Some(cell) = self.get(cx, cy) else {
                continue;
            };
            if selected_color(&cell, use_bg) != target {
                continue;
            }
            self.draw_pixel(cx, cy, z, pixel);
            filled += 1;
            for (nx, ny) in [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)] {
                if visited.len() >= MAX_FLOOD_FILL_CELLS {
                    break;
                }
                if self.clip.contains(nx, ny) && visited.insert((nx, ny)) {
                    stack.push((nx, ny));
                }
            }
        }
    }

    /// Captures a screen rectangle as an image, independent of the current clip.
    /// Cells outside the screen become transparent. Zero-area rectangles return
    /// `None`.
    pub fn capture_rect(&self, rect: Rect) -> Option<Image> {
        if rect.is_empty() {
            return None;
        }
        let count = usize::from(rect.width).checked_mul(usize::from(rect.height))?;
        let mut pixels = Vec::with_capacity(count);
        for local_y in 0..usize::from(rect.height) {
            let y = i64::from(rect.y) + local_y as i64;
            for local_x in 0..usize::from(rect.width) {
                let x = i64::from(rect.x) + local_x as i64;
                let pixel = match (i32::try_from(x), i32::try_from(y)) {
                    (Ok(x), Ok(y)) => self
                        .screen
                        .get(x, y)
                        .map(|cell| cell.pixel)
                        .unwrap_or_else(Pixel::transparent),
                    _ => Pixel::transparent(),
                };
                pixels.push(pixel);
            }
        }
        // A well-formed pixel count should always be accepted. The blank-image
        // fallback keeps capture total even if a backend rejects a size.
        Some(
            Image::from_pixels(rect.width, rect.height, pixels)
                .unwrap_or_else(|_| Image::new(rect.width, rect.height, Pixel::transparent())),
        )
    }

    /// Draws an image with its top-left corner at `(x, y)`. Transparent image
    /// pixels are skipped; a missing background inherits the destination cell's
    /// background through normal compositing.
    pub fn draw_image(&mut self, x: i32, y: i32, z: i32, image: &Image) {
        let Some(visible) = Rect::new(x, y, image.width(), image.height()).intersection(self.clip)
        else {
            return;
        };
        let origin_x = i64::from(x);
        let origin_y = i64::from(y);
        for draw_y in i64::from(visible.y)..visible.bottom() {
            let local_y = (draw_y - origin_y) as i32;
            for draw_x in i64::from(visible.x)..visible.right() {
                let local_x = (draw_x - origin_x) as i32;
                if let Some(pixel) = image.get(local_x, local_y) {
                    if !pixel.is_transparent() {
                        self.draw_pixel_i64(draw_x, draw_y, z, pixel);
                    }
                }
            }
        }
    }

    /// Draws an ASCII-art polyline using `|`, `_`, `/`, and `\` according to
    /// each point's local direction. A one-point polyline is a `.`.
    pub fn art_line(&mut self, points: &[(i32, i32)], z: i32, pixel: Pixel) {
        if pixel.is_transparent() {
            return;
        }
        match points {
            [] => {}
            [point] => {
                self.draw_pixel(point.0, point.1, z, Pixel::new('.', pixel.fg, pixel.bg));
            }
            _ => {
                for pair in points.windows(2) {
                    self.draw_art_segment(pair[0], pair[1], z, pixel);
                }
            }
        }
    }

    fn draw_art_segment(&mut self, start: (i32, i32), end: (i32, i32), z: i32, pixel: Pixel) {
        if pixel.is_transparent() || self.clip.is_empty() {
            return;
        }
        let dx = (i64::from(end.0) - i64::from(start.0)).abs();
        let dy = (i64::from(end.1) - i64::from(start.1)).abs();
        if dx == 0 && dy == 0 {
            self.draw_pixel(start.0, start.1, z, Pixel::new('.', pixel.fg, pixel.bg));
            return;
        }
        let steep = dy > dx;
        let (a0, b0, a1, b1, low, high) = if steep {
            (
                start.1,
                start.0,
                end.1,
                end.0,
                i64::from(self.clip.y),
                self.clip.bottom() - 1,
            )
        } else {
            (
                start.0,
                start.1,
                end.0,
                end.1,
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
        let point_at = |step: i64| -> (i64, i64) {
            let advance = ((i128::from(step) * i128::from(minor) + i128::from(major / 2))
                / i128::from(major)) as i64;
            let a = a0 + da.signum() * step;
            let b = b0 + db.signum() * advance;
            if steep {
                (b, a)
            } else {
                (a, b)
            }
        };
        for step in first..=last {
            let current = point_at(step);
            let neighbor = if step > 0 {
                point_at(step - 1)
            } else {
                point_at(step + 1)
            };
            let glyph = art_direction_char(neighbor, current);
            self.draw_pixel_i64(
                current.0,
                current.1,
                z,
                Pixel::new(glyph, pixel.fg, pixel.bg),
            );
        }
    }
}

fn selected_color(cell: &Cell, use_bg: bool) -> Option<Color> {
    if use_bg {
        cell.pixel.bg
    } else {
        Some(cell.pixel.fg)
    }
}

fn dda_round(value: f64) -> i64 {
    (value + 0.5).floor() as i64
}

fn isqrt_i128(value: i128) -> i64 {
    debug_assert!(value >= 0);
    let mut root = (value as f64).sqrt() as i128;
    while (root + 1) * (root + 1) <= value {
        root += 1;
    }
    while root * root > value {
        root -= 1;
    }
    root as i64
}

fn art_direction_char(from: (i64, i64), to: (i64, i64)) -> char {
    if from.0 == to.0 {
        '|'
    } else if from.1 == to.1 {
        '_'
    } else if from.0 > to.0 {
        if from.1 < to.1 {
            '/'
        } else {
            '\\'
        }
    } else if from.1 < to.1 {
        '\\'
    } else {
        '/'
    }
}

/// Liang-Barsky clipping against a one-cell-padded clip rectangle. The padding
/// keeps endpoints on clip edges (whose neighbors are just outside) visible.
fn line_clip_range(x0: f64, y0: f64, x1: f64, y1: f64, clip: Rect) -> Option<(f64, f64)> {
    if clip.is_empty() {
        return None;
    }
    let left = i64::from(clip.x) as f64 - 1.0;
    let right = clip.right() as f64 + 1.0;
    let top = i64::from(clip.y) as f64 - 1.0;
    let bottom = clip.bottom() as f64 + 1.0;
    let dx = x1 - x0;
    let dy = y1 - y0;
    let mut enter = 0.0_f64;
    let mut exit = 1.0_f64;
    for (p, q) in [
        (-dx, x0 - left),
        (dx, right - x0),
        (-dy, y0 - top),
        (dy, bottom - y0),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let ratio = q / p;
            if p < 0.0 {
                if ratio > exit {
                    return None;
                }
                enter = enter.max(ratio);
            } else {
                if ratio < enter {
                    return None;
                }
                exit = exit.min(ratio);
            }
        }
    }
    (enter <= exit).then_some((enter, exit))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RED: Color = Color::rgb(255, 0, 0);
    const BLUE: Color = Color::rgb(0, 0, 255);

    fn px(ch: char) -> Pixel {
        Pixel::new(ch, Color::WHITE, None)
    }

    fn screen_of(width: u16, height: u16) -> Screen {
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
        let mut screen = screen_of(1, 1);
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
        let mut screen = screen_of(5, 3);
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
        let mut screen = screen_of(7, 3);
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
        let mut screen = screen_of(9, 6);
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
        let mut screen = screen_of(5, 4);
        let mut canvas = screen.canvas();
        canvas.draw_rect(Rect::new(-1, -1, 7, 6), 0, px('!'));
        canvas.set_clip(Rect::new(1, 1, 3, 2));
        canvas.draw_rect(Rect::new(0, 0, 4, 3), 1, px('#'));
        assert_eq!(screen.snapshot(), ".....\n...#.\n.###.\n.....");
    }

    #[test]
    fn extreme_rectangles_are_clipped_before_iteration() {
        let mut screen = screen_of(4, 3);
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
        let mut screen = screen_of(9, 5);
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
                            let mut actual = screen_of(5, 5);
                            let mut expected = screen_of(5, 5);
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
        let mut screen = screen_of(5, 5);
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
        let mut screen = screen_of(3, 3);
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
            let mut screen = screen_of(width, height);
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

    #[test]
    fn color_constants_and_hex_parsing() {
        assert_eq!(Color::GRAY1, Color::rgb(0x11, 0x11, 0x11));
        assert_eq!(Color::GRAY2, Color::rgb(0x22, 0x22, 0x22));
        assert_eq!(Color::GRAY3, Color::rgb(0x33, 0x33, 0x33));
        assert_eq!(Color::GRAY4, Color::rgb(0x44, 0x44, 0x44));
        assert_eq!(Color::DFLT_BG, Color::GRAY1);
        assert_eq!(Color::gray1, Color::GRAY1);
        assert_eq!(Color::dflt_bg, Color::DFLT_BG);

        let color = Color::rgb(0x12, 0xab, 0xff);
        assert_eq!(color.hex(), "#12abff");
        assert_eq!(color.css_hex(), "#12abff");
        assert_eq!(color.to_string(), "#12abff");
        for valid in [
            "#12abff",
            "0x12ABFF",
            "12abff",
            "  #12ABFF  ",
            "#abc",
            "0xABC",
            "abc",
        ] {
            let parsed = Color::from_hex(valid).unwrap();
            if valid.ends_with("abc") || valid.ends_with("ABC") {
                assert_eq!(parsed, Color::rgb(0xaa, 0xbb, 0xcc), "{valid}");
            } else {
                assert_eq!(parsed, color, "{valid}");
            }
        }
        assert_eq!("#ff0000".parse::<Color>().unwrap(), RED);
        assert_eq!(
            Color::from_hex(""),
            Err(ColorParseError::Empty),
            "empty string"
        );
        for invalid in ["#", "0x"] {
            assert_eq!(
                Color::from_hex(invalid),
                Err(ColorParseError::Empty),
                "{invalid}"
            );
        }
        for invalid in ["#12", "1234567", "#12345", "#123456789"] {
            assert_eq!(
                Color::from_hex(invalid),
                Err(ColorParseError::InvalidLength),
                "{invalid}"
            );
        }
        assert_eq!(
            Color::from_hex("#gg0000"),
            Err(ColorParseError::InvalidDigit('g'))
        );
        assert_eq!(
            Color::from_hex("\u{3b1}\u{3b2}\u{3b3}"),
            Err(ColorParseError::InvalidDigit('\u{3b1}'))
        );
        assert!(!ColorParseError::Empty.to_string().is_empty());
        assert!(!ColorParseError::InvalidLength.to_string().is_empty());
        assert!(!ColorParseError::InvalidDigit('x').to_string().is_empty());
        fn assert_error<T: std::error::Error>() {}
        assert_error::<ColorParseError>();
    }

    #[test]
    fn luma_channels_inversion_and_brightness_helpers() {
        assert_eq!(Color::BLACK.luma(), 0);
        assert_eq!(Color::WHITE.luma(), 255);
        assert_eq!(Color::rgb(255, 0, 0).luma(), 54);
        assert_eq!(Color::rgb(0, 255, 0).luma(), 182);
        assert_eq!(Color::rgb(0, 0, 255).luma(), 18);

        let color = Color::rgb(10, 20, 30);
        assert_eq!(color.with_r(200), Color::rgb(200, 20, 30));
        assert_eq!(color.with_g(200), Color::rgb(10, 200, 30));
        assert_eq!(color.with_b(200), Color::rgb(10, 20, 200));
        assert_eq!(color.invert(), Color::rgb(245, 235, 225));
        assert_eq!(color.invert().invert(), color);

        assert_eq!(RED.darker(0.0), RED);
        assert_eq!(RED.darker(1.0), Color::BLACK);
        assert_eq!(RED.darker(2.0), Color::BLACK);
        assert_eq!(RED.darker(-1.0), RED);
        assert_eq!(RED.darker(f32::NAN), RED);
        assert_eq!(RED.darker(f32::NEG_INFINITY), RED);
        assert_eq!(RED.darker(f32::INFINITY), Color::BLACK);
        assert_eq!(RED.lighter(0.0), RED);
        assert_eq!(RED.lighter(1.0), Color::WHITE);
        assert_eq!(RED.lighter(2.0), Color::WHITE);
        assert_eq!(RED.lighter(-1.0), RED);
        assert_eq!(RED.lighter(f32::NAN), RED);
        assert_eq!(RED.lighter(f32::INFINITY), Color::WHITE);
        assert_eq!(
            Color::rgb(10, 20, 30).lighter(0.5),
            Color::rgb(133, 138, 143)
        );
        assert_eq!(Color::rgb(10, 20, 30).darker(0.5), Color::rgb(5, 10, 15));

        assert_eq!(RED.mixture(BLUE, 0.0), RED);
        assert_eq!(RED.mixture(BLUE, 1.0), BLUE);
        assert_eq!(RED.mixture(BLUE, 0.5), Color::rgb(128, 0, 128));
        assert_eq!(RED.mixture(BLUE, f32::NAN), RED);
        assert_eq!(RED.mixture(BLUE, 99.0), BLUE);
    }

    #[test]
    fn hsb_roundtrips_and_clamps() {
        for color in [
            Color::BLACK,
            Color::WHITE,
            RED,
            Color::rgb(0, 255, 0),
            Color::rgb(0, 0, 255),
            Color::rgb(128, 128, 128),
            Color::rgb(10, 20, 30),
            Color::rgb(255, 128, 0),
            Color::rgb(255, 0, 255),
            Color::rgb(0, 255, 255),
            Color::rgb(255, 255, 0),
            Color::rgb(1, 2, 3),
        ] {
            let (h, s, b) = color.rgb_to_hsb();
            assert!((0.0..=1.0).contains(&h), "{color:?}: {h}");
            assert!((0.0..=1.0).contains(&s), "{color:?}: {s}");
            assert!((0.0..=1.0).contains(&b), "{color:?}: {b}");
            assert_eq!(Color::from_hsb(h, s, b), color, "{color:?}");
        }
        assert_eq!(RED.rgb_to_hsb(), (0.0, 1.0, 1.0));
        let (green_hue, _, _) = Color::rgb(0, 255, 0).rgb_to_hsb();
        assert!((green_hue - 1.0 / 3.0).abs() < 1e-5);
        let (blue_hue, _, _) = Color::rgb(0, 0, 255).rgb_to_hsb();
        assert!((blue_hue - 2.0 / 3.0).abs() < 1e-5);
        assert_eq!(Color::rgb(0, 0, 0).rgb_to_hsb(), (0.0, 0.0, 0.0));
        assert_eq!(Color::WHITE.rgb_to_hsb(), (0.0, 0.0, 1.0));

        assert_eq!(Color::from_hsb(0.0, 1.0, 1.0), RED);
        assert_eq!(Color::from_hsb(1.0, 1.0, 1.0), RED);
        assert_eq!(Color::from_hsb(-1.0 / 3.0, 1.0, 1.0), Color::rgb(0, 0, 255));
        assert_eq!(Color::from_hsb(f32::NAN, 1.0, 1.0), RED);
        assert_eq!(Color::from_hsb(f32::INFINITY, 1.0, 1.0), RED);
        assert_eq!(Color::from_hsb(0.0, 9.0, 1.0), RED);
        assert_eq!(Color::from_hsb(0.0, 1.0, 9.0), RED);
        assert_eq!(Color::from_hsb(0.0, -1.0, 1.0), Color::WHITE);
        assert_eq!(Color::from_hsb(0.0, 0.0, 0.5), Color::rgb(128, 128, 128));
    }

    #[test]
    fn gradients_exclude_start_and_include_end() {
        assert!(Color::gradient(RED, BLUE, 0).is_empty());
        assert_eq!(Color::gradient(RED, BLUE, 1), vec![BLUE]);
        let gradient = Color::gradient(Color::BLACK, Color::WHITE, 4);
        assert_eq!(
            gradient,
            vec![
                Color::rgb(64, 64, 64),
                Color::rgb(128, 128, 128),
                Color::rgb(191, 191, 191),
                Color::WHITE,
            ]
        );
        assert_eq!(gradient[0], Color::rgb(64, 64, 64));
        assert_eq!(*gradient.last().unwrap(), Color::WHITE);
        let reverse = Color::gradient(Color::WHITE, Color::BLACK, 2);
        assert_eq!(reverse, vec![Color::rgb(128, 128, 128), Color::BLACK]);
    }

    #[test]
    fn xterm_conversion_matches_cosplay() {
        assert_eq!(Color::BLACK.nearest_xterm(), 16);
        assert_eq!(Color::WHITE.nearest_xterm(), 231);
        assert_eq!(Color::rgb(0, 0, 95).nearest_xterm(), 17);
        assert_eq!(Color::rgb(0, 0, 135).nearest_xterm(), 18);
        assert_eq!(Color::rgb(0, 0, 175).nearest_xterm(), 19);
        assert_eq!(Color::rgb(0, 0, 215).nearest_xterm(), 20);
        assert_eq!(Color::rgb(0, 0, 255).nearest_xterm(), 21);
        assert_eq!(Color::rgb(95, 0, 0).nearest_xterm(), 52);
        assert_eq!(Color::rgb(255, 0, 0).nearest_xterm(), 196);
        assert_eq!(Color::rgb(0, 255, 0).nearest_xterm(), 46);
        assert_eq!(Color::rgb(0, 0, 255).nearest_xterm(), 21);
        assert_eq!(Color::rgb(128, 128, 128).nearest_xterm(), 244);
        assert_eq!(Color::rgb(255, 255, 255).nearest_xterm(), 231);
        for gray in (0..=255).step_by(7) {
            let index = Color::rgb(gray, gray, gray).nearest_xterm();
            assert!((16..=255).contains(&index));
        }
    }

    #[test]
    fn rectangle_edges_area_and_center() {
        let rect = Rect::new(-2, 3, 4, 2);
        assert_eq!(rect.right(), 2);
        assert_eq!(rect.bottom(), 5);
        assert_eq!(rect.area(), 8);
        assert_eq!(rect.center(), (-1, 3));
        assert!(!rect.is_empty());

        let wide = Rect::new(i32::MAX, i32::MIN, u16::MAX, u16::MAX);
        assert_eq!(wide.right(), i64::from(i32::MAX) + i64::from(u16::MAX));
        assert_eq!(wide.bottom(), i64::from(i32::MIN) + i64::from(u16::MAX));
        assert_eq!(wide.area(), u64::from(u16::MAX) * u64::from(u16::MAX));
        assert!((i32::MIN..=i32::MAX).contains(&wide.center().0));

        let empty = Rect::new(7, -3, 0, 9);
        assert!(empty.is_empty());
        assert_eq!(empty.area(), 0);
        assert_eq!(empty.center(), (7, 1));
        assert_eq!(Rect::new(7, -3, 9, 0).center(), (11, -3));
        assert!(!empty.contains(7, -3));
        assert!(!empty.contains_rect(empty));
    }

    #[test]
    fn rectangle_union_contains_helpers_and_extremes() {
        let a = Rect::new(5, 6, 3, 2);
        let b = Rect::new(0, 2, 4, 8);
        assert_eq!(a.union(b), Rect::new(0, 2, 8, 8));
        assert_eq!(a.union(b), b.union(a));
        assert_eq!(a.union(Rect::new(0, 0, 0, 0)), a);
        assert_eq!(Rect::new(0, 0, 0, 0).union(a), a);
        assert_eq!(
            Rect::new(0, 0, 1, 1).union(Rect::new(0, 0, 0, 1)),
            Rect::new(0, 0, 1, 1)
        );

        let outer = Rect::new(0, 0, 10, 8);
        assert!(outer.contains_point(0, 0));
        assert!(outer.contains_point(9, 7));
        assert!(!outer.contains_point(10, 7));
        assert!(outer.contains_rect(Rect::new(1, 1, 8, 6)));
        assert!(outer.contains_rect(outer));
        assert!(!outer.contains_rect(Rect::new(9, 7, 2, 1)));
        assert!(!outer.contains_rect(Rect::new(1, 1, 0, 0)));
        assert!(!Rect::new(0, 0, 0, 0).contains_rect(Rect::new(0, 0, 0, 0)));
        assert!(outer.overlaps(Rect::new(9, 7, 5, 5)));
        assert!(!outer.overlaps(Rect::new(10, 0, 1, 1)));
        assert_eq!(
            outer.overlaps(Rect::new(9, 7, 5, 5)),
            outer.intersects(Rect::new(9, 7, 5, 5))
        );

        let extreme = Rect::new(i32::MIN, i32::MIN, u16::MAX, u16::MAX).union(Rect::new(
            i32::MAX,
            i32::MAX,
            u16::MAX,
            u16::MAX,
        ));
        assert_eq!(extreme.x, i32::MIN);
        assert_eq!(extreme.y, i32::MIN);
        assert_eq!(extreme.width, u16::MAX);
        assert_eq!(extreme.height, u16::MAX);
        assert!(Rect::new(i32::MAX, i32::MAX, u16::MAX, u16::MAX).area() > 0);
    }

    #[test]
    fn rectangle_inset_and_grow_are_safe() {
        let rect = Rect::new(10, 10, 10, 6);
        assert_eq!(
            rect.inset(2, 1, 3, 2),
            Rect::new(12, 11, 5, 3),
            "positive insets shrink each named side"
        );
        assert_eq!(
            rect.inset(-1, -2, -3, -4),
            Rect::new(9, 8, 14, 12),
            "negative insets grow each named side"
        );
        assert_eq!(rect.inset(100, 0, 0, 0).width, 0);
        assert_eq!(rect.inset(0, 100, 0, 0).height, 0);
        assert_eq!(rect.inset(100, 100, 100, 100), Rect::new(110, 110, 0, 0));

        assert_eq!(Rect::new(5, 5, 4, 2).grow(2, 1), Rect::new(3, 4, 8, 4));
        assert_eq!(Rect::new(5, 5, 4, 2).grow(-2, -1), Rect::new(7, 6, 0, 0));
        assert_eq!(Rect::new(0, 0, 1, 1).grow(-1, 0), Rect::new(1, 0, 0, 1));

        let huge = Rect::new(i32::MAX, i32::MIN, u16::MAX, u16::MAX);
        let _ = huge.inset(i32::MIN, i32::MAX, i32::MIN, i32::MAX);
        let _ = huge.grow(i32::MAX, i32::MIN);
    }

    #[test]
    fn canvas_get_ignores_clip_and_clip_stack_restores() {
        let mut screen = screen_of(5, 4);
        let mut canvas = screen.canvas();
        canvas.draw_pixel(0, 0, 3, px('A'));
        canvas.draw_pixel(4, 3, 4, px('Z'));
        assert_eq!(
            canvas.get(0, 0),
            Some(Cell {
                pixel: px('A'),
                z: 3
            })
        );
        assert_eq!(canvas.get(4, 3).unwrap().pixel.ch, 'Z');
        assert_eq!(canvas.get(5, 0), None);

        canvas.set_clip(Rect::new(1, 1, 2, 2));
        assert_eq!(canvas.current_clip(), Rect::new(1, 1, 2, 2));
        assert_eq!(canvas.get(0, 0).unwrap().pixel.ch, 'A');

        canvas.push_clip(Rect::new(2, 0, 3, 3));
        assert_eq!(canvas.current_clip(), Rect::new(2, 1, 1, 2));
        canvas.push_clip(Rect::new(0, 0, 1, 1));
        assert!(canvas.current_clip().is_empty());
        canvas.fill_rect(canvas.bounds(), 9, px('!'));
        assert!(canvas.pop_clip());
        assert_eq!(canvas.current_clip(), Rect::new(2, 1, 1, 2));
        assert!(canvas.pop_clip());
        assert_eq!(canvas.current_clip(), Rect::new(1, 1, 2, 2));
        assert!(!canvas.pop_clip());
        assert_eq!(canvas.current_clip(), Rect::new(1, 1, 2, 2));

        canvas.push_clip(Rect::new(0, 0, 0, 0));
        assert!(canvas.current_clip().is_empty());
        canvas.reset_clip();
        assert_eq!(canvas.current_clip(), canvas.bounds());
        assert!(!canvas.pop_clip(), "reset discards saved clips");
        assert_eq!(screen.snapshot(), "A....\n.....\n.....\n....Z");
    }

    #[test]
    fn draw_cell_uses_argument_z_and_sanitizes() {
        let mut screen = Screen::new(2, 1, Pixel::new('.', Color::WHITE, Some(BLUE)));
        {
            let mut canvas = screen.canvas();
            let invalid = Cell {
                pixel: Pixel {
                    ch: '\n',
                    fg: RED,
                    bg: None,
                },
                z: i32::MAX,
            };
            canvas.draw_cell(0, 0, 5, invalid);
            assert_eq!(
                canvas.get(0, 0),
                Some(Cell {
                    pixel: Pixel::new('?', RED, Some(BLUE)),
                    z: 5,
                })
            );
            canvas.draw_cell(
                0,
                0,
                4,
                Cell {
                    pixel: px('x'),
                    z: i32::MIN,
                },
            );
            assert_eq!(canvas.get(0, 0).unwrap().pixel.ch, '?');
            canvas.draw_cell(
                0,
                0,
                5,
                Cell {
                    pixel: px('y'),
                    z: i32::MIN,
                },
            );
            assert_eq!(canvas.get(0, 0).unwrap().pixel.ch, 'y');
            canvas.draw_cell(
                0,
                0,
                i32::MAX,
                Cell {
                    pixel: Pixel::transparent(),
                    z: 0,
                },
            );
            assert_eq!(canvas.get(0, 0).unwrap().pixel.ch, 'y');
            canvas.draw_cell(
                2,
                0,
                0,
                Cell {
                    pixel: px('!'),
                    z: 0,
                },
            );
        }
        assert_eq!(screen.snapshot(), "y.");
    }

    #[test]
    fn fill_is_an_alias_for_fill_rect() {
        let mut screen = screen_of(5, 3);
        let mut canvas = screen.canvas();
        canvas.fill(Rect::new(1, 1, 3, 2), 2, px('#'));
        canvas.fill(Rect::new(0, 0, 5, 1), 3, px('-'));
        canvas.fill(Rect::new(0, 0, 5, 3), 1, px('x'));
        assert_eq!(screen.snapshot(), "-----\nx###x\nx###x");
        assert_eq!(screen.get(1, 1).unwrap().z, 2);
    }

    #[test]
    fn circle_draws_symmetric_outline_and_handles_limits() {
        let mut screen = screen_of(9, 9);
        {
            let mut canvas = screen.canvas();
            canvas.draw_circle(4, 4, 3, 0, px('*'));
        }
        for (x, y) in [(4, 1), (4, 7), (1, 4), (7, 4)] {
            assert_eq!(screen.get(x, y).unwrap().pixel.ch, '*', "{x},{y}");
        }
        assert_eq!(screen.get(4, 4).unwrap().pixel.ch, '.');
        for (x, y) in [
            (2, 3),
            (3, 2),
            (5, 2),
            (6, 3),
            (6, 5),
            (5, 6),
            (3, 6),
            (2, 5),
        ] {
            assert_eq!(screen.get(x, y).unwrap().pixel.ch, '*', "{x},{y}");
        }
        for (x, y) in [(2, 2), (6, 2), (2, 6), (6, 6)] {
            assert_eq!(screen.get(x, y).unwrap().pixel.ch, '.', "{x},{y}");
        }
        let points = screen
            .cells
            .iter()
            .filter(|cell| cell.pixel.ch == '*')
            .count();
        assert_eq!(points, 12, "radius 3 midpoint pixels");

        let mut dots = screen_of(5, 5);
        dots.canvas().draw_circle(2, 2, 0, 0, px('o'));
        assert_eq!(dots.snapshot(), ".....\n.....\n..o..\n.....\n.....");
        dots.canvas().draw_circle(2, 2, -1, 0, px('x'));
        assert_eq!(dots.snapshot(), ".....\n.....\n..o..\n.....\n.....");

        let mut clipped = screen_of(7, 7);
        {
            let mut canvas = clipped.canvas();
            canvas.set_clip(Rect::new(2, 2, 3, 3));
            canvas.draw_circle(3, 3, 2, 0, px('#'));
        }
        assert_eq!(
            clipped.snapshot(),
            ".......\n.......\n..#.#..\n.......\n..#.#..\n.......\n......."
        );
    }

    #[test]
    fn enormous_circles_are_bounded() {
        let mut screen = screen_of(10, 6);
        {
            let mut canvas = screen.canvas();
            canvas.draw_circle(5, 3, i32::MAX, 0, px('!'));
        }
        assert!(screen.cells.iter().all(|cell| cell.pixel.ch == '.'));
        {
            let mut canvas = screen.canvas();
            canvas.draw_circle(-100_000, 3, 100_001, 1, px('!'));
        }
        assert!(screen.cells.iter().any(|cell| cell.pixel.ch == '!'));
        {
            let mut canvas = screen.canvas();
            canvas.draw_circle(i32::MAX, i32::MIN, i32::MAX, 2, px('?'));
        }
        let _ = screen.snapshot();
    }

    #[test]
    fn polyline_draws_each_segment_and_endpoints() {
        let mut screen = screen_of(7, 5);
        {
            let mut canvas = screen.canvas();
            canvas.draw_polyline(&[(1, 1), (3, 1), (3, 3)], 0, px('#'));
            canvas.draw_polyline(&[(5, 4)], 0, px('o'));
            canvas.draw_polyline(&[], 0, px('!'));
        }
        assert_eq!(
            screen.snapshot(),
            ".......\n.###...\n...#...\n...#...\n.....o."
        );
        screen.canvas().draw_polyline(&[(0, 0), (6, 4)], 1, px('+'));
        assert_eq!(screen.get(0, 0).unwrap().pixel.ch, '+');
        assert_eq!(screen.get(6, 4).unwrap().pixel.ch, '+');
    }

    #[test]
    fn vector_is_inclusive_dda_and_reports_progress() {
        let mut screen = screen_of(6, 4);
        let mut progress = Vec::new();
        {
            let mut canvas = screen.canvas();
            canvas.draw_vector(0, 0, 4, 0, 0, |t| {
                progress.push(t);
                px(if t < 0.5 { 'a' } else { 'b' })
            });
            canvas.draw_vector(0, 1, 3, 3, 1, |_| px('#'));
            canvas.draw_vector(5, 3, 5, 3, 2, |_| px('o'));
        }
        assert_eq!(progress, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(screen.snapshot(), "aabbb.\n#.....\n.##...\n...#.o");

        let mut out = Vec::new();
        let mut other = screen_of(3, 3);
        other
            .canvas()
            .draw_vector(i32::MIN, i32::MIN, i32::MAX, i32::MAX, 0, |t| {
                out.push(t);
                px('x')
            });
        assert!(!out.is_empty());
        assert_eq!(other.snapshot(), "x..\n.x.\n..x");

        let mut called = false;
        other.canvas().draw_vector(-1, -1, -1, -1, 0, |_| {
            called = true;
            px('!')
        });
        assert!(
            !called,
            "an offscreen point does not invoke the pixel producer"
        );
    }

    #[test]
    fn vector_matches_naive_dda_when_clipped() {
        fn reference(screen: &mut Screen, clip: Rect, start: (i32, i32), end: (i32, i32)) {
            let dx = end.0 - start.0;
            let dy = end.1 - start.1;
            let steps = dx.abs().max(dy.abs());
            let mut canvas = screen.canvas();
            canvas.set_clip(clip);
            if steps == 0 {
                canvas.draw_pixel(start.0, start.1, 0, px('#'));
                return;
            }
            for step in 0..=steps {
                let t = step as f64 / f64::from(steps);
                let x = (f64::from(start.0) + f64::from(dx) * t + 0.5).floor() as i32;
                let y = (f64::from(start.1) + f64::from(dy) * t + 0.5).floor() as i32;
                canvas.draw_pixel(x, y, 0, px('#'));
            }
        }

        for clip in [Rect::new(0, 0, 5, 5), Rect::new(1, 2, 3, 2)] {
            for x0 in -2..=6 {
                for y0 in -2..=6 {
                    for x1 in -2..=6 {
                        for y1 in -2..=6 {
                            let mut actual = screen_of(5, 5);
                            let mut expected = screen_of(5, 5);
                            let mut canvas = actual.canvas();
                            canvas.set_clip(clip);
                            canvas.draw_vector(x0, y0, x1, y1, 0, |_| px('#'));
                            reference(&mut expected, clip, (x0, y0), (x1, y1));
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
    fn flood_fill_respects_foreground_boundaries_and_clip() {
        let mut screen = screen_of(7, 5);
        {
            let mut canvas = screen.canvas();
            canvas.draw_rect(Rect::new(0, 0, 7, 5), 0, Pixel::new('#', RED, None));
            canvas.flood_fill(3, 2, 1, px('*'), None, false);
        }
        assert_eq!(
            screen.snapshot(),
            "#######\n#*****#\n#*****#\n#*****#\n#######"
        );
        assert!(screen.get(3, 2).unwrap().pixel.fg.eq(&Color::WHITE));

        let mut clipped = screen_of(7, 1);
        {
            let mut canvas = clipped.canvas();
            canvas.set_clip(Rect::new(2, 0, 3, 1));
            canvas.flood_fill(3, 0, 1, px('*'), None, false);
        }
        assert_eq!(clipped.snapshot(), "..***..");

        let mut blocked = screen_of(5, 3);
        {
            let mut canvas = blocked.canvas();
            canvas.draw_line(2, 0, 2, 2, 1, Pixel::new('|', RED, None));
            canvas.flood_fill(0, 1, 0, px('L'), None, false);
            canvas.flood_fill(4, 1, 0, px('R'), None, false);
        }
        assert_eq!(blocked.snapshot(), "LL|RR\nLL|RR\nLL|RR");

        let mut untouched = screen_of(3, 3);
        untouched.canvas().flood_fill(1, 1, 0, px('!'), None, true);
        assert_eq!(
            untouched.snapshot(),
            "!!!\n!!!\n!!!",
            "an absent background is a matchable color"
        );
    }

    #[test]
    fn flood_fill_selection_background_and_explicit_color() {
        let mut screen = Screen::new(5, 1, Pixel::new(' ', Color::WHITE, Some(BLUE)));
        {
            let mut canvas = screen.canvas();
            canvas.draw_pixel(2, 0, 5, Pixel::new('o', RED, Some(RED)));
            canvas.flood_fill(
                0,
                0,
                1,
                Pixel::new('*', Color::WHITE, Some(Color::rgb(0, 255, 0))),
                None,
                true,
            );
        }
        assert_eq!(screen.snapshot(), "**o  ");
        assert_eq!(
            screen.get(0, 0).unwrap().pixel.bg,
            Some(Color::rgb(0, 255, 0))
        );
        assert_eq!(screen.get(2, 0).unwrap().pixel.bg, Some(RED));
        assert_eq!(screen.get(3, 0).unwrap().pixel.bg, Some(BLUE));

        let mut explicit = screen_of(5, 1);
        let mut canvas = explicit.canvas();
        canvas.draw_pixel(0, 0, 0, Pixel::new('a', RED, None));
        canvas.draw_pixel(1, 0, 0, Pixel::new('b', BLUE, None));
        canvas.draw_pixel(2, 0, 0, Pixel::new('c', RED, None));
        canvas.draw_pixel(3, 0, 0, Pixel::new('d', RED, None));
        canvas.flood_fill(2, 0, 1, px('x'), Some(RED), false);
        assert_eq!(explicit.snapshot(), "abxx.");

        let mut mismatch = screen_of(3, 1);
        mismatch
            .canvas()
            .flood_fill(0, 0, 1, px('!'), Some(RED), false);
        assert_eq!(mismatch.snapshot(), "...");

        let mut outside = screen_of(3, 1);
        outside.canvas().flood_fill(-1, 0, 1, px('!'), None, false);
        outside.canvas().flood_fill(3, 0, 1, px('!'), None, false);
        assert_eq!(outside.snapshot(), "...");
    }

    #[test]
    fn flood_fill_stops_at_the_cell_limit() {
        let mut screen = screen_of(520, 520);
        screen
            .canvas()
            .flood_fill(260, 260, 1, px('*'), None, false);
        let filled = screen
            .cells
            .iter()
            .filter(|cell| cell.pixel.ch == '*')
            .count();
        assert_eq!(filled, MAX_FLOOD_FILL_CELLS);
        assert_eq!(screen.get(0, 0).unwrap().pixel.ch, '.');
    }

    fn make_image(width: u16, height: u16, pixels: Vec<Pixel>) -> Image {
        match Image::from_pixels(width, height, pixels) {
            Ok(image) => image,
            Err(_) => panic!("valid image construction failed"),
        }
    }

    #[test]
    fn capture_rect_reads_screen_independent_of_clip() {
        let mut screen = screen_of(4, 2);
        {
            let mut canvas = screen.canvas();
            canvas.draw_pixel(0, 0, 0, px('A'));
            canvas.draw_pixel(3, 1, 0, px('D'));
            canvas.set_clip(Rect::new(3, 1, 1, 1));

            let image = canvas.capture_rect(Rect::new(0, 0, 4, 2)).unwrap();
            assert_eq!((image.width(), image.height()), (4, 2));
            assert_eq!(image.get(0, 0).unwrap().ch, 'A');
            assert_eq!(image.get(3, 1).unwrap().ch, 'D');
            assert_eq!(image.get(1, 0).unwrap().ch, '.');

            let partial = canvas.capture_rect(Rect::new(3, 1, 3, 2)).unwrap();
            assert_eq!((partial.width(), partial.height()), (3, 2));
            assert_eq!(partial.get(0, 0).unwrap().ch, 'D');
            assert!(partial.get(1, 0).unwrap().is_transparent());
            assert!(partial.get(2, 1).unwrap().is_transparent());

            assert!(canvas.capture_rect(Rect::new(0, 0, 0, 4)).is_none());
            assert!(canvas.capture_rect(Rect::new(0, 0, 4, 0)).is_none());
        }
    }

    #[test]
    fn draw_image_skips_transparency_clips_and_inherits_background() {
        let image = make_image(
            3,
            2,
            vec![
                Pixel::transparent(),
                px('a'),
                Pixel::new('b', RED, None),
                Pixel::new('c', BLUE, None),
                Pixel::transparent(),
                px('d'),
            ],
        );
        let mut screen = Screen::new(5, 4, Pixel::new('.', Color::WHITE, Some(BLUE)));
        {
            let mut canvas = screen.canvas();
            canvas.draw_image(1, 1, 7, &image);
            canvas.draw_image(-2, -2, 0, &image);
            canvas.draw_image(100, 100, 0, &image);
        }
        assert_eq!(screen.snapshot(), ".....\n..ab.\n.c.d.\n.....");
        assert_eq!(screen.get(2, 1).unwrap().pixel.bg, Some(BLUE));
        assert_eq!(screen.get(3, 2).unwrap().pixel.bg, Some(BLUE));
        assert_eq!(screen.get(2, 1).unwrap().z, 7);
        assert_eq!(screen.get(2, 2).unwrap().z, i32::MIN);

        let mut clipped = Screen::new(4, 3, Pixel::new('.', Color::WHITE, Some(BLUE)));
        {
            let mut canvas = clipped.canvas();
            canvas.set_clip(Rect::new(0, 0, 2, 2));
            canvas.draw_image(0, 0, 1, &image);
        }
        assert_eq!(clipped.snapshot(), ".a..\nc...\n....");

        let mut offset = Screen::new(3, 2, Pixel::new('.', Color::WHITE, Some(BLUE)));
        offset.canvas().draw_image(-1, -1, 3, &image);
        assert_eq!(
            offset.snapshot(),
            ".d.\n...",
            "local coordinates stay relative to the image origin"
        );
    }

    #[test]
    fn art_line_uses_directional_glyphs_and_handles_edges() {
        let mut screen = screen_of(7, 5);
        {
            let mut canvas = screen.canvas();
            canvas.art_line(&[(0, 0), (6, 0)], 0, px('#'));
            canvas.art_line(&[(0, 1), (0, 4)], 0, px('#'));
            canvas.art_line(&[(1, 4), (5, 0)], 0, px('#'));
            canvas.art_line(&[(1, 1), (4, 4)], 0, px('#'));
        }
        assert_eq!(
            screen.snapshot(),
            "_____/_\n|\\../..\n|.\\/...\n|./\\...\n|/..\\.."
        );

        let mut single = screen_of(3, 3);
        {
            let mut canvas = single.canvas();
            canvas.art_line(&[(1, 1)], 0, px('#'));
            canvas.art_line(&[], 0, px('#'));
            canvas.art_line(&[(0, 0)], 0, Pixel::transparent());
        }
        assert_eq!(single.snapshot(), "...\n...\n...");
        let single_cell = single.get(1, 1).unwrap();
        assert_eq!(single_cell.pixel.ch, '.');
        assert_eq!(single_cell.z, 0, "a one-point line draws its cell");
        assert_eq!(single.get(0, 0).unwrap().z, i32::MIN);

        let mut huge = screen_of(9, 5);
        huge.canvas()
            .art_line(&[(i32::MIN, i32::MIN), (i32::MAX, i32::MAX)], 0, px('!'));
        assert_eq!(
            huge.snapshot(),
            "\\........\n.\\.......\n..\\......\n...\\.....\n....\\...."
        );
    }

    #[test]
    fn new_primitives_respect_transparency_and_empty_clips() {
        let mut screen = screen_of(4, 4);
        let mut canvas = screen.canvas();
        for iterator in 0..2 {
            if iterator == 1 {
                canvas.set_clip(Rect::new(0, 0, 0, 4));
            }
            let pixel = if iterator == 0 {
                Pixel::transparent()
            } else {
                px('!')
            };
            canvas.draw_circle(2, 2, 3, i32::MAX, pixel);
            canvas.draw_polyline(&[(0, 0), (3, 3)], i32::MAX, pixel);
            canvas.draw_vector(0, 0, 3, 3, i32::MAX, |_| pixel);
            canvas.flood_fill(2, 2, i32::MAX, pixel, None, false);
            canvas.fill(canvas.bounds(), i32::MAX, pixel);
            canvas.art_line(&[(0, 0), (3, 3)], i32::MAX, pixel);
        }
        assert_eq!(screen.snapshot(), "....\n....\n....\n....");
        assert!(screen.cells.iter().all(|cell| cell.z == i32::MIN));
    }
}
