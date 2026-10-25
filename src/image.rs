// SPDX-License-Identifier: Apache-2.0
//! Backend-neutral character-cell images.
//!
//! [`Image`] is a row-major rectangle of [`Pixel`] cells. It is independent of
//! the terminal renderer and the native GPU renderer, so images can be created,
//! transformed, loaded, and saved without a live screen.
//!
//! Transparent cells use the NUL character, matching [`Pixel::transparent`].
//! Text output always renders those cells as ordinary spaces.

use crate::graphics::{Color, Pixel, Rect};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::Path;

/// REXPaint's built-in transparent background color.
const REX_MAGENTA: Color = Color::rgb(255, 0, 255);
const REX_XP_VERSION: i32 = -1;

/// An image error.
#[derive(Debug)]
pub enum ImageError {
    /// A filesystem or stream operation failed.
    Io(std::io::Error),
    /// The input did not follow the requested image format.
    Format(String),
    /// An image dimension cannot be represented or its pixel count overflowed.
    InvalidDimensions { width: u16, height: u16 },
    /// Gzip decompression of a REXPaint XP image failed.
    Decompress,
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Format(message) => write!(f, "invalid image format: {message}"),
            Self::InvalidDimensions { width, height } => {
                write!(f, "invalid image dimensions: {width}x{height}")
            }
            Self::Decompress => write!(f, "failed to decompress image data"),
        }
    }
}

impl Error for ImageError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ImageError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// A rectangular character-cell image.
///
/// Pixels are stored row-major. Coordinates passed to [`Image::get`],
/// [`Image::set`], and drawing methods are relative to the image's top-left
/// corner. The optional origin is metadata for the path or resource an image
/// came from; it does not affect image equality.
#[derive(Debug, Clone, Default)]
pub struct Image {
    width: u16,
    height: u16,
    pixels: Vec<Pixel>,
    origin: Option<String>,
}

impl PartialEq for Image {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width && self.height == other.height && self.pixels == other.pixels
    }
}

impl Eq for Image {}

impl Image {
    /// Creates a filled image.
    ///
    /// The fill pixel is sanitized with [`Pixel::new`].
    pub fn new(width: u16, height: u16, fill: Pixel) -> Self {
        let fill = sanitize(fill);
        let len = cell_count(width, height);
        Self {
            width,
            height,
            pixels: vec![fill; len],
            origin: None,
        }
    }

    /// Creates an image from row-major pixels.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::Format`] when `pixels.len()` is not exactly
    /// `width * height`, and [`ImageError::InvalidDimensions`] if the
    /// multiplication overflows `usize`.
    pub fn from_pixels(width: u16, height: u16, pixels: Vec<Pixel>) -> Result<Self, ImageError> {
        let expected = checked_cell_count(width, height)?;
        if pixels.len() != expected {
            return Err(ImageError::Format(format!(
                "expected {expected} pixels for {width}x{height}, got {}",
                pixels.len()
            )));
        }
        Ok(Self {
            width,
            height,
            pixels: pixels.into_iter().map(sanitize).collect(),
            origin: None,
        })
    }

    /// Creates an image from text.
    ///
    /// Every line is padded on the right to the width of the widest line. A
    /// trailing newline does not create an extra row. When
    /// `spaces_transparent` is true, ASCII spaces become transparent NUL
    /// cells; otherwise they are ordinary space cells with the requested
    /// colors.
    pub fn from_text(text: &str, fg: Color, bg: Option<Color>, spaces_transparent: bool) -> Self {
        let lines: Vec<&str> = text.lines().collect();
        Self::from_line_strs(lines, fg, bg, spaces_transparent)
    }

    /// Creates an image from individual lines.
    ///
    /// Lines are padded with spaces to a common width. See [`Image::from_text`]
    /// for the meaning of `spaces_transparent`.
    pub fn from_lines<I, S>(
        lines: I,
        fg: Color,
        bg: Option<Color>,
        spaces_transparent: bool,
    ) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let owned: Vec<String> = lines
            .into_iter()
            .map(|line| line.as_ref().to_owned())
            .collect();
        let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
        Self::from_line_strs(refs, fg, bg, spaces_transparent)
    }

    fn from_line_strs(
        lines: Vec<&str>,
        fg: Color,
        bg: Option<Color>,
        spaces_transparent: bool,
    ) -> Self {
        let height = lines.len().min(usize::from(u16::MAX)) as u16;
        let widest = lines
            .iter()
            .take(usize::from(height))
            .map(|line| line.chars().count())
            .max()
            .unwrap_or(0)
            .min(usize::from(u16::MAX));
        let width = widest as u16;
        let mut pixels = Vec::with_capacity(widest.saturating_mul(usize::from(height)));

        for line in lines.iter().take(usize::from(height)) {
            let mut written = 0;
            for ch in line.chars().take(widest) {
                pixels.push(if spaces_transparent && ch == ' ' {
                    Pixel::transparent()
                } else {
                    Pixel::new(ch, fg, bg)
                });
                written += 1;
            }
            for _ in written..widest {
                pixels.push(if spaces_transparent {
                    Pixel::transparent()
                } else {
                    Pixel::new(' ', fg, bg)
                });
            }
        }

        Self {
            width,
            height,
            pixels,
            origin: None,
        }
    }

    /// Returns the image width in cells.
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Returns the image height in cells.
    pub fn height(&self) -> u16 {
        self.height
    }

    /// Returns the image bounds: `(0, 0, width, height)`.
    pub fn bounds(&self) -> Rect {
        Rect::new(0, 0, self.width, self.height)
    }

    /// Returns the optional source path, URL, or other origin.
    pub fn origin(&self) -> Option<&str> {
        self.origin.as_deref()
    }

    /// Sets the origin, returning the image.
    pub fn with_origin(mut self, origin: impl Into<String>) -> Self {
        self.origin = Some(origin.into());
        self
    }

    /// Sets the origin in place.
    pub fn set_origin(&mut self, origin: impl Into<String>) {
        self.origin = Some(origin.into());
    }

    /// Removes the origin metadata.
    pub fn clear_origin(&mut self) {
        self.origin = None;
    }

    /// Gets a pixel at an image coordinate.
    pub fn get(&self, x: i32, y: i32) -> Option<Pixel> {
        if !self.bounds().contains(x, y) {
            return None;
        }
        Some(self.pixels[self.index(x as usize, y as usize)])
    }

    /// Sets a pixel at an image coordinate.
    ///
    /// Returns `false` (without changing anything) when the coordinate is
    /// outside the image. The pixel is sanitized with [`Pixel::new`].
    pub fn set(&mut self, x: i32, y: i32, pixel: Pixel) -> bool {
        if !self.bounds().contains(x, y) {
            return false;
        }
        let index = self.index(x as usize, y as usize);
        self.pixels[index] = sanitize(pixel);
        true
    }

    /// Iterates over `(x, y, pixel)` in row-major order.
    pub fn pixels(&self) -> impl Iterator<Item = (i32, i32, Pixel)> + '_ {
        let width = usize::from(self.width);
        self.pixels
            .iter()
            .enumerate()
            .map(move |(index, pixel)| ((index % width) as i32, (index / width) as i32, *pixel))
    }

    /// Returns all pixels in row-major order.
    pub fn pixels_slice(&self) -> &[Pixel] {
        &self.pixels
    }

    /// Iterates over rows, each as a slice of pixels.
    pub fn rows(&self) -> impl Iterator<Item = &[Pixel]> + '_ {
        let width = usize::from(self.width);
        (0..usize::from(self.height)).map(move |y| &self.pixels[y * width..(y + 1) * width])
    }

    /// Returns one row, or `None` when `y` is outside the image.
    pub fn row(&self, y: i32) -> Option<&[Pixel]> {
        if y < 0 || y >= i32::from(self.height) {
            return None;
        }
        let width = usize::from(self.width);
        let start = y as usize * width;
        Some(&self.pixels[start..start + width])
    }

    /// Returns all rows as strings, with transparent cells rendered as spaces.
    pub fn row_strings(&self) -> Vec<String> {
        self.rows().map(row_to_string).collect()
    }

    /// Returns one row as a string, or `None` when `y` is outside the image.
    pub fn row_string(&self, y: i32) -> Option<String> {
        self.row(y).map(row_to_string)
    }

    /// Returns the image as newline-separated text, without a trailing newline.
    ///
    /// Transparent NUL cells are rendered as spaces.
    pub fn to_text(&self) -> String {
        self.row_strings().join("\n")
    }

    /// Fills every cell with `pixel`.
    pub fn fill(&mut self, pixel: Pixel) {
        self.pixels.fill(sanitize(pixel));
    }

    /// Fills the part of `rect` that overlaps this image.
    pub fn fill_rect(&mut self, rect: Rect, pixel: Pixel) {
        let Some(visible) = self.bounds().intersection(rect) else {
            return;
        };
        let pixel = sanitize(pixel);
        for y in visible.y..visible.y + i32::from(visible.height) {
            for x in visible.x..visible.x + i32::from(visible.width) {
                let index = self.index(x as usize, y as usize);
                self.pixels[index] = pixel;
            }
        }
    }

    /// Returns a deep copy of this image.
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Copies a rectangle, padding cells outside the source with transparency.
    pub fn copy_rect(&self, rect: Rect) -> Self {
        self.crop(rect)
    }

    /// Crops a rectangle, padding cells outside the source with transparency.
    ///
    /// The returned image has exactly `rect.width` by `rect.height` cells.
    /// Negative source coordinates are allowed.
    pub fn crop(&self, rect: Rect) -> Self {
        let width = rect.width;
        let height = rect.height;
        let len = cell_count(width, height);
        let mut pixels = Vec::with_capacity(len);
        for y in 0..i64::from(height) {
            let source_y = i64::from(rect.y) + y;
            for x in 0..i64::from(width) {
                let source_x = i64::from(rect.x) + x;
                pixels.push(
                    if source_x >= 0
                        && source_y >= 0
                        && source_x < i64::from(self.width)
                        && source_y < i64::from(self.height)
                    {
                        self.pixels[self.index(source_x as usize, source_y as usize)]
                    } else {
                        Pixel::transparent()
                    },
                );
            }
        }
        Self {
            width,
            height,
            pixels,
            origin: self.origin.clone(),
        }
    }

    /// Splits the image into fixed-size frames in row-major order.
    ///
    /// Partial frames at the right and bottom edges are padded with
    /// transparency. A zero frame dimension or empty source yields no frames.
    pub fn split(&self, frame_width: u16, frame_height: u16) -> Vec<Self> {
        if frame_width == 0 || frame_height == 0 || self.width == 0 || self.height == 0 {
            return Vec::new();
        }
        let columns = div_ceil(self.width, frame_width);
        let rows = div_ceil(self.height, frame_height);
        let mut frames = Vec::with_capacity(usize::from(columns) * usize::from(rows));
        for frame_y in 0..rows {
            for frame_x in 0..columns {
                frames.push(self.crop(Rect::new(
                    i32::from(frame_x) * i32::from(frame_width),
                    i32::from(frame_y) * i32::from(frame_height),
                    frame_width,
                    frame_height,
                )));
            }
        }
        frames
    }

    /// Pads the image with transparent cells.
    ///
    /// # Panics
    ///
    /// Panics when a resulting dimension does not fit in `u16`.
    pub fn pad(&self, left: u16, top: u16, right: u16, bottom: u16) -> Self {
        let width = self
            .width
            .checked_add(left)
            .and_then(|width| width.checked_add(right))
            .expect("padded image width exceeds u16");
        let height = self
            .height
            .checked_add(top)
            .and_then(|height| height.checked_add(bottom))
            .expect("padded image height exceeds u16");
        let mut image = Self::new(width, height, Pixel::transparent());
        image.origin = self.origin.clone();
        image.draw(i32::from(left), i32::from(top), self);
        image
    }

    /// Resizes using nearest-neighbor sampling.
    ///
    /// If either source dimension is zero, the result is transparent.
    pub fn resize_nearest(&self, width: u16, height: u16) -> Self {
        let len = cell_count(width, height);
        let mut pixels = Vec::with_capacity(len);
        for y in 0..u32::from(height) {
            let source_y = if self.height == 0 {
                None
            } else {
                Some(
                    (y as u64 * u64::from(self.height) / u64::from(height))
                        .min(u64::from(self.height - 1)) as usize,
                )
            };
            for x in 0..u32::from(width) {
                let source_x = if self.width == 0 {
                    None
                } else {
                    Some(
                        (x as u64 * u64::from(self.width) / u64::from(width))
                            .min(u64::from(self.width - 1)) as usize,
                    )
                };
                pixels.push(match (source_x, source_y) {
                    (Some(sx), Some(sy)) => self.pixels[self.index(sx, sy)],
                    _ => Pixel::transparent(),
                });
            }
        }
        Self {
            width,
            height,
            pixels,
            origin: self.origin.clone(),
        }
    }

    /// Trims transparent rows and columns from all four sides.
    ///
    /// An entirely transparent image becomes `0x0`. If no transparent border
    /// exists, the image is returned unchanged (with its origin preserved).
    pub fn trim(&self) -> Self {
        self.trim_by(Pixel::is_transparent)
    }

    /// Trims rows and columns whose pixels have `bg` as their background.
    ///
    /// Transparent cells also count as background. This crops the image; it
    /// does not change colors. See [`Image::replace_bg`] for an exterior flood
    /// fill.
    pub fn trim_bg(&self, bg: Color) -> Self {
        self.trim_by(|pixel| pixel.is_transparent() || pixel.bg == Some(bg))
    }

    /// Trims rows and columns containing only pixels matching `is_blank`.
    pub fn trim_by(&self, is_blank: impl Fn(&Pixel) -> bool) -> Self {
        let Some((min_x, min_y, max_x, max_y)) = self.ink_bounds(&is_blank) else {
            let mut image = Self::new(0, 0, Pixel::transparent());
            image.origin = self.origin.clone();
            return image;
        };
        self.crop(Rect::new(
            min_x,
            min_y,
            (max_x - min_x + 1) as u16,
            (max_y - min_y + 1) as u16,
        ))
    }

    fn ink_bounds(&self, is_blank: impl Fn(&Pixel) -> bool) -> Option<(i32, i32, i32, i32)> {
        let mut min_x = self.width;
        let mut min_y = self.height;
        let mut max_x = 0;
        let mut max_y = 0;
        let mut found = false;
        for y in 0..self.height {
            for x in 0..self.width {
                let pixel = self.pixels[self.index(x as usize, y as usize)];
                if !is_blank(&pixel) {
                    found = true;
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x);
                    max_y = max_y.max(y);
                }
            }
        }
        found.then(|| {
            (
                i32::from(min_x),
                i32::from(min_y),
                i32::from(max_x),
                i32::from(max_y),
            )
        })
    }

    /// Replaces exterior background cells whose background matches `old`.
    ///
    /// A background cell is transparent, or a space with the requested
    /// background. Only the 4-connected region reachable from the image edge
    /// is changed; enclosed matching regions and glyph cells are left
    /// untouched.
    pub fn replace_bg(&self, old: Option<Color>, new: Option<Color>) -> Self {
        let mut image = self.clone();
        if old == new || image.pixels.is_empty() {
            return image;
        }
        let width = usize::from(image.width);
        let height = usize::from(image.height);
        let mut visited = vec![false; image.pixels.len()];
        let mut stack = Vec::new();

        let seed = |index: usize, stack: &mut Vec<usize>, visited: &mut [bool]| {
            if !visited[index] && background_matches(&image.pixels[index], old) {
                visited[index] = true;
                stack.push(index);
            }
        };
        for x in 0..width {
            seed(x, &mut stack, &mut visited);
            seed((height - 1) * width + x, &mut stack, &mut visited);
        }
        for y in 0..height {
            seed(y * width, &mut stack, &mut visited);
            seed(y * width + width - 1, &mut stack, &mut visited);
        }

        while let Some(index) = stack.pop() {
            let pixel = &mut image.pixels[index];
            *pixel = Pixel::new(pixel.ch, pixel.fg, new);
            let x = index % width;
            let y = index / width;
            let mut push = |next: usize| {
                if !visited[next] && background_matches(&image.pixels[next], old) {
                    visited[next] = true;
                    stack.push(next);
                }
            };
            if x > 0 {
                push(index - 1);
            }
            if x + 1 < width {
                push(index + 1);
            }
            if y > 0 {
                push(index - width);
            }
            if y + 1 < height {
                push(index + width);
            }
        }

        image
    }

    /// Applies character-cell edge smoothing.
    ///
    /// Solid cells (anything other than transparent or space) with a missing
    /// orthogonal neighbor receive the same directional edge glyphs as
    /// CosPlay. `char_ramp` is an optional light-to-dark fallback ramp for
    /// edge patterns without a directional glyph. When the eight-neighbor
    /// coverage is below `threshold` (clamped to `0.0..=1.0`), the ramp is
    /// indexed by that coverage. An empty ramp disables the fallback.
    pub fn antialias(&self, char_ramp: &str, threshold: f32) -> Self {
        let ramp: Vec<char> = char_ramp.chars().collect();
        let threshold = if threshold.is_nan() {
            0.0
        } else {
            threshold.clamp(0.0, 1.0)
        };
        let is_blank = |pixel: Pixel| pixel.is_transparent() || pixel.ch == ' ';
        let mut image = self.clone();

        for y in 0..i32::from(self.height) {
            for x in 0..i32::from(self.width) {
                let pixel = self.get(x, y).expect("coordinates are in bounds");
                if is_blank(pixel) {
                    continue;
                }
                let top = y > 0 && !is_blank(self.get(x, y - 1).unwrap());
                let left = x > 0 && !is_blank(self.get(x - 1, y).unwrap());
                let bottom =
                    y + 1 < i32::from(self.height) && !is_blank(self.get(x, y + 1).unwrap());
                let right = x + 1 < i32::from(self.width) && !is_blank(self.get(x + 1, y).unwrap());

                if top && left && bottom && right {
                    continue;
                }

                let edge = match (top, left, bottom, right) {
                    (false, false, true, true) => Some('d'),
                    (false, true, true, false) => Some('b'),
                    (true, true, false, false) => Some('F'),
                    (true, false, false, true) => Some('Y'),
                    (false, false, false, true) => Some(';'),
                    (true, false, false, false) => Some('V'),
                    (false, true, false, false) => Some('>'),
                    (false, false, true, false) => Some('<'),
                    (false, false, false, false) => Some('@'),
                    _ => None,
                };

                let glyph = edge.or_else(|| {
                    if ramp.is_empty() {
                        return None;
                    }
                    let mut solid = 0_u32;
                    for ny in (y - 1).max(0)..=(y + 1).min(i32::from(self.height) - 1) {
                        for nx in (x - 1).max(0)..=(x + 1).min(i32::from(self.width) - 1) {
                            if (nx != x || ny != y) && !is_blank(self.get(nx, ny).unwrap()) {
                                solid += 1;
                            }
                        }
                    }
                    let coverage = solid as f32 / 8.0;
                    if coverage < threshold {
                        let index = ((coverage * (ramp.len() - 1) as f32).round() as usize)
                            .min(ramp.len() - 1);
                        Some(ramp[index])
                    } else {
                        None
                    }
                });

                if let Some(glyph) = glyph {
                    image.set(x, y, Pixel::new(glyph, pixel.fg, pixel.bg));
                }
            }
        }
        image
    }

    /// Flips horizontally, substituting glyphs that have a mirrored form.
    pub fn hor_flip(&self) -> Self {
        let mut image = self.copy();
        for y in 0..i32::from(self.height) {
            for x in 0..i32::from(self.width) {
                let source = self.get(i32::from(self.width) - 1 - x, y).unwrap();
                image.set(
                    x,
                    y,
                    Pixel::new(flip_glyph(source.ch, HOR_FLIP_MAP), source.fg, source.bg),
                );
            }
        }
        image
    }

    /// Flips vertically, substituting glyphs that have an upside-down form.
    pub fn ver_flip(&self) -> Self {
        let mut image = self.copy();
        for y in 0..i32::from(self.height) {
            for x in 0..i32::from(self.width) {
                let source = self.get(x, i32::from(self.height) - 1 - y).unwrap();
                image.set(
                    x,
                    y,
                    Pixel::new(flip_glyph(source.ch, VER_FLIP_MAP), source.fg, source.bg),
                );
            }
        }
        image
    }

    /// Places `other` directly below this image.
    ///
    /// The result is as wide as the wider image. Unused cells are filled with
    /// `bg`. Transparent pixels in either input do not overwrite `bg`.
    ///
    /// # Panics
    ///
    /// Panics when the combined dimensions do not fit in `u16`.
    pub fn stitch_below(&self, other: &Self, bg: Pixel) -> Self {
        let width = self.width.max(other.width);
        let height = self
            .height
            .checked_add(other.height)
            .expect("stitched image height exceeds u16");
        let mut image = Self::new(width, height, bg);
        image.origin = self.origin.clone();
        image.draw(0, 0, self);
        image.draw(0, i32::from(self.height), other);
        image
    }

    /// Places `other` directly to the right of this image.
    ///
    /// The result is as tall as the taller image. Unused cells are filled with
    /// `bg`. Transparent pixels in either input do not overwrite `bg`.
    ///
    /// # Panics
    ///
    /// Panics when the combined dimensions do not fit in `u16`.
    pub fn stitch_right(&self, other: &Self, bg: Pixel) -> Self {
        let width = self
            .width
            .checked_add(other.width)
            .expect("stitched image width exceeds u16");
        let height = self.height.max(other.height);
        let mut image = Self::new(width, height, bg);
        image.origin = self.origin.clone();
        image.draw(0, 0, self);
        image.draw(i32::from(self.width), 0, other);
        image
    }

    /// Draws `source` onto this image, skipping transparent pixels.
    ///
    /// Coordinates may be negative or outside the destination; clipping is
    /// per cell.
    pub fn draw(&mut self, x: i32, y: i32, source: &Self) {
        for (sx, sy, pixel) in source.pixels() {
            if pixel.is_transparent() {
                continue;
            }
            let Some(dx) = i64::from(x).checked_add(i64::from(sx)) else {
                continue;
            };
            let Some(dy) = i64::from(y).checked_add(i64::from(sy)) else {
                continue;
            };
            if let (Ok(dx), Ok(dy)) = (i32::try_from(dx), i32::try_from(dy)) {
                self.set(dx, dy, pixel);
            }
        }
    }

    /// Alias for [`Image::draw`].
    pub fn draw_image(&mut self, x: i32, y: i32, source: &Self) {
        self.draw(x, y, source);
    }

    /// Alias for [`Image::draw`].
    pub fn blit(&mut self, x: i32, y: i32, source: &Self) {
        self.draw(x, y, source);
    }

    /// Loads an image, selecting the format from a case-insensitive extension.
    ///
    /// Recognized extensions are `.csv`, `.xp`, and `.txt`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ImageError> {
        let path = path.as_ref();
        match extension(path)?.as_str() {
            "csv" => Self::load_rexpaint_csv(path),
            "xp" => Self::load_rexpaint_xp(path),
            "txt" => Self::load_txt(path),
            other => Err(ImageError::Format(format!(
                "unsupported image extension .{other}: {}",
                path.display()
            ))),
        }
    }

    /// Alias for [`Image::load`].
    pub fn load_auto(path: impl AsRef<Path>) -> Result<Self, ImageError> {
        Self::load(path)
    }

    /// Saves an image, selecting the format from a case-insensitive extension.
    ///
    /// Recognized extensions are `.csv`, `.xp`, and `.txt`.
    pub fn save(path: impl AsRef<Path>, image: &Self) -> Result<(), ImageError> {
        let path = path.as_ref();
        match extension(path)?.as_str() {
            "csv" => image.save_rexpaint_csv(path),
            "xp" => image.save_rexpaint_xp(path),
            "txt" => image.save_txt(path),
            other => Err(ImageError::Format(format!(
                "unsupported image extension .{other}: {}",
                path.display()
            ))),
        }
    }

    /// Alias for [`Image::save`].
    pub fn save_auto(path: impl AsRef<Path>, image: &Self) -> Result<(), ImageError> {
        Self::save(path, image)
    }

    /// Loads a plain-text image using white foreground and no background.
    pub fn load_txt(path: impl AsRef<Path>) -> Result<Self, ImageError> {
        let path = path.as_ref();
        let bytes = fs::read(path)?;
        Self::load_txt_bytes(&bytes).map(|image| image.with_origin(path.to_string_lossy()))
    }

    /// Saves this image as plain text.
    ///
    /// Transparent cells are written as spaces. Each row is newline-terminated.
    pub fn save_txt(&self, path: impl AsRef<Path>) -> Result<(), ImageError> {
        let bytes = self.save_txt_bytes();
        fs::write(path, bytes)?;
        Ok(())
    }

    /// Loads a plain-text image from bytes.
    pub fn load_txt_bytes(bytes: &[u8]) -> Result<Self, ImageError> {
        let text = decode_utf8(bytes, "text image")?;
        Ok(Self::from_text(text, Color::WHITE, None, false))
    }

    /// Returns this image as plain-text bytes. Each row is newline-terminated.
    pub fn save_txt_bytes(&self) -> Vec<u8> {
        let mut text = self.to_text();
        if self.height != 0 {
            text.push('\n');
        }
        text.into_bytes()
    }

    /// Loads a REXPaint/CosPlay CSV image.
    pub fn load_rexpaint_csv(path: impl AsRef<Path>) -> Result<Self, ImageError> {
        let path = path.as_ref();
        let bytes = fs::read(path)?;
        Self::load_rexpaint_csv_bytes(&bytes).map(|image| image.with_origin(path.to_string_lossy()))
    }

    /// Saves this image as a REXPaint/CosPlay CSV image.
    pub fn save_rexpaint_csv(&self, path: impl AsRef<Path>) -> Result<(), ImageError> {
        fs::write(path, self.save_rexpaint_csv_bytes())?;
        Ok(())
    }

    /// Loads a REXPaint/CosPlay CSV image from bytes.
    ///
    /// The format is a CosPlay header followed by
    /// `x,y,char_code,0xRRGGBB,0xRRGGBB` rows. A magenta background is read
    /// as no background. Coordinates do not need to be complete or sorted.
    pub fn load_rexpaint_csv_bytes(bytes: &[u8]) -> Result<Self, ImageError> {
        let text = decode_utf8(bytes, "REXPaint CSV")?;
        let mut records = Vec::new();
        let mut saw_data = false;

        for (line_index, raw_line) in text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            if !saw_data && line.to_ascii_lowercase().starts_with("cosplay") {
                continue;
            }
            saw_data = true;
            let fields: Vec<&str> = line.split(',').map(str::trim).collect();
            if fields.len() != 5 {
                return Err(ImageError::Format(format!(
                    "CSV line {}: expected 5 fields, got {}",
                    line_index + 1,
                    fields.len()
                )));
            }
            let x = parse_c_int(fields[0]).map_err(|message| {
                ImageError::Format(format!("CSV line {}: x: {message}", line_index + 1))
            })?;
            let y = parse_c_int(fields[1]).map_err(|message| {
                ImageError::Format(format!("CSV line {}: y: {message}", line_index + 1))
            })?;
            if !(0..=i64::from(u16::MAX - 1)).contains(&x)
                || !(0..=i64::from(u16::MAX - 1)).contains(&y)
            {
                return Err(ImageError::Format(format!(
                    "CSV line {}: coordinate ({x}, {y}) is outside 0..={}",
                    line_index + 1,
                    u16::MAX - 1
                )));
            }
            let ch_code = parse_c_int(fields[2]).map_err(|message| {
                ImageError::Format(format!("CSV line {}: character: {message}", line_index + 1))
            })?;
            let ch = u32::try_from(ch_code)
                .ok()
                .and_then(char::from_u32)
                .ok_or_else(|| {
                    ImageError::Format(format!(
                        "CSV line {}: invalid character code {ch_code}",
                        line_index + 1
                    ))
                })?;
            let fg = parse_color(fields[3]).map_err(|message| {
                ImageError::Format(format!(
                    "CSV line {}: foreground: {message}",
                    line_index + 1
                ))
            })?;
            let bg = parse_color(fields[4]).map_err(|message| {
                ImageError::Format(format!(
                    "CSV line {}: background: {message}",
                    line_index + 1
                ))
            })?;
            let bg = (bg != REX_MAGENTA).then_some(bg);
            records.push((x as u16, y as u16, Pixel::new(ch, fg, bg)));
        }

        if records.is_empty() {
            return Err(ImageError::Format("CSV contains no pixel rows".to_owned()));
        }

        let width = records.iter().map(|(x, _, _)| *x).max().unwrap() + 1;
        let height = records.iter().map(|(_, y, _)| *y).max().unwrap() + 1;
        let len = cell_count(width, height);
        let mut pixels = vec![Pixel::transparent(); len];
        for (x, y, pixel) in records {
            let index = usize::from(y) * usize::from(width) + usize::from(x);
            pixels[index] = pixel;
        }
        Self::from_pixels(width, height, pixels)
    }

    /// Returns this image in REXPaint/CosPlay CSV format.
    ///
    /// Transparent backgrounds are written as REXPaint's transparent magenta.
    pub fn save_rexpaint_csv_bytes(&self) -> Vec<u8> {
        let origin = self.origin().unwrap_or("code");
        let mut result = format!(
            "CosPlay image [{}x{}, origin={}, bg=0xFF00FF]\n",
            self.width, self.height, origin
        );
        for (x, y, pixel) in self.pixels() {
            let bg = pixel.bg.unwrap_or(REX_MAGENTA);
            result.push_str(&format!(
                "{x},{y},{},0x{:02X}{:02X}{:02X},0x{:02X}{:02X}{:02X}\n",
                u32::from(pixel.ch),
                pixel.fg.r,
                pixel.fg.g,
                pixel.fg.b,
                bg.r,
                bg.g,
                bg.b
            ));
        }
        result.into_bytes()
    }

    /// Loads a gzip-compressed REXPaint XP image.
    ///
    /// Later layers overwrite non-transparent cells of earlier layers.
    pub fn load_rexpaint_xp(path: impl AsRef<Path>) -> Result<Self, ImageError> {
        let path = path.as_ref();
        let bytes = fs::read(path)?;
        Self::load_rexpaint_xp_bytes(&bytes).map(|image| image.with_origin(path.to_string_lossy()))
    }

    /// Saves this image as a gzip-compressed, single-layer REXPaint XP image.
    pub fn save_rexpaint_xp(&self, path: impl AsRef<Path>) -> Result<(), ImageError> {
        let bytes = self.save_rexpaint_xp_bytes()?;
        fs::write(path, bytes)?;
        Ok(())
    }

    /// Loads a gzip-compressed REXPaint XP image from bytes.
    pub fn load_rexpaint_xp_bytes(bytes: &[u8]) -> Result<Self, ImageError> {
        let mut decoder = GzDecoder::new(bytes);
        let mut raw = Vec::new();
        decoder
            .read_to_end(&mut raw)
            .map_err(|_| ImageError::Decompress)?;
        let mut cursor = Cursor::new(raw.as_slice());

        let _version = read_xp_i32(&mut cursor, "version")?;
        let layer_count = read_xp_i32(&mut cursor, "layer count")?;
        if layer_count <= 0 {
            return Err(ImageError::Format(
                "XP image must contain at least one layer".to_owned(),
            ));
        }

        let mut result: Option<Self> = None;
        for layer_index in 0..layer_count {
            let width_i32 = read_xp_i32(&mut cursor, "layer width")?;
            let height_i32 = read_xp_i32(&mut cursor, "layer height")?;
            let width = xp_dimension(width_i32, height_i32)?;
            let height = xp_dimension(height_i32, width_i32)?;
            checked_cell_count(width, height)?;

            if let Some(image) = &result {
                if image.width != width || image.height != height {
                    return Err(ImageError::Format(format!(
                        "XP layer {layer_index} is {width}x{height}, expected {}x{}",
                        image.width, image.height
                    )));
                }
            } else {
                result = Some(Self::new(width, height, Pixel::transparent()));
            }
            let image = result.as_mut().expect("layer zero initializes the image");

            for x in 0..usize::from(width) {
                for y in 0..usize::from(height) {
                    let ch_code = read_xp_i32(&mut cursor, "cell character")?;
                    let ch = u32::try_from(ch_code)
                        .ok()
                        .and_then(char::from_u32)
                        .ok_or_else(|| {
                            ImageError::Format(format!(
                                "XP layer {layer_index} cell ({x}, {y}) has invalid character code {ch_code}"
                            ))
                        })?;
                    let fg = Color::rgb(
                        read_xp_u8(&mut cursor, "foreground red")?,
                        read_xp_u8(&mut cursor, "foreground green")?,
                        read_xp_u8(&mut cursor, "foreground blue")?,
                    );
                    let bg = Color::rgb(
                        read_xp_u8(&mut cursor, "background red")?,
                        read_xp_u8(&mut cursor, "background green")?,
                        read_xp_u8(&mut cursor, "background blue")?,
                    );
                    let bg = (bg != REX_MAGENTA).then_some(bg);
                    let pixel = Pixel::new(ch, fg, bg);
                    // Layer zero establishes transparency; later layers only
                    // overwrite cells that actually contain a glyph.
                    if layer_index == 0 || !pixel.is_transparent() {
                        let index = image.index(x, y);
                        image.pixels[index] = pixel;
                    }
                }
            }
        }

        result.ok_or_else(|| ImageError::Format("XP image has no layers".to_owned()))
    }

    /// Returns this image as a gzip-compressed, single-layer REXPaint XP image.
    pub fn save_rexpaint_xp_bytes(&self) -> Result<Vec<u8>, ImageError> {
        let area = cell_count(self.width, self.height);
        let mut raw = Vec::with_capacity(12 + area.saturating_mul(10));
        raw.extend_from_slice(&REX_XP_VERSION.to_le_bytes());
        raw.extend_from_slice(&1_i32.to_le_bytes());
        raw.extend_from_slice(&i32::from(self.width).to_le_bytes());
        raw.extend_from_slice(&i32::from(self.height).to_le_bytes());

        for x in 0..usize::from(self.width) {
            for y in 0..usize::from(self.height) {
                let pixel = self.pixels[self.index(x, y)];
                raw.extend_from_slice(&(u32::from(pixel.ch) as i32).to_le_bytes());
                let fg = color_hex(pixel.fg);
                raw.extend_from_slice(&[fg.0, fg.1, fg.2]);
                let bg = pixel.bg.unwrap_or(REX_MAGENTA);
                raw.extend_from_slice(&[bg.r, bg.g, bg.b]);
            }
        }

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&raw)?;
        Ok(encoder.finish()?)
    }

    /// Returns this image as a plain string.
    pub fn to_txt(&self) -> String {
        self.to_text()
    }

    /// Returns this image as REXPaint/CosPlay CSV text.
    pub fn to_rexpaint_csv(&self) -> String {
        String::from_utf8(self.save_rexpaint_csv_bytes()).expect("CSV output is always valid UTF-8")
    }

    /// Returns this image as REXPaint XP bytes.
    pub fn to_rexpaint_xp(&self) -> Result<Vec<u8>, ImageError> {
        self.save_rexpaint_xp_bytes()
    }

    fn index(&self, x: usize, y: usize) -> usize {
        y * usize::from(self.width) + x
    }
}

impl fmt::Display for Image {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_text())
    }
}

/// Loads an image, selecting the format from a case-insensitive extension.
pub fn load(path: impl AsRef<Path>) -> Result<Image, ImageError> {
    Image::load(path)
}

/// Saves an image, selecting the format from a case-insensitive extension.
pub fn save(path: impl AsRef<Path>, image: &Image) -> Result<(), ImageError> {
    Image::save(path, image)
}

/// Alias for [`load`].
pub fn load_auto(path: impl AsRef<Path>) -> Result<Image, ImageError> {
    Image::load_auto(path)
}

/// Alias for [`save`].
pub fn save_auto(path: impl AsRef<Path>, image: &Image) -> Result<(), ImageError> {
    Image::save_auto(path, image)
}

/// Loads a plain-text image.
pub fn load_txt(path: impl AsRef<Path>) -> Result<Image, ImageError> {
    Image::load_txt(path)
}

/// Saves an image as plain text.
pub fn save_txt(path: impl AsRef<Path>, image: &Image) -> Result<(), ImageError> {
    image.save_txt(path)
}

/// Loads a plain-text image from bytes.
pub fn load_txt_bytes(bytes: &[u8]) -> Result<Image, ImageError> {
    Image::load_txt_bytes(bytes)
}

/// Returns a plain-text byte representation of an image.
pub fn save_txt_bytes(image: &Image) -> Vec<u8> {
    image.save_txt_bytes()
}

/// Loads a REXPaint/CosPlay CSV image.
pub fn load_rexpaint_csv(path: impl AsRef<Path>) -> Result<Image, ImageError> {
    Image::load_rexpaint_csv(path)
}

/// Saves an image as a REXPaint/CosPlay CSV image.
pub fn save_rexpaint_csv(path: impl AsRef<Path>, image: &Image) -> Result<(), ImageError> {
    image.save_rexpaint_csv(path)
}

/// Loads a REXPaint/CosPlay CSV image from bytes.
pub fn load_rexpaint_csv_bytes(bytes: &[u8]) -> Result<Image, ImageError> {
    Image::load_rexpaint_csv_bytes(bytes)
}

/// Returns a REXPaint/CosPlay CSV byte representation of an image.
pub fn save_rexpaint_csv_bytes(image: &Image) -> Vec<u8> {
    image.save_rexpaint_csv_bytes()
}

/// Loads a gzip-compressed REXPaint XP image.
pub fn load_rexpaint_xp(path: impl AsRef<Path>) -> Result<Image, ImageError> {
    Image::load_rexpaint_xp(path)
}

/// Saves an image as a gzip-compressed REXPaint XP image.
pub fn save_rexpaint_xp(path: impl AsRef<Path>, image: &Image) -> Result<(), ImageError> {
    image.save_rexpaint_xp(path)
}

/// Loads a gzip-compressed REXPaint XP image from bytes.
pub fn load_rexpaint_xp_bytes(bytes: &[u8]) -> Result<Image, ImageError> {
    Image::load_rexpaint_xp_bytes(bytes)
}

/// Returns a gzip-compressed REXPaint XP byte representation of an image.
pub fn save_rexpaint_xp_bytes(image: &Image) -> Result<Vec<u8>, ImageError> {
    image.save_rexpaint_xp_bytes()
}

const HOR_FLIP_MAP: &[(char, char)] = &[
    ('{', '}'),
    ('/', '\\'),
    ('7', '\\'),
    ('<', '>'),
    ('(', ')'),
    ('[', ']'),
    ('`', '\''),
    ('\'', '\''),
    ('e', 'g'),
    ('J', 'L'),
    ('q', 'p'),
    ('d', 'b'),
    ('\\', '/'),
];

const VER_FLIP_MAP: &[(char, char)] = &[
    ('-', '-'),
    ('_', '-'),
    ('`', '.'),
    ('.', '.'),
    ('"', ','),
    (',', '\''),
    ('\'', ','),
    ('/', '\\'),
    ('a', 'e'),
    ('b', 'p'),
    ('V', '^'),
    ('M', 'W'),
    ('9', '6'),
    ('P', 'd'),
    ('z', 's'),
];

fn flip_glyph(ch: char, map: &[(char, char)]) -> char {
    if let Some((_, replacement)) = map.iter().find(|(from, _)| *from == ch) {
        *replacement
    } else if let Some((replacement, _)) = map.iter().find(|(_, to)| *to == ch) {
        *replacement
    } else {
        ch
    }
}

fn sanitize(pixel: Pixel) -> Pixel {
    Pixel::new(pixel.ch, pixel.fg, pixel.bg)
}

fn cell_count(width: u16, height: u16) -> usize {
    checked_cell_count(width, height).expect("image dimensions overflow usize")
}

fn checked_cell_count(width: u16, height: u16) -> Result<usize, ImageError> {
    let _ = u32::from(width)
        .checked_mul(u32::from(height))
        .ok_or(ImageError::InvalidDimensions { width, height })?;
    usize::from(width)
        .checked_mul(usize::from(height))
        .ok_or(ImageError::InvalidDimensions { width, height })
}

fn div_ceil(value: u16, divisor: u16) -> u16 {
    ((u32::from(value) + u32::from(divisor) - 1) / u32::from(divisor)) as u16
}

fn row_to_string(row: &[Pixel]) -> String {
    row.iter()
        .map(|pixel| {
            if pixel.is_transparent() {
                ' '
            } else {
                pixel.ch
            }
        })
        .collect()
}

fn background_matches(pixel: &Pixel, old: Option<Color>) -> bool {
    let is_background_cell = pixel.is_transparent() || pixel.ch == ' ';
    is_background_cell
        && match old {
            Some(color) => pixel.bg == Some(color),
            None => pixel.bg.is_none(),
        }
}

fn extension(path: &Path) -> Result<String, ImageError> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .ok_or_else(|| {
            ImageError::Format(format!(
                "image path has no recognized extension: {}",
                path.display()
            ))
        })?;
    Ok(extension.to_ascii_lowercase())
}

fn decode_utf8<'a>(bytes: &'a [u8], label: &str) -> Result<&'a str, ImageError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| ImageError::Format(format!("{label} is not valid UTF-8: {error}")))?;
    Ok(text.strip_prefix('\u{feff}').unwrap_or(text))
}

fn parse_c_int(input: &str) -> Result<i64, String> {
    let input = input.trim();
    if input.is_empty() {
        return Err("empty integer".to_owned());
    }
    let (negative, unsigned) = match input.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, input.strip_prefix('+').unwrap_or(input)),
    };
    let value = if let Some(hex) = unsigned
        .strip_prefix("0x")
        .or_else(|| unsigned.strip_prefix("0X"))
    {
        i64::from_str_radix(hex, 16).map_err(|_| format!("invalid integer {input:?}"))?
    } else {
        unsigned
            .parse::<i64>()
            .map_err(|_| format!("invalid integer {input:?}"))?
    };
    Ok(if negative { -value } else { value })
}

fn parse_color(input: &str) -> Result<Color, String> {
    let input = input.trim();
    let value = if let Some(hex) = input
        .strip_prefix("0x")
        .or_else(|| input.strip_prefix("0X"))
        .or_else(|| input.strip_prefix('#'))
    {
        i64::from_str_radix(hex, 16).map_err(|_| format!("invalid color {input:?}"))?
    } else {
        parse_c_int(input)?
    };
    if !(0..=0xFF_FFFF).contains(&value) {
        return Err(format!("color {input:?} is outside 0x000000..=0xFFFFFF"));
    }
    Ok(Color::rgb(
        ((value >> 16) & 0xFF) as u8,
        ((value >> 8) & 0xFF) as u8,
        (value & 0xFF) as u8,
    ))
}

fn color_hex(color: Color) -> (u8, u8, u8) {
    (color.r, color.g, color.b)
}

fn read_xp_i32(cursor: &mut Cursor<&[u8]>, what: &str) -> Result<i32, ImageError> {
    let mut bytes = [0_u8; 4];
    cursor
        .read_exact(&mut bytes)
        .map_err(|_| ImageError::Format(format!("truncated XP image while reading {what}")))?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_xp_u8(cursor: &mut Cursor<&[u8]>, what: &str) -> Result<u8, ImageError> {
    let mut byte = [0_u8; 1];
    cursor
        .read_exact(&mut byte)
        .map_err(|_| ImageError::Format(format!("truncated XP image while reading {what}")))?;
    Ok(byte[0])
}

fn xp_dimension(value: i32, other: i32) -> Result<u16, ImageError> {
    if value < 0 {
        return Err(ImageError::Format(format!(
            "negative XP dimension {value}x{other}"
        )));
    }
    u16::try_from(value).map_err(|_| ImageError::InvalidDimensions {
        width: u16::try_from(value).unwrap_or(u16::MAX),
        height: u16::try_from(other).unwrap_or(u16::MAX),
    })
}

#[cfg(test)]
type XpLayer = Vec<Vec<(char, Color, Option<Color>)>>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    const RED: Color = Color::rgb(255, 0, 0);
    const GREEN: Color = Color::rgb(0, 255, 0);
    const BLUE: Color = Color::rgb(0, 0, 255);

    fn px(ch: char) -> Pixel {
        Pixel::new(ch, Color::WHITE, None)
    }

    fn temp_file(name: &str) -> (PathBuf, PathBuf) {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before Unix epoch")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "cosplay-image-test-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).expect("create temp directory");
        (directory.clone(), directory.join(name))
    }

    fn remove_temp(directory: &Path) {
        let _ = fs::remove_dir_all(directory);
    }

    fn xp_with_layers(layers: &[XpLayer]) -> Vec<u8> {
        let height = layers[0].len();
        let width = layers[0][0].len();
        let mut raw = Vec::new();
        raw.extend_from_slice(&(-1_i32).to_le_bytes());
        raw.extend_from_slice(&(layers.len() as i32).to_le_bytes());
        for layer in layers {
            assert_eq!(layer.len(), height);
            raw.extend_from_slice(&(width as i32).to_le_bytes());
            raw.extend_from_slice(&(height as i32).to_le_bytes());
            for x in 0..width {
                for row in layer {
                    assert_eq!(row.len(), width);
                    let (ch, fg, bg) = row[x];
                    raw.extend_from_slice(&(ch as i32).to_le_bytes());
                    raw.extend_from_slice(&[fg.r, fg.g, fg.b]);
                    let bg = bg.unwrap_or(REX_MAGENTA);
                    raw.extend_from_slice(&[bg.r, bg.g, bg.b]);
                }
            }
        }
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&raw).expect("compress XP fixture");
        encoder.finish().expect("finish XP fixture")
    }

    #[test]
    fn constructors_sanitize_and_validate() {
        let image = Image::new(
            2,
            2,
            Pixel {
                ch: '\n',
                fg: RED,
                bg: Some(BLUE),
            },
        );
        assert_eq!(image.bounds(), Rect::new(0, 0, 2, 2));
        assert_eq!(image.get(0, 0).unwrap().ch, '?');
        assert_eq!(image.get(1, 1).unwrap().bg, Some(BLUE));

        let error = Image::from_pixels(2, 2, vec![px('a'); 3]).unwrap_err();
        assert!(matches!(error, ImageError::Format(_)));
        let image = Image::from_pixels(
            2,
            1,
            vec![
                Pixel {
                    ch: '\x1b',
                    fg: RED,
                    bg: None,
                },
                px('b'),
            ],
        )
        .unwrap();
        assert_eq!(image.row_string(0).unwrap(), "?b");
        assert_eq!(Image::default(), Image::new(0, 0, px('x')));
    }

    #[test]
    fn from_text_pads_and_can_make_spaces_transparent() {
        let opaque = Image::from_text("ab\nc  d", RED, Some(BLUE), false);
        assert_eq!((opaque.width(), opaque.height()), (4, 2));
        assert_eq!(opaque.row_strings(), vec!["ab  ", "c  d"]);
        assert_eq!(opaque.get(2, 0).unwrap().ch, ' ');
        assert_eq!(opaque.get(2, 0).unwrap().bg, Some(BLUE));

        let transparent = Image::from_text("ab\nc  d", RED, Some(BLUE), true);
        assert_eq!(transparent.row_strings(), vec!["ab  ", "c  d"]);
        assert_eq!(transparent.get(2, 0).unwrap(), Pixel::transparent());
        assert_eq!(transparent.get(1, 1).unwrap(), Pixel::transparent());
        assert_eq!(transparent.get(3, 1).unwrap().ch, 'd');
        assert_eq!(transparent.get(3, 1).unwrap().fg, RED);
        assert_eq!(transparent.get(3, 1).unwrap().bg, Some(BLUE));
    }

    #[test]
    fn from_text_and_lines_handle_empty_and_trailing_newlines() {
        assert_eq!(Image::from_text("", RED, None, false), Image::default());
        let one = Image::from_text("a\n", RED, None, false);
        assert_eq!((one.width(), one.height()), (1, 1));
        let blank_line = Image::from_text("\n", RED, None, false);
        assert_eq!((blank_line.width(), blank_line.height()), (0, 1));
        assert_eq!(blank_line.row_strings(), vec![""]);
        let lines = Image::from_lines(["x", "yz"], RED, None, false);
        assert_eq!(lines.row_strings(), vec!["x ", "yz"]);
    }

    #[test]
    fn coordinates_pixels_rows_and_origin_are_exposed() {
        let mut image = Image::default().with_origin("sprite.txt");
        assert_eq!(image.origin(), Some("sprite.txt"));
        image = Image::from_pixels(2, 2, vec![px('a'), px('b'), px('c'), px('d')]).unwrap();
        assert_eq!(image.origin(), None);
        image.set_origin("replacement");
        assert_eq!(image.origin(), Some("replacement"));
        image.clear_origin();
        assert_eq!(image.origin(), None);

        assert_eq!(image.get(-1, 0), None);
        assert!(!image.set(2, 0, px('x')));
        assert!(image.set(1, 1, px('X')));
        assert_eq!(
            image.pixels().collect::<Vec<_>>(),
            vec![
                (0, 0, px('a')),
                (1, 0, px('b')),
                (0, 1, px('c')),
                (1, 1, px('X'))
            ]
        );
        assert_eq!(
            image.rows().map(|row| row.len()).collect::<Vec<_>>(),
            [2, 2]
        );
        assert_eq!(image.row_strings(), vec!["ab", "cX"]);
        assert_eq!(image.row_string(2), None);
        assert_eq!(image.to_text(), "ab\ncX");
    }

    #[test]
    fn crop_pads_with_transparency_and_negative_coordinates() {
        let image = Image::from_text("ab\ncd", RED, None, false);
        let cropped = image.crop(Rect::new(-1, -1, 4, 4));
        assert_eq!((cropped.width(), cropped.height()), (4, 4));
        assert_eq!(cropped.row_strings(), vec!["    ", " ab ", " cd ", "    "]);
        assert_eq!(cropped.get(0, 0).unwrap(), Pixel::transparent());
        assert_eq!(image.copy_rect(Rect::new(1, 1, 2, 1)).row_strings(), ["d "]);
    }

    #[test]
    fn pad_and_resize_nearest_behave_predictably() {
        let image = Image::from_text("ab\ncd", RED, None, false);
        let padded = image.pad(1, 2, 0, 1);
        assert_eq!((padded.width(), padded.height()), (3, 5));
        assert_eq!(padded.get(0, 0).unwrap(), Pixel::transparent());
        assert_eq!(
            padded.row_strings(),
            vec!["   ", "   ", " ab", " cd", "   "]
        );

        let resized = image.resize_nearest(4, 1);
        assert_eq!(resized.row_strings(), vec!["aabb"]);
        let empty = Image::default().resize_nearest(2, 2);
        assert_eq!(empty.pixels_slice(), [Pixel::transparent(); 4]);
    }

    #[test]
    fn split_is_row_major_and_pads_partial_frames() {
        let image = Image::from_text("abc\ndef\nghi", RED, None, false);
        let frames = image.split(2, 2);
        assert_eq!(frames.len(), 4);
        assert_eq!(frames[0].row_strings(), vec!["ab", "de"]);
        assert_eq!(frames[1].row_strings(), vec!["c ", "f "]);
        assert_eq!(frames[2].row_strings(), vec!["gh", "  "]);
        assert_eq!(frames[3].row_strings(), vec!["i ", "  "]);
        assert!(image.split(0, 2).is_empty());
    }

    #[test]
    fn flips_substitute_upstream_glyphs_and_preserve_colors() {
        let image = Image::from_text("(/<[{}>])dbpq", RED, Some(BLUE), false);
        let flipped = image.hor_flip();
        assert_eq!(flipped.row_string(0).unwrap(), "pqdb([<{}]>\\)");
        assert_eq!(flipped.get(0, 0).unwrap().fg, RED);
        assert_eq!(flipped.get(0, 0).unwrap().bg, Some(BLUE));

        let path = Image::from_text("a_bV/", RED, None, false);
        let vertical = path.ver_flip();
        assert_eq!(vertical.row_string(0).unwrap(), "e-p^\\");
        assert_eq!(
            Image::from_text("dbpq", RED, None, false)
                .ver_flip()
                .row_string(0)
                .unwrap(),
            "Ppbq"
        );
    }

    #[test]
    fn trim_and_trim_bg_remove_empty_borders() {
        let mut image = Image::new(5, 4, Pixel::new(' ', RED, Some(BLUE)));
        image.set(2, 1, px('x'));
        image.set(3, 2, px('y'));
        let trimmed = image.trim_bg(BLUE);
        assert_eq!((trimmed.width(), trimmed.height()), (2, 2));
        assert_eq!(trimmed.row_strings(), vec!["x ", " y"]);

        let mut transparent = Image::new(3, 3, Pixel::transparent());
        transparent.set(1, 1, px('z'));
        let trimmed = transparent.trim();
        assert_eq!((trimmed.width(), trimmed.height()), (1, 1));
        assert_eq!(trimmed.row_string(0).unwrap(), "z");
        assert_eq!(
            Image::new(2, 2, Pixel::transparent()).trim(),
            Image::default()
        );
    }

    #[test]
    fn replace_bg_only_changes_exterior_connected_pixels() {
        let bg = Color::rgb(1, 2, 3);
        let replacement = Color::rgb(9, 8, 7);
        let mut image = Image::new(5, 5, Pixel::new(' ', RED, Some(bg)));
        for x in 1..4 {
            for y in 1..4 {
                if x == 1 || y == 1 || x == 3 || y == 3 {
                    image.set(x, y, Pixel::new('#', GREEN, Some(bg)));
                }
            }
        }
        let changed = image.replace_bg(Some(bg), Some(replacement));
        assert_eq!(changed.get(0, 0).unwrap().bg, Some(replacement));
        assert_eq!(changed.get(2, 2).unwrap().bg, Some(bg));
        assert_eq!(changed.get(1, 1).unwrap().bg, Some(bg));
        assert_eq!(changed.get(1, 1).unwrap().ch, '#');

        let none_changed = image.replace_bg(None, Some(replacement));
        assert_eq!(none_changed, image);
    }

    #[test]
    fn stitch_combines_dimensions_with_background_fill() {
        let left = Image::from_text("ab", RED, None, false);
        let right = Image::from_text("c\nd", GREEN, None, false);
        let below = left.stitch_below(&right, px('.'));
        assert_eq!(below.row_strings(), vec!["ab", "c.", "d."]);
        let beside = left.stitch_right(&right, px('.'));
        assert_eq!(beside.row_strings(), vec!["abc", "..d"]);
    }

    #[test]
    fn draw_skips_transparency_and_clips() {
        let source = Image::from_text("x y", RED, None, true);
        let mut target = Image::new(3, 2, px('.'));
        target.draw(-1, 1, &source);
        assert_eq!(target.row_strings(), vec!["...", ".y."]);
        target.blit(2, 0, &source);
        assert_eq!(target.row_strings(), vec!["..x", ".y."]);
    }

    #[test]
    fn antialias_uses_directional_edge_glyphs() {
        let image = Image::from_text("x", RED, None, false);
        let result = image.antialias(" .#", 0.0);
        assert_eq!(result.row_string(0).unwrap(), "@");
        let solid = Image::from_text("xxx\nxxx\nxxx", RED, None, false);
        let smoothed = solid.antialias("", 0.0);
        assert_eq!(smoothed.get(1, 1), solid.get(1, 1));
        assert_eq!(smoothed.get(0, 0).unwrap().ch, 'd');
        assert_eq!(smoothed.get(1, 0).unwrap().ch, 'x');
    }

    #[test]
    fn txt_roundtrip_and_transparent_spaces() {
        let (directory, path) = temp_file("image.txt");
        let image = Image::from_text("ab\ncd", RED, Some(BLUE), false);
        image.save_txt(&path).unwrap();
        let loaded = Image::load_txt(&path).unwrap();
        assert_eq!(loaded.row_strings(), image.row_strings());
        assert_eq!(loaded.origin(), Some(path.to_string_lossy().as_ref()));

        let mut transparent = Image::from_text("x y", RED, None, true);
        transparent.set(1, 0, Pixel::transparent());
        let bytes = transparent.save_txt_bytes();
        assert_eq!(String::from_utf8(bytes.clone()).unwrap(), "x y\n");
        let loaded = Image::load_txt_bytes(&bytes).unwrap();
        assert_eq!(loaded.row_string(0).unwrap(), "x y");
        remove_temp(&directory);
    }

    #[test]
    fn csv_roundtrip_preserves_colors_and_transparency() {
        let (directory, path) = temp_file("IMAGE.CSV");
        let mut image = Image::from_text("ab\ncd", RED, None, false);
        image.set(1, 0, Pixel::new('b', GREEN, Some(BLUE)));
        image.set(0, 1, Pixel::transparent());
        image.set_origin("sprite");
        image.save_rexpaint_csv(&path).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("CosPlay image [2x2, origin=sprite, bg=0xFF00FF]"));
        assert!(text.contains("1,0,98,0x00FF00,0x0000FF"));
        assert!(text.contains("0,1,0,0x000000,0xFF00FF"));

        let loaded = Image::load(&path).unwrap();
        assert_eq!(loaded.row_strings(), image.row_strings());
        assert_eq!(loaded.get(1, 0), image.get(1, 0));
        assert_eq!(loaded.get(0, 1), Some(Pixel::transparent()));
        assert_eq!(loaded.get(1, 1).unwrap().bg, None);
        assert_eq!(loaded.origin(), Some(path.to_string_lossy().as_ref()));
        remove_temp(&directory);
    }

    #[test]
    fn csv_loader_accepts_unordered_sparse_rows_and_hex_char_codes() {
        let csv = b"CosPlay image [2x2, origin=code, bg=0xFF00FF]\n1,1,0x62,0x112233,0xFF00FF\n0,0,97,0x445566,0x778899\n";
        let image = Image::load_rexpaint_csv_bytes(csv).unwrap();
        assert_eq!((image.width(), image.height()), (2, 2));
        assert_eq!(image.row_strings(), vec!["a ", " b"]);
        assert_eq!(image.get(1, 1).unwrap().bg, None);
        assert_eq!(
            image.get(0, 0).unwrap().bg,
            Some(Color::rgb(0x77, 0x88, 0x99))
        );
    }

    #[test]
    fn malformed_csv_is_rejected() {
        for csv in [
            &b"header only"[..],
            b"0,0,97,0xFFFFFF\n",
            b"x,0,97,0xFFFFFF,0x000000\n",
            b"0,0,1114112,0xFFFFFF,0x000000\n",
            b"-1,0,97,0xFFFFFF,0x000000\n",
        ] {
            assert!(Image::load_rexpaint_csv_bytes(csv).is_err(), "{csv:?}");
        }
    }

    #[test]
    fn xp_roundtrip_preserves_transparency_and_is_gzip() {
        let (directory, path) = temp_file("image.XP");
        let mut image = Image::from_text("ab\ncd", RED, None, false);
        image.set(0, 0, Pixel::new('a', GREEN, Some(BLUE)));
        image.set(1, 1, Pixel::transparent());
        image.save_rexpaint_xp(&path).unwrap();
        let bytes = fs::read(&path).unwrap();
        assert_eq!(&bytes[..2], &[0x1f, 0x8b]);

        let loaded = Image::load_rexpaint_xp(&path).unwrap();
        assert_eq!(loaded, image);
        assert_eq!(loaded.origin(), Some(path.to_string_lossy().as_ref()));
        remove_temp(&directory);
    }

    #[test]
    fn xp_flattens_layers_with_later_opaque_cells_winning() {
        let layer0 = vec![
            vec![('a', RED, Some(BLUE)), ('b', RED, Some(BLUE))],
            vec![('c', RED, Some(BLUE)), ('d', RED, Some(BLUE))],
        ];
        let layer1 = vec![
            vec![('\0', GREEN, None), ('X', GREEN, None)],
            vec![(' ', GREEN, Some(BLUE)), ('\0', GREEN, None)],
        ];
        let bytes = xp_with_layers(&[layer0, layer1]);
        let image = Image::load_rexpaint_xp_bytes(&bytes).unwrap();
        assert_eq!(image.row_strings(), vec!["aX", " d"]);
        assert_eq!(image.get(1, 0).unwrap().fg, GREEN);
        assert_eq!(image.get(1, 0).unwrap().bg, None);
        assert_eq!(image.get(0, 1).unwrap().ch, ' ');
        assert_eq!(image.get(0, 1).unwrap().fg, GREEN);
    }

    #[test]
    fn malformed_xp_and_extensions_are_rejected() {
        assert!(matches!(
            Image::load_rexpaint_xp_bytes(b"not gzip"),
            Err(ImageError::Decompress)
        ));

        let truncated = {
            let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(&[0_u8; 8]).unwrap();
            encoder.finish().unwrap()
        };
        assert!(matches!(
            Image::load_rexpaint_xp_bytes(&truncated),
            Err(ImageError::Format(_))
        ));

        let invalid_width = {
            let mut raw = Vec::new();
            raw.extend_from_slice(&(-1_i32).to_le_bytes());
            raw.extend_from_slice(&1_i32.to_le_bytes());
            raw.extend_from_slice(&70_000_i32.to_le_bytes());
            raw.extend_from_slice(&1_i32.to_le_bytes());
            let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(&raw).unwrap();
            encoder.finish().unwrap()
        };
        assert!(matches!(
            Image::load_rexpaint_xp_bytes(&invalid_width),
            Err(ImageError::InvalidDimensions { .. })
        ));
        assert!(matches!(
            Image::load_txt_bytes(&[0xFF, 0xFE]),
            Err(ImageError::Format(_))
        ));

        let (directory, path) = temp_file("image.bmp");
        fs::write(&path, b"x").unwrap();
        assert!(matches!(Image::load(&path), Err(ImageError::Format(_))));
        assert!(matches!(
            Image::save(&path, &Image::default()),
            Err(ImageError::Format(_))
        ));
        assert!(Image::load(temporary_path_without_extension()).is_err());
        remove_temp(&directory);
    }

    fn temporary_path_without_extension() -> PathBuf {
        let (directory, path) = temp_file("no_extension");
        remove_temp(&directory);
        path
    }

    #[test]
    fn top_level_save_dispatches_case_insensitively() {
        let (directory, path) = temp_file("image.TxT");
        let image = Image::from_text("hi", RED, None, false);
        save(&path, &image).unwrap();
        assert_eq!(load(&path).unwrap().row_strings(), image.row_strings());
        remove_temp(&directory);
    }

    #[test]
    fn errors_display_and_wrap_io() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let error = ImageError::from(io);
        assert!(error.to_string().contains("I/O error"));
        assert!(error.source().is_some());
        assert_eq!(
            ImageError::InvalidDimensions {
                width: 2,
                height: 3
            }
            .to_string(),
            "invalid image dimensions: 2x3"
        );
        assert_eq!(
            ImageError::Decompress.to_string(),
            "failed to decompress image data"
        );
    }
}
