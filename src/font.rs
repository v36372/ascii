// SPDX-License-Identifier: Apache-2.0
//! Bitmap-style text fonts: FIGlet fonts and the one-row system font.
//!
//! A [`FigletFont`] parses the FIGfont 2 (`flf2a`) format and renders text as
//! an [`Image`] whose cells contain the FIGlet sub-characters. The bundled
//! registry exposes the 277 fonts under `assets/figlets` by name. [`SystemFont`]
//! is the one-cell-high font used for ordinary text.
//!
//! Rendering is intentionally independent of the terminal and the GPU
//! renderer. The returned image can be drawn on a canvas, stitched into a
//! sprite, or saved with [`Image`]'s existing helpers.

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use crate::graphics::{Color, Pixel};
use crate::image::{Image, ImageError};

include!("../assets/figlets/registry.rs");

/// Horizontal alignment for [`Font::render_lines`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Align {
    /// Lines start at the left edge.
    #[default]
    Left,
    /// Lines are centered; odd extra padding goes on the right.
    Center,
    /// Lines end at the right edge.
    Right,
}

impl Align {
    /// Splits `amount` padding cells into left and right padding.
    fn split(self, amount: usize) -> (usize, usize) {
        match self {
            Self::Left => (0, amount),
            Self::Center => (amount / 2, amount - amount / 2),
            Self::Right => (amount, 0),
        }
    }
}

/// An error produced while loading or rendering a font.
#[derive(Debug)]
pub enum FontError {
    /// The FIGfont header is missing required fields or contains invalid data.
    InvalidHeader(String),
    /// The font does not contain a glyph for the requested character.
    MissingGlyph(char),
    /// A FIGlet glyph line is malformed or the file ended in the middle of a
    /// glyph. `line` is one-based.
    InvalidGlyph {
        /// One-based physical line number in the FLF source.
        line: usize,
        /// Description of the problem.
        message: String,
    },
    /// The underlying image could not be created.
    Image(ImageError),
}

impl fmt::Display for FontError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHeader(message) => write!(f, "invalid FIGlet header: {message}"),
            Self::MissingGlyph(ch) => write!(f, "font has no glyph for {ch:?}"),
            Self::InvalidGlyph { line, message } => {
                write!(f, "invalid FIGlet glyph at line {line}: {message}")
            }
            Self::Image(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for FontError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Image(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ImageError> for FontError {
    fn from(error: ImageError) -> Self {
        Self::Image(error)
    }
}

/// A font that renders text into a character-cell [`Image`].
///
/// `render` renders a single line. Newlines are not glyphs; implementations
/// use the first line of `text` and callers that need multiple rows should use
/// [`Font::render_lines`].
pub trait Font {
    /// Human-readable font name.
    fn name(&self) -> &str;

    /// Height of one rendered text line in cells.
    fn height(&self) -> u16;

    /// Baseline row, measured from the top of a line.
    fn baseline(&self) -> u16;

    /// Whether this is the one-row system font.
    fn is_system(&self) -> bool;

    /// Renders the first line of `text`.
    fn render(&self, text: &str, fg: Color, bg: Option<Color>) -> Result<Image, FontError>;

    /// Renders one image row per input line, padding shorter lines according
    /// to `align`.
    fn render_lines(
        &self,
        lines: &[&str],
        fg: Color,
        bg: Option<Color>,
        align: Align,
    ) -> Result<Image, FontError>;
}

/// The built-in one-row system font.
///
/// Each input line becomes one image row. Characters are passed through
/// [`Pixel::new`], so the same glyph sanitizing rules as the rest of the crate
/// apply.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemFont;

impl SystemFont {
    /// Creates the system font.
    pub const fn new() -> Self {
        Self
    }

    /// Renders the first line of `text`.
    pub fn render(&self, text: &str, fg: Color, bg: Option<Color>) -> Result<Image, FontError> {
        self.render_lines(&[first_line(text)], fg, bg, Align::Left)
    }

    /// Renders one image row per input line.
    pub fn render_lines(
        &self,
        lines: &[&str],
        fg: Color,
        bg: Option<Color>,
        align: Align,
    ) -> Result<Image, FontError> {
        if lines.is_empty() {
            return Ok(Image::new(0, 0, Pixel::new(' ', fg, bg)));
        }

        let max_width = lines
            .iter()
            .map(|line| line.chars().count())
            .max()
            .unwrap_or(0);
        check_dimensions(max_width, lines.len())?;

        let mut rows = Vec::with_capacity(lines.len());
        for line in lines {
            let width = line.chars().count();
            let (left, right) = align.split(max_width - width);
            let mut row = String::with_capacity(max_width);
            row.extend(std::iter::repeat_n(' ', left));
            row.push_str(line);
            row.extend(std::iter::repeat_n(' ', right));
            rows.push(row);
        }
        rows_to_image(rows, fg, bg)
    }
}

impl Font for SystemFont {
    fn name(&self) -> &str {
        "System"
    }

    fn height(&self) -> u16 {
        1
    }

    fn baseline(&self) -> u16 {
        1
    }

    fn is_system(&self) -> bool {
        true
    }

    fn render(&self, text: &str, fg: Color, bg: Option<Color>) -> Result<Image, FontError> {
        SystemFont::render(self, text, fg, bg)
    }

    fn render_lines(
        &self,
        lines: &[&str],
        fg: Color,
        bg: Option<Color>,
        align: Align,
    ) -> Result<Image, FontError> {
        SystemFont::render_lines(self, lines, fg, bg, align)
    }
}

/// Horizontal fitting mode for a [`FigletFont`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fitting {
    /// Every FIGcharacter keeps its full designed width.
    FullWidth,
    /// FIGcharacters are moved together until they touch. Hardblanks still
    /// count as visible sub-characters.
    Kerning,
    /// Universal smushing: visible sub-characters may overlap, with the later
    /// character winning. Hardblanks never hide visible sub-characters.
    UniversalSmush,
    /// Controlled smushing using the six horizontal FIGlet rules.
    ControlledSmush,
}

/// A parsed FIGfont 2 font.
#[derive(Clone, Debug)]
pub struct FigletFont {
    name: String,
    height: u16,
    baseline: u16,
    hardblank: char,
    max_length: u16,
    comment_lines: Vec<String>,
    fitting: Fitting,
    glyphs: HashMap<char, Glyph>,
}

#[derive(Clone, Debug)]
struct Glyph {
    rows: Vec<String>,
    cells: Vec<Vec<char>>,
    width: usize,
}

#[derive(Clone, Copy, Debug)]
struct Header {
    hardblank: char,
    height: u16,
    baseline: u16,
    max_length: u16,
    old_layout: i32,
    comment_lines: usize,
    full_layout: Option<i32>,
}

const GERMAN_CHARS: [char; 7] = ['Ä', 'Ö', 'Ü', 'ä', 'ö', 'ü', 'ß'];
const REQUIRED_GLYPH_COUNT: usize = 102;

impl FigletFont {
    /// Parses an FLF font from bytes.
    ///
    /// The bytes are decoded as UTF-8 when possible. Otherwise each byte is
    /// treated as Latin-1 (the lossless `windows-1252`-compatible fallback
    /// required for older FIGlet fonts).
    pub fn from_flf(bytes: &[u8]) -> Result<Self, FontError> {
        let decoded;
        let text = match std::str::from_utf8(bytes) {
            Ok(text) => text,
            Err(_) => {
                decoded = bytes
                    .iter()
                    .map(|byte| char::from(*byte))
                    .collect::<String>();
                &decoded
            }
        };
        Self::from_str(text)
    }

    /// Parses an FLF font from text.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(text: &str) -> Result<Self, FontError> {
        let normalized = normalize_newlines(text);
        let text = normalized.strip_prefix('\u{feff}').unwrap_or(&normalized);
        let lines: Vec<&str> = text.split('\n').collect();
        let header_line = lines.first().copied().unwrap_or("");
        let header = parse_header(header_line)?;

        let comments_start = 1usize;
        let comments_end = comments_start
            .checked_add(header.comment_lines)
            .ok_or_else(|| {
                FontError::InvalidHeader("comment line count is too large".to_owned())
            })?;
        if comments_end > lines.len() {
            return Err(FontError::InvalidGlyph {
                line: lines.len().saturating_add(1),
                message: format!(
                    "unexpected end of file while reading {} comment lines",
                    header.comment_lines
                ),
            });
        }
        let comment_lines = lines[comments_start..comments_end]
            .iter()
            .map(|line| line.trim_end().to_owned())
            .collect();

        let mut glyphs = HashMap::with_capacity(REQUIRED_GLYPH_COUNT);
        let mut line_index = comments_end;
        for ch in required_chars() {
            let (glyph, next) = parse_glyph(&lines, line_index, header.height, ch)?;
            glyphs.insert(ch, glyph);
            line_index = next;
        }

        Ok(Self {
            name: "FIGlet".to_owned(),
            height: header.height,
            baseline: header.baseline,
            hardblank: header.hardblank,
            max_length: header.max_length,
            comment_lines,
            fitting: fitting_for_header(&header),
            glyphs,
        })
    }

    /// Loads and parses an FLF font from a path.
    ///
    /// The font name is taken from the file stem when it is valid UTF-8.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, FontError> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|error| FontError::Image(ImageError::Io(error)))?;
        let mut font = Self::from_flf(&bytes)?;
        if let Some(name) = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .filter(|stem| !stem.is_empty())
        {
            font.name = name.to_owned();
        }
        Ok(font)
    }

    /// Returns the current horizontal fitting mode.
    pub fn fitting(&self) -> Fitting {
        self.fitting
    }

    /// Replaces the horizontal fitting mode.
    pub fn with_fitting(mut self, fitting: Fitting) -> Self {
        self.fitting = fitting;
        self
    }

    /// Returns a copy using kerning.
    pub fn with_kerning(self) -> Self {
        self.with_fitting(Fitting::Kerning)
    }

    /// Returns a copy using full-width spacing.
    pub fn with_full_width(self) -> Self {
        self.with_fitting(Fitting::FullWidth)
    }

    /// Returns a copy using universal smushing.
    pub fn with_universal_smushing(self) -> Self {
        self.with_fitting(Fitting::UniversalSmush)
    }

    /// Returns a copy using controlled smushing with all six horizontal rules.
    pub fn with_control_smushing(self) -> Self {
        self.with_fitting(Fitting::ControlledSmush)
    }

    /// Renders the first line of `text`.
    pub fn render(&self, text: &str, fg: Color, bg: Option<Color>) -> Result<Image, FontError> {
        self.render_line(first_line(text), fg, bg)
    }

    /// Renders a single FIGlet line.
    pub fn render_line(
        &self,
        text: &str,
        fg: Color,
        bg: Option<Color>,
    ) -> Result<Image, FontError> {
        let rows = self.render_line_rows(text)?;
        rows_to_image(rows, fg, bg)
    }

    /// Renders one FIGlet line per input line, padding shorter lines according
    /// to `align`.
    pub fn render_lines(
        &self,
        lines: &[&str],
        fg: Color,
        bg: Option<Color>,
        align: Align,
    ) -> Result<Image, FontError> {
        render_figlet_lines(self, lines, fg, bg, align)
    }

    /// The hardblank character used by this font.
    pub fn hardblank(&self) -> char {
        self.hardblank
    }

    /// The header's maximum line length, in sub-characters.
    pub fn max_length(&self) -> u16 {
        self.max_length
    }

    /// Comment lines from immediately after the header.
    pub fn comment_lines(&self) -> &[String] {
        &self.comment_lines
    }

    /// Returns a glyph's rows without endmarks, or `None` when absent.
    pub fn glyph(&self, ch: char) -> Option<&[String]> {
        self.glyphs.get(&ch).map(|glyph| glyph.rows.as_slice())
    }

    fn render_line_rows(&self, text: &str) -> Result<Vec<String>, FontError> {
        let mut rows: Vec<Vec<char>> = Vec::new();
        let mut width = 0usize;
        let mut last_width = 0usize;
        let mut has_glyph = false;

        for ch in text.chars() {
            let glyph = self.glyphs.get(&ch).ok_or(FontError::MissingGlyph(ch))?;
            if !has_glyph {
                rows = glyph.cells.clone();
                width = glyph.width;
                last_width = glyph.width;
                has_glyph = true;
                continue;
            }

            let amount = self.overlap_amount(&rows, width, glyph, last_width);
            let start = width - amount;
            let new_width = start
                .checked_add(glyph.width)
                .ok_or_else(|| dimension_error(usize::MAX, usize::from(self.height)))?;
            check_width(new_width, usize::from(self.height))?;

            for row in &mut rows {
                row.resize(new_width, ' ');
            }
            for (y, glyph_row) in glyph.cells.iter().enumerate() {
                for (x, &right) in glyph_row.iter().enumerate() {
                    let position = start + x;
                    let left = rows[y][position];
                    rows[y][position] = self.merge_chars(left, right);
                }
            }
            width = new_width;
            last_width = glyph.width;
        }

        if !has_glyph {
            return Ok(vec![String::new(); usize::from(self.height)]);
        }

        Ok(rows
            .into_iter()
            .map(|row| {
                let rendered: String = row
                    .into_iter()
                    .map(|ch| if ch == self.hardblank { ' ' } else { ch })
                    .collect();
                rendered.trim_end_matches(' ').to_owned()
            })
            .collect())
    }

    fn overlap_amount(
        &self,
        rows: &[Vec<char>],
        width: usize,
        glyph: &Glyph,
        last_width: usize,
    ) -> usize {
        if self.fitting == Fitting::FullWidth {
            return 0;
        }
        let maximum = width.min(glyph.width);
        for amount in (0..=maximum).rev() {
            if self.can_overlap(rows, width, glyph, amount, last_width) {
                return amount;
            }
        }
        0
    }

    fn can_overlap(
        &self,
        rows: &[Vec<char>],
        width: usize,
        glyph: &Glyph,
        amount: usize,
        last_width: usize,
    ) -> bool {
        if amount == 0 {
            return true;
        }
        let start = width - amount;
        for (y, glyph_row) in glyph.cells.iter().enumerate() {
            for x in 0..amount {
                let left = rows[y][start + x];
                let right = glyph_row[x];
                if !self.can_merge(left, right, last_width, glyph.width) {
                    return false;
                }
            }
        }
        true
    }

    fn can_merge(&self, left: char, right: char, last_width: usize, current_width: usize) -> bool {
        if left == ' ' || right == ' ' {
            return true;
        }
        match self.fitting {
            Fitting::FullWidth | Fitting::Kerning => false,
            Fitting::UniversalSmush => last_width >= 2 && current_width >= 2,
            Fitting::ControlledSmush => {
                last_width >= 2
                    && current_width >= 2
                    && controlled_smush(left, right, self.hardblank).is_some()
            }
        }
    }

    fn merge_chars(&self, left: char, right: char) -> char {
        if left == ' ' {
            return right;
        }
        if right == ' ' {
            return left;
        }
        match self.fitting {
            Fitting::FullWidth | Fitting::Kerning => left,
            Fitting::UniversalSmush => {
                if left == self.hardblank {
                    right
                } else if right == self.hardblank {
                    left
                } else {
                    right
                }
            }
            Fitting::ControlledSmush => {
                controlled_smush(left, right, self.hardblank).unwrap_or(left)
            }
        }
    }
}

impl Font for FigletFont {
    fn name(&self) -> &str {
        &self.name
    }

    fn height(&self) -> u16 {
        self.height
    }

    fn baseline(&self) -> u16 {
        self.baseline
    }

    fn is_system(&self) -> bool {
        false
    }

    fn render(&self, text: &str, fg: Color, bg: Option<Color>) -> Result<Image, FontError> {
        FigletFont::render(self, text, fg, bg)
    }

    fn render_lines(
        &self,
        lines: &[&str],
        fg: Color,
        bg: Option<Color>,
        align: Align,
    ) -> Result<Image, FontError> {
        FigletFont::render_lines(self, lines, fg, bg, align)
    }
}

/// Returns the names of all bundled FIGlet fonts, sorted by name.
pub fn builtin_font_names() -> &'static [&'static str] {
    static NAMES: OnceLock<Vec<&'static str>> = OnceLock::new();
    NAMES
        .get_or_init(|| {
            BUILTIN_FIGLETS
                .iter()
                .map(|(name, _)| *name)
                .collect::<Vec<_>>()
        })
        .as_slice()
}

/// Returns the FLF bytes for a bundled font.
///
/// Lookup is ASCII-case-insensitive and accepts an optional `.flf` extension.
pub fn builtin_font(name: &str) -> Option<&'static [u8]> {
    let name = name.trim();
    let lower = name.to_ascii_lowercase();
    let stem = if lower.ends_with(".flf") {
        &name[..name.len() - 4]
    } else {
        name
    };
    BUILTIN_FIGLETS
        .iter()
        .find(|(candidate, _)| candidate.eq_ignore_ascii_case(stem))
        .map(|(_, bytes)| *bytes)
}

fn required_chars() -> impl Iterator<Item = char> {
    (b' '..=b'~').map(char::from).chain(GERMAN_CHARS)
}

fn parse_header(line: &str) -> Result<Header, FontError> {
    let line = line.trim_end();
    let rest = line.strip_prefix("flf2a").ok_or_else(|| {
        FontError::InvalidHeader(format!("missing \"flf2a\" signature in header {line:?}"))
    })?;
    let mut rest_chars = rest.chars();
    let hardblank = rest_chars
        .next()
        .ok_or_else(|| FontError::InvalidHeader("missing hardblank character".to_owned()))?;
    if hardblank == '\0' || hardblank.is_whitespace() {
        return Err(FontError::InvalidHeader(format!(
            "invalid hardblank character {hardblank:?}"
        )));
    }

    let fields: Vec<&str> = rest_chars.as_str().split_whitespace().collect();
    if fields.len() < 5 {
        return Err(FontError::InvalidHeader(format!(
            "expected at least 5 numeric fields after the signature, got {}",
            fields.len()
        )));
    }

    let height = parse_u16(fields[0], "height")?;
    if height == 0 {
        return Err(FontError::InvalidHeader(
            "height must be greater than zero".to_owned(),
        ));
    }
    let baseline = parse_u16(fields[1], "baseline")?;
    if baseline > height {
        return Err(FontError::InvalidHeader(format!(
            "baseline {baseline} exceeds height {height}"
        )));
    }
    let max_length = parse_u16(fields[2], "max length")?;
    if max_length == 0 {
        return Err(FontError::InvalidHeader(
            "max length must be greater than zero".to_owned(),
        ));
    }
    let old_layout = parse_i32(fields[3], "old layout")?;
    if !(-1..=63).contains(&old_layout) {
        return Err(FontError::InvalidHeader(format!(
            "old layout {old_layout} is outside -1..=63"
        )));
    }
    let comment_lines = fields[4].parse::<usize>().map_err(|_| {
        FontError::InvalidHeader(format!("invalid comment line count {:?}", fields[4]))
    })?;

    let print_direction = match fields.get(5) {
        Some(field) => parse_i32(field, "print direction")?,
        None => 0,
    };
    if print_direction != 0 {
        return Err(FontError::InvalidHeader(
            "right-to-left FIGlet fonts are not supported".to_owned(),
        ));
    }

    let full_layout = match fields.get(6) {
        Some(field) => Some(parse_i32(field, "full layout")?),
        None => None,
    };
    if let Some(field) = fields.get(7) {
        parse_i32(field, "code tag count")?;
    }

    Ok(Header {
        hardblank,
        height,
        baseline,
        max_length,
        old_layout,
        comment_lines,
        full_layout,
    })
}

fn parse_u16(field: &str, name: &str) -> Result<u16, FontError> {
    field
        .parse::<u16>()
        .map_err(|_| FontError::InvalidHeader(format!("invalid {name} {field:?}")))
}

fn parse_i32(field: &str, name: &str) -> Result<i32, FontError> {
    field
        .parse::<i32>()
        .map_err(|_| FontError::InvalidHeader(format!("invalid {name} {field:?}")))
}

fn parse_glyph(
    lines: &[&str],
    start: usize,
    height: u16,
    ch: char,
) -> Result<(Glyph, usize), FontError> {
    let height = usize::from(height);
    let end = start
        .checked_add(height)
        .ok_or_else(|| FontError::InvalidGlyph {
            line: start.saturating_add(1),
            message: format!("glyph height overflows for {ch:?}"),
        })?;
    if end > lines.len() {
        return Err(FontError::InvalidGlyph {
            line: lines.len().saturating_add(1),
            message: format!(
                "unexpected end of file while reading {height} lines for glyph {ch:?}"
            ),
        });
    }

    let first = lines[start].trim_end();
    let endmark = first
        .chars()
        .last()
        .ok_or_else(|| FontError::InvalidGlyph {
            line: start.saturating_add(1),
            message: format!("glyph {ch:?} starts with an empty line"),
        })?;
    if endmark == '\0' || endmark.is_whitespace() {
        return Err(FontError::InvalidGlyph {
            line: start.saturating_add(1),
            message: format!("glyph {ch:?} has invalid endmark {endmark:?}"),
        });
    }

    let mut cells: Vec<Vec<char>> = Vec::with_capacity(height);
    let mut width = 0usize;
    for row in 0..height {
        let physical_line = start + row;
        let line = lines[physical_line].trim_end();
        if line.is_empty() {
            return Err(FontError::InvalidGlyph {
                line: physical_line.saturating_add(1),
                message: format!("empty line in glyph {ch:?}"),
            });
        }
        let trailing = line
            .chars()
            .rev()
            .take_while(|candidate| *candidate == endmark)
            .count();
        if trailing == 0 {
            return Err(FontError::InvalidGlyph {
                line: physical_line.saturating_add(1),
                message: format!("glyph line for {ch:?} does not end with endmark {endmark:?}"),
            });
        }
        let remove = if row + 1 == height {
            trailing.min(2)
        } else {
            1
        };
        let mut body = line.to_owned();
        for _ in 0..remove {
            body.pop();
        }
        let row_cells: Vec<char> = body.chars().collect();
        width = width.max(row_cells.len());
        cells.push(row_cells);
    }

    if width > usize::from(u16::MAX) {
        return Err(FontError::InvalidGlyph {
            line: start.saturating_add(1),
            message: format!("glyph {ch:?} is wider than u16::MAX cells"),
        });
    }
    for row in &mut cells {
        row.resize(width, ' ');
    }
    let rows = cells
        .iter()
        .map(|row| row.iter().copied().collect::<String>())
        .collect();

    Ok((Glyph { rows, cells, width }, end))
}

fn fitting_for_header(header: &Header) -> Fitting {
    match header.full_layout {
        Some(full_layout) if full_layout > 0 => {
            if header.old_layout == -1 && full_layout & 64 == 0 && full_layout & 128 == 0 {
                Fitting::FullWidth
            } else if header.old_layout == 0 && full_layout & 64 != 0 && full_layout & 128 == 0 {
                Fitting::Kerning
            } else if header.old_layout == 0 && full_layout & 0x3f == 0 && full_layout & 128 != 0 {
                Fitting::UniversalSmush
            } else if header.old_layout > 0 && full_layout & 128 != 0 {
                Fitting::ControlledSmush
            } else {
                fallback_fitting(header.old_layout, full_layout)
            }
        }
        _ => fallback_fitting(header.old_layout, header.full_layout.unwrap_or(0)),
    }
}

fn fallback_fitting(old_layout: i32, full_layout: i32) -> Fitting {
    if old_layout == -1 {
        Fitting::FullWidth
    } else if old_layout == 0 {
        if full_layout & 128 != 0 {
            if full_layout & 0x3f != 0 {
                Fitting::ControlledSmush
            } else {
                Fitting::UniversalSmush
            }
        } else {
            Fitting::Kerning
        }
    } else if old_layout > 0 {
        Fitting::ControlledSmush
    } else if full_layout & 128 != 0 {
        if full_layout & 0x3f != 0 {
            Fitting::ControlledSmush
        } else {
            Fitting::UniversalSmush
        }
    } else if full_layout & 64 != 0 {
        Fitting::Kerning
    } else {
        Fitting::FullWidth
    }
}

fn controlled_smush(left: char, right: char, hardblank: char) -> Option<char> {
    if left == ' ' {
        return Some(right);
    }
    if right == ' ' {
        return Some(left);
    }

    // Rule 6: only two hardblanks may smush with each other.
    if left == hardblank && right == hardblank {
        return Some(hardblank);
    }
    if left == hardblank || right == hardblank {
        return None;
    }

    // Rule 1: equal characters.
    if left == right {
        return Some(left);
    }

    // Rule 2: an underscore is replaced by a border character.
    if left == '_' && is_border_char(right) {
        return Some(right);
    }
    if right == '_' && is_border_char(left) {
        return Some(left);
    }

    // Rule 3: hierarchy smushing.
    if let (Some(left_class), Some(right_class)) = (border_class(left), border_class(right)) {
        if left_class != right_class {
            return Some(if left_class > right_class {
                left
            } else {
                right
            });
        }
    }

    // Rule 4: opposing brackets become a vertical bar.
    if matches!(
        (left, right),
        ('[', ']') | (']', '[') | ('{', '}') | ('}', '{') | ('(', ')') | (')', '(')
    ) {
        return Some('|');
    }

    // Rule 5: the "big X" pairs.
    match (left, right) {
        ('/', '\\') => Some('|'),
        ('\\', '/') => Some('Y'),
        ('>', '<') => Some('X'),
        _ => None,
    }
}

fn is_border_char(ch: char) -> bool {
    border_class(ch).is_some()
}

fn border_class(ch: char) -> Option<u8> {
    match ch {
        '|' => Some(0),
        '/' | '\\' => Some(1),
        '[' | ']' => Some(2),
        '{' | '}' => Some(3),
        '(' | ')' => Some(4),
        '<' | '>' => Some(5),
        _ => None,
    }
}

fn render_figlet_lines(
    font: &FigletFont,
    lines: &[&str],
    fg: Color,
    bg: Option<Color>,
    align: Align,
) -> Result<Image, FontError> {
    if lines.is_empty() {
        return Ok(Image::new(0, 0, Pixel::new(' ', fg, bg)));
    }

    let mut rendered = Vec::with_capacity(lines.len());
    let mut max_width = 0usize;
    for line in lines {
        let rows = font.render_line_rows(line)?;
        let width = rows
            .iter()
            .map(|row| row.chars().count())
            .max()
            .unwrap_or(0);
        max_width = max_width.max(width);
        rendered.push(rows);
    }

    let total_height = lines
        .len()
        .checked_mul(usize::from(font.height))
        .ok_or_else(|| dimension_error(max_width, usize::MAX))?;
    check_dimensions(max_width, total_height)?;

    let mut all_rows = Vec::with_capacity(total_height);
    for rows in rendered {
        for row in rows {
            let width = row.chars().count();
            let (left, right) = align.split(max_width - width);
            let mut aligned = String::with_capacity(max_width);
            aligned.extend(std::iter::repeat_n(' ', left));
            aligned.push_str(&row);
            aligned.extend(std::iter::repeat_n(' ', right));
            all_rows.push(aligned);
        }
    }
    rows_to_image(all_rows, fg, bg)
}

fn rows_to_image(rows: Vec<String>, fg: Color, bg: Option<Color>) -> Result<Image, FontError> {
    let width = rows
        .iter()
        .map(|row| row.chars().count())
        .max()
        .unwrap_or(0);
    check_dimensions(width, rows.len())?;
    Ok(Image::from_lines(rows, fg, bg, false))
}

fn check_dimensions(width: usize, height: usize) -> Result<(), FontError> {
    if width > usize::from(u16::MAX) || height > usize::from(u16::MAX) {
        return Err(dimension_error(width, height));
    }
    Ok(())
}

fn check_width(width: usize, height: usize) -> Result<(), FontError> {
    if width > usize::from(u16::MAX) {
        return Err(dimension_error(width, height));
    }
    Ok(())
}

fn dimension_error(width: usize, height: usize) -> FontError {
    FontError::Image(ImageError::InvalidDimensions {
        width: u16::try_from(width).unwrap_or(u16::MAX),
        height: u16::try_from(height).unwrap_or(u16::MAX),
    })
}

fn first_line(text: &str) -> &str {
    text.split(['\n', '\r']).next().unwrap_or("")
}

fn normalize_newlines(text: &str) -> String {
    if !text.contains('\r') {
        return text.to_owned();
    }
    let mut normalized = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\r' {
            if chars.peek() == Some(&'\n') {
                chars.next();
            }
            normalized.push('\n');
        } else {
            normalized.push(ch);
        }
    }
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    const RED: Color = Color::rgb(255, 0, 0);
    const BLUE: Color = Color::rgb(0, 0, 255);

    fn standard() -> FigletFont {
        FigletFont::from_flf(builtin_font("Standard").expect("bundled Standard font"))
            .expect("Standard parses")
    }

    fn ogre() -> FigletFont {
        FigletFont::from_flf(builtin_font("Ogre").expect("bundled Ogre font")).expect("Ogre parses")
    }

    fn first_ink(image: &Image, row: usize) -> Option<usize> {
        let row = i32::try_from(row).ok()?;
        image.row(row).and_then(|pixels| {
            pixels
                .iter()
                .position(|pixel| !pixel.is_transparent() && pixel.ch != ' ')
        })
    }

    #[test]
    fn standard_header_and_glyphs_parse() {
        let font = standard();
        assert_eq!(font.height(), 6);
        assert_eq!(font.baseline(), 5);
        assert_eq!(font.hardblank(), '$');
        assert_eq!(font.max_length(), 16);
        assert_eq!(font.comment_lines().len(), 13);
        assert!(!font.is_system());
        assert_eq!(font.glyph('A').expect("A glyph").len(), 6);
        assert!(font.glyph('Ä').is_some());
        assert!(font.glyph('ß').is_some());
        assert!(font.glyph('\u{1f980}').is_none());
    }

    #[test]
    fn standard_and_ogre_render_nonempty_text() {
        for font in [standard(), ogre()] {
            let image = font.render("Hi", Color::WHITE, Some(BLUE)).unwrap();
            assert!(image.width() > 0, "{} rendered no width", font.name());
            assert_eq!(image.height(), font.height());
            assert!(image
                .row_strings()
                .iter()
                .any(|row| row.chars().any(|ch| ch != ' ')));
            assert!(!image.to_text().contains(font.hardblank()));
            assert_eq!(image.get(0, 0).unwrap().fg, Color::WHITE);
        }
    }

    #[test]
    fn fitting_modes_are_selectable_and_ordered() {
        let full = standard().with_full_width();
        let kern = standard().with_kerning();
        let controlled = standard().with_control_smushing();
        let universal = standard().with_universal_smushing();

        assert_eq!(full.fitting(), Fitting::FullWidth);
        assert_eq!(kern.fitting(), Fitting::Kerning);
        assert_eq!(controlled.fitting(), Fitting::ControlledSmush);
        assert_eq!(universal.fitting(), Fitting::UniversalSmush);
        assert_eq!(
            standard().with_fitting(Fitting::Kerning).fitting(),
            Fitting::Kerning
        );

        let full_width = full.render_line("AV", RED, None).unwrap().width();
        let kern_width = kern.render_line("AV", RED, None).unwrap().width();
        let controlled_width = controlled.render_line("AV", RED, None).unwrap().width();
        let universal_width = universal.render_line("AV", RED, None).unwrap().width();
        assert!(full_width >= kern_width);
        assert!(kern_width >= controlled_width);
        assert!(controlled_width >= universal_width);
    }

    #[test]
    fn hardblanks_are_rendered_as_spaces() {
        let font = standard();
        assert!(
            font.glyph(' ')
                .expect("space glyph")
                .iter()
                .any(|row| row.contains('$')),
            "Standard's space glyph should use hardblanks"
        );
        let image = font.render("A A", Color::WHITE, None).unwrap();
        assert!(!image.to_text().contains('$'));
        assert!(!image.to_text().contains(font.hardblank()));
    }

    #[test]
    fn multiline_rendering_aligns_to_the_widest_line() {
        let font = standard();
        let left = font
            .render_lines(&["I", "WW"], Color::WHITE, Some(BLUE), Align::Left)
            .unwrap();
        let center = font
            .render_lines(&["I", "WW"], Color::WHITE, Some(BLUE), Align::Center)
            .unwrap();
        let right = font
            .render_lines(&["I", "WW"], Color::WHITE, Some(BLUE), Align::Right)
            .unwrap();

        assert_eq!(left.width(), right.width());
        assert_eq!(left.width(), center.width());
        assert_eq!(left.height(), font.height() * 2);
        assert_eq!(left.get(0, 0).unwrap().bg, Some(BLUE));

        let left_start = first_ink(&left, 0).unwrap();
        let center_start = first_ink(&center, 0).unwrap();
        let right_start = first_ink(&right, 0).unwrap();
        assert!(left_start <= center_start);
        assert!(center_start <= right_start);
        assert!(left_start < right_start);
    }

    #[test]
    fn system_font_is_one_row_and_handles_multiline_alignment() {
        let font = SystemFont::new();
        assert_eq!(font.name(), "System");
        assert_eq!(font.height(), 1);
        assert_eq!(font.baseline(), 1);
        assert!(font.is_system());

        let single = font.render("abc", RED, Some(BLUE)).unwrap();
        assert_eq!((single.width(), single.height()), (3, 1));
        assert_eq!(single.row_string(0).unwrap(), "abc");
        assert_eq!(single.get(1, 0).unwrap().fg, RED);
        assert_eq!(single.get(1, 0).unwrap().bg, Some(BLUE));

        let first = font.render("one\ntwo", RED, None).unwrap();
        assert_eq!(first.row_string(0).unwrap(), "one");

        let right = font
            .render_lines(&["a", "bbb"], Color::WHITE, None, Align::Right)
            .unwrap();
        assert_eq!((right.width(), right.height()), (3, 2));
        assert_eq!(right.row_strings(), vec!["  a", "bbb"]);

        let center = font
            .render_lines(&["a", "bbb"], Color::WHITE, None, Align::Center)
            .unwrap();
        assert_eq!(center.row_strings(), vec![" a ", "bbb"]);

        let empty = font
            .render_lines(&[], Color::WHITE, None, Align::Left)
            .unwrap();
        assert_eq!((empty.width(), empty.height()), (0, 0));
    }

    #[test]
    fn malformed_headers_and_truncated_glyphs_are_errors() {
        assert!(matches!(
            FigletFont::from_flf(b"not a FIGlet font"),
            Err(FontError::InvalidHeader(_))
        ));
        assert!(matches!(
            FigletFont::from_flf(b"flf2a$ 6 5 20 15"),
            Err(FontError::InvalidHeader(_))
        ));
        assert!(matches!(
            FigletFont::from_flf(b"flf2a$ 6 5 20 15 0\n"),
            Err(FontError::InvalidGlyph { .. })
        ));
        assert!(matches!(
            FigletFont::from_flf(b"flf2a$ 0 0 20 15 0\n"),
            Err(FontError::InvalidHeader(_))
        ));
    }

    #[test]
    fn latin1_fallback_accepts_non_utf8_comments() {
        let mut bytes = builtin_font("Standard")
            .expect("bundled Standard font")
            .to_vec();
        let header_end = bytes
            .iter()
            .position(|byte| *byte == b'\n')
            .expect("header newline");
        let comment_start = header_end + 1;
        let comment_end = bytes[comment_start..]
            .iter()
            .position(|byte| *byte == b'\r' || *byte == b'\n')
            .map(|offset| comment_start + offset)
            .expect("comment newline");
        bytes.insert(comment_end, 0xE4);
        assert!(FigletFont::from_flf(&bytes).is_ok());
    }

    #[test]
    fn controlled_smushing_implements_all_six_rules() {
        let hardblank = '$';

        // Rule 1: equal characters.
        assert_eq!(controlled_smush('|', '|', hardblank), Some('|'));
        // Rule 2: an underscore is replaced by a border character.
        assert_eq!(controlled_smush('_', '|', hardblank), Some('|'));
        assert_eq!(controlled_smush('[', '_', hardblank), Some('['));
        // Rule 3: the latter hierarchy class wins.
        assert_eq!(controlled_smush('|', '/', hardblank), Some('/'));
        assert_eq!(controlled_smush('{', '(', hardblank), Some('('));
        // Rule 4: opposing brackets become a vertical bar.
        assert_eq!(controlled_smush('[', ']', hardblank), Some('|'));
        assert_eq!(controlled_smush(')', '(', hardblank), Some('|'));
        // Rule 5: big X pairs.
        assert_eq!(controlled_smush('/', '\\', hardblank), Some('|'));
        assert_eq!(controlled_smush('\\', '/', hardblank), Some('Y'));
        assert_eq!(controlled_smush('>', '<', hardblank), Some('X'));
        assert_eq!(controlled_smush('<', '>', hardblank), None);
        // Rule 6: only hardblanks smush with each other.
        assert_eq!(controlled_smush('$', '$', hardblank), Some('$'));
        assert_eq!(controlled_smush('$', '|', hardblank), None);
        assert_eq!(controlled_smush('|', '$', hardblank), None);
        // Characters outside the rules do not smush.
        assert_eq!(controlled_smush('A', 'B', hardblank), None);
    }

    #[test]
    fn universal_smushing_lets_visible_characters_override_hardblanks() {
        let font = standard().with_universal_smushing();
        assert_eq!(font.merge_chars('$', '|'), '|');
        assert_eq!(font.merge_chars('|', '$'), '|');
        assert_eq!(font.merge_chars('$', '$'), '$');
        assert_eq!(font.merge_chars('A', 'B'), 'B');
        assert_eq!(font.merge_chars(' ', 'B'), 'B');
        assert_eq!(font.merge_chars('A', ' '), 'A');
    }

    #[test]
    fn builtin_registry_is_complete_sorted_and_unique() {
        let names = builtin_font_names();
        assert_eq!(names.len(), 277, "expected all 277 bundled fonts");
        let unique: HashSet<&str> = names.iter().copied().collect();
        assert_eq!(unique.len(), names.len());
        assert!(names.windows(2).all(|pair| pair[0] <= pair[1]));

        assert!(builtin_font("Standard").is_some());
        assert!(builtin_font("standard.flf").is_some());
        assert!(builtin_font("STANDARD.FLF").is_some());
        assert!(builtin_font("Ogre").is_some());
        assert!(builtin_font("not-a-font").is_none());
    }

    #[test]
    fn all_bundled_fonts_parse_and_render() {
        for name in builtin_font_names() {
            let bytes = builtin_font(name).expect("registry lookup");
            let font = FigletFont::from_flf(bytes)
                .unwrap_or_else(|error| panic!("font {name:?} did not parse: {error}"));
            assert!(font.height() > 0, "font {name:?} has zero height");
            assert!(font.glyph('A').is_some(), "font {name:?} has no A glyph");
            let image = font
                .render_line("A", Color::WHITE, None)
                .unwrap_or_else(|error| panic!("font {name:?} did not render: {error}"));
            assert!(image.width() > 0, "font {name:?} rendered an empty A glyph");
        }
    }

    #[test]
    fn load_uses_the_file_stem_as_the_name() {
        let font = FigletFont::load("assets/figlets/Standard.flf").unwrap();
        assert_eq!(font.name(), "Standard");
        assert_eq!(font.height(), 6);
    }
}
