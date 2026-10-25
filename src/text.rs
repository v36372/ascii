// SPDX-License-Identifier: Apache-2.0
//! Text composition utilities: markup skins, styled strings, and ASCII tables.
//!
//! The utilities in this module are deliberately independent of the renderer:
//! they produce [`Pixel`] values or strings that can be handed to an
//! [`Image`], a canvas, or a terminal writer.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::graphics::{Color, Pixel};
use crate::image::Image;

/// A cloneable character-to-pixel converter used by markup elements.
///
/// Skins are stored behind an [`Arc`], so cloning a [`Markup`] is cheap and
/// does not clone the closure itself. The closure must be thread-safe, which
/// keeps markup values usable from multiple threads.
#[derive(Clone)]
pub struct Skin(Arc<dyn Fn(char) -> Pixel + Send + Sync>);

impl Skin {
    /// Creates a skin from a character-to-pixel function.
    pub fn new<F>(skin: F) -> Self
    where
        F: Fn(char) -> Pixel + Send + Sync + 'static,
    {
        Self(Arc::new(skin))
    }

    /// Creates a skin that renders every character with the given colors.
    pub fn solid<B>(fg: Color, bg: B) -> Self
    where
        B: Into<Option<Color>>,
    {
        let bg = bg.into();
        Self::new(move |ch| Pixel::new(ch, fg, bg))
    }

    /// Applies the skin to one character.
    pub fn apply(&self, ch: char) -> Pixel {
        (self.0)(ch)
    }
}

impl fmt::Debug for Skin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Skin(..)")
    }
}

impl<F> From<F> for Skin
where
    F: Fn(char) -> Pixel + Send + Sync + 'static,
{
    fn from(skin: F) -> Self {
        Self::new(skin)
    }
}

/// One tagged region in a [`Markup`].
///
/// Text between `open_tag` and `close_tag` is converted to pixels with `skin`.
/// Tags are removed from the output and may be nested.
#[derive(Clone, Debug)]
pub struct MarkupElement {
    /// Opening tag, for example `"<red>"`.
    pub open_tag: String,
    /// Closing tag, for example `"</red>"`.
    pub close_tag: String,
    /// Character-to-pixel converter for this region.
    pub skin: Skin,
}

impl MarkupElement {
    /// Creates a markup element from an opening tag, a closing tag, and a
    /// character-to-pixel function (or an existing [`Skin`]).
    pub fn new<O, C, S>(open_tag: O, close_tag: C, skin: S) -> Self
    where
        O: Into<String>,
        C: Into<String>,
        S: Into<Skin>,
    {
        Self {
            open_tag: open_tag.into(),
            close_tag: close_tag.into(),
            skin: skin.into(),
        }
    }

    /// Creates an element whose skin uses the given foreground and background
    /// colors for every character in the region.
    pub fn colored<O, C, B>(open_tag: O, close_tag: C, fg: Color, bg: B) -> Self
    where
        O: Into<String>,
        C: Into<String>,
        B: Into<Option<Color>>,
    {
        Self::new(open_tag, close_tag, Skin::solid(fg, bg))
    }
}

/// Errors reported while validating or processing a [`Markup`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MarkupError {
    /// An opening or closing tag is empty.
    EmptyTag {
        /// Index of the offending element.
        element: usize,
        /// Either `"opening"` or `"closing"`.
        which: &'static str,
    },
    /// A tag contains a literal space.
    TagContainsSpace {
        /// Index of the offending element.
        element: usize,
        /// The offending tag.
        tag: String,
    },
    /// The same opening or closing tag is used more than once.
    DuplicateTag {
        /// The duplicated tag.
        tag: String,
    },
    /// One tag contains another tag used by a different element.
    IntersectingTags {
        /// First tag involved in the intersection.
        first: String,
        /// Second tag involved in the intersection.
        second: String,
    },
    /// A closing tag was found with no corresponding opening tag.
    UnmatchedCloseTag {
        /// The unmatched closing tag.
        tag: String,
    },
    /// A closing tag did not match the most recently opened element.
    MismatchedCloseTag {
        /// Closing tag expected from the innermost open element.
        expected: String,
        /// Closing tag actually found.
        found: String,
    },
    /// Input ended while an element was still open.
    UnclosedTag {
        /// Opening tag that was never closed.
        tag: String,
    },
    /// The image dimensions or pixel buffer could not be created.
    Image(String),
}

impl fmt::Display for MarkupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTag { element, which } => {
                write!(f, "markup element {element} has an empty {which} tag")
            }
            Self::TagContainsSpace { element, tag } => {
                write!(f, "markup element {element} tag contains a space: {tag:?}")
            }
            Self::DuplicateTag { tag } => write!(f, "duplicate markup tag: {tag:?}"),
            Self::IntersectingTags { first, second } => {
                write!(f, "intersecting markup tags: {first:?} and {second:?}")
            }
            Self::UnmatchedCloseTag { tag } => {
                write!(f, "unmatched markup closing tag: {tag:?}")
            }
            Self::MismatchedCloseTag { expected, found } => {
                write!(
                    f,
                    "mismatched markup closing tag: expected {expected:?}, found {found:?}"
                )
            }
            Self::UnclosedTag { tag } => write!(f, "unclosed markup tag: {tag:?}"),
            Self::Image(message) => write!(f, "could not create text image: {message}"),
        }
    }
}

impl std::error::Error for MarkupError {}

/// A set of tagged character skins plus default colors.
///
/// Markup elements may be nested. Tags are matched exactly, in the order in
/// which elements were supplied; text belonging to a tag is never emitted.
///
/// ```
/// use cosplay::{graphics::{Color, Pixel}, text::{Markup, MarkupElement}};
///
/// let red = Color::rgb(255, 0, 0);
/// let markup = Markup::new(
///     Color::WHITE,
///     None,
///     vec![MarkupElement::colored("<r>", "</r>", red, None)],
/// )
/// .unwrap();
/// let pixels = markup.process("a<r>b</r>c").unwrap();
/// assert_eq!(pixels.iter().map(|p| p.ch).collect::<String>(), "abc");
/// assert_eq!(pixels[1].fg, red);
/// ```
#[derive(Clone, Debug)]
pub struct Markup {
    /// Default foreground color.
    pub fg: Color,
    /// Default optional background color.
    pub bg: Option<Color>,
    /// Tagged skins. The list may be empty.
    pub elements: Vec<MarkupElement>,
}

impl Markup {
    /// Creates and validates a markup specification.
    pub fn new<I, B>(fg: Color, bg: B, elements: I) -> Result<Self, MarkupError>
    where
        I: IntoIterator<Item = MarkupElement>,
        B: Into<Option<Color>>,
    {
        let markup = Self {
            fg,
            bg: bg.into(),
            elements: elements.into_iter().collect(),
        };
        markup.validate()?;
        Ok(markup)
    }

    /// Alias for [`Markup::new`].
    pub fn from_elements<I, B>(fg: Color, bg: B, elements: I) -> Result<Self, MarkupError>
    where
        I: IntoIterator<Item = MarkupElement>,
        B: Into<Option<Color>>,
    {
        Self::new(fg, bg, elements)
    }

    /// Creates an empty markup specification using only the default colors.
    pub fn empty<B>(fg: Color, bg: B) -> Self
    where
        B: Into<Option<Color>>,
    {
        Self {
            fg,
            bg: bg.into(),
            elements: Vec::new(),
        }
    }

    /// Creates a markup specification with one element.
    pub fn with_element<B>(fg: Color, bg: B, element: MarkupElement) -> Result<Self, MarkupError>
    where
        B: Into<Option<Color>>,
    {
        Self::new(fg, bg, vec![element])
    }

    /// Adds one element, validating the resulting specification.
    ///
    /// The element is not added when validation fails.
    pub fn push(&mut self, element: MarkupElement) -> Result<(), MarkupError> {
        let mut elements = self.elements.clone();
        elements.push(element);
        let candidate = Self {
            fg: self.fg,
            bg: self.bg,
            elements,
        };
        candidate.validate()?;
        self.elements = candidate.elements;
        Ok(())
    }

    /// Validates tags against the rules of the CosPlay markup format.
    ///
    /// Retained tags must be non-empty, contain no spaces, be unique, and no
    /// tag may contain a tag belonging to a different element.
    pub fn validate(&self) -> Result<(), MarkupError> {
        for (element, item) in self.elements.iter().enumerate() {
            if item.open_tag.is_empty() {
                return Err(MarkupError::EmptyTag {
                    element,
                    which: "opening",
                });
            }
            if item.close_tag.is_empty() {
                return Err(MarkupError::EmptyTag {
                    element,
                    which: "closing",
                });
            }
            if item.open_tag.contains(' ') {
                return Err(MarkupError::TagContainsSpace {
                    element,
                    tag: item.open_tag.clone(),
                });
            }
            if item.close_tag.contains(' ') {
                return Err(MarkupError::TagContainsSpace {
                    element,
                    tag: item.close_tag.clone(),
                });
            }
        }

        let mut seen = HashSet::new();
        for item in &self.elements {
            for tag in [&item.open_tag, &item.close_tag] {
                if !seen.insert(tag.as_str()) {
                    return Err(MarkupError::DuplicateTag { tag: tag.clone() });
                }
            }
        }

        for (index, first) in self.elements.iter().enumerate() {
            for second in &self.elements[index + 1..] {
                for first_tag in [&first.open_tag, &first.close_tag] {
                    for second_tag in [&second.open_tag, &second.close_tag] {
                        if first_tag.contains(second_tag.as_str()) {
                            return Err(MarkupError::IntersectingTags {
                                first: first_tag.clone(),
                                second: second_tag.clone(),
                            });
                        }
                        if second_tag.contains(first_tag.as_str()) {
                            return Err(MarkupError::IntersectingTags {
                                first: second_tag.clone(),
                                second: first_tag.clone(),
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Converts input text to pixels, removing markup tags.
    ///
    /// Newlines are treated like ordinary characters by this method; use
    /// [`Markup::to_image`] when line breaks should produce separate rows.
    pub fn process(&self, input: &str) -> Result<Vec<Pixel>, MarkupError> {
        self.validate()?;
        let mut pixels = Vec::new();
        self.walk(input, |pixel| match pixel {
            Some(pixel) => pixels.push(pixel),
            None => pixels.push(Pixel::new('\n', self.fg, self.bg)),
        })?;
        Ok(pixels)
    }

    /// Converts input text to an image, padding shorter rows according to
    /// `align`.
    ///
    /// # Panics
    ///
    /// Panics when the markup or input is invalid, or when the image cannot be
    /// created. Use [`Markup::try_to_image`] to handle those errors.
    pub fn to_image<A>(&self, input: &str, align: A) -> Image
    where
        A: Into<Align>,
    {
        match self.try_to_image(input, align.into()) {
            Ok(image) => image,
            Err(error) => panic!("could not render markup image: {error}"),
        }
    }

    /// Fallible variant of [`Markup::to_image`].
    pub fn try_to_image<A>(&self, input: &str, align: A) -> Result<Image, MarkupError>
    where
        A: Into<Align>,
    {
        let align = align.into();
        self.validate()?;

        let mut lines = vec![Vec::new()];
        self.walk(input, |pixel| match pixel {
            Some(pixel) => {
                if let Some(line) = lines.last_mut() {
                    line.push(pixel);
                }
            }
            None => lines.push(Vec::new()),
        })?;

        let width = lines.iter().map(Vec::len).max().unwrap_or(0);
        let space = Pixel::new(' ', self.fg, self.bg);
        let mut pixels = Vec::new();
        for line in &lines {
            let padding = width.saturating_sub(line.len());
            let (left, right) = align.split(padding);
            pixels.extend(std::iter::repeat_n(space, left));
            pixels.extend_from_slice(line);
            pixels.extend(std::iter::repeat_n(space, right));
        }

        let width = u16::try_from(width)
            .map_err(|_| MarkupError::Image("markup image is wider than u16::MAX".to_string()))?;
        let height = u16::try_from(lines.len())
            .map_err(|_| MarkupError::Image("markup image is taller than u16::MAX".to_string()))?;

        if width == 0 {
            return Ok(Image::new(0, 0, space));
        }

        Image::from_pixels(width, height, pixels)
            .map_err(|error| MarkupError::Image(format!("{error:?}")))
    }

    /// Shared tag-aware walker. A `None` event denotes a line break; a `Some`
    /// event contains the pixel produced for an ordinary character.
    fn walk<F>(&self, input: &str, mut emit: F) -> Result<(), MarkupError>
    where
        F: FnMut(Option<Pixel>),
    {
        let mut stack: Vec<usize> = Vec::new();
        let mut offset = 0;

        while offset < input.len() {
            if let Some((element, is_open, tag)) = self.match_tag(input, offset) {
                if is_open {
                    stack.push(element);
                } else {
                    let Some(&innermost) = stack.last() else {
                        return Err(MarkupError::UnmatchedCloseTag {
                            tag: tag.to_string(),
                        });
                    };
                    let expected = &self.elements[innermost].close_tag;
                    if expected != tag {
                        return Err(MarkupError::MismatchedCloseTag {
                            expected: expected.clone(),
                            found: tag.to_string(),
                        });
                    }
                    stack.pop();
                }
                offset += tag.len();
                continue;
            }

            let Some(ch) = input[offset..].chars().next() else {
                break;
            };
            offset += ch.len_utf8();
            if ch == '\n' {
                emit(None);
            } else {
                emit(Some(self.pixel_for(stack.last().copied(), ch)));
            }
        }

        if let Some(&innermost) = stack.last() {
            return Err(MarkupError::UnclosedTag {
                tag: self.elements[innermost].open_tag.clone(),
            });
        }
        Ok(())
    }

    fn pixel_for(&self, element: Option<usize>, ch: char) -> Pixel {
        match element {
            Some(index) => self.elements[index].skin.apply(ch),
            None => Pixel::new(ch, self.fg, self.bg),
        }
    }

    fn match_tag<'a>(&'a self, input: &str, offset: usize) -> Option<(usize, bool, &'a str)> {
        for (index, element) in self.elements.iter().enumerate() {
            if input[offset..].starts_with(&element.open_tag) {
                return Some((index, true, element.open_tag.as_str()));
            }
            if input[offset..].starts_with(&element.close_tag) {
                return Some((index, false, element.close_tag.as_str()));
            }
        }
        None
    }
}

/// A builder for a single row of colored text.
///
/// [`StyledString`] is useful for constructing labels and HUD fragments whose
/// characters carry different foreground or background colors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StyledString {
    fg: Color,
    bg: Option<Color>,
    pixels: Vec<Pixel>,
}

impl Default for StyledString {
    fn default() -> Self {
        Self::new(Color::WHITE, None)
    }
}

impl StyledString {
    /// Creates an empty styled string with default foreground and background
    /// colors. The background may be a [`Color`] or an `Option<Color>`.
    pub fn new<B>(fg: Color, bg: B) -> Self
    where
        B: Into<Option<Color>>,
    {
        Self {
            fg,
            bg: bg.into(),
            pixels: Vec::new(),
        }
    }

    /// Current default foreground color.
    pub fn fg(&self) -> Color {
        self.fg
    }

    /// Current default background color.
    pub fn bg(&self) -> Option<Color> {
        self.bg
    }

    /// Replaces the default foreground color.
    pub fn set_fg(&mut self, fg: Color) -> &mut Self {
        self.fg = fg;
        self
    }

    /// Replaces the default background color.
    pub fn set_bg<B>(&mut self, bg: B) -> &mut Self
    where
        B: Into<Option<Color>>,
    {
        self.bg = bg.into();
        self
    }

    /// Appends text using `fg` and the current background color.
    pub fn push_str(&mut self, text: &str, fg: Color) -> &mut Self {
        self.push_text(text, fg, self.bg)
    }

    /// Appends text using an explicit foreground and background color.
    pub fn push_text<B>(&mut self, text: &str, fg: Color, bg: B) -> &mut Self
    where
        B: Into<Option<Color>>,
    {
        let bg = bg.into();
        self.pixels
            .extend(text.chars().map(|ch| Pixel::new(ch, fg, bg)));
        self
    }

    /// Appends a single character with explicit colors.
    pub fn push_char<B>(&mut self, ch: char, fg: Color, bg: B) -> &mut Self
    where
        B: Into<Option<Color>>,
    {
        self.pixels.push(Pixel::new(ch, fg, bg.into()));
        self
    }

    /// Appends an already constructed pixel.
    pub fn push_pixel(&mut self, pixel: Pixel) -> &mut Self {
        self.pixels.push(pixel);
        self
    }

    /// Appends a slice of pixels.
    pub fn push_pixels(&mut self, pixels: &[Pixel]) -> &mut Self {
        self.pixels.extend_from_slice(pixels);
        self
    }

    /// Removes all pixels, retaining the current colors.
    pub fn clear(&mut self) -> &mut Self {
        self.pixels.clear();
        self
    }

    /// Number of character cells in the string.
    pub fn width(&self) -> usize {
        self.pixels.len()
    }

    /// Number of character cells in the string.
    pub fn len(&self) -> usize {
        self.pixels.len()
    }

    /// Whether the string contains no pixels.
    pub fn is_empty(&self) -> bool {
        self.pixels.is_empty()
    }

    /// Returns the built pixels without cloning.
    pub fn into_pixels(self) -> Vec<Pixel> {
        self.pixels
    }

    /// Returns a copy of the built pixels.
    pub fn build(&self) -> Vec<Pixel> {
        self.pixels.clone()
    }

    /// Returns the built pixels by reference.
    pub fn pixels(&self) -> &[Pixel] {
        &self.pixels
    }

    /// Builds a row from `(color, text)` pairs using no background color.
    pub fn styled(parts: &[(Color, &str)]) -> Vec<Pixel> {
        styled(parts)
    }

    /// Builds a row from one plain text fragment.
    pub fn style_str(text: &str, fg: Color) -> Vec<Pixel> {
        styled(&[(fg, text)])
    }

    /// Builds a row from one text fragment with a background color.
    pub fn style_str_bg<B>(text: &str, fg: Color, bg: B) -> Vec<Pixel>
    where
        B: Into<Option<Color>>,
    {
        let bg = bg.into();
        text.chars().map(|ch| Pixel::new(ch, fg, bg)).collect()
    }
}

/// Builds a single-row pixel buffer from `(foreground, text)` pairs.
///
/// ```
/// use cosplay::{graphics::Color, text::styled};
///
/// let pixels = styled(&[(Color::rgb(255, 0, 0), "red"), (Color::WHITE, "!")]);
/// assert_eq!(pixels.iter().map(|p| p.ch).collect::<String>(), "red!");
/// ```
pub fn styled(parts: &[(Color, &str)]) -> Vec<Pixel> {
    let mut pixels = Vec::new();
    for &(fg, text) in parts {
        pixels.extend(text.chars().map(|ch| Pixel::new(ch, fg, None)));
    }
    pixels
}

/// Horizontal alignment for table cells and text images.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Align {
    /// Content starts at the left edge.
    #[default]
    Left,
    /// Content is centered with any odd extra cell placed on the right.
    Center,
    /// Content ends at the right edge.
    Right,
}

impl Align {
    /// Parses `left`, `center`, or `right`, ignoring ASCII case.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "left" => Some(Self::Left),
            "center" | "centre" => Some(Self::Center),
            "right" => Some(Self::Right),
            _ => None,
        }
    }

    /// Canonical lowercase name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Left => "left",
            Self::Center => "center",
            Self::Right => "right",
        }
    }

    /// Splits `amount` padding cells into left and right padding.
    fn split(self, amount: usize) -> (usize, usize) {
        match self {
            Self::Left => (0, amount),
            Self::Center => (amount / 2, amount - amount / 2),
            Self::Right => (amount, 0),
        }
    }
}

impl std::str::FromStr for Align {
    type Err = StyleError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value).ok_or_else(|| StyleError::InvalidAlign(value.trim().to_string()))
    }
}

impl From<i32> for Align {
    /// Maps the original CosPlay image alignment values: `-1` and `2` are
    /// left-aligned, `0` is centered, and `1` is right-aligned.
    fn from(value: i32) -> Self {
        match value {
            0 => Self::Center,
            1 => Self::Right,
            _ => Self::Left,
        }
    }
}

/// Wrapping behavior for table cells.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Wrap {
    /// Do not wrap text. Text that does not fit is truncated.
    #[default]
    None,
    /// Wrap at a width, preferring whitespace boundaries.
    Words(usize),
    /// Wrap at a width without regard to word boundaries.
    Fixed(usize),
}

impl Wrap {
    /// The width requested by this wrapping mode, if any.
    pub const fn width(self) -> Option<usize> {
        match self {
            Self::None => None,
            Self::Words(width) | Self::Fixed(width) => Some(width),
        }
    }
}

/// Errors reported while parsing a table style.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StyleError {
    /// An entry did not have the form `property: value`.
    InvalidEntry(String),
    /// An unknown style property was used.
    UnknownProperty(String),
    /// A property value could not be parsed.
    InvalidValue {
        /// The property name.
        property: String,
        /// The rejected value.
        value: String,
    },
    /// The `align` property was not `left`, `center`, or `right`.
    InvalidAlign(String),
}

impl fmt::Display for StyleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidEntry(entry) => {
                write!(f, "invalid table style entry: {entry:?}")
            }
            Self::UnknownProperty(property) => {
                write!(f, "unknown table style property: {property:?}")
            }
            Self::InvalidValue { property, value } => {
                write!(f, "invalid value for table style {property:?}: {value:?}")
            }
            Self::InvalidAlign(value) => {
                write!(f, "invalid table alignment: {value:?}")
            }
        }
    }
}

impl std::error::Error for StyleError {}

/// Padding, maximum width, and alignment for one kind of table cell.
///
/// Styles can be constructed directly or parsed from the compact syntax used
/// by the original CosPlay engine:
///
/// ```text
/// leftPad: 1, rightPad: 1, maxWidth: 20, align: left
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellStyle {
    /// Spaces inserted before content.
    pub left_pad: usize,
    /// Spaces inserted after content.
    pub right_pad: usize,
    /// Maximum content width, excluding padding.
    pub max_width: Option<usize>,
    /// Horizontal alignment.
    pub align: Align,
}

impl Default for CellStyle {
    fn default() -> Self {
        Self {
            left_pad: 1,
            right_pad: 1,
            max_width: None,
            align: Align::Left,
        }
    }
}

impl CellStyle {
    /// Creates a style with one cell of padding on each side.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets left and right padding.
    pub const fn padded(mut self, left_pad: usize, right_pad: usize) -> Self {
        self.left_pad = left_pad;
        self.right_pad = right_pad;
        self
    }

    /// Sets the maximum content width.
    pub const fn max_width(mut self, max_width: usize) -> Self {
        self.max_width = Some(max_width);
        self
    }

    /// Sets the horizontal alignment.
    pub const fn align(mut self, align: Align) -> Self {
        self.align = align;
        self
    }

    /// Total horizontal padding.
    pub const fn padding(self) -> usize {
        self.left_pad + self.right_pad
    }

    /// Parses a comma-separated style string on top of the default style.
    ///
    /// Supported properties are `leftPad`, `rightPad`, `maxWidth`, and
    /// `align`. Property names are ASCII-case-insensitive.
    pub fn parse(style: &str) -> Result<Self, StyleError> {
        Self::parse_with(style, Self::default())
    }

    /// Parses a style string on top of `base`.
    pub fn parse_with(style: &str, mut base: Self) -> Result<Self, StyleError> {
        for entry in style.split(',') {
            let entry = entry.trim();
            if entry.is_empty() {
                continue;
            }
            let Some((property, value)) = entry.split_once(':') else {
                return Err(StyleError::InvalidEntry(entry.to_string()));
            };
            let property = property.trim();
            let value = value.trim();
            let normalized = property
                .chars()
                .filter(|ch| *ch != '_')
                .flat_map(char::to_lowercase)
                .collect::<String>();

            match normalized.as_str() {
                "leftpad" => {
                    base.left_pad = parse_usize(property, value)?;
                }
                "rightpad" => {
                    base.right_pad = parse_usize(property, value)?;
                }
                "maxwidth" => {
                    let width = parse_usize(property, value)?;
                    if width == 0 {
                        return Err(StyleError::InvalidValue {
                            property: property.to_string(),
                            value: value.to_string(),
                        });
                    }
                    base.max_width = Some(width);
                }
                "align" => {
                    base.align = Align::parse(value)
                        .ok_or_else(|| StyleError::InvalidAlign(value.to_string()))?;
                }
                _ => return Err(StyleError::UnknownProperty(property.to_string())),
            }
        }
        Ok(base)
    }
}

impl std::str::FromStr for CellStyle {
    type Err = StyleError;

    fn from_str(style: &str) -> Result<Self, Self::Err> {
        Self::parse(style)
    }
}

fn parse_usize(property: &str, value: &str) -> Result<usize, StyleError> {
    value
        .parse::<usize>()
        .map_err(|_| StyleError::InvalidValue {
            property: property.to_string(),
            value: value.to_string(),
        })
}

/// Box-drawing characters used by [`AsciiTable`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BorderChars {
    /// Horizontal line character.
    pub horizontal: char,
    /// Vertical line character.
    pub vertical: char,
    /// Top-left corner.
    pub top_left: char,
    /// Top-edge column intersection.
    pub top_join: char,
    /// Top-right corner.
    pub top_right: char,
    /// Left edge row intersection.
    pub middle_left: char,
    /// Interior row/column intersection.
    pub middle_join: char,
    /// Right edge row intersection.
    pub middle_right: char,
    /// Bottom-left corner.
    pub bottom_left: char,
    /// Bottom-edge column intersection.
    pub bottom_join: char,
    /// Bottom-right corner.
    pub bottom_right: char,
}

impl BorderChars {
    /// Conventional seven-bit ASCII table characters.
    pub const fn ascii() -> Self {
        Self {
            horizontal: '-',
            vertical: '|',
            top_left: '+',
            top_join: '+',
            top_right: '+',
            middle_left: '+',
            middle_join: '+',
            middle_right: '+',
            bottom_left: '+',
            bottom_join: '+',
            bottom_right: '+',
        }
    }

    /// Unicode light box-drawing characters.
    pub const fn unicode() -> Self {
        Self {
            horizontal: '─',
            vertical: '│',
            top_left: '┌',
            top_join: '┬',
            top_right: '┐',
            middle_left: '├',
            middle_join: '┼',
            middle_right: '┤',
            bottom_left: '└',
            bottom_join: '┴',
            bottom_right: '┘',
        }
    }
}

impl Default for BorderChars {
    fn default() -> Self {
        Self::ascii()
    }
}

/// Predefined or custom table border characters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BorderStyle {
    /// `+`, `-`, and `|`.
    #[default]
    Ascii,
    /// Unicode light box-drawing characters.
    Unicode,
    /// Caller-supplied characters.
    Custom(BorderChars),
}

impl BorderStyle {
    /// Returns the concrete characters for this style.
    pub const fn chars(self) -> BorderChars {
        match self {
            Self::Ascii => BorderChars::ascii(),
            Self::Unicode => BorderChars::unicode(),
            Self::Custom(chars) => chars,
        }
    }
}

/// Table margins, in character cells.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Margins {
    /// Blank rows above the table.
    pub top: usize,
    /// Spaces after each table row.
    pub right: usize,
    /// Blank rows below the table.
    pub bottom: usize,
    /// Spaces before each table row.
    pub left: usize,
}

/// Alias for [`Margins`], matching the spelling used by the original engine.
pub type Margin = Margins;

impl Margins {
    /// Creates margins from top, right, bottom, and left cell counts.
    pub const fn new(top: usize, right: usize, bottom: usize, left: usize) -> Self {
        Self {
            top,
            right,
            bottom,
            left,
        }
    }

    /// Uses the same margin on all four sides.
    pub const fn all(cells: usize) -> Self {
        Self::new(cells, cells, cells, cells)
    }

    /// Uses the given horizontal and vertical margins.
    pub const fn symmetric(horizontal: usize, vertical: usize) -> Self {
        Self::new(vertical, horizontal, vertical, horizontal)
    }
}

/// A deterministic character-cell table.
///
/// By default headers are centered, data cells are left-aligned, and every
/// cell has one space of horizontal padding. Rows without a matching number of
/// cells are padded with empty cells; cells may contain `'\n'` to request
/// multiple lines.
#[derive(Clone, Debug)]
pub struct AsciiTable {
    /// Header cells. An empty vector renders a table without a header row.
    pub headers: Vec<String>,
    /// Data rows. Rows may have different lengths.
    pub rows: Vec<Vec<String>>,
    /// Exact content widths per column, excluding cell padding.
    ///
    /// Missing entries, or `None`, use the natural content width. Widths may
    /// be reduced further by [`AsciiTable::max_width`].
    pub widths: Option<Vec<usize>>,
    /// Table margins.
    pub margins: Margins,
    /// Border character set.
    pub border: BorderStyle,
    /// Wrapping behavior.
    pub wrap: Wrap,
    /// Maximum total rendered width, including borders and left/right margins.
    pub max_width: Option<usize>,
    /// Whether to draw horizontal lines between data rows.
    pub inside_border: bool,
    /// Style for data cells.
    pub style: CellStyle,
    /// Style for header cells.
    pub header_style: CellStyle,
}

impl Default for AsciiTable {
    fn default() -> Self {
        Self {
            headers: Vec::new(),
            rows: Vec::new(),
            widths: None,
            margins: Margins::default(),
            border: BorderStyle::Ascii,
            wrap: Wrap::None,
            max_width: None,
            inside_border: false,
            style: CellStyle::default(),
            header_style: CellStyle {
                align: Align::Center,
                ..CellStyle::default()
            },
        }
    }
}

impl AsciiTable {
    /// Creates an empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a table from headers and rows.
    ///
    /// ```
    /// use cosplay::text::AsciiTable;
    ///
    /// let table = AsciiTable::of(["Name"], [["Ada"]]);
    /// assert_eq!(table.render(), "+------+\n| Name |\n+------+\n| Ada  |\n+------+");
    /// ```
    pub fn of<H, R>(headers: H, rows: R) -> Self
    where
        H: IntoIterator,
        H::Item: ToString,
        R: IntoIterator,
        R::Item: IntoIterator,
        <R::Item as IntoIterator>::Item: ToString,
    {
        let headers = headers.into_iter().map(|cell| cell.to_string()).collect();
        let rows = rows
            .into_iter()
            .map(|row| row.into_iter().map(|cell| cell.to_string()).collect())
            .collect();
        Self {
            headers,
            rows,
            ..Self::default()
        }
    }

    /// Creates a table without headers.
    pub fn from_rows<R>(rows: R) -> Self
    where
        R: IntoIterator,
        R::Item: IntoIterator,
        <R::Item as IntoIterator>::Item: ToString,
    {
        Self {
            headers: Vec::new(),
            rows: rows
                .into_iter()
                .map(|row| row.into_iter().map(|cell| cell.to_string()).collect())
                .collect(),
            ..Self::default()
        }
    }

    /// Replaces the headers.
    pub fn with_headers<H>(mut self, headers: H) -> Self
    where
        H: IntoIterator,
        H::Item: ToString,
    {
        self.headers = headers.into_iter().map(|cell| cell.to_string()).collect();
        self
    }

    /// Appends one data row.
    pub fn add_row<I>(&mut self, row: I) -> &mut Self
    where
        I: IntoIterator,
        I::Item: ToString,
    {
        self.rows
            .push(row.into_iter().map(|cell| cell.to_string()).collect());
        self
    }

    /// Appends one data row, returning the table for chaining.
    pub fn push_row<I>(mut self, row: I) -> Self
    where
        I: IntoIterator,
        I::Item: ToString,
    {
        self.add_row(row);
        self
    }

    /// Sets exact column content widths.
    pub fn with_widths<I>(mut self, widths: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        self.widths = Some(widths.into_iter().collect());
        self
    }

    /// Replaces the table margins.
    pub const fn with_margins(mut self, margins: Margins) -> Self {
        self.margins = margins;
        self
    }

    /// Sets the four margins in top, right, bottom, left order.
    pub const fn margin(mut self, top: usize, right: usize, bottom: usize, left: usize) -> Self {
        self.margins = Margins::new(top, right, bottom, left);
        self
    }

    /// Alias for [`AsciiTable::margin`].
    pub const fn with_margin(self, top: usize, right: usize, bottom: usize, left: usize) -> Self {
        self.margin(top, right, bottom, left)
    }

    /// Sets the border style.
    pub const fn with_border(mut self, border: BorderStyle) -> Self {
        self.border = border;
        self
    }

    /// Alias for [`AsciiTable::with_border`].
    pub const fn with_border_style(self, border: BorderStyle) -> Self {
        self.with_border(border)
    }

    /// Uses caller-supplied border characters.
    pub const fn with_border_chars(mut self, chars: BorderChars) -> Self {
        self.border = BorderStyle::Custom(chars);
        self
    }

    /// Sets the wrapping behavior.
    pub const fn with_wrap(mut self, wrap: Wrap) -> Self {
        self.wrap = wrap;
        self
    }

    /// Sets (or clears, with `None`) the maximum total rendered table width.
    pub fn with_max_width<M>(mut self, max_width: M) -> Self
    where
        M: Into<Option<usize>>,
    {
        self.max_width = max_width.into();
        self
    }

    /// Enables or disables horizontal lines between data rows.
    pub const fn with_inside_border(mut self, inside_border: bool) -> Self {
        self.inside_border = inside_border;
        self
    }

    /// Sets the data-cell alignment.
    pub const fn with_align(mut self, align: Align) -> Self {
        self.style.align = align;
        self
    }

    /// Alias for [`AsciiTable::with_align`].
    pub const fn align(self, align: Align) -> Self {
        self.with_align(align)
    }

    /// Alias for [`AsciiTable::with_align`].
    pub const fn with_alignment(self, align: Align) -> Self {
        self.with_align(align)
    }

    /// Sets the header alignment.
    pub const fn with_header_align(mut self, align: Align) -> Self {
        self.header_style.align = align;
        self
    }

    /// Sets exact column content widths. Alias for [`AsciiTable::with_widths`].
    pub fn widths<I>(self, widths: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        self.with_widths(widths)
    }

    /// Sets the table margins. Alias for [`AsciiTable::with_margins`].
    pub const fn margins(self, margins: Margins) -> Self {
        self.with_margins(margins)
    }

    /// Sets the border style. Alias for [`AsciiTable::with_border`].
    pub const fn border(self, border: BorderStyle) -> Self {
        self.with_border(border)
    }

    /// Sets the wrapping behavior. Alias for [`AsciiTable::with_wrap`].
    pub const fn wrap(self, wrap: Wrap) -> Self {
        self.with_wrap(wrap)
    }

    /// Sets the maximum total rendered width. Alias for
    /// [`AsciiTable::with_max_width`].
    pub fn max_width<M>(self, max_width: M) -> Self
    where
        M: Into<Option<usize>>,
    {
        self.with_max_width(max_width)
    }

    /// Enables or disables horizontal lines between rows. Alias for
    /// [`AsciiTable::with_inside_border`].
    pub const fn inside_border(self, inside_border: bool) -> Self {
        self.with_inside_border(inside_border)
    }

    /// Alias for [`AsciiTable::with_header_align`].
    pub const fn header_align(self, align: Align) -> Self {
        self.with_header_align(align)
    }

    /// Alias for [`AsciiTable::with_header_align`].
    pub const fn with_header_alignment(self, align: Align) -> Self {
        self.with_header_align(align)
    }

    /// Alias for [`AsciiTable::with_align`].
    pub const fn alignment(self, align: Align) -> Self {
        self.with_align(align)
    }

    /// Parses and applies a data-cell style string, preserving unspecified
    /// fields from the current style.
    pub fn with_style(self, style: &str) -> Result<Self, StyleError> {
        let style = CellStyle::parse_with(style, self.style)?;
        Ok(Self { style, ..self })
    }

    /// Alias for [`AsciiTable::with_style`].
    pub fn style(self, style: &str) -> Result<Self, StyleError> {
        self.with_style(style)
    }

    /// Parses and applies a header style string, preserving unspecified fields
    /// from the current header style.
    pub fn with_header_style(self, style: &str) -> Result<Self, StyleError> {
        let header_style = CellStyle::parse_with(style, self.header_style)?;
        Ok(Self {
            header_style,
            ..self
        })
    }

    /// Alias for [`AsciiTable::with_header_style`].
    pub fn header_style(self, style: &str) -> Result<Self, StyleError> {
        self.with_header_style(style)
    }

    /// Renders the table with its rows joined by `'\n'`.
    ///
    /// The final border has no newline of its own; a non-zero bottom margin
    /// adds blank lines after it. An entirely empty table renders as the empty
    /// string.
    pub fn render(&self) -> String {
        let columns = self.column_count();
        if columns == 0 {
            return String::new();
        }

        let has_header = !self.headers.is_empty();
        let chars = self.border.chars();
        let pads = self.column_pads(columns, has_header);
        let widths = self.column_widths(columns, has_header, &pads);

        let mut lines = Vec::new();
        for _ in 0..self.margins.top {
            lines.push(String::new());
        }

        lines.push(self.with_margins_around(self.border_line(
            chars,
            chars.top_left,
            chars.top_join,
            chars.top_right,
            &widths,
            &pads,
        )));

        if has_header {
            lines.extend(
                self.render_record(&self.headers, self.header_style, &widths, &pads, chars)
                    .into_iter()
                    .map(|line| self.with_margins_around(line)),
            );
        }

        let draw_header_separator = has_header && !self.rows.is_empty();
        if draw_header_separator {
            lines.push(self.with_margins_around(self.border_line(
                chars,
                chars.middle_left,
                chars.middle_join,
                chars.middle_right,
                &widths,
                &pads,
            )));
        }

        for (index, row) in self.rows.iter().enumerate() {
            lines.extend(
                self.render_record(row, self.style, &widths, &pads, chars)
                    .into_iter()
                    .map(|line| self.with_margins_around(line)),
            );
            if self.inside_border && index + 1 < self.rows.len() {
                lines.push(self.with_margins_around(self.border_line(
                    chars,
                    chars.middle_left,
                    chars.middle_join,
                    chars.middle_right,
                    &widths,
                    &pads,
                )));
            }
        }

        lines.push(self.with_margins_around(self.border_line(
            chars,
            chars.bottom_left,
            chars.bottom_join,
            chars.bottom_right,
            &widths,
            &pads,
        )));

        for _ in 0..self.margins.bottom {
            lines.push(String::new());
        }

        lines.join("\n")
    }

    /// Renders the table into an image with one pixel per rendered character.
    ///
    /// The background may be a [`Color`] or an `Option<Color>`.
    pub fn to_image<B>(&self, fg: Color, bg: B) -> Image
    where
        B: Into<Option<Color>>,
    {
        Image::from_text(&self.render(), fg, bg.into(), false)
    }

    fn column_count(&self) -> usize {
        self.rows
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(0)
            .max(self.headers.len())
    }

    fn column_pads(&self, columns: usize, has_header: bool) -> Vec<usize> {
        (0..columns)
            .map(|_| {
                let mut padding = self.style.padding();
                if has_header {
                    padding = padding.max(self.header_style.padding());
                }
                padding
            })
            .collect()
    }

    fn column_widths(&self, columns: usize, has_header: bool, pads: &[usize]) -> Vec<usize> {
        let mut widths = vec![0; columns];

        if has_header {
            for (index, width) in widths.iter_mut().enumerate() {
                let text = self.headers.get(index).map_or("", String::as_str);
                *width = cell_natural_width(text, self.header_style, self.wrap);
            }
        }

        for row in &self.rows {
            for (index, width) in widths.iter_mut().enumerate() {
                let text = row.get(index).map_or("", String::as_str);
                *width = (*width).max(cell_natural_width(text, self.style, self.wrap));
            }
        }

        if let Some(explicit) = &self.widths {
            for (index, width) in explicit.iter().take(columns).enumerate() {
                widths[index] = *width;
            }
        }

        if let Some(max_width) = self.max_width {
            let overhead = self
                .margins
                .left
                .saturating_add(self.margins.right)
                .saturating_add(columns.saturating_add(1))
                .saturating_add(pads.iter().copied().fold(0usize, usize::saturating_add));
            let available = max_width.saturating_sub(overhead);
            widths = fit_widths(&widths, available);
        }

        widths
    }

    fn with_margins_around(&self, line: String) -> String {
        let mut result = String::with_capacity(
            self.margins
                .left
                .saturating_add(line.chars().count())
                .saturating_add(self.margins.right),
        );
        result.push_str(&" ".repeat(self.margins.left));
        result.push_str(&line);
        result.push_str(&" ".repeat(self.margins.right));
        result
    }

    fn border_line(
        &self,
        chars: BorderChars,
        left: char,
        join: char,
        right: char,
        widths: &[usize],
        pads: &[usize],
    ) -> String {
        let mut line = String::new();
        line.push(left);
        for (index, (width, pad)) in widths.iter().zip(pads).enumerate() {
            if index != 0 {
                line.push(join);
            }
            for _ in 0..width.saturating_add(*pad) {
                line.push(chars.horizontal);
            }
        }
        line.push(right);
        line
    }

    fn render_record(
        &self,
        cells: &[String],
        style: CellStyle,
        widths: &[usize],
        pads: &[usize],
        chars: BorderChars,
    ) -> Vec<String> {
        let cell_lines: Vec<Vec<String>> = widths
            .iter()
            .enumerate()
            .map(|(index, width)| {
                let text = cells.get(index).map_or("", String::as_str);
                cell_lines(text, style, self.wrap, *width)
            })
            .collect();
        let height = cell_lines.iter().map(Vec::len).max().unwrap_or(1).max(1);
        let mut lines = Vec::with_capacity(height);

        for line_index in 0..height {
            let mut line = String::new();
            line.push(chars.vertical);
            for (index, width) in widths.iter().enumerate() {
                let text = cell_lines[index].get(line_index).map_or("", String::as_str);
                line.push_str(&align_text(text, width.saturating_add(pads[index]), style));
                line.push(chars.vertical);
            }
            lines.push(line);
        }

        lines
    }
}

impl fmt::Display for AsciiTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render())
    }
}

fn cell_natural_width(text: &str, style: CellStyle, wrap: Wrap) -> usize {
    let limit = style
        .max_width
        .unwrap_or(usize::MAX)
        .min(wrap.width().unwrap_or(usize::MAX));
    text.split('\n')
        .map(|line| display_width(line).min(limit))
        .max()
        .unwrap_or(0)
}

fn cell_lines(text: &str, style: CellStyle, wrap: Wrap, width: usize) -> Vec<String> {
    let limit = width
        .min(style.max_width.unwrap_or(usize::MAX))
        .min(wrap.width().unwrap_or(usize::MAX));
    let mut lines = Vec::new();

    for raw in text.split('\n') {
        if raw.is_empty() {
            lines.push(String::new());
        } else if display_width(raw) <= limit {
            lines.push(raw.to_string());
        } else {
            match wrap {
                Wrap::None => lines.push(truncate_to_width(raw, limit)),
                Wrap::Fixed(_) => lines.extend(wrap_fixed(raw, limit)),
                Wrap::Words(_) => lines.extend(wrap_words(raw, limit)),
            }
        }
    }

    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn align_text(text: &str, width: usize, style: CellStyle) -> String {
    let text = truncate_to_width(text, width);
    let text_width = display_width(&text);
    let free = width.saturating_sub(text_width);
    let (left, right) = match style.align {
        Align::Left => {
            let left = style.left_pad.min(free);
            (left, free - left)
        }
        Align::Right => {
            let right = style.right_pad.min(free);
            (free - right, right)
        }
        Align::Center => (free / 2, free - free / 2),
    };

    let mut result = String::with_capacity(width);
    result.push_str(&" ".repeat(left));
    result.push_str(&text);
    result.push_str(&" ".repeat(right));
    result
}

fn fit_widths(widths: &[usize], available: usize) -> Vec<usize> {
    let total = widths.iter().copied().fold(0usize, usize::saturating_add);
    if total <= available {
        return widths.to_vec();
    }
    if available == 0 || widths.is_empty() {
        return vec![0; widths.len()];
    }

    let maximum = widths.iter().copied().max().unwrap_or(0);
    let mut low = 0;
    let mut high = maximum;
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        let used = widths
            .iter()
            .map(|width| (*width).min(middle))
            .sum::<usize>();
        if used <= available {
            low = middle;
        } else {
            high = middle - 1;
        }
    }

    let mut result = widths
        .iter()
        .map(|width| (*width).min(low))
        .collect::<Vec<_>>();
    let mut remaining =
        available.saturating_sub(result.iter().copied().fold(0usize, usize::saturating_add));
    while remaining != 0 {
        let mut progressed = false;
        for (index, width) in widths.iter().enumerate() {
            if remaining == 0 {
                break;
            }
            if result[index] < *width {
                result[index] += 1;
                remaining -= 1;
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
    result
}

fn display_width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

fn truncate_to_width(text: &str, width: usize) -> String {
    let mut result = String::new();
    let mut used = 0usize;
    for ch in text.chars() {
        let char_width = ch.width().unwrap_or(0);
        if used.saturating_add(char_width) > width {
            break;
        }
        used += char_width;
        result.push(ch);
    }
    result
}

fn split_index(text: &str, width: usize) -> usize {
    let mut used = 0usize;
    for (index, ch) in text.char_indices() {
        let char_width = ch.width().unwrap_or(0);
        if used.saturating_add(char_width) > width {
            return index;
        }
        used += char_width;
    }
    text.len()
}

fn wrap_fixed(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }

    let mut lines = Vec::new();
    let mut current = String::new();
    let mut used = 0usize;
    for ch in text.chars() {
        let char_width = ch.width().unwrap_or(0);
        if !current.is_empty() && used.saturating_add(char_width) > width {
            lines.push(std::mem::take(&mut current));
            used = 0;
        }
        current.push(ch);
        used += char_width;
        if used > width {
            lines.push(std::mem::take(&mut current));
            used = 0;
        }
    }
    if !current.is_empty() || lines.is_empty() {
        lines.push(current);
    }
    lines
}

fn wrap_words(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }

    let mut lines = Vec::new();
    let mut rest = text;

    while !rest.is_empty() {
        if display_width(rest) <= width {
            lines.push(rest.to_string());
            break;
        }

        let mut end = split_index(rest, width);
        if end == 0 {
            end = rest.chars().next().map_or(rest.len(), char::len_utf8);
        }
        let chunk = &rest[..end];
        let remainder = &rest[end..];
        let clean_boundary = remainder.chars().next().is_none_or(char::is_whitespace);

        if clean_boundary {
            let head = chunk.trim_end();
            if !head.is_empty() {
                lines.push(head.to_string());
            }
            rest = remainder.trim_start();
            continue;
        }

        if let Some((offset, _)) = chunk
            .char_indices()
            .rev()
            .find(|(_, ch)| ch.is_whitespace())
        {
            let head = chunk[..offset].trim_end();
            if !head.is_empty() {
                lines.push(head.to_string());
                rest = rest[offset..].trim_start();
                continue;
            }
        }

        lines.push(chunk.to_string());
        rest = remainder;
    }

    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    const RED: Color = Color::rgb(255, 0, 0);
    const GREEN: Color = Color::rgb(0, 255, 0);
    const BLUE: Color = Color::rgb(0, 0, 255);

    fn chars(pixels: &[Pixel]) -> String {
        pixels.iter().map(|pixel| pixel.ch).collect()
    }

    // Markup -----------------------------------------------------------------

    fn colored_markup() -> Markup {
        Markup::new(
            Color::WHITE,
            Some(Color::BLACK),
            vec![
                MarkupElement::colored("<r>", "</r>", RED, Some(Color::BLACK)),
                MarkupElement::colored("<b>", "</b>", BLUE, None),
            ],
        )
        .unwrap()
    }

    #[test]
    fn markup_accepts_custom_closure_skins() {
        let element = MarkupElement::new("<g>", "</g>", |ch| {
            Pixel::new(ch, GREEN, Some(Color::BLACK))
        });
        let markup = Markup::new(Color::WHITE, None, [element]).unwrap();
        let pixels = markup.process("<g>ok</g>").unwrap();
        assert_eq!(chars(&pixels), "ok");
        assert!(pixels.iter().all(|pixel| pixel.fg == GREEN));
    }

    #[test]
    fn markup_removes_tags_and_applies_skins() {
        let markup = colored_markup();
        let pixels = markup.process("a<r>red</r>b").unwrap();
        assert_eq!(chars(&pixels), "aredb");
        assert_eq!(pixels[0].fg, Color::WHITE);
        assert_eq!(pixels[0].bg, Some(Color::BLACK));
        for pixel in &pixels[1..4] {
            assert_eq!(pixel.fg, RED);
            assert_eq!(pixel.bg, Some(Color::BLACK));
        }
        assert_eq!(pixels[4].fg, Color::WHITE);
    }

    #[test]
    fn markup_nests_and_restores_the_outer_skin() {
        let markup = colored_markup();
        let pixels = markup.process("a<r>r<b>blue</b>r</r>z").unwrap();
        assert_eq!(chars(&pixels), "arbluerz");
        assert_eq!(pixels[1].fg, RED);
        for pixel in &pixels[2..6] {
            assert_eq!(pixel.fg, BLUE);
            assert_eq!(pixel.bg, None);
        }
        assert_eq!(pixels[6].fg, RED);
        assert_eq!(pixels[7].fg, Color::WHITE);
    }

    #[test]
    fn markup_reports_unmatched_mismatched_and_unclosed_tags() {
        let markup = colored_markup();
        assert!(matches!(
            markup.process("x</r>"),
            Err(MarkupError::UnmatchedCloseTag { .. })
        ));
        assert!(matches!(
            markup.process("<r><b>x</r></b>"),
            Err(MarkupError::MismatchedCloseTag { .. })
        ));
        assert!(matches!(
            markup.process("<r>x"),
            Err(MarkupError::UnclosedTag { .. })
        ));
        assert!(markup.process("<r>x</r>").is_ok());
    }

    #[test]
    fn markup_validation_matches_upstream_rules() {
        let element = |open: &str, close: &str| MarkupElement::colored(open, close, RED, None);

        assert!(matches!(
            Markup::new(Color::WHITE, None, vec![element("", ">x")]),
            Err(MarkupError::EmptyTag {
                which: "opening",
                ..
            })
        ));
        assert!(matches!(
            Markup::new(Color::WHITE, None, vec![element("<x>", "")]),
            Err(MarkupError::EmptyTag {
                which: "closing",
                ..
            })
        ));
        assert!(matches!(
            Markup::new(Color::WHITE, None, vec![element("< x>", "> x")]),
            Err(MarkupError::TagContainsSpace { .. })
        ));
        assert!(matches!(
            Markup::new(
                Color::WHITE,
                None,
                vec![element("<x>", "</x>"), element("<x>", "</y>")]
            ),
            Err(MarkupError::DuplicateTag { .. })
        ));
        assert!(matches!(
            Markup::new(
                Color::WHITE,
                None,
                vec![element("<x>", "</x>"), element("<x>y", "</x>y")]
            ),
            Err(MarkupError::IntersectingTags { .. })
        ));
        assert!(matches!(
            Markup::new(Color::WHITE, None, vec![element("<x>", "<x>")]),
            Err(MarkupError::DuplicateTag { .. })
        ));
        assert!(Markup::new(Color::WHITE, None, vec![element("<x>", "</x>")]).is_ok());
    }

    #[test]
    fn markup_to_image_aligns_and_keeps_skin_colors() {
        let markup = colored_markup();
        let image = markup.to_image("<r>red</r>\nz", Align::Right);
        assert_eq!((image.width(), image.height()), (3, 2));
        assert_eq!(image.get(0, 0).unwrap().ch, 'r');
        assert_eq!(image.get(1, 0).unwrap().fg, RED);
        assert_eq!(image.get(0, 1).unwrap().ch, ' ');
        assert_eq!(image.get(2, 1).unwrap().ch, 'z');

        let centered = markup.try_to_image("x\nyyy", Align::Center).unwrap();
        assert_eq!((centered.width(), centered.height()), (3, 2));
        assert_eq!(centered.get(0, 0).unwrap().ch, ' ');
        assert_eq!(centered.get(1, 0).unwrap().ch, 'x');
        assert_eq!(centered.get(0, 1).unwrap().ch, 'y');
    }

    // Styled strings ---------------------------------------------------------

    #[test]
    fn styled_string_builds_a_single_colored_row() {
        let mut label = StyledString::new(Color::WHITE, Some(Color::BLACK));
        label
            .push_str("Hi", GREEN)
            .push_text("!", RED, None)
            .push_pixel(Pixel::new('#', BLUE, Some(Color::BLACK)));
        assert_eq!(label.width(), 4);
        assert_eq!(label.len(), 4);
        assert!(!label.is_empty());
        assert_eq!(chars(&label.build()), "Hi!#");
        assert_eq!(label.build()[0].fg, GREEN);
        assert_eq!(label.build()[0].bg, Some(Color::BLACK));
        assert_eq!(label.build()[2].fg, RED);
        assert_eq!(label.build()[2].bg, None);
        assert_eq!(label.build()[3].fg, BLUE);

        let copied = label.clone();
        assert_eq!(copied, label);
        label.clear();
        assert!(label.is_empty());
        assert_eq!(copied.width(), 4);
    }

    #[test]
    fn styled_helper_and_defaults() {
        let pixels = styled(&[(RED, "ab"), (BLUE, "c")]);
        assert_eq!(chars(&pixels), "abc");
        assert_eq!(pixels[0].fg, RED);
        assert_eq!(pixels[2].fg, BLUE);
        assert_eq!(pixels[0].bg, None);

        let pixels = StyledString::styled(&[(GREEN, "x")]);
        assert_eq!(chars(&pixels), "x");
        assert_eq!(StyledString::default().width(), 0);
        assert_eq!(StyledString::style_str("ab", RED)[1].fg, RED);
    }

    // Tables -----------------------------------------------------------------

    fn table() -> AsciiTable {
        AsciiTable::of(["Name", "Value"], [["Ada", "1"], ["Grace", "22"]])
    }

    #[test]
    fn ascii_table_renders_a_deterministic_grid() {
        let rendered = table().render();
        assert_eq!(
            rendered,
            concat!(
                "+-------+-------+\n",
                "| Name  | Value |\n",
                "+-------+-------+\n",
                "| Ada   | 1     |\n",
                "| Grace | 22    |\n",
                "+-------+-------+"
            )
        );
        assert_eq!(table().to_string(), rendered);
    }

    #[test]
    fn table_centers_headers_and_honors_alignment_and_padding() {
        let table = AsciiTable::of(["H"], [["abcd"]])
            .with_style("leftPad: 0, rightPad: 0, align: right")
            .unwrap()
            .with_header_style("leftPad: 0, rightPad: 0, align: center")
            .unwrap();
        assert_eq!(
            table.render(),
            concat!("+----+\n", "| H  |\n", "+----+\n", "|abcd|\n", "+----+")
        );

        let table = AsciiTable::of(["x"], [["ab"]]).with_align(Align::Center);
        assert_eq!(table.render(), "+----+\n| x  |\n+----+\n| ab |\n+----+");
    }

    #[test]
    fn table_wraps_words_and_falls_back_to_hard_breaks() {
        let words = AsciiTable::of(["Text"], [["one two three"]])
            .with_wrap(Wrap::Words(7))
            .with_header_align(Align::Left);
        assert_eq!(
            words.render(),
            concat!(
                "+---------+\n",
                "| Text    |\n",
                "+---------+\n",
                "| one two |\n",
                "| three   |\n",
                "+---------+"
            )
        );

        let long = AsciiTable::of(["x"], [["abcdefgh"]]).with_wrap(Wrap::Words(5));
        assert_eq!(
            long.render(),
            concat!(
                "+-------+\n",
                "|   x   |\n",
                "+-------+\n",
                "| abcde |\n",
                "| fgh   |\n",
                "+-------+"
            )
        );
    }

    #[test]
    fn table_wraps_fixed_widths_and_multiline_cells() {
        let table = AsciiTable::of(["A", "B"], [["abcdef", "x\ny"]])
            .with_wrap(Wrap::Fixed(3))
            .with_header_align(Align::Left);
        assert_eq!(
            table.render(),
            concat!(
                "+-----+---+\n",
                "| A   | B |\n",
                "+-----+---+\n",
                "| abc | x |\n",
                "| def | y |\n",
                "+-----+---+"
            )
        );
    }

    #[test]
    fn table_truncates_to_explicit_and_global_maximum_widths() {
        let explicit = AsciiTable::of(["A"], [["abcdef"]])
            .with_widths([3])
            .with_header_align(Align::Left);
        assert_eq!(
            explicit.render(),
            "+-----+\n| A   |\n+-----+\n| abc |\n+-----+"
        );

        let bounded = AsciiTable::of(["A", "B"], [["hello", "world"]])
            .with_max_width(11)
            .with_header_align(Align::Left);
        assert_eq!(
            bounded.render(),
            "+----+----+\n| A  | B  |\n+----+----+\n| he | wo |\n+----+----+"
        );
    }

    #[test]
    fn table_handles_empty_and_uneven_input() {
        assert_eq!(AsciiTable::new().render(), "");
        assert_eq!(
            AsciiTable::from_rows(Vec::<Vec<String>>::new()).render(),
            ""
        );

        let uneven = AsciiTable::of(["A", "B", "C"], [vec!["1"], vec!["2", "3"]]);
        assert_eq!(
            uneven.render(),
            concat!(
                "+---+---+---+\n",
                "| A | B | C |\n",
                "+---+---+---+\n",
                "| 1 |   |   |\n",
                "| 2 | 3 |   |\n",
                "+---+---+---+"
            )
        );

        let header_only = AsciiTable::of(["A"], Vec::<Vec<String>>::new());
        assert_eq!(header_only.render(), "+---+\n| A |\n+---+");
    }

    #[test]
    fn table_draws_unicode_and_custom_borders_and_inside_lines() {
        let unicode = AsciiTable::of(["A"], [["1"], ["2"]])
            .with_border(BorderStyle::Unicode)
            .with_inside_border(true)
            .with_header_align(Align::Left);
        assert_eq!(
            unicode.render(),
            concat!(
                "┌───┐\n",
                "│ A │\n",
                "├───┤\n",
                "│ 1 │\n",
                "├───┤\n",
                "│ 2 │\n",
                "└───┘"
            )
        );

        let custom = BorderChars {
            horizontal: '=',
            vertical: '#',
            top_left: '*',
            top_join: '*',
            top_right: '*',
            middle_left: '*',
            middle_join: '*',
            middle_right: '*',
            bottom_left: '*',
            bottom_join: '*',
            bottom_right: '*',
        };
        let table = AsciiTable::of(["A"], [["1"]]).with_border_chars(custom);
        assert_eq!(table.render(), "*===*\n# A #\n*===*\n# 1 #\n*===*");
    }

    #[test]
    fn table_applies_margins_without_trailing_whitespace_on_rows() {
        let table = AsciiTable::of(["A"], [["1"]])
            .margin(1, 2, 1, 3)
            .with_header_align(Align::Left);
        assert_eq!(
            table.render(),
            concat!(
                "\n",
                "   +---+  \n",
                "   | A |  \n",
                "   +---+  \n",
                "   | 1 |  \n",
                "   +---+  \n"
            )
        );
    }

    #[test]
    fn table_style_parser_supports_the_upstream_keys() {
        let style = CellStyle::parse("leftPad: 2, rightPad:0, maxWidth: 8, align: right").unwrap();
        assert_eq!(
            style,
            CellStyle {
                left_pad: 2,
                right_pad: 0,
                max_width: Some(8),
                align: Align::Right,
            }
        );
        assert_eq!(CellStyle::parse("").unwrap(), CellStyle::default());
        assert_eq!(
            CellStyle::parse("ALIGN: Center, left_pad: 3")
                .unwrap()
                .align,
            Align::Center
        );

        assert!(matches!(
            CellStyle::parse("maxWidth: 0"),
            Err(StyleError::InvalidValue { .. })
        ));
        assert!(matches!(
            CellStyle::parse("align: sideways"),
            Err(StyleError::InvalidAlign(_))
        ));
        assert!(matches!(
            CellStyle::parse("wat: 1"),
            Err(StyleError::UnknownProperty(_))
        ));
        assert!(matches!(
            CellStyle::parse("leftPad"),
            Err(StyleError::InvalidEntry(_))
        ));

        let table = AsciiTable::of(["A"], [["1"]])
            .with_style("leftPad: 0, rightPad: 0, align: right")
            .unwrap();
        assert_eq!(table.style.left_pad, 0);
        assert_eq!(table.style.align, Align::Right);
        assert_eq!(table.header_style.align, Align::Center);
    }

    #[test]
    fn table_to_image_uses_the_rendered_dimensions() {
        let table = AsciiTable::of(["A"], [["1"]]);
        let image = table.to_image(Color::WHITE, Some(Color::BLACK));
        let rendered = table.render();
        let lines = rendered.lines().count();
        let width = rendered
            .lines()
            .map(|line| line.chars().count())
            .max()
            .unwrap();
        assert_eq!(image.width(), width as u16);
        assert_eq!(image.height(), lines as u16);
    }

    #[test]
    fn helpers_wrap_and_truncate_by_display_width() {
        assert_eq!(truncate_to_width("abcdef", 3), "abc");
        assert_eq!(truncate_to_width("abc", 0), "");
        assert_eq!(wrap_fixed("abcdefg", 3), vec!["abc", "def", "g"]);
        assert_eq!(wrap_words("one two three", 7), vec!["one two", "three"]);
        assert_eq!(wrap_words("abcdefgh", 5), vec!["abcde", "fgh"]);
        assert_eq!(fit_widths(&[5, 5], 4), vec![2, 2]);
        assert_eq!(fit_widths(&[3, 4], 10), vec![3, 4]);
    }
}
