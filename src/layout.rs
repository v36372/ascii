// SPDX-License-Identifier: Apache-2.0
//! A deterministic, dependency-driven layout engine inspired by CosPlay's
//! `CPLayoutSprite`/`CPLayoutCompiler`.
//!
//! A [`Layout`] maps object IDs to [`LayoutSpec`] values. Resolving a layout
//! computes each object's top-left position from a canvas, a map of external
//! rectangles, and the desired size shared by all laid-out objects. Objects may
//! depend on other laid-out objects, so chains are resolved with a stable
//! depth-first traversal of the ID-ordered map. Unknown targets and cycles are
//! reported as [`LayoutError`] rather than panicking.
//!
//! A target of `""` or `"@"` refers to the canvas. Every coordinate is an
//! `i32`; all derived totals use saturating arithmetic, so objects may be
//! placed outside the canvas without overflow. A scalar offset shifts the
//! relation's primary axis: x for horizontal relations and
//! [`LayoutRelation::Center`], y for vertical relations. An offset also shifts
//! x for [`LayoutRelation::Same`] without an axis and for
//! [`LayoutRelation::At`].
//!
//! # Text format
//!
//! This is intentionally a practical subset of CosPlay's parser, not its full
//! ANTLR grammar. IDs and targets are bare, whitespace-delimited tokens (there
//! is no quoting). Supported lines are:
//!
//! ```text
//! id: at X Y [OFFSET]
//! id: left|right of TARGET [OFFSET]
//! id: above|below [of] TARGET [OFFSET]
//! id: top|bottom of TARGET [OFFSET]
//! id: center of TARGET [OFFSET]
//! id: same as TARGET [OFFSET]
//! id: same x|y as TARGET [OFFSET]
//! id: inside|outside left|right|top|bottom of TARGET [OFFSET]
//! ```
//!
//! `top` is an alias for `above`, and `bottom` is an alias for `below`.
//! Offsets may be written as `+ 2`, `+2`, `- 3`, or `-3`. Blank lines and
//! trailing comments beginning with `#` or `--` are ignored. Parse failures
//! carry their one-based line number.
//!
//! ```
//! use cosplay::layout::{Layout, LayoutSpec};
//! use std::collections::BTreeMap;
//!
//! let mut layout = Layout::new();
//! layout.insert("panel", LayoutSpec::at(2, 1));
//! layout.insert("title", LayoutSpec::inside_top("panel").with_offset(1));
//! layout.insert("button", LayoutSpec::after("panel").with_offset(2));
//!
//! let positions = layout
//!     .resolve(
//!         &BTreeMap::new(),
//!         cosplay::graphics::Rect::new(0, 0, 80, 24),
//!         (6, 2),
//!     )
//!     .unwrap();
//! assert_eq!(positions["panel"], (2, 1));
//! assert_eq!(positions["title"], (2, 2));
//! assert_eq!(positions["button"], (10, 1));
//! ```

use crate::graphics::Rect;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

/// The axis used by [`LayoutRelation::Same`] and the shorthand constructors on
/// [`LayoutRelation`]. Horizontal is the x axis; vertical is the y axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LayoutDirection {
    Horizontal,
    Vertical,
}

/// A placement rule relative to a target rectangle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LayoutRelation {
    /// Align the own rectangle's top-left with the target's inside left edge.
    InsideLeft,
    /// Align the own rectangle's right edge with the target's inside right edge.
    InsideRight,
    /// Align the own rectangle's top-left with the target's inside top edge.
    InsideTop,
    /// Align the own rectangle's bottom edge with the target's inside bottom edge.
    InsideBottom,
    /// Place immediately to the left of the target.
    Before,
    /// Place immediately to the right of the target.
    After,
    /// Place immediately above the target.
    Above,
    /// Place immediately below the target.
    Below,
    /// Center the own rectangle on both axes of the target.
    Center,
    /// Copy both coordinates from the target. With an axis, copy only that
    /// axis and resolve the other coordinate to the canvas origin.
    Same { axis: Option<LayoutDirection> },
    /// An absolute position. This variant exists mainly for the text format;
    /// its target is ignored.
    At { x: i32, y: i32 },
}

impl LayoutRelation {
    /// Start-edge shorthand: horizontal selects [`Self::InsideLeft`], vertical
    /// selects [`Self::InsideTop`].
    pub const fn inside(direction: LayoutDirection) -> Self {
        match direction {
            LayoutDirection::Horizontal => Self::InsideLeft,
            LayoutDirection::Vertical => Self::InsideTop,
        }
    }

    /// Start-edge shorthand: horizontal selects [`Self::Before`], vertical
    /// selects [`Self::Above`].
    pub const fn outside(direction: LayoutDirection) -> Self {
        match direction {
            LayoutDirection::Horizontal => Self::Before,
            LayoutDirection::Vertical => Self::Above,
        }
    }
}

/// One object's placement rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayoutSpec {
    /// The target object's ID. Either `""` or `"@"` means the canvas.
    pub target: String,
    pub relation: LayoutRelation,
    /// A signed shift along the relation's primary axis.
    pub offset: i32,
}

/// Alias for [`LayoutSpec`], matching CosPlay's anchor terminology.
pub type Anchor = LayoutSpec;

impl LayoutSpec {
    pub fn new(target: impl Into<String>, relation: LayoutRelation) -> Self {
        Self {
            target: normalize_target(target.into(), relation),
            relation,
            offset: 0,
        }
    }

    pub fn at(x: i32, y: i32) -> Self {
        Self::new("", LayoutRelation::At { x, y })
    }

    pub fn before(target: impl Into<String>) -> Self {
        Self::new(target, LayoutRelation::Before)
    }

    pub fn after(target: impl Into<String>) -> Self {
        Self::new(target, LayoutRelation::After)
    }

    pub fn above(target: impl Into<String>) -> Self {
        Self::new(target, LayoutRelation::Above)
    }

    pub fn below(target: impl Into<String>) -> Self {
        Self::new(target, LayoutRelation::Below)
    }

    pub fn center_of(target: impl Into<String>) -> Self {
        Self::new(target, LayoutRelation::Center)
    }

    pub fn same_as(target: impl Into<String>) -> Self {
        Self::new(target, LayoutRelation::Same { axis: None })
    }

    pub fn same_axis(target: impl Into<String>, axis: LayoutDirection) -> Self {
        Self::new(target, LayoutRelation::Same { axis: Some(axis) })
    }

    pub fn inside_left(target: impl Into<String>) -> Self {
        Self::new(target, LayoutRelation::InsideLeft)
    }

    pub fn inside_right(target: impl Into<String>) -> Self {
        Self::new(target, LayoutRelation::InsideRight)
    }

    pub fn inside_top(target: impl Into<String>) -> Self {
        Self::new(target, LayoutRelation::InsideTop)
    }

    pub fn inside_bottom(target: impl Into<String>) -> Self {
        Self::new(target, LayoutRelation::InsideBottom)
    }

    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = normalize_target(target.into(), self.relation);
        self
    }

    pub fn with_relation(mut self, relation: LayoutRelation) -> Self {
        self.relation = relation;
        self.target = normalize_target(std::mem::take(&mut self.target), relation);
        self
    }

    pub fn with_offset(mut self, offset: i32) -> Self {
        self.offset = offset;
        self
    }
}

/// A deterministic collection of object placement rules.
///
/// The internal `BTreeMap` makes both resolution and serialization independent
/// of insertion order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Layout {
    specs: BTreeMap<String, LayoutSpec>,
}
impl Layout {
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts or replaces a rule. Canvas targets are canonicalized to `""`.
    pub fn insert(&mut self, id: impl Into<String>, mut spec: LayoutSpec) -> Option<LayoutSpec> {
        spec.target = normalize_target(spec.target, spec.relation);
        self.specs.insert(id.into(), spec)
    }

    pub fn remove(&mut self, id: &str) -> Option<LayoutSpec> {
        self.specs.remove(id)
    }

    pub fn get(&self, id: &str) -> Option<&LayoutSpec> {
        self.specs.get(id)
    }

    pub fn contains(&self, id: &str) -> bool {
        self.specs.contains_key(id)
    }

    pub fn len(&self) -> usize {
        self.specs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    pub fn clear(&mut self) {
        self.specs.clear();
    }

    /// Iterates in deterministic ID order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &LayoutSpec)> {
        self.specs.iter().map(|(id, spec)| (id.as_str(), spec))
    }

    /// Resolves every rule to its top-left position.
    ///
    /// An object target may name another laid-out object, in which case its
    /// resolved rectangle uses `own_size`, or an ID present in `rects`, which is
    /// treated as a fixed external anchor. A laid-out rule takes precedence over
    /// an external rectangle with the same ID, ensuring dependency cycles are
    /// still detected.
    pub fn resolve(
        &self,
        rects: &BTreeMap<String, Rect>,
        canvas: Rect,
        own_size: (u16, u16),
    ) -> Result<BTreeMap<String, (i32, i32)>, LayoutError> {
        let mut resolver = Resolver {
            layout: self,
            rects,
            canvas,
            own_size,
            visiting: BTreeSet::new(),
            stack: Vec::new(),
            done: BTreeSet::new(),
            positions: BTreeMap::new(),
        };
        for id in self.specs.keys() {
            resolver.resolve_one(id)?;
        }
        Ok(resolver.positions)
    }

    /// Resolves and visits positions in deterministic ID order.
    pub fn apply(
        &self,
        rects: &BTreeMap<String, Rect>,
        canvas: Rect,
        own_size: (u16, u16),
        mut update: impl FnMut(&str, (i32, i32)),
    ) -> Result<(), LayoutError> {
        let positions = self.resolve(rects, canvas, own_size)?;
        for (id, position) in positions {
            update(&id, position);
        }
        Ok(())
    }

    /// Parses the documented practical subset of the text layout language.
    pub fn parse(input: &str) -> Result<Self, LayoutError> {
        let mut layout = Self::new();
        for (index, raw) in input.lines().enumerate() {
            let line = index + 1;
            let text = strip_comment(raw).trim();
            if text.is_empty() {
                continue;
            }
            let (id, rest) = text.split_once(':').ok_or_else(|| {
                LayoutError::parse(line, "expected `id: relation` (missing colon)")
            })?;
            let id = id.trim();
            if id.is_empty() {
                return Err(LayoutError::parse(line, "object ID cannot be empty"));
            }
            if id.chars().any(char::is_whitespace) {
                return Err(LayoutError::parse(
                    line,
                    format!("object ID `{id}` must be a single token"),
                ));
            }
            if layout.specs.contains_key(id) {
                return Err(LayoutError::parse(
                    line,
                    format!("duplicate object ID `{id}`"),
                ));
            }
            let tokens: Vec<&str> = rest.split_whitespace().collect();
            if tokens.is_empty() {
                return Err(LayoutError::parse(line, "missing layout relation"));
            }
            let spec = parse_spec(&tokens, line)?;
            layout.insert(id, spec);
        }
        Ok(layout)
    }
}

impl FromStr for Layout {
    type Err = LayoutError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        Self::parse(input)
    }
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (id, spec) in &self.specs {
            let target = if spec.target.is_empty() {
                "@"
            } else {
                spec.target.as_str()
            };
            let offset = offset_suffix(spec.offset);
            match spec.relation {
                LayoutRelation::InsideLeft => writeln!(f, "{id}: inside left of {target}{offset}")?,
                LayoutRelation::InsideRight => {
                    writeln!(f, "{id}: inside right of {target}{offset}")?
                }
                LayoutRelation::InsideTop => writeln!(f, "{id}: inside top of {target}{offset}")?,
                LayoutRelation::InsideBottom => {
                    writeln!(f, "{id}: inside bottom of {target}{offset}")?
                }
                LayoutRelation::Before => writeln!(f, "{id}: left of {target}{offset}")?,
                LayoutRelation::After => writeln!(f, "{id}: right of {target}{offset}")?,
                LayoutRelation::Above => writeln!(f, "{id}: above {target}{offset}")?,
                LayoutRelation::Below => writeln!(f, "{id}: below {target}{offset}")?,
                LayoutRelation::Center => writeln!(f, "{id}: center of {target}{offset}")?,
                LayoutRelation::Same { axis: None } => {
                    writeln!(f, "{id}: same as {target}{offset}")?
                }
                LayoutRelation::Same {
                    axis: Some(LayoutDirection::Horizontal),
                } => writeln!(f, "{id}: same x as {target}{offset}")?,
                LayoutRelation::Same {
                    axis: Some(LayoutDirection::Vertical),
                } => writeln!(f, "{id}: same y as {target}{offset}")?,
                LayoutRelation::At { x, y } => writeln!(f, "{id}: at {x} {y}{offset}")?,
            }
        }
        Ok(())
    }
}

/// Errors from dependency resolution or text parsing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayoutError {
    UnknownTarget {
        object: String,
        target: String,
    },
    Cycle {
        /// The cycle path, including the repeated first ID.
        path: Vec<String>,
    },
    Parse {
        /// One-based source line.
        line: usize,
        message: String,
    },
}

impl LayoutError {
    fn parse(line: usize, message: impl Into<String>) -> Self {
        Self::Parse {
            line,
            message: message.into(),
        }
    }
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTarget { object, target } => {
                write!(
                    f,
                    "layout object `{object}` references unknown target `{target}`"
                )
            }
            Self::Cycle { path } => write!(f, "layout cycle: {}", path.join(" -> ")),
            Self::Parse { line, message } => write!(f, "layout line {line}: {message}"),
        }
    }
}

impl std::error::Error for LayoutError {}

struct Resolver<'a> {
    layout: &'a Layout,
    rects: &'a BTreeMap<String, Rect>,
    canvas: Rect,
    own_size: (u16, u16),
    visiting: BTreeSet<String>,
    stack: Vec<String>,
    done: BTreeSet<String>,
    positions: BTreeMap<String, (i32, i32)>,
}

impl Resolver<'_> {
    fn resolve_one(&mut self, id: &str) -> Result<(i32, i32), LayoutError> {
        if let Some(position) = self.positions.get(id) {
            return Ok(*position);
        }
        if self.visiting.contains(id) {
            let start = self.stack.iter().position(|entry| entry == id).unwrap_or(0);
            let mut path = self.stack[start..].to_vec();
            path.push(id.to_owned());
            return Err(LayoutError::Cycle { path });
        }
        let spec =
            self.layout
                .specs
                .get(id)
                .cloned()
                .ok_or_else(|| LayoutError::UnknownTarget {
                    object: id.to_owned(),
                    target: id.to_owned(),
                })?;

        self.visiting.insert(id.to_owned());
        self.stack.push(id.to_owned());

        let position = match spec.relation {
            LayoutRelation::At { x, y } => (x.saturating_add(spec.offset), y),
            relation => {
                let target = if is_canvas_target(&spec.target) {
                    self.canvas
                } else if self.layout.specs.contains_key(&spec.target) {
                    let (x, y) = self.resolve_one(&spec.target)?;
                    Rect::new(x, y, self.own_size.0, self.own_size.1)
                } else if let Some(rect) = self.rects.get(&spec.target) {
                    *rect
                } else {
                    return Err(LayoutError::UnknownTarget {
                        object: id.to_owned(),
                        target: spec.target.clone(),
                    });
                };
                place(relation, target, self.own_size, spec.offset, self.canvas)
            }
        };

        self.stack.pop();
        self.visiting.remove(id);
        self.done.insert(id.to_owned());
        self.positions.insert(id.to_owned(), position);
        Ok(position)
    }
}

fn place(
    relation: LayoutRelation,
    target: Rect,
    own: (u16, u16),
    offset: i32,
    canvas: Rect,
) -> (i32, i32) {
    let own_width = i32::from(own.0);
    let own_height = i32::from(own.1);
    let right = target.x.saturating_add(i32::from(target.width));
    let bottom = target.y.saturating_add(i32::from(target.height));
    match relation {
        LayoutRelation::InsideLeft => (target.x.saturating_add(offset), target.y),
        LayoutRelation::InsideRight => (
            right.saturating_sub(own_width).saturating_add(offset),
            target.y,
        ),
        LayoutRelation::InsideTop => (target.x, target.y.saturating_add(offset)),
        LayoutRelation::InsideBottom => (
            target.x,
            bottom.saturating_sub(own_height).saturating_add(offset),
        ),
        LayoutRelation::Before => (
            target.x.saturating_sub(own_width).saturating_add(offset),
            target.y,
        ),
        LayoutRelation::After => (right.saturating_add(offset), target.y),
        LayoutRelation::Above => (
            target.x,
            target.y.saturating_sub(own_height).saturating_add(offset),
        ),
        LayoutRelation::Below => (target.x, bottom.saturating_add(offset)),
        LayoutRelation::Center => (
            center(target.x, target.width, own.0).saturating_add(offset),
            center(target.y, target.height, own.1),
        ),
        LayoutRelation::Same { axis: None } => (target.x.saturating_add(offset), target.y),
        LayoutRelation::Same {
            axis: Some(LayoutDirection::Horizontal),
        } => (target.x.saturating_add(offset), canvas.y),
        LayoutRelation::Same {
            axis: Some(LayoutDirection::Vertical),
        } => (canvas.x, target.y.saturating_add(offset)),
        LayoutRelation::At { x, y } => (x.saturating_add(offset), y),
    }
}

fn center(start: i32, container: u16, item: u16) -> i32 {
    let remaining = i32::from(container).saturating_sub(i32::from(item));
    start.saturating_add(remaining / 2)
}

fn is_canvas_target(target: &str) -> bool {
    target.is_empty() || target == "@"
}

fn normalize_target(mut target: String, relation: LayoutRelation) -> String {
    if is_canvas_target(&target) || matches!(relation, LayoutRelation::At { .. }) {
        target.clear();
    }
    target
}

/// Horizontal alignment inside a container.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Align {
    Start,
    Center,
    End,
}

/// Returns the signed offset that aligns `item` inside `container`.
pub fn align_offset(container: u16, item: u16, align: Align) -> i32 {
    let remaining = i32::from(container) - i32::from(item);
    match align {
        Align::Start => 0,
        Align::Center => remaining / 2,
        Align::End => remaining,
    }
}

fn strip_comment(line: &str) -> &str {
    let hash = line.find('#');
    let dash = line.find("--");
    let cut = match (hash, dash) {
        (Some(a), Some(b)) => a.min(b),
        (Some(a), None) | (None, Some(a)) => a,
        (None, None) => line.len(),
    };
    &line[..cut]
}

fn offset_suffix(offset: i32) -> String {
    match offset {
        0 => String::new(),
        value if value > 0 => format!(" + {value}"),
        value => format!(" - {}", value.unsigned_abs()),
    }
}

fn parse_spec(tokens: &[&str], line: usize) -> Result<LayoutSpec, LayoutError> {
    let keyword = tokens
        .first()
        .copied()
        .ok_or_else(|| LayoutError::parse(line, "missing layout relation"))?;
    match keyword {
        "at" => parse_at(tokens, line),
        "inside" | "outside" => parse_edge(tokens, line),
        "left" | "right" | "top" | "bottom" | "above" | "below" | "center" | "same" => {
            parse_relative(tokens, line)
        }
        other => Err(LayoutError::parse(
            line,
            format!("unknown layout keyword `{other}`"),
        )),
    }
}

fn parse_at(tokens: &[&str], line: usize) -> Result<LayoutSpec, LayoutError> {
    if tokens.len() < 3 {
        return Err(LayoutError::parse(
            line,
            "`at` expects an x and y coordinate",
        ));
    }
    let x = parse_i32(tokens[1], line, "x coordinate")?;
    let y = parse_i32(tokens[2], line, "y coordinate")?;
    let offset = parse_offset(&tokens[3..], line)?;
    Ok(LayoutSpec::at(x, y).with_offset(offset))
}

fn parse_edge(tokens: &[&str], line: usize) -> Result<LayoutSpec, LayoutError> {
    let keyword = tokens
        .first()
        .copied()
        .ok_or_else(|| LayoutError::parse(line, "missing layout relation"))?;
    let inside = keyword == "inside";
    let edge = tokens
        .get(1)
        .ok_or_else(|| LayoutError::parse(line, "expected left, right, top, or bottom"))?;
    let relation = match (inside, *edge) {
        (true, "left") => LayoutRelation::InsideLeft,
        (true, "right") => LayoutRelation::InsideRight,
        (true, "top") => LayoutRelation::InsideTop,
        (true, "bottom") => LayoutRelation::InsideBottom,
        (false, "left") => LayoutRelation::Before,
        (false, "right") => LayoutRelation::After,
        (false, "top") => LayoutRelation::Above,
        (false, "bottom") => LayoutRelation::Below,
        _ => {
            return Err(LayoutError::parse(
                line,
                format!("unknown edge `{edge}`; expected left, right, top, or bottom"),
            ));
        }
    };
    if tokens.get(2) != Some(&"of") {
        return Err(LayoutError::parse(
            line,
            format!("expected `of` after `{keyword} {edge}`"),
        ));
    }
    let target = target_at(tokens, 3, line)?;
    let offset = parse_offset(&tokens[4..], line)?;
    Ok(LayoutSpec::new(target, relation).with_offset(offset))
}

fn parse_relative(tokens: &[&str], line: usize) -> Result<LayoutSpec, LayoutError> {
    match tokens[0] {
        "left" | "right" | "top" | "bottom" => {
            let relation = match tokens[0] {
                "left" => LayoutRelation::Before,
                "right" => LayoutRelation::After,
                "top" => LayoutRelation::Above,
                "bottom" => LayoutRelation::Below,
                _ => {
                    return Err(LayoutError::parse(
                        line,
                        format!("unknown layout keyword `{}`", tokens[0]),
                    ));
                }
            };
            if tokens.get(1) != Some(&"of") {
                return Err(LayoutError::parse(
                    line,
                    format!("expected `of` after `{}`", tokens[0]),
                ));
            }
            let target = target_at(tokens, 2, line)?;
            let offset = parse_offset(&tokens[3..], line)?;
            Ok(LayoutSpec::new(target, relation).with_offset(offset))
        }
        "above" | "below" => {
            let relation = if tokens[0] == "above" {
                LayoutRelation::Above
            } else {
                LayoutRelation::Below
            };
            let target_index = if tokens.get(1) == Some(&"of") { 2 } else { 1 };
            let target = target_at(tokens, target_index, line)?;
            let offset = parse_offset(&tokens[target_index + 1..], line)?;
            Ok(LayoutSpec::new(target, relation).with_offset(offset))
        }
        "center" => {
            if tokens.get(1) != Some(&"of") {
                return Err(LayoutError::parse(line, "expected `of` after `center`"));
            }
            let target = target_at(tokens, 2, line)?;
            let offset = parse_offset(&tokens[3..], line)?;
            Ok(LayoutSpec::new(target, LayoutRelation::Center).with_offset(offset))
        }
        "same" => parse_same(tokens, line),
        other => Err(LayoutError::parse(
            line,
            format!("unknown layout keyword `{other}`"),
        )),
    }
}

fn parse_same(tokens: &[&str], line: usize) -> Result<LayoutSpec, LayoutError> {
    let (axis, as_index) = match tokens.get(1) {
        Some(&"as") => (None, 1),
        Some(&"x") => (Some(LayoutDirection::Horizontal), 2),
        Some(&"y") => (Some(LayoutDirection::Vertical), 2),
        _ => {
            return Err(LayoutError::parse(
                line,
                "expected `as`, `x as`, or `y as` after `same`",
            ));
        }
    };
    if tokens.get(as_index) != Some(&"as") {
        return Err(LayoutError::parse(line, "expected `as` in `same` relation"));
    }
    let target = target_at(tokens, as_index + 1, line)?;
    let offset = parse_offset(&tokens[as_index + 2..], line)?;
    Ok(LayoutSpec::new(target, LayoutRelation::Same { axis }).with_offset(offset))
}

fn target_at<'a>(tokens: &[&'a str], index: usize, line: usize) -> Result<&'a str, LayoutError> {
    tokens
        .get(index)
        .copied()
        .filter(|target| !target.is_empty())
        .ok_or_else(|| LayoutError::parse(line, "missing target object ID"))
}

fn parse_offset(tokens: &[&str], line: usize) -> Result<i32, LayoutError> {
    match tokens {
        [] => Ok(0),
        [value] => parse_i32(value, line, "offset"),
        [sign @ ("+" | "-"), value] => {
            let text = format!("{sign}{value}");
            text.parse::<i32>()
                .map_err(|_| LayoutError::parse(line, format!("invalid offset `{sign} {value}`")))
        }
        _ => Err(LayoutError::parse(
            line,
            format!("unexpected trailing tokens `{}`", tokens.join(" ")),
        )),
    }
}

fn parse_i32(token: &str, line: usize, what: &str) -> Result<i32, LayoutError> {
    token
        .parse::<i32>()
        .map_err(|_| LayoutError::parse(line, format!("invalid {what} `{token}`")))
}
#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: i32, y: i32, width: u16, height: u16) -> Rect {
        Rect::new(x, y, width, height)
    }

    fn external(id: &str, x: i32, y: i32, width: u16, height: u16) -> BTreeMap<String, Rect> {
        BTreeMap::from([(id.to_owned(), rect(x, y, width, height))])
    }

    fn resolve(
        layout: &Layout,
        rects: &BTreeMap<String, Rect>,
        canvas: Rect,
        own: (u16, u16),
    ) -> BTreeMap<String, (i32, i32)> {
        layout.resolve(rects, canvas, own).unwrap()
    }

    #[test]
    fn inside_relations_use_the_matching_target_edge() {
        let rects = external("target", 10, 20, 10, 6);
        let canvas = rect(0, 0, 100, 50);
        let mut layout = Layout::new();
        layout.insert("left", LayoutSpec::inside_left("target"));
        layout.insert("right", LayoutSpec::inside_right("target"));
        layout.insert("top", LayoutSpec::inside_top("target"));
        layout.insert("bottom", LayoutSpec::inside_bottom("target"));

        let positions = resolve(&layout, &rects, canvas, (4, 2));
        assert_eq!(positions["left"], (10, 20));
        assert_eq!(positions["right"], (16, 20));
        assert_eq!(positions["top"], (10, 20));
        assert_eq!(positions["bottom"], (10, 24));
    }

    #[test]
    fn outside_relations_are_adjacent_and_align_the_other_axis() {
        let rects = external("target", 10, 20, 6, 4);
        let canvas = rect(0, 0, 100, 50);
        let mut layout = Layout::new();
        layout.insert("before", LayoutSpec::before("target"));
        layout.insert("after", LayoutSpec::after("target"));
        layout.insert("above", LayoutSpec::above("target"));
        layout.insert("below", LayoutSpec::below("target"));

        let positions = resolve(&layout, &rects, canvas, (3, 2));
        assert_eq!(positions["before"], (7, 20));
        assert_eq!(positions["after"], (16, 20));
        assert_eq!(positions["above"], (10, 18));
        assert_eq!(positions["below"], (10, 24));
    }

    #[test]
    fn center_same_and_absolute_relations() {
        let rects = external("target", 10, 20, 10, 6);
        let canvas = rect(0, 0, 100, 50);
        let mut layout = Layout::new();
        layout.insert("center", LayoutSpec::center_of("target"));
        layout.insert("same", LayoutSpec::same_as("target"));
        layout.insert(
            "same_x",
            LayoutSpec::same_axis("target", LayoutDirection::Horizontal),
        );
        layout.insert(
            "same_y",
            LayoutSpec::same_axis("target", LayoutDirection::Vertical),
        );
        layout.insert("at", LayoutSpec::at(-4, 7));

        let positions = resolve(&layout, &rects, canvas, (4, 2));
        assert_eq!(positions["center"], (13, 22));
        assert_eq!(positions["same"], (10, 20));
        assert_eq!(positions["same_x"], (10, 0));
        assert_eq!(positions["same_y"], (0, 20));
        assert_eq!(positions["at"], (-4, 7));
    }

    #[test]
    fn offsets_move_only_the_relations_primary_axis() {
        let rects = external("target", 10, 20, 10, 6);
        let canvas = rect(0, 0, 100, 50);
        let mut layout = Layout::new();
        layout.insert(
            "inside_left",
            LayoutSpec::inside_left("target").with_offset(2),
        );
        layout.insert(
            "inside_right",
            LayoutSpec::inside_right("target").with_offset(-2),
        );
        layout.insert(
            "inside_top",
            LayoutSpec::inside_top("target").with_offset(3),
        );
        layout.insert(
            "inside_bottom",
            LayoutSpec::inside_bottom("target").with_offset(-3),
        );
        layout.insert("before", LayoutSpec::before("target").with_offset(2));
        layout.insert("after", LayoutSpec::after("target").with_offset(-2));
        layout.insert("above", LayoutSpec::above("target").with_offset(2));
        layout.insert("below", LayoutSpec::below("target").with_offset(-2));
        layout.insert("center", LayoutSpec::center_of("target").with_offset(2));
        layout.insert("same", LayoutSpec::same_as("target").with_offset(2));
        layout.insert("at", LayoutSpec::at(4, 5).with_offset(3));

        let positions = resolve(&layout, &rects, canvas, (4, 2));
        assert_eq!(positions["inside_left"], (12, 20));
        assert_eq!(positions["inside_right"], (14, 20));
        assert_eq!(positions["inside_top"], (10, 23));
        assert_eq!(positions["inside_bottom"], (10, 21));
        assert_eq!(positions["before"], (8, 20));
        assert_eq!(positions["after"], (18, 20));
        assert_eq!(positions["above"], (10, 20));
        assert_eq!(positions["below"], (10, 24));
        assert_eq!(positions["center"], (15, 22));
        assert_eq!(positions["same"], (12, 20));
        assert_eq!(positions["at"], (7, 5));
    }

    #[test]
    fn canvas_targets_accept_empty_and_at_sign() {
        let canvas = rect(5, 6, 20, 10);
        let mut layout = Layout::new();
        layout.insert("center", LayoutSpec::center_of(""));
        layout.insert("after_sign", LayoutSpec::after("@"));
        layout.insert("before_sign", LayoutSpec::before("@"));
        layout.insert("at", LayoutSpec::at(1, 2));

        let positions = resolve(&layout, &BTreeMap::new(), canvas, (4, 2));
        assert_eq!(positions["center"], (13, 10));
        assert_eq!(positions["after_sign"], (25, 6));
        assert_eq!(positions["before_sign"], (1, 6));
        assert_eq!(positions["at"], (1, 2));
    }

    #[test]
    fn resolution_of_chains_is_independent_of_insertion_order() {
        let canvas = rect(0, 0, 100, 40);
        let build_a = || {
            let mut layout = Layout::new();
            layout.insert("root", LayoutSpec::inside_left("@"));
            layout.insert("middle", LayoutSpec::after("root"));
            layout.insert("leaf", LayoutSpec::after("middle"));
            layout
        };
        let build_b = || {
            let mut layout = Layout::new();
            layout.insert("leaf", LayoutSpec::after("middle"));
            layout.insert("middle", LayoutSpec::after("root"));
            layout.insert("root", LayoutSpec::inside_left("@"));
            layout
        };

        let a = resolve(&build_a(), &BTreeMap::new(), canvas, (2, 2));
        let b = resolve(&build_b(), &BTreeMap::new(), canvas, (2, 2));
        assert_eq!(a, b);
        assert_eq!(a["root"], (0, 0));
        assert_eq!(a["middle"], (2, 0));
        assert_eq!(a["leaf"], (4, 0));
    }

    #[test]
    fn laid_out_targets_take_precedence_over_external_rects() {
        let mut layout = Layout::new();
        layout.insert("anchor", LayoutSpec::after("@"));
        layout.insert("item", LayoutSpec::after("anchor"));
        let rects = external("anchor", 500, 500, 10, 10);

        let positions = resolve(&layout, &rects, rect(4, 5, 20, 10), (3, 2));
        assert_eq!(positions["anchor"], (24, 5));
        assert_eq!(positions["item"], (27, 5));
    }

    #[test]
    fn unknown_targets_are_clear_errors() {
        let mut layout = Layout::new();
        layout.insert("a", LayoutSpec::before("missing"));
        let error = layout
            .resolve(&BTreeMap::new(), rect(0, 0, 10, 10), (1, 1))
            .unwrap_err();
        assert_eq!(
            error,
            LayoutError::UnknownTarget {
                object: "a".to_owned(),
                target: "missing".to_owned(),
            }
        );
        assert!(error.to_string().contains("unknown target `missing`"));

        let mut chain = Layout::new();
        chain.insert("a", LayoutSpec::before("b"));
        chain.insert("b", LayoutSpec::center_of("absent"));
        let error = chain
            .resolve(&BTreeMap::new(), rect(0, 0, 10, 10), (1, 1))
            .unwrap_err();
        assert_eq!(
            error,
            LayoutError::UnknownTarget {
                object: "b".to_owned(),
                target: "absent".to_owned(),
            }
        );
    }

    #[test]
    fn cycles_and_self_cycles_are_reported_with_paths() {
        let mut self_cycle = Layout::new();
        self_cycle.insert("a", LayoutSpec::after("a"));
        assert_eq!(
            self_cycle
                .resolve(&BTreeMap::new(), rect(0, 0, 20, 20), (1, 1))
                .unwrap_err(),
            LayoutError::Cycle {
                path: vec!["a".to_owned(), "a".to_owned()]
            }
        );

        let mut pair = Layout::new();
        pair.insert("a", LayoutSpec::after("b"));
        pair.insert("b", LayoutSpec::before("a"));
        let error = pair
            .resolve(&BTreeMap::new(), rect(0, 0, 20, 20), (1, 1))
            .unwrap_err();
        assert_eq!(
            error,
            LayoutError::Cycle {
                path: vec!["a".to_owned(), "b".to_owned(), "a".to_owned()]
            }
        );
        assert!(error.to_string().contains("a -> b -> a"));

        // A rule always wins over an external rectangle with the same ID.
        let mut precedence = Layout::new();
        precedence.insert("loop_a", LayoutSpec::after("loop_b"));
        precedence.insert("loop_b", LayoutSpec::after("loop_a"));
        assert!(matches!(
            precedence.resolve(&external("loop_a", 0, 0, 1, 1), rect(0, 0, 20, 20), (1, 1)),
            Err(LayoutError::Cycle { .. })
        ));
    }

    #[test]
    fn apply_visits_in_deterministic_id_order() {
        let mut layout = Layout::new();
        for id in ["c", "a", "b"] {
            layout.insert(id, LayoutSpec::inside_left("@"));
        }
        let mut visited = Vec::new();
        layout
            .apply(
                &BTreeMap::new(),
                rect(0, 0, 10, 10),
                (1, 1),
                |id, position| visited.push((id.to_owned(), position)),
            )
            .unwrap();
        assert_eq!(
            visited,
            vec![
                ("a".to_owned(), (0, 0)),
                ("b".to_owned(), (0, 0)),
                ("c".to_owned(), (0, 0)),
            ]
        );
    }
    #[test]
    fn parser_accepts_the_documented_subset() {
        let text = "\
# a full-line comment
obj: at 4 5
lefty: left of obj + 2
righty: right of obj - 3
abovey: above obj + 4
belowy: below of obj - 5
topy: top of obj + 6
bottomy: bottom of obj - 7
center: center of obj
same: same as obj
same_x: same x as obj
same_y: same y as obj
in_l: inside left of obj + 8
in_r: inside right of obj
in_t: inside top of obj
in_b: inside bottom of obj
out_l: outside left of obj
out_r: outside right of obj
out_t: outside top of obj
out_b: outside bottom of obj
canvas: center of @ -- canvas target
";
        let layout = Layout::parse(text).unwrap();
        assert_eq!(layout.len(), 20);

        assert_eq!(layout.get("obj").unwrap(), &LayoutSpec::at(4, 5));
        assert_eq!(
            layout.get("lefty").unwrap(),
            &LayoutSpec::before("obj").with_offset(2)
        );
        assert_eq!(
            layout.get("righty").unwrap(),
            &LayoutSpec::after("obj").with_offset(-3)
        );
        assert_eq!(
            layout.get("abovey").unwrap(),
            &LayoutSpec::above("obj").with_offset(4)
        );
        assert_eq!(
            layout.get("belowy").unwrap(),
            &LayoutSpec::below("obj").with_offset(-5)
        );
        assert_eq!(
            layout.get("topy").unwrap(),
            &LayoutSpec::above("obj").with_offset(6)
        );
        assert_eq!(
            layout.get("bottomy").unwrap(),
            &LayoutSpec::below("obj").with_offset(-7)
        );
        assert_eq!(layout.get("center").unwrap(), &LayoutSpec::center_of("obj"));
        assert_eq!(layout.get("same").unwrap(), &LayoutSpec::same_as("obj"));
        assert_eq!(
            layout.get("same_x").unwrap(),
            &LayoutSpec::same_axis("obj", LayoutDirection::Horizontal)
        );
        assert_eq!(
            layout.get("same_y").unwrap(),
            &LayoutSpec::same_axis("obj", LayoutDirection::Vertical)
        );
        assert_eq!(
            layout.get("in_l").unwrap(),
            &LayoutSpec::inside_left("obj").with_offset(8)
        );
        assert_eq!(
            layout.get("in_r").unwrap(),
            &LayoutSpec::inside_right("obj")
        );
        assert_eq!(layout.get("in_t").unwrap(), &LayoutSpec::inside_top("obj"));
        assert_eq!(
            layout.get("in_b").unwrap(),
            &LayoutSpec::inside_bottom("obj")
        );
        assert_eq!(layout.get("out_l").unwrap(), &LayoutSpec::before("obj"));
        assert_eq!(layout.get("out_r").unwrap(), &LayoutSpec::after("obj"));
        assert_eq!(layout.get("out_t").unwrap(), &LayoutSpec::above("obj"));
        assert_eq!(layout.get("out_b").unwrap(), &LayoutSpec::below("obj"));
        assert!(layout.get("canvas").unwrap().target.is_empty());
    }

    #[test]
    fn comments_can_be_trailing_or_whole_lines() {
        let layout: Layout = "\
# leading

a: right of b # trailing hash
b: at 1 2 -- trailing dashes
"
        .parse()
        .unwrap();
        assert_eq!(layout.len(), 2);
        assert_eq!(layout.get("a").unwrap().offset, 0);
        assert_eq!(layout.get("b").unwrap(), &LayoutSpec::at(1, 2));
    }

    #[test]
    fn parser_roundtrip_of_every_relation_is_canonical() {
        let mut layout = Layout::new();
        layout.insert(
            "absolute",
            LayoutSpec::at(i32::MIN, i32::MAX).with_offset(-7),
        );
        layout.insert("after", LayoutSpec::after("absolute").with_offset(3));
        layout.insert("before", LayoutSpec::before("absolute").with_offset(-2));
        layout.insert("above", LayoutSpec::above("absolute"));
        layout.insert("below", LayoutSpec::below("absolute").with_offset(1));
        layout.insert("center", LayoutSpec::center_of("absolute").with_offset(4));
        layout.insert("inside_l", LayoutSpec::inside_left("absolute"));
        layout.insert("inside_r", LayoutSpec::inside_right("absolute"));
        layout.insert("inside_t", LayoutSpec::inside_top("absolute"));
        layout.insert("inside_b", LayoutSpec::inside_bottom("absolute"));
        layout.insert("same", LayoutSpec::same_as("absolute").with_offset(-3));
        layout.insert(
            "same_x",
            LayoutSpec::same_axis("absolute", LayoutDirection::Horizontal),
        );
        layout.insert(
            "same_y",
            LayoutSpec::same_axis("absolute", LayoutDirection::Vertical),
        );
        layout.insert("canvas", LayoutSpec::center_of(""));
        layout.insert("canvas_sign", LayoutSpec::after("@").with_offset(i32::MIN));

        let text = layout.to_string();
        let parsed: Layout = text.parse().unwrap();
        assert_eq!(parsed, layout);
        assert_eq!(parsed.to_string(), text);
    }

    #[test]
    fn display_emits_sorted_canonical_specs() {
        let mut layout = Layout::new();
        layout.insert("c", LayoutSpec::inside_bottom("a"));
        layout.insert("a", LayoutSpec::after("b").with_offset(2));
        layout.insert("b", LayoutSpec::at(4, -5));
        assert_eq!(
            layout.to_string(),
            "a: right of b + 2\nb: at 4 -5\nc: inside bottom of a\n"
        );
        assert_eq!(layout.to_string().parse::<Layout>().unwrap(), layout);
    }

    #[test]
    fn parse_errors_carry_one_based_line_numbers() {
        fn error(input: &str) -> LayoutError {
            input.parse::<Layout>().unwrap_err()
        }

        assert_eq!(
            error("broken"),
            LayoutError::Parse {
                line: 1,
                message: "expected `id: relation` (missing colon)".to_owned()
            }
        );
        assert_eq!(
            error("\n# comment\nb: at 1"),
            LayoutError::Parse {
                line: 3,
                message: "`at` expects an x and y coordinate".to_owned()
            }
        );
        assert_eq!(
            error("a: left of"),
            LayoutError::Parse {
                line: 1,
                message: "missing target object ID".to_owned()
            }
        );
        assert_eq!(
            error("a: at x 2"),
            LayoutError::Parse {
                line: 1,
                message: "invalid x coordinate `x`".to_owned()
            }
        );
        assert_eq!(
            error("a: left of b + nope"),
            LayoutError::Parse {
                line: 1,
                message: "invalid offset `+ nope`".to_owned()
            }
        );
        assert!(matches!(
            error("a: at 1 2\na: at 3 4"),
            LayoutError::Parse { line: 2, .. }
        ));
        assert!(matches!(
            error(": at 1 2"),
            LayoutError::Parse { line: 1, .. }
        ));
        assert!(matches!(
            error("a b: at 1 2"),
            LayoutError::Parse { line: 1, .. }
        ));
        assert!(matches!(
            error("a: at 2147483648 0"),
            LayoutError::Parse { line: 1, .. }
        ));

        let message = error("\n\nunknown").to_string();
        assert!(message.contains("line 3"), "{message}");
    }

    #[test]
    fn parser_and_resolver_handle_extreme_signed_values() {
        let layout: Layout = "\
low: at -2147483648 2147483647 + 2147483647
high: left of low - 2147483648
"
        .parse()
        .unwrap();
        assert_eq!(
            layout.get("low").unwrap(),
            &LayoutSpec::at(i32::MIN, i32::MAX).with_offset(i32::MAX)
        );
        assert_eq!(
            layout.get("high").unwrap(),
            &LayoutSpec::before("low").with_offset(i32::MIN)
        );
        let positions = resolve(&layout, &BTreeMap::new(), rect(0, 0, 10, 10), (0, 0));
        assert_eq!(
            positions["low"],
            (i32::MIN.saturating_add(i32::MAX), i32::MAX)
        );
        assert_eq!(positions["high"], (i32::MIN, i32::MAX));
    }

    #[test]
    fn align_offset_covers_start_center_and_end() {
        assert_eq!(align_offset(10, 4, Align::Start), 0);
        assert_eq!(align_offset(10, 4, Align::Center), 3);
        assert_eq!(align_offset(10, 4, Align::End), 6);
        assert_eq!(align_offset(4, 4, Align::Center), 0);
        assert_eq!(align_offset(2, 5, Align::Start), 0);
        assert_eq!(align_offset(2, 5, Align::Center), -1);
        assert_eq!(align_offset(2, 5, Align::End), -3);
        assert_eq!(align_offset(0, u16::MAX, Align::End), -i32::from(u16::MAX));
    }

    #[test]
    fn extreme_rects_saturate_in_every_direction() {
        let max_target = rect(i32::MAX - 1, i32::MAX - 1, u16::MAX, u16::MAX);
        let mut layout = Layout::new();
        layout.insert("after", LayoutSpec::after("target"));
        layout.insert("below", LayoutSpec::below("target"));
        layout.insert("inside_right", LayoutSpec::inside_right("target"));
        layout.insert("center", LayoutSpec::center_of("target"));
        let positions = resolve(
            &layout,
            &BTreeMap::from([("target".to_owned(), max_target)]),
            rect(0, 0, 100, 50),
            (u16::MAX, u16::MAX),
        );
        assert_eq!(positions["after"], (i32::MAX, i32::MAX - 1));
        assert_eq!(positions["below"], (i32::MAX - 1, i32::MAX));
        assert_eq!(
            positions["inside_right"],
            (i32::MAX - i32::from(u16::MAX), i32::MAX - 1)
        );
        assert_eq!(positions["center"], (i32::MAX - 1, i32::MAX - 1));

        let min_target = rect(i32::MIN, i32::MIN, u16::MAX, u16::MAX);
        let mut layout = Layout::new();
        layout.insert("before", LayoutSpec::before("target"));
        layout.insert("above", LayoutSpec::above("target"));
        layout.insert("inside_right", LayoutSpec::inside_right("target"));
        layout.insert("center", LayoutSpec::center_of("target"));
        let positions = resolve(
            &layout,
            &BTreeMap::from([("target".to_owned(), min_target)]),
            rect(0, 0, 100, 50),
            (u16::MAX, u16::MAX),
        );
        assert_eq!(positions["before"], (i32::MIN, i32::MIN));
        assert_eq!(positions["above"], (i32::MIN, i32::MIN));
        assert_eq!(positions["inside_right"], (i32::MIN, i32::MIN));
        assert_eq!(positions["center"], (i32::MIN, i32::MIN));

        let mut offsets = Layout::new();
        offsets.insert("before", LayoutSpec::before("target").with_offset(i32::MIN));
        offsets.insert("after", LayoutSpec::after("target").with_offset(i32::MAX));
        offsets.insert("at", LayoutSpec::at(0, 0).with_offset(i32::MIN));
        let positions = resolve(
            &offsets,
            &external("target", 0, 0, 0, 0),
            rect(i32::MAX - 1, i32::MAX - 1, 10, 10),
            (0, 0),
        );
        assert_eq!(positions["before"], (i32::MIN, 0));
        assert_eq!(positions["after"], (i32::MAX, 0));
        assert_eq!(positions["at"], (i32::MIN, 0));

        let canvas = rect(i32::MAX - 1, i32::MAX - 1, 10, 10);
        let mut limits = Layout::new();
        limits.insert("after", LayoutSpec::after("@"));
        limits.insert("center", LayoutSpec::center_of("@"));
        limits.insert("before", LayoutSpec::before("@").with_offset(i32::MAX));
        let positions = resolve(&limits, &BTreeMap::new(), canvas, (0, 0));
        assert_eq!(positions["after"], (i32::MAX, i32::MAX - 1));
        assert_eq!(positions["center"], (i32::MAX, i32::MAX));
        assert_eq!(positions["before"], (i32::MAX, i32::MAX - 1));
    }

    #[test]
    fn specs_normalize_canvas_targets_and_relation_helpers_work() {
        assert_eq!(LayoutSpec::new("@", LayoutRelation::Before).target, "");
        assert_eq!(LayoutSpec::at(1, 2).target, "");
        assert_eq!(LayoutSpec::before("other").with_target("@").target, "");
        assert_eq!(
            LayoutSpec::before("other")
                .with_relation(LayoutRelation::At { x: 1, y: 2 })
                .target,
            ""
        );
        assert_eq!(
            LayoutRelation::inside(LayoutDirection::Horizontal),
            LayoutRelation::InsideLeft
        );
        assert_eq!(
            LayoutRelation::inside(LayoutDirection::Vertical),
            LayoutRelation::InsideTop
        );
        assert_eq!(
            LayoutRelation::outside(LayoutDirection::Horizontal),
            LayoutRelation::Before
        );
        assert_eq!(
            LayoutRelation::outside(LayoutDirection::Vertical),
            LayoutRelation::Above
        );

        let mut layout = Layout::new();
        layout.insert("a", LayoutSpec::before("b"));
        layout.insert(
            "b",
            LayoutSpec {
                target: "@".to_owned(),
                relation: LayoutRelation::After,
                offset: 0,
            },
        );
        assert!(layout.get("b").unwrap().target.is_empty());
        assert_eq!(
            layout.insert("a", LayoutSpec::center_of("b")),
            Some(LayoutSpec::before("b"))
        );
        assert_eq!(layout.len(), 2);
        assert!(layout.remove("a").is_some());
        assert!(!layout.contains("a"));
        layout.clear();
        assert!(layout.is_empty());
    }
}
