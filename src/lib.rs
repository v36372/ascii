//! A terminal-first Rust port of CosPlay.
//!
//! The renderer and deterministic engine can run without a terminal. See the
//! `pong` binary for a small interactive game and headless smoke runner.
//!
//! ```
//! use cosplay::{
//!     engine::{Engine, FrameInfo, Scene, SceneObject},
//!     graphics::{Canvas, Color, Pixel, Rect},
//! };
//! use std::time::Duration;
//!
//! struct Star;
//! impl SceneObject for Star {
//!     fn rect(&self) -> Rect { Rect::new(1, 0, 1, 1) }
//!     fn render(&self, canvas: &mut Canvas<'_>, _: &FrameInfo) {
//!         canvas.draw_pixel(1, 0, 0, Pixel::new('*', Color::WHITE, None));
//!     }
//! }
//! let mut scene = Scene::new("demo", Pixel::new('.', Color::WHITE, Some(Color::BLACK)));
//! scene.add_object(Star);
//! let mut engine = Engine::new(4, 1);
//! engine.add_scene(scene)?;
//! engine.start("demo")?;
//! engine.tick(Duration::from_millis(33), &[])?;
//! assert_eq!(engine.screen().snapshot(), ".*..");
//! # Ok::<(), cosplay::engine::EngineError>(())
//! ```
#![forbid(unsafe_code)]

pub mod animation;
pub mod audio;
pub mod curve;
pub mod engine;
pub mod font;
pub mod graphics;
pub mod image;
pub mod layout;
pub mod particles;
pub mod pong;
pub mod random;
pub mod runtime;
pub mod shader;
pub mod sprites;
pub mod terminal;
pub mod text;
pub mod ui;

#[cfg(feature = "meadow")]
pub mod meadow;
