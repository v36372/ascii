# Rust port: terminal-first milestone

The Scala implementation remains the reference and is not modified. This milestone
implements a native character-cell renderer, scene runtime, terminal backend, and
playable Pong. It is not full API/feature parity. Audio, FIGlet, REXPaint, particles,
layout language, and desktop emulator/debugger are deferred.

## Implementation contract

Crate: `cosplay`, edition 2021, under `rust/`. Modules: `graphics`, `terminal`,
`engine`, `pong`. No unsafe code. Only `crossterm = "0.29"` and
`unicode-width = "0.2"` initially. Public APIs below are coordination boundaries.

### Graphics (`src/graphics.rs`)

- `Color { pub r: u8, pub g: u8, pub b: u8 }`: Copy/Eq, `const fn rgb(...)`,
  constants BLACK, WHITE.
- `Pixel { pub ch: char, pub fg: Color, pub bg: Option<Color> }`: Copy/Eq,
  `Pixel::new(ch, fg, bg)`, `Pixel::transparent()` (NUL sentinel),
  `is_transparent()`. Only single-column printable characters are rendered;
  invalid/control/wide/zero-width glyphs become '?' (except transparency sentinel).
- `Rect { pub x: i32, pub y: i32, pub width: u16, pub height: u16 }`: Copy/Eq,
  `new(x,y,width,height)`, `contains(x,y)`, `intersects(other)`,
  `intersection(other) -> Option<Rect>`. Half-open coordinates; empty rects don't
  intersect. Arithmetic must be safe at i32 extremes.
- `Cell { pub pixel: Pixel, pub z: i32 }`: Copy/Eq.
- `Screen::new(width:u16,height:u16,background:Pixel)`; `width()`, `height()`,
  `bounds()->Rect`, `get(x:i32,y:i32)->Option<&Cell>`, `clear()`,
  `canvas()->Canvas<'_>`, `snapshot()->String` (glyph rows joined by '\n', no trailing newline).
- `Canvas<'a>` borrows a Screen, `bounds()->Rect`,
  `set_clip(Rect)` (intersect screen bounds), `draw_pixel(x,y,z,pixel)`,
  `draw_text(x,y,z,text:&str,fg:Color,bg:Option<Color>)`,
  `fill_rect(rect,z,pixel)`, `draw_rect(rect,z,pixel)`,
  `draw_line(x0,y0,x1,y1,z,pixel)`.
- Larger z wins; equal z is last-write-wins. Transparent pixels do not occlude.
  Missing background inherits the already-composited cell background.
  Drawing outside screen/clip is ignored. Bound line/rectangle work to visible
  regions, not enormous offscreen coordinates. Tests cover these contracts.

### Terminal (`src/terminal.rs`)

- `Key`: Copy/Eq enum `Up, Down, Left, Right, Char(char), Enter, Escape, Space`.
- `InputEvent`: Copy/Eq enum `Key(Key), Resize(u16,u16), Quit`.
- `Terminal::new() -> std::io::Result<Self>` enters raw mode/alternate screen,
  hides cursor and disables wrapping. Reject non-TTY stdin/stdout cleanly.
- `size(&self)->io::Result<(u16,u16)>`.
- `poll_events(&mut self, timeout:Duration)->io::Result<Vec<InputEvent>>`;
  ignore release events, map Ctrl-C to Quit, drain ready events with a bound.
- `render(&mut self, screen:&Screen, camera:Rect, force:bool)->io::Result<()>`.
  Crop camera, center viewport when terminal is larger, clear exposed borders;
  output changed cells only, invalidate on size/viewport change. Never write an
  unsafe glyph even if public Pixel fields are constructed directly.
- `Drop` and explicit `restore(&mut self)->io::Result<()>` restore terminal state;
  rollback partial initialization. Provide `install_panic_hook()` for binary use
  (not called automatically by library). Restore before printing panic diagnostics.
- Keep ANSI diff generation separately testable via a writer/pure encoder; test
  unchanged frames, resize, camera offsets, background resets, error retry.

### Runtime and game (coordinator-owned)

An owned Engine, explicit scene ownership, frame time/input and command queues.
Single-threaded deterministic tick supports a headless backend. Start/activate,
update/monitor/render/shader, deactivate/stop are distinct lifecycle phases.
Scene/object mutations apply at frame boundaries; no global engine singleton.
Pong includes keyboard controls, computer opponent, scoring, pause/restart/quit,
resize handling and deterministic headless smoke mode. Input/time are injectable
for tests. No ECS framework required.

## Validation

`cargo fmt --check`, `cargo test --all-targets`, `cargo clippy --all-targets -- -D warnings`.
Headless Pong execution and framebuffer tests require no TTY or JVM. CI covers
Linux/macOS/Windows; interactive behavior must be verified separately on each.
