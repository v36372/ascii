# Building native glyph scenes

Enable the **`spatial`** Cargo feature and import `cosplay::spatial`. The engine
owns its renderer, atlas, shaders, frame types, controls, and application runner;
it does not import either demo. A spatial-only build does not compile the meadow
module. The old `meadow` feature enables `spatial` plus that demo for compatibility.
The terminal `SceneObject`/`Canvas` APIs remain separate.

```text
meadow scene ───┐
               ├──> spatial primitives + renderer + application runner
cyberpunk scene┘
```

## Start here

```rust
use cosplay::spatial::{
    Camera2d, Depth, GlyphGroup, LinearRgba, SceneFrame, Transform2d,
};

let art = GlyphGroup::from_text(
    " /\\\n/__\\\n|[]|", [8.0, 12.0],
    LinearRgba::from_srgb8(180, 210, 200), None,
)?;
let mut frame = SceneFrame::new(
    Camera2d { center: [0.0, 0.0], view_height: 320.0 },
    LinearRgba::from_srgb8(8, 12, 18),
);
frame.add(&art, Transform2d::at(-20.0, 24.0), Depth::ground(0.0)?)?;
```

This is frame construction only: no GPU is needed until rendering. Cache the art,
then build a frame for each simulation tick. Use `frame.clear()` to reuse its
instance allocation; it clears instances only, not camera/background/knight.

The complete second-scene recipe is **`examples/spatial_courtyard.rs`**. It builds
walls, reusable columns, lanterns, moving embers, grass patches, a tree, and rocks
without importing the meadow `World` or constructing raw `GlyphInstance`s:

```sh
cargo run --features spatial --example spatial_courtyard -- /tmp/courtyard.png 2
```

Arguments are output path and animation time in seconds. This captures a native
GPU-rendered PNG, not a terminal screenshot; a working graphics adapter is
required. The example's CPU-only test exercises deterministic scene construction.

## Building blocks

| API | Purpose |
| --- | --- |
| `AsciiGlyph::new(char)` | Checked printable ASCII atlas character |
| `LinearRgba::from_srgb8` / `new` | sRGB byte conversion / checked linear color |
| `Quad::glyph` / `Quad::solid` | Character or solid rectangular backing |
| `GlyphGroup::push` | Build a local-space visual asset |
| `GlyphGroup::from_text` | Multiline ASCII art, optional cell backings |
| `GlyphGroup::extend` | Compose another group through a local transform |
| `GlyphGroup::bounds` | Conservative bounds of all transformed/rotated quads |
| `Transform2d` | Translation, rotation, positive uniform scale |
| `Depth::ground` / `new` | Ground-based occlusion / explicit normalized depth |
| `SceneFrame::add` | Place a group without changing the cached source |
| `Camera2d::bounds` | View rectangle for an explicit target aspect ratio |
| `prefabs::GrassBlade` | Segmented blade with explicit bend and color |
| `prefabs::GrassPatch` | Seeded placements, each blade sorted by its own root |
| `prefabs::OrganicProp` | Cached seeded tree or rock surface; explicit time/wind |
| `GpuRenderer::render_scene` | Draw into a caller-owned compatible texture view |
| `GpuRenderer::capture_scene_png` | Render and capture without a window |
| `app::Application` | Game-owned update, reset, and frame callbacks |
| `app::run` | Shared window, held-key input, resize, pause/zoom, presentation |
| `app::prepare_capture` / `capture` | Fixed-step CPU simulation / GPU PNG capture |

## Contracts agents should rely on

### Geometry

- World coordinates: **+X right, +Y up**. No character-grid snapping.
- Quad position is its **center**, not its top-left corner.
- Rotation is **counterclockwise radians**.
- A group transform applies **scale → rotation → translation**.
- Group scale is positive and uniform. Nonuniform group scaling after rotating
  child quads would create shear, which the renderer cannot represent; it is
  deliberately not offered. Individual quad width/height remain independent.
- Groups are flat reusable assets, not a live parent/child scene graph.
  `extend()` copies/transforms another group; later source edits do not propagate.
- ASCII art's first cell is centered at `[0, 0]`; subsequent rows go **down (-Y)**.
  Spaces are holes unless a backing is requested. Only printable ASCII and `\n`
  are accepted; normalize CRLF, expand tabs, and choose ASCII replacements yourself.
- Empty groups are valid; their bounds are `None`. Invalid geometry is rejected
  even when applying a transform to an empty group.
- Bounds cover complete quads, not just origins. Use them for culling tall props
  whose ground point is outside the camera. These are visual bounds, not collision
  shapes. `Bounds2d::intersects` includes touching edges.

### Depth and backings

**Smaller depth is nearer.** `Depth::ground(y)` maps world Y to normalized depth:
objects lower on screen occlude objects farther up. It saturates at 0.01/0.99 for
extreme coordinates. Use explicit depths for HUDs or other projection conventions.

Every glyph of an object should share its **ground anchor's depth**, not its own
Y. This allows foliage in front of a character's feet to occlude its body correctly.
A group's transform does not silently recalculate depth: the caller supplies it.

`Quad::order` adds a small local offset: positive 1 is 0.000001 nearer. Backed
text uses order 0 for the backing and 1 for ink. Organic props use 1 and 2,
matching the meadow. Keep offsets small; they are not independent world layers.
Offset results outside 0..=1 return an error. Equal depth is last-write-wins.

Grass patches are the intentional exception to one anchor per group: `add_to()`
transforms each root and computes that blade's ground depth independently. All
segments of each blade still share its root depth.

### Color and transparency

Colors stored in instances are **linear** RGB, in 0..=1. Prefer `from_srgb8()` for
artist-picked RGB bytes. All four channels are finite and checked by `new()`.

The current renderer uses alpha blending **with depth writes**. Atlas ink below
alpha 0.5 is discarded; solid backings discard at alpha <= 0.001. This supports
cutout glyphs, but is **not order-independent translucent compositing**. Use opaque
backings and opaque glyph ink for predictable overlapping objects. Arbitrary
smoke/fade compositing will require a dedicated transparency pass. No general
lighting or shadow system is implied by a quad's color.

### Validation and determinism

- Invalid glyphs, NaN/infinite geometry, nonpositive sizes, invalid scales,
  transformation overflow, and invalid depth offsets return `SceneError`.
- `push`, `extend`, `SceneFrame::add`, and `GrassPatch::add_to` are atomic on error:
  a failed operation does not leave partially appended geometry.
- Public camera/knight fields are validated at render/capture time. Camera bounds
  can also be checked without a GPU via `Camera2d::bounds(aspect)`.
- Prefab geometry has explicit seeds and bounded construction sizes. Tree/rock
  sampling and blade segmentation are shared with the existing meadow, not
  independent copies of its algorithms.
- Prefabs have no hidden player or clock. Callers supply time/bend/wind. Rocks
  ignore time/wind; trees and grass animate. Shadows, light pools, collision,
  movement, and player displacement remain game-owned behavior.
- Same seed/configuration/time produces the same frame data on the same platform.
  PNG equality is tested on the same graphics adapter; cross-driver pixel identity
  is not promised.

## Rendering and optional characters

```rust
use cosplay::spatial::GpuRenderer;
let instance = wgpu::Instance::default();
let mut renderer = pollster::block_on(GpuRenderer::new(&instance, None))?;
renderer.capture_scene_png(1000, 700, &frame, std::path::Path::new("scene.png"))?;
```

For interactive games, implement `spatial::app::Application` and call `app::run`.
Do **not** copy a demo's window loop. Both `src/bin/meadow.rs` and
`src/bin/cyberpunk.rs` are now thin CLI/game adapters to the same runner.

The trait requires:

- `type Frame: FrameSource` — normally `SceneFrame`.
- `update(delta_seconds, Controls)` — game-owned simulation.
- `reset()` — game-owned reconstruction from its seed/configuration.
- `frame(View)` — scene/HUD generation using current pixel dimensions, zoom,
  pause state, and autoplay state. Errors propagate to the caller.

`AppOptions` configures title, initial size, zoom, autoplay, and an optional
presented-frame limit. It supports initial sizes 1..4096 by 1..2160 and zoom
0.5..3. The default bindings are WASD/arrows, Shift, Space, R, F1, wheel/+/- and
Escape. The runner clears held keys on focus loss/suspend, suppresses repeated
toggle actions, handles surface loss/resize, and tears down GPU/window state
before the event-loop connection closes. This is a common default controller,
not yet a configurable arbitrary key-binding/action system.

For capture, call `app::capture(&mut game, &options, steps, path)`; it uses the
same frame contract with N fixed 1/30-second updates. `prepare_capture` does just
the simulation/frame construction without a GPU. Neither function silently resets
the game. Zero steps captures its current state; steps are bounded at 100000.

For custom hosts, the lower-level renderer is still available: pass a surface to
`GpuRenderer::new`, configure it with `renderer.format`, and call `render_scene`
on each acquired texture view. `FrameSource`/`FrameView` also allow existing raw
frame buffers to render without copying into a `SceneFrame`.

A new `SceneFrame` contains **no knight**. Set
`frame.knight = Some(KnightPose { ... })` to opt into the existing live 3D character;
set it back to `None` to remove it. This is one specialized knight, not arbitrary
model loading or multi-character rendering. The optional `KnightPass` creates its
pipeline, mesh buffer, and offscreen color/depth targets **only on first use**.
Glyph-only scenes bind a one-texel placeholder instead. Once created, the pass is
cached; `None` skips mesh generation/drawing and never leaks a previous knight
into a new frame.

The old `meadow::MeadowFrame`, `GpuRenderer::render`, and `capture_png` continue to
work unchanged, with their original mandatory knight and background. The meadow
module implements the engine's `FrameSource` trait and re-exports old type/renderer
paths; the dependency points from demo to engine, never the reverse.

## Verification

```sh
cargo test --all-features --all-targets --locked
# Feature-isolation regression check: meadow is not compiled here.
cargo test --no-default-features --features spatial --all-targets --locked
cargo test --all-features --doc --locked
cargo clippy --all-features --all-targets --locked -- -D warnings
# Explicit GPU opt-in: blank background/no implicit hero, layering, text/backings,
# group rotation, deterministic capture, resizing and optional-character removal.
cargo test --features spatial --test spatial_gpu --locked -- --ignored --nocapture
cargo test --features spatial --lib spatial::renderer --locked -- --ignored --nocapture
cargo test --features spatial --test cyberpunk_cli --locked -- --ignored --nocapture
cargo test --features meadow --test meadow_cli --locked -- --ignored --nocapture
```

This is an authoring layer, not an ECS, scene editor, general model pipeline, or a
bridge for terminal widgets/shaders. New props and visual effects can now live in
game code, using public primitives rather than modifying the renderer.
