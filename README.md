# CosPlay Rust — native glyph meadow

The main graphical demo is a **2D top-down world with a real animated 3D knight
rendered as glyphs**, following the supplied `ascii-demo.zip` reference. This is
not a terminal approximation or a fullscreen ASCII post-processing filter.

The original Scala engine remains unchanged. This Rust implementation is not
source-compatible with Scala and does not yet implement its full feature set.

![Native Rust meadow capture](docs/meadow.png)

*Native Metal capture, seed 2024, initial frame, 1280×800.*

## Play the meadow

Install current stable Rust, then from the repository root:

```sh
cargo run --manifest-path rust/Cargo.toml --release --features meadow --bin meadow
```

This opens a native GPU window (Metal / Vulkan / DirectX, depending on platform).
Use **release mode** for interactive performance. No JVM, Node.js, browser, or
reference ZIP is needed at runtime.

| Control | Action |
|---|---|
| WASD / arrow keys | Move |
| Shift | Run |
| Mouse wheel / + / - | Zoom |
| Space | Pause / resume |
| R | Reset the seeded world |
| F1 | Toggle automatic walking |
| Esc | Quit |

It is an exploration/visual demo, not a finished RPG: combat, quests, networking,
and persistence are not implemented.

### Scene and rendering

- Dense procedural teal/sage grass with curved blades, rolling gusts, tip sheen,
  and displacement around the player.
- Flowers, green/red/teal trees, rocks, simulated moonlight and soft blob shadows.
- Ground-position depth ordering: foreground grass occludes the knight's legs;
  tree/rock backings hide the foliage behind them.
- Smooth movement, turning, camera follow, held-key input, running, and trunk/rock
  collision. The world is bounded; the grass tile wraps around the camera.
- Actual low-poly 3D knight geometry, animated limbs/cape, lighting, and a depth
  buffer. The character is rendered into a small offscreen texture, then sampled
  by a model-anchored glyph grid. There is no baked character sprite sheet.
- Independently positioned, rotated, scaled glyph quads; no terminal-cell snapping.

### Reproducible GPU captures

```sh
cargo run --manifest-path rust/Cargo.toml --release --features meadow --bin meadow -- \
  --headless --frames 120 --auto --size 1280x800 --output /tmp/meadow.png
```

`--headless` skips the window, **not the graphics device**: it requires a supported
adapter and renders the same scene/character pipelines as interactive mode. The
simulation uses fixed 1/30-second steps. `--frames 0` captures the initial scene.
The same seed/frame/adapter is reproducible; bit-identical pixels across different
GPU drivers are not guaranteed.

Other options: `--seed 2024`, `--density 0.25..2`, `--zoom 0.5..3`. Use `--help` for
all options. A positive `--frames N` without `--headless` closes the window after
N presented frames, useful for native-window smoke testing.

## Architecture

The optional `meadow` feature adds the spatial glyph renderer and graphical demo;
the default build remains small and terminal-only.

- `src/meadow/mod.rs`: renderer-neutral glyph instances, frame/pose/input data.
- `src/meadow/world.rs`: deterministic procedural world and simulation; no window
  or GPU calls. Static foliage data is cached, then culled and animated per frame.
- `src/meadow/renderer.rs` and helpers: glyph atlas, GPU resources, depth-tested
  instancing, offscreen 3D character pass, and PNG readback.
- `src/bin/meadow.rs`: native winit window, input, CLI, timing, resize and surface
  lifecycle, HUD and headless capture.

The spatial renderer deliberately does **not** route through the terminal
`Screen`/`Canvas`: a single fixed character cell cannot represent overlapping,
rotated, subpixel glyph sprites. Both paths are part of the `cosplay` library,
with separate frame representations. Grass animation currently runs on the CPU;
the GPU draws instances and renders/converts the 3D knight. This is not a claim of
an exact Three.js shader port or pixel-identical reproduction of the reference.

## Terminal foundation

Pong remains as a small regression/example for the original terminal milestone:

```sh
cargo run --manifest-path rust/Cargo.toml --release --bin pong
cargo run --manifest-path rust/Cargo.toml --bin pong -- --headless --frames 120 --size 60x20
```

Interactive Pong requires a true-color ANSI terminal of at least 40x16. W/S or
arrows move, Space/P pauses, R restarts, Q/Esc/Ctrl-C quits. Terminal modes are
restored on normal exit, errors and panics; SIGKILL/direct abort cannot run cleanup.

The terminal API includes RGB pixels, transparency/background inheritance,
Z-order, clipping, bounded lines/rectangles/text, framebuffer snapshots, diff
output, and an owned single-threaded scene engine. Objects have lifecycle,
update/monitor/render/shader callbacks and deferred structural commands. Each
drawing callback gets independent clipping. Update queries use a phase snapshot;
render/shader ordering uses post-monitor `(z, id)` metadata. This intentionally
replaces Scala's unrestricted mutable object references.

## Validation

```sh
cargo fmt --manifest-path rust/Cargo.toml --check
cargo test --manifest-path rust/Cargo.toml --all-features --all-targets --locked
cargo test --manifest-path rust/Cargo.toml --all-features --doc --locked
cargo clippy --manifest-path rust/Cargo.toml --all-features --all-targets --locked -- -D warnings
# Requires a usable graphics adapter; validates real rendered PNGs and live 3D poses.
cargo test --manifest-path rust/Cargo.toml --features meadow --lib meadow::renderer --locked -- --ignored --nocapture
cargo test --manifest-path rust/Cargo.toml --features meadow --test meadow_cli --locked -- --ignored --nocapture
# Unix/macOS: startup/input/resize/normal/error/panic terminal lifecycle.
cargo build --manifest-path rust/Cargo.toml --bin pong --example terminal_lifecycle --locked
python3 rust/tests/pty_smoke.py rust/target/debug/pong rust/target/debug/examples/terminal_lifecycle
```

Normal unit/CLI tests do not require a GPU or display. GPU execution is opt-in;
missing adapters produce an error rather than substituting a fake renderer.
Cross-platform CI checks compilation/tests; visual fidelity and interactive input
still need validation on real target machines.

## Scope and licenses

Further Scala parity (audio, FIGlet, REXPaint, particle/shader prefabs,
message/cache/timer APIs, layout language, and GUI debugger) remains deferred.
The meadow is the current visual/gameplay direction; Pong is only a foundation test.

The supplied reference credits Ansenjeo's ASCII game clips for its visual style.
The original project uses Apache-2.0 (`../LICENSE`). The embedded Source Code Pro
fonts already present in the repository use the SIL Open Font License, included
in `assets/SOURCE-CODE-PRO-LICENSE.txt` and available through `meadow --licenses`.

`PORTING.md` records the terminal milestone. `MEADOW_PLAN.md` describes the native
spatial rendering milestone.
