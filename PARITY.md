# CosPlay parity assessment

This document measures **functional parity** against the upstream Scala CosPlay
engine. It is not a claim of source compatibility, byte-identical rendering, or
complete content/tooling parity. "Complete" means the capability exists with a
usable native API and tests; "partial" means important capabilities work but a
documented subset is missing; "missing" means no implementation is provided.

## Score

| # | Capability family | Status | Native implementation |
|---|---|---|---|
| 1 | Color, pixel, rectangle and tuple math | Complete | `graphics`: RGB/HSB, gradients, 256-color mapping, hex, luma, safe rectangles and clipping |
| 2 | Z-ordered canvas drawing | Complete | Pixels, text, images, lines, vectors, circles, polylines, boxes, flood fill, art lines, capture |
| 3 | Styled text, markup and ASCII tables | Complete | `text`: nested markup, styled rows, wrapping, alignment, borders, tables |
| 4 | Character image operations | Complete | `image`: crop, pad, split, trim, resize, antialias, flips, stitching, background flood fill |
| 5 | Image formats | Complete | TXT, REXPaint CSV, gzipped multilayer XP (read), single-layer XP (write) |
| 6 | Fonts and FIGlet | Complete | `font`: system font, FLF parser, all fitting modes, **277 bundled fonts** |
| 7 | Scene runtime and lifecycle | Complete | `engine`: scenes, objects, lifecycle, camera, deterministic fixed-step tick, headless |
| 8 | Cache, timers and messages | Complete | `runtime` + `engine`: delayed caches, next-frame/delayed timers, typed messages |
| 9 | Animation | Complete | `animation`: time-based/filmstrip, loop/bounce, completion, named animation sprites |
| 10 | ASCII video | Complete | `animation`: video traits, frame timing, seek, loop, player events and sprites |
| 11 | Built-in sprites | Complete | `sprites`: image, static, label, animation, video, particle, bubble, keyboard, canvas, centered, tile map, singleton |
| 12 | CPU shaders and transitions | Complete | `shader`: fade, slide (all 13 directions), shimmer, sparkle, border, flashlight, star streaks, beat |
| 13 | Particle effects | Complete | `particles`: generic particles/emitters, deterministic confetti, callbacks, bounds |
| 14 | Declarative layout | Partial | Programmatic relations and practical parser subset; not the full ANTLR grammar |
| 15 | UI widgets | Partial | Panel, list box, text input, spacer, layout group; no complete focus traversal/scrollbars |
| 16 | Keyboard input | Partial | Arrows, text, modifiers, F1-F12, navigation keys; no mouse or terminal emulator input |
| 17 | Audio | Partial | WAV/AIFF/AU decode, playback, volume/speed/seek, beat envelope, owned registry; no balance, URL streaming or asynchronous fades |
| 18 | Native terminal backend | Complete | Raw mode, true color, diff output, resize, centering, panic/error restoration |
| 19 | Logging, statistics and metadata | Complete | `runtime`: leveled/throttled logs, frame statistics, game metadata |
| 20 | Randomness and curves | Complete | `random` and `curve`: reproducible RNG, GUIDs, shuffling, easing, Lagrange and gradients |
| 21 | Native GPU spatial renderer | Complete | Meadow glyph instancing, depth, 3D character pipeline, PNG capture (beyond the Scala backend) |
| 22 | Prefab art/video library | Missing | A few composable primitives are supplied, but the upstream 58 images and moon video are not bundled |
| 23 | Built-in emulator and GUI debugger | Missing | Requires a separate virtual terminal GUI and inspector |
| 24 | Cross-platform verification | Partial | Linux build/runtime/GPU/PTY tests pass; macOS/Windows still need real-machine validation |
| 25 | Documentation and validation | Complete | API docs, parity accounting, showcase, unit/integration/PTY/GPU test instructions |

Equal-weight score: **20.5 / 25 = 82%**. Complete engine capability families are
18/25; 5 are partial and 2 are absent. The 80% target is therefore met for
functional engine parity, not for every prefab asset or desktop tool.

## Deliberate differences

- The Rust API is idiomatic and owned; it is not a source-compatible translation
  of Scala names such as `CPCanvas` or `CPSceneObjectContext`.
- Rendering is deterministic and headless-first. GPU pixels are reproducible per
  adapter, not guaranteed bit-identical across drivers.
- Obvious upstream defects are not copied. For example, particle bounds are
  inclusive and "all emitters paused" matches its documentation.
- The layout language supports documented common relations rather than the
  complete ANTLR grammar.
- Visual parity with the reference ZIP is approximate; fonts, shader math,
  lighting and GPU rasterization differ.

## Remaining work beyond the 80% target

1. Bundle/port the upstream image, animation-atlas, video and logo-scene prefabs.
2. Implement the terminal emulator and GUI debugger.
3. Complete layout grammar, focus traversal, scrollbars and richer widgets.
4. Add audio balance/URL streaming and device-backed integration tests.
5. Validate interactive and GPU behavior on macOS and Windows.

## Verification

```sh
cargo fmt --check
cargo test --all-features --all-targets --locked
cargo test --all-features --doc --locked
cargo clippy --all-features --all-targets --locked -- -D warnings
cargo run --example terminal_showcase
```

Optional GPU and terminal-lifecycle checks remain documented in `README.md`.
