# Meadow rendering design

## Target

The supplied `ascii-demo.zip` defines the visual reference: a 2D top-down glyph
world with a real animated 3D character. It is not a terminal scene and not a
fullscreen ASCII filter. The native Rust implementation preserves those two
important properties, while leaving the Scala engine and terminal backend intact.

## Data flow

```text
held keys / fixed-step capture input
               |
         World::step
     player, velocity, yaw,
     walk phase, camera, time
               |
         World::frame
    culled glyph instances + HeroPose
               |
         GpuRenderer
        /           \
  glyph atlas    posed 3D knight triangles
                    |
              lit/depth-tested RGBA16F 58x56 target
        \           /
       glyph instancing pass
       + 29x28 live knight cells
               |
        window or PNG readback
```

## Why a separate spatial frame

The terminal `Screen` stores one glyph per fixed cell. The reference requires
floating-point positions, rotation, scale, overlapping glyphs and texture alpha.
Routing the meadow through `Screen` would lose that information before drawing.

`GlyphInstance` instead carries center/size/color/rotation/depth/ASCII index and
kind (glyph, solid backing, live knight cell). `MeadowFrame` owns instances,
camera/view height and a character pose. `World` is deterministic and independent
of GPU/window handles. The GPU renderer is part of the same `cosplay` library,
behind the optional `meadow` feature, but does not use the terminal scene API.

## Depth and coordinates

World x points right, y points up/further away. Instances carry root/ground depth:
`clamp(0.5 + ground_y * 0.00005, 0.01, 0.99)`; smaller values are nearer. The GPU
uses `LessEqual` depth testing. Glyphs sit slightly in front of their solid object
backings. All character cells share the feet's depth, allowing foreground grass
to cover legs without grass behind the character showing through its armor.

The camera is floating-point and smoothed. The player is bounded to +/-2800 world
units. Ground vegetation repeats in indexed 1800x1100 tiles; tree locations and
tree glyph geometry are generated once. Culling accounts for canopy height and
blade reach rather than only object roots.

## Character pipeline

The knight is constructed from low-poly cuboids, cylinders and sphere patches.
Per-frame posing changes limb rotations, body bounce, cape and character yaw.
Triangle normals, moon/rim/ambient lighting and a real depth buffer produce a
transparent 58x56 offscreen image. The glyph fragment shader averages each 2x2
texel block, selects a symbol by luminance, and uses coverage to discard empty
cells. The 29x28 grid is pinned to projected model feet. No character framebuffer
readback or baked animation sheet is used during ordinary rendering.

## Atlas, shading and performance

The existing Source Code Pro Regular/Bold fonts are embedded under the SIL OFL.
`fontdue` rasterizes an ASCII atlas; the character symbol ramp is sorted by
measured ink coverage. Source Code Pro replaces the reference's browser-resolved
font stack, so glyph shapes/coverage need not be pixel-identical.

Grass generation, wind deformation, analytical shadows and visibility currently
run on the CPU. Static tree height-field shading is cached. The GPU draws the
result as instances and runs the character's actual 3D render/conversion passes.
The reference's TSL/MaterialX noise and browser renderer are not reproduced
bit-for-bit. Colors are linear and the output is sRGB.

## Validation and boundaries

- Unit tests exercise deterministic generation, input/motion, collision, wrapping,
  culling, finite geometry, depth, mesh posing/normals, atlas and readback layout.
- Opt-in GPU tests exercise actual WGSL pipelines, live character animation,
  depth/alpha behavior, reproducible captures and nonblank meadow PNGs.
- `meadow --headless` renders the same pipeline without a window; it still needs
  a graphics adapter. Errors are surfaced rather than replacing the GPU renderer.
- The app clears held keys on focus loss, ignores one-shot actions from synthetic
  key events, handles zero-size windows/surface loss and avoids overdue timer spins.
- `rust/docs/meadow.png` is a native initial-frame capture, not the reference image.

This milestone is a traversable visual demo, not a complete RPG. Gameplay systems,
asset authoring/import, additional characters, post-processing and common scene
abstractions across the terminal/spatial paths remain future work.
