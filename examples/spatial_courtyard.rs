// SPDX-License-Identifier: Apache-2.0
//! A second GPU scene built ONLY with public authoring APIs. No World or raw
//! GlyphInstance construction. Run with `--features spatial`; optional positional
//! arguments are output PNG path and animation time in seconds.
use cosplay::spatial::{
    prefabs::{GrassPatch, GrassPatchConfig, OrganicConfig, OrganicKind, OrganicProp},
    AsciiGlyph, Camera2d, Depth, GlyphGroup, GpuRenderer, LinearRgba, Quad, SceneError, SceneFrame,
    Transform2d,
};
use std::path::PathBuf;

struct Courtyard {
    ground: GlyphGroup,
    wall: GlyphGroup,
    pillar: GlyphGroup,
    lantern: GlyphGroup,
    ember: GlyphGroup,
    grass: GrassPatch,
    tree: OrganicProp,
    rock: OrganicProp,
}
impl Courtyard {
    fn new() -> Result<Self, SceneError> {
        let stone = LinearRgba::from_srgb8(118, 138, 154);
        let dark = LinearRgba::from_srgb8(24, 32, 41);
        let floor = LinearRgba::from_srgb8(18, 27, 31);
        let mut ground = GlyphGroup::new();
        ground.push(Quad::solid([780.0, 380.0], floor))?;
        for row in 0..18 {
            for col in 0..39 {
                ground.push(
                    Quad::glyph(
                        AsciiGlyph::new(if (row + col) % 7 == 0 { ',' } else { '.' })?,
                        [8.0, 10.0],
                        LinearRgba::from_srgb8(45, 63, 63),
                    )
                    .at(col as f32 * 20.0 - 380.0, row as f32 * 20.0 - 170.0)
                    .with_order(1),
                )?;
            }
        }
        let masonry = GlyphGroup::from_text(
            "##  ###     ##  ####\n###################\n#_#___#___#___#___##\n##__#___#___#___#_#\n#_#___#___#___#___##",
            [9.0, 12.0], stone, Some(dark))?;
        let mut wall = GlyphGroup::new();
        // Raise art so the bottom row is at the asset's ground anchor.
        wall.extend(&masonry, Transform2d::at(0.0, 48.0))?;
        let column = GlyphGroup::from_text(
            "[==]\n || \n || \n || \n[__]",
            [9.0, 12.0],
            stone,
            Some(dark),
        )?;
        let mut pillar = GlyphGroup::new();
        pillar.extend(&column, Transform2d::at(-13.5, 48.0))?;
        let mut lantern = GlyphGroup::new();
        lantern.push(Quad::solid([4.0, 40.0], stone).at(0.0, 20.0))?;
        lantern.push(Quad::solid([15.0, 20.0], dark).at(0.0, 47.0))?;
        lantern.push(
            Quad::glyph(
                AsciiGlyph::new('*')?,
                [15.0, 20.0],
                LinearRgba::from_srgb8(255, 186, 80),
            )
            .at(0.0, 47.0)
            .with_order(1),
        )?;
        let mut ember = GlyphGroup::new();
        ember.push(
            Quad::glyph(
                AsciiGlyph::new('.')?,
                [7.0, 10.0],
                LinearRgba::from_srgb8(255, 157, 65),
            )
            .with_order(2),
        )?;
        Ok(Self {
            ground,
            wall,
            pillar,
            lantern,
            ember,
            grass: GrassPatch::new(GrassPatchConfig {
                size: [160.0, 45.0],
                count: 170,
                seed: 91,
                ..GrassPatchConfig::default()
            })?,
            tree: OrganicProp::new(OrganicConfig {
                radius: 57.0,
                height: 155.0,
                seed: 12,
                ..OrganicConfig::default()
            })?,
            rock: OrganicProp::new(OrganicConfig {
                kind: OrganicKind::Rock,
                radius: 25.0,
                height: 42.0,
                seed: 83,
                dark: LinearRgba::from_srgb8(45, 51, 65),
                lit: LinearRgba::from_srgb8(160, 170, 185),
            })?,
        })
    }

    fn frame(&self, time: f32) -> Result<SceneFrame, SceneError> {
        let mut frame = SceneFrame::new(
            Camera2d {
                center: [0.0, 35.0],
                view_height: 440.0,
            },
            LinearRgba::from_srgb8(8, 12, 18),
        );
        frame.add(&self.ground, Transform2d::default(), Depth::new(0.95)?)?;
        for x in [-260.0, 85.0] {
            frame.add(&self.wall, Transform2d::at(x, 90.0), Depth::ground(90.0)?)?;
        }
        for (x, y, angle) in [(-70.0, 85.0, 0.0), (70.0, 85.0, 0.0), (180.0, -105.0, -0.3)] {
            frame.add(
                &self.pillar,
                Transform2d {
                    rotation: angle,
                    ..Transform2d::at(x, y)
                },
                Depth::ground(y)?,
            )?;
        }
        frame.add(
            &self.tree.group(time, (time * 0.7).sin() * 0.14)?,
            Transform2d::at(-295.0, -10.0),
            Depth::ground(-10.0)?,
        )?;
        let rock = self.rock.group(time, 0.0)?;
        for (x, y, scale) in [
            (220.0, -45.0, 1.0),
            (-150.0, -90.0, 0.65),
            (270.0, 30.0, 0.7),
        ] {
            frame.add(
                &rock,
                Transform2d {
                    scale,
                    ..Transform2d::at(x, y)
                },
                Depth::ground(y)?,
            )?;
        }
        for (x, y) in [(-200.0, -110.0), (210.0, -70.0), (-250.0, 70.0)] {
            self.grass.add_to(
                &mut frame,
                Transform2d::at(x, y),
                time,
                (time * 0.8).sin() * 0.2,
            )?;
        }
        for x in [-55.0, 55.0] {
            frame.add(
                &self.lantern,
                Transform2d::at(x, 35.0),
                Depth::ground(35.0)?,
            )?;
            for i in 0..8 {
                let age = (time * 0.25 + i as f32 / 8.0).rem_euclid(1.0);
                frame.add(
                    &self.ember,
                    Transform2d::at(x + (age * 12.0 + i as f32).sin() * 10.0, 83.0 + age * 42.0),
                    Depth::ground(35.0)?,
                )?;
            }
        }
        // No knight, implicit player, global world, or renderer modifications.
        Ok(frame)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().unwrap_or_else(|| "courtyard.png".into()));
    let time: f32 = args.next().unwrap_or_else(|| "0".into()).parse()?;
    if !time.is_finite() || args.next().is_some() {
        return Err("usage: spatial_courtyard [output.png] [finite time in seconds]".into());
    }
    let scene = Courtyard::new()?;
    let frame = scene.frame(time)?;
    let instance = wgpu::Instance::default();
    let mut renderer = pollster::block_on(GpuRenderer::new(&instance, None))?;
    renderer.capture_scene_png(1200, 700, &frame, &path)?;
    println!(
        "{} ({} glyph/backing instances)",
        path.display(),
        frame.instances().len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_second_scene_uses_only_public_primitives_and_is_reproducible() {
        let courtyard = Courtyard::new().unwrap();
        let a = courtyard.frame(0.0).unwrap();
        assert_eq!(a, Courtyard::new().unwrap().frame(0.0).unwrap());
        assert!(a.knight.is_none());
        assert!(a.instances().len() > 2000);
        assert_ne!(a, courtyard.frame(1.0).unwrap());
        assert!(courtyard.frame(f32::NAN).is_err());
    }
}
