// SPDX-License-Identifier: Apache-2.0
use super::*;
use prefabs::*;
use std::f32::consts::FRAC_PI_2;

fn close(a: f32, b: f32) {
    assert!((a - b).abs() < 0.00001, "{a} != {b}");
}
fn glyph() -> Quad {
    Quad::glyph(AsciiGlyph::new('#').unwrap(), [4.0, 8.0], LinearRgba::WHITE)
}

#[test]
fn checked_characters_colors_and_depth() {
    for ch in [' ', '|', '~'] {
        assert_eq!(AsciiGlyph::new(ch).unwrap().character(), ch);
    }
    for ch in ['é', '\n', '\0', '\t', '\u{7f}'] {
        assert!(AsciiGlyph::new(ch).is_err());
    }
    for v in [f32::NAN, f32::INFINITY, -0.01, 1.01] {
        assert!(LinearRgba::new(v, 0.0, 0.0, 1.0).is_err());
        assert!(Depth::new(v).is_err());
    }
    assert_eq!(
        LinearRgba::from_srgb8(255, 0, 0).channels(),
        [1.0, 0.0, 0.0, 1.0]
    );
    close(
        LinearRgba::from_srgb8(128, 128, 128).channels()[0],
        0.2158605,
    );
    assert!(Depth::ground(-10.0).unwrap().value() < Depth::ground(10.0).unwrap().value());
    assert!(Depth::ground(f32::NAN).is_err());
}

#[test]
fn transform_and_group_composition_preserve_assets_and_anchor_depth() {
    let mut group = GlyphGroup::new();
    group.push(glyph().at(2.0, 3.0)).unwrap();
    let original = group.clone();
    let transform = Transform2d {
        translation: [10.0, 20.0],
        rotation: FRAC_PI_2,
        scale: 2.0,
    };
    let mut frame = SceneFrame::default();
    frame
        .add(&group, transform, Depth::ground(20.0).unwrap())
        .unwrap();
    let instance = frame.instances()[0];
    close(instance.position[0], 4.0);
    close(instance.position[1], 24.0);
    assert_eq!(instance.size, [8.0, 16.0]);
    close(instance.rotation, FRAC_PI_2);
    assert_eq!(instance.depth, Depth::ground(20.0).unwrap().value());
    assert_eq!(instance.kind, 0);
    assert_eq!(instance.glyph, u32::from(b'#'));
    assert_eq!(group, original);
    let bounds = group.bounds(transform).unwrap().unwrap();
    close(bounds.min[0], -4.0);
    close(bounds.max[0], 12.0);
    close(bounds.min[1], 20.0);
    close(bounds.max[1], 28.0);

    let mut combined = GlyphGroup::new();
    combined.extend(&group, transform).unwrap();
    let mut flat = SceneFrame::default();
    flat.add(
        &combined,
        Transform2d::default(),
        Depth::ground(20.0).unwrap(),
    )
    .unwrap();
    assert_eq!(frame.instances(), flat.instances());
    frame.clear();
    assert!(frame.instances().is_empty());
}

#[test]
fn text_orientation_spaces_and_backings_have_explicit_contracts() {
    let art = GlyphGroup::from_text("A \n B", [6.0, 10.0], LinearRgba::WHITE, None).unwrap();
    assert_eq!(art.quads().len(), 2);
    assert_eq!(art.quads()[1].position, [6.0, -10.0]);
    let backed = GlyphGroup::from_text(
        "A ",
        [6.0, 10.0],
        LinearRgba::WHITE,
        Some(LinearRgba::BLACK),
    )
    .unwrap();
    assert_eq!(backed.quads().len(), 3);
    let mut frame = SceneFrame::default();
    frame
        .add(&backed, Transform2d::default(), Depth::new(0.5).unwrap())
        .unwrap();
    let instances = frame.instances();
    assert_eq!(instances[0].kind, 1);
    assert!(instances[1].depth < instances[0].depth);
    assert_eq!(instances[2].kind, 1);
    for text in ["okay\n🦀", "x\r\ny", "x\ty"] {
        assert!(GlyphGroup::from_text(text, [1.0; 2], LinearRgba::WHITE, None).is_err());
    }
    assert!(GlyphGroup::from_text("", [0.0; 2], LinearRgba::WHITE, None).is_err());
}

#[test]
fn composition_errors_are_atomic_including_overflow_and_depth_offsets() {
    let mut group = GlyphGroup::new();
    group.push(glyph()).unwrap();
    let snapshot = group.clone();
    assert!(group
        .push(Quad {
            rotation: f32::NAN,
            ..glyph()
        })
        .is_err());
    assert_eq!(snapshot, group);
    let mut frame = SceneFrame::default();
    frame
        .add(&group, Transform2d::default(), Depth::new(0.5).unwrap())
        .unwrap();
    let original = frame.clone();
    for transform in [
        Transform2d {
            scale: 0.0,
            ..Transform2d::default()
        },
        Transform2d {
            scale: -1.0,
            ..Transform2d::default()
        },
        Transform2d {
            rotation: f32::INFINITY,
            ..Transform2d::default()
        },
        Transform2d {
            scale: f32::MAX,
            ..Transform2d::default()
        },
        Transform2d::at(f32::NAN, 0.0),
    ] {
        assert!(frame
            .add(&group, transform, Depth::new(0.5).unwrap())
            .is_err());
        assert!(group.extend(&snapshot, transform).is_err());
        assert_eq!(frame, original);
        assert_eq!(group, snapshot);
    }
    group.push(glyph().with_order(1)).unwrap();
    assert!(frame
        .add(&group, Transform2d::default(), Depth::new(0.0).unwrap())
        .is_err());
    assert_eq!(
        frame, original,
        "invalid second quad must not append the first"
    );
}

#[test]
fn camera_bounds_empty_groups_and_overhanging_objects() {
    let camera = Camera2d {
        center: [0.0, 10.0],
        view_height: 100.0,
    };
    let view = camera.bounds(2.0).unwrap();
    assert_eq!(
        view,
        Bounds2d {
            min: [-100.0, -40.0],
            max: [100.0, 60.0]
        }
    );
    for aspect in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(camera.bounds(aspect).is_err());
    }
    assert_eq!(
        GlyphGroup::new().bounds(Transform2d::default()).unwrap(),
        None
    );
    let mut group = GlyphGroup::new();
    group.push(glyph().at(0.0, 65.0)).unwrap();
    let overhang = group.bounds(Transform2d::at(0.0, -110.0)).unwrap().unwrap();
    assert!(!overhang.intersects(view));
    let overhang = group.bounds(Transform2d::at(0.0, -105.0)).unwrap().unwrap();
    assert!(
        overhang.intersects(view),
        "root is outside the view but geometry intersects it"
    );
    assert!(SceneFrame::default().knight.is_none());
}

#[test]
fn seeded_props_are_reusable_visuals_not_worlds() {
    let config = OrganicConfig::default();
    let tree = OrganicProp::new(config).unwrap();
    assert_eq!(tree, OrganicProp::new(config).unwrap());
    assert_ne!(
        tree,
        OrganicProp::new(OrganicConfig { seed: 9, ..config }).unwrap()
    );
    let still = tree.group(0.0, 0.0).unwrap();
    assert!(!still.quads().is_empty());
    assert_eq!(still, tree.group(0.0, 0.0).unwrap());
    assert_ne!(still, tree.group(1.0, 0.3).unwrap());
    let rock = OrganicProp::new(OrganicConfig {
        kind: OrganicKind::Rock,
        radius: 25.0,
        height: 40.0,
        ..config
    })
    .unwrap();
    assert_eq!(
        rock.group(0.0, 0.0).unwrap(),
        rock.group(10.0, 0.8).unwrap()
    );
    for radius in [0.0, 257.0, f32::NAN] {
        assert!(OrganicProp::new(OrganicConfig { radius, ..config }).is_err());
    }
    assert!(tree.group(f32::NAN, 0.0).is_err());
}

#[test]
fn grass_is_seeded_animated_and_root_sorted() {
    let config = GrassPatchConfig {
        count: 12,
        ..GrassPatchConfig::default()
    };
    let patch = GrassPatch::new(config).unwrap();
    assert_eq!(patch, GrassPatch::new(config).unwrap());
    assert_ne!(
        patch,
        GrassPatch::new(GrassPatchConfig { seed: 1, ..config }).unwrap()
    );
    let transform = Transform2d {
        translation: [7.0, 30.0],
        rotation: 0.2,
        scale: 1.5,
    };
    let mut a = SceneFrame::default();
    let mut b = SceneFrame::default();
    patch.add_to(&mut a, transform, 0.0, 0.0).unwrap();
    patch.add_to(&mut b, transform, 0.0, 0.0).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.instances().len(), config.count * 3);
    for blade in a.instances().as_chunks::<3>().0 {
        assert!(blade.iter().all(|segment| segment.depth == blade[0].depth));
    }
    assert_ne!(a.instances()[0].depth, a.instances()[3].depth);
    b.clear();
    patch.add_to(&mut b, transform, 1.0, 0.2).unwrap();
    assert_ne!(a, b);
    assert!(a
        .instances()
        .iter()
        .zip(b.instances())
        .all(|(a, b)| a.depth == b.depth));
    let snapshot = a.clone();
    assert!(patch.add_to(&mut a, transform, f32::NAN, 0.0).is_err());
    assert_eq!(a, snapshot);
    assert!(GrassPatch::new(GrassPatchConfig {
        count: 10_001,
        ..config
    })
    .is_err());
    assert!(GrassBlade {
        segments: 0,
        ..GrassBlade::default()
    }
    .group(0.0, LinearRgba::WHITE)
    .is_err());
}
