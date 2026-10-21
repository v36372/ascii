// SPDX-License-Identifier: Apache-2.0
//! Deterministic ASCII atlas, generated from the bundled Source Code Pro fonts.

use fontdue::{Font, FontSettings};

pub(super) const CELL_WIDTH: u32 = 40;
pub(super) const CELL_HEIGHT: u32 = 64;
pub(super) const WIDTH: u32 = CELL_WIDTH * 16;
pub(super) const HEIGHT: u32 = CELL_HEIGHT * 8;
const REGULAR: &[u8] =
    include_bytes!("../../../modules/cosplay/src/main/resources/fonts/SourceCodePro-Regular.ttf");
const BOLD: &[u8] =
    include_bytes!("../../../modules/cosplay/src/main/resources/fonts/SourceCodePro-Bold.ttf");

pub(super) struct Atlas {
    pub pixels: Vec<u8>,
    pub ramp: [[u32; 4]; 4],
}

pub(super) fn build() -> Result<Atlas, String> {
    let regular = Font::from_bytes(REGULAR, FontSettings::default())
        .map_err(|e| format!("regular atlas font: {e}"))?;
    let bold = Font::from_bytes(BOLD, FontSettings::default())
        .map_err(|e| format!("bold atlas font: {e}"))?;
    let mut pixels = vec![0; (WIDTH * HEIGHT) as usize];
    let mut ink = [0u64; 128];
    for code in 33u8..127 {
        let ch = char::from(code);
        let thin = "|/\\()'`,.-".contains(ch);
        let (font, size) = if thin {
            (&regular, 55.0)
        } else {
            (&bold, 46.0)
        };
        let (metrics, bitmap) = font.rasterize(ch, size);
        let line = font
            .horizontal_line_metrics(size)
            .ok_or("atlas font has no horizontal metrics")?;
        let baseline = (CELL_HEIGHT as f32 - (line.ascent - line.descent)) * 0.5 + line.ascent;
        let x0 = ((CELL_WIDTH as f32 - metrics.advance_width) * 0.5).round() as i32 + metrics.xmin;
        let y0 = baseline.round() as i32 - metrics.height as i32 - metrics.ymin;
        for y in 0..metrics.height {
            for x in 0..metrics.width {
                let xx = x0 + x as i32;
                let yy = y0 + y as i32;
                // A one-texel transparent gutter prevents adjacent-cell bleed.
                if xx < 1 || yy < 1 || xx >= CELL_WIDTH as i32 - 1 || yy >= CELL_HEIGHT as i32 - 1 {
                    continue;
                }
                let px = u32::from(code % 16) * CELL_WIDTH + xx as u32;
                let py = u32::from(code / 16) * CELL_HEIGHT + yy as u32;
                let alpha = bitmap[y * metrics.width + x];
                pixels[(py * WIDTH + px) as usize] = alpha;
                ink[code as usize] += u64::from(alpha);
            }
        }
    }
    let mut symbols = *b".-:=+*[]|I#%@8HM";
    symbols.sort_by_key(|code| (ink[*code as usize], *code));
    let mut ramp = [[0; 4]; 4];
    for (i, code) in symbols.into_iter().enumerate() {
        ramp[i / 4][i % 4] = u32::from(code);
    }
    Ok(Atlas { pixels, ramp })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ink(atlas: &Atlas, code: u32) -> u64 {
        (0..CELL_HEIGHT)
            .flat_map(|y| (0..CELL_WIDTH).map(move |x| (x, y)))
            .map(|(x, y)| {
                u64::from(
                    atlas.pixels[((code / 16 * CELL_HEIGHT + y) * WIDTH
                        + code % 16 * CELL_WIDTH
                        + x) as usize],
                )
            })
            .sum()
    }
    #[test]
    fn atlas_has_gutters_printable_ascii_and_coverage_sorted_ramp() {
        let atlas = build().unwrap();
        assert_eq!(atlas.pixels.len(), (WIDTH * HEIGHT) as usize);
        for code in 0..128 {
            if !(33..127).contains(&code) {
                assert_eq!(ink(&atlas, code), 0);
            } else {
                assert!(ink(&atlas, code) > 0, "ASCII {code}");
            }
            for y in 0..CELL_HEIGHT {
                assert_eq!(
                    atlas.pixels
                        [((code / 16 * CELL_HEIGHT + y) * WIDTH + code % 16 * CELL_WIDTH) as usize],
                    0
                );
            }
        }
        let coverage: Vec<_> = atlas
            .ramp
            .iter()
            .flatten()
            .map(|&code| ink(&atlas, code))
            .collect();
        assert!(coverage.windows(2).all(|w| w[0] <= w[1]));
    }
}
