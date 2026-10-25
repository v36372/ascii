// SPDX-License-Identifier: Apache-2.0
//! Deterministic, injectable randomness used by games, shaders and particles.
//!
//! Prefer an owned [`Rand`] seeded explicitly for reproducible tests. The
//! [`default_rng`] helper is only for interactive programs.

use crate::graphics::Color;
use std::time::{SystemTime, UNIX_EPOCH};

/// Small, fast deterministic PRNG (SplitMix64).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rand {
    state: u64,
}

const GLYPHS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

impl Rand {
    pub const fn from_seed(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn new(seed: u64) -> Self {
        Self::from_seed(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform float in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Uniform float in `[0, 1)` with 53 bits of precision.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A uniform integer in `[min, max)`. Returns `min` when the range is empty.
    pub fn int(&mut self, min: i64, max: i64) -> i64 {
        if max <= min {
            return min;
        }
        let span = (max as i128 - min as i128) as u128;
        min + (u128::from(self.next_u64()) % span) as i64
    }

    /// A uniform integer in the inclusive range `[min, max]`.
    pub fn int_inclusive(&mut self, min: i64, max: i64) -> i64 {
        if max <= min {
            return min;
        }
        let span = (max as i128 - min as i128 + 1) as u128;
        min + (u128::from(self.next_u64()) % span) as i64
    }

    /// A uniform `usize` in `[0, bound)`. Returns zero for an empty range.
    pub fn index(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            self.next_u64() as usize % bound
        }
    }

    /// A uniform `f32` in `[min, max)`, returning `min` when invalid.
    pub fn range(&mut self, min: f32, max: f32) -> f32 {
        if !min.is_finite() || !max.is_finite() || max <= min {
            min
        } else {
            min + (max - min) * self.next_f32()
        }
    }

    /// A probability clamped to `[0, 1]`; `1` is always true, `0` always false.
    pub fn chance(&mut self, probability: f32) -> bool {
        let probability = if probability.is_finite() {
            probability.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.next_f32() < probability
    }

    pub fn coin_flip(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    pub fn choose<'a, T>(&mut self, values: &'a [T]) -> Option<&'a T> {
        if values.is_empty() {
            None
        } else {
            Some(&values[self.index(values.len())])
        }
    }

    pub fn choose_cloned<T: Clone>(&mut self, values: &[T]) -> Option<T> {
        self.choose(values).cloned()
    }

    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = self.index(i + 1);
            values.swap(i, j);
        }
    }

    /// A random sample of at most `count` items, preserving no particular order.
    pub fn sample<T: Clone>(&mut self, values: &[T], count: usize) -> Vec<T> {
        let mut result = values.to_vec();
        self.shuffle(&mut result);
        result.truncate(count.min(result.len()));
        result
    }

    pub fn byte(&mut self) -> u8 {
        self.next_u32() as u8
    }

    pub fn color(&mut self, colors: &[Color]) -> Option<Color> {
        self.choose(colors).copied()
    }

    pub fn random_color(&mut self) -> Color {
        Color::rgb(self.byte(), self.byte(), self.byte())
    }

    pub fn ascii_char(&mut self) -> char {
        char::from(33 + (self.next_u32() % 94) as u8)
    }

    pub fn printable_char(&mut self) -> char {
        char::from(32 + (self.next_u32() % 95) as u8)
    }

    /// Six uppercase alphanumeric characters.
    pub fn guid6(&mut self) -> String {
        (0..6)
            .map(|_| char::from(GLYPHS[self.index(GLYPHS.len())]))
            .collect()
    }

    /// A UUID-v4-shaped identifier derived from this generator.
    pub fn guid(&mut self) -> String {
        let mut bytes = [0u8; 16];
        for chunk in bytes.chunks_mut(8) {
            chunk.copy_from_slice(&self.next_u64().to_le_bytes()[..chunk.len()]);
        }
        bytes[6] = (bytes[6] & 0x0f) | 0x40;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        format!(
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
        )
    }
}

/// An RNG seeded from the clock and process ID for interactive programs.
pub fn default_rng() -> Rand {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    Rand::from_seed(now ^ (u64::from(std::process::id()) << 32))
}

/// Convenience wrappers over an explicit generator.
pub fn seq_rand<'a, T>(rng: &mut Rand, values: &'a [T]) -> Option<&'a T> {
    rng.choose(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_is_reproducible_and_sequence_changes() {
        let mut a = Rand::new(42);
        let mut b = Rand::new(42);
        let mut c = Rand::new(43);
        let seq_a: Vec<_> = (0..8).map(|_| a.next_u64()).collect();
        let seq_b: Vec<_> = (0..8).map(|_| b.next_u64()).collect();
        let seq_c: Vec<_> = (0..8).map(|_| c.next_u64()).collect();
        assert_eq!(seq_a, seq_b);
        assert_ne!(seq_a, seq_c);
        assert!(seq_a.windows(2).all(|w| w[0] != w[1]));
    }

    #[test]
    fn numeric_ranges_are_bounded_and_invalid_ranges_are_safe() {
        let mut rng = Rand::new(7);
        for _ in 0..1000 {
            let n = rng.int(-3, 4);
            assert!((-3..4).contains(&n));
            let n = rng.int_inclusive(5, 5);
            assert_eq!(n, 5);
            let x = rng.range(-1.0, 1.0);
            assert!((-1.0..1.0).contains(&x));
        }
        assert_eq!(rng.int(9, 9), 9);
        assert_eq!(rng.int(9, 2), 9);
        assert_eq!(rng.index(0), 0);
        assert!(rng.range(f32::NAN, 3.0).is_nan());
    }

    #[test]
    fn chance_is_clamped_and_coin_flip_is_unbiased_enough() {
        let mut rng = Rand::new(11);
        assert!(!rng.chance(0.0));
        assert!(rng.chance(1.0));
        assert!(!rng.chance(f32::NAN));
        let heads = (0..10_000).filter(|_| rng.coin_flip()).count();
        assert!((4500..5500).contains(&heads), "{heads}");
    }

    #[test]
    fn shuffle_is_a_permutation_and_sample_is_bounded() {
        let mut rng = Rand::new(3);
        let mut values: Vec<_> = (0..32).collect();
        rng.shuffle(&mut values);
        let mut sorted = values.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..32).collect::<Vec<_>>());
        assert_eq!(rng.sample(&values, 5).len(), 5);
        assert_eq!(rng.sample(&values, 100).len(), 32);
        assert!(rng.sample::<u8>(&[], 5).is_empty());
    }

    #[test]
    fn identifiers_have_documented_shape_and_are_unique() {
        let mut rng = Rand::new(99);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..1000 {
            let id = rng.guid6();
            assert_eq!(id.len(), 6);
            assert!(id
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit()));
            seen.insert(id);
            let guid = rng.guid();
            assert_eq!(guid.len(), 36);
            assert_eq!(guid.as_bytes()[8], b'-');
            assert_eq!(guid.as_bytes()[13], b'-');
            assert_eq!(guid.as_bytes()[14], b'4');
            assert!(matches!(guid.as_bytes()[19], b'8' | b'9' | b'a' | b'b'));
        }
        assert_eq!(seen.len(), 1000);
    }

    #[test]
    fn characters_and_colors_are_in_range() {
        let mut rng = Rand::new(5);
        for _ in 0..1000 {
            assert!((33..=126).contains(&(rng.ascii_char() as u32)));
            assert!((32..=126).contains(&(rng.printable_char() as u32)));
            let c = rng.random_color();
            assert!(!c.hex().is_empty());
        }
        let palette = [Color::BLACK, Color::WHITE];
        assert!(palette.contains(&rng.color(&palette).unwrap()));
        assert!(rng.color(&[]).is_none());
    }

    #[test]
    fn default_rng_is_seeded_and_usable() {
        let mut rng = default_rng();
        assert_ne!(rng.next_u64(), 0);
    }
}
