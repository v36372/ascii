// SPDX-License-Identifier: Apache-2.0
//! Interpolation, easing and cycling color gradients.

use crate::graphics::Color;

/// Linear interpolation between two scalars.
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    if !a.is_finite() || !b.is_finite() || !t.is_finite() {
        return 0.0;
    }
    a + (b - a) * t
}

/// Clamped linear interpolation between two scalars.
pub fn lerp_clamped(a: f32, b: f32, t: f32) -> f32 {
    lerp(a, b, t.clamp(0.0, 1.0))
}

/// Common easing curves over a normalized progress value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Easing {
    #[default]
    Linear,
    Smooth,
    EaseIn,
    EaseOut,
    EaseInOut,
    QuadraticIn,
    QuadraticOut,
}

impl Easing {
    pub fn apply(self, t: f32) -> f32 {
        let t = if t.is_finite() {
            t.clamp(0.0, 1.0)
        } else {
            0.0
        };
        match self {
            Self::Linear => t,
            Self::Smooth => t * t * (3.0 - 2.0 * t),
            Self::EaseIn => t * t,
            Self::EaseOut => 1.0 - (1.0 - t) * (1.0 - t),
            Self::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
                }
            }
            Self::QuadraticIn => t * t,
            Self::QuadraticOut => 1.0 - (1.0 - t) * (1.0 - t),
        }
    }
}

/// A deterministic polynomial through the supplied points.
///
/// The polynomial uses Lagrange interpolation. Duplicate or non-finite x
/// coordinates are rejected so evaluation never divides by zero.
#[derive(Clone, Debug, PartialEq)]
pub struct LagrangePolynomial {
    points: Vec<(f32, f32)>,
}

impl LagrangePolynomial {
    pub fn new(points: impl IntoIterator<Item = (f32, f32)>) -> Result<Self, CurveError> {
        let points: Vec<_> = points.into_iter().collect();
        if points.is_empty() {
            return Err(CurveError::NoPoints);
        }
        if points.iter().any(|(x, y)| !x.is_finite() || !y.is_finite()) {
            return Err(CurveError::NonFinite);
        }
        for (index, (x, _)) in points.iter().enumerate() {
            if points[..index].iter().any(|(other, _)| other == x) {
                return Err(CurveError::DuplicateX(*x));
            }
        }
        Ok(Self { points })
    }

    pub fn points(&self) -> &[(f32, f32)] {
        &self.points
    }

    pub fn eval(&self, x: f32) -> f32 {
        if !x.is_finite() {
            return 0.0;
        }
        let mut result = 0.0;
        for (i, (xi, yi)) in self.points.iter().enumerate() {
            let mut term = *yi;
            for (j, (xj, _)) in self.points.iter().enumerate() {
                if i != j {
                    term *= (x - xj) / (xi - xj);
                }
            }
            result += term;
        }
        result
    }
}

/// A color gradient sampled by a normalized position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gradient {
    stops: Vec<Color>,
}

impl Gradient {
    pub fn new(stops: impl IntoIterator<Item = Color>) -> Result<Self, CurveError> {
        let stops: Vec<_> = stops.into_iter().collect();
        if stops.is_empty() {
            return Err(CurveError::NoPoints);
        }
        Ok(Self { stops })
    }

    pub fn two(start: Color, end: Color, steps: usize) -> Self {
        Self {
            stops: Color::gradient(start, end, steps.max(1)),
        }
    }

    pub fn stops(&self) -> &[Color] {
        &self.stops
    }

    /// Samples the gradient at normalized `position` in `[0, 1]`.
    pub fn at(&self, position: f32) -> Color {
        if self.stops.len() == 1 || !position.is_finite() {
            return self.stops[0];
        }
        let position = position.clamp(0.0, 1.0);
        let scaled = position * (self.stops.len() - 1) as f32;
        let index = (scaled.floor() as usize).min(self.stops.len() - 2);
        self.stops[index].mixture(self.stops[index + 1], scaled - index as f32)
    }

    /// Samples cyclically: positions outside `[0, 1]` wrap around.
    pub fn cycle(&self, position: f32) -> Color {
        if !position.is_finite() {
            return self.stops[0];
        }
        let wrapped = position.rem_euclid(1.0);
        self.at(wrapped)
    }

    /// An iterator of exactly `steps` colors, including both ends.
    pub fn sample(&self, steps: usize) -> Vec<Color> {
        if steps == 0 {
            return Vec::new();
        }
        if steps == 1 {
            return vec![self.stops[0]];
        }
        (0..steps)
            .map(|index| self.at(index as f32 / (steps - 1) as f32))
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum CurveError {
    NoPoints,
    NonFinite,
    DuplicateX(f32),
}

impl std::fmt::Display for CurveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoPoints => write!(f, "curve needs at least one point"),
            Self::NonFinite => write!(f, "curve points must be finite"),
            Self::DuplicateX(x) => write!(f, "duplicate curve x coordinate: {x}"),
        }
    }
}

impl std::error::Error for CurveError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lagrange_interpolates_known_points_and_quadratic() {
        let line = LagrangePolynomial::new([(0.0, 1.0), (1.0, 3.0)]).unwrap();
        assert!((line.eval(0.5) - 2.0).abs() < 1e-6);
        let parabola = LagrangePolynomial::new([(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)]).unwrap();
        for x in [-1.0, 0.5, 1.5, 3.0] {
            assert!((parabola.eval(x) - x * x).abs() < 1e-4, "{x}");
        }
    }

    #[test]
    fn lagrange_rejects_bad_points() {
        assert_eq!(LagrangePolynomial::new([]), Err(CurveError::NoPoints));
        assert_eq!(
            LagrangePolynomial::new([(0.0, 1.0), (0.0, 2.0)]),
            Err(CurveError::DuplicateX(0.0))
        );
        assert_eq!(
            LagrangePolynomial::new([(f32::NAN, 1.0)]),
            Err(CurveError::NonFinite)
        );
    }

    #[test]
    fn easing_is_clamped_and_monotonic_at_key_points() {
        for easing in [
            Easing::Linear,
            Easing::Smooth,
            Easing::EaseIn,
            Easing::EaseOut,
            Easing::EaseInOut,
            Easing::QuadraticIn,
            Easing::QuadraticOut,
        ] {
            assert_eq!(easing.apply(-1.0), 0.0);
            assert_eq!(easing.apply(2.0), 1.0);
            assert!((0.0..=1.0).contains(&easing.apply(0.5)));
        }
    }

    #[test]
    fn gradient_samples_endpoints_cycles_and_validation() {
        let red = Color::rgb(255, 0, 0);
        let blue = Color::rgb(0, 0, 255);
        let gradient = Gradient::new([red, Color::WHITE, blue]).unwrap();
        assert_eq!(gradient.at(0.0), red);
        assert_eq!(gradient.at(1.0), blue);
        assert_eq!(gradient.cycle(1.0), red);
        assert_eq!(gradient.cycle(-0.5), gradient.at(0.5));
        let samples = gradient.sample(5);
        assert_eq!(samples.len(), 5);
        assert_eq!(samples[0], red);
        assert_eq!(samples[4], blue);
        assert_eq!(Gradient::two(red, blue, 4).stops().len(), 4);
        assert!(Gradient::new([]).is_err());
    }
}
