//! 2D Easom function.
//!
//! `f(x, y) = −cos(x)·cos(y)·exp(−((x − π)² + (y − π)²))`
//!
//! Smooth function that is almost entirely flat (≈ 0) except for a single
//! narrow, deep spike at `(x, y) = (π, π)` where `f = −1`. The tiny basin makes
//! it a hard target for solvers without good initialization or a global search
//! phase—a first-order method started far away sees a near-zero gradient and
//! drifts. Usual search domain is `x, y ∈ [-100, 100]`.

use core::marker::PhantomData;

use super::spec::{Dimensionality, HasSpec, ProblemSpec, Properties, Reference};
use crate::{CostFunction, Gradient};

/// Standard lower bound on each coordinate.
pub const STANDARD_LOWER: f64 = -100.0;
/// Standard upper bound on each coordinate.
pub const STANDARD_UPPER: f64 = 100.0;

/// Evaluates the Easom function at `x`. Requires `x.len() == 2`.
pub fn easom(x: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), 2);
    let pi = core::f64::consts::PI;
    let (a, b) = (x[0], x[1]);
    let e = (-((a - pi) * (a - pi) + (b - pi) * (b - pi))).exp();
    -a.cos() * b.cos() * e
}

/// Writes the Easom gradient at `x` into `out`. Both slices must have length 2.
pub fn easom_gradient(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), 2);
    debug_assert_eq!(out.len(), 2);
    let pi = core::f64::consts::PI;
    let (a, b) = (x[0], x[1]);
    let e = (-((a - pi) * (a - pi) + (b - pi) * (b - pi))).exp();
    // f = −cos(x)·cos(y)·E, with E = exp(−((x−π)² + (y−π)²)).
    // ∂f/∂x = E·cos(y)·[sin(x) + 2(x−π)·cos(x)]
    out[0] = e * b.cos() * (a.sin() + 2.0 * (a - pi) * a.cos());
    // ∂f/∂y = E·cos(x)·[sin(y) + 2(y−π)·cos(y)]
    out[1] = e * a.cos() * (b.sin() + 2.0 * (b - pi) * b.cos());
}

/// Pre-wrapped Easom problem (fixed 2D). Generic over the parameter backend
/// `P`; the default `P = Vec<f64>` lets you write `Easom::default()` for the
/// common case. Backend impls (`nalgebra::DVector<f64>`, `ndarray::Array1<f64>`,
/// `faer::Col<f64>`) are gated behind their respective features.
pub struct Easom<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> Easom<P> {
    /// Build a freshly typed problem instance. Pair with one of the
    /// backend-specific impl blocks (Vec, nalgebra, ndarray, faer).
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for Easom<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Catalog entry for this problem.
pub static EASOM_SPEC: ProblemSpec = ProblemSpec {
    name: "Easom",
    dim: Dimensionality::Fixed(2),
    properties: Properties {
        smooth: true,
        differentiable: true,
        convex: false,
        // A single minimum, but vast flat regions with near-zero gradient—
        // keep the conservative call rather than claiming clean unimodality.
        unimodal: false,
        separable: false,
        scalable: false,
    },
    references: &[Reference {
        citation: "Easom (1990)",
        title: "A survey of global optimization techniques",
        source: "M.Eng. thesis, University of Louisville, Louisville, KY",
        doi: None,
        url: None,
    }],
    description: "Mostly-flat surface with a single narrow, deep spike: global \
                  minimum at (x, y) = (π, π), value −1, surrounded by near-zero \
                  plateaus. Usual search domain is x, y ∈ [-100, 100]; needs \
                  good initialization or a global phase to locate the basin.",
};

impl<P> HasSpec for Easom<P> {
    const SPEC: &'static ProblemSpec = &EASOM_SPEC;
}

impl CostFunction for Easom<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(easom(x))
    }
}

impl Gradient for Easom<Vec<f64>> {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; x.len()];
        easom_gradient(x, &mut out);
        Ok(out)
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_impl {
    use super::{Easom, easom, easom_gradient};
    use crate::{CostFunction, Gradient};
    use nalgebra::DVector;

    impl CostFunction for Easom<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(easom(x.as_slice()))
        }
    }

    impl Gradient for Easom<DVector<f64>> {
        type Gradient = DVector<f64>;
        fn gradient(&self, x: &DVector<f64>) -> Result<DVector<f64>, std::convert::Infallible> {
            let mut out = DVector::zeros(x.len());
            easom_gradient(x.as_slice(), out.as_mut_slice());
            Ok(out)
        }
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::{Easom, easom, easom_gradient};
    use crate::{CostFunction, Gradient};
    use ndarray::Array1;

    impl CostFunction for Easom<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Array1<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(easom(x.as_slice().expect("Array1 is contiguous")))
        }
    }

    impl Gradient for Easom<Array1<f64>> {
        type Gradient = Array1<f64>;
        fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, std::convert::Infallible> {
            let mut out = Array1::zeros(x.len());
            easom_gradient(
                x.as_slice().expect("Array1 is contiguous"),
                out.as_slice_mut().expect("Array1 is contiguous"),
            );
            Ok(out)
        }
    }
}

#[cfg(feature = "faer")]
mod faer_impl {
    use super::Easom;
    use crate::{CostFunction, Gradient};
    use faer::Col;

    // faer's `Col` doesn't expose a `&[f64]` directly across all 0.24 APIs we
    // care about, so we evaluate elementwise here rather than routing through
    // the slice-based primitives.
    impl CostFunction for Easom<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            debug_assert_eq!(x.nrows(), 2);
            let pi = core::f64::consts::PI;
            let (a, b) = (x[0], x[1]);
            let e = (-((a - pi) * (a - pi) + (b - pi) * (b - pi))).exp();
            Ok(-a.cos() * b.cos() * e)
        }
    }

    impl Gradient for Easom<Col<f64>> {
        type Gradient = Col<f64>;
        fn gradient(&self, x: &Col<f64>) -> Result<Col<f64>, std::convert::Infallible> {
            debug_assert_eq!(x.nrows(), 2);
            let pi = core::f64::consts::PI;
            let (a, b) = (x[0], x[1]);
            let e = (-((a - pi) * (a - pi) + (b - pi) * (b - pi))).exp();
            let g0 = e * b.cos() * (a.sin() + 2.0 * (a - pi) * a.cos());
            let g1 = e * a.cos() * (b.sin() + 2.0 * (b - pi) * b.cos());
            Ok(Col::<f64>::from_fn(2, |i| if i == 0 { g0 } else { g1 }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn minimum_is_minus_one_at_pi_pi() {
        assert!((easom(&[PI, PI]) - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn known_value_at_origin() {
        // f(0, 0) = −cos(0)·cos(0)·exp(−(π² + π²)) = −exp(−2π²)
        let expected = -(-2.0 * PI * PI).exp();
        assert!((easom(&[0.0, 0.0]) - expected).abs() < 1e-15);
    }

    #[test]
    fn gradient_zero_at_minimum() {
        let mut g = vec![0.0; 2];
        easom_gradient(&[PI, PI], &mut g);
        for v in g {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn gradient_matches_finite_difference() {
        // Pick a point inside the active basin so the gradient is non-trivial.
        let x = [2.0, 3.0];
        let mut g = vec![0.0; x.len()];
        easom_gradient(&x, &mut g);
        let h = 1e-6;
        for i in 0..x.len() {
            let mut xp = x;
            let mut xm = x;
            xp[i] += h;
            xm[i] -= h;
            let fd = (easom(&xp) - easom(&xm)) / (2.0 * h);
            assert!((g[i] - fd).abs() < 1e-5, "i={i}, g={}, fd={fd}", g[i]);
        }
    }

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        let spec = <Easom<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(spec.name, "Easom");
        assert!(spec.properties.smooth);
        assert!(spec.properties.differentiable);
        assert!(matches!(spec.dim, Dimensionality::Fixed(2)));
        assert!(!spec.references.is_empty());
    }
}
