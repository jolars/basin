//! 2D Beale function.
//!
//! `f(x, y) = (1.5 − x + xy)² + (2.25 − x + xy²)² + (2.625 − x + xy³)²`
//!
//! Smooth multimodal 2D test function with a long, near-flat valley. Global
//! minimum at `(x, y) = (3, 0.5)` with `f = 0`. Usual search domain is
//! `x, y ∈ [-4.5, 4.5]`. Useful as a smooth, non-quadratic, fixed-2D
//! complement to Rosenbrock—first-order methods slow noticeably in the
//! flat region near the optimum.

use core::marker::PhantomData;

use super::spec::{Dimensionality, HasSpec, ProblemSpec, Properties, Reference};
use crate::{CostFunction, Gradient};

/// Evaluates the Beale function at `x`. Requires `x.len() == 2`.
pub fn beale(x: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), 2);
    let (a, b) = (x[0], x[1]);
    let t1 = 1.5 - a + a * b;
    let t2 = 2.25 - a + a * b * b;
    let t3 = 2.625 - a + a * b * b * b;
    t1 * t1 + t2 * t2 + t3 * t3
}

/// Writes the Beale gradient at `x` into `out`. Both slices must have length 2.
pub fn beale_gradient(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), 2);
    debug_assert_eq!(out.len(), 2);
    let (a, b) = (x[0], x[1]);
    let b2 = b * b;
    let b3 = b2 * b;
    let t1 = 1.5 - a + a * b;
    let t2 = 2.25 - a + a * b2;
    let t3 = 2.625 - a + a * b3;
    // ∂f/∂x = 2·t1·(y − 1) + 2·t2·(y² − 1) + 2·t3·(y³ − 1)
    out[0] = 2.0 * (t1 * (b - 1.0) + t2 * (b2 - 1.0) + t3 * (b3 - 1.0));
    // ∂f/∂y = 2x·(t1 + 2y·t2 + 3y²·t3)
    out[1] = 2.0 * a * (t1 + 2.0 * b * t2 + 3.0 * b2 * t3);
}

/// Pre-wrapped Beale problem (fixed 2D). Generic over the parameter backend
/// `P`; the default `P = Vec<f64>` lets you write `Beale::default()` for the
/// common case. Backend impls (`nalgebra::DVector<f64>`, `ndarray::Array1<f64>`,
/// `faer::Col<f64>`) are gated behind their respective features.
pub struct Beale<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> Beale<P> {
    /// Build a freshly typed problem instance. Pair with one of the
    /// backend-specific impl blocks (Vec, nalgebra, ndarray, faer).
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for Beale<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Catalog entry for this problem.
pub static BEALE_SPEC: ProblemSpec = ProblemSpec {
    name: "Beale",
    dim: Dimensionality::Fixed(2),
    properties: Properties {
        smooth: true,
        differentiable: true,
        convex: false,
        // Three saddle points exist in addition to the global minimum, but no
        // other strict local minima—keeping `unimodal: false` is the
        // conservative call given the literature's mixed usage of the term.
        unimodal: false,
        separable: false,
        scalable: false,
    },
    references: &[Reference {
        citation: "Beale (1958)",
        title: "On an iterative method for finding a local minimum of a function of more than one variable",
        source: "Technical Report 25, Statistical Techniques Research Group, Princeton University",
        doi: None,
        url: None,
    }],
    description: "Smooth 2D polynomial test function with a long, near-flat \
                  valley. Global minimum at (x, y) = (3, 0.5), value 0. \
                  Usual search domain is x, y ∈ [-4.5, 4.5]; first-order \
                  methods slow noticeably in the flat region near the optimum.",
};

impl<P> HasSpec for Beale<P> {
    const SPEC: &'static ProblemSpec = &BEALE_SPEC;
}

impl CostFunction for Beale<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(beale(x))
    }
}

impl Gradient for Beale<Vec<f64>> {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; x.len()];
        beale_gradient(x, &mut out);
        Ok(out)
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_impl {
    use super::{Beale, beale, beale_gradient};
    use crate::{CostFunction, Gradient};
    use nalgebra::DVector;

    impl CostFunction for Beale<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(beale(x.as_slice()))
        }
    }

    impl Gradient for Beale<DVector<f64>> {
        type Gradient = DVector<f64>;
        fn gradient(&self, x: &DVector<f64>) -> Result<DVector<f64>, std::convert::Infallible> {
            let mut out = DVector::zeros(x.len());
            beale_gradient(x.as_slice(), out.as_mut_slice());
            Ok(out)
        }
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::{Beale, beale, beale_gradient};
    use crate::{CostFunction, Gradient};
    use ndarray::Array1;

    impl CostFunction for Beale<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Array1<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(beale(x.as_slice().expect("Array1 is contiguous")))
        }
    }

    impl Gradient for Beale<Array1<f64>> {
        type Gradient = Array1<f64>;
        fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, std::convert::Infallible> {
            let mut out = Array1::zeros(x.len());
            beale_gradient(
                x.as_slice().expect("Array1 is contiguous"),
                out.as_slice_mut().expect("Array1 is contiguous"),
            );
            Ok(out)
        }
    }
}

#[cfg(feature = "faer")]
mod faer_impl {
    use super::Beale;
    use crate::{CostFunction, Gradient};
    use faer::Col;

    // faer's `Col` doesn't expose a `&[f64]` directly across all 0.24 APIs we
    // care about, so we evaluate elementwise here rather than routing through
    // the slice-based primitives.
    impl CostFunction for Beale<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            debug_assert_eq!(x.nrows(), 2);
            let (a, b) = (x[0], x[1]);
            let t1 = 1.5 - a + a * b;
            let t2 = 2.25 - a + a * b * b;
            let t3 = 2.625 - a + a * b * b * b;
            Ok(t1 * t1 + t2 * t2 + t3 * t3)
        }
    }

    impl Gradient for Beale<Col<f64>> {
        type Gradient = Col<f64>;
        fn gradient(&self, x: &Col<f64>) -> Result<Col<f64>, std::convert::Infallible> {
            debug_assert_eq!(x.nrows(), 2);
            let (a, b) = (x[0], x[1]);
            let b2 = b * b;
            let b3 = b2 * b;
            let t1 = 1.5 - a + a * b;
            let t2 = 2.25 - a + a * b2;
            let t3 = 2.625 - a + a * b3;
            let g0 = 2.0 * (t1 * (b - 1.0) + t2 * (b2 - 1.0) + t3 * (b3 - 1.0));
            let g1 = 2.0 * a * (t1 + 2.0 * b * t2 + 3.0 * b2 * t3);
            Ok(Col::<f64>::from_fn(2, |i| if i == 0 { g0 } else { g1 }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beale_minimum_is_zero_at_known_optimum() {
        assert!(beale(&[3.0, 0.5]).abs() < 1e-12);
    }

    #[test]
    fn beale_known_value_at_origin() {
        // f(0, 0) = 1.5² + 2.25² + 2.625² = 2.25 + 5.0625 + 6.890625 = 14.203125
        assert!((beale(&[0.0, 0.0]) - 14.203125).abs() < 1e-12);
    }

    #[test]
    fn gradient_zero_at_minimum() {
        let mut g = vec![0.0; 2];
        beale_gradient(&[3.0, 0.5], &mut g);
        for v in g {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let x = [-1.2, 0.7];
        let mut g = vec![0.0; x.len()];
        beale_gradient(&x, &mut g);

        let h = 1e-6;
        for i in 0..x.len() {
            let mut xp = x;
            let mut xm = x;
            xp[i] += h;
            xm[i] -= h;
            let fd = (beale(&xp) - beale(&xm)) / (2.0 * h);
            assert!((g[i] - fd).abs() < 1e-5, "i={i}, g={}, fd={fd}", g[i]);
        }
    }

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        let spec = <Beale<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(spec.name, "Beale");
        assert!(spec.properties.smooth);
        assert!(spec.properties.differentiable);
        assert!(matches!(spec.dim, Dimensionality::Fixed(2)));
        assert!(!spec.references.is_empty());
    }
}
