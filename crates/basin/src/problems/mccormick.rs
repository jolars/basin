//! 2D McCormick function.
//!
//! `f(x, y) = sin(x + y) + (x − y)² − 1.5·x + 2.5·y + 1`
//!
//! Smooth 2D test function with a single global minimum on its standard
//! search domain `x ∈ [-1.5, 4], y ∈ [-3, 4]`. The exact minimizer is
//! `(x*, y*) = (0.5 − π/3, −0.5 − π/3) ≈ (-0.54719, -1.54719)` with value
//! `f* = -√3/2 − π/3 ≈ -1.9133`. Mixes a quadratic basin with a sine ripple,
//! so it isn't globally convex but behaves nicely as a single-basin test
//! for first-order methods.
//!
//! The formula and recommended domain match the catalog entry in Adorio
//! (2005); see also Surjanovic & Bingham
//! (`https://www.sfu.ca/~ssurjano/mccorm.html`).

use core::marker::PhantomData;

use super::spec::{
    Dimensionality, HasSpec, ProblemSpec, Properties, Reference,
};
use crate::{CostFunction, Gradient};

/// Evaluates the McCormick function at `x`. Requires `x.len() == 2`.
pub fn mccormick(x: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), 2);
    let (a, b) = (x[0], x[1]);
    let d = a - b;
    (a + b).sin() + d * d - 1.5 * a + 2.5 * b + 1.0
}

/// Writes the McCormick gradient at `x` into `out`. Both slices must have
/// length 2.
pub fn mccormick_gradient(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), 2);
    debug_assert_eq!(out.len(), 2);
    let (a, b) = (x[0], x[1]);
    let c = (a + b).cos();
    let d = a - b;
    // ∂f/∂x = cos(x + y) + 2·(x − y) − 1.5
    out[0] = c + 2.0 * d - 1.5;
    // ∂f/∂y = cos(x + y) − 2·(x − y) + 2.5
    out[1] = c - 2.0 * d + 2.5;
}

/// Pre-wrapped McCormick problem (fixed 2D). Generic over the parameter
/// backend `P`; the default `P = Vec<f64>` lets you write
/// `McCormick::default()` for the common case. Backend impls
/// (`nalgebra::DVector<f64>`, `ndarray::Array1<f64>`, `faer::Col<f64>`) are
/// gated behind their respective features.
pub struct McCormick<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> McCormick<P> {
    /// Build a freshly typed problem instance. Pair with one of the
    /// backend-specific impl blocks (Vec, nalgebra, ndarray, faer).
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for McCormick<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Catalog entry for this problem.
pub static MCCORMICK_SPEC: ProblemSpec = ProblemSpec {
    name: "McCormick",
    dim: Dimensionality::Fixed(2),
    properties: Properties {
        smooth: true,
        differentiable: true,
        // The sin(x + y) term makes the Hessian indefinite at points where
        // sin(x + y) > 0, so f is not convex on the standard domain.
        convex: false,
        // On the standard search domain x ∈ [-1.5, 4], y ∈ [-3, 4] there is
        // a single strict local (and global) minimum.
        unimodal: true,
        // Coupling via sin(x + y) and (x − y)² blocks per-coordinate
        // decomposition.
        separable: false,
        scalable: false,
    },
    references: &[
        Reference {
            citation: "Adorio (2005)",
            title: "MVF — Multivariate Test Functions Library in C for Unconstrained Global Optimization",
            source: "Department of Mathematics, U.P. Diliman",
            doi: None,
            url: Some("http://www.geocities.ws/eadorio/mvf.pdf"),
        },
        Reference {
            citation: "Jamil & Yang (2013)",
            title: "A Literature Survey of Benchmark Functions For Global Optimisation Problems",
            source: "International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150–194",
            doi: Some("10.1504/IJMMNO.2013.055204"),
            url: Some("https://arxiv.org/abs/1308.4008"),
        },
    ],
    description: "Smooth 2D test function: f(x, y) = sin(x + y) + (x − y)² \
                  − 1.5·x + 2.5·y + 1. Standard search domain is \
                  x ∈ [-1.5, 4], y ∈ [-3, 4], on which it has a single \
                  global minimum at (0.5 − π/3, −0.5 − π/3) ≈ \
                  (-0.54719, -1.54719) with value −√3/2 − π/3 ≈ -1.9133. \
                  Not convex (sin term flips Hessian sign), but behaves as a \
                  single-basin test for first-order methods.",
};

impl<P> HasSpec for McCormick<P> {
    const SPEC: &'static ProblemSpec = &MCCORMICK_SPEC;
}

impl CostFunction for McCormick<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(mccormick(x))
    }
}

impl Gradient for McCormick<Vec<f64>> {
    type Gradient = Vec<f64>;
    fn gradient(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; x.len()];
        mccormick_gradient(x, &mut out);
        Ok(out)
    }
}

#[cfg(feature = "nalgebra_all")]
mod nalgebra_impl {
    use super::{McCormick, mccormick, mccormick_gradient};
    use crate::{CostFunction, Gradient};
    use nalgebra::DVector;

    impl CostFunction for McCormick<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &DVector<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(mccormick(x.as_slice()))
        }
    }

    impl Gradient for McCormick<DVector<f64>> {
        type Gradient = DVector<f64>;
        fn gradient(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, std::convert::Infallible> {
            let mut out = DVector::zeros(x.len());
            mccormick_gradient(x.as_slice(), out.as_mut_slice());
            Ok(out)
        }
    }
}

#[cfg(feature = "ndarray_all")]
mod ndarray_impl {
    use super::{McCormick, mccormick, mccormick_gradient};
    use crate::{CostFunction, Gradient};
    use ndarray::Array1;

    impl CostFunction for McCormick<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &Array1<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(mccormick(x.as_slice().expect("Array1 is contiguous")))
        }
    }

    impl Gradient for McCormick<Array1<f64>> {
        type Gradient = Array1<f64>;
        fn gradient(
            &self,
            x: &Array1<f64>,
        ) -> Result<Array1<f64>, std::convert::Infallible> {
            let mut out = Array1::zeros(x.len());
            mccormick_gradient(
                x.as_slice().expect("Array1 is contiguous"),
                out.as_slice_mut().expect("Array1 is contiguous"),
            );
            Ok(out)
        }
    }
}

#[cfg(feature = "faer_all")]
mod faer_impl {
    use super::McCormick;
    use crate::{CostFunction, Gradient};
    use faer::Col;

    // faer's `Col` doesn't expose a `&[f64]` directly across all 0.24 APIs we
    // care about, so we evaluate elementwise here rather than routing through
    // the slice-based primitives.
    impl CostFunction for McCormick<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            debug_assert_eq!(x.nrows(), 2);
            let (a, b) = (x[0], x[1]);
            let d = a - b;
            Ok((a + b).sin() + d * d - 1.5 * a + 2.5 * b + 1.0)
        }
    }

    impl Gradient for McCormick<Col<f64>> {
        type Gradient = Col<f64>;
        fn gradient(
            &self,
            x: &Col<f64>,
        ) -> Result<Col<f64>, std::convert::Infallible> {
            debug_assert_eq!(x.nrows(), 2);
            let (a, b) = (x[0], x[1]);
            let c = (a + b).cos();
            let d = a - b;
            let g0 = c + 2.0 * d - 1.5;
            let g1 = c - 2.0 * d + 2.5;
            Ok(Col::<f64>::from_fn(2, |i| if i == 0 { g0 } else { g1 }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Exact minimizer: x* = 0.5 − π/3, y* = −0.5 − π/3.
    fn minimizer() -> [f64; 2] {
        [0.5 - PI / 3.0, -0.5 - PI / 3.0]
    }

    #[test]
    fn mccormick_value_at_known_minimum() {
        // f* = sin(-2π/3) + 1 + (-1.5)(0.5 - π/3) + 2.5(-0.5 - π/3) + 1
        //    = -√3/2 - π/3.
        let expected = -((3.0_f64).sqrt()) / 2.0 - PI / 3.0;
        let v = mccormick(&minimizer());
        assert!((v - expected).abs() < 1e-12, "got {v}, expected {expected}");
    }

    #[test]
    fn mccormick_known_value_at_origin() {
        // f(0, 0) = sin(0) + 0 - 0 + 0 + 1 = 1.
        assert!((mccormick(&[0.0, 0.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn gradient_zero_at_minimum() {
        let mut g = vec![0.0; 2];
        mccormick_gradient(&minimizer(), &mut g);
        for v in g {
            assert!(v.abs() < 1e-12, "component = {v}");
        }
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let x = [-1.2, 0.7];
        let mut g = vec![0.0; x.len()];
        mccormick_gradient(&x, &mut g);

        let h = 1e-6;
        for i in 0..x.len() {
            let mut xp = x;
            let mut xm = x;
            xp[i] += h;
            xm[i] -= h;
            let fd = (mccormick(&xp) - mccormick(&xm)) / (2.0 * h);
            assert!((g[i] - fd).abs() < 1e-5, "i={i}, g={}, fd={fd}", g[i]);
        }
    }

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        let spec = <McCormick<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(spec.name, "McCormick");
        assert!(spec.properties.smooth);
        assert!(spec.properties.differentiable);
        assert!(spec.properties.unimodal);
        assert!(matches!(spec.dim, Dimensionality::Fixed(2)));
        assert!(!spec.references.is_empty());
    }
}
