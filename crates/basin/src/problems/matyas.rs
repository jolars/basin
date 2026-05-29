//! 2D Matyas function.
//!
//! `f(x, y) = 0.26·(x² + y²) − 0.48·x·y`
//!
//! Smooth, plate-like 2D quadratic with a single global minimum at
//! `(x, y) = (0, 0)` with `f = 0`. Usual search domain is
//! `x, y ∈ [-10, 10]`. The Hessian `[[0.52, -0.48], [-0.48, 0.52]]` is
//! constant and positive definite (eigenvalues `0.04` and `1.0`), so the
//! function is strictly convex but mildly ill-conditioned along the
//! `x = y` direction. Useful as a near-trivial sanity check that's
//! distinct from the separable Sphere.

use core::marker::PhantomData;

use super::spec::{Dimensionality, HasSpec, ProblemSpec, Properties, Reference};
use crate::{CostFunction, Gradient};

/// Evaluates the Matyas function at `x`. Requires `x.len() == 2`.
pub fn matyas(x: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), 2);
    let (a, b) = (x[0], x[1]);
    0.26 * (a * a + b * b) - 0.48 * a * b
}

/// Writes the Matyas gradient at `x` into `out`. Both slices must have length 2.
pub fn matyas_gradient(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), 2);
    debug_assert_eq!(out.len(), 2);
    let (a, b) = (x[0], x[1]);
    // ∂f/∂x = 0.52·x − 0.48·y
    out[0] = 0.52 * a - 0.48 * b;
    // ∂f/∂y = 0.52·y − 0.48·x
    out[1] = 0.52 * b - 0.48 * a;
}

/// Pre-wrapped Matyas problem (fixed 2D). Generic over the parameter backend
/// `P`; the default `P = Vec<f64>` lets you write `Matyas::default()` for the
/// common case. Backend impls (`nalgebra::DVector<f64>`, `ndarray::Array1<f64>`,
/// `faer::Col<f64>`) are gated behind their respective features.
pub struct Matyas<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> Matyas<P> {
    /// Build a freshly typed problem instance. Pair with one of the
    /// backend-specific impl blocks (Vec, nalgebra, ndarray, faer).
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for Matyas<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Catalogue entry for this problem.
pub static MATYAS_SPEC: ProblemSpec = ProblemSpec {
    name: "Matyas",
    dim: Dimensionality::Fixed(2),
    properties: Properties {
        smooth: true,
        differentiable: true,
        // Hessian [[0.52, -0.48], [-0.48, 0.52]] is constant SPD
        // (eigenvalues 0.04 and 1.0), so f is strictly convex on R².
        convex: true,
        unimodal: true,
        // Cross term -0.48·x·y prevents per-coordinate decomposition.
        separable: false,
        scalable: false,
    },
    references: &[Reference {
        citation: "Jamil & Yang (2013)",
        title: "A Literature Survey of Benchmark Functions For Global Optimisation Problems",
        source: "International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150–194",
        doi: Some("10.1504/IJMMNO.2013.055204"),
        url: Some("https://arxiv.org/abs/1308.4008"),
    }],
    description: "Smooth plate-like 2D quadratic: f(x, y) = 0.26·(x² + y²) \
                  − 0.48·x·y. Global minimum at (x, y) = (0, 0), value 0. \
                  Usual search domain is x, y ∈ [-10, 10]. Strictly convex \
                  but mildly ill-conditioned (Hessian eigenvalues 0.04 and \
                  1.0), so first-order methods crawl along the x = y \
                  direction.",
};

impl<P> HasSpec for Matyas<P> {
    const SPEC: &'static ProblemSpec = &MATYAS_SPEC;
}

impl CostFunction for Matyas<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(matyas(x))
    }
}

impl Gradient for Matyas<Vec<f64>> {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; x.len()];
        matyas_gradient(x, &mut out);
        Ok(out)
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_impl {
    use super::{Matyas, matyas, matyas_gradient};
    use crate::{CostFunction, Gradient};
    use nalgebra::DVector;

    impl CostFunction for Matyas<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(matyas(x.as_slice()))
        }
    }

    impl Gradient for Matyas<DVector<f64>> {
        type Gradient = DVector<f64>;
        fn gradient(&self, x: &DVector<f64>) -> Result<DVector<f64>, std::convert::Infallible> {
            let mut out = DVector::zeros(x.len());
            matyas_gradient(x.as_slice(), out.as_mut_slice());
            Ok(out)
        }
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::{Matyas, matyas, matyas_gradient};
    use crate::{CostFunction, Gradient};
    use ndarray::Array1;

    impl CostFunction for Matyas<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Array1<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(matyas(x.as_slice().expect("Array1 is contiguous")))
        }
    }

    impl Gradient for Matyas<Array1<f64>> {
        type Gradient = Array1<f64>;
        fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, std::convert::Infallible> {
            let mut out = Array1::zeros(x.len());
            matyas_gradient(
                x.as_slice().expect("Array1 is contiguous"),
                out.as_slice_mut().expect("Array1 is contiguous"),
            );
            Ok(out)
        }
    }
}

#[cfg(feature = "faer")]
mod faer_impl {
    use super::Matyas;
    use crate::{CostFunction, Gradient};
    use faer::Col;

    // faer's `Col` doesn't expose a `&[f64]` directly across all 0.24 APIs we
    // care about, so we evaluate elementwise here rather than routing through
    // the slice-based primitives.
    impl CostFunction for Matyas<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            debug_assert_eq!(x.nrows(), 2);
            let (a, b) = (x[0], x[1]);
            Ok(0.26 * (a * a + b * b) - 0.48 * a * b)
        }
    }

    impl Gradient for Matyas<Col<f64>> {
        type Gradient = Col<f64>;
        fn gradient(&self, x: &Col<f64>) -> Result<Col<f64>, std::convert::Infallible> {
            debug_assert_eq!(x.nrows(), 2);
            let (a, b) = (x[0], x[1]);
            let g0 = 0.52 * a - 0.48 * b;
            let g1 = 0.52 * b - 0.48 * a;
            Ok(Col::<f64>::from_fn(2, |i| if i == 0 { g0 } else { g1 }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matyas_minimum_is_zero_at_known_optimum() {
        assert!(matyas(&[0.0, 0.0]).abs() < 1e-12);
    }

    #[test]
    fn matyas_known_value_at_one_one() {
        // f(1, 1) = 0.26·(1 + 1) − 0.48·1·1 = 0.52 − 0.48 = 0.04.
        assert!((matyas(&[1.0, 1.0]) - 0.04).abs() < 1e-12);
    }

    #[test]
    fn gradient_zero_at_minimum() {
        let mut g = vec![0.0; 2];
        matyas_gradient(&[0.0, 0.0], &mut g);
        for v in g {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let x = [-1.2, 0.7];
        let mut g = vec![0.0; x.len()];
        matyas_gradient(&x, &mut g);

        let h = 1e-6;
        for i in 0..x.len() {
            let mut xp = x;
            let mut xm = x;
            xp[i] += h;
            xm[i] -= h;
            let fd = (matyas(&xp) - matyas(&xm)) / (2.0 * h);
            assert!((g[i] - fd).abs() < 1e-5, "i={i}, g={}, fd={fd}", g[i]);
        }
    }

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        let spec = <Matyas<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(spec.name, "Matyas");
        assert!(spec.properties.smooth);
        assert!(spec.properties.differentiable);
        assert!(spec.properties.convex);
        assert!(spec.properties.unimodal);
        assert!(matches!(spec.dim, Dimensionality::Fixed(2)));
        assert!(!spec.references.is_empty());
    }
}
