//! 2D Himmelblau function.
//!
//! `f(x, y) = (x² + y − 11)² + (x + y² − 7)²`
//!
//! Smooth quartic polynomial with four equal global minima (value 0), arranged
//! symmetrically around a local maximum near the origin. The classic "which
//! minimum does the solver find?" test: most interesting once a global solver
//! makes the choice between basins meaningful. Global minima:
//! `(3, 2)`, `(−2.805118, 3.131312)`, `(−3.779310, −3.283186)`,
//! `(3.584428, −1.848127)`. Usual search domain is `x, y ∈ [-5, 5]`.

use core::marker::PhantomData;

use super::spec::{
    Dimensionality, HasSpec, ProblemSpec, Properties, Reference,
};
use crate::{CostFunction, Gradient};

/// Standard lower bound on each coordinate.
pub const STANDARD_LOWER: f64 = -5.0;
/// Standard upper bound on each coordinate.
pub const STANDARD_UPPER: f64 = 5.0;

/// Evaluates the Himmelblau function at `x`. Requires `x.len() == 2`.
pub fn himmelblau(x: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), 2);
    let (a, b) = (x[0], x[1]);
    let t1 = a * a + b - 11.0;
    let t2 = a + b * b - 7.0;
    t1 * t1 + t2 * t2
}

/// Writes the Himmelblau gradient at `x` into `out`. Both slices must have
/// length 2.
pub fn himmelblau_gradient(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), 2);
    debug_assert_eq!(out.len(), 2);
    let (a, b) = (x[0], x[1]);
    let t1 = a * a + b - 11.0;
    let t2 = a + b * b - 7.0;
    // ∂f/∂x = 2·t1·(2x) + 2·t2·(1) = 4x·t1 + 2·t2
    out[0] = 4.0 * a * t1 + 2.0 * t2;
    // ∂f/∂y = 2·t1·(1) + 2·t2·(2y) = 2·t1 + 4y·t2
    out[1] = 2.0 * t1 + 4.0 * b * t2;
}

/// Pre-wrapped Himmelblau problem (fixed 2D). Generic over the parameter
/// backend `P`; the default `P = Vec<f64>` lets you write
/// `Himmelblau::default()` for the common case. Backend impls
/// (`nalgebra::DVector<f64>`, `ndarray::Array1<f64>`, `faer::Col<f64>`) are
/// gated behind their respective features.
pub struct Himmelblau<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> Himmelblau<P> {
    /// Build a freshly typed problem instance. Pair with one of the
    /// backend-specific impl blocks (Vec, nalgebra, ndarray, faer).
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for Himmelblau<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Catalog entry for this problem.
pub static HIMMELBLAU_SPEC: ProblemSpec = ProblemSpec {
    name: "Himmelblau",
    dim: Dimensionality::Fixed(2),
    properties: Properties {
        smooth: true,
        differentiable: true,
        convex: false,
        // Four equal global minima.
        unimodal: false,
        separable: false,
        scalable: false,
    },
    references: &[Reference {
        citation: "Himmelblau (1972)",
        title: "Applied Nonlinear Programming",
        source: "McGraw-Hill, New York",
        doi: None,
        url: None,
    }],
    description: "Smooth quartic polynomial with four equal global minima \
                  (value 0) at (3, 2), (−2.805118, 3.131312), \
                  (−3.779310, −3.283186) and (3.584428, −1.848127), arranged \
                  around a central local maximum. Usual search domain is \
                  x, y ∈ [-5, 5]; the classic 'which minimum?' test for global \
                  solvers.",
};

impl<P> HasSpec for Himmelblau<P> {
    const SPEC: &'static ProblemSpec = &HIMMELBLAU_SPEC;
}

impl CostFunction for Himmelblau<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(himmelblau(x))
    }
}

impl Gradient for Himmelblau<Vec<f64>> {
    type Gradient = Vec<f64>;
    fn gradient(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; x.len()];
        himmelblau_gradient(x, &mut out);
        Ok(out)
    }
}

#[cfg(feature = "nalgebra_all")]
mod nalgebra_impl {
    use super::{Himmelblau, himmelblau, himmelblau_gradient};
    use crate::{CostFunction, Gradient};
    use nalgebra::DVector;

    impl CostFunction for Himmelblau<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &DVector<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(himmelblau(x.as_slice()))
        }
    }

    impl Gradient for Himmelblau<DVector<f64>> {
        type Gradient = DVector<f64>;
        fn gradient(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, std::convert::Infallible> {
            let mut out = DVector::zeros(x.len());
            himmelblau_gradient(x.as_slice(), out.as_mut_slice());
            Ok(out)
        }
    }
}

#[cfg(feature = "ndarray_all")]
mod ndarray_impl {
    use super::{Himmelblau, himmelblau, himmelblau_gradient};
    use crate::{CostFunction, Gradient};
    use ndarray::Array1;

    impl CostFunction for Himmelblau<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &Array1<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(himmelblau(x.as_slice().expect("Array1 is contiguous")))
        }
    }

    impl Gradient for Himmelblau<Array1<f64>> {
        type Gradient = Array1<f64>;
        fn gradient(
            &self,
            x: &Array1<f64>,
        ) -> Result<Array1<f64>, std::convert::Infallible> {
            let mut out = Array1::zeros(x.len());
            himmelblau_gradient(
                x.as_slice().expect("Array1 is contiguous"),
                out.as_slice_mut().expect("Array1 is contiguous"),
            );
            Ok(out)
        }
    }
}

#[cfg(feature = "faer_all")]
mod faer_impl {
    use super::Himmelblau;
    use crate::{CostFunction, Gradient};
    use faer::Col;

    // faer's `Col` doesn't expose a `&[f64]` directly across all 0.24 APIs we
    // care about, so we evaluate elementwise here rather than routing through
    // the slice-based primitives.
    impl CostFunction for Himmelblau<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            debug_assert_eq!(x.nrows(), 2);
            let (a, b) = (x[0], x[1]);
            let t1 = a * a + b - 11.0;
            let t2 = a + b * b - 7.0;
            Ok(t1 * t1 + t2 * t2)
        }
    }

    impl Gradient for Himmelblau<Col<f64>> {
        type Gradient = Col<f64>;
        fn gradient(
            &self,
            x: &Col<f64>,
        ) -> Result<Col<f64>, std::convert::Infallible> {
            debug_assert_eq!(x.nrows(), 2);
            let (a, b) = (x[0], x[1]);
            let t1 = a * a + b - 11.0;
            let t2 = a + b * b - 7.0;
            let g0 = 4.0 * a * t1 + 2.0 * t2;
            let g1 = 2.0 * t1 + 4.0 * b * t2;
            Ok(Col::<f64>::from_fn(2, |i| if i == 0 { g0 } else { g1 }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimum_is_zero_at_known_optima() {
        assert!(himmelblau(&[3.0, 2.0]).abs() < 1e-12);
        // The other three minima are irrational; check to the documented digits.
        assert!(himmelblau(&[-2.805118, 3.131312]).abs() < 1e-8);
        assert!(himmelblau(&[-3.779310, -3.283186]).abs() < 1e-8);
        assert!(himmelblau(&[3.584428, -1.848127]).abs() < 1e-8);
    }

    #[test]
    fn known_value_at_origin() {
        // f(0, 0) = (0 + 0 − 11)² + (0 + 0 − 7)² = 121 + 49 = 170
        assert!((himmelblau(&[0.0, 0.0]) - 170.0).abs() < 1e-12);
    }

    #[test]
    fn gradient_zero_at_minimum() {
        let mut g = vec![0.0; 2];
        himmelblau_gradient(&[3.0, 2.0], &mut g);
        for v in g {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let x = [-1.2, 0.7];
        let mut g = vec![0.0; x.len()];
        himmelblau_gradient(&x, &mut g);
        let h = 1e-6;
        for i in 0..x.len() {
            let mut xp = x;
            let mut xm = x;
            xp[i] += h;
            xm[i] -= h;
            let fd = (himmelblau(&xp) - himmelblau(&xm)) / (2.0 * h);
            assert!((g[i] - fd).abs() < 1e-5, "i={i}, g={}, fd={fd}", g[i]);
        }
    }

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        let spec = <Himmelblau<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(spec.name, "Himmelblau");
        assert!(spec.properties.smooth);
        assert!(spec.properties.differentiable);
        assert!(!spec.properties.unimodal);
        assert!(matches!(spec.dim, Dimensionality::Fixed(2)));
        assert!(!spec.references.is_empty());
    }
}
