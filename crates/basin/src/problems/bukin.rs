//! 2D Bukin function N.6.
//!
//! `f(x, y) = 100·√|y − 0.01·x²| + 0.01·|x + 10|`
//!
//! A notoriously hard 2D function: the global minimum at `(x, y) = (−10, 1)`
//! with `f = 0` lies along a sharp, nearly-flat ridge `y = 0.01·x²` where the
//! `√|·|` term creates a non-differentiable crease. First-order methods are
//! useless here (the ridge is non-smooth and the gradient is huge off it); it's
//! a target for derivative-free and global solvers. The standard search domain
//! is the asymmetric box `x ∈ [-15, -5]`, `y ∈ [-3, 3]`. Cost-only.

use core::marker::PhantomData;

use super::spec::{Dimensionality, HasSpec, ProblemSpec, Properties, Reference};
use crate::CostFunction;

/// Standard lower bound on `x` (the asymmetric Bukin domain).
pub const X_LOWER: f64 = -15.0;
/// Standard upper bound on `x` (the asymmetric Bukin domain).
pub const X_UPPER: f64 = -5.0;
/// Standard lower bound on `y` (the asymmetric Bukin domain).
pub const Y_LOWER: f64 = -3.0;
/// Standard upper bound on `y` (the asymmetric Bukin domain).
pub const Y_UPPER: f64 = 3.0;

/// Evaluates the Bukin N.6 function at `x`. Requires `x.len() == 2`.
pub fn bukin_n6(x: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), 2);
    let (a, b) = (x[0], x[1]);
    100.0 * (b - 0.01 * a * a).abs().sqrt() + 0.01 * (a + 10.0).abs()
}

/// Pre-wrapped Bukin N.6 problem (fixed 2D). Cost-only: the `√|·|` ridge makes
/// it non-differentiable, so no `Gradient` impl is provided. Generic over the
/// parameter backend `P`; the default `P = Vec<f64>` lets you write
/// `BukinN6::default()` for the common case.
pub struct BukinN6<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> BukinN6<P> {
    /// Build a freshly typed problem instance. Pair with one of the
    /// backend-specific impl blocks (Vec, nalgebra, ndarray, faer).
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for BukinN6<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Catalogue entry for this problem.
pub static BUKIN_N6_SPEC: ProblemSpec = ProblemSpec {
    name: "Bukin N.6",
    dim: Dimensionality::Fixed(2),
    properties: Properties {
        // √|·| ridge ⇒ non-smooth and non-differentiable along y = 0.01·x².
        smooth: false,
        differentiable: false,
        convex: false,
        unimodal: false,
        separable: false,
        scalable: false,
    },
    references: &[Reference {
        citation: "Jamil & Yang (2013)",
        title: "A literature survey of benchmark functions for global optimisation problems",
        source: "International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150–194",
        doi: Some("10.1504/IJMMNO.2013.055204"),
        url: Some("https://arxiv.org/abs/1308.4008"),
    }],
    description: "f(x, y) = 100·√|y − 0.01·x²| + 0.01·|x + 10|. Global minimum \
                  at (x, y) = (−10, 1), value 0, along a sharp non-differentiable \
                  ridge y = 0.01·x². Standard domain is the asymmetric box \
                  x ∈ [-15, -5], y ∈ [-3, 3]; pathological for first-order \
                  methods, a target for derivative-free / global solvers.",
};

impl<P> HasSpec for BukinN6<P> {
    const SPEC: &'static ProblemSpec = &BUKIN_N6_SPEC;
}

impl CostFunction for BukinN6<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(bukin_n6(x))
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_impl {
    use super::{BukinN6, bukin_n6};
    use crate::CostFunction;
    use nalgebra::DVector;

    impl CostFunction for BukinN6<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(bukin_n6(x.as_slice()))
        }
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::{BukinN6, bukin_n6};
    use crate::CostFunction;
    use ndarray::Array1;

    impl CostFunction for BukinN6<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Array1<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(bukin_n6(x.as_slice().expect("Array1 is contiguous")))
        }
    }
}

#[cfg(feature = "faer")]
mod faer_impl {
    use super::BukinN6;
    use crate::CostFunction;
    use faer::Col;

    impl CostFunction for BukinN6<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            debug_assert_eq!(x.nrows(), 2);
            let (a, b) = (x[0], x[1]);
            Ok(100.0 * (b - 0.01 * a * a).abs().sqrt() + 0.01 * (a + 10.0).abs())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimum_is_zero_at_known_optimum() {
        assert!(bukin_n6(&[-10.0, 1.0]).abs() < 1e-12);
    }

    #[test]
    fn known_value_in_domain() {
        // f(−5, 0) = 100·√|0 − 0.01·25| + 0.01·|−5 + 10|
        //          = 100·√0.25 + 0.01·5 = 100·0.5 + 0.05 = 50.05
        assert!((bukin_n6(&[-5.0, 0.0]) - 50.05).abs() < 1e-12);
    }

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        let spec = <BukinN6<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(spec.name, "Bukin N.6");
        assert!(!spec.properties.smooth);
        assert!(!spec.properties.differentiable);
        assert!(matches!(spec.dim, Dimensionality::Fixed(2)));
        assert!(!spec.references.is_empty());
    }
}
