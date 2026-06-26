//! 2D Cross-in-tray function.
//!
//! ```text
//! f(x, y) = −0.0001·(|sin(x)·sin(y)·exp(|100 − √(x²+y²)/π|)| + 1)^0.1
//! ```
//!
//! Multimodal function whose surface looks like a tray with a cross-shaped
//! pattern of deep wells. It has four equal global minima at
//! `(±1.34941, ±1.34941)` with `f ≈ −2.06261`. The nested `|·|` terms make it
//! non-differentiable, so it is cost-only: a target for derivative-free and
//! global solvers. Usual search domain is `x, y ∈ [-10, 10]`.

use core::marker::PhantomData;

use super::spec::{Dimensionality, HasSpec, ProblemSpec, Properties, Reference};
use crate::CostFunction;

/// Standard lower bound on each coordinate.
pub const STANDARD_LOWER: f64 = -10.0;
/// Standard upper bound on each coordinate.
pub const STANDARD_UPPER: f64 = 10.0;

/// Evaluates the Cross-in-tray function at `x`. Requires `x.len() == 2`.
pub fn cross_in_tray(x: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), 2);
    let pi = core::f64::consts::PI;
    let (a, b) = (x[0], x[1]);
    let inner = (100.0 - (a * a + b * b).sqrt() / pi).abs();
    let g = (a.sin() * b.sin() * inner.exp()).abs();
    -0.0001 * (g + 1.0).powf(0.1)
}

/// Pre-wrapped Cross-in-tray problem (fixed 2D). Cost-only: the nested `|·|`
/// terms make it non-differentiable, so no `Gradient` impl is provided.
pub struct CrossInTray<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> CrossInTray<P> {
    /// Build a freshly typed problem instance. Pair with one of the
    /// backend-specific impl blocks (Vec, nalgebra, ndarray, faer).
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for CrossInTray<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Catalog entry for this problem.
pub static CROSS_IN_TRAY_SPEC: ProblemSpec = ProblemSpec {
    name: "Cross-in-tray",
    dim: Dimensionality::Fixed(2),
    properties: Properties {
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
    description: "Tray-shaped multimodal surface with four equal global minima \
                  at (±1.34941, ±1.34941), value ≈ −2.06261. Non-differentiable \
                  (nested |·| terms); usual search domain is x, y ∈ [-10, 10]. \
                  Cost-only, for derivative-free and global solvers.",
};

impl<P> HasSpec for CrossInTray<P> {
    const SPEC: &'static ProblemSpec = &CROSS_IN_TRAY_SPEC;
}

impl CostFunction for CrossInTray<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(cross_in_tray(x))
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_impl {
    use super::{CrossInTray, cross_in_tray};
    use crate::CostFunction;
    use nalgebra::DVector;

    impl CostFunction for CrossInTray<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(cross_in_tray(x.as_slice()))
        }
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::{CrossInTray, cross_in_tray};
    use crate::CostFunction;
    use ndarray::Array1;

    impl CostFunction for CrossInTray<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Array1<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(cross_in_tray(x.as_slice().expect("Array1 is contiguous")))
        }
    }
}

#[cfg(feature = "faer")]
mod faer_impl {
    use super::CrossInTray;
    use crate::CostFunction;
    use faer::Col;

    impl CostFunction for CrossInTray<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            debug_assert_eq!(x.nrows(), 2);
            let pi = core::f64::consts::PI;
            let (a, b) = (x[0], x[1]);
            let inner = (100.0 - (a * a + b * b).sqrt() / pi).abs();
            let g = (a.sin() * b.sin() * inner.exp()).abs();
            Ok(-0.0001 * (g + 1.0).powf(0.1))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_value_at_origin() {
        // sin(0) = 0 ⇒ the inner product is 0 ⇒ (0 + 1)^0.1 = 1 ⇒ f = −0.0001.
        assert!((cross_in_tray(&[0.0, 0.0]) - (-0.0001)).abs() < 1e-15);
    }

    #[test]
    fn minimum_value_at_documented_optimum() {
        let f = cross_in_tray(&[1.34941, 1.34941]);
        assert!((f - (-2.06261)).abs() < 1e-4, "got {f}");
    }

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        let spec = <CrossInTray<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(spec.name, "Cross-in-tray");
        assert!(!spec.properties.differentiable);
        assert!(matches!(spec.dim, Dimensionality::Fixed(2)));
        assert!(!spec.references.is_empty());
    }
}
