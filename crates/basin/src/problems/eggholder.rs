//! 2D Eggholder function.
//!
//! ```text
//! f(x, y) = −(y + 47)·sin(√|x/2 + (y + 47)|) − x·sin(√|x − (y + 47)|)
//! ```
//!
//! Highly multimodal "egg carton" surface with a vast number of local minima
//! and a single deep global minimum at `(x, y) = (512, 404.2319)` with
//! `f ≈ −959.6407` tucked into a corner of the domain. The nested `√|·|` terms
//! make it non-differentiable, so it is cost-only: a hard target for global
//! solvers. Usual search domain is `x, y ∈ [-512, 512]`.

use core::marker::PhantomData;

use super::spec::{
    Dimensionality, HasSpec, ProblemSpec, Properties, Reference,
};
use crate::CostFunction;

/// Standard lower bound on each coordinate.
pub const STANDARD_LOWER: f64 = -512.0;
/// Standard upper bound on each coordinate.
pub const STANDARD_UPPER: f64 = 512.0;

/// Evaluates the Eggholder function at `x`. Requires `x.len() == 2`.
pub fn eggholder(x: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), 2);
    let (a, b) = (x[0], x[1]);
    let c = b + 47.0;
    -c * (a / 2.0 + c).abs().sqrt().sin() - a * (a - c).abs().sqrt().sin()
}

/// Pre-wrapped Eggholder problem (fixed 2D). Cost-only: the nested `√|·|` terms
/// make it non-differentiable, so no `Gradient` impl is provided.
pub struct Eggholder<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> Eggholder<P> {
    /// Build a freshly typed problem instance. Pair with one of the
    /// backend-specific impl blocks (Vec, nalgebra, ndarray, faer).
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for Eggholder<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Catalog entry for this problem.
pub static EGGHOLDER_SPEC: ProblemSpec = ProblemSpec {
    name: "Eggholder",
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
    description: "Highly multimodal 'egg carton' surface with many local minima \
                  and a single deep global minimum at (x, y) = (512, 404.2319), \
                  value ≈ −959.6407, in a corner of the domain. \
                  Non-differentiable; usual search domain is x, y ∈ [-512, 512]. \
                  Cost-only, for global solvers.",
};

impl<P> HasSpec for Eggholder<P> {
    const SPEC: &'static ProblemSpec = &EGGHOLDER_SPEC;
}

impl CostFunction for Eggholder<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(eggholder(x))
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_impl {
    use super::{Eggholder, eggholder};
    use crate::CostFunction;
    use nalgebra::DVector;

    impl CostFunction for Eggholder<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &DVector<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(eggholder(x.as_slice()))
        }
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::{Eggholder, eggholder};
    use crate::CostFunction;
    use ndarray::Array1;

    impl CostFunction for Eggholder<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &Array1<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(eggholder(x.as_slice().expect("Array1 is contiguous")))
        }
    }
}

#[cfg(feature = "faer")]
mod faer_impl {
    use super::Eggholder;
    use crate::CostFunction;
    use faer::Col;

    impl CostFunction for Eggholder<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            debug_assert_eq!(x.nrows(), 2);
            let (a, b) = (x[0], x[1]);
            let c = b + 47.0;
            Ok(-c * (a / 2.0 + c).abs().sqrt().sin()
                - a * (a - c).abs().sqrt().sin())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_value_at_origin() {
        // f(0, 0) = −47·sin(√47) − 0 = −47·sin(√47).
        let expected = -47.0 * (47.0_f64.sqrt()).sin();
        assert!((eggholder(&[0.0, 0.0]) - expected).abs() < 1e-12);
    }

    #[test]
    fn minimum_value_at_documented_optimum() {
        let f = eggholder(&[512.0, 404.2319]);
        assert!((f - (-959.6407)).abs() < 1e-3, "got {f}");
    }

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        let spec = <Eggholder<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(spec.name, "Eggholder");
        assert!(!spec.properties.differentiable);
        assert!(matches!(spec.dim, Dimensionality::Fixed(2)));
        assert!(!spec.references.is_empty());
    }
}
