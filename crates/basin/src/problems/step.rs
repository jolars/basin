//! N-dimensional Step function (De Jong's discontinuous staircase).
//!
//! `f(x) = Σᵢ ⌊xᵢ + 0.5⌋²`
//!
//! A piecewise-constant, **discontinuous** function: the floor makes it jump
//! between flat plateaus. The global minimum is `f = 0` on the central cube
//! `xᵢ ∈ [−0.5, 0.5)`. It is the canonical test for the case where smooth- and
//! continuity-based methods break down: the Nelder-Mead simplex degenerates on a
//! plateau (all vertices tie) and stalls, whereas a mesh-based direct search
//! (MADS) steps down the staircase to the minimum. Cost-only (discontinuous, so
//! no gradient).

use core::marker::PhantomData;

use super::spec::{Dimensionality, HasSpec, ProblemSpec, Properties, Reference};
use crate::CostFunction;

/// Evaluates the Step function at `x` (any dimension `≥ 1`).
pub fn step(x: &[f64]) -> f64 {
    x.iter().map(|xi| (xi + 0.5).floor().powi(2)).sum()
}

/// Pre-wrapped Step problem (scalable N-D). Cost-only: the floor makes it
/// discontinuous, so no `Gradient` impl is provided. Generic over the parameter
/// backend `P`; the default `P = Vec<f64>` lets you write `Step::default()` for
/// the common case.
pub struct Step<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> Step<P> {
    /// Build a freshly typed problem instance. Pair with one of the
    /// backend-specific impl blocks (Vec, nalgebra, ndarray, faer).
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for Step<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Catalog entry for this problem.
pub static STEP_SPEC: ProblemSpec = ProblemSpec {
    name: "Step",
    dim: Dimensionality::NDimensional { min: 1 },
    properties: Properties {
        // Floor ⇒ piecewise constant: discontinuous, non-differentiable.
        smooth: false,
        differentiable: false,
        convex: false,
        // Flat plateaus everywhere; a single global-minimum cube but not
        // strictly unimodal.
        unimodal: false,
        // Separable sum of per-coordinate floors.
        separable: true,
        scalable: true,
    },
    references: &[Reference {
        citation: "Jamil & Yang (2013)",
        title: "A literature survey of benchmark functions for global optimisation problems",
        source: "International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150–194",
        doi: Some("10.1504/IJMMNO.2013.055204"),
        url: Some("https://arxiv.org/abs/1308.4008"),
    }],
    description: "Discontinuous staircase f(x) = Σ ⌊xᵢ + 0.5⌋² (De Jong's step \
                  function). Global minimum 0 on the central cube xᵢ ∈ [−0.5, \
                  0.5). Piecewise-constant plateaus make it a target for \
                  direct-search/derivative-free solvers and a trap for the \
                  Nelder-Mead simplex, which degenerates on a plateau.",
};

impl<P> HasSpec for Step<P> {
    const SPEC: &'static ProblemSpec = &STEP_SPEC;
}

impl CostFunction for Step<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(step(x))
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_impl {
    use super::{Step, step};
    use crate::CostFunction;
    use nalgebra::DVector;

    impl CostFunction for Step<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(step(x.as_slice()))
        }
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::{Step, step};
    use crate::CostFunction;
    use ndarray::Array1;

    impl CostFunction for Step<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Array1<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(step(x.as_slice().expect("Array1 is contiguous")))
        }
    }
}

#[cfg(feature = "faer")]
mod faer_impl {
    use super::Step;
    use crate::CostFunction;
    use faer::Col;

    impl CostFunction for Step<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            Ok((0..x.nrows()).map(|i| (x[i] + 0.5).floor().powi(2)).sum())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimum_is_zero_on_central_cube() {
        assert_eq!(step(&[0.0, 0.0]), 0.0);
        // Anywhere in [−0.5, 0.5) is the global minimum.
        assert_eq!(step(&[-0.4, 0.49]), 0.0);
    }

    #[test]
    fn known_values_at_simple_points() {
        // f(2, 2) = ⌊2.5⌋² + ⌊2.5⌋² = 2² + 2² = 8.
        assert_eq!(step(&[2.0, 2.0]), 8.0);
        // f(−1.6, 0.0) = ⌊−1.1⌋² + ⌊0.5⌋² = (−2)² + 0² = 4.
        assert_eq!(step(&[-1.6, 0.0]), 4.0);
    }

    #[test]
    fn scalable_dimension() {
        assert_eq!(step(&[3.0, 3.0, 3.0, 3.0]), 4.0 * 9.0); // ⌊3.5⌋² = 9 each
    }

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        let spec = <Step<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(spec.name, "Step");
        assert!(!spec.properties.smooth);
        assert!(!spec.properties.differentiable);
        assert!(spec.properties.separable);
        assert!(matches!(spec.dim, Dimensionality::NDimensional { min: 1 }));
        assert!(!spec.references.is_empty());
    }
}
