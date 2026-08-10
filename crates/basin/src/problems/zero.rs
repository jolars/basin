//! N-dimensional Zero function: `f(x) = 0` everywhere.
//!
//! A degenerate sanity or termination edge case. The cost is identically zero
//! and the gradient is the zero vector, so *every* point is a global minimizer
//! (value 0). Useful for exercising solver bookkeeping in the limit: a solver
//! should terminate immediately on a gradient- or step-tolerance criterion
//! without taking a meaningful step, and shouldn't divide by a zero gradient
//! norm or loop forever. Defined for any `n ≥ 1`.

use core::marker::PhantomData;

use super::spec::{
    Dimensionality, HasSpec, ProblemSpec, Properties, Reference,
};
use crate::{CostFunction, Gradient};

/// Evaluates the Zero function at `x`: always `0`. Works for any `n >= 1`.
pub fn zero(_x: &[f64]) -> f64 {
    0.0
}

/// Writes the Zero gradient at `x` into `out`: always the zero vector.
pub fn zero_gradient(_x: &[f64], out: &mut [f64]) {
    for g in out.iter_mut() {
        *g = 0.0;
    }
}

/// Pre-wrapped Zero problem. Generic over the parameter backend `P`; the
/// default `P = Vec<f64>` lets you write `Zero::default()` for the common case.
/// Backend impls (`nalgebra::DVector<f64>`, `ndarray::Array1<f64>`,
/// `faer::Col<f64>`) are gated behind their respective features.
pub struct Zero<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> Zero<P> {
    /// Build a freshly typed problem instance. Pair with one of the
    /// backend-specific impl blocks (Vec, nalgebra, ndarray, faer).
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for Zero<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Catalog entry for this problem.
pub static ZERO_SPEC: ProblemSpec = ProblemSpec {
    name: "Zero",
    dim: Dimensionality::NDimensional { min: 1 },
    properties: Properties {
        smooth: true,
        differentiable: true,
        // A constant function is (degenerately) convex.
        convex: true,
        // A continuum of minimizers, not a single isolated minimum.
        unimodal: false,
        separable: true,
        scalable: true,
    },
    references: &[Reference {
        citation: "basin test corpus",
        title: "Zero (constant) function: termination and sanity edge case",
        source: "basin (internal); no published origin",
        doi: None,
        url: None,
    }],
    description: "Constant function f(x) = 0 everywhere; the gradient is \
                  identically zero. A degenerate sanity/termination edge \
                  case: every point is a global minimizer (value 0), so any \
                  solver should terminate immediately on a gradient or step \
                  tolerance. Defined for any n ≥ 1.",
};

impl<P> HasSpec for Zero<P> {
    const SPEC: &'static ProblemSpec = &ZERO_SPEC;
}

impl CostFunction for Zero<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(zero(x))
    }
}

impl Gradient for Zero<Vec<f64>> {
    type Gradient = Vec<f64>;
    fn gradient(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; x.len()];
        zero_gradient(x, &mut out);
        Ok(out)
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_impl {
    use super::{Zero, zero, zero_gradient};
    use crate::{CostFunction, Gradient};
    use nalgebra::DVector;

    impl CostFunction for Zero<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &DVector<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(zero(x.as_slice()))
        }
    }

    impl Gradient for Zero<DVector<f64>> {
        type Gradient = DVector<f64>;
        fn gradient(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, std::convert::Infallible> {
            let mut out = DVector::zeros(x.len());
            zero_gradient(x.as_slice(), out.as_mut_slice());
            Ok(out)
        }
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::{Zero, zero, zero_gradient};
    use crate::{CostFunction, Gradient};
    use ndarray::Array1;

    impl CostFunction for Zero<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &Array1<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(zero(x.as_slice().expect("Array1 is contiguous")))
        }
    }

    impl Gradient for Zero<Array1<f64>> {
        type Gradient = Array1<f64>;
        fn gradient(
            &self,
            x: &Array1<f64>,
        ) -> Result<Array1<f64>, std::convert::Infallible> {
            let mut out = Array1::zeros(x.len());
            zero_gradient(
                x.as_slice().expect("Array1 is contiguous"),
                out.as_slice_mut().expect("Array1 is contiguous"),
            );
            Ok(out)
        }
    }
}

#[cfg(feature = "faer")]
mod faer_impl {
    use super::Zero;
    use crate::{CostFunction, Gradient};
    use faer::Col;

    impl CostFunction for Zero<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, _x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(0.0)
        }
    }

    impl Gradient for Zero<Col<f64>> {
        type Gradient = Col<f64>;
        fn gradient(
            &self,
            x: &Col<f64>,
        ) -> Result<Col<f64>, std::convert::Infallible> {
            Ok(Col::<f64>::zeros(x.nrows()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_is_zero_everywhere() {
        assert_eq!(zero(&[0.0]), 0.0);
        assert_eq!(zero(&[1.0, -2.0, 3.5]), 0.0);
        assert_eq!(zero(&[1e9; 8]), 0.0);
    }

    #[test]
    fn gradient_is_zero_everywhere() {
        let mut g = vec![1.0; 5];
        zero_gradient(&[3.0, -1.0, 0.7, 42.0, -8.0], &mut g);
        for v in g {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn gradient_matches_finite_difference() {
        // Trivially zero, but keep the convention's FD check honest.
        let x = [-1.2, 1.0, 0.7];
        let mut g = vec![0.0; x.len()];
        zero_gradient(&x, &mut g);
        let h = 1e-6;
        for i in 0..x.len() {
            let mut xp = x;
            let mut xm = x;
            xp[i] += h;
            xm[i] -= h;
            let fd = (zero(&xp) - zero(&xm)) / (2.0 * h);
            assert!((g[i] - fd).abs() < 1e-12, "i={i}, g={}, fd={fd}", g[i]);
        }
    }

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        let spec = <Zero<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(spec.name, "Zero");
        assert!(spec.properties.smooth);
        assert!(spec.properties.convex);
        assert!(spec.properties.scalable);
        assert!(!spec.properties.unimodal);
        assert!(matches!(spec.dim, Dimensionality::NDimensional { min: 1 }));
        assert!(!spec.references.is_empty());
    }
}
