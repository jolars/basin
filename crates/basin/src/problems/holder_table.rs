//! 2D Holder table function.
//!
//! ```text
//! f(x, y) = −|sin(x)·cos(y)·exp(|1 − √(x²+y²)/π|)|
//! ```
//!
//! Multimodal function with many local minima and four equal global minima at
//! `(±8.05502, ±9.66459)` with `f ≈ −19.2085`, arranged at the corners of a
//! flat "table". The nested `|·|` terms make it non-differentiable, so it is
//! cost-only—a target for derivative-free and global solvers. Usual search
//! domain is `x, y ∈ [-10, 10]`.

use core::marker::PhantomData;

use super::spec::{Dimensionality, HasSpec, ProblemSpec, Properties, Reference};
use crate::CostFunction;

/// Standard lower bound on each coordinate.
pub const STANDARD_LOWER: f64 = -10.0;
/// Standard upper bound on each coordinate.
pub const STANDARD_UPPER: f64 = 10.0;

/// Evaluates the Holder table function at `x`. Requires `x.len() == 2`.
pub fn holder_table(x: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), 2);
    let pi = core::f64::consts::PI;
    let (a, b) = (x[0], x[1]);
    let inner = (1.0 - (a * a + b * b).sqrt() / pi).abs();
    -(a.sin() * b.cos() * inner.exp()).abs()
}

/// Pre-wrapped Holder table problem (fixed 2D). Cost-only: the nested `|·|`
/// terms make it non-differentiable, so no `Gradient` impl is provided.
pub struct HolderTable<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> HolderTable<P> {
    /// Build a freshly typed problem instance. Pair with one of the
    /// backend-specific impl blocks (Vec, nalgebra, ndarray, faer).
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for HolderTable<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Catalog entry for this problem.
pub static HOLDER_TABLE_SPEC: ProblemSpec = ProblemSpec {
    name: "Holder table",
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
    description: "Multimodal surface with four equal global minima at \
                  (±8.05502, ±9.66459), value ≈ −19.2085, at the corners of a \
                  flat table. Non-differentiable (nested |·| terms); usual \
                  search domain is x, y ∈ [-10, 10]. Cost-only, for \
                  derivative-free and global solvers.",
};

impl<P> HasSpec for HolderTable<P> {
    const SPEC: &'static ProblemSpec = &HOLDER_TABLE_SPEC;
}

impl CostFunction for HolderTable<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(holder_table(x))
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_impl {
    use super::{HolderTable, holder_table};
    use crate::CostFunction;
    use nalgebra::DVector;

    impl CostFunction for HolderTable<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(holder_table(x.as_slice()))
        }
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::{HolderTable, holder_table};
    use crate::CostFunction;
    use ndarray::Array1;

    impl CostFunction for HolderTable<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Array1<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(holder_table(x.as_slice().expect("Array1 is contiguous")))
        }
    }
}

#[cfg(feature = "faer")]
mod faer_impl {
    use super::HolderTable;
    use crate::CostFunction;
    use faer::Col;

    impl CostFunction for HolderTable<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            debug_assert_eq!(x.nrows(), 2);
            let pi = core::f64::consts::PI;
            let (a, b) = (x[0], x[1]);
            let inner = (1.0 - (a * a + b * b).sqrt() / pi).abs();
            Ok(-(a.sin() * b.cos() * inner.exp()).abs())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_value_at_origin() {
        // sin(0) = 0 ⇒ the product is 0 ⇒ f(0, 0) = 0.
        assert!(holder_table(&[0.0, 0.0]).abs() < 1e-15);
    }

    #[test]
    fn minimum_value_at_documented_optimum() {
        let f = holder_table(&[8.05502, 9.66459]);
        assert!((f - (-19.2085)).abs() < 1e-3, "got {f}");
    }

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        let spec = <HolderTable<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(spec.name, "Holder table");
        assert!(!spec.properties.differentiable);
        assert!(matches!(spec.dim, Dimensionality::Fixed(2)));
        assert!(!spec.references.is_empty());
    }
}
