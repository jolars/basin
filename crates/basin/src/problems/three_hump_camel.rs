//! 2D Three-hump camel function.
//!
//! `f(x, y) = 2x² − 1.05x⁴ + x⁶/6 + xy + y²`
//!
//! Smooth polynomial with three local minima: the central global minimum at
//! `(x, y) = (0, 0)` with `f = 0`, flanked by two symmetric "humps". Useful as
//! a small, smooth basin-of-attraction test for local solvers: depending on the
//! start, a descent method settles into the central bowl or stalls near a hump.
//! Usual search domain is `x, y ∈ [-5, 5]`.

use core::marker::PhantomData;

use super::spec::{
    Dimensionality, HasSpec, ProblemSpec, Properties, Reference,
};
use crate::{CostFunction, Gradient};

/// Standard lower bound on each coordinate.
pub const STANDARD_LOWER: f64 = -5.0;
/// Standard upper bound on each coordinate.
pub const STANDARD_UPPER: f64 = 5.0;

/// Evaluates the Three-hump camel function at `x`. Requires `x.len() == 2`.
pub fn three_hump_camel(x: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), 2);
    let (a, b) = (x[0], x[1]);
    let a2 = a * a;
    let a4 = a2 * a2;
    let a6 = a4 * a2;
    2.0 * a2 - 1.05 * a4 + a6 / 6.0 + a * b + b * b
}

/// Writes the Three-hump camel gradient at `x` into `out`. Both slices must
/// have length 2.
pub fn three_hump_camel_gradient(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), 2);
    debug_assert_eq!(out.len(), 2);
    let (a, b) = (x[0], x[1]);
    let a2 = a * a;
    let a3 = a2 * a;
    let a5 = a2 * a3;
    // ∂f/∂x = 4x − 4.2x³ + x⁵ + y
    out[0] = 4.0 * a - 4.2 * a3 + a5 + b;
    // ∂f/∂y = x + 2y
    out[1] = a + 2.0 * b;
}

/// Pre-wrapped Three-hump camel problem (fixed 2D). Generic over the parameter
/// backend `P`; the default `P = Vec<f64>` lets you write
/// `ThreeHumpCamel::default()` for the common case. Backend impls
/// (`nalgebra::DVector<f64>`, `ndarray::Array1<f64>`, `faer::Col<f64>`) are
/// gated behind their respective features.
pub struct ThreeHumpCamel<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> ThreeHumpCamel<P> {
    /// Build a freshly typed problem instance. Pair with one of the
    /// backend-specific impl blocks (Vec, nalgebra, ndarray, faer).
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for ThreeHumpCamel<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Catalog entry for this problem.
pub static THREE_HUMP_CAMEL_SPEC: ProblemSpec = ProblemSpec {
    name: "Three-hump camel",
    dim: Dimensionality::Fixed(2),
    properties: Properties {
        smooth: true,
        differentiable: true,
        convex: false,
        // Three local minima (one global, two humps).
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
    description: "Smooth 2D polynomial with three local minima. Global minimum \
                  at (x, y) = (0, 0), value 0, flanked by two symmetric humps. \
                  Usual search domain is x, y ∈ [-5, 5]; a compact \
                  basin-of-attraction test for local descent methods.",
};

impl<P> HasSpec for ThreeHumpCamel<P> {
    const SPEC: &'static ProblemSpec = &THREE_HUMP_CAMEL_SPEC;
}

impl CostFunction for ThreeHumpCamel<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(three_hump_camel(x))
    }
}

impl Gradient for ThreeHumpCamel<Vec<f64>> {
    type Gradient = Vec<f64>;
    fn gradient(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; x.len()];
        three_hump_camel_gradient(x, &mut out);
        Ok(out)
    }
}

#[cfg(feature = "nalgebra_all")]
mod nalgebra_impl {
    use super::{ThreeHumpCamel, three_hump_camel, three_hump_camel_gradient};
    use crate::{CostFunction, Gradient};
    use nalgebra::DVector;

    impl CostFunction for ThreeHumpCamel<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &DVector<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(three_hump_camel(x.as_slice()))
        }
    }

    impl Gradient for ThreeHumpCamel<DVector<f64>> {
        type Gradient = DVector<f64>;
        fn gradient(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, std::convert::Infallible> {
            let mut out = DVector::zeros(x.len());
            three_hump_camel_gradient(x.as_slice(), out.as_mut_slice());
            Ok(out)
        }
    }
}

#[cfg(feature = "ndarray_all")]
mod ndarray_impl {
    use super::{ThreeHumpCamel, three_hump_camel, three_hump_camel_gradient};
    use crate::{CostFunction, Gradient};
    use ndarray::Array1;

    impl CostFunction for ThreeHumpCamel<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &Array1<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(three_hump_camel(
                x.as_slice().expect("Array1 is contiguous"),
            ))
        }
    }

    impl Gradient for ThreeHumpCamel<Array1<f64>> {
        type Gradient = Array1<f64>;
        fn gradient(
            &self,
            x: &Array1<f64>,
        ) -> Result<Array1<f64>, std::convert::Infallible> {
            let mut out = Array1::zeros(x.len());
            three_hump_camel_gradient(
                x.as_slice().expect("Array1 is contiguous"),
                out.as_slice_mut().expect("Array1 is contiguous"),
            );
            Ok(out)
        }
    }
}

#[cfg(feature = "faer_all")]
mod faer_impl {
    use super::ThreeHumpCamel;
    use crate::{CostFunction, Gradient};
    use faer::Col;

    // faer's `Col` doesn't expose a `&[f64]` directly across all 0.24 APIs we
    // care about, so we evaluate elementwise here rather than routing through
    // the slice-based primitives.
    impl CostFunction for ThreeHumpCamel<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            debug_assert_eq!(x.nrows(), 2);
            let (a, b) = (x[0], x[1]);
            let a2 = a * a;
            let a4 = a2 * a2;
            let a6 = a4 * a2;
            Ok(2.0 * a2 - 1.05 * a4 + a6 / 6.0 + a * b + b * b)
        }
    }

    impl Gradient for ThreeHumpCamel<Col<f64>> {
        type Gradient = Col<f64>;
        fn gradient(
            &self,
            x: &Col<f64>,
        ) -> Result<Col<f64>, std::convert::Infallible> {
            debug_assert_eq!(x.nrows(), 2);
            let (a, b) = (x[0], x[1]);
            let a2 = a * a;
            let a3 = a2 * a;
            let a5 = a2 * a3;
            let g0 = 4.0 * a - 4.2 * a3 + a5 + b;
            let g1 = a + 2.0 * b;
            Ok(Col::<f64>::from_fn(2, |i| if i == 0 { g0 } else { g1 }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimum_is_zero_at_origin() {
        assert!(three_hump_camel(&[0.0, 0.0]).abs() < 1e-12);
    }

    #[test]
    fn known_value_at_unit_point() {
        // f(1, 1) = 2 − 1.05 + 1/6 + 1 + 1 = 3.11666…
        let f = three_hump_camel(&[1.0, 1.0]);
        assert!((f - 3.116666666666667).abs() < 1e-12, "got {f}");
    }

    #[test]
    fn gradient_zero_at_minimum() {
        let mut g = vec![0.0; 2];
        three_hump_camel_gradient(&[0.0, 0.0], &mut g);
        for v in g {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let x = [-1.2, 0.7];
        let mut g = vec![0.0; x.len()];
        three_hump_camel_gradient(&x, &mut g);
        let h = 1e-6;
        for i in 0..x.len() {
            let mut xp = x;
            let mut xm = x;
            xp[i] += h;
            xm[i] -= h;
            let fd =
                (three_hump_camel(&xp) - three_hump_camel(&xm)) / (2.0 * h);
            assert!((g[i] - fd).abs() < 1e-5, "i={i}, g={}, fd={fd}", g[i]);
        }
    }

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        let spec = <ThreeHumpCamel<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(spec.name, "Three-hump camel");
        assert!(spec.properties.smooth);
        assert!(spec.properties.differentiable);
        assert!(!spec.properties.convex);
        assert!(matches!(spec.dim, Dimensionality::Fixed(2)));
        assert!(!spec.references.is_empty());
    }
}
