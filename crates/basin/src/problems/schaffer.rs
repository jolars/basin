//! 2D Schaffer functions N.2 and N.4.
//!
//! Two closely related multimodal functions built from concentric circular
//! ripples that decay toward the origin, sharing the rational envelope
//! `(1 + 0.001(x² + y²))²`:
//!
//! ```text
//! N.2:  f(x, y) = 0.5 + (sin²(x² − y²) − 0.5) / (1 + 0.001(x² + y²))²
//! N.4:  f(x, y) = 0.5 + (cos²(sin(|x² − y²|)) − 0.5) / (1 + 0.001(x² + y²))²
//! ```
//!
//! N.2 has its global minimum at `(0, 0)` with `f = 0`; it is smooth, so an
//! analytic gradient is provided. N.4 has four global minima near
//! `(0, ±1.25313)` and `(±1.25313, 0)` with `f ≈ 0.292579`; the `|·|` inside
//! makes it non-differentiable along `x² = y²`, so it is cost-only. Usual
//! search domain for both is `x, y ∈ [-100, 100]`.

use core::marker::PhantomData;

use super::spec::{
    Dimensionality, HasSpec, ProblemSpec, Properties, Reference,
};
use crate::{CostFunction, Gradient};

/// Standard lower bound on each coordinate.
pub const STANDARD_LOWER: f64 = -100.0;
/// Standard upper bound on each coordinate.
pub const STANDARD_UPPER: f64 = 100.0;

// Schaffer N.2 (smooth, gradient provided)

/// Evaluates the Schaffer N.2 function at `x`. Requires `x.len() == 2`.
pub fn schaffer_n2(x: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), 2);
    let (a, b) = (x[0], x[1]);
    let u = a * a - b * b;
    let r2 = a * a + b * b;
    let d = 1.0 + 0.001 * r2;
    let su = u.sin();
    0.5 + (su * su - 0.5) / (d * d)
}

/// Writes the Schaffer N.2 gradient at `x` into `out`. Both slices must have
/// length 2.
pub fn schaffer_n2_gradient(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), 2);
    debug_assert_eq!(out.len(), 2);
    let (a, b) = (x[0], x[1]);
    let u = a * a - b * b;
    let r2 = a * a + b * b;
    let d = 1.0 + 0.001 * r2;
    let su = u.sin();
    let num = su * su - 0.5;
    let s2u = (2.0 * u).sin();
    let d3 = d * d * d;
    // f = 0.5 + num/d², d = 1 + 0.001·r². With num'_x = 2x·sin(2u),
    // d'_x = 0.002x:  ∂f/∂x = (num'_x·d − 0.004x·num) / d³.
    out[0] = (2.0 * a * s2u * d - 0.004 * a * num) / d3;
    // num'_y = −2y·sin(2u), d'_y = 0.002y.
    out[1] = (-2.0 * b * s2u * d - 0.004 * b * num) / d3;
}

/// Pre-wrapped Schaffer N.2 problem (fixed 2D). Generic over the parameter
/// backend `P`; the default `P = Vec<f64>` lets you write
/// `SchafferN2::default()` for the common case.
pub struct SchafferN2<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> SchafferN2<P> {
    /// Build a freshly typed problem instance.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for SchafferN2<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Catalog entry for Schaffer N.2.
pub static SCHAFFER_N2_SPEC: ProblemSpec = ProblemSpec {
    name: "Schaffer N.2",
    dim: Dimensionality::Fixed(2),
    properties: Properties {
        smooth: true,
        differentiable: true,
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
    description: "Concentric circular ripples decaying toward the origin: \
                  f(x, y) = 0.5 + (sin²(x²−y²) − 0.5)/(1+0.001(x²+y²))². Smooth \
                  with global minimum at (0, 0), value 0. Usual search domain \
                  is x, y ∈ [-100, 100].",
};

impl<P> HasSpec for SchafferN2<P> {
    const SPEC: &'static ProblemSpec = &SCHAFFER_N2_SPEC;
}

impl CostFunction for SchafferN2<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(schaffer_n2(x))
    }
}

impl Gradient for SchafferN2<Vec<f64>> {
    type Gradient = Vec<f64>;
    fn gradient(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; x.len()];
        schaffer_n2_gradient(x, &mut out);
        Ok(out)
    }
}

// Schaffer N.4 (non-differentiable, cost-only)

/// Evaluates the Schaffer N.4 function at `x`. Requires `x.len() == 2`.
pub fn schaffer_n4(x: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), 2);
    let (a, b) = (x[0], x[1]);
    let t = (a * a - b * b).abs();
    let r2 = a * a + b * b;
    let d = 1.0 + 0.001 * r2;
    let c = t.sin().cos();
    0.5 + (c * c - 0.5) / (d * d)
}

/// Pre-wrapped Schaffer N.4 problem (fixed 2D). Cost-only: the `|·|` term makes
/// it non-differentiable, so no [`Gradient`] impl is provided.
pub struct SchafferN4<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> SchafferN4<P> {
    /// Build a freshly typed problem instance.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for SchafferN4<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Catalog entry for Schaffer N.4.
pub static SCHAFFER_N4_SPEC: ProblemSpec = ProblemSpec {
    name: "Schaffer N.4",
    dim: Dimensionality::Fixed(2),
    properties: Properties {
        // |·| inside the ripple ⇒ non-differentiable along x² = y².
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
    description: "Non-differentiable variant of Schaffer N.2: \
                  f(x, y) = 0.5 + (cos²(sin(|x²−y²|)) − 0.5)/(1+0.001(x²+y²))². \
                  Four global minima near (0, ±1.25313) and (±1.25313, 0), \
                  value ≈ 0.292579. Usual search domain is x, y ∈ [-100, 100]; \
                  cost-only (sharp ridge along x² = y²).",
};

impl<P> HasSpec for SchafferN4<P> {
    const SPEC: &'static ProblemSpec = &SCHAFFER_N4_SPEC;
}

impl CostFunction for SchafferN4<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(schaffer_n4(x))
    }
}

#[cfg(feature = "nalgebra_all")]
mod nalgebra_impl {
    use super::{
        SchafferN2, SchafferN4, schaffer_n2, schaffer_n2_gradient, schaffer_n4,
    };
    use crate::{CostFunction, Gradient};
    use nalgebra::DVector;

    impl CostFunction for SchafferN2<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &DVector<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(schaffer_n2(x.as_slice()))
        }
    }

    impl Gradient for SchafferN2<DVector<f64>> {
        type Gradient = DVector<f64>;
        fn gradient(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, std::convert::Infallible> {
            let mut out = DVector::zeros(x.len());
            schaffer_n2_gradient(x.as_slice(), out.as_mut_slice());
            Ok(out)
        }
    }

    impl CostFunction for SchafferN4<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &DVector<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(schaffer_n4(x.as_slice()))
        }
    }
}

#[cfg(feature = "ndarray_all")]
mod ndarray_impl {
    use super::{
        SchafferN2, SchafferN4, schaffer_n2, schaffer_n2_gradient, schaffer_n4,
    };
    use crate::{CostFunction, Gradient};
    use ndarray::Array1;

    impl CostFunction for SchafferN2<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &Array1<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(schaffer_n2(x.as_slice().expect("Array1 is contiguous")))
        }
    }

    impl Gradient for SchafferN2<Array1<f64>> {
        type Gradient = Array1<f64>;
        fn gradient(
            &self,
            x: &Array1<f64>,
        ) -> Result<Array1<f64>, std::convert::Infallible> {
            let mut out = Array1::zeros(x.len());
            schaffer_n2_gradient(
                x.as_slice().expect("Array1 is contiguous"),
                out.as_slice_mut().expect("Array1 is contiguous"),
            );
            Ok(out)
        }
    }

    impl CostFunction for SchafferN4<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &Array1<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(schaffer_n4(x.as_slice().expect("Array1 is contiguous")))
        }
    }
}

#[cfg(feature = "faer_all")]
mod faer_impl {
    use super::{SchafferN2, SchafferN4};
    use crate::{CostFunction, Gradient};
    use faer::Col;

    // faer's `Col` doesn't expose a `&[f64]` directly across all 0.24 APIs we
    // care about, so we evaluate elementwise here rather than routing through
    // the slice-based primitives.
    impl CostFunction for SchafferN2<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            debug_assert_eq!(x.nrows(), 2);
            let (a, b) = (x[0], x[1]);
            let u = a * a - b * b;
            let d = 1.0 + 0.001 * (a * a + b * b);
            let su = u.sin();
            Ok(0.5 + (su * su - 0.5) / (d * d))
        }
    }

    impl Gradient for SchafferN2<Col<f64>> {
        type Gradient = Col<f64>;
        fn gradient(
            &self,
            x: &Col<f64>,
        ) -> Result<Col<f64>, std::convert::Infallible> {
            debug_assert_eq!(x.nrows(), 2);
            let (a, b) = (x[0], x[1]);
            let u = a * a - b * b;
            let d = 1.0 + 0.001 * (a * a + b * b);
            let su = u.sin();
            let num = su * su - 0.5;
            let s2u = (2.0 * u).sin();
            let d3 = d * d * d;
            let g0 = (2.0 * a * s2u * d - 0.004 * a * num) / d3;
            let g1 = (-2.0 * b * s2u * d - 0.004 * b * num) / d3;
            Ok(Col::<f64>::from_fn(2, |i| if i == 0 { g0 } else { g1 }))
        }
    }

    impl CostFunction for SchafferN4<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            debug_assert_eq!(x.nrows(), 2);
            let (a, b) = (x[0], x[1]);
            let t = (a * a - b * b).abs();
            let d = 1.0 + 0.001 * (a * a + b * b);
            let c = t.sin().cos();
            Ok(0.5 + (c * c - 0.5) / (d * d))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n2_minimum_is_zero_at_origin() {
        assert!(schaffer_n2(&[0.0, 0.0]).abs() < 1e-12);
    }

    #[test]
    fn n2_known_value_on_axis() {
        // f(1, 0) = 0.5 + (sin²(1) − 0.5) / (1 + 0.001)²
        assert!((schaffer_n2(&[1.0, 0.0]) - 0.7076579).abs() < 1e-6);
    }

    #[test]
    fn n2_gradient_zero_at_minimum() {
        let mut g = vec![0.0; 2];
        schaffer_n2_gradient(&[0.0, 0.0], &mut g);
        for v in g {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn n2_gradient_matches_finite_difference() {
        let x = [1.2, -0.7];
        let mut g = vec![0.0; x.len()];
        schaffer_n2_gradient(&x, &mut g);
        let h = 1e-6;
        for i in 0..x.len() {
            let mut xp = x;
            let mut xm = x;
            xp[i] += h;
            xm[i] -= h;
            let fd = (schaffer_n2(&xp) - schaffer_n2(&xm)) / (2.0 * h);
            assert!((g[i] - fd).abs() < 1e-5, "i={i}, g={}, fd={fd}", g[i]);
        }
    }

    #[test]
    fn n4_known_value_at_origin() {
        // f(0, 0) = 0.5 + (cos²(sin(0)) − 0.5) / 1 = 0.5 + (1 − 0.5) = 1.
        assert!((schaffer_n4(&[0.0, 0.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn n4_minimum_value_at_documented_optimum() {
        let f = schaffer_n4(&[0.0, 1.25313]);
        assert!((f - 0.292579).abs() < 1e-4, "got {f}");
    }

    #[test]
    fn specs_are_wired_up_via_has_spec_trait() {
        let s2 = <SchafferN2<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(s2.name, "Schaffer N.2");
        assert!(s2.properties.smooth);
        assert!(s2.properties.differentiable);
        assert!(matches!(s2.dim, Dimensionality::Fixed(2)));
        assert!(!s2.references.is_empty());

        let s4 = <SchafferN4<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(s4.name, "Schaffer N.4");
        assert!(!s4.properties.differentiable);
        assert!(matches!(s4.dim, Dimensionality::Fixed(2)));
        assert!(!s4.references.is_empty());
    }
}
