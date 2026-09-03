//! N-dimensional Styblinski-Tang function.
//!
//! `f(x) = ½·Σᵢ (xᵢ⁴ − 16·xᵢ² + 5·xᵢ)`
//!
//! Separable quartic: each coordinate contributes an identical double-well
//! `½(t⁴ − 16t² + 5t)` with a deeper minimum near `t ≈ −2.903534` and a
//! shallower one near `t ≈ +2.7`, so the surface has `2ⁿ` local minima at the
//! well combinations. The global minimum is at `xᵢ ≈ −2.903534` for all `i`,
//! value `≈ −39.16599·n`. Smooth and differentiable (analytic gradient
//! provided); separable, so coordinate-wise methods do well. Standard search
//! domain is `[−5, 5]^n`.

use core::marker::PhantomData;

use super::spec::{
    Dimensionality, HasSpec, ProblemSpec, Properties, Reference,
};
use crate::{BoxConstraints, CostFunction, Gradient};

/// Standard lower bound on each coordinate.
pub const STANDARD_LOWER: f64 = -5.0;
/// Standard upper bound on each coordinate.
pub const STANDARD_UPPER: f64 = 5.0;

/// Per-coordinate value of the global minimizer (root of `2t³ − 16t + 2.5`).
pub const MINIMIZER: f64 = -2.903534;

/// Evaluates the Styblinski-Tang function at `x`. Works for any `n >= 1`.
pub fn styblinski_tang(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for &v in x.iter() {
        let v2 = v * v;
        s += v2 * v2 - 16.0 * v2 + 5.0 * v;
    }
    0.5 * s
}

/// Writes the Styblinski-Tang gradient at `x` into `out`. Both slices must have
/// the same length.
pub fn styblinski_tang_gradient(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), out.len());
    // ∂f/∂xᵢ = ½(4xᵢ³ − 32xᵢ + 5) = 2xᵢ³ − 16xᵢ + 2.5
    for (g, &v) in out.iter_mut().zip(x.iter()) {
        *g = 2.0 * v * v * v - 16.0 * v + 2.5;
    }
}

/// Pre-wrapped Styblinski-Tang problem. Generic over the parameter backend `P`;
/// the default `P = Vec<f64>` lets you write `StyblinskiTang::default()` for the
/// common case. Backend impls (`nalgebra::DVector<f64>`, `ndarray::Array1<f64>`,
/// `faer::Col<f64>`) are gated behind their respective features.
///
/// Carries no constraint metadata. For solvers that need explicit box bounds,
/// use [`StyblinskiTangBoxed`].
pub struct StyblinskiTang<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> StyblinskiTang<P> {
    /// Build a freshly typed problem instance. Pair with one of the
    /// backend-specific impl blocks (Vec, nalgebra, ndarray, faer).
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for StyblinskiTang<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Catalog entry for this problem.
pub static STYBLINSKI_TANG_SPEC: ProblemSpec = ProblemSpec {
    name: "Styblinski-Tang",
    dim: Dimensionality::NDimensional { min: 1 },
    properties: Properties {
        smooth: true,
        differentiable: true,
        convex: false,
        // 2ⁿ local minima (per-coordinate double wells).
        unimodal: false,
        separable: true,
        scalable: true,
    },
    references: &[Reference {
        citation: "Styblinski & Tang (1990)",
        title: "Experiments in nonconvex optimization: Stochastic approximation with function smoothing and simulated annealing",
        source: "Neural Networks, 3(4), 467–483",
        doi: Some("10.1016/0893-6080(90)90029-K"),
        url: None,
    }],
    description: "Separable quartic double-well per coordinate: f(x) = \
                  ½·Σ(xᵢ⁴ − 16xᵢ² + 5xᵢ), with 2ⁿ local minima. Global minimum \
                  at xᵢ ≈ −2.903534 for all i, value ≈ −39.16599·n. Smooth, \
                  differentiable, separable; standard search domain [−5, 5]ⁿ.",
};

impl<P> HasSpec for StyblinskiTang<P> {
    const SPEC: &'static ProblemSpec = &STYBLINSKI_TANG_SPEC;
}

impl CostFunction for StyblinskiTang<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(styblinski_tang(x))
    }
}

impl Gradient for StyblinskiTang<Vec<f64>> {
    type Gradient = Vec<f64>;
    fn gradient(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; x.len()];
        styblinski_tang_gradient(x, &mut out);
        Ok(out)
    }
}

#[cfg(feature = "nalgebra_all")]
mod nalgebra_impl {
    use super::{StyblinskiTang, styblinski_tang, styblinski_tang_gradient};
    use crate::{CostFunction, Gradient};
    use nalgebra::DVector;

    impl CostFunction for StyblinskiTang<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &DVector<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(styblinski_tang(x.as_slice()))
        }
    }

    impl Gradient for StyblinskiTang<DVector<f64>> {
        type Gradient = DVector<f64>;
        fn gradient(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, std::convert::Infallible> {
            let mut out = DVector::zeros(x.len());
            styblinski_tang_gradient(x.as_slice(), out.as_mut_slice());
            Ok(out)
        }
    }
}

#[cfg(feature = "ndarray_all")]
mod ndarray_impl {
    use super::{StyblinskiTang, styblinski_tang, styblinski_tang_gradient};
    use crate::{CostFunction, Gradient};
    use ndarray::Array1;

    impl CostFunction for StyblinskiTang<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &Array1<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(styblinski_tang(x.as_slice().expect("Array1 is contiguous")))
        }
    }

    impl Gradient for StyblinskiTang<Array1<f64>> {
        type Gradient = Array1<f64>;
        fn gradient(
            &self,
            x: &Array1<f64>,
        ) -> Result<Array1<f64>, std::convert::Infallible> {
            let mut out = Array1::zeros(x.len());
            styblinski_tang_gradient(
                x.as_slice().expect("Array1 is contiguous"),
                out.as_slice_mut().expect("Array1 is contiguous"),
            );
            Ok(out)
        }
    }
}

#[cfg(feature = "faer_all")]
mod faer_impl {
    use super::StyblinskiTang;
    use crate::{CostFunction, Gradient};
    use faer::Col;

    // faer's `Col` doesn't expose a `&[f64]` directly across all 0.24 APIs we
    // care about, so we evaluate elementwise here rather than routing through
    // the slice-based primitives.
    impl CostFunction for StyblinskiTang<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            let mut s = 0.0;
            for i in 0..x.nrows() {
                let v = x[i];
                let v2 = v * v;
                s += v2 * v2 - 16.0 * v2 + 5.0 * v;
            }
            Ok(0.5 * s)
        }
    }

    impl Gradient for StyblinskiTang<Col<f64>> {
        type Gradient = Col<f64>;
        fn gradient(
            &self,
            x: &Col<f64>,
        ) -> Result<Col<f64>, std::convert::Infallible> {
            Ok(Col::<f64>::from_fn(x.nrows(), |i| {
                let v = x[i];
                2.0 * v * v * v - 16.0 * v + 2.5
            }))
        }
    }
}

// Boxed (constrained) form

/// Styblinski-Tang function with explicit element-wise box bounds, suitable for
/// box-constrained solvers (L-BFGS-B, projected gradient, CMA-ES variants).
/// Implements both [`CostFunction`] and [`Gradient`] plus [`BoxConstraints`],
/// routing through the same raw [`styblinski_tang`]/[`styblinski_tang_gradient`]
/// free functions as the unconstrained [`StyblinskiTang`]. The standard search
/// domain `[−5, 5]ⁿ` is the common case; build it with
/// [`StyblinskiTangBoxed::with_standard_bounds`].
pub struct StyblinskiTangBoxed<P> {
    lower: P,
    upper: P,
}

impl<P> StyblinskiTangBoxed<P> {
    /// Build a Styblinski-Tang problem with arbitrary element-wise bounds.
    /// Caller must ensure `lower[i] ≤ upper[i]` per component.
    pub fn new(lower: P, upper: P) -> Self {
        Self { lower, upper }
    }
}

impl<P> HasSpec for StyblinskiTangBoxed<P> {
    const SPEC: &'static ProblemSpec = &STYBLINSKI_TANG_SPEC;
}

impl StyblinskiTangBoxed<Vec<f64>> {
    /// Build the canonical Styblinski-Tang instance on `[−5, 5]ⁿ` for the
    /// requested dimension `n`.
    pub fn with_standard_bounds(n: usize) -> Self {
        Self {
            lower: vec![STANDARD_LOWER; n],
            upper: vec![STANDARD_UPPER; n],
        }
    }
}

impl CostFunction for StyblinskiTangBoxed<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(styblinski_tang(x))
    }
}

impl Gradient for StyblinskiTangBoxed<Vec<f64>> {
    type Gradient = Vec<f64>;
    fn gradient(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; x.len()];
        styblinski_tang_gradient(x, &mut out);
        Ok(out)
    }
}

impl BoxConstraints for StyblinskiTangBoxed<Vec<f64>> {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

#[cfg(feature = "nalgebra_all")]
mod nalgebra_boxed_impl {
    use super::{
        STANDARD_LOWER, STANDARD_UPPER, StyblinskiTangBoxed, styblinski_tang,
        styblinski_tang_gradient,
    };
    use crate::{BoxConstraints, CostFunction, Gradient};
    use nalgebra::DVector;

    impl StyblinskiTangBoxed<DVector<f64>> {
        /// Build the canonical Styblinski-Tang instance on `[−5, 5]ⁿ` for the
        /// requested dimension `n`.
        pub fn with_standard_bounds(n: usize) -> Self {
            Self {
                lower: DVector::from_element(n, STANDARD_LOWER),
                upper: DVector::from_element(n, STANDARD_UPPER),
            }
        }
    }

    impl CostFunction for StyblinskiTangBoxed<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &DVector<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(styblinski_tang(x.as_slice()))
        }
    }

    impl Gradient for StyblinskiTangBoxed<DVector<f64>> {
        type Gradient = DVector<f64>;
        fn gradient(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, std::convert::Infallible> {
            let mut out = DVector::zeros(x.len());
            styblinski_tang_gradient(x.as_slice(), out.as_mut_slice());
            Ok(out)
        }
    }

    impl BoxConstraints for StyblinskiTangBoxed<DVector<f64>> {
        fn lower(&self) -> &DVector<f64> {
            &self.lower
        }
        fn upper(&self) -> &DVector<f64> {
            &self.upper
        }
    }
}

#[cfg(feature = "ndarray_all")]
mod ndarray_boxed_impl {
    use super::{
        STANDARD_LOWER, STANDARD_UPPER, StyblinskiTangBoxed, styblinski_tang,
        styblinski_tang_gradient,
    };
    use crate::{BoxConstraints, CostFunction, Gradient};
    use ndarray::Array1;

    impl StyblinskiTangBoxed<Array1<f64>> {
        /// Build the canonical Styblinski-Tang instance on `[−5, 5]ⁿ` for the
        /// requested dimension `n`.
        pub fn with_standard_bounds(n: usize) -> Self {
            Self {
                lower: Array1::from_elem(n, STANDARD_LOWER),
                upper: Array1::from_elem(n, STANDARD_UPPER),
            }
        }
    }

    impl CostFunction for StyblinskiTangBoxed<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &Array1<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(styblinski_tang(x.as_slice().expect("Array1 is contiguous")))
        }
    }

    impl Gradient for StyblinskiTangBoxed<Array1<f64>> {
        type Gradient = Array1<f64>;
        fn gradient(
            &self,
            x: &Array1<f64>,
        ) -> Result<Array1<f64>, std::convert::Infallible> {
            let mut out = Array1::zeros(x.len());
            styblinski_tang_gradient(
                x.as_slice().expect("Array1 is contiguous"),
                out.as_slice_mut().expect("Array1 is contiguous"),
            );
            Ok(out)
        }
    }

    impl BoxConstraints for StyblinskiTangBoxed<Array1<f64>> {
        fn lower(&self) -> &Array1<f64> {
            &self.lower
        }
        fn upper(&self) -> &Array1<f64> {
            &self.upper
        }
    }
}

#[cfg(feature = "faer_all")]
mod faer_boxed_impl {
    use super::{STANDARD_LOWER, STANDARD_UPPER, StyblinskiTangBoxed};
    use crate::{BoxConstraints, CostFunction, Gradient};
    use faer::Col;

    impl StyblinskiTangBoxed<Col<f64>> {
        /// Build the canonical Styblinski-Tang instance on `[−5, 5]ⁿ` for the
        /// requested dimension `n`.
        pub fn with_standard_bounds(n: usize) -> Self {
            Self {
                lower: Col::<f64>::from_fn(n, |_| STANDARD_LOWER),
                upper: Col::<f64>::from_fn(n, |_| STANDARD_UPPER),
            }
        }
    }

    impl CostFunction for StyblinskiTangBoxed<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            let mut s = 0.0;
            for i in 0..x.nrows() {
                let v = x[i];
                let v2 = v * v;
                s += v2 * v2 - 16.0 * v2 + 5.0 * v;
            }
            Ok(0.5 * s)
        }
    }

    impl Gradient for StyblinskiTangBoxed<Col<f64>> {
        type Gradient = Col<f64>;
        fn gradient(
            &self,
            x: &Col<f64>,
        ) -> Result<Col<f64>, std::convert::Infallible> {
            Ok(Col::<f64>::from_fn(x.nrows(), |i| {
                let v = x[i];
                2.0 * v * v * v - 16.0 * v + 2.5
            }))
        }
    }

    impl BoxConstraints for StyblinskiTangBoxed<Col<f64>> {
        fn lower(&self) -> &Col<f64> {
            &self.lower
        }
        fn upper(&self) -> &Col<f64> {
            &self.upper
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_zero_at_origin() {
        assert!(styblinski_tang(&[0.0]).abs() < 1e-12);
        assert!(styblinski_tang(&[0.0, 0.0, 0.0]).abs() < 1e-12);
    }

    #[test]
    fn known_value_at_ones() {
        // Per coordinate: ½(1 − 16 + 5) = −5. For n = 2 that's −10.
        assert!((styblinski_tang(&[1.0, 1.0]) - (-10.0)).abs() < 1e-12);
    }

    #[test]
    fn minimum_value_at_documented_optimum() {
        // f(MINIMIZER, …) ≈ −39.16599·n.
        let x = [MINIMIZER, MINIMIZER];
        let f = styblinski_tang(&x);
        assert!((f - (-39.16599 * 2.0)).abs() < 1e-2, "got {f}");
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let x = [-1.2, 0.7, 2.3];
        let mut g = vec![0.0; x.len()];
        styblinski_tang_gradient(&x, &mut g);
        let h = 1e-6;
        for i in 0..x.len() {
            let mut xp = x;
            let mut xm = x;
            xp[i] += h;
            xm[i] -= h;
            let fd = (styblinski_tang(&xp) - styblinski_tang(&xm)) / (2.0 * h);
            assert!((g[i] - fd).abs() < 1e-5, "i={i}, g={}, fd={fd}", g[i]);
        }
    }

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        let spec = <StyblinskiTang<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(spec.name, "Styblinski-Tang");
        assert!(spec.properties.smooth);
        assert!(spec.properties.separable);
        assert!(spec.properties.scalable);
        assert!(matches!(spec.dim, Dimensionality::NDimensional { min: 1 }));
        assert!(!spec.references.is_empty());
    }

    #[test]
    fn boxed_form_exposes_standard_bounds_and_shares_math() {
        let p = StyblinskiTangBoxed::<Vec<f64>>::with_standard_bounds(3);
        let lo = <StyblinskiTangBoxed<Vec<f64>> as BoxConstraints>::lower(&p);
        let hi = <StyblinskiTangBoxed<Vec<f64>> as BoxConstraints>::upper(&p);
        assert!(lo.iter().all(|&v| v == STANDARD_LOWER));
        assert!(hi.iter().all(|&v| v == STANDARD_UPPER));

        let x = vec![0.3, -0.7, 1.2];
        assert!((p.cost(&x).unwrap() - styblinski_tang(&x)).abs() < 1e-12);
        let mut g = vec![0.0; x.len()];
        styblinski_tang_gradient(&x, &mut g);
        assert_eq!(p.gradient(&x).unwrap(), g);
    }

    #[test]
    fn boxed_form_reuses_spec() {
        let spec = <StyblinskiTangBoxed<Vec<f64>> as HasSpec>::SPEC;
        assert!(core::ptr::eq(spec, &STYBLINSKI_TANG_SPEC));
    }
}
