//! N-dimensional Levy function.
//!
//! ```text
//! wᵢ = 1 + (xᵢ − 1)/4
//! f(x) = sin²(π·w₁)
//!      + Σᵢ₌₁ⁿ⁻¹ (wᵢ − 1)²·[1 + 10·sin²(π·wᵢ + 1)]
//!      + (wₙ − 1)²·[1 + sin²(2π·wₙ)]
//! ```
//!
//! Multimodal but smooth: the affine `wᵢ` reparameterization places the global
//! minimum at `x = (1, …, 1)` with `f = 0`, surrounded by many local minima
//! from the `sin²` ripple terms. Unlike [`Ackley`](super::Ackley), Levy is
//! differentiable everywhere, so an analytic gradient is provided (a local
//! method started near the optimum converges; from far away it stalls in a
//! ripple). Standard search domain is `[−10, 10]^n`.

use core::marker::PhantomData;

use super::spec::{
    Dimensionality, HasSpec, ProblemSpec, Properties, Reference,
};
use crate::{BoxConstraints, CostFunction, Gradient};

/// Standard lower bound on each coordinate.
pub const STANDARD_LOWER: f64 = -10.0;
/// Standard upper bound on each coordinate.
pub const STANDARD_UPPER: f64 = 10.0;

/// Evaluates the Levy function at `x`. Works for any `n >= 1`.
pub fn levy(x: &[f64]) -> f64 {
    debug_assert!(!x.is_empty());
    let pi = core::f64::consts::PI;
    let n = x.len();
    let w = |i: usize| 1.0 + (x[i] - 1.0) / 4.0;

    let mut s = (pi * w(0)).sin().powi(2);
    for i in 0..n - 1 {
        let wi = w(i);
        let t = (pi * wi + 1.0).sin();
        s += (wi - 1.0).powi(2) * (1.0 + 10.0 * t * t);
    }
    let wl = w(n - 1);
    let tl = (2.0 * pi * wl).sin();
    s += (wl - 1.0).powi(2) * (1.0 + tl * tl);
    s
}

/// Writes the Levy gradient at `x` into `out`. Both slices must have the same
/// length `n >= 1`.
pub fn levy_gradient(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), out.len());
    debug_assert!(!x.is_empty());
    let pi = core::f64::consts::PI;
    let n = x.len();
    let w = |i: usize| 1.0 + (x[i] - 1.0) / 4.0;
    // dwᵢ/dxᵢ = 1/4 throughout.
    for g in out.iter_mut() {
        *g = 0.0;
    }

    // Term 1: sin²(π·w₁), depends on w₀.
    out[0] += 0.25 * pi * (2.0 * pi * w(0)).sin();

    // Middle terms i = 1..n−1 (zero-indexed 0..n−1):
    //   Mᵢ = (wᵢ − 1)²·[1 + 10·sin²(θ)], θ = π·wᵢ + 1.
    //   dMᵢ/dwᵢ = 2(wᵢ−1)(1 + 10·sin²θ) + (wᵢ−1)²·10π·sin(2θ).
    for (g, &xi) in out.iter_mut().zip(x.iter()).take(n - 1) {
        let wi = 1.0 + (xi - 1.0) / 4.0;
        let theta = pi * wi + 1.0;
        let s = theta.sin();
        let dm = 2.0 * (wi - 1.0) * (1.0 + 10.0 * s * s)
            + (wi - 1.0).powi(2) * 10.0 * pi * (2.0 * theta).sin();
        *g += 0.25 * dm;
    }

    // Last term: (wₙ − 1)²·[1 + sin²(φ)], φ = 2π·wₙ.
    //   dL/dwₙ = 2(wₙ−1)(1 + sin²φ) + (wₙ−1)²·2π·sin(2φ).
    let wl = w(n - 1);
    let phi = 2.0 * pi * wl;
    let sl = phi.sin();
    let dl = 2.0 * (wl - 1.0) * (1.0 + sl * sl)
        + (wl - 1.0).powi(2) * 2.0 * pi * (2.0 * phi).sin();
    out[n - 1] += 0.25 * dl;
}

/// Pre-wrapped Levy problem. Generic over the parameter backend `P`; the
/// default `P = Vec<f64>` lets you write `Levy::default()` for the common case.
/// Backend impls (`nalgebra::DVector<f64>`, `ndarray::Array1<f64>`,
/// `faer::Col<f64>`) are gated behind their respective features.
///
/// Carries no constraint metadata. For solvers that need explicit box bounds,
/// use [`LevyBoxed`].
pub struct Levy<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> Levy<P> {
    /// Build a freshly typed problem instance. Pair with one of the
    /// backend-specific impl blocks (Vec, nalgebra, ndarray, faer).
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for Levy<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Catalog entry for this problem.
pub static LEVY_SPEC: ProblemSpec = ProblemSpec {
    name: "Levy",
    dim: Dimensionality::NDimensional { min: 1 },
    properties: Properties {
        smooth: true,
        differentiable: true,
        convex: false,
        unimodal: false,
        separable: false,
        scalable: true,
    },
    references: &[Reference {
        citation: "Laguna & Martí (2005)",
        title: "Experimental testing of advanced scatter search designs for global optimization of multimodal functions",
        source: "Journal of Global Optimization, 33(2), 235–255",
        doi: Some("10.1007/s10898-004-1936-z"),
        url: None,
    }],
    description: "Smooth but multimodal: an affine reparameterization wᵢ = 1 + \
                  (xᵢ−1)/4 places the global minimum at x = (1, …, 1), value 0, \
                  surrounded by sin² ripple minima. Standard search domain is \
                  [−10, 10]ⁿ. Differentiable everywhere (analytic gradient).",
};

impl<P> HasSpec for Levy<P> {
    const SPEC: &'static ProblemSpec = &LEVY_SPEC;
}

impl CostFunction for Levy<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(levy(x))
    }
}

impl Gradient for Levy<Vec<f64>> {
    type Gradient = Vec<f64>;
    fn gradient(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; x.len()];
        levy_gradient(x, &mut out);
        Ok(out)
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_impl {
    use super::{Levy, levy, levy_gradient};
    use crate::{CostFunction, Gradient};
    use nalgebra::DVector;

    impl CostFunction for Levy<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &DVector<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(levy(x.as_slice()))
        }
    }

    impl Gradient for Levy<DVector<f64>> {
        type Gradient = DVector<f64>;
        fn gradient(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, std::convert::Infallible> {
            let mut out = DVector::zeros(x.len());
            levy_gradient(x.as_slice(), out.as_mut_slice());
            Ok(out)
        }
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::{Levy, levy, levy_gradient};
    use crate::{CostFunction, Gradient};
    use ndarray::Array1;

    impl CostFunction for Levy<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &Array1<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(levy(x.as_slice().expect("Array1 is contiguous")))
        }
    }

    impl Gradient for Levy<Array1<f64>> {
        type Gradient = Array1<f64>;
        fn gradient(
            &self,
            x: &Array1<f64>,
        ) -> Result<Array1<f64>, std::convert::Infallible> {
            let mut out = Array1::zeros(x.len());
            levy_gradient(
                x.as_slice().expect("Array1 is contiguous"),
                out.as_slice_mut().expect("Array1 is contiguous"),
            );
            Ok(out)
        }
    }
}

#[cfg(feature = "faer")]
mod faer_impl {
    use super::{Levy, levy, levy_gradient};
    use crate::{CostFunction, Gradient};
    use faer::Col;

    // The Levy math is index-coupled (first/middle/last terms differ), so we
    // collect into a `Vec` and route through the slice primitives rather than
    // duplicating the recurrence elementwise.
    impl CostFunction for Levy<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            let v: Vec<f64> = (0..x.nrows()).map(|i| x[i]).collect();
            Ok(levy(&v))
        }
    }

    impl Gradient for Levy<Col<f64>> {
        type Gradient = Col<f64>;
        fn gradient(
            &self,
            x: &Col<f64>,
        ) -> Result<Col<f64>, std::convert::Infallible> {
            let v: Vec<f64> = (0..x.nrows()).map(|i| x[i]).collect();
            let mut g = vec![0.0; v.len()];
            levy_gradient(&v, &mut g);
            Ok(Col::<f64>::from_fn(g.len(), |i| g[i]))
        }
    }
}

// ----------------------------------------------------------------------
// Boxed (constrained) form
// ----------------------------------------------------------------------

/// Levy function with explicit element-wise box bounds, suitable for
/// box-constrained solvers (L-BFGS-B, projected gradient, CMA-ES variants).
/// Implements both [`CostFunction`] and [`Gradient`] plus [`BoxConstraints`],
/// routing through the same raw [`levy`]/[`levy_gradient`] free functions as
/// the unconstrained [`Levy`]. The standard search domain `[−10, 10]ⁿ` is the
/// common case; build it with [`LevyBoxed::with_standard_bounds`].
pub struct LevyBoxed<P> {
    lower: P,
    upper: P,
}

impl<P> LevyBoxed<P> {
    /// Build a Levy problem with arbitrary element-wise bounds. Caller must
    /// ensure `lower[i] ≤ upper[i]` per component.
    pub fn new(lower: P, upper: P) -> Self {
        Self { lower, upper }
    }
}

impl<P> HasSpec for LevyBoxed<P> {
    const SPEC: &'static ProblemSpec = &LEVY_SPEC;
}

impl LevyBoxed<Vec<f64>> {
    /// Build the canonical Levy instance on `[−10, 10]ⁿ` for the requested
    /// dimension `n`.
    pub fn with_standard_bounds(n: usize) -> Self {
        Self {
            lower: vec![STANDARD_LOWER; n],
            upper: vec![STANDARD_UPPER; n],
        }
    }
}

impl CostFunction for LevyBoxed<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(levy(x))
    }
}

impl Gradient for LevyBoxed<Vec<f64>> {
    type Gradient = Vec<f64>;
    fn gradient(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; x.len()];
        levy_gradient(x, &mut out);
        Ok(out)
    }
}

impl BoxConstraints for LevyBoxed<Vec<f64>> {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_boxed_impl {
    use super::{
        LevyBoxed, STANDARD_LOWER, STANDARD_UPPER, levy, levy_gradient,
    };
    use crate::{BoxConstraints, CostFunction, Gradient};
    use nalgebra::DVector;

    impl LevyBoxed<DVector<f64>> {
        /// Build the canonical Levy instance on `[−10, 10]ⁿ` for the requested
        /// dimension `n`.
        pub fn with_standard_bounds(n: usize) -> Self {
            Self {
                lower: DVector::from_element(n, STANDARD_LOWER),
                upper: DVector::from_element(n, STANDARD_UPPER),
            }
        }
    }

    impl CostFunction for LevyBoxed<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &DVector<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(levy(x.as_slice()))
        }
    }

    impl Gradient for LevyBoxed<DVector<f64>> {
        type Gradient = DVector<f64>;
        fn gradient(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, std::convert::Infallible> {
            let mut out = DVector::zeros(x.len());
            levy_gradient(x.as_slice(), out.as_mut_slice());
            Ok(out)
        }
    }

    impl BoxConstraints for LevyBoxed<DVector<f64>> {
        fn lower(&self) -> &DVector<f64> {
            &self.lower
        }
        fn upper(&self) -> &DVector<f64> {
            &self.upper
        }
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_boxed_impl {
    use super::{
        LevyBoxed, STANDARD_LOWER, STANDARD_UPPER, levy, levy_gradient,
    };
    use crate::{BoxConstraints, CostFunction, Gradient};
    use ndarray::Array1;

    impl LevyBoxed<Array1<f64>> {
        /// Build the canonical Levy instance on `[−10, 10]ⁿ` for the requested
        /// dimension `n`.
        pub fn with_standard_bounds(n: usize) -> Self {
            Self {
                lower: Array1::from_elem(n, STANDARD_LOWER),
                upper: Array1::from_elem(n, STANDARD_UPPER),
            }
        }
    }

    impl CostFunction for LevyBoxed<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &Array1<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(levy(x.as_slice().expect("Array1 is contiguous")))
        }
    }

    impl Gradient for LevyBoxed<Array1<f64>> {
        type Gradient = Array1<f64>;
        fn gradient(
            &self,
            x: &Array1<f64>,
        ) -> Result<Array1<f64>, std::convert::Infallible> {
            let mut out = Array1::zeros(x.len());
            levy_gradient(
                x.as_slice().expect("Array1 is contiguous"),
                out.as_slice_mut().expect("Array1 is contiguous"),
            );
            Ok(out)
        }
    }

    impl BoxConstraints for LevyBoxed<Array1<f64>> {
        fn lower(&self) -> &Array1<f64> {
            &self.lower
        }
        fn upper(&self) -> &Array1<f64> {
            &self.upper
        }
    }
}

#[cfg(feature = "faer")]
mod faer_boxed_impl {
    use super::{
        LevyBoxed, STANDARD_LOWER, STANDARD_UPPER, levy, levy_gradient,
    };
    use crate::{BoxConstraints, CostFunction, Gradient};
    use faer::Col;

    impl LevyBoxed<Col<f64>> {
        /// Build the canonical Levy instance on `[−10, 10]ⁿ` for the requested
        /// dimension `n`.
        pub fn with_standard_bounds(n: usize) -> Self {
            Self {
                lower: Col::<f64>::from_fn(n, |_| STANDARD_LOWER),
                upper: Col::<f64>::from_fn(n, |_| STANDARD_UPPER),
            }
        }
    }

    impl CostFunction for LevyBoxed<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            let v: Vec<f64> = (0..x.nrows()).map(|i| x[i]).collect();
            Ok(levy(&v))
        }
    }

    impl Gradient for LevyBoxed<Col<f64>> {
        type Gradient = Col<f64>;
        fn gradient(
            &self,
            x: &Col<f64>,
        ) -> Result<Col<f64>, std::convert::Infallible> {
            let v: Vec<f64> = (0..x.nrows()).map(|i| x[i]).collect();
            let mut g = vec![0.0; v.len()];
            levy_gradient(&v, &mut g);
            Ok(Col::<f64>::from_fn(g.len(), |i| g[i]))
        }
    }

    impl BoxConstraints for LevyBoxed<Col<f64>> {
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
    fn minimum_is_zero_at_ones() {
        assert!(levy(&[1.0]).abs() < 1e-12);
        assert!(levy(&[1.0, 1.0]).abs() < 1e-12);
        assert!(levy(&[1.0; 7]).abs() < 1e-12);
    }

    #[test]
    fn known_value_at_fives() {
        // x = (5, 5) ⇒ w = (2, 2). Term1 = sin²(2π) = 0.
        // Middle (i = 1): (2−1)²·[1 + 10·sin²(2π+1)] = 1 + 10·sin²(1).
        // Last: (2−1)²·[1 + sin²(4π)] = 1.
        // f = 0 + (1 + 10·sin²1) + 1 = 2 + 10·sin²1 ≈ 9.0807342.
        let expected = 2.0 + 10.0 * 1.0_f64.sin().powi(2);
        assert!((levy(&[5.0, 5.0]) - expected).abs() < 1e-9);
    }

    #[test]
    fn gradient_zero_at_minimum() {
        let mut g = vec![0.0; 4];
        levy_gradient(&[1.0; 4], &mut g);
        for v in g {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let x = [-1.2, 0.7, 2.3];
        let mut g = vec![0.0; x.len()];
        levy_gradient(&x, &mut g);
        let h = 1e-6;
        for i in 0..x.len() {
            let mut xp = x;
            let mut xm = x;
            xp[i] += h;
            xm[i] -= h;
            let fd = (levy(&xp) - levy(&xm)) / (2.0 * h);
            assert!((g[i] - fd).abs() < 1e-5, "i={i}, g={}, fd={fd}", g[i]);
        }
    }

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        let spec = <Levy<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(spec.name, "Levy");
        assert!(spec.properties.smooth);
        assert!(spec.properties.differentiable);
        assert!(spec.properties.scalable);
        assert!(matches!(spec.dim, Dimensionality::NDimensional { min: 1 }));
        assert!(!spec.references.is_empty());
    }

    #[test]
    fn boxed_form_exposes_standard_bounds_and_shares_math() {
        let p = LevyBoxed::<Vec<f64>>::with_standard_bounds(4);
        let lo = <LevyBoxed<Vec<f64>> as BoxConstraints>::lower(&p);
        let hi = <LevyBoxed<Vec<f64>> as BoxConstraints>::upper(&p);
        assert!(lo.iter().all(|&v| v == STANDARD_LOWER));
        assert!(hi.iter().all(|&v| v == STANDARD_UPPER));

        let x = vec![0.3, -0.7, 1.2, 2.1];
        assert!((p.cost(&x).unwrap() - levy(&x)).abs() < 1e-12);
        let mut g = vec![0.0; x.len()];
        levy_gradient(&x, &mut g);
        assert_eq!(p.gradient(&x).unwrap(), g);
    }

    #[test]
    fn boxed_form_reuses_levy_spec() {
        let spec = <LevyBoxed<Vec<f64>> as HasSpec>::SPEC;
        assert!(core::ptr::eq(spec, &LEVY_SPEC));
    }
}
