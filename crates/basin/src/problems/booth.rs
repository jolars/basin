//! 2D Booth function.
//!
//! `f(x, y) = (x + 2y − 7)² + (2x + y − 5)²`
//!
//! Smooth, convex 2D quadratic. Global minimum at `(x, y) = (1, 3)` with
//! `f = 0`. Usual search domain is `x, y ∈ [-10, 10]`. The cost is the squared
//! residual of the linear system `[[1, 2], [2, 1]] · [x, y]ᵀ = [7, 5]ᵀ`, so
//! the Hessian is constant SPD and any reasonable first-order method should
//! converge in a handful of steps. Useful as an easy convex 2D gradient sanity
//! check distinct from the separable Sphere.

use core::marker::PhantomData;

use super::spec::{Dimensionality, HasSpec, ProblemSpec, Properties, Reference};
use crate::{BoxConstraints, CostFunction, Gradient, Residual};

/// Evaluates the Booth function at `x`. Requires `x.len() == 2`.
pub fn booth(x: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), 2);
    let (a, b) = (x[0], x[1]);
    let t1 = a + 2.0 * b - 7.0;
    let t2 = 2.0 * a + b - 5.0;
    t1 * t1 + t2 * t2
}

/// Writes the Booth residuals at `x` into `out`. Both slices must have
/// length 2. `r(x, y) = [x + 2y − 7, 2x + y − 5]` so
/// `Σ rᵢ² = booth(x, y)` exactly — the unscaled-sum convention shared
/// with `RosenbrockResiduals`. Zero at `(1, 3)`.
pub fn booth_residuals(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), 2);
    debug_assert_eq!(out.len(), 2);
    out[0] = x[0] + 2.0 * x[1] - 7.0;
    out[1] = 2.0 * x[0] + x[1] - 5.0;
}

/// Writes the constant 2×2 Booth Jacobian `[[1, 2], [2, 1]]` into `out`
/// in row-major order. `out.len()` must be 4.
pub fn booth_residuals_jacobian(out: &mut [f64]) {
    debug_assert_eq!(out.len(), 4);
    out[0] = 1.0;
    out[1] = 2.0;
    out[2] = 2.0;
    out[3] = 1.0;
}

/// Writes the Booth gradient at `x` into `out`. Both slices must have length 2.
pub fn booth_gradient(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), 2);
    debug_assert_eq!(out.len(), 2);
    let (a, b) = (x[0], x[1]);
    let t1 = a + 2.0 * b - 7.0;
    let t2 = 2.0 * a + b - 5.0;
    // ∂f/∂x = 2·t1 + 4·t2
    out[0] = 2.0 * t1 + 4.0 * t2;
    // ∂f/∂y = 4·t1 + 2·t2
    out[1] = 4.0 * t1 + 2.0 * t2;
}

/// Pre-wrapped Booth problem (fixed 2D). Generic over the parameter backend
/// `P`; the default `P = Vec<f64>` lets you write `Booth::default()` for the
/// common case. Backend impls (`nalgebra::DVector<f64>`, `ndarray::Array1<f64>`,
/// `faer::Col<f64>`) are gated behind their respective features.
pub struct Booth<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> Booth<P> {
    /// Build a freshly typed problem instance. Pair with one of the
    /// backend-specific impl blocks (Vec, nalgebra, ndarray, faer).
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for Booth<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Catalogue entry for this problem.
pub static BOOTH_SPEC: ProblemSpec = ProblemSpec {
    name: "Booth",
    dim: Dimensionality::Fixed(2),
    properties: Properties {
        smooth: true,
        differentiable: true,
        // Sum of squared affine forms in (x, y); the Hessian [[10, 8], [8, 10]]
        // is constant and positive definite, so f is strictly convex on R².
        convex: true,
        unimodal: true,
        // Cross term 8xy in the expansion prevents per-coordinate decomposition.
        separable: false,
        scalable: false,
    },
    references: &[Reference {
        citation: "Jamil & Yang (2013)",
        title: "A Literature Survey of Benchmark Functions For Global Optimisation Problems",
        source: "International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150–194",
        doi: Some("10.1504/IJMMNO.2013.055204"),
        url: Some("https://arxiv.org/abs/1308.4008"),
    }],
    description: "Smooth convex 2D quadratic: f(x, y) = (x + 2y − 7)² + \
                  (2x + y − 5)². Global minimum at (x, y) = (1, 3), value 0. \
                  Usual search domain is x, y ∈ [-10, 10]. The Hessian is \
                  constant and positive definite, so first-order methods \
                  converge in a handful of steps.",
};

impl<P> HasSpec for Booth<P> {
    const SPEC: &'static ProblemSpec = &BOOTH_SPEC;
}

impl CostFunction for Booth<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(booth(x))
    }
}

impl Gradient for Booth<Vec<f64>> {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; x.len()];
        booth_gradient(x, &mut out);
        Ok(out)
    }
}

/// Booth function with explicit element-wise box bounds, suitable for
/// constrained solvers ([`ProjectedGradientDescent`](crate::solver::ProjectedGradientDescent)).
///
/// Carries the bounds as data on the problem (tenet 4 in `CONTRIBUTING.md`)
/// and routes cost and gradient through the same raw [`booth`] /
/// [`booth_gradient`] free fns as the unconstrained [`Booth`]. The
/// global minimum `(1, 3)` of unconstrained Booth is on the interior
/// of `[-10, 10]²` but lies *outside* tighter boxes such as
/// `[-1, 1]²`, where the constrained optimum is the box corner
/// `(1, 1)` — see the integration tests for projection-active
/// behavior.
pub struct BoothBoxed<P> {
    lower: P,
    upper: P,
}

impl<P> BoothBoxed<P> {
    /// Build a Booth problem with the given element-wise bounds.
    /// Caller must ensure `lower[i] ≤ upper[i]` for each component
    /// (the projection primitive [`f64::clamp`] panics on violation).
    pub fn new(lower: P, upper: P) -> Self {
        Self { lower, upper }
    }
}

impl<P> HasSpec for BoothBoxed<P> {
    const SPEC: &'static ProblemSpec = &BOOTH_SPEC;
}

impl CostFunction for BoothBoxed<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(booth(x))
    }
}

impl Gradient for BoothBoxed<Vec<f64>> {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; x.len()];
        booth_gradient(x, &mut out);
        Ok(out)
    }
}

impl BoxConstraints for BoothBoxed<Vec<f64>> {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

// ----------------------------------------------------------------------
// Residual-form Booth (n = 2)
// ----------------------------------------------------------------------
// Booth factors as a 2-residual least-squares problem
// `r = [x+2y−7, 2x+y−5]` with constant Jacobian `[[1,2],[2,1]]` and
// `Σ rᵢ² == booth(x, y)` exactly (unscaled-sum convention shared with
// `RosenbrockResiduals`). Used as a fixture for `Trf` (S6) — the
// constrained optimum on `[-1, 1]²` is the corner (1, 1), giving a
// load-bearing edge-active test case.

/// Booth function exposed as a least-squares problem (2 residuals, 2
/// parameters). Shares [`BOOTH_SPEC`] with the cost-form [`Booth`]
/// wrapper.
pub struct BoothResiduals<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> BoothResiduals<P> {
    /// Build a freshly typed Booth-as-residuals instance.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for BoothResiduals<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P> HasSpec for BoothResiduals<P> {
    const SPEC: &'static ProblemSpec = &BOOTH_SPEC;
}

/// Booth-as-residuals with explicit element-wise box bounds, suitable
/// for [`Trf`](crate::solver::Trf). The unconstrained min `(1, 3)`
/// lies outside tighter boxes (e.g. `[-1, 1]²`), where the constrained
/// optimum sits at the box corner `(1, 1)` — a load-bearing edge-active
/// test case where the unprojected `‖∇f‖_∞` is large but the BCL
/// scaled-gradient measure `‖D · Jᵀr‖_∞` vanishes.
pub struct BoothBoxedResiduals<P> {
    lower: P,
    upper: P,
}

impl<P> BoothBoxedResiduals<P> {
    /// Build a Booth-as-residuals problem with the given element-wise
    /// bounds. Caller must ensure `lower[i] ≤ upper[i]`.
    pub fn new(lower: P, upper: P) -> Self {
        Self { lower, upper }
    }
}

impl<P> HasSpec for BoothBoxedResiduals<P> {
    const SPEC: &'static ProblemSpec = &BOOTH_SPEC;
}

impl CostFunction for BoothResiduals<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(booth(x))
    }
}

impl Residual for BoothResiduals<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = std::convert::Infallible;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; 2];
        booth_residuals(x, &mut out);
        Ok(out)
    }
}

impl CostFunction for BoothBoxedResiduals<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(booth(x))
    }
}

impl Residual for BoothBoxedResiduals<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = std::convert::Infallible;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; 2];
        booth_residuals(x, &mut out);
        Ok(out)
    }
}

impl BoxConstraints for BoothBoxedResiduals<Vec<f64>> {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_impl {
    use super::{
        Booth, BoothBoxed, BoothBoxedResiduals, BoothResiduals, booth, booth_gradient,
        booth_residuals, booth_residuals_jacobian,
    };
    use crate::{BoxConstraints, CostFunction, Gradient, Jacobian, Residual};
    use nalgebra::{DMatrix, DVector};

    impl CostFunction for Booth<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(booth(x.as_slice()))
        }
    }

    impl Gradient for Booth<DVector<f64>> {
        type Gradient = DVector<f64>;
        fn gradient(&self, x: &DVector<f64>) -> Result<DVector<f64>, std::convert::Infallible> {
            let mut out = DVector::zeros(x.len());
            booth_gradient(x.as_slice(), out.as_mut_slice());
            Ok(out)
        }
    }

    impl CostFunction for BoothBoxed<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(booth(x.as_slice()))
        }
    }

    impl Gradient for BoothBoxed<DVector<f64>> {
        type Gradient = DVector<f64>;
        fn gradient(&self, x: &DVector<f64>) -> Result<DVector<f64>, std::convert::Infallible> {
            let mut out = DVector::zeros(x.len());
            booth_gradient(x.as_slice(), out.as_mut_slice());
            Ok(out)
        }
    }

    impl BoxConstraints for BoothBoxed<DVector<f64>> {
        fn lower(&self) -> &DVector<f64> {
            &self.lower
        }
        fn upper(&self) -> &DVector<f64> {
            &self.upper
        }
    }

    impl CostFunction for BoothResiduals<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(booth(x.as_slice()))
        }
    }

    impl Residual for BoothResiduals<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = DVector<f64>;
        type Error = std::convert::Infallible;
        fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, std::convert::Infallible> {
            let mut out = DVector::zeros(2);
            booth_residuals(x.as_slice(), out.as_mut_slice());
            Ok(out)
        }
    }

    impl Jacobian for BoothResiduals<DVector<f64>> {
        type Jacobian = DMatrix<f64>;
        fn jacobian(&self, _x: &DVector<f64>) -> Result<DMatrix<f64>, std::convert::Infallible> {
            // Constant 2×2 Jacobian — independent of x.
            let mut buf = [0.0_f64; 4];
            booth_residuals_jacobian(&mut buf);
            Ok(DMatrix::from_row_slice(2, 2, &buf))
        }
    }

    impl CostFunction for BoothBoxedResiduals<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(booth(x.as_slice()))
        }
    }

    impl Residual for BoothBoxedResiduals<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = DVector<f64>;
        type Error = std::convert::Infallible;
        fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, std::convert::Infallible> {
            let mut out = DVector::zeros(2);
            booth_residuals(x.as_slice(), out.as_mut_slice());
            Ok(out)
        }
    }

    impl Jacobian for BoothBoxedResiduals<DVector<f64>> {
        type Jacobian = DMatrix<f64>;
        fn jacobian(&self, _x: &DVector<f64>) -> Result<DMatrix<f64>, std::convert::Infallible> {
            let mut buf = [0.0_f64; 4];
            booth_residuals_jacobian(&mut buf);
            Ok(DMatrix::from_row_slice(2, 2, &buf))
        }
    }

    impl BoxConstraints for BoothBoxedResiduals<DVector<f64>> {
        fn lower(&self) -> &DVector<f64> {
            &self.lower
        }
        fn upper(&self) -> &DVector<f64> {
            &self.upper
        }
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::{Booth, BoothBoxed, booth, booth_gradient};
    use crate::{BoxConstraints, CostFunction, Gradient};
    use ndarray::Array1;

    impl CostFunction for Booth<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Array1<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(booth(x.as_slice().expect("Array1 is contiguous")))
        }
    }

    impl Gradient for Booth<Array1<f64>> {
        type Gradient = Array1<f64>;
        fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, std::convert::Infallible> {
            let mut out = Array1::zeros(x.len());
            booth_gradient(
                x.as_slice().expect("Array1 is contiguous"),
                out.as_slice_mut().expect("Array1 is contiguous"),
            );
            Ok(out)
        }
    }

    impl CostFunction for BoothBoxed<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Array1<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(booth(x.as_slice().expect("Array1 is contiguous")))
        }
    }

    impl Gradient for BoothBoxed<Array1<f64>> {
        type Gradient = Array1<f64>;
        fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, std::convert::Infallible> {
            let mut out = Array1::zeros(x.len());
            booth_gradient(
                x.as_slice().expect("Array1 is contiguous"),
                out.as_slice_mut().expect("Array1 is contiguous"),
            );
            Ok(out)
        }
    }

    impl BoxConstraints for BoothBoxed<Array1<f64>> {
        fn lower(&self) -> &Array1<f64> {
            &self.lower
        }
        fn upper(&self) -> &Array1<f64> {
            &self.upper
        }
    }
}

#[cfg(feature = "faer")]
mod faer_impl {
    use super::{Booth, BoothBoxed, BoothBoxedResiduals, BoothResiduals, booth_residuals_jacobian};
    use crate::{BoxConstraints, CostFunction, Gradient, Jacobian, Residual};
    use faer::{Col, Mat};

    fn cost_inline(x: &Col<f64>) -> f64 {
        debug_assert_eq!(x.nrows(), 2);
        let (a, b) = (x[0], x[1]);
        let t1 = a + 2.0 * b - 7.0;
        let t2 = 2.0 * a + b - 5.0;
        t1 * t1 + t2 * t2
    }

    fn grad_inline(x: &Col<f64>) -> Col<f64> {
        debug_assert_eq!(x.nrows(), 2);
        let (a, b) = (x[0], x[1]);
        let t1 = a + 2.0 * b - 7.0;
        let t2 = 2.0 * a + b - 5.0;
        let g0 = 2.0 * t1 + 4.0 * t2;
        let g1 = 4.0 * t1 + 2.0 * t2;
        Col::<f64>::from_fn(2, |i| if i == 0 { g0 } else { g1 })
    }

    // faer's `Col` doesn't expose a `&[f64]` directly across all 0.24 APIs we
    // care about, so we evaluate elementwise here rather than routing through
    // the slice-based primitives.
    impl CostFunction for Booth<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(cost_inline(x))
        }
    }

    impl Gradient for Booth<Col<f64>> {
        type Gradient = Col<f64>;
        fn gradient(&self, x: &Col<f64>) -> Result<Col<f64>, std::convert::Infallible> {
            Ok(grad_inline(x))
        }
    }

    impl CostFunction for BoothBoxed<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(cost_inline(x))
        }
    }

    impl Gradient for BoothBoxed<Col<f64>> {
        type Gradient = Col<f64>;
        fn gradient(&self, x: &Col<f64>) -> Result<Col<f64>, std::convert::Infallible> {
            Ok(grad_inline(x))
        }
    }

    impl BoxConstraints for BoothBoxed<Col<f64>> {
        fn lower(&self) -> &Col<f64> {
            &self.lower
        }
        fn upper(&self) -> &Col<f64> {
            &self.upper
        }
    }

    fn residuals_inline(x: &Col<f64>) -> Col<f64> {
        debug_assert_eq!(x.nrows(), 2);
        let r0 = x[0] + 2.0 * x[1] - 7.0;
        let r1 = 2.0 * x[0] + x[1] - 5.0;
        Col::<f64>::from_fn(2, |i| if i == 0 { r0 } else { r1 })
    }

    impl CostFunction for BoothResiduals<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(cost_inline(x))
        }
    }

    impl Residual for BoothResiduals<Col<f64>> {
        type Param = Col<f64>;
        type Output = Col<f64>;
        type Error = std::convert::Infallible;
        fn residual(&self, x: &Col<f64>) -> Result<Col<f64>, std::convert::Infallible> {
            Ok(residuals_inline(x))
        }
    }

    impl Jacobian for BoothResiduals<Col<f64>> {
        type Jacobian = Mat<f64>;
        fn jacobian(&self, _x: &Col<f64>) -> Result<Mat<f64>, std::convert::Infallible> {
            let mut buf = [0.0_f64; 4];
            booth_residuals_jacobian(&mut buf);
            Ok(Mat::from_fn(2, 2, |i, j| buf[i * 2 + j]))
        }
    }

    impl CostFunction for BoothBoxedResiduals<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(cost_inline(x))
        }
    }

    impl Residual for BoothBoxedResiduals<Col<f64>> {
        type Param = Col<f64>;
        type Output = Col<f64>;
        type Error = std::convert::Infallible;
        fn residual(&self, x: &Col<f64>) -> Result<Col<f64>, std::convert::Infallible> {
            Ok(residuals_inline(x))
        }
    }

    impl Jacobian for BoothBoxedResiduals<Col<f64>> {
        type Jacobian = Mat<f64>;
        fn jacobian(&self, _x: &Col<f64>) -> Result<Mat<f64>, std::convert::Infallible> {
            let mut buf = [0.0_f64; 4];
            booth_residuals_jacobian(&mut buf);
            Ok(Mat::from_fn(2, 2, |i, j| buf[i * 2 + j]))
        }
    }

    impl BoxConstraints for BoothBoxedResiduals<Col<f64>> {
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
    fn booth_minimum_is_zero_at_known_optimum() {
        assert!(booth(&[1.0, 3.0]).abs() < 1e-12);
    }

    #[test]
    fn booth_known_value_at_origin() {
        // f(0, 0) = (-7)² + (-5)² = 49 + 25 = 74.
        assert!((booth(&[0.0, 0.0]) - 74.0).abs() < 1e-12);
    }

    #[test]
    fn gradient_zero_at_minimum() {
        let mut g = vec![0.0; 2];
        booth_gradient(&[1.0, 3.0], &mut g);
        for v in g {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let x = [-1.2, 0.7];
        let mut g = vec![0.0; x.len()];
        booth_gradient(&x, &mut g);

        let h = 1e-6;
        for i in 0..x.len() {
            let mut xp = x;
            let mut xm = x;
            xp[i] += h;
            xm[i] -= h;
            let fd = (booth(&xp) - booth(&xm)) / (2.0 * h);
            assert!((g[i] - fd).abs() < 1e-5, "i={i}, g={}, fd={fd}", g[i]);
        }
    }

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        let spec = <Booth<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(spec.name, "Booth");
        assert!(spec.properties.smooth);
        assert!(spec.properties.differentiable);
        assert!(spec.properties.convex);
        assert!(spec.properties.unimodal);
        assert!(matches!(spec.dim, Dimensionality::Fixed(2)));
        assert!(!spec.references.is_empty());
    }
}
