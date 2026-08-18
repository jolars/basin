//! N-dimensional Rosenbrock function.
//!
//! `f(x) = Σ_{i=0}^{n-2} [100·(x_{i+1} − x_i²)² + (1 − x_i)²]`
//!
//! Global minimum at `x = (1, …, 1)` with `f = 0`. The classical 2D start is
//! `(−1.2, 1.0)`. The 1D case (`n = 1`) is degenerate and returns 0.

use core::marker::PhantomData;

use super::spec::{
    Dimensionality, HasSpec, ProblemSpec, Properties, Reference,
};
use crate::{CostFunction, DenseMatrix, Gradient, Jacobian, Residual};

/// Catalog entry for the Rosenbrock function.
pub static ROSENBROCK_SPEC: ProblemSpec = ProblemSpec {
    name: "Rosenbrock",
    dim: Dimensionality::NDimensional { min: 2 },
    properties: Properties {
        smooth: true,
        differentiable: true,
        convex: false,
        // Unimodal in 2D and 3D, but for n ≥ 4 a second local minimum
        // appears near (-1, 1, …, 1). False is the safe N-agnostic claim.
        unimodal: false,
        separable: false,
        scalable: true,
    },
    references: &[Reference {
        citation: "Rosenbrock (1960)",
        title: "An automatic method for finding the greatest or least value of a function",
        source: "The Computer Journal, 3(3), 175–184",
        doi: Some("10.1093/comjnl/3.3.175"),
        url: None,
    }],
    description: "Banana-shaped curved valley with global minimum at \
                  x = (1, …, 1), value 0. Standard hard test for first- and \
                  second-order methods due to the narrow, ill-conditioned valley.",
};

impl<P> HasSpec for Rosenbrock<P> {
    const SPEC: &'static ProblemSpec = &ROSENBROCK_SPEC;
}

/// Evaluates Rosenbrock's function at `x`.
pub fn rosenbrock(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len().saturating_sub(1) {
        let a = x[i + 1] - x[i] * x[i];
        let b = 1.0 - x[i];
        s += 100.0 * a * a + b * b;
    }
    s
}

/// Writes the Rosenbrock gradient at `x` into `out`. Lengths must match.
pub fn rosenbrock_gradient(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), out.len());
    for v in out.iter_mut() {
        *v = 0.0;
    }
    for i in 0..x.len().saturating_sub(1) {
        let a = x[i + 1] - x[i] * x[i];
        out[i] += -400.0 * x[i] * a - 2.0 * (1.0 - x[i]);
        out[i + 1] += 200.0 * a;
    }
}

/// Pre-wrapped Rosenbrock problem. Generic over the parameter backend `P`;
/// the default `P = Vec<f64>` lets you write `Rosenbrock::default()` for the
/// common case. Backend impls (`nalgebra::DVector<f64>`, `ndarray::Array1<f64>`,
/// `faer::Col<f64>`) are gated behind their respective features.
pub struct Rosenbrock<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> Rosenbrock<P> {
    /// Build a freshly typed Rosenbrock instance. Pair with one of the
    /// backend-specific impl blocks (Vec, nalgebra, ndarray, faer).
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for Rosenbrock<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl CostFunction for Rosenbrock<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(rosenbrock(x))
    }
}

impl Gradient for Rosenbrock<Vec<f64>> {
    type Gradient = Vec<f64>;
    fn gradient(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; x.len()];
        rosenbrock_gradient(x, &mut out);
        Ok(out)
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_impl {
    use super::{Rosenbrock, rosenbrock, rosenbrock_gradient};
    use crate::{CostFunction, Gradient};
    use nalgebra::DVector;

    impl CostFunction for Rosenbrock<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &DVector<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(rosenbrock(x.as_slice()))
        }
    }

    impl Gradient for Rosenbrock<DVector<f64>> {
        type Gradient = DVector<f64>;
        fn gradient(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, std::convert::Infallible> {
            let mut out = DVector::zeros(x.len());
            rosenbrock_gradient(x.as_slice(), out.as_mut_slice());
            Ok(out)
        }
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::{Rosenbrock, rosenbrock, rosenbrock_gradient};
    use crate::{CostFunction, Gradient};
    use ndarray::Array1;

    // Array1 owns a contiguous buffer, so the `as_slice` calls always succeed.
    impl CostFunction for Rosenbrock<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &Array1<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(rosenbrock(x.as_slice().expect("Array1 is contiguous")))
        }
    }

    impl Gradient for Rosenbrock<Array1<f64>> {
        type Gradient = Array1<f64>;
        fn gradient(
            &self,
            x: &Array1<f64>,
        ) -> Result<Array1<f64>, std::convert::Infallible> {
            let mut out = Array1::zeros(x.len());
            rosenbrock_gradient(
                x.as_slice().expect("Array1 is contiguous"),
                out.as_slice_mut().expect("Array1 is contiguous"),
            );
            Ok(out)
        }
    }
}

#[cfg(feature = "faer")]
mod faer_impl {
    use super::Rosenbrock;
    use crate::{CostFunction, Gradient};
    use faer::Col;

    // faer's `Col` doesn't expose a `&[f64]` directly across all 0.24 APIs we
    // care about, so we evaluate elementwise here rather than routing through
    // the slice-based primitives.
    impl CostFunction for Rosenbrock<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            let n = x.nrows();
            let mut s = 0.0;
            for i in 0..n.saturating_sub(1) {
                let a = x[i + 1] - x[i] * x[i];
                let b = 1.0 - x[i];
                s += 100.0 * a * a + b * b;
            }
            Ok(s)
        }
    }

    impl Gradient for Rosenbrock<Col<f64>> {
        type Gradient = Col<f64>;
        fn gradient(
            &self,
            x: &Col<f64>,
        ) -> Result<Col<f64>, std::convert::Infallible> {
            let n = x.nrows();
            let mut out = Col::<f64>::zeros(n);
            for i in 0..n.saturating_sub(1) {
                let a = x[i + 1] - x[i] * x[i];
                out[i] += -400.0 * x[i] * a - 2.0 * (1.0 - x[i]);
                out[i + 1] += 200.0 * a;
            }
            Ok(out)
        }
    }
}

// Residual-form Rosenbrock (n = 2 only)
// The 2D Rosenbrock factors as a 2-residual least-squares problem:
//   r₀ = 10·(x₁ − x₀²)
//   r₁ = 1 − x₀
// with `Σ rᵢ² = rosenbrock(x)` exactly (note: unscaled sum, matching the
// published Rosenbrock cost rather than the LM ½‖r‖² convention; see
// the `Residual` trait contract). Used as a fixture for the LM track:
// same minimum (1, 1), same shape, but exposed in the form Gauss-Newton
// and Levenberg-Marquardt expect. n > 2 is not supported here; the
// existing `Rosenbrock` wrapper covers the cost form for general n.

/// Writes the 2D Rosenbrock residual at `x` into `out`. Both must have
/// length 2.
///
/// `r = [10·(x₁ − x₀²), 1 − x₀]`. Zero at `(1, 1)`.
pub fn rosenbrock_residuals(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), 2);
    debug_assert_eq!(out.len(), 2);
    out[0] = 10.0 * (x[1] - x[0] * x[0]);
    out[1] = 1.0 - x[0];
}

/// Writes the 2×2 Jacobian `∂rᵢ/∂xⱼ` at `x` into `out` in row-major
/// order: `out[i*2 + j] = ∂rᵢ/∂xⱼ`. `x.len()` must be 2 and
/// `out.len()` must be 4.
///
/// ```text
///        ∂x₀      ∂x₁
/// r₀:    −20·x₀   10
/// r₁:    −1       0
/// ```
pub fn rosenbrock_residuals_jacobian(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), 2);
    debug_assert_eq!(out.len(), 4);
    out[0] = -20.0 * x[0];
    out[1] = 10.0;
    out[2] = -1.0;
    out[3] = 0.0;
}

/// 2D Rosenbrock exposed as a least-squares problem (2 residuals, 2
/// parameters). Shares [`ROSENBROCK_SPEC`] with the cost-form
/// [`Rosenbrock`] wrapper: this is just a different *interface* over
/// the same function. Restricted to `param.len() == 2`; passing any
/// other length will trip a debug assertion in the raw functions.
///
/// `Jacobian` is implemented for every dense backend: the `Vec<f64>` default
/// (over the hand-rolled [`DenseMatrix<f64>`](crate::DenseMatrix)), nalgebra
/// (`DMatrix<f64>`), faer (`Mat<f64>`), and ndarray (`Array2<f64>`); see the
/// trait's `# Backends` note for the param→Jacobian pairings.
pub struct RosenbrockResiduals<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> RosenbrockResiduals<P> {
    /// Build a freshly typed Rosenbrock-as-residuals instance. Pair with
    /// the dense backend impls (Vec, nalgebra, faer, ndarray) that supply the
    /// `Jacobian` matrix type.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for RosenbrockResiduals<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P> HasSpec for RosenbrockResiduals<P> {
    const SPEC: &'static ProblemSpec = &ROSENBROCK_SPEC;
}

impl CostFunction for RosenbrockResiduals<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(rosenbrock(x))
    }
}

impl Residual for RosenbrockResiduals<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = std::convert::Infallible;
    fn residual(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; 2];
        rosenbrock_residuals(x, &mut out);
        Ok(out)
    }
}

impl Jacobian for RosenbrockResiduals<Vec<f64>> {
    type Jacobian = DenseMatrix<f64>;
    fn jacobian(
        &self,
        x: &Vec<f64>,
    ) -> Result<DenseMatrix<f64>, std::convert::Infallible> {
        let mut buf = [0.0_f64; 4];
        rosenbrock_residuals_jacobian(x, &mut buf);
        Ok(DenseMatrix::from_row_slice(2, 2, &buf))
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_residuals_impl {
    use super::{
        RosenbrockResiduals, rosenbrock, rosenbrock_residuals,
        rosenbrock_residuals_jacobian,
    };
    use crate::{CostFunction, Jacobian, Residual};
    use nalgebra::{DMatrix, DVector};

    impl CostFunction for RosenbrockResiduals<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &DVector<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(rosenbrock(x.as_slice()))
        }
    }

    impl Residual for RosenbrockResiduals<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = DVector<f64>;
        type Error = std::convert::Infallible;
        fn residual(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, std::convert::Infallible> {
            let mut out = DVector::zeros(2);
            rosenbrock_residuals(x.as_slice(), out.as_mut_slice());
            Ok(out)
        }
    }

    impl Jacobian for RosenbrockResiduals<DVector<f64>> {
        type Jacobian = DMatrix<f64>;
        fn jacobian(
            &self,
            x: &DVector<f64>,
        ) -> Result<DMatrix<f64>, std::convert::Infallible> {
            let mut buf = [0.0_f64; 4];
            rosenbrock_residuals_jacobian(x.as_slice(), &mut buf);
            Ok(DMatrix::from_row_slice(2, 2, &buf))
        }
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_residuals_impl {
    use super::{
        RosenbrockResiduals, rosenbrock, rosenbrock_residuals,
        rosenbrock_residuals_jacobian,
    };
    use crate::{CostFunction, Jacobian, Residual};
    use ndarray::{Array1, Array2};

    impl CostFunction for RosenbrockResiduals<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &Array1<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(rosenbrock(x.as_slice().expect("Array1 is contiguous")))
        }
    }

    impl Residual for RosenbrockResiduals<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = Array1<f64>;
        type Error = std::convert::Infallible;
        fn residual(
            &self,
            x: &Array1<f64>,
        ) -> Result<Array1<f64>, std::convert::Infallible> {
            let mut out = Array1::zeros(2);
            rosenbrock_residuals(
                x.as_slice().expect("Array1 is contiguous"),
                out.as_slice_mut().expect("Array1 is contiguous"),
            );
            Ok(out)
        }
    }

    impl Jacobian for RosenbrockResiduals<Array1<f64>> {
        type Jacobian = Array2<f64>;
        fn jacobian(
            &self,
            x: &Array1<f64>,
        ) -> Result<Array2<f64>, std::convert::Infallible> {
            let mut buf = [0.0_f64; 4];
            rosenbrock_residuals_jacobian(
                x.as_slice().expect("Array1 is contiguous"),
                &mut buf,
            );
            // `buf` is row-major (2×2); the default C-order `from_shape_vec`
            // matches the nalgebra `from_row_slice` mirror above.
            Ok(Array2::from_shape_vec((2, 2), buf.to_vec())
                .expect("4 entries for a 2×2"))
        }
    }
}

#[cfg(feature = "faer")]
mod faer_residuals_impl {
    use super::{RosenbrockResiduals, rosenbrock_residuals_jacobian};
    use crate::{CostFunction, Jacobian, Residual};
    use faer::{Col, Mat};

    // faer's `Col` doesn't expose `&[f64]` across all 0.24 APIs we care
    // about; evaluate elementwise to mirror the cost-form impl above.
    impl CostFunction for RosenbrockResiduals<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            let a = x[1] - x[0] * x[0];
            let b = 1.0 - x[0];
            Ok(100.0 * a * a + b * b)
        }
    }

    impl Residual for RosenbrockResiduals<Col<f64>> {
        type Param = Col<f64>;
        type Output = Col<f64>;
        type Error = std::convert::Infallible;
        fn residual(
            &self,
            x: &Col<f64>,
        ) -> Result<Col<f64>, std::convert::Infallible> {
            let mut out = Col::<f64>::zeros(2);
            out[0] = 10.0 * (x[1] - x[0] * x[0]);
            out[1] = 1.0 - x[0];
            Ok(out)
        }
    }

    impl Jacobian for RosenbrockResiduals<Col<f64>> {
        type Jacobian = Mat<f64>;
        fn jacobian(
            &self,
            x: &Col<f64>,
        ) -> Result<Mat<f64>, std::convert::Infallible> {
            let xs = [x[0], x[1]];
            let mut buf = [0.0_f64; 4];
            rosenbrock_residuals_jacobian(&xs, &mut buf);
            Ok(Mat::from_fn(2, 2, |i, j| buf[i * 2 + j]))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rosenbrock_minimum_is_zero_at_ones() {
        assert_eq!(rosenbrock(&[1.0, 1.0]), 0.0);
        assert_eq!(rosenbrock(&[1.0, 1.0, 1.0, 1.0]), 0.0);
    }

    #[test]
    fn rosenbrock_known_value_at_classical_start() {
        // f(-1.2, 1.0) = (1 - (-1.2))^2 + 100*(1 - 1.44)^2 = 4.84 + 19.36 = 24.2
        assert!((rosenbrock(&[-1.2, 1.0]) - 24.2).abs() < 1e-12);
    }

    #[test]
    fn gradient_zero_at_minimum() {
        let mut g = vec![0.0; 4];
        rosenbrock_gradient(&[1.0, 1.0, 1.0, 1.0], &mut g);
        for v in g {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        let spec = <Rosenbrock<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(spec.name, "Rosenbrock");
        assert!(spec.properties.smooth);
        assert!(spec.properties.scalable);
        assert!(matches!(spec.dim, Dimensionality::NDimensional { min: 2 }));
        assert!(!spec.references.is_empty());
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let x = [-1.2, 1.0, 0.7, 0.4];
        let mut g = vec![0.0; x.len()];
        rosenbrock_gradient(&x, &mut g);

        let h = 1e-6;
        for i in 0..x.len() {
            let mut xp = x;
            let mut xm = x;
            xp[i] += h;
            xm[i] -= h;
            let fd = (rosenbrock(&xp) - rosenbrock(&xm)) / (2.0 * h);
            assert!((g[i] - fd).abs() < 1e-5, "i={i}, g={}, fd={fd}", g[i]);
        }
    }

    // -- residual-form tests --

    #[test]
    fn residuals_are_zero_at_optimum() {
        let mut r = vec![0.0; 2];
        rosenbrock_residuals(&[1.0, 1.0], &mut r);
        assert!(r[0].abs() < 1e-12);
        assert!(r[1].abs() < 1e-12);
    }

    #[test]
    fn residuals_match_cost_at_n2() {
        // For n = 2: rosenbrock(x) = 100·(x₁−x₀²)² + (1−x₀)² = Σ rᵢ².
        for x in [[-1.2, 1.0], [0.5, 0.25], [2.0, 4.0], [0.0, 0.0]] {
            let mut r = vec![0.0; 2];
            rosenbrock_residuals(&x, &mut r);
            let sum_sq = r[0] * r[0] + r[1] * r[1];
            let c = rosenbrock(&x);
            assert!(
                (c - sum_sq).abs() < 1e-12,
                "x={x:?}, c={c}, sum_sq={sum_sq}"
            );
        }
    }

    #[test]
    fn residual_jacobian_matches_finite_difference() {
        let x = [-1.2, 1.0];
        let mut j = vec![0.0; 4];
        rosenbrock_residuals_jacobian(&x, &mut j);

        let h = 1e-6;
        for i in 0..2 {
            for k in 0..2 {
                let mut xp = x;
                let mut xm = x;
                xp[k] += h;
                xm[k] -= h;
                let mut rp = vec![0.0; 2];
                let mut rm = vec![0.0; 2];
                rosenbrock_residuals(&xp, &mut rp);
                rosenbrock_residuals(&xm, &mut rm);
                let fd = (rp[i] - rm[i]) / (2.0 * h);
                assert!(
                    (j[i * 2 + k] - fd).abs() < 1e-5,
                    "i={i}, k={k}, j={}, fd={fd}",
                    j[i * 2 + k]
                );
            }
        }
    }

    #[test]
    fn residual_wrapper_reuses_rosenbrock_spec() {
        let spec = <RosenbrockResiduals<Vec<f64>> as HasSpec>::SPEC;
        // Same static: both wrappers point at the one Rosenbrock entry.
        assert!(core::ptr::eq(spec, &ROSENBROCK_SPEC));
    }

    #[test]
    fn residual_trait_returns_expected_vector() {
        let p: RosenbrockResiduals = RosenbrockResiduals::default();
        let r = p.residual(&vec![1.0, 1.0]).unwrap();
        assert_eq!(r.len(), 2);
        for v in r {
            assert!(v.abs() < 1e-12);
        }
    }

    #[cfg(feature = "nalgebra")]
    mod nalgebra_jacobian_tests {
        use super::super::RosenbrockResiduals;
        use crate::{
            GramMatrix, Jacobian, LinearSolveSpd, MatTransposeVec, Residual,
        };
        use nalgebra::{DMatrix, DVector};

        #[test]
        fn jacobian_at_minimum_matches_documented_layout() {
            let p: RosenbrockResiduals<DVector<f64>> =
                RosenbrockResiduals::new();
            let x = DVector::from_vec(vec![1.0, 1.0]);
            let j: DMatrix<f64> = p.jacobian(&x).unwrap();
            assert_eq!(j.shape(), (2, 2));
            // Layout at x = (1, 1): [[−20·1, 10], [−1, 0]]
            assert!((j[(0, 0)] + 20.0).abs() < 1e-12);
            assert!((j[(0, 1)] - 10.0).abs() < 1e-12);
            assert!((j[(1, 0)] + 1.0).abs() < 1e-12);
            assert!(j[(1, 1)].abs() < 1e-12);
        }

        #[test]
        fn gauss_newton_step_at_classical_start_is_well_defined() {
            // At the classical start (-1.2, 1.0), J has full column rank,
            // so JᵀJ is SPD and the GN step δ = (JᵀJ)⁻¹ Jᵀ r is well
            // defined. Smallest credible end-to-end exercise of the
            // linalg layer through a real Jacobian.
            let p: RosenbrockResiduals<DVector<f64>> =
                RosenbrockResiduals::new();
            let x = DVector::from_vec(vec![-1.2, 1.0]);
            let j = p.jacobian(&x).unwrap();
            let r = p.residual(&x).unwrap();
            let g = j.gram();
            let rhs = j.mat_transpose_vec(&r);
            let delta = g.solve_spd(&rhs).expect("JᵀJ at (-1.2, 1.0) is SPD");
            assert_eq!(delta.len(), 2);
            // Hand-computed: J = [[24, 10], [-1, 0]], r = [-4.4, 2.2].
            // JᵀJ = [[577, 240], [240, 100]]  (det 100).
            // Jᵀr = [-107.8, -44].
            // δ = (JᵀJ)⁻¹·(Jᵀr) = (1/100)·[[100,-240],[-240,577]]·[-107.8,-44]
            //                   = (1/100)·[-220, 484] = [-2.2, 4.84].
            // The Gauss-Newton update is x ← x − δ, so this δ is the
            // un-negated normal-equation solution.
            assert!((delta[0] + 2.2).abs() < 1e-9, "delta[0] = {}", delta[0]);
            assert!((delta[1] - 4.84).abs() < 1e-9, "delta[1] = {}", delta[1]);
        }
    }

    #[cfg(feature = "faer")]
    mod faer_jacobian_tests {
        use super::super::RosenbrockResiduals;
        use crate::{Jacobian, Residual};
        use faer::{Col, Mat};

        #[test]
        fn jacobian_at_minimum_matches_documented_layout() {
            let p: RosenbrockResiduals<Col<f64>> = RosenbrockResiduals::new();
            let x = Col::<f64>::from_fn(2, |i| [1.0, 1.0][i]);
            let j: Mat<f64> = p.jacobian(&x).unwrap();
            assert_eq!((j.nrows(), j.ncols()), (2, 2));
            assert!((j[(0, 0)] + 20.0).abs() < 1e-12);
            assert!((j[(0, 1)] - 10.0).abs() < 1e-12);
            assert!((j[(1, 0)] + 1.0).abs() < 1e-12);
            assert!(j[(1, 1)].abs() < 1e-12);
        }

        #[test]
        fn jacobian_agrees_with_residual_via_finite_difference() {
            // End-to-end sanity: the per-backend Jacobian must match a
            // central-difference estimate of the residual it pairs with.
            let p: RosenbrockResiduals<Col<f64>> = RosenbrockResiduals::new();
            let x = Col::<f64>::from_fn(2, |i| [-1.2, 1.0][i]);
            let j = p.jacobian(&x).unwrap();
            let h = 1e-6;
            for k in 0..2 {
                let mut xp = x.clone();
                let mut xm = x.clone();
                xp[k] += h;
                xm[k] -= h;
                let rp = p.residual(&xp).unwrap();
                let rm = p.residual(&xm).unwrap();
                for i in 0..2 {
                    let fd = (rp[i] - rm[i]) / (2.0 * h);
                    assert!(
                        (j[(i, k)] - fd).abs() < 1e-5,
                        "i={i} k={k} j={} fd={fd}",
                        j[(i, k)]
                    );
                }
            }
        }
    }
}
