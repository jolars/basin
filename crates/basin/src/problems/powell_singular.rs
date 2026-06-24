//! Powell's singular function — a 4-variable, 4-residual least-squares
//! problem from Powell (1962). Classic Levenberg–Marquardt benchmark
//! because the Jacobian is rank-deficient at the optimum (hence
//! "singular"), so naive Gauss–Newton stalls and the damping in LM
//! actually has to do work.
//!
//! Residuals at `x = (x₀, x₁, x₂, x₃)`:
//! ```text
//! r₀ = x₀ + 10·x₁
//! r₁ = √5 · (x₂ − x₃)
//! r₂ = (x₁ − 2·x₂)²
//! r₃ = √10 · (x₀ − x₃)²
//! ```
//! Cost (LM convention) `f(x) = ½ Σ rᵢ(x)²`. Global minimum at
//! `x = (0, 0, 0, 0)` with `f = 0`. Standard initial point for LM
//! benchmarks is `x₀ = (3, −1, 0, 1)`.
//!
//! S2a wires `Jacobian` impls for the LA-heavy backends (nalgebra and
//! faer dense). `Vec<f64>` and `ndarray` deliberately don't implement
//! `Jacobian` — see the trait's `# Backends` note. The raw
//! `powell_singular_jacobian` function below stays backend-agnostic and
//! is the single source of truth that the per-backend impls reshape
//! into their matrix types.

use core::marker::PhantomData;

use super::spec::{Dimensionality, HasSpec, ProblemSpec, Properties, Reference};
use crate::{CostFunction, DenseMatrix, Jacobian, Residual};

/// Catalogue entry for Powell's singular function.
pub static POWELL_SINGULAR_SPEC: ProblemSpec = ProblemSpec {
    name: "Powell singular",
    dim: Dimensionality::Fixed(4),
    properties: Properties {
        smooth: true,
        differentiable: true,
        // Sum of squares with a quartic term in two of the residuals;
        // not convex globally.
        convex: false,
        // Single global minimum at the origin. Jacobian rank-deficiency
        // there is the interesting bit, not multimodality.
        unimodal: true,
        separable: false,
        scalable: false,
    },
    references: &[Reference {
        citation: "Powell (1962)",
        title: "An iterative method for finding stationary values of a function of several variables",
        source: "The Computer Journal, 5(2), 147–151",
        doi: Some("10.1093/comjnl/5.2.147"),
        url: None,
    }],
    description: "4-variable / 4-residual least-squares problem with a \
                  rank-deficient Jacobian at the optimum x = (0, 0, 0, 0). \
                  Standard LM benchmark — naive Gauss–Newton stalls; LM \
                  damping recovers convergence.",
};

impl<P> HasSpec for PowellSingular<P> {
    const SPEC: &'static ProblemSpec = &POWELL_SINGULAR_SPEC;
}

const SQRT_5: f64 = 2.236_067_977_499_79; // √5
const SQRT_10: f64 = 3.162_277_660_168_379_5; // √10

/// Evaluates Powell's singular function as a scalar cost,
/// `f(x) = ½ Σ rᵢ(x)²` (LM convention). Requires `x.len() == 4`.
pub fn powell_singular(x: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), 4);
    let r0 = x[0] + 10.0 * x[1];
    let r1 = SQRT_5 * (x[2] - x[3]);
    let d12 = x[1] - 2.0 * x[2];
    let r2 = d12 * d12;
    let d03 = x[0] - x[3];
    let r3 = SQRT_10 * d03 * d03;
    0.5 * (r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3)
}

/// Writes Powell's residual vector at `x` into `out`. Both must have
/// length 4.
pub fn powell_singular_residuals(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), 4);
    debug_assert_eq!(out.len(), 4);
    out[0] = x[0] + 10.0 * x[1];
    out[1] = SQRT_5 * (x[2] - x[3]);
    let d12 = x[1] - 2.0 * x[2];
    out[2] = d12 * d12;
    let d03 = x[0] - x[3];
    out[3] = SQRT_10 * d03 * d03;
}

/// Writes the 4×4 Jacobian `∂rᵢ/∂xⱼ` at `x` into `out` in row-major
/// order: `out[i*4 + j] = ∂rᵢ/∂xⱼ`. `x.len()` must be 4 and
/// `out.len()` must be 16.
///
/// The matrix layout:
/// ```text
///        ∂x₀          ∂x₁          ∂x₂           ∂x₃
/// r₀:    1            10           0             0
/// r₁:    0            0            √5            −√5
/// r₂:    0            2(x₁−2x₂)    −4(x₁−2x₂)    0
/// r₃:    2√10(x₀−x₃)  0            0             −2√10(x₀−x₃)
/// ```
/// The Jacobian becomes rank-deficient at the optimum (rows 2 and 3
/// vanish there), which is what makes this problem hard.
pub fn powell_singular_jacobian(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), 4);
    debug_assert_eq!(out.len(), 16);
    for v in out.iter_mut() {
        *v = 0.0;
    }
    // Row 0: r₀ = x₀ + 10·x₁
    out[0] = 1.0;
    out[1] = 10.0;
    // Row 1: r₁ = √5·(x₂ − x₃)
    out[4 + 2] = SQRT_5;
    out[4 + 3] = -SQRT_5;
    // Row 2: r₂ = (x₁ − 2·x₂)²
    let d12 = x[1] - 2.0 * x[2];
    out[8 + 1] = 2.0 * d12;
    out[8 + 2] = -4.0 * d12;
    // Row 3: r₃ = √10·(x₀ − x₃)²
    let d03 = x[0] - x[3];
    out[12] = 2.0 * SQRT_10 * d03;
    out[12 + 3] = -2.0 * SQRT_10 * d03;
}

/// Pre-wrapped Powell-singular problem. Generic over the parameter
/// backend `P`; the default `P = Vec<f64>` lets you write
/// `PowellSingular::default()` for the common case. Backend impls
/// (`nalgebra::DVector<f64>`, `ndarray::Array1<f64>`, `faer::Col<f64>`)
/// are gated behind their respective features.
///
/// `Jacobian` is implemented for every dense backend: the `Vec<f64>` default
/// (over the hand-rolled [`DenseMatrix<f64>`](crate::DenseMatrix)), nalgebra
/// (`DMatrix<f64>`), faer (`Mat<f64>`), and ndarray (`Array2<f64>`); see the
/// trait's `# Backends` note for the param→Jacobian pairings.
pub struct PowellSingular<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> PowellSingular<P> {
    /// Build a freshly typed Powell-singular instance. Pair with one of
    /// the backend-specific impl blocks; every dense backend (Vec, nalgebra,
    /// faer, ndarray) supplies the `Jacobian` matrix type.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for PowellSingular<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl CostFunction for PowellSingular<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(powell_singular(x))
    }
}

impl Residual for PowellSingular<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = std::convert::Infallible;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; 4];
        powell_singular_residuals(x, &mut out);
        Ok(out)
    }
}

impl Jacobian for PowellSingular<Vec<f64>> {
    type Jacobian = DenseMatrix<f64>;
    fn jacobian(&self, x: &Vec<f64>) -> Result<DenseMatrix<f64>, std::convert::Infallible> {
        let mut buf = [0.0_f64; 16];
        powell_singular_jacobian(x, &mut buf);
        // `from_row_slice` interprets `buf` in row-major order, matching
        // the layout `powell_singular_jacobian` produces.
        Ok(DenseMatrix::from_row_slice(4, 4, &buf))
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_impl {
    use super::{
        PowellSingular, powell_singular, powell_singular_jacobian, powell_singular_residuals,
    };
    use crate::{CostFunction, Jacobian, Residual};
    use nalgebra::{DMatrix, DVector};

    impl CostFunction for PowellSingular<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(powell_singular(x.as_slice()))
        }
    }

    impl Residual for PowellSingular<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = DVector<f64>;
        type Error = std::convert::Infallible;
        fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, std::convert::Infallible> {
            let mut out = DVector::zeros(4);
            powell_singular_residuals(x.as_slice(), out.as_mut_slice());
            Ok(out)
        }
    }

    impl Jacobian for PowellSingular<DVector<f64>> {
        type Jacobian = DMatrix<f64>;
        fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, std::convert::Infallible> {
            let mut buf = [0.0_f64; 16];
            powell_singular_jacobian(x.as_slice(), &mut buf);
            // `from_row_slice` interprets `buf` in row-major order, matching
            // the layout `powell_singular_jacobian` produces.
            Ok(DMatrix::from_row_slice(4, 4, &buf))
        }
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::{
        PowellSingular, powell_singular, powell_singular_jacobian, powell_singular_residuals,
    };
    use crate::{CostFunction, Jacobian, Residual};
    use ndarray::{Array1, Array2};

    // Array1 owns a contiguous buffer, so `as_slice` always succeeds.
    impl CostFunction for PowellSingular<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Array1<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(powell_singular(x.as_slice().expect("Array1 is contiguous")))
        }
    }

    impl Residual for PowellSingular<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = Array1<f64>;
        type Error = std::convert::Infallible;
        fn residual(&self, x: &Array1<f64>) -> Result<Array1<f64>, std::convert::Infallible> {
            let mut out = Array1::zeros(4);
            powell_singular_residuals(
                x.as_slice().expect("Array1 is contiguous"),
                out.as_slice_mut().expect("Array1 is contiguous"),
            );
            Ok(out)
        }
    }

    impl Jacobian for PowellSingular<Array1<f64>> {
        type Jacobian = Array2<f64>;
        fn jacobian(&self, x: &Array1<f64>) -> Result<Array2<f64>, std::convert::Infallible> {
            let mut buf = [0.0_f64; 16];
            powell_singular_jacobian(x.as_slice().expect("Array1 is contiguous"), &mut buf);
            // `buf` is row-major (4×4); the default C-order `from_shape_vec`
            // matches the nalgebra `from_row_slice` mirror above.
            Ok(Array2::from_shape_vec((4, 4), buf.to_vec()).expect("16 entries for a 4×4"))
        }
    }
}

#[cfg(feature = "faer")]
mod faer_impl {
    use super::{PowellSingular, SQRT_5, SQRT_10, powell_singular_jacobian};
    use crate::{CostFunction, Jacobian, Residual};
    use faer::{Col, Mat};

    // faer's `Col` doesn't expose a `&[f64]` directly across all 0.24 APIs we
    // care about, so we evaluate elementwise here rather than routing through
    // the slice-based primitives.
    impl CostFunction for PowellSingular<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            let r0 = x[0] + 10.0 * x[1];
            let r1 = SQRT_5 * (x[2] - x[3]);
            let d12 = x[1] - 2.0 * x[2];
            let r2 = d12 * d12;
            let d03 = x[0] - x[3];
            let r3 = SQRT_10 * d03 * d03;
            Ok(0.5 * (r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3))
        }
    }

    impl Residual for PowellSingular<Col<f64>> {
        type Param = Col<f64>;
        type Output = Col<f64>;
        type Error = std::convert::Infallible;
        fn residual(&self, x: &Col<f64>) -> Result<Col<f64>, std::convert::Infallible> {
            let mut out = Col::<f64>::zeros(4);
            out[0] = x[0] + 10.0 * x[1];
            out[1] = SQRT_5 * (x[2] - x[3]);
            let d12 = x[1] - 2.0 * x[2];
            out[2] = d12 * d12;
            let d03 = x[0] - x[3];
            out[3] = SQRT_10 * d03 * d03;
            Ok(out)
        }
    }

    impl Jacobian for PowellSingular<Col<f64>> {
        type Jacobian = Mat<f64>;
        fn jacobian(&self, x: &Col<f64>) -> Result<Mat<f64>, std::convert::Infallible> {
            // Route through the row-major raw fn for a single source of
            // truth. The Col → slice copy is 4 entries.
            let xs = [x[0], x[1], x[2], x[3]];
            let mut buf = [0.0_f64; 16];
            powell_singular_jacobian(&xs, &mut buf);
            Ok(Mat::from_fn(4, 4, |i, j| buf[i * 4 + j]))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residuals_are_zero_at_origin() {
        let mut r = vec![0.0; 4];
        powell_singular_residuals(&[0.0, 0.0, 0.0, 0.0], &mut r);
        for v in r {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn cost_is_zero_at_origin() {
        assert_eq!(powell_singular(&[0.0, 0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn residuals_at_classical_start() {
        // x = (3, -1, 0, 1):
        //   r₀ = 3 + 10·(-1) = -7
        //   r₁ = √5·(0 - 1) = -√5
        //   r₂ = (-1 - 0)² = 1
        //   r₃ = √10·(3 - 1)² = 4·√10
        let mut r = vec![0.0; 4];
        powell_singular_residuals(&[3.0, -1.0, 0.0, 1.0], &mut r);
        assert!((r[0] - (-7.0)).abs() < 1e-12);
        assert!((r[1] - (-SQRT_5)).abs() < 1e-12);
        assert!((r[2] - 1.0).abs() < 1e-12);
        assert!((r[3] - 4.0 * SQRT_10).abs() < 1e-12);
    }

    #[test]
    fn cost_matches_half_sum_of_squared_residuals() {
        // Pick a few non-optimum points.
        for x in [
            [3.0, -1.0, 0.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
            [-2.0, 1.0, 3.0, -0.5],
        ] {
            let mut r = vec![0.0; 4];
            powell_singular_residuals(&x, &mut r);
            let half_sum_sq = 0.5 * r.iter().map(|ri| ri * ri).sum::<f64>();
            let c = powell_singular(&x);
            assert!(
                (c - half_sum_sq).abs() < 1e-12,
                "x={x:?}, c={c}, half_sum_sq={half_sum_sq}"
            );
        }
    }

    #[test]
    fn jacobian_matches_finite_difference() {
        let x = [3.0, -1.0, 0.0, 1.0];
        let mut j = vec![0.0; 16];
        powell_singular_jacobian(&x, &mut j);

        let h = 1e-6;
        for i in 0..4 {
            for k in 0..4 {
                let mut xp = x;
                let mut xm = x;
                xp[k] += h;
                xm[k] -= h;
                let mut rp = vec![0.0; 4];
                let mut rm = vec![0.0; 4];
                powell_singular_residuals(&xp, &mut rp);
                powell_singular_residuals(&xm, &mut rm);
                let fd = (rp[i] - rm[i]) / (2.0 * h);
                assert!(
                    (j[i * 4 + k] - fd).abs() < 1e-5,
                    "i={i}, k={k}, j={}, fd={fd}",
                    j[i * 4 + k]
                );
            }
        }
    }

    #[test]
    fn jacobian_is_rank_deficient_at_optimum() {
        // Rows 2 and 3 vanish at x = 0 because they're derivatives of squared
        // terms evaluated where the inner expression is zero. That's the
        // "singular" in Powell's singular function.
        let mut j = vec![0.0; 16];
        powell_singular_jacobian(&[0.0; 4], &mut j);
        for k in 0..4 {
            assert_eq!(j[8 + k], 0.0, "row 2 col {k} should be zero at origin");
            assert_eq!(j[12 + k], 0.0, "row 3 col {k} should be zero at origin");
        }
    }

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        let spec = <PowellSingular<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(spec.name, "Powell singular");
        assert!(spec.properties.smooth);
        assert!(matches!(spec.dim, Dimensionality::Fixed(4)));
        assert!(!spec.references.is_empty());
    }

    #[test]
    fn residual_trait_returns_expected_vector() {
        let p: PowellSingular = PowellSingular::default();
        let r = p.residual(&vec![0.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(r.len(), 4);
        for v in r {
            assert!(v.abs() < 1e-12);
        }
    }

    #[cfg(feature = "nalgebra")]
    mod nalgebra_jacobian_tests {
        use super::super::PowellSingular;
        use crate::{GramMatrix, Jacobian, LinearSolveError, LinearSolveSpd};
        use nalgebra::{DMatrix, DVector};

        #[test]
        fn jacobian_at_classical_start_matches_documented_layout() {
            let p: PowellSingular<DVector<f64>> = PowellSingular::new();
            let x = DVector::from_vec(vec![3.0, -1.0, 0.0, 1.0]);
            let j: DMatrix<f64> = p.jacobian(&x).unwrap();
            assert_eq!(j.shape(), (4, 4));
            // d12 = x₁ − 2·x₂ = -1; d03 = x₀ − x₃ = 2.
            // Row 0: [1, 10, 0, 0]
            assert!((j[(0, 0)] - 1.0).abs() < 1e-12);
            assert!((j[(0, 1)] - 10.0).abs() < 1e-12);
            // Row 1: [0, 0, √5, −√5]
            assert!((j[(1, 2)] - super::SQRT_5).abs() < 1e-12);
            assert!((j[(1, 3)] + super::SQRT_5).abs() < 1e-12);
            // Row 2: [0, 2·d12, −4·d12, 0] = [0, -2, 4, 0]
            assert!((j[(2, 1)] + 2.0).abs() < 1e-12);
            assert!((j[(2, 2)] - 4.0).abs() < 1e-12);
            // Row 3: [2√10·d03, 0, 0, −2√10·d03] = [4√10, 0, 0, −4√10]
            assert!((j[(3, 0)] - 4.0 * super::SQRT_10).abs() < 1e-12);
            assert!((j[(3, 3)] + 4.0 * super::SQRT_10).abs() < 1e-12);
        }

        #[test]
        fn gram_at_origin_is_singular() {
            // Powell's "singular": Jᵀ J at the optimum drops rank, so
            // Cholesky must fail. This is what makes plain Gauss-Newton
            // stall here — the LM track will use the same property to
            // exercise the damping.
            let p: PowellSingular<DVector<f64>> = PowellSingular::new();
            let x = DVector::zeros(4);
            let j = p.jacobian(&x).unwrap();
            let g = j.gram();
            let b = DVector::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
            let err = g
                .solve_spd(&b)
                .expect_err("Jᵀ J at origin must be singular");
            assert_eq!(err, LinearSolveError::NotPositiveDefinite);
        }
    }

    #[cfg(feature = "faer")]
    mod faer_jacobian_tests {
        use super::super::PowellSingular;
        use crate::{GramMatrix, Jacobian, LinearSolveError, LinearSolveSpd};
        use faer::{Col, Mat};

        #[test]
        fn jacobian_at_classical_start_matches_documented_layout() {
            let p: PowellSingular<Col<f64>> = PowellSingular::new();
            let x = Col::<f64>::from_fn(4, |i| [3.0, -1.0, 0.0, 1.0][i]);
            let j: Mat<f64> = p.jacobian(&x).unwrap();
            assert_eq!((j.nrows(), j.ncols()), (4, 4));
            assert!((j[(0, 0)] - 1.0).abs() < 1e-12);
            assert!((j[(0, 1)] - 10.0).abs() < 1e-12);
            assert!((j[(1, 2)] - super::SQRT_5).abs() < 1e-12);
            assert!((j[(1, 3)] + super::SQRT_5).abs() < 1e-12);
            assert!((j[(2, 1)] + 2.0).abs() < 1e-12);
            assert!((j[(2, 2)] - 4.0).abs() < 1e-12);
            assert!((j[(3, 0)] - 4.0 * super::SQRT_10).abs() < 1e-12);
            assert!((j[(3, 3)] + 4.0 * super::SQRT_10).abs() < 1e-12);
        }

        #[test]
        fn gram_at_origin_is_singular() {
            let p: PowellSingular<Col<f64>> = PowellSingular::new();
            let x = Col::<f64>::zeros(4);
            let j = p.jacobian(&x).unwrap();
            let g = j.gram();
            let b = Col::<f64>::from_fn(4, |_| 1.0);
            let err = g
                .solve_spd(&b)
                .expect_err("Jᵀ J at origin must be singular");
            assert_eq!(err, LinearSolveError::NotPositiveDefinite);
        }
    }
}
