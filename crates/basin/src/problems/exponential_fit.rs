//! Exponential data-fitting least-squares problem—the canonical
//! poorly-scaled nonlinear least-squares benchmark.
//!
//! Fit the one-term exponential model `ŷ(t) = a · exp(b · t)` to data
//! `(tᵢ, yᵢ)` by minimizing `f(a, b) = ½ Σ rᵢ²` with residuals
//!
//! ```text
//! rᵢ(a, b) = a · exp(b · tᵢ) − yᵢ
//! ```
//!
//! The Jacobian columns have very different magnitudes—`∂rᵢ/∂a =
//! exp(b·tᵢ)` is `O(1)` while `∂rᵢ/∂b = a·tᵢ·exp(b·tᵢ)` scales with the
//! amplitude `a` (often hundreds or thousands). That column-scale
//! disparity is exactly what isotropic Levenberg-Marquardt damping
//! (`μI`) handles badly and Marquardt diagonal damping (`μ·diag(JᵀJ)`)
//! is invariant to—see
//! [`LevenbergMarquardt`](crate::solver::LevenbergMarquardt). It is the
//! exponential-fit family used as the running example in Madsen,
//! Nielsen, Tingleff (2004).
//!
//! Like [`SparseLeastSquares`](super::SparseLeastSquares), the problem
//! carries its data (`t`, `y`) on the struct. With exact data
//! (`yᵢ = a* · exp(b* · tᵢ)`) the global minimum sits at `(a*, b*)`
//! with `f = 0`. The generic parameter `V` pins the parameter vector
//! backend; `Jacobian` is implemented for every dense backend (nalgebra
//! `DVector<f64>` → `DMatrix<f64>`, faer `Col<f64>` → `Mat<f64>`, and
//! ndarray `Array1<f64>` → `Array2<f64>`).

use core::marker::PhantomData;

use super::spec::{Dimensionality, HasSpec, ProblemSpec, Properties, Reference};

/// Catalog entry for the exponential-fit problem.
pub static EXPONENTIAL_FIT_SPEC: ProblemSpec = ProblemSpec {
    name: "Exponential fit",
    dim: Dimensionality::Fixed(2),
    properties: Properties {
        smooth: true,
        differentiable: true,
        // Sum-of-squares of an exponential model—nonconvex in (a, b).
        convex: false,
        // The cost surface has flat valleys and a poorly-scaled
        // Jacobian; not reliably unimodal in practice.
        unimodal: false,
        separable: false,
        // Two parameters fixed by the model; the data length is free
        // but the parameter dimension is not.
        scalable: false,
    },
    references: &[Reference {
        citation: "Madsen, Nielsen, Tingleff (2004)",
        title: "Methods for Non-Linear Least Squares Problems (2nd ed.)",
        source: "IMM, Technical University of Denmark",
        doi: None,
        url: Some("https://www2.compute.dtu.dk/pubdb/pubs/3215-full.html"),
    }],
    description: "Fit ŷ(t) = a·exp(b·t) to data by least squares. The \
                  Jacobian columns differ in scale by the amplitude a, \
                  making this the canonical poorly-scaled NLLS benchmark \
                  that distinguishes Marquardt diagonal damping from \
                  isotropic damping.",
};

/// Writes the exponential-fit residual vector `rᵢ = a·exp(b·tᵢ) − yᵢ`
/// at `params = [a, b]` into `out`. `t` and `y` must have equal length,
/// and `out.len()` must equal `t.len()`.
pub fn exponential_fit_residuals(params: &[f64], t: &[f64], y: &[f64], out: &mut [f64]) {
    debug_assert_eq!(params.len(), 2);
    debug_assert_eq!(t.len(), y.len());
    debug_assert_eq!(out.len(), t.len());
    let (a, b) = (params[0], params[1]);
    for i in 0..t.len() {
        out[i] = a * (b * t[i]).exp() - y[i];
    }
}

/// Evaluates the exponential-fit cost `f(a, b) = ½ Σ rᵢ²` (LM
/// convention) at `params = [a, b]`.
pub fn exponential_fit(params: &[f64], t: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(params.len(), 2);
    debug_assert_eq!(t.len(), y.len());
    let (a, b) = (params[0], params[1]);
    0.5 * t
        .iter()
        .zip(y.iter())
        .map(|(&ti, &yi)| {
            let r = a * (b * ti).exp() - yi;
            r * r
        })
        .sum::<f64>()
}

/// Writes the `m × 2` Jacobian `∂rᵢ/∂[a, b]` at `params = [a, b]` into
/// `out` in row-major order (`out[i*2 + j]`), where `m = t.len()`:
///
/// ```text
///            ∂a              ∂b
/// rᵢ:   exp(b·tᵢ)     a·tᵢ·exp(b·tᵢ)
/// ```
///
/// `out.len()` must equal `2 · t.len()`.
pub fn exponential_fit_jacobian(params: &[f64], t: &[f64], out: &mut [f64]) {
    debug_assert_eq!(params.len(), 2);
    debug_assert_eq!(out.len(), t.len() * 2);
    let (a, b) = (params[0], params[1]);
    for i in 0..t.len() {
        let e = (b * t[i]).exp();
        out[i * 2] = e;
        out[i * 2 + 1] = a * t[i] * e;
    }
}

/// Exponential data-fitting problem `min_{a,b} ½ Σ (a·exp(b·tᵢ) − yᵢ)²`.
/// Carries the data `(t, y)` on the struct; the generic parameter `V`
/// pins the parameter-vector backend. Construct with [`new`](Self::new).
///
/// `Jacobian` is implemented for the feature-gated backends (nalgebra
/// `DVector<f64>`, faer `Col<f64>`, and ndarray `Array1<f64>`); the
/// default `Vec<f64>` backend supplies [`CostFunction`](crate::CostFunction)
/// and [`Residual`](crate::Residual) but not [`Jacobian`](crate::Jacobian).
pub struct ExponentialFit<V = Vec<f64>> {
    /// Sample abscissae `tᵢ`.
    pub t: Vec<f64>,
    /// Observed values `yᵢ`.
    pub y: Vec<f64>,
    _backend: PhantomData<fn() -> V>,
}

impl<V> ExponentialFit<V> {
    /// Build an exponential-fit problem from data. Panics if `t` and
    /// `y` have different lengths.
    pub fn new(t: Vec<f64>, y: Vec<f64>) -> Self {
        assert_eq!(t.len(), y.len(), "ExponentialFit: t and y length mismatch");
        Self {
            t,
            y,
            _backend: PhantomData,
        }
    }

    /// Build the exact-data instance whose global minimum is `(a, b)`
    /// with `f = 0`: samples `tᵢ = i · dt` for `i ∈ 0..m` and sets
    /// `yᵢ = a · exp(b · tᵢ)`. Handy for tests and benchmarks that need
    /// a known optimum.
    pub fn sampled(a: f64, b: f64, m: usize, dt: f64) -> Self {
        let t: Vec<f64> = (0..m).map(|i| i as f64 * dt).collect();
        let y: Vec<f64> = t.iter().map(|&ti| a * (b * ti).exp()).collect();
        Self::new(t, y)
    }
}

impl<V> HasSpec for ExponentialFit<V> {
    const SPEC: &'static ProblemSpec = &EXPONENTIAL_FIT_SPEC;
}

mod vec_impl {
    use super::{ExponentialFit, exponential_fit, exponential_fit_residuals};
    use crate::{CostFunction, Residual};

    impl CostFunction for ExponentialFit<Vec<f64>> {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(exponential_fit(x, &self.t, &self.y))
        }
    }

    impl Residual for ExponentialFit<Vec<f64>> {
        type Param = Vec<f64>;
        type Output = Vec<f64>;
        type Error = std::convert::Infallible;
        fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
            let mut out = vec![0.0; self.t.len()];
            exponential_fit_residuals(x, &self.t, &self.y, &mut out);
            Ok(out)
        }
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_impl {
    use super::{
        ExponentialFit, exponential_fit, exponential_fit_jacobian, exponential_fit_residuals,
    };
    use crate::{CostFunction, Jacobian, Residual};
    use nalgebra::{DMatrix, DVector};

    impl CostFunction for ExponentialFit<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(exponential_fit(x.as_slice(), &self.t, &self.y))
        }
    }

    impl Residual for ExponentialFit<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = DVector<f64>;
        type Error = std::convert::Infallible;
        fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, std::convert::Infallible> {
            let mut out = DVector::zeros(self.t.len());
            exponential_fit_residuals(x.as_slice(), &self.t, &self.y, out.as_mut_slice());
            Ok(out)
        }
    }

    impl Jacobian for ExponentialFit<DVector<f64>> {
        type Jacobian = DMatrix<f64>;
        fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, std::convert::Infallible> {
            let m = self.t.len();
            let mut buf = vec![0.0_f64; m * 2];
            exponential_fit_jacobian(x.as_slice(), &self.t, &mut buf);
            // Row-major buffer, m×2 layout.
            Ok(DMatrix::from_row_slice(m, 2, &buf))
        }
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::{
        ExponentialFit, exponential_fit, exponential_fit_jacobian, exponential_fit_residuals,
    };
    use crate::{CostFunction, Jacobian, Residual};
    use ndarray::{Array1, Array2};

    impl CostFunction for ExponentialFit<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Array1<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(exponential_fit(
                x.as_slice().expect("Array1 is contiguous"),
                &self.t,
                &self.y,
            ))
        }
    }

    impl Residual for ExponentialFit<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = Array1<f64>;
        type Error = std::convert::Infallible;
        fn residual(&self, x: &Array1<f64>) -> Result<Array1<f64>, std::convert::Infallible> {
            let mut out = Array1::zeros(self.t.len());
            exponential_fit_residuals(
                x.as_slice().expect("Array1 is contiguous"),
                &self.t,
                &self.y,
                out.as_slice_mut().expect("Array1 is contiguous"),
            );
            Ok(out)
        }
    }

    impl Jacobian for ExponentialFit<Array1<f64>> {
        type Jacobian = Array2<f64>;
        fn jacobian(&self, x: &Array1<f64>) -> Result<Array2<f64>, std::convert::Infallible> {
            let m = self.t.len();
            let mut buf = vec![0.0_f64; m * 2];
            exponential_fit_jacobian(
                x.as_slice().expect("Array1 is contiguous"),
                &self.t,
                &mut buf,
            );
            // Row-major buffer, m×2 layout—the default C-order
            // `from_shape_vec` matches the nalgebra `from_row_slice` mirror.
            Ok(Array2::from_shape_vec((m, 2), buf).expect("m*2 entries for an m×2"))
        }
    }
}

#[cfg(feature = "faer")]
mod faer_impl {
    use super::{
        ExponentialFit, exponential_fit, exponential_fit_jacobian, exponential_fit_residuals,
    };
    use crate::{CostFunction, Jacobian, Residual};
    use faer::{Col, Mat};

    impl CostFunction for ExponentialFit<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(exponential_fit(&[x[0], x[1]], &self.t, &self.y))
        }
    }

    impl Residual for ExponentialFit<Col<f64>> {
        type Param = Col<f64>;
        type Output = Col<f64>;
        type Error = std::convert::Infallible;
        fn residual(&self, x: &Col<f64>) -> Result<Col<f64>, std::convert::Infallible> {
            let m = self.t.len();
            let mut buf = vec![0.0_f64; m];
            exponential_fit_residuals(&[x[0], x[1]], &self.t, &self.y, &mut buf);
            Ok(Col::from_fn(m, |i| buf[i]))
        }
    }

    impl Jacobian for ExponentialFit<Col<f64>> {
        type Jacobian = Mat<f64>;
        fn jacobian(&self, x: &Col<f64>) -> Result<Mat<f64>, std::convert::Infallible> {
            let m = self.t.len();
            let mut buf = vec![0.0_f64; m * 2];
            exponential_fit_jacobian(&[x[0], x[1]], &self.t, &mut buf);
            Ok(Mat::from_fn(m, 2, |i, j| buf[i * 2 + j]))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residuals_vanish_at_true_params() {
        // Exact data sampled from (a, b) = (1000, −1): residuals are 0.
        let p: ExponentialFit = ExponentialFit::sampled(1000.0, -1.0, 8, 0.5);
        let mut r = vec![0.0; p.t.len()];
        exponential_fit_residuals(&[1000.0, -1.0], &p.t, &p.y, &mut r);
        for v in r {
            assert!(v.abs() < 1e-9, "residual = {v}");
        }
    }

    #[test]
    fn cost_is_zero_at_true_params() {
        let p: ExponentialFit = ExponentialFit::sampled(1000.0, -1.0, 8, 0.5);
        assert!(exponential_fit(&[1000.0, -1.0], &p.t, &p.y) < 1e-15);
    }

    #[test]
    fn cost_matches_half_sum_of_squared_residuals() {
        let p: ExponentialFit = ExponentialFit::sampled(1000.0, -1.0, 8, 0.5);
        let params = [800.0, -0.7];
        let mut r = vec![0.0; p.t.len()];
        exponential_fit_residuals(&params, &p.t, &p.y, &mut r);
        let half_sum_sq = 0.5 * r.iter().map(|ri| ri * ri).sum::<f64>();
        let c = exponential_fit(&params, &p.t, &p.y);
        assert!((c - half_sum_sq).abs() < 1e-6, "c={c}, hss={half_sum_sq}");
    }

    #[test]
    fn jacobian_matches_finite_difference() {
        let t = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let params = [500.0, -0.6];
        let mut j = vec![0.0; t.len() * 2];
        exponential_fit_jacobian(&params, &t, &mut j);

        let y = vec![0.0; t.len()]; // y irrelevant for the derivative
        let h = 1e-6;
        for col in 0..2 {
            let mut pp = params;
            let mut pm = params;
            pp[col] += h;
            pm[col] -= h;
            let mut rp = vec![0.0; t.len()];
            let mut rm = vec![0.0; t.len()];
            exponential_fit_residuals(&pp, &t, &y, &mut rp);
            exponential_fit_residuals(&pm, &t, &y, &mut rm);
            for i in 0..t.len() {
                let fd = (rp[i] - rm[i]) / (2.0 * h);
                let rel = (j[i * 2 + col] - fd).abs() / fd.abs().max(1.0);
                assert!(
                    rel < 1e-5,
                    "i={i}, col={col}, j={}, fd={fd}",
                    j[i * 2 + col]
                );
            }
        }
    }

    #[test]
    fn columns_are_poorly_scaled() {
        // The whole point: ‖∂r/∂b‖ ≫ ‖∂r/∂a‖ when the amplitude is large.
        let p: ExponentialFit = ExponentialFit::sampled(1000.0, -1.0, 8, 0.5);
        let mut j = vec![0.0; p.t.len() * 2];
        exponential_fit_jacobian(&[1000.0, -1.0], &p.t, &mut j);
        let col_a: f64 = (0..p.t.len()).map(|i| j[i * 2].powi(2)).sum();
        let col_b: f64 = (0..p.t.len()).map(|i| j[i * 2 + 1].powi(2)).sum();
        assert!(
            col_b / col_a > 1e3,
            "expected strongly disparate column norms, got ratio {}",
            col_b / col_a
        );
    }

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        let spec = <ExponentialFit<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(spec.name, "Exponential fit");
        assert!(spec.properties.smooth);
        assert!(!spec.properties.convex);
        assert!(matches!(spec.dim, Dimensionality::Fixed(2)));
        assert!(!spec.references.is_empty());
    }
}
