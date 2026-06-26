//! Benchmark-support library for basin's cross-framework comparisons
//! (axis 3 of the bench plan): basin vs the `levenberg-marquardt` crate
//! (NLLS) and vs `argmin` and `gomez` (gradient-based and derivative-free).
//!
//! The pattern throughout is to reuse basin's corpus as the source of
//! truth and build thin adapters for the competitors, so every framework
//! attacks bit-identical math:
//!
//! - **lm crate** ([`LmExponentialFit`], [`LmPowellSingular`],
//!   [`LmVarDim`]): NLLS problems implemented against
//!   [`levenberg_marquardt::LeastSquaresProblem`] with formulas copied
//!   from basin's corpus. Both crates use nalgebra ^0.34 (shared
//!   lockfile), so they operate on the very same `DVector`/`DMatrix`.
//! - **argmin** ([`ArgminProblem`]): a single fn-pointer adapter that
//!   wraps any of basin's `pub` raw cost/gradient functions
//!   (`fn(&[f64]) -> f64`/`fn(&[f64], &mut [f64])`) into argmin's
//!   [`CostFunction`](argmin::core::CostFunction) +
//!   [`Gradient`](argmin::core::Gradient) on the `Vec<f64>` backend—
//!   the backend both basin and argmin support natively, so GD and
//!   Nelder-Mead run on identical param types.
//! - **gomez** ([`GomezProblem`]): a single fn-pointer adapter that
//!   wraps a basin raw cost function into gomez's
//!   [`Problem`](gomez::Problem) + [`Function`](gomez::Function), so
//!   gomez's derivative-free optimizers see the same math basin and
//!   argmin do. gomez has no gradient-based optimizer that lines up
//!   with basin's GD or L-BFGS, so the adapter is cost-only.
//!
//! `benches/compare.rs` (LM) and `benches/gd_nm.rs` (argmin) hold the
//! timing harnesses; the `verify*` binaries print one-shot convergence
//! comparisons to confirm the solvers reach the same optimum in
//! comparable work before the timings are trusted.

use std::marker::PhantomData;

use argmin::core::{CostFunction, Error, Gradient};
use basin::{Jacobian, Residual};
use faer::{Col, Mat};
use gomez::nalgebra as gna;
use gomez::{Domain, Function, Problem};
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::storage::Owned;
use nalgebra::{DMatrix, DVector, Dyn};

/// argmin-side adapter wrapping a pair of basin raw functions (a cost
/// `fn(&[f64]) -> f64` and a gradient `fn(&[f64], &mut [f64])`) into
/// argmin's [`CostFunction`] + [`Gradient`] on the `Vec<f64>` backend.
/// One type covers every problem in basin's corpus, so the argmin side
/// computes exactly the same math basin does.
pub struct ArgminProblem {
    cost: fn(&[f64]) -> f64,
    grad: fn(&[f64], &mut [f64]),
}

impl ArgminProblem {
    pub fn new(cost: fn(&[f64]) -> f64, grad: fn(&[f64], &mut [f64])) -> Self {
        Self { cost, grad }
    }
}

impl CostFunction for ArgminProblem {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, p: &Vec<f64>) -> Result<f64, Error> {
        Ok((self.cost)(p))
    }
}

impl Gradient for ArgminProblem {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;
    fn gradient(&self, p: &Vec<f64>) -> Result<Vec<f64>, Error> {
        let mut g = vec![0.0; p.len()];
        (self.grad)(p, &mut g);
        Ok(g)
    }
}

/// gomez-side adapter wrapping a basin raw cost `fn(&[f64]) -> f64` into
/// gomez's [`Problem`] + [`Function`] over an unconstrained `n`-D domain.
/// Cost-only—gomez has no gradient-using optimizer that lines up with
/// basin's first-order solvers, so the adapter exposes only the value;
/// gomez approximates derivatives internally if a method ever needs them.
///
/// Operates on `gomez::nalgebra::DVector<f64>` (gomez's bundled nalgebra
/// 0.32), which is a separate version from the lm/basin nalgebra 0.34
/// elsewhere in this crate—the two majors coexist in the lockfile and
/// never meet, because gomez only ever sees its own type.
pub struct GomezProblem {
    cost: fn(&[f64]) -> f64,
    n: usize,
}

impl GomezProblem {
    pub fn new(cost: fn(&[f64]) -> f64, n: usize) -> Self {
        Self { cost, n }
    }
}

impl Problem for GomezProblem {
    type Field = f64;
    fn domain(&self) -> Domain<f64> {
        Domain::unconstrained(self.n)
    }
}

impl Function for GomezProblem {
    fn apply<Sx>(&self, x: &gna::Vector<f64, gna::Dyn, Sx>) -> f64
    where
        Sx: gna::Storage<f64, gna::Dyn> + gna::IsContiguous,
    {
        (self.cost)(x.as_slice())
    }
}

/// The lm crate's default convergence tolerance (`30·ε`, set for
/// `ftol`/`xtol`/`gtol` in `LevenbergMarquardt::new()`). basin's LM is
/// configured to the same value on all three relative tests so the two
/// solvers stop at comparable points.
pub const LM_DEFAULT_TOL: f64 = 30.0 * f64::EPSILON;

/// Exponential fit `rᵢ = a·exp(b·tᵢ) − yᵢ` (mirrors
/// `basin::problems::ExponentialFit`). `n = 2`, `m = t.len()`.
pub struct LmExponentialFit {
    t: Vec<f64>,
    y: Vec<f64>,
    p: DVector<f64>,
}

impl LmExponentialFit {
    /// Same exact-data instance as `ExponentialFit::sampled`
    /// (`tᵢ = i·dt`, `yᵢ = a·exp(b·tᵢ)`), starting at `x0 = [a0, b0]`.
    pub fn sampled(a: f64, b: f64, m: usize, dt: f64, a0: f64, b0: f64) -> Self {
        let t: Vec<f64> = (0..m).map(|i| i as f64 * dt).collect();
        let y: Vec<f64> = t.iter().map(|&ti| a * (b * ti).exp()).collect();
        Self {
            t,
            y,
            p: DVector::from_vec(vec![a0, b0]),
        }
    }
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for LmExponentialFit {
    type ParameterStorage = Owned<f64, Dyn>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;

    fn set_params(&mut self, x: &DVector<f64>) {
        self.p.copy_from(x);
    }

    fn params(&self) -> DVector<f64> {
        self.p.clone()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        let (a, b) = (self.p[0], self.p[1]);
        Some(DVector::from_iterator(
            self.t.len(),
            self.t
                .iter()
                .zip(&self.y)
                .map(|(&ti, &yi)| a * (b * ti).exp() - yi),
        ))
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
        let (a, b) = (self.p[0], self.p[1]);
        let mut j = DMatrix::zeros(self.t.len(), 2);
        for (i, &ti) in self.t.iter().enumerate() {
            let e = (b * ti).exp();
            j[(i, 0)] = e;
            j[(i, 1)] = a * ti * e;
        }
        Some(j)
    }
}

// ---------------------------------------------------------------------
// Variably Dimensioned function (Moré-Garbow-Hillstrom 1981, #25): the
// scalable, well-conditioned, full-rank problem used to probe eunoia's
// n ∈ {10, 20, 30} regime. `n` params, `m = n + 2` residuals, unique
// minimum f = 0 at x = (1, …, 1). Because it converges cleanly (unlike
// the rank-deficient Powell), all three solvers reach the optimum in
// comparable iteration counts—so the timing reflects per-iteration
// cost, and any iteration-count difference is itself a clean signal.
//
//   rᵢ      = xᵢ − 1                  (i = 0 … n−1)
//   r_n     = Σⱼ (j+1)(xⱼ − 1) =: s
//   r_{n+1} = s²
// ---------------------------------------------------------------------

/// Standard MGH starting point `xⱼ = 1 − (j+1)/n`.
pub fn vardim_start(n: usize) -> Vec<f64> {
    (0..n).map(|j| 1.0 - (j as f64 + 1.0) / n as f64).collect()
}

fn vardim_s(x: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(j, &xj)| (j as f64 + 1.0) * (xj - 1.0))
        .sum()
}

fn vardim_residual(x: &[f64], out: &mut [f64]) {
    let n = x.len();
    let s = vardim_s(x);
    for j in 0..n {
        out[j] = x[j] - 1.0;
    }
    out[n] = s;
    out[n + 1] = s * s;
}

/// Row-major `(n+2) × n` Jacobian: `out[i*n + j] = ∂rᵢ/∂xⱼ`.
fn vardim_jacobian_row_major(x: &[f64], out: &mut [f64]) {
    let n = x.len();
    for v in out.iter_mut() {
        *v = 0.0;
    }
    let s = vardim_s(x);
    for j in 0..n {
        let w = j as f64 + 1.0;
        out[j * n + j] = 1.0; // identity block, rows 0..n
        out[n * n + j] = w; // row n: ∂s/∂xⱼ = (j+1)
        out[(n + 1) * n + j] = 2.0 * s * w; // row n+1: ∂s²/∂xⱼ = 2s(j+1)
    }
}

/// basin-side Variably Dimensioned problem, generic over the parameter
/// backend `V` (mirrors `basin::problems::ExponentialFit<V>`).
pub struct VarDim<V> {
    n: usize,
    _backend: PhantomData<fn() -> V>,
}

impl<V> VarDim<V> {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            _backend: PhantomData,
        }
    }
}

impl Residual for VarDim<DVector<f64>> {
    type Param = DVector<f64>;
    type Output = DVector<f64>;
    type Error = std::convert::Infallible;
    fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; self.n + 2];
        vardim_residual(x.as_slice(), &mut out);
        Ok(DVector::from_vec(out))
    }
}

impl Jacobian for VarDim<DVector<f64>> {
    type Jacobian = DMatrix<f64>;
    fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; (self.n + 2) * self.n];
        vardim_jacobian_row_major(x.as_slice(), &mut out);
        Ok(DMatrix::from_row_slice(self.n + 2, self.n, &out))
    }
}

impl Residual for VarDim<Col<f64>> {
    type Param = Col<f64>;
    type Output = Col<f64>;
    type Error = std::convert::Infallible;
    fn residual(&self, x: &Col<f64>) -> Result<Col<f64>, std::convert::Infallible> {
        let xs: Vec<f64> = (0..self.n).map(|i| x[i]).collect();
        let mut out = vec![0.0; self.n + 2];
        vardim_residual(&xs, &mut out);
        Ok(Col::from_fn(self.n + 2, |i| out[i]))
    }
}

impl Jacobian for VarDim<Col<f64>> {
    type Jacobian = Mat<f64>;
    fn jacobian(&self, x: &Col<f64>) -> Result<Mat<f64>, std::convert::Infallible> {
        let xs: Vec<f64> = (0..self.n).map(|i| x[i]).collect();
        let mut out = vec![0.0; (self.n + 2) * self.n];
        vardim_jacobian_row_major(&xs, &mut out);
        Ok(Mat::from_fn(self.n + 2, self.n, |i, j| out[i * self.n + j]))
    }
}

/// lm-crate-side Variably Dimensioned problem.
pub struct LmVarDim {
    n: usize,
    p: DVector<f64>,
}

impl LmVarDim {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            p: DVector::from_vec(vardim_start(n)),
        }
    }
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for LmVarDim {
    type ParameterStorage = Owned<f64, Dyn>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;

    fn set_params(&mut self, x: &DVector<f64>) {
        self.p.copy_from(x);
    }

    fn params(&self) -> DVector<f64> {
        self.p.clone()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        let mut out = vec![0.0; self.n + 2];
        vardim_residual(self.p.as_slice(), &mut out);
        Some(DVector::from_vec(out))
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
        let mut out = vec![0.0; (self.n + 2) * self.n];
        vardim_jacobian_row_major(self.p.as_slice(), &mut out);
        Some(DMatrix::from_row_slice(self.n + 2, self.n, &out))
    }
}

// ---------------------------------------------------------------------
// Underdetermined trigonometric least-squares (issue #10): a smooth
// nonlinear problem with `m < n`, so `JᵀJ` is rank-deficient (singular
// without the LM damping `μ·D`). This is the regime where eunoia's
// ellipse fits live and where basin's LM showed a large per-iteration
// cost gap vs the lm crate. The target `b` is deliberately outside the
// range of `A·sin(x)` (whose entries are bounded by `Σⱼ|Aᵢⱼ|`), so the
// problem is *infeasible* and the solver genuinely iterates (with
// rejected steps that bump `μ`) instead of hitting a zero-residual fit
// in one step.
//
//   rᵢ(x)   = Σⱼ Aᵢⱼ·sin(xⱼ) − bᵢ            (i = 0 … m−1)
//   ∂rᵢ/∂xⱼ = Aᵢⱼ·cos(xⱼ)
//
// `A`, `b`, and the start `x₀` are fixed deterministic splitmix64 draws,
// so the basin and lm-crate sides attack bit-identical math.
// ---------------------------------------------------------------------

/// splitmix64-style deterministic pseudo-random in `[-0.5, 0.5)`. Shared
/// by both sides so the generated `A`/`b`/`x₀` are bit-identical.
fn splitmix(i: u64) -> f64 {
    let mut x = i.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}

/// Shared data + math for the underdetermined problem, reused verbatim by
/// the basin and lm-crate adapters.
pub struct UnderDetData {
    m: usize,
    n: usize,
    a: Vec<f64>, // row-major m × n: a[i*n + j] = Aᵢⱼ
    b: Vec<f64>, // length m
}

impl UnderDetData {
    /// `m` residuals, `n` parameters (use `m < n` for the rank-deficient
    /// regime). Deterministic, so repeated construction is identical.
    pub fn new(m: usize, n: usize) -> Self {
        let a = (0..m * n).map(|k| splitmix(k as u64)).collect();
        // b ∈ 3 ± 2.5, well outside |A·sin(x)| ≤ Σⱼ|Aᵢⱼ| (≈ n/4 here),
        // so the fit is infeasible and the solver must iterate.
        let b = (0..m)
            .map(|i| 3.0 + 5.0 * splitmix(1_000_000 + i as u64))
            .collect();
        Self { m, n, a, b }
    }

    /// Start point `x₀ⱼ = 0.1·splitmix(42+j)`—small, generic, away from
    /// any stationary point.
    pub fn start(&self) -> Vec<f64> {
        (0..self.n).map(|j| 0.1 * splitmix(42 + j as u64)).collect()
    }

    fn residual(&self, x: &[f64], out: &mut [f64]) {
        for (i, out_i) in out.iter_mut().enumerate().take(self.m) {
            let row = &self.a[i * self.n..(i + 1) * self.n];
            *out_i = row
                .iter()
                .zip(x)
                .map(|(&aij, &xj)| aij * xj.sin())
                .sum::<f64>()
                - self.b[i];
        }
    }

    /// Row-major `m × n` Jacobian: `out[i*n + j] = Aᵢⱼ·cos(xⱼ)`.
    fn jacobian_row_major(&self, x: &[f64], out: &mut [f64]) {
        for i in 0..self.m {
            for j in 0..self.n {
                out[i * self.n + j] = self.a[i * self.n + j] * x[j].cos();
            }
        }
    }
}

/// basin-side underdetermined problem, generic over the param backend
/// `V` (mirrors [`VarDim`]).
pub struct UnderDet<V> {
    data: UnderDetData,
    _backend: PhantomData<fn() -> V>,
}

impl<V> UnderDet<V> {
    pub fn new(m: usize, n: usize) -> Self {
        Self {
            data: UnderDetData::new(m, n),
            _backend: PhantomData,
        }
    }

    pub fn start(&self) -> Vec<f64> {
        self.data.start()
    }
}

impl Residual for UnderDet<DVector<f64>> {
    type Param = DVector<f64>;
    type Output = DVector<f64>;
    type Error = std::convert::Infallible;
    fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; self.data.m];
        self.data.residual(x.as_slice(), &mut out);
        Ok(DVector::from_vec(out))
    }
}

impl Jacobian for UnderDet<DVector<f64>> {
    type Jacobian = DMatrix<f64>;
    fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; self.data.m * self.data.n];
        self.data.jacobian_row_major(x.as_slice(), &mut out);
        Ok(DMatrix::from_row_slice(self.data.m, self.data.n, &out))
    }
}

impl Residual for UnderDet<Col<f64>> {
    type Param = Col<f64>;
    type Output = Col<f64>;
    type Error = std::convert::Infallible;
    fn residual(&self, x: &Col<f64>) -> Result<Col<f64>, std::convert::Infallible> {
        let xs: Vec<f64> = (0..self.data.n).map(|i| x[i]).collect();
        let mut out = vec![0.0; self.data.m];
        self.data.residual(&xs, &mut out);
        Ok(Col::from_fn(self.data.m, |i| out[i]))
    }
}

impl Jacobian for UnderDet<Col<f64>> {
    type Jacobian = Mat<f64>;
    fn jacobian(&self, x: &Col<f64>) -> Result<Mat<f64>, std::convert::Infallible> {
        let xs: Vec<f64> = (0..self.data.n).map(|i| x[i]).collect();
        let mut out = vec![0.0; self.data.m * self.data.n];
        self.data.jacobian_row_major(&xs, &mut out);
        Ok(Mat::from_fn(self.data.m, self.data.n, |i, j| {
            out[i * self.data.n + j]
        }))
    }
}

/// lm-crate-side underdetermined problem.
pub struct LmUnderDet {
    data: UnderDetData,
    p: DVector<f64>,
}

impl LmUnderDet {
    pub fn new(m: usize, n: usize) -> Self {
        let data = UnderDetData::new(m, n);
        let p = DVector::from_vec(data.start());
        Self { data, p }
    }
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for LmUnderDet {
    type ParameterStorage = Owned<f64, Dyn>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;

    fn set_params(&mut self, x: &DVector<f64>) {
        self.p.copy_from(x);
    }

    fn params(&self) -> DVector<f64> {
        self.p.clone()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        let mut out = vec![0.0; self.data.m];
        self.data.residual(self.p.as_slice(), &mut out);
        Some(DVector::from_vec(out))
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
        let mut out = vec![0.0; self.data.m * self.data.n];
        self.data.jacobian_row_major(self.p.as_slice(), &mut out);
        Some(DMatrix::from_row_slice(self.data.m, self.data.n, &out))
    }
}

const SQRT_5: f64 = 2.236_067_977_499_79;
const SQRT_10: f64 = 3.162_277_660_168_379_5;

/// Powell's singular function as 4 residuals in 4 unknowns (mirrors
/// `basin::problems::PowellSingular`). Rank-deficient at the optimum.
pub struct LmPowellSingular {
    p: DVector<f64>,
}

impl LmPowellSingular {
    /// Start at `x0` (the classical start is `[3, −1, 0, 1]`).
    pub fn new(x0: [f64; 4]) -> Self {
        Self {
            p: DVector::from_row_slice(&x0),
        }
    }
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for LmPowellSingular {
    type ParameterStorage = Owned<f64, Dyn>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;

    fn set_params(&mut self, x: &DVector<f64>) {
        self.p.copy_from(x);
    }

    fn params(&self) -> DVector<f64> {
        self.p.clone()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        let x = &self.p;
        let d12 = x[1] - 2.0 * x[2];
        let d03 = x[0] - x[3];
        Some(DVector::from_vec(vec![
            x[0] + 10.0 * x[1],
            SQRT_5 * (x[2] - x[3]),
            d12 * d12,
            SQRT_10 * d03 * d03,
        ]))
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
        let x = &self.p;
        let d12 = x[1] - 2.0 * x[2];
        let d03 = x[0] - x[3];
        let mut j = DMatrix::zeros(4, 4);
        j[(0, 0)] = 1.0;
        j[(0, 1)] = 10.0;
        j[(1, 2)] = SQRT_5;
        j[(1, 3)] = -SQRT_5;
        j[(2, 1)] = 2.0 * d12;
        j[(2, 2)] = -4.0 * d12;
        j[(3, 0)] = 2.0 * SQRT_10 * d03;
        j[(3, 3)] = -2.0 * SQRT_10 * d03;
        Some(j)
    }
}
