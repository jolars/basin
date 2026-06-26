// Index-based loops mirror Powell's exposition (LINCOA, Powell 2015) and the
// dense index arithmetic of the active-set QR / H-factorization algebra;
// blanket-allowed here.
#![allow(clippy::needless_range_loop)]

//! LINCOA (Powell 2015): linearly-constrained model-based derivative-free
//! optimization.
//!
//! [`Lincoa`] reuses the shared Powell quadratic-model core (the crate-internal
//! `powell` module, shared with NEWUOA and BOBYQA) and swaps the trust-region
//! subproblem for the linearly-constrained `trstep` (projected truncated-CG) with
//! the active-set step `getact`/`ActiveSetQr`. It binds
//! [`CostFunction`](crate::core::problem::CostFunction) +
//! [`LinearConstraints`](crate::core::constraint::LinearConstraints) (the
//! general linear form: box bounds, equalities, and inequalities).
//!
//! # Architecture
//!
//! Like its siblings, the model is maintained through a factored inverse-KKT
//! matrix `H`; the driver is the Figure-1 loop with the ρ/Δ schedule, the LINCOA
//! geometry step (`geostep`), origin shifts, and the `tryqalt` alternative-model
//! swap. The linear feasible region enters through the active-set QR
//! (`ActiveSetQr`) threaded across iterations and the residual vector `rescon`.
//! Box bounds and linear equalities fold into the inequality form `A x ≤ b`
//! exactly as PRIMA's `get_lincon` does (`fold_constraints`); the public solver
//! binds the general-form `LinearConstraints` aggregator, which carries all
//! three kinds at once (see [`Lincoa`]).
//!
//! # Backends
//!
//! Backend-generic over the parameter vector; the constraint matrix `A` is read
//! through [`MatTransposeVec`](crate::core::math::MatTransposeVec) only (each
//! constraint normal is `Aᵀ eⱼ`), which every backend's matrix type provides
//! (`DenseMatrix` for `Vec<f64>`, `Array2`, `DMatrix`, `Mat`), so all four work.
//! The model algebra is internal pure-Rust `Vec<f64>` scratch; wasm-clean.
//!
//! # References
//!
//! M. J. D. Powell, *On fast trust region methods for quadratic models with
//! linear constraints*, Math. Program. Comput. 7:237–267 (2015). Ported from
//! [PRIMA](https://github.com/libprima/prima) v0.7.2.

pub(crate) mod driver;
pub(crate) mod geometry;
pub(crate) mod getact;
pub(crate) mod init;
pub(crate) mod trstep;

#[cfg(test)]
mod parity;

use crate::core::constraint::LinearConstraints;
use crate::core::inner::InitialState;
use crate::core::math::{MatTransposeVec, Scalar, VectorLen};
use crate::core::problem::{CostFunction, Problem};
use crate::core::solver::Solver;
use crate::core::state::LincoaState;
use crate::core::termination::TerminationReason;

use driver::{LincoaWork, Transition};
use init::fold_constraints;

/// LINCOA (Powell 2015): linearly-constrained model-based derivative-free
/// trust-region optimization.
///
/// LINCOA minimizes a smooth objective subject to linear constraints (box
/// bounds, linear equalities, and linear inequalities, all reduced internally to
/// `A x ≤ b`) using only objective values, no derivatives. It maintains the same
/// least-Frobenius-norm quadratic surrogate as [`Newuoa`](crate::Newuoa) /
/// [`Bobyqa`](crate::Bobyqa), and keeps every trust-region iterate feasible via a
/// projected truncated-CG subproblem with an active-set QR.
///
/// Drive it with an [`Executor`](crate::Executor) over a [`LincoaState`] on a
/// problem implementing [`CostFunction`] and
/// [`LinearConstraints`] (here only the inequality block is present):
///
/// ```
/// use basin::{CostFunction, DenseMatrix, Executor, Lincoa, LincoaState, MaxCostEvals};
/// use basin::core::constraint::LinearConstraints;
///
/// // min ‖x − (2, 2)‖²  s.t.  x0 + x1 ≤ 2   (optimum: the projection (1, 1)).
/// // The pure-Rust `DenseMatrix`/`Vec<f64>` carriers need no backend feature.
/// struct Proj { a: DenseMatrix<f64>, b: Vec<f64> }
/// impl CostFunction for Proj {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
///         Ok((x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2))
///     }
/// }
/// impl LinearConstraints for Proj {
///     type Matrix = DenseMatrix<f64>;
///     fn inequalities(&self) -> Option<(&DenseMatrix<f64>, &Vec<f64>)> {
///         Some((&self.a, &self.b))
///     }
/// }
///
/// let problem = Proj { a: DenseMatrix::from_row_slice(1, 2, &[1.0, 1.0]), b: vec![2.0] };
/// let solver = Lincoa::new().with_rho_beg(0.5).with_rho_end(1e-7);
/// let state = LincoaState::new(vec![0.0, 0.0]);
/// let result = Executor::new(problem, solver, state)
///     .terminate_on(MaxCostEvals(500))
///     .run()
///     .unwrap();
/// assert!((result.best_param()[0] - 1.0).abs() < 1e-3);
/// assert!((result.best_param()[1] - 1.0).abs() < 1e-3);
/// ```
///
/// # Configuration
///
/// - [`with_rho_beg`](Self::with_rho_beg): initial trust-region radius `ρ_beg`
///   (a reasonable initial change to the variables; default `1.0`).
/// - [`with_rho_end`](Self::with_rho_end): final radius `ρ_end`, the required
///   accuracy (default `1e-6`); must satisfy `ρ_beg > ρ_end > 0`.
/// - [`with_npt`](Self::with_npt): interpolation-set size `npt`, in
///   `[n+2, ½(n+1)(n+2)]` (default `2n+1`).
///
/// # Constraints
///
/// LINCOA binds the general-form
/// [`LinearConstraints`]: a problem
/// exposes any combination of box bounds, linear equalities, and linear
/// inequalities (each accessor is optional). All present blocks are folded into
/// one unit-normalized `A x ≤ b` system: bounds as `±eᵢ` rows, an equality
/// `aᵀx = β` as the pair `aᵀx ≤ β`, `−aᵀx ≤ −β`, and every iterate is kept
/// feasible by the active-set projection. The start point should be feasible;
/// if not, LINCOA relaxes `b` to make it feasible (Powell's behavior).
///
/// # Termination
///
/// Natural convergence is `ρ` reaching `ρ_end`, signalled as
/// [`TerminationReason::SolverConverged`]. Add
/// [`MaxCostEvals`](crate::MaxCostEvals) to cap the budget or
/// [`RhoTolerance`](crate::RhoTolerance) to stop at a coarser `ρ`.
///
/// # Backends
///
/// Backend-generic: the parameter vector needs only [`Clone`], [`VectorLen`], and
/// indexing, and the constraint matrix is read via
/// [`MatTransposeVec`] (`Aᵀ eⱼ`), which every
/// backend's matrix type implements, so `Vec<f64>`, nalgebra, ndarray, and faer
/// all work. wasm-clean (the model algebra is pure-Rust `Vec<f64>` scratch).
///
/// # References
///
/// M. J. D. Powell, *On fast trust region methods for quadratic models with
/// linear constraints*, Math. Program. Comput. 7:237–267 (2015). Cross-validated
/// against [PRIMA](https://github.com/libprima/prima) v0.7.2, the authoritative
/// source for the exact formulas. PRIMA is BSD 3-Clause licensed; its required
/// notice is retained in the crate's `COPYRIGHT` file.
///
/// [`CostFunction`]: crate::core::problem::CostFunction
/// [`TerminationReason::SolverConverged`]: crate::TerminationReason::SolverConverged
pub struct Lincoa<F = f64> {
    rho_beg: F,
    rho_end: F,
    npt: Option<usize>,
    /// Built in [`Solver::init`]; the resumable model + constraints + schedule.
    work: Option<LincoaWork<F>>,
}

impl<F: Scalar> Lincoa<F> {
    /// A LINCOA solver with the default schedule (`ρ_beg = 1`, `ρ_end = 1e-6`,
    /// `npt = 2n+1`). Tune with the `with_*` builders.
    pub fn new() -> Self {
        Self {
            rho_beg: F::from_f64(1.0).expect("1.0 representable"),
            rho_end: F::from_f64(1e-6).expect("1e-6 representable"),
            npt: None,
            work: None,
        }
    }

    /// Set the initial trust-region radius `ρ_beg` (also the initial `Δ`).
    pub fn with_rho_beg(mut self, rho_beg: F) -> Self {
        self.rho_beg = rho_beg;
        self
    }

    /// Set the final trust-region radius `ρ_end`. Must satisfy `ρ_beg > ρ_end > 0`.
    pub fn with_rho_end(mut self, rho_end: F) -> Self {
        self.rho_end = rho_end;
        self
    }

    /// Set the interpolation-set size `npt`, in `[n+2, ½(n+1)(n+2)]`. Defaults to
    /// `2n+1` when unset.
    pub fn with_npt(mut self, npt: usize) -> Self {
        self.npt = Some(npt);
        self
    }
}

impl<F: Scalar> Default for Lincoa<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a `V` from a flat `&[F]`, reusing `template` for type and length.
fn fill_from<V, F>(template: &V, slice: &[F]) -> V
where
    V: Clone + std::ops::IndexMut<usize, Output = F>,
    F: Copy,
{
    let mut v = template.clone();
    for (i, &x) in slice.iter().enumerate() {
        v[i] = x;
    }
    v
}

impl<V, F> InitialState<V> for Lincoa<F>
where
    F: Scalar,
    V: Clone,
{
    type State = LincoaState<V, F>;
    fn seed(&self, x: &V) -> Self::State {
        LincoaState::new(x.clone())
    }
}

impl<P, V, F> Solver<P, LincoaState<V, F>> for Lincoa<F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F> + LinearConstraints,
    P::Matrix: MatTransposeVec<V>,
    V: Clone
        + VectorLen
        + std::ops::Index<usize, Output = F>
        + std::ops::IndexMut<usize, Output = F>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: LincoaState<V, F>,
    ) -> Result<LincoaState<V, F>, Self::Error> {
        let n = state.param.vec_len();
        assert!(n >= 1, "Lincoa requires a non-empty start point");
        let npt = self.npt.unwrap_or(2 * n + 1);

        let x0: Vec<F> = (0..n).map(|i| state.param[i]).collect();
        let template = state.param.clone();

        // Extract each linear block into `(normal, rhs)` rows. A constraint
        // normal is row `j` of `A` = `Aᵀ eⱼ` (one `mat_transpose_vec` per row);
        // `b` is read element-wise. `fold_constraints` then folds box bounds,
        // equalities, and inequalities into one unit-normalized `A x ≤ b` system
        // and relaxes `b` for a feasible start.
        let extract_rows = |a: &P::Matrix, b: &V| -> Vec<(Vec<F>, F)> {
            let m = b.vec_len();
            (0..m)
                .map(|j| {
                    let mut ej = b.clone();
                    for i in 0..m {
                        ej[i] = if i == j { F::one() } else { F::zero() };
                    }
                    let normal = a.mat_transpose_vec(&ej);
                    ((0..n).map(|i| normal[i]).collect(), b[j])
                })
                .collect()
        };
        let (amat, bvec_abs) = {
            let inner = problem.inner();
            let ineq = inner
                .inequalities()
                .map(|(a, b)| extract_rows(a, b))
                .unwrap_or_default();
            let eq = inner
                .equalities()
                .map(|(a, b)| extract_rows(a, b))
                .unwrap_or_default();
            let lower: Option<Vec<F>> = inner.lower().map(|v| (0..n).map(|i| v[i]).collect());
            let upper: Option<Vec<F>> = inner.upper().map(|v| (0..n).map(|i| v[i]).collect());
            fold_constraints(n, &x0, lower.as_deref(), upper.as_deref(), &eq, &ineq)
        };

        let (work, best_x, best_f) = {
            let mut eval =
                |slice: &[F]| -> Result<F, P::Error> { problem.cost(&fill_from(&template, slice)) };
            LincoaWork::try_init(
                x0,
                amat,
                bvec_abs,
                self.rho_beg,
                self.rho_end,
                npt,
                &mut eval,
            )?
        };

        state.param = fill_from(&template, &best_x);
        state.cost = Some(best_f);
        state.rho = work.rho();
        self.work = Some(work);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: LincoaState<V, F>,
    ) -> Result<(LincoaState<V, F>, Option<TerminationReason>), Self::Error> {
        let template = state.param.clone();
        let work = self
            .work
            .as_mut()
            .expect("Lincoa::init must run before next_iter");

        let out = {
            let mut eval =
                |slice: &[F]| -> Result<F, P::Error> { problem.cost(&fill_from(&template, slice)) };
            work.step(&mut eval)?
        };
        state.rho = work.rho();

        // The reported iterate is the model's best *feasible* point (x_opt); the
        // step also evaluates possibly-infeasible geometry points, which must not
        // be reported. Eval counting is automatic through the `Problem` wrapper.
        let (best_x, best_f) = work.best();
        state.param = fill_from(&template, &best_x);
        state.cost = Some(best_f);

        let reason = match out.transition {
            Transition::Converged => Some(TerminationReason::SolverConverged),
            Transition::Continue | Transition::RhoReduced => None,
        };
        Ok((state, reason))
    }
}
