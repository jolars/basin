// Index-based loops mirror Powell's exposition (NEWUOA, Powell 2006) and the
// dense index arithmetic of the H-factorization algebra; the lint is
// blanket-allowed for this module.
#![allow(clippy::needless_range_loop)]
// The standalone `minimize` driver, `NewuoaConfig` / `NewuoaOutcome`, and the
// model-core read surface have only `#[cfg(test)]` callers (the in-module tests
// and the PRIMA parity fixtures); the public `Newuoa` solver drives the model
// through `NewuoaWork` instead. Blanket-allow so that test-only `pub(crate)`
// surface does not trip dead-code analysis in the non-test build.
#![allow(dead_code)]

//! NEWUOA (Powell 2006) — model-based derivative-free optimization.
//!
//! [`Newuoa`] is Powell's NEWUOA: an unconstrained trust-region method for
//! smooth objectives whose derivatives are unavailable. It maintains a quadratic
//! surrogate `Q` interpolating the objective on a set of `npt` points and updates
//! it by the least-Frobenius-norm rule, so each iteration needs only **one** new
//! objective value. It binds [`CostFunction`] only — no gradient.
//!
//! # Architecture
//!
//! The module is the spine of Powell's model-based family (NEWUOA → BOBYQA →
//! LINCOA): an internal `QuadraticModel`, maintained through a factored
//! inverse-KKT matrix `H = W⁻¹`, plus a swappable trust-region subproblem (here
//! TRSAPP). The two operations that maintain the model are the closed-form
//! initialization (§3) and the least-Frobenius-norm rank-2 update (§4; derivation
//! in Powell 2004a, `references/frobenius-update/`). The Figure-1 driver loop adds
//! the ρ/Δ schedule (§7), the BIGLAG/BIGDEN geometry steps (§6), origin shifts
//! (§7), and the Qint robustness modification (§8). These pieces are
//! crate-internal; only [`Newuoa`] and [`NewuoaState`](crate::NewuoaState) are
//! public.
//!
//! # State / solver split
//!
//! The iterate and the trust-region radius `ρ` live on
//! [`NewuoaState`](crate::NewuoaState); the quadratic model, the factored `H`,
//! and the ρ/Δ schedule are solver-internal scratch the solver carries — the same
//! split Levenberg-Marquardt uses for its μ/ν/diag working state. This is why
//! [`NewuoaState`](crate::NewuoaState) is generic over the parameter vector `V`
//! only, not the backend matrix `M`: NEWUOA's model algebra is internal `Vec<F>`
//! scratch and needs no `linalg`-tier ops from `V`.
//!
//! # Termination
//!
//! NEWUOA's natural convergence is `ρ` reaching `ρ_end` (configured on the
//! solver, where it also drives the eq-7.6 schedule); the solver signals it via
//! [`TerminationReason::SolverConverged`]. Add
//! [`MaxCostEvals`](crate::MaxCostEvals) to cap the evaluation budget, or
//! [`RhoTolerance`](crate::RhoTolerance) to stop early at a coarser `ρ`.
//!
//! # Backends
//!
//! Backend-generic over the parameter vector: `Vec<f64>`, nalgebra, ndarray, and
//! faer all work (the parameter type needs only element access and length —
//! [`Clone`], [`VectorLen`](crate::core::math::VectorLen), and indexing). wasm-
//! clean: the model algebra is pure-Rust `Vec<F>` with no BLAS/LAPACK.
//!
//! [`CostFunction`]: crate::core::problem::CostFunction
//! [`TerminationReason::SolverConverged`]: crate::TerminationReason::SolverConverged

pub(crate) mod bigden;
pub(crate) mod biglag;
pub(crate) mod driver;
pub(crate) mod init;
pub(crate) mod trsapp;

#[cfg(test)]
mod parity;

use crate::core::math::{Scalar, VectorLen};
use crate::core::problem::{CostFunction, Problem};
use crate::core::solver::Solver;
use crate::core::state::NewuoaState;
use crate::core::termination::TerminationReason;

use driver::{NewuoaWork, Transition};

/// NEWUOA (Powell 2006): model-based derivative-free trust-region optimization.
///
/// Configure the trust-region radii and interpolation-set size, then drive it
/// with an [`Executor`](crate::Executor) over a [`NewuoaState`]:
///
/// ```
/// use basin::{CostFunction, Executor, Newuoa, NewuoaState, MaxCostEvals};
///
/// struct Quadratic;
/// impl CostFunction for Quadratic {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
///         Ok((x[0] - 1.0).powi(2) + 2.0 * (x[1] + 2.0).powi(2))
///     }
/// }
///
/// let solver = Newuoa::new().with_rho_beg(0.5).with_rho_end(1e-8);
/// let state = NewuoaState::new(vec![0.0, 0.0]);
/// let result = Executor::new(Quadratic, solver, state)
///     .terminate_on(MaxCostEvals(500))
///     .run()
///     .unwrap();
/// assert!(result.best_cost() < 1e-10);
/// ```
///
/// # Configuration
///
/// - [`with_rho_beg`](Self::with_rho_beg) — initial trust-region radius `ρ_beg`
///   (a reasonable initial change to the variables; default `1.0`). Also the
///   initial `Δ`.
/// - [`with_rho_end`](Self::with_rho_end) — final radius `ρ_end`, the required
///   accuracy in the variables (default `1e-6`). The run converges once `ρ`
///   reaches it; `ρ_end` also drives the eq-7.6 schedule, so it must satisfy
///   `ρ_beg > ρ_end > 0`.
/// - [`with_npt`](Self::with_npt) — interpolation-set size `npt`, in
///   `[n+2, ½(n+1)(n+2)]` (default `2n+1`, Powell's recommendation).
///
/// # Backends
///
/// Backend-generic over the parameter vector: `Vec<f64>`, nalgebra, ndarray, and
/// faer all work. The model algebra is internal pure-Rust `Vec<f64>` scratch, so
/// the parameter type needs only [`Clone`], [`VectorLen`], and `Index`/`IndexMut`
/// element access — never any `linalg`-tier matrix op. wasm-clean (no
/// BLAS/LAPACK).
///
/// # References
///
/// M. J. D. Powell, *The NEWUOA software for unconstrained optimization without
/// derivatives*, in Large-Scale Nonlinear Optimization (2006), pp. 255–297.
/// Cross-validated against [PRIMA](https://github.com/libprima/prima) v0.7.2.
pub struct Newuoa<F = f64> {
    rho_beg: F,
    rho_end: F,
    npt: Option<usize>,
    /// Built in [`Solver::init`]; the resumable model + ρ/Δ schedule.
    work: Option<NewuoaWork<F>>,
}

impl<F: Scalar> Newuoa<F> {
    /// A NEWUOA solver with the default schedule (`ρ_beg = 1`, `ρ_end = 1e-6`,
    /// `npt = 2n+1`). Tune with the `with_*` builders.
    pub fn new() -> Self {
        Self {
            rho_beg: F::from_f64(1.0).expect("1.0 representable"),
            rho_end: F::from_f64(1e-6).expect("1e-6 representable"),
            npt: None,
            work: None,
        }
    }

    /// Set the initial trust-region radius `ρ_beg` (also the initial `Δ`) — a
    /// reasonable initial change to the variables.
    pub fn with_rho_beg(mut self, rho_beg: F) -> Self {
        self.rho_beg = rho_beg;
        self
    }

    /// Set the final trust-region radius `ρ_end` — the required accuracy in the
    /// variables. Must satisfy `ρ_beg > ρ_end > 0`.
    pub fn with_rho_end(mut self, rho_end: F) -> Self {
        self.rho_end = rho_end;
        self
    }

    /// Set the interpolation-set size `npt`, in `[n+2, ½(n+1)(n+2)]`. Defaults to
    /// `2n+1` (Powell's recommendation) when unset.
    pub fn with_npt(mut self, npt: usize) -> Self {
        self.npt = Some(npt);
        self
    }
}

impl<F: Scalar> Default for Newuoa<F> {
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

impl<P, V, F> Solver<P, NewuoaState<V, F>> for Newuoa<F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>,
    V: Clone
        + VectorLen
        + std::ops::Index<usize, Output = F>
        + std::ops::IndexMut<usize, Output = F>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: NewuoaState<V, F>,
    ) -> Result<NewuoaState<V, F>, Self::Error> {
        let n = state.param.vec_len();
        assert!(n >= 1, "Newuoa requires a non-empty start point");
        let npt = self.npt.unwrap_or(2 * n + 1);

        let x0: Vec<F> = (0..n).map(|i| state.param[i]).collect();
        let template = state.param.clone();

        let (work, best_x, best_f) = {
            let mut eval =
                |slice: &[F]| -> Result<F, P::Error> { problem.cost(&fill_from(&template, slice)) };
            NewuoaWork::try_init(x0, self.rho_beg, self.rho_end, npt, &mut eval)?
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
        mut state: NewuoaState<V, F>,
    ) -> Result<(NewuoaState<V, F>, Option<TerminationReason>), Self::Error> {
        let template = state.param.clone();
        let work = self
            .work
            .as_mut()
            .expect("Newuoa::init must run before next_iter");

        let out = {
            let mut eval =
                |slice: &[F]| -> Result<F, P::Error> { problem.cost(&fill_from(&template, slice)) };
            work.step(&mut eval)?
        };
        state.rho = work.rho();

        // Fold the step's evaluations into the running best (the reported
        // iterate). NEWUOA's reported point is the least-F seen, so param/cost
        // are monotone and coincide with best_param/best_cost.
        let mut best_f = state.cost.expect("Newuoa::init seeds the cost");
        for (xabs, f_new) in &out.evaluated {
            if *f_new < best_f {
                best_f = *f_new;
                state.param = fill_from(&template, xabs);
                state.cost = Some(best_f);
            }
        }

        // ρ reaching ρ_end (with the work there complete) is NEWUOA's natural
        // convergence; signal it as a clean stop.
        let reason = match out.transition {
            Transition::Converged => Some(TerminationReason::SolverConverged),
            Transition::Continue | Transition::RhoReduced => None,
        };
        Ok((state, reason))
    }
}
