// Index-based loops and many-arg helpers mirror the Fortran source for
// parity. Both lints are blanket-allowed for this module.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

//! Limited-memory BFGS — both modes.
//!
//! [`Lbfgs`](crate::Lbfgs) is generic over a type-state
//! [`Mode`](crate::solver::lbfgs::Bounded) marker:
//!
//! - [`Lbfgs<Bounded>`](crate::Lbfgs) is a faithful port of Nocedal–Zhu's L-BFGS-B
//!   v3.0 Fortran source (`references/lbfgsb-v3.0/`). The Fortran
//!   subroutines map to submodules below, and the top-level solver
//!   mirrors the `mainlb` iteration loop (with the goto-style
//!   coroutine flattened to a Rust `loop`). `Lbfgsb` is the type
//!   alias for this mode.
//! - [`Lbfgs<Unbounded>`](crate::Lbfgs) is unconstrained limited-memory BFGS via
//!   Nocedal–Wright's two-loop recursion (Algorithm 7.4). Uses the
//!   same [`LbfgsState`](crate::LbfgsState) history fields (`ws`, `wy`, `sy`, `theta`)
//!   but skips the Cauchy / `freev` / `subsm` machinery — those are
//!   box-constraint-specific.
//!
//! Submodules backing the bounded path (all `pub(crate)`):
//!
//! - `cauchy` — generalized Cauchy point along the projected gradient
//!   path. Port of `cauchy.f`.
//! - `subsm` — subspace minimization with `iword == 1` bound-
//!   backtracking (v3.0 deviation). Port of `subsm.f`.
//! - `formk` — `L·E·Lᵀ` factorization of the indefinite middle
//!   matrix `K`. Port of `formk.f`.
//! - `compact` — compact-form helpers (`formt`, `bmv`, pure-Rust
//!   Cholesky and triangular solves).
//! - `backend` — the `AsFloatSliceMut` trait that lets
//!   the slice-based numerics work generically over `Vec<f64>`,
//!   `nalgebra::DVector<f64>`, `faer::Col<f64>`, and
//!   `ndarray::Array1<f64>`.

pub(crate) mod backend;
pub(crate) mod cauchy;
pub(crate) mod compact;
pub(crate) mod formk;
pub(crate) mod subsm;

use core::marker::PhantomData;

use crate::core::constraint::BoxConstraints;
use crate::core::math::{Dot, Scalar, ScaledAdd};
use crate::core::problem::{CostFunction, Gradient, Problem};
use crate::core::solver::Solver;
use crate::core::state::lbfgs::{LbfgsState, LbfgsbWork};
use crate::core::termination::TerminationReason;
use crate::line_search::{LineSearch, MoreThuente};

use self::backend::{AsFloatSlice, AsFloatSliceMut};
use self::cauchy::{cauchy, iwhere as iwh};
use self::compact::{bmv, formt};
use self::formk::formk;
use self::subsm::subsm;

/// Limited-memory BFGS, parameterised over a type-state mode marker.
///
/// `Lbfgs<Bounded>` (aliased as [`Lbfgsb`]) is a faithful port of
/// Byrd–Lu–Nocedal 1995 / Zhu–Byrd–Lu–Nocedal 1997 (ACM TOMS Alg. 778),
/// with the Nocedal–Morales 2011 v3.0 directional-derivative + bound-
/// backtracking deviation in subspace minimization. Iteration-wise
/// parity with the Fortran v3.0 reference (`references/lbfgsb-v3.0/`)
/// is verified by `tests/lbfgsb_iter_parity.rs`.
///
/// `Lbfgs<Unbounded>` is unconstrained limited-memory BFGS via
/// Nocedal–Wright's two-loop recursion (Algorithm 7.4). It reuses the
/// same [`LbfgsState`] history machinery but skips the Cauchy /
/// `freev` / `subsm` phases — those are box-constraint-specific.
/// Construct with [`Lbfgs::<Unbounded>::new()`], or transition from
/// the default [`Lbfgs::<Bounded>::new()`] via [`Lbfgs::unbounded`].
///
/// # Bounded mode — per-iteration outline
///
/// 1. Walks the projected gradient ray, building a piecewise-quadratic
///    model and identifying the **generalized Cauchy point** `xcp` —
///    the minimizer along the path (see the `cauchy` submodule).
/// 2. Restricts to the free variables at `xcp` and computes an
///    approximate **subspace minimizer** via a structured Newton step
///    against the limited-memory compact-form Hessian (see the `subsm` submodule).
///    If the projected Newton step is infeasible, a uniform-α
///    bound-backtracking branch fires.
/// 3. Performs a **Moré–Thuente line search** along `d = z − x`,
///    safeguarded so the step is feasible.
/// 4. Accepts the step, updates the limited-memory `(s, y)` history
///    when the curvature condition holds, and rebuilds the
///    compact-form middle matrix `T` via Cholesky factorization
///    (see `compact::formt`).
///
/// On a singular middle-matrix or non-positive-definite `T`, the
/// solver clears the history and retries the iteration (Fortran's
/// `goto 222` reset path). One retry is enough — after clearing,
/// `col = 0` falls through to the line-search-only path, which
/// either succeeds with the projected steepest-descent direction or
/// fails the whole solve.
///
/// # Unbounded mode — per-iteration outline
///
/// 1. Two-loop recursion (Nocedal–Wright Alg. 7.4) over the
///    `(s_i, y_i)` history with initial Hessian `H₀ = (1/θ)·I`
///    (`θ = (y_last·y_last) / (s_last·y_last)` after the first
///    accepted update) to produce `d = −H_k · ∇f`.
/// 2. Moré–Thuente line search along `d`.
/// 3. Curvature-conditioned limited-memory update (same as bounded).
///
/// # Memory parameter
///
/// The history capacity `m` lives on [`LbfgsState`]:
/// `LbfgsState::new(x0, m)`. Fortran recommends `m ∈ [3, 20]`;
/// `m = 10` is a reasonable default.
///
/// # Termination
///
/// No solver-internal optimality test on the unbounded path — pair with
/// the framework-level
/// [`GradientTolerance`](crate::core::termination::GradientTolerance).
/// On the bounded path, a built-in projected-gradient check fires
/// when `‖projgr(x, g, l, u)‖_∞ ≤ tol_pg` (Fortran-`pgtol` parity);
/// the framework-level
/// [`ProjectedGradientTolerance`](crate::core::termination::ProjectedGradientTolerance)
/// is the canonical companion when external bookkeeping wants the same
/// metric. Pair either mode with
/// [`MaxIter`](crate::core::termination::MaxIter),
/// [`MaxCostEvals`](crate::core::termination::MaxCostEvals), and
/// [`CostTolerance`](crate::core::termination::CostTolerance) as
/// desired.
///
/// # Backends
///
/// Generic over any parameter type implementing
/// `AsFloatSliceMut<F>` + [`Clone`] + [`Dot<F>`] +
/// [`ScaledAdd<F>`]. Built-in impls cover `Vec<F>`,
/// `nalgebra::DVector<F>` (feature `nalgebra`), `faer::Col<F>`
/// (feature `faer`), and `ndarray::Array1<F>` (feature `ndarray`).
/// Other backends can implement the trait if their storage is
/// contiguous.
///
/// # Examples
///
/// See [`Bfgs`](crate::Bfgs) for the quasi-Newton `Executor` pattern.
/// L-BFGS iterates an `LbfgsState` sized to the history length `m`
/// (`LbfgsState::new(x0, m)`); construct the solver with `Lbfgs::new()`
/// (L-BFGS-B, the default) or `Lbfgs::<Unbounded>::new()`.
pub struct Lbfgs<Mode = Bounded, S = MoreThuente, F = f64> {
    line_search: S,
    /// Fortran `dr ≤ epsmch · ddum` curvature-skip threshold
    /// (`lbfgsb.f:875`). Defaults to `F::epsilon()`. Consulted
    /// in both modes for the limited-memory update acceptance test.
    epsilon: F,
    /// Built-in projected-gradient convergence tolerance. Bounded mode
    /// only — emits [`TerminationReason::SolverConverged`] at the top
    /// of an iteration when `‖projgr(x, g, l, u)‖_∞ ≤ tol_pg`. Default
    /// `1e-10`. Set to `0.0` to disable (matches Fortran `pgtol = 0`
    /// — required for the iteration-wise parity test against the
    /// reference, which doesn't terminate on the projected gradient).
    /// Stored on the shared struct; the field is unused (and the
    /// builder unavailable) in [`Unbounded`] mode, where users wire
    /// the framework-level
    /// [`GradientTolerance`](crate::core::termination::GradientTolerance)
    /// instead.
    tol_pg: F,
    /// Default limited-memory history capacity (Fortran `m`,
    /// `references/lbfgsb-v3.0/`). Default `10`. Only consulted when
    /// the solver constructs the state itself — e.g. as a
    /// [`MemeticInner`](crate::solver::MemeticInner) seeding a fresh
    /// [`LbfgsState`] for a CMA-ES injection refinement. Standalone
    /// users supply `m_capacity` directly to `LbfgsState::new(x, m)`,
    /// in which case `init` reads it off the state and this field is
    /// unused.
    ///
    /// `pub(crate)` so the `MemeticInner` impl in
    /// `solver/cma_inject.rs` can read it.
    pub(crate) m_capacity: usize,
    /// Type-state marker; carries the mode at the type level only.
    _mode: PhantomData<fn() -> Mode>,
}

/// Type-state marker for box-constrained L-BFGS-B (the default).
/// Constructors live on [`Lbfgs<Bounded, MoreThuente>`]; the
/// [`Solver`] impl requires `P: BoxConstraints` and the full
/// `AsFloatSliceMut<F>` + [`Dot<F>`] + [`ScaledAdd<F>`] backend.
/// [`Lbfgsb`] is the canonical type alias for this mode.
pub struct Bounded;

/// Type-state marker for unconstrained L-BFGS. Constructors live on
/// [`Lbfgs<Unbounded, MoreThuente>`]; the [`Solver`] impl has the same
/// backend bounds as [`Bounded`] but **no** [`BoxConstraints`]
/// requirement. The algorithm is Nocedal–Wright's two-loop recursion
/// over the [`LbfgsState`] history with `H₀ = (1/θ)·I`.
pub struct Unbounded;

/// Canonical name for the bounded-mode L-BFGS-B solver. Equivalent to
/// [`Lbfgs<Bounded, S>`]; call sites can use either name interchangeably.
pub type Lbfgsb<S = MoreThuente> = Lbfgs<Bounded, S>;

impl Default for Lbfgs<Bounded, MoreThuente> {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for Lbfgs<Unbounded, MoreThuente> {
    fn default() -> Self {
        Self::new()
    }
}

impl Lbfgs<Bounded, MoreThuente> {
    /// L-BFGS-B with Moré–Thuente line search and Fortran v3.0
    /// defaults (`ftol = 1e-3`, `gtol = 0.9`, `xtol = 0.1`). Built-in
    /// projected-gradient tolerance is `1e-10`; the seed history
    /// capacity (used only when the solver constructs the state, e.g.
    /// for memetic injection) is `10`.
    pub fn new() -> Self {
        Self {
            line_search: MoreThuente::new(),
            epsilon: f64::EPSILON,
            tol_pg: 1e-10,
            m_capacity: 10,
            _mode: PhantomData,
        }
    }
}

impl Lbfgs<Unbounded, MoreThuente> {
    /// Unconstrained L-BFGS with Moré–Thuente line search and the same
    /// curvature-skip / history defaults as the bounded path. The
    /// `tol_pg` field is unused in this mode — terminate via the
    /// framework-level
    /// [`GradientTolerance`](crate::core::termination::GradientTolerance).
    pub fn new() -> Self {
        Self {
            line_search: MoreThuente::new(),
            epsilon: f64::EPSILON,
            tol_pg: 1e-10,
            m_capacity: 10,
            _mode: PhantomData,
        }
    }
}

impl<S, F: Scalar> Lbfgs<Bounded, S, F> {
    /// L-BFGS-B with an explicit line-search strategy. Note: using
    /// anything other than [`MoreThuente`] forfeits iteration-wise
    /// parity with the Fortran reference. The curvature-skip threshold
    /// defaults to `F::epsilon()` (matching `f64::EPSILON` when
    /// `F = f64`); `tol_pg` defaults to `1e-10`.
    pub fn with_line_search(line_search: S) -> Self {
        Self {
            line_search,
            epsilon: F::epsilon(),
            tol_pg: F::from_f64(1e-10).unwrap(),
            m_capacity: 10,
            _mode: PhantomData,
        }
    }

    /// Override the built-in projected-gradient convergence tolerance.
    /// Default `1e-10`; pass `0.0` to disable (Fortran-`pgtol=0`
    /// semantics, used by the iteration-wise parity test). Bounded
    /// mode only — the unbounded path doesn't compute a projected
    /// gradient.
    pub fn tol_pg(mut self, tol_pg: F) -> Self {
        assert!(tol_pg >= F::zero(), "tol_pg must be ≥ 0");
        self.tol_pg = tol_pg;
        self
    }

    /// Switch to the unconstrained [`Unbounded`] mode while preserving
    /// the configured line search, curvature threshold, and history
    /// capacity. Mirrors [`NelderMead::projected`](crate::solver::NelderMead::projected)'s
    /// type-state transition.
    pub fn unbounded(self) -> Lbfgs<Unbounded, S, F> {
        Lbfgs {
            line_search: self.line_search,
            epsilon: self.epsilon,
            tol_pg: self.tol_pg,
            m_capacity: self.m_capacity,
            _mode: PhantomData,
        }
    }
}

impl<S, F: Scalar> Lbfgs<Unbounded, S, F> {
    /// Unconstrained L-BFGS with an explicit line-search strategy. The
    /// curvature-skip threshold defaults to `F::epsilon()` (matching
    /// `f64::EPSILON` when `F = f64`).
    pub fn with_line_search(line_search: S) -> Self {
        Self {
            line_search,
            epsilon: F::epsilon(),
            tol_pg: F::from_f64(1e-10).unwrap(),
            m_capacity: 10,
            _mode: PhantomData,
        }
    }

    /// Switch to box-constrained [`Bounded`] mode while preserving the
    /// configured line search, curvature threshold, and history
    /// capacity. The resulting solver requires the problem to
    /// implement [`BoxConstraints`].
    pub fn bounded(self) -> Lbfgs<Bounded, S, F> {
        Lbfgs {
            line_search: self.line_search,
            epsilon: self.epsilon,
            tol_pg: self.tol_pg,
            m_capacity: self.m_capacity,
            _mode: PhantomData,
        }
    }
}

impl<Mode, S, F: Scalar> Lbfgs<Mode, S, F> {
    /// Override the curvature-skip threshold. Default `F::epsilon()`
    /// (= `f64::EPSILON` when `F = f64`), matching Fortran's
    /// `dr ≤ epsmch · ddum` test.
    pub fn epsilon(mut self, epsilon: F) -> Self {
        assert!(epsilon >= F::zero(), "epsilon must be ≥ 0");
        self.epsilon = epsilon;
        self
    }

    /// Override the default limited-memory history capacity used when
    /// the solver constructs its own [`LbfgsState`] (memetic seeding).
    /// Standalone usage that hands in a state via `LbfgsState::new(x, m)`
    /// is unaffected. Default `10`; Nocedal recommends `[3, 20]`.
    ///
    /// # Panics
    ///
    /// Panics if `m_capacity == 0`.
    pub fn m_capacity(mut self, m_capacity: usize) -> Self {
        assert!(m_capacity >= 1, "m_capacity must be ≥ 1");
        self.m_capacity = m_capacity;
        self
    }
}

impl<P, V, S, F> Solver<P, LbfgsState<V, F>> for Lbfgs<Bounded, S, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V> + BoxConstraints,
    V: AsFloatSliceMut<F> + Clone + Dot<F> + ScaledAdd<F>,
    S: LineSearch<P, V, F, Error = P::Error>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: LbfgsState<V, F>,
    ) -> Result<LbfgsState<V, F>, Self::Error> {
        let n = state.param.as_float_slice().len();
        let m = state.m_capacity;
        let mut work = LbfgsbWork::<F>::new(n, m);

        // Project the initial iterate onto the feasible box and
        // initialise `iwhere`, `cnstnd`, `boxed` (Fortran `active`,
        // `lbfgsb.f:1004`).
        active_init(
            state.param.as_float_slice_mut(),
            problem.inner().lower().as_float_slice(),
            problem.inner().upper().as_float_slice(),
            &mut work.iwhere,
            &mut work.cnstnd,
            &mut work.boxed,
        );

        let (cost, grad) = problem.cost_and_gradient(&state.param)?;
        state.cost = Some(cost);
        state.gradient = Some(grad);
        state.work = Some(work);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: LbfgsState<V, F>,
    ) -> Result<(LbfgsState<V, F>, Option<TerminationReason>), Self::Error> {
        // Take the gradient and cost cached at the current `param`;
        // restore them on early exits.
        let g_v = state
            .gradient
            .take()
            .expect("gradient not set: Solver::init must run before next_iter");
        let f_old = state
            .cost
            .expect("cost not set: Solver::init must run before next_iter");

        let n = state.param.as_float_slice().len();
        let m = state.m_capacity;

        // Inner restart loop — Fortran's `goto 222` path. At most one
        // restart per iteration: after clearing history we either
        // succeed with the (col == 0) line-search-only path or bail.
        let mut restart_budget = 1u8;

        loop {
            let work = state.work.as_mut().expect("work missing");

            // -------------------------------------------------------
            // Phase A — projected gradient norm. Drives both the
            // built-in convergence check below and the cauchy
            // short-circuit further down. The framework-side
            // `ProjectedGradientTolerance` criterion does the same
            // calculation, but checking it inline here lets a memetic
            // wrapper (CMA-ES injection, `BoundedCmaInject`) skip
            // having to register an external criterion against bounds
            // it can't see at solver-build time.
            // -------------------------------------------------------
            let sbgnrm = projected_gradient_norm(
                state.param.as_float_slice(),
                g_v.as_float_slice(),
                problem.inner().lower().as_float_slice(),
                problem.inner().upper().as_float_slice(),
            );

            // Built-in convergence: emit `SolverConverged` when the
            // projected-gradient infinity-norm sits at the tolerance.
            // Restore the borrowed cost/gradient so callers reading
            // `state.gradient()` / `state.cost()` on the final result
            // see the values at the converged iterate. Set
            // `tol_pg = 0.0` (Fortran `pgtol = 0`) to disable.
            if sbgnrm <= self.tol_pg {
                state.gradient = Some(g_v);
                state.cost = Some(f_old);
                return Ok((state, Some(TerminationReason::SolverConverged)));
            }

            let col = state.ws.len();
            let theta = state.theta;
            let cnstnd = work.cnstnd;
            let boxed = work.boxed;
            let updatd = work.updatd;

            // -------------------------------------------------------
            // Phase B — generalized Cauchy point (or skip when no
            // bounds are active and we already have history).
            // -------------------------------------------------------
            let mut wrk = updatd;
            if !cnstnd && col > 0 {
                // Unbounded with history: skip GCP, set z := x.
                work.z.copy_from_slice(state.param.as_float_slice());
            } else {
                let ws_cols: Vec<&[F]> = state.ws.iter().map(|v| v.as_float_slice()).collect();
                let wy_cols: Vec<&[F]> = state.wy.iter().map(|v| v.as_float_slice()).collect();
                let cauchy_res = cauchy(
                    state.param.as_float_slice(),
                    problem.inner().lower().as_float_slice(),
                    problem.inner().upper().as_float_slice(),
                    g_v.as_float_slice(),
                    &ws_cols,
                    &wy_cols,
                    &state.sy,
                    &work.wt,
                    m,
                    theta,
                    sbgnrm,
                    &mut work.z,
                    &mut work.d,
                    &mut work.t_buf,
                    &mut work.iwhere,
                    &mut work.indx2,
                    &mut work.wa_p,
                    &mut work.wa_c,
                    &mut work.wa_wbp,
                    &mut work.wa_v,
                );
                if cauchy_res.is_err() {
                    if try_restart(&mut state, &g_v, f_old, &mut restart_budget) {
                        continue;
                    } else {
                        return Ok((state, Some(TerminationReason::SolverFailed)));
                    }
                }
            }

            // -------------------------------------------------------
            // Phase C — free / active partition (Fortran `freev`).
            // -------------------------------------------------------
            let (nfree, nenter, ileave) = freev(
                n,
                &work.iwhere,
                &mut work.index,
                &mut work.indx2,
                work.nfree,
                state.iter,
                cnstnd,
            );
            work.nfree = nfree;
            let wrk_local = (ileave < n) || (nenter > 0) || wrk;
            wrk = wrk_local;

            // -------------------------------------------------------
            // Phase D — subspace minimization (when there are free
            // variables and history to use).
            // -------------------------------------------------------
            if nfree > 0 && col > 0 {
                if wrk {
                    let ws_cols: Vec<&[F]> = state.ws.iter().map(|v| v.as_float_slice()).collect();
                    let wy_cols: Vec<&[F]> = state.wy.iter().map(|v| v.as_float_slice()).collect();
                    if formk(
                        &mut work.wn,
                        &mut work.wn1,
                        m,
                        col,
                        theta,
                        &state.sy,
                        &ws_cols,
                        &wy_cols,
                        nfree,
                        &work.index,
                        nenter,
                        ileave,
                        &work.indx2,
                        work.iupdat,
                        updatd,
                    )
                    .is_err()
                    {
                        if try_restart(&mut state, &g_v, f_old, &mut restart_budget) {
                            continue;
                        } else {
                            return Ok((state, Some(TerminationReason::SolverFailed)));
                        }
                    }
                }

                // cmprlb: r = −Z'B(z − x) − Z'g, indexed by the free
                // subspace.
                if cmprlb(
                    state.param.as_float_slice(),
                    g_v.as_float_slice(),
                    &work.z,
                    &mut work.r,
                    &mut work.wa_c,
                    &mut work.wa_p,
                    &state.sy,
                    &work.wt,
                    &state.ws,
                    &state.wy,
                    &work.index,
                    nfree,
                    cnstnd,
                    col,
                    theta,
                    m,
                )
                .is_err()
                {
                    if try_restart(&mut state, &g_v, f_old, &mut restart_budget) {
                        continue;
                    } else {
                        return Ok((state, Some(TerminationReason::SolverFailed)));
                    }
                }

                // subsm: writes the subspace minimizer into `z`.
                let ws_cols: Vec<&[F]> = state.ws.iter().map(|v| v.as_float_slice()).collect();
                let wy_cols: Vec<&[F]> = state.wy.iter().map(|v| v.as_float_slice()).collect();
                let subsm_res = subsm(
                    &mut work.z,
                    &mut work.r,
                    &mut work.xp,
                    state.param.as_float_slice(),
                    g_v.as_float_slice(),
                    &work.index[0..nfree],
                    problem.inner().lower().as_float_slice(),
                    problem.inner().upper().as_float_slice(),
                    &ws_cols,
                    &wy_cols,
                    &work.wn,
                    &mut work.wa_v,
                    m,
                    col,
                    theta,
                );
                if subsm_res.is_err() {
                    if try_restart(&mut state, &g_v, f_old, &mut restart_budget) {
                        continue;
                    } else {
                        return Ok((state, Some(TerminationReason::SolverFailed)));
                    }
                }
                // We don't read `SubsmStatus` — the projected step
                // is already applied to `work.z`, and the
                // line-search step cap below re-enforces feasibility.
            }

            // -------------------------------------------------------
            // Phase E — line search. Direction d = z − x.
            // -------------------------------------------------------
            for i in 0..n {
                work.d[i] = work.z[i] - state.param.as_float_slice()[i];
            }
            let dtd: F = work.d.iter().fold(F::zero(), |acc, x| acc + (*x) * (*x));
            let dnorm = dtd.sqrt();
            work.dnorm = dnorm;

            // Maximum feasible step (Fortran lnsrlb `stpmx`).
            let stpmx = if cnstnd {
                if state.iter == 0 {
                    F::one()
                } else {
                    feasible_step_cap(
                        state.param.as_float_slice(),
                        problem.inner().lower().as_float_slice(),
                        problem.inner().upper().as_float_slice(),
                        &work.d,
                    )
                }
            } else {
                F::from_f64(1.0e10).unwrap()
            };

            let alpha_init = if state.iter == 0 && !boxed {
                (F::one() / dnorm).min(stpmx)
            } else {
                F::one()
            };

            // Build a V-typed direction for the line search.
            let mut d_v = state.param.clone();
            d_v.as_float_slice_mut().copy_from_slice(&work.d);

            // Save previous iterate before stepping (Fortran `t = x;
            // r = g`).
            work.t_buf.copy_from_slice(state.param.as_float_slice());
            work.r.copy_from_slice(g_v.as_float_slice());
            work.gdold = work
                .d
                .iter()
                .zip(g_v.as_float_slice())
                .fold(F::zero(), |acc, (a, b)| acc + (*a) * (*b));

            // Drive the line search. Fortran `lnsrlb` sets the
            // initial trial step and the feasibility cap on
            // `dcsrch`'s `stp` / `stpmax`; we don't have a generic
            // hook for that on [`LineSearch`], so for now we let the
            // configured line search keep its own initial / max-step
            // settings. Parity holds on the Rosenbrock 5D fixture
            // because the natural Newton step stays interior, but
            // tight-bound problems may need a constraint-aware
            // wrapper down the road.
            let _ = (alpha_init, stpmx);
            let stp = self
                .line_search
                .next(problem, &state.param, f_old, &g_v, &d_v)?;

            if !(stp.is_finite() && stp > F::zero()) {
                // Line search bailed. If col == 0, abnormal
                // termination — there's no compact-form state to
                // reset. Otherwise restart with cleared history. The
                // clone of `g_v` is fine: we're on the cold-path exit
                // either way.
                state.gradient = Some(g_v.clone());
                state.cost = Some(f_old);
                if state.ws.is_empty() {
                    return Ok((state, Some(TerminationReason::SolverFailed)));
                }
                if try_restart_after_lnsrch(&mut state, &mut restart_budget) {
                    continue;
                } else {
                    return Ok((state, Some(TerminationReason::SolverFailed)));
                }
            }

            // Apply the step. x ← x + stp · d.
            state.param.scaled_add(stp, &d_v);

            // Recompute f, g at the new iterate. (MoreThuente
            // discards the final trial's values; cleanest workaround
            // is one fused cost+grad eval per iter.)
            let (f_new, g_new) = problem.cost_and_gradient(&state.param)?;

            // -------------------------------------------------------
            // Phase F — limited-memory update. Curvature check
            // matches Fortran's `dr ≤ epsmch · ddum`.
            // -------------------------------------------------------
            // s = stp · d  (in slice form, d holds the unscaled
            // direction; the s vector lives in `d` scaled by stp).
            // y = g_new − g_old.
            // dr = y · s, ddum = −gdold · stp (Fortran convention).
            let work = state.work.as_mut().unwrap();
            let g_new_slice = g_new.as_float_slice();
            let g_old_slice = work.r.as_slice(); // saved earlier

            let mut dr = F::zero();
            for i in 0..n {
                let yi = g_new_slice[i] - g_old_slice[i];
                let si = stp * work.d[i];
                dr = dr + yi * si;
            }
            let ddum = -work.gdold * stp;

            if dr > self.epsilon * ddum.abs() {
                // Accept the (s, y) pair. Build s, y as V then push.
                let mut s_v = state.param.clone();
                let s_slice = s_v.as_float_slice_mut();
                for i in 0..n {
                    s_slice[i] = stp * work.d[i];
                }
                let mut y_v = g_new.clone();
                let y_slice = y_v.as_float_slice_mut();
                for i in 0..n {
                    y_slice[i] = g_new_slice[i] - g_old_slice[i];
                }
                let appended = state.append_pair(s_v, y_v);
                if appended {
                    let work = state.work.as_mut().unwrap();
                    work.updatd = true;
                    work.iupdat = work.iupdat.saturating_add(1);

                    // Rebuild T = θ SᵀS + L D⁻¹ Lᵀ. On failure, reset.
                    let new_col = state.ws.len();
                    if formt(state.theta, &state.sy, &state.ss, new_col, m, &mut work.wt).is_err() {
                        // Reset history; the next iter starts fresh.
                        state.ws.clear();
                        state.wy.clear();
                        for v in state.sy.iter_mut() {
                            *v = F::zero();
                        }
                        for v in state.ss.iter_mut() {
                            *v = F::zero();
                        }
                        state.theta = F::one();
                        work.reset_history();
                    }
                } else {
                    // append_pair refused (s·y ≤ 0 numerically) —
                    // treat as a skipped update.
                    let work = state.work.as_mut().unwrap();
                    work.updatd = false;
                }
            } else {
                // Skip the update; matches Fortran's `nskip += 1` path.
                let work = state.work.as_mut().unwrap();
                work.updatd = false;
            }

            state.cost = Some(f_new);
            state.gradient = Some(g_new);
            return Ok((state, None));
        }
    }
}

impl<P, V, S, F> Solver<P, LbfgsState<V, F>> for Lbfgs<Unbounded, S, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V>,
    V: AsFloatSliceMut<F> + Clone + Dot<F> + ScaledAdd<F>,
    S: LineSearch<P, V, F, Error = P::Error>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: LbfgsState<V, F>,
    ) -> Result<LbfgsState<V, F>, Self::Error> {
        // Cache cost and gradient at the initial iterate. `state.work`
        // stays `None` — the box-constrained scratch buffers are
        // never touched on the unbounded path.
        let (cost, grad) = problem.cost_and_gradient(&state.param)?;
        state.cost = Some(cost);
        state.gradient = Some(grad);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: LbfgsState<V, F>,
    ) -> Result<(LbfgsState<V, F>, Option<TerminationReason>), Self::Error> {
        let g_v = state
            .gradient
            .take()
            .expect("gradient not set: Solver::init must run before next_iter");
        let f_old = state
            .cost
            .expect("cost not set: Solver::init must run before next_iter");

        let n = state.param.as_float_slice().len();
        let m = state.m_capacity;
        let col = state.ws.len();
        let theta = state.theta;

        // Nocedal–Wright two-loop recursion (Algorithm 7.4). Computes
        // d = −H_k · g into `d_v` via an in-place accumulator that
        // starts at `g`, has the BFGS history applied, scales by
        // `H₀ = (1/θ)·I`, then negates.
        let mut d_v = state.param.clone();
        {
            let d_slice = d_v.as_float_slice_mut();
            let g_slice = g_v.as_float_slice();
            // q ← g.
            d_slice.copy_from_slice(g_slice);

            if col > 0 {
                let mut alpha = vec![F::zero(); col];

                // Backward pass: q ← q − αᵢ yᵢ for i = col-1 .. 0.
                for i in (0..col).rev() {
                    let rho_i = F::one() / state.sy[i * m + i];
                    let s_i = state.ws[i].as_float_slice();
                    let y_i = state.wy[i].as_float_slice();
                    let mut s_dot_q = F::zero();
                    for k in 0..n {
                        s_dot_q = s_dot_q + s_i[k] * d_slice[k];
                    }
                    let a = rho_i * s_dot_q;
                    alpha[i] = a;
                    for k in 0..n {
                        d_slice[k] = d_slice[k] - a * y_i[k];
                    }
                }

                // r ← H₀ · q with H₀ = (1/θ)·I.
                let inv_theta = F::one() / theta;
                for k in 0..n {
                    d_slice[k] = d_slice[k] * inv_theta;
                }

                // Forward pass: r ← r + (αᵢ − β) sᵢ for i = 0 .. col-1.
                for i in 0..col {
                    let rho_i = F::one() / state.sy[i * m + i];
                    let s_i = state.ws[i].as_float_slice();
                    let y_i = state.wy[i].as_float_slice();
                    let mut y_dot_r = F::zero();
                    for k in 0..n {
                        y_dot_r = y_dot_r + y_i[k] * d_slice[k];
                    }
                    let beta = rho_i * y_dot_r;
                    let coef = alpha[i] - beta;
                    for k in 0..n {
                        d_slice[k] = d_slice[k] + coef * s_i[k];
                    }
                }
            }

            // d = −H · g.
            for k in 0..n {
                d_slice[k] = -d_slice[k];
            }
        }

        // gᵀd for the curvature-skip threshold (Fortran `gdold`).
        let gdold: F = {
            let g_slice = g_v.as_float_slice();
            let d_slice = d_v.as_float_slice();
            let mut acc = F::zero();
            for i in 0..n {
                acc = acc + g_slice[i] * d_slice[i];
            }
            acc
        };

        let stp = self
            .line_search
            .next(problem, &state.param, f_old, &g_v, &d_v)?;

        if !(stp.is_finite() && stp > F::zero()) {
            // Line search bailed. Restore cached cost / gradient so
            // the caller's final state is consistent with the last
            // accepted iterate, and bubble the failure.
            state.gradient = Some(g_v);
            state.cost = Some(f_old);
            return Ok((state, Some(TerminationReason::SolverFailed)));
        }

        // x ← x + stp · d.
        state.param.scaled_add(stp, &d_v);
        let (f_new, g_new) = problem.cost_and_gradient(&state.param)?;

        // Curvature-conditioned limited-memory update. Matches Fortran
        // `dr ≤ epsmch · |ddum|` with `dr = y·s` and `ddum = −gdold·stp`.
        let g_new_slice = g_new.as_float_slice();
        let g_old_slice = g_v.as_float_slice();
        let d_slice = d_v.as_float_slice();
        let mut dr = F::zero();
        for i in 0..n {
            let yi = g_new_slice[i] - g_old_slice[i];
            let si = stp * d_slice[i];
            dr = dr + yi * si;
        }
        let ddum = -gdold * stp;

        if dr > self.epsilon * ddum.abs() {
            // Build s = stp · d and y = g_new − g_old as V-typed
            // vectors, then push. `append_pair` re-runs the s·y > 0
            // check as a final safeguard and refreshes `theta`.
            let mut s_v = state.param.clone();
            {
                let s_slice = s_v.as_float_slice_mut();
                for i in 0..n {
                    s_slice[i] = stp * d_slice[i];
                }
            }
            let mut y_v = g_new.clone();
            {
                let y_slice = y_v.as_float_slice_mut();
                for i in 0..n {
                    y_slice[i] = g_new_slice[i] - g_old_slice[i];
                }
            }
            state.append_pair(s_v, y_v);
        }

        state.cost = Some(f_new);
        state.gradient = Some(g_new);
        Ok((state, None))
    }
}

/// Project a candidate `x` into `[l, u]` and initialize `iwhere`
/// per Fortran `active` (`lbfgsb.f:1004`).
fn active_init<F: Scalar>(
    x: &mut [F],
    l: &[F],
    u: &[F],
    iwhere: &mut [i8],
    cnstnd: &mut bool,
    boxed: &mut bool,
) {
    let n = x.len();
    *cnstnd = false;
    *boxed = true;
    for i in 0..n {
        let lo = l[i];
        let hi = u[i];
        let lower_finite = lo.is_finite();
        let upper_finite = hi.is_finite();
        if lower_finite && x[i] < lo {
            x[i] = lo;
        }
        if upper_finite && x[i] > hi {
            x[i] = hi;
        }
        if !(lower_finite && upper_finite) {
            *boxed = false;
        }
        if !lower_finite && !upper_finite {
            iwhere[i] = iwh::ALWAYS_FREE;
        } else {
            *cnstnd = true;
            if lower_finite && upper_finite && lo == hi {
                iwhere[i] = iwh::ALWAYS_FIXED;
            } else {
                iwhere[i] = iwh::FREE_MOVED;
            }
        }
    }
}

/// Infinity-norm of the projected gradient (Fortran `projgr`,
/// `lbfgsb.f:2942`).
fn projected_gradient_norm<F: Scalar>(x: &[F], g: &[F], l: &[F], u: &[F]) -> F {
    let zero = F::zero();
    let mut sbgnrm = zero;
    for i in 0..x.len() {
        let mut gi = g[i];
        let lower_finite = l[i].is_finite();
        let upper_finite = u[i].is_finite();
        if lower_finite || upper_finite {
            if gi < zero {
                if upper_finite {
                    gi = (x[i] - u[i]).max(gi);
                }
            } else if lower_finite {
                gi = (x[i] - l[i]).min(gi);
            }
        }
        let abs_gi = gi.abs();
        if abs_gi > sbgnrm {
            sbgnrm = abs_gi;
        }
    }
    sbgnrm
}

/// Largest feasible step `stpmx` along `d` such that `x + stpmx · d`
/// stays inside `[l, u]`. Fortran `lnsrlb` initial-step calculation
/// (`lbfgsb.f:2511-2530`).
fn feasible_step_cap<F: Scalar>(x: &[F], l: &[F], u: &[F], d: &[F]) -> F {
    let zero = F::zero();
    let mut stpmx = F::from_f64(1.0e10).unwrap();
    for i in 0..x.len() {
        let di = d[i];
        let lower_finite = l[i].is_finite();
        let upper_finite = u[i].is_finite();
        if di < zero && lower_finite {
            let gap = l[i] - x[i];
            if gap >= zero {
                stpmx = zero;
            } else if di * stpmx < gap {
                stpmx = gap / di;
            }
        } else if di > zero && upper_finite {
            let gap = u[i] - x[i];
            if gap <= zero {
                stpmx = zero;
            } else if di * stpmx > gap {
                stpmx = gap / di;
            }
        }
    }
    stpmx
}

/// Count entering / leaving variables and rebuild the free + active
/// partition in `index`. Port of Fortran `freev` (`lbfgsb.f:2241`).
///
/// Returns `(nfree, nenter, ileave)` where `indx2[0..nenter]` holds
/// the entering variables and `indx2[ileave..n]` the leaving ones.
fn freev(
    n: usize,
    iwhere: &[i8],
    index: &mut [usize],
    indx2: &mut [usize],
    prev_nfree: usize,
    iter: u64,
    cnstnd: bool,
) -> (usize, usize, usize) {
    let mut nenter = 0usize;
    let mut ileave = n;
    if iter > 0 && cnstnd {
        // Variables that were free, now active ⇒ leaving.
        for i in 0..prev_nfree {
            let k = index[i];
            if iwhere[k] > 0 {
                ileave -= 1;
                indx2[ileave] = k;
            }
        }
        // Variables that were active, now free ⇒ entering.
        for i in prev_nfree..n {
            let k = index[i];
            if iwhere[k] <= 0 {
                indx2[nenter] = k;
                nenter += 1;
            }
        }
    }
    // Rebuild free + active partition.
    let mut nfree = 0usize;
    let mut iact = n;
    for i in 0..n {
        if iwhere[i] <= 0 {
            index[nfree] = i;
            nfree += 1;
        } else {
            iact -= 1;
            index[iact] = i;
        }
    }
    (nfree, nenter, ileave)
}

/// Compute the reduced gradient `r = −Z'B(xcp − x) − Z'g` for the
/// free subspace (Fortran `cmprlb`, `lbfgsb.f:1720`). Writes
/// `r[0..nfree]`; consumes the compact-form Cauchy correction stored
/// in `wa_c` (= `W'(xcp − x)` from cauchy).
#[allow(clippy::too_many_arguments)]
fn cmprlb<V, F: Scalar>(
    x: &[F],
    g: &[F],
    z: &[F],
    r: &mut [F],
    wa_c: &mut [F],
    wa_p: &mut [F],
    sy: &[F],
    wt: &[F],
    ws: &[V],
    wy: &[V],
    index: &[usize],
    nfree: usize,
    cnstnd: bool,
    col: usize,
    theta: F,
    m: usize,
) -> Result<(), ()>
where
    V: AsFloatSlice<F>,
{
    if !cnstnd && col > 0 {
        for i in 0..x.len() {
            r[i] = -g[i];
        }
        return Ok(());
    }
    for i in 0..nfree {
        let k = index[i];
        r[i] = -theta * (z[k] - x[k]) - g[k];
    }
    // Apply M⁻¹ to `wa_c` → `wa_p`, then add the correction.
    bmv(sy, wt, col, m, wa_c, wa_p).map_err(|_| ())?;
    for j in 0..col {
        let a1 = wa_p[j];
        let a2 = theta * wa_p[col + j];
        let wy_j = wy[j].as_float_slice();
        let ws_j = ws[j].as_float_slice();
        for i in 0..nfree {
            let k = index[i];
            r[i] = r[i] + wy_j[k] * a1 + ws_j[k] * a2;
        }
    }
    Ok(())
}

/// Reset the limited-memory history of `state` and bail or continue
/// based on `restart_budget`. Returns `true` if a restart was budgeted
/// and the caller should `continue` the outer loop, `false` if budget
/// was exhausted.
fn try_restart<V, F: Scalar>(
    state: &mut LbfgsState<V, F>,
    g_v: &V,
    f_old: F,
    restart_budget: &mut u8,
) -> bool
where
    V: Clone,
{
    if *restart_budget == 0 {
        // Restore the cached gradient / cost so the state stays
        // consistent for the caller.
        state.gradient = Some(g_v.clone());
        state.cost = Some(f_old);
        return false;
    }
    *restart_budget -= 1;
    // Clear history & reset theta.
    state.ws.clear();
    state.wy.clear();
    for v in state.sy.iter_mut() {
        *v = F::zero();
    }
    for v in state.ss.iter_mut() {
        *v = F::zero();
    }
    state.theta = F::one();
    if let Some(work) = state.work.as_mut() {
        work.reset_history();
    }
    true
}

/// Same as `try_restart`, but used after the line search has already
/// applied side effects we need to leave intact (state.gradient / cost
/// already restored by the caller).
fn try_restart_after_lnsrch<V, F: Scalar>(
    state: &mut LbfgsState<V, F>,
    restart_budget: &mut u8,
) -> bool {
    if *restart_budget == 0 {
        return false;
    }
    *restart_budget -= 1;
    state.ws.clear();
    state.wy.clear();
    for v in state.sy.iter_mut() {
        *v = F::zero();
    }
    for v in state.ss.iter_mut() {
        *v = F::zero();
    }
    state.theta = F::one();
    if let Some(work) = state.work.as_mut() {
        work.reset_history();
    }
    true
}

/// Deprecated all-caps alias preserved for downstream code that imported
/// the original [`LBFGS`] name. Resolves to [`Lbfgs`]; will be removed in
/// a future release after the rename window closes.
#[deprecated(
    since = "0.8.0",
    note = "use `Lbfgs` (PascalCase) — the all-caps alias will be removed in a future release"
)]
#[allow(non_camel_case_types)]
pub type LBFGS<Mode = Bounded, S = MoreThuente> = Lbfgs<Mode, S>;

/// Deprecated all-caps alias preserved for downstream code that imported
/// the original [`LBFGSB`] name. Resolves to [`Lbfgsb`]; will be removed
/// in a future release after the rename window closes.
#[deprecated(
    since = "0.8.0",
    note = "use `Lbfgsb` (PascalCase) — the all-caps alias will be removed in a future release"
)]
#[allow(non_camel_case_types)]
pub type LBFGSB<S = MoreThuente> = Lbfgsb<S>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::executor::Executor;
    use crate::core::termination::{MaxIter, ProjectedGradientTolerance};

    /// Smoke test on a small box-constrained quadratic. The problem
    /// `f(x) = (x − c)ᵀ(x − c)` with bounds `[l, u]` has minimizer
    /// `clamp(c, l, u)`. Verifies the full pipeline (cauchy + subsm +
    /// formk + line search + matupd) returns the analytical answer.
    #[test]
    fn shifted_quadratic_in_box_converges_to_clamp() {
        use crate::core::constraint::BoxConstraints;
        use crate::core::problem::{CostFunction, Gradient};

        struct Quad {
            c: Vec<f64>,
            l: Vec<f64>,
            u: Vec<f64>,
        }
        impl CostFunction for Quad {
            type Param = Vec<f64>;
            type Output = f64;
            type Error = std::convert::Infallible;
            fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
                Ok(x.iter().zip(&self.c).map(|(a, b)| (a - b).powi(2)).sum())
            }
        }
        impl Gradient for Quad {
            type Gradient = Vec<f64>;
            fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
                Ok(x.iter().zip(&self.c).map(|(a, b)| 2.0 * (a - b)).collect())
            }
        }
        impl BoxConstraints for Quad {
            fn lower(&self) -> &Vec<f64> {
                &self.l
            }
            fn upper(&self) -> &Vec<f64> {
                &self.u
            }
        }

        // Unconstrained minimizer at (3, -1); both clipped by bounds.
        let problem = Quad {
            c: vec![3.0, -1.0],
            l: vec![0.0, 0.0],
            u: vec![2.0, 2.0],
        };

        let state = LbfgsState::new(vec![1.0, 1.0], 5);
        let solver = Lbfgsb::new();
        let lower = problem.lower().clone();
        let upper = problem.upper().clone();
        let result = Executor::new(problem, solver, state)
            .terminate_on(MaxIter(50))
            .terminate_on(ProjectedGradientTolerance::new(lower, upper, 1e-10))
            .run()
            .unwrap();
        let final_x = result.state.param.clone();
        // Optimum: clamp((3, -1), [0,0], [2, 2]) = (2, 0).
        assert!((final_x[0] - 2.0).abs() < 1e-6, "x0 = {}", final_x[0]);
        assert!(final_x[1].abs() < 1e-6, "x1 = {}", final_x[1]);
    }

    /// Unbounded `−∞ ≤ x ≤ +∞` problem: behaves as L-BFGS on
    /// Rosenbrock 2D. The compact-form path with `cnstnd == false`
    /// skips GCP entirely after the first iteration.
    #[test]
    fn unbounded_rosenbrock_2d_converges() {
        use crate::core::constraint::BoxConstraints;
        use crate::core::problem::{CostFunction, Gradient};

        struct Rosen {
            l: Vec<f64>,
            u: Vec<f64>,
        }
        impl CostFunction for Rosen {
            type Param = Vec<f64>;
            type Output = f64;
            type Error = std::convert::Infallible;
            fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
                Ok((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2))
            }
        }
        impl Gradient for Rosen {
            type Gradient = Vec<f64>;
            fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
                let dfdx0 = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
                let dfdx1 = 200.0 * (x[1] - x[0] * x[0]);
                Ok(vec![dfdx0, dfdx1])
            }
        }
        impl BoxConstraints for Rosen {
            fn lower(&self) -> &Vec<f64> {
                &self.l
            }
            fn upper(&self) -> &Vec<f64> {
                &self.u
            }
        }

        let problem = Rosen {
            l: vec![f64::NEG_INFINITY; 2],
            u: vec![f64::INFINITY; 2],
        };
        let state = LbfgsState::new(vec![-1.2, 1.0], 5);
        let solver = Lbfgsb::new();
        let lower = problem.lower().clone();
        let upper = problem.upper().clone();
        let result = Executor::new(problem, solver, state)
            .terminate_on(MaxIter(200))
            .terminate_on(ProjectedGradientTolerance::new(lower, upper, 1e-8))
            .run()
            .unwrap();
        let final_x = result.state.param.clone();
        assert!(
            (final_x[0] - 1.0).abs() < 1e-3 && (final_x[1] - 1.0).abs() < 1e-3,
            "x = {:?}",
            final_x
        );
    }
}
