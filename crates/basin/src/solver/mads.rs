// Index-based loops mirror the OrthoMADS exposition (Abramson et al. 2009) and
// the flat poll/direction arithmetic; the lint is blanket-allowed here.
#![allow(clippy::needless_range_loop)]

//! MADS (Audet & Dennis 2006): mesh adaptive direct search, the deterministic
//! OrthoMADS instance (Abramson, Audet, Dennis & Le Digabel 2009).
//!
//! [`Mads`] is a directional direct-search method for **nonsmooth or
//! non-continuous** objectives, where Nelder-Mead has no convergence theory and
//! is known to stagnate. Each iteration polls a positive spanning set of `2n`
//! directions on a shrinking mesh around the incumbent; an improving mesh point
//! coarsens the mesh, a failed poll refines it. The set of poll directions
//! becomes asymptotically dense, which gives MADS its Clarke-stationarity
//! convergence guarantee on nonsmooth functions. It binds [`CostFunction`] only;
//! no derivatives.
//!
//! # Architecture
//!
//! This implements the **OrthoMADS** instance: the poll directions are generated
//! deterministically (no RNG) from the quasi-random Halton sequence (`halton`)
//! and a scaled Householder reflection (`geometry`), producing an orthogonal
//! integer basis whose `±` columns form the maximal positive basis
//! `D_k = [H −H]`. Two parameters drive the mesh: a fine mesh size `Δᵐ` and a
//! coarser poll size `Δᵖ`, linked through the integer mesh index `ℓ` by
//! `Δᵖ = 2^{-ℓ}`, `Δᵐ = min{1, 4^{-ℓ}}` (OrthoMADS eq. (1)); `Δᵐ` shrinks faster
//! than `Δᵖ`, so the number of available poll directions grows without bound.
//! Because the directions are integer and trial points are `x_k + Δᵐ·d`, every
//! poll point lies exactly on the mesh. The poll loop and bookkeeping live in
//! the `driver` submodule.
//!
//! # State and solver split
//!
//! The incumbent and the poll size `Δᵖ` live on [`MadsState`](crate::MadsState);
//! the mesh index, Halton-index schedule, and flat-scratch incumbent live on the
//! solver (`Mads`'s `MadsWork`): the same split the Powell-family DFO solvers
//! use. So [`MadsState`](crate::MadsState) is generic over the parameter vector
//! `V` only; the direction algebra is internal `Vec<f64>`/`Vec<i64>` scratch.
//!
//! # Termination
//!
//! MADS's natural convergence is the poll size `Δᵖ` reaching the configured floor
//! `poll_size_min`; the solver signals it via
//! [`TerminationReason::SolverConverged`]. Add [`MaxCostEvals`](crate::MaxCostEvals)
//! to cap the evaluation budget, or [`MeshTolerance`](crate::MeshTolerance) to
//! stop early at a coarser poll size.
//!
//! # Backends
//!
//! Backend-generic over the parameter vector: `Vec<f64>`, nalgebra, ndarray, and
//! faer all work (the parameter type needs only [`Clone`], [`VectorLen`], and
//! `Index`/`IndexMut`). The poll geometry is pure-Rust integer/`f64` scratch with
//! no RNG, no `linalg`-tier ops, and no time: fully deterministic and wasm-clean.
//!
//! [`CostFunction`]: crate::core::problem::CostFunction
//! [`VectorLen`]: crate::core::math::VectorLen
//! [`TerminationReason::SolverConverged`]: crate::TerminationReason::SolverConverged

pub(crate) mod driver;
pub(crate) mod geometry;
pub(crate) mod halton;
pub(crate) mod progressive_barrier;

use std::marker::PhantomData;

use crate::core::constraint::{BoxConstraints, NonlinearInequalityConstraints};
use crate::core::inner::InitialState;
use crate::core::math::{Scalar, VectorLen};
use crate::core::problem::{CostFunction, Problem};
use crate::core::solver::Solver;
use crate::core::state::{ConstrainedMadsState, MadsState};
use crate::core::termination::TerminationReason;
use crate::solver::nelder_mead::Unbounded;

use driver::{MadsWork, Transition};
use progressive_barrier::PbWork;

/// Type-state marker: box-bound-constrained MADS (`Mads<Bounded>`), reached via
/// [`Mads::bounded`]. Bounds are enforced by the **extreme barrier**: infeasible
/// poll points are assigned `f = +∞` and rejected without evaluating the
/// objective (Audet & Dennis 2006, §2). Pair with a problem implementing
/// [`BoxConstraints`].
pub struct Bounded;

/// Type-state marker: nonlinearly-constrained MADS (`Mads<Constrained>`),
/// reached via [`Mads::constrained`]. General nonlinear inequality constraints
/// `c(x) ≤ 0` are handled by the **progressive barrier** (Audet & Dennis 2009):
/// the solver tracks the aggregate violation `h(x) = Σⱼ max(cⱼ, 0)²` and walks a
/// barrier threshold down to zero, so an **infeasible start is allowed**. Pair
/// with a problem implementing [`NonlinearInequalityConstraints`].
pub struct Constrained;

/// MADS (Audet & Dennis 2006): mesh adaptive direct search, deterministic
/// OrthoMADS instance: derivative-free local optimization for nonsmooth or
/// non-continuous objectives.
///
/// Configure the poll-size schedule, then drive it with an
/// [`Executor`](crate::Executor) over a [`MadsState`]:
///
/// ```
/// use basin::{CostFunction, Executor, Mads, MadsState, MaxCostEvals};
///
/// // A nonsmooth objective (an L1 "valley") with minimizer (1, 2).
/// struct AbsValley;
/// impl CostFunction for AbsValley {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
///         Ok((x[0] - 1.0).abs() + 2.0 * (x[1] - 2.0).abs())
///     }
/// }
///
/// let solver = Mads::new().with_initial_poll_size(1.0).with_min_poll_size(1e-9);
/// let state = MadsState::new(vec![0.0, 0.0]);
/// let result = Executor::new(AbsValley, solver, state)
///     .terminate_on(MaxCostEvals(5_000))
///     .run()
///     .unwrap();
/// assert!(result.best_cost() < 1e-6);
/// ```
///
/// # Configuration
///
/// - [`with_initial_poll_size`](Self::with_initial_poll_size): initial poll
///   size `Δ₀` (a reasonable initial change to the variables; default `1.0`). It
///   uniformly scales the mesh, so trial points stay on the scaled integer mesh.
/// - [`with_min_poll_size`](Self::with_min_poll_size): convergence floor on the
///   poll size (default `1e-6`). The run converges once `Δᵖ` reaches it; must be
///   `> 0` and `< Δ₀`.
///
/// # Constraints
///
/// Two opt-in constrained modes specialize the default unconstrained solver:
///
/// - [`bounded`](Self::bounded) → `Mads<Bounded>` handles box bounds by the
///   **extreme barrier** (infeasible poll points get `f = +∞`); needs a problem
///   implementing [`BoxConstraints`] and a feasible (or clamped) start.
/// - [`constrained`](Self::constrained) → `Mads<Constrained>` handles general
///   nonlinear inequality constraints `c(x) ≤ 0` by the **progressive barrier**
///   (Audet & Dennis 2009): it tracks the aggregate violation
///   `h(x) = Σⱼ max(cⱼ, 0)²` and drives a barrier threshold down to zero,
///   exploring around both a feasible and an infeasible incumbent. Needs a
///   problem implementing [`NonlinearInequalityConstraints`] and a
///   [`ConstrainedMadsState`]; an **infeasible start is allowed**.
///
/// # Backends
///
/// Backend-generic over the parameter vector: `Vec<f64>`, nalgebra, ndarray, and
/// faer all work. The poll geometry is internal pure-Rust integer/`f64` scratch,
/// so the parameter type needs only [`Clone`], [`VectorLen`], and `Index`/`IndexMut`,
/// never any `linalg`-tier op. Deterministic (no RNG) and wasm-clean.
///
/// # References
///
/// C. Audet & J. E. Dennis, Jr., *Mesh adaptive direct search algorithms for
/// constrained optimization*, SIAM J. Optim. 17 (2006), pp. 188–217 (with the
/// 2008 erratum). M. A. Abramson, C. Audet, J. E. Dennis, Jr. & S. Le Digabel,
/// *OrthoMADS: A deterministic MADS instance with orthogonal directions*, SIAM
/// J. Optim. 20 (2009), pp. 948–966. C. Audet & J. E. Dennis, Jr., *A
/// progressive barrier for derivative-free nonlinear programming*, SIAM J.
/// Optim. 20 (2009), pp. 445–472 (the constrained mode).
pub struct Mads<Mode = Unbounded, F = f64> {
    poll_size_init: F,
    poll_size_min: F,
    /// Built in [`Solver::init`]; the resumable poll loop + mesh schedule
    /// (unconstrained or box-bounded modes).
    work: Option<MadsWork<F>>,
    /// Built in [`Solver::init`] for the constrained mode; the resumable
    /// progressive-barrier poll loop. `None` in the other modes.
    pb_work: Option<PbWork<F>>,
    _mode: PhantomData<fn() -> Mode>,
}

impl<F: Scalar> Mads<Unbounded, F> {
    /// A MADS solver with the default schedule (`Δ₀ = 1`, `poll_size_min =
    /// 1e-6`). Tune with the `with_*` builders.
    pub fn new() -> Self {
        Self {
            poll_size_init: F::from_f64(1.0).expect("1.0 representable"),
            poll_size_min: F::from_f64(1e-6).expect("1e-6 representable"),
            work: None,
            pb_work: None,
            _mode: PhantomData,
        }
    }
}

impl<Mode, F: Scalar> Mads<Mode, F> {
    /// Set the initial poll size `Δ₀`, a reasonable initial change to the
    /// variables. Uniformly scales the mesh.
    pub fn with_initial_poll_size(mut self, poll_size_init: F) -> Self {
        self.poll_size_init = poll_size_init;
        self
    }

    /// Set the convergence floor on the poll size. Must satisfy `0 < floor < Δ₀`.
    pub fn with_min_poll_size(mut self, poll_size_min: F) -> Self {
        self.poll_size_min = poll_size_min;
        self
    }
}

impl<F: Scalar> Mads<Unbounded, F> {
    /// Switch to the box-bound-constrained variant ([`Bounded`]). The resulting
    /// `Mads<Bounded>` requires a problem implementing [`BoxConstraints`] and
    /// enforces the bounds by the extreme barrier (infeasible poll points get
    /// `f = +∞`).
    /// An infeasible start is clamped into the box.
    pub fn bounded(self) -> Mads<Bounded, F> {
        Mads {
            poll_size_init: self.poll_size_init,
            poll_size_min: self.poll_size_min,
            work: None,
            pb_work: None,
            _mode: PhantomData,
        }
    }

    /// Switch to the nonlinearly-constrained variant ([`Constrained`]). The
    /// resulting `Mads<Constrained>` requires a problem implementing
    /// [`NonlinearInequalityConstraints`] (`c(x) ≤ 0`) and handles the
    /// constraints by the **progressive barrier**, so an **infeasible start is
    /// allowed**. Drive it with a [`ConstrainedMadsState`].
    pub fn constrained(self) -> Mads<Constrained, F> {
        Mads {
            poll_size_init: self.poll_size_init,
            poll_size_min: self.poll_size_min,
            work: None,
            pb_work: None,
            _mode: PhantomData,
        }
    }
}

impl<F: Scalar> Default for Mads<Unbounded, F> {
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

/// Copy a backend vector `V` into a flat `Vec<F>` of length `n`.
fn to_vec<V, F>(v: &V, n: usize) -> Vec<F>
where
    V: std::ops::Index<usize, Output = F>,
    F: Copy,
{
    (0..n).map(|i| v[i]).collect()
}

/// Whether any component of `x` lies outside `[lower, upper]`.
fn out_of_box<F: Scalar>(x: &[F], lower: &[F], upper: &[F]) -> bool {
    x.iter()
        .zip(lower)
        .zip(upper)
        .any(|((&xi, &lo), &up)| xi < lo || xi > up)
}

impl<V, F> InitialState<V> for Mads<Unbounded, F>
where
    F: Scalar,
    V: Clone,
{
    type State = MadsState<V, F>;
    fn seed(&self, x: &V) -> Self::State {
        MadsState::new(x.clone())
    }
}

impl<V, F> InitialState<V> for Mads<Bounded, F>
where
    F: Scalar,
    V: Clone,
{
    type State = MadsState<V, F>;
    fn seed(&self, x: &V) -> Self::State {
        MadsState::new(x.clone())
    }
}

impl<V, F> InitialState<V> for Mads<Constrained, F>
where
    F: Scalar,
    V: Clone,
{
    type State = ConstrainedMadsState<V, F>;
    fn seed(&self, x: &V) -> Self::State {
        ConstrainedMadsState::new(x.clone())
    }
}

impl<P, V, F> Solver<P, MadsState<V, F>> for Mads<Unbounded, F>
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
        mut state: MadsState<V, F>,
    ) -> Result<MadsState<V, F>, Self::Error> {
        let n = state.param.vec_len();
        assert!(n >= 1, "Mads requires a non-empty start point");

        let x0: Vec<F> = (0..n).map(|i| state.param[i]).collect();
        let template = state.param.clone();

        let (work, best_x, best_f) = {
            let mut eval = |slice: &[F]| -> Result<F, P::Error> {
                problem.cost(&fill_from(&template, slice))
            };
            MadsWork::try_init(
                x0,
                self.poll_size_init,
                self.poll_size_min,
                &mut eval,
            )?
        };

        state.param = fill_from(&template, &best_x);
        state.cost = Some(best_f);
        state.poll_size = work.poll_size();
        state.mesh_index = work.mesh_index();
        self.work = Some(work);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: MadsState<V, F>,
    ) -> Result<(MadsState<V, F>, Option<TerminationReason>), Self::Error> {
        let template = state.param.clone();
        let work = self
            .work
            .as_mut()
            .expect("Mads::init must run before next_iter");

        let out = {
            let mut eval = |slice: &[F]| -> Result<F, P::Error> {
                problem.cost(&fill_from(&template, slice))
            };
            work.step(&mut eval)?
        };
        state.poll_size = work.poll_size();
        state.mesh_index = work.mesh_index();

        // Fold the step's improving evaluation into the running best (the
        // reported iterate). MADS accepts only improving mesh points, so
        // param/cost are monotone and coincide with best_param/best_cost.
        let mut best_f = state.cost.expect("Mads::init seeds the cost");
        for (xabs, f_new) in &out.evaluated {
            if *f_new < best_f {
                best_f = *f_new;
                state.param = fill_from(&template, xabs);
                state.cost = Some(best_f);
            }
        }

        // The poll size reaching its floor is MADS's natural convergence.
        let reason = match out.transition {
            Transition::Converged => Some(TerminationReason::SolverConverged),
            Transition::Continue => None,
        };
        Ok((state, reason))
    }
}

/// Aggregate constraint violation `h(x) = Σⱼ max(cⱼ(x), 0)²` from the constraint
/// vector `c(x)` (length `m`). `h = 0` exactly on the feasible set `c(x) ≤ 0`.
fn violation<V, F>(c: &V, m: usize) -> F
where
    V: std::ops::Index<usize, Output = F>,
    F: Scalar,
{
    let mut h = F::zero();
    for j in 0..m {
        let cj = c[j];
        if cj > F::zero() {
            h = h + cj * cj;
        }
    }
    h
}

impl<P, V, F> Solver<P, ConstrainedMadsState<V, F>> for Mads<Constrained, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F> + NonlinearInequalityConstraints,
    V: Clone
        + VectorLen
        + std::ops::Index<usize, Output = F>
        + std::ops::IndexMut<usize, Output = F>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: ConstrainedMadsState<V, F>,
    ) -> Result<ConstrainedMadsState<V, F>, Self::Error> {
        let n = state.param.vec_len();
        assert!(n >= 1, "Mads requires a non-empty start point");

        let m = problem.inner().num_constraints();
        let x0: Vec<F> = (0..n).map(|i| state.param[i]).collect();
        let template = state.param.clone();

        let (work, best_x, best_f, best_h) = {
            let mut eval = |slice: &[F]| -> Result<(F, F), P::Error> {
                let xv = fill_from(&template, slice);
                let f = problem.cost(&xv)?;
                let c = problem.inner().constraints(&xv)?;
                Ok((f, violation(&c, m)))
            };
            PbWork::try_init(
                x0,
                self.poll_size_init,
                self.poll_size_min,
                &mut eval,
            )?
        };

        state.param = fill_from(&template, &best_x);
        state.cost = Some(best_f);
        state.constraint_violation = best_h;
        state.poll_size = work.poll_size();
        state.mesh_index = work.mesh_index();
        self.pb_work = Some(work);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: ConstrainedMadsState<V, F>,
    ) -> Result<
        (ConstrainedMadsState<V, F>, Option<TerminationReason>),
        Self::Error,
    > {
        let m = problem.inner().num_constraints();
        let template = state.param.clone();
        let work = self
            .pb_work
            .as_mut()
            .expect("Mads::init must run before next_iter");

        let out = {
            let mut eval = |slice: &[F]| -> Result<(F, F), P::Error> {
                let xv = fill_from(&template, slice);
                let f = problem.cost(&xv)?;
                let c = problem.inner().constraints(&xv)?;
                Ok((f, violation(&c, m)))
            };
            work.step(&mut eval)?
        };
        state.poll_size = work.poll_size();
        state.mesh_index = work.mesh_index();

        // The progressive-barrier driver owns incumbent selection; adopt its
        // reported incumbent (feasible if found, else best infeasible).
        let (x_star, f_star, h_star) = out.incumbent;
        state.param = fill_from(&template, &x_star);
        state.cost = Some(f_star);
        state.constraint_violation = h_star;

        let reason = match out.transition {
            Transition::Converged => Some(TerminationReason::SolverConverged),
            Transition::Continue => None,
        };
        Ok((state, reason))
    }
}

impl<P, V, F> Solver<P, MadsState<V, F>> for Mads<Bounded, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F> + BoxConstraints,
    V: Clone
        + VectorLen
        + std::ops::Index<usize, Output = F>
        + std::ops::IndexMut<usize, Output = F>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: MadsState<V, F>,
    ) -> Result<MadsState<V, F>, Self::Error> {
        let n = state.param.vec_len();
        assert!(n >= 1, "Mads requires a non-empty start point");

        let lower = to_vec(problem.inner().lower(), n);
        let upper = to_vec(problem.inner().upper(), n);
        // Clamp an infeasible start into the box so the extreme barrier has a
        // feasible incumbent to descend from.
        let x0: Vec<F> = (0..n)
            .map(|i| {
                let xi = state.param[i];
                if xi < lower[i] {
                    lower[i]
                } else if xi > upper[i] {
                    upper[i]
                } else {
                    xi
                }
            })
            .collect();
        let template = state.param.clone();

        let (work, best_x, best_f) = {
            let mut eval = |slice: &[F]| -> Result<F, P::Error> {
                if out_of_box(slice, &lower, &upper) {
                    Ok(F::infinity()) // extreme barrier: reject without evaluating
                } else {
                    problem.cost(&fill_from(&template, slice))
                }
            };
            MadsWork::try_init(
                x0,
                self.poll_size_init,
                self.poll_size_min,
                &mut eval,
            )?
        };

        state.param = fill_from(&template, &best_x);
        state.cost = Some(best_f);
        state.poll_size = work.poll_size();
        state.mesh_index = work.mesh_index();
        self.work = Some(work);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: MadsState<V, F>,
    ) -> Result<(MadsState<V, F>, Option<TerminationReason>), Self::Error> {
        let n = state.param.vec_len();
        let lower = to_vec(problem.inner().lower(), n);
        let upper = to_vec(problem.inner().upper(), n);
        let template = state.param.clone();
        let work = self
            .work
            .as_mut()
            .expect("Mads::init must run before next_iter");

        let out = {
            let mut eval = |slice: &[F]| -> Result<F, P::Error> {
                if out_of_box(slice, &lower, &upper) {
                    Ok(F::infinity())
                } else {
                    problem.cost(&fill_from(&template, slice))
                }
            };
            work.step(&mut eval)?
        };
        state.poll_size = work.poll_size();
        state.mesh_index = work.mesh_index();

        let mut best_f = state.cost.expect("Mads::init seeds the cost");
        for (xabs, f_new) in &out.evaluated {
            if *f_new < best_f {
                best_f = *f_new;
                state.param = fill_from(&template, xabs);
                state.cost = Some(best_f);
            }
        }

        let reason = match out.transition {
            Transition::Converged => Some(TerminationReason::SolverConverged),
            Transition::Continue => None,
        };
        Ok((state, reason))
    }
}
