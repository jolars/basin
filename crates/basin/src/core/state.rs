//! Solver state shapes.
//!
//! Every [`Solver`](crate::core::solver::Solver) carries its iterate as a
//! [`State`]. The base [`State`] trait is the minimum the executor and
//! generic termination criteria need to read; richer state shapes extend
//! it ([`GradientState`] for first-order solvers, [`SimplexState`] for
//! simplex-based solvers like Nelder-Mead) so termination criteria can
//! bound on the minimum capability they need (tenet 3 in `AGENTS.md`).
//!
//! `State::Float` is generic across the trait. The vector-tier-only states
//! ([`BasicState`], [`BasicSimplexState`], [`BasicPopulationState`]) and the
//! linalg-tier-using [`QuasiNewtonState`] / [`LbfgsState`] take an `F: Scalar`
//! parameter that defaults to `f64`, so existing call sites resolve unchanged
//! while opening the door to `f32`. Solvers (`GradientDescent`, `Bfgs`,
//! both `Lbfgs` modes, the NLLS family, CMA-ES, barrier / AL, etc.) and the
//! shipped termination criteria all carry the same `F = f64` default. See
//! `tests/f32_round_trip.rs` for an end-to-end demonstration that the full
//! pipeline composes at `F = f32`, and the *Provisional choices* section of
//! `AGENTS.md`.

/// Limited-memory BFGS / L-BFGS-B state (`LbfgsState`).
pub mod lbfgs;

pub use lbfgs::LbfgsState;

use crate::core::math::{MatrixIdentity, Scalar, VectorLen};
use crate::core::problem::EvalCounts;

/// Minimum information the executor and generic termination criteria
/// need to read from a solver's iterate.
///
/// # Contract
///
/// - **Caller must:** construct via the appropriate concrete state
///   constructor (e.g. [`BasicState::new`]) before handing the state to
///   [`Executor`](crate::core::executor::Executor). The executor's `init`
///   call populates derived fields (cost, gradient) before any termination
///   check sees the state.
/// - **Implementor must:** keep [`param`](Self::param) stable between
///   iterations — the returned reference is valid until the next
///   [`Solver::next_iter`](crate::core::solver::Solver::next_iter)
///   returns. [`cost_evals`](Self::cost_evals) counts every call to the
///   problem's cost function, not iterations: a single
///   [`Solver::next_iter`](crate::core::solver::Solver::next_iter) may
///   evaluate the cost many times (line searches, Nelder-Mead shrinks),
///   and users budget against this counter rather than
///   [`iter`](Self::iter).
pub trait State {
    /// The parameter type the solver iterates over (e.g. `Vec<f64>`,
    /// `nalgebra::DVector<f64>`).
    type Param;
    /// The scalar type of the objective. In practice always `f64` (see
    /// the module docs).
    type Float;

    /// Number of fully completed iterations. A
    /// [`Solver::next_iter`](crate::core::solver::Solver::next_iter)
    /// that bails mid-iteration with `Some(reason)` does not increment
    /// this counter — see the
    /// [`executor`](crate::core::executor) module for the exact ordering.
    fn iter(&self) -> u64;
    /// Increment [`iter`](Self::iter) by one. Called by the executor
    /// after a successful [`Solver::next_iter`](crate::core::solver::Solver::next_iter).
    fn increment_iter(&mut self);
    /// Cumulative count of cost-function evaluations performed so far.
    /// Diverges from `iter()` whenever a single iteration evaluates the
    /// cost more than once (line searches, Nelder-Mead shrinks, etc.) —
    /// this is what users actually budget against.
    ///
    /// Populated by the
    /// [`Executor`](crate::core::executor::Executor) from the wrapper's
    /// `EvalCounts` after every
    /// successful
    /// [`Solver::init`](crate::core::solver::Solver::init) /
    /// [`Solver::next_iter`](crate::core::solver::Solver::next_iter) —
    /// the per-state mapping is defined by the state's
    /// [`CountsMirror`] impl. Solvers never write to this counter
    /// directly; the wrapper's counts are authoritative.
    fn cost_evals(&self) -> u64;
    /// Current iterate. Stable between
    /// [`Solver::next_iter`](crate::core::solver::Solver::next_iter)
    /// calls; safe to read at any iteration including iter 0.
    fn param(&self) -> &Self::Param;
    /// Cost at the current [`param`](Self::param).
    ///
    /// # Panics
    ///
    /// Concrete states ([`BasicState`], [`BasicSimplexState`], and
    /// `QuasiNewtonState` under the `nalgebra` feature) panic if
    /// `cost()` is read before
    /// [`Solver::init`](crate::core::solver::Solver::init) has populated
    /// the cached cost. By contract the executor calls `init` before any
    /// termination criterion check, so reads from criteria and from
    /// [`OptimizationResult`](crate::core::executor::OptimizationResult)
    /// are safe.
    fn cost(&self) -> Self::Float;
}

/// States that carry a gradient at the current [`param`](State::param).
///
/// # Contract
///
/// - **Implementor must:** at the end of every successful
///   [`Solver::next_iter`](crate::core::solver::Solver::next_iter)
///   (and at the end of [`Solver::init`](crate::core::solver::Solver::init)
///   for first-order solvers), populate
///   [`gradient`](Self::gradient) so it corresponds to the *current*
///   [`param`](State::param). Termination criteria read it; if it lags
///   behind the param they will fire on stale data.
/// - `None` means "no gradient available at this iterate yet" — the
///   only legitimate case is before
///   [`Solver::init`](crate::core::solver::Solver::init) has run, used
///   by criteria like [`GradientTolerance`](crate::core::termination::GradientTolerance)
///   to silently skip the check.
pub trait GradientState: State {
    /// Gradient at the current [`param`](State::param), if populated.
    fn gradient(&self) -> Option<&Self::Param>;
    /// Cumulative count of gradient evaluations performed so far. Lives
    /// on `GradientState` rather than `State` so derivative-free states
    /// don't carry a counter they can never increment.
    ///
    /// Populated by the
    /// [`Executor`](crate::core::executor::Executor) from the wrapper's
    /// `EvalCounts`; see
    /// [`State::cost_evals`] for the broader rule and
    /// [`CountsMirror`] for the per-state mapping.
    fn gradient_evals(&self) -> u64;
}

/// Bridge from the wrapper's
/// `EvalCounts` to the state's
/// [`State::cost_evals`] / [`GradientState::gradient_evals`] counters.
/// The [`Executor`](crate::core::executor::Executor) calls
/// [`mirror`](Self::mirror) after every successful
/// [`Solver::init`](crate::core::solver::Solver::init) /
/// [`Solver::next_iter`](crate::core::solver::Solver::next_iter),
/// passing the per-run delta of the wrapper's counts so the state
/// reflects work-since-this-run-started (rather than cumulative across
/// nested [`run_loop`](crate::core::executor::run_loop) calls).
///
/// Public (rather than crate-private) so user-defined state types can
/// be plugged into the [`Executor`](crate::core::executor::Executor) —
/// the trait must be impl'able outside basin. Most users won't need
/// this: the shipped state types (`BasicState`, `QuasiNewtonState`,
/// `LbfgsState`, `BasicSimplexState`, `BasicPopulationState`) already
/// implement it.
///
/// # Per-state mapping
///
/// - **[`BasicState`] / [`QuasiNewtonState`] / [`LbfgsState`]** (carry
///   both cost and gradient counters):
///   `cost_evals = cost + residual`,
///   `gradient_evals = gradient + jacobian + hessian`.
///   The residual / Jacobian / Hessian counters fold into the
///   cost / gradient slots, preserving today's NLLS convention where
///   residual calls counted against `cost_evals` and Jacobian calls
///   against `gradient_evals` on `BasicState`.
/// - **[`BasicSimplexState`] / [`BasicPopulationState`] / `MaLsChState`**
///   (derivative-free outer, no `gradient_evals` field):
///   `cost_evals = total_work` (every kind folded in). Lets a CMA-ES
///   outer running e.g. an L-BFGS inner have `state.cost_evals`
///   reflect total computational work without a per-trait cross-type
///   fold.
pub trait CountsMirror: State {
    /// Overwrite the state's counters from the per-run wrapper delta.
    /// Called by the executor after every successful
    /// [`Solver::init`](crate::core::solver::Solver::init) /
    /// [`Solver::next_iter`](crate::core::solver::Solver::next_iter).
    fn mirror(&mut self, delta: &EvalCounts);
}

/// States built around a simplex of `n + 1` vertices and parallel costs.
///
/// Mirrors [`GradientState`]: the trait exists so termination criteria
/// (e.g. the simplex-collapse test of Lagarias et al. 1998, eq. T1, in
/// [`SimplexTolerance`](crate::core::termination::SimplexTolerance)) can
/// bound on a richer view than [`State::param`] / [`State::cost`], which
/// only see the best vertex.
///
/// # Contract
///
/// - **Implementor must:** keep [`vertices`](Self::vertices) and
///   [`costs`](Self::costs) sorted by **ascending cost** at the start and
///   end of every [`Solver::next_iter`](crate::core::solver::Solver::next_iter)
///   call (and at the end of [`Solver::init`](crate::core::solver::Solver::init)).
///   So [`State::param`] / [`State::cost`] always return the current best
///   vertex (`vertices[0]` / `costs[0]`).
/// - **Implementor must:** sort `NaN` costs *last*, so a single bad
///   evaluation can't drag itself to the front and become the
///   "best" vertex.
/// - **Implementor must:** keep the two slices the same length and in
///   parallel order — `costs[i]` is the cost at `vertices[i]`.
pub trait SimplexState: State {
    /// All `n + 1` vertices, sorted by ascending cost.
    fn vertices(&self) -> &[Self::Param];
    /// Costs in parallel with [`vertices`](Self::vertices), sorted ascending.
    fn costs(&self) -> &[Self::Float];
}

/// States built around a population of `λ` candidate parameters and
/// parallel costs.
///
/// Mirrors [`SimplexState`]: the trait exists so termination criteria
/// that need to inspect the whole population (diversity, generation
/// spread, stall counters) can bound on a richer view than
/// [`State::param`] / [`State::cost`], which only see the best
/// candidate. The vehicle for stochastic solvers
/// ([`RandomSearch`](crate::solver::RandomSearch); CMA-ES once it lands).
///
/// # Contract
///
/// - **Implementor must:** keep [`candidates`](Self::candidates) and
///   [`costs`](Self::costs) sorted by **ascending cost** at the start
///   and end of every
///   [`Solver::next_iter`](crate::core::solver::Solver::next_iter)
///   call (and at the end of [`Solver::init`](crate::core::solver::Solver::init)).
///   So [`State::param`] / [`State::cost`] always return the current
///   best candidate (`candidates[0]` / `costs[0]`).
/// - **Implementor must:** sort `NaN` costs *last*, so a single bad
///   evaluation can't drag itself to the front and become the
///   "best" candidate.
/// - **Implementor must:** keep the two slices the same length and in
///   parallel order — `costs[i]` is the cost at `candidates[i]`.
pub trait PopulationState: State {
    /// All `λ` candidates, sorted by ascending cost.
    fn candidates(&self) -> &[Self::Param];
    /// Costs in parallel with [`candidates`](Self::candidates), sorted
    /// ascending.
    fn costs(&self) -> &[Self::Float];
}

/// Default state for single-iterate solvers (gradient descent,
/// Gauss-Newton, …): one `param`, optional cached cost and gradient,
/// plus iteration / evaluation counters.
///
/// The scalar `F` defaults to `f64` so existing `BasicState<P>` call
/// sites resolve unchanged.
pub struct BasicState<P, F = f64> {
    pub(crate) param: P,
    pub(crate) cost: Option<F>,
    pub(crate) gradient: Option<P>,
    pub(crate) iter: u64,
    pub(crate) cost_evals: u64,
    pub(crate) gradient_evals: u64,
}

impl<P, F> BasicState<P, F> {
    /// Build a state at the given starting point. Cost and gradient
    /// are filled in by [`Solver::init`](crate::core::solver::Solver::init).
    pub fn new(param: P) -> Self {
        Self {
            param,
            cost: None,
            gradient: None,
            iter: 0,
            cost_evals: 0,
            gradient_evals: 0,
        }
    }
}

impl<P, F: Scalar> State for BasicState<P, F> {
    type Param = P;
    type Float = F;

    fn iter(&self) -> u64 {
        self.iter
    }

    fn increment_iter(&mut self) {
        self.iter += 1;
    }

    fn cost_evals(&self) -> u64 {
        self.cost_evals
    }

    fn param(&self) -> &P {
        &self.param
    }

    /// Reads the cost cached at the current `param`.
    ///
    /// # Panics
    ///
    /// Panics if accessed before
    /// [`Solver::init`](crate::core::solver::Solver::init) has populated
    /// the cached cost. By contract,
    /// [`Executor`](crate::core::executor::Executor) calls `init` before
    /// any termination-criterion check (see the
    /// [`executor`](crate::core::executor) module docs for the full
    /// ordering), so reads from inside criteria and from
    /// [`OptimizationResult`](crate::core::executor::OptimizationResult)
    /// are safe.
    fn cost(&self) -> F {
        self.cost
            .expect("BasicState::cost read before Solver::init populated it")
    }
}

impl<P, F: Scalar> GradientState for BasicState<P, F> {
    fn gradient(&self) -> Option<&P> {
        self.gradient.as_ref()
    }

    fn gradient_evals(&self) -> u64 {
        self.gradient_evals
    }
}

impl<P, F> CountsMirror for BasicState<P, F>
where
    BasicState<P, F>: State,
{
    fn mirror(&mut self, delta: &EvalCounts) {
        // NLLS convention preserved: residual calls fold into the cost
        // counter, Jacobian / Hessian into gradient. (Today's
        // Gauss-Newton / LM / TRF impls manually bumped cost_evals on
        // residual() and gradient_evals on jacobian().)
        self.cost_evals = delta.cost_evals + delta.residual_evals;
        self.gradient_evals = delta.gradient_evals + delta.jacobian_evals + delta.hessian_evals;
    }
}

/// Default `SimplexState` implementation: `n + 1` vertices and their costs
/// in parallel `Vec`s. The solver keeps both sorted by ascending cost at
/// the start and end of every `next_iter`, so `param()` / `cost()` always
/// return the current best vertex.
///
/// The scalar `F` defaults to `f64` so existing `BasicSimplexState<V>`
/// call sites resolve unchanged.
pub struct BasicSimplexState<V, F = f64> {
    pub(crate) vertices: Vec<V>,
    pub(crate) costs: Vec<F>,
    pub(crate) iter: u64,
    pub(crate) cost_evals: u64,
}

impl<V, F: Scalar> BasicSimplexState<V, F> {
    /// Build from a pre-constructed simplex (advanced users / non-default
    /// initial geometries). For the common case of "I just have a starting
    /// point", prefer the backend-specific `BasicSimplexState::new`
    /// constructors.
    pub fn from_simplex(vertices: Vec<V>) -> Self {
        assert!(
            vertices.len() >= 2,
            "BasicSimplexState requires at least 2 vertices (n+1 for an n-D problem)"
        );
        let n = vertices.len();
        Self {
            vertices,
            costs: vec![F::infinity(); n],
            iter: 0,
            cost_evals: 0,
        }
    }
}

/// FMINSEARCH/SciPy-style initial simplex from a single starting point.
///
/// Implemented per backend (`Vec<f64>`, `nalgebra::DVector<f64>`, …) so a
/// single `BasicSimplexState::new(x0)` constructor works uniformly across
/// backends. The default step is 5% on non-zero coordinates and an
/// absolute `0.00025` on zero coordinates.
pub trait IntoInitialSimplex<V> {
    /// Build a simplex of `n + 1` vertices around `self`, perturbing each
    /// coordinate by `relative_step`.
    fn into_initial_simplex(self, relative_step: f64) -> Vec<V>;
}

impl IntoInitialSimplex<Self> for Vec<f64> {
    fn into_initial_simplex(self, relative_step: f64) -> Vec<Self> {
        let n = self.len();
        let mut simplex = Vec::with_capacity(n + 1);
        simplex.push(self.clone());
        for i in 0..n {
            let mut v = self.clone();
            v[i] = if self[i] != 0.0 {
                (1.0 + relative_step) * self[i]
            } else {
                0.00025
            };
            simplex.push(v);
        }
        simplex
    }
}

#[cfg(feature = "nalgebra")]
impl IntoInitialSimplex<Self> for nalgebra::DVector<f64> {
    fn into_initial_simplex(self, relative_step: f64) -> Vec<Self> {
        let n = self.len();
        let mut simplex = Vec::with_capacity(n + 1);
        simplex.push(self.clone());
        for i in 0..n {
            let mut v = self.clone();
            v[i] = if self[i] != 0.0 {
                (1.0 + relative_step) * self[i]
            } else {
                0.00025
            };
            simplex.push(v);
        }
        simplex
    }
}

#[cfg(feature = "faer")]
impl IntoInitialSimplex<Self> for faer::Col<f64> {
    fn into_initial_simplex(self, relative_step: f64) -> Vec<Self> {
        let n = self.nrows();
        let mut simplex = Vec::with_capacity(n + 1);
        simplex.push(self.clone());
        for i in 0..n {
            let mut v = self.clone();
            v[i] = if self[i] != 0.0 {
                (1.0 + relative_step) * self[i]
            } else {
                0.00025
            };
            simplex.push(v);
        }
        simplex
    }
}

#[cfg(feature = "ndarray")]
impl IntoInitialSimplex<ndarray::Array1<f64>> for ndarray::Array1<f64> {
    fn into_initial_simplex(self, relative_step: f64) -> Vec<ndarray::Array1<f64>> {
        let n = self.len();
        let mut simplex = Vec::with_capacity(n + 1);
        simplex.push(self.clone());
        for i in 0..n {
            let mut v = self.clone();
            v[i] = if self[i] != 0.0 {
                (1.0 + relative_step) * self[i]
            } else {
                0.00025
            };
            simplex.push(v);
        }
        simplex
    }
}

impl<V, F: Scalar> BasicSimplexState<V, F> {
    /// Build an FMINSEARCH/SciPy-style simplex around a starting point
    /// `x0`. Mirrors `BasicState::new` ergonomically — the solver infers
    /// dimension from the simplex during `init`.
    pub fn new<X: IntoInitialSimplex<V>>(x0: X) -> Self {
        Self::from_simplex(x0.into_initial_simplex(0.05))
    }

    /// Like `new`, but with a custom relative step (default is `0.05`).
    /// Zero coordinates still use the FMINSEARCH absolute step `0.00025`.
    pub fn with_step<X: IntoInitialSimplex<V>>(x0: X, relative_step: f64) -> Self {
        Self::from_simplex(x0.into_initial_simplex(relative_step))
    }
}

impl<V, F: Scalar> State for BasicSimplexState<V, F> {
    type Param = V;
    type Float = F;

    fn iter(&self) -> u64 {
        self.iter
    }

    fn increment_iter(&mut self) {
        self.iter += 1;
    }

    fn cost_evals(&self) -> u64 {
        self.cost_evals
    }

    fn param(&self) -> &V {
        &self.vertices[0]
    }

    fn cost(&self) -> F {
        self.costs[0]
    }
}

impl<V, F> CountsMirror for BasicSimplexState<V, F>
where
    BasicSimplexState<V, F>: State,
{
    fn mirror(&mut self, delta: &EvalCounts) {
        // Derivative-free state: any work folds into the single
        // `cost_evals` counter (a simplex solver only calls `cost`
        // today; the generalization matters when a future composed
        // outer drives a gradient-based inner against this state).
        self.cost_evals = delta.total_work();
    }
}

impl<V, F: Scalar> SimplexState for BasicSimplexState<V, F> {
    fn vertices(&self) -> &[V] {
        &self.vertices
    }

    fn costs(&self) -> &[F] {
        &self.costs
    }
}

/// State for quasi-Newton solvers that maintain a dense inverse-Hessian
/// approximation `H ≈ ∇²f(x)⁻¹` (BFGS, DFP, SR1).
///
/// Generic over the param vector `V` and dense matrix `M`. Constructors
/// ship for the `Vec<f64>` / [`DenseMatrix`](crate::DenseMatrix) backend
/// (always available) and
/// the nalgebra `DVector<f64>` / `DMatrix<f64>` backend (feature `nalgebra`);
/// faer is reached via the generic [`State`] / [`GradientState`] impls below.
/// (L-BFGS uses a different state shape — a history of `(s, y)` pairs — see
/// [`LbfgsState`].)
///
/// `initial_scaling_done` tracks whether we've applied the standard
/// `H₀ ← (sᵀy / yᵀy)·I` rescaling after the first accepted step (Nocedal
/// & Wright (6.20)). This makes the unit step well-scaled on poorly
/// conditioned problems where plain identity initialization stalls.
///
/// The scalar `F` defaults to `f64` so existing `QuasiNewtonState<V, M>`
/// call sites resolve unchanged.
pub struct QuasiNewtonState<V, M, F = f64> {
    pub(crate) param: V,
    pub(crate) cost: Option<F>,
    pub(crate) gradient: Option<V>,
    pub(crate) inverse_hessian: M,
    pub(crate) initial_scaling_done: bool,
    pub(crate) iter: u64,
    pub(crate) cost_evals: u64,
    pub(crate) gradient_evals: u64,
}

impl<V: VectorLen, M: MatrixIdentity, F> QuasiNewtonState<V, M, F> {
    /// Build a state at the given starting point with the inverse-Hessian
    /// approximation initialised to the identity.
    ///
    /// Generic over the backend: `M` is the dense matrix paired with the
    /// param vector `V` — [`DenseMatrix`](crate::core::math::DenseMatrix) for
    /// `Vec<f64>`, `DMatrix<f64>` for nalgebra, `Mat<f64>` for faer. Since
    /// `M` is not an argument, annotate it at the call site when it can't be
    /// inferred from context, e.g.
    /// `QuasiNewtonState::<Vec<f64>, DenseMatrix>::new(x)`.
    pub fn new(param: V) -> Self {
        let n = param.vec_len();
        Self {
            param,
            cost: None,
            gradient: None,
            inverse_hessian: M::identity(n),
            initial_scaling_done: false,
            iter: 0,
            cost_evals: 0,
            gradient_evals: 0,
        }
    }
}

impl<V, M, F: Scalar> State for QuasiNewtonState<V, M, F> {
    type Param = V;
    type Float = F;

    fn iter(&self) -> u64 {
        self.iter
    }

    fn increment_iter(&mut self) {
        self.iter += 1;
    }

    fn cost_evals(&self) -> u64 {
        self.cost_evals
    }

    fn param(&self) -> &V {
        &self.param
    }

    /// Reads the cost cached at the current `param`.
    ///
    /// # Panics
    ///
    /// Panics if accessed before
    /// [`Solver::init`](crate::core::solver::Solver::init) has populated
    /// the cached cost. See [`BasicState::cost`] for the full safety
    /// argument — same contract.
    fn cost(&self) -> F {
        self.cost
            .expect("QuasiNewtonState::cost read before Solver::init populated it")
    }
}

impl<V, M, F: Scalar> GradientState for QuasiNewtonState<V, M, F> {
    fn gradient(&self) -> Option<&V> {
        self.gradient.as_ref()
    }

    fn gradient_evals(&self) -> u64 {
        self.gradient_evals
    }
}

impl<V, M, F: Scalar> CountsMirror for QuasiNewtonState<V, M, F> {
    fn mirror(&mut self, delta: &EvalCounts) {
        self.cost_evals = delta.cost_evals + delta.residual_evals;
        self.gradient_evals = delta.gradient_evals + delta.jacobian_evals + delta.hessian_evals;
    }
}

/// Default [`PopulationState`] implementation: `λ` candidate parameters
/// and parallel costs. The solver keeps both sorted by ascending cost
/// at the start and end of every `next_iter`, so [`State::param`] /
/// [`State::cost`] always return the current best candidate.
///
/// Vehicle for [`RandomSearch`](crate::solver::RandomSearch); will be
/// reused by CMA-ES (S8) without changes.
///
/// The scalar `F` defaults to `f64` so existing `BasicPopulationState<V>`
/// call sites resolve unchanged.
pub struct BasicPopulationState<V, F = f64> {
    pub(crate) candidates: Vec<V>,
    pub(crate) costs: Vec<F>,
    pub(crate) iter: u64,
    pub(crate) cost_evals: u64,
}

impl<V, F: Scalar> BasicPopulationState<V, F> {
    /// Build from a pre-constructed population (advanced users; custom
    /// initial distributions). Costs are filled by the solver in
    /// [`Solver::init`](crate::core::solver::Solver::init).
    ///
    /// # Panics
    ///
    /// Panics if `candidates` is empty — a population must have at
    /// least one member.
    pub fn from_population(candidates: Vec<V>) -> Self {
        assert!(
            !candidates.is_empty(),
            "BasicPopulationState requires a non-empty population"
        );
        let n = candidates.len();
        Self {
            candidates,
            costs: vec![F::infinity(); n],
            iter: 0,
            cost_evals: 0,
        }
    }

    /// Empty container with `lambda` capacity reserved. The solver
    /// fills it in [`Solver::init`](crate::core::solver::Solver::init)
    /// (e.g. by sampling uniformly in the problem's box).
    ///
    /// Use this constructor when the *solver* owns the initial-
    /// population distribution (the random-search style); use
    /// [`from_population`](Self::from_population) when the *caller* owns
    /// it.
    ///
    /// # Panics
    ///
    /// Panics if `lambda == 0`.
    pub fn with_size(lambda: usize) -> Self {
        assert!(lambda >= 1, "BasicPopulationState requires lambda >= 1");
        Self {
            candidates: Vec::with_capacity(lambda),
            costs: Vec::with_capacity(lambda),
            iter: 0,
            cost_evals: 0,
        }
    }
}

impl<V, F: Scalar> State for BasicPopulationState<V, F> {
    type Param = V;
    type Float = F;

    fn iter(&self) -> u64 {
        self.iter
    }

    fn increment_iter(&mut self) {
        self.iter += 1;
    }

    fn cost_evals(&self) -> u64 {
        self.cost_evals
    }

    fn param(&self) -> &V {
        &self.candidates[0]
    }

    fn cost(&self) -> F {
        self.costs[0]
    }
}

impl<V, F> CountsMirror for BasicPopulationState<V, F>
where
    BasicPopulationState<V, F>: State,
{
    fn mirror(&mut self, delta: &EvalCounts) {
        // Derivative-free outer: any kind of work (e.g. an L-BFGS
        // inner's gradient calls inside a CMA-injection wrapper) folds
        // into the single `cost_evals` counter so `state.cost_evals`
        // reflects total work, with no per-trait cross-type fold.
        self.cost_evals = delta.total_work();
    }
}

impl<V, F: Scalar> PopulationState for BasicPopulationState<V, F> {
    fn candidates(&self) -> &[V] {
        &self.candidates
    }

    fn costs(&self) -> &[F] {
        &self.costs
    }
}
