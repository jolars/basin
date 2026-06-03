//! Framework: traits, state shapes, the iteration driver, and the
//! termination layer. The slot taxonomy is:
//!
//! - [`problem`] — what the *user* implements about their objective:
//!   [`CostFunction`](problem::CostFunction),
//!   [`Gradient`](problem::Gradient),
//!   [`MiniBatchGradient`](problem::MiniBatchGradient) (finite-sum,
//!   for mini-batch SGD),
//!   [`Residual`](problem::Residual) / [`Jacobian`](problem::Jacobian)
//!   (least squares), and [`Hessian`](problem::Hessian) (second order).
//!   Future: operators (matrix-free).
//! - [`numdiff`] — finite-difference derivative synthesis: the
//!   [`FiniteDiff`](numdiff::FiniteDiff) wrapper adds
//!   [`Gradient`](problem::Gradient) / [`Jacobian`](problem::Jacobian) /
//!   [`Hessian`](problem::Hessian) to a problem that only exposes
//!   function values.
//! - [`constraint`] — constraint markers carried on the problem (tenet 4
//!   in `CONTRIBUTING.md`): [`BoxConstraints`](constraint::BoxConstraints) and
//!   [`LinearInequalityConstraints`](constraint::LinearInequalityConstraints).
//! - [`barrier`] — the [`LogBarrier`](barrier::LogBarrier) adapter that
//!   rewrites a linearly-constrained problem as the unconstrained
//!   log-barrier objective consumed by the
//!   [`BarrierMethod`](crate::solver::BarrierMethod).
//! - [`state`] — what a solver carries between iterations:
//!   [`State`](state::State) for the minimum,
//!   [`GradientState`](state::GradientState) /
//!   [`SimplexState`](state::SimplexState) when a solver carries
//!   richer info that termination criteria can read.
//! - [`solver`] — the [`Solver`](solver::Solver) trait every concrete
//!   solver implements. Lifecycle is `init` once, then repeated
//!   `next_iter`, with an optional `terminate` hook.
//! - [`termination`] — the framework-level
//!   [`TerminationCriterion`](termination::TerminationCriterion)
//!   trait plus shipped criteria. Each criterion bounds on the minimum
//!   state shape it needs (tenet 3), so mismatches are compile errors
//!   rather than runtime no-ops.
//! - [`observer`] — read-only side-effect hooks fired around the loop
//!   ([`Observe`](observer::Observe) + [`ObserverMode`](observer::ObserverMode)).
//!   Sibling to [`termination`]: observers watch, criteria decide.
//! - [`executor`] — the driver: [`Executor`](executor::Executor) /
//!   [`Stepper`](executor::Stepper) / [`run_loop`](executor::run_loop).
//!   The canonical iteration ordering is documented on the
//!   [`executor`] module.
//! - [`inner`] — the composition adapter:
//!   [`InnerExecutor`](inner::InnerExecutor) wraps `run_loop` for outer
//!   solvers that drive an inner solver per outer iteration. See
//!   `CONTRIBUTING.md` "Solver composition" for the contracts.
//! - [`math`] — the small shared math layer
//!   ([`ScaledAdd`](math::ScaledAdd), [`NormSquared`](math::NormSquared),
//!   …) that backend-generic solvers depend on. Per tenet 5, this stays
//!   honest: only ops every backend can implement well live here. LA-heavy
//!   ops will live in a separate tier when the first solver wants them.

pub mod augmented_lagrangian;
pub mod barrier;
pub mod constraint;
pub mod executor;
pub mod inner;
pub mod math;
pub mod numdiff;
pub mod observer;
#[doc(hidden)]
pub mod parallel;
pub mod problem;
pub mod rng;
pub mod solver;
pub mod state;
pub mod termination;
