//! Single-iterate state for nonlinear least-squares solvers.

use crate::core::math::Scalar;
use crate::core::problem::EvalCounts;
use crate::core::state::{CountsMirror, State};

/// State for the nonlinear least-squares solvers
/// ([`GaussNewton`](crate::solver::GaussNewton),
/// [`LevenbergMarquardt`](crate::solver::LevenbergMarquardt),
/// [`Trf`](crate::solver::Trf)): one `param`, an optional cached cost, and
/// iteration/evaluation counters split into residual and Jacobian work.
///
/// # Why not [`BasicState`](crate::core::state::BasicState)?
///
/// `NllsState` deliberately does **not** impl
/// [`GradientState`](crate::core::state::GradientState). The NLLS solvers
/// have no L2 gradient to populate: their first-order optimality test is the
/// ∞-norm of `Jᵀr`, exposed through each solver's own `with_tol_grad`/
/// `with_tol_grad_rel`, not the framework's
/// [`GradientTolerance`](crate::core::termination::GradientTolerance). Were
/// these solvers to run on `BasicState` (which *is* a `GradientState`), a user
/// could attach `GradientTolerance` and have it **silently never fire**:
/// `gradient()` stays `None`, so the criterion short-circuits and falls
/// through to `MaxIter`. Binding NLLS to a state that is not a `GradientState`
/// turns that misconfiguration into a compile error (tenet 3), the same guard
/// that already keeps gradient criteria off derivative-free solvers.
///
/// # Eval counters
///
/// [`cost_evals`](State::cost_evals) folds residual evaluations into the cost
/// counter (a residual evaluation *is* the cost work for a least-squares
/// objective), so [`MaxCostEvals`](crate::core::termination::MaxCostEvals) and
/// [`OptimizationResult::cost_evals`](crate::core::executor::OptimizationResult::cost_evals)
/// behave exactly as they did on `BasicState`. The Jacobian work is surfaced
/// separately through [`jacobian_evals`](Self::jacobian_evals): the MINPACK
/// `njev` to `cost_evals`'s `nfev`. Hessian evaluations (unused by today's
/// NLLS solvers) fold into the Jacobian counter, preserving the historical
/// "Jacobian/Hessian → second-order work" convention.
///
/// The scalar `F` defaults to `f64` so `NllsState<P>` call sites resolve
/// unchanged.
pub struct NllsState<P, F = f64> {
    pub(crate) param: P,
    pub(crate) cost: Option<F>,
    pub(crate) iter: u64,
    pub(crate) cost_evals: u64,
    pub(crate) residual_evals: u64,
    pub(crate) jacobian_evals: u64,
    pub(crate) best_param: Option<P>,
    pub(crate) best_cost: F,
    pub(crate) best_iter: u64,
    pub(crate) best_cost_evals: u64,
}

impl<P, F: Scalar> NllsState<P, F> {
    /// Build a state at the given starting point. The cost is filled in by
    /// [`Solver::init`](crate::core::solver::Solver::init).
    pub fn new(param: P) -> Self {
        Self {
            param,
            cost: None,
            iter: 0,
            cost_evals: 0,
            residual_evals: 0,
            jacobian_evals: 0,
            best_param: None,
            best_cost: F::infinity(),
            best_iter: 0,
            best_cost_evals: 0,
        }
    }

    /// Cumulative residual evaluations across the run: the MINPACK `nfev`.
    ///
    /// Equal to [`cost_evals`](State::cost_evals) for a pure least-squares run
    /// (every cost is a residual evaluation); exposed under its own name so
    /// consumers porting MINPACK/`levenberg-marquardt`-crate diagnostics can
    /// read the residual and Jacobian counts independently.
    pub fn residual_evals(&self) -> u64 {
        self.residual_evals
    }

    /// Cumulative Jacobian evaluations across the run—the MINPACK `njev`.
    pub fn jacobian_evals(&self) -> u64 {
        self.jacobian_evals
    }
}

impl<P: Clone, F: Scalar> State for NllsState<P, F> {
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
    /// [`Solver::init`](crate::core::solver::Solver::init) has populated the
    /// cached cost. By contract,
    /// [`Executor`](crate::core::executor::Executor) calls `init` before any
    /// termination-criterion check, so reads from inside criteria and from
    /// [`OptimizationResult`](crate::core::executor::OptimizationResult) are
    /// safe.
    fn cost(&self) -> F {
        self.cost
            .expect("NllsState::cost read before Solver::init populated it")
    }

    fn best_param(&self) -> &P {
        self.best_param
            .as_ref()
            .expect("NllsState::best_param read before Solver::init populated it")
    }

    fn best_cost(&self) -> F {
        self.best_cost
    }

    fn best_iter(&self) -> u64 {
        self.best_iter
    }

    fn best_cost_evals(&self) -> u64 {
        self.best_cost_evals
    }

    fn update_best(&mut self) {
        if let Some(curr) = self.cost {
            if self.best_param.is_none() || curr < self.best_cost {
                self.best_param = Some(self.param.clone());
                self.best_cost = curr;
                self.best_iter = self.iter;
                self.best_cost_evals = self.cost_evals;
            }
        }
    }

    fn reset_best(&mut self) {
        self.best_param = None;
        self.best_cost = F::infinity();
        self.best_iter = 0;
        self.best_cost_evals = 0;
    }
}

impl<P, F> CountsMirror for NllsState<P, F>
where
    NllsState<P, F>: State,
{
    fn mirror(&mut self, delta: &EvalCounts) {
        // Residual calls are the cost work for a least-squares objective, so
        // they fold into `cost_evals`; the Jacobian (and, for any future
        // second-order NLLS solver, the Hessian) is tracked separately.
        self.cost_evals = delta.cost_evals + delta.residual_evals;
        self.residual_evals = delta.residual_evals;
        self.jacobian_evals = delta.jacobian_evals + delta.hessian_evals;
    }
}
