//! Single-iterate state for one-dimensional solvers that use a derivative.

use crate::core::math::Scalar;
use crate::core::problem::EvalCounts;
use crate::core::state::{CountsMirror, GradientState, State};

/// State for one-dimensional solvers that carry a first derivative
/// ([`BrentDerivative`](crate::solver::BrentDerivative)): a scalar `param`, an
/// optional cached cost, an optional cached gradient `f'(x)`, and
/// cost and gradient evaluation counters. `Param` and `Float` are the same scalar
/// `F`.
///
/// It is the gradient-carrying sibling of
/// [`ScalarState`](crate::core::state::ScalarState): same scalar single-iterate
/// shape, but it *does* impl [`GradientState`] (its gradient is the scalar
/// `f'(x)`), so first-order termination criteria such as
/// [`GradientTolerance`](crate::core::termination::GradientTolerance) work on a
/// 1D solver: the natural "stop when `|f'(x)| ≤ tol`" test. A scalar gradient
/// satisfies the criteria's norm bounds directly (`f64::norm_squared()` is
/// `f' · f'`).
///
/// [`best_param`](State::best_param) tracks the lowest-cost probe, which for a
/// bracketing search can differ from the final iterate.
///
/// The scalar `F` defaults to `f64` so `ScalarGradientState` call sites resolve
/// unchanged.
pub struct ScalarGradientState<F = f64> {
    pub(crate) param: F,
    pub(crate) cost: Option<F>,
    pub(crate) gradient: Option<F>,
    pub(crate) iter: u64,
    pub(crate) cost_evals: u64,
    pub(crate) gradient_evals: u64,
    pub(crate) best_param: Option<F>,
    pub(crate) best_cost: F,
    pub(crate) best_iter: u64,
    pub(crate) best_cost_evals: u64,
    pub(crate) best_gradient_evals: u64,
}

impl<F: Scalar> ScalarGradientState<F> {
    /// Build a state at the given starting point. The cost and gradient are
    /// filled in by [`Solver::init`](crate::core::solver::Solver::init).
    pub fn new(param: F) -> Self {
        Self {
            param,
            cost: None,
            gradient: None,
            iter: 0,
            cost_evals: 0,
            gradient_evals: 0,
            best_param: None,
            best_cost: F::infinity(),
            best_iter: 0,
            best_cost_evals: 0,
            best_gradient_evals: 0,
        }
    }
}

impl<F: Scalar> State for ScalarGradientState<F> {
    type Param = F;
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

    fn param(&self) -> &F {
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
            .expect("ScalarGradientState::cost read before Solver::init populated it")
    }

    fn best_param(&self) -> &F {
        self.best_param
            .as_ref()
            .expect("ScalarGradientState::best_param read before Solver::init populated it")
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
                self.best_param = Some(self.param);
                self.best_cost = curr;
                self.best_iter = self.iter;
                self.best_cost_evals = self.cost_evals;
                self.best_gradient_evals = self.gradient_evals;
            }
        }
    }

    fn reset_best(&mut self) {
        self.best_param = None;
        self.best_cost = F::infinity();
        self.best_iter = 0;
        self.best_cost_evals = 0;
        self.best_gradient_evals = 0;
    }
}

impl<F: Scalar> GradientState for ScalarGradientState<F> {
    fn gradient(&self) -> Option<&F> {
        self.gradient.as_ref()
    }

    fn gradient_evals(&self) -> u64 {
        self.gradient_evals
    }

    fn best_gradient_evals(&self) -> u64 {
        self.best_gradient_evals
    }
}

impl<F> CountsMirror for ScalarGradientState<F>
where
    ScalarGradientState<F>: State,
{
    fn mirror(&mut self, delta: &EvalCounts) {
        // Same rule as `BasicState`: residual folds into cost, Jacobian/
        // Hessian into gradient. A 1D derivative solver only does cost +
        // gradient work, so the residual/Jacobian/Hessian terms are 0 in
        // practice.
        self.cost_evals = delta.cost_evals + delta.residual_evals;
        self.gradient_evals = delta.gradient_evals + delta.jacobian_evals + delta.hessian_evals;
    }
}
