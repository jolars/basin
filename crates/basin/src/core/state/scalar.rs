//! Single-iterate state for one-dimensional solvers.

use crate::core::math::Scalar;
use crate::core::problem::EvalCounts;
use crate::core::state::{CountsMirror, State};

/// State for one-dimensional, derivative-free solvers
/// ([`Brent`](crate::solver::Brent)): a scalar `param`, an optional cached
/// cost, and cost-evaluation counters. `Param` and `Float` are the same
/// scalar `F`.
///
/// This is the leanest single-iterate state — only cost evaluations, no
/// gradient and no residual / Jacobian. It deliberately does **not** impl
/// [`GradientState`](crate::core::state::GradientState): a 1D minimizer has no
/// gradient to populate, so attaching
/// [`GradientTolerance`](crate::core::termination::GradientTolerance) is a
/// compile error rather than a criterion that silently never fires (tenet 3).
///
/// [`best_param`](State::best_param) tracks the lowest-cost probe, which for a
/// bracketing search like Brent can differ from the final iterate.
///
/// The scalar `F` defaults to `f64` so `ScalarState` call sites resolve
/// unchanged.
pub struct ScalarState<F = f64> {
    pub(crate) param: F,
    pub(crate) cost: Option<F>,
    pub(crate) iter: u64,
    pub(crate) cost_evals: u64,
    pub(crate) best_param: Option<F>,
    pub(crate) best_cost: F,
    pub(crate) best_iter: u64,
    pub(crate) best_cost_evals: u64,
}

impl<F: Scalar> ScalarState<F> {
    /// Build a state at the given starting point. The cost is filled in by
    /// [`Solver::init`](crate::core::solver::Solver::init).
    pub fn new(param: F) -> Self {
        Self {
            param,
            cost: None,
            iter: 0,
            cost_evals: 0,
            best_param: None,
            best_cost: F::infinity(),
            best_iter: 0,
            best_cost_evals: 0,
        }
    }
}

impl<F: Scalar> State for ScalarState<F> {
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
            .expect("ScalarState::cost read before Solver::init populated it")
    }

    fn best_param(&self) -> &F {
        self.best_param
            .as_ref()
            .expect("ScalarState::best_param read before Solver::init populated it")
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

impl<F> CountsMirror for ScalarState<F>
where
    ScalarState<F>: State,
{
    fn mirror(&mut self, delta: &EvalCounts) {
        // A 1D minimizer does only cost work; the residual term is kept for
        // parity with the other single-iterate states and is 0 in practice.
        self.cost_evals = delta.cost_evals + delta.residual_evals;
    }
}
