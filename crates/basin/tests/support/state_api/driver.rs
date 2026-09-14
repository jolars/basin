use super::state::{ObjectiveBest, Progress};
use basin::core::math::Scalar;
use basin::{EvalCounts, ExactCheckpoint, Problem, Solver, TerminationReason};
use num_traits::Float;

/// A narrow experiment with publication and ownership-returning completion.
/// Execution hooks and clocks continue to belong to the production executor.
pub struct Run<P, S, So> {
    problem: Problem<P>,
    state: Option<S>,
    solver: So,
    stopped: Option<TerminationReason>,
}

pub struct Completed<S, So> {
    pub state: S,
    pub solver: So,
    pub counts: EvalCounts,
    pub reason: TerminationReason,
}

impl<S, So> Completed<S, So> {
    pub fn into_checkpoint(self) -> ExactCheckpoint<So, S> {
        ExactCheckpoint::from_parts(self.solver, self.state, self.counts)
    }
}

impl<P, S, So> Run<P, S, So>
where
    S: Progress,
    So: Solver<P, S>,
{
    pub fn from_start(
        problem: P,
        solver: So,
        point: S::Param,
    ) -> Result<Self, So::Error>
    where
        So: basin::InitialState<S::Param, State = S>,
    {
        let state = solver.seed(&point);
        Self::new(problem, solver, state)
    }

    pub fn new(
        problem: P,
        mut solver: So,
        mut state: S,
    ) -> Result<Self, So::Error> {
        let mut problem = Problem::new(problem);
        state.fresh();
        solver.reset_convergence();
        let mut state = solver.init(&mut problem, state)?;
        assert!(state.is_ready(), "solver published incomplete state");
        state.publish(false, *problem.counts());
        Ok(Self {
            problem,
            state: Some(state),
            solver,
            stopped: None,
        })
    }

    pub fn resume(problem: P, checkpoint: ExactCheckpoint<So, S>) -> Self {
        let (solver, state, counts) = checkpoint.into_parts();
        let mut problem = Problem::new(problem);
        *problem.counts_mut() = counts;
        Self {
            problem,
            state: Some(state),
            solver,
            stopped: None,
        }
    }

    pub fn state(&self) -> Option<&S> {
        self.state.as_ref()
    }
    pub fn solver(&self) -> &So {
        &self.solver
    }
    pub fn counts(&self) -> &EvalCounts {
        self.problem.counts()
    }

    pub fn step(&mut self) -> Result<Option<TerminationReason>, So::Error> {
        if let Some(reason) = self.stopped {
            return Ok(Some(reason));
        }
        let state =
            self.state.as_ref().expect("a failed run cannot be stepped");
        if let Some(reason) =
            self.solver.check_convergence(&self.problem, state)
        {
            self.stopped = Some(reason);
            return Ok(Some(reason));
        }
        let (mut state, reason) = self
            .solver
            .next_iter(&mut self.problem, self.state.take().unwrap())?;
        assert!(state.is_ready(), "solver published incomplete state");
        state.publish(reason.is_none(), *self.problem.counts());
        self.state = Some(state);
        self.stopped = reason;
        Ok(reason)
    }

    pub fn into_checkpoint(self) -> Option<ExactCheckpoint<So, S>> {
        Some(ExactCheckpoint::from_parts(
            self.solver,
            self.state?,
            *self.problem.counts(),
        ))
    }

    pub fn run(mut self, max_iter: u64) -> Result<Completed<S, So>, So::Error> {
        let reason = loop {
            if let Some(reason) = self.stopped {
                break reason;
            }
            if self.state.as_ref().unwrap().iter() >= max_iter {
                break TerminationReason::MaxIter;
            }
            if let Some(reason) = self.step()? {
                break reason;
            }
        };
        Ok(Completed {
            state: self.state.unwrap(),
            solver: self.solver,
            counts: *self.problem.counts(),
            reason,
        })
    }

    pub fn cost_budget_reached(&self, limit: u64) -> bool {
        self.problem.counts().cost_evals >= limit
    }

    pub fn target_reached(&self, target: S::Float) -> bool
    where
        S: ObjectiveBest,
        S::Float: Scalar,
    {
        assert!(target.is_finite());
        self.state
            .as_ref()
            .and_then(ObjectiveBest::objective_best)
            .is_some_and(|cost| cost <= target)
    }
}
