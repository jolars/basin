use super::{ObservationEvent, Observe, ObserveSolver};
use crate::core::termination::TerminationReason;

// Adapting state observers keeps both kinds in one registration-ordered list
// with one dynamic call per callback and no bounds on state or solver storage.
pub(crate) struct StateObserver<O>(pub(crate) O);

impl<S, So, O: Observe<S>> ObserveSolver<S, So> for StateObserver<O> {
    fn observe_init(&mut self, state: &S, _solver: &So) {
        self.0.observe_init(state);
    }

    fn observe_iter(&mut self, state: &S, _solver: &So) {
        self.0.observe_iter(state);
    }

    fn observe_final(
        &mut self,
        state: &S,
        _solver: &So,
        reason: &TerminationReason,
    ) {
        self.0.observe_final(state, reason);
    }
}

pub(crate) struct SolverCallback<C>(pub(crate) C);

impl<S, So, C> ObserveSolver<S, So> for SolverCallback<C>
where
    C: FnMut(&S, &So, ObservationEvent),
{
    fn observe_init(&mut self, state: &S, solver: &So) {
        (self.0)(state, solver, ObservationEvent::Init);
    }

    fn observe_iter(&mut self, state: &S, solver: &So) {
        (self.0)(state, solver, ObservationEvent::Iter);
    }

    fn observe_final(
        &mut self,
        state: &S,
        solver: &So,
        reason: &TerminationReason,
    ) {
        (self.0)(state, solver, ObservationEvent::Final(*reason));
    }
}
