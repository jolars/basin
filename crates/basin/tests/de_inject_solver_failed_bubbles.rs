//! [`DeInject`] must bubble `SolverFailed` from the inner solver out
//! through the outer's mid-iter `Option<TerminationReason>` return
//! (CONTRIBUTING.md "Solver composition" rule 3).
//!
//! Mirror of `cma_inject_solver_failed_bubbles.rs`: same
//! `AlwaysFails` fixture wrapped in [`ClosureInner`] for the seeder.

#![cfg(feature = "nalgebra")]

use basin::problems::RastriginBoxed;
use basin::{
    BasicPopulationState, BasicState, ClosureInner, De, DeInject, Executor,
    Problem, Solver, State, TerminationReason,
};
use nalgebra::DVector;

/// Inner solver that always returns `SolverFailed` on the first
/// `next_iter` call. Same shape as the `AlwaysFails` in
/// `tests/inner_executor.rs` and `tests/cma_inject_solver_failed_bubbles.rs`.
struct AlwaysFails;

impl<P, S: State> Solver<P, S> for AlwaysFails {
    type Error = std::convert::Infallible;
    fn next_iter(
        &mut self,
        _problem: &mut Problem<P>,
        state: S,
    ) -> Result<(S, Option<TerminationReason>), Self::Error> {
        Ok((state, Some(TerminationReason::SolverFailed)))
    }
}

#[test]
fn bubbles_inner_failure() {
    let problem = RastriginBoxed::<DVector<f64>>::with_standard_bounds(3);

    let de = De::new(5).with_pop_size(8);
    let inner =
        ClosureInner::new(AlwaysFails, |x: &DVector<f64>, _sigma: f64| {
            BasicState::new(x.clone())
        });
    let solver = DeInject::with_inner_solver(de, inner);

    let result = Executor::new(
        problem,
        solver,
        BasicPopulationState::<DVector<f64>>::with_size(1),
    )
    .max_iter(20)
    .run()
    .unwrap();

    assert_eq!(
        result.reason,
        TerminationReason::SolverFailed,
        "outer should bubble SolverFailed from the inner; got {:?}",
        result.reason
    );
    // The first injection runs inside the first call to
    // `DeInject::next_iter`, which bails mid-iter with SolverFailed;
    // per the executor contract the iter counter is left untouched,
    // so iter == 0.
    assert_eq!(result.iter(), 0, "expected iter = 0 (mid-iter bail)");
}
