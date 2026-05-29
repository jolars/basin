//! [`CmaInject`] must bubble `SolverFailed` from the inner solver out
//! through the outer's mid-iter `Option<TerminationReason>` return
//! (AGENTS.md "Solver composition" rule 3).
//!
//! Uses the `AlwaysFails` harness (sibling of the one in
//! `tests/inner_executor.rs`) wrapped in [`ClosureInner`] for the
//! seeder + work-unit closures — `AlwaysFails` is a one-off fixture
//! that doesn't have a dedicated [`MemeticInner`] impl. This is the
//! S11-era deferred test promoted to a real fixture (S11 hardwired
//! NelderMead and NelderMead never returns `SolverFailed`).

#![cfg(feature = "nalgebra")]

use basin::problems::Sphere;
use basin::{
    BasicPopulationState, BasicState, ClosureInner, CmaEs, CmaInject, Executor, Problem, Solver,
    State, TerminationReason,
};
use nalgebra::{DMatrix, DVector};

/// Inner solver that always returns `SolverFailed` on the first
/// `next_iter` call. Same shape as the `AlwaysFails` in
/// `tests/inner_executor.rs:164`.
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
    let m0 = DVector::from_vec(vec![1.0; 3]);
    let n = 3usize;
    let lambda = CmaEs::<DVector<f64>, DMatrix<f64>>::default_lambda(n);

    let cma = CmaEs::<DVector<f64>, DMatrix<f64>>::new(m0, 0.3, 5);

    // Wrap AlwaysFails in ClosureInner with a BasicState seeder. The
    // work-unit closure reads cost_evals only since AlwaysFails
    // doesn't touch the gradient counter.
    let inner = ClosureInner::new(
        AlwaysFails,
        |x: &DVector<f64>, _sigma: f64| BasicState::new(x.clone()),
        |s: &BasicState<DVector<f64>>| s.cost_evals(),
    );
    let solver = CmaInject::with_inner_solver(cma, inner);

    let result = Executor::new(
        Sphere::<DVector<f64>>::new(),
        solver,
        BasicPopulationState::<DVector<f64>>::with_size(lambda),
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
    // `CmaInject::next_iter`, which bails mid-iter with SolverFailed;
    // per the executor contract the iter counter is left untouched,
    // so iter == 0.
    assert_eq!(result.iter(), 0, "expected iter = 0 (mid-iter bail)");
}
