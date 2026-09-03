//! [`CmaInject`] must bubble `SolverFailed` from the inner solver out
//! through the outer's mid-iter `Option<TerminationReason>` return
//! (CONTRIBUTING.md "Solver composition" rule 3).
//!
//! Uses the `AlwaysFails` harness (sibling of the one in
//! `tests/inner_executor.rs`) wrapped in [`ClosureInner`] for the
//! seeder closure; `AlwaysFails` is a one-off fixture that doesn't
//! have a dedicated [`MemeticInner`] impl. This is the S11-era
//! deferred test promoted to a real fixture (S11 hardwired NelderMead
//! and NelderMead never returns `SolverFailed`).

#![cfg(feature = "nalgebra_all")]

use crate::backend_aliases::nalgebra::{DMatrix, DVector};
use basin::problems::Sphere;
use basin::{
    BasicState, ClosureInner, CmaEs, CmaEsState, CmaInject, Executor, Problem,
    Solver, State, TerminationReason,
};

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

    let cma = CmaEs::<DVector<f64>, DMatrix<f64>>::new(5);

    // Wrap AlwaysFails in ClosureInner with a BasicState seeder.
    let inner =
        ClosureInner::new(AlwaysFails, |x: &DVector<f64>, _sigma: f64| {
            BasicState::new(x.clone())
        });
    let solver = CmaInject::with_inner_solver(cma, inner);

    let result = Executor::new(
        Sphere::<DVector<f64>>::new(),
        solver,
        CmaEsState::<DVector<f64>, DMatrix<f64>>::new(m0, 0.3),
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

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
