//! [`CmaInject`] must bubble `SolverFailed` from the inner solver out
//! through the outer stopping report, retaining the nested failure
//! (CONTRIBUTING.md "Solver composition" rule 3).
//!
//! Uses the `AlwaysFails` harness (sibling of the one in
//! `tests/inner_executor.rs`) wrapped in [`ClosureInner`] for the
//! seeder closure; `AlwaysFails` is a one-off fixture that doesn't
//! have a dedicated [`MemeticInner`] impl. This is the S11-era
//! deferred test promoted to a real fixture (S11 hardwired NelderMead
//! and NelderMead never returns `SolverFailed`).

#![cfg(all(feature = "nalgebra_all", feature = "problems"))]

use crate::backend_aliases::nalgebra::{DMatrix, DVector};
use basin::problems::Sphere;
use basin::{
    ClosureInner, CmaEs, CmaInject, Executor, PointState, PopulationProgress,
    Problem, Solver, State, TerminationCode,
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
    ) -> Result<basin::SolverStep<S>, Self::Error> {
        Ok(basin::SolverStep::from((
            state,
            Some(basin::Termination::numerical_failure(
                "External solver reported a numerical failure.",
            )),
        )))
    }
}

#[test]
fn bubbles_inner_failure() {
    let m0 = DVector::from_vec(vec![1.0; 3]);

    let cma = CmaEs::<DVector<f64>, DMatrix<f64>>::new(5, 0.3);

    // Wrap AlwaysFails in ClosureInner with a PointState seeder.
    let inner =
        ClosureInner::new(AlwaysFails, |x: &DVector<f64>, _sigma: f64| {
            PointState::new(x.clone())
        });
    let solver = CmaInject::with_inner_solver(cma, inner);

    let result = Executor::new(
        Sphere::<DVector<f64>>::new(),
        solver,
        PopulationProgress::<DVector<f64>>::from_point(m0),
    )
    .max_iter(20)
    .run()
    .unwrap();

    assert_eq!(
        result.report.code(),
        TerminationCode::SolverFailed,
        "outer should bubble SolverFailed from the inner; got {:?}",
        result.report.code()
    );
    let basin::Termination::Failed(basin::NumericalFailure::Inner { report }) =
        &result.report.termination
    else {
        panic!("expected nested inner failure");
    };
    assert_eq!(
        report.stage,
        basin::TerminationStage::Step { completed: false }
    );
    assert!(
        matches!(&report.termination, basin::Termination::Failed(basin::NumericalFailure::Solver {message, ..}) if message == "External solver reported a numerical failure.")
    );
    // The first injection runs inside the first call to
    // `CmaInject::next_iter`, which bails mid-iter with SolverFailed;
    // per the executor contract the iter counter is left untouched,
    // so iter == 0.
    assert_eq!(result.iter(), 0, "expected iter = 0 (mid-iter bail)");
}

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
