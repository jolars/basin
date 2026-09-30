//! GBNM progress and solver-owned restart history.

use basin::{BoxConstraints, CostFunction, Executor, Gbnm, PointState, State};
use std::convert::Infallible;

struct FlatBox {
    lower: Vec<f64>,
    upper: Vec<f64>,
    constant: Option<f64>,
}
impl FlatBox {
    fn new(n: usize) -> Self {
        Self {
            lower: vec![-5.0; n],
            upper: vec![5.0; n],
            constant: Some(1.0),
        }
    }
    fn quadratic(n: usize) -> Self {
        Self {
            constant: None,
            ..Self::new(n)
        }
    }
}
impl CostFunction for FlatBox {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        assert!(x.iter().all(|x| (-5.0..=5.0).contains(x)));
        Ok(self
            .constant
            .unwrap_or_else(|| x.iter().map(|x| x * x).sum()))
    }
}
impl BoxConstraints for FlatBox {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

#[test]
fn fresh_gbnm_reevaluates_the_current_point_and_resets_progress() {
    let prior =
        Executor::from_start(FlatBox::new(2), Gbnm::new(31), vec![2.0, 2.0])
            .max_iter(7)
            .run_with_solver()
            .unwrap();
    let seed = prior.state.param().clone();
    let fresh = Executor::new(FlatBox::new(2), prior.solver, prior.state)
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    assert_eq!(fresh.iter(), 0);
    assert_eq!(fresh.state.best_iter(), 0);
    assert_eq!(fresh.counts.cost_evals, 3);
    assert_eq!(fresh.solver.search_starts(), &[seed]);
    assert_eq!(fresh.solver.restart_count(), 0);
}

#[test]
fn reused_gbnm_restarts_its_rng_and_history() {
    let make =
        || Executor::from_start(FlatBox::new(2), Gbnm::new(31), vec![2.0, 2.0]);
    let expected = make().max_iter(20).run_with_solver().unwrap();
    let used = make().max_iter(7).run_with_solver().unwrap();
    let restarted = Executor::new(
        FlatBox::new(2),
        used.solver,
        PointState::new(vec![2.0, 2.0]),
    )
    .max_iter(20)
    .run_with_solver()
    .unwrap();
    assert_eq!(
        restarted.solver.search_starts(),
        expected.solver.search_starts()
    );
    assert_eq!(restarted.solver.vertices(), expected.solver.vertices());
    assert_eq!(restarted.solver.costs(), expected.solver.costs());
    assert_eq!(restarted.counts, expected.counts);
}

#[test]
fn worse_restarts_retain_matching_incumbent_and_raw_publication_metadata() {
    let result = Executor::from_start(
        FlatBox::quadratic(2),
        Gbnm::new(31).with_absolute_simplex_cost_tolerance(100.0),
        vec![0.0, 0.0],
    )
    .require_evaluated_state()
    .max_iter(5)
    .run_with_solver()
    .unwrap();
    assert_eq!(result.solver.restart_count(), 5);
    assert!(result.state.current().unwrap().1 > 0.0);
    assert_eq!(result.state.best(), Some((&vec![0.0, 0.0], 0.0)));
    assert_eq!(result.state.best_iter(), 0);
    assert_eq!(result.state.best_counts().unwrap().cost_evals, 3);
    assert_eq!(result.state.counts(), &result.counts);
    assert_eq!(result.state.cost_evals(), 18);
    let (point, cost) = result.state.current().unwrap();
    assert_eq!(cost, point.iter().map(|x| x * x).sum());
    assert_eq!(point, &result.solver.vertices()[0]);
    assert_eq!(cost, result.solver.costs()[0]);
}

#[test]
fn nonfinite_publications_have_checked_incumbent_availability() {
    for cost in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let result = Executor::from_start(
            FlatBox {
                constant: Some(cost),
                ..FlatBox::new(2)
            },
            Gbnm::new(7),
            vec![0.0, 0.0],
        )
        .require_evaluated_state()
        .max_iter(0)
        .run_with_solver()
        .unwrap();
        assert!(result.state.current().is_some());
        if cost == f64::NEG_INFINITY {
            assert_eq!(result.state.best().unwrap().1, cost);
        } else {
            assert!(result.state.best().is_none());
            assert!(result.state.best_counts().is_none());
        }
        assert_eq!(result.state.counts().cost_evals, 3);
        assert_eq!(result.state.counts().total_work(), 3);
    }
}

#[test]
fn reused_solver_rebuilds_local_geometry_for_a_different_dimension() {
    let used = Executor::from_start(
        FlatBox::quadratic(2),
        Gbnm::new(31),
        vec![2.0, 2.0],
    )
    .max_iter(20)
    .run_with_solver()
    .unwrap();
    let fresh = Executor::from_start(
        FlatBox::quadratic(3),
        used.solver,
        vec![1.0, 2.0, 3.0],
    )
    .require_evaluated_state()
    .max_iter(50)
    .run_with_solver()
    .unwrap();
    let expected = Executor::from_start(
        FlatBox::quadratic(3),
        Gbnm::new(31),
        vec![1.0, 2.0, 3.0],
    )
    .require_evaluated_state()
    .max_iter(50)
    .run_with_solver()
    .unwrap();
    assert_eq!(fresh.state, expected.state);
    assert_eq!(fresh.solver.vertices(), expected.solver.vertices());
    assert_eq!(fresh.solver.costs(), expected.solver.costs());
    assert_eq!(
        fresh.solver.search_starts(),
        expected.solver.search_starts()
    );
    assert_eq!(fresh.counts, expected.counts);
}

#[test]
fn owned_checkpoints_preserve_local_search_and_restart_sequences() {
    for constant in [None, Some(1.0)] {
        let problem = || FlatBox {
            constant,
            ..FlatBox::new(2)
        };
        let make =
            || Executor::from_start(problem(), Gbnm::new(31), vec![2.0, 2.0]);
        let expected = make().max_iter(60).run_with_solver().unwrap();
        for split in [0, 1, 7, 23, 59] {
            let checkpoint = make()
                .max_iter(split)
                .run_with_solver()
                .unwrap()
                .into_checkpoint();
            let resumed =
                Executor::resume_from_checkpoint(problem(), checkpoint)
                    .require_evaluated_state()
                    .max_iter(60)
                    .run_with_solver()
                    .unwrap();
            assert_eq!(resumed.state, expected.state);
            assert_eq!(resumed.solver.vertices(), expected.solver.vertices());
            assert_eq!(resumed.solver.costs(), expected.solver.costs());
            assert_eq!(
                resumed.solver.search_starts(),
                expected.solver.search_starts()
            );
            assert_eq!(
                resumed.solver.local_optima(),
                expected.solver.local_optima()
            );
            assert_eq!(
                resumed.solver.restart_count(),
                expected.solver.restart_count()
            );
            assert_eq!(resumed.counts, expected.counts);
        }
    }
}

#[cfg(feature = "serde")]
#[test]
fn serialized_checkpoints_retain_models_while_progress_only_restarts() {
    type Checkpoint =
        basin::ExactCheckpoint<Gbnm<Vec<f64>>, PointState<Vec<f64>>>;
    let make = || {
        Executor::from_start(
            FlatBox::quadratic(2),
            Gbnm::new(31),
            vec![2.0, 2.0],
        )
    };
    let expected = make().max_iter(100).run_with_solver().unwrap();
    let checkpoint = make()
        .max_iter(71)
        .run_with_solver()
        .unwrap()
        .into_checkpoint();
    let bytes = postcard::to_allocvec(&checkpoint).unwrap();
    let restored: Checkpoint = postcard::from_bytes(&bytes).unwrap();
    let resumed =
        Executor::resume_from_checkpoint(FlatBox::quadratic(2), restored)
            .require_evaluated_state()
            .max_iter(100)
            .run_with_solver()
            .unwrap();
    assert_eq!(resumed.state, expected.state);
    assert_eq!(resumed.counts, expected.counts);
    assert_eq!(
        postcard::to_allocvec(&resumed.solver).unwrap(),
        postcard::to_allocvec(&expected.solver).unwrap()
    );
    let seed = resumed.state.param().clone();
    let fresh =
        Executor::new(FlatBox::quadratic(2), resumed.solver, resumed.state)
            .require_evaluated_state()
            .max_iter(0)
            .run_with_solver()
            .unwrap();
    assert_eq!(fresh.iter(), 0);
    assert_eq!(fresh.solver.search_starts(), &[seed]);
    assert!(fresh.solver.local_optima().is_empty());
    assert_eq!(fresh.solver.restart_count(), 0);
    assert_eq!(fresh.counts.cost_evals, 3);
}
