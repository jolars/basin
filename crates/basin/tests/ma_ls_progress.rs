//! Shared outer progress and solver-owned persistent local-search chains.
use basin::{
    BoxConstraints, CostFunction, CountsMirror, EvalCounts, Executor, MaLsChSw,
    PopulationProgress, State,
};
use std::convert::Infallible;
struct Sphere {
    lo: Vec<f64>,
    hi: Vec<f64>,
}
impl Sphere {
    fn new(n: usize) -> Self {
        Self {
            lo: vec![-5.0; n],
            hi: vec![5.0; n],
        }
    }
}
impl CostFunction for Sphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        assert_eq!(x.len(), self.lo.len());
        Ok(x.iter().map(|v| v * v).sum())
    }
}
impl BoxConstraints for Sphere {
    fn lower(&self) -> &Vec<f64> {
        &self.lo
    }
    fn upper(&self) -> &Vec<f64> {
        &self.hi
    }
}
#[test]
fn fresh_run_resets_outer_iteration_and_chain_history() {
    let prior = Executor::new(
        Sphere::new(2),
        MaLsChSw::new(71)
            .with_pop_size(4)
            .with_ls_intensity(8)
            .with_nfrec(2),
        PopulationProgress::empty(),
    )
    .max_iter(4)
    .run_with_solver()
    .unwrap();
    let fresh = Executor::new(Sphere::new(2), prior.solver, prior.state)
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    assert_eq!(fresh.iter(), 0);
    assert_eq!(fresh.counts.cost_evals, 4);
}
#[test]
fn all_categories_remain_separate() {
    let mut state = PopulationProgress::<Vec<f64>>::empty();
    let counts = EvalCounts {
        cost_evals: 2,
        gradient_evals: 3,
        residual_evals: 4,
        jacobian_evals: 5,
        hessian_evals: 6,
        hessian_product_evals: 7,
    };
    state.mirror(&counts);
    assert_eq!(state.cost_evals(), 2);
}

fn coherent<S: basin::ResumableInner<Vec<f64>>>(
    state: &PopulationProgress<Vec<f64>>,
    solver: &basin::MaLsCh<Vec<f64>, S>,
) {
    for (i, (x, &f)) in state.candidates().iter().zip(state.costs()).enumerate()
    {
        assert_eq!(f, x.iter().map(|v| v * v).sum::<f64>());
        if let Some((_, inner)) = solver.chain(i) {
            assert!(solver.ls_application_count(i) > 0);
            assert_eq!(inner.best_cost(), f);
            assert_eq!(inner.best_param(), x);
        }
    }
    let (best, f) = state.best().unwrap();
    assert_eq!(f, best.iter().map(|v| v * v).sum::<f64>());
}

macro_rules! lifecycle {
    ($name:ident, $constructor:expr) => {
        #[test]
        fn $name() {
            let solver = || {
                $constructor
                    .with_pop_size(4)
                    .with_nfrec(3)
                    .with_ls_intensity(18)
                    .with_ls_improvement_threshold(0.0)
            };
            let make = || {
                Executor::new(
                    Sphere::new(2),
                    solver(),
                    PopulationProgress::empty(),
                )
            };
            let expected = make().max_iter(14).run_with_solver().unwrap();
            coherent(&expected.state, &expected.solver);
            for split in [0, 1, 4, 13] {
                let checkpoint = make()
                    .max_iter(split)
                    .run_with_solver()
                    .unwrap()
                    .into_checkpoint();
                #[cfg(feature = "serde")]
                let checkpoint = {
                    fn restore<
                        T: serde::Serialize + serde::de::DeserializeOwned,
                    >(
                        value: T,
                    ) -> T {
                        postcard::from_bytes(
                            &postcard::to_allocvec(&value).unwrap(),
                        )
                        .unwrap()
                    }
                    restore(checkpoint)
                };
                let resumed = Executor::resume_from_checkpoint(
                    Sphere::new(2),
                    checkpoint,
                )
                .require_evaluated_state()
                .max_iter(14)
                .run_with_solver()
                .unwrap();
                coherent(&resumed.state, &resumed.solver);
                assert_eq!(resumed.state, expected.state);
                assert_eq!(resumed.counts, expected.counts);
                #[cfg(feature = "serde")]
                assert_eq!(
                    postcard::to_allocvec(&resumed.solver).unwrap(),
                    postcard::to_allocvec(&expected.solver).unwrap()
                );
            }
            let warm = expected.state.candidates().to_vec();
            let fresh =
                Executor::new(Sphere::new(2), expected.solver, expected.state)
                    .max_iter(0)
                    .run_with_solver()
                    .unwrap();
            let initialized = Executor::new(
                Sphere::new(2),
                solver(),
                PopulationProgress::from_population(warm),
            )
            .max_iter(0)
            .run_with_solver()
            .unwrap();
            assert_eq!(fresh.state, initialized.state);
            for i in 0..4 {
                assert_eq!(fresh.solver.ls_application_count(i), 0);
                assert!(fresh.solver.chain(i).is_none());
            }
            #[cfg(feature = "serde")]
            assert_eq!(
                postcard::to_allocvec(&fresh.solver).unwrap(),
                postcard::to_allocvec(&initialized.solver).unwrap()
            );
            let resized = Executor::new(
                Sphere::new(3),
                fresh.solver,
                PopulationProgress::empty(),
            )
            .max_iter(2)
            .run_with_solver()
            .unwrap();
            let initialized = Executor::new(
                Sphere::new(3),
                solver(),
                PopulationProgress::empty(),
            )
            .max_iter(2)
            .run_with_solver()
            .unwrap();
            assert_eq!(resized.state, initialized.state);
            #[cfg(feature = "serde")]
            assert_eq!(
                postcard::to_allocvec(&resized.solver).unwrap(),
                postcard::to_allocvec(&initialized.solver).unwrap()
            );
        }
    };
}
lifecycle!(
    sw_chains_continue_exactly_and_fresh_runs_drop_history,
    basin::MaLsChSw::<Vec<f64>>::new(71)
);
lifecycle!(
    cma_chains_continue_exactly_and_fresh_runs_drop_history,
    basin::MaLsChCma::<Vec<f64>, basin::DenseMatrix>::new(71)
);

#[test]
fn explicit_population_is_repaired_and_incumbent_counts_match_publication() {
    let result = Executor::new(
        Sphere::new(2),
        MaLsChSw::new(71).with_pop_size(4),
        PopulationProgress::from_population(vec![
            vec![9.0, 0.0],
            vec![-9.0, 0.0],
            vec![1.0; 2],
            vec![2.0; 2],
        ]),
    )
    .max_iter(0)
    .run_with_solver()
    .unwrap();
    coherent(&result.state, &result.solver);
    assert_eq!(
        result.state.candidates(),
        &[vec![1.0; 2], vec![2.0; 2], vec![5.0, 0.0], vec![-5.0, 0.0]]
    );
    assert_eq!(result.state.best_counts(), Some(&result.counts));
    assert_eq!(result.state.best_iter(), 0);
    assert_eq!(result.counts.cost_evals, 4);
}

#[test]
fn nonfinite_chain_without_an_incumbent_does_not_read_best_param() {
    struct Nonfinite(Sphere);
    impl CostFunction for Nonfinite {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = Infallible;
        fn cost(&self, _: &Vec<f64>) -> Result<f64, Infallible> {
            Ok(f64::INFINITY)
        }
    }
    impl BoxConstraints for Nonfinite {
        fn lower(&self) -> &Vec<f64> {
            &self.0.lo
        }
        fn upper(&self) -> &Vec<f64> {
            &self.0.hi
        }
    }
    let result = Executor::new(
        Nonfinite(Sphere::new(2)),
        basin::MaLsChCma::<_, basin::DenseMatrix>::new(71)
            .with_pop_size(4)
            .with_nfrec(1)
            .with_ls_intensity(8),
        PopulationProgress::empty(),
    )
    .max_iter(3)
    .run()
    .unwrap();
    assert_eq!(result.state.best(), None);
    assert_eq!(result.state.best_counts(), None);
    assert_eq!(result.iter(), 3);
}

#[test]
fn inner_failure_preserves_every_category_and_current_population() {
    struct Failing;
    impl basin::ResumableInner<Vec<f64>> for Failing {
        type State = basin::PointState<Vec<f64>>;
        fn seed_chain(
            &self,
            x: &Vec<f64>,
            _: f64,
            _: f64,
            _: u64,
        ) -> (Self, Self::State) {
            (Self, basin::PointState::new(x.clone()))
        }
        fn prepare_resume(&self, _: &mut Self::State) {}
    }
    impl basin::Solver<Sphere, basin::PointState<Vec<f64>>> for Failing {
        type Error = Infallible;
        fn init(
            &mut self,
            problem: &mut basin::Problem<Sphere>,
            mut state: basin::PointState<Vec<f64>>,
        ) -> Result<basin::PointState<Vec<f64>>, Infallible> {
            let cost = problem.cost(state.param())?;
            state.replace(state.param().clone(), cost);
            Ok(state)
        }
        fn next_iter(
            &mut self,
            problem: &mut basin::Problem<Sphere>,
            state: basin::PointState<Vec<f64>>,
        ) -> Result<
            (
                basin::PointState<Vec<f64>>,
                Option<basin::TerminationReason>,
            ),
            Infallible,
        > {
            // A custom adapter can report work performed by an external evaluator.
            let counts = problem.counts_mut();
            counts.gradient_evals += 2;
            counts.residual_evals += 3;
            counts.jacobian_evals += 4;
            counts.hessian_evals += 5;
            counts.hessian_product_evals += 6;
            Ok((state, Some(basin::TerminationReason::SolverFailed)))
        }
    }
    let solver = basin::MaLsCh::with_inner(71, Failing)
        .with_pop_size(4)
        .with_nfrec(1)
        .with_ls_intensity(8);
    let result =
        Executor::new(Sphere::new(2), solver, PopulationProgress::empty())
            .max_iter(4)
            .run_with_solver()
            .unwrap();
    coherent(&result.state, &result.solver);
    assert_eq!(result.reason, basin::TerminationReason::SolverFailed);
    assert_eq!(
        result.counts,
        EvalCounts {
            cost_evals: 6,
            gradient_evals: 2,
            residual_evals: 3,
            jacobian_evals: 4,
            hessian_evals: 5,
            hessian_product_evals: 6
        }
    );
    assert_eq!(result.state.counts(), &result.counts);
    assert_eq!(result.state.cost_evals(), 6);
}
