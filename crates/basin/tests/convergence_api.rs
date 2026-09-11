use basin::{
    BasicState, CostFunction, Executor, Gradient, GradientDescent, NelderMead,
    TerminationReason,
};

struct Sphere;

impl CostFunction for Sphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;

    fn cost(&self, x: &Self::Param) -> Result<f64, Self::Error> {
        Ok(x.iter().map(|x| x * x).sum())
    }
}

impl Gradient for Sphere {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Self::Param) -> Result<Self::Gradient, Self::Error> {
        Ok(x.iter().map(|x| 2.0 * x).collect())
    }
}

struct ConstantCost;

impl CostFunction for ConstantCost {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;

    fn cost(&self, _: &Vec<f64>) -> Result<f64, Self::Error> {
        Ok(0.0)
    }
}

impl Gradient for ConstantCost {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(vec![0.0; x.len()])
    }
}

#[test]
fn exact_step_checks_allow_overflowing_parameter_norms() {
    for (absolute, relative, expected) in [
        (Some(0.0), None, TerminationReason::ParamTolerance),
        (None, Some(0.0), TerminationReason::RelativeParamTolerance),
    ] {
        let solver = GradientDescent::new(0.1)
            .with_absolute_step_tolerance(absolute)
            .with_relative_step_tolerance(relative);
        let result = Executor::from_start(ConstantCost, solver, vec![1e200])
            .max_iter(2)
            .run()
            .unwrap();
        assert_eq!(result.reason, expected);
        assert_eq!(result.iter(), 1);
    }
}

#[test]
fn relative_step_checks_scale_large_parameters_before_taking_the_norm() {
    struct Linear;
    impl CostFunction for Linear {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
            Ok(x[1])
        }
    }
    impl Gradient for Linear {
        type Gradient = Vec<f64>;
        fn gradient(&self, _: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
            Ok(vec![0.0, 1.0])
        }
    }
    for (tolerance, expected) in [
        (0.0, TerminationReason::MaxIter),
        (1e-300, TerminationReason::MaxIter),
        (1e-200, TerminationReason::RelativeParamTolerance),
    ] {
        let solver =
            GradientDescent::new(0.5).with_relative_step_tolerance(tolerance);
        let result = Executor::from_start(Linear, solver, vec![1e200, 1.0])
            .max_iter(2)
            .run()
            .unwrap();
        assert_eq!(result.reason, expected);
    }
}

#[test]
fn step_checks_reject_nonfinite_steps() {
    for nonfinite in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let solver = GradientDescent::<_, Vec<f64>>::new(0.1)
            .with_absolute_step_tolerance(1e200)
            .with_relative_step_tolerance(1.0);
        let result =
            Executor::from_start(ConstantCost, solver, vec![nonfinite])
                .max_iter(2)
                .run()
                .unwrap();
        assert_eq!(result.reason, TerminationReason::MaxIter);
    }
}

#[test]
fn simplex_cost_check_does_not_require_vector_norms() {
    use basin::{BasicSimplexState, ScaleInPlace, ScaledAdd};

    #[derive(Clone)]
    struct Minimal(f64);
    impl ScaleInPlace for Minimal {
        fn scale_in_place(&mut self, scale: f64) {
            self.0 *= scale;
        }
    }
    impl ScaledAdd for Minimal {
        fn scaled_add(&mut self, scale: f64, other: &Self) {
            self.0 += scale * other.0;
        }
    }
    struct MinimalSphere;
    impl CostFunction for MinimalSphere {
        type Param = Minimal;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Minimal) -> Result<f64, Self::Error> {
            Ok(x.0 * x.0)
        }
    }

    for (tolerance, expected) in [
        (Some(0.0), TerminationReason::SimplexTolerance),
        (None, TerminationReason::MaxIter),
    ] {
        let direct = NelderMead::new()
            .with_absolute_simplex_cost_tolerance(1.0)
            .with_absolute_simplex_cost_tolerance(tolerance)
            .with_absolute_cost_change_tolerance(None);
        let configured = NelderMead::new()
            .with_absolute_cost_change_tolerance(None)
            .with_absolute_simplex_cost_tolerance(tolerance);
        for solver in [direct, configured] {
            let result = Executor::new(
                MinimalSphere,
                solver,
                BasicSimplexState::from_simplex(vec![
                    Minimal(-1.0),
                    Minimal(1.0),
                ]),
            )
            .max_iter(1)
            .run()
            .unwrap();
            assert_eq!(result.reason, expected);
        }
    }
}

#[test]
fn simplex_group_preserves_settings_in_both_setter_orders() {
    use basin::BasicSimplexState;

    for (size, cost, vertices, expected) in [
        (
            Some(0.0),
            Some(0.0),
            [-1.0, 1.0],
            TerminationReason::MaxIter,
        ),
        (Some(1.0), Some(0.0), [1.0, 2.0], TerminationReason::MaxIter),
        (
            Some(2.0),
            Some(0.0),
            [-1.0, 1.0],
            TerminationReason::SimplexTolerance,
        ),
        (
            None,
            Some(0.0),
            [-1.0, 1.0],
            TerminationReason::SimplexTolerance,
        ),
        (
            Some(1.0),
            None,
            [1.0, 2.0],
            TerminationReason::SimplexTolerance,
        ),
        (None, None, [1.0, 2.0], TerminationReason::MaxIter),
    ] {
        let size_first = NelderMead::new()
            .with_absolute_simplex_size_tolerance(100.0)
            .with_absolute_simplex_cost_tolerance(100.0)
            .with_absolute_simplex_size_tolerance(size)
            .with_absolute_simplex_cost_tolerance(cost);
        let cost_first = NelderMead::new()
            .with_absolute_simplex_cost_tolerance(cost)
            .with_absolute_simplex_size_tolerance(size);
        for solver in [size_first, cost_first] {
            let result = Executor::new(
                Sphere,
                solver,
                BasicSimplexState::from_simplex(
                    vertices.map(|x| vec![x]).to_vec(),
                ),
            )
            .max_iter(1)
            .run()
            .unwrap();
            assert_eq!(result.reason, expected);
        }
    }
}

#[test]
fn solver_gradient_tolerance_stops_before_first_step() {
    let solver =
        GradientDescent::new(0.1).with_absolute_gradient_tolerance(1e-8);
    let result = Executor::from_start(Sphere, solver, vec![0.0])
        .run()
        .unwrap();
    assert_eq!(result.reason, TerminationReason::GradientTolerance);
    assert_eq!(result.iter(), 0);
    assert_eq!(result.cost_evals(), 1);
}

#[test]
fn repeated_setters_replace_and_none_disables() {
    let solver = GradientDescent::new(0.1)
        .with_absolute_gradient_tolerance(100.0)
        .with_absolute_gradient_tolerance(None);
    let result = Executor::from_start(Sphere, solver, vec![1.0])
        .max_iter(2)
        .run()
        .unwrap();
    assert_eq!(result.reason, TerminationReason::MaxIter);
    assert_eq!(result.iter(), 2);
}

#[test]
fn direct_budget_observes_initialization() {
    let result =
        Executor::from_start(Sphere, GradientDescent::new(0.1), vec![1.0])
            .max_cost_evals(0)
            .run()
            .unwrap();
    assert_eq!(result.reason, TerminationReason::MaxCostEvals);
    assert_eq!(result.cost_evals(), 1);
    assert_eq!(result.iter(), 0);
}

#[test]
fn custom_stop_sees_initialized_state() {
    let result =
        Executor::from_start(Sphere, GradientDescent::new(0.1), vec![1.0])
            .stop_when(|state: &BasicState<Vec<f64>>| {
                (basin::State::cost(state) == 1.0)
                    .then_some(TerminationReason::UserRequested)
            })
            .run()
            .unwrap();
    assert_eq!(result.reason, TerminationReason::UserRequested);
    assert_eq!(result.iter(), 0);
}

#[test]
fn simplex_settings_converge_together() {
    let solver = NelderMead::new()
        .with_absolute_simplex_size_tolerance(1e-8)
        .with_absolute_simplex_cost_tolerance(1e-10);
    let result = Executor::from_start(Sphere, solver, vec![1.0, -2.0])
        .run()
        .unwrap();
    assert_eq!(result.reason, TerminationReason::SimplexTolerance);
    assert!(result.cost() < 1e-12);
}

#[test]
fn relative_gradient_history_resets_for_fresh_borrowed_runs() {
    use basin::{Problem, RunControl, run_loop_with_control};
    let mut solver =
        GradientDescent::new(0.1).with_relative_gradient_tolerance(0.25);
    let mut control = RunControl::new().max_iter(100);
    let mut problem = Problem::new(Sphere);
    for start in [1.0, 100.0, 0.001] {
        let result = run_loop_with_control(
            &mut problem,
            BasicState::new(vec![start]),
            &mut solver,
            &mut control,
        )
        .unwrap();
        assert_eq!(result.reason, TerminationReason::RelativeGradientTolerance);
        assert_eq!(result.iter(), 7);
        assert_eq!(result.cost_evals(), 8);
    }
}

#[test]
fn exact_resume_preserves_relative_gradient_anchor() {
    use basin::{ExactCheckpoint, Problem, RunControl, run_loop_with_control};
    let mut solver =
        GradientDescent::new(0.1).with_relative_gradient_tolerance(0.25);
    let mut problem = Problem::new(Sphere);
    let first = run_loop_with_control(
        &mut problem,
        BasicState::new(vec![1.0]),
        &mut solver,
        &mut RunControl::new().max_iter(3),
    )
    .unwrap();
    let checkpoint =
        ExactCheckpoint::from_parts(solver, first.state, *problem.counts());
    let exact = Executor::resume_from_checkpoint(Sphere, checkpoint)
        .max_iter(100)
        .run()
        .unwrap();
    assert_eq!(exact.iter(), 7);
    assert_eq!(exact.reason, TerminationReason::RelativeGradientTolerance);
}

#[test]
fn repeated_boundary_checks_do_not_create_zero_changes() {
    use basin::{
        ExactCheckpoint, Problem, RunControl, Solver, run_loop_with_control,
    };
    let mut solver = GradientDescent::new(0.1)
        .with_absolute_step_tolerance(0.0)
        .with_absolute_cost_change_tolerance(0.0);
    let mut problem = Problem::new(Sphere);
    let first = run_loop_with_control(
        &mut problem,
        BasicState::new(vec![1.0]),
        &mut solver,
        &mut RunControl::new().max_iter(1),
    )
    .unwrap();
    assert_eq!(solver.check_convergence(&problem, &first.state), None);
    assert_eq!(solver.check_convergence(&problem, &first.state), None);
    let checkpoint =
        ExactCheckpoint::from_parts(solver, first.state, *problem.counts());
    let resumed = Executor::resume_from_checkpoint(Sphere, checkpoint)
        .max_iter(2)
        .run()
        .unwrap();
    assert_eq!(resumed.reason, TerminationReason::MaxIter);
}

#[test]
fn inner_stop_factory_restarts_history_and_counts() {
    use basin::{InnerExecutor, Problem, State};
    let mut inner = InnerExecutor::new(GradientDescent::new(0.1))
        .max_gradient_evals(100)
        .stop_when_factory(|| {
            let mut calls = 0;
            move |_: &BasicState<Vec<f64>>| {
                calls += 1;
                (calls == 3).then_some(TerminationReason::UserRequested)
            }
        });
    let mut problem = Problem::new(Sphere);
    for _ in 0..2 {
        let result =
            inner.run(&mut problem, BasicState::new(vec![1.0])).unwrap();
        assert_eq!(result.reason, TerminationReason::UserRequested);
        assert_eq!(result.state.iter(), 2);
        assert_eq!(result.cost_evals(), 3);
    }
    assert_eq!(problem.counts().gradient_evals, 6);
}

#[test]
fn direct_budgets_take_precedence_over_convergence_and_hooks() {
    let solver =
        GradientDescent::new(0.1).with_absolute_gradient_tolerance(0.0);
    let result = Executor::from_start(Sphere, solver, vec![0.0])
        .max_gradient_evals(1)
        .stop_when(|_| Some(TerminationReason::UserRequested))
        .run()
        .unwrap();
    assert_eq!(result.reason, TerminationReason::MaxGradientEvals);
    assert_eq!(result.iter(), 0);
}

#[test]
fn simplex_size_and_cost_are_an_and_group() {
    use basin::BasicSimplexState;
    let solver = NelderMead::new()
        .with_absolute_simplex_size_tolerance(100.0)
        .with_absolute_simplex_cost_tolerance(0.0);
    let result = Executor::new(
        Sphere,
        solver,
        BasicSimplexState::from_simplex(vec![vec![1.0], vec![2.0]]),
    )
    .max_iter(1)
    .run()
    .unwrap();
    assert_eq!(result.reason, TerminationReason::MaxIter);
}

#[test]
#[allow(deprecated)] // Verify that the old zero-disable contract survives the migration.
fn native_zero_is_exact_and_legacy_zero_still_disables() {
    use basin::{DenseMatrix, GaussNewton, Jacobian, Residual};
    struct Linear;
    impl Residual for Linear {
        type Param = Vec<f64>;
        type Output = Vec<f64>;
        type Error = std::convert::Infallible;
        fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
            Ok(x.clone())
        }
    }
    impl Jacobian for Linear {
        type Jacobian = DenseMatrix;
        fn jacobian(&self, _: &Vec<f64>) -> Result<DenseMatrix, Self::Error> {
            Ok(DenseMatrix::from_row_slice(1, 1, &[1.0]))
        }
    }
    let exact = Executor::from_start(
        Linear,
        GaussNewton::new().with_absolute_gradient_tolerance(0.0),
        vec![0.0],
    )
    .max_iter(1)
    .run()
    .unwrap();
    assert_eq!(exact.reason, TerminationReason::SolverConverged);
    for solver in [
        GaussNewton::new().with_tol_grad(0.0),
        GaussNewton::new().with_absolute_gradient_tolerance(None),
    ] {
        let result = Executor::from_start(Linear, solver, vec![0.0])
            .max_iter(1)
            .run()
            .unwrap();
        assert_eq!(result.reason, TerminationReason::MaxIter);
    }
}

#[test]
fn nonfinite_gradient_is_not_convergence() {
    use basin::{InitialState, Problem, Solver};
    struct Nonfinite;
    impl CostFunction for Nonfinite {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, _: &Vec<f64>) -> Result<f64, Self::Error> {
            Ok(0.0)
        }
    }
    impl Gradient for Nonfinite {
        type Gradient = Vec<f64>;
        fn gradient(&self, _: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
            Ok(vec![f64::INFINITY])
        }
    }
    let mut solver =
        GradientDescent::new(0.1).with_relative_gradient_tolerance(1.0);
    let mut problem = Problem::new(Nonfinite);
    let state = solver.init(&mut problem, solver.seed(&vec![0.0])).unwrap();
    assert_eq!(solver.check_convergence(&problem, &state), None);
}

#[test]
fn cost_check_does_not_require_vector_norms() {
    use basin::{NegInPlace, ScaleInPlace, ScaledAdd};
    #[derive(Clone)]
    struct Minimal(f64);
    impl ScaledAdd for Minimal {
        fn scaled_add(&mut self, scale: f64, other: &Self) {
            self.0 += scale * other.0;
        }
    }
    impl ScaleInPlace for Minimal {
        fn scale_in_place(&mut self, scale: f64) {
            self.0 *= scale;
        }
    }
    impl NegInPlace for Minimal {
        fn neg_in_place(&mut self) {
            self.0 = -self.0;
        }
    }
    struct MinimalSphere;
    impl CostFunction for MinimalSphere {
        type Param = Minimal;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Minimal) -> Result<f64, Self::Error> {
            Ok(x.0 * x.0)
        }
    }
    impl Gradient for MinimalSphere {
        type Gradient = Minimal;
        fn gradient(&self, x: &Minimal) -> Result<Minimal, Self::Error> {
            Ok(Minimal(2.0 * x.0))
        }
    }
    let solver =
        GradientDescent::new(0.1).with_absolute_cost_change_tolerance(1e-6);
    let result = Executor::from_start(MinimalSphere, solver, Minimal(1.0))
        .run()
        .unwrap();
    assert_eq!(result.reason, TerminationReason::CostTolerance);
}

#[cfg(feature = "serde")]
#[test]
fn configured_solver_and_inner_budgets_round_trip() {
    use basin::{BasicSimplexState, InnerExecutor, Problem};
    let mut inner = InnerExecutor::new(
        NelderMead::new()
            .with_absolute_simplex_size_tolerance(1e-8)
            .with_absolute_cost_change_tolerance(1e-12),
    )
    .max_cost_evals(50);
    let encoded =
        bincode::serde::encode_to_vec(&inner, bincode::config::standard())
            .unwrap();
    let (mut decoded, _): (InnerExecutor<BasicSimplexState<Vec<f64>>, _>, _) =
        bincode::serde::decode_from_slice(
            &encoded,
            bincode::config::standard(),
        )
        .unwrap();
    std::mem::swap(&mut inner, &mut decoded);
    let expected = inner
        .run(&mut Problem::new(Sphere), BasicSimplexState::new(vec![1.0]))
        .unwrap();
    let result = decoded
        .run(&mut Problem::new(Sphere), BasicSimplexState::new(vec![1.0]))
        .unwrap();
    assert_eq!(result.iter(), expected.iter());
    assert_eq!(result.reason, expected.reason);
    assert_eq!(result.param(), expected.param());
    let with_hook = decoded.stop_when(|_| None);
    assert!(
        bincode::serde::encode_to_vec(&with_hook, bincode::config::standard())
            .is_err()
    );
}

#[test]
fn projected_check_uses_each_inner_problems_bounds() {
    use basin::{
        BoxConstraints, Problem, ProjectedGradientDescent, RunControl,
        run_loop_with_control,
    };
    struct BoundedSphere {
        lower: Vec<f64>,
        upper: Vec<f64>,
    }
    impl CostFunction for BoundedSphere {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
            Sphere.cost(x)
        }
    }
    impl Gradient for BoundedSphere {
        type Gradient = Vec<f64>;
        fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
            Sphere.gradient(x)
        }
    }
    impl BoxConstraints for BoundedSphere {
        fn lower(&self) -> &Vec<f64> {
            &self.lower
        }
        fn upper(&self) -> &Vec<f64> {
            &self.upper
        }
    }
    let mut solver = ProjectedGradientDescent::new(0.1)
        .with_absolute_projected_gradient_tolerance(0.0);
    let mut control = RunControl::new().max_iter(1);
    for (lower, expected) in [
        (1.0, TerminationReason::ProjectedGradientTolerance),
        (0.0, TerminationReason::MaxIter),
    ] {
        let mut problem = Problem::new(BoundedSphere {
            lower: vec![lower],
            upper: vec![2.0],
        });
        let result = run_loop_with_control(
            &mut problem,
            BasicState::new(vec![1.0]),
            &mut solver,
            &mut control,
        )
        .unwrap();
        assert_eq!(result.reason, expected);
    }
}

#[cfg(feature = "serde")]
#[test]
fn serialized_checkpoint_preserves_observed_convergence_history() {
    use basin::{
        BasicSimplexState, ExactCheckpoint, Problem, RunControl,
        run_loop_with_control,
    };
    fn round_trip<T: serde::Serialize + serde::de::DeserializeOwned>(
        value: &T,
    ) -> T {
        let bytes =
            bincode::serde::encode_to_vec(value, bincode::config::standard())
                .unwrap();
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
            .unwrap()
            .0
    }
    let mut solver = NelderMead::new()
        .with_absolute_step_tolerance(1e-10)
        .with_absolute_cost_change_tolerance(1e-14);
    let mut problem = Problem::new(Sphere);
    let first = run_loop_with_control(
        &mut problem,
        BasicSimplexState::new(vec![1.0, 2.0]),
        &mut solver,
        &mut RunControl::new().max_iter(4),
    )
    .unwrap();
    assert_eq!(first.reason, TerminationReason::MaxIter);
    let checkpoint =
        ExactCheckpoint::from_parts(solver, first.state, *problem.counts());
    let restored = round_trip(&checkpoint);
    let direct = Executor::resume_from_checkpoint(Sphere, checkpoint)
        .run()
        .unwrap();
    let decoded = Executor::resume_from_checkpoint(Sphere, restored)
        .run()
        .unwrap();
    assert_eq!(direct.reason, decoded.reason);
    assert_eq!(direct.iter(), decoded.iter());
    assert_eq!(direct.param(), decoded.param());
    assert_eq!(direct.cost_evals(), decoded.cost_evals());
}

#[test]
#[allow(deprecated)] // Preserve the historical owned-versus-borrowed reset contract.
fn legacy_criterion_history_is_preserved_by_owned_executors() {
    use basin::{InnerExecutor, Problem, TerminationCriterion};
    struct AlreadyChecked {
        ready: bool,
    }
    impl TerminationCriterion<BasicState<Vec<f64>>> for AlreadyChecked {
        fn check(
            &mut self,
            _: &BasicState<Vec<f64>>,
        ) -> Option<TerminationReason> {
            self.ready.then_some(TerminationReason::UserRequested)
        }
        fn reset(&mut self) {
            self.ready = false;
        }
    }
    let owned =
        Executor::from_start(Sphere, GradientDescent::new(0.1), vec![1.0])
            .max_iter(1)
            .terminate_on(AlreadyChecked { ready: true })
            .run()
            .unwrap();
    assert_eq!(owned.reason, TerminationReason::UserRequested);
    assert_eq!(owned.iter(), 0);
    let borrowed = InnerExecutor::new(GradientDescent::new(0.1))
        .max_iter(1)
        .terminate_on(AlreadyChecked { ready: true })
        .run(&mut Problem::new(Sphere), BasicState::new(vec![1.0]))
        .unwrap();
    assert_eq!(borrowed.reason, TerminationReason::MaxIter);
}
