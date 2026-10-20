use basin::{
    AugmentedLagrangianMethod, BarrierMethod, CostFunction, DenseMatrix,
    Executor, Gradient, GradientDescent, Scalar, State, TerminationCode,
};
use std::convert::Infallible;

struct Quadratic<F> {
    a: DenseMatrix<F>,
    b: Vec<F>,
    scale: F,
}

impl<F: Scalar> basin::Constraints for Quadratic<F> {
    type Set = basin::NoConstraints;
    fn constraints(&self) -> basin::NoConstraints {
        basin::NoConstraints::EMPTY
    }
}

impl<F: Scalar> CostFunction for Quadratic<F> {
    type Param = Vec<F>;
    type Output = F;
    type Error = Infallible;

    fn cost(&self, x: &Vec<F>) -> Result<F, Infallible> {
        Ok(self.scale * (x[0] * x[0] + (x[1] - F::one()).powi(2)))
    }
}

impl<F: Scalar> Gradient for Quadratic<F> {
    type Gradient = Vec<F>;

    fn gradient(&self, x: &Vec<F>) -> Result<Vec<F>, Infallible> {
        let two = (F::one() + F::one()) * self.scale;
        Ok(vec![two * x[0], two * (x[1] - F::one())])
    }
}

type Equality<F> = basin::Constrained<
    Quadratic<F>,
    basin::ConstraintSet<
        basin::NoBounds,
        basin::LinearBlocks<
            basin::LinearBlock<DenseMatrix<F>, Vec<F>>,
            basin::NoLinearBlock,
        >,
    >,
>;
type Inequality<F> = basin::Constrained<
    Quadratic<F>,
    basin::ConstraintSet<
        basin::NoBounds,
        basin::LinearBlocks<
            basin::NoLinearBlock,
            basin::LinearBlock<DenseMatrix<F>, Vec<F>>,
        >,
    >,
>;
fn with_eq<F: Scalar>(p: Quadratic<F>) -> Equality<F> {
    let block = basin::LinearBlock::new(p.a.clone(), p.b.clone());
    basin::Constrained::new(
        p,
        basin::ConstraintSet::new().with_linear_equalities(block),
    )
}
fn with_ineq<F: Scalar>(p: Quadratic<F>) -> Inequality<F> {
    let block = basin::LinearBlock::new(p.a.clone(), p.b.clone());
    basin::Constrained::new(
        p,
        basin::ConstraintSet::new().with_linear_inequalities(block),
    )
}
fn equality<F: Scalar>() -> Equality<F> {
    with_eq(Quadratic {
        a: DenseMatrix::from_row_slice(1, 2, &[F::one(), F::zero()]),
        b: vec![F::zero()],
        scale: F::one(),
    })
}

fn inequality<F: Scalar>() -> Inequality<F> {
    with_ineq(Quadratic {
        a: DenseMatrix::from_row_slice(1, 2, &[F::one(), F::zero()]),
        b: vec![F::one()],
        scale: F::one(),
    })
}

fn false_outer_success<F: Scalar>() {
    let start = vec![F::zero(), F::from_f64(3.0).unwrap()];
    let alm = AugmentedLagrangianMethod::with_inner_solver(
        GradientDescent::new(F::zero())
            .with_absolute_cost_change_tolerance(F::zero()),
    )
    .with_inner_max_iter(2)
    .with_absolute_cost_change_tolerance(F::zero())
    .with_absolute_step_tolerance(F::zero());
    let result = Executor::from_start(equality::<F>(), alm, start.clone())
        .max_iter(12)
        .run()
        .unwrap();
    assert_eq!(result.report.code(), TerminationCode::MaxIter);
    assert_eq!(result.param(), &start);

    let barrier = BarrierMethod::with_inner_solver(
        GradientDescent::new(F::zero())
            .with_absolute_cost_change_tolerance(F::zero()),
    )
    .with_inner_max_iter(2)
    .with_absolute_cost_change_tolerance(F::zero())
    .with_absolute_step_tolerance(F::zero());
    let result =
        Executor::from_start(inequality::<F>(), barrier, start.clone())
            .max_iter(12)
            .run()
            .unwrap();
    assert_eq!(result.report.code(), TerminationCode::MaxIter);
    assert_eq!(result.param(), &start);
}

fn accurate_partial_inner<F: Scalar>() {
    let start = vec![F::zero(), F::from_f64(3.0).unwrap()];
    let alm = AugmentedLagrangianMethod::with_inner_solver(
        GradientDescent::new(F::from_f64(0.5).unwrap()),
    )
    .with_inner_max_iter(1);
    let result = Executor::from_start(equality::<F>(), alm, start.clone())
        .max_iter(2)
        .run()
        .unwrap();
    assert_eq!(result.report.code(), TerminationCode::SolverConverged);
    assert_eq!(result.param(), &vec![F::zero(), F::one()]);

    let problem = Quadratic {
        a: DenseMatrix::from_row_slice(1, 2, &[F::zero(), F::zero()]),
        b: vec![F::one()],
        scale: F::one(),
    };
    let barrier = BarrierMethod::with_inner_solver(GradientDescent::new(
        F::from_f64(0.5).unwrap(),
    ))
    .with_inner_max_iter(1)
    .with_absolute_duality_gap_tolerance(F::one());
    let result = Executor::from_start(with_ineq(problem), barrier, start)
        .max_iter(2)
        .run()
        .unwrap();
    assert_eq!(result.report.code(), TerminationCode::SolverConverged);
    assert_eq!(result.param(), &vec![F::zero(), F::one()]);
}

#[test]
fn progress_only_inners_do_not_establish_outer_stationarity() {
    false_outer_success::<f64>();
    false_outer_success::<f32>();
}

#[test]
fn accurate_budget_limited_inners_remain_usable() {
    accurate_partial_inner::<f64>();
    accurate_partial_inner::<f32>();
}

fn exact_and_disabled<F: Scalar>(tiny: F) {
    let mut problem = equality::<F>().into_parts().0;
    problem.scale = tiny;
    let problem = with_eq(problem);
    let solver = AugmentedLagrangianMethod::with_inner_solver(
        GradientDescent::new(F::zero()),
    )
    .with_inner_max_iter(1)
    .with_absolute_stationarity_tolerance(F::zero())
    .with_relative_stationarity_tolerance(F::zero());
    let result = Executor::from_start(
        problem,
        solver,
        vec![F::zero(), F::from_f64(3.0).unwrap()],
    )
    .max_iter(2)
    .run()
    .unwrap();
    assert_eq!(result.report.code(), TerminationCode::MaxIter);

    let solver = AugmentedLagrangianMethod::with_inner_solver(
        GradientDescent::new(F::zero()),
    )
    .with_inner_max_iter(1)
    .with_absolute_stationarity_tolerance(None)
    .with_relative_stationarity_tolerance(None);
    let result = Executor::from_start(
        equality::<F>(),
        solver,
        vec![F::zero(), F::from_f64(3.0).unwrap()],
    )
    .max_iter(2)
    .run()
    .unwrap();
    assert_eq!(result.report.code(), TerminationCode::SolverConverged);

    for tolerance in [Some(F::zero()), None] {
        // Constant positive slack isolates the objective gradient, including
        // subnormal components that would disappear in a squared norm.
        let problem = Quadratic {
            a: DenseMatrix::from_row_slice(1, 2, &[F::zero(), F::zero()]),
            b: vec![F::one()],
            scale: tiny,
        };
        let solver =
            BarrierMethod::with_inner_solver(GradientDescent::new(F::zero()))
                .with_inner_max_iter(1)
                .with_absolute_stationarity_tolerance(tolerance)
                .with_relative_stationarity_tolerance(tolerance)
                .with_absolute_duality_gap_tolerance(F::one());
        let result = Executor::from_start(
            with_ineq(problem),
            solver,
            vec![F::zero(), F::from_f64(3.0).unwrap()],
        )
        .max_iter(2)
        .run()
        .unwrap();
        assert_eq!(
            result.report.code(),
            if tolerance.is_some() {
                TerminationCode::MaxIter
            } else {
                TerminationCode::SolverConverged
            }
        );
    }
}

#[test]
fn exact_stationarity_does_not_underflow_and_both_checks_can_be_disabled() {
    exact_and_disabled::<f64>(f64::from_bits(1));
    exact_and_disabled::<f32>(f32::from_bits(1));
}

fn relative_scaling<F: Scalar>() {
    for scale in [1e-8, 1.0, 1e8] {
        let scale = F::from_f64(scale).unwrap();
        for step in [F::zero(), F::from_f64(0.5).unwrap() / scale] {
            let mut problem = equality::<F>().into_parts().0;
            problem.scale = scale;
            let problem = with_eq(problem);
            let solver = AugmentedLagrangianMethod::with_inner_solver(
                GradientDescent::new(step),
            )
            .with_inner_max_iter(1)
            .with_absolute_stationarity_tolerance(None);
            let result = Executor::from_start(
                problem,
                solver,
                vec![F::zero(), F::from_f64(3.0).unwrap()],
            )
            .max_iter(2)
            .run()
            .unwrap();
            assert_eq!(
                result.report.code(),
                if step == F::zero() {
                    TerminationCode::MaxIter
                } else {
                    TerminationCode::SolverConverged
                }
            );
        }
    }
}

#[test]
fn relative_stationarity_handles_objective_scaling_in_both_precisions() {
    relative_scaling::<f64>();
    relative_scaling::<f32>();
}

#[test]
fn barrier_retries_keep_the_first_reference_for_the_same_weight() {
    let solver = BarrierMethod::with_inner_solver(GradientDescent::new(0.1))
        .with_inner_max_iter(1)
        .with_absolute_stationarity_tolerance(None)
        .with_relative_stationarity_tolerance(1e-3)
        .with_absolute_duality_gap_tolerance(1e-3);
    let result =
        Executor::from_start(inequality::<f64>(), solver, vec![0.0, 3.0])
            .max_iter(200)
            .run()
            .unwrap();
    assert_eq!(result.report.code(), TerminationCode::SolverConverged);
    assert!((result.param()[1] - 1.0).abs() < 1e-3);
    assert!(result.param()[0].abs() < 1e-3);
}

struct OuterProbe {
    a: DenseMatrix,
    b: Vec<f64>,
    calls: std::cell::Cell<u64>,
    nonfinite: bool,
    abort_gradient: bool,
    gradient_value: f64,
}

impl basin::Constraints for OuterProbe {
    type Set = basin::NoConstraints;
    fn constraints(&self) -> basin::NoConstraints {
        basin::NoConstraints::EMPTY
    }
}
#[allow(clippy::type_complexity)]
fn probe_eq(
    p: OuterProbe,
) -> basin::Constrained<
    OuterProbe,
    basin::ConstraintSet<
        basin::NoBounds,
        basin::LinearBlocks<basin::LinearBlock<DenseMatrix, Vec<f64>>>,
    >,
> {
    let block = basin::LinearBlock::new(p.a.clone(), p.b.clone());
    basin::Constrained::new(
        p,
        basin::ConstraintSet::new().with_linear_equalities(block),
    )
}
#[allow(clippy::type_complexity)]
fn probe_ineq(
    p: OuterProbe,
) -> basin::Constrained<
    OuterProbe,
    basin::ConstraintSet<
        basin::NoBounds,
        basin::LinearBlocks<
            basin::NoLinearBlock,
            basin::LinearBlock<DenseMatrix, Vec<f64>>,
        >,
    >,
> {
    let block = basin::LinearBlock::new(p.a.clone(), p.b.clone());
    basin::Constrained::new(
        p,
        basin::ConstraintSet::new().with_linear_inequalities(block),
    )
}
impl CostFunction for OuterProbe {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = &'static str;
    fn cost(&self, _: &Vec<f64>) -> Result<f64, Self::Error> {
        let calls = self.calls.get();
        self.calls.set(calls + 1);
        Ok(if self.nonfinite && calls > 0 {
            f64::NAN
        } else {
            0.0
        })
    }
}
impl Gradient for OuterProbe {
    type Gradient = Vec<f64>;
    fn gradient(&self, _: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        if self.abort_gradient {
            Err("stationarity aborted")
        } else {
            Ok(vec![self.gradient_value])
        }
    }
}

struct ClaimedConverged;
impl basin::InitialState<Vec<f64>> for ClaimedConverged {
    type State = basin::PointState<Vec<f64>>;
    fn seed(&self, x: &Vec<f64>) -> Self::State {
        basin::PointState::new(x.clone())
    }
}
impl basin::WarmStart<Vec<f64>> for ClaimedConverged {}
impl<P: CostFunction<Param = Vec<f64>, Output = f64>>
    basin::Solver<P, basin::PointState<Vec<f64>>> for ClaimedConverged
{
    type Error = P::Error;
    fn init(
        &mut self,
        problem: &mut basin::Problem<P>,
        mut state: basin::PointState<Vec<f64>>,
    ) -> Result<basin::PointState<Vec<f64>>, Self::Error> {
        use basin::State;
        let cost = problem.cost(state.param())?;
        state.replace(state.param().clone(), cost);
        Ok(state)
    }
    fn next_iter(
        &mut self,
        _: &mut basin::Problem<P>,
        _: basin::PointState<Vec<f64>>,
    ) -> Result<basin::SolverStep<basin::PointState<Vec<f64>>>, Self::Error>
    {
        unreachable!()
    }
    fn terminate(
        &self,
        _: &basin::PointState<Vec<f64>>,
    ) -> Option<basin::Termination> {
        Some(basin::Termination::custom(
            "inner.claim",
            "An inner completion label is not independent outer evidence.",
            vec![],
        ))
    }
}

#[test]
fn stationarity_callback_errors_are_charged_and_propagated() {
    use basin::{InitialState, Problem, Solver};
    for barrier_mode in [false, true] {
        let problem = OuterProbe {
            a: DenseMatrix::from_row_slice(1, 1, &[1.0]),
            b: vec![if barrier_mode { 1.0 } else { 0.0 }],
            calls: std::cell::Cell::new(0),
            nonfinite: false,
            abort_gradient: true,
            gradient_value: 0.0,
        };
        let (error, counts) = if barrier_mode {
            let mut problem = Problem::new(probe_ineq(problem));
            let mut solver = BarrierMethod::with_inner_solver(ClaimedConverged);
            let state =
                solver.init(&mut problem, solver.seed(&vec![0.0])).unwrap();
            let error = solver.next_iter(&mut problem, state).err().unwrap();
            (error, *problem.counts())
        } else {
            let mut problem = Problem::new(probe_eq(problem));
            let mut solver =
                AugmentedLagrangianMethod::with_inner_solver(ClaimedConverged);
            let state =
                solver.init(&mut problem, solver.seed(&vec![0.0])).unwrap();
            let error = solver.next_iter(&mut problem, state).err().unwrap();
            (error, *problem.counts())
        };
        assert_eq!(
            error,
            basin::ConstraintSolveError::Evaluation("stationarity aborted")
        );
        assert_eq!(counts.cost_evals, 2);
        assert_eq!(counts.gradient_evals, 1);
    }
}

#[test]
fn nonfinite_original_costs_do_not_replace_a_coherent_outer_point() {
    for barrier_mode in [false, true] {
        let problem = OuterProbe {
            a: DenseMatrix::from_row_slice(1, 1, &[1.0]),
            b: vec![if barrier_mode { 1.0 } else { 0.0 }],
            calls: std::cell::Cell::new(0),
            nonfinite: true,
            abort_gradient: false,
            gradient_value: 0.0,
        };
        let (code, cost, iter) = if barrier_mode {
            let result = Executor::from_start(
                probe_ineq(problem),
                BarrierMethod::with_inner_solver(ClaimedConverged),
                vec![0.0],
            )
            .max_iter(2)
            .run()
            .unwrap();
            (result.report.code(), result.cost(), result.state.iter())
        } else {
            let result = Executor::from_start(
                probe_eq(problem),
                AugmentedLagrangianMethod::with_inner_solver(ClaimedConverged),
                vec![0.0],
            )
            .max_iter(2)
            .run()
            .unwrap();
            (result.report.code(), result.cost(), result.state.iter())
        };
        assert_eq!(code, TerminationCode::SolverFailed);
        assert_eq!(cost, 0.0);
        assert_eq!(iter, 0);
    }
}

#[test]
fn nonfinite_gradients_cannot_establish_outer_convergence() {
    for barrier_mode in [false, true] {
        for gradient_value in [f64::NAN, f64::INFINITY] {
            let problem = OuterProbe {
                a: DenseMatrix::from_row_slice(1, 1, &[1.0]),
                b: vec![if barrier_mode { 1.0 } else { 0.0 }],
                calls: std::cell::Cell::new(0),
                nonfinite: false,
                abort_gradient: false,
                gradient_value,
            };
            let code = if barrier_mode {
                Executor::from_start(
                    probe_ineq(problem),
                    BarrierMethod::with_inner_solver(ClaimedConverged),
                    vec![0.0],
                )
                .max_iter(2)
                .run()
                .unwrap()
                .report
                .code()
            } else {
                Executor::from_start(
                    probe_eq(problem),
                    AugmentedLagrangianMethod::with_inner_solver(
                        ClaimedConverged,
                    ),
                    vec![0.0],
                )
                .max_iter(2)
                .run()
                .unwrap()
                .report
                .code()
            };
            assert_eq!(code, TerminationCode::MaxIter);
        }
    }
}
