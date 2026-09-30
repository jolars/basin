//! Rejected trials must not look like converged cost or parameter changes.

use basin::{
    BoxConstraints, CostFunction, DenseMatrix, ExactCheckpoint, Executor,
    FirstOrderState, Gradient, Hessian, HessianProduct, Jacobian,
    LevenbergMarquardt, LmDamping, PointState, Problem, Residual,
    RobustLeastSquares, RunControl, Solver, SquaredLoss, State,
    TerminationReason, Trf, TrustRegion, run_loop_with_control,
};
use std::convert::Infallible;

struct SquareResidual {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl SquareResidual {
    fn new() -> Self {
        Self {
            lower: vec![f64::NEG_INFINITY],
            upper: vec![f64::INFINITY],
        }
    }
}

impl Residual for SquareResidual {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = Infallible;

    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(vec![x[0] * x[0] - 1.])
    }
}

impl Jacobian for SquareResidual {
    type Jacobian = DenseMatrix;

    fn jacobian(&self, x: &Vec<f64>) -> Result<DenseMatrix, Infallible> {
        Ok(DenseMatrix::from_row_slice(1, 1, &[2. * x[0]]))
    }
}

impl CostFunction for SquareResidual {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(0.5 * (x[0] * x[0] - 1.).powi(2))
    }
}

impl Gradient for SquareResidual {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(vec![2. * x[0] * (x[0] * x[0] - 1.)])
    }
}

impl Hessian for SquareResidual {
    type Hessian = DenseMatrix;

    fn hessian(&self, x: &Vec<f64>) -> Result<DenseMatrix, Infallible> {
        Ok(DenseMatrix::from_row_slice(1, 1, &[6. * x[0] * x[0] - 2.]))
    }
}

impl HessianProduct for SquareResidual {
    fn hessian_product(
        &self,
        x: &Vec<f64>,
        v: &Vec<f64>,
    ) -> Result<Vec<f64>, Infallible> {
        Ok(vec![(6. * x[0] * x[0] - 2.) * v[0]])
    }
}

impl BoxConstraints for SquareResidual {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }

    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

fn check_rejections<P, S, So>(
    make_problem: impl Fn() -> P,
    make_state: impl Fn() -> S,
    mut solver: So,
    accepted_reason: TerminationReason,
) where
    S: State<Param = Vec<f64>, Float = f64> + basin::CountsMirror,
    So: Solver<P, S, Error = Infallible>,
{
    let mut problem = Problem::new(make_problem());
    // Both fresh reuse and exact continuation must retain the right anchor.
    for _ in 0..2 {
        let first = run_loop_with_control(
            &mut problem,
            make_state(),
            &mut solver,
            &mut RunControl::new().max_iter(1),
        )
        .unwrap();
        assert_eq!(first.reason, TerminationReason::MaxIter);
        assert_eq!(first.state.param(), &[0.1]);
        assert_eq!(first.state.cost(), 0.5 * 0.99_f64.powi(2));
        assert_eq!(solver.check_convergence(&problem, &first.state), None);
        assert_eq!(solver.check_convergence(&problem, &first.state), None);

        let checkpoint =
            ExactCheckpoint::from_parts(solver, first.state, *problem.counts());
        let result =
            Executor::resume_from_checkpoint(make_problem(), checkpoint)
                .max_iter(100)
                .run_with_solver()
                .unwrap();
        assert_eq!(result.reason, accepted_reason);
        assert!(result.state.cost() < 0.5 * 0.99_f64.powi(2));
        assert!(result.state.param()[0] > 0.1);
        solver = result.solver;
    }
}

macro_rules! rejection_checks {
    ($name:ident, $problem:expr, $state:ident, $solver:expr, [$($setter:ident => $reason:ident),+ $(,)?]) => {
        mod $name {
            use super::*;
            $(
                #[test]
                fn $setter() {
                // A loose threshold must fire on the first accepted step,
                // while even an exact-zero threshold must ignore rejection.
                check_rejections(
                    || $problem,
                    || $state::new(vec![0.1]),
                    ($solver).$setter(10.),
                    TerminationReason::$reason,
                );
                let result = Executor::new(
                    $problem,
                    ($solver).$setter(0.),
                    $state::new(vec![0.1]),
                )
                .max_iter(2)
                .run()
                .unwrap();
                assert_eq!(result.reason, TerminationReason::MaxIter);
                let result = Executor::new(
                    $problem,
                    ($solver).$setter(1e-13),
                    $state::new(vec![0.1]),
                )
                .max_iter(100)
                .run()
                .unwrap();
                assert!(result.cost() < 1e-12, "cost = {}", result.cost());
                }
            )+
        }
    };
}

macro_rules! all_change_checks {
    ($name:ident, $state:ident, $solver:expr) => {
        rejection_checks!($name, SquareResidual::new(), $state, $solver, [
            with_absolute_cost_change_tolerance => CostTolerance,
            with_relative_cost_change_tolerance => RelativeCostTolerance,
            with_absolute_step_tolerance => ParamTolerance,
            with_relative_step_tolerance => RelativeParamTolerance,
        ]);
    };
}

all_change_checks!(trf, PointState, Trf::new());
all_change_checks!(
    trust_region,
    FirstOrderState,
    TrustRegion::new()
        .with_radius(10.)
        .with_max_inner_attempts(1)
);
all_change_checks!(
    matrix_free,
    FirstOrderState,
    TrustRegion::matrix_free()
        .with_radius(10.)
        .with_max_inner_attempts(1)
);

macro_rules! lm_checks {
    ($name:ident, $problem:expr, $solver:expr) => {
        rejection_checks!($name, $problem, PointState, $solver, [
            with_absolute_cost_change_tolerance => CostTolerance,
            with_relative_cost_change_tolerance => RelativeCostTolerance,
            with_absolute_step_tolerance => ParamTolerance,
        ]);
    };
}

lm_checks!(lm, SquareResidual::new(), LevenbergMarquardt::new());
lm_checks!(
    lm_qr,
    SquareResidual::new(),
    LevenbergMarquardt::new().with_pivoted_qr()
);
lm_checks!(
    lm_trust_radius,
    SquareResidual::new(),
    LevenbergMarquardt::new().with_damping(LmDamping::TrustRegion)
);
lm_checks!(
    lm_qr_trust_radius,
    SquareResidual::new(),
    LevenbergMarquardt::new()
        .with_damping(LmDamping::TrustRegion)
        .with_pivoted_qr()
);
lm_checks!(
    robust_lm,
    RobustLeastSquares::new(SquareResidual::new(), SquaredLoss),
    LevenbergMarquardt::new()
);
lm_checks!(
    robust_lm_qr,
    RobustLeastSquares::new(SquareResidual::new(), SquaredLoss),
    LevenbergMarquardt::new().with_pivoted_qr()
);
rejection_checks!(robust_trf,
RobustLeastSquares::new(SquareResidual::new(), SquaredLoss), PointState, Trf::new(), [
    with_absolute_cost_change_tolerance => CostTolerance,
    with_relative_cost_change_tolerance => RelativeCostTolerance,
    with_absolute_step_tolerance => ParamTolerance,
    with_relative_step_tolerance => RelativeParamTolerance,
]);

#[test]
fn rejection_after_accepted_steps_does_not_establish_convergence() {
    use std::{cell::RefCell, rc::Rc};

    fn check<So>(solver: So)
    where
        So: Solver<
                SquareResidual,
                FirstOrderState<Vec<f64>>,
                Error = Infallible,
            >,
    {
        let iterates = Rc::new(RefCell::new(Vec::new()));
        let observed = iterates.clone();
        let result = Executor::new(
            SquareResidual::new(),
            solver,
            FirstOrderState::new(vec![0.1]),
        )
        .max_iter(100)
        .stop_when(move |state| {
            observed.borrow_mut().push(state.param()[0]);
            None
        })
        .run()
        .unwrap();

        assert!(
            iterates
                .borrow()
                .windows(2)
                .any(|pair| { pair[0] != 0.1 && pair[0] == pair[1] })
        );
        assert_eq!(result.reason, TerminationReason::GradientTolerance);
        assert!(result.cost() < 1e-16);
    }

    check(
        TrustRegion::new()
            .with_radius(0.01)
            .with_max_inner_attempts(1)
            .with_absolute_gradient_tolerance(1e-8)
            .with_absolute_cost_change_tolerance(0.)
            .with_absolute_step_tolerance(0.),
    );
    check(
        TrustRegion::matrix_free()
            .with_radius(0.01)
            .with_max_inner_attempts(1)
            .with_absolute_gradient_tolerance(1e-8)
            .with_absolute_cost_change_tolerance(0.)
            .with_absolute_step_tolerance(0.),
    );
}
