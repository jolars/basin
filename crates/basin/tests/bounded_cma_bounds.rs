//! Gaussian box sampling accepts unbounded sides while validating the box.

use basin::{
    BoundedCmaEs, BoundedCmaInject, BoxBounds, Constrained, ConstraintError,
    ConstraintSolveError, Constraints, CostFunction, DenseMatrix, Executor,
    NelderMead, NoConstraints, PopulationProgress, Problem, ProblemConstraints,
    RelaxedInner, Solver,
};
use std::convert::Infallible;

struct Quadratic;
impl ProblemConstraints for Quadratic {
    type Constraints = NoConstraints;
    fn constraint_set(&self) -> NoConstraints {
        Constraints::new()
    }
}
impl CostFunction for Quadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        assert!(x[0].is_finite());
        Ok(x[0] * x[0])
    }
}

type BoxedQuadratic = Constrained<Quadratic, Constraints<BoxBounds<Vec<f64>>>>;

fn boxed(lower: Vec<f64>, upper: Vec<f64>) -> BoxedQuadratic {
    Constrained::new(
        Quadratic,
        Constraints::new()
            .with_bounds(BoxBounds::new(Some(lower), Some(upper))),
    )
}

fn check<S>(make_solver: impl Fn() -> S)
where
    S: Solver<
            BoxedQuadratic,
            PopulationProgress<Vec<f64>>,
            Error = ConstraintSolveError<Infallible>,
        >,
{
    for (lower, upper) in [
        (0.0, f64::INFINITY),
        (f64::NEG_INFINITY, 0.0),
        (f64::NEG_INFINITY, f64::INFINITY),
    ] {
        let result = Executor::new(
            boxed(vec![lower], vec![upper]),
            make_solver(),
            PopulationProgress::from_point(vec![2.0]),
        )
        .max_iter(300)
        .run()
        .unwrap();
        assert!(
            result.best_param()[0] >= lower && result.best_param()[0] <= upper
        );
        assert!(result.best_cost() < 1e-8, "cost = {}", result.best_cost());
    }

    for (lower, upper, expected) in [
        (
            vec![f64::NAN],
            vec![1.0],
            ConstraintError::NaNBound {
                side: basin::BoundSide::Lower,
                coordinate: 0,
            },
        ),
        (
            vec![f64::INFINITY],
            vec![f64::INFINITY],
            ConstraintError::InvalidInfinity {
                side: basin::BoundSide::Lower,
                coordinate: 0,
            },
        ),
        (
            vec![f64::NEG_INFINITY],
            vec![f64::NEG_INFINITY],
            ConstraintError::InvalidInfinity {
                side: basin::BoundSide::Upper,
                coordinate: 0,
            },
        ),
        (
            vec![2.0],
            vec![1.0],
            ConstraintError::InvertedBounds { coordinate: 0 },
        ),
        (
            vec![0.0],
            vec![],
            ConstraintError::DimensionMismatch {
                block: basin::ConstraintBlock::Bounds,
                axis: basin::ConstraintAxis::Length,
                expected: 1,
                actual: 0,
            },
        ),
    ] {
        let mut problem = Problem::new(boxed(lower, upper));
        let error = make_solver()
            .init(&mut problem, &mut PopulationProgress::from_point(vec![0.0]))
            .unwrap_err();
        assert_eq!(
            error,
            basin::EvaluationError::Evaluation(
                ConstraintSolveError::InvalidConstraints(expected)
            )
        );
        assert_eq!(problem.counts().total_work(), 0);
    }
}

#[test]
fn bounded_cma_accepts_unbounded_sides() {
    check(|| BoundedCmaEs::<Vec<f64>, DenseMatrix>::new(42, 0.3));
}

#[test]
fn bounded_cma_inject_accepts_unbounded_sides() {
    check(|| {
        BoundedCmaInject::with_inner_solver(
            BoundedCmaEs::<Vec<f64>, DenseMatrix>::new(42, 0.3),
            RelaxedInner::new(NelderMead::new()),
        )
    });
}
