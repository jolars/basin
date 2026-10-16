//! Coordinate-based solvers must not require reference-returning indexing.

use basin::{
    BoxBounds, ClampInPlace, Constrained, ConstraintError,
    ConstraintSolveError, Constraints, CostFunction, Direct, Executor, Gbnm,
    NoConstraints, PointState, Problem, ProblemConstraints, SampleUniformBox,
    ScaleInPlace, ScaledAdd, Solver, VectorIndex, VectorLen,
};
use std::convert::Infallible;

#[derive(Clone, Debug)]
struct Coordinates(Vec<f64>);

impl VectorLen for Coordinates {
    fn vec_len(&self) -> usize {
        self.0.len()
    }
}
impl VectorIndex for Coordinates {
    fn get_scalar(&self, i: usize) -> f64 {
        self.0[i]
    }
    fn set_scalar(&mut self, i: usize, value: f64) {
        self.0[i] = value;
    }
}
impl ClampInPlace for Coordinates {
    fn clamp_in_place(&mut self, lower: &Self, upper: &Self) {
        self.0.clamp_in_place(&lower.0, &upper.0);
    }
}
impl ScaleInPlace for Coordinates {
    fn scale_in_place(&mut self, scalar: f64) {
        self.0.scale_in_place(scalar);
    }
}
impl ScaledAdd for Coordinates {
    fn scaled_add(&mut self, scalar: f64, other: &Self) {
        self.0.scaled_add(scalar, &other.0);
    }
}
impl SampleUniformBox for Coordinates {
    fn sample_uniform_box<R: rand::Rng + ?Sized>(
        lower: &Self,
        upper: &Self,
        rng: &mut R,
    ) -> Self {
        Self(Vec::sample_uniform_box(&lower.0, &upper.0, rng))
    }
}

struct Quadratic;
impl ProblemConstraints for Quadratic {
    type Constraints = NoConstraints;
    fn constraint_set(&self) -> NoConstraints {
        Constraints::new()
    }
}
impl CostFunction for Quadratic {
    type Param = Coordinates;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Coordinates) -> Result<f64, Infallible> {
        Ok((x.get_scalar(0) - 0.25).powi(2))
    }
}

type BoxedQuadratic =
    Constrained<Quadratic, Constraints<BoxBounds<Coordinates>>>;

fn boxed(lower: Vec<f64>, upper: Vec<f64>) -> BoxedQuadratic {
    Constrained::new(
        Quadratic,
        Constraints::new().with_bounds(BoxBounds::new(
            Some(Coordinates(lower)),
            Some(Coordinates(upper)),
        )),
    )
}

#[test]
fn direct_accepts_coordinate_only_vectors() {
    let result = Executor::new(
        boxed(vec![-1.0], vec![1.0]),
        Direct::new(),
        PointState::new(Coordinates(vec![0.8])),
    )
    .max_iter(100)
    .run()
    .unwrap();
    assert!((result.param().get_scalar(0) - 0.25).abs() < 1e-4);
}

#[test]
fn gbnm_accepts_coordinate_only_vectors() {
    let result = Executor::new(
        boxed(vec![-1.0], vec![1.0]),
        Gbnm::new(42),
        PointState::new(Coordinates(vec![0.8])),
    )
    .max_iter(100)
    .run()
    .unwrap();
    assert!((result.best_param().get_scalar(0) - 0.25).abs() < 1e-4);
}

#[test]
fn coordinate_validation_preserves_bound_errors() {
    let cases = [
        (
            vec![f64::NAN],
            vec![1.0],
            ConstraintError::NaNBound {
                side: basin::BoundSide::Lower,
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
            vec![f64::INFINITY],
            ConstraintError::NonFiniteInterval { coordinate: 0 },
        ),
        (
            vec![],
            vec![1.0],
            ConstraintError::DimensionMismatch {
                block: basin::ConstraintBlock::Bounds,
                axis: basin::ConstraintAxis::Length,
                expected: 1,
                actual: 0,
            },
        ),
    ];
    for (lower, upper, expected) in cases {
        let mut direct_problem =
            Problem::new(boxed(lower.clone(), upper.clone()));
        let error = Direct::new()
            .init(&mut direct_problem, PointState::new(Coordinates(vec![0.0])))
            .unwrap_err();
        assert_eq!(
            error,
            ConstraintSolveError::InvalidConstraints(expected.clone())
        );
        assert_eq!(direct_problem.counts().total_work(), 0);
        let mut gbnm_problem = Problem::new(boxed(lower, upper));
        let error = Gbnm::new(42)
            .init(&mut gbnm_problem, PointState::new(Coordinates(vec![0.0])))
            .unwrap_err();
        assert_eq!(error, ConstraintSolveError::InvalidConstraints(expected));
        assert_eq!(gbnm_problem.counts().total_work(), 0);
    }
}
