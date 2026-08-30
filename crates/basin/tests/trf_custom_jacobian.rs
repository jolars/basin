use std::convert::Infallible;

use basin::{
    AddDiagonalVectorInPlace, BoxConstraints, CostFunction, Executor,
    GramMatrix, Jacobian, LinearSolveError, LinearSolveSpd, MatTransposeVec,
    MaxDiagonal, Residual, TerminationReason, Trf,
};

#[derive(Clone)]
struct ScalarMatrix(f64);

impl GramMatrix for ScalarMatrix {
    fn gram(&self) -> Self {
        Self(self.0 * self.0)
    }
}

impl MatTransposeVec<Vec<f64>> for ScalarMatrix {
    fn mat_transpose_vec(&self, x: &Vec<f64>) -> Vec<f64> {
        assert_eq!(x.len(), 1);
        vec![self.0 * x[0]]
    }
}

impl LinearSolveSpd<Vec<f64>> for ScalarMatrix {
    fn solve_spd(&self, b: &Vec<f64>) -> Result<Vec<f64>, LinearSolveError> {
        assert_eq!(b.len(), 1);
        if self.0 <= 0.0 {
            return Err(LinearSolveError::NotPositiveDefinite);
        }
        Ok(vec![b[0] / self.0])
    }
}

impl AddDiagonalVectorInPlace<Vec<f64>> for ScalarMatrix {
    fn add_diagonal_vector_in_place(&mut self, diagonal: &Vec<f64>) {
        assert_eq!(diagonal.len(), 1);
        self.0 += diagonal[0];
    }
}

impl MaxDiagonal for ScalarMatrix {
    fn max_diagonal(&self) -> f64 {
        self.0
    }
}

struct BoundedResidual {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl CostFunction for BoundedResidual {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        Ok(0.5 * (x[0] - 2.0).powi(2))
    }
}

impl Residual for BoundedResidual {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = Infallible;

    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(vec![x[0] - 2.0])
    }
}

impl Jacobian for BoundedResidual {
    type Jacobian = ScalarMatrix;

    fn jacobian(&self, _x: &Vec<f64>) -> Result<ScalarMatrix, Self::Error> {
        Ok(ScalarMatrix(1.0))
    }
}

impl BoxConstraints for BoundedResidual {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }

    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

#[test]
fn trf_accepts_a_downstream_jacobian_type() {
    let problem = BoundedResidual {
        lower: vec![-10.0],
        upper: vec![10.0],
    };

    let result = Executor::from_start(
        problem,
        Trf::<Vec<f64>, ScalarMatrix>::new(),
        vec![0.0],
    )
    .max_iter(50)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!((result.param()[0] - 2.0).abs() < 1e-8);
}
