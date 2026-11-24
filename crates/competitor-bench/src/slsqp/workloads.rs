//! Shared constrained workloads for the allocation probe and diagnostics audit.

use std::convert::Infallible;

use basin::{
    BoxBounds, ConstraintDescription, ConstraintJacobian, ConstraintSet,
    ConstraintSolveError, Constraints, CostFunction, DenseMatrix, Executor,
    Gradient, NoLinearConstraints, NonlinearBlock, NonlinearBlocks,
    NonlinearEqualityValues, NonlinearInequalityValues,
    OptimizationResultWithSolver, SelectedFirstOrderState, Slsqp, State,
    TerminationCode,
};

pub type Result =
    OptimizationResultWithSolver<SelectedFirstOrderState<Vec<f64>>, Slsqp>;

#[derive(Clone, Copy, Debug)]
pub enum Case {
    Hs71,
    Quadratic,
}

impl Case {
    pub fn name(self) -> &'static str {
        match self {
            Self::Hs71 => "hs71",
            Self::Quadratic => "quadratic",
        }
    }

    pub fn start(self) -> Vec<f64> {
        match self {
            Self::Hs71 => vec![1.0, 5.0, 5.0, 1.0],
            Self::Quadratic => {
                (0..30).map(|i| 0.3 + 0.4 * i as f64 / 30.0).collect()
            }
        }
    }

    pub fn objective(self, x: &[f64]) -> f64 {
        match self {
            Self::Hs71 => x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2],
            Self::Quadratic => super::quadratic(x),
        }
    }

    pub fn gradient(self, x: &[f64], g: &mut [f64]) {
        match self {
            Self::Hs71 => g.copy_from_slice(&[
                x[3] * (2.0 * x[0] + x[1] + x[2]),
                x[0] * x[3],
                x[0] * x[3] + 1.0,
                x[0] * (x[0] + x[1] + x[2]),
            ]),
            Self::Quadratic => super::quadratic_gradient(x, g),
        }
    }

    pub fn has_equality(self) -> bool {
        matches!(self, Self::Hs71)
    }

    pub fn equality(x: &[f64], g: Option<&mut [f64]>) -> f64 {
        if let Some(g) = g {
            g.iter_mut().zip(x).for_each(|(g, x)| *g = 2.0 * x);
        }
        x.iter().map(|v| v * v).sum::<f64>() - 40.0
    }

    pub fn inequality(self, x: &[f64], g: Option<&mut [f64]>) -> f64 {
        match self {
            Self::Hs71 => {
                if let Some(g) = g {
                    for (j, g) in g.iter_mut().enumerate() {
                        *g = -(0..4)
                            .filter(|&k| k != j)
                            .map(|k| x[k])
                            .product::<f64>();
                    }
                }
                25.0 - x.iter().product::<f64>()
            }
            Self::Quadratic => super::ball_value(x, g),
        }
    }

    pub fn solve(
        self,
        start: Vec<f64>,
        lower: Vec<f64>,
        upper: Vec<f64>,
    ) -> Result {
        Executor::from_start(
            Workload {
                case: self,
                lower,
                upper,
            },
            Slsqp::new().with_absolute_accuracy_tolerance(super::ACCURACY),
            start,
        )
        .max_evaluations(basin::EvaluationKind::Cost, super::BUDGET as u64)
        .max_iter(super::BUDGET as u64)
        .run_with_solver()
        .unwrap()
    }

    pub fn bounds(self, n: usize) -> (Vec<f64>, Vec<f64>) {
        let (lower, upper) = match self {
            Self::Hs71 => (1.0, 5.0),
            Self::Quadratic => (f64::NEG_INFINITY, f64::INFINITY),
        };
        (vec![lower; n], vec![upper; n])
    }
}

struct Workload {
    case: Case,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl Constraints for Workload {
    type Set = ConstraintSet<
        BoxBounds<Vec<f64>>,
        NoLinearConstraints,
        NonlinearBlocks<NonlinearBlock, NonlinearBlock>,
    >;

    fn constraints(
        &self,
    ) -> <Self::Set as ConstraintDescription>::Borrowed<'_> {
        ConstraintSet::new()
            .with_bounds(BoxBounds::new(Some(&self.lower), Some(&self.upper)))
            .with_nonlinear_equalities(usize::from(self.case.has_equality()))
            .with_nonlinear_inequalities(1)
    }
}

impl CostFunction for Workload {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> std::result::Result<f64, Infallible> {
        Ok(self.case.objective(x))
    }
}

impl Gradient for Workload {
    type Gradient = Vec<f64>;

    fn gradient(
        &self,
        x: &Vec<f64>,
    ) -> std::result::Result<Vec<f64>, Infallible> {
        let mut g = vec![0.0; x.len()];
        self.case.gradient(x, &mut g);
        Ok(g)
    }
}

impl NonlinearInequalityValues for Workload {
    type Param = Vec<f64>;
    type Values = Vec<f64>;
    type Error = Infallible;

    fn nonlinear_inequalities(
        &self,
        x: &Self::Param,
    ) -> std::result::Result<Self::Values, ConstraintSolveError<Self::Error>>
    {
        Ok(vec![self.case.inequality(x, None)])
    }
}

impl NonlinearEqualityValues for Workload {
    type Param = Vec<f64>;
    type Values = Vec<f64>;
    type Error = Infallible;

    fn nonlinear_equalities(
        &self,
        x: &Self::Param,
    ) -> std::result::Result<Self::Values, ConstraintSolveError<Self::Error>>
    {
        Ok(if self.case.has_equality() {
            vec![Case::equality(x, None)]
        } else {
            vec![]
        })
    }
}

impl ConstraintJacobian for Workload {
    type Param = Vec<f64>;
    type Jacobian = DenseMatrix;
    type Error = Infallible;

    fn constraint_jacobian(
        &self,
        x: &Self::Param,
    ) -> std::result::Result<Self::Jacobian, ConstraintSolveError<Self::Error>>
    {
        let rows = 1 + usize::from(self.case.has_equality());
        let mut entries = vec![0.0; rows * x.len()];
        if self.case.has_equality() {
            Case::equality(x, Some(&mut entries[..x.len()]));
        }
        self.case
            .inequality(x, Some(&mut entries[(rows - 1) * x.len()..]));
        Ok(DenseMatrix::from_row_slice(rows, x.len(), &entries))
    }
}

pub fn hs71() -> Result {
    let case = Case::Hs71;
    let start = case.start();
    let (lower, upper) = case.bounds(start.len());
    case.solve(start, lower, upper)
}

pub fn verify_hs71(result: &Result) {
    assert_eq!(result.report.code(), TerminationCode::SolverConverged);
    let x = result.state.param();
    assert!(x.iter().all(|v| v.is_finite() && (1.0..=5.0).contains(v)));
    let cost = Case::Hs71.objective(x);
    assert!((cost - result.state.cost()).abs() < 1e-12);
    assert!((cost - 17.014_017_289_156).abs() < 1e-8);
    assert!(Case::equality(x, None).abs() < 1e-9);
    assert!(Case::Hs71.inequality(x, None) < 1e-9);
    assert!(result.solver.stationarity().unwrap() < 1e-5);
    assert!(result.solver.complementarity().unwrap() < 1e-9);
}
