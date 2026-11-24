use basin::{
    BoxBounds, ConstraintDescription, ConstraintSet, Constraints, CostFunction,
    Executor, Gradient, Lbfgsb, Termination,
};
use std::convert::Infallible;

struct BoundedQuadratic {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl Constraints for BoundedQuadratic {
    type Set = ConstraintSet<BoxBounds<Vec<f64>>>;

    fn constraints(
        &self,
    ) -> <Self::Set as ConstraintDescription>::Borrowed<'_> {
        ConstraintSet::new()
            .with_bounds(BoxBounds::new(Some(&self.lower), Some(&self.upper)))
    }
}

impl CostFunction for BoundedQuadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<f64, Self::Error> {
        Ok((x[0] - 1.0).powi(2) + 4.0 * (x[1] + 2.0).powi(2))
    }
}

impl Gradient for BoundedQuadratic {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Self::Param) -> Result<Self::Gradient, Self::Error> {
        Ok(vec![2.0 * (x[0] - 1.0), 8.0 * (x[1] + 2.0)])
    }
}

fn main() {
    let problem = BoundedQuadratic {
        lower: vec![-1.0, -1.0],
        upper: vec![0.5, 1.0],
    };
    let solver = Lbfgsb::new()
        .with_m_capacity(10)
        .with_absolute_projected_gradient_tolerance(1e-8)
        .with_relative_cost_reduction_tolerance(None);

    let result = Executor::from_start(problem, solver, vec![-3.0, 5.0])
        .max_iter(200)
        .run()
        .unwrap();

    assert!(matches!(
        result.report.termination,
        Termination::Converged(_)
    ));
    assert!((result.param()[0] - 0.5).abs() < 1e-7);
    assert!((result.param()[1] + 1.0).abs() < 1e-7);
}
