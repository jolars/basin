use basin::{
    Bobyqa, BoxBounds, ConstraintDescription, ConstraintSet, Constraints,
    CostFunction, Executor, Termination,
};
use std::convert::Infallible;

struct BoundedQuadratic {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl CostFunction for BoundedQuadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<f64, Self::Error> {
        Ok((x[0] - 2.0).powi(2) + 4.0 * (x[1] + 1.0).powi(2))
    }
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

fn main() {
    let problem = BoundedQuadratic {
        lower: vec![-2.0, -3.0],
        upper: vec![1.0, 2.0],
    };
    let solver = Bobyqa::new()
        .with_initial_radius(0.5)
        .with_final_radius(1e-8);

    let result = Executor::from_start(problem, solver, vec![0.0, 1.0])
        .max_cost_evals(500)
        .run()
        .unwrap();

    assert!(matches!(
        result.report.termination,
        Termination::Converged(_)
    ));
    let incumbent = result.selected_record().expect("expected an incumbent");
    let x = incumbent.param;
    assert!((-2.0..=1.0).contains(&x[0]));
    assert!((-3.0..=2.0).contains(&x[1]));
    assert!((x[0] - 1.0).abs() < 1e-6);
    assert!((x[1] + 1.0).abs() < 1e-6);
    assert!((incumbent.cost.unwrap() - 1.0).abs() < 1e-10);
}
