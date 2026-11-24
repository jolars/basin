use basin::solver::lbfgs::model_strategy::ProjectedSearch;
use basin::{Lbfgsb, line_search::projected_armijo::ProjectedArmijo};

fn main() {
    let solver = Lbfgsb::<Vec<f64>>::new()
        .with_m_capacity(10)
        .with_absolute_projected_gradient_tolerance(1e-8)
        .with_relative_cost_reduction_tolerance(None)
        .with_compact_model()
        .with_projected_search(ProjectedSearch::Armijo(
            ProjectedArmijo::new().with_fused_evaluations(),
        ));

    assert!(solver.curvature_retention());
}
