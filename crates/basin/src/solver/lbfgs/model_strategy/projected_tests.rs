use super::*;
use crate::{BoxBounds, Constrained, ConstraintSet};
use std::convert::Infallible;

struct Quadratic;
impl CostFunction for Quadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(0.5 * (x[0] - 2.5).powi(2) + 10.0 * (x[1] - 0.51).powi(2))
    }
}
impl Gradient for Quadratic {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(vec![x[0] - 2.5, 20.0 * (x[1] - 0.51)])
    }
}
fn problem() -> Constrained<Quadratic, ConstraintSet<BoxBounds<Vec<f64>>>> {
    Constrained::new(
        Quadratic,
        ConstraintSet::new().with_bounds(BoxBounds::new(
            Some(vec![0.0; 2]),
            Some(vec![1.0; 2]),
        )),
    )
}
#[test]
fn ray_preserves_component_ratios_and_projected_pairs_use_actual_displacement()
{
    for search in [
        ProjectedSearch::FeasibleRay,
        ProjectedSearch::Chord,
        ProjectedSearch::Armijo(
            ProjectedArmijo::new().with_fused_evaluations(),
        ),
    ] {
        let ray = matches!(search, ProjectedSearch::FeasibleRay);
        let mut solver = Lbfgsb::new()
            .with_relative_cost_reduction_tolerance(None)
            .with_compact_model()
            .with_projected_search(search);
        let mut p = Problem::new(problem());
        let initial = vec![0.5, 0.5];
        let state = solver
            .init(&mut p, FirstOrderState::new(initial.clone()))
            .unwrap();
        let next = solver.next_iter(&mut p, state).unwrap().state;
        let expected = if ray { vec![1.0, 0.55] } else { vec![1.0, 0.7] };
        for (a, b) in next.param().iter().zip(&expected) {
            assert!((a - b).abs() < 1e-14);
        }
        let h = solver.model.solver.history.as_ref().unwrap();
        assert_eq!(h.ws.len(), 1);
        for i in 0..2 {
            assert_eq!(h.ws[0][i], next.param()[i] - initial[i]);
        }
        assert_eq!(p.counts().cost_evals, 2);
        assert_eq!(p.counts().gradient_evals, 2);
    }
}
#[test]
fn retention_switch_only_resets_on_mask_changes_and_checkpoint_preserves_mask()
{
    for retain in [true, false] {
        let p = problem();
        let solver = Lbfgsb::new()
            .with_relative_cost_reduction_tolerance(None)
            .with_compact_model()
            .with_projected_search(ProjectedSearch::Armijo(
                ProjectedArmijo::new().with_fused_evaluations(),
            ))
            .with_curvature_retention(retain);
        let mut stepper = crate::Executor::new(
            p,
            solver,
            FirstOrderState::new(vec![0.5, 0.5]),
        )
        .max_iter(2)
        .into_stepper()
        .unwrap();
        stepper.step().unwrap();
        let checkpoint = stepper.into_checkpoint().unwrap();
        #[cfg(feature = "serde")]
        let checkpoint = {
            let bytes = postcard::to_allocvec(&checkpoint).unwrap();
            postcard::from_bytes::<
                crate::ExactCheckpoint<
                    ProjectedLbfgs<Vec<f64>>,
                    FirstOrderState<Vec<f64>>,
                >,
            >(&bytes)
            .unwrap()
        };
        let result =
            crate::Executor::resume_from_checkpoint(problem(), checkpoint)
                .max_iter(2)
                .run_with_solver()
                .unwrap();
        assert_eq!(result.solver.history_len(), if retain { 2 } else { 1 });
        let fresh = crate::Executor::new(
            problem(),
            Lbfgsb::new()
                .with_relative_cost_reduction_tolerance(None)
                .with_compact_model()
                .with_projected_search(ProjectedSearch::Armijo(
                    ProjectedArmijo::new().with_fused_evaluations(),
                ))
                .with_curvature_retention(retain),
            FirstOrderState::new(vec![0.5, 0.5]),
        )
        .max_iter(2)
        .run()
        .unwrap();
        assert_eq!(result.param(), fresh.param());
        assert_eq!(result.state.counts(), fresh.state.counts());
    }
}
