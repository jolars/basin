//! Shared simplex progress and solver-owned Nelder-Mead workspaces.

use basin::{CostFunction, Executor, NelderMead, SimplexProgress, State};
use std::convert::Infallible;

struct Sphere;
impl CostFunction for Sphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(x.iter().map(|x| x * x).sum())
    }
}

#[test]
fn fresh_simplex_run_resets_iteration_and_reevaluates_vertices() {
    let prior = Executor::new(
        Sphere,
        NelderMead::adaptive(),
        SimplexProgress::new(vec![2.0, -1.0]),
    )
    .max_iter(5)
    .run_with_solver()
    .unwrap();
    let fresh = Executor::new(Sphere, prior.solver, prior.state)
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    assert_eq!(fresh.state.iter(), 0);
    assert_eq!(fresh.state.best_iter(), 0);
    assert_eq!(fresh.counts.cost_evals, 3);
}

fn coherent(state: &basin::SimplexProgress<Vec<f64>>) {
    let (vertices, costs) = state.evaluated_vertices().unwrap();
    for (x, &cost) in vertices.iter().zip(costs) {
        assert_eq!(Sphere.cost(x).unwrap(), cost);
    }
    assert!(costs.windows(2).all(|pair| pair[0] <= pair[1]));
    let (x, cost) = state.best().unwrap();
    assert_eq!(Sphere.cost(x).unwrap(), cost);
}

#[test]
fn exact_continuation_retains_simplex_and_workspace() {
    let make = || {
        Executor::from_start(Sphere, NelderMead::adaptive(), vec![2.0, -1.0])
    };
    let expected = make().max_iter(60).run_with_solver().unwrap();
    for split in [0, 1, 7, 23, 59] {
        let checkpoint = make()
            .max_iter(split)
            .run_with_solver()
            .unwrap()
            .into_checkpoint();
        let resumed = Executor::resume_from_checkpoint(Sphere, checkpoint)
            .require_evaluated_state()
            .max_iter(60)
            .run_with_solver()
            .unwrap();
        coherent(&resumed.state);
        assert_eq!(resumed.state, expected.state);
        assert_eq!(resumed.counts, expected.counts);
    }
}

#[test]
fn reused_solver_rebuilds_scratch_and_adaptive_parameters_for_new_dimension() {
    let prior =
        Executor::from_start(Sphere, NelderMead::adaptive(), vec![2.0, -1.0])
            .max_iter(10)
            .run_with_solver()
            .unwrap();
    let seed = vec![3.0, -2.0, 1.0];
    let restarted = Executor::from_start(Sphere, prior.solver, seed.clone())
        .require_evaluated_state()
        .max_iter(80)
        .run_with_solver()
        .unwrap();
    let rebuilt = Executor::from_start(Sphere, NelderMead::adaptive(), seed)
        .require_evaluated_state()
        .max_iter(80)
        .run_with_solver()
        .unwrap();
    coherent(&restarted.state);
    assert_eq!(restarted.state, rebuilt.state);
    assert_eq!(restarted.counts, rebuilt.counts);
}

#[test]
fn simplex_replacements_preserve_historical_pairs_and_raw_metadata() {
    use basin::{CountsMirror, EvalCounts, IncumbentState, SimplexProgress};
    let mut state = SimplexProgress::from_simplex(vec![vec![1.0], vec![2.0]]);
    assert!(state.current().is_none());
    assert!(state.evaluated_vertices().is_none());
    assert!(state.best().is_none());
    assert!(state.costs().is_empty());
    state
        .replace(vec![vec![2.0], vec![1.0]], vec![4.0, 1.0])
        .unwrap();
    let counts = EvalCounts {
        cost_evals: 2,
        gradient_evals: 3,
        residual_evals: 4,
        jacobian_evals: 5,
        hessian_evals: 6,
        hessian_product_evals: 7,
    };
    state.mirror(&counts);
    state.update_best();
    assert_eq!(state.current(), Some((&vec![1.0], 1.0)));
    assert_eq!(state.best_counts(), Some(&counts));
    let vertices_ptr = state.vertices().as_ptr();
    let (mut vertices, mut costs) = state.take_vertices();
    assert_eq!(vertices.as_ptr(), vertices_ptr);
    assert!(state.current().is_none());
    vertices[0][0] = 3.0;
    vertices[1][0] = 4.0;
    costs[0] = 9.0;
    costs[1] = 16.0;
    state.replace(vertices, costs).unwrap();
    assert_eq!(state.vertices().as_ptr(), vertices_ptr);
    state.increment_iter();
    state.mirror(&EvalCounts {
        cost_evals: 4,
        ..counts
    });
    state.update_best();
    coherent(&state);
    assert_eq!(state.current(), Some((&vec![3.0], 9.0)));
    assert_eq!(state.best(), Some((&vec![1.0], 1.0)));
    assert_eq!(state.incumbent_record().unwrap().iter, 0);
    assert_eq!(state.best_counts(), Some(&counts));
    state
        .replace(vec![vec![-1.0], vec![0.0]], vec![1.0, 0.0])
        .unwrap();
    let final_counts = EvalCounts {
        cost_evals: 6,
        ..counts
    };
    state.mirror(&final_counts);
    // This boundary models a clean mid-step stop, with no completed iteration.
    state.update_best();
    assert_eq!(state.best_iter(), 1);
    assert_eq!(state.best_counts(), Some(&final_counts));
    state.increment_iter();
    state.update_best();
    assert_eq!(state.best_iter(), 1);
    state.reset();
    assert!(state.best().is_none());
    assert!(state.current().is_none());
    assert_eq!(state.iter(), 0);
    assert_eq!(state.counts(), &EvalCounts::default());
}

#[test]
fn invalid_replacements_preserve_the_complete_previous_state() {
    use basin::SimplexProgress;
    let mut state = SimplexProgress::from_simplex(vec![vec![1.0], vec![2.0]]);
    state
        .replace(vec![vec![1.0], vec![2.0]], vec![1.0, 4.0])
        .unwrap();
    state.update_best();
    let before = state.clone();
    for (vertices, costs) in [
        (vec![], vec![]),
        (vec![vec![1.0]], vec![1.0]),
        (vec![vec![1.0], vec![2.0], vec![3.0]], vec![1.0, 4.0, 9.0]),
        (vec![vec![1.0], vec![2.0, 3.0]], vec![1.0, 13.0]),
        (vec![vec![1.0], vec![2.0]], vec![1.0]),
    ] {
        assert!(state.replace(vertices, costs).is_err());
        assert_eq!(state, before);
    }
}

#[test]
fn nonfinite_costs_and_ties_have_stable_selection_semantics() {
    use basin::SimplexProgress;
    let mut state = SimplexProgress::from_simplex(vec![vec![0.0], vec![1.0]]);
    state
        .replace(vec![vec![0.0], vec![1.0]], vec![f64::NAN, f64::INFINITY])
        .unwrap();
    state.update_best();
    assert_eq!(state.current().unwrap().1, f64::INFINITY);
    assert!(state.costs()[1].is_nan());
    assert!(state.best().is_none());
    state
        .replace(vec![vec![1.0], vec![-1.0]], vec![1.0, 1.0])
        .unwrap();
    state.update_best();
    assert_eq!(state.best(), Some((&vec![1.0], 1.0)));
    state.increment_iter();
    state
        .replace(vec![vec![-1.0], vec![1.0]], vec![1.0, 1.0])
        .unwrap();
    state.update_best();
    assert_eq!(state.current(), Some((&vec![-1.0], 1.0)));
    assert_eq!(state.best(), Some((&vec![1.0], 1.0)));
    assert_eq!(state.best_iter(), 0);
    state
        .replace(vec![vec![0.0], vec![1.0]], vec![f64::NEG_INFINITY, 1.0])
        .unwrap();
    state.update_best();
    assert_eq!(state.best(), Some((&vec![0.0], f64::NEG_INFINITY)));
}

#[cfg(feature = "serde")]
#[test]
fn serialized_checkpoint_keeps_workspace_while_progress_only_restarts() {
    type Checkpoint = basin::ExactCheckpoint<
        NelderMead<Vec<f64>>,
        basin::SimplexProgress<Vec<f64>>,
    >;
    let make =
        || Executor::from_start(Sphere, NelderMead::new(), vec![2.0, -1.0]);
    let expected = make().max_iter(50).run_with_solver().unwrap();
    let checkpoint = make()
        .max_iter(17)
        .run_with_solver()
        .unwrap()
        .into_checkpoint();
    let bytes = postcard::to_allocvec(&checkpoint).unwrap();
    let restored: Checkpoint = postcard::from_bytes(&bytes).unwrap();
    let resumed = Executor::resume_from_checkpoint(Sphere, restored)
        .require_evaluated_state()
        .max_iter(50)
        .run_with_solver()
        .unwrap();
    assert_eq!(resumed.state, expected.state);
    assert_eq!(resumed.counts, expected.counts);
    assert_eq!(
        postcard::to_allocvec(&resumed.solver).unwrap(),
        postcard::to_allocvec(&expected.solver).unwrap()
    );
    let restarted = Executor::new(Sphere, resumed.solver, resumed.state)
        .require_evaluated_state()
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    assert_eq!(restarted.iter(), 0);
    assert_eq!(restarted.counts.cost_evals, 3);
}

impl basin::Gradient for Sphere {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(x.iter().map(|x| 2.0 * x).collect())
    }
}
impl basin::Residual for Sphere {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = Infallible;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(x.clone())
    }
}
impl basin::Jacobian for Sphere {
    type Jacobian = basin::DenseMatrix;
    fn jacobian(&self, x: &Vec<f64>) -> Result<basin::DenseMatrix, Infallible> {
        Ok(basin::MatrixIdentity::identity(x.len()))
    }
}
impl basin::Hessian for Sphere {
    type Hessian = basin::DenseMatrix;
    fn hessian(&self, x: &Vec<f64>) -> Result<basin::DenseMatrix, Infallible> {
        Ok(basin::MatrixFromDiagonal::from_diagonal(&vec![
            2.0;
            x.len()
        ]))
    }
}
impl basin::HessianProduct for Sphere {
    fn hessian_product(
        &self,
        _: &Vec<f64>,
        v: &Vec<f64>,
    ) -> Result<Vec<f64>, Infallible> {
        Ok(v.iter().map(|x| 2.0 * x).collect())
    }
}

struct ExternalSimplex;
impl basin::Solver<Sphere, basin::SimplexProgress<Vec<f64>>>
    for ExternalSimplex
{
    type Error = Infallible;
    fn init(
        &mut self,
        p: &mut basin::Problem<Sphere>,
        mut state: basin::SimplexProgress<Vec<f64>>,
    ) -> Result<basin::SimplexProgress<Vec<f64>>, Infallible> {
        state.reset();
        let (vertices, _) = state.take_vertices();
        let costs = vertices
            .iter()
            .map(|x| p.cost(x))
            .collect::<Result<Vec<_>, _>>()?;
        state.replace(vertices, costs).unwrap();
        Ok(state)
    }
    fn next_iter(
        &mut self,
        p: &mut basin::Problem<Sphere>,
        mut state: basin::SimplexProgress<Vec<f64>>,
    ) -> Result<
        (
            basin::SimplexProgress<Vec<f64>>,
            Option<basin::TerminationReason>,
        ),
        Infallible,
    > {
        let (mut vertices, mut costs) = state.take_vertices();
        for (v, cost) in vertices.iter_mut().zip(&mut costs) {
            for x in v.iter_mut() {
                *x *= 0.5;
            }
            *cost = p.cost(v)?;
        }
        p.cost_and_gradient_and_hessian(&vertices[0])?;
        p.residual_and_jacobian(&vertices[0])?;
        p.hessian_product(&vertices[0], &vertices[0])?;
        state.replace(vertices, costs).unwrap();
        Ok((state, Some(basin::TerminationReason::UserRequested)))
    }
}

#[test]
fn external_simplex_updates_preserve_all_categories_at_midstep_publication() {
    let result = Executor::new(
        Sphere,
        ExternalSimplex,
        basin::SimplexProgress::from_simplex(vec![vec![4.0], vec![5.0]]),
    )
    .require_evaluated_state()
    .run_with_solver()
    .unwrap();
    assert_eq!(result.reason, basin::TerminationReason::UserRequested);
    assert_eq!(result.iter(), 0);
    coherent(&result.state);
    assert_eq!(result.state.best(), Some((&vec![2.0], 4.0)));
    assert_eq!(
        result.counts,
        basin::EvalCounts {
            cost_evals: 5,
            gradient_evals: 1,
            residual_evals: 1,
            jacobian_evals: 1,
            hessian_evals: 1,
            hessian_product_evals: 1,
        }
    );
    assert_eq!(result.state.counts(), &result.counts);
    assert_eq!(result.state.best_counts(), Some(&result.counts));
    assert_eq!(result.state.cost_evals(), 5);
    assert_eq!(result.state.best_iter(), 0);
}
