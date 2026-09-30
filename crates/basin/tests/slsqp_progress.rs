//! Shared constrained first-order records and raw evaluation categories.

use basin::{
    CountsMirror, EvalCounts, GradientState, SelectedFirstOrderState, State,
};

#[test]
fn category_readers_do_not_fold_constraint_work() {
    let counts = EvalCounts {
        cost_evals: 2,
        gradient_evals: 3,
        residual_evals: 5,
        jacobian_evals: 7,
        hessian_evals: 11,
        hessian_product_evals: 13,
    };
    let mut state = SelectedFirstOrderState::<Vec<f64>>::new(vec![0.0]);
    state.mirror(&counts);
    assert_eq!(state.cost_evals(), 2);
    assert_eq!(state.gradient_evals(), 3);
    assert_eq!(state.counts(), &counts);
}

#[test]
fn public_records_validate_dimensions_and_preserve_pending_selections() {
    let mut state = SelectedFirstOrderState::new(vec![0.0]);
    assert!(state.current().is_none());
    assert!(state.gradient().is_none());
    assert!(!state.select_current());
    state.replace(vec![0.0], 0.0, vec![0.0], 1.0).unwrap();
    state.select_current();
    state.mirror(&EvalCounts {
        cost_evals: 1,
        gradient_evals: 2,
        residual_evals: 3,
        jacobian_evals: 4,
        hessian_evals: 5,
        hessian_product_evals: 6,
    });
    state.update_best();
    state.replace(vec![1.0], 1.0, vec![2.0], 0.0).unwrap();
    state.select_current();
    let unchanged = state.clone();
    let error = state.replace(vec![2.0], 4.0, vec![], 1.0).unwrap_err();
    assert_eq!(error.param_len, 1);
    assert_eq!(error.gradient_len, 0);
    assert_eq!(state, unchanged);
    state.replace(vec![2.0], 4.0, vec![4.0], 1.0).unwrap();
    state.increment_iter();
    let counts = EvalCounts {
        cost_evals: 7,
        gradient_evals: 8,
        residual_evals: 9,
        jacobian_evals: 10,
        hessian_evals: 11,
        hessian_product_evals: 12,
    };
    state.mirror(&counts);
    state.update_best();
    assert_eq!(state.best(), Some((&vec![1.0], 1.0, 0.0)));
    assert_eq!(state.current(), Some((&vec![2.0], 4.0, &vec![4.0], 1.0)));
    assert_eq!(state.best_counts(), Some(&counts));
    assert_eq!(state.best_cost_evals(), 7);
    assert_eq!(state.best_gradient_evals(), 8);
    state.increment_iter();
    state.update_best();
    assert_eq!(state.best_iter(), 1);
    state.reset();
    assert!(state.current().is_none());
    assert!(state.best().is_none());
    assert!(state.gradient().is_none());
    assert_eq!(state.param(), &vec![2.0]);
    assert_eq!(state.counts(), &EvalCounts::default());
}

struct Equality;
impl basin::CostFunction for Equality {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        Ok(x.iter().map(|x| x * x).sum())
    }
}
impl basin::Gradient for Equality {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(x.iter().map(|x| 2.0 * x).collect())
    }
}
impl basin::NonlinearConstraints for Equality {
    type Matrix = basin::DenseMatrix;
    fn num_nonlinear_constraints(&self) -> usize {
        0
    }
    fn nonlinear_constraints(
        &self,
        _: &Vec<f64>,
    ) -> Result<Vec<f64>, Self::Error> {
        Ok(vec![])
    }
    fn num_nonlinear_equalities(&self) -> usize {
        1
    }
    fn nonlinear_equalities(
        &self,
        x: &Vec<f64>,
    ) -> Result<Option<Vec<f64>>, Self::Error> {
        Ok(Some(vec![x[0] - 1.0]))
    }
}
impl basin::ConstraintJacobian for Equality {
    fn constraint_jacobian(
        &self,
        x: &Vec<f64>,
    ) -> Result<basin::DenseMatrix, Self::Error> {
        let mut row = vec![0.0; x.len()];
        row[0] = 1.0;
        Ok(basin::DenseMatrix::from_row_slice(1, x.len(), &row))
    }
}

#[test]
fn fresh_slsqp_rebuilds_diagnostics_and_reevaluates_progress() {
    use basin::{Executor, Slsqp};
    let used = Executor::from_start(Equality, Slsqp::new(), vec![0.0; 2])
        .max_iter(5)
        .run_with_solver()
        .unwrap();
    assert!(used.state.best_cost() > 0.99);
    assert_eq!(used.state.current().unwrap().2, &vec![2.0, 0.0]);
    let mut state = used.state;
    state
        .replace(vec![0.0; 3], -100.0, vec![100.0; 3], 0.0)
        .unwrap();
    let fresh = Executor::new(Equality, used.solver, state)
        .require_evaluated_state()
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    assert_eq!(
        fresh.state.current(),
        Some((&vec![0.0; 3], 0.0, &vec![0.0; 3], 1.0))
    );
    assert_eq!(fresh.state.best_iter(), 0);
    assert_eq!(fresh.iter(), 0);
    assert_eq!(fresh.state.counts(), &fresh.counts);
    assert_eq!(fresh.state.cost_evals(), 1);
    assert_eq!(fresh.state.gradient_evals(), 1);
    assert_eq!(fresh.solver.constraint_violation(), Some(1.0));
    assert_eq!(fresh.solver.failure(), None);
}
