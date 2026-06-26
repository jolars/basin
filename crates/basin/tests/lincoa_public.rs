//! Public-API integration tests for the LINCOA solver.
//!
//! Exercises [`Lincoa`] through the framework: [`Executor`] over a
//! [`LincoaState`], with framework termination ([`MaxCostEvals`],
//! [`RhoTolerance`]) and a problem carrying linear constraints. These confirm
//! the public wiring: init/next_iter, the constraint extraction + folding (all
//! of inequalities, box bounds, and equalities through the general-form
//! [`LinearConstraints`] trait), the V↔Vec bridge, count mirroring, feasibility
//! of the returned point, and the convergence/budget/early-stop termination
//! paths.

use basin::core::constraint::LinearConstraints;
use basin::{
    CostFunction, DenseMatrix, Executor, Lincoa, LincoaState, MaxCostEvals, RhoTolerance,
    TerminationReason,
};

/// `min ‖x − c‖²` subject to `A x ≤ b`, on `Vec<f64>` with the pure-Rust
/// [`DenseMatrix`] constraint carrier (default features, no backend needed).
struct ConstrainedQuadratic {
    c: Vec<f64>,
    a: DenseMatrix<f64>,
    b: Vec<f64>,
}

impl CostFunction for ConstrainedQuadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok((0..x.len()).map(|i| (x[i] - self.c[i]).powi(2)).sum())
    }
}

impl LinearConstraints for ConstrainedQuadratic {
    type Matrix = DenseMatrix<f64>;
    fn inequalities(&self) -> Option<(&DenseMatrix<f64>, &Vec<f64>)> {
        Some((&self.a, &self.b))
    }
}

/// `min ‖x − (2, 2)‖²` s.t. `x0 + x1 ≤ 2` converges to the projection `(1, 1)`,
/// with `f = 2` and a feasible returned point.
#[test]
fn converges_to_projection() {
    let problem = ConstrainedQuadratic {
        c: vec![2.0, 2.0],
        a: DenseMatrix::from_row_slice(1, 2, &[1.0, 1.0]),
        b: vec![2.0],
    };
    let result = Executor::new(
        problem,
        Lincoa::new().with_rho_beg(0.5).with_rho_end(1e-7),
        LincoaState::new(vec![0.0, 0.0]),
    )
    .terminate_on(MaxCostEvals(500))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    let x = result.best_param();
    assert!(
        (x[0] - 1.0).abs() < 1e-3 && (x[1] - 1.0).abs() < 1e-3,
        "x = {x:?}"
    );
    assert!(
        (result.best_cost() - 2.0).abs() < 1e-3,
        "f = {}",
        result.best_cost()
    );
    // The returned point is feasible: x0 + x1 ≤ 2.
    assert!(x[0] + x[1] <= 2.0 + 1e-6, "infeasible point {x:?}");
    assert!(
        result.cost_evals() < 500,
        "cost_evals = {}",
        result.cost_evals()
    );
}

/// Two constraints active at once: `min ‖x − (3, 3)‖²` s.t. `x0 ≤ 1`, `x1 ≤ 1`
/// converges to the corner `(1, 1)`.
#[test]
fn converges_with_two_active_constraints() {
    let problem = ConstrainedQuadratic {
        c: vec![3.0, 3.0],
        a: DenseMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]),
        b: vec![1.0, 1.0],
    };
    let result = Executor::new(
        problem,
        Lincoa::new().with_rho_beg(0.3).with_rho_end(1e-7),
        LincoaState::new(vec![0.0, 0.0]),
    )
    .terminate_on(MaxCostEvals(500))
    .run()
    .unwrap();

    let x = result.best_param();
    assert!(
        (x[0] - 1.0).abs() < 1e-3 && (x[1] - 1.0).abs() < 1e-3,
        "x = {x:?}"
    );
    assert!(x[0] <= 1.0 + 1e-6 && x[1] <= 1.0 + 1e-6, "infeasible {x:?}");
}

/// Box bounds only (no matrix blocks): `min ‖x − (3, 3)‖²` s.t. `x ≤ (1, 1)`
/// folds the upper bounds into `+eᵢ` rows and converges to the corner `(1, 1)`.
/// First public-API coverage of the box-bounds fold path.
struct BoxConstrained {
    c: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl CostFunction for BoxConstrained {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok((0..x.len()).map(|i| (x[i] - self.c[i]).powi(2)).sum())
    }
}

impl LinearConstraints for BoxConstrained {
    // The matrix type is required by the trait but unused: no equality /
    // inequality blocks, only box bounds.
    type Matrix = DenseMatrix<f64>;
    fn lower(&self) -> Option<&Vec<f64>> {
        Some(&self.lower)
    }
    fn upper(&self) -> Option<&Vec<f64>> {
        Some(&self.upper)
    }
}

#[test]
fn box_bounds_fold_and_converge_to_corner() {
    let problem = BoxConstrained {
        c: vec![3.0, 3.0],
        lower: vec![-10.0, -10.0],
        upper: vec![1.0, 1.0],
    };
    let result = Executor::new(
        problem,
        Lincoa::new().with_rho_beg(0.3).with_rho_end(1e-7),
        LincoaState::new(vec![0.0, 0.0]),
    )
    .terminate_on(MaxCostEvals(500))
    .run()
    .unwrap();

    let x = result.best_param();
    assert!(
        (x[0] - 1.0).abs() < 1e-3 && (x[1] - 1.0).abs() < 1e-3,
        "x = {x:?}"
    );
    // The returned point respects the upper bounds.
    assert!(x[0] <= 1.0 + 1e-6 && x[1] <= 1.0 + 1e-6, "infeasible {x:?}");
}

/// Linear equality only: `min ‖x − (2, 2)‖²` s.t. `x0 + x1 = 2` folds to the
/// inequality pair `±(x0+x1) ≤ ±2` and converges to the projection `(1, 1)`.
/// First public-API coverage of the equality fold path.
struct EqualityConstrained {
    c: Vec<f64>,
    a: DenseMatrix<f64>,
    b: Vec<f64>,
}

impl CostFunction for EqualityConstrained {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok((0..x.len()).map(|i| (x[i] - self.c[i]).powi(2)).sum())
    }
}

impl LinearConstraints for EqualityConstrained {
    type Matrix = DenseMatrix<f64>;
    fn equalities(&self) -> Option<(&DenseMatrix<f64>, &Vec<f64>)> {
        Some((&self.a, &self.b))
    }
}

#[test]
fn equality_folds_and_converges_to_projection() {
    let problem = EqualityConstrained {
        c: vec![2.0, 2.0],
        a: DenseMatrix::from_row_slice(1, 2, &[1.0, 1.0]),
        b: vec![2.0],
    };
    // Start on the constraint line x0 + x1 = 2.
    let result = Executor::new(
        problem,
        Lincoa::new().with_rho_beg(0.3).with_rho_end(1e-8),
        LincoaState::new(vec![0.0, 2.0]),
    )
    .terminate_on(MaxCostEvals(500))
    .run()
    .unwrap();

    let x = result.best_param();
    assert!(
        (x[0] - 1.0).abs() < 1e-2 && (x[1] - 1.0).abs() < 1e-2,
        "x = {x:?}"
    );
    // The returned point stays on the equality line.
    assert!((x[0] + x[1] - 2.0).abs() < 1e-4, "off-line {x:?}");
}

#[test]
fn respects_cost_eval_budget() {
    let problem = ConstrainedQuadratic {
        c: vec![2.0, 2.0],
        a: DenseMatrix::from_row_slice(1, 2, &[1.0, 1.0]),
        b: vec![2.0],
    };
    let result = Executor::new(
        problem,
        Lincoa::new().with_rho_beg(0.5).with_rho_end(1e-12),
        LincoaState::new(vec![0.0, 0.0]),
    )
    .terminate_on(MaxCostEvals(15))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxCostEvals);
    assert!(
        result.cost_evals() >= 15,
        "cost_evals = {}",
        result.cost_evals()
    );
}

#[test]
fn rho_tolerance_stops_early() {
    let problem = ConstrainedQuadratic {
        c: vec![2.0, 2.0],
        a: DenseMatrix::from_row_slice(1, 2, &[1.0, 1.0]),
        b: vec![2.0],
    };
    let result = Executor::new(
        problem,
        Lincoa::new().with_rho_beg(0.5).with_rho_end(1e-12),
        LincoaState::new(vec![0.0, 0.0]),
    )
    .terminate_on(RhoTolerance::new(1e-3))
    .terminate_on(MaxCostEvals(5000))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::RhoTolerance);
    assert!(result.state.rho() <= 1e-3, "rho = {}", result.state.rho());
}

/// Backend-generic: drive LINCOA on nalgebra `DMatrix`/`DVector`.
#[cfg(feature = "nalgebra")]
#[test]
fn backend_generic_nalgebra() {
    use nalgebra::{DMatrix, DVector};

    struct Proj {
        a: DMatrix<f64>,
        b: DVector<f64>,
    }
    impl CostFunction for Proj {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
            Ok((x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2))
        }
    }
    impl LinearConstraints for Proj {
        type Matrix = DMatrix<f64>;
        fn inequalities(&self) -> Option<(&DMatrix<f64>, &DVector<f64>)> {
            Some((&self.a, &self.b))
        }
    }

    let problem = Proj {
        a: DMatrix::from_row_slice(1, 2, &[1.0, 1.0]),
        b: DVector::from_vec(vec![2.0]),
    };
    let result = Executor::new(
        problem,
        Lincoa::new().with_rho_beg(0.5).with_rho_end(1e-7),
        LincoaState::new(DVector::from_vec(vec![0.0, 0.0])),
    )
    .terminate_on(MaxCostEvals(500))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    let x = result.best_param();
    assert!(
        (x[0] - 1.0).abs() < 1e-3 && (x[1] - 1.0).abs() < 1e-3,
        "x = {x:?}"
    );
}

/// Backend-generic: drive LINCOA on ndarray `Array2`/`Array1`. Guards the
/// support-matrix ✓ for ndarray; the param vector must satisfy
/// `VectorLen + IndexMut` and the constraint matrix `MatTransposeVec`.
#[cfg(feature = "ndarray")]
#[test]
fn backend_generic_ndarray() {
    use ndarray::{Array1, Array2};

    struct Proj {
        a: Array2<f64>,
        b: Array1<f64>,
    }
    impl CostFunction for Proj {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Array1<f64>) -> Result<f64, std::convert::Infallible> {
            Ok((x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2))
        }
    }
    impl LinearConstraints for Proj {
        type Matrix = Array2<f64>;
        fn inequalities(&self) -> Option<(&Array2<f64>, &Array1<f64>)> {
            Some((&self.a, &self.b))
        }
    }

    let problem = Proj {
        a: Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap(),
        b: Array1::from_vec(vec![2.0]),
    };
    let result = Executor::new(
        problem,
        Lincoa::new().with_rho_beg(0.5).with_rho_end(1e-7),
        LincoaState::new(Array1::from_vec(vec![0.0, 0.0])),
    )
    .terminate_on(MaxCostEvals(500))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    let x = result.best_param();
    assert!(
        (x[0] - 1.0).abs() < 1e-3 && (x[1] - 1.0).abs() < 1e-3,
        "x = {x:?}"
    );
}

/// Backend-generic: drive LINCOA on faer `Mat`/`Col`. Guards the
/// support-matrix ✓ for faer.
#[cfg(feature = "faer")]
#[test]
fn backend_generic_faer() {
    use faer::{Col, Mat};

    struct Proj {
        a: Mat<f64>,
        b: Col<f64>,
    }
    impl CostFunction for Proj {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            Ok((x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2))
        }
    }
    impl LinearConstraints for Proj {
        type Matrix = Mat<f64>;
        fn inequalities(&self) -> Option<(&Mat<f64>, &Col<f64>)> {
            Some((&self.a, &self.b))
        }
    }

    let problem = Proj {
        a: Mat::from_fn(1, 2, |_, _| 1.0),
        b: Col::from_fn(1, |_| 2.0),
    };
    let result = Executor::new(
        problem,
        Lincoa::new().with_rho_beg(0.5).with_rho_end(1e-7),
        LincoaState::new(Col::from_fn(2, |_| 0.0)),
    )
    .terminate_on(MaxCostEvals(500))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    let x = result.best_param();
    assert!(
        (x[0] - 1.0).abs() < 1e-3 && (x[1] - 1.0).abs() < 1e-3,
        "x = {x:?}"
    );
}
