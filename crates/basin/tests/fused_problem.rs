//! Fused-evaluation tests for the defaulted methods on
//! [`Gradient::cost_and_gradient`], [`Hessian::cost_and_gradient_and_hessian`],
//! and [`Jacobian::residual_and_jacobian`].
//!
//! Three things to verify:
//!
//! 1. A user who impls only `CostFunction + Gradient` (no override)
//!    transparently gets `cost_and_gradient` for free, and a migrated
//!    gradient solver runs against them with zero opt-in.
//! 2. A user who *overrides* `cost_and_gradient` is actually called by
//!    the solver (proven via a shared counter).
//! 3. `FiniteDiff` flows into the migrated solver bound; a values-only
//!    problem still works through the fused method.

use std::cell::Cell;
use std::rc::Rc;

use basin::{
    BasicState, CostFunction, Executor, FiniteDiff, Gradient, GradientDescent,
    GradientTolerance, MaxIter,
};

// ---------------------------------------------------------------------
// 1. No-opt-in user: defaulted body works.
// ---------------------------------------------------------------------

struct Sphere;
impl CostFunction for Sphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(x.iter().map(|xi| xi * xi).sum())
    }
}
impl Gradient for Sphere {
    type Gradient = Vec<f64>;
    fn gradient(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, std::convert::Infallible> {
        Ok(x.iter().map(|xi| 2.0 * xi).collect())
    }
}

#[test]
fn default_fused_matches_separate_calls() {
    let p = Sphere;
    let x = vec![1.5, -2.25, 0.5];
    let (fused_cost, fused_grad) = p.cost_and_gradient(&x).unwrap();
    assert_eq!(fused_cost, p.cost(&x).unwrap());
    assert_eq!(fused_grad, p.gradient(&x).unwrap());
}

#[test]
fn gradient_descent_runs_with_no_opt_in() {
    // Sphere has no `cost_and_gradient` override; the defaulted body
    // is what the solver hits. No extra trait impl required.
    let solver = GradientDescent::new(0.1);
    let state = BasicState::new(vec![1.0, 1.0]);
    let result = Executor::new(Sphere, solver, state)
        .terminate_on(MaxIter(200))
        .terminate_on(GradientTolerance(1e-10))
        .run()
        .unwrap();
    assert!(result.cost() < 1e-8, "got cost {}", result.cost());
}

// ---------------------------------------------------------------------
// 2. Override is called by the solver.
// ---------------------------------------------------------------------

struct Counted {
    fused_calls: Rc<Cell<usize>>,
}
impl CostFunction for Counted {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(x.iter().map(|xi| xi * xi).sum())
    }
}
impl Gradient for Counted {
    type Gradient = Vec<f64>;
    fn gradient(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, std::convert::Infallible> {
        Ok(x.iter().map(|xi| 2.0 * xi).collect())
    }
    fn cost_and_gradient(
        &self,
        x: &Vec<f64>,
    ) -> Result<(f64, Vec<f64>), std::convert::Infallible> {
        Ok({
            self.fused_calls.set(self.fused_calls.get() + 1);
            let c = x.iter().map(|xi| xi * xi).sum();
            let g = x.iter().map(|xi| 2.0 * xi).collect();
            (c, g)
        })
    }
}

#[test]
fn solver_calls_fused_override() {
    let counter = Rc::new(Cell::new(0_usize));
    let problem = Counted {
        fused_calls: counter.clone(),
    };
    let solver = GradientDescent::new(0.1);
    let state = BasicState::new(vec![1.0, 1.0]);
    let result = Executor::new(problem, solver, state)
        .terminate_on(MaxIter(200))
        .terminate_on(GradientTolerance(1e-10))
        .run()
        .unwrap();

    // GD's init + each next_iter goes through the fused call.
    let expected_min = result.iter() as usize + 1;
    let calls = counter.get();
    assert!(
        calls >= expected_min,
        "expected at least {expected_min} fused calls, got {calls}"
    );
    assert!(result.cost() < 1e-8, "got cost {}", result.cost());
}

// ---------------------------------------------------------------------
// 3. CostAndGradientAndHessian: defaulted body equals separate calls.
// ---------------------------------------------------------------------

#[cfg(feature = "nalgebra_all")]
mod hessian {
    use super::*;
    use crate::backend_aliases::nalgebra::{DMatrix, DVector};
    use basin::Hessian;

    struct SphereN;
    impl CostFunction for SphereN {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &DVector<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(x.dot(x))
        }
    }
    impl Gradient for SphereN {
        type Gradient = DVector<f64>;
        fn gradient(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, std::convert::Infallible> {
            Ok(2.0 * x)
        }
    }
    impl Hessian for SphereN {
        type Hessian = DMatrix<f64>;
        fn hessian(
            &self,
            x: &DVector<f64>,
        ) -> Result<DMatrix<f64>, std::convert::Infallible> {
            Ok(2.0 * DMatrix::identity(x.len(), x.len()))
        }
    }

    #[test]
    fn default_triple_matches_separate_calls() {
        let p = SphereN;
        let x = DVector::from_vec(vec![1.0, -2.0, 3.0]);
        let (c, g, h) = p.cost_and_gradient_and_hessian(&x).unwrap();
        assert_eq!(c, p.cost(&x).unwrap());
        assert_eq!(g, p.gradient(&x).unwrap());
        assert_eq!(h, p.hessian(&x).unwrap());
    }

    /// User overrides `cost_and_gradient_and_hessian` to share work.
    /// No Newton solver ships yet; exercise the override directly.
    struct CountedTriple {
        fused_calls: Rc<Cell<usize>>,
    }
    impl CostFunction for CountedTriple {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &DVector<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(x.dot(x))
        }
    }
    impl Gradient for CountedTriple {
        type Gradient = DVector<f64>;
        fn gradient(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, std::convert::Infallible> {
            Ok(2.0 * x)
        }
    }
    impl Hessian for CountedTriple {
        type Hessian = DMatrix<f64>;
        fn hessian(
            &self,
            x: &DVector<f64>,
        ) -> Result<DMatrix<f64>, std::convert::Infallible> {
            Ok(2.0 * DMatrix::identity(x.len(), x.len()))
        }
        fn cost_and_gradient_and_hessian(
            &self,
            x: &DVector<f64>,
        ) -> Result<(f64, DVector<f64>, DMatrix<f64>), std::convert::Infallible>
        {
            Ok({
                self.fused_calls.set(self.fused_calls.get() + 1);
                let c = x.dot(x);
                let g = 2.0 * x;
                let h = 2.0 * DMatrix::identity(x.len(), x.len());
                (c, g, h)
            })
        }
    }

    #[test]
    fn override_triple_is_called() {
        let counter = Rc::new(Cell::new(0_usize));
        let p = CountedTriple {
            fused_calls: counter.clone(),
        };
        let x = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let _ = p.cost_and_gradient_and_hessian(&x).unwrap();
        let _ = p.cost_and_gradient_and_hessian(&x).unwrap();
        assert_eq!(counter.get(), 2);
    }
}

// ---------------------------------------------------------------------
// 4. ResidualAndJacobian: LM actually calls the fused override.
// ---------------------------------------------------------------------

#[cfg(feature = "nalgebra_all")]
mod lsq {
    use super::*;
    use crate::backend_aliases::nalgebra::{DMatrix, DVector};
    use basin::{Jacobian, LevenbergMarquardt, NllsState, Residual};

    /// `r(x) = (x₀ − 1, x₁ − 2)`, J = I. Minimum at (1, 2).
    struct Affine {
        fused_calls: Rc<Cell<usize>>,
    }
    impl CostFunction for Affine {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &DVector<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(0.5 * ((x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2)))
        }
    }
    impl Residual for Affine {
        type Param = DVector<f64>;
        type Output = DVector<f64>;
        type Error = std::convert::Infallible;
        fn residual(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, std::convert::Infallible> {
            Ok(DVector::from_vec(vec![x[0] - 1.0, x[1] - 2.0]))
        }
    }
    impl Jacobian for Affine {
        type Jacobian = DMatrix<f64>;
        fn jacobian(
            &self,
            _x: &DVector<f64>,
        ) -> Result<DMatrix<f64>, std::convert::Infallible> {
            Ok(DMatrix::identity(2, 2))
        }
        fn residual_and_jacobian(
            &self,
            x: &DVector<f64>,
        ) -> Result<(DVector<f64>, DMatrix<f64>), std::convert::Infallible>
        {
            Ok({
                self.fused_calls.set(self.fused_calls.get() + 1);
                (
                    DVector::from_vec(vec![x[0] - 1.0, x[1] - 2.0]),
                    DMatrix::identity(2, 2),
                )
            })
        }
    }

    #[test]
    fn lm_calls_fused_residual_and_jacobian() {
        let counter = Rc::new(Cell::new(0_usize));
        let problem = Affine {
            fused_calls: counter.clone(),
        };
        let state = NllsState::new(DVector::from_vec(vec![0.0, 0.0]));
        let solver: LevenbergMarquardt<DVector<f64>, DMatrix<f64>> =
            LevenbergMarquardt::new();
        let result = Executor::new(problem, solver, state)
            .terminate_on(MaxIter(10))
            .run()
            .unwrap();

        // LM init does one fused call; subsequent iters reuse caches.
        assert!(counter.get() >= 1);
        let x = result.param();
        assert!((x[0] - 1.0).abs() < 1e-8);
        assert!((x[1] - 2.0).abs() < 1e-8);
    }
}

// ---------------------------------------------------------------------
// 5. `FiniteDiff` is usable with the migrated solver bound.
// ---------------------------------------------------------------------

/// Cost-only problem (no `Gradient` impl).
struct CostOnly;
impl CostFunction for CostOnly {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(x.iter().map(|xi| (xi - 1.0).powi(2)).sum())
    }
}

#[test]
fn finite_diff_runs_through_solver() {
    let problem = FiniteDiff::new(CostOnly);
    let solver = GradientDescent::new(0.1);
    let state = BasicState::new(vec![0.0, 0.0]);
    let result = Executor::new(problem, solver, state)
        .terminate_on(MaxIter(200))
        .terminate_on(GradientTolerance(1e-8))
        .run()
        .unwrap();

    let x = result.param();
    assert!((x[0] - 1.0).abs() < 1e-3);
    assert!((x[1] - 1.0).abs() < 1e-3);
}

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
