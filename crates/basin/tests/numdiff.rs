//! End-to-end: drive real solvers on problems whose derivatives are
//! synthesized by `FiniteDiff` from function values only.

#![cfg(feature = "problems")]

use basin::problems::Sphere;
use basin::{
    BasicState, Executor, FiniteDiff, GradientDescent, GradientTolerance, TerminationReason,
};

#[test]
fn gradient_descent_on_finite_diff_sphere_converges() {
    // Sphere exposes only `cost`; `FiniteDiff` supplies the gradient by
    // central differences. Gradient descent should still march to the
    // origin — the FD gradient is accurate enough to drive a first-order
    // solver and routes through the backend-generic `V: ScaledAdd + …`
    // bounds.
    let problem = FiniteDiff::new(Sphere::<Vec<f64>>::new());
    let initial = vec![1.5, -2.0, 0.75, 3.0];

    let result = Executor::new(problem, GradientDescent::new(0.2), BasicState::new(initial))
        .max_iter(500)
        .terminate_on(GradientTolerance(1e-9))
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::GradientTolerance);
    assert!(result.cost() < 1e-12, "cost = {}", result.cost());
    for (i, &xi) in result.param().iter().enumerate() {
        assert!(xi.abs() < 1e-6, "x[{i}] = {xi} (expected near 0)");
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra {
    use basin::problems::{Rosenbrock, RosenbrockResiduals};
    use basin::{
        BasicState, Executor, FiniteDiff, GradientTolerance, LevenbergMarquardt, Method, NllsState,
        TerminationReason, TrustRegion,
    };
    use nalgebra::DVector;

    #[test]
    fn trust_region_on_finite_diff_hessian_minimizes_rosenbrock() {
        // Rosenbrock exposes only `cost`; `FiniteDiff` supplies both the
        // gradient and the (central-difference) Hessian the trust-region
        // Newton solver consumes. The FD Hessian is only ~√ε accurate, so a
        // strong-but-not-machine-precision bound is the honest check.
        let problem =
            FiniteDiff::new(Rosenbrock::<DVector<f64>>::new()).hessian_method(Method::Central);

        let result = Executor::new(
            problem,
            TrustRegion::new(),
            BasicState::new(DVector::from_vec(vec![-1.2, 1.0])),
        )
        .max_iter(300)
        .terminate_on(GradientTolerance(1e-6))
        .run()
        .unwrap();

        assert!(result.cost() < 1e-8, "cost = {}", result.cost());
        assert!((result.param()[0] - 1.0).abs() < 1e-3);
        assert!((result.param()[1] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn levenberg_marquardt_on_finite_diff_jacobian_matches_analytic() {
        // The eunoia/lmdif-parity smoke test: LM with a forward-difference
        // (MINPACK fdjac2) Jacobian must converge to (1, 1) and track the
        // analytic-Jacobian run closely.
        let initial = DVector::from_vec(vec![-1.2, 1.0]);

        let analytic = Executor::new(
            RosenbrockResiduals::<DVector<f64>>::new(),
            LevenbergMarquardt::new(),
            NllsState::new(initial.clone()),
        )
        .max_iter(100)
        .run()
        .unwrap();

        let fd = Executor::new(
            FiniteDiff::new(RosenbrockResiduals::<DVector<f64>>::new()),
            LevenbergMarquardt::new(),
            NllsState::new(initial),
        )
        .max_iter(100)
        .run()
        .unwrap();

        assert_eq!(fd.reason, TerminationReason::SolverConverged);
        assert!(fd.cost() < 1e-12, "fd cost = {}", fd.cost());
        assert!(
            (fd.param()[0] - 1.0).abs() < 1e-6,
            "x[0] = {}",
            fd.param()[0]
        );
        assert!(
            (fd.param()[1] - 1.0).abs() < 1e-6,
            "x[1] = {}",
            fd.param()[1]
        );

        // FD and analytic Jacobians should land on essentially the same
        // optimum.
        assert!((fd.param()[0] - analytic.param()[0]).abs() < 1e-5);
        assert!((fd.param()[1] - analytic.param()[1]).abs() < 1e-5);
    }
}

#[cfg(feature = "faer")]
mod faer {
    use basin::problems::RosenbrockResiduals;
    use basin::{Executor, FiniteDiff, LevenbergMarquardt, NllsState, TerminationReason};
    use faer::Col;

    #[test]
    fn levenberg_marquardt_on_finite_diff_jacobian_matches_analytic() {
        let initial = Col::<f64>::from_fn(2, |i| [-1.2, 1.0][i]);

        let fd = Executor::new(
            FiniteDiff::new(RosenbrockResiduals::<Col<f64>>::new()),
            LevenbergMarquardt::new(),
            NllsState::new(initial),
        )
        .max_iter(100)
        .run()
        .unwrap();

        assert_eq!(fd.reason, TerminationReason::SolverConverged);
        assert!(fd.cost() < 1e-12, "fd cost = {}", fd.cost());
        assert!(
            (fd.param()[0] - 1.0).abs() < 1e-6,
            "x[0] = {}",
            fd.param()[0]
        );
        assert!(
            (fd.param()[1] - 1.0).abs() < 1e-6,
            "x[1] = {}",
            fd.param()[1]
        );
    }
}
