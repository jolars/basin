//! End-to-end smoke test that solvers, states, and termination criteria
//! compose at `F = f32` over the `Vec<F>` backend. Demonstrates the
//! provisional-choice trigger from `CONTRIBUTING.md` is now satisfiable: the
//! whole pipeline runs at a non-`f64` scalar without further refactor.

use basin::core::executor::Executor;
use basin::core::math::DenseMatrix;
use basin::core::problem::{CostFunction, Gradient, Hessian, HessianProduct};
use basin::core::state::{BasicState, LbfgsState, State};
use basin::core::termination::{
    CostTolerance, GradientTolerance, MaxIter, RelativeCostTolerance,
    TargetCost,
};
use basin::line_search::{Backtracking, HagerZhang, MoreThuente};
use basin::solver::lbfgs::{Lbfgs, Unbounded};
use basin::{
    BoxConstraints, Cobyla, CobylaState, Gbnm, GbnmState, GradientDescent,
    MatrixFree, MoreSorensen, NonlinearInequalityConstraints, Steihaug,
    TerminationReason, TrustRegion,
};

/// `f(x) = ‖x − c‖²` with `c = (1, 2, 3)`. Minimum at `c`, cost 0.
struct ShiftedQuadF32 {
    c: Vec<f32>,
}

impl CostFunction for ShiftedQuadF32 {
    type Param = Vec<f32>;
    type Output = f32;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f32>) -> Result<f32, Self::Error> {
        Ok(x.iter().zip(&self.c).map(|(a, b)| (a - b).powi(2)).sum())
    }
}

impl Gradient for ShiftedQuadF32 {
    type Gradient = Vec<f32>;
    fn gradient(&self, x: &Vec<f32>) -> Result<Vec<f32>, Self::Error> {
        Ok(x.iter().zip(&self.c).map(|(a, b)| 2.0 * (a - b)).collect())
    }
}

struct BoundedSphereF32 {
    lower: Vec<f32>,
    upper: Vec<f32>,
}

impl CostFunction for BoundedSphereF32 {
    type Param = Vec<f32>;
    type Output = f32;
    type Error = std::convert::Infallible;

    fn cost(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
        Ok(x.iter().map(|value| value * value).sum())
    }
}

impl BoxConstraints for BoundedSphereF32 {
    fn lower(&self) -> &Self::Param {
        &self.lower
    }

    fn upper(&self) -> &Self::Param {
        &self.upper
    }
}

impl NonlinearInequalityConstraints for BoundedSphereF32 {
    fn num_constraints(&self) -> usize {
        2 * self.lower.len()
    }

    fn constraints(&self, x: &Vec<f32>) -> Result<Vec<f32>, Self::Error> {
        Ok(x.iter()
            .zip(&self.lower)
            .zip(&self.upper)
            .flat_map(|((&v, &l), &u)| [l - v, v - u])
            .collect())
    }
}

#[test]
fn cobyla_f32_round_trips_small_and_cached_inverse_paths() {
    for n in [2, 3, 5] {
        let problem = BoundedSphereF32 {
            lower: vec![1.0; n],
            upper: vec![5.0; n],
        };
        let result = Executor::new(
            problem,
            Cobyla::<f32>::new().with_rho_beg(0.5).with_rho_end(1e-4),
            CobylaState::<Vec<f32>, f32>::new(vec![3.0; n]),
        )
        .terminate_on(MaxIter(1000))
        .run()
        .unwrap();
        assert_eq!(result.reason, TerminationReason::SolverConverged);
        let x = result.best_param();
        assert!(x.iter().all(|v| v.is_finite() && (*v - 1.0).abs() < 5e-4));
        assert!((result.best_cost() - n as f32).abs() < 5e-3);
        assert_eq!(result.best_cost(), x.iter().map(|v| v * v).sum::<f32>());
    }
}

#[test]
fn gbnm_f32_round_trips_solver_state_and_bounds() {
    let problem = BoundedSphereF32 {
        lower: vec![-5.0; 2],
        upper: vec![5.0; 2],
    };
    let solver = Gbnm::<f32>::new(42);
    let state = GbnmState::<Vec<f32>, f32>::new(vec![3.0, 3.0]);

    let result = Executor::new(problem, solver, state)
        .terminate_on(MaxIter(300))
        .run()
        .unwrap();

    assert!(result.state.best_cost() < 1e-5);
    assert!(
        result
            .state
            .vertices()
            .iter()
            .flatten()
            .all(|value| (-5.0..=5.0).contains(value))
    );
}

#[test]
fn gradient_descent_f32_with_f32_termination_converges() {
    let problem = ShiftedQuadF32 {
        c: vec![1.0_f32, 2.0, 3.0],
    };
    let state = BasicState::<Vec<f32>, f32>::new(vec![0.0_f32; 3]);
    let solver: GradientDescent<Backtracking<f32>, Vec<f32>, f32> =
        GradientDescent::with_line_search(Backtracking::new());

    let result = Executor::new(problem, solver, state)
        .terminate_on(MaxIter(500))
        .terminate_on(GradientTolerance::<f32>(1e-3))
        .run()
        .unwrap();

    let final_x = result.state.param();
    assert!((final_x[0] - 1.0).abs() < 1e-2);
    assert!((final_x[1] - 2.0).abs() < 1e-2);
    assert!((final_x[2] - 3.0).abs() < 1e-2);
}

#[test]
fn unbounded_lbfgs_f32_round_trips_state_solver_termination() {
    let problem = ShiftedQuadF32 {
        c: vec![1.0_f32, 2.0, 3.0],
    };
    let state = LbfgsState::<Vec<f32>, f32>::new(vec![0.0_f32; 3], 5);
    let solver: Lbfgs<Unbounded, MoreThuente<f32>, f32> =
        Lbfgs::<Unbounded, MoreThuente<f32>, f32>::with_line_search(
            MoreThuente::new(),
        );

    let result = Executor::new(problem, solver, state)
        .terminate_on(MaxIter(100))
        .terminate_on(GradientTolerance::<f32>(1e-3))
        .terminate_on(CostTolerance::<f32>::new(1e-6))
        .terminate_on(RelativeCostTolerance::<f32>::new(1e-6))
        .terminate_on(TargetCost::<f32>(1e-6))
        .run()
        .unwrap();

    let final_x = result.state.param();
    assert!((final_x[0] - 1.0).abs() < 1e-3);
    assert!((final_x[1] - 2.0).abs() < 1e-3);
    assert!((final_x[2] - 3.0).abs() < 1e-3);
}

#[test]
fn hager_zhang_f32_round_trips_line_search_and_lbfgs() {
    let problem = ShiftedQuadF32 {
        c: vec![1.0_f32, 2.0, 3.0],
    };
    let state = LbfgsState::<Vec<f32>, f32>::new(vec![0.0_f32; 3], 5);
    let solver: Lbfgs<Unbounded, HagerZhang<f32>, f32> =
        Lbfgs::<Unbounded, HagerZhang<f32>, f32>::with_line_search(
            HagerZhang::new(),
        );

    let result = Executor::new(problem, solver, state)
        .terminate_on(MaxIter(100))
        .terminate_on(GradientTolerance::<f32>(1e-3))
        .run()
        .unwrap();

    let final_x = result.state.param();
    assert!((final_x[0] - 1.0).abs() < 1e-3);
    assert!((final_x[1] - 2.0).abs() < 1e-3);
    assert!((final_x[2] - 3.0).abs() < 1e-3);
}

impl Hessian for ShiftedQuadF32 {
    type Hessian = DenseMatrix<f32>;
    fn hessian(&self, _x: &Vec<f32>) -> Result<DenseMatrix<f32>, Self::Error> {
        // f(x) = Σ (xᵢ − cᵢ)², so ∇²f = 2 I.
        Ok(DenseMatrix::from_fn(
            3,
            3,
            |i, j| if i == j { 2.0 } else { 0.0 },
        ))
    }
}

impl HessianProduct for ShiftedQuadF32 {
    fn hessian_product(
        &self,
        _x: &Vec<f32>,
        v: &Vec<f32>,
    ) -> Result<Vec<f32>, Self::Error> {
        // ∇²f = 2 I, so ∇²f·v = 2 v.
        Ok(v.iter().map(|vi| 2.0 * vi).collect())
    }
}

#[test]
fn matrix_free_trust_region_f32_round_trips_state_solver_termination() {
    // The matrix-free second-order pipeline (HessianProduct trait, Steihaug
    // CG via products) runs end-to-end at F = f32 over Vec<f32>, with no
    // matrix type anywhere.
    let problem = ShiftedQuadF32 {
        c: vec![1.0_f32, 2.0, 3.0],
    };
    let state = BasicState::<Vec<f32>, f32>::new(vec![0.0_f32; 3]);
    let solver: TrustRegion<Steihaug, f32, MatrixFree> =
        TrustRegion::matrix_free_with(Steihaug::new());

    let result = Executor::new(problem, solver, state)
        .terminate_on(MaxIter(100))
        .terminate_on(GradientTolerance::<f32>(1e-4))
        .run()
        .unwrap();

    let final_x = result.state.param();
    assert!((final_x[0] - 1.0).abs() < 1e-3);
    assert!((final_x[1] - 2.0).abs() < 1e-3);
    assert!((final_x[2] - 3.0).abs() < 1e-3);
}

#[test]
fn solis_wets_f32_round_trips_state_solver_termination() {
    // The derivative-free adaptive-random-search pipeline (SolisWets,
    // SolisWetsState, RhoTolerance via RhoState) runs end-to-end at
    // F = f32 over Vec<f32>.
    use basin::SolisWets;
    use basin::core::termination::RhoTolerance;

    let problem = ShiftedQuadF32 {
        c: vec![1.0_f32, 2.0, 3.0],
    };
    let solver = SolisWets::<f32>::new(42);

    let result = Executor::from_start(problem, solver, vec![0.0_f32; 3])
        .terminate_on(MaxIter(20_000))
        .terminate_on(RhoTolerance::<f32>::new(1e-5))
        .run()
        .unwrap();

    let final_x = result.state.best_param();
    assert!((final_x[0] - 1.0).abs() < 1e-1);
    assert!((final_x[1] - 2.0).abs() < 1e-1);
    assert!((final_x[2] - 3.0).abs() < 1e-1);
}

#[test]
fn trust_region_f32_round_trips_state_solver_termination() {
    // The whole second-order pipeline (Hessian trait, DenseMatrix<f32>
    // matvec, Steihaug CG) runs end-to-end at F = f32 over Vec<f32>.
    let problem = ShiftedQuadF32 {
        c: vec![1.0_f32, 2.0, 3.0],
    };
    let state = BasicState::<Vec<f32>, f32>::new(vec![0.0_f32; 3]);
    let solver: TrustRegion<Steihaug, f32> =
        TrustRegion::with_subproblem(Steihaug::new());

    let result = Executor::new(problem, solver, state)
        .terminate_on(MaxIter(100))
        .terminate_on(GradientTolerance::<f32>(1e-4))
        .run()
        .unwrap();

    let final_x = result.state.param();
    assert!((final_x[0] - 1.0).abs() < 1e-3);
    assert!((final_x[1] - 2.0).abs() < 1e-3);
    assert!((final_x[2] - 3.0).abs() < 1e-3);
}

#[test]
fn more_sorensen_f32_round_trips_state_solver_termination() {
    let problem = ShiftedQuadF32 {
        c: vec![1.0_f32, 2.0, 3.0],
    };
    let state = BasicState::<Vec<f32>, f32>::new(vec![0.0_f32; 3]);
    let solver: TrustRegion<MoreSorensen, f32> =
        TrustRegion::with_subproblem(MoreSorensen::new());

    let result = Executor::new(problem, solver, state)
        .terminate_on(MaxIter(100))
        .terminate_on(GradientTolerance::<f32>(1e-4))
        .run()
        .unwrap();

    let final_x = result.state.param();
    assert!((final_x[0] - 1.0).abs() < 1e-3);
    assert!((final_x[1] - 2.0).abs() < 1e-3);
    assert!((final_x[2] - 3.0).abs() < 1e-3);
}

#[test]
fn levenberg_marquardt_qr_f32_round_trip() {
    struct Fit;
    impl basin::Residual for Fit {
        type Param = Vec<f32>;
        type Output = Vec<f32>;
        type Error = std::convert::Infallible;
        fn residual(&self, x: &Vec<f32>) -> Result<Vec<f32>, Self::Error> {
            Ok(vec![x[0] - 1., 2. * (x[1] - 2.)])
        }
    }
    impl basin::Jacobian for Fit {
        type Jacobian = DenseMatrix<f32>;
        fn jacobian(
            &self,
            _: &Vec<f32>,
        ) -> Result<Self::Jacobian, Self::Error> {
            Ok(DenseMatrix::from_row_slice(2, 2, &[1., 0., 0., 2.]))
        }
    }
    let solver =
        basin::LevenbergMarquardtQr::<_, _, f32>::new().with_tol_grad(1e-5);
    let result = Executor::from_start(Fit, solver, vec![0., 0.])
        .max_iter(50)
        .run()
        .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!((result.param()[0] - 1.).abs() < 1e-5);
    assert!((result.param()[1] - 2.).abs() < 1e-5);
}
