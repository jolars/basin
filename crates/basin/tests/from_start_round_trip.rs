//! `Executor::from_start` equivalence tests.
//!
//! `Executor::from_start(problem, solver, x0)` must produce exactly the same
//! run as `Executor::new(problem, solver, ExplicitState::new(x0))`. It is
//! sugar that calls the solver's
//! [`InitialState::seed`](basin::InitialState::seed) instead of making the
//! caller name the state type. Each test runs both forms with identical
//! settings and asserts bit-identical final iterate and cost, covering one
//! solver per state family.
//!
//! Solvers whose natural initialization needs more than a point, namely CMA-ES
//! (σ), the population GA, DE, and random search (they sample the box), and the
//! bracketing scalar solvers (Brent, golden-section), deliberately do **not**
//! implement [`InitialState`](basin::InitialState), so `from_start` with one
//! is a compile error. That exclusion is enforced by the type system; e.g.
//! `Executor::from_start(problem, CmaEs::new(...), x0)` does not compile
//! because `CmaEs: InitialState<_>` is unimplemented.

use basin::MoreThuente;
use basin::core::math::DenseMatrix;
use basin::solver::lbfgs::Unbounded;
use basin::{
    BasicSimplexState, BasicState, Bfgs, CostFunction, Executor, Gradient,
    GradientDescent, Hessian, Jacobian, Lbfgs, LbfgsState, LevenbergMarquardt,
    Mads, MadsState, MaxIter, NelderMead, Newuoa, NewuoaState, NllsState,
    QuasiNewtonState, Residual, Steihaug, TrustRegion,
};

/// f(x) = Σ (xᵢ − cᵢ)², minimum 0 at x = c. Smooth, with an exact gradient
/// and a constant `2 I` Hessian, so it drives the first-order, second-order,
/// and derivative-free solvers alike.
struct Quad {
    c: Vec<f64>,
}

impl CostFunction for Quad {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        Ok(x.iter()
            .zip(&self.c)
            .map(|(xi, ci)| (xi - ci).powi(2))
            .sum())
    }
}

impl Gradient for Quad {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(x.iter()
            .zip(&self.c)
            .map(|(xi, ci)| 2.0 * (xi - ci))
            .collect())
    }
}

impl Hessian for Quad {
    type Hessian = DenseMatrix<f64>;
    fn hessian(&self, x: &Vec<f64>) -> Result<DenseMatrix<f64>, Self::Error> {
        let n = x.len();
        Ok(DenseMatrix::from_fn(
            n,
            n,
            |i, j| if i == j { 2.0 } else { 0.0 },
        ))
    }
}

/// Residual form rᵢ(x) = xᵢ − cᵢ with Jacobian = I, for the NLLS family.
struct LinearResiduals {
    c: Vec<f64>,
}

impl Residual for LinearResiduals {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = std::convert::Infallible;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(x.iter().zip(&self.c).map(|(xi, ci)| xi - ci).collect())
    }
}

impl Jacobian for LinearResiduals {
    type Jacobian = DenseMatrix<f64>;
    fn jacobian(&self, x: &Vec<f64>) -> Result<DenseMatrix<f64>, Self::Error> {
        let n = x.len();
        Ok(DenseMatrix::from_fn(
            n,
            n,
            |i, j| if i == j { 1.0 } else { 0.0 },
        ))
    }
}

fn problem() -> Quad {
    Quad {
        c: vec![1.0, 2.0, 3.0],
    }
}

fn x0() -> Vec<f64> {
    vec![0.0, 0.0, 0.0]
}

#[test]
fn gradient_descent_basic_state() {
    let a = Executor::from_start(problem(), GradientDescent::new(0.01), x0())
        .terminate_on(MaxIter(50))
        .run()
        .unwrap();
    let b = Executor::new(
        problem(),
        GradientDescent::new(0.01),
        BasicState::new(x0()),
    )
    .terminate_on(MaxIter(50))
    .run()
    .unwrap();
    assert_eq!(a.param(), b.param());
    assert_eq!(a.cost(), b.cost());
}

#[test]
fn trust_region_basic_state_second_order() {
    let a = Executor::from_start(
        problem(),
        TrustRegion::with_subproblem(Steihaug::new()),
        x0(),
    )
    .terminate_on(MaxIter(50))
    .run()
    .unwrap();
    let b = Executor::new(
        problem(),
        TrustRegion::with_subproblem(Steihaug::new()),
        BasicState::new(x0()),
    )
    .terminate_on(MaxIter(50))
    .run()
    .unwrap();
    assert_eq!(a.param(), b.param());
    assert_eq!(a.cost(), b.cost());
    // Sanity: the second-order solver actually converged on the quadratic.
    assert!(a.cost() < 1e-12);
}

#[test]
fn nelder_mead_simplex_state() {
    let a = Executor::from_start(problem(), NelderMead::new(), x0())
        .terminate_on(MaxIter(50))
        .run()
        .unwrap();
    let b = Executor::new(
        problem(),
        NelderMead::new(),
        BasicSimplexState::new(x0()),
    )
    .terminate_on(MaxIter(50))
    .run()
    .unwrap();
    assert_eq!(a.param(), b.param());
    assert_eq!(a.cost(), b.cost());
}

#[test]
fn lbfgs_unbounded_history_state() {
    let a = Executor::from_start(
        problem(),
        Lbfgs::<Unbounded, MoreThuente>::new(),
        x0(),
    )
    .terminate_on(MaxIter(50))
    .run()
    .unwrap();
    let b = Executor::new(
        problem(),
        Lbfgs::<Unbounded, MoreThuente>::new(),
        LbfgsState::new(x0(), 10),
    )
    .terminate_on(MaxIter(50))
    .run()
    .unwrap();
    assert_eq!(a.param(), b.param());
    assert_eq!(a.cost(), b.cost());
}

#[test]
fn bfgs_quasi_newton_state_vec_backend() {
    // Also covers the `Vec<f64>` BFGS seed impl added with `from_start`
    // (previously `WarmStart` existed only for nalgebra).
    let a = Executor::from_start(problem(), Bfgs::new(), x0())
        .terminate_on(MaxIter(50))
        .run()
        .unwrap();
    let b = Executor::new(
        problem(),
        Bfgs::new(),
        QuasiNewtonState::<Vec<f64>, DenseMatrix<f64>, f64>::new(x0()),
    )
    .terminate_on(MaxIter(50))
    .run()
    .unwrap();
    assert_eq!(a.param(), b.param());
    assert_eq!(a.cost(), b.cost());
}

#[test]
fn levenberg_marquardt_nlls_state() {
    let prob = || LinearResiduals {
        c: vec![1.0, 2.0, 3.0],
    };
    let a = Executor::from_start(prob(), LevenbergMarquardt::new(), x0())
        .terminate_on(MaxIter(50))
        .run()
        .unwrap();
    let b =
        Executor::new(prob(), LevenbergMarquardt::new(), NllsState::new(x0()))
            .terminate_on(MaxIter(50))
            .run()
            .unwrap();
    assert_eq!(a.param(), b.param());
    assert_eq!(a.cost(), b.cost());
}

#[test]
fn newuoa_state() {
    let a = Executor::from_start(problem(), Newuoa::new(), x0())
        .terminate_on(MaxIter(50))
        .run()
        .unwrap();
    let b = Executor::new(problem(), Newuoa::new(), NewuoaState::new(x0()))
        .terminate_on(MaxIter(50))
        .run()
        .unwrap();
    assert_eq!(a.param(), b.param());
    assert_eq!(a.cost(), b.cost());
}

#[test]
fn mads_state() {
    let a = Executor::from_start(problem(), Mads::new(), x0())
        .terminate_on(MaxIter(50))
        .run()
        .unwrap();
    let b = Executor::new(problem(), Mads::new(), MadsState::new(x0()))
        .terminate_on(MaxIter(50))
        .run()
        .unwrap();
    assert_eq!(a.param(), b.param());
    assert_eq!(a.cost(), b.cost());
}

/// BFGS now seeds uniformly across every backend it runs on. The `Vec<f64>`
/// case is covered by [`bfgs_quasi_newton_state_vec_backend`]; these confirm
/// the nalgebra and faer seed impls added alongside it (`WarmStart` was
/// nalgebra-only before). `ndarray` is intentionally absent: BFGS does not
/// run on `Array2` (no `GeneralRankOneUpdate`).
#[cfg(feature = "nalgebra")]
#[test]
fn bfgs_from_start_nalgebra_backend() {
    use nalgebra::DVector;

    struct QuadN;
    impl CostFunction for QuadN {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, Self::Error> {
            Ok((0..x.len())
                .map(|i| (x[i] - (i as f64 + 1.0)).powi(2))
                .sum())
        }
    }
    impl Gradient for QuadN {
        type Gradient = DVector<f64>;
        fn gradient(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, Self::Error> {
            Ok(DVector::from_fn(x.len(), |i, _| {
                2.0 * (x[i] - (i as f64 + 1.0))
            }))
        }
    }

    let x0 = || DVector::from_vec(vec![0.0, 0.0, 0.0]);
    let a = Executor::from_start(QuadN, Bfgs::new(), x0())
        .terminate_on(MaxIter(50))
        .run()
        .unwrap();
    let b = Executor::new(
        QuadN,
        Bfgs::new(),
        basin::NalgebraQuasiNewtonState::new(x0()),
    )
    .terminate_on(MaxIter(50))
    .run()
    .unwrap();
    assert_eq!(a.param(), b.param());
    assert_eq!(a.cost(), b.cost());
    assert!(a.cost() < 1e-12);
}

#[cfg(feature = "faer")]
#[test]
fn bfgs_from_start_faer_backend() {
    use faer::Col;

    struct QuadF;
    impl CostFunction for QuadF {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, Self::Error> {
            Ok((0..x.nrows())
                .map(|i| (x[i] - (i as f64 + 1.0)).powi(2))
                .sum())
        }
    }
    impl Gradient for QuadF {
        type Gradient = Col<f64>;
        fn gradient(&self, x: &Col<f64>) -> Result<Col<f64>, Self::Error> {
            Ok(Col::from_fn(x.nrows(), |i| 2.0 * (x[i] - (i as f64 + 1.0))))
        }
    }

    let x0 = || Col::from_fn(3, |_| 0.0);
    let a = Executor::from_start(QuadF, Bfgs::new(), x0())
        .terminate_on(MaxIter(50))
        .run()
        .unwrap();
    let b = Executor::new(
        QuadF,
        Bfgs::new(),
        basin::FaerQuasiNewtonState::new(x0()),
    )
    .terminate_on(MaxIter(50))
    .run()
    .unwrap();
    assert_eq!(a.param(), b.param());
    assert_eq!(a.cost(), b.cost());
    assert!(a.cost() < 1e-12);
}

/// `from_start` is fully scalar-generic: it round-trips at `F = f32` over the
/// `Vec<f32>` backend exactly like the `f64` cases above.
#[test]
fn from_start_round_trips_at_f32() {
    struct QuadF32 {
        c: Vec<f32>,
    }
    impl CostFunction for QuadF32 {
        type Param = Vec<f32>;
        type Output = f32;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f32>) -> Result<f32, Self::Error> {
            Ok(x.iter()
                .zip(&self.c)
                .map(|(xi, ci)| (xi - ci).powi(2))
                .sum())
        }
    }
    impl Gradient for QuadF32 {
        type Gradient = Vec<f32>;
        fn gradient(&self, x: &Vec<f32>) -> Result<Vec<f32>, Self::Error> {
            Ok(x.iter()
                .zip(&self.c)
                .map(|(xi, ci)| 2.0 * (xi - ci))
                .collect())
        }
    }

    let prob = || QuadF32 {
        c: vec![1.0_f32, 2.0, 3.0],
    };
    let x0 = || vec![0.0_f32, 0.0, 0.0];

    let a = Executor::from_start(prob(), GradientDescent::new(0.01_f32), x0())
        .terminate_on(MaxIter(50))
        .run()
        .unwrap();
    let b = Executor::new(
        prob(),
        GradientDescent::new(0.01_f32),
        BasicState::<Vec<f32>, f32>::new(x0()),
    )
    .terminate_on(MaxIter(50))
    .run()
    .unwrap();
    assert_eq!(a.param(), b.param());
    assert_eq!(a.cost(), b.cost());
}
