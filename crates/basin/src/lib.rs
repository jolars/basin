//! Basin: a numerical optimization library.
//!
//! The framework lives in [`core`]: problem traits the user implements
//! ([`CostFunction`], [`Gradient`], [`BoxConstraints`],
//! [`LinearInequalityConstraints`], [`LinearEqualityConstraints`],
//! [`LinearConstraints`], [`NonlinearInequalityConstraints`]), state shapes
//! solvers iterate over ([`State`], [`GradientState`], [`SimplexState`]),
//! the [`Solver`] trait, a pluggable termination layer
//! ([`TerminationCriterion`]), and a read-only observer layer
//! ([`Observe`]). Concrete solvers are in [`solver`]; line searches in
//! [`line_search`].
//!
//! Start at [`Executor`] for the user-facing driver, or [`core`] for the
//! trait taxonomy and the iteration-loop contract.
//!
//! See `CONTRIBUTING.md` at the repo root for the design tenets that shape
//! these APIs (notably tenet 3 on framework-level termination, tenet 4
//! on first-class constraints, and tenet 5 on backend tiering).
//!
//! # Example
//!
//! Implement [`CostFunction`] (and [`Gradient`] when the solver needs
//! derivatives), then hand the problem, a solver, and an initial state to
//! the [`Executor`]:
//!
//! ```
//! use basin::{BasicState, CostFunction, Executor, Gradient, GradientDescent, GradientTolerance};
//!
//! struct Sphere;
//! impl CostFunction for Sphere {
//!     type Param = Vec<f64>;
//!     type Output = f64;
//!     type Error = std::convert::Infallible;
//!     fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
//!         Ok(x.iter().map(|xi| xi * xi).sum())
//!     }
//! }
//! impl Gradient for Sphere {
//!     type Gradient = Vec<f64>;
//!     fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
//!         Ok(x.iter().map(|xi| 2.0 * xi).collect())
//!     }
//! }
//!
//! let result = Executor::new(Sphere, GradientDescent::new(0.1), BasicState::new(vec![1.0, 1.0]))
//!     .max_iter(1_000)
//!     .terminate_on(GradientTolerance(1e-8))
//!     .run()
//!     .unwrap();
//! assert!(result.cost() < 1e-12);
//! ```
//!
//! # Seeding the initial state
//!
//! [`Executor::new`] takes a fully-built [`State`] so you control the initial
//! iterate (a custom simplex, a warm-started inverse Hessian, an anisotropic
//! CMA-ES covariance). For the common case (start at a point, use the
//! solver's natural defaults) [`Executor::from_start`] takes the bare
//! starting vector instead and builds the state for you via
//! [`InitialState::seed`], so you never name the concrete state type:
//!
//! ```
//! use basin::{Executor, MaxIter, NelderMead};
//! # struct Sphere;
//! # impl basin::CostFunction for Sphere {
//! #     type Param = Vec<f64>;
//! #     type Output = f64;
//! #     type Error = std::convert::Infallible;
//! #     fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
//! #         Ok(x.iter().map(|xi| xi * xi).sum())
//! #     }
//! # }
//! let result = Executor::from_start(Sphere, NelderMead::new(), vec![1.0, 1.0])
//!     .terminate_on(MaxIter(500))
//!     .run()
//!     .unwrap();
//! ```
//!
//! Most solvers support `from_start`; the few whose natural initialization
//! needs more than a point do not implement [`InitialState`], so calling
//! `from_start` with one is a compile error (use [`Executor::new`] with an
//! explicit state). The map:
//!
//! | Solver | State | `from_start` |
//! | ------ | ----- | ------------ |
//! | `GradientDescent`, `Sgd` | `BasicState` | ✓ |
//! | `ProjectedGradientDescent` | `BasicState` | ✓ (`f64` only) |
//! | `Bfgs` | `QuasiNewtonState` | ✓ (`Vec`/nalgebra/ndarray/faer) |
//! | `Lbfgs`, `Lbfgsb` | `LbfgsState` | ✓ |
//! | `TrustRegion` | `BasicState` | ✓ |
//! | `GaussNewton`, `LevenbergMarquardt`, `Trf` | `NllsState` | ✓ |
//! | `NelderMead` | `BasicSimplexState` | ✓ |
//! | `Newuoa`, `Bobyqa`, `Lincoa`, `Cobyla` | `NewuoaState`/… | ✓ |
//! | `Mads` | `MadsState`/`ConstrainedMadsState` | ✓ |
//! | `SolisWets` | `SolisWetsState` | ✓ |
//! | `BarrierMethod`, `AugmentedLagrangianMethod` | `BasicState` | ✓ |
//! | `CmaEs`, `BoundedCmaEs`, `CmaInject`, `BoundedCmaInject`, `MaLsChCma`, `MaLsChSw` | `CmaEsState`/`MaLsChState`/… | ✗ (needs a step-size σ or samples the box) |
//! | `RandomSearch`, `Ssga`, `De`, `DeInject` | `BasicPopulationState` | ✗ (sample the box, ignore a point) |
//! | `Brent`, `BrentDerivative`, `GoldenSection` | `ScalarState` | ✗ (bracket, not a point) |
//!
//! # Error model
//!
//! Basin distinguishes *three* outcomes a run can produce. The split is a
//! stable part of the public contract; downstream code can rely on it:
//!
//! - **Soft reject**: return `Ok(f64::INFINITY)` from [`CostFunction::cost`]
//!   to reject a *single point* without stopping the solve. Line searches treat
//!   `+∞` as worse and retreat; population solvers treat it as worst fitness.
//!   This is the channel for "this `x` is outside my domain, but the solve
//!   should continue."
//! - **Clean stop**: the run ends *normally* with a
//!   [`TerminationReason`], either
//!   because a [`TerminationCriterion`] fired or because the [`Solver`] reported
//!   a mid-iteration stop. [`Executor::run`] returns
//!   `Ok(`[`OptimizationResult`]`)` carrying that reason. This is **not** an
//!   error.
//! - **Hard abort**: return `Err(_)` from a problem-trait method to terminate
//!   the *entire* solve. The error is your own type and bubbles out of
//!   [`Executor::run`] untouched, typed as `Result<_, P::Error>`. Use it when
//!   the failure is not about a particular `x`: a downstream service vanished,
//!   the user pressed cancel, an early-stop condition in your own problem state
//!   fired.
//!
//! ## One error type, threaded through
//!
//! The hard-abort error is chosen *once*, on the problem
//! ([`CostFunction::Error`], or
//! [`Residual::Error`] for nonlinear
//! least squares). Every downstream trait mirrors it: [`Solver::Error`] and
//! [`LineSearch::Error`] are set to `P::Error`, so a custom problem error flows
//! through the solver and line search out to the caller with no conversion glue.
//! Problems that cannot fail pick [`std::convert::Infallible`]; its niche
//! optimization keeps `Result<f64, Infallible>` the same layout as a bare `f64`,
//! so the happy path stays zero-cost.
//!
//! The [`problem`](crate::core::problem) module docs carry the per-trait detail.
//!
//! # Backends
//!
//! Parameters and linear algebra are generic over the backend. `Vec<f64>` needs
//! no features; nalgebra, ndarray, and faer are enabled one feature each, each
//! pinning a single major version. Basin pins one major version per backend;
//! each Basin 1.x release supports exactly these versions:
//!
//! | Backend    | Feature      | Version                            |
//! | ---------- | ------------ | ---------------------------------- |
//! | nalgebra   | `nalgebra`   | 0.34 (with `nalgebra-sparse` 0.11) |
//! | ndarray    | `ndarray`    | 0.17                               |
//! | faer       | `faer`       | 0.24                               |
//!
//! `Vec<f64>` is the built-in default backend, so no feature is needed for that.
//!
//! A backend major-version bump is a breaking change and ships only in a Basin
//! major release; within the 1.x series these pins are fixed.
//!
//! Two backends have opt-in, BLAS/LAPACK-backed acceleration. Both are off by
//! default and not wasm-compatible (each links a Fortran/BLAS toolchain), and
//! both expect you to bring your own BLAS/LAPACK source crate:
//!
//! | Feature           | Effect                                                                                        |
//! | ----------------- | --------------------------------------------------------------------------------------------- |
//! | `ndarray-blas`    | Forwards `ndarray/blas` for BLAS-backed ndarray linear algebra.                                |
//! | `nalgebra-lapack` | Swaps the nalgebra backend's Cholesky/symmetric eigendecomposition for LAPACK-backed ones.   |
//!
//! The default build is wasm-friendly: no BLAS/LAPACK and no threads.
//! Parallelism is behind the opt-in `parallel` feature.
#![cfg_attr(docsrs, feature(doc_cfg), doc(auto_cfg))]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod core;
pub mod line_search;
/// Catalog of test problems used by the example tests and benchmarks.
#[cfg(feature = "problems")]
pub mod problems;
/// Concrete solver implementations.
pub mod solver;

pub use crate::core::augmented_lagrangian::AugmentedLagrangian;
pub use crate::core::barrier::LogBarrier;
pub use crate::core::constraint::{
    BoxConstraints, LinearConstraints, LinearEqualityConstraints, LinearInequalityConstraints,
    NonlinearInequalityConstraints,
};
pub use crate::core::executor::{Executor, OptimizationResult, StepOutcome, Stepper, run_loop};
pub use crate::core::inner::{InitialState, InnerExecutor, ResumableInner, WarmStart};
pub use crate::core::math::{
    ClampInPlace, ComponentMulAssign, DenseMatrix, DenseMatrixFromFn, Dot, GramMatrix,
    LinearSolveError, LinearSolveLstsq, LinearSolveSpd, MatTransposeVec, MatVec,
    MatrixFromDiagonal, MatrixIdentity, NegInPlace, NormInfinity, NormSquared,
    SampleStandardNormal, SampleUniformBox, Scalar, ScaleInPlace, ScaledAdd, SymmetricEigen,
    SymmetricEigenError, VectorIndex, VectorLen,
};
pub use crate::core::numdiff::{
    FiniteDiff, Method, central_difference_gradient, central_difference_hessian,
    central_difference_hessian_product, central_difference_jacobian, forward_difference_gradient,
    forward_difference_hessian, forward_difference_hessian_product, forward_difference_jacobian,
};
#[cfg(all(feature = "serde", not(target_arch = "wasm32")))]
pub use crate::core::observer::{CheckpointWriter, read_checkpoint};
pub use crate::core::observer::{History, Observe, ObserverMode, Report};
pub use crate::core::problem::{
    CostFunction, EvalCounts, Gradient, Hessian, HessianProduct, Jacobian, MiniBatchGradient,
    Problem, Residual,
};
pub use crate::core::solver::Solver;
#[cfg(feature = "faer")]
pub use crate::core::state::FaerQuasiNewtonState;
#[cfg(feature = "nalgebra")]
pub use crate::core::state::NalgebraQuasiNewtonState;
#[cfg(feature = "ndarray")]
pub use crate::core::state::NdarrayQuasiNewtonState;
pub use crate::core::state::{
    BasicPopulationState, BasicSimplexState, BasicState, BobyqaState, CmaEsState, CobylaState,
    ConstrainedMadsState, CountsMirror, GradientState, IntoInitialSimplex, LbfgsState, LincoaState,
    MadsState, MeshState, NewuoaState, NllsState, PopulationState, RhoState, ScalarGradientState,
    ScalarState, SimplexState, SolisWetsState, State,
};
pub use crate::core::state::{DenseQuasiNewtonState, QuasiNewtonState};
pub use crate::core::termination::{
    CmaEsTolerance, CostTolerance, GradientTolerance, MaxCostEvals, MaxGradientEvals, MaxIter,
    MaxTime, MeshTolerance, NoImprovement, ParamTolerance, ProjectedGradientTolerance,
    RelativeCostTolerance, RelativeGradientTolerance, RelativeParamTolerance, RhoTolerance,
    SimplexTolerance, TargetCost, TerminationCriterion, TerminationReason,
};
pub use crate::line_search::{Backtracking, Constant, LineSearch, MoreThuente, Wolfe};
pub use crate::solver::Bfgs;
pub use crate::solver::lbfgs::{Lbfgs, Lbfgsb};
pub use crate::solver::trust_region::{
    CauchyPoint, Dogleg, ExactHessian, MatrixFree, Steihaug, TrustRegion,
};
pub use crate::solver::{
    AcceptanceTest, AugmentedLagrangianMethod, BarrierMethod, BasinHopping, Bobyqa, BoundedCmaEs,
    BoundedCmaInject, Brent, BrentDerivative, ClosureInner, CmaEs, CmaInject, Cobyla, De, DeInject,
    GaussNewton, GoldenSection, GradientDescent, LevenbergMarquardt, Lincoa, MaLsCh, MaLsChCma,
    MaLsChGenericState, MaLsChState, MaLsChSw, MaLsChSwState, Mads, MemeticInner, Metropolis,
    NelderMead, Newuoa, ProjectedGradientDescent, RandomDisplacement, RandomSearch, Sgd, SolisWets,
    Ssga, StepTaker, Trf,
};
