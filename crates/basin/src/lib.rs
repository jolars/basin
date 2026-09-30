//! Basin: a numerical optimization library.
//!
//! The framework lives in [`core`]: problem traits the user implements
//! ([`CostFunction`], [`Gradient`], [`BoxConstraints`],
//! [`LinearInequalityConstraints`], [`LinearEqualityConstraints`],
//! [`LinearConstraints`], [`NonlinearInequalityConstraints`],
//! [`NonlinearConstraints`]), state shapes
//! solvers iterate over ([`State`], [`GradientState`], [`SimplexState`]),
//! the [`Solver`] trait, solver-owned convergence, execution controls
//! ([`RunControl`]), and a read-only observer layer
//! ([`Observe`]). Concrete optimization solvers are in [`solver`]; line
//! searches in [`line_search`]; and direct scalar root finders in [`root`].
//!
//! Start at [`Executor`] for optimization runs, [`root`] for scalar equations,
//! or [`core`] for the trait taxonomy and the iteration-loop contract.
//!
//! See `CONTRIBUTING.md` at the repo root for the design tenets that shape
//! these APIs (notably tenet 3 on solver-owned convergence, tenet 4
//! on first-class constraints, and tenet 5 on backend tiering).
//!
//! # Example
//!
//! Implement [`CostFunction`] (and [`Gradient`] when the solver needs
//! derivatives), then hand the problem, a solver, and an initial state to
//! the [`Executor`]:
//!
//! ```
//! use basin::{
//!     FirstOrderState, CostFunction, Executor, Gradient, GradientDescent,
//! };
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
//!     fn gradient(
//!         &self,
//!         x: &Vec<f64>,
//!     ) -> Result<Vec<f64>, std::convert::Infallible> {
//!         Ok(x.iter().map(|xi| 2.0 * xi).collect())
//!     }
//! }
//!
//! let result = Executor::new(
//!     Sphere,
//!     (GradientDescent::new(0.1)).with_absolute_gradient_tolerance(1e-8),
//!     FirstOrderState::new(vec![1.0, 1.0]),
//! )
//! .max_iter(1_000)
//! .run()
//! .unwrap();
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
//! use basin::{Executor, NelderMead};
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
//!     .max_iter(500)
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
//! | `GradientDescent` | `FirstOrderState` | ✓ |
//! | `Sgd` | `PointState` | ✓ |
//! | `NonlinearCg` | `FirstOrderState` | ✓ |
//! | `ProjectedGradientDescent` | `FirstOrderState` | ✓ |
//! | `Bfgs` | `FirstOrderState` | ✓ (`Vec`/nalgebra/ndarray/faer) |
//! | `Lbfgs`, `Lbfgsb` | `FirstOrderState` | ✓ |
//! | `TrustRegion` | `FirstOrderState` | ✓ |
//! | `GaussNewton`, `LevenbergMarquardt`, `LevenbergMarquardtQr`, `Trf`, `TrustRegionReflective` | `PointState` | ✓ |
//! | `NelderMead` | `SimplexProgress` | ✓ |
//! | `Gbnm` | `PointState` | ✓ |
//! | `Newuoa`, `Bobyqa`, `Lincoa` | `PointState` | ✓ |
//! | `Cobyla` | `SelectedState` | ✓ |
//! | `Mads` | `PointState` or `SelectedState` | ✓ |
//! | `BasinHopping` | `PointState` | ✓ |
//! | `SolisWets` | `PointState` | ✓ |
//! | `SimulatedAnnealing` | `ProposalState` | ✓ |
//! | `BarrierMethod` | `PointState` | ✓ |
//! | `Slsqp` | `SelectedFirstOrderState` | ✓ |
//! | `AugmentedLagrangianMethod` | `SelectedState` | ✓ |
//! | `CmaEs`, `BoundedCmaEs` | `PopulationProgress` | ✓ (scale is configured on the solver) |
//! | `CmaInject`, `BoundedCmaInject` | `PopulationProgress` | ✗ (explicit mean seed) |
//! | `MaLsChCma`, `MaLsChSw` | `PopulationProgress` | ✗ (samples the box) |
//! | `GlobalBestPso` | `PopulationProgress` | ✗ (samples a swarm from the box) |
//! | `RandomSearch`, `Ssga`, `De`, `DeInject` | `PopulationProgress` | ✗ (samples the box or accepts explicit members) |
//! | `Direct` | `PointState` | ✗ (starts at the box midpoint) |
//! | `BrentDerivative` | `FirstOrderState<F, F>` | ✗ (bracket, not a point) |
//! | `Brent`, `GoldenSection` | `PointState<F, F>` | ✗ (bracket, not a point) |
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
//!   because convergence or an execution limit was reached, an attached
//!   [`CancellationToken`] was cancelled, or the [`Solver`] reported a
//!   mid-iteration stop. [`Executor::run`] returns
//!   `Ok(`[`OptimizationResult`]`)` carrying that reason. This is **not** an
//!   error.
//! - **Hard abort**: return `Err(_)` from a problem-trait method to terminate
//!   the *entire* solve. The error is your own type and bubbles out of
//!   [`Executor::run`] untouched, typed as `Result<_, P::Error>`. Use it when
//!   the failure is not about a particular `x`: a downstream service vanished,
//!   an expensive evaluation detected a fine-grained cancellation request, or
//!   an early-stop condition in your own problem state fired.
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
//! The direct scalar [`root`] APIs follow the same typed-error principle without
//! using optimization state: callback failures are wrapped in
//! [`BrentRootError::Evaluation`] or [`RootError::Evaluation`], structural
//! bracket failures have distinct variants, and an iteration-limit exit is a
//! clean [`RootResult`].
//!
//! The [`problem`](crate::core::problem) module docs carry the per-trait detail.
//!
//! # Features
//!
//! Basin enables no features by default. Enable `problems` to use the standard
//! test-function corpus through `basin::problems`, for example with
//! `cargo add basin --features problems`.
//!
//! # Backends
//!
//! Parameters and linear algebra are generic over the backend. `Vec<f64>` needs
//! no features. Each external backend has exact version features and a moving
//! alias:
//!
//! | Backend  | Exact features                            | Moving alias      |
//! | -------- | ----------------------------------------- | ----------------- |
//! | nalgebra | `nalgebra_v0_32` through `nalgebra_v0_35` | `nalgebra_latest` |
//! | ndarray  | `ndarray_v0_15` through `ndarray_v0_17`   | `ndarray_latest`  |
//! | faer     | `faer_v0_22` through `faer_v0_24`         | `faer_latest`     |
//!
//! The original features remain frozen for Basin 1.x compatibility:
//! `nalgebra` selects 0.34, `ndarray` selects 0.17, and `faer` selects 0.24.
//! If dependency feature unification enables several releases of one backend,
//! Basin implements every enabled release independently.
//!
//! BFGS uses [`FirstOrderState`] with a solver-owned inverse Hessian. Its
//! [`DenseBackend`] association selects the matrix belonging to the parameter's
//! backend version, so feature unification cannot change the concrete type.
//!
//! Each nalgebra release includes its matching `nalgebra-sparse` release:
//! 0.32/0.9, 0.33/0.10, 0.34/0.11, and 0.35/0.12. Versioned acceleration uses
//! the `nalgebra_v0_XX-lapack` and `ndarray_v0_XX-blas` features. The moving
//! aliases are `nalgebra_latest-lapack` and `ndarray_latest-blas`; the original
//! `nalgebra-lapack` and `ndarray-blas` features remain frozen at nalgebra 0.34
//! and ndarray 0.17.
//!
//! BLAS/LAPACK acceleration is off by default, is not wasm-compatible, and
//! expects the application to supply BLAS/LAPACK symbols at link time. The
//! default build is wasm-friendly and single-threaded; parallelism is behind
//! the opt-in `parallel` feature.
//!
//! Basin's package MSRV is Rust 1.87. `nalgebra_v0_35` and
//! `nalgebra_latest` require Rust 1.89 because nalgebra 0.35 does.
//!
//! # Citation
//!
//! If you use Basin in your research, please cite the paper:
//!
//! > Larsson, J. (2026). *Basin: Efficient and Extensible Numerical
//! > Optimization in Rust* (arXiv:2608.11279). arXiv.
//! > <https://doi.org/10.48550/arXiv.2608.11279>
//!
//! ```bibtex
//! @misc{larsson2026basin,
//!   title         = {Basin: Efficient and Extensible Numerical Optimization in {{Rust}}},
//!   shorttitle    = {Basin},
//!   author        = {Larsson, Johan},
//!   year          = {2026},
//!   month         = aug,
//!   number        = {arXiv:2608.11279},
//!   eprint        = {2608.11279},
//!   primaryclass  = {cs.LG},
//!   publisher     = {arXiv},
//!   doi           = {10.48550/arXiv.2608.11279},
//!   archiveprefix = {arXiv}
//! }
//! ```
//!
//! `CITATION.cff` at the repo root carries the same reference in
//! machine-readable form.
#![cfg_attr(docsrs, feature(doc_cfg), doc(auto_cfg))]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

mod backend_macros;

// Legacy state aliases and single-version examples retain their selected
// backend. Implementations bind every enabled version independently.
#[cfg(feature = "nalgebra_v0_35")]
extern crate nalgebra;
#[cfg(all(
    not(any(
        feature = "nalgebra_v0_35",
        feature = "nalgebra_v0_34",
        feature = "nalgebra_v0_33"
    )),
    feature = "nalgebra_v0_32"
))]
extern crate nalgebra_0_32 as nalgebra;
#[cfg(all(
    not(any(feature = "nalgebra_v0_35", feature = "nalgebra_v0_34")),
    feature = "nalgebra_v0_33"
))]
extern crate nalgebra_0_33 as nalgebra;
#[cfg(all(not(feature = "nalgebra_v0_35"), feature = "nalgebra_v0_34"))]
extern crate nalgebra_0_34 as nalgebra;

#[cfg(feature = "ndarray_v0_17")]
extern crate ndarray;
#[cfg(all(
    not(any(feature = "ndarray_v0_17", feature = "ndarray_v0_16")),
    feature = "ndarray_v0_15"
))]
extern crate ndarray_0_15 as ndarray;
#[cfg(all(not(feature = "ndarray_v0_17"), feature = "ndarray_v0_16"))]
extern crate ndarray_0_16 as ndarray;

#[cfg(feature = "faer_v0_24")]
extern crate faer;
#[cfg(all(
    not(any(feature = "faer_v0_24", feature = "faer_v0_23")),
    feature = "faer_v0_22"
))]
extern crate faer_0_22 as faer;
#[cfg(all(not(feature = "faer_v0_24"), feature = "faer_v0_23"))]
extern crate faer_0_23 as faer;

#[cfg(all(
    feature = "nalgebra_all",
    not(any(
        feature = "nalgebra_v0_32",
        feature = "nalgebra_v0_33",
        feature = "nalgebra_v0_34",
        feature = "nalgebra_v0_35"
    ))
))]
compile_error!("`nalgebra_all` is internal; enable a `nalgebra_v*` feature");
#[cfg(all(
    feature = "ndarray_all",
    not(any(
        feature = "ndarray_v0_15",
        feature = "ndarray_v0_16",
        feature = "ndarray_v0_17"
    ))
))]
compile_error!("`ndarray_all` is internal; enable an `ndarray_v*` feature");
#[cfg(all(
    feature = "faer_all",
    not(any(
        feature = "faer_v0_22",
        feature = "faer_v0_23",
        feature = "faer_v0_24"
    ))
))]
compile_error!("`faer_all` is internal; enable a `faer_v*` feature");

/// Automatic scalar root and minimum bracketing.
pub mod bracket;
pub mod core;
pub mod line_search;
/// Catalog of test problems used by the example tests and benchmarks.
///
/// Available with the opt-in `problems` feature.
#[cfg(feature = "problems")]
pub mod problems;
/// Scalar root-finding algorithms with direct solve APIs.
pub mod root;
/// Concrete solver implementations.
pub mod solver;

pub use crate::bracket::{
    BracketError, BracketTerminationReason, MinimumBracketResult,
    MinimumBracketer, RootBracketResult, RootBracketer,
};
pub use crate::core::augmented_lagrangian::AugmentedLagrangian;
pub use crate::core::barrier::LogBarrier;
pub use crate::core::checkpoint::{CheckpointSink, ExactCheckpoint};
#[cfg(all(feature = "serde", not(target_arch = "wasm32")))]
pub use crate::core::checkpoint::{
    CheckpointStatus, CheckpointWriteError, ExactCheckpointWriter,
    read_exact_checkpoint,
};
pub use crate::core::constraint::{
    BoxConstraints, ConstraintJacobian, FoldedConstraints, LinearConstraints,
    LinearEqualityConstraints, LinearInequalityConstraints,
    NonlinearConstraints, NonlinearInequalityConstraints,
};
pub use crate::core::convergence::{
    ConfiguredSolver, NativeConvergenceDiagnostics, NativeConvergenceTest,
};
pub use crate::core::executor::{
    CancellationToken, Executor, OptimizationResult,
    OptimizationResultWithSolver, StepOutcome, Stepper, run_loop_with_control,
};
pub use crate::core::inner::{
    InitialState, InnerExecutor, ResumableInner, WarmStart,
};
pub use crate::core::least_squares::{
    ArctanLoss, CauchyLoss, HuberLoss, LossEvaluation, LossFunction,
    RobustLeastSquares, SoftL1Loss, SquaredLoss,
};
pub use crate::core::math::{
    AddDiagonalVectorInPlace, ClampInPlace, ComponentMulAssign, DenseBackend,
    DenseMatrix, DenseMatrixFromFn, Dot, FactorizePivotedQr,
    GeneralRankOneUpdate, GramMatrix, LinearSolveError, LinearSolveLstsq,
    LinearSolveSpd, MatTransposeVec, MatVec, MatrixFromDiagonal,
    MatrixIdentity, MatrixIndex, MaxDiagonal, NegInPlace, NormInfinity,
    NormSquared, QrFactorization, QrSolveError, RegularizedQrSolve,
    SampleStandardNormal, SampleUniformBox, Scalar, ScaleInPlace,
    ScaleRowsInPlace, ScaledAdd, SymmetricEigen, SymmetricEigenError,
    VectorIndex, VectorLen,
};
pub use crate::core::numdiff::{
    BoundedFiniteDiff, DerivativeCheckError, DerivativeCheckReport,
    DerivativeChecker, DerivativeComparison, DerivativeSource, FiniteDiff,
    Method, central_difference_gradient, central_difference_hessian,
    central_difference_hessian_product, central_difference_jacobian,
    forward_difference_gradient, forward_difference_hessian,
    forward_difference_hessian_product, forward_difference_jacobian,
};
#[cfg(all(feature = "serde", not(target_arch = "wasm32")))]
pub use crate::core::observer::{CheckpointWriter, read_checkpoint};
pub use crate::core::observer::{
    History, ObservationEvent, Observe, ObserveSolver, ObserverMode, Report,
};
pub use crate::core::problem::{
    CostFunction, EvalCounts, EvaluationKind, Gradient, Hessian,
    HessianProduct, Jacobian, MiniBatchGradient, Problem, Residual,
};
pub use crate::core::run_control::RunControl;
pub use crate::core::solver::Solver;
pub use crate::core::state::{
    AcceptanceState, CountsMirror, EvaluatedGradientState, EvaluatedState,
    ExactResumeState, FirstOrderState, GradientDimensionMismatch,
    GradientState, IncumbentRef, IncumbentState, IntoInitialSimplex, MeshState,
    ObjectiveIncumbentState, PointState, PopulationProgress,
    PopulationShapeError, PopulationState, ProposalState, RawEvaluationState,
    RhoState, SelectedFirstOrderState, SelectedState, SimplexProgress,
    SimplexShapeError, SimplexState, State,
};
pub use crate::core::termination::TerminationReason;
pub use crate::line_search::{
    Backtracking, Constant, HagerZhang, LineSearch, LineSearchBounds,
    LineSearchEvaluation, LineSearchOutcome, LineSearchResult, MoreThuente,
    Wolfe,
};
pub use crate::root::{
    BrentRoot, BrentRootError, HalleyRoot, NewtonRoot, RootError, RootResult,
    RootTerminationReason, SecantRoot, Toms748Root,
};
pub use crate::solver::Bfgs;
pub use crate::solver::lbfgs::{Lbfgs, Lbfgsb};
pub use crate::solver::trust_region::{
    CauchyPoint, Dogleg, ExactHessian, MatrixFree, MoreSorensen, Steihaug,
    TrustRegion,
};
pub use crate::solver::{
    AcceptanceTest, AugmentedLagrangianMethod, BarrierMethod, BasinHopping,
    Bobyqa, BoundedCmaEs, BoundedCmaInject, Brent, BrentDerivative, CgUpdate,
    ClosureInner, CmaEs, CmaInject, Cobyla, De, DeCrossover, DeInject,
    DeMutation, Direct, GaussNewton, Gbnm, GlobalBestPso, GoldenSection,
    GradientDescent, LevenbergMarquardt, LevenbergMarquardtQr, Lincoa,
    LmDamping, MaLsCh, MaLsChCma, MaLsChSw, Mads, MemeticInner, Metropolis,
    Neighbor, NelderMead, Newuoa, NonlinearCg, ProjectedGradientDescent,
    PsoBoundaryHandling, PsoVelocityLimit, RandomDisplacement, RandomSearch,
    Reannealing, Sgd, SimulatedAnnealing, Slsqp, SlsqpFailure, SolisWets, Ssga,
    StepTaker, TemperatureSchedule, Trf, TrustRegionReflective,
};
