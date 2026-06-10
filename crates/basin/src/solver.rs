/// Augmented-Lagrangian method for linear equality constraints `A x = b`
/// (penalty + multiplier updates over any unconstrained inner solver;
/// tolerates infeasible starts).
pub mod augmented_lagrangian_method;
/// Log-barrier method for linear inequality constraints `A x ≤ b`
/// (`constrOptim`-style layer over any unconstrained inner solver).
pub mod barrier_method;
/// Box-constrained CMA-ES with adaptive quadratic boundary penalty
/// (Hansen `BoundPenalty`, the default in pycma).
pub mod bounded_cma_es;
/// Memetic [`BoundedCmaEs`] with
/// Hansen-2011 injection — sibling of [`cma_inject`] over the bounded
/// outer. Inners: Nelder-Mead, Levenberg-Marquardt, L-BFGS-B.
pub mod bounded_cma_inject;
/// Brent's method (1D root / minimum bracketing).
pub mod brent;
/// Brent's method using first derivatives ("dbrent", 1D minimization).
pub mod brent_derivative;
/// Hansen 2016 (µ/µ_W, λ)-CMA-ES with negative weights.
pub mod cma_es;
/// Memetic CMA-ES with Hansen-2011 injection. Inners: Nelder-Mead,
/// Levenberg-Marquardt. For L-BFGS-B inner with consistent bound
/// handling, see [`bounded_cma_inject`].
pub mod cma_inject;
/// Differential Evolution (DE/rand/1/bin) — Storn-Price 1997 global
/// optimizer on a feasible box, fully backend-generic.
pub mod de;
/// Memetic [`De`] with per-generation top-k local refinement —
/// DE-flavored sibling of [`cma_inject`]. Inners: Nelder-Mead,
/// Levenberg-Marquardt, L-BFGS-B.
pub mod de_inject;
/// Pure Gauss-Newton solver for nonlinear least squares.
pub mod gauss_newton;
/// Golden-section search (1D minimization on a bracketed interval).
pub mod golden_section;
/// Steepest-descent solver with a pluggable line search and optional
/// heavy-ball momentum.
pub mod gradient_descent;
/// L-BFGS family — unconstrained `Lbfgs<Unbounded>` (two-loop
/// recursion) and box-constrained `Lbfgs<Bounded>` (faithful port of
/// Nocedal's L-BFGS-B v3.0). `Lbfgsb` is a type alias for
/// `Lbfgs<Bounded>`.
pub mod lbfgs;
/// Levenberg-Marquardt solver for nonlinear least squares with
/// Nielsen 1999 damping update.
pub mod levenberg_marquardt;
/// MA-LSCh-CMA — memetic algorithm with LS chains (inner: CMA-ES).
pub mod ma_ls_ch_cma;
/// Nelder-Mead derivative-free simplex solver.
pub mod nelder_mead;
/// Projected gradient descent for box-constrained problems.
pub mod projected_gradient_descent;
/// Elitist (1+λ) random search over a feasible box.
pub mod random_search;
/// Mini-batch stochastic gradient descent with constant learning rate
/// and optional Polyak heavy-ball momentum.
pub mod sgd;
/// Steady-state real-coded GA with BLX-α + NAM + BGA + replace-worst.
pub mod ssga;
/// Levenberg-Marquardt with box bounds (TRF — trust-region-reflective).
pub mod trf;

/// BFGS quasi-Newton solver (dense inverse-Hessian; `Vec<f64>`, nalgebra,
/// faer).
pub mod bfgs;

pub use augmented_lagrangian_method::AugmentedLagrangianMethod;
pub use barrier_method::BarrierMethod;
pub use bfgs::Bfgs;
pub use bounded_cma_es::BoundedCmaEs;
pub use bounded_cma_inject::BoundedCmaInject;
pub use brent::Brent;
pub use brent_derivative::BrentDerivative;
pub use cma_es::CmaEs;
pub use cma_inject::{ClosureInner, CmaInject, MemeticInner};
pub use de::De;
pub use de_inject::DeInject;
pub use gauss_newton::GaussNewton;
pub use golden_section::GoldenSection;
pub use gradient_descent::GradientDescent;
pub use levenberg_marquardt::LevenbergMarquardt;
pub use ma_ls_ch_cma::{MaLsChCma, MaLsChState};
pub use nelder_mead::{NelderMead, Projected, Unbounded};
pub use projected_gradient_descent::ProjectedGradientDescent;
pub use random_search::RandomSearch;
pub use sgd::Sgd;
pub use ssga::Ssga;
pub use trf::Trf;
