// Index-based loops mirror Powell's exposition (NEWUOA, Powell 2006; the
// H-factorization derivation, Powell 2004a) and the dense index arithmetic of
// the H-algebra; the lint is blanket-allowed for this module.
#![allow(clippy::needless_range_loop)]
// The model-core read surface and several H-update entry points have only
// `#[cfg(test)]` callers in the non-`bobyqa` build (the in-module tests, and —
// for the §8 `Q_int` path — NEWUOA's driver alone); BOBYQA adds more callers.
// Blanket-allow so test-only `pub(crate)` surface does not trip dead-code
// analysis.
#![allow(dead_code)]

//! Shared core of Powell's model-based derivative-free family (NEWUOA →
//! BOBYQA → LINCOA).
//!
//! All three solvers maintain the same least-Frobenius-norm quadratic surrogate
//! `Q` through the same factored inverse-KKT matrix `H = W⁻¹`. That spine lives
//! here so each solver only supplies the pieces that genuinely differ:
//!
//! - [`model`] — [`QuadraticModel`]: storage and read surface (model evaluation,
//!   the implicit-Hessian matvec, Lagrange-function coefficients).
//! - [`update`] — the §4 closed-form least-Frobenius-norm rank-2 `H` update
//!   (`prepare_update` / `update_params` / `commit_update`), plus the §8
//!   alternative-model (`Q_int`) operations the *caller* may choose to adopt.
//! - [`origin`] — the §7 origin shift that re-centres `x0` on `x_opt`.
//! - [`subproblem`] — the [`TrustRegionSubproblem`] seam: NEWUOA supplies TRSAPP
//!   (unconstrained), BOBYQA supplies TRSBOX (box), LINCOA a projected solver.
//!
//! Initialization (the choice and placement of the interpolation set) is *not*
//! shared: NEWUOA seeds a coordinate cross, BOBYQA clips to the box, so each
//! solver owns its own `init`.

pub(crate) mod model;
pub(crate) mod origin;
pub(crate) mod subproblem;
pub(crate) mod update;

#[cfg(test)]
pub(crate) mod kkt;

pub(crate) use model::QuadraticModel;
pub(crate) use subproblem::{TrustRegionStep, TrustRegionSubproblem};
