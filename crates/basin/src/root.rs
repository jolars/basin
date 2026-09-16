//! Scalar root-finding algorithms.
//!
//! Root finding is deliberately separate from Basin's optimization
//! [`Solver`](crate::Solver)/[`Executor`](crate::Executor) loop. Optimization
//! states rank points by a scalar cost, whereas a bracketed root finder must
//! retain the signed function value and a sign-changing interval. The direct
//! APIs in this module preserve those semantics without treating `|f(x)|` as
//! an optimization objective.
//!
//! All solvers require a finite, sign-changing interval (or an exact endpoint
//! root). Use [`crate::RootBracketer`] to discover one first. [`BrentRoot`],
//! [`SecantRoot`], and [`Toms748Root`] need only function values. [`NewtonRoot`]
//! uses a first derivative, and [`HalleyRoot`] also uses a second derivative;
//! both accept separate or combined callbacks. Derivative steps retain the
//! bracket and fall back to bisection when necessary. Small steps alone do
//! not establish convergence.
//!
//! Results keep the signed value, final bracket, termination reason, and
//! evaluation counts. An iteration limit is a clean result, while invalid
//! inputs and application errors return typed errors. Each stage starts fresh
//! and counts its own work, including reevaluating bracket endpoints.

/// Brent's bracketed scalar root finder.
pub mod brent;

mod common;
/// Errors shared by the secant, Newton, Halley, and TOMS 748 solvers.
pub mod error;
/// Safeguarded Halley iteration.
pub mod halley;
/// Safeguarded Newton iteration.
pub mod newton;
/// Safeguarded secant iteration.
pub mod secant;
/// TOMS Algorithm 748 (`k = 2`).
pub mod toms748;

pub use brent::{BrentRoot, BrentRootError, RootResult, RootTerminationReason};
pub use error::RootError;
pub use halley::HalleyRoot;
pub use newton::NewtonRoot;
pub use secant::SecantRoot;
pub use toms748::Toms748Root;
