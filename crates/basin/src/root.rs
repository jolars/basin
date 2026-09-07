//! Scalar root-finding algorithms.
//!
//! Root finding is deliberately separate from Basin's optimization
//! [`Solver`](crate::Solver)/[`Executor`](crate::Executor) loop. Optimization
//! states rank points by a scalar cost, whereas a bracketed root finder must
//! retain the signed function value and a sign-changing interval. The direct
//! APIs in this module preserve those semantics without treating `|f(x)|` as
//! an optimization objective.

/// Brent's bracketed scalar root finder.
pub mod brent;

pub use brent::{BrentRoot, BrentRootError, RootResult, RootTerminationReason};
