//! Canonical problem-side constraint sets and checked evaluation capabilities.
//!
//! [`ProblemConstraints`] declares the complete constraint set. Solvers accept
//! only block combinations they support. Bounds, linear equalities and
//! inequalities, and nonlinear equalities and inequalities remain distinct.

//!
//! An unconstrained solver cannot discard a declared box:
//!
//! ```compile_fail
//! use basin::{BoxBounds, Constrained, Constraints, CostFunction, Executor,
//!             NelderMead, NoConstraints, ProblemConstraints};
//! struct Sphere;
//! impl ProblemConstraints for Sphere {
//!     type Constraints = NoConstraints;
//!     fn constraint_set(&self) -> basin::NoConstraints { basin::NoConstraints::EMPTY }
//! }
//! impl CostFunction for Sphere {
//!     type Param = Vec<f64>;
//!     type Output = f64;
//!     type Error = std::convert::Infallible;
//!     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
//!         Ok(x.iter().map(|v| v * v).sum())
//!     }
//! }
//! let constraints = Constraints::new()
//!     .with_bounds(BoxBounds::new(Some(vec![-1.0]), Some(vec![1.0])));
//! let problem = Constrained::new(Sphere, constraints);
//! Executor::from_start(problem, NelderMead::new(), vec![0.5]).run();
//! ```
//!
//! A box-only solver also rejects an additional nonlinear block, even when
//! the problem supplies the required value callback:
//!
//! ```compile_fail
//! use basin::{BoxBounds, Constrained, ConstraintSolveError, Constraints,
//!             CostFunction, Executor, NelderMead, NoConstraints,
//!             NonlinearInequalityValues, ProblemConstraints};
//! struct Sphere;
//! impl ProblemConstraints for Sphere {
//!     type Constraints = NoConstraints;
//!     fn constraint_set(&self) -> basin::NoConstraints { basin::NoConstraints::EMPTY }
//! }
//! impl CostFunction for Sphere {
//!     type Param = Vec<f64>;
//!     type Output = f64;
//!     type Error = std::convert::Infallible;
//!     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
//!         Ok(x.iter().map(|v| v * v).sum())
//!     }
//! }
//! impl NonlinearInequalityValues for Sphere {
//!     type Param = Vec<f64>;
//!     type Values = Vec<f64>;
//!     type Error = std::convert::Infallible;
//!     fn nonlinear_inequalities(&self, x: &Vec<f64>)
//!         -> Result<Vec<f64>, ConstraintSolveError<Self::Error>> {
//!         Ok(vec![x[0] - 0.25])
//!     }
//! }
//! let constraints = Constraints::new()
//!     .with_bounds(BoxBounds::new(Some(vec![-1.0]), Some(vec![1.0])))
//!     .with_nonlinear_inequalities(1);
//! let problem = Constrained::new(Sphere, constraints);
//! Executor::from_start(problem, NelderMead::new().projected(), vec![0.5]).run();
//! ```

pub(crate) mod model;
pub use model::*;
mod adapters;
pub(crate) mod evaluation;
pub use adapters::*;

/// Analytic Jacobian of the declared nonlinear blocks.
///
/// Rows contain nonlinear equalities first, followed by nonlinear inequalities
/// in callback order. Columns correspond to parameters. Linear blocks supply
/// their own derivatives and are excluded. This storage is independent of the
/// linear matrices. Finite-difference adapters synthesize this capability.
pub trait ConstraintJacobian: ProblemConstraints {
    /// Parameter type.
    type Param;
    /// Nonlinear Jacobian storage.
    type Jacobian;
    /// Original callback error type.
    type Error;
    /// Evaluate equality rows first, followed by inequality rows.
    fn constraint_jacobian(
        &self,
        x: &Self::Param,
    ) -> Result<Self::Jacobian, ConstraintSolveError<Self::Error>>;
    /// Evaluate through a shared hard-budget context.
    fn constraint_jacobian_with_context(
        &self,
        x: &Self::Param,
        context: &crate::EvaluationContext,
    ) -> Result<
        Self::Jacobian,
        crate::ConstraintSolveError<crate::EvaluationError<Self::Error>>,
    > {
        context
            .reserve_kinds(&[crate::EvaluationKind::Jacobian])
            .map_err(|error| {
                crate::ConstraintSolveError::Evaluation(error.into())
            })?;
        self.constraint_jacobian(x)
            .map_err(crate::core::evaluation::constraint_error)
    }
}
