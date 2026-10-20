//! Canonical problem-side constraint sets and checked evaluation capabilities.
//!
//! [`Constraints`] optionally declares the complete constraint set. Constrained
//! solvers accept only block combinations they support. Unconstrained solvers
//! ignore the declaration. Bounds, linear equalities and
//! inequalities, and nonlinear equalities and inequalities remain distinct.

//!
//! An unconstrained solver can optimize an objective without enforcing its box:
//!
//! ```
//! use basin::{BoxBounds, Constrained, ConstraintSet, CostFunction, Executor,
//!             NelderMead};
//! struct Sphere;
//! impl CostFunction for Sphere {
//!     type Param = Vec<f64>;
//!     type Output = f64;
//!     type Error = std::convert::Infallible;
//!     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
//!         Ok(x.iter().map(|v| (v - 2.0).powi(2)).sum())
//!     }
//! }
//! let constraints = ConstraintSet::new()
//!     .with_bounds(BoxBounds::new(Some(vec![-1.0]), Some(vec![1.0])));
//! let problem = Constrained::new(Sphere, constraints);
//! let result = Executor::from_start(problem, NelderMead::new(), vec![0.5])
//!     .max_iter(1000).run().unwrap();
//! assert!(result.param()[0] > 1.0);
//! ```
//!
//! A box-only solver also rejects an additional nonlinear block, even when
//! the problem supplies the required value callback:
//!
//! ```compile_fail
//! use basin::{BoxBounds, Constrained, ConstraintSolveError, ConstraintSet,
//!             CostFunction, Executor, NelderMead,
//!             NonlinearInequalityValues};
//! struct Sphere;
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
//! let constraints = ConstraintSet::new()
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
pub trait ConstraintJacobian {
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
}
