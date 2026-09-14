//! Test-only prototype: no exports, features, or dependencies are added to Basin.
//! The small driver uses Basin's `Solver`, `Problem`, and observation traits.
//! Numerical steps mirror the existing solvers so ownership can be tested
//! against their trajectories before choosing a production implementation.

#[path = "state_api/driver.rs"]
pub mod driver;
#[path = "state_api/solvers.rs"]
pub mod solvers;
#[path = "state_api/state.rs"]
pub mod state;
