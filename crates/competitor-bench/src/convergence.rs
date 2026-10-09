//! CDP-1 measurement tools, independent of production stopping settings.
//!
//! This first harness stage covers analytic smooth objectives and box constraints.
//! General KKT certificates, NIST precision certificates, and native trial diagnostics remain
//! separate validation gates. Instrumented elapsed time is diagnostic only.

pub mod fixtures;
pub mod least_squares;
pub mod ledger;
pub mod nist;
pub mod quality;
pub mod runner;
