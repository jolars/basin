//! CDP-1 measurement tools, independent of production stopping settings.
//!
//! This first harness stage covers analytic smooth objectives and box constraints.
//! General KKT certificates, NIST models, and native trial diagnostics remain
//! separate validation gates. Instrumented elapsed time is diagnostic only.

pub mod fixtures;
pub mod ledger;
pub mod quality;
pub mod runner;
