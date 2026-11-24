//! Audit native and reconstructed evidence at the returned point.

#[path = "support/bounded_fit_diagnostics.rs"]
mod bounded_fit;
#[path = "support/slsqp_diagnostics.rs"]
mod slsqp;

fn main() {
    slsqp::run();
    bounded_fit::run();
}
