//! Integration test for [`DeInject`] with [`Lbfgsb`] inner on the
//! nalgebra backend.
//!
//! Booth `[-5, 5]²` — global min at `(1, 3)` strictly interior — with
//! a small DE population and L-Bfgs-B polishing. DE places candidates
//! in the basin within a handful of generations; L-Bfgs-B drives them
//! to gradient-descent precision. Assert `‖x* − (1, 3)‖_∞ ≤ 1e-6`.

#![cfg(feature = "nalgebra")]

use basin::problems::BoothBoxed;
use basin::{BasicPopulationState, De, DeInject, Executor, Lbfgsb};
use nalgebra::DVector;

#[test]
fn converges_on_booth_boxed_to_lbfgsb_precision() {
    let lower = DVector::from_vec(vec![-5.0, -5.0]);
    let upper = DVector::from_vec(vec![5.0, 5.0]);
    let problem = BoothBoxed::<DVector<f64>>::new(lower, upper);

    let de = De::new(19).with_pop_size(12);
    let solver = DeInject::with_inner_solver(de, Lbfgsb::new())
        .with_k(1)
        .with_inner_max_iter(50);

    let result = Executor::new(
        problem,
        solver,
        BasicPopulationState::<DVector<f64>>::with_size(1),
    )
    .max_iter(40)
    .run()
    .unwrap();

    let p = result.param();
    let err = (p[0] - 1.0).abs().max((p[1] - 3.0).abs());
    assert!(
        err <= 1e-6,
        "booth-boxed iterate = ({}, {}), expected ≈ (1, 3) within 1e-6 (err = {})",
        p[0],
        p[1],
        err
    );
}
