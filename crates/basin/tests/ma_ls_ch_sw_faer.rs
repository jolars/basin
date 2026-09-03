#![cfg(feature = "faer_all")]

//! faer-backend smoke mirror for [`MaLsChSw`]; the deep tests live in
//! `tests/ma_ls_ch_sw_nalgebra.rs`. Vector type only—no `Mat`.

use crate::backend_aliases::faer::Col;
use basin::problems::SphereBoxed;
use basin::{Executor, MaLsChSw, MaLsChSwState, MaxCostEvals};

#[test]
fn converges_on_sphere_d10() {
    let problem =
        SphereBoxed::new(Col::from_fn(10, |_| -5.0), Col::from_fn(10, |_| 5.0));
    let solver = MaLsChSw::<Col<f64>>::new(7).with_pop_size(20);
    let result = Executor::new(problem, solver, MaLsChSwState::new())
        .max_iter(u64::MAX)
        .terminate_on(MaxCostEvals(20_000))
        .run()
        .unwrap();

    assert!(
        result.cost() < 1e-6,
        "Sphere(D=10) faer cost {} should be < 1e-6 within 20k evals",
        result.cost()
    );
}

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
