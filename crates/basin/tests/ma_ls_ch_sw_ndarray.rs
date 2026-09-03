#![cfg(feature = "ndarray_all")]

//! ndarray-backend smoke mirror for [`MaLsChSw`]; the deep tests live
//! in `tests/ma_ls_ch_sw_nalgebra.rs`. Vector type only—no `Array2`.

use crate::backend_aliases::ndarray::Array1;
use basin::problems::SphereBoxed;
use basin::{Executor, MaLsChSw, MaLsChSwState, MaxCostEvals};

#[test]
fn converges_on_sphere_d10() {
    let problem = SphereBoxed::new(
        Array1::from_elem(10, -5.0),
        Array1::from_elem(10, 5.0),
    );
    let solver = MaLsChSw::<Array1<f64>>::new(7).with_pop_size(20);
    let result = Executor::new(problem, solver, MaLsChSwState::new())
        .max_iter(u64::MAX)
        .terminate_on(MaxCostEvals(20_000))
        .run()
        .unwrap();

    assert!(
        result.cost() < 1e-6,
        "Sphere(D=10) ndarray cost {} should be < 1e-6 within 20k evals",
        result.cost()
    );
}

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
