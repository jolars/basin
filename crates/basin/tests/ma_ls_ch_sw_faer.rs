#![cfg(feature = "faer")]

//! faer-backend smoke mirror for [`MaLsChSw`]; the deep tests live in
//! `tests/ma_ls_ch_sw_nalgebra.rs`. Vector type only — no `Mat`.

use basin::{CostFunction, Executor, MaLsChSw, MaLsChSwState, MaxCostEvals};
use faer::Col;

struct BoxedSphere {
    lower: Col<f64>,
    upper: Col<f64>,
}

impl CostFunction for BoxedSphere {
    type Param = Col<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(x.iter().map(|v| v * v).sum())
    }
}

impl basin::BoxConstraints for BoxedSphere {
    fn lower(&self) -> &Col<f64> {
        &self.lower
    }
    fn upper(&self) -> &Col<f64> {
        &self.upper
    }
}

#[test]
fn converges_on_sphere_d10() {
    let problem = BoxedSphere {
        lower: Col::from_fn(10, |_| -5.0),
        upper: Col::from_fn(10, |_| 5.0),
    };
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
