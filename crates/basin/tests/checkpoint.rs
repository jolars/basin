//! Round-trip test for the `serde`-gated [`CheckpointWriter`] observer: run,
//! checkpoint, reload, resume, and confirm the resumed run matches an
//! uninterrupted one.
#![cfg(all(feature = "serde", not(target_arch = "wasm32")))]

use basin::{
    BasicState, CheckpointWriter, CostFunction, Executor, Gradient,
    GradientDescent, ObserverMode, State, read_checkpoint,
};

struct Quadratic;

impl CostFunction for Quadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(0.5 * x.iter().map(|v| v * v).sum::<f64>())
    }
}

impl Gradient for Quadratic {
    type Gradient = Vec<f64>;

    fn gradient(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, std::convert::Infallible> {
        Ok(x.clone())
    }
}

#[test]
fn checkpoint_resume_matches_uninterrupted_run() {
    let dir =
        std::env::temp_dir().join(format!("basin-ckpt-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("run.ckpt");

    let start = vec![5.0, -3.0, 2.0];
    let step = 0.1;

    // Reference: 20 uninterrupted iterations.
    let reference = Executor::new(
        Quadratic,
        GradientDescent::new(step),
        BasicState::new(start.clone()),
    )
    .max_iter(20)
    .run()
    .unwrap();

    // Split run: 12 iterations, checkpointing the final state, then reload and
    // run 8 more.
    Executor::new(
        Quadratic,
        GradientDescent::new(step),
        BasicState::new(start),
    )
    .max_iter(12)
    .observe_with(CheckpointWriter::new(&path), ObserverMode::Every(4))
    .run()
    .unwrap();

    let reloaded: BasicState<Vec<f64>> = read_checkpoint(&path).unwrap();
    // The checkpoint captured the 12th iterate.
    assert_eq!(reloaded.iter(), 12);

    // `max_iter` is checked against the absolute `state.iter()`, and the
    // reloaded state already stands at iter 12, so 20 means "8 more".
    let resumed =
        Executor::new(Quadratic, GradientDescent::new(step), reloaded)
            .max_iter(20)
            .run()
            .unwrap();

    // Same optimum, reached identically.
    assert_eq!(resumed.iter(), 20);
    for (a, b) in resumed.param().iter().zip(reference.param()) {
        assert!((a - b).abs() < 1e-12, "resumed {a} vs reference {b}");
    }
    assert!((resumed.cost() - reference.cost()).abs() < 1e-12);

    std::fs::remove_dir_all(&dir).ok();
}
