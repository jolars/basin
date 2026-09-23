//! Round-trip test for the `serde`-gated [`CheckpointWriter`] observer: run,
//! checkpoint, reload, warm-start, and confirm this deterministic solver
//! matches an uninterrupted run.
#![cfg(all(feature = "serde", not(target_arch = "wasm32")))]

#[path = "support/backend_aliases.rs"]
mod backend_aliases;

use basin::{
    BasicState, CheckpointWriter, CostFunction, Executor, Gradient,
    GradientDescent, Observe, ObserverMode, State, read_checkpoint,
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
fn checkpoint_warm_start_matches_uninterrupted_run() {
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
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], b"BASINST\0");
    assert_eq!(&bytes[8..12], &1_u32.to_le_bytes());
    assert_eq!(&bytes[12..], postcard::to_allocvec(&reloaded).unwrap());
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

#[test]
fn legacy_checkpoint_can_be_read_and_rewritten_as_postcard() {
    let path = std::env::temp_dir()
        .join(format!("basin-legacy-{}.ckpt", std::process::id()));
    let state = BasicState::<Vec<f64>>::new(vec![1.0; 300]);
    let legacy =
        bincode::serde::encode_to_vec(&state, bincode::config::standard())
            .unwrap();
    std::fs::write(&path, &legacy).unwrap();
    let restored: BasicState<Vec<f64>> = read_checkpoint(&path).unwrap();
    assert_eq!(restored.param(), state.param());
    assert_eq!(std::fs::read(&path).unwrap(), legacy);

    // The legacy reader has always permitted trailing bytes.
    let mut trailing = legacy;
    trailing.push(0);
    std::fs::write(&path, trailing).unwrap();
    let restored: BasicState<Vec<f64>> = read_checkpoint(&path).unwrap();
    CheckpointWriter::new(&path).observe_iter(&restored);
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], b"BASINST\0");
    assert_eq!(&bytes[12..], postcard::to_allocvec(&state).unwrap());
    let restored: BasicState<Vec<f64>> = read_checkpoint(&path).unwrap();
    assert_eq!(restored.param(), state.param());
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn invalid_postcard_checkpoints_do_not_fall_back_to_bincode() {
    let path = std::env::temp_dir()
        .join(format!("basin-invalid-{}.ckpt", std::process::id()));
    let prefix = [b"BASINST\0".as_slice(), &1_u32.to_le_bytes()].concat();
    let mut unsupported = prefix.clone();
    unsupported[8..12].copy_from_slice(&2_u32.to_le_bytes());

    for bytes in [
        prefix[..8].to_vec(),
        prefix[..10].to_vec(),
        unsupported,
        prefix.clone(),
        [prefix.as_slice(), &[1, 0]].concat(),
    ] {
        std::fs::write(&path, bytes).unwrap();
        // A bincode fallback would accept the first magic byte as a u8.
        let error = read_checkpoint::<u8>(&path).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }
    std::fs::write(&path, [prefix.as_slice(), &[2]].concat()).unwrap();
    assert_eq!(
        read_checkpoint::<bool>(&path).unwrap_err().kind(),
        std::io::ErrorKind::InvalidData
    );
    std::fs::remove_file(&path).unwrap();
}

fn round_trip<T: serde::Serialize + serde::de::DeserializeOwned>(
    value: &T,
) -> T {
    let bytes = postcard::to_allocvec(value).unwrap();
    let (restored, remaining) = postcard::take_from_bytes(&bytes).unwrap();
    assert!(remaining.is_empty());
    restored
}

macro_rules! check_float_bits {
    ($float:ty, $nan_bits:expr, $vector:expr) => {{
        let nan = <$float>::from_bits($nan_bits);
        let values: &[$float] = &[
            0.0,
            -0.0,
            1.5,
            <$float>::INFINITY,
            <$float>::NEG_INFINITY,
            nan,
            -nan,
        ];
        let state = BasicState::<_, $float>::new(($vector)(values));
        let restored = round_trip(&state);
        let expected = values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(
            restored
                .param()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
    }};
}

#[test]
fn postcard_preserves_vec_float_bits() {
    check_float_bits!(f32, 0x7fc0_0042, |values: &[f32]| values.to_vec());
    check_float_bits!(f64, 0x7ff8_0000_0000_0042, |values: &[f64]| values
        .to_vec());
}

#[cfg(feature = "nalgebra_all")]
#[test]
fn postcard_preserves_nalgebra_float_bits() {
    use backend_aliases::nalgebra::DVector;
    check_float_bits!(f32, 0x7fc0_0042, DVector::from_column_slice);
    check_float_bits!(f64, 0x7ff8_0000_0000_0042, DVector::from_column_slice);
}

#[cfg(feature = "ndarray_all")]
#[test]
fn postcard_preserves_ndarray_float_bits() {
    use backend_aliases::ndarray::Array1;
    check_float_bits!(f32, 0x7fc0_0042, |values: &[f32]| Array1::from_vec(
        values.to_vec()
    ));
    check_float_bits!(f64, 0x7ff8_0000_0000_0042, |values: &[f64]| {
        Array1::from_vec(values.to_vec())
    });
}
