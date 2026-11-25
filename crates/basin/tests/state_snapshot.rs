//! State snapshot files and Serde payload compatibility.
#![cfg(feature = "serde")]

#[path = "support/backend_aliases.rs"]
mod backend_aliases;

use basin::{FirstOrderState, State};

#[cfg(all(feature = "checkpoint-postcard", not(target_arch = "wasm32")))]
mod file {
    use super::*;
    use basin::{
        CostFunction, Executor, Gradient, GradientDescent, ObserverMode,
        StateSnapshotWriter, read_checkpoint, read_state_snapshot,
    };

    struct Quadratic;

    impl basin::Constraints for Quadratic {
        type Set = basin::NoConstraints;
        fn constraints(
            &self,
        ) -> <Self::Set as basin::ConstraintDescription>::Borrowed<'_> {
            basin::ConstraintSet::new()
        }
    }
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
    fn state_snapshot_warm_start_matches_uninterrupted_run() {
        let dir = std::env::temp_dir()
            .join(format!("basin-ckpt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("run.ckpt");

        let start = vec![5.0, -3.0, 2.0];
        let step = 0.1;

        // Reference: 20 uninterrupted iterations.
        let reference = Executor::new(
            Quadratic,
            GradientDescent::new(step),
            FirstOrderState::new(start.clone()),
        )
        .max_iter(20)
        .run()
        .unwrap();

        // Save the final state after 12 iterations, then start from it and run
        // 8 more iterations.
        Executor::new(
            Quadratic,
            GradientDescent::new(step),
            FirstOrderState::new(start),
        )
        .max_iter(12)
        .observe_with(StateSnapshotWriter::new(&path), ObserverMode::Every(4))
        .run()
        .unwrap();

        let reloaded: FirstOrderState<Vec<f64>> =
            read_state_snapshot(&path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..8], b"BASINST\0");
        assert_eq!(&bytes[8..12], &1_u32.to_le_bytes());
        assert_eq!(&bytes[12..], postcard::to_allocvec(&reloaded).unwrap());
        let error = read_checkpoint::<u8, u8>(&path).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("magic"));
        assert_eq!(reloaded.iter(), 12);

        // A state-only snapshot starts a fresh run: its point is reevaluated,
        // and iteration and evaluation counters restart at zero.
        let resumed =
            Executor::new(Quadratic, GradientDescent::new(step), reloaded)
                .max_iter(8)
                .run()
                .unwrap();

        // Same optimum, reached identically.
        assert_eq!(resumed.iter(), 8);
        assert_eq!(resumed.state.counts().cost_evals, 9);
        assert_eq!(resumed.state.counts().gradient_evals, 9);
        for (a, b) in resumed.param().iter().zip(reference.param()) {
            assert!((a - b).abs() < 1e-12, "resumed {a} vs reference {b}");
        }
        assert!((resumed.cost() - reference.cost()).abs() < 1e-12);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn unprefixed_state_snapshots_are_rejected() {
        let path = std::env::temp_dir()
            .join(format!("basin-unprefixed-{}.ckpt", std::process::id()));
        // The scalar payload is valid by itself, but a file needs a prefix.
        let bytes = [1];
        std::fs::write(&path, bytes).unwrap();
        let error = read_state_snapshot::<u8>(&path).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("magic"));
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn invalid_postcard_state_snapshots_are_rejected() {
        let path = std::env::temp_dir()
            .join(format!("basin-invalid-{}.ckpt", std::process::id()));
        let prefix = [b"BASINST\0".as_slice(), &1_u32.to_le_bytes()].concat();
        let mut unsupported = prefix.clone();
        unsupported[8..12].copy_from_slice(&2_u32.to_le_bytes());

        for bytes in [
            vec![],
            prefix[..3].to_vec(),
            prefix[..8].to_vec(),
            prefix[..10].to_vec(),
            unsupported,
            prefix.clone(),
            [prefix.as_slice(), &[1, 0]].concat(),
        ] {
            std::fs::write(&path, bytes).unwrap();
            let error = read_state_snapshot::<u8>(&path).unwrap_err();
            assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        }
        std::fs::write(&path, [prefix.as_slice(), &[2]].concat()).unwrap();
        assert_eq!(
            read_state_snapshot::<bool>(&path).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        std::fs::remove_file(&path).unwrap();
    }
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
        let state = FirstOrderState::<_, $float>::new(($vector)(values));
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

#[test]
fn release_eval_counts_keep_the_six_counter_postcard_layout() {
    let mut counts = basin::EvalCounts::default();
    counts.cost_evals = 1;
    counts.gradient_evals = 2;
    counts.residual_evals = 3;
    counts.jacobian_evals = 4;
    counts.hessian_evals = 5;
    counts.hessian_product_evals = 6;
    assert_eq!(postcard::to_allocvec(&counts).unwrap(), [1, 2, 3, 4, 5, 6]);
    assert_eq!(
        postcard::from_bytes::<basin::EvalCounts>(&[1, 2, 3, 4, 5, 6]).unwrap(),
        counts
    );
}
