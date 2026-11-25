#![cfg(feature = "serde-codecs")]

use basin::{
    CountsMirror, ExactCheckpoint, Executor, Solver, State, StepOutcome,
    Stepper, Termination,
};
use bincode_next::{Decode, Encode, Fingerprint, config, error::DecodeError};
use checkpoint_codecs::{
    BfgsCheckpoint, MAX_ITER, Quadratic, SPLIT_ITER, SerdeCodec, bfgs, cma_es,
};
use serde::{Serialize, de::DeserializeOwned};
use std::convert::Infallible;
use std::marker::PhantomData;

fn assert_same_boundary<S, So>(
    expected: &Stepper<Quadratic, S, So>,
    actual: &Stepper<Quadratic, S, So>,
) where
    S: State<Float = f64> + CountsMirror + Serialize,
    So: Solver<Quadratic, S> + Serialize,
{
    assert_eq!(actual.iter(), expected.iter());
    assert_eq!(actual.counts(), expected.counts());
    // Compare every serialized field and preserve distinctions like signed zero.
    assert_eq!(
        postcard::to_allocvec(&(actual.solver(), actual.state())).unwrap(),
        postcard::to_allocvec(&(expected.solver(), expected.state())).unwrap(),
        "iteration {}",
        expected.iter(),
    );
}

fn assert_continuation<S, So>(
    make: impl Fn() -> Stepper<Quadratic, S, So>,
    round_trip: impl FnOnce(ExactCheckpoint<So, S>) -> ExactCheckpoint<So, S>,
) where
    S: State<Float = f64> + CountsMirror + Serialize,
    So: Solver<Quadratic, S, Error = Infallible> + Serialize,
{
    let mut reference = make();
    let mut split = make();
    for iter in 1..=SPLIT_ITER {
        assert_eq!(reference.step().unwrap(), StepOutcome::Continue);
        assert_eq!(split.step().unwrap(), StepOutcome::Continue);
        assert_eq!(split.iter(), iter);
    }
    let restored = round_trip(split.into_checkpoint().unwrap());
    let mut resumed = Executor::resume_from_checkpoint(Quadratic, restored)
        .max_iter(MAX_ITER)
        .into_stepper()
        .unwrap();
    assert_same_boundary(&reference, &resumed);
    loop {
        let expected = reference.step().unwrap();
        assert_eq!(resumed.step().unwrap(), expected);
        assert_same_boundary(&reference, &resumed);
        if let StepOutcome::Stopped(report) = expected {
            assert_eq!(report.iteration, MAX_ITER);
            assert!(matches!(report.termination, Termination::Limit(_)));
            break;
        }
    }
}

fn round_trip<T: Serialize + DeserializeOwned>(
    codec: SerdeCodec,
    checkpoint: T,
) -> T {
    let name = codec.name();
    let mut bytes = codec.encode(&checkpoint).expect(name);
    assert!(
        codec.decode::<T>(&bytes[..bytes.len() - 1]).is_err(),
        "{name}"
    );
    let restored = codec.decode(&bytes).expect(name);
    bytes.push(0);
    assert!(codec.decode::<T>(&bytes).is_err(), "{name}");
    restored
}

#[test]
fn ndarray_bfgs_resumes_after_inverse_hessian_updates() {
    let make = || bfgs(8);
    let initial = make();
    for codec in SerdeCodec::ALL {
        assert_continuation(make, |checkpoint| {
            assert_ne!(
                checkpoint.solver().inverse_hessian(),
                initial.solver().inverse_hessian(),
            );
            round_trip(codec, checkpoint)
        });
    }
}

#[test]
fn seeded_cma_es_resumes_between_eigen_updates() {
    for codec in SerdeCodec::ALL {
        assert_continuation(
            || cma_es(8),
            |checkpoint| {
                assert_eq!(
                    checkpoint.solver().distribution().unwrap().generation,
                    SPLIT_ITER
                );
                round_trip(codec, checkpoint)
            },
        );
    }
}

// Use a manually versioned marker because bincode-next cannot inspect the
// field layout of a Serde payload.
#[derive(Encode, Decode)]
struct LiveBfgsCheckpoint<Layout>(
    #[bincode(with_serde)] BfgsCheckpoint,
    PhantomData<Layout>,
);

#[derive(Encode, Decode, Fingerprint)]
struct NdarrayBfgsSerdeLayoutV1;

#[derive(Decode, Fingerprint)]
struct NdarrayBfgsSerdeLayoutV2;

impl<C: config::Config, Layout: Fingerprint<C>> Fingerprint<C>
    for LiveBfgsCheckpoint<Layout>
{
    const SCHEMA_HASH: u64 = Layout::SCHEMA_HASH;
}

#[test]
fn bincode_fingerprints_a_live_checkpoint_with_an_explicit_serde_revision() {
    let config = config::standard().with_fingerprint();
    assert_continuation(
        || bfgs(8),
        |checkpoint| {
            assert_eq!(
                bincode_next::serde::encode_to_vec(&checkpoint, config)
                    .unwrap(),
                bincode_next::serde::encode_to_vec(
                    &checkpoint,
                    config::standard()
                )
                .unwrap(),
            );
            let bytes = bincode_next::encode_to_vec(
                LiveBfgsCheckpoint::<NdarrayBfgsSerdeLayoutV1>(
                    checkpoint,
                    PhantomData,
                ),
                config,
            )
            .unwrap();
            assert!(matches!(
                bincode_next::decode_from_slice::<
                    LiveBfgsCheckpoint<NdarrayBfgsSerdeLayoutV2>,
                    _,
                >(&bytes, config),
                Err(DecodeError::SchemaMismatch { .. })
            ));
            let (LiveBfgsCheckpoint(restored, _), consumed) =
                bincode_next::decode_from_slice::<
                    LiveBfgsCheckpoint<NdarrayBfgsSerdeLayoutV1>,
                    _,
                >(&bytes, config)
                .unwrap();
            assert_eq!(consumed, bytes.len());
            restored
        },
    );
}
