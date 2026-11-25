use std::{hint::black_box, time::Duration};

use checkpoint_codecs::{SPLIT_ITER, SerdeCodec, bfgs, cma_es};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use serde::{Serialize, de::DeserializeOwned};

fn measure<T: Serialize + DeserializeOwned>(
    criterion: &mut Criterion,
    solver: &str,
    dimension: usize,
    checkpoint: &T,
) {
    let mut group = criterion.benchmark_group(format!("{solver}/{dimension}"));
    for codec in SerdeCodec::ALL {
        let bytes = codec.encode(checkpoint).unwrap();
        assert!(codec.decode::<T>(&bytes).is_ok());
        println!("{solver},{dimension},{},{}", codec.name(), bytes.len());
        group.bench_function(BenchmarkId::new("encode", codec.name()), |b| {
            b.iter(|| codec.encode(black_box(checkpoint)).unwrap())
        });
        group.bench_function(BenchmarkId::new("decode", codec.name()), |b| {
            b.iter(|| codec.decode::<T>(black_box(&bytes)).unwrap())
        });
    }
    group.finish();
}

fn payload(criterion: &mut Criterion) {
    println!("solver,dimension,codec,payload_bytes");
    for dimension in [8, 32, 128] {
        let mut bfgs = bfgs(dimension);
        let mut cma_es = cma_es(dimension);
        for _ in 0..SPLIT_ITER {
            assert!(bfgs.step().unwrap().report().is_none());
            assert!(cma_es.step().unwrap().report().is_none());
        }
        assert_eq!(bfgs.iter(), SPLIT_ITER);
        assert_eq!(cma_es.iter(), SPLIT_ITER);
        measure(
            criterion,
            "bfgs",
            dimension,
            &bfgs.into_checkpoint().unwrap(),
        );
        measure(
            criterion,
            "cma-es",
            dimension,
            &cma_es.into_checkpoint().unwrap(),
        );
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .without_plots()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1));
    targets = payload
}
criterion_main!(benches);
