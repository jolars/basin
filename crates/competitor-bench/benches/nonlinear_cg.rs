use std::hint::black_box;
use std::time::Duration;

use competitor_bench::nonlinear_cg::{DIMENSION, Library, run};
use criterion::{Criterion, criterion_group, criterion_main};

fn benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("nonlinear_cg_pr_plus_rosenbrock_n100");
    for library in [Library::Basin, Library::Argmin] {
        let check = run(library, DIMENSION, false);
        assert!(check.summary.verified, "{:?}", check.summary);
        group.bench_function(library.name(), |b| {
            b.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    let result = black_box(run(library, DIMENSION, false));
                    assert!(result.summary.verified, "{:?}", result.summary);
                    let (nanos, _) = result.points[0];
                    elapsed +=
                        Duration::from_nanos(u64::try_from(nanos).unwrap());
                }
                elapsed
            });
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
