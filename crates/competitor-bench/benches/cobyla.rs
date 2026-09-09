//! Fixed-budget cases from the GlobalSearch migration, without executor overhead.

pub use basin::core;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

#[path = "support/cobyla.rs"]
// Cargo also checks benchmarks with cfg(test), but without test functions.
#[allow(unused_imports)]
mod cobyla;

fn benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cobyla_driver");
    for (case, name) in cobyla::CASES.into_iter().chain(cobyla::SCALING_CASES) {
        group.bench_function(name, |b| b.iter(|| black_box(case).solve()));
    }
    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
