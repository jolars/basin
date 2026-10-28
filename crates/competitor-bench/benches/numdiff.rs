//! Uninstrumented finite-difference timings. See `NUMDIFF.md` for the protocol.

#[path = "support/numdiff.rs"]
mod support;

use basin::MatrixIndex;
use criterion::{Criterion, criterion_group, criterion_main};
use support::{BenchVector, Fixture, Function};

fn backend<V>(c: &mut Criterion, backend: &str)
where
    V: BenchVector,
    V::Matrix: MatrixIndex,
{
    let mut group = c.benchmark_group(format!("numdiff/{backend}"));
    for case in support::cases() {
        for &call in case.calls() {
            let name = format!("{}/{}", case.name(), call.name());
            group.bench_function(name, |b| {
                let mut fixture =
                    Fixture::<V, _>::new(case, Function::new(case));
                fixture.run(call);
                fixture.verify();
                b.iter(|| fixture.run(call));
            });
        }
    }
    group.finish();
}

fn benchmark(c: &mut Criterion) {
    eprintln!("numdiff: {} (f64)", support::execution());
    backend::<Vec<f64>>(c, "vec");
    backend::<support::NalgebraVector>(c, "nalgebra");
    backend::<ndarray::Array1<f64>>(c, "ndarray");
    backend::<faer::Col<f64>>(c, "faer");
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
