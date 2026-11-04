//! Fresh solves of the L-BFGS-B 3.0 reference drivers, including initialization.

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
#[path = "support/lbfgsb.rs"]
mod support;

// Compile the private production kernels directly, avoiding a separate copy
// of their implementation or a public API solely for benchmarking.
pub use basin::core;
#[allow(dead_code, unused_imports)]
#[path = "../src/solver/lbfgs/compact.rs"]
mod compact;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn benchmark(c: &mut Criterion) {
    let mut kernels = c.benchmark_group("lbfgsb_kernels");
    for n in [1, 2, 5, 10, 20, 40] {
        let stride = 2 * n + 4;
        let source: Vec<f64> = (0..n * stride)
            .map(|i| ((i * 17 + 3) % 31) as f64 / 16.0 - 1.0)
            .collect();
        let initial = vec![0.125_f64; n * stride];
        kernels.bench_function(format!("gram/n{n}"), |b| {
            b.iter_batched(
                || initial.clone(),
                |mut target| {
                    compact::add_gram_upper(
                        black_box(&source),
                        &mut target,
                        n,
                        stride,
                    );
                    black_box(target)
                },
                BatchSize::SmallInput,
            );
        });
        let mut positive = vec![-77.0_f64; n * stride];
        for i in 0..n {
            for j in i..n {
                positive[i * stride + j] = if i == j {
                    n as f64 + 1.0
                } else {
                    0.125 / (j - i + 1) as f64
                };
            }
        }
        kernels.bench_function(format!("cholesky/n{n}"), |b| {
            b.iter_batched(
                || positive.clone(),
                |mut target| {
                    let succeeded = compact::cholesky_upper_in_place(
                        &mut target,
                        n,
                        stride,
                    );
                    black_box((succeeded, target))
                },
                BatchSize::SmallInput,
            );
        });
    }
    kernels.finish();

    let mut group = c.benchmark_group("lbfgsb_reference");
    for number in 1..=3 {
        let mut problem = support::Driver::new(number);
        problem.check_feasibility = true;
        problem.verify(&problem.solve());
        problem.check_feasibility = false;
        group.bench_function(format!("driver{number}/vec"), |b| {
            b.iter(|| black_box(problem.solve()));
        });
        #[cfg(feature = "faer_all")]
        {
            let mut problem = support::Driver::<backend_aliases::faer::Col<f64>>::with_backend(number);
            problem.check_feasibility = true;
            problem.verify(&problem.solve());
            problem.check_feasibility = false;
            group.bench_function(format!("driver{number}/faer"), |b| {
                b.iter(|| black_box(problem.solve()));
            });
        }
    }
    for case in [support::short::Case::Mixed, support::short::Case::Rollover] {
        let mut problem = support::short::Short::<Vec<f64>>::new(case);
        for adapter in [
            support::short::Adapter::Reference,
            support::short::Adapter::Trimmed,
            support::short::Adapter::RequiredStops,
        ] {
            problem.check_feasibility = true;
            problem.verify(&problem.solve(adapter));
            problem.check_feasibility = false;
            let name = format!("{}/vec/{}", case.name(), adapter.name());
            group.bench_function(format!("{name}/solve"), |b| {
                b.iter(|| {
                    let result = problem.solve(adapter);
                    black_box(problem.extract(&result));
                });
            });
            // Initialization includes its first fused evaluation and teardown.
            group.bench_function(format!("{name}/initialize_and_drop"), |b| {
                b.iter(|| {
                    black_box(problem.initialize(adapter));
                });
            });
            // Setup is excluded; iteration, extraction, and teardown are timed.
            // Batched phase measurements need not sum to the full-solve timing.
            group.bench_function(format!("{name}/iterations_and_drop"), |b| {
                b.iter_batched(
                    || problem.initialize(adapter),
                    |stepper| {
                        let result = stepper.run_to_end().unwrap();
                        black_box(problem.extract(&result));
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
