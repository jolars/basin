//! Fresh solves of the L-BFGS-B 3.0 reference drivers, including initialization.

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
#[path = "support/lbfgsb.rs"]
mod support;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn benchmark(c: &mut Criterion) {
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
