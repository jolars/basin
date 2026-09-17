//! Fresh solves of the L-BFGS-B 3.0 reference drivers, including initialization.

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
#[path = "support/lbfgsb.rs"]
mod support;

use criterion::{Criterion, criterion_group, criterion_main};
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
    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
