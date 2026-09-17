//! Fresh solves of the L-BFGS-B 3.0 reference drivers, including initialization.

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
        group.bench_function(format!("driver{number}"), |b| {
            b.iter(|| black_box(problem.solve()));
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
