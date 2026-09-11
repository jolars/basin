//! Timing comparisons for gradient descent, Nelder-Mead, and COBYLA on
//! `Vec<f64>` (axis 3 of the bench plan).
//!
//! Both frameworks solve the *same* problem (basin's raw Rosenbrock
//! functions, wrapped for argmin via `ArgminProblem`) from the *same*
//! start with *matched* configuration:
//!   * GD:      steepest descent + More-Thuente line search.
//!   * NM:      standard coefficients (α=1, exp=2, con=0.5, shrink=0.5)
//!     and a bit-identical initial simplex (basin's `IntoInitialSimplex`,
//!     relative step 0.05).
//!
//! GD/NM use a *fixed* iteration budget with no early stop on either side
//! (`src/bin/verify_gd_nm.rs` confirms both run the full budget and reach
//! comparable cost), so the timing is a clean per-iteration
//! implementation-cost comparison.
//!
//! COBYLA compares Basin's PRIMA port with the `cobyla` crate on the original
//! migration workloads. It uses matched objective budgets and final radii,
//! with independent result checks before timing. The variants take different
//! trajectories, so these are whole-solve comparisons at a common quality
//! target. Timing includes setup, initialization, extraction, and teardown.
//!
//! Run: `cargo bench -p competitor-bench --bench gd_nm`.

use std::hint::black_box;

use argmin::core::Executor as ArgminExecutor;
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::neldermead::NelderMead as ArgminNelderMead;
use basin::problems::{Rosenbrock, rosenbrock, rosenbrock_gradient};
use basin::{
    BasicSimplexState, BasicState, Executor, GradientDescent,
    IntoInitialSimplex, MoreThuente, NelderMead,
};
use competitor_bench::ArgminProblem;
use criterion::{
    BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main,
};

const MAX_ITERS: u64 = 200;

/// Classic Rosenbrock start, extended to `n` dims by repeating the
/// `(−1.2, 1.0)` pair.
fn rosenbrock_start(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
        .collect()
}

fn bench_gd(c: &mut Criterion) {
    for n in [2usize, 10] {
        let mut g = c.benchmark_group(format!("gd_rosenbrock_n{n}"));

        g.bench_function(BenchmarkId::from_parameter("basin"), |b| {
            b.iter_batched(
                || rosenbrock_start(n),
                |x0| {
                    black_box(
                        Executor::new(
                            Rosenbrock::<Vec<f64>>::default(),
                            GradientDescent::with_line_search(
                                MoreThuente::new(),
                            ),
                            BasicState::new(x0),
                        )
                        .max_iter(MAX_ITERS)
                        .run(),
                    )
                },
                BatchSize::SmallInput,
            )
        });

        g.bench_function(BenchmarkId::from_parameter("argmin"), |b| {
            b.iter_batched(
                || rosenbrock_start(n),
                |x0| {
                    let ls: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> =
                        MoreThuenteLineSearch::new();
                    black_box(
                        ArgminExecutor::new(
                            ArgminProblem::new(rosenbrock, rosenbrock_gradient),
                            SteepestDescent::new(ls),
                        )
                        .configure(|s| s.param(x0).max_iters(MAX_ITERS))
                        .run()
                        .unwrap(),
                    )
                },
                BatchSize::SmallInput,
            )
        });

        g.finish();
    }
}

fn bench_nm(c: &mut Criterion) {
    for n in [2usize, 6] {
        let mut g = c.benchmark_group(format!("nm_rosenbrock_n{n}"));

        g.bench_function(BenchmarkId::from_parameter("basin"), |b| {
            b.iter_batched(
                || rosenbrock_start(n),
                |x0| {
                    black_box(
                        Executor::new(
                            Rosenbrock::<Vec<f64>>::default(),
                            NelderMead::new(),
                            BasicSimplexState::new(x0),
                        )
                        .max_iter(MAX_ITERS)
                        .run(),
                    )
                },
                BatchSize::SmallInput,
            )
        });

        g.bench_function(BenchmarkId::from_parameter("argmin"), |b| {
            b.iter_batched(
                || {
                    IntoInitialSimplex::into_initial_simplex(
                        rosenbrock_start(n),
                        0.05,
                    )
                },
                |simplex| {
                    let nm = ArgminNelderMead::new(simplex)
                        .with_sd_tolerance(0.0)
                        .unwrap();
                    black_box(
                        ArgminExecutor::new(
                            ArgminProblem::new(rosenbrock, rosenbrock_gradient),
                            nm,
                        )
                        .configure(|s| s.max_iters(MAX_ITERS))
                        .run()
                        .unwrap(),
                    )
                },
                BatchSize::SmallInput,
            )
        });

        g.finish();
    }
}

fn bench_cobyla(c: &mut Criterion) {
    for (case, name) in competitor_bench::cobyla::CASES {
        case.verify(&case.solve_basin());
        case.verify(&case.solve_reference());
        let mut group = c.benchmark_group(format!("cobyla_{name}"));
        group.bench_function("basin", |b| {
            b.iter(|| black_box(black_box(case).solve_basin()))
        });
        group.bench_function("cobyla", |b| {
            b.iter(|| black_box(black_box(case).solve_reference()))
        });
        group.finish();
    }
}

criterion_group!(benches, bench_gd, bench_nm, bench_cobyla);
criterion_main!(benches);
