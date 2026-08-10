//! Timing comparison (issue #9): basin's Levenberg-Marquardt vs the
//! `levenberg-marquardt` crate on identical small dense NLLS problems.
//!
//! Three contestants per problem:
//!   * `lm-crate`:      the external crate (nalgebra 0.34).
//!   * `basin/nalgebra`: basin on nalgebra 0.34 (same LA library as the
//!     lm crate, so this isolates *solver-loop* overhead).
//!   * `basin/faer`:    basin on faer 0.24 (adds the backend variable).
//!
//! All three use MINPACK gtol/ftol/xtol at `30·ε` (the lm crate's
//! default; basin configured to match) so they stop at comparable
//! points, confirmed by `src/bin/verify.rs`.
//!
//! Run: `cargo bench -p lm-bench`.

use std::hint::black_box;

use basin::problems::{ExponentialFit, PowellSingular};
use basin::{Executor, LevenbergMarquardt, NllsState};
use competitor_bench::{
    LM_DEFAULT_TOL, LmExponentialFit, LmPowellSingular, LmUnderDet, LmVarDim,
    UnderDet, VarDim, vardim_start,
};
use criterion::{
    BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main,
};
use faer::Col;
use nalgebra::DVector;

fn basin_lm<V, M>() -> LevenbergMarquardt<V, M> {
    LevenbergMarquardt::new()
        .with_tol_grad(0.0)
        .with_tol_grad_rel(LM_DEFAULT_TOL)
        .with_tol_cost_rel(LM_DEFAULT_TOL)
        .with_tol_step_rel(LM_DEFAULT_TOL)
}

fn bench_exp_fit(c: &mut Criterion) {
    let mut g = c.benchmark_group("exp_fit");

    g.bench_function(BenchmarkId::from_parameter("lm-crate"), |b| {
        b.iter_batched(
            || LmExponentialFit::sampled(1.0e5, -1.0, 10, 0.4, 5.0e4, -0.3),
            |p| {
                black_box(
                    levenberg_marquardt::LevenbergMarquardt::new().minimize(p),
                )
            },
            BatchSize::SmallInput,
        )
    });

    g.bench_function(BenchmarkId::from_parameter("basin/nalgebra"), |b| {
        b.iter_batched(
            || {
                (
                    ExponentialFit::<DVector<f64>>::sampled(
                        1.0e5, -1.0, 10, 0.4,
                    ),
                    DVector::from_vec(vec![5.0e4, -0.3]),
                )
            },
            |(p, x0)| {
                black_box(
                    Executor::new(p, basin_lm(), NllsState::new(x0))
                        .max_iter(200)
                        .run(),
                )
            },
            BatchSize::SmallInput,
        )
    });

    g.bench_function(BenchmarkId::from_parameter("basin/faer"), |b| {
        b.iter_batched(
            || {
                (
                    ExponentialFit::<Col<f64>>::sampled(1.0e5, -1.0, 10, 0.4),
                    Col::from_fn(2, |i| if i == 0 { 5.0e4 } else { -0.3 }),
                )
            },
            |(p, x0)| {
                black_box(
                    Executor::new(p, basin_lm(), NllsState::new(x0))
                        .max_iter(200)
                        .run(),
                )
            },
            BatchSize::SmallInput,
        )
    });

    g.finish();
}

fn bench_powell(c: &mut Criterion) {
    let mut g = c.benchmark_group("powell");

    g.bench_function(BenchmarkId::from_parameter("lm-crate"), |b| {
        b.iter_batched(
            || LmPowellSingular::new([3.0, -1.0, 0.0, 1.0]),
            |p| {
                black_box(
                    levenberg_marquardt::LevenbergMarquardt::new().minimize(p),
                )
            },
            BatchSize::SmallInput,
        )
    });

    g.bench_function(BenchmarkId::from_parameter("basin/nalgebra"), |b| {
        b.iter_batched(
            || {
                (
                    PowellSingular::<DVector<f64>>::new(),
                    DVector::from_vec(vec![3.0, -1.0, 0.0, 1.0]),
                )
            },
            |(p, x0)| {
                black_box(
                    Executor::new(p, basin_lm(), NllsState::new(x0))
                        .max_iter(200)
                        .run(),
                )
            },
            BatchSize::SmallInput,
        )
    });

    g.bench_function(BenchmarkId::from_parameter("basin/faer"), |b| {
        b.iter_batched(
            || {
                (
                    PowellSingular::<Col<f64>>::new(),
                    Col::from_fn(4, |i| [3.0, -1.0, 0.0, 1.0][i]),
                )
            },
            |(p, x0)| {
                black_box(
                    Executor::new(p, basin_lm(), NllsState::new(x0))
                        .max_iter(200)
                        .run(),
                )
            },
            BatchSize::SmallInput,
        )
    });

    g.finish();
}

/// Variably Dimensioned at eunoia's regime `n ∈ {10, 20, 30}`. The
/// well-conditioned, full-rank problem where all three converge cleanly,
/// so timings reflect per-iteration cost (and the known ~1.4× iteration
/// gap of basin vs the lm crate). basin/nalgebra and basin/faer run the
/// *same* iteration count, so their ratio is the pure faer penalty.
fn bench_vardim(c: &mut Criterion) {
    for n in [10usize, 20, 30] {
        let mut g = c.benchmark_group(format!("vardim_n{n}"));

        g.bench_function(BenchmarkId::from_parameter("lm-crate"), |b| {
            b.iter_batched(
                || LmVarDim::new(n),
                |p| {
                    black_box(
                        levenberg_marquardt::LevenbergMarquardt::new()
                            .minimize(p),
                    )
                },
                BatchSize::SmallInput,
            )
        });

        g.bench_function(BenchmarkId::from_parameter("basin/nalgebra"), |b| {
            b.iter_batched(
                || {
                    (
                        VarDim::<DVector<f64>>::new(n),
                        DVector::from_vec(vardim_start(n)),
                    )
                },
                |(p, x0)| {
                    black_box(
                        Executor::new(p, basin_lm(), NllsState::new(x0))
                            .max_iter(500)
                            .run(),
                    )
                },
                BatchSize::SmallInput,
            )
        });

        g.bench_function(BenchmarkId::from_parameter("basin/faer"), |b| {
            let start = vardim_start(n);
            b.iter_batched(
                || (VarDim::<Col<f64>>::new(n), Col::from_fn(n, |i| start[i])),
                |(p, x0)| {
                    black_box(
                        Executor::new(p, basin_lm(), NllsState::new(x0))
                            .max_iter(500)
                            .run(),
                    )
                },
                BatchSize::SmallInput,
            )
        });

        g.finish();
    }
}

/// Underdetermined trigonometric (`m < n`, rank-deficient `JᵀJ`) at
/// eunoia's regime (issue #10). Both solvers reach the same cost (see
/// `verify`), so on the sizes where iteration counts also match
/// (`(15,7)`, `(25,12)`) the timing ratio is per-iteration cost. This is
/// the regime where basin reformed `JᵀJ` on every rejected step, which
/// the lm crate avoids by factoring `J` once per outer iteration.
fn bench_underdet(c: &mut Criterion) {
    for (m, n) in [(7usize, 15usize), (15, 20), (12, 25)] {
        let mut g = c.benchmark_group(format!("underdet_n{n}_m{m}"));

        g.bench_function(BenchmarkId::from_parameter("lm-crate"), |b| {
            b.iter_batched(
                || LmUnderDet::new(m, n),
                |p| {
                    black_box(
                        levenberg_marquardt::LevenbergMarquardt::new()
                            .minimize(p),
                    )
                },
                BatchSize::SmallInput,
            )
        });

        g.bench_function(BenchmarkId::from_parameter("basin/nalgebra"), |b| {
            b.iter_batched(
                || {
                    let p = UnderDet::<DVector<f64>>::new(m, n);
                    let x0 = DVector::from_vec(p.start());
                    (p, x0)
                },
                |(p, x0)| {
                    black_box(
                        Executor::new(p, basin_lm(), NllsState::new(x0))
                            .max_iter(500)
                            .run(),
                    )
                },
                BatchSize::SmallInput,
            )
        });

        g.bench_function(BenchmarkId::from_parameter("basin/faer"), |b| {
            b.iter_batched(
                || {
                    let p = UnderDet::<Col<f64>>::new(m, n);
                    let start = p.start();
                    let x0 = Col::from_fn(n, |i| start[i]);
                    (p, x0)
                },
                |(p, x0)| {
                    black_box(
                        Executor::new(p, basin_lm(), NllsState::new(x0))
                            .max_iter(500)
                            .run(),
                    )
                },
                BatchSize::SmallInput,
            )
        });

        g.finish();
    }
}

criterion_group!(
    benches,
    bench_exp_fit,
    bench_powell,
    bench_vardim,
    bench_underdet
);
criterion_main!(benches);
