//! Timing comparison (issue #9): basin's Levenberg-Marquardt vs the
//! `levenberg-marquardt` crate on identical small dense NLLS problems.
//!
//! Three contestants per problem:
//!   * `lm-crate`:      the external crate (nalgebra 0.34).
//!   * `basin/nalgebra`: basin on nalgebra 0.34 (same LA library as the
//!     lm crate, so this isolates *solver-loop* overhead).
//!   * `basin/faer`:    basin on faer 0.24 (adds the backend variable).
//!
//! QR variants use the same Basin damping and stopping rules. Equal named
//! tolerances do not imply the same MINPACK convergence trajectory; use
//! `verify_lm_qr` to compare convergence and callback counts separately.
//!
//! Run: `cargo bench -p competitor-bench --bench compare`.

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
#[cfg(not(feature = "basin-latest"))]
use nalgebra::DVector as BasinDVector;
#[cfg(feature = "basin-latest")]
use nalgebra_latest::DVector as BasinDVector;

fn basin_lm<V, M>() -> LevenbergMarquardt<V, M> {
    LevenbergMarquardt::new()
        .with_absolute_gradient_tolerance(None)
        .with_gradient_orthogonality_tolerance(LM_DEFAULT_TOL)
        .with_relative_model_reduction_tolerance(LM_DEFAULT_TOL)
        .with_relative_step_tolerance(LM_DEFAULT_TOL)
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
                    ExponentialFit::<BasinDVector<f64>>::sampled(
                        1.0e5, -1.0, 10, 0.4,
                    ),
                    BasinDVector::from_vec(vec![5.0e4, -0.3]),
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

    g.bench_function(BenchmarkId::from_parameter("basin/nalgebra-qr"), |b| {
        b.iter_batched(
            || {
                (
                    ExponentialFit::<BasinDVector<f64>>::sampled(
                        1.0e5, -1.0, 10, 0.4,
                    ),
                    BasinDVector::from_vec(vec![5.0e4, -0.3]),
                )
            },
            |(p, x0)| {
                black_box(
                    Executor::new(
                        p,
                        basin_lm().with_pivoted_qr(),
                        NllsState::new(x0),
                    )
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

    g.bench_function(BenchmarkId::from_parameter("basin/faer-qr"), |b| {
        b.iter_batched(
            || {
                (
                    ExponentialFit::<Col<f64>>::sampled(1.0e5, -1.0, 10, 0.4),
                    Col::from_fn(2, |i| if i == 0 { 5.0e4 } else { -0.3 }),
                )
            },
            |(p, x0)| {
                black_box(
                    Executor::new(
                        p,
                        basin_lm().with_pivoted_qr(),
                        NllsState::new(x0),
                    )
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
                    PowellSingular::<BasinDVector<f64>>::new(),
                    BasinDVector::from_vec(vec![3.0, -1.0, 0.0, 1.0]),
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

    g.bench_function(BenchmarkId::from_parameter("basin/nalgebra-qr"), |b| {
        b.iter_batched(
            || {
                (
                    PowellSingular::<BasinDVector<f64>>::new(),
                    BasinDVector::from_vec(vec![3.0, -1.0, 0.0, 1.0]),
                )
            },
            |(p, x0)| {
                black_box(
                    Executor::new(
                        p,
                        basin_lm().with_pivoted_qr(),
                        NllsState::new(x0),
                    )
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

    g.bench_function(BenchmarkId::from_parameter("basin/faer-qr"), |b| {
        b.iter_batched(
            || {
                (
                    PowellSingular::<Col<f64>>::new(),
                    Col::from_fn(4, |i| [3.0, -1.0, 0.0, 1.0][i]),
                )
            },
            |(p, x0)| {
                black_box(
                    Executor::new(
                        p,
                        basin_lm().with_pivoted_qr(),
                        NllsState::new(x0),
                    )
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
                        VarDim::<BasinDVector<f64>>::new(n),
                        BasinDVector::from_vec(vardim_start(n)),
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

        g.bench_function(
            BenchmarkId::from_parameter("basin/nalgebra-qr"),
            |b| {
                b.iter_batched(
                    || {
                        (
                            VarDim::<BasinDVector<f64>>::new(n),
                            BasinDVector::from_vec(vardim_start(n)),
                        )
                    },
                    |(p, x0)| {
                        black_box(
                            Executor::new(
                                p,
                                basin_lm().with_pivoted_qr(),
                                NllsState::new(x0),
                            )
                            .max_iter(500)
                            .run(),
                        )
                    },
                    BatchSize::SmallInput,
                )
            },
        );

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

        g.bench_function(BenchmarkId::from_parameter("basin/faer-qr"), |b| {
            let start = vardim_start(n);
            b.iter_batched(
                || (VarDim::<Col<f64>>::new(n), Col::from_fn(n, |i| start[i])),
                |(p, x0)| {
                    black_box(
                        Executor::new(
                            p,
                            basin_lm().with_pivoted_qr(),
                            NllsState::new(x0),
                        )
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
                    let p = UnderDet::<BasinDVector<f64>>::new(m, n);
                    let x0 = BasinDVector::from_vec(p.start());
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

        g.bench_function(
            BenchmarkId::from_parameter("basin/nalgebra-qr"),
            |b| {
                b.iter_batched(
                    || {
                        let p = UnderDet::<BasinDVector<f64>>::new(m, n);
                        let x0 = BasinDVector::from_vec(p.start());
                        (p, x0)
                    },
                    |(p, x0)| {
                        black_box(
                            Executor::new(
                                p,
                                basin_lm().with_pivoted_qr(),
                                NllsState::new(x0),
                            )
                            .max_iter(500)
                            .run(),
                        )
                    },
                    BatchSize::SmallInput,
                )
            },
        );

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

        g.bench_function(BenchmarkId::from_parameter("basin/faer-qr"), |b| {
            b.iter_batched(
                || {
                    let p = UnderDet::<Col<f64>>::new(m, n);
                    let start = p.start();
                    let x0 = Col::from_fn(n, |i| start[i]);
                    (p, x0)
                },
                |(p, x0)| {
                    black_box(
                        Executor::new(
                            p,
                            basin_lm().with_pivoted_qr(),
                            NllsState::new(x0),
                        )
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

fn bench_qr_regularization(c: &mut Criterion) {
    use basin::{
        AddDiagonalVectorInPlace, DenseMatrix, FactorizePivotedQr, GramMatrix,
        LinearSolveSpd, MatTransposeVec, RegularizedQrSolve,
    };
    let a = DenseMatrix::from_fn(128, 8, |i, j| {
        ((i * 13 + j * 7) as f64).sin() + if i == j { 1. } else { 0. }
    });
    let b: Vec<f64> = (0..128).map(|i| (i as f64).cos()).collect();
    let qr = a.factorize_pivoted_qr(&b).unwrap();
    let d: Vec<f64> = qr.column_norms_squared();
    let gram = a.gram();
    let atb = a.mat_transpose_vec(&b);
    let mut group = c.benchmark_group("qr_regularization_128x8");
    group.bench_function("factorize", |bench| {
        bench.iter(|| black_box(a.factorize_pivoted_qr(black_box(&b)).unwrap()))
    });
    group.bench_function("qr_retry", |bench| {
        bench.iter(|| {
            black_box(qr.solve_regularized(black_box(1e-3), &d, None).unwrap())
        })
    });
    group.bench_function("cholesky_retry", |bench| {
        bench.iter(|| {
            let mut damped = gram.clone();
            let damping: Vec<_> = d.iter().map(|x| x * 1e-3).collect();
            damped.add_diagonal_vector_in_place(&damping);
            black_box(damped.solve_spd(black_box(&atb)).unwrap())
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_exp_fit,
    bench_powell,
    bench_vardim,
    bench_underdet,
    bench_qr_regularization
);
criterion_main!(benches);
