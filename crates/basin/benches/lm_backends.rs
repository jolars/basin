//! Backend microbenchmarks for the Levenberg-Marquardt hot path.
//!
//! Matrix kernels use identical deterministic inputs across backends. QR
//! measurements separate preparation from undamped and regularized solves,
//! covering the tiny exponential-fit shape, a tall Jacobian, and a larger
//! parameter count. End-to-end solves also capture solver and callback work;
//! use the competitor harness's verification binaries to compare trajectories.
//!
//! Run with
//! `cargo bench --features nalgebra,faer,problems --bench lm_backends`.

use std::hint::black_box;

use basin::problems::{ExponentialFit, PowellSingular};
use basin::{
    DenseMatrix, Executor, FactorizePivotedQr, GramMatrix, LevenbergMarquardt,
    LinearSolveSpd, MatTransposeVec, PointState, RegularizedQrSolve,
};
use criterion::{
    BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main,
};

use crate::backend_aliases::faer::{Col, Mat};
use crate::backend_aliases::nalgebra::{DMatrix, DVector};

/// splitmix64-style deterministic pseudo-random in `[-0.5, 0.5)`, so the
/// nalgebra and faer inputs are bit-identical (fair timing, same
/// conditioning).
fn rng(i: u64) -> f64 {
    let mut x = i.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}

/// Column-major `m × n` Jacobian data: entry `(i, j)` at flat index
/// `j * m + i`.
fn jac_data(m: usize, n: usize) -> Vec<f64> {
    (0..m * n).map(|k| rng(k as u64)).collect()
}

/// Length-`m` residual vector.
fn vec_data(m: usize, seed: u64) -> Vec<f64> {
    (0..m).map(|i| rng(seed.wrapping_add(i as u64))).collect()
}

/// Column-major symmetric, diagonally-dominant (hence SPD) `n × n`
/// matrix, mirroring a damped normal-equations matrix `JᵀJ + μ·D`.
fn spd_data(n: usize) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for j in 0..n {
        for i in j..n {
            let v = rng((i * n + j) as u64);
            a[i + j * n] = v;
            a[j + i * n] = v;
        }
        // Diagonal dominance ⇒ positive definite.
        a[j + j * n] += n as f64;
    }
    a
}

/// The three Jacobian shapes from eunoia's corpus (issue #9): `m`
/// residuals, `n = 5·n_sets` parameters, so the normal equations are
/// `n × n ≤ 30 × 30`.
const SHAPES: [(usize, usize); 3] = [(4, 10), (16, 20), (64, 30)];

fn bench_qr(c: &mut Criterion) {
    for preparation in [true, false] {
        let mut group = c.benchmark_group(if preparation {
            "lm_qr_prepare"
        } else {
            "lm_qr_solve"
        });
        for (m, n) in [(10, 2), (1000, 2), (64, 30)] {
            let data = jac_data(m, n);
            let rhs = vec_data(m, 1 << 20);
            macro_rules! backend {
                ($name:literal, $vector_type:ty, $matrix:expr, $vector:expr) => {{
                    let matrix = $matrix;
                    let rhs = $vector;
                    if preparation {
                        group.bench_function(
                            BenchmarkId::new($name, format!("{m}x{n}")),
                            |b| {
                                b.iter(|| {
                                    black_box(
                                        black_box(&matrix)
                                            .factorize_pivoted_qr(black_box(
                                                &rhs,
                                            ))
                                            .unwrap(),
                                    )
                                })
                            },
                        );
                    } else {
                        let factor = matrix.factorize_pivoted_qr(&rhs).unwrap();
                        let diagonal: $vector_type =
                            factor.column_norms_squared();
                        for mu in [0., 1e-3] {
                            group.bench_function(
                                BenchmarkId::new(
                                    $name,
                                    format!("{m}x{n}_mu{mu}"),
                                ),
                                |b| {
                                    b.iter(|| {
                                        black_box(
                                            factor
                                                .solve_regularized(
                                                    black_box(mu),
                                                    black_box(&diagonal),
                                                    None,
                                                )
                                                .unwrap(),
                                        )
                                    })
                                },
                            );
                        }
                    }
                }};
            }
            backend!(
                "vec",
                Vec<f64>,
                DenseMatrix::from_fn(m, n, |i, j| data[j * m + i]),
                rhs.clone()
            );
            backend!(
                "nalgebra",
                DVector<f64>,
                DMatrix::from_column_slice(m, n, &data),
                DVector::from_column_slice(&rhs)
            );
            backend!(
                "faer",
                Col<f64>,
                Mat::from_fn(m, n, |i, j| data[j * m + i]),
                Col::from_fn(m, |i| rhs[i])
            );
        }
        group.finish();
    }
}

fn bench_gram(c: &mut Criterion) {
    let mut g = c.benchmark_group("lm_gram");
    for (m, n) in SHAPES {
        let label = format!("{m}x{n}");
        let data = jac_data(m, n);

        let jn = DMatrix::from_column_slice(m, n, &data);
        g.bench_with_input(
            BenchmarkId::new("nalgebra", &label),
            &jn,
            |b, j| b.iter(|| black_box(j.gram())),
        );

        let jf = Mat::from_fn(m, n, |i, j| data[i + j * m]);
        g.bench_with_input(BenchmarkId::new("faer", &label), &jf, |b, j| {
            b.iter(|| black_box(j.gram()))
        });
    }
    g.finish();
}

fn bench_mat_transpose_vec(c: &mut Criterion) {
    let mut g = c.benchmark_group("lm_jt_r");
    for (m, n) in SHAPES {
        let label = format!("{m}x{n}");
        let data = jac_data(m, n);
        let rdata = vec_data(m, 1 << 20);

        let jn = DMatrix::from_column_slice(m, n, &data);
        let rn = DVector::from_column_slice(&rdata);
        g.bench_with_input(
            BenchmarkId::new("nalgebra", &label),
            &(jn, rn),
            |b, (j, r)| b.iter(|| black_box(j.mat_transpose_vec(r))),
        );

        let jf = Mat::from_fn(m, n, |i, j| data[i + j * m]);
        let rf = Col::from_fn(m, |i| rdata[i]);
        g.bench_with_input(
            BenchmarkId::new("faer", &label),
            &(jf, rf),
            |b, (j, r)| b.iter(|| black_box(j.mat_transpose_vec(r))),
        );
    }
    g.finish();
}

fn bench_solve_spd(c: &mut Criterion) {
    let mut g = c.benchmark_group("lm_solve_spd");
    for (_, n) in SHAPES {
        let label = format!("{n}x{n}");
        let adata = spd_data(n);
        let bdata = vec_data(n, 1 << 21);

        let an = DMatrix::from_column_slice(n, n, &adata);
        let bn = DVector::from_column_slice(&bdata);
        g.bench_with_input(
            BenchmarkId::new("nalgebra", &label),
            &(an, bn),
            |b, (a, rhs)| b.iter(|| black_box(a.solve_spd(rhs).unwrap())),
        );

        let af = Mat::from_fn(n, n, |i, j| adata[i + j * n]);
        let bf = Col::from_fn(n, |i| bdata[i]);
        g.bench_with_input(
            BenchmarkId::new("faer", &label),
            &(af, bf),
            |b, (a, rhs)| b.iter(|| black_box(a.solve_spd(rhs).unwrap())),
        );
    }
    g.finish();
}

fn bench_full_solve(c: &mut Criterion) {
    let mut g = c.benchmark_group("lm_full_solve");

    // ExponentialFit (n = 2): the poorly-scaled fit from the LM tests.
    g.bench_function(BenchmarkId::new("nalgebra", "exp_fit"), |b| {
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
                Executor::new(p, LevenbergMarquardt::new(), PointState::new(x0))
                    .max_iter(200)
                    .run()
            },
            BatchSize::SmallInput,
        )
    });
    g.bench_function(BenchmarkId::new("faer", "exp_fit"), |b| {
        b.iter_batched(
            || {
                (
                    ExponentialFit::<Col<f64>>::sampled(1.0e5, -1.0, 10, 0.4),
                    Col::from_fn(2, |i| if i == 0 { 5.0e4 } else { -0.3 }),
                )
            },
            |(p, x0)| {
                Executor::new(p, LevenbergMarquardt::new(), PointState::new(x0))
                    .max_iter(200)
                    .run()
            },
            BatchSize::SmallInput,
        )
    });

    // PowellSingular (n = 4): rank-deficient at the optimum, exercises
    // the damping path.
    g.bench_function(BenchmarkId::new("nalgebra", "powell"), |b| {
        b.iter_batched(
            || {
                (
                    PowellSingular::<DVector<f64>>::new(),
                    DVector::from_vec(vec![3.0, -1.0, 0.0, 1.0]),
                )
            },
            |(p, x0)| {
                Executor::new(p, LevenbergMarquardt::new(), PointState::new(x0))
                    .max_iter(200)
                    .run()
            },
            BatchSize::SmallInput,
        )
    });
    g.bench_function(BenchmarkId::new("faer", "powell"), |b| {
        b.iter_batched(
            || {
                (
                    PowellSingular::<Col<f64>>::new(),
                    Col::from_fn(4, |i| [3.0, -1.0, 0.0, 1.0][i]),
                )
            },
            |(p, x0)| {
                Executor::new(p, LevenbergMarquardt::new(), PointState::new(x0))
                    .max_iter(200)
                    .run()
            },
            BatchSize::SmallInput,
        )
    });

    g.finish();
}

criterion_group!(
    benches,
    bench_gram,
    bench_mat_transpose_vec,
    bench_solve_spd,
    bench_qr,
    bench_full_solve
);
criterion_main!(benches);

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
