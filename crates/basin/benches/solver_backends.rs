//! Backend axis (axis 1 of the bench plan): fix the solver + problem, vary
//! only the linear-algebra backend, isolating the cost of the LA layer.
//!
//! A *curated* set of (solver, problem) cases chosen so backend coverage
//! *narrows* as the solver's linear-algebra needs grow — and for different
//! reasons. Every case scales in the problem size `n`, so each renders as a
//! time-vs-`n` curve, one line per backend. Each group is named
//! `{solver}_{problem}_n{N}` with one `bench_function` per backend it supports:
//!
//! | solver        | problem          | backends         | why a backend is absent  |
//! |---------------|------------------|------------------|--------------------------|
//! | gradient desc | Rosenbrock       | vec/nalg/nd/faer | vector tier — all four   |
//! | Nelder–Mead   | Ackley           | vec/nalg/nd/faer | vector tier — all four   |
//! | L-Bfgs        | Styblinski–Tang  | vec/nalg/nd/faer | compact form, all four   |
//! | Bfgs          | Levy             | vec/nalg/faer    | ndarray lacks the dense  |
//! |               |                  |                  | rank-1 update + identity |
//! | CMA-ES        | Rastrigin        | vec/nalg/faer    | ndarray lacks the        |
//! |               |                  |                  | symmetric eigensolver    |
//! | Levenberg–M.  | sparse least-sq. | nalgebra/faer    | only these two carry     |
//! | Gauss–Newton  | sparse least-sq. | nalgebra/faer    | sparse matrices          |
//!
//! The least-squares cases use a scalable sparse `SparseLeastSquares` fixture
//! (the dense residual problems — Rosenbrock, Powell, Booth — are all fixed
//! size, so they can't drive an `n`-scaling chart).
//!
//! A *fixed* iteration budget with no tolerance criterion (`MAX_ITERS`) means
//! every backend does identical algorithmic work, so the ratio across a group
//! is pure per-iteration backend cost. The least-squares cases and CMA-ES
//! *converge before* the budget, so there `MAX_ITERS` is only a cap and the
//! comparison is per-solve backend cost. Problem + start-state construction is
//! charged to `iter_batched` setup, not the timed routine.
//!
//! Run with
//! `cargo bench --features nalgebra,ndarray,faer --bench solver_backends`.

use std::hint::black_box;
use std::time::Duration;

use basin::problems::{Ackley, Levy, Rastrigin, Rosenbrock, SparseLeastSquares, StyblinskiTang};
use basin::{
    BasicPopulationState, BasicSimplexState, BasicState, Bfgs, CmaEs, DenseMatrix, Executor,
    GaussNewton, GradientDescent, LbfgsState, Lbfgsb, LevenbergMarquardt, MoreThuente, NelderMead,
    QuasiNewtonState,
};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};

use faer::sparse::{SparseColMat, Triplet};
use faer::{Col, Mat};
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::{CooMatrix, CscMatrix};
use ndarray::Array1;

/// Fixed budget, no early stop — identical work across backends (a cap for the
/// solvers that converge sooner; see the module docs).
const MAX_ITERS: u64 = 200;

/// Problem sizes. Larger `n` surfaces the backends' vectorization differences;
/// `n = 2` is the classic small case. Log-spaced so the per-`n` curves render
/// cleanly on the log–log chart.
const DIMS: [usize; 5] = [2, 6, 20, 60, 200];

/// CMA-ES sizes. Kept small: each generation does λ cost-evals plus an O(n³)
/// eigendecomposition, so high `n` is dominated by the eigensolve.
const CMA_DIMS: [usize; 5] = [2, 4, 8, 16, 32];

/// Classic Rosenbrock start, extended to `n` dims by repeating `(−1.2, 1.0)`
/// (matches `competitor-bench`'s `gd_nm.rs`); also the start for the
/// least-squares Rosenbrock cases.
fn rosenbrock_start(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
        .collect()
}

/// Emit one criterion contestant: a full solve to `MAX_ITERS` on `$prob`
/// (the backend-specific problem type), building the start state from
/// `$setup` and wrapping it in `$state`. `$solver` is reconstructed each
/// iteration (cheap, and identical across backends).
macro_rules! contestant {
    ($g:expr_2021, $label:literal, $setup:expr_2021, $prob:ty, $solver:expr_2021, $state:expr_2021 $(,)?) => {
        $g.bench_function(BenchmarkId::from_parameter($label), |b| {
            b.iter_batched(
                $setup,
                |x0| {
                    black_box(
                        Executor::new(<$prob>::default(), $solver, $state(x0))
                            .max_iter(MAX_ITERS)
                            .run(),
                    )
                },
                BatchSize::SmallInput,
            )
        });
    };
}

/// Run `contestant!` on all four backends for problem `$Prob<V>`, using the
/// canonical per-backend start-vector constructors. For vector-tier solvers
/// that share one state constructor across backends.
macro_rules! backends_all4 {
    ($g:expr_2021, $Prob:ident, $start:expr_2021, $n:expr_2021, $solver:expr_2021, $state:expr_2021 $(,)?) => {{
        let start = $start;
        contestant!(
            $g,
            "vec",
            || start.clone(),
            $Prob<Vec<f64>>,
            $solver,
            $state
        );
        contestant!(
            $g,
            "nalgebra",
            || DVector::from_vec(start.clone()),
            $Prob<DVector<f64>>,
            $solver,
            $state,
        );
        contestant!(
            $g,
            "ndarray",
            || Array1::from_vec(start.clone()),
            $Prob<Array1<f64>>,
            $solver,
            $state,
        );
        contestant!(
            $g,
            "faer",
            || Col::<f64>::from_fn($n, |i| start[i]),
            $Prob<Col<f64>>,
            $solver,
            $state,
        );
    }};
}

/// Scalable sparse least-squares design for `n` parameters: an `n×n` identity
/// block stacked on `n−1` adjacent-pair coupling rows (`m = 2n−1` residuals,
/// `3n−2` nonzeros). With `b = A·x*` for `x*ᵢ = i+1` the minimum is at `x*`
/// with zero residual, and the identity block keeps `A` full column rank so
/// `JᵀJ` is SPD — Gauss-Newton and LM both converge.
fn sparse_pattern(n: usize) -> (Vec<(usize, usize, f64)>, Vec<f64>) {
    let xstar: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let mut entries = Vec::with_capacity(3 * n - 2);
    let mut b = vec![0.0; 2 * n - 1];
    for i in 0..n {
        entries.push((i, i, 1.0));
        b[i] = xstar[i];
    }
    for i in 0..n - 1 {
        entries.push((n + i, i, 1.0));
        entries.push((n + i, i + 1, 1.0));
        b[n + i] = xstar[i] + xstar[i + 1];
    }
    (entries, b)
}

type NalgebraSparseLsq = SparseLeastSquares<CscMatrix<f64>, DVector<f64>>;
type FaerSparseLsq = SparseLeastSquares<SparseColMat<usize, f64>, Col<f64>>;

/// nalgebra-sparse fixture + zero start.
fn sparse_lsq_nalgebra(n: usize) -> (NalgebraSparseLsq, DVector<f64>) {
    let (entries, b) = sparse_pattern(n);
    let mut coo = CooMatrix::<f64>::new(2 * n - 1, n);
    for (i, j, v) in entries {
        coo.push(i, j, v);
    }
    let problem = SparseLeastSquares::new(CscMatrix::from(&coo), DVector::from_vec(b));
    (problem, DVector::zeros(n))
}

/// faer-sparse fixture + zero start (same pattern as the nalgebra one).
fn sparse_lsq_faer(n: usize) -> (FaerSparseLsq, Col<f64>) {
    let (entries, b) = sparse_pattern(n);
    let triplets: Vec<Triplet<usize, usize, f64>> = entries
        .into_iter()
        .map(|(i, j, v)| Triplet::new(i, j, v))
        .collect();
    let a = SparseColMat::try_new_from_triplets(2 * n - 1, n, &triplets)
        .expect("sparse least-squares triplets valid");
    let problem = SparseLeastSquares::new(a, Col::from_fn(b.len(), |i| b[i]));
    (problem, Col::zeros(n))
}

/// Gradient descent (More-Thuente line search) on Rosenbrock — all four
/// backends, vector tier only.
fn bench_gd(c: &mut Criterion) {
    for n in DIMS {
        let mut g = c.benchmark_group(format!("gd_rosenbrock_n{n}"));
        backends_all4!(
            g,
            Rosenbrock,
            rosenbrock_start(n),
            n,
            GradientDescent::with_line_search(MoreThuente::new()),
            BasicState::new,
        );
        g.finish();
    }
}

/// Nelder–Mead (standard simplex) on multimodal Ackley — all four backends,
/// vector tier only, derivative-free.
fn bench_nm(c: &mut Criterion) {
    for n in DIMS {
        let mut g = c.benchmark_group(format!("nm_ackley_n{n}"));
        backends_all4!(
            g,
            Ackley,
            vec![2.0; n],
            n,
            NelderMead::standard(),
            BasicSimplexState::new,
        );
        g.finish();
    }
}

/// Unconstrained L-Bfgs on multimodal Styblinski–Tang — all four backends,
/// compact form (no dense matrix).
fn bench_lbfgs(c: &mut Criterion) {
    for n in DIMS {
        let mut g = c.benchmark_group(format!("lbfgs_styblinski_n{n}"));
        backends_all4!(
            g,
            StyblinskiTang,
            vec![0.0; n],
            n,
            Lbfgsb::new().unbounded(),
            |x0| LbfgsState::new(x0, 10),
        );
        g.finish();
    }
}

/// Bfgs on multimodal Levy — dense O(n²) rank-2 inverse-Hessian update. No
/// ndarray: `Array2` implements neither `GeneralRankOneUpdate` nor
/// `MatrixIdentity`, so that pairing is a compile error. The dense matrix type
/// pairs with each vector backend (`DenseMatrix` / `DMatrix` / `Mat`).
fn bench_bfgs(c: &mut Criterion) {
    for n in DIMS {
        let mut g = c.benchmark_group(format!("bfgs_levy_n{n}"));
        let start = vec![5.0; n];
        contestant!(
            g,
            "vec",
            || start.clone(),
            Levy<Vec<f64>>,
            Bfgs::new(),
            QuasiNewtonState::<Vec<f64>, DenseMatrix>::new,
        );
        contestant!(
            g,
            "nalgebra",
            || DVector::from_vec(start.clone()),
            Levy<DVector<f64>>,
            Bfgs::new(),
            QuasiNewtonState::<DVector<f64>, DMatrix<f64>>::new,
        );
        contestant!(
            g,
            "faer",
            || Col::<f64>::from_fn(n, |i| start[i]),
            Levy<Col<f64>>,
            Bfgs::new(),
            QuasiNewtonState::<Col<f64>, Mat<f64>>::new,
        );
        g.finish();
    }
}

/// CMA-ES on multimodal Rastrigin — derivative-free, population-based, with a
/// per-generation symmetric eigendecomposition. No ndarray: `Array2` lacks
/// `SymmetricEigen` / `RankOneUpdate`. `Vec<f64>` works via the hand-rolled
/// cyclic-Jacobi eigensolver. A fixed `seed` makes the sampling identical
/// across backends, so the comparison is pure per-iteration backend cost.
fn bench_cmaes(c: &mut Criterion) {
    for n in CMA_DIMS {
        let mut g = c.benchmark_group(format!("cmaes_rastrigin_n{n}"));
        // In-domain start away from the global optimum at the origin.
        let m0 = vec![3.0; n];
        // λ is backend-independent; match the solver's internal default in the
        // population state so the contract holds.
        let lambda = CmaEs::<Vec<f64>, DenseMatrix>::default_lambda(n);

        contestant!(
            g,
            "vec",
            || m0.clone(),
            Rastrigin<Vec<f64>>,
            CmaEs::<Vec<f64>, DenseMatrix>::new(m0.clone(), 0.3, 42),
            |_x0| BasicPopulationState::<Vec<f64>>::with_size(lambda),
        );

        let m0n = DVector::from_vec(m0.clone());
        contestant!(
            g,
            "nalgebra",
            || m0n.clone(),
            Rastrigin<DVector<f64>>,
            CmaEs::<DVector<f64>, DMatrix<f64>>::new(m0n.clone(), 0.3, 42),
            |_x0| BasicPopulationState::<DVector<f64>>::with_size(lambda),
        );

        let m0f = Col::<f64>::from_fn(n, |i| m0[i]);
        contestant!(
            g,
            "faer",
            || m0f.clone(),
            Rastrigin<Col<f64>>,
            CmaEs::<Col<f64>, Mat<f64>>::new(m0f.clone(), 0.3, 42),
            |_x0| BasicPopulationState::<Col<f64>>::with_size(lambda),
        );
        g.finish();
    }
}

/// Levenberg–Marquardt on a scalable sparse least-squares problem — damped
/// normal equations with a sparse `JᵀJ` and sparse Cholesky. Only nalgebra and
/// faer carry sparse matrices, so Vec/ndarray are out by construction. The
/// fixture is rebuilt in `iter_batched` setup, off the timed path.
fn bench_lm(c: &mut Criterion) {
    for n in DIMS {
        let mut g = c.benchmark_group(format!("lm_sparselsq_n{n}"));
        g.bench_function(BenchmarkId::from_parameter("nalgebra"), |b| {
            b.iter_batched(
                || sparse_lsq_nalgebra(n),
                |(p, x0)| {
                    black_box(
                        Executor::new(p, LevenbergMarquardt::new(), BasicState::new(x0))
                            .max_iter(MAX_ITERS)
                            .run(),
                    )
                },
                BatchSize::SmallInput,
            )
        });
        g.bench_function(BenchmarkId::from_parameter("faer"), |b| {
            b.iter_batched(
                || sparse_lsq_faer(n),
                |(p, x0)| {
                    black_box(
                        Executor::new(p, LevenbergMarquardt::new(), BasicState::new(x0))
                            .max_iter(MAX_ITERS)
                            .run(),
                    )
                },
                BatchSize::SmallInput,
            )
        });
        g.finish();
    }
}

/// Gauss–Newton on the same scalable sparse least-squares problem — undamped,
/// but the design is full column rank and zero-residual so `JᵀJ` stays SPD and
/// the solver converges. nalgebra + faer only (sparse matrices).
fn bench_gn(c: &mut Criterion) {
    for n in DIMS {
        let mut g = c.benchmark_group(format!("gn_sparselsq_n{n}"));
        g.bench_function(BenchmarkId::from_parameter("nalgebra"), |b| {
            b.iter_batched(
                || sparse_lsq_nalgebra(n),
                |(p, x0)| {
                    black_box(
                        Executor::new(p, GaussNewton::new(), BasicState::new(x0))
                            .max_iter(MAX_ITERS)
                            .run(),
                    )
                },
                BatchSize::SmallInput,
            )
        });
        g.bench_function(BenchmarkId::from_parameter("faer"), |b| {
            b.iter_batched(
                || sparse_lsq_faer(n),
                |(p, x0)| {
                    black_box(
                        Executor::new(p, GaussNewton::new(), BasicState::new(x0))
                            .max_iter(MAX_ITERS)
                            .run(),
                    )
                },
                BatchSize::SmallInput,
            )
        });
        g.finish();
    }
}

/// Trimmed criterion budget: 30 samples × 2 s measurement (default is 100 × 5 s)
/// with 1 s warm-up (default 3 s). Cuts per-bench overhead by ~3× without
/// meaningfully widening the mean's confidence interval — the chart consumes
/// the mean only, and the 5-dim × {2..4}-backend grid (≈110 contestants)
/// otherwise pushes total wall-clock past 30 min on a 12-core desktop.
fn bench_config() -> Criterion {
    Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2))
}

criterion_group! {
    name = benches;
    config = bench_config();
    targets = bench_gd, bench_nm, bench_lbfgs, bench_bfgs, bench_cmaes, bench_lm, bench_gn
}
criterion_main!(benches);
