//! Serial-vs-parallel scaling of basin's population solvers (CMA-ES, DE,
//! Random Search) when the `parallel` feature fans each generation's fitness
//! evaluation across the rayon pool.
//!
//! # Why this exists
//!
//! basin's `parallel` feature is a *compile-time* switch with no runtime
//! threads knob—thread count follows rayon's global pool
//! (`RAYON_NUM_THREADS`, default = all cores). So you cannot put a "serial"
//! and a "parallel" contestant side by side in one criterion run. Instead the
//! comparison is **two runs of this single `parallel`-built binary**, differing
//! only by the rayon pool size, compared via criterion baselines:
//!
//! ```text
//! # serial baseline: one rayon worker (≈ the feature-off path, minus the
//! # plain-map vs par-iter overhead)
//! RAYON_NUM_THREADS=1 cargo bench -p competitor-bench --features parallel \
//!     --bench population_scaling -- --save-baseline serial
//!
//! # parallel: all cores, compared against the saved serial baseline
//! cargo bench -p competitor-bench --features parallel \
//!     --bench population_scaling -- --baseline serial
//! ```
//!
//! Criterion then reports the per-benchmark speedup (`change: -NN%`). The
//! benchmark IDs are stable across the two runs (they encode only the solver
//! and problem size, never the thread count), which is what lets the baseline
//! comparison line up.
//!
//! # The objective
//!
//! A sphere wrapped in a fixed busy-compute loop ([`ExpensiveSphere`]) so one
//! evaluation costs ~microseconds—the regime where fanning a generation
//! across cores pays for the rayon overhead. For near-free objectives the
//! batch overhead dominates and parallelism is a net loss; that crossover is
//! the whole reason the feature is opt-in, so it is worth eyeballing by
//! dialing `WORK` down to a small value.

use std::hint::black_box;
use std::sync::Once;

use basin::{
    BasicPopulationState, BoxConstraints, CmaEs, CmaEsState, CostFunction, De, DenseMatrix,
    Executor, RandomSearch,
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

/// Cost per evaluation, in busy-loop iterations. ~thousands ≈ a few µs per
/// `cost` call. Turn this down toward 0 to watch parallelism stop paying off.
const WORK: u64 = 4_000;
/// Generations to run. Fixed budget, no early stop, so each solve does exactly
/// `λ·(GENERATIONS+1)` evaluations and the timing is a clean per-generation
/// comparison.
const GENERATIONS: u64 = 30;

/// Sphere with a deliberately expensive, data-dependent inner loop and a box
/// constraint (so it drives the bounded DE and Random Search as well as the
/// unconstrained CMA-ES). The loop result feeds the cost at a negligible scale
/// so the optimizers still see a clean sphere and cannot hoist it away.
struct ExpensiveSphere {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl ExpensiveSphere {
    fn new(dim: usize) -> Self {
        Self {
            lower: vec![-5.0; dim],
            upper: vec![5.0; dim],
        }
    }
}

impl CostFunction for ExpensiveSphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        let mut acc = 0.0f64;
        for k in 0..WORK {
            acc += ((k as f64) * 1e-9 + x[k as usize % x.len()]).sin().abs();
        }
        let base: f64 = x.iter().map(|xi| xi * xi).sum();
        Ok(base + acc * 1e-12)
    }
}

impl BoxConstraints for ExpensiveSphere {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

/// Print the active rayon pool size once, so a glance at the bench output
/// confirms which configuration (serial vs all-cores) this run measured.
fn announce_threads() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "population_scaling: rayon pool = {} thread(s) \
             (set RAYON_NUM_THREADS=1 for the serial baseline)",
            rayon::current_num_threads()
        );
    });
}

fn bench_cma_es(c: &mut Criterion) {
    announce_threads();
    let mut g = c.benchmark_group("cma_es_expensive_sphere");
    for &(dim, lambda) in &[(10usize, 128usize), (20, 256)] {
        g.bench_function(
            BenchmarkId::from_parameter(format!("d{dim}_l{lambda}")),
            |b| {
                b.iter(|| {
                    black_box(
                        Executor::new(
                            ExpensiveSphere::new(dim),
                            CmaEs::<Vec<f64>, DenseMatrix>::new(7).with_lambda(lambda),
                            CmaEsState::<Vec<f64>, DenseMatrix>::new(vec![1.0; dim], 0.5),
                        )
                        .max_iter(GENERATIONS)
                        .run(),
                    )
                })
            },
        );
    }
    g.finish();
}

fn bench_de(c: &mut Criterion) {
    announce_threads();
    let mut g = c.benchmark_group("de_expensive_sphere");
    for &(dim, np) in &[(10usize, 128usize), (20, 256)] {
        g.bench_function(BenchmarkId::from_parameter(format!("d{dim}_np{np}")), |b| {
            b.iter(|| {
                black_box(
                    Executor::new(
                        ExpensiveSphere::new(dim),
                        De::<f64>::new(99).with_pop_size(np),
                        BasicPopulationState::<Vec<f64>>::with_size(np),
                    )
                    .max_iter(GENERATIONS)
                    .run(),
                )
            })
        });
    }
    g.finish();
}

fn bench_random_search(c: &mut Criterion) {
    announce_threads();
    let mut g = c.benchmark_group("random_search_expensive_sphere");
    for &(dim, lambda) in &[(10usize, 128usize), (20, 256)] {
        g.bench_function(
            BenchmarkId::from_parameter(format!("d{dim}_l{lambda}")),
            |b| {
                b.iter(|| {
                    black_box(
                        Executor::new(
                            ExpensiveSphere::new(dim),
                            RandomSearch::new(lambda, 2024),
                            BasicPopulationState::<Vec<f64>>::with_size(lambda),
                        )
                        .max_iter(GENERATIONS)
                        .run(),
                    )
                })
            },
        );
    }
    g.finish();
}

criterion_group!(benches, bench_cma_es, bench_de, bench_random_search);
criterion_main!(benches);
