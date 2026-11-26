//! The default serial population solvers must be reproducible and count cost
//! evaluations identically with and without the `parallel` feature.
//!
//! Each generation's candidates are evaluated with
//! [`Problem::cost_batch`](basin::Problem::cost_batch), which collects results
//! in input order and bumps `cost_evals` by the batch size on the calling
//! thread. So:
//!
//! - **Same seed → identical trajectory.** The downstream evaluation fixture
//!   also compares a seeded CMA-ES run with explicit threaded evaluation.
//! - **`cost_evals` matches a fixed per-solver formula.** DE, global-best PSO,
//!   and RandomSearch hit `λ · (iters + 1)`; CMA-ES adds one `f(m)` mean evaluation per
//!   generation, so it hits `(λ + 1) · (iters + 1)`. The constants below are
//!   feature-independent: CI runs this file with `--features parallel` *and*
//!   without, and both must hit the same numbers, proving the batched path
//!   counts exactly as the old per-candidate loop did.
//!
//! These run on the default `Vec<f64>` backend without optional linear
//! algebra features.

#![cfg(feature = "problems")]

use basin::problems::{RastriginBoxed, Rosenbrock};
use basin::{
    CmaEs, De, DenseMatrix, Executor, GlobalBestPso, OptimizationResult,
    PopulationProgress, RandomSearch, State,
};
use std::fmt::Debug;

/// Number of `next_iter` calls a `MaxIter(m)` run performs: termination is
/// checked before each iteration (including iter 0), so the solver completes
/// exactly `m` iterations on top of `init`. DE and RandomSearch evaluate `λ`
/// fresh candidates in `init` and `λ` again per iteration, so a run that is not
/// cut short by a solver-internal stop performs `λ · (m + 1)` cost evaluations.
/// (CMA-ES additionally evaluates the mean once per generation; see its test.)
fn expected_cost_evals(lambda: u64, max_iter: u64) -> u64 {
    lambda * (max_iter + 1)
}

#[test]
fn cma_es_reproducible_and_counts_match() {
    let m0 = vec![-1.0, 1.0];
    let lambda = CmaEs::<Vec<f64>, DenseMatrix>::default_lambda(2);
    // Small budget on Rosenbrock: well short of the TolX stop, so every run
    // performs the full `max_iter` iterations and the eval count is exact.
    let max_iter = 5;

    let run = || {
        Executor::new(
            Rosenbrock::<Vec<f64>>::new(),
            CmaEs::<Vec<f64>, DenseMatrix>::new(17, 0.3),
            PopulationProgress::<Vec<f64>>::from_point(m0.clone()),
        )
        .max_iter(max_iter)
        .run()
        .unwrap()
    };
    let (a, b) = (run(), run());

    assert_bit_identical(&a, &b);
    // CMA-ES additionally evaluates `f(m)` (the distribution mean, reported as
    // `param()`/`xfavorite`) once per generation, so its count is
    // `(λ + 1)·(iters + 1)`, not `λ·(iters + 1)` like the other population
    // solvers. Still feature-independent.
    assert_eq!(
        a.counts.cost_evals,
        (lambda as u64 + 1) * (max_iter + 1),
        "CMA-ES cost_evals must equal (λ+1)·(iters+1) regardless of `parallel`"
    );
}

#[test]
fn de_reproducible_and_counts_match() {
    let lambda = 16; // DE pop size, set explicitly so the count is predictable.
    let max_iter = 8;

    let run = || {
        Executor::new(
            RastriginBoxed::<Vec<f64>>::with_standard_bounds(4),
            De::<f64>::new(99).with_pop_size(lambda),
            PopulationProgress::<Vec<f64>>::empty(),
        )
        .max_iter(max_iter)
        .run()
        .unwrap()
    };
    let (a, b) = (run(), run());

    assert_bit_identical(&a, &b);
    assert_eq!(
        a.counts.cost_evals,
        expected_cost_evals(lambda as u64, max_iter),
        "DE cost_evals must equal pop·(iters+1) regardless of `parallel`"
    );
}

#[test]
fn global_best_pso_reproducible_and_counts_match() {
    let swarm_size = 16;
    let max_iter = 8;

    let run = || {
        Executor::new(
            RastriginBoxed::<Vec<f64>>::with_standard_bounds(4),
            GlobalBestPso::new(99).with_swarm_size(swarm_size),
            PopulationProgress::<Vec<f64>>::empty(),
        )
        .max_iter(max_iter)
        .run()
        .unwrap()
    };
    let (a, b) = (run(), run());

    assert_bit_identical(&a, &b);
    assert_eq!(
        a.counts.cost_evals,
        expected_cost_evals(swarm_size as u64, max_iter),
        "global-best PSO cost_evals must equal swarm·(iters+1) regardless of `parallel`"
    );
}

#[cfg(feature = "parallel")]
#[test]
fn de_parallel_opt_in_preserves_cost_tolerance() {
    let solver = De::new(99)
        .with_pop_size(16)
        .with_absolute_cost_change_tolerance(1e6)
        .with_parallel_evaluation();
    let result = Executor::new(
        RastriginBoxed::<Vec<f64>>::with_standard_bounds(4),
        solver,
        PopulationProgress::empty(),
    )
    .max_iter(8)
    .run()
    .unwrap();

    assert_eq!(result.report.code(), basin::TerminationCode::CostTolerance);
    assert_eq!(result.iter(), 1);
    assert_eq!(result.counts.cost_evals, 32);
}

#[test]
fn random_search_reproducible_and_counts_match() {
    let lambda = 20;
    let max_iter = 6;

    let run = || {
        Executor::new(
            RastriginBoxed::<Vec<f64>>::with_standard_bounds(3),
            RandomSearch::new(lambda, 2024),
            PopulationProgress::<Vec<f64>>::empty(),
        )
        .max_iter(max_iter)
        .run()
        .unwrap()
    };
    let (a, b) = (run(), run());

    assert_bit_identical(&a, &b);
    // RandomSearch keeps the elite (already costed) and evaluates `λ` fresh
    // draws per iteration plus `λ` in `init`, so the formula still holds.
    assert_eq!(
        a.counts.cost_evals,
        expected_cost_evals(lambda as u64, max_iter),
        "RandomSearch cost_evals must equal λ·(iters+1) regardless of `parallel`"
    );
}

fn assert_bit_identical<S: State>(
    a: &OptimizationResult<S>,
    b: &OptimizationResult<S>,
) where
    S::Param: PartialEq + Debug,
    S::Float: PartialEq + Debug,
{
    assert_eq!(
        a.current_record().unwrap().cost.unwrap(),
        b.current_record().unwrap().cost.unwrap(),
        "best cost differs across same-seed runs"
    );
    assert_eq!(
        a.param(),
        b.param(),
        "iterate differs across same-seed runs"
    );
    assert_eq!(
        a.counts.cost_evals, b.counts.cost_evals,
        "cost_evals differs across same-seed runs"
    );
}
