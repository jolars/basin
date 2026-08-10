//! Population solvers must be bit-reproducible and count cost evaluations
//! identically whether or not the `parallel` feature fans candidate
//! evaluation across the `rayon` pool.
//!
//! Each generation's candidates are evaluated with
//! [`Problem::cost_batch`](basin::Problem::cost_batch), which collects results
//! in input order and bumps `cost_evals` by the batch size on the calling
//! thread. So:
//!
//! - **Same seed → identical trajectory**, regardless of thread scheduling.
//!   This is the parallel analogue of numdiff's
//!   `gradient_is_bitwise_reproducible_across_calls`: it catches any
//!   nondeterminism a reordered parallel reduction could introduce.
//! - **`cost_evals` matches a fixed per-solver formula.** DE and RandomSearch
//!   hit `λ · (iters + 1)`; CMA-ES adds one `f(m)` mean evaluation per
//!   generation, so it hits `(λ + 1) · (iters + 1)`. The constants below are
//!   feature-independent: CI runs this file with `--features parallel` *and*
//!   without, and both must hit the same numbers, proving the batched path
//!   counts exactly as the old per-candidate loop did.
//!
//! These run on the default `Vec<f64>` backend so the file needs no LA
//! feature; the `parallel` matrix entry exercises the rayon path.

use basin::problems::{RastriginBoxed, Rosenbrock};
use basin::{
    BasicPopulationState, CmaEs, CmaEsState, De, DenseMatrix, Executor,
    OptimizationResult, RandomSearch, State,
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
            CmaEs::<Vec<f64>, DenseMatrix>::new(17),
            CmaEsState::<Vec<f64>, DenseMatrix>::new(m0.clone(), 0.3),
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
        a.cost_evals(),
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
            BasicPopulationState::<Vec<f64>>::with_size(lambda),
        )
        .max_iter(max_iter)
        .run()
        .unwrap()
    };
    let (a, b) = (run(), run());

    assert_bit_identical(&a, &b);
    assert_eq!(
        a.cost_evals(),
        expected_cost_evals(lambda as u64, max_iter),
        "DE cost_evals must equal pop·(iters+1) regardless of `parallel`"
    );
}

#[test]
fn random_search_reproducible_and_counts_match() {
    let lambda = 20;
    let max_iter = 6;

    let run = || {
        Executor::new(
            RastriginBoxed::<Vec<f64>>::with_standard_bounds(3),
            RandomSearch::new(lambda, 2024),
            BasicPopulationState::<Vec<f64>>::with_size(lambda),
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
        a.cost_evals(),
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
        a.cost(),
        b.cost(),
        "best cost differs across same-seed runs"
    );
    assert_eq!(
        a.param(),
        b.param(),
        "iterate differs across same-seed runs"
    );
    assert_eq!(
        a.cost_evals(),
        b.cost_evals(),
        "cost_evals differs across same-seed runs"
    );
}
