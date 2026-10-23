//! Uninstrumented solves and separate callback accounting for the published NM case.

use std::cell::Cell;

use basin::problems::{Rosenbrock, rosenbrock};
use basin::{
    Executor, IntoInitialSimplex, NelderMead, SimplexProgress, State,
    StepOutcome,
};
use gomez::nalgebra::{Dyn, IsContiguous, Storage, Vector};
use gomez::{Domain, Function, OptimizerDriver, Problem};

use crate::GomezProblem;

pub const TARGET: f64 = 1e-10;
pub const MAX_ITERS: u64 = 10_000;

pub fn start(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
        .collect()
}

pub fn initial_simplex(n: usize) -> Vec<Vec<f64>> {
    start(n).into_initial_simplex(0.05)
}

/// Derive additive offsets from the shared vertices, preserving their rounding.
pub fn initial_steps(n: usize) -> Vec<f64> {
    let simplex = initial_simplex(n);
    (0..n).map(|i| simplex[i + 1][i] - simplex[0][i]).collect()
}

pub fn gomez_solver<P: Problem<Field = f64>>(
    problem: &P,
    domain: &Domain<f64>,
) -> gomez::algo::NelderMead<P> {
    // Gomez copies the domain scale as additive simplex offsets at construction.
    // Derive them from Basin's vertices instead of its unit-step default.
    let domain = domain.clone().with_scale(initial_steps(domain.dim()));
    gomez::algo::NelderMead::new(problem, &domain)
}

#[derive(Debug)]
pub struct Summary {
    pub point: Vec<f64>,
    pub cost: f64,
    pub iterations: u64,
    pub objective_calls: u64,
    pub stop: String,
}

impl Summary {
    /// This verification evaluation is separate from the solver's callback count.
    pub fn verify(&self) -> bool {
        assert!(self.point.iter().all(|x| x.is_finite()));
        let checked = rosenbrock(&self.point);
        assert!(checked.is_finite());
        assert!((checked - self.cost).abs() <= 1e-12 * checked.abs().max(1.0));
        checked <= TARGET
    }
}

pub fn solve_basin(n: usize) -> Summary {
    let mut stepper = Executor::new(
        Rosenbrock::<Vec<f64>>::default(),
        NelderMead::new(),
        SimplexProgress::from_simplex(initial_simplex(n)),
    )
    .max_iter(MAX_ITERS)
    .into_stepper()
    .unwrap();
    let stop = loop {
        let outcome = stepper.step().unwrap();
        if (0.0..=TARGET).contains(&stepper.state().cost()) {
            break "target".to_owned();
        }
        if let StepOutcome::Stopped(reason) = outcome {
            break format!("{reason:?}");
        }
    };
    let state = stepper.state();
    Summary {
        point: state.param().clone(),
        cost: state.cost(),
        iterations: state.iter(),
        objective_calls: stepper.counts().cost_evals,
        stop,
    }
}

struct GomezCounted<const COUNT: bool> {
    inner: GomezProblem,
    calls: Cell<u64>,
}

impl<const COUNT: bool> Problem for GomezCounted<COUNT> {
    type Field = f64;
    fn domain(&self) -> Domain<f64> {
        self.inner.domain()
    }
}

impl<const COUNT: bool> Function for GomezCounted<COUNT> {
    fn apply<S>(&self, x: &Vector<f64, Dyn, S>) -> f64
    where
        S: Storage<f64, Dyn> + IsContiguous,
    {
        if COUNT {
            self.calls.set(self.calls.get() + 1);
        }
        self.inner.apply(x)
    }
}

/// Gomez starts from the same vertices as Basin, using standard coefficients.
/// Its subsequent rounding, inner-contraction tie rule, and collapse stop differ.
/// Disable counting for timing; verify with counting in a separate run.
pub fn solve_gomez<const COUNT: bool>(n: usize) -> Summary {
    let problem = GomezCounted::<COUNT> {
        inner: GomezProblem::new(rosenbrock, n),
        calls: Cell::new(0),
    };
    let mut optimizer = OptimizerDriver::builder(&problem)
        .with_initial(start(n))
        .with_algo(gomez_solver)
        .build();
    let mut iterations = 0;
    let mut last_cost = f64::NAN;
    let mut stop = "budget".to_owned();
    for _ in 0..MAX_ITERS {
        match optimizer.next() {
            Ok((_, cost)) => {
                iterations += 1;
                last_cost = cost;
                if (0.0..=TARGET).contains(&cost) {
                    stop = "target".to_owned();
                    break;
                }
            }
            Err(error) => {
                stop = format!("{error:?}");
                break;
            }
        }
    }
    Summary {
        point: optimizer.x().to_vec(),
        cost: last_cost,
        iterations,
        objective_calls: problem.calls.get(),
        stop,
    }
}
