//! Standalone active CMA-ES on an ill-conditioned quadratic and multimodal Ackley.
//!
//! The reference is `cmaes` 0.2.2 with explicit negative recombination weights.
//! Both drivers stop at the same best-objective target or evaluation budget.
//! Basin also evaluates each mean and refreshes its eigendecomposition every
//! generation; the reference samples only and refreshes eigenpairs periodically.

use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use basin::{CmaEs, CostFunction, Executor, PopulationProgress, StepOutcome};

use crate::convergence::{Run, Summary};

#[cfg(not(feature = "basin-latest"))]
use nalgebra::{DMatrix as BasinMatrix, DVector as BasinVector};
#[cfg(feature = "basin-latest")]
use nalgebra_latest::{DMatrix as BasinMatrix, DVector as BasinVector};

pub const DIMENSION: usize = 40;
pub const POPULATION: usize = 16;
pub const SIGMA: f64 = 2.0;
pub const TARGET: f64 = 1e-8;
pub const MAX_EVALUATIONS: usize = 100_000;
pub const MAX_GENERATIONS: usize = 10_000;
pub const SEEDS: [u64; 5] = [0, 1, 2, 3, 4];

#[derive(Clone, Copy, Debug)]
pub enum Case {
    RotatedEllipsoid,
    Ackley,
}

impl Case {
    pub fn name(self) -> &'static str {
        match self {
            Self::RotatedEllipsoid => "rotated_ellipsoid",
            Self::Ackley => "ackley",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Library {
    Basin,
    Reference,
}

impl Library {
    pub fn name(self) -> &'static str {
        match self {
            Self::Basin => "basin",
            Self::Reference => "cmaes",
        }
    }
}

struct Workload {
    case: Case,
    rotation: Vec<Vec<f64>>,
    weights: Vec<f64>,
    calls: AtomicUsize,
}

impl Workload {
    fn new(case: Case, n: usize) -> Self {
        assert!(n >= 2);
        // An orthonormal DCT-II fixes the dense rotation independently of seeds
        // and library-specific random-number generators.
        let rotation = (0..n)
            .map(|i| {
                let scale = if i == 0 { 1.0 } else { 2.0 };
                (0..n)
                    .map(|j| {
                        (scale / n as f64).sqrt()
                            * (std::f64::consts::PI
                                * i as f64
                                * (j as f64 + 0.5)
                                / n as f64)
                                .cos()
                    })
                    .collect()
            })
            .collect();
        Self {
            case,
            rotation,
            weights: (0..n)
                .map(|i| 10_f64.powf(6.0 * i as f64 / (n - 1) as f64))
                .collect(),
            calls: AtomicUsize::new(0),
        }
    }

    fn cost(&self, x: &[f64]) -> f64 {
        match self.case {
            Case::Ackley => basin::problems::ackley(x),
            Case::RotatedEllipsoid => self
                .rotation
                .iter()
                .zip(&self.weights)
                .map(|(row, weight)| {
                    let y = row.iter().zip(x).map(|(q, x)| q * x).sum::<f64>();
                    weight * y * y
                })
                .sum(),
        }
    }

    fn evaluate(&self, x: &[f64]) -> f64 {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.cost(x)
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}

impl basin::Constraints for &Workload {
    type Set = basin::NoConstraints;
    fn constraints(&self) -> basin::NoConstraints {
        basin::NoConstraints::EMPTY
    }
}

impl CostFunction for &Workload {
    type Param = BasinVector<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<f64, Infallible> {
        Ok(self.evaluate(x.as_slice()))
    }
}

struct Outcome {
    point: Vec<f64>,
    cost: f64,
    generations: usize,
    status: String,
}

fn basin_solve(
    problem: &Workload,
    seed: u64,
    record: &mut impl FnMut(f64),
) -> Outcome {
    let solver = CmaEs::<BasinVector<f64>, BasinMatrix<f64>>::new(seed, SIGMA)
        .with_lambda(POPULATION)
        .with_absolute_distribution_size_tolerance(None);
    let state = PopulationProgress::from_point(BasinVector::from_element(
        problem.weights.len(),
        3.0,
    ));
    let mut stepper = Executor::new(problem, solver, state)
        .max_iter((MAX_GENERATIONS - 1) as u64)
        .into_stepper()
        .unwrap();
    let status = loop {
        let (_, cost) = stepper.state().best().unwrap();
        record(cost);
        if cost <= TARGET {
            break "TargetReached".into();
        }
        if problem.calls() >= MAX_EVALUATIONS {
            break "EvaluationBudget".into();
        }
        if let StepOutcome::Stopped(report) = stepper.step().unwrap() {
            record(stepper.state().best().unwrap().1);
            break format!("{:?}", report.code());
        }
    };
    assert_eq!(stepper.counts().cost_evals as usize, problem.calls());
    let (point, cost) = stepper.state().best().unwrap();
    Outcome {
        point: point.as_slice().to_vec(),
        cost,
        // Initialization samples the first population before any iteration.
        generations: stepper.iter() as usize + 1,
        status,
    }
}

fn reference_options(n: usize, seed: u64) -> ::cmaes::CMAESOptions {
    ::cmaes::CMAESOptions::new(vec![3.0; n], SIGMA)
        .population_size(POPULATION)
        .weights(::cmaes::Weights::Negative)
        .parallel_update(false)
        .seed(seed)
        .max_function_evals(MAX_EVALUATIONS)
        .max_generations(MAX_GENERATIONS)
        .tol_fun(0.0)
        .tol_fun_rel(0.0)
        .tol_fun_hist(0.0)
        .tol_x(0.0)
        // Keep optional stagnation checks beyond the common safety cap.
        .tol_stagnation(MAX_GENERATIONS + 1)
}

fn reference_solve(
    problem: &Workload,
    seed: u64,
    record: &mut impl FnMut(f64),
) -> Outcome {
    let mut solver = reference_options(problem.weights.len(), seed)
        .build(|x: &::cmaes::DVector<f64>| problem.evaluate(x.as_slice()))
        .unwrap();
    let status = loop {
        let termination = solver.next();
        let cost = solver
            .overall_best_individual()
            .map_or(f64::NAN, |best| best.value);
        record(cost);
        if let Some(termination) = termination {
            break format!("{:?}", termination.reasons);
        }
        if cost <= TARGET {
            break "TargetReached".into();
        }
    };
    assert_eq!(solver.function_evals(), problem.calls());
    let best = solver.overall_best_individual();
    Outcome {
        point: best.map_or_else(
            || vec![f64::NAN; problem.weights.len()],
            |best| best.point.as_slice().to_vec(),
        ),
        cost: best.map_or(f64::NAN, |best| best.value),
        generations: solver.generation(),
        status,
    }
}

/// Time one fresh solve; fixture setup and independent verification are untimed.
pub fn run(library: Library, case: Case, seed: u64) -> Run {
    run_logged(library, case, seed, true)
}

pub fn run_logged(
    library: Library,
    case: Case,
    seed: u64,
    tracing: bool,
) -> Run {
    run_sized(library, case, seed, DIMENSION, tracing)
}

fn run_sized(
    library: Library,
    case: Case,
    seed: u64,
    n: usize,
    tracing: bool,
) -> Run {
    let problem = Workload::new(case, n);
    let mut points = Vec::new();
    let start = Instant::now();
    let mut record = |cost: f64| {
        if tracing && cost.is_finite() {
            points.push((start.elapsed().as_nanos(), cost));
        }
    };
    let outcome = match library {
        Library::Basin => basin_solve(&problem, seed, &mut record),
        Library::Reference => reference_solve(&problem, seed, &mut record),
    };
    let elapsed = start.elapsed().as_nanos();
    if outcome.cost.is_finite() {
        points.push((elapsed, outcome.cost));
    }
    let cost = problem.cost(&outcome.point);
    let verified = outcome.point.iter().all(|x| x.is_finite())
        && cost.is_finite()
        && (0.0..=TARGET).contains(&cost)
        && (cost - outcome.cost).abs() <= 1e-12 * cost.abs().max(1.0);
    Run {
        points,
        violations: Vec::new(),
        summary: Summary {
            point: outcome.point,
            cost,
            violation: 0.0,
            iterations: Some(outcome.generations as u64),
            projected_gradient: None,
            objective_calls: problem.calls(),
            residual_calls: 0,
            jacobian_calls: 0,
            gradient_calls: 0,
            constraint_calls: 0,
            status: outcome.status,
            verified,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotated_fixture_has_the_declared_spectrum_and_minimum() {
        let problem = Workload::new(Case::RotatedEllipsoid, 4);
        assert_eq!(problem.cost(&[0.0; 4]), 0.0);
        for (i, row) in problem.rotation.iter().enumerate() {
            for (j, other) in problem.rotation.iter().enumerate() {
                let dot =
                    row.iter().zip(other).map(|(a, b)| a * b).sum::<f64>();
                assert!((dot - f64::from(i == j)).abs() < 1e-14);
            }
            assert!(
                (problem.cost(row) / problem.weights[i] - 1.0).abs() < 1e-12
            );
        }
        assert_eq!(problem.weights[3] / problem.weights[0], 1e6);
        assert!(problem.cost(&[f64::NAN; 4]).is_nan());
    }

    #[test]
    fn reference_explicitly_uses_active_serial_updates() {
        let options = reference_options(DIMENSION, 42);
        assert!(matches!(options.weights, ::cmaes::Weights::Negative));
        assert!(!options.parallel_update);
        assert_eq!(options.seed, Some(42));
        assert_eq!(options.population_size, POPULATION);
        assert_eq!(options.initial_step_size, SIGMA);
    }

    #[test]
    fn both_drivers_reach_the_target_and_count_all_objective_calls() {
        for case in [Case::RotatedEllipsoid, Case::Ackley] {
            for library in [Library::Basin, Library::Reference] {
                let run = run_sized(library, case, 0, 4, true);
                assert!(
                    run.summary.verified,
                    "{case:?}/{library:?}: {:?}",
                    run.summary
                );
                let generations = run.summary.iterations.unwrap() as usize;
                let calls_per_generation = match library {
                    Library::Basin => POPULATION + 1,
                    Library::Reference => POPULATION,
                };
                assert_eq!(
                    run.summary.objective_calls,
                    generations * calls_per_generation
                );
                assert!(
                    run.points
                        .windows(2)
                        .all(|p| p[0].0 <= p[1].0 && p[0].1 >= p[1].1)
                );
            }
        }
    }
}
