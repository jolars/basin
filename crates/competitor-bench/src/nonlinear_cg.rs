//! Nonlinear conjugate gradient with Polak–Ribière+ on chained Rosenbrock.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use argmin::core::observers::{Observe, ObserverMode};
use argmin::core::{Error, KV, State as ArgminState};
use argmin::solver::conjugategradient::{
    NonlinearConjugateGradient, beta::PolakRibierePlus,
};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use basin::problems::rosenbrock;
use basin::{CgUpdate, Executor, MoreThuente, NonlinearCg};

use crate::convergence::{RosenbrockWorkload as Workload, Run, Summary};

pub const DIMENSION: usize = 100;
pub const TARGET: f64 = 1e-10;
const MAX_ITERS: u64 = 10_000;
const CURVATURE: f64 = 0.1;

#[derive(Clone, Copy, Debug)]
pub enum Library {
    Basin,
    Argmin,
}

impl Library {
    pub fn name(self) -> &'static str {
        match self {
            Self::Basin => "basin",
            Self::Argmin => "argmin",
        }
    }
}

struct Progress {
    point: Vec<f64>,
    cost: f64,
    iterations: u64,
}

struct Samples {
    start: Instant,
    points: Vec<(u128, f64)>,
    iterations: u64,
}

impl Samples {
    fn record(&mut self, cost: f64, iterations: u64) {
        self.points.push((self.start.elapsed().as_nanos(), cost));
        self.iterations = iterations;
    }
}

struct Observer(Arc<Mutex<Samples>>);

impl<I: ArgminState<Param = Vec<f64>, Float = f64>> Observe<I> for Observer {
    fn observe_init(
        &mut self,
        _: &str,
        state: &I,
        _: &KV,
    ) -> Result<(), Error> {
        self.0.lock().unwrap().record(state.get_cost(), 0);
        Ok(())
    }

    fn observe_iter(&mut self, state: &I, _: &KV) -> Result<(), Error> {
        self.0
            .lock()
            .unwrap()
            .record(state.get_cost(), state.get_iter() + 1);
        Ok(())
    }
}

fn basin_solve(
    problem: &Workload,
    n: usize,
    samples: Option<&Arc<Mutex<Samples>>>,
) -> (String, Option<Progress>) {
    let solver = NonlinearCg::with_line_search(
        MoreThuente::new()
            .with_sufficient_decrease_coefficient(1e-4)
            .with_curvature_coefficient(CURVATURE)
            .with_relative_bracket_tolerance(1e-10)
            .stpmin(f64::EPSILON.sqrt())
            .stpmax(1e10),
    )
    .with_update(CgUpdate::PolakRibierePlus);
    let mut exec = Executor::from_start(problem, solver, crate::nm::start(n))
        .max_iter(MAX_ITERS)
        .target_objective(TARGET);
    if let Some(samples) = samples {
        use basin::State;
        let samples = Arc::clone(samples);
        exec = exec.observe_solver(
            move |state, _, event| {
                if !matches!(event, basin::ObservationEvent::Final(_)) {
                    samples.lock().unwrap().record(state.cost(), state.iter());
                }
            },
            basin::ObserverMode::Always,
        );
    }
    let result = exec.run().unwrap();
    assert_eq!(
        result.counts.cost_evals as usize,
        problem.costs.load(Ordering::Relaxed)
    );
    assert_eq!(
        result.counts.gradient_evals as usize,
        problem.gradients.load(Ordering::Relaxed)
    );
    let progress = Progress {
        point: result.param().clone(),
        cost: result.cost(),
        iterations: result.iter(),
    };
    (format!("{:?}", result.report.code()), Some(progress))
}

fn argmin_solve(
    problem: &Workload,
    n: usize,
    samples: Option<&Arc<Mutex<Samples>>>,
) -> (String, Option<Progress>) {
    let search = MoreThuenteLineSearch::new()
        .with_c(1e-4, CURVATURE)
        .unwrap()
        .with_width_tolerance(1e-10)
        .unwrap()
        .with_bounds(f64::EPSILON.sqrt(), 1e10)
        .unwrap();
    let solver =
        NonlinearConjugateGradient::new(search, PolakRibierePlus::new());
    let mut exec = argmin::core::Executor::new(problem, solver)
        .configure(|s| {
            s.param(crate::nm::start(n))
                .max_iters(MAX_ITERS)
                .target_cost(TARGET)
        })
        .ctrlc(false);
    if let Some(samples) = samples {
        exec = exec
            .add_observer(Observer(Arc::clone(samples)), ObserverMode::Always);
    }
    match exec.run() {
        Ok(result) => {
            let state = &result.state;
            let progress = Progress {
                point: state.get_param().unwrap().clone(),
                cost: state.get_cost(),
                iterations: state.get_iter(),
            };
            assert_eq!(
                result
                    .problem
                    .counts
                    .get("cost_count")
                    .copied()
                    .unwrap_or(0) as usize,
                problem.costs.load(Ordering::Relaxed)
            );
            assert_eq!(
                result
                    .problem
                    .counts
                    .get("gradient_count")
                    .copied()
                    .unwrap_or(0) as usize,
                problem.gradients.load(Ordering::Relaxed)
            );
            (
                format!("{:?}", state.get_termination_status()),
                Some(progress),
            )
        }
        Err(error) => (format!("SolverFailed: {error}"), None),
    }
}

/// Include construction, initialization, accepted-iterate sampling, and teardown.
/// Independent objective checks do not contribute to callback counts or timing.
pub fn run(library: Library, n: usize, tracing: bool) -> Run {
    assert!(n >= 2);
    let problem = Workload::default();
    let start = Instant::now();
    let samples = tracing.then(|| {
        Arc::new(Mutex::new(Samples {
            start,
            points: Vec::new(),
            iterations: 0,
        }))
    });
    let (status, returned) = match library {
        Library::Basin => basin_solve(&problem, n, samples.as_ref()),
        Library::Argmin => argmin_solve(&problem, n, samples.as_ref()),
    };
    let samples = samples.map(|samples| {
        Arc::try_unwrap(samples).ok().unwrap().into_inner().unwrap()
    });
    let elapsed = start.elapsed().as_nanos();
    let (mut points, observed_iterations) = samples.map_or_else(
        || (Vec::new(), None),
        |samples| (samples.points, Some(samples.iterations)),
    );
    let returned_point = returned.is_some();
    let iterations = returned
        .as_ref()
        .map(|progress| progress.iterations)
        .or(observed_iterations);
    let progress = returned.unwrap_or_else(|| Progress {
        point: vec![f64::NAN; n],
        cost: f64::NAN,
        iterations: 0,
    });
    points.push((elapsed, progress.cost));
    let mut best = f64::INFINITY;
    points.retain_mut(|(_, cost)| {
        if cost.is_finite() {
            best = best.min(*cost);
        }
        *cost = best;
        best.is_finite()
    });
    let checked = rosenbrock(&progress.point);
    let verified = returned_point
        && progress.point.iter().all(|x| x.is_finite())
        && checked.is_finite()
        && (0.0..=TARGET).contains(&checked)
        && (checked - progress.cost).abs() <= 1e-12 * checked.abs().max(1.0);
    Run {
        points,
        violations: Vec::new(),
        summary: Summary {
            point: progress.point,
            cost: checked,
            violation: 0.0,
            iterations,
            projected_gradient: None,
            objective_calls: problem.costs.load(Ordering::Relaxed),
            residual_calls: 0,
            jacobian_calls: 0,
            gradient_calls: problem.gradients.load(Ordering::Relaxed),
            constraint_calls: 0,
            status,
            verified,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_drivers_reach_the_target_with_consistent_counts_and_traces() {
        for library in [Library::Basin, Library::Argmin] {
            let traced = run(library, DIMENSION, true);
            let untraced = run(library, DIMENSION, false);
            assert!(traced.summary.verified, "{:?}", traced.summary);
            assert!(untraced.summary.verified, "{:?}", untraced.summary);
            assert_eq!(traced.summary.point, untraced.summary.point);
            assert_eq!(traced.summary.cost, untraced.summary.cost);
            assert_eq!(traced.summary.iterations, untraced.summary.iterations);
            assert_eq!(
                traced.summary.objective_calls,
                untraced.summary.objective_calls
            );
            assert_eq!(
                traced.summary.gradient_calls,
                untraced.summary.gradient_calls
            );
            assert!(traced.summary.objective_calls > 1);
            assert!(traced.summary.gradient_calls > 1);
            assert_eq!(
                traced.points[0].1,
                rosenbrock(&crate::nm::start(DIMENSION))
            );
            assert_eq!(traced.points.last().unwrap().1, traced.summary.cost);
            assert!(
                traced
                    .points
                    .windows(2)
                    .all(|p| p[0].0 <= p[1].0 && p[0].1 >= p[1].1)
            );
            assert_eq!(untraced.points.len(), 1);
        }
    }
}
