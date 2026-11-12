//! Dense BFGS on chained Rosenbrock with matched seeds and line-search settings.

use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use argmin::core::observers::{Observe, ObserverMode};
use argmin::core::{Error, KV, State as ArgminState};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::BFGS;
use basin::problems::{rosenbrock, rosenbrock_gradient};
use basin::solver::bfgs::InitialInverseHessian;
use basin::{
    Bfgs, CostFunction, DenseMatrix, Executor, FirstOrderState, Gradient,
    MatrixIdentity, MoreThuente, State, StepOutcome,
};

use crate::convergence::{Run, Summary};

pub const DIMENSION: usize = 100;
pub const TARGET: f64 = 1e-10;
pub const MAX_ITERS: u64 = 10_000;

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

#[derive(Default)]
struct Workload {
    costs: AtomicUsize,
    gradients: AtomicUsize,
}

impl CostFunction for &Workload {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        self.costs.fetch_add(1, Ordering::Relaxed);
        Ok(rosenbrock(x))
    }
}

impl Gradient for &Workload {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        self.gradients.fetch_add(1, Ordering::Relaxed);
        let mut g = vec![0.0; x.len()];
        rosenbrock_gradient(x, &mut g);
        Ok(g)
    }
}

impl argmin::core::CostFunction for &Workload {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Error> {
        Ok(CostFunction::cost(self, x)?)
    }
}

impl argmin::core::Gradient for &Workload {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Error> {
        Ok(Gradient::gradient(self, x)?)
    }
}

/// Match argmin's identity model without Basin's optional first-update rescaling.
struct UnscaledIdentity;

impl InitialInverseHessian<Vec<f64>, DenseMatrix> for UnscaledIdentity {
    fn initialize(&self, x: &Vec<f64>, _: &Vec<f64>) -> DenseMatrix {
        DenseMatrix::identity(x.len())
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
    progress: Progress,
    tracing: bool,
}

impl Samples {
    fn record(&mut self, point: &[f64], cost: f64, iterations: u64) {
        if self.tracing {
            self.points.push((self.start.elapsed().as_nanos(), cost));
        }
        self.progress = Progress {
            point: point.to_vec(),
            cost,
            iterations,
        };
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
        self.observe_iter(state, &KV::new())
    }

    fn observe_iter(&mut self, state: &I, _: &KV) -> Result<(), Error> {
        self.0.lock().unwrap().record(
            state.get_param().unwrap(),
            state.get_cost(),
            state.get_iter(),
        );
        Ok(())
    }
}

fn basin_solve(problem: &Workload, n: usize, samples: &mut Samples) -> String {
    let solver = Bfgs::with_line_search(
        MoreThuente::new()
            .with_sufficient_decrease_coefficient(1e-4)
            .with_relative_bracket_tolerance(1e-10)
            .stpmin(f64::EPSILON.sqrt())
            .stpmax(1e10),
    )
    .with_initial_inverse_hessian_initializer(UnscaledIdentity);
    let mut stepper = Executor::new(
        problem,
        solver,
        FirstOrderState::new(crate::nm::start(n)),
    )
    .max_iter(MAX_ITERS)
    .into_stepper()
    .unwrap();
    let status = loop {
        let state = stepper.state();
        samples.record(state.param(), state.cost(), stepper.iter());
        if (0.0..=TARGET).contains(&state.cost()) {
            break "TargetReached".into();
        }
        if let StepOutcome::Stopped(report) = stepper.step().unwrap() {
            let state = stepper.state();
            samples.record(state.param(), state.cost(), stepper.iter());
            break format!("{:?}", report.code());
        }
    };
    assert_eq!(
        stepper.counts().cost_evals as usize,
        problem.costs.load(Ordering::Relaxed)
    );
    assert_eq!(
        stepper.counts().gradient_evals as usize,
        problem.gradients.load(Ordering::Relaxed)
    );
    status
}

fn argmin_solve(
    problem: &Workload,
    n: usize,
    samples: Arc<Mutex<Samples>>,
) -> String {
    let search: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> =
        MoreThuenteLineSearch::new()
            .with_c(1e-4, 0.9)
            .unwrap()
            .with_width_tolerance(1e-10)
            .unwrap()
            .with_bounds(f64::EPSILON.sqrt(), 1e10)
            .unwrap();
    let solver = BFGS::new(search)
        .with_tolerance_grad(0.0)
        .unwrap()
        .with_tolerance_cost(0.0)
        .unwrap();
    let result = argmin::core::Executor::new(problem, solver)
        .configure(|s| {
            s.param(crate::nm::start(n))
                .inv_hessian(
                    (0..n)
                        .map(|i| (0..n).map(|j| f64::from(i == j)).collect())
                        .collect::<Vec<Vec<f64>>>(),
                )
                .max_iters(MAX_ITERS)
                .target_cost(TARGET)
        })
        .ctrlc(false)
        .add_observer(Observer(Arc::clone(&samples)), ObserverMode::Always)
        .run();
    match result {
        Ok(result) => {
            let state = &result.state;
            samples.lock().unwrap().record(
                state.get_param().unwrap(),
                state.get_cost(),
                state.get_iter(),
            );
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
            format!("{:?}", state.get_termination_status())
        }
        Err(error) => format!("SolverFailed: {error}"),
    }
}

/// Include construction, initialization, accepted-iterate sampling, and teardown.
/// Independent objective checks do not contribute to callback counts or timing.
pub fn run(library: Library, n: usize, tracing: bool) -> Run {
    assert!(n >= 2);
    let problem = Workload::default();
    let mut samples = Samples {
        start: Instant::now(),
        points: Vec::new(),
        progress: Progress {
            point: vec![f64::NAN; n],
            cost: f64::NAN,
            iterations: 0,
        },
        tracing,
    };
    let start = Instant::now();
    samples.start = start;
    let (status, mut samples) = match library {
        Library::Basin => (basin_solve(&problem, n, &mut samples), samples),
        Library::Argmin => {
            let samples = Arc::new(Mutex::new(samples));
            let status = argmin_solve(&problem, n, Arc::clone(&samples));
            (
                status,
                Arc::try_unwrap(samples).ok().unwrap().into_inner().unwrap(),
            )
        }
    };
    let elapsed = start.elapsed().as_nanos();
    let progress = samples.progress;
    samples.points.push((elapsed, progress.cost));
    let mut best = f64::INFINITY;
    samples.points.retain_mut(|(_, cost)| {
        if cost.is_finite() {
            best = best.min(*cost);
        }
        *cost = best;
        best.is_finite()
    });
    let checked = rosenbrock(&progress.point);
    let verified = progress.point.iter().all(|x| x.is_finite())
        && checked.is_finite()
        && (0.0..=TARGET).contains(&checked)
        && (checked - progress.cost).abs() <= 1e-12 * checked.abs().max(1.0);
    Run {
        points: samples.points,
        violations: Vec::new(),
        summary: Summary {
            point: progress.point,
            cost: checked,
            violation: 0.0,
            iterations: Some(progress.iterations),
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
            let traced = run(library, 4, true);
            let untraced = run(library, 4, false);
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
            assert_eq!(traced.points[0].1, rosenbrock(&crate::nm::start(4)));
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
