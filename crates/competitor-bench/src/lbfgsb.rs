//! One bounded L-BFGS-B comparison against Becker's C translation, via
//! `lbfgsb` 0.1.1. The extended Rosenbrock objective and bounds follow the
//! reference's driver 3 and Basin's `benches/support/lbfgsb.rs`.

use std::{
    cell::Cell, convert::Infallible, os::raw::c_long, sync::Mutex,
    time::Instant,
};

use basin::{
    CostFunction, Executor, FirstOrderState, Gradient, Lbfgsb, State,
    StepOutcome,
};

use crate::convergence::{Run, Summary};

pub const DIMENSION: usize = 20_000;
pub const MEMORY: usize = 10;
pub const PROJECTED_GRADIENT_TOLERANCE: f64 = 1e-8;
const MAX_ITERATIONS: u64 = 2_000;
const MAX_EVALUATIONS: u64 = 10_000;

// The translated C routines retain mutable static scratch variables.
static NATIVE: Mutex<()> = Mutex::new(());

#[derive(Clone, Copy, Debug)]
pub enum Library {
    Basin,
    Reference,
}

impl Library {
    pub fn name(self) -> &'static str {
        match self {
            Self::Basin => "basin",
            Self::Reference => "lbfgsb",
        }
    }
}

struct Workload {
    lower: Vec<f64>,
    upper: Vec<f64>,
    max_trial_violation: Cell<f64>,
}

impl Workload {
    fn new(n: usize) -> Self {
        assert!(n >= 2);
        Self {
            lower: (0..n)
                .map(|i| if i % 2 == 0 { 1.0 } else { -100.0 })
                .collect(),
            upper: vec![100.0; n],
            max_trial_violation: Cell::new(0.0),
        }
    }

    fn violation(&self, x: &[f64]) -> f64 {
        if x.len() != self.lower.len() || x.iter().any(|v| !v.is_finite()) {
            return f64::INFINITY;
        }
        x.iter().enumerate().fold(0.0_f64, |v, (i, &x)| {
            v.max(self.lower[i] - x).max(x - self.upper[i])
        })
    }

    fn evaluate(&self, x: &[f64], g: &mut [f64]) -> f64 {
        self.max_trial_violation
            .set(self.max_trial_violation.get().max(self.violation(x)));
        let n = x.len();
        let mut value = 0.25 * (x[0] - 1.0).powi(2);
        for i in 1..n {
            value += (x[i] - x[i - 1].powi(2)).powi(2);
        }
        let mut t1 = x[1] - x[0].powi(2);
        g[0] = 2.0 * (x[0] - 1.0) - 16.0 * x[0] * t1;
        for i in 1..n - 1 {
            let t2 = t1;
            t1 = x[i + 1] - x[i].powi(2);
            g[i] = 8.0 * t2 - 16.0 * x[i] * t1;
        }
        g[n - 1] = 8.0 * t1;
        4.0 * value
    }

    fn projected_gradient(&self, x: &[f64], g: &[f64]) -> f64 {
        if g.iter().any(|v| !v.is_finite()) || !self.violation(x).is_finite() {
            return f64::INFINITY;
        }
        x.iter()
            .zip(g)
            .enumerate()
            .map(|(i, (&x, &g))| {
                if g < 0.0 {
                    g.max(x - self.upper[i]).abs()
                } else {
                    g.min(x - self.lower[i]).abs()
                }
            })
            .fold(0.0, f64::max)
    }
}

impl basin::Constraints for &Workload {
    type Set = basin::ConstraintSet<basin::BoxBounds<Vec<f64>>>;
    fn constraints(
        &self,
    ) -> <Self::Set as basin::ConstraintDescription>::Borrowed<'_> {
        basin::ConstraintSet::new().with_bounds(basin::BoxBounds::new(
            Some(&self.lower),
            Some(&self.upper),
        ))
    }
}

impl CostFunction for &Workload {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(self.evaluate(x, &mut vec![0.0; x.len()]))
    }
}

impl Gradient for &Workload {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(self.cost_and_gradient(x)?.1)
    }
    fn cost_and_gradient(
        &self,
        x: &Vec<f64>,
    ) -> Result<(f64, Vec<f64>), Infallible> {
        let mut g = vec![0.0; x.len()];
        Ok((self.evaluate(x, &mut g), g))
    }
}

struct Outcome {
    point: Vec<f64>,
    cost: f64,
    iterations: u64,
    objective_calls: usize,
    gradient_calls: usize,
    status: String,
}

fn basin_solve(
    problem: &Workload,
    limits: (u64, u64),
    record: &mut impl FnMut(f64),
) -> Outcome {
    let solver = Lbfgsb::new()
        .with_m_capacity(MEMORY)
        .with_absolute_projected_gradient_tolerance(
            PROJECTED_GRADIENT_TOLERANCE,
        )
        .with_relative_cost_reduction_tolerance(0.0);
    let mut stepper = Executor::new(
        problem,
        solver,
        FirstOrderState::new(vec![3.0; problem.lower.len()]),
    )
    .max_iter(limits.0)
    .max_cost_evals(limits.1)
    .into_stepper()
    .unwrap();
    record(stepper.state().cost());
    loop {
        let iteration = stepper.iter();
        let outcome = stepper.step().unwrap();
        if stepper.iter() != iteration {
            record(stepper.state().cost());
        }
        if let StepOutcome::Stopped(report) = outcome {
            return Outcome {
                point: stepper.state().param().clone(),
                cost: stepper.state().cost(),
                iterations: stepper.iter(),
                objective_calls: stepper.state().cost_evals() as usize,
                gradient_calls: stepper.counts().gradient_evals as usize,
                status: format!("{:?}", report.code()),
            };
        }
    }
}

fn native_solve(
    problem: &Workload,
    limits: (u64, u64),
    record: &mut impl FnMut(f64),
) -> Outcome {
    let _guard = NATIVE.lock().unwrap();
    let size = problem.lower.len();
    let n = c_long::try_from(size).unwrap();
    let m = MEMORY as c_long;
    let mut x = vec![3.0; size];
    let mut g = vec![0.0; size];
    let mut f = f64::NAN;
    let nbd = vec![2 as c_long; size];
    let mut wa =
        vec![0.0; (2 * MEMORY + 5) * size + 11 * MEMORY * MEMORY + 8 * MEMORY];
    let mut iwa = vec![0 as c_long; 3 * size];
    let mut task = 1 as c_long;
    let mut csave = 0 as c_long;
    let mut lsave = [0 as c_long; 4];
    let mut isave = [0 as c_long; 44];
    let mut dsave = [0.0; 29];
    let mut calls = 0;
    let status = loop {
        // SAFETY: All buffers have the sizes documented by setulb and remain
        // live and disjoint across reverse-communication calls. c_long matches
        // the bundled header's integer/logical typedefs. The mutex protects
        // the C routines' static scratch storage for the entire solve.
        unsafe {
            ::lbfgsb::setulb(
                &n,
                &m,
                x.as_mut_ptr(),
                problem.lower.as_ptr(),
                problem.upper.as_ptr(),
                nbd.as_ptr(),
                &mut f,
                g.as_mut_ptr(),
                &0.0,
                &PROJECTED_GRADIENT_TOLERANCE,
                wa.as_mut_ptr(),
                iwa.as_mut_ptr(),
                &mut task,
                &-1,
                &mut csave,
                lsave.as_mut_ptr(),
                isave.as_mut_ptr(),
                dsave.as_mut_ptr(),
            );
        }
        match task {
            10..=15 => {
                f = problem.evaluate(&x, &mut g);
                calls += 1;
                if calls == 1 {
                    record(f);
                    if calls >= limits.1 {
                        break "MaxCostEvals".into();
                    }
                }
            }
            2 => {
                record(f);
                if isave[29] as u64 >= limits.0 {
                    break "MaxIter".into();
                }
                if calls >= limits.1 {
                    break "MaxCostEvals".into();
                }
            }
            21 => break "ProjectedGradientTolerance".into(),
            22 => break "RelativeCostReductionTolerance".into(),
            _ => break format!("NativeTask({task})"),
        }
    };
    Outcome {
        point: x,
        cost: f,
        iterations: isave[29] as u64,
        objective_calls: calls as usize,
        gradient_calls: calls as usize,
        status,
    }
}

/// A fresh solve including initialization, accepted-iterate tracing, and teardown.
pub fn run(library: Library, n: usize) -> Run {
    run_logged(library, n, true)
}

/// Disable intermediate samples when calibrating the uninstrumented solve size.
pub fn run_logged(library: Library, n: usize, tracing: bool) -> Run {
    run_with_limits(library, n, tracing, (MAX_ITERATIONS, MAX_EVALUATIONS))
}

fn run_with_limits(
    library: Library,
    n: usize,
    tracing: bool,
    limits: (u64, u64),
) -> Run {
    let mut points = Vec::new();
    let start = Instant::now();
    let problem = Workload::new(n);
    let mut record = |cost| {
        if tracing {
            points.push((start.elapsed().as_nanos(), cost));
        }
    };
    let outcome = match library {
        Library::Basin => basin_solve(&problem, limits, &mut record),
        Library::Reference => native_solve(&problem, limits, &mut record),
    };
    let elapsed = start.elapsed().as_nanos();
    points.push((elapsed, outcome.cost));
    // Independent checks neither add to solver counters nor enter the timer.
    let mut gradient = vec![0.0; n];
    let cost = problem.evaluate(&outcome.point, &mut gradient);
    let violation = problem.violation(&outcome.point);
    let pg = problem.projected_gradient(&outcome.point, &gradient);
    let verified = cost.is_finite()
        && outcome.cost.is_finite()
        && (cost - outcome.cost).abs() <= 1e-13 * cost.abs().max(1.0)
        && cost <= 1e-10
        && violation <= 1e-12
        && problem.max_trial_violation.get() <= 1e-12
        && pg <= PROJECTED_GRADIENT_TOLERANCE * (1.0 + 1e-6);
    Run {
        points,
        violations: Vec::new(),
        summary: Summary {
            point: outcome.point,
            cost: outcome.cost,
            violation,
            iterations: Some(outcome.iterations),
            projected_gradient: Some(pg),
            objective_calls: outcome.objective_calls,
            gradient_calls: outcome.gradient_calls,
            residual_calls: 0,
            jacobian_calls: 0,
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
    fn objective_and_gradient_match_the_reference_formula() {
        let problem = Workload::new(6);
        let x = vec![1.3, 0.8, 1.1, -0.2, 1.4, 0.7];
        let mut g = vec![0.0; x.len()];
        let cost = problem.evaluate(&x, &mut g);
        let expected = (x[0] - 1.0).powi(2)
            + 4.0
                * x.windows(2)
                    .map(|p| (p[1] - p[0] * p[0]).powi(2))
                    .sum::<f64>();
        assert!((cost - expected).abs() < 1e-13);
        for i in 0..x.len() {
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus[i] += 1e-6;
            minus[i] -= 1e-6;
            let numeric = (problem.evaluate(&plus, &mut vec![0.0; x.len()])
                - problem.evaluate(&minus, &mut vec![0.0; x.len()]))
                / 2e-6;
            assert!((numeric - g[i]).abs() <= 1e-7 * g[i].abs().max(1.0));
        }
        assert_eq!(problem.evaluate(&[1.0; 6], &mut [0.0; 6]), 0.0);
    }

    #[test]
    fn both_adapters_reach_the_target_with_matched_work() {
        for n in [25, 1_000] {
            let a = run(Library::Basin, n);
            let b = run(Library::Reference, n);
            for run in [&a, &b] {
                assert!(run.summary.verified, "{:?}", run.summary);
                assert_eq!(run.summary.status, "ProjectedGradientTolerance");
                assert_eq!(
                    run.summary.objective_calls,
                    run.summary.gradient_calls
                );
                assert_eq!(run.summary.violation, 0.0);
                if n == 1_000 {
                    assert!(
                        run.summary.point.iter().step_by(2).any(|&x| x == 1.0)
                    );
                }
                assert!(
                    run.points
                        .windows(2)
                        .all(|p| p[0].0 <= p[1].0 && p[0].1 >= p[1].1)
                );
            }
            assert_eq!(a.summary.iterations, b.summary.iterations);
            assert_eq!(a.summary.objective_calls, b.summary.objective_calls);
            assert!((a.summary.cost - b.summary.cost).abs() < 1e-15);
        }
    }

    #[test]
    fn budgets_and_nonfinite_points_are_not_verified_successes() {
        for library in [Library::Basin, Library::Reference] {
            let run = run_with_limits(library, 25, true, (2_000, 1));
            assert!(!run.summary.verified);
            assert_eq!(run.summary.objective_calls, 1);
            assert_eq!(run.summary.status, "MaxCostEvals");
        }
        let problem = Workload::new(2);
        assert_eq!(problem.violation(&[f64::NAN, 1.0]), f64::INFINITY);
        assert_eq!(
            problem.projected_gradient(&[1.0, 1.0], &[f64::NAN, 0.0]),
            f64::INFINITY
        );
    }
}
