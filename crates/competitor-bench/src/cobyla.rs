//! Public COBYLA comparisons on migration workloads and a larger coupled quadratic.
//!
//! Basin follows PRIMA; the `cobyla` crate is derived from NLopt, while `nlopt`
//! wraps native NLopt 2.9.1. Equal objective budgets and final radii do not imply
//! equal trajectories or numerical work.

use basin::{Cobyla, CostFunction, Executor, SelectedState, TerminationCode};
use std::{
    cell::{Cell, RefCell},
    convert::Infallible,
};

use crate::convergence::{Log, Run, Summary};

const FINAL_RADIUS: f64 = 7.450_580_596_923_828e-9;
pub const FEASIBILITY_TOLERANCE: f64 = 1.490_116_119_384_765_6e-8;

#[derive(Clone, Copy)]
pub enum Library {
    Basin,
    Reference,
    Nlopt,
}

impl Library {
    pub fn name(self) -> &'static str {
        match self {
            Self::Basin => "basin",
            Self::Reference => "cobyla",
            Self::Nlopt => "nlopt",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Case {
    Camel,
    Sphere,
    Quadratic,
    QuadraticNd(usize),
}

pub const CASES: [(Case, &str); 3] = [
    (Case::Camel, "camel_2d"),
    (Case::Sphere, "sphere_10d"),
    (Case::Quadratic, "quadratic_2d"),
];

#[derive(Debug, PartialEq)]
pub enum Stop {
    Budget,
    Radius,
    Other(String),
}

#[derive(Debug)]
pub struct Outcome {
    pub point: Vec<f64>,
    pub cost: f64,
    pub objective_calls: usize,
    /// Vector callbacks in Basin; scalar nonlinear callbacks in the references.
    /// Native bound rows are evaluated internally by both references.
    pub constraint_calls: usize,
    pub iterations: Option<u64>,
    pub stop: Stop,
}

impl Case {
    fn start(self) -> Vec<f64> {
        match self {
            Self::Camel => vec![0.0, 0.0],
            Self::Sphere => {
                vec![2.5, -2.0, 1.5, -1.0, 0.5, 2.25, -1.75, 1.25, -0.75, 0.25]
            }
            Self::Quadratic => vec![0.5, 0.5],
            Self::QuadraticNd(n) => {
                (0..n).map(|i| 0.3 + 0.4 * i as f64 / n as f64).collect()
            }
        }
    }

    fn bounds(self) -> Vec<(f64, f64)> {
        match self {
            Self::Camel => vec![(-3.0, 3.0), (-2.0, 2.0)],
            Self::Sphere => vec![(-5.0, 5.0); 10],
            Self::Quadratic => vec![(0.0, 2.0); 2],
            Self::QuadraticNd(n) => vec![(0.0, 2.0); n],
        }
    }

    pub fn budget(self) -> usize {
        match self {
            Self::Camel => 50,
            Self::Sphere => 200,
            Self::Quadratic => 100,
            Self::QuadraticNd(_) => 10_000,
        }
    }

    fn objective(self, x: &[f64]) -> f64 {
        match self {
            Self::Camel => {
                (4.0 - 2.1 * x[0].powi(2) + x[0].powi(4) / 3.0) * x[0].powi(2)
                    + x[0] * x[1]
                    + (-4.0 + 4.0 * x[1].powi(2)) * x[1].powi(2)
            }
            Self::Sphere => x.iter().map(|v| v * v).sum(),
            Self::Quadratic => (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2),
            Self::QuadraticNd(_) => {
                x.iter().map(|v| (v - 1.0).powi(2)).sum::<f64>()
                    + x.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum::<f64>()
            }
        }
    }

    pub fn violation(self, x: &[f64]) -> f64 {
        if x.iter().any(|v| !v.is_finite()) {
            return f64::INFINITY;
        }
        let mut violation = 0.0_f64;
        for (&v, (lower, upper)) in x.iter().zip(self.bounds()) {
            violation = violation.max(lower - v).max(v - upper);
        }
        if matches!(self, Self::Quadratic | Self::QuadraticNd(_)) {
            violation =
                violation.max(x.iter().sum::<f64>() - 0.75 * x.len() as f64);
        }
        violation
    }

    /// Check the returned point independently, outside the measured solve.
    pub fn verify(self, result: &Outcome) {
        assert_eq!(result.point.len(), self.start().len());
        assert!(result.point.iter().all(|v| v.is_finite()));
        assert!(result.cost.is_finite());
        assert!(result.objective_calls > 0);
        assert!(result.objective_calls <= self.budget());
        let cost = self.objective(&result.point);
        assert!((cost - result.cost).abs() <= 1e-13 * cost.abs().max(1.0));
        assert!(self.violation(&result.point) <= FEASIBILITY_TOLERANCE);
        let (target, tolerance) = match self {
            Self::Camel => (-1.031_628_453_489_877, 1e-5),
            Self::Sphere => (0.0, 1e-5),
            Self::Quadratic => (0.125, 1e-6),
            Self::QuadraticNd(n) => (n as f64 / 16.0, 1e-6),
        };
        assert!((cost - target).abs() < tolerance);
    }

    pub fn solve_basin(self) -> Outcome {
        self.solve_basin_logged(None)
    }

    fn solve_basin_logged(self, log: Option<&RefCell<Log>>) -> Outcome {
        let mut problem = Problem::new(self);
        problem.log = log;
        let start = self.start();
        if let Some(log) = log {
            log.borrow_mut().restart();
        }
        let result = Executor::new(
            &problem,
            Cobyla::new()
                .with_initial_radius(0.5)
                .with_final_radius(FINAL_RADIUS),
            SelectedState::new(start),
        )
        .max_iter(self.budget() as u64)
        .max_cost_evals(self.budget() as u64)
        .run()
        .unwrap();
        Outcome {
            // COBYLA's current state contains its feasibility-filter selection.
            point: result.param().clone(),
            cost: result.cost(),
            objective_calls: problem.objective_calls.get(),
            constraint_calls: problem.constraint_calls.get(),
            iterations: Some(result.iter()),
            stop: match result.report.code() {
                TerminationCode::MaxCostEvals => Stop::Budget,
                TerminationCode::SolverConverged => Stop::Radius,
                other => Stop::Other(format!("{other:?}")),
            },
        }
    }

    pub fn solve_reference(self) -> Outcome {
        self.solve_reference_logged(None)
    }

    fn solve_reference_logged(self, log: Option<&RefCell<Log>>) -> Outcome {
        let mut problem = Problem::new(self);
        problem.log = log;
        let start = self.start();
        let constraint = |x: &[f64], _: &mut ()| {
            problem
                .constraint_calls
                .set(problem.constraint_calls.get() + 1);
            // The reference accepts nonnegative constraints.
            0.75 * x.len() as f64 - x.iter().sum::<f64>()
        };
        let constraints =
            if matches!(self, Self::Quadratic | Self::QuadraticNd(_)) {
                std::slice::from_ref(&constraint)
            } else {
                &[]
            };
        if let Some(log) = log {
            log.borrow_mut().restart();
        }
        let result = ::cobyla::minimize(
            |x: &[f64], _: &mut ()| problem.objective(x),
            &start,
            &problem.bounds,
            constraints,
            (),
            self.budget(),
            ::cobyla::RhoBeg::All(0.5),
            Some(::cobyla::StopTols {
                xtol_abs: vec![FINAL_RADIUS; start.len()],
                ..Default::default()
            }),
        );
        let (point, cost, stop) = match result {
            Ok((status, point, cost)) => (
                point,
                cost,
                match status {
                    ::cobyla::SuccessStatus::MaxEvalReached => Stop::Budget,
                    ::cobyla::SuccessStatus::XtolReached => Stop::Radius,
                    other => Stop::Other(format!("{other:?}")),
                },
            ),
            Err((status, point, cost)) => {
                (point, cost, Stop::Other(format!("{status:?}")))
            }
        };
        Outcome {
            point,
            cost,
            objective_calls: problem.objective_calls.get(),
            constraint_calls: problem.constraint_calls.get(),
            iterations: None,
            stop,
        }
    }

    pub fn solve_nlopt(self) -> Outcome {
        self.solve_nlopt_logged(None)
    }

    fn solve_nlopt_logged(self, log: Option<&RefCell<Log>>) -> Outcome {
        let mut problem = Problem::new(self);
        problem.log = log;
        let mut point = self.start();
        if let Some(log) = log {
            log.borrow_mut().restart();
        }
        let mut opt = nlopt::Nlopt::new(
            nlopt::Algorithm::Cobyla,
            point.len(),
            |x: &[f64], _: Option<&mut [f64]>, _: &mut ()| problem.objective(x),
            nlopt::Target::Minimize,
            (),
        );
        opt.set_lower_bounds(
            &problem.bounds.iter().map(|b| b.0).collect::<Vec<_>>(),
        )
        .unwrap();
        opt.set_upper_bounds(
            &problem.bounds.iter().map(|b| b.1).collect::<Vec<_>>(),
        )
        .unwrap();
        if matches!(self, Self::Quadratic | Self::QuadraticNd(_)) {
            opt.add_inequality_constraint(
                |x: &[f64], _: Option<&mut [f64]>, _: &mut ()| {
                    problem
                        .constraint_calls
                        .set(problem.constraint_calls.get() + 1);
                    // NLopt uses c(x) <= 0, opposite to the `cobyla` crate.
                    x.iter().sum::<f64>() - 0.75 * x.len() as f64
                },
                (),
                // Match the `cobyla` crate's zero constraint tolerance.
                0.0,
            )
            .unwrap();
        }
        opt.set_initial_step1(0.5).unwrap();
        opt.set_xtol_abs1(FINAL_RADIUS).unwrap();
        opt.set_maxeval(self.budget() as u32).unwrap();
        let (cost, stop) = match opt.optimize(&mut point) {
            Ok((status, cost)) => (
                cost,
                match status {
                    nlopt::SuccessState::MaxEvalReached => Stop::Budget,
                    nlopt::SuccessState::XtolReached => Stop::Radius,
                    other => Stop::Other(format!("{other:?}")),
                },
            ),
            Err((status, cost)) => (cost, Stop::Other(format!("{status:?}"))),
        };
        // Include native optimizer teardown in the trace, as on Basin's side.
        drop(opt);
        Outcome {
            point,
            cost,
            objective_calls: problem.objective_calls.get(),
            constraint_calls: problem.constraint_calls.get(),
            iterations: None,
            stop,
        }
    }

    pub fn trace(self, library: Library) -> Run {
        let log = RefCell::new(Log::new());
        let result = match library {
            Library::Basin => self.solve_basin_logged(Some(&log)),
            Library::Reference => self.solve_reference_logged(Some(&log)),
            Library::Nlopt => self.solve_nlopt_logged(Some(&log)),
        };
        let elapsed = log.borrow().elapsed();
        let violation = self.violation(&result.point);
        let cost = self.objective(&result.point);
        let target = match self {
            Self::Camel => -1.031_628_453_489_877,
            Self::Sphere => 0.0,
            Self::Quadratic => 0.125,
            Self::QuadraticNd(n) => n as f64 / 16.0,
        };
        let verified = result.point.iter().all(|x| x.is_finite())
            && result.cost.is_finite()
            && (cost - result.cost).abs() <= 1e-13 * cost.abs().max(1.0)
            && (cost - target).abs() < 1e-6
            && violation <= FEASIBILITY_TOLERANCE;
        let status = match (library, &result.stop) {
            (Library::Basin, Stop::Budget) => "MaxCostEvals".into(),
            (Library::Basin, Stop::Radius) => "SolverConverged".into(),
            (Library::Reference | Library::Nlopt, Stop::Budget) => {
                "MaxEvalReached".into()
            }
            (Library::Reference | Library::Nlopt, Stop::Radius) => {
                "XtolReached".into()
            }
            (_, Stop::Other(status)) => status.clone(),
        };
        log.into_inner().finish(
            elapsed,
            Summary {
                point: result.point,
                cost: result.cost,
                violation,
                objective_calls: result.objective_calls,
                residual_calls: 0,
                jacobian_calls: 0,
                gradient_calls: 0,
                constraint_calls: result.constraint_calls,
                status,
                verified,
            },
            Some(FEASIBILITY_TOLERANCE),
            |x| self.violation(x),
        )
    }
}

struct Problem<'a> {
    case: Case,
    bounds: Vec<(f64, f64)>,
    objective_calls: Cell<usize>,
    constraint_calls: Cell<usize>,
    log: Option<&'a RefCell<Log>>,
}

impl Problem<'_> {
    fn new(case: Case) -> Self {
        Self {
            case,
            bounds: case.bounds(),
            objective_calls: Cell::new(0),
            constraint_calls: Cell::new(0),
            log: None,
        }
    }

    fn objective(&self, x: &[f64]) -> f64 {
        self.objective_calls.set(self.objective_calls.get() + 1);
        let cost = self.case.objective(x);
        if let Some(log) = self.log {
            log.borrow_mut().record(cost, x);
        }
        cost
    }
}

impl basin::Constraints for &Problem<'_> {
    type Set = basin::ConstraintSet<
        basin::NoBounds,
        basin::NoLinearConstraints,
        basin::NonlinearBlocks<basin::NoNonlinearBlock, basin::NonlinearBlock>,
    >;
    fn constraints(
        &self,
    ) -> <Self::Set as basin::ConstraintDescription>::Borrowed<'_> {
        basin::ConstraintSet::new().with_nonlinear_inequalities({
            2 * self.bounds.len()
                + usize::from(matches!(
                    self.case,
                    Case::Quadratic | Case::QuadraticNd(_)
                ))
        })
    }
}
impl CostFunction for &Problem<'_> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        // Executor budgets are checked between iterations. The callback guard
        // preserves the strict objective budget used in the migration workload.
        Ok(if self.objective_calls.get() >= self.case.budget() {
            f64::INFINITY
        } else {
            self.objective(x)
        })
    }
}

impl basin::NonlinearInequalityValues for &Problem<'_> {
    type Param = Vec<f64>;
    type Values = Vec<f64>;
    type Error = Infallible;
    fn nonlinear_inequalities(
        &self,
        x: &Self::Param,
    ) -> Result<Self::Values, basin::ConstraintSolveError<Self::Error>> {
        {
            self.constraint_calls.set(self.constraint_calls.get() + 1);
            let mut values = Vec::with_capacity(
                basin::Constraints::constraints(self)
                    .nonlinear()
                    .inequalities()
                    .count(),
            );
            if matches!(self.case, Case::Quadratic | Case::QuadraticNd(_)) {
                values.push(-(0.75 * x.len() as f64 - x.iter().sum::<f64>()));
            }
            for (&v, &(lower, upper)) in x.iter().zip(&self.bounds) {
                values.extend([lower - v, v - upper]);
            }
            Ok(values)
        }
        .map_err(basin::ConstraintSolveError::Evaluation)
    }
}

#[cfg(test)]
mod tests {
    use super::{CASES, Stop};

    #[test]
    fn larger_coupled_quadratic_recovers_the_constrained_optimum() {
        let case = super::Case::QuadraticNd(30);
        for library in [
            super::Library::Basin,
            super::Library::Reference,
            super::Library::Nlopt,
        ] {
            let result = case.trace(library);
            assert!(result.summary.verified);
            assert_eq!(result.summary.point.len(), 30);
            assert!(result.summary.objective_calls <= case.budget());
            assert!(
                result
                    .violations
                    .iter()
                    .any(|p| p.1 > super::FEASIBILITY_TOLERANCE)
            );
        }
    }

    #[test]
    fn quadratic_tracing_preserves_solutions_and_raw_callback_counts() {
        use super::{Case, Library};
        for library in [Library::Basin, Library::Reference, Library::Nlopt] {
            let plain = match library {
                Library::Basin => Case::Quadratic.solve_basin(),
                Library::Reference => Case::Quadratic.solve_reference(),
                Library::Nlopt => Case::Quadratic.solve_nlopt(),
            };
            let traced = Case::Quadratic.trace(library);
            assert!(traced.summary.verified);
            assert_eq!(plain.point, traced.summary.point);
            assert_eq!(plain.cost, traced.summary.cost);
            assert_eq!(plain.objective_calls, traced.summary.objective_calls);
            assert_eq!(plain.constraint_calls, traced.summary.constraint_calls);
            assert_eq!(traced.violations.len(), plain.objective_calls + 1);
            assert!(traced.violations.iter().any(|p| p.1 > 0.0));
            assert!(traced.points.iter().all(|p| p.1 >= 0.125 - 1e-8));
        }
    }

    #[test]
    fn migration_comparison_preserves_quality_budgets_and_stops() {
        for ((case, _), (basin_calls, constraints)) in
            CASES.into_iter().zip([(50, 50), (200, 201), (61, 61)])
        {
            let basin = case.solve_basin();
            case.verify(&basin);
            assert_eq!(basin.objective_calls, basin_calls);
            assert_eq!(basin.constraint_calls, constraints);
            let stop = if basin_calls == case.budget() {
                Stop::Budget
            } else {
                Stop::Radius
            };
            assert_eq!(basin.stop, stop);
            for reference in [case.solve_reference(), case.solve_nlopt()] {
                case.verify(&reference);
                assert!(matches!(reference.stop, Stop::Budget | Stop::Radius));
                if reference.stop == Stop::Budget {
                    assert_eq!(reference.objective_calls, case.budget());
                }
                assert_eq!(
                    reference.constraint_calls,
                    if matches!(case, super::Case::Quadratic) {
                        reference.objective_calls
                    } else {
                        0
                    }
                );
            }
        }
    }
}
