//! Public COBYLA comparisons on the three GlobalSearch migration workloads.
//!
//! Basin follows PRIMA; the `cobyla` crate is derived from NLopt. Equal objective
//! budgets and final radii do not imply equal trajectories or numerical work.

use basin::{
    Cobyla, CobylaState, CostFunction, Executor,
    NonlinearInequalityConstraints, TerminationReason,
};
use std::{cell::Cell, convert::Infallible};

const FINAL_RADIUS: f64 = 7.450_580_596_923_828e-9;
const FEASIBILITY_TOLERANCE: f64 = 1.490_116_119_384_765_6e-8;

#[derive(Clone, Copy, Debug)]
pub enum Case {
    Camel,
    Sphere,
    Quadratic,
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
}

#[derive(Debug)]
pub struct Outcome {
    pub point: Vec<f64>,
    pub cost: f64,
    pub objective_calls: usize,
    /// Vector callbacks in Basin; scalar nonlinear callbacks in `cobyla`.
    /// Native bound rows are evaluated internally by the latter.
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
        }
    }

    fn bounds(self) -> Vec<(f64, f64)> {
        match self {
            Self::Camel => vec![(-3.0, 3.0), (-2.0, 2.0)],
            Self::Sphere => vec![(-5.0, 5.0); 10],
            Self::Quadratic => vec![(0.0, 2.0); 2],
        }
    }

    pub fn budget(self) -> usize {
        match self {
            Self::Camel => 50,
            Self::Sphere => 200,
            Self::Quadratic => 100,
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
        }
    }

    pub fn violation(self, x: &[f64]) -> f64 {
        let mut violation = 0.0_f64;
        for (&v, (lower, upper)) in x.iter().zip(self.bounds()) {
            violation = violation.max(lower - v).max(v - upper);
        }
        if matches!(self, Self::Quadratic) {
            violation = violation.max(-(1.5 - x[0] - x[1]));
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
        };
        assert!((cost - target).abs() < tolerance);
    }

    pub fn solve_basin(self) -> Outcome {
        let problem = Problem::new(self);
        let result = Executor::new(
            &problem,
            Cobyla::new()
                .with_initial_radius(0.5)
                .with_final_radius(FINAL_RADIUS),
            CobylaState::new(self.start()),
        )
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
            stop: match result.reason {
                TerminationReason::MaxCostEvals => Stop::Budget,
                TerminationReason::SolverConverged => Stop::Radius,
                other => panic!("unexpected Basin termination: {other:?}"),
            },
        }
    }

    pub fn solve_reference(self) -> Outcome {
        let problem = Problem::new(self);
        let start = self.start();
        let constraint = |x: &[f64], _: &mut ()| {
            problem
                .constraint_calls
                .set(problem.constraint_calls.get() + 1);
            // The reference accepts nonnegative constraints.
            1.5 - x[0] - x[1]
        };
        let constraints = if matches!(self, Self::Quadratic) {
            std::slice::from_ref(&constraint)
        } else {
            &[]
        };
        let (status, point, cost) = ::cobyla::minimize(
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
        )
        .expect("reference COBYLA failed");
        Outcome {
            point,
            cost,
            objective_calls: problem.objective_calls.get(),
            constraint_calls: problem.constraint_calls.get(),
            iterations: None,
            stop: match status {
                ::cobyla::SuccessStatus::MaxEvalReached => Stop::Budget,
                ::cobyla::SuccessStatus::XtolReached => Stop::Radius,
                other => panic!("unexpected reference termination: {other:?}"),
            },
        }
    }
}

struct Problem {
    case: Case,
    bounds: Vec<(f64, f64)>,
    objective_calls: Cell<usize>,
    constraint_calls: Cell<usize>,
}

impl Problem {
    fn new(case: Case) -> Self {
        Self {
            case,
            bounds: case.bounds(),
            objective_calls: Cell::new(0),
            constraint_calls: Cell::new(0),
        }
    }

    fn objective(&self, x: &[f64]) -> f64 {
        self.objective_calls.set(self.objective_calls.get() + 1);
        self.case.objective(x)
    }
}

impl CostFunction for &Problem {
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

impl NonlinearInequalityConstraints for &Problem {
    fn num_constraints(&self) -> usize {
        2 * self.bounds.len()
            + usize::from(matches!(self.case, Case::Quadratic))
    }

    fn constraints(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        self.constraint_calls.set(self.constraint_calls.get() + 1);
        let mut values = Vec::with_capacity(self.num_constraints());
        if matches!(self.case, Case::Quadratic) {
            values.push(-(1.5 - x[0] - x[1]));
        }
        for (&v, &(lower, upper)) in x.iter().zip(&self.bounds) {
            values.extend([lower - v, v - upper]);
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::{CASES, Stop};

    #[test]
    fn migration_comparison_preserves_quality_budgets_and_stops() {
        for ((case, _), (basin_calls, constraints)) in
            CASES.into_iter().zip([(50, 50), (200, 201), (61, 61)])
        {
            let basin = case.solve_basin();
            let reference = case.solve_reference();
            case.verify(&basin);
            case.verify(&reference);
            assert_eq!(basin.objective_calls, basin_calls);
            assert_eq!(basin.constraint_calls, constraints);
            let stop = if basin_calls == case.budget() {
                Stop::Budget
            } else {
                Stop::Radius
            };
            assert_eq!(basin.stop, stop);
            if reference.stop == Stop::Budget {
                assert_eq!(reference.objective_calls, case.budget());
            }
        }
    }
}
