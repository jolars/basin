//! SLSQP comparison on unconstrained Rosenbrock with analytic derivatives.
//!
//! All three implementations record objective improvements at callback
//! granularity and include solve initialization in the elapsed time. Basin's
//! Kraft accuracy test differs from the references' absolute function-change
//! test; equal numeric tolerances do not imply identical stopping rules.

use std::cell::RefCell;
use std::convert::Infallible;
use std::rc::Rc;
use std::time::Instant;

use basin::problems::{rosenbrock, rosenbrock_gradient};
use basin::{CostFunction, Executor, Gradient, Slsqp, SlsqpState, State};

use crate::unconstrained::Unconstrained;

pub const START: [f64; 2] = [-1.2, 1.0];
pub const ACCURACY: f64 = 1e-10;
pub const BUDGET: usize = 200;

#[derive(Clone, Copy, Debug)]
pub enum Library {
    Basin,
    Slsqp,
    Nlopt,
}

impl Library {
    pub fn name(self) -> &'static str {
        match self {
            Self::Basin => "basin",
            Self::Slsqp => "slsqp",
            Self::Nlopt => "nlopt",
        }
    }
}

pub struct Run {
    pub points: Vec<(u128, f64)>,
    pub x: Vec<f64>,
    pub cost: f64,
    pub cost_evals: usize,
    pub gradient_evals: usize,
    pub status: String,
    pub converged: bool,
}

impl Run {
    /// Reevaluate the returned point outside the timer, independently of the
    /// best trial recorded by the callback. Budget exhaustion is not success.
    pub fn verify(&self) {
        assert!(self.converged, "SLSQP stopped with {}", self.status);
        assert_eq!(self.x.len(), START.len());
        assert!(self.x.iter().all(|x| x.is_finite()));
        let cost = rosenbrock(&self.x);
        assert!(cost.is_finite() && cost < 1e-8, "cost={cost}");
        assert!((cost - self.cost).abs() < 1e-12);
        let mut g = vec![0.0; self.x.len()];
        rosenbrock_gradient(&self.x, &mut g);
        assert!(g.iter().all(|v| v.is_finite() && v.abs() < 1e-3));
        assert!(self.x.iter().all(|x| (x - 1.0).abs() < 1e-4));
    }
}

struct Log {
    start: Instant,
    best: f64,
    points: Vec<(u128, f64)>,
    cost_evals: usize,
    gradient_evals: usize,
}

#[derive(Clone)]
struct Objective(Rc<RefCell<Log>>);

impl CostFunction for Objective {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<f64, Infallible> {
        Ok(self.evaluate(x, None))
    }
}

impl Gradient for Objective {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        let mut g = vec![0.0; x.len()];
        rosenbrock_gradient(x, &mut g);
        self.0.borrow_mut().gradient_evals += 1;
        Ok(g)
    }
}

fn callback(x: &[f64], g: Option<&mut [f64]>, obj: &mut Objective) -> f64 {
    obj.evaluate(x, g)
}

impl Objective {
    fn evaluate(&self, x: &[f64], g: Option<&mut [f64]>) -> f64 {
        let f = rosenbrock(x);
        let mut log = self.0.borrow_mut();
        log.cost_evals += 1;
        if let Some(g) = g {
            rosenbrock_gradient(x, g);
            log.gradient_evals += 1;
        }
        assert!(f.is_finite(), "non-finite SLSQP objective");
        if f < log.best {
            log.best = f;
            let elapsed = log.start.elapsed().as_nanos();
            log.points.push((elapsed, f));
        }
        f
    }
}

pub fn run(library: Library) -> Run {
    let f0 = rosenbrock(&START);
    let log = Rc::new(RefCell::new(Log {
        start: Instant::now(),
        best: f0,
        points: vec![(0, f0)],
        cost_evals: 0,
        gradient_evals: 0,
    }));
    let obj = Objective(Rc::clone(&log));
    let (x, cost, status, converged) = match library {
        Library::Basin => {
            let result = Executor::new(
                Unconstrained(obj),
                Slsqp::new().with_absolute_accuracy_tolerance(ACCURACY),
                SlsqpState::new(START.to_vec()),
            )
            .max_iter(BUDGET as u64)
            .run()
            .unwrap();
            (
                result.state.param().clone(),
                result.state.cost(),
                format!("{:?}", result.reason),
                result.reason == basin::TerminationReason::SolverConverged,
            )
        }
        Library::Slsqp => {
            let constraints: [&dyn slsqp::Func<Objective>; 0] = [];
            let result = slsqp::minimize(
                callback,
                &START,
                &[(f64::NEG_INFINITY, f64::INFINITY); 2],
                &constraints,
                obj,
                BUDGET,
                Some(slsqp::StopTols {
                    ftol_abs: ACCURACY,
                    ..Default::default()
                }),
            );
            match result {
                Ok((status, x, f)) => (
                    x,
                    f,
                    format!("{status:?}"),
                    matches!(status, slsqp::SuccessStatus::FtolReached),
                ),
                Err((status, x, f)) => (x, f, format!("{status:?}"), false),
            }
        }
        Library::Nlopt => {
            let mut opt = nlopt::Nlopt::new(
                nlopt::Algorithm::Slsqp,
                START.len(),
                callback,
                nlopt::Target::Minimize,
                obj,
            );
            opt.set_maxeval(BUDGET as u32).unwrap();
            opt.set_ftol_abs(ACCURACY).unwrap();
            let mut x = START.to_vec();
            let result = opt.optimize(&mut x);
            match result {
                Ok((status, f)) => (
                    x,
                    f,
                    format!("{status:?}"),
                    matches!(status, nlopt::SuccessState::FtolReached),
                ),
                Err((status, f)) => (x, f, format!("{status:?}"), false),
            }
        }
    };
    let mut log = log.borrow_mut();
    // Include the final gradient and termination work after the last improvement.
    let elapsed = log.start.elapsed().as_nanos();
    let best = log.best;
    log.points.push((elapsed, best));
    Run {
        points: std::mem::take(&mut log.points),
        x,
        cost,
        cost_evals: log.cost_evals,
        gradient_evals: log.gradient_evals,
        status,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn references_reach_the_rosenbrock_minimum() {
        for library in [Library::Basin, Library::Slsqp, Library::Nlopt] {
            let result = run(library);
            result.verify();
            assert!(result.cost_evals > 0 && result.gradient_evals > 0);
            assert!(
                result
                    .points
                    .windows(2)
                    .all(|p| { p[0].0 <= p[1].0 && p[0].1 >= p[1].1 })
            );
        }
    }
}
