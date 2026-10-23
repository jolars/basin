//! SLSQP comparisons on Rosenbrock and a nonlinear constrained quadratic.
//!
//! All three implementations record objective improvements at callback
//! granularity and include solve initialization in the elapsed time. Basin's
//! Kraft accuracy test differs from the references' absolute function-change
//! test; equal numeric tolerances do not imply identical stopping rules.

pub mod workloads;

use std::cell::RefCell;
use std::convert::Infallible;
use std::rc::Rc;
use std::time::Instant;

use basin::problems::{rosenbrock, rosenbrock_gradient};
use basin::{
    CostFunction, CountsMirror, Executor, Gradient, Problem,
    SelectedFirstOrderState, Slsqp, Solver, State, TerminationCode,
};

pub const START: [f64; 2] = [-1.2, 1.0];
pub const ACCURACY: f64 = 1e-10;
pub const BUDGET: usize = 200;

#[derive(Clone, Copy, Debug)]
pub enum Library {
    Basin,
    BasinManual,
    Slsqp,
    Nlopt,
}

impl Library {
    pub fn name(self) -> &'static str {
        match self {
            Self::Basin => "basin",
            Self::BasinManual => "basin_manual",
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
    pub iterations: Option<u64>,
    pub status: String,
    pub converged: bool,
    pub constraint_evals: usize,
    pub violations: Vec<(u128, f64)>,
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

impl Run {
    pub fn verify_constrained(&self) -> bool {
        let n = self.x.len();
        let violation = (self.x.iter().map(|v| v * v).sum::<f64>() / n as f64
            - 0.5625)
            .max(0.0);
        self.x
            .iter()
            .all(|v| v.is_finite() && (v - 0.75).abs() < 1e-4)
            && self.cost.is_finite()
            && (quadratic(&self.x) - self.cost).abs() < 1e-10
            && (self.cost - n as f64 / 16.0).abs() < 1e-6
            && violation <= 1e-8
    }
}

struct Log {
    start: Instant,
    best: f64,
    points: Vec<(u128, f64)>,
    cost_evals: usize,
    gradient_evals: usize,
    constraint_evals: usize,
    trials: Vec<(u128, f64, Vec<f64>)>,
}

#[derive(Clone)]
struct Objective(Rc<RefCell<Log>>, bool);

impl basin::Constraints for Objective {
    type Set = basin::ConstraintSet<
        basin::NoBounds,
        basin::NoLinearConstraints,
        basin::NonlinearBlocks<basin::NoNonlinearBlock, basin::NonlinearBlock>,
    >;
    fn constraints(
        &self,
    ) -> <Self::Set as basin::ConstraintDescription>::Borrowed<'_> {
        basin::ConstraintSet::new()
            .with_nonlinear_inequalities(usize::from(self.1))
    }
}

fn quadratic(x: &[f64]) -> f64 {
    x.iter().map(|v| (v - 1.0).powi(2)).sum::<f64>()
        + x.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum::<f64>()
}

fn quadratic_gradient(x: &[f64], g: &mut [f64]) {
    for i in 0..x.len() {
        g[i] = 2.0 * (x[i] - 1.0);
        if i > 0 {
            g[i] += 2.0 * (x[i] - x[i - 1]);
        }
        if i + 1 < x.len() {
            g[i] += 2.0 * (x[i] - x[i + 1]);
        }
    }
}

fn ball(x: &[f64], g: Option<&mut [f64]>, obj: &mut Objective) -> f64 {
    obj.0.borrow_mut().constraint_evals += 1;
    if let Some(g) = g {
        for (g, v) in g.iter_mut().zip(x) {
            *g = 2.0 * v / x.len() as f64;
        }
    }
    x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 - 0.5625
}

impl basin::NonlinearInequalityValues for Objective {
    type Param = Vec<f64>;
    type Values = Vec<f64>;
    type Error = Infallible;
    fn nonlinear_inequalities(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, basin::ConstraintSolveError<Infallible>> {
        Ok(if self.1 {
            vec![ball(x, None, &mut self.clone())]
        } else {
            vec![]
        })
    }
}

impl basin::ConstraintJacobian for Objective {
    type Param = Vec<f64>;
    type Jacobian = basin::DenseMatrix;
    type Error = Infallible;
    fn constraint_jacobian(
        &self,
        x: &Vec<f64>,
    ) -> Result<Self::Jacobian, basin::ConstraintSolveError<Infallible>> {
        Ok(basin::DenseMatrix::from_row_slice(
            usize::from(self.1),
            x.len(),
            &if self.1 {
                x.iter().map(|v| 2.0 * v / x.len() as f64).collect()
            } else {
                vec![]
            },
        ))
    }
}
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
        if self.1 {
            quadratic_gradient(x, &mut g);
        } else {
            rosenbrock_gradient(x, &mut g);
        }
        self.0.borrow_mut().gradient_evals += 1;
        Ok(g)
    }
}

fn callback(x: &[f64], g: Option<&mut [f64]>, obj: &mut Objective) -> f64 {
    obj.evaluate(x, g)
}

impl Objective {
    fn evaluate(&self, x: &[f64], g: Option<&mut [f64]>) -> f64 {
        let f = if self.1 { quadratic(x) } else { rosenbrock(x) };
        let mut log = self.0.borrow_mut();
        log.cost_evals += 1;
        if let Some(g) = g {
            if self.1 {
                quadratic_gradient(x, g);
            } else {
                rosenbrock_gradient(x, g);
            }
            log.gradient_evals += 1;
        }
        assert!(f.is_finite(), "non-finite SLSQP objective");
        let time = log.start.elapsed().as_nanos();
        log.trials.push((time, f, x.to_vec()));
        if f < log.best {
            log.best = f;
            let elapsed = log.start.elapsed().as_nanos();
            log.points.push((elapsed, f));
        }
        f
    }
}

fn manual(
    obj: Objective,
    start: Vec<f64>,
) -> (SelectedFirstOrderState<Vec<f64>>, TerminationCode) {
    let mut problem = Problem::new(obj);
    let mut solver = Slsqp::new().with_absolute_accuracy_tolerance(ACCURACY);
    let mut state = solver
        .init(&mut problem, SelectedFirstOrderState::new(start))
        .unwrap();
    state.mirror(problem.counts());
    state.update_best();
    loop {
        if state.iter() >= BUDGET as u64 {
            return (state, TerminationCode::MaxIter);
        }
        if let Some(reason) = solver.check_convergence(&problem, &state) {
            return (state, reason.code());
        }
        let (mut next, completed, reason) =
            solver.next_iter(&mut problem, state).unwrap().into_parts();
        next.mirror(problem.counts());
        if completed {
            next.increment_iter();
        }
        if let Some(reason) = reason {
            next.update_best();
            return (next, reason.code());
        }
        next.update_best();
        state = next;
    }
}

pub fn run(library: Library) -> Run {
    run_case(library, 2, false)
}

pub fn run_constrained(library: Library, n: usize) -> Run {
    run_case(library, n, true)
}

fn run_case(library: Library, n: usize, constrained: bool) -> Run {
    let start = if constrained {
        (0..n).map(|i| 0.3 + 0.4 * i as f64 / n as f64).collect()
    } else {
        START.to_vec()
    };
    let f0 = if constrained {
        quadratic(&start)
    } else {
        rosenbrock(&start)
    };
    let log = Rc::new(RefCell::new(Log {
        start: Instant::now(),
        best: f0,
        points: vec![(0, f0)],
        cost_evals: 0,
        gradient_evals: 0,
        constraint_evals: 0,
        trials: Vec::new(),
    }));
    let obj = Objective(Rc::clone(&log), constrained);
    let mut iterations = None;
    let (x, cost, status, converged) = match library {
        Library::Basin | Library::BasinManual => {
            let (state, reason) = if matches!(library, Library::BasinManual) {
                manual(obj, start.clone())
            } else {
                let result = Executor::new(
                    obj,
                    Slsqp::new().with_absolute_accuracy_tolerance(ACCURACY),
                    SelectedFirstOrderState::new(start.clone()),
                )
                .max_iter(BUDGET as u64)
                .run()
                .unwrap();
                (result.state, result.report.code())
            };
            iterations = Some(state.iter());
            (
                state.param().clone(),
                state.cost(),
                format!("{reason:?}"),
                reason == TerminationCode::SolverConverged,
            )
        }
        Library::Slsqp => {
            let constraint: &dyn slsqp::Func<Objective> = &ball;
            let constraints = if constrained {
                vec![constraint]
            } else {
                vec![]
            };
            let result = slsqp::minimize(
                callback,
                &start,
                &vec![(f64::NEG_INFINITY, f64::INFINITY); n],
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
                n,
                callback,
                nlopt::Target::Minimize,
                obj.clone(),
            );
            if constrained {
                opt.add_inequality_constraint(ball, obj, 1e-8).unwrap();
            }
            opt.set_maxeval(BUDGET as u32).unwrap();
            opt.set_ftol_abs(ACCURACY).unwrap();
            let mut x = start;
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
    let mut violations = Vec::new();
    if constrained {
        let mut best = f64::INFINITY;
        let mut points = Vec::new();
        for (time, f, x) in &log.trials {
            let violation = (x.iter().map(|v| v * v).sum::<f64>() / n as f64
                - 0.5625)
                .max(0.0);
            violations.push((*time, violation));
            if violation <= 1e-8 {
                best = best.min(*f);
            }
            if best.is_finite() {
                points.push((*time, best));
            }
        }
        if best.is_finite() {
            points.push((elapsed, best));
        }
        violations.push((
            elapsed,
            (x.iter().map(|v| v * v).sum::<f64>() / n as f64 - 0.5625).max(0.0),
        ));
        log.points = points;
    }
    Run {
        constraint_evals: log.constraint_evals,
        violations,
        points: std::mem::take(&mut log.points),
        x,
        cost,
        cost_evals: log.cost_evals,
        gradient_evals: log.gradient_evals,
        iterations,
        status,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constrained_derivatives_and_known_optimum_agree() {
        let x = vec![0.3, 0.6, 0.5, 0.8];
        let mut gradient = vec![0.0; x.len()];
        quadratic_gradient(&x, &mut gradient);
        for col in 0..x.len() {
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus[col] += 1e-6;
            minus[col] -= 1e-6;
            let numeric = (quadratic(&plus) - quadratic(&minus)) / 2e-6;
            assert!((numeric - gradient[col]).abs() < 1e-8);
        }
        assert_eq!(quadratic(&[0.75; 100]), 6.25);
        for library in [Library::Basin, Library::Slsqp, Library::Nlopt] {
            let result = run_constrained(library, 100);
            assert!(
                result.verify_constrained(),
                "{}: {}",
                library.name(),
                result.status
            );
            assert!(result.constraint_evals > 0);
            assert!(result.violations.iter().any(|p| p.1 > 1e-8));
            assert!(
                result
                    .points
                    .windows(2)
                    .all(|p| p[0].0 <= p[1].0 && p[0].1 >= p[1].1)
            );
        }
    }

    #[test]
    fn manual_loop_matches_executor_work_and_trace() {
        let executor = run(Library::Basin);
        let manual = run(Library::BasinManual);
        executor.verify();
        manual.verify();
        assert_eq!(executor.x, manual.x);
        assert_eq!(executor.cost, manual.cost);
        assert_eq!(executor.cost_evals, manual.cost_evals);
        assert_eq!(executor.gradient_evals, manual.gradient_evals);
        assert_eq!(executor.iterations, Some(35));
        assert_eq!(executor.iterations, manual.iterations);
        assert_eq!(executor.status, manual.status);
        assert_eq!(
            executor.points.iter().map(|p| p.1).collect::<Vec<_>>(),
            manual.points.iter().map(|p| p.1).collect::<Vec<_>>(),
        );
    }

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
