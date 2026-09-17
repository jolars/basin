//! The three L-BFGS-B 3.0 extended Rosenbrock drivers used by Ariadne.
//! Settings match `crates/theseus/tests/optimizer_reference.rs` at Ariadne
//! commit c8e957cfe58648f07ce4222e4f6e925563d834b2. Objective and gradient
//! evaluations are fused, and each run includes fresh solver initialization.

use basin::{
    BoxConstraints, CostFunction, Executor, Gradient, GradientState,
    LbfgsState, Lbfgsb, OptimizationResult, State, TerminationReason,
};
use std::convert::Infallible;

pub struct Driver {
    number: usize,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub check_feasibility: bool,
}

impl Driver {
    pub fn new(number: usize) -> Self {
        assert!((1..=3).contains(&number));
        let n = if number == 3 { 1000 } else { 25 };
        let mut lower = vec![-100.0; n];
        for i in (0..n).step_by(2) {
            lower[i] = 1.0;
        }
        Self {
            number,
            lower,
            upper: vec![100.0; n],
            check_feasibility: false,
        }
    }

    pub fn evaluate(&self, x: &[f64], g: &mut [f64]) -> f64 {
        if self.check_feasibility {
            assert!(x.iter().enumerate().all(|(i, &v)| v.is_finite()
                && self.lower[i] <= v
                && v <= self.upper[i]));
        }
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

    pub fn projected_gradient(&self, x: &[f64], g: &[f64]) -> f64 {
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

    pub fn solve(&self) -> OptimizationResult<LbfgsState<Vec<f64>>> {
        let first = self.number == 1;
        let evaluation_limit = if self.number == 3 { 900 } else { 99 };
        let history = if self.number == 3 { 10 } else { 5 };
        let solver = Lbfgsb::new()
            .with_absolute_projected_gradient_tolerance(first.then_some(1e-5));
        let mut previous: Option<f64> = None;
        let lower = self.lower.clone();
        let upper = self.upper.clone();
        Executor::new(
            self,
            solver,
            LbfgsState::new(vec![3.0; lower.len()], history),
        )
        .max_iter(2000)
        .stop_when(move |state: &LbfgsState<Vec<f64>>| {
            let value = state.cost();
            let old = previous.replace(value);
            let pg = state
                .param()
                .iter()
                .zip(state.gradient().unwrap())
                .enumerate()
                .map(|(i, (&x, &g))| {
                    if g < 0.0 {
                        g.max(x - upper[i]).abs()
                    } else {
                        g.min(x - lower[i]).abs()
                    }
                })
                .fold(0.0, f64::max);
            if !first
                && state.iter() > 0
                && (state.cost_evals() >= evaluation_limit
                    || pg <= 1e-10 * (1.0 + value.abs()))
            {
                return Some(TerminationReason::UserRequested);
            }
            if first
                && old.is_some_and(|old| {
                    old - value
                        <= 1e7
                            * f64::EPSILON
                            * old.abs().max(value.abs()).max(1.0)
                })
            {
                return Some(TerminationReason::RelativeCostTolerance);
            }
            None
        })
        .run()
        .unwrap()
    }

    pub fn verify(&self, result: &OptimizationResult<LbfgsState<Vec<f64>>>) {
        let (iterations, evaluations, objective, objective_tolerance, pg_limit) =
            match self.number {
                1 => (23, 28, 1.08349008e-9, 1e-16, 1.8e-4),
                2 => (46, 53, 5.80702313e-15, 1e-20, 1e-10),
                3 => (49, 58, 5.35e-22, 1e-22, 1e-10),
                _ => unreachable!(),
            };
        assert_eq!(result.iter(), iterations);
        assert_eq!(result.cost_evals(), evaluations);
        assert_eq!(result.state.gradient_evals(), evaluations);
        assert_eq!(
            result.reason,
            if self.number == 1 {
                TerminationReason::RelativeCostTolerance
            } else {
                TerminationReason::UserRequested
            }
        );
        let mut gradient = vec![0.0; self.lower.len()];
        // Independent verification calls are outside the solve's counters/timer.
        let cost = self.evaluate(result.param(), &mut gradient);
        assert!((cost - objective).abs() <= objective_tolerance);
        assert_eq!(cost, result.cost());
        assert_eq!(&gradient, result.state.gradient().unwrap());
        assert!(self.projected_gradient(result.param(), &gradient) <= pg_limit);
        assert_eq!(result.state.best_param(), result.param());
        assert_eq!(result.state.best_cost(), result.cost());
    }
}

impl CostFunction for &Driver {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(self.evaluate(x, &mut vec![0.0; x.len()]))
    }
}
impl Gradient for &Driver {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(self.cost_and_gradient(x)?.1)
    }
    fn cost_and_gradient(
        &self,
        x: &Vec<f64>,
    ) -> Result<(f64, Vec<f64>), Infallible> {
        let mut gradient = vec![0.0; x.len()];
        let cost = self.evaluate(x, &mut gradient);
        Ok((cost, gradient))
    }
}
impl BoxConstraints for &Driver {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}
