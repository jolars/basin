//! The three L-BFGS-B 3.0 extended Rosenbrock drivers used by Ariadne.
//! Settings match `crates/theseus/tests/optimizer_reference.rs` at Ariadne
//! commit c8e957cfe58648f07ce4222e4f6e925563d834b2. Objective and gradient
//! evaluations are fused, and each run includes fresh solver initialization.

use basin::{
    BoxConstraints, CostFunction, Executor, Gradient, GradientState,
    LbfgsState, Lbfgsb, OptimizationResult, Solver, State, TerminationReason,
};
use std::convert::Infallible;

pub trait Vector: Clone + 'static {
    fn filled(n: usize, value: f64) -> Self;
    fn as_slice(&self) -> &[f64];
    fn as_mut_slice(&mut self) -> &mut [f64];
}

impl Vector for Vec<f64> {
    fn filled(n: usize, value: f64) -> Self {
        vec![value; n]
    }
    fn as_slice(&self) -> &[f64] {
        self
    }
    fn as_mut_slice(&mut self) -> &mut [f64] {
        self
    }
}

#[cfg(feature = "nalgebra_all")]
impl Vector for crate::backend_aliases::nalgebra::DVector<f64> {
    fn filled(n: usize, value: f64) -> Self {
        Self::from_element(n, value)
    }
    fn as_slice(&self) -> &[f64] {
        self.as_slice()
    }
    fn as_mut_slice(&mut self) -> &mut [f64] {
        self.as_mut_slice()
    }
}

#[cfg(feature = "ndarray_all")]
impl Vector for crate::backend_aliases::ndarray::Array1<f64> {
    fn filled(n: usize, value: f64) -> Self {
        Self::from_elem(n, value)
    }
    fn as_slice(&self) -> &[f64] {
        self.as_slice_memory_order().unwrap()
    }
    fn as_mut_slice(&mut self) -> &mut [f64] {
        self.as_slice_mut().unwrap()
    }
}

#[cfg(feature = "faer_all")]
impl Vector for crate::backend_aliases::faer::Col<f64> {
    fn filled(n: usize, value: f64) -> Self {
        Self::from_fn(n, |_| value)
    }
    fn as_slice(&self) -> &[f64] {
        self.try_as_col_major().unwrap().as_slice()
    }
    fn as_mut_slice(&mut self) -> &mut [f64] {
        self.try_as_col_major_mut().unwrap().as_slice_mut()
    }
}

pub struct Driver<V = Vec<f64>> {
    number: usize,
    pub lower: V,
    pub upper: V,
    pub check_feasibility: bool,
}

impl Driver {
    pub fn new(number: usize) -> Self {
        Self::with_backend(number)
    }
}

impl<V: Vector> Driver<V> {
    pub fn with_backend(number: usize) -> Self {
        assert!((1..=3).contains(&number));
        let n = if number == 3 { 1000 } else { 25 };
        let mut lower = V::filled(n, -100.0);
        for i in (0..n).step_by(2) {
            lower.as_mut_slice()[i] = 1.0;
        }
        Self {
            number,
            lower,
            upper: V::filled(n, 100.0),
            check_feasibility: false,
        }
    }

    pub fn evaluate(&self, x: &[f64], g: &mut [f64]) -> f64 {
        if self.check_feasibility {
            assert!(x.iter().enumerate().all(|(i, &v)| v.is_finite()
                && self.lower.as_slice()[i] <= v
                && v <= self.upper.as_slice()[i]));
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
                    g.max(x - self.upper.as_slice()[i]).abs()
                } else {
                    g.min(x - self.lower.as_slice()[i]).abs()
                }
            })
            .fold(0.0, f64::max)
    }

    pub fn solve(&self) -> OptimizationResult<LbfgsState<V>>
    where
        Lbfgsb: for<'a> Solver<&'a Self, LbfgsState<V>, Error = Infallible>,
    {
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
            LbfgsState::new(V::filled(lower.as_slice().len(), 3.0), history),
        )
        .max_iter(2000)
        .stop_when(move |state: &LbfgsState<V>| {
            let value = state.cost();
            let old = previous.replace(value);
            let pg = state
                .param()
                .as_slice()
                .iter()
                .zip(state.gradient().unwrap().as_slice())
                .enumerate()
                .map(|(i, (&x, &g))| {
                    if g < 0.0 {
                        g.max(x - upper.as_slice()[i]).abs()
                    } else {
                        g.min(x - lower.as_slice()[i]).abs()
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

    pub fn verify(&self, result: &OptimizationResult<LbfgsState<V>>) {
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
        let mut gradient = vec![0.0; self.lower.as_slice().len()];
        // Independent verification calls are outside the solve's counters/timer.
        let cost = self.evaluate(result.param().as_slice(), &mut gradient);
        assert!((cost - objective).abs() <= objective_tolerance);
        assert_eq!(cost, result.cost());
        assert_eq!(&gradient, result.state.gradient().unwrap().as_slice());
        assert!(
            self.projected_gradient(result.param().as_slice(), &gradient)
                <= pg_limit
        );
        assert_eq!(
            result.state.best_param().as_slice(),
            result.param().as_slice()
        );
        assert_eq!(result.state.best_cost(), result.cost());
    }
}

impl<V: Vector> CostFunction for &Driver<V> {
    type Param = V;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &V) -> Result<f64, Infallible> {
        Ok(self.evaluate(x.as_slice(), &mut vec![0.0; x.as_slice().len()]))
    }
}
impl<V: Vector> Gradient for &Driver<V> {
    type Gradient = V;
    fn gradient(&self, x: &V) -> Result<V, Infallible> {
        Ok(self.cost_and_gradient(x)?.1)
    }
    fn cost_and_gradient(&self, x: &V) -> Result<(f64, V), Infallible> {
        let mut gradient = V::filled(x.as_slice().len(), 0.0);
        let cost = self.evaluate(x.as_slice(), gradient.as_mut_slice());
        Ok((cost, gradient))
    }
}
impl<V: Vector> BoxConstraints for &Driver<V> {
    fn lower(&self) -> &V {
        &self.lower
    }
    fn upper(&self) -> &V {
        &self.upper
    }
}
