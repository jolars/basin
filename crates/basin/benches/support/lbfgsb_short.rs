//! Short L-BFGS-B 3.0 cases from Ariadne commit c8e957c.
//!
//! The mixed-bound quadratic and fixed-variable/history-rollover quadratic
//! are described in `crates/lbfgsb/tests/reference/PROVENANCE.md` there. The
//! analytic solutions below provide independent checks without a native solver.

use super::Vector;
use basin::{
    BoxConstraints, CostFunction, Executor, Gradient, GradientState,
    LbfgsState, Lbfgsb, OptimizationResult, Solver, State, Stepper,
    TerminationReason,
};
use std::convert::Infallible;

#[derive(Clone, Copy, Debug)]
pub enum Case {
    Mixed,
    Rollover,
}

impl Case {
    pub fn name(self) -> &'static str {
        match self {
            Self::Mixed => "mixed",
            Self::Rollover => "rollover",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Adapter {
    Reference,
    Trimmed,
    RequiredStops,
}

impl Adapter {
    pub fn name(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Trimmed => "trimmed",
            Self::RequiredStops => "required_stops",
        }
    }
}

pub struct Short<V = Vec<f64>> {
    case: Case,
    start: V,
    gradient_tolerance: f64,
    lower: V,
    upper: V,
    pub check_feasibility: bool,
}

impl<V: Vector> Short<V> {
    pub fn new(case: Case) -> Self {
        let (start, lower, upper): (&[f64], &[f64], &[f64]) = match case {
            Case::Mixed => (
                &[-3.0, 9.0, 7.0, -8.0],
                &[0.0, -1.0, f64::NEG_INFINITY, 0.0],
                &[2.0, 4.0, f64::INFINITY, f64::INFINITY],
            ),
            Case::Rollover => (
                &[-3.0, 8.0, 4.0, -6.0, 9.0, 0.0, 0.0, 0.0],
                &[
                    0.0,
                    0.0,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                    1.5,
                    -1.0,
                    -0.5,
                    f64::NEG_INFINITY,
                ],
                &[
                    2.0,
                    f64::INFINITY,
                    2.0,
                    f64::INFINITY,
                    1.5,
                    1.0,
                    f64::INFINITY,
                    0.5,
                ],
            ),
        };
        let vector = |values: &[f64]| {
            let mut result = V::filled(values.len(), 0.0);
            result.as_mut_slice().copy_from_slice(values);
            result
        };
        Self {
            case,
            gradient_tolerance: match case {
                Case::Mixed => 1e-12,
                Case::Rollover => 1e-10,
            },
            start: vector(start),
            lower: vector(lower),
            upper: vector(upper),
            check_feasibility: false,
        }
    }

    fn evaluate(&self, x: &[f64], g: &mut [f64]) -> f64 {
        if self.check_feasibility {
            self.verify_feasibility(x);
        }
        let (target, scale): (&[f64], &[f64]) = match self.case {
            Case::Mixed => (&[1.0, 2.0, -2.0, 3.0], &[2.0; 4]),
            Case::Rollover => (
                &[1.0, -2.0, 4.0, -1.0, 9.0, 2.0, -1.0, 1.0],
                &[1.0, 50.0, 0.1, 20.0, 1.0, 0.5, 0.5, 0.5],
            ),
        };
        let mut value = 0.0;
        for i in 0..x.len() {
            let d = x[i] - target[i];
            value += 0.5 * scale[i] * d * d;
            g[i] = scale[i] * d;
        }
        value
    }

    fn verify_feasibility(&self, x: &[f64]) {
        assert_eq!(x.len(), self.start.as_slice().len());
        assert!(x.iter().enumerate().all(|(i, &v)| v.is_finite()
            && self.lower.as_slice()[i] <= v
            && v <= self.upper.as_slice()[i]));
    }

    fn executor(
        &self,
        adapter: Adapter,
    ) -> Executor<&Self, LbfgsState<V>, Lbfgsb>
    where
        Lbfgsb: for<'a> Solver<&'a Self, LbfgsState<V>, Error = Infallible>,
    {
        let (history, cost_tolerance) = match self.case {
            Case::Mixed => (5, 0.0),
            Case::Rollover => (2, 1e7 * f64::EPSILON),
        };
        let gradient_tolerance = self.gradient_tolerance;
        let custom_stop = gradient_tolerance == 0.0;
        let solver = Lbfgsb::new().with_absolute_projected_gradient_tolerance(
            Some(gradient_tolerance),
        );
        let bounds = matches!(adapter, Adapter::Reference)
            .then(|| (self.lower.clone(), self.upper.clone()));
        let mut previous: Option<f64> = None;
        let executor = Executor::new(
            self,
            solver,
            LbfgsState::new(self.start.clone(), history),
        )
        .max_iter(2000);
        // A disabled application stop still allocates a closure and dispatches
        // it at every boundary. Let the solver handle this case on its own.
        if matches!(adapter, Adapter::RequiredStops)
            && !custom_stop
            && cost_tolerance == 0.0
        {
            return executor;
        }
        executor.stop_when(move |state: &LbfgsState<V>| {
            // The original adapter computes this even when the solver
            // already owns the projected-gradient convergence test.
            let pg = if let Some((lower, upper)) = &bounds {
                projected_gradient(
                    state.param().as_slice(),
                    state.gradient().unwrap().as_slice(),
                    lower.as_slice(),
                    upper.as_slice(),
                )
            } else {
                0.0
            };
            let value = state.cost();
            let old = previous.replace(value);
            if custom_stop
                && state.iter() > 0
                && (state.cost_evals() >= 1000
                    || pg <= 1e-10 * (1.0 + value.abs()))
            {
                return Some(TerminationReason::UserRequested);
            }
            if cost_tolerance > 0.0
                && old.is_some_and(|old| {
                    old - value
                        <= cost_tolerance * old.abs().max(value.abs()).max(1.0)
                })
            {
                Some(TerminationReason::RelativeCostTolerance)
            } else {
                None
            }
        })
    }

    pub fn initialize(
        &self,
        adapter: Adapter,
    ) -> Stepper<&Self, LbfgsState<V>, Lbfgsb>
    where
        Lbfgsb: for<'a> Solver<&'a Self, LbfgsState<V>, Error = Infallible>,
    {
        self.executor(adapter).into_stepper().unwrap()
    }

    pub fn solve(&self, adapter: Adapter) -> OptimizationResult<LbfgsState<V>>
    where
        Lbfgsb: for<'a> Solver<&'a Self, LbfgsState<V>, Error = Infallible>,
    {
        self.executor(adapter).run().unwrap()
    }

    /// Match an adapter that returns an owned point and discards solver state.
    pub fn extract(
        &self,
        result: &OptimizationResult<LbfgsState<V>>,
    ) -> (V, f64, f64) {
        let pg = projected_gradient(
            result.param().as_slice(),
            result.state.gradient().unwrap().as_slice(),
            self.lower.as_slice(),
            self.upper.as_slice(),
        );
        (result.param().clone(), result.cost(), pg)
    }

    pub fn verify(&self, result: &OptimizationResult<LbfgsState<V>>) {
        let (iterations, evaluations, target, objective): (_, _, &[f64], _) =
            match self.case {
                Case::Mixed => (2, 3, &[1.0, 2.0, -2.0, 3.0], 0.0),
                Case::Rollover => {
                    (8, 11, &[1.0, 0.0, 2.0, -1.0, 1.5, 1.0, -0.5, 0.5], 128.7)
                }
            };
        assert_eq!(result.iter(), iterations);
        assert_eq!(result.cost_evals(), evaluations);
        assert_eq!(result.state.gradient_evals(), evaluations);
        assert_eq!(
            result.reason,
            TerminationReason::ProjectedGradientTolerance
        );
        self.verify_feasibility(result.param().as_slice());
        for (&value, &expected) in result.param().as_slice().iter().zip(target)
        {
            assert!((value - expected).abs() <= 1e-12);
        }
        // These verification calls are outside the solve's timer and counters.
        let mut gradient = vec![0.0; target.len()];
        let value = self.evaluate(result.param().as_slice(), &mut gradient);
        assert!((value - objective).abs() <= 1e-12);
        assert_eq!(value, result.cost());
        assert_eq!(gradient, result.state.gradient().unwrap().as_slice());
        assert!(
            projected_gradient(
                result.param().as_slice(),
                &gradient,
                self.lower.as_slice(),
                self.upper.as_slice()
            ) <= 1e-10
        );
        assert_eq!(result.best_param().as_slice(), result.param().as_slice());
        assert_eq!(result.best_cost(), result.cost());
    }
}

fn projected_gradient(
    x: &[f64],
    g: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> f64 {
    x.iter()
        .zip(g)
        .enumerate()
        .map(|(i, (&x, &g))| {
            if g < 0.0 {
                g.max(x - upper[i]).abs()
            } else {
                g.min(x - lower[i]).abs()
            }
        })
        .fold(0.0, f64::max)
}

impl<V: Vector> CostFunction for &Short<V> {
    type Param = V;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &V) -> Result<f64, Infallible> {
        Ok(self.evaluate(x.as_slice(), &mut vec![0.0; x.as_slice().len()]))
    }
}
impl<V: Vector> Gradient for &Short<V> {
    type Gradient = V;
    fn gradient(&self, x: &V) -> Result<V, Infallible> {
        Ok(self.cost_and_gradient(x)?.1)
    }
    fn cost_and_gradient(&self, x: &V) -> Result<(f64, V), Infallible> {
        let mut gradient = V::filled(x.as_slice().len(), 0.0);
        let value = self.evaluate(x.as_slice(), gradient.as_mut_slice());
        Ok((value, gradient))
    }
}
impl<V: Vector> BoxConstraints for &Short<V> {
    fn lower(&self) -> &V {
        &self.lower
    }
    fn upper(&self) -> &V {
        &self.upper
    }
}
