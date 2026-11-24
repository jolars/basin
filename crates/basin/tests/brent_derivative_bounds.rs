use basin::{
    BoxConstraints, BrentDerivative, CostFunction, Gradient, Problem, Scalar,
    ScalarGradientState, Solver, State,
};
use std::convert::Infallible;

struct Linear<F> {
    lower: F,
    upper: F,
}

impl<F: Scalar> CostFunction for Linear<F> {
    type Param = F;
    type Output = F;
    type Error = Infallible;

    fn cost(&self, x: &F) -> Result<F, Infallible> {
        assert!(*x >= self.lower && *x <= self.upper);
        Ok(*x)
    }
}

impl<F: Scalar> Gradient for Linear<F> {
    type Gradient = F;

    fn gradient(&self, x: &F) -> Result<F, Infallible> {
        assert!(*x >= self.lower && *x <= self.upper);
        Ok(F::one())
    }
}

impl<F: Scalar> BoxConstraints for Linear<F> {
    fn lower(&self) -> &F {
        &self.lower
    }
    fn upper(&self) -> &F {
        &self.upper
    }
}

fn brent_domain<F: Scalar>() {
    let mut problem = Problem::new(Linear {
        lower: F::zero(),
        upper: F::one(),
    });
    let mut solver =
        BrentDerivative::new().with_absolute_position_tolerance(F::one());
    let mut state = solver
        .init(
            &mut problem,
            ScalarGradientState::new(F::from_f64(0.5).unwrap()),
        )
        .unwrap();
    // A composed policy can suppress the native width stop and request more steps.
    for _ in 0..5 {
        let step = solver.next_iter(&mut problem, state).unwrap();
        state = step.0;
        assert!(*state.param() >= F::zero() && *state.param() <= F::one());
        if step.1.is_some() {
            break;
        }
    }
}

#[test]
fn brent_trials_remain_inside_the_domain_when_width_stopping_is_suppressed() {
    brent_domain::<f64>();
    brent_domain::<f32>();
}

struct Quartic;

impl CostFunction for Quartic {
    type Param = f64;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &f64) -> Result<f64, Infallible> {
        assert!((0.0..=1.0).contains(x));
        Ok((x - 0.5).powi(4))
    }
}

impl Gradient for Quartic {
    type Gradient = f64;

    fn gradient(&self, x: &f64) -> Result<f64, Infallible> {
        Ok(4.0 * (x - 0.5).powi(3))
    }
}

impl BoxConstraints for Quartic {
    fn lower(&self) -> &f64 {
        &0.0
    }

    fn upper(&self) -> &f64 {
        &1.0
    }
}

#[test]
fn brent_shrinks_the_bracket_when_the_minimum_step_reaches_an_endpoint() {
    let result = basin::Executor::new(
        Quartic,
        BrentDerivative::new(),
        ScalarGradientState::new(0.25),
    )
    .max_iter(100)
    .run()
    .unwrap();

    assert_eq!(result.reason, basin::TerminationReason::SolverConverged);
    assert!((result.best_param() - 0.5).abs() < 1e-8);
}
