use std::convert::Infallible;

use basin::solver::lbfgs::{Lbfgs, Unbounded};
use basin::{
    CostFunction, Executor, Gradient, HagerZhang, LbfgsState, LineSearch,
    LineSearchOutcome, Problem, TerminationReason,
};

struct ShiftedQuadratic {
    center: f64,
}

impl CostFunction for ShiftedQuadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        Ok((x[0] - self.center).powi(2))
    }
}

impl Gradient for ShiftedQuadratic {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(vec![2.0 * (x[0] - self.center)])
    }
}

#[test]
fn shifted_quadratic_finds_stationary_step() {
    let mut problem = Problem::new(ShiftedQuadratic { center: 3.0 });
    let mut search = HagerZhang::new();

    let alpha = search
        .next(&mut problem, &vec![0.0], 9.0, &vec![-6.0], &vec![6.0])
        .unwrap();

    assert!((alpha - 0.5).abs() < 1e-12, "alpha = {alpha}");
    assert_eq!(problem.counts().cost_evals, 2);
    assert_eq!(problem.counts().gradient_evals, 2);
}

#[test]
fn expands_before_bracketing_a_distant_minimum() {
    let mut problem = Problem::new(ShiftedQuadratic { center: 10.0 });
    let mut search = HagerZhang::new();

    let alpha = search
        .next(&mut problem, &vec![0.0], 100.0, &vec![-20.0], &vec![1.0])
        .unwrap();

    assert!(alpha > 0.0);
    let phi = (alpha - 10.0).powi(2);
    let dphi = 2.0 * (alpha - 10.0);
    assert!(phi - 100.0 <= 0.1 * alpha * -20.0);
    assert!(dphi >= 0.9 * -20.0);
}

struct ShiftedQuartic;

impl CostFunction for ShiftedQuartic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        Ok((x[0] - 10.0).powi(4))
    }
}

impl Gradient for ShiftedQuartic {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(vec![4.0 * (x[0] - 10.0).powi(3)])
    }
}

#[test]
fn expansion_and_secant_refinement_handle_a_nonlinear_profile() {
    let mut problem = Problem::new(ShiftedQuartic);
    let mut search = HagerZhang::new().with_wolfe_coefficients(0.1, 0.1);
    let alpha = search
        .next(
            &mut problem,
            &vec![0.0],
            10_000.0,
            &vec![-4_000.0],
            &vec![1.0],
        )
        .unwrap();

    let phi = (alpha - 10.0).powi(4);
    let dphi = 4.0 * (alpha - 10.0).powi(3);
    assert!(alpha > 5.0, "alpha = {alpha}");
    assert!(phi - 10_000.0 <= 0.1 * alpha * -4_000.0);
    assert!(dphi >= 0.1 * -4_000.0);
    assert!(problem.counts().cost_evals >= 3);
}

struct CancellationProneQuadratic;

impl CostFunction for CancellationProneQuadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        // The expanded quadratic is deliberately added to a scale at which
        // its decrease rounds away. Its derivative remains informative.
        Ok(1e20 + (1.0 - 2.0 * x[0]) + x[0] * x[0])
    }
}

impl Gradient for CancellationProneQuadratic {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(vec![2.0 * x[0] - 2.0])
    }
}

#[test]
fn approximate_wolfe_accepts_when_armijo_decrease_rounds_away() {
    let mut problem = Problem::new(CancellationProneQuadratic);
    let mut search = HagerZhang::new();
    let alpha = search
        .next(
            &mut problem,
            &vec![0.0],
            1e20 + 1.0,
            &vec![-2.0],
            &vec![1.0],
        )
        .unwrap();

    assert_eq!(alpha, 1.0);
    // Both costs round to 1e20, so ordinary Armijo cannot hold for a
    // negative initial slope. The accepted unit step therefore exercises
    // the approximate-Wolfe branch.
    assert!(0.0 > 0.1 * alpha * -2.0);
    assert_eq!(problem.counts().cost_evals, 1);
    assert_eq!(problem.counts().gradient_evals, 1);
}

struct Linear;

impl CostFunction for Linear {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        Ok(-x[0])
    }
}

impl Gradient for Linear {
    type Gradient = Vec<f64>;

    fn gradient(&self, _x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(vec![-1.0])
    }
}

#[test]
fn returns_zero_when_the_evaluation_budget_is_exhausted() {
    let mut problem = Problem::new(Linear);
    let mut search = HagerZhang::new().maxfev(3);

    let alpha = search
        .next(&mut problem, &vec![0.0], 0.0, &vec![-1.0], &vec![1.0])
        .unwrap();

    assert_eq!(alpha, 0.0);
    assert_eq!(problem.counts().cost_evals, 3);
    assert_eq!(problem.counts().gradient_evals, 3);
}

#[test]
fn reports_failure_when_the_evaluation_budget_is_exhausted() {
    let mut problem = Problem::new(Linear);
    let mut search = HagerZhang::new().maxfev(3);

    let outcome = search
        .next_with_outcome(
            &mut problem,
            &vec![0.0],
            0.0,
            &vec![-1.0],
            &vec![1.0],
        )
        .unwrap();

    assert_eq!(outcome, LineSearchOutcome::Failed);
    assert_eq!(problem.counts().cost_evals, 3);
    assert_eq!(problem.counts().gradient_evals, 3);
}

#[test]
fn invalid_initial_slopes_do_not_probe_the_problem() {
    for slope in [0.0, 1.0, f64::NAN] {
        let mut problem = Problem::new(Linear);
        let mut search = HagerZhang::new();
        let alpha = search
            .next(&mut problem, &vec![0.0], 0.0, &vec![slope], &vec![1.0])
            .unwrap();

        assert_eq!(alpha, 0.0);
        assert_eq!(problem.counts().cost_evals, 0);
        assert_eq!(problem.counts().gradient_evals, 0);
    }
}

#[test]
fn non_finite_initial_costs_do_not_probe_the_problem() {
    for cost in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut problem = Problem::new(Linear);
        let mut search = HagerZhang::new();
        let alpha = search
            .next(&mut problem, &vec![0.0], cost, &vec![-1.0], &vec![1.0])
            .unwrap();

        assert_eq!(alpha, 0.0);
        assert_eq!(problem.counts().cost_evals, 0);
        assert_eq!(problem.counts().gradient_evals, 0);
    }
}

#[test]
fn invalid_public_configuration_does_not_probe_the_problem() {
    let mut problem = Problem::new(Linear);
    let mut search = HagerZhang::new();
    search.delta = f64::NAN;

    let alpha = search
        .next(&mut problem, &vec![0.0], 0.0, &vec![-1.0], &vec![1.0])
        .unwrap();

    assert_eq!(alpha, 0.0);
    assert_eq!(problem.counts().cost_evals, 0);
    assert_eq!(problem.counts().gradient_evals, 0);
}

#[test]
fn zero_evaluation_budget_does_not_probe_the_problem() {
    let mut problem = Problem::new(Linear);
    let mut search = HagerZhang::new().maxfev(0);
    let alpha = search
        .next(&mut problem, &vec![0.0], 0.0, &vec![-1.0], &vec![1.0])
        .unwrap();

    assert_eq!(alpha, 0.0);
    assert_eq!(problem.counts().cost_evals, 0);
    assert_eq!(problem.counts().gradient_evals, 0);
}

#[test]
fn defaults_and_builders_expose_the_reference_parameters() {
    let search = HagerZhang::new()
        .with_wolfe_coefficients(0.2, 0.8)
        .with_relative_cost_relaxation_tolerance(1e-8)
        .theta(0.4)
        .gamma(0.7)
        .alpha_init(0.75)
        .bounds(1e-12, 100.0)
        .rho(4.0)
        .maxfev(40);

    assert_eq!(search.delta, 0.2);
    assert_eq!(search.sigma, 0.8);
    assert_eq!(search.epsilon, 1e-8);
    assert_eq!(search.theta, 0.4);
    assert_eq!(search.gamma, 0.7);
    assert_eq!(search.alpha_init, 0.75);
    assert_eq!(search.step_min, 1e-12);
    assert_eq!(search.step_max, 100.0);
    assert_eq!(search.rho, 4.0);
    assert_eq!(search.maxfev, 40);

    let defaults = HagerZhang::<f64>::default();
    assert_eq!(defaults.delta, 0.1);
    assert_eq!(defaults.sigma, 0.9);
    assert_eq!(defaults.epsilon, 1e-6);
    assert_eq!(defaults.theta, 0.5);
    assert_eq!(defaults.gamma, 0.66);
    assert_eq!(defaults.alpha_init, 1.0);
    assert_eq!(defaults.step_min, f64::EPSILON);
    assert_eq!(defaults.step_max, 1e5);
    assert_eq!(defaults.rho, 5.0);
    assert_eq!(defaults.maxfev, 100);
}

#[test]
#[should_panic(expected = "delta and sigma must satisfy")]
fn rejects_invalid_wolfe_parameters() {
    let _ = HagerZhang::new().with_wolfe_coefficients(0.5, 0.9);
}

struct BoundedQuadratic;

impl CostFunction for BoundedQuadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        if x[0] > 0.5 {
            Ok(f64::INFINITY)
        } else {
            Ok((x[0] - 0.25).powi(2))
        }
    }
}

impl Gradient for BoundedQuadratic {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        if x[0] > 0.5 {
            Ok(vec![f64::INFINITY])
        } else {
            Ok(vec![2.0 * (x[0] - 0.25)])
        }
    }
}

#[test]
fn retreats_from_a_soft_rejected_trial() {
    let mut problem = Problem::new(BoundedQuadratic);
    let mut search = HagerZhang::new();

    let alpha = search
        .next(&mut problem, &vec![0.0], 0.0625, &vec![-0.5], &vec![1.0])
        .unwrap();

    assert!(alpha > 0.0 && alpha <= 0.5, "alpha = {alpha}");
    assert!((alpha - 0.25).abs() < 1e-12, "alpha = {alpha}");
}

#[derive(Debug)]
struct BelowStepBound;

struct StepBoundGuard;

impl CostFunction for StepBoundGuard {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = BelowStepBound;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        if x[0] > 0.0 && x[0] < 0.4 {
            return Err(BelowStepBound);
        }
        if x[0] > 0.5 {
            return Ok(f64::INFINITY);
        }
        Ok((x[0] - 0.25).powi(2))
    }
}

impl Gradient for StepBoundGuard {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        if x[0] > 0.0 && x[0] < 0.4 {
            return Err(BelowStepBound);
        }
        if x[0] > 0.5 {
            return Ok(vec![f64::INFINITY]);
        }
        Ok(vec![2.0 * (x[0] - 0.25)])
    }
}

#[test]
fn every_trial_respects_the_configured_step_bounds() {
    let mut problem = Problem::new(StepBoundGuard);
    let mut search = HagerZhang::new().bounds(0.4, 1.0);
    let alpha = search
        .next(&mut problem, &vec![0.0], 0.0625, &vec![-0.5], &vec![1.0])
        .expect("the search must not probe below step_min");

    assert!((0.4..=1.0).contains(&alpha), "alpha = {alpha}");
}

#[derive(Debug, PartialEq, Eq)]
enum ProbeError {
    Cancelled,
}

struct Bailing;

impl CostFunction for Bailing {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = ProbeError;

    fn cost(&self, _x: &Vec<f64>) -> Result<f64, Self::Error> {
        Err(ProbeError::Cancelled)
    }
}

impl Gradient for Bailing {
    type Gradient = Vec<f64>;

    fn gradient(&self, _x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        unreachable!("the default fused evaluation stops after the cost error")
    }
}

#[test]
fn propagates_typed_problem_errors() {
    let mut problem = Problem::new(Bailing);
    let mut search = HagerZhang::new();

    let error = search
        .next(&mut problem, &vec![0.0], 1.0, &vec![-1.0], &vec![1.0])
        .unwrap_err();

    assert_eq!(error, ProbeError::Cancelled);
    assert_eq!(problem.counts().cost_evals, 1);
    assert_eq!(problem.counts().gradient_evals, 1);
}

#[test]
fn one_instance_can_be_reused_without_retained_search_state() {
    let mut search = HagerZhang::new();

    for center in [3.0, 6.0] {
        let mut problem = Problem::new(ShiftedQuadratic { center });
        let gradient = vec![-2.0 * center];
        let direction = vec![2.0 * center];
        let alpha = search
            .next(
                &mut problem,
                &vec![0.0],
                center * center,
                &gradient,
                &direction,
            )
            .unwrap();
        assert!((alpha - 0.5).abs() < 1e-12, "alpha = {alpha}");
    }
}

struct DifferentiableAckley;

fn ackley_cost_and_gradient(x: &[f64]) -> (f64, Vec<f64>) {
    let n = x.len() as f64;
    let c = 2.0 * core::f64::consts::PI;
    let sum_sq: f64 = x.iter().map(|xi| xi * xi).sum();
    let sum_cos: f64 = x.iter().map(|xi| (c * xi).cos()).sum();
    let radius = (sum_sq / n).sqrt();
    let exp_radius = (-0.2 * radius).exp();
    let exp_cos = (sum_cos / n).exp();
    let cost = -20.0 * exp_radius - exp_cos + 20.0 + core::f64::consts::E;
    let gradient = x
        .iter()
        .map(|xi| {
            4.0 * exp_radius * xi / (n * radius)
                + c * exp_cos * (c * xi).sin() / n
        })
        .collect();
    (cost, gradient)
}

impl CostFunction for DifferentiableAckley {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        Ok(ackley_cost_and_gradient(x).0)
    }
}

impl Gradient for DifferentiableAckley {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(ackley_cost_and_gradient(x).1)
    }

    fn cost_and_gradient(
        &self,
        x: &Vec<f64>,
    ) -> Result<(f64, Vec<f64>), Self::Error> {
        Ok(ackley_cost_and_gradient(x))
    }
}

#[test]
fn global_search_ackley_direction_gets_a_decreasing_step() {
    let x = vec![0.25, -0.4, 0.15];
    let (cost, gradient) = ackley_cost_and_gradient(&x);
    let direction: Vec<_> = gradient.iter().map(|gi| -gi).collect();
    let mut problem = Problem::new(DifferentiableAckley);
    let alpha = HagerZhang::new()
        .next(&mut problem, &x, cost, &gradient, &direction)
        .unwrap();
    let candidate: Vec<_> = x
        .iter()
        .zip(&direction)
        .map(|(xi, di)| xi + alpha * di)
        .collect();

    assert!(alpha.is_finite() && alpha > 0.0, "alpha = {alpha}");
    assert!(
        ackley_cost_and_gradient(&candidate).0 < cost,
        "accepted step did not decrease the Ackley cost"
    );
}

#[test]
fn unbounded_lbfgs_with_hager_zhang_progresses_on_ackley() {
    let initial = vec![0.25, -0.4, 0.15];
    let initial_cost = ackley_cost_and_gradient(&initial).0;
    let solver =
        Lbfgs::<Unbounded, HagerZhang>::with_line_search(HagerZhang::new());
    let result = Executor::new(
        DifferentiableAckley,
        solver,
        LbfgsState::new(initial, 5),
    )
    .max_iter(5)
    .run()
    .unwrap();

    assert_ne!(result.reason, TerminationReason::SolverFailed);
    assert!(
        result.cost() < initial_cost,
        "cost {} did not improve on {initial_cost}",
        result.cost()
    );
}
