use basin::{Bfgs, CostFunction, Executor, Gradient};
use std::convert::Infallible;
struct ScaledQuadratic;

impl CostFunction for ScaledQuadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(0.5 * 1e-19 * x[0] * x[0])
    }
}

impl Gradient for ScaledQuadratic {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(vec![1e-19 * x[0]])
    }
}

#[test]
fn more_thuente_expands_rounded_away_bfgs_step() {
    use basin::MoreThuente;

    let target = 1e-8;
    let result = Executor::from_start(
        ScaledQuadratic,
        Bfgs::with_line_search(MoreThuente::new())
            .with_absolute_cost_change_tolerance(target),
        vec![1e12],
    )
    .max_iter(20)
    .run()
    .unwrap();

    assert!(result.cost() <= target, "cost = {}", result.cost());
}

#[test]
fn more_thuente_rounded_trials_respect_step_and_evaluation_limits() {
    use basin::{LineSearch, LineSearchBounds, LineSearchOutcome, MoreThuente};

    let x = vec![1e12];
    let f = ScaledQuadratic.cost(&x).unwrap();
    let g = ScaledQuadratic.gradient(&x).unwrap();
    let d = vec![-g[0]];
    for (cap, maxfev, evaluations, step) in [
        (1024.0, 1, 1, 1024.0),
        (512.0, 1, 0, 0.0),
        (1024.0, 0, 0, 0.0),
    ] {
        for configured_cap in [false, true] {
            let mut p = basin::Problem::new(ScaledQuadratic);
            let mut search = MoreThuente::new()
                .stpmax(if configured_cap { cap } else { 1e10 })
                .maxfev(maxfev);
            let result = search
                .next_with_bounds(
                    &mut p,
                    &x,
                    f,
                    &g,
                    &d,
                    LineSearchBounds::new(
                        1.0,
                        if configured_cap { 1e10 } else { cap },
                    ),
                )
                .unwrap();

            assert_eq!(result.outcome, LineSearchOutcome::Step(step));
            assert_eq!(p.counts().cost_evals, evaluations);
            assert_eq!(p.counts().gradient_evals, evaluations);
            if let Some(evaluation) = result.evaluation {
                assert_eq!(evaluations, 1);
                assert_ne!(evaluation.param, x);
            } else {
                assert_eq!(evaluations, 0);
            }
        }
    }
}

struct OverflowQuadratic;

impl CostFunction for OverflowQuadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        assert!(x.iter().all(|value| value.is_finite()));
        Ok((x[1] - 1.0_f64.next_down()).powi(2))
    }
}

impl Gradient for OverflowQuadratic {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(vec![0.0, 2.0 * (x[1] - 1.0_f64.next_down())])
    }
}

#[test]
fn more_thuente_bisects_an_overflowing_expansion_before_evaluation() {
    use basin::{LineSearch, LineSearchOutcome, MoreThuente, Problem};

    let x = vec![f64::MAX, 1.0];
    let f = OverflowQuadratic.cost(&x).unwrap();
    let g = OverflowQuadratic.gradient(&x).unwrap();
    let d = vec![3e291, -2.1e-17];
    for search in [
        MoreThuente::new(),
        MoreThuente::new().alpha_init(4.0).stpmin(3.0),
    ] {
        let mut problem = Problem::new(OverflowQuadratic);
        let result = search
            .maxfev(1)
            .next_with_evaluation(&mut problem, &x, f, &g, &d)
            .unwrap();

        assert_eq!(result.outcome, LineSearchOutcome::Step(3.0));
        let evaluation = result.evaluation.unwrap();
        assert_eq!(evaluation.param, vec![f64::MAX, 1.0_f64.next_down()]);
        assert_eq!(evaluation.cost, 0.0);
        assert_eq!(problem.counts().cost_evals, 1);
        assert_eq!(problem.counts().gradient_evals, 1);
    }
}

#[test]
fn more_thuente_skips_evaluation_if_all_changed_trials_overflow() {
    use basin::{LineSearch, LineSearchOutcome, MoreThuente, Problem};

    let x = vec![f64::MAX, 1.0];
    let f = OverflowQuadratic.cost(&x).unwrap();
    let g = OverflowQuadratic.gradient(&x).unwrap();
    let d = vec![1e292, -2.1e-17];
    let mut problem = Problem::new(OverflowQuadratic);

    let result = MoreThuente::new()
        .next_with_evaluation(&mut problem, &x, f, &g, &d)
        .unwrap();

    assert_eq!(result.outcome, LineSearchOutcome::Step(0.0));
    assert!(result.evaluation.is_none());
    assert_eq!(problem.counts().cost_evals, 0);
    assert_eq!(problem.counts().gradient_evals, 0);
}
