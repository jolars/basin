use std::{cell::RefCell, convert::Infallible, rc::Rc};

use basin::{
    Backtracking, BoxConstraints, Constant, CostFunction, Executor, Gradient,
    GradientState, HagerZhang, LbfgsState, Lbfgsb, LineSearch,
    LineSearchBounds, LineSearchOutcome, LineSearchResult, MoreThuente,
    Problem, State, TerminationReason, Wolfe,
};

#[derive(Clone)]
struct Quadratic {
    center: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    scale: f64,
    probes: Rc<RefCell<Vec<Vec<f64>>>>,
}

impl Quadratic {
    fn new(center: Vec<f64>) -> Self {
        let n = center.len();
        Self {
            center,
            lower: vec![0.0; n],
            upper: vec![1.0; n],
            scale: 1.0,
            probes: Rc::default(),
        }
    }

    fn assert_feasible(&self, x: &[f64]) {
        for (i, &xi) in x.iter().enumerate() {
            assert!(
                xi.is_finite() && self.lower[i] <= xi && xi <= self.upper[i],
                "infeasible evaluation: x[{i}] = {xi}, bounds = [{}, {}]",
                self.lower[i],
                self.upper[i],
            );
        }
    }
}

impl CostFunction for Quadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        self.assert_feasible(x);
        self.probes.borrow_mut().push(x.clone());
        Ok(self.scale
            * x.iter()
                .zip(&self.center)
                .map(|(xi, ci)| (xi - ci).powi(2))
                .sum::<f64>())
    }
}

impl Gradient for Quadratic {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        self.assert_feasible(x);
        Ok(x.iter()
            .zip(&self.center)
            .map(|(xi, ci)| 2.0 * self.scale * (xi - ci))
            .collect())
    }
}

impl BoxConstraints for Quadratic {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }

    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

#[test]
fn distant_optima_keep_every_evaluation_and_cached_value_feasible() {
    for center in [
        vec![5.0; 2],
        vec![8.0; 2],
        vec![50.0; 2],
        vec![-50.0; 2],
        vec![-50.0, 50.0],
    ] {
        for scale in [1.0, 0.001] {
            let expected: Vec<_> =
                center.iter().map(|&c: &f64| c.clamp(0.0, 1.0)).collect();
            let mut problem = Quadratic::new(center.clone());
            problem.scale = scale;
            let result = Executor::new(
                problem.clone(),
                Lbfgsb::default(),
                LbfgsState::new(vec![0.5; 2], 10),
            )
            .max_iter(100)
            .run()
            .unwrap();
            assert_eq!(result.reason, TerminationReason::SolverConverged);
            assert_eq!(result.param(), &expected);
            assert_eq!(result.best_param(), &expected);
            assert_eq!(result.cost(), problem.cost(result.param()).unwrap());
            assert_eq!(
                result.state.best_cost(),
                problem.cost(result.best_param()).unwrap()
            );
            assert_eq!(
                result.state.gradient().unwrap(),
                &problem.gradient(result.param()).unwrap()
            );
        }
    }
}

#[test]
fn one_sided_bounds_use_the_normalized_initial_step() {
    let mut problem = Quadratic::new(vec![50.0; 2]);
    problem.lower.fill(f64::NEG_INFINITY);
    problem.upper.fill(100.0);
    let probes = problem.probes.clone();
    Executor::new(
        problem,
        Lbfgsb::default(),
        LbfgsState::new(vec![0.0; 2], 10),
    )
    .max_iter(1)
    .run()
    .unwrap();
    let probes = probes.borrow();
    let norm = probes[1].iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-14,
        "initial displacement norm = {norm}"
    );
}

#[test]
fn rounded_endpoints_are_feasible_before_evaluation() {
    for sign in [-1.0, 1.0] {
        let mut problem = Quadratic::new(vec![sign * 50.0]);
        let bound = sign * 7.963295525789484;
        if sign < 0.0 {
            problem.lower[0] = bound;
            problem.upper[0] = 10.0;
        } else {
            problem.lower[0] = -10.0;
            problem.upper[0] = bound;
        }
        let result = Executor::new(
            problem.clone(),
            Lbfgsb::default(),
            LbfgsState::new(vec![-sign * 9.269342184833151], 10),
        )
        .max_iter(100)
        .run()
        .unwrap();
        assert_eq!(result.reason, TerminationReason::SolverConverged);
        assert!((result.param()[0] - bound).abs() < 1e-12);
        assert_eq!(result.cost(), problem.cost(result.param()).unwrap());
        assert_eq!(
            result.state.gradient().unwrap(),
            &problem.gradient(result.param()).unwrap()
        );
    }
}

#[test]
fn one_sided_and_fixed_bounds_keep_evaluations_feasible() {
    for (center, lower, upper, expected) in [
        (-50.0, 0.0, f64::INFINITY, 0.0),
        (50.0, f64::NEG_INFINITY, 1.0, 1.0),
        (50.0, 0.25, 0.25, 0.25),
    ] {
        let mut problem = Quadratic::new(vec![center; 2]);
        problem.lower.fill(lower);
        problem.upper.fill(upper);
        let result = Executor::new(
            problem,
            Lbfgsb::default(),
            LbfgsState::new(vec![0.5; 2], 10),
        )
        .max_iter(100)
        .run()
        .unwrap();
        assert_eq!(result.reason, TerminationReason::SolverConverged);
        assert_eq!(result.param(), &[expected; 2]);
        assert_eq!(result.best_param(), result.param());
    }
}

fn check_builtin<S: LineSearch<Quadratic, Vec<f64>, Error = Infallible>>(
    search: S,
) {
    let problem = Quadratic::new(vec![-50.0, 50.0]);
    let result = Executor::new(
        problem.clone(),
        Lbfgsb::with_line_search(search),
        LbfgsState::new(vec![0.5; 2], 10),
    )
    .max_iter(100)
    .run()
    .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert_eq!(result.param(), &[0.0, 1.0]);
    assert_eq!(result.best_param(), result.param());
    assert_eq!(result.cost(), problem.cost(result.param()).unwrap());
    assert_eq!(
        result.state.gradient().unwrap(),
        &problem.gradient(result.param()).unwrap()
    );
}

#[test]
fn all_builtin_searches_respect_the_box() {
    check_builtin(MoreThuente::new().alpha_init(100.0));
    check_builtin(Backtracking::new().alpha_init(100.0));
    check_builtin(Wolfe::new().alpha_init(100.0));
    check_builtin(HagerZhang::new().alpha_init(100.0));
    check_builtin(Constant::new(100.0));
}

struct LegacySearch;

impl LineSearch<Quadratic, Vec<f64>> for LegacySearch {
    type Error = Infallible;

    fn next(
        &mut self,
        _: &mut Problem<Quadratic>,
        _: &Vec<f64>,
        _: f64,
        _: &Vec<f64>,
        _: &Vec<f64>,
    ) -> Result<f64, Infallible> {
        panic!(
            "an unsupported custom search must not probe the bounded problem"
        );
    }
}

#[test]
fn unsupported_custom_search_fails_without_evaluating_or_changing_state() {
    let problem = Quadratic::new(vec![50.0; 2]);
    let result = Executor::new(
        problem,
        Lbfgsb::with_line_search(LegacySearch),
        LbfgsState::new(vec![0.5; 2], 10),
    )
    .max_iter(100)
    .run()
    .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverFailed);
    assert_eq!(result.param(), &[0.5; 2]);
    assert_eq!(result.best_param(), result.param());
    assert_eq!(result.cost_evals(), 1);
    assert_eq!(result.state.gradient_evals(), 1);
}

struct InvalidSearch(LineSearchResult<Vec<f64>, f64>);

impl LineSearch<Quadratic, Vec<f64>> for InvalidSearch {
    type Error = Infallible;

    fn next(
        &mut self,
        _: &mut Problem<Quadratic>,
        _: &Vec<f64>,
        _: f64,
        _: &Vec<f64>,
        _: &Vec<f64>,
    ) -> Result<f64, Infallible> {
        unreachable!()
    }

    fn next_with_bounds(
        &mut self,
        _: &mut Problem<Quadratic>,
        _: &Vec<f64>,
        _: f64,
        _: &Vec<f64>,
        _: &Vec<f64>,
        _: LineSearchBounds,
    ) -> Result<LineSearchResult<Vec<f64>, f64>, Infallible> {
        Ok(self.0.clone())
    }
}

#[test]
fn invalid_selected_steps_and_cached_points_leave_a_feasible_best() {
    let mut results: Vec<_> = [0.0, -1.0, 2.0, f64::NAN, f64::INFINITY]
        .into_iter()
        .map(|step| LineSearchResult::new(LineSearchOutcome::Step(step)))
        .collect();
    results.push(LineSearchResult::with_evaluation(
        1.0,
        vec![2.0; 2],
        -1.0,
        vec![0.0; 2],
    ));
    for search_result in results {
        let problem = Quadratic::new(vec![50.0; 2]);
        let result = Executor::new(
            problem.clone(),
            Lbfgsb::with_line_search(InvalidSearch(search_result)),
            LbfgsState::new(vec![0.5; 2], 10),
        )
        .max_iter(100)
        .run()
        .unwrap();
        assert_eq!(result.reason, TerminationReason::SolverFailed);
        assert_eq!(result.param(), &[0.5; 2]);
        assert_eq!(result.best_param(), result.param());
        assert_eq!(result.cost_evals(), 1);
        assert_eq!(result.state.gradient_evals(), 1);
        assert_eq!(result.cost(), problem.cost(result.param()).unwrap());
        assert_eq!(
            result.state.gradient().unwrap(),
            &problem.gradient(result.param()).unwrap()
        );
    }
}

#[test]
fn per_call_bounds_do_not_change_more_thuente_configuration() {
    let mut problem = Problem::new(Quadratic::new(vec![50.0]));
    let param = vec![0.5];
    let (cost, gradient) = problem.cost_and_gradient(&param).unwrap();
    let mut search = MoreThuente::new().alpha_init(3.0).stpmax(4.0);
    let result = search
        .next_with_bounds(
            &mut problem,
            &param,
            cost,
            &gradient,
            &vec![2.0],
            LineSearchBounds::new(0.1, 0.25),
        )
        .unwrap();
    assert_eq!(result.outcome, LineSearchOutcome::Step(0.25));
    assert_eq!(result.evaluation.unwrap().param, &[1.0]);
    assert_eq!(problem.inner().probes.borrow()[1], &[0.7]);
    assert_eq!(search.alpha_init, 3.0);
    assert_eq!(search.stpmax, 4.0);
}

macro_rules! backend_regression {
    ($name:ident, $float:ty, $vector:ty, $make:expr) => {
        mod $name {
            use super::*;
            type Float = $float;
            type Vector = $vector;

            struct BoundedQuadratic {
                lower: Vector,
                upper: Vector,
            }

            impl CostFunction for BoundedQuadratic {
                type Param = Vector;
                type Output = Float;
                type Error = Infallible;

                fn cost(&self, x: &Vector) -> Result<Float, Infallible> {
                    for i in 0..2 {
                        assert!(0.0 <= x[i] && x[i] <= 1.0);
                    }
                    Ok((x[0] + 50.0).powi(2) + (x[1] - 50.0).powi(2))
                }
            }

            impl Gradient for BoundedQuadratic {
                type Gradient = Vector;
                fn gradient(&self, x: &Vector) -> Result<Vector, Infallible> {
                    for i in 0..2 {
                        assert!(0.0 <= x[i] && x[i] <= 1.0);
                    }
                    Ok(($make)(vec![2.0 * (x[0] + 50.0), 2.0 * (x[1] - 50.0)]))
                }
            }

            impl BoxConstraints for BoundedQuadratic {
                fn lower(&self) -> &Vector {
                    &self.lower
                }
                fn upper(&self) -> &Vector {
                    &self.upper
                }
            }

            #[test]
            fn lower_and_upper_active_bounds() {
                let problem = BoundedQuadratic {
                    lower: ($make)(vec![0.0; 2]),
                    upper: ($make)(vec![1.0; 2]),
                };
                let solver = basin::Lbfgs::<
                    basin::solver::lbfgs::Bounded,
                    _,
                    Float,
                >::with_line_search(
                    MoreThuente::<Float>::new()
                );
                let result = Executor::new(
                    problem,
                    solver,
                    LbfgsState::new(($make)(vec![0.5; 2]), 10),
                )
                .max_iter(100)
                .run()
                .unwrap();
                assert_eq!(result.reason, TerminationReason::SolverConverged);
                assert_eq!(result.param()[0], 0.0);
                assert_eq!(result.param()[1], 1.0);
                assert_eq!(result.best_param(), result.param());
                assert_eq!(result.cost(), 50.0 * 50.0 + 49.0 * 49.0);
                assert_eq!(result.state.gradient().unwrap()[0], 100.0);
                assert_eq!(result.state.gradient().unwrap()[1], -98.0);
            }
        }
    };
}

backend_regression!(vec_f32, f32, Vec<f32>, |v| v);

#[cfg(feature = "nalgebra_all")]
backend_regression!(
    nalgebra_f64,
    f64,
    crate::backend_aliases::nalgebra::DVector<f64>,
    crate::backend_aliases::nalgebra::DVector::from_vec
);
#[cfg(feature = "ndarray_all")]
backend_regression!(
    ndarray_f64,
    f64,
    crate::backend_aliases::ndarray::Array1<f64>,
    crate::backend_aliases::ndarray::Array1::from_vec
);
#[cfg(feature = "faer_all")]
backend_regression!(
    faer_f64,
    f64,
    crate::backend_aliases::faer::Col<f64>,
    |v: Vec<f64>| crate::backend_aliases::faer::Col::from_fn(v.len(), |i| v[i])
);

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
