use basin::{
    BoxConstraints, CostFunction, DenseMatrix, Executor, Jacobian, NllsState,
    Residual, State, TerminationReason, TrustRegionReflective,
};

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
#[path = "support/reflective_backend.rs"]
mod reflective_backend;

#[test]
fn vec_backend() {
    reflective_backend::check::<_, f64>(|x| x.to_vec());
    reflective_backend::check::<_, f32>(|x| x.to_vec());
}
#[cfg(feature = "nalgebra_all")]
#[test]
fn nalgebra_backend() {
    reflective_backend::check::<_, f64>(
        backend_aliases::nalgebra::DVector::from_column_slice,
    );
    reflective_backend::check::<_, f32>(
        backend_aliases::nalgebra::DVector::from_column_slice,
    );
}
#[cfg(feature = "ndarray_all")]
#[test]
fn ndarray_backend() {
    reflective_backend::check::<_, f64>(|x| {
        backend_aliases::ndarray::Array1::from_vec(x.to_vec())
    });
    reflective_backend::check::<_, f32>(|x| {
        backend_aliases::ndarray::Array1::from_vec(x.to_vec())
    });
}
#[cfg(feature = "faer_all")]
#[test]
fn faer_backend() {
    reflective_backend::check::<_, f64>(|x| {
        backend_aliases::faer::Col::from_fn(x.len(), |i| x[i])
    });
    reflective_backend::check::<_, f32>(|x| {
        backend_aliases::faer::Col::from_fn(x.len(), |i| x[i])
    });
}

#[derive(Clone)]
enum Function {
    Linear { a: Vec<f64>, b: Vec<f64> },
    Rosenbrock,
}

#[derive(Clone)]
struct Model {
    function: Function,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl Model {
    fn linear(a: &[f64], b: &[f64], lower: &[f64], upper: &[f64]) -> Self {
        Self {
            function: Function::Linear {
                a: a.to_vec(),
                b: b.to_vec(),
            },
            lower: lower.to_vec(),
            upper: upper.to_vec(),
        }
    }

    fn rosenbrock() -> Self {
        Self {
            function: Function::Rosenbrock,
            lower: vec![-2.0, 1.5],
            upper: vec![2.0, 3.0],
        }
    }
}

impl CostFunction for Model {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = &'static str;
    fn cost(&self, _: &Vec<f64>) -> Result<f64, Self::Error> {
        panic!("NLLS must derive its cost from residuals")
    }
}

impl Residual for Model {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = &'static str;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        for i in 0..x.len() {
            assert!(
                x[i] >= self.lower[i] && x[i] <= self.upper[i],
                "infeasible callback: {x:?}"
            );
            if self.lower[i] != self.upper[i] {
                assert!(
                    x[i] > self.lower[i] && x[i] < self.upper[i],
                    "noninterior callback: {x:?}"
                );
            }
        }
        Ok(match &self.function {
            Function::Linear { a, b } => a
                .chunks(x.len())
                .zip(b)
                .map(|(row, b)| {
                    row.iter().zip(x).map(|(a, x)| a * x).sum::<f64>() - b
                })
                .collect(),
            Function::Rosenbrock => {
                vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]]
            }
        })
    }
}

impl Jacobian for Model {
    type Jacobian = DenseMatrix;
    fn jacobian(&self, x: &Vec<f64>) -> Result<DenseMatrix, Self::Error> {
        Ok(match &self.function {
            Function::Linear { a, b } => {
                DenseMatrix::from_row_slice(b.len(), x.len(), a)
            }
            Function::Rosenbrock => DenseMatrix::from_row_slice(
                2,
                2,
                &[-20.0 * x[0], 10.0, -1.0, 0.0],
            ),
        })
    }
}

impl BoxConstraints for Model {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

#[test]
fn interior_active_and_fixed_solutions() {
    for (lower, upper, expected) in [
        (vec![-10.0; 2], vec![10.0; 2], vec![1.0, 3.0]),
        (vec![-1.0; 2], vec![1.0; 2], vec![1.0, 1.0]),
        (vec![0.5, -10.0], vec![0.5, 10.0], vec![0.5, 3.0]),
        (vec![0.5, 2.0], vec![0.5, 2.0], vec![0.5, 2.0]),
    ] {
        let model =
            Model::linear(&[1.0, 0.0, 0.0, 1.0], &[1.0, 3.0], &lower, &upper);
        let result = Executor::from_start(
            model.clone(),
            TrustRegionReflective::new(),
            vec![-20.0, 20.0],
        )
        .max_iter(200)
        .run()
        .unwrap();
        assert_eq!(result.reason, TerminationReason::SolverConverged);
        for (x, want) in result.param().iter().zip(&expected) {
            assert!((x - want).abs() < 2e-4, "{x} != {want}");
        }
        let cost = model
            .residual(result.param())
            .unwrap()
            .iter()
            .map(|r| 0.5 * r * r)
            .sum::<f64>();
        assert!(
            (result.cost() - cost).abs() <= 4.0 * f64::EPSILON * cost.max(1.0)
        );
        if lower == upper {
            assert_eq!(result.state.residual_evals(), 1);
            assert_eq!(result.state.jacobian_evals(), 0);
        }
    }
}

#[test]
fn rank_deficient_and_underdetermined() {
    for (a, b) in [
        (vec![1.0, 1.0, 2.0, 2.0], vec![1.0, 2.0]),
        (vec![1.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
    ] {
        let model =
            Model::linear(&a, &b, &[f64::NEG_INFINITY; 2], &[f64::INFINITY; 2]);
        let result = Executor::from_start(
            model,
            TrustRegionReflective::new(),
            vec![0.0; 2],
        )
        .max_iter(100)
        .run()
        .unwrap();
        assert_eq!(result.reason, TerminationReason::SolverConverged);
        assert!(result.cost() < 1e-20, "{}", result.cost());
        assert!(result.param().iter().all(|x| x.is_finite()));
    }
}

#[test]
fn bounded_rosenbrock() {
    let result = Executor::from_start(
        Model::rosenbrock(),
        TrustRegionReflective::new(),
        vec![2.0, 2.0],
    )
    .max_iter(200)
    .run()
    .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!((result.param()[0] - 1.22437075).abs() < 1e-6);
    assert!((result.param()[1] - 1.5).abs() < 1e-8);
    assert!((result.cost() - 0.0252130939468).abs() < 1e-10);
}

#[test]
fn narrow_interval_and_no_representable_interior() {
    let model = Model::linear(&[1.0], &[2.0], &[1.0], &[1.0 + 1e-12]);
    let result =
        Executor::from_start(model, TrustRegionReflective::new(), vec![0.0])
            .max_iter(100)
            .run()
            .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverConverged);
    let model = Model::linear(
        &[1.0],
        &[2.0],
        &[1.0],
        &[f64::from_bits(1.0_f64.to_bits() + 1)],
    );
    let result =
        Executor::from_start(model, TrustRegionReflective::new(), vec![1.0])
            .max_iter(100)
            .run()
            .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverFailed);
    assert_eq!(result.state.residual_evals(), 0);
}

#[test]
fn initialization_counts_and_stationarity() {
    let model =
        Model::linear(&[1.0], &[0.0], &[f64::NEG_INFINITY], &[f64::INFINITY]);
    let result =
        Executor::from_start(model, TrustRegionReflective::new(), vec![0.0])
            .run()
            .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert_eq!(result.iter(), 0);
    assert_eq!(result.state.residual_evals(), 1);
    assert_eq!(result.state.jacobian_evals(), 1);
}

#[test]
fn exact_continuation_and_fresh_reuse() {
    let full = Executor::from_start(
        Model::rosenbrock(),
        TrustRegionReflective::new(),
        vec![2.0, 2.0],
    )
    .max_iter(200)
    .run()
    .unwrap();
    let partial = Executor::from_start(
        Model::rosenbrock(),
        TrustRegionReflective::new(),
        vec![2.0, 2.0],
    )
    .max_iter(2)
    .run_with_solver()
    .unwrap();
    let resumed = Executor::resume_from_checkpoint(
        Model::rosenbrock(),
        partial.into_checkpoint(),
    )
    .max_iter(200)
    .run_with_solver()
    .unwrap();
    assert_eq!(full.param(), resumed.state.param());
    assert_eq!(full.state.residual_evals(), resumed.state.residual_evals());
    let fresh = Executor::new(
        Model::rosenbrock(),
        resumed.solver,
        NllsState::new(vec![2.0, 2.0]),
    )
    .max_iter(200)
    .run()
    .unwrap();
    assert_eq!(fresh.param(), full.param());
    assert_eq!(fresh.state.residual_evals(), full.state.residual_evals());
}

#[test]
fn settings_compose_in_both_orders() {
    let solver = TrustRegionReflective::new()
        .with_relative_step_tolerance(1e-12)
        .with_initial_radius(0.5)
        .with_rank_tolerance(None)
        .with_max_inner_attempts(30)
        .with_max_subproblem_iterations(60)
        .with_absolute_scaled_gradient_tolerance(1e-8)
        .with_absolute_cost_change_tolerance(None);
    let result =
        Executor::from_start(Model::rosenbrock(), solver, vec![2.0, 2.0])
            .max_iter(200)
            .run()
            .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverConverged);
    let _ = TrustRegionReflective::<f64>::new()
        .with_initial_radius(None)
        .with_absolute_scaled_gradient_tolerance(None)
        .with_relative_step_tolerance::<Vec<f64>, f64>(1e-12);
}

#[test]
fn scipy_reference_fixtures() {
    for line in include_str!("fixtures/trust_region_reflective_reference.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let case: Vec<_> = line.split('|').collect();
        let numbers = |i: usize| -> Vec<f64> {
            case[i]
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|s| s.parse().unwrap())
                .collect()
        };
        let (lower, upper) = (numbers(2), numbers(3));
        let model = if case[0] == "rosenbrock" {
            Model::rosenbrock()
        } else {
            Model::linear(&numbers(5), &numbers(6), &lower, &upper)
        };
        let result = Executor::from_start(
            model.clone(),
            TrustRegionReflective::new()
                .with_absolute_scaled_gradient_tolerance(1e-10),
            numbers(4),
        )
        .max_iter(500)
        .run()
        .unwrap();
        assert_eq!(
            result.reason,
            TerminationReason::SolverConverged,
            "{}",
            case[0]
        );
        let reference: f64 = case[8].parse().unwrap();
        assert!(
            (result.cost() - reference).abs() < 1e-8 * reference.max(1.0),
            "{}: {} vs {reference}",
            case[0],
            result.cost()
        );
        if case[1] == "1" {
            for (x, y) in result.param().iter().zip(numbers(7)) {
                assert!((x - y).abs() < 2e-5, "{}: {x} vs {y}", case[0]);
            }
        }
        let r = model.residual(result.param()).unwrap();
        let j = model.jacobian(result.param()).unwrap();
        let mut optimality = 0.0_f64;
        for i in 0..lower.len() {
            if lower[i] == upper[i] {
                continue;
            }
            let g: f64 =
                r.iter().enumerate().map(|(k, r)| j.get(k, i) * r).sum();
            let v = if g < 0.0 && upper[i].is_finite() {
                upper[i] - result.param()[i]
            } else if g > 0.0 && lower[i].is_finite() {
                result.param()[i] - lower[i]
            } else {
                1.0
            };
            optimality = optimality.max((v * g).abs());
        }
        assert!(
            optimality <= 1.01e-10,
            "{}: optimality {optimality}",
            case[0]
        );
    }
}

type Callback = fn(f64) -> Result<f64, &'static str>;
struct Callbacks {
    residual: Callback,
    jacobian: Callback,
    lower: Vec<f64>,
    upper: Vec<f64>,
}
impl Callbacks {
    fn new(residual: Callback, jacobian: Callback) -> Self {
        Self {
            residual,
            jacobian,
            lower: vec![-10.0],
            upper: vec![10.0],
        }
    }
}
impl CostFunction for Callbacks {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = &'static str;
    fn cost(&self, _: &Vec<f64>) -> Result<f64, Self::Error> {
        panic!("residual-only solver")
    }
}
impl Residual for Callbacks {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = &'static str;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        (self.residual)(x[0]).map(|v| vec![v])
    }
}
impl Jacobian for Callbacks {
    type Jacobian = DenseMatrix;
    fn jacobian(&self, x: &Vec<f64>) -> Result<DenseMatrix, Self::Error> {
        (self.jacobian)(x[0]).map(|v| DenseMatrix::from_row_slice(1, 1, &[v]))
    }
}
impl BoxConstraints for Callbacks {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

fn guarded_exp(x: f64) -> Result<f64, &'static str> {
    Ok(if x > 1.0 {
        f64::INFINITY
    } else {
        x.exp() - 2.0
    })
}

#[test]
fn nonfinite_trials_shrink_radius_and_reuse_jacobian() {
    let problem = Callbacks::new(guarded_exp, |x| Ok(x.exp()));
    let result = Executor::from_start(
        problem,
        TrustRegionReflective::new().with_initial_radius(10.0),
        vec![-1.0],
    )
    .max_iter(100)
    .run()
    .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!((result.param()[0] - 2.0_f64.ln()).abs() < 1e-8);
    assert!(result.state.residual_evals() > result.state.jacobian_evals());
    assert_eq!(result.state.jacobian_evals(), result.iter() + 1);
}

#[test]
fn retry_exhaustion_preserves_the_current_state() {
    let problem = Callbacks::new(guarded_exp, |x| Ok(x.exp()));
    let result = Executor::from_start(
        problem,
        TrustRegionReflective::new()
            .with_initial_radius(10.0)
            .with_max_inner_attempts(1),
        vec![-1.0],
    )
    .run()
    .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverFailed);
    assert_eq!(result.param(), &vec![-1.0]);
    assert_eq!(result.iter(), 0);
    assert_eq!(result.state.residual_evals(), 2);
    assert_eq!(result.state.jacobian_evals(), 1);
    assert!(
        (result.cost() - 0.5 * guarded_exp(-1.0).unwrap().powi(2)).abs()
            < 1e-15
    );
}

#[test]
fn invalid_evaluations_and_typed_errors() {
    for problem in [
        Callbacks::new(|_| Ok(f64::NAN), |_| Ok(1.0)),
        Callbacks::new(|x| Ok(x - 2.0), |_| Ok(f64::INFINITY)),
        Callbacks::new(
            |x| Ok(x - 2.0),
            |x| Ok(if x == 0.0 { 1.0 } else { f64::NAN }),
        ),
    ] {
        let result = Executor::from_start(
            problem,
            TrustRegionReflective::new(),
            vec![0.0],
        )
        .max_iter(10)
        .run()
        .unwrap();
        assert_eq!(result.reason, TerminationReason::SolverFailed);
        assert_eq!(result.param(), &vec![0.0]);
        assert_eq!(result.iter(), 0);
    }
    for problem in [
        Callbacks::new(|_| Err("residual abort"), |_| Ok(1.0)),
        Callbacks::new(|x| Ok(x - 2.0), |_| Err("Jacobian abort")),
        Callbacks::new(
            |x| {
                if x == 0.0 {
                    Ok(-2.0)
                } else {
                    Err("trial abort")
                }
            },
            |_| Ok(1.0),
        ),
    ] {
        assert!(
            Executor::from_start(
                problem,
                TrustRegionReflective::new(),
                vec![0.0]
            )
            .run()
            .is_err()
        );
    }
}

#[test]
fn disabled_and_exact_gradient_tolerances() {
    for (tol, reason) in [
        (None, TerminationReason::SolverFailed),
        (Some(0.0), TerminationReason::SolverConverged),
    ] {
        let problem = Callbacks::new(Ok, |_| Ok(1.0));
        let result = Executor::from_start(
            problem,
            TrustRegionReflective::new()
                .with_absolute_scaled_gradient_tolerance(tol),
            vec![0.0],
        )
        .max_iter(2)
        .run()
        .unwrap();
        assert_eq!(result.reason, reason);
    }
}

#[test]
fn invalid_configuration_panics() {
    for value in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(
            std::panic::catch_unwind(|| TrustRegionReflective::new()
                .with_absolute_scaled_gradient_tolerance(value))
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(
                || TrustRegionReflective::new().with_initial_radius(value)
            )
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(
                || TrustRegionReflective::new().with_rank_tolerance(value)
            )
            .is_err()
        );
    }
    assert!(
        std::panic::catch_unwind(
            || TrustRegionReflective::<f64>::new().with_max_inner_attempts(0)
        )
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| TrustRegionReflective::<f64>::new()
            .with_max_subproblem_iterations(0))
        .is_err()
    );
}

#[test]
fn bound_aware_numerical_jacobian() {
    let mut p = Callbacks::new(
        |x| {
            if (0.0..=1.0).contains(&x) {
                Ok(x - 2.0)
            } else {
                Err("outside bounds")
            }
        },
        |_| panic!("use numerical Jacobian"),
    );
    p.lower = vec![0.0];
    p.upper = vec![1.0];
    // A fixed probe avoids cancellation in MINPACK-style relative steps at
    // the tiny initial displacement from the lower bound.
    let p =
        basin::BoundedFiniteDiff::new(p, vec![0.0], vec![1.0]).with_step(1e-6);
    let result =
        Executor::from_start(p, TrustRegionReflective::new(), vec![0.0])
            .max_iter(100)
            .run()
            .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!((result.param()[0] - 1.0).abs() < 1e-7);
}

#[test]
fn malformed_shapes_and_bounds_panic() {
    for p in [
        Model::linear(&[1.0], &[0.0], &[1.0], &[0.0]),
        Model::linear(&[1.0], &[0.0], &[f64::NAN], &[1.0]),
        Model::linear(&[1.0], &[0.0], &[], &[1.0]),
    ] {
        assert!(
            std::panic::catch_unwind(|| Executor::from_start(
                p,
                TrustRegionReflective::new(),
                vec![0.0]
            )
            .run())
            .is_err()
        );
    }
    assert!(
        std::panic::catch_unwind(|| Executor::from_start(
            Model::rosenbrock(),
            TrustRegionReflective::new(),
            vec![f64::INFINITY, 2.0]
        )
        .run())
        .is_err()
    );
}
