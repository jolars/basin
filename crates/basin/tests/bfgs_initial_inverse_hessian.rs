#[path = "support/bfgs_initial_inverse_hessian.rs"]
mod backend;
#[path = "support/backend_aliases.rs"]
mod backend_aliases;

use backend::Quadratic;
#[cfg(feature = "serde")]
use basin::FirstOrderState;
#[cfg(feature = "serde")]
use basin::solver::bfgs::{FixedInverseHessian, InitialInverseHessian};
use basin::{
    Bfgs, Constant, CostFunction, DenseMatrix, ExactCheckpoint, Executor,
    Gradient, MatrixIdentity, State,
};
use std::{cell::Cell, convert::Infallible};

#[test]
fn vec_f32() {
    backend::check::<_, f32>(|x| x.to_vec());
}

#[test]
fn vec_f64() {
    backend::check::<_, f64>(|x| x.to_vec());
}

#[cfg(feature = "nalgebra_all")]
#[test]
fn nalgebra_both_scalars() {
    use backend_aliases::nalgebra::DVector;
    backend::check::<_, f32>(DVector::from_column_slice);
    backend::check::<_, f64>(DVector::from_column_slice);
}

#[cfg(feature = "ndarray_all")]
#[test]
fn ndarray_both_scalars() {
    use backend_aliases::ndarray::Array1;
    backend::check::<_, f32>(|x| Array1::from_vec(x.to_vec()));
    backend::check::<_, f64>(|x| Array1::from_vec(x.to_vec()));
}

#[cfg(feature = "faer_all")]
#[test]
fn faer_both_scalars() {
    use backend_aliases::faer::Col;
    backend::check::<_, f32>(|x| Col::from_fn(x.len(), |i| x[i]));
    backend::check::<_, f64>(|x| Col::from_fn(x.len(), |i| x[i]));
}

fn quadratic() -> Quadratic<Vec<f64>, f64> {
    Quadratic {
        make: |x| x.to_vec(),
    }
}

fn seed() -> DenseMatrix {
    DenseMatrix::from_row_slice(2, 2, &[0.3, 0.1, 0.1, 0.7])
}

fn assert_matrix_close(a: &DenseMatrix, b: &DenseMatrix) {
    assert_eq!(a.nrows(), b.nrows());
    assert_eq!(a.ncols(), b.ncols());
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            assert!((a.get(i, j) - b.get(i, j)).abs() < 1e-13);
        }
    }
}

// Use the factored inverse-BFGS formula as an independent reference for the
// production implementation's three rank-one updates.
fn reference_update(h: &DenseMatrix, s: &[f64], y: &[f64]) -> DenseMatrix {
    let n = s.len();
    let rho = 1.0 / s.iter().zip(y).map(|(a, b)| a * b).sum::<f64>();
    DenseMatrix::from_fn(n, n, |i, j| {
        let mut value = rho * s[i] * s[j];
        for k in 0..n {
            for l in 0..n {
                let left = f64::from(i == k) - rho * s[i] * y[k];
                let right = f64::from(l == j) - rho * y[l] * s[j];
                value += left * h.get(k, l) * right;
            }
        }
        value
    })
}

#[test]
fn default_seed_rescales_and_custom_seeds_retain_their_scaling() {
    let initial = vec![2.0, -1.0];
    let first = Executor::from_start(
        quadratic(),
        Bfgs::with_line_search(Constant(0.25)),
        initial.clone(),
    )
    .max_iter(1)
    .run_with_solver()
    .unwrap();
    let s = [-1.75, 0.0];
    let y = [-7.0, -1.75];
    let h0 =
        DenseMatrix::from_row_slice(2, 2, &[4.0 / 17.0, 0.0, 0.0, 4.0 / 17.0]);
    assert_matrix_close(
        first.solver.inverse_hessian().unwrap(),
        &reference_update(&h0, &s, &y),
    );

    let fixed = Executor::from_start(
        quadratic(),
        Bfgs::with_line_search(Constant(0.25))
            .with_initial_inverse_hessian(seed()),
        initial.clone(),
    )
    .max_iter(1)
    .run_with_solver()
    .unwrap();
    let from_gradient = Executor::from_start(
        quadratic(),
        Bfgs::with_line_search(Constant(0.25))
            .with_initial_inverse_hessian_initializer(
                |_: &Vec<f64>, _: &Vec<f64>| seed(),
            ),
        initial,
    )
    .max_iter(1)
    .run_with_solver()
    .unwrap();
    let s = [-0.525, -0.175];
    let y = [-2.275, -0.875];
    assert_matrix_close(
        fixed.solver.inverse_hessian().unwrap(),
        &reference_update(&seed(), &s, &y),
    );
    assert_matrix_close(
        from_gradient.solver.inverse_hessian().unwrap(),
        fixed.solver.inverse_hessian().unwrap(),
    );
}

#[test]
fn skipped_curvature_does_not_replace_custom_seed() {
    let first = Executor::from_start(
        quadratic(),
        Bfgs::with_line_search(Constant(0.25))
            .with_initial_inverse_hessian(seed())
            .with_relative_curvature_tolerance(2.0),
        vec![2.0, -1.0],
    )
    .max_iter(2)
    .run_with_solver()
    .unwrap();
    assert_eq!(first.solver.inverse_hessian(), Some(&seed()));
    let (solver, state, counts) = first.into_checkpoint().into_parts();
    let (x, _, g) = state.current().unwrap();
    let direction = basin::MatVec::matvec(&seed(), g);
    let s: Vec<_> = direction.iter().map(|v| -0.25 * v).collect();
    let next: Vec<_> = x.iter().zip(&s).map(|(x, s)| x + s).collect();
    let next_gradient = quadratic().gradient(&next).unwrap();
    let y: Vec<_> = next_gradient.iter().zip(g).map(|(a, b)| a - b).collect();
    let resumed = Executor::resume_from_checkpoint(
        quadratic(),
        ExactCheckpoint::from_parts(
            solver.with_relative_curvature_tolerance(0.0),
            state,
            counts,
        ),
    )
    .max_iter(3)
    .run_with_solver()
    .unwrap();
    assert_matrix_close(
        resumed.solver.inverse_hessian().unwrap(),
        &reference_update(&seed(), &s, &y),
    );
}

#[test]
fn skipped_default_curvature_defers_rescaling_until_the_first_update() {
    let first = Executor::from_start(
        quadratic(),
        Bfgs::with_line_search(Constant(0.1))
            .with_relative_curvature_tolerance(2.0),
        vec![2.0, -1.0],
    )
    .max_iter(1)
    .run_with_solver()
    .unwrap();
    assert_eq!(
        first.solver.inverse_hessian(),
        Some(&DenseMatrix::identity(2))
    );
    let (solver, state, counts) = first.into_checkpoint().into_parts();
    let (x, _, gradient) = state.current().unwrap();
    let s: Vec<_> = gradient.iter().map(|v| -0.1 * v).collect();
    let next: Vec<_> = x.iter().zip(&s).map(|(x, s)| x + s).collect();
    let next_gradient = quadratic().gradient(&next).unwrap();
    let y: Vec<_> = next_gradient
        .iter()
        .zip(gradient)
        .map(|(a, b)| a - b)
        .collect();
    let sy: f64 = s.iter().zip(&y).map(|(s, y)| s * y).sum();
    let yy: f64 = y.iter().map(|y| y * y).sum();
    let h0 =
        DenseMatrix::from_fn(2, 2, |i, j| if i == j { sy / yy } else { 0.0 });
    let resumed = Executor::resume_from_checkpoint(
        quadratic(),
        ExactCheckpoint::from_parts(
            solver.with_relative_curvature_tolerance(0.0),
            state,
            counts,
        ),
    )
    .max_iter(2)
    .run_with_solver()
    .unwrap();
    assert_matrix_close(
        resumed.solver.inverse_hessian().unwrap(),
        &reference_update(&h0, &s, &y),
    );
}

#[test]
fn changing_initializer_affects_fresh_solves_but_preserves_the_live_model() {
    let make = || {
        Executor::from_start(
            quadratic(),
            Bfgs::with_line_search(Constant(0.25)),
            vec![2.0, -1.0],
        )
    };
    let expected = make().max_iter(3).run_with_solver().unwrap();
    // Even at iteration zero, setting a custom seed on a retained solver
    // must preserve the default model's pending first-update rescaling.
    let first = make().max_iter(0).run_with_solver().unwrap();
    let (solver, state, counts) = first.into_checkpoint().into_parts();
    let checkpoint = ExactCheckpoint::from_parts(
        solver.with_initial_inverse_hessian(seed()),
        state,
        counts,
    );
    assert_eq!(
        checkpoint.solver().inverse_hessian(),
        Some(&DenseMatrix::identity(2))
    );
    let resumed = Executor::resume_from_checkpoint(quadratic(), checkpoint)
        .max_iter(3)
        .run_with_solver()
        .unwrap();
    assert_eq!(resumed.state, expected.state);
    assert_eq!(resumed.counts, expected.counts);
    assert_eq!(
        resumed.solver.inverse_hessian(),
        expected.solver.inverse_hessian()
    );
    let fresh = Executor::new(quadratic(), resumed.solver, resumed.state)
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    assert_eq!(fresh.solver.inverse_hessian(), Some(&seed()));
}

#[test]
fn fresh_fixed_seed_is_independent_of_previous_updates() {
    let first = Executor::from_start(
        quadratic(),
        Bfgs::with_line_search(Constant(0.25))
            .with_initial_inverse_hessian(seed()),
        vec![2.0, -1.0],
    )
    .max_iter(2)
    .run_with_solver()
    .unwrap();
    assert_ne!(first.solver.inverse_hessian(), Some(&seed()));
    let fresh = Executor::new(quadratic(), first.solver, first.state)
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    assert_eq!(fresh.solver.inverse_hessian(), Some(&seed()));
    assert_eq!(fresh.state.iter(), 0);
    assert_eq!(fresh.counts.cost_evals, 1);
    assert_eq!(fresh.counts.gradient_evals, 1);
}

struct Sphere;
impl basin::Constraints for Sphere {
    type Set = basin::NoConstraints;
    fn constraints(&self) -> basin::NoConstraints {
        basin::NoConstraints::EMPTY
    }
}
impl CostFunction for Sphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(x.iter().map(|v| v * v).sum())
    }
}
impl Gradient for Sphere {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(x.iter().map(|v| 2.0 * v).collect())
    }
}

#[test]
fn fresh_initializers_follow_the_new_dimension_and_exact_resume_skips_them() {
    let calls = Cell::new(0);
    let make_solver = || {
        Bfgs::with_line_search(Constant(0.1))
            .with_initial_inverse_hessian_initializer(
                |x: &Vec<f64>, g: &Vec<f64>| {
                    calls.set(calls.get() + 1);
                    assert_eq!(*g, Sphere.gradient(x).unwrap());
                    DenseMatrix::from_fn(x.len(), x.len(), |i, j| {
                        if i == j {
                            0.1 / g[i].abs().max(1e-3)
                        } else {
                            0.0
                        }
                    })
                },
            )
    };
    let first = Executor::from_start(Sphere, make_solver(), vec![3.0, -2.0])
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    assert_eq!(calls.get(), 1);
    assert_eq!(first.counts.cost_evals, 1);
    assert_eq!(first.counts.gradient_evals, 1);
    let resumed =
        Executor::resume_from_checkpoint(Sphere, first.into_checkpoint())
            .max_iter(3)
            .run_with_solver()
            .unwrap();
    assert_eq!(calls.get(), 1);
    let expected = Executor::from_start(Sphere, make_solver(), vec![3.0, -2.0])
        .max_iter(3)
        .run_with_solver()
        .unwrap();
    assert_eq!(calls.get(), 2);
    assert_eq!(resumed.state, expected.state);
    assert_eq!(resumed.counts, expected.counts);
    assert_eq!(
        resumed.solver.inverse_hessian(),
        expected.solver.inverse_hessian()
    );

    let fresh =
        Executor::from_start(Sphere, resumed.solver, vec![0.0, 2e-5, -4.0])
            .max_iter(0)
            .run_with_solver()
            .unwrap();
    assert_eq!(calls.get(), 3);
    assert_eq!(fresh.state.iter(), 0);
    assert_eq!(fresh.counts.cost_evals, 1);
    assert_eq!(fresh.counts.gradient_evals, 1);
    assert_matrix_close(
        fresh.solver.inverse_hessian().unwrap(),
        &DenseMatrix::from_row_slice(
            3,
            3,
            &[100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0125],
        ),
    );
}

#[test]
fn initialization_accepts_empty_and_stationary_points() {
    for x in [vec![], vec![0.0; 2]] {
        let result = Executor::from_start(
            Sphere,
            Bfgs::new()
                .with_initial_inverse_hessian(DenseMatrix::identity(x.len()))
                .with_absolute_gradient_tolerance(0.0),
            x,
        )
        .max_iter(5)
        .run()
        .unwrap();
        assert_eq!(result.iter(), 0);
        assert_eq!(result.counts.cost_evals, 1);
        assert_eq!(result.counts.gradient_evals, 1);
    }
}

#[test]
fn custom_seed_builders_and_convergence_settings_compose_in_either_order() {
    let fixed = Executor::from_start(
        quadratic(),
        Bfgs::with_line_search(Constant(0.25))
            .with_absolute_gradient_tolerance(0.0)
            .with_initial_inverse_hessian(seed())
            .with_relative_curvature_tolerance(1e-10)
            .with_absolute_cost_change_tolerance(0.0)
            .with_relative_step_tolerance(0.0),
        vec![2.0, -1.0],
    )
    .max_iter(2)
    .run()
    .unwrap();
    let factory = Executor::from_start(
        quadratic(),
        Bfgs::with_line_search(Constant(0.25))
            .with_absolute_cost_change_tolerance(0.0)
            .with_initial_inverse_hessian_initializer(
                |_: &Vec<f64>, _: &Vec<f64>| seed(),
            )
            .with_relative_curvature_tolerance(1e-10)
            .with_absolute_gradient_tolerance(0.0)
            .with_relative_step_tolerance(0.0),
        vec![2.0, -1.0],
    )
    .max_iter(2)
    .run()
    .unwrap();
    assert_eq!(fixed.state, factory.state);
    assert_eq!(fixed.counts, factory.counts);
}

#[test]
fn invalid_fixed_and_callback_seeds_panic_at_initialization() {
    let invalid = [
        DenseMatrix::identity(1),
        DenseMatrix::from_fn(2, 3, |_, _| 1.0),
        DenseMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.1, 1.0]),
        DenseMatrix::from_row_slice(2, 2, &[f64::NAN, 0.0, 0.0, 1.0]),
        DenseMatrix::from_row_slice(2, 2, &[1.0, f64::NAN, 0.0, 1.0]),
        DenseMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, f64::INFINITY]),
        DenseMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 1.0]),
        DenseMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, 1.0]),
        DenseMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]),
        DenseMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]),
    ];
    for matrix in invalid {
        assert!(
            std::panic::catch_unwind(|| {
                Executor::from_start(
                    quadratic(),
                    Bfgs::new().with_initial_inverse_hessian(matrix.clone()),
                    vec![2.0, -1.0],
                )
                .max_iter(0)
                .run()
                .unwrap();
            })
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| {
                Executor::from_start(
                    quadratic(),
                    Bfgs::new().with_initial_inverse_hessian_initializer(
                        |_: &Vec<f64>, _: &Vec<f64>| matrix.clone(),
                    ),
                    vec![2.0, -1.0],
                )
                .max_iter(0)
                .run()
                .unwrap();
            })
            .is_err()
        );
    }
}

struct FailingEvaluation;
impl basin::Constraints for FailingEvaluation {
    type Set = basin::NoConstraints;
    fn constraints(&self) -> basin::NoConstraints {
        basin::NoConstraints::EMPTY
    }
}
impl CostFunction for FailingEvaluation {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = &'static str;
    fn cost(&self, _: &Vec<f64>) -> Result<f64, Self::Error> {
        Err("initial evaluation failed")
    }
}
impl Gradient for FailingEvaluation {
    type Gradient = Vec<f64>;
    fn gradient(&self, _: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        unreachable!("cost evaluation must fail first")
    }
}

#[test]
fn failed_initial_evaluation_propagates_without_calling_initializer() {
    let called = Cell::new(false);
    let result = Executor::from_start(
        FailingEvaluation,
        Bfgs::new().with_initial_inverse_hessian_initializer(
            |x: &Vec<f64>, _: &Vec<f64>| {
                called.set(true);
                DenseMatrix::identity(x.len())
            },
        ),
        vec![2.0, -1.0],
    )
    .run();
    assert!(matches!(result, Err("initial evaluation failed")));
    assert!(!called.get());
}

#[cfg(feature = "serde")]
#[test]
fn serialized_fixed_seed_preserves_model_then_restarts_from_original_seed() {
    type Solver = Bfgs<
        Vec<f64>,
        f64,
        DenseMatrix,
        Constant,
        FixedInverseHessian<DenseMatrix>,
    >;
    type Checkpoint = ExactCheckpoint<Solver, FirstOrderState<Vec<f64>>>;
    let make = || {
        Executor::from_start(
            quadratic(),
            Bfgs::with_line_search(Constant(0.25))
                .with_initial_inverse_hessian(seed()),
            vec![2.0, -1.0],
        )
    };
    let expected = make().max_iter(4).run_with_solver().unwrap();
    for steps in [0, 2] {
        let checkpoint = make()
            .max_iter(steps)
            .run_with_solver()
            .unwrap()
            .into_checkpoint();
        let bytes = postcard::to_allocvec(&checkpoint).unwrap();
        let restored: Checkpoint = postcard::from_bytes(&bytes).unwrap();
        let resumed = Executor::resume_from_checkpoint(quadratic(), restored)
            .max_iter(4)
            .run_with_solver()
            .unwrap();
        assert_eq!(resumed.state, expected.state);
        assert_eq!(resumed.counts, expected.counts);
        assert_eq!(
            resumed.solver.inverse_hessian(),
            expected.solver.inverse_hessian()
        );
        let fresh = Executor::new(quadratic(), resumed.solver, resumed.state)
            .max_iter(0)
            .run_with_solver()
            .unwrap();
        assert_eq!(fresh.solver.inverse_hessian(), Some(&seed()));
    }
}

#[cfg(feature = "serde")]
#[test]
fn default_identity_policy_preserves_the_previous_postcard_layout() {
    #[derive(serde::Serialize)]
    struct PreviousBfgs {
        line_search: Constant,
        epsilon: f64,
        inverse_hessian: Option<DenseMatrix>,
        initial_scaling_done: bool,
        param: std::marker::PhantomData<Vec<f64>>,
    }
    #[derive(serde::Serialize)]
    struct PreviousCheckpoint<'a> {
        solver: PreviousBfgs,
        state: &'a FirstOrderState<Vec<f64>>,
        counts: &'a basin::EvalCounts,
    }
    type Solver = Bfgs<Vec<f64>, f64, DenseMatrix, Constant>;
    type Checkpoint = ExactCheckpoint<Solver, FirstOrderState<Vec<f64>>>;
    assert!(
        !std::any::type_name::<Solver>().contains("IdentityInverseHessian")
    );
    let make = || {
        Executor::from_start(
            quadratic(),
            Bfgs::with_line_search(Constant(0.25)),
            vec![2.0, -1.0],
        )
    };
    let expected = make().max_iter(4).run_with_solver().unwrap();
    for steps in [0, 2] {
        let checkpoint = make()
            .max_iter(steps)
            .run_with_solver()
            .unwrap()
            .into_checkpoint();
        let previous = PreviousCheckpoint {
            solver: PreviousBfgs {
                line_search: Constant(0.25),
                epsilon: 1e-10,
                inverse_hessian: checkpoint.solver().inverse_hessian().cloned(),
                initial_scaling_done: steps > 0,
                param: std::marker::PhantomData,
            },
            state: checkpoint.state(),
            counts: checkpoint.counts(),
        };
        let bytes = postcard::to_allocvec(&previous).unwrap();
        assert_eq!(bytes, postcard::to_allocvec(&checkpoint).unwrap());
        let restored: Checkpoint = postcard::from_bytes(&bytes).unwrap();
        let resumed = Executor::resume_from_checkpoint(quadratic(), restored)
            .max_iter(4)
            .run_with_solver()
            .unwrap();
        assert_eq!(resumed.state, expected.state);
        assert_eq!(resumed.counts, expected.counts);
        assert_eq!(
            resumed.solver.inverse_hessian(),
            expected.solver.inverse_hessian()
        );
    }
}

#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
struct GradientSeed {
    scale: f64,
    floor: f64,
}

#[cfg(feature = "serde")]
impl InitialInverseHessian<Vec<f64>, DenseMatrix> for GradientSeed {
    fn initialize(&self, x: &Vec<f64>, g: &Vec<f64>) -> DenseMatrix {
        let initializer = |x: &Vec<f64>, g: &Vec<f64>| {
            DenseMatrix::from_fn(x.len(), x.len(), |i, j| {
                if i == j {
                    self.scale / g[i].abs().max(self.floor)
                } else {
                    0.0
                }
            })
        };
        initializer.initialize(x, g)
    }
}

#[cfg(feature = "serde")]
#[test]
fn named_gradient_initializer_survives_serialization_and_fresh_reuse() {
    type Checkpoint = ExactCheckpoint<
        Bfgs<Vec<f64>, f64, DenseMatrix, Constant, GradientSeed>,
        FirstOrderState<Vec<f64>>,
    >;
    let make = || {
        Executor::from_start(
            Sphere,
            Bfgs::with_line_search(Constant(0.25))
                .with_initial_inverse_hessian_initializer(GradientSeed {
                    scale: 0.1,
                    floor: 1e-3,
                }),
            vec![3.0, -2.0],
        )
    };
    let expected = make().max_iter(4).run_with_solver().unwrap();
    for steps in [0, 2] {
        let checkpoint = make()
            .max_iter(steps)
            .run_with_solver()
            .unwrap()
            .into_checkpoint();
        let bytes = postcard::to_allocvec(&checkpoint).unwrap();
        let restored: Checkpoint = postcard::from_bytes(&bytes).unwrap();
        let resumed = Executor::resume_from_checkpoint(Sphere, restored)
            .max_iter(4)
            .run_with_solver()
            .unwrap();
        assert_eq!(resumed.state, expected.state);
        assert_eq!(resumed.counts, expected.counts);
        assert_eq!(
            resumed.solver.inverse_hessian(),
            expected.solver.inverse_hessian()
        );
        let fresh =
            Executor::from_start(Sphere, resumed.solver, vec![0.0, 4.0, -5.0])
                .max_iter(0)
                .run_with_solver()
                .unwrap();
        assert_matrix_close(
            fresh.solver.inverse_hessian().unwrap(),
            &DenseMatrix::from_row_slice(
                3,
                3,
                &[100.0, 0.0, 0.0, 0.0, 0.0125, 0.0, 0.0, 0.0, 0.01],
            ),
        );
    }
}
