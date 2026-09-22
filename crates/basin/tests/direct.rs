#[path = "support/direct_backend.rs"]
mod backend;
#[path = "support/backend_aliases.rs"]
mod backend_aliases;

use basin::{
    BoxConstraints, CostFunction, Direct, EvaluationKind, Executor, PointState,
    Problem, Solver, State, TerminationReason,
};
use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
};

#[derive(Clone)]
struct Objective<C> {
    lower: Vec<f64>,
    upper: Vec<f64>,
    cost: C,
}

impl<C, E> CostFunction for Objective<C>
where
    C: Fn(&[f64]) -> Result<f64, E>,
{
    type Param = Vec<f64>;
    type Output = f64;
    type Error = E;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, E> {
        for ((&x, &lo), &hi) in x.iter().zip(&self.lower).zip(&self.upper) {
            assert!(x.is_finite() && x >= lo && x <= hi);
        }
        (self.cost)(x)
    }
}

impl<C, E> BoxConstraints for Objective<C>
where
    C: Fn(&[f64]) -> Result<f64, E>,
{
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

fn sphere(x: &[f64]) -> Result<f64, Infallible> {
    Ok(x.iter().map(|x| (x - 0.37).powi(2)).sum())
}

type ObjectiveFn = fn(&[f64]) -> Result<f64, Infallible>;

fn quadratic(n: usize) -> Objective<ObjectiveFn> {
    Objective {
        lower: vec![-1.0; n],
        upper: vec![1.0; n],
        cost: sphere,
    }
}

fn no_geometry() -> Direct {
    Direct::new().with_absolute_radius_tolerance(None)
}

#[test]
fn vec_f64() {
    backend::check::<_, f64>(|x| x.to_vec());
}

#[test]
fn f32_mixed_magnitude_costs_refine_the_best_rectangle() {
    struct Penalized {
        lower: Vec<f32>,
        upper: Vec<f32>,
    }

    impl CostFunction for Penalized {
        type Param = Vec<f32>;
        type Output = f32;
        type Error = Infallible;

        fn cost(&self, x: &Vec<f32>) -> Result<f32, Infallible> {
            Ok(if x[0] >= 2.0 / 3.0 {
                1e38
            } else if x[0] == 0.5 {
                -1e-9
            } else if (x[0] - 0.5).abs() < 0.02 {
                -2e-9
            } else {
                0.0
            })
        }
    }

    impl BoxConstraints for Penalized {
        fn lower(&self) -> &Vec<f32> {
            &self.lower
        }

        fn upper(&self) -> &Vec<f32> {
            &self.upper
        }
    }

    // The fourth sweep must refine the small negative incumbent even while
    // a larger rectangle retains a large finite penalty.
    let result = Executor::new(
        Penalized {
            lower: vec![0.0],
            upper: vec![1.0],
        },
        Direct::new(),
        PointState::new(vec![0.0]),
    )
    .max_iter(4)
    .run()
    .unwrap();
    assert_eq!(result.cost(), -2e-9);
    assert_eq!(result.reason, TerminationReason::MaxIter);
}

#[cfg(feature = "nalgebra_all")]
#[test]
fn nalgebra_scalars() {
    use backend_aliases::nalgebra::DVector;
    backend::check::<_, f64>(DVector::from_column_slice);
    backend::check::<_, f32>(DVector::from_column_slice);
}

#[cfg(feature = "ndarray_all")]
#[test]
fn ndarray_scalars() {
    use backend_aliases::ndarray::Array1;
    backend::check::<_, f64>(|x| Array1::from_vec(x.to_vec()));
    backend::check::<_, f32>(|x| Array1::from_vec(x.to_vec()));
}

#[cfg(feature = "faer_all")]
#[test]
fn faer_scalars() {
    use backend_aliases::faer::Col;
    backend::check::<_, f64>(|x| Col::from_fn(x.len(), |i| x[i]));
    backend::check::<_, f32>(|x| Col::from_fn(x.len(), |i| x[i]));
}

#[test]
fn midpoint_then_first_cross_and_budget_overshoot() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let log = calls.clone();
    let problem = Objective {
        lower: vec![0.0, 2.0, -3.0],
        upper: vec![6.0, 2.0, 3.0],
        cost: move |x: &[f64]| {
            log.lock().unwrap().push(x.to_vec());
            Ok::<_, Infallible>(x[0] + 2.0 * x[2])
        },
    };
    let stepper =
        Executor::new(problem, no_geometry(), PointState::new(vec![99.0; 3]))
            .max_evaluations(EvaluationKind::Cost, 2)
            .into_stepper()
            .unwrap();
    assert_eq!(stepper.state().param(), &[3.0, 2.0, 0.0]);
    assert_eq!(stepper.state().cost(), 3.0);
    assert_eq!(stepper.state().counts().cost_evals, 1);
    let result = stepper.run_to_end().unwrap();
    assert_eq!(result.reason, TerminationReason::MaxEvaluations);
    assert_eq!(result.iter(), 1);
    assert_eq!(result.state.counts().cost_evals, 5);
    let mut points = calls.lock().unwrap().clone();
    points.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let expected = [
        vec![1.0, 2.0, 0.0],
        vec![3.0, 2.0, -2.0],
        vec![3.0, 2.0, 0.0],
        vec![3.0, 2.0, 2.0],
        vec![5.0, 2.0, 0.0],
    ];
    for (actual, expected) in points.iter().zip(expected) {
        for (&a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-14);
        }
    }
    assert_eq!(result.cost(), result.param()[0] + 2.0 * result.param()[2]);
    assert_eq!(result.state.best_cost(), result.cost());
}

#[test]
fn initialization_only_and_fixed_box() {
    let result = Executor::new(
        quadratic(2),
        Direct::new(),
        PointState::new(vec![99.0; 2]),
    )
    .max_iter(0)
    .run()
    .unwrap();
    assert_eq!(result.iter(), 0);
    assert_eq!(result.state.counts().cost_evals, 1);
    assert_eq!(result.param(), &[0.0, 0.0]);
    for value in [1.0, f64::INFINITY, f64::NAN, f64::NEG_INFINITY] {
        let problem = Objective {
            lower: vec![2.0],
            upper: vec![2.0],
            cost: move |_: &[f64]| Ok::<_, Infallible>(value),
        };
        let result =
            Executor::new(problem, no_geometry(), PointState::new(vec![0.0]))
                .max_iter(10)
                .run()
                .unwrap();
        assert_eq!(result.iter(), 0);
        assert_eq!(result.state.counts().cost_evals, 1);
        assert_eq!(
            result.reason,
            if value < f64::INFINITY {
                TerminationReason::SolverConverged
            } else {
                TerminationReason::SolverFailed
            }
        );
    }
}

#[test]
fn tolerances_are_optional_and_use_normalized_geometry() {
    for solver in [
        Direct::new().with_absolute_radius_tolerance(0.5),
        no_geometry().with_relative_volume_tolerance(1.0),
    ] {
        let result =
            Executor::new(quadratic(1), solver, PointState::new(vec![99.0]))
                .max_iter(20)
                .run()
                .unwrap();
        assert_eq!(result.reason, TerminationReason::SolverConverged);
        assert_eq!(result.iter(), 0);
    }
    for solver in [
        no_geometry(),
        Direct::new()
            .with_absolute_radius_tolerance(0.0)
            .with_relative_volume_tolerance(0.0),
    ] {
        let result =
            Executor::new(quadratic(1), solver, PointState::new(vec![99.0]))
                .max_iter(3)
                .run()
                .unwrap();
        assert_eq!(result.reason, TerminationReason::MaxIter);
        assert_eq!(result.iter(), 3);
    }
}

#[test]
fn rejected_midpoint_can_recover_and_all_rejections_do_not_converge() {
    for rejection in [f64::INFINITY, f64::NAN] {
        let problem = Objective {
            lower: vec![0.0],
            upper: vec![1.0],
            cost: move |x: &[f64]| {
                Ok::<_, Infallible>(if x[0] >= 0.25 {
                    rejection
                } else {
                    (x[0] - 0.1).powi(2)
                })
            },
        };
        let result =
            Executor::new(problem, Direct::new(), PointState::new(vec![0.0]))
                .target_objective(1e-5)
                .max_iter(30)
                .run()
                .unwrap();
        assert_eq!(result.reason, TerminationReason::TargetCost);
        assert!(result.cost().is_finite());
        let problem = Objective {
            lower: vec![0.0],
            upper: vec![1.0],
            cost: move |_: &[f64]| Ok::<_, Infallible>(rejection),
        };
        let result = Executor::new(
            problem,
            Direct::new().with_absolute_radius_tolerance(1.0),
            PointState::new(vec![0.0]),
        )
        .max_iter(4)
        .run()
        .unwrap();
        assert_eq!(result.reason, TerminationReason::MaxIter);
        assert!(result.state.best().is_none());
        assert_eq!(result.state.counts().cost_evals, 9);
    }
}

#[test]
fn negative_infinity_and_typed_errors() {
    #[derive(Debug, PartialEq)]
    struct Abort;
    for at_init in [true, false] {
        let problem = Objective {
            lower: vec![0.0],
            upper: vec![1.0],
            cost: move |x: &[f64]| {
                if at_init || x[0] != 0.5 {
                    Err(Abort)
                } else {
                    Ok(1.0)
                }
            },
        };
        assert!(matches!(
            Executor::new(problem, Direct::new(), PointState::new(vec![0.0]))
                .run(),
            Err(Abort)
        ));
        let problem = Objective {
            lower: vec![0.0],
            upper: vec![1.0],
            cost: move |x: &[f64]| {
                Ok::<_, Infallible>(if at_init || x[0] != 0.5 {
                    f64::NEG_INFINITY
                } else {
                    1.0
                })
            },
        };
        let result =
            Executor::new(problem, Direct::new(), PointState::new(vec![0.0]))
                .run()
                .unwrap();
        assert_eq!(result.reason, TerminationReason::SolverConverged);
        assert_eq!(result.cost(), f64::NEG_INFINITY);
        assert_eq!(result.state.best_cost(), result.cost());
    }
}

#[test]
fn extreme_bounds_and_roundoff_stop() {
    for (lo, hi) in [
        (-f64::MAX, f64::MAX),
        (f64::MAX / 2.0, f64::MAX),
        (-f64::MAX, -f64::MAX / 2.0),
    ] {
        let problem = Objective {
            lower: vec![lo],
            upper: vec![hi],
            cost: |_: &[f64]| Ok::<_, Infallible>(1.0),
        };
        let result =
            Executor::new(problem, no_geometry(), PointState::new(vec![0.0]))
                .max_iter(2)
                .run()
                .unwrap();
        assert_eq!(result.reason, TerminationReason::MaxIter);
    }
    let problem = Objective {
        lower: vec![1.0],
        upper: vec![f64::from_bits(1.0_f64.to_bits() + 1)],
        cost: |_: &[f64]| Ok::<_, Infallible>(1.0),
    };
    let result =
        Executor::new(problem, no_geometry(), PointState::new(vec![0.0]))
            .max_iter(2)
            .run()
            .unwrap();
    assert_eq!(result.reason, TerminationReason::NumericalNoProgress);
    assert_eq!(result.iter(), 0);
    assert_eq!(result.state.counts().cost_evals, 1);
}

#[test]
fn invalid_inputs_panic_before_evaluation() {
    for (lower, upper, seed) in [
        (vec![], vec![], vec![]),
        (vec![0.0], vec![1.0, 1.0], vec![0.0]),
        (vec![0.0], vec![1.0], vec![]),
        (vec![2.0], vec![1.0], vec![0.0]),
        (vec![f64::NEG_INFINITY], vec![1.0], vec![0.0]),
        (vec![0.0], vec![f64::INFINITY], vec![0.0]),
        (vec![f64::NAN], vec![1.0], vec![0.0]),
    ] {
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let problem = Objective {
            lower,
            upper,
            cost: |_: &[f64]| {
                calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok::<_, Infallible>(1.0)
            },
        };
        assert!(
            std::panic::catch_unwind(|| Executor::new(
                problem,
                Direct::new(),
                PointState::new(seed)
            )
            .run())
            .is_err()
        );
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    }
    for value in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(
            std::panic::catch_unwind(|| Direct::new().with_epsilon(value))
                .is_err()
        );
        assert!(
            std::panic::catch_unwind(
                || Direct::new().with_absolute_radius_tolerance(value)
            )
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(
                || Direct::new().with_relative_volume_tolerance(value)
            )
            .is_err()
        );
    }
}

#[test]
fn checkpoint_continuation_and_fresh_reset() {
    let run = || {
        Executor::new(
            quadratic(2),
            no_geometry(),
            PointState::new(vec![0.0; 2]),
        )
    };
    let full = run().max_iter(12).run_with_solver().unwrap();
    let partial = run().max_iter(4).run_with_solver().unwrap();
    let checkpoint = partial.into_checkpoint();
    #[cfg(feature = "serde")]
    let checkpoint = {
        let bytes = bincode::serde::encode_to_vec(
            &checkpoint,
            bincode::config::standard(),
        )
        .unwrap();
        bincode::serde::decode_from_slice::<
            basin::ExactCheckpoint<Direct, PointState<Vec<f64>>>,
            _,
        >(&bytes, bincode::config::standard())
        .unwrap()
        .0
    };
    let resumed = Executor::resume_from_checkpoint(quadratic(2), checkpoint)
        .max_iter(12)
        .run_with_solver()
        .unwrap();
    assert_eq!(full.state, resumed.state);
    assert_eq!(full.counts, resumed.counts);
    let (mut solver, state, _) = resumed.into_checkpoint().into_parts();
    let mut problem = Problem::new(quadratic(2));
    let reset = solver.init(&mut problem, state).unwrap();
    assert_eq!(reset.iter(), 0);
    assert_eq!(reset.param(), &[0.0, 0.0]);
    assert_eq!(reset.cost(), sphere(&[0.0, 0.0]).unwrap());
    assert_eq!(problem.counts().cost_evals, 1);
}

fn reference_cost(name: &str, x: &[f64]) -> f64 {
    if name.starts_with("styblinski_tang") {
        0.5 * x
            .iter()
            .map(|t| t.powi(4) - 16.0 * t * t + 5.0 * t)
            .sum::<f64>()
    } else {
        let n = x.len() as f64;
        let squared = x.iter().map(|x| (x - 0.37).powi(2)).sum::<f64>() / n;
        let cosine = x
            .iter()
            .map(|x| (2.0 * std::f64::consts::PI * (x - 0.37)).cos())
            .sum::<f64>()
            / n;
        -20.0 * (-0.2 * squared.sqrt()).exp() - cosine.exp()
            + 20.0
            + std::f64::consts::E
    }
}

#[test]
fn scipy_solution_quality_and_evaluation_counts() {
    for line in include_str!("fixtures/direct_reference.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let fields: Vec<_> = line.split('|').collect();
        let name = fields[0];
        let dimension: usize = fields[1].parse().unwrap();
        let budget: u64 = fields[2].parse().unwrap();
        let tolerance: f64 = fields[3].parse().unwrap();
        let minimum: f64 = fields[4].parse().unwrap();
        let reference: f64 = fields[5].parse().unwrap();
        let reference_count: u64 = fields[6].parse().unwrap();
        let reference_x: Vec<f64> =
            fields[8].split(',').map(|x| x.parse().unwrap()).collect();
        assert_eq!(reference_x.len(), dimension);
        assert!((reference_cost(name, &reference_x) - reference).abs() < 1e-11);
        let problem = Objective {
            lower: vec![-5.0; dimension],
            upper: vec![5.0; dimension],
            cost: |x: &[f64]| Ok::<_, Infallible>(reference_cost(name, x)),
        };
        let result = Executor::new(
            problem,
            no_geometry(),
            PointState::new(vec![0.0; dimension]),
        )
        .max_cost_evals(budget)
        .max_iter(1000)
        .run()
        .unwrap();
        let gap = result.cost() - minimum;
        eprintln!(
            "{name}: Basin gap={gap:.8e}, evaluations={}; SciPy gap={:.8e}, evaluations={reference_count}",
            result.state.cost_evals(),
            reference - minimum
        );
        assert_eq!(result.reason, TerminationReason::MaxCostEvals);
        assert!(
            gap >= -1e-11 && gap <= tolerance,
            "{name}: gap={gap}, tolerance={tolerance}"
        );
        assert!(result.state.cost_evals() >= budget);
        assert!(
            result.state.cost_evals() <= 2 * reference_count,
            "{name}: excessive final sweep"
        );
        assert_eq!(reference_cost(name, result.param()), result.cost());
    }
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_pool_sizes_preserve_the_trajectory() {
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                Executor::new(
                    quadratic(3),
                    no_geometry(),
                    PointState::new(vec![0.0; 3]),
                )
                .max_iter(12)
                .run()
                .unwrap()
                .state
            })
    };
    assert_eq!(run(1), run(4));
}
