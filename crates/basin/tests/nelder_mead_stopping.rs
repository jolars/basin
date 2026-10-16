//! Default simplex stopping, scaling, evidence, and continuation contracts.

#[path = "support/nelder_mead_stopping.rs"]
mod backend;

use basin::{
    ConvergenceEvidence, CostFunction, Executor, NelderMead, Problem,
    SimplexProgress, Solver, Termination, TerminationCode, Threshold,
};
use std::convert::Infallible;

struct Sphere;

impl basin::ProblemConstraints for Sphere {
    type Constraints = basin::NoConstraints;
    fn constraint_set(
        &self,
    ) -> <Self::Constraints as basin::ConstraintDescription>::Borrowed<'_> {
        basin::Constraints::new()
    }
}
impl CostFunction for Sphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(x.iter().map(|x| x * x).sum())
    }
}

fn disabled() -> NelderMead<Vec<f64>> {
    NelderMead::new()
        .with_absolute_simplex_size_tolerance(None)
        .with_relative_simplex_size_tolerance(None)
        .with_absolute_simplex_cost_tolerance(None)
        .with_relative_simplex_cost_tolerance(None)
}

fn thresholds(
    termination: &Termination,
) -> (Option<&Threshold>, Option<&Threshold>) {
    let Termination::Converged(convergence) = termination else {
        panic!("expected convergence");
    };
    assert_eq!(convergence.criteria().len(), 1);
    let ConvergenceEvidence::Simplex { size, cost_spread } =
        &convergence.criteria()[0].evidence
    else {
        panic!("expected simplex evidence");
    };
    (size.as_ref(), cost_spread.as_ref())
}

#[test]
fn mixed_thresholds_use_or_within_each_dimension_and_and_between_them() {
    for (absolute, relative, cost_relative, expected) in [
        (1.0, 0.2, 0.5, true),
        (3.0, 0.1, 0.5, true),
        (1.0, 0.2, 0.1, false),
        (1.0, 0.0, 0.5, false),
    ] {
        let mut solver = disabled()
            .with_absolute_simplex_size_tolerance(absolute)
            .with_relative_simplex_size_tolerance(relative)
            .with_relative_simplex_cost_tolerance(cost_relative);
        let mut problem = Problem::new(Sphere);
        let state = solver
            .init(
                &mut problem,
                SimplexProgress::from_simplex(vec![vec![10.0], vec![12.0]]),
            )
            .unwrap();
        let counts = *problem.counts();
        for _ in 0..3 {
            let termination = solver.check_convergence(&problem, &state);
            assert_eq!(termination.is_some(), expected);
            if let Some(termination) = termination {
                let (size, cost) = thresholds(&termination);
                let size = size.unwrap();
                let cost = cost.unwrap();
                assert_eq!(size.value, 2.0);
                assert_eq!(size.tolerance, absolute.max(relative * 10.0));
                assert_eq!(size.absolute_tolerance, Some(absolute));
                assert_eq!(size.relative_tolerance, Some(relative));
                assert_eq!(size.reference, Some(10.0));
                assert_eq!(cost.value, 44.0);
                assert_eq!(cost.reference, Some(100.0));
                assert_eq!(cost.tolerance, cost_relative * 100.0);
            }
            assert_eq!(problem.counts(), &counts);
        }
    }
}

#[test]
fn wrappers_and_projection_preserve_one_owner_in_both_builder_orders() {
    struct BoundedSphere {
        lower: Vec<f64>,
        upper: Vec<f64>,
    }

    impl basin::ProblemConstraints for BoundedSphere {
        type Constraints = basin::Constraints<basin::BoxBounds<Vec<f64>>>;
        fn constraint_set(
            &self,
        ) -> <Self::Constraints as basin::ConstraintDescription>::Borrowed<'_>
        {
            basin::Constraints::new().with_bounds(basin::BoxBounds::new(
                Some(&self.lower),
                Some(&self.upper),
            ))
        }
    }
    impl CostFunction for BoundedSphere {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
            Sphere.cost(x)
        }
    }

    let direct = NelderMead::new()
        .with_absolute_simplex_size_tolerance(None)
        .with_absolute_simplex_cost_tolerance(None)
        .with_relative_simplex_size_tolerance(0.2)
        .with_relative_simplex_cost_tolerance(0.5)
        .projected()
        .with_absolute_cost_change_tolerance(None);
    let wrapped = NelderMead::new()
        .with_absolute_cost_change_tolerance(None)
        .with_relative_simplex_cost_tolerance(100.0)
        .with_absolute_simplex_cost_tolerance(None)
        .projected()
        .with_relative_simplex_size_tolerance(0.2)
        .with_absolute_simplex_size_tolerance(None)
        .with_relative_simplex_cost_tolerance(0.5);
    for solver in [direct, wrapped] {
        let result = Executor::new(
            BoundedSphere {
                lower: vec![0.0],
                upper: vec![100.0],
            },
            solver,
            SimplexProgress::from_simplex(vec![vec![10.0], vec![12.0]]),
        )
        .max_iter(1)
        .run()
        .unwrap();
        let (size, cost) = thresholds(&result.report.termination);
        assert_eq!(size.unwrap().relative_tolerance, Some(0.2));
        assert_eq!(cost.unwrap().relative_tolerance, Some(0.5));
        assert_eq!(size.unwrap().absolute_tolerance, None);
        assert_eq!(cost.unwrap().absolute_tolerance, None);
        assert_eq!(result.iter(), 0);
        assert_eq!(result.cost_evals(), 2);
    }
}

#[test]
fn initial_spread_survives_checks_and_setters_and_resets_on_fresh_init() {
    let mut solver = disabled().with_relative_simplex_cost_tolerance(0.25);
    let mut problem = Problem::new(Sphere);
    let mut state = solver
        .init(
            &mut problem,
            SimplexProgress::from_simplex(vec![vec![0.0], vec![4.0]]),
        )
        .unwrap();
    assert!(solver.check_convergence(&problem, &state).is_none());
    state
        .replace(vec![vec![0.0], vec![1.0]], vec![0.0, 1.0])
        .unwrap();
    solver = solver.with_relative_simplex_cost_tolerance(0.125);
    let termination = solver.check_convergence(&problem, &state).unwrap();
    let (size, cost) = thresholds(&termination);
    assert!(size.is_none());
    assert_eq!(cost.unwrap().reference, Some(16.0));
    assert_eq!(cost.unwrap().tolerance, 2.0);

    let mut fresh = solver
        .init(
            &mut problem,
            SimplexProgress::from_simplex(vec![vec![0.0], vec![2.0]]),
        )
        .unwrap();
    fresh
        .replace(vec![vec![0.0], vec![0.5]], vec![0.0, 0.25])
        .unwrap();
    let termination = solver.check_convergence(&problem, &fresh).unwrap();
    let (_, cost) = thresholds(&termination);
    assert_eq!(cost.unwrap().reference, Some(4.0));
    assert_eq!(cost.unwrap().tolerance, 0.5);
}

#[test]
fn nonfinite_initial_costs_defer_reference_until_first_finite_spread() {
    struct Domain;

    impl basin::ProblemConstraints for Domain {
        type Constraints = basin::NoConstraints;
        fn constraint_set(
            &self,
        ) -> <Self::Constraints as basin::ConstraintDescription>::Borrowed<'_>
        {
            basin::Constraints::new()
        }
    }
    impl CostFunction for Domain {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
            Ok(if x[0] <= 1.0 {
                x[0] * x[0]
            } else {
                f64::INFINITY
            })
        }
    }
    let mut solver = disabled().with_relative_simplex_cost_tolerance(0.25);
    let mut problem = Problem::new(Domain);
    let mut state = solver
        .init(
            &mut problem,
            SimplexProgress::from_simplex(vec![vec![0.0], vec![2.0]]),
        )
        .unwrap();
    assert!(solver.check_convergence(&problem, &state).is_none());
    state
        .replace(vec![vec![0.0], vec![1.0]], vec![0.0, 1.0])
        .unwrap();
    assert!(solver.check_convergence(&problem, &state).is_none());
    state
        .replace(vec![vec![0.0], vec![0.5]], vec![0.0, 0.25])
        .unwrap();
    let termination = solver.check_convergence(&problem, &state).unwrap();
    let (_, cost) = thresholds(&termination);
    assert_eq!(cost.unwrap().reference, Some(1.0));
}

#[test]
fn zero_is_exact_and_complete_opt_out_reaches_budget() {
    for (vertices, size_enabled, expected) in [
        ([-1.0, 1.0], true, false),
        ([-1.0, 1.0], false, true),
        ([-1.0, -1.0], true, true),
    ] {
        let mut solver = disabled()
            .with_relative_simplex_size_tolerance(size_enabled.then_some(0.0))
            .with_relative_simplex_cost_tolerance(0.0);
        let mut problem = Problem::new(Sphere);
        let state = solver
            .init(
                &mut problem,
                SimplexProgress::from_simplex(
                    vertices.map(|x| vec![x]).to_vec(),
                ),
            )
            .unwrap();
        assert_eq!(
            solver.check_convergence(&problem, &state).is_some(),
            expected
        );
    }
    let result = Executor::new(
        Sphere,
        disabled(),
        SimplexProgress::from_simplex(vec![vec![0.0]; 2]),
    )
    .max_iter(7)
    .run()
    .unwrap();
    assert_eq!(result.report.code(), TerminationCode::MaxIter);
    assert_eq!(result.iter(), 7);
}

#[test]
fn executor_limits_precede_default_convergence() {
    let make = || {
        Executor::new(
            Sphere,
            NelderMead::new(),
            SimplexProgress::from_simplex(vec![vec![0.0]; 2]),
        )
    };
    let result = make().max_iter(0).run().unwrap();
    assert_eq!(result.report.code(), TerminationCode::MaxIter);
    let result = make().max_cost_evals(2).run().unwrap();
    assert_eq!(result.report.code(), TerminationCode::MaxCostEvals);
    assert_eq!(result.cost_evals(), 2);
}

#[test]
fn nonfinite_measurements_and_overflowed_bounds_never_converge() {
    let mut solver = NelderMead::<Vec<f64>>::new();
    let problem = Problem::new(Sphere);
    for nonfinite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for costs in [vec![0.0, nonfinite], vec![nonfinite, 0.0]] {
            let mut state = SimplexProgress::from_simplex(vec![vec![0.0]; 2]);
            state.replace(vec![vec![0.0]; 2], costs).unwrap();
            assert!(solver.check_convergence(&problem, &state).is_none());
        }
        let mut state = SimplexProgress::from_simplex(vec![vec![nonfinite]; 2]);
        state
            .replace(vec![vec![nonfinite]; 2], vec![0.0; 2])
            .unwrap();
        assert!(solver.check_convergence(&problem, &state).is_none());
    }
    for (vertices, costs) in [
        (vec![vec![f64::MAX], vec![-f64::MAX]], vec![0.0; 2]),
        (vec![vec![0.0]; 2], vec![f64::MAX, -f64::MAX]),
    ] {
        let mut state = SimplexProgress::from_simplex(vertices.clone());
        state.replace(vertices, costs).unwrap();
        assert!(solver.check_convergence(&problem, &state).is_none());
    }
    let mut problem = Problem::new(Sphere);
    let state = solver
        .init(
            &mut problem,
            SimplexProgress::from_simplex(vec![vec![2.0]; 2]),
        )
        .unwrap();
    for mut solver in [
        NelderMead::new().with_relative_simplex_size_tolerance(f64::MAX),
        NelderMead::new().with_relative_simplex_cost_tolerance(f64::MAX),
    ] {
        assert!(solver.check_convergence(&problem, &state).is_none());
    }
}

#[test]
fn all_four_setters_reject_invalid_tolerances() {
    type Setter = fn(NelderMead<Vec<f64>>, f64) -> NelderMead<Vec<f64>>;
    let setters: [Setter; 4] = [
        |s, v| s.with_absolute_simplex_size_tolerance(v),
        |s, v| s.with_relative_simplex_size_tolerance(v),
        |s, v| s.with_absolute_simplex_cost_tolerance(v),
        |s, v| s.with_relative_simplex_cost_tolerance(v),
    ];
    for setter in setters {
        for invalid in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                std::panic::catch_unwind(|| setter(NelderMead::new(), invalid))
                    .is_err()
            );
        }
    }
}

#[test]
fn exact_continuation_preserves_reference_state_counts_and_report() {
    let make =
        || Executor::from_start(Sphere, NelderMead::new(), vec![2.0, -1.0]);
    let reference = make().max_iter(2_000).run_with_solver().unwrap();
    assert_eq!(reference.report.code(), TerminationCode::SimplexTolerance);
    for split in [0, 1, 7, 30, reference.iter() - 1] {
        let checkpoint = make()
            .max_iter(split)
            .run_with_solver()
            .unwrap()
            .into_checkpoint();
        let resumed = Executor::resume_from_checkpoint(Sphere, checkpoint)
            .require_evaluated_state()
            .max_iter(2_000)
            .run_with_solver()
            .unwrap();
        assert_eq!(resumed.state, reference.state);
        assert_eq!(resumed.counts, reference.counts);
        assert_eq!(resumed.report, reference.report);
    }
}

#[cfg(feature = "serde")]
#[test]
fn serialized_checkpoint_preserves_reference_and_evidence() {
    let make =
        || Executor::from_start(Sphere, NelderMead::new(), vec![2.0, -1.0]);
    let reference = make().max_iter(2_000).run_with_solver().unwrap();
    for split in [0, 7, 30] {
        let checkpoint = make()
            .max_iter(split)
            .run_with_solver()
            .unwrap()
            .into_checkpoint();
        let bytes = postcard::to_allocvec(&checkpoint).unwrap();
        let decoded: basin::ExactCheckpoint<
            NelderMead<Vec<f64>>,
            SimplexProgress<Vec<f64>>,
        > = postcard::from_bytes(&bytes).unwrap();
        let resumed = Executor::resume_from_checkpoint(Sphere, decoded)
            .require_evaluated_state()
            .max_iter(2_000)
            .run_with_solver()
            .unwrap();
        assert_eq!(resumed.state, reference.state);
        assert_eq!(resumed.counts, reference.counts);
        assert_eq!(resumed.report, reference.report);
        assert_eq!(
            postcard::to_allocvec(&resumed.solver).unwrap(),
            postcard::to_allocvec(&reference.solver).unwrap()
        );
        let report_bytes = postcard::to_allocvec(&resumed.report).unwrap();
        let report: basin::TerminationReport =
            postcard::from_bytes(&report_bytes).unwrap();
        assert_eq!(report, reference.report);
    }
}

#[test]
fn vec_f32_defaults() {
    backend::check::<Vec<f32>, f32>(|x| x.to_vec(), 32.0 * f32::EPSILON);
}

#[test]
fn vec_f64_defaults() {
    backend::check::<Vec<f64>, f64>(|x| x.to_vec(), f64::EPSILON.sqrt());
}

macro_rules! nalgebra_tests {
    ($feature:literal, $module:ident, $backend:ident) => {
        #[cfg(feature = $feature)]
        mod $module {
            #[test]
            fn f32_defaults() {
                super::backend::check::<$backend::DVector<f32>, f32>(
                    |x| $backend::DVector::from_column_slice(x),
                    32.0 * f32::EPSILON,
                );
            }
            #[test]
            fn f64_defaults() {
                super::backend::check::<$backend::DVector<f64>, f64>(
                    |x| $backend::DVector::from_column_slice(x),
                    f64::EPSILON.sqrt(),
                );
            }
        }
    };
}

macro_rules! ndarray_tests {
    ($feature:literal, $module:ident, $backend:ident) => {
        #[cfg(feature = $feature)]
        mod $module {
            #[test]
            fn f32_defaults() {
                super::backend::check::<$backend::Array1<f32>, f32>(
                    |x| $backend::Array1::from_vec(x.to_vec()),
                    32.0 * f32::EPSILON,
                );
            }
            #[test]
            fn f64_defaults() {
                super::backend::check::<$backend::Array1<f64>, f64>(
                    |x| $backend::Array1::from_vec(x.to_vec()),
                    f64::EPSILON.sqrt(),
                );
            }
        }
    };
}

macro_rules! faer_tests {
    ($feature:literal, $module:ident, $backend:ident) => {
        #[cfg(feature = $feature)]
        mod $module {
            #[test]
            fn f32_defaults() {
                super::backend::check::<$backend::Col<f32>, f32>(
                    |x| $backend::Col::from_fn(x.len(), |i| x[i]),
                    32.0 * f32::EPSILON,
                );
            }
            #[test]
            fn f64_defaults() {
                super::backend::check::<$backend::Col<f64>, f64>(
                    |x| $backend::Col::from_fn(x.len(), |i| x[i]),
                    f64::EPSILON.sqrt(),
                );
            }
        }
    };
}

nalgebra_tests!("nalgebra_v0_32", nalgebra32, nalgebra_0_32);
nalgebra_tests!("nalgebra_v0_33", nalgebra33, nalgebra_0_33);
nalgebra_tests!("nalgebra_v0_34", nalgebra34, nalgebra_0_34);
nalgebra_tests!("nalgebra_v0_35", nalgebra35, nalgebra);
ndarray_tests!("ndarray_v0_15", ndarray15, ndarray_0_15);
ndarray_tests!("ndarray_v0_16", ndarray16, ndarray_0_16);
ndarray_tests!("ndarray_v0_17", ndarray17, ndarray);
faer_tests!("faer_v0_22", faer22, faer_0_22);
faer_tests!("faer_v0_23", faer23, faer_0_23);
faer_tests!("faer_v0_24", faer24, faer);
