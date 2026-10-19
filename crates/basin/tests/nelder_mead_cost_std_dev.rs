//! Sample cost dispersion, composition, and continuation across dense backends.

use basin::{
    ClampInPlace, ConvergenceEvidence, ConvergenceTest, CostFunction, Executor,
    NelderMead, Problem, SamplingDomain, Scalar, ScaleInPlace, ScaledAdd,
    SimplexProgress, Solver, State, Termination, TerminationCode, VectorIndex,
    VectorLen,
};
use std::{convert::Infallible, marker::PhantomData};

struct Quadratic<V, F> {
    lower: V,
    upper: V,
    _float: PhantomData<F>,
}

impl<V, F> basin::ProblemConstraints for Quadratic<V, F> {
    type Constraints = basin::Constraints<basin::BoxBounds<V>>;

    fn constraint_set(
        &self,
    ) -> <Self::Constraints as basin::ConstraintDescription>::Borrowed<'_> {
        basin::Constraints::new().with_bounds(basin::BoxBounds::new(
            Some(&self.lower),
            Some(&self.upper),
        ))
    }
}

impl<V: VectorLen + VectorIndex<F>, F: Scalar> CostFunction
    for Quadratic<V, F>
{
    type Param = V;
    type Output = F;
    type Error = Infallible;

    fn cost(&self, x: &V) -> Result<F, Infallible> {
        Ok((0..x.vec_len()).map(|i| x.get_scalar(i).powi(2)).sum())
    }
}

fn quadratic<V, F: Scalar>(make: impl Fn(&[F]) -> V) -> Quadratic<V, F> {
    Quadratic {
        lower: make(&[F::zero(); 2]),
        upper: make(&[F::one(); 2]),
        _float: PhantomData,
    }
}

fn std_dev_only<V, F: Scalar>() -> NelderMead<V, F> {
    NelderMead::new()
        .with_absolute_simplex_size_tolerance(None)
        .with_relative_simplex_size_tolerance(None)
        .with_absolute_simplex_cost_tolerance(None)
        .with_relative_simplex_cost_tolerance(None)
}

fn evaluated<V: VectorLen, F: Scalar>(
    make: impl Fn(&[F]) -> V,
    costs: &[F],
) -> SimplexProgress<V, F> {
    let vertices = || {
        (0..costs.len())
            .map(|i| make(&vec![F::from_usize(i).unwrap(); costs.len() - 1]))
            .collect()
    };
    let mut state = SimplexProgress::from_simplex(vertices());
    state.replace(vertices(), costs.to_vec()).unwrap();
    state
}

fn tests<F: Scalar>(termination: &Termination<F>) -> Vec<ConvergenceTest> {
    let Termination::Converged(convergence) = termination else {
        panic!("expected convergence, got {termination:?}");
    };
    convergence
        .criteria()
        .iter()
        .map(|c| c.test.clone())
        .collect()
}

fn std_dev_value<F: Scalar>(termination: &Termination<F>, expected: F) -> F {
    let Termination::Converged(convergence) = termination else {
        panic!("expected convergence, got {termination:?}");
    };
    let criterion = convergence
        .criteria()
        .iter()
        .find(|c| c.test == ConvergenceTest::AbsoluteSimplexCostStdDev)
        .expect("standard-deviation evidence");
    let ConvergenceEvidence::UpperBound {
        value,
        bound,
        tolerance,
        reference,
    } = criterion.evidence
    else {
        panic!("expected upper-bound evidence");
    };
    assert_eq!(bound, expected);
    assert_eq!(tolerance, expected);
    assert_eq!(reference, None);
    assert!(value.is_finite() && value <= bound);
    assert_eq!(termination.code(), TerminationCode::SimplexTolerance);
    value
}

fn check_backend<V, F>(make: impl Fn(&[F]) -> V)
where
    F: Scalar,
    V: Clone
        + VectorLen
        + VectorIndex<F>
        + ScaleInPlace<F>
        + ScaledAdd<F>
        + ClampInPlace,
{
    let f = |value| F::from_f64(value).unwrap();
    let problem = Problem::new(SamplingDomain::new(quadratic(&make)));
    let check = |costs: &[F], expected: F, tolerance: F| {
        let state = evaluated(&make, costs);
        let mut solver = std_dev_only()
            .with_absolute_simplex_cost_std_dev_tolerance(tolerance);
        let counts = *problem.counts();
        for _ in 0..3 {
            let termination =
                solver.check_convergence(&problem, &state).unwrap();
            assert_eq!(
                tests(&termination),
                [ConvergenceTest::AbsoluteSimplexCostStdDev]
            );
            let value = std_dev_value(&termination, tolerance);
            if expected == F::zero() {
                assert_eq!(value, F::zero());
            } else {
                assert!(
                    (value / expected - F::one()).abs()
                        <= f(16.0) * F::epsilon()
                );
            }
            assert_eq!(problem.counts(), &counts);
        }
    };

    // A population denominator would pass at 0.9; the sample deviation is one.
    let state = evaluated(&make, &[f(1.0), f(2.0), f(3.0)]);
    let mut solver =
        std_dev_only().with_absolute_simplex_cost_std_dev_tolerance(f(0.9));
    assert!(solver.check_convergence(&problem, &state).is_none());
    check(&[f(1.0), f(2.0), f(3.0)], F::one(), F::one());
    check(&[F::zero(), f(2.0)], f(2.0).sqrt(), f(2.0));
    check(&[f(-2.0), F::one(), f(7.0), f(6.0)], f(18.0).sqrt(), f(5.0));

    for scale in [F::min_positive_value(), F::one(), F::max_value() / f(16.0)] {
        check(&[-scale, F::zero(), scale], scale, scale);
    }
    let offset = F::epsilon().recip();
    check(&[offset, offset + f(2.0), offset + f(4.0)], f(2.0), f(2.0));
    check(&[F::max_value(); 3], F::zero(), F::zero());
    check(
        &[-F::max_value(), F::zero(), F::max_value()],
        F::max_value(),
        F::max_value(),
    );
    let state = evaluated(&make, &[-F::max_value(), F::zero(), F::max_value()]);
    let mut solver = NelderMead::new()
        .with_absolute_simplex_cost_std_dev_tolerance(F::max_value());
    let termination = solver.check_convergence(&problem, &state).unwrap();
    assert_eq!(
        tests(&termination),
        [ConvergenceTest::AbsoluteSimplexCostStdDev]
    );

    let mut solver = std_dev_only()
        .with_absolute_simplex_cost_std_dev_tolerance(F::max_value());
    let state = evaluated(&make, &[-F::max_value(), F::max_value()]);
    assert!(solver.check_convergence(&problem, &state).is_none());
    for nonfinite in [F::nan(), F::infinity(), F::neg_infinity()] {
        for costs in [
            vec![nonfinite, F::one()],
            vec![F::one(), nonfinite],
            vec![nonfinite; 3],
        ] {
            let state = evaluated(&make, &costs);
            assert!(solver.check_convergence(&problem, &state).is_none());
        }
    }
    let unevaluated =
        SimplexProgress::from_simplex(vec![make(&[F::zero()]); 2]);
    assert!(solver.check_convergence(&problem, &unevaluated).is_none());

    // Final rounding can erase a subnormal deviation, but zero stays exact.
    let tiny = F::min_positive_value() * F::epsilon();
    let state =
        evaluated(&make, &[F::zero(), F::zero(), F::zero(), F::zero(), tiny]);
    let mut solver =
        std_dev_only().with_absolute_simplex_cost_std_dev_tolerance(F::zero());
    assert!(solver.check_convergence(&problem, &state).is_none());
    solver = solver.with_absolute_simplex_cost_std_dev_tolerance(tiny);
    assert!(solver.check_convergence(&problem, &state).is_some());

    let flat = evaluated(&make, &[F::one(); 3]);
    for mut solver in [
        NelderMead::new(),
        NelderMead::adaptive(),
        NelderMead::default(),
        NelderMead::with_params(F::one(), f(2.0), f(0.5), f(0.5)),
    ] {
        assert!(solver.check_convergence(&problem, &flat).is_none());
        solver = solver.with_absolute_simplex_cost_std_dev_tolerance(F::zero());
        assert!(solver.check_convergence(&problem, &flat).is_some());
        solver = solver.with_absolute_simplex_cost_std_dev_tolerance(None);
        assert!(solver.check_convergence(&problem, &flat).is_none());

        // Projected initialization must evaluate clipped vertices first.
        let seed = SimplexProgress::from_simplex(vec![
            make(&[f(-10.0), F::zero()]),
            make(&[f(0.5), F::zero()]),
            make(&[f(2.0), F::zero()]),
        ]);
        let result = Executor::new(
            quadratic(&make),
            solver
                .projected()
                .with_absolute_simplex_cost_std_dev_tolerance(f(0.6)),
            seed,
        )
        .max_iter(1)
        .run()
        .unwrap();
        assert_eq!(result.iter(), 0);
        assert_eq!(result.cost_evals(), 3);
        assert_eq!(
            tests(&result.report.termination),
            [ConvergenceTest::AbsoluteSimplexCostStdDev]
        );
        let value = std_dev_value(&result.report.termination, f(0.6));
        assert!(
            (value - (f(13.0) / f(48.0)).sqrt()).abs() <= f(4.0) * F::epsilon()
        );
    }
}

#[test]
fn vec_f32() {
    check_backend::<Vec<f32>, f32>(|x| x.to_vec());
}

#[test]
fn vec_f64() {
    check_backend::<Vec<f64>, f64>(|x| x.to_vec());
}

macro_rules! backend_tests {
    ($feature:literal, $module:ident, $vector:ty, $make:expr) => {
        #[cfg(feature = $feature)]
        mod $module {
            type V<F> = $vector;
            #[test]
            fn f32() {
                super::check_backend::<V<f32>, f32>($make);
            }
            #[test]
            fn f64() {
                super::check_backend::<V<f64>, f64>($make);
            }
        }
    };
}

backend_tests!(
    "nalgebra_v0_32",
    nalgebra32,
    nalgebra_0_32::DVector<F>,
    nalgebra_0_32::DVector::from_column_slice
);
backend_tests!(
    "nalgebra_v0_33",
    nalgebra33,
    nalgebra_0_33::DVector<F>,
    nalgebra_0_33::DVector::from_column_slice
);
backend_tests!(
    "nalgebra_v0_34",
    nalgebra34,
    nalgebra_0_34::DVector<F>,
    nalgebra_0_34::DVector::from_column_slice
);
backend_tests!(
    "nalgebra_v0_35",
    nalgebra35,
    nalgebra::DVector<F>,
    nalgebra::DVector::from_column_slice
);
backend_tests!("ndarray_v0_15", ndarray15, ndarray_0_15::Array1<F>, |x| {
    ndarray_0_15::Array1::from_vec(x.to_vec())
});
backend_tests!("ndarray_v0_16", ndarray16, ndarray_0_16::Array1<F>, |x| {
    ndarray_0_16::Array1::from_vec(x.to_vec())
});
backend_tests!("ndarray_v0_17", ndarray17, ndarray::Array1<F>, |x| {
    ndarray::Array1::from_vec(x.to_vec())
});
backend_tests!("faer_v0_22", faer22, faer_0_22::Col<F>, |x| {
    faer_0_22::Col::from_fn(x.len(), |i| x[i])
});
backend_tests!("faer_v0_23", faer23, faer_0_23::Col<F>, |x| {
    faer_0_23::Col::from_fn(x.len(), |i| x[i])
});
backend_tests!("faer_v0_24", faer24, faer::Col<F>, |x| faer::Col::from_fn(
    x.len(),
    |i| x[i]
));

#[test]
fn simplex_predicates_use_or_and_retain_both_passing_reports() {
    let problem =
        Problem::new(SamplingDomain::new(quadratic(|x: &[f64]| x.to_vec())));
    let state = evaluated(|x| x.to_vec(), &[1.0, 2.0, 3.0]);
    for (spread, std_dev, expected) in [
        (2.0, 0.5, vec![ConvergenceTest::Simplex]),
        (0.5, 1.0, vec![ConvergenceTest::AbsoluteSimplexCostStdDev]),
        (
            2.0,
            1.0,
            vec![
                ConvergenceTest::Simplex,
                ConvergenceTest::AbsoluteSimplexCostStdDev,
            ],
        ),
        (0.5, 0.5, vec![]),
    ] {
        let mut solver = std_dev_only()
            .with_absolute_simplex_cost_tolerance(spread)
            .with_absolute_simplex_cost_std_dev_tolerance(std_dev);
        let termination = solver.check_convergence(&problem, &state);
        assert_eq!(
            termination.as_ref().map(tests).unwrap_or_default(),
            expected
        );
    }
}

#[test]
fn wrappers_forward_replacements_and_preserve_other_tests() {
    let problem =
        Problem::new(SamplingDomain::new(quadratic(|x: &[f64]| x.to_vec())));
    let mut state = evaluated(|x| x.to_vec(), &[1.0, 2.0, 3.0]);
    let direct = std_dev_only()
        .with_absolute_simplex_cost_std_dev_tolerance(1.0)
        .with_absolute_cost_change_tolerance(0.0);
    let wrapped = std_dev_only()
        .with_absolute_cost_change_tolerance(0.0)
        .with_absolute_simplex_cost_std_dev_tolerance(100.0)
        .with_absolute_simplex_cost_std_dev_tolerance(1.0);
    for mut solver in [direct, wrapped] {
        let termination = solver.check_convergence(&problem, &state).unwrap();
        assert_eq!(
            tests(&termination),
            [ConvergenceTest::AbsoluteSimplexCostStdDev]
        );
        state.increment_iter();
        let termination = solver.check_convergence(&problem, &state).unwrap();
        assert_eq!(
            tests(&termination),
            [
                ConvergenceTest::AbsoluteCostChange,
                ConvergenceTest::AbsoluteSimplexCostStdDev
            ]
        );
        solver = solver.with_absolute_simplex_cost_std_dev_tolerance(None);
        state.increment_iter();
        let termination = solver.check_convergence(&problem, &state).unwrap();
        assert_eq!(tests(&termination), [ConvergenceTest::AbsoluteCostChange]);
    }

    let direct = std_dev_only()
        .with_absolute_simplex_cost_std_dev_tolerance(1.0)
        .projected()
        .with_absolute_cost_change_tolerance(None);
    let wrapped = std_dev_only()
        .with_absolute_cost_change_tolerance(None)
        .with_absolute_simplex_cost_std_dev_tolerance(0.5)
        .projected()
        .with_absolute_simplex_cost_std_dev_tolerance(1.0);
    let problem = Problem::new(quadratic(|x: &[f64]| x.to_vec()));
    for mut solver in [direct, wrapped] {
        let termination = solver.check_convergence(&problem, &state).unwrap();
        assert_eq!(std_dev_value(&termination, 1.0), 1.0);
    }
}

#[test]
fn setter_rejects_invalid_tolerances() {
    for tolerance in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            std::panic::catch_unwind(|| {
                NelderMead::<Vec<f64>>::new()
                    .with_absolute_simplex_cost_std_dev_tolerance(tolerance)
            })
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| {
                NelderMead::<Vec<f64>>::new()
                    .with_absolute_cost_change_tolerance::<f64>(None)
                    .with_absolute_simplex_cost_std_dev_tolerance(tolerance)
            })
            .is_err()
        );
    }
}

#[test]
fn executor_limits_precede_the_optional_check() {
    let result = Executor::new(
        SamplingDomain::new(quadratic(|x: &[f64]| x.to_vec())),
        std_dev_only().with_absolute_simplex_cost_std_dev_tolerance(0.0),
        SimplexProgress::from_simplex(vec![
            vec![1.0, 0.0],
            vec![-1.0, 0.0],
            vec![0.0, 1.0],
        ]),
    )
    .max_iter(0)
    .run()
    .unwrap();
    assert_eq!(result.report.code(), TerminationCode::MaxIter);
    assert_eq!(result.cost_evals(), 3);
}

#[test]
fn fresh_and_exact_continuation_preserve_the_setting_and_result() {
    let problem = || SamplingDomain::new(quadratic(|x: &[f64]| x.to_vec()));
    let solver =
        || std_dev_only().with_absolute_simplex_cost_std_dev_tolerance(1e-5);
    let make = || Executor::from_start(problem(), solver(), vec![2.0, -1.0]);
    let reference = make().max_iter(2_000).run_with_solver().unwrap();
    assert_eq!(
        tests(&reference.report.termination),
        [ConvergenceTest::AbsoluteSimplexCostStdDev]
    );
    assert!(reference.iter() > 1 && reference.iter() < 2_000);
    assert!(reference.cost() < 1e-4);
    for split in [0, 1, 7, reference.iter() - 1] {
        let checkpoint = make()
            .max_iter(split)
            .run_with_solver()
            .unwrap()
            .into_checkpoint();
        #[cfg(feature = "serde")]
        let checkpoint = {
            let bytes = postcard::to_allocvec(&checkpoint).unwrap();
            postcard::from_bytes::<
                basin::ExactCheckpoint<
                    NelderMead<Vec<f64>>,
                    SimplexProgress<Vec<f64>>,
                >,
            >(&bytes)
            .unwrap()
        };
        let resumed = Executor::resume_from_checkpoint(problem(), checkpoint)
            .max_iter(2_000)
            .run_with_solver()
            .unwrap();
        assert_eq!(resumed.state, reference.state);
        assert_eq!(resumed.counts, reference.counts);
        assert_eq!(resumed.report, reference.report);
    }
    #[cfg(feature = "serde")]
    {
        let bytes = postcard::to_allocvec(&reference.report).unwrap();
        let decoded: basin::TerminationReport =
            postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, reference.report);
    }
    let fresh =
        Executor::from_start(problem(), reference.solver, vec![2.0, -1.0])
            .max_iter(2_000)
            .run_with_solver()
            .unwrap();
    assert_eq!(fresh.state, reference.state);
    assert_eq!(fresh.counts, reference.counts);
    assert_eq!(fresh.report, reference.report);
}
