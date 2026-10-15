//! Source stopping predicates, finite publication, and convergence history.

use basin::{
    Constant, ConvergenceEvidence, ConvergenceTest, CostFunction, Executor,
    FirstOrderState, Gradient, Lbfgsb, State, Termination, TerminationCode,
};
use std::convert::Infallible;

macro_rules! stopping_backend {
    ($name:ident, $float:ty, $vector:ty, $make:expr) => {
        mod $name {
            use super::*;
            type Float = $float;
            type Vector = $vector;

            struct Quadratic {
                lower: Vector,
                upper: Vector,
                cost: Option<Float>,
                gradient: Option<Float>,
            }
            impl Quadratic {
                fn new() -> Self {
                    Self {
                        lower: ($make)(vec![Float::NEG_INFINITY]),
                        upper: ($make)(vec![Float::INFINITY]),
                        cost: None,
                        gradient: None,
                    }
                }
            }

            impl basin::ProblemConstraints for Quadratic {
                type Constraints = basin::Constraints<basin::BoxBounds<Vector>>;
                fn constraint_set(
                    &self,
                ) -> <Self::Constraints as basin::ConstraintDescription>::Borrowed<'_> {
                    basin::Constraints::new().with_bounds(basin::BoxBounds::new(
                        Some(&self.lower),
                        Some(&self.upper),
                    ))
                }
            }
            impl CostFunction for Quadratic {
                type Param = Vector;
                type Output = Float;
                type Error = Infallible;
                fn cost(&self, x: &Vector) -> Result<Float, Infallible> {
                    Ok(self.cost.unwrap_or(0.5 * x[0] * x[0]))
                }
            }
            impl Gradient for Quadratic {
                type Gradient = Vector;
                fn gradient(&self, x: &Vector) -> Result<Vector, Infallible> {
                    Ok(($make)(vec![self.gradient.unwrap_or(x[0])]))
                }
            }


            #[test]
            fn nonfinite_initial_cost_or_gradient_cannot_converge_at_fixed_bounds() {
                for invalid in [Float::NAN, Float::INFINITY, Float::NEG_INFINITY] {
                    for bad_cost in [false, true] {
                        let mut problem = Quadratic::new();
                        problem.lower = ($make)(vec![0.5]);
                        problem.upper = ($make)(vec![0.5]);
                        if bad_cost {
                            problem.cost = Some(invalid);
                        } else {
                            problem.gradient = Some(invalid);
                        }
                        let result = Executor::new(
                            problem,
                            Lbfgsb::new(),
                            FirstOrderState::new(($make)(vec![0.5])),
                        )
                        .max_iter(2)
                        .run()
                        .unwrap();
                        assert_eq!(result.report.code(), TerminationCode::SolverFailed);
                        assert_eq!(result.cost_evals(), 1);
                    }
                }
            }

            #[test]
            fn source_cost_floor_stops_after_an_accepted_step_without_stationarity() {
                let result = Executor::new(
                    Quadratic::new(),
                    Lbfgsb::with_line_search(Constant::new(1e-4 as Float))
                        .with_absolute_projected_gradient_tolerance(None)
                        .with_relative_cost_reduction_tolerance(1e-4 as Float),
                    FirstOrderState::new(($make)(vec![0.5])),
                )
                .max_iter(5)
                .run()
                .unwrap();
                assert_eq!(result.state.iter(), 1);
                assert_eq!(result.cost_evals(), 2);
                let Termination::Converged(c) = &result.report.termination else {
                    panic!("not converged");
                };
                assert_eq!(c.criteria()[0].test, ConvergenceTest::RelativeCostReduction);
                let ConvergenceEvidence::UpperBound { value, bound, reference, .. } = c.criteria()[0].evidence else {
                    panic!("missing evidence");
                };
                assert_eq!(reference, Some(1.));
                assert!(value <= bound && value > bound * 0.125);
                assert_eq!(result.report.code(), TerminationCode::RelativeCostTolerance);
            }

            #[test]
            fn signed_reduction_and_gradient_precedence_match_the_source() {
                let increased = Executor::new(
                    Quadratic::new(),
                    Lbfgsb::with_line_search(Constant::new(3. as Float))
                        .with_absolute_projected_gradient_tolerance(None)
                        .with_relative_cost_reduction_tolerance(0. as Float),
                    FirstOrderState::new(($make)(vec![0.5])),
                )
                .max_iter(5)
                .run()
                .unwrap();
                assert!(increased.cost() > 0.125);
                assert_eq!(increased.report.code(), TerminationCode::RelativeCostTolerance);
                let stationary = Executor::new(
                    Quadratic::new(),
                    Lbfgsb::with_line_search(Constant::new(1. as Float))
                        .with_absolute_projected_gradient_tolerance(0. as Float)
                        .with_relative_cost_reduction_tolerance(1. as Float),
                    FirstOrderState::new(($make)(vec![0.5])),
                )
                .max_iter(5)
                .run()
                .unwrap();
                let Termination::Converged(c) = &stationary.report.termination else {
                    panic!("not converged");
                };
                assert_eq!(c.criteria().len(), 1);
                assert_eq!(c.criteria()[0].test, ConvergenceTest::BoundClippedGradient);
            }
        }
    };
}

stopping_backend!(vec_f64, f64, Vec<f64>, |v| v);
stopping_backend!(vec_f32, f32, Vec<f32>, |v| v);
#[cfg(feature = "nalgebra_all")]
stopping_backend!(
    nalgebra_f64,
    f64,
    backend_aliases::nalgebra::DVector<f64>,
    backend_aliases::nalgebra::DVector::from_vec
);
#[cfg(feature = "nalgebra_all")]
stopping_backend!(
    nalgebra_f32,
    f32,
    backend_aliases::nalgebra::DVector<f32>,
    backend_aliases::nalgebra::DVector::from_vec
);
#[cfg(feature = "ndarray_all")]
stopping_backend!(
    ndarray_f64,
    f64,
    backend_aliases::ndarray::Array1<f64>,
    backend_aliases::ndarray::Array1::from_vec
);
#[cfg(feature = "ndarray_all")]
stopping_backend!(
    ndarray_f32,
    f32,
    backend_aliases::ndarray::Array1<f32>,
    backend_aliases::ndarray::Array1::from_vec
);
#[cfg(feature = "faer_all")]
stopping_backend!(faer_f64, f64, backend_aliases::faer::Col<f64>, |v: Vec<
    f64,
>| {
    backend_aliases::faer::Col::from_fn(v.len(), |i| v[i])
});
#[cfg(feature = "faer_all")]
stopping_backend!(faer_f32, f32, backend_aliases::faer::Col<f32>, |v: Vec<
    f32,
>| {
    backend_aliases::faer::Col::from_fn(v.len(), |i| v[i])
});

#[path = "support/backend_aliases.rs"]
mod backend_aliases;

struct NonfiniteTrial {
    invalid: f64,
    bad_cost: bool,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl basin::ProblemConstraints for NonfiniteTrial {
    type Constraints = basin::Constraints<basin::BoxBounds<Vec<f64>>>;
    fn constraint_set(
        &self,
    ) -> <Self::Constraints as basin::ConstraintDescription>::Borrowed<'_> {
        basin::Constraints::new().with_bounds(basin::BoxBounds::new(
            Some(&self.lower),
            Some(&self.upper),
        ))
    }
}
impl CostFunction for NonfiniteTrial {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(if x[0] != 1. && self.bad_cost {
            self.invalid
        } else {
            0.5 * x[0] * x[0]
        })
    }
}

impl Gradient for NonfiniteTrial {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(vec![if x[0] != 1. && !self.bad_cost {
            self.invalid
        } else {
            x[0]
        }])
    }
}

#[test]
fn uncached_nonfinite_trials_preserve_the_last_finite_iterate_in_both_modes() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for bad_cost in [false, true] {
            for bounded in [false, true] {
                let problem = NonfiniteTrial {
                    invalid,
                    bad_cost,
                    lower: vec![f64::NEG_INFINITY],
                    upper: vec![f64::INFINITY],
                };
                let solver = Lbfgsb::with_line_search(Constant::new(0.5));
                let state = FirstOrderState::new(vec![1.]);
                let result = if bounded {
                    Executor::new(problem, solver, state)
                        .max_iter(3)
                        .run()
                        .unwrap()
                } else {
                    Executor::new(
                        basin::SamplingDomain::new(problem),
                        solver.unbounded(),
                        state,
                    )
                    .max_iter(3)
                    .run()
                    .unwrap()
                };
                assert_eq!(result.report.code(), TerminationCode::SolverFailed);
                assert_eq!(result.param(), &[1.]);
                assert_eq!(result.best_param(), &[1.]);
                assert_eq!(result.cost(), 0.5);
                assert_eq!(
                    basin::GradientState::gradient(&result.state),
                    Some(&vec![1.])
                );
                assert_eq!(result.cost_evals(), 2);
                assert_eq!(result.counts.gradient_evals, 2);
            }
        }
    }
}

struct RoundedLinear<F> {
    base: F,
    slope: F,
    lower: Vec<F>,
    upper: Vec<F>,
}

impl<F: basin::Scalar> RoundedLinear<F> {
    fn new(base: F, slope: F) -> Self {
        Self {
            base,
            slope,
            lower: vec![F::neg_infinity()],
            upper: vec![F::infinity()],
        }
    }
}

impl<F: basin::Scalar> basin::ProblemConstraints for RoundedLinear<F> {
    type Constraints = basin::Constraints<basin::BoxBounds<Vec<F>>>;
    fn constraint_set(
        &self,
    ) -> <Self::Constraints as basin::ConstraintDescription>::Borrowed<'_> {
        basin::Constraints::new().with_bounds(basin::BoxBounds::new(
            Some(&self.lower),
            Some(&self.upper),
        ))
    }
}
impl<F: basin::Scalar> CostFunction for RoundedLinear<F> {
    type Param = Vec<F>;
    type Output = F;
    type Error = Infallible;
    fn cost(&self, x: &Vec<F>) -> Result<F, Infallible> {
        Ok(self.base + self.slope * x[0])
    }
}
impl<F: basin::Scalar> Gradient for RoundedLinear<F> {
    type Gradient = Vec<F>;
    fn gradient(&self, _: &Vec<F>) -> Result<Vec<F>, Infallible> {
        Ok(vec![self.slope])
    }
}

#[test]
fn precision_defaults_and_explicit_legacy_configuration_remain_distinct() {
    let double = Executor::new(
        RoundedLinear::new(1., 1e-9),
        Lbfgsb::with_line_search(Constant::new(0.1)),
        FirstOrderState::new(vec![0.5]),
    )
    .max_iter(2)
    .run()
    .unwrap();
    assert_eq!(
        double.report.code(),
        TerminationCode::ProjectedGradientTolerance
    );
    assert_eq!(double.state.iter(), 0);
    let single = Executor::new(
        RoundedLinear::new(1_f32, 1e-9_f32),
        Lbfgsb::with_line_search(Constant::new(0.1_f32)),
        FirstOrderState::new(vec![0.5_f32]),
    )
    .max_iter(2)
    .run()
    .unwrap();
    assert_eq!(single.report.code(), TerminationCode::MaxIter);
    let legacy = Executor::new(
        RoundedLinear::new(1., 1e-9),
        Lbfgsb::with_line_search(Constant::new(0.1))
            .with_absolute_projected_gradient_tolerance(1e-10)
            .with_relative_cost_reduction_tolerance(None),
        FirstOrderState::new(vec![0.5]),
    )
    .max_iter(2)
    .run()
    .unwrap();
    assert_eq!(legacy.report.code(), TerminationCode::MaxIter);

    let double_cost = Executor::new(
        RoundedLinear::new(1e20, 1.),
        Lbfgsb::with_line_search(Constant::new(0.1)),
        FirstOrderState::new(vec![0.5]),
    )
    .max_iter(2)
    .run()
    .unwrap();
    assert_eq!(
        double_cost.report.code(),
        TerminationCode::RelativeCostTolerance
    );
    let single_cost = Executor::new(
        RoundedLinear::new(1e20_f32, 1_f32),
        Lbfgsb::with_line_search(Constant::new(0.1_f32)),
        FirstOrderState::new(vec![0.5_f32]),
    )
    .max_iter(2)
    .run()
    .unwrap();
    assert_eq!(single_cost.report.code(), TerminationCode::MaxIter);
}

#[test]
fn zero_disabling_and_configured_setter_order_preserve_the_native_setting() {
    let make = || {
        Executor::new(
            RoundedLinear::new(1e20, 1.),
            Lbfgsb::with_line_search(Constant::new(0.1))
                .with_relative_step_tolerance::<Vec<f64>, f64>(None)
                .with_absolute_projected_gradient_tolerance(None)
                .with_relative_cost_reduction_tolerance(0.)
                .unbounded()
                .bounded(),
            FirstOrderState::new(vec![0.5]),
        )
    };
    let exact = make().max_iter(3).run().unwrap();
    assert_eq!(exact.report.code(), TerminationCode::RelativeCostTolerance);
    assert_eq!(exact.state.iter(), 1);
    let disabled = Executor::new(
        RoundedLinear::new(1e20, 1.),
        Lbfgsb::with_line_search(Constant::new(0.1))
            .with_relative_cost_reduction_tolerance(None)
            .with_absolute_projected_gradient_tolerance(None),
        FirstOrderState::new(vec![0.5]),
    )
    .max_iter(3)
    .run()
    .unwrap();
    assert_eq!(disabled.report.code(), TerminationCode::MaxIter);
}

#[test]
fn cost_history_survives_exact_resume_and_resets_for_fresh_runs() {
    let make = || {
        Executor::new(
            RoundedLinear::new(1e20, 1.),
            Lbfgsb::with_line_search(Constant::new(0.1))
                .with_absolute_projected_gradient_tolerance(None)
                .with_relative_cost_reduction_tolerance(0.),
            FirstOrderState::new(vec![0.5]),
        )
    };
    let mut expected = make().max_iter(3).run_with_solver().unwrap();
    let problem = basin::Problem::new(RoundedLinear::new(1e20, 1.));
    let first_check = basin::Solver::check_convergence(
        &mut expected.solver,
        &problem,
        &expected.state,
    );
    let repeated_check = basin::Solver::check_convergence(
        &mut expected.solver,
        &problem,
        &expected.state,
    );
    assert!(first_check.is_some());
    assert_eq!(first_check, repeated_check);
    let first = make().max_iter(1).run_with_solver().unwrap();
    assert_eq!(first.report.code(), TerminationCode::MaxIter);
    let checkpoint = first.into_checkpoint();
    #[cfg(feature = "serde")]
    let checkpoint = {
        fn round_trip<T: serde::Serialize + serde::de::DeserializeOwned>(
            value: &T,
        ) -> T {
            postcard::from_bytes(&postcard::to_allocvec(value).unwrap())
                .unwrap()
        }
        round_trip(&checkpoint)
    };
    let resumed = Executor::resume_from_checkpoint(
        RoundedLinear::new(1e20, 1.),
        checkpoint,
    )
    .max_iter(3)
    .run_with_solver()
    .unwrap();
    assert_eq!(resumed.state, expected.state);
    assert_eq!(resumed.counts, expected.counts);
    assert_eq!(
        resumed.report.code(),
        TerminationCode::RelativeCostTolerance
    );
    let fresh = Executor::new(
        RoundedLinear::new(1e20, 1.),
        resumed.solver,
        resumed.state,
    )
    .max_iter(3)
    .run()
    .unwrap();
    assert_eq!(fresh.state.iter(), 1);
    assert_eq!(fresh.counts.cost_evals, 2);
}
