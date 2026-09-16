use std::cell::Cell;
use std::convert::Infallible;

use basin::{
    HalleyRoot, NewtonRoot, RootError, RootResult, RootTerminationReason,
    SecantRoot, Toms748Root,
};

fn check(result: RootResult, f: impl Fn(f64) -> f64, expected: f64) {
    assert!(result.converged(), "{result:?}");
    assert!((result.root() - expected).abs() <= 2e-12, "{result:?}");
    assert_eq!(result.value(), f(result.root()));
    let (a, b) = result.bracket();
    assert!(a <= result.root() && result.root() <= b);
    assert!(
        f(a) == 0.0
            || f(b) == 0.0
            || f(a).is_sign_positive() != f(b).is_sign_positive()
    );
    assert!(
        result.value() == 0.0
            || b - a <= 1e-12 + 4.0 * f64::EPSILON * result.root().abs()
    );
}

macro_rules! root_contract {
    ($module:ident, $solver:ident, $($derivative:expr),*) => {
        mod $module {
            use super::*;

            #[test]
            fn solves_and_counts_evaluations() {
                let calls = Cell::new(0);
                let f = |x: f64| x * x * x - x - 2.0;
                let result = $solver::new(1.0, 2.0).solve(|x| {
                    calls.set(calls.get() + 1);
                    Ok::<_, Infallible>(f(x))
                } $(, $derivative)*).unwrap();
                check(result, f, 1.521_379_706_804_567_6);
                assert_eq!(result.function_evals(), calls.get());
                assert_eq!(result.callback_evals(), result.function_evals() + result.derivative_evals() + result.second_derivative_evals());
            }

            #[test]
            fn endpoints_and_zero_budget() {
                let result = $solver::new(1.0, 2.0).with_max_iter(0)
                    .solve(|x| Ok::<_, Infallible>(x - 1.0) $(, $derivative)*).unwrap();
                assert!(result.converged());
                assert_eq!(result.function_evals(), 1);
                assert_eq!(result.iterations(), 0);
                assert_eq!(result.derivative_evals(), 0);
                let result = $solver::new(1.0, 2.0).with_max_iter(0)
                    .solve(|x| Ok::<_, Infallible>(x - 2.0) $(, $derivative)*).unwrap();
                assert!(result.converged());
                assert_eq!(result.function_evals(), 2);
                let result = $solver::new(1.0, 2.0).with_max_iter(0)
                    .solve(|x| Ok::<_, Infallible>(x * x - 2.0) $(, $derivative)*).unwrap();
                assert_eq!(result.reason(), RootTerminationReason::MaxIter);
                assert_eq!(result.function_evals(), 2);
                assert_eq!(result.derivative_evals(), 0);
                assert_eq!(result.root(), 1.0);
            }

            #[test]
            fn rejects_invalid_inputs_before_evaluation() {
                for (a,b) in [(2.0, 1.0), (1.0, 1.0), (f64::NAN, 1.0), (-f64::MAX, f64::MAX)] {
                    let error = $solver::new(a, b).solve(|_| -> Result<f64, Infallible> {
                        panic!("invalid intervals must not be evaluated")
                    } $(, $derivative)*).unwrap_err();
                    assert!(matches!(error, RootError::InvalidInterval { .. }));
                }
                let error = $solver::new(1.0, 2.0).solve(|_| Ok::<_, Infallible>(1e300) $(, $derivative)*).unwrap_err();
                assert!(matches!(error, RootError::NotBracketed { .. }));
                let error = $solver::new(1.0, 2.0).solve(|_| Ok::<_, Infallible>(f64::NAN) $(, $derivative)*).unwrap_err();
                assert!(matches!(error, RootError::NonFiniteValue { .. }));
            }

            #[test]
            fn adjacent_endpoints_terminate_without_repeated_evaluations() {
                let a = 1.0_f64;
                let b = f64::from_bits(a.to_bits() + 1);
                let result = $solver::new(a, b).solve(|x| Ok::<_, Infallible>(if x == a { -1.0 } else { 1.0 }) $(, $derivative)*).unwrap();
                assert!(result.converged());
                assert_eq!(result.function_evals(), 2);
            }
        }
    };
}

root_contract!(secant, SecantRoot,);
root_contract!(toms748, Toms748Root,);
root_contract!(newton, NewtonRoot, |x: f64| Ok::<_, Infallible>(
    3.0 * x * x - 1.0
));
root_contract!(
    halley,
    HalleyRoot,
    |x: f64| Ok::<_, Infallible>(3.0 * x * x - 1.0),
    |x: f64| Ok::<_, Infallible>(6.0 * x)
);

#[test]
fn combined_callbacks_share_work_and_match_separate_callbacks() {
    let f = |x: f64| Ok::<_, Infallible>(x.cos() - x);
    let d = |x: f64| Ok::<_, Infallible>(-x.sin() - 1.0);
    let dd = |x: f64| Ok::<_, Infallible>(-x.cos());
    let calls = Cell::new(0);
    let newton = NewtonRoot::new(0.0, 1.0);
    let separate = newton.solve(f, d).unwrap();
    let combined = newton
        .solve_combined(|x| {
            calls.set(calls.get() + 1);
            Ok::<_, Infallible>((f(x)?, d(x)?))
        })
        .unwrap();
    check(combined, |x| x.cos() - x, 0.739_085_133_215_160_7);
    assert_eq!(combined.root(), separate.root());
    assert_eq!(combined.function_evals(), separate.function_evals());
    assert_eq!(combined.callback_evals(), calls.get());
    assert_eq!(combined.derivative_evals(), calls.get());
    assert_eq!(combined.second_derivative_evals(), 0);

    calls.set(0);
    let halley = HalleyRoot::new(0.0, 1.0);
    let separate = halley.solve(f, d, dd).unwrap();
    let combined = halley
        .solve_combined(|x| {
            calls.set(calls.get() + 1);
            Ok::<_, Infallible>((f(x)?, d(x)?, dd(x)?))
        })
        .unwrap();
    check(combined, |x| x.cos() - x, 0.739_085_133_215_160_7);
    assert_eq!(combined.root(), separate.root());
    assert_eq!(combined.function_evals(), separate.function_evals());
    assert_eq!(combined.callback_evals(), calls.get());
    assert_eq!(combined.derivative_evals(), calls.get());
    assert_eq!(combined.second_derivative_evals(), calls.get());
}

#[test]
fn unusable_derivatives_fall_back_without_leaving_bracket() {
    for d in [0.0, f64::INFINITY, f64::NAN, 1e300, -1e-300] {
        let f = |x: f64| {
            assert!((-2.0..=2.0).contains(&x));
            Ok::<_, Infallible>(x * x * x - 1.0)
        };
        let result = NewtonRoot::new(-2.0, 2.0).solve(f, |_| Ok(d)).unwrap();
        check(result, |x| x * x * x - 1.0, 1.0);
        let result = HalleyRoot::new(-2.0, 2.0)
            .solve(f, |_| Ok(d), |_| Ok(d))
            .unwrap();
        check(result, |x| x * x * x - 1.0, 1.0);
    }
}

#[test]
fn validates_initial_guesses_before_callbacks() {
    for x in [-1.0, 0.0, 2.0, 3.0, f64::NAN] {
        let error = NewtonRoot::new(0.0, 2.0)
            .with_initial_guess(x)
            .solve_combined(|_| -> Result<(f64, f64), Infallible> {
                panic!("invalid guess")
            })
            .unwrap_err();
        assert!(matches!(error, RootError::InvalidInitialGuess { .. }));
    }
}

#[test]
fn preserves_errors_from_every_callback() {
    let error = SecantRoot::new(0.0, 2.0)
        .solve(|_| Err::<f64, _>("value"))
        .unwrap_err();
    assert!(matches!(error, RootError::Evaluation("value")));
    let error = NewtonRoot::new(0.0, 2.0)
        .solve(|x| Ok(x - 1.5), |_| Err::<f64, _>("derivative"))
        .unwrap_err();
    assert!(matches!(error, RootError::Evaluation("derivative")));
    let error = HalleyRoot::new(0.0, 2.0)
        .solve(
            |x| Ok(x - 1.5),
            |_| Ok(1.0),
            |_| Err::<f64, _>("second derivative"),
        )
        .unwrap_err();
    assert!(matches!(error, RootError::Evaluation("second derivative")));
    let error = HalleyRoot::new(0.0, 2.0)
        .solve_combined(|_| Err::<(f64, f64, f64), _>("combined"))
        .unwrap_err();
    assert!(matches!(error, RootError::Evaluation("combined")));
}

#[test]
fn scaling_function_values_does_not_break_interpolation() {
    for scale in [1e-300, 1.0, 1e300] {
        let f = |x| Ok::<_, Infallible>(scale * (x * x - 2.0));
        for result in [
            SecantRoot::new(0.0, 2.0).solve(f).unwrap(),
            Toms748Root::new(0.0, 2.0).solve(f).unwrap(),
        ] {
            check(result, |x| scale * (x * x - 2.0), 2.0_f64.sqrt());
        }
    }
}

#[test]
fn toms748_contracts_interval_each_cycle() {
    let mut previous = 2.0;
    for budget in 1..15 {
        let result = Toms748Root::new(0.0, 2.0)
            .with_max_iter(budget)
            .solve(|x: f64| Ok::<_, Infallible>(x.powi(9) - 1.1))
            .unwrap();
        let (a, b) = result.bracket();
        if budget > 1 {
            assert!(b - a <= 0.5 * previous + 1e-15, "{result:?}");
        }
        if result.converged() {
            break;
        }
        assert_eq!(result.iterations(), budget);
        previous = b - a;
    }
}

#[test]
fn toms748_matches_scipy_reference_roots_and_selected_trajectories() {
    for line in include_str!("fixtures/scalar_roots.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let fields: Vec<_> = line.split('\t').collect();
        let name = fields[0];
        let a: f64 = fields[1].parse().unwrap();
        let b: f64 = fields[2].parse().unwrap();
        let expected: f64 = fields[3].parse().unwrap();
        let f = |x: f64| match name {
            "cubic" => x * x * x - x - 2.0,
            "cosine" => x.cos() - x,
            "exponential" => x.exp() - 7.0,
            "flat" => (x - 0.7).powi(3),
            "steep" => x.powi(9) - 1.1,
            _ => panic!("unknown fixture {name}"),
        };
        let mut trace = Vec::new();
        let result = Toms748Root::new(a, b)
            .solve(|x| {
                trace.push(x);
                Ok::<_, Infallible>(f(x))
            })
            .unwrap();
        check(result, f, expected);
        assert_eq!(result.function_evals() as usize, trace.len());
        // These smooth cases share the paper's initial interpolation sequence.
        // Other fixtures exercise different floating-point safeguards.
        if matches!(name, "cubic" | "cosine") {
            let reference: Vec<f64> =
                fields[7].split(',').map(|v| v.parse().unwrap()).collect();
            assert!(trace.len() >= reference.len());
            for (actual, expected) in trace.iter().zip(reference) {
                assert!(
                    (actual - expected).abs() <= 2e-14,
                    "{name}: {actual} != {expected}"
                );
            }
        }
        let (mut lower, mut upper) = (a, b);
        for &x in &trace[2..] {
            assert!(
                lower < x && x < upper,
                "{name}: trial {x} outside [{lower}, {upper}]"
            );
            if f(x) == 0.0 {
                break;
            }
            if f(x).is_sign_positive() == f(lower).is_sign_positive() {
                lower = x;
            } else {
                upper = x;
            }
        }
    }
}

#[test]
fn halley_reversing_correction_uses_newton_step() {
    let mut points = Vec::new();
    let result = HalleyRoot::new(0.0, 2.0)
        .with_initial_guess(1.0)
        .with_max_iter(2)
        .solve(
            |x| {
                points.push(x);
                Ok::<_, Infallible>(x * x - 2.0)
            },
            |x| Ok(2.0 * x),
            |_| Ok(-100.0),
        )
        .unwrap();
    assert_eq!(points, vec![0.0, 2.0, 1.0, 1.5]);
    assert_eq!(result.iterations(), 2);
    assert_eq!(result.reason(), RootTerminationReason::MaxIter);
}

#[test]
fn small_steps_do_not_establish_root_convergence() {
    let f = |x: f64| Ok::<_, Infallible>(x - 0.3);
    let result = NewtonRoot::new(0.0, 1.0)
        .with_max_iter(2)
        .solve(f, |_| Ok(1e15))
        .unwrap();
    assert_eq!(result.reason(), RootTerminationReason::MaxIter);
    assert!(result.value().abs() > 0.1);
    let result = HalleyRoot::new(0.0, 1.0)
        .with_max_iter(2)
        .solve(f, |_| Ok(1e15), |_| Ok(0.0))
        .unwrap();
    assert_eq!(result.reason(), RootTerminationReason::MaxIter);
}

#[test]
fn non_finite_interior_values_and_typed_toms_errors_are_hard_errors() {
    let f = |x| {
        Ok::<_, Infallible>(if x == 0.0 {
            -1.0
        } else if x == 1.0 {
            1.0
        } else {
            f64::NAN
        })
    };
    assert!(matches!(
        Toms748Root::new(0.0, 1.0).solve(f),
        Err(RootError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        NewtonRoot::new(0.0, 1.0).solve(f, |_| Ok(1.0)),
        Err(RootError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        Toms748Root::new(0.0, 1.0).solve(|_| Err::<f64, _>("user")),
        Err(RootError::Evaluation("user"))
    ));
}

#[test]
fn extreme_coordinates_and_subnormal_intervals() {
    for (a, b, target) in [
        (1e300, 1.5e300, 1.25e300),
        (-1e-300, 1e-300, 3e-301),
        (0.0, 1e-320, 3e-321),
    ] {
        let scale = b - a;
        let f = |x| Ok::<_, Infallible>((x - target) / scale);
        for result in [
            SecantRoot::new(a, b)
                .with_absolute_position_tolerance(f64::from_bits(1))
                .solve(f)
                .unwrap(),
            Toms748Root::new(a, b)
                .with_absolute_position_tolerance(f64::from_bits(1))
                .solve(f)
                .unwrap(),
            NewtonRoot::new(a, b)
                .with_absolute_position_tolerance(f64::from_bits(1))
                .solve(f, |_| Ok(1.0 / scale))
                .unwrap(),
            HalleyRoot::new(a, b)
                .with_absolute_position_tolerance(f64::from_bits(1))
                .solve(f, |_| Ok(1.0 / scale), |_| Ok(0.0))
                .unwrap(),
        ] {
            assert!(result.converged(), "{result:?}");
            assert!(result.root().is_finite());
            assert_eq!(result.value(), f(result.root()).unwrap());
            assert!(
                (result.root() - target).abs()
                    <= (8.0 * f64::EPSILON * target.abs())
                        .max(f64::from_bits(1))
            );
        }
    }
}
