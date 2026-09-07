use std::convert::Infallible;

use basin::{BrentRoot, BrentRootError, RootTerminationReason};

#[test]
fn finds_an_interior_root() {
    let result = BrentRoot::new(1.0_f64, 2.0)
        .with_tol(4.0 * f64::EPSILON, 1e-14)
        .solve(|x| Ok::<_, Infallible>(x * x * x - x - 2.0))
        .unwrap();

    assert_eq!(result.reason(), RootTerminationReason::Converged);
    assert!(result.converged());
    assert!(
        (result.root() - 1.521_379_706_804_567_6).abs() < 1e-12,
        "{result:?}"
    );
    assert!(result.value().abs() < 1e-12, "{result:?}");
    let (lower, upper) = result.bracket();
    assert!(lower <= result.root() && result.root() <= upper);
    assert!(result.iterations() > 0);
    assert_eq!(result.function_evals(), result.iterations() + 2);
}

#[test]
fn reports_endpoint_roots_without_iterating() {
    let lower = BrentRoot::new(0.0_f64, 2.0)
        .solve(|x| Ok::<_, Infallible>(x * (x - 2.0)))
        .unwrap();
    assert_eq!(lower.root(), 0.0);
    assert_eq!(lower.value(), 0.0);
    assert_eq!(lower.iterations(), 0);
    assert_eq!(lower.function_evals(), 1);

    let upper = BrentRoot::new(-1.0_f64, 0.0)
        .solve(Ok::<_, Infallible>)
        .unwrap();
    assert_eq!(upper.root(), 0.0);
    assert_eq!(upper.value(), 0.0);
    assert_eq!(upper.iterations(), 0);
    assert_eq!(upper.function_evals(), 2);
}

#[test]
fn rejects_an_invalid_interval_before_evaluation() {
    let mut calls = 0;
    let error = BrentRoot::new(2.0_f64, 1.0)
        .solve(|x| {
            calls += 1;
            Ok::<_, Infallible>(x)
        })
        .unwrap_err();

    assert!(matches!(
        error,
        BrentRootError::InvalidInterval {
            lower: 2.0,
            upper: 1.0
        }
    ));
    assert_eq!(calls, 0);
}

#[test]
fn rejects_an_interval_whose_width_overflows() {
    let error = BrentRoot::new(-f64::MAX, f64::MAX)
        .solve(Ok::<_, Infallible>)
        .unwrap_err();

    assert!(matches!(error, BrentRootError::InvalidInterval { .. }));
}

#[test]
fn rejects_same_sign_endpoints_without_multiplying_them() {
    let error = BrentRoot::new(-1.0_f64, 1.0)
        .solve(|_| Ok::<_, Infallible>(1.0e200))
        .unwrap_err();

    assert!(matches!(
        error,
        BrentRootError::NotBracketed {
            lower: -1.0,
            upper: 1.0,
            ..
        }
    ));
}

#[test]
fn rejects_non_finite_function_values() {
    let error = BrentRoot::new(-1.0_f64, 1.0)
        .solve(|x| Ok::<_, Infallible>(if x < 0.0 { f64::NAN } else { x }))
        .unwrap_err();

    assert!(matches!(
        error,
        BrentRootError::NonFiniteValue { x: -1.0, value } if value.is_nan()
    ));
}

#[derive(Debug, PartialEq, Eq)]
struct EvaluationError;

impl std::fmt::Display for EvaluationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("evaluation failed")
    }
}

impl std::error::Error for EvaluationError {}

#[test]
fn preserves_typed_evaluation_errors() {
    let error = BrentRoot::new(-1.0_f64, 1.0)
        .solve(|_| Err::<f64, _>(EvaluationError))
        .unwrap_err();

    assert!(matches!(error, BrentRootError::Evaluation(EvaluationError)));
}

#[test]
fn iteration_limit_is_a_clean_result() {
    let result = BrentRoot::new(1.0_f64, 2.0)
        .with_max_iter(1)
        .solve(|x| Ok::<_, Infallible>(x * x - 2.0))
        .unwrap();

    assert_eq!(result.reason(), RootTerminationReason::MaxIter);
    assert!(!result.converged());
    assert_eq!(result.iterations(), 1);
    assert_eq!(result.function_evals(), 3);
}

#[test]
fn zero_iteration_limit_returns_the_better_endpoint() {
    let result = BrentRoot::new(1.0_f64, 4.0)
        .with_max_iter(0)
        .solve(|x| Ok::<_, Infallible>(x * x - 2.0))
        .unwrap();

    assert_eq!(result.reason(), RootTerminationReason::MaxIter);
    assert_eq!(result.root(), 1.0);
    assert_eq!(result.value(), -1.0);
    assert_eq!(result.bracket(), (1.0, 4.0));
    assert_eq!(result.iterations(), 0);
    assert_eq!(result.function_evals(), 2);
}

#[test]
fn supports_f32() {
    let result = BrentRoot::new(0.0_f32, 2.0)
        .solve(|x| Ok::<_, Infallible>(x.cos() - x))
        .unwrap();

    assert!(result.converged());
    assert!((result.root() - 0.739_085_14).abs() < 2e-6, "{result:?}");
}

#[test]
#[should_panic(expected = "at least four times machine epsilon")]
fn rejects_an_unrepresentable_relative_tolerance() {
    let _ = BrentRoot::new(0.0_f64, 2.0).with_tol(f64::EPSILON, 1e-12);
}
