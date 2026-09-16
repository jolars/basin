use std::convert::Infallible;

use basin::{
    BracketError, BracketTerminationReason, BrentRoot, MinimumBracketer,
    RootBracketer,
};

#[test]
fn roots_in_both_directions_compose_with_brent() {
    for target in [-100.5, 100.5, 0.0, 1.0] {
        let mut calls = 0;
        let result = RootBracketer::new(0.0, 1.0)
            .bracket(|x| {
                calls += 1;
                Ok::<_, Infallible>(x - target)
            })
            .unwrap();
        assert!(result.bracketed(), "{result:?}");
        assert_eq!(result.function_evals(), calls);
        let (a, b) = result.bracket();
        assert!(a < b && a <= target && target <= b);
        assert_eq!(result.values(), (a - target, b - target));
        let root = BrentRoot::new(a, b)
            .solve(|x| Ok::<_, Infallible>(x - target))
            .unwrap();
        assert_eq!(root.root(), target);
    }
}

#[test]
fn minima_in_both_directions() {
    for target in [-100.5, 100.5, 0.0] {
        let mut calls = 0;
        let result = MinimumBracketer::new(-1.0, 0.0, 1.0)
            .bracket(|x: f64| {
                calls += 1;
                Ok::<_, Infallible>((x - target).powi(2))
            })
            .unwrap();
        assert!(result.bracketed(), "{result:?}");
        assert_eq!(result.function_evals(), calls);
        let (a, m, b) = result.bracket();
        let (fa, fm, fb) = result.values();
        assert!(a < m && m < b && a < target && target < b);
        assert!(fm <= fa && fm <= fb && (fm < fa || fm < fb));
    }
}

#[test]
fn respects_domain_limits_and_reports_boundary_minima() {
    let root = RootBracketer::new(0.2, 0.8)
        .with_max_iter(2000)
        .with_lower_bound(0.0)
        .with_upper_bound(1.0)
        .bracket(|x| {
            assert!((0.0..=1.0).contains(&x));
            Ok::<_, Infallible>(x + 1.0)
        })
        .unwrap();
    assert_eq!(root.reason(), BracketTerminationReason::BoundsReached);
    assert_eq!(root.bracket(), (0.0, 1.0));
    for sign in [-1.0, 1.0] {
        let minimum = MinimumBracketer::new(0.2, 0.5, 0.8)
            .with_max_iter(2000)
            .with_lower_bound(0.0)
            .with_upper_bound(1.0)
            .bracket(|x| {
                assert!((0.0..=1.0).contains(&x));
                Ok::<_, Infallible>(sign * x)
            })
            .unwrap();
        assert_eq!(minimum.reason(), BracketTerminationReason::BoundsReached);
        assert!(!minimum.bracketed());
    }
}

#[test]
fn finds_a_minimum_close_to_a_boundary() {
    let result = MinimumBracketer::new(0.2, 0.5, 0.8)
        .with_upper_bound(1.0)
        .bracket(|x: f64| Ok::<_, Infallible>((x - 0.999).powi(2)))
        .unwrap();
    assert!(result.bracketed(), "{result:?}");
    let (a, _, b) = result.bracket();
    assert!(a < 0.999 && 0.999 < b);
}

#[test]
fn limits_and_flat_functions_are_not_success() {
    let root = RootBracketer::new(0.0, 1.0)
        .with_max_iter(0)
        .bracket(|x| Ok::<_, Infallible>(x - 10.0))
        .unwrap();
    assert_eq!(root.reason(), BracketTerminationReason::MaxIter);
    assert_eq!(root.function_evals(), 2);
    let minimum = MinimumBracketer::new(-1.0, 0.0, 1.0)
        .with_max_iter(2)
        .bracket(|_| Ok::<_, Infallible>(1.0))
        .unwrap();
    assert_eq!(minimum.reason(), BracketTerminationReason::MaxIter);
    assert_eq!(minimum.function_evals(), 5);
    assert_eq!(minimum.iterations(), 2);
}

#[test]
fn errors_and_invalid_inputs() {
    let error = RootBracketer::new(2.0, 1.0)
        .bracket(|_| -> Result<f64, Infallible> { panic!("invalid") })
        .unwrap_err();
    assert!(matches!(error, BracketError::InvalidInitialPoints));
    let error = MinimumBracketer::new(0.0, 0.0, 1.0)
        .bracket(|_| -> Result<f64, Infallible> { panic!("invalid") })
        .unwrap_err();
    assert!(matches!(error, BracketError::InvalidInitialPoints));
    let error = RootBracketer::new(0.0, 1.0)
        .bracket(|_| Err::<f64, _>("user"))
        .unwrap_err();
    assert!(matches!(error, BracketError::Evaluation("user")));
    let error = MinimumBracketer::new(-1.0, 0.0, 1.0)
        .bracket(|_| Ok::<_, Infallible>(f64::INFINITY))
        .unwrap_err();
    assert!(matches!(error, BracketError::NonFiniteValue { .. }));
}

#[test]
fn unbounded_coordinate_overflow_stops_before_callback() {
    let result = RootBracketer::new(-1e308_f64, 0.0)
        .bracket(|x| {
            assert!(x.is_finite());
            Ok::<_, Infallible>(1.0)
        })
        .unwrap();
    assert_eq!(result.reason(), BracketTerminationReason::NoProgress);
}

#[test]
fn matches_scipy_bracketing_fixtures() {
    for line in include_str!("fixtures/scalar_brackets.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let fields: Vec<_> = line.split('\t').collect();
        let target: f64 = fields[1].parse().unwrap();
        let expected: Vec<f64> =
            fields[3..9].iter().map(|v| v.parse().unwrap()).collect();
        let (points, values, nfev, nit) = if fields[0] == "root" {
            let result = RootBracketer::new(0.0, 1.0)
                .bracket(|x| Ok::<_, Infallible>(x - target))
                .unwrap();
            assert!(result.bracketed());
            let (a, b) = result.bracket();
            let (fa, fb) = result.values();
            (
                vec![a, b],
                vec![fa, fb],
                result.function_evals(),
                result.iterations(),
            )
        } else {
            let mut bracketer = MinimumBracketer::new(-0.5, 0.0, 0.5);
            if fields[2] != "none" {
                bracketer =
                    bracketer.with_upper_bound(fields[2].parse().unwrap());
            }
            let result = bracketer
                .bracket(|x| Ok::<_, Infallible>((x - target).powi(2)))
                .unwrap();
            assert!(result.bracketed());
            let (a, m, b) = result.bracket();
            let (fa, fm, fb) = result.values();
            (
                vec![a, m, b],
                vec![fa, fm, fb],
                result.function_evals(),
                result.iterations(),
            )
        };
        let expected: Vec<_> =
            expected.into_iter().filter(|v| !v.is_nan()).collect();
        for (actual, expected) in points.into_iter().chain(values).zip(expected)
        {
            assert!(
                (actual - expected).abs()
                    <= 8.0 * f64::EPSILON * expected.abs().max(1.0),
                "{line}: {actual} != {expected}"
            );
        }
        assert_eq!(nfev, fields[9].parse::<u64>().unwrap(), "{line}");
        assert_eq!(nit, fields[10].parse::<u64>().unwrap(), "{line}");
    }
}

#[test]
fn failed_bounded_search_never_uses_out_of_domain_seed() {
    let result = RootBracketer::new(0.0, 1.0)
        .with_upper_bound(0.5)
        .bracket(|_| -> Result<f64, Infallible> { panic!("outside domain") });
    assert!(matches!(result, Err(BracketError::InvalidInitialPoints)));
    let result = MinimumBracketer::new(0.0, 0.5, 1.0)
        .with_lower_bound(0.25)
        .bracket(|_| -> Result<f64, Infallible> { panic!("outside domain") });
    assert!(matches!(result, Err(BracketError::InvalidInitialPoints)));
}

#[test]
fn bracket_at_zero_budget_is_successful_but_a_flat_triple_is_not() {
    assert!(
        RootBracketer::new(-1.0, 1.0)
            .with_max_iter(0)
            .bracket(Ok::<_, Infallible>)
            .unwrap()
            .bracketed()
    );
    let result = MinimumBracketer::new(-1.0, 0.0, 1.0)
        .with_max_iter(0)
        .bracket(|x| Ok::<_, Infallible>(x * x))
        .unwrap();
    assert!(result.bracketed());
    assert_eq!(result.function_evals(), 3);
    let result = MinimumBracketer::new(-1.0, 0.0, 1.0)
        .with_max_iter(0)
        .bracket(|_| Ok::<_, Infallible>(1.0))
        .unwrap();
    assert_eq!(result.reason(), BracketTerminationReason::MaxIter);
}
