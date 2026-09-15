use super::*;
use crate::{CountsMirror, FirstOrderState, PointState};

fn point(cost: f64) -> PointState<f64> {
    let mut state = PointState::new(cost);
    state.replace(cost, cost);
    state.update_best();
    state
}

#[test]
fn raw_categories_read_the_state_without_legacy_folds() {
    let mut state = point(1.0);
    state.mirror(&EvalCounts {
        cost_evals: 2,
        gradient_evals: 3,
        residual_evals: 5,
        jacobian_evals: 7,
        hessian_evals: 11,
        hessian_product_evals: 13,
    });
    for (kind, count) in EvaluationKind::ALL
        .into_iter()
        .zip([2, 3, 5, 7, 11, 13, 41])
    {
        let mut control = RunControl::new().max_evaluations(kind, count + 1);
        assert_eq!(control.check(&state, &EvalCounts::default()), None);
        control = control.max_evaluations(kind, count);
        assert_eq!(
            control.check(&state, &EvalCounts::default()),
            Some(TerminationReason::MaxEvaluations)
        );
    }
}

#[test]
fn raw_setters_replace_and_preserve_budget_precedence() {
    let mut state = FirstOrderState::new(vec![1.0]);
    state.replace(vec![1.0], 1.0, vec![2.0]).unwrap();
    let counts = EvalCounts {
        cost_evals: 2,
        gradient_evals: 3,
        ..EvalCounts::default()
    };
    state.mirror(&counts);
    state.update_best();
    let mut control = RunControl::new()
        .max_evaluations(EvaluationKind::Cost, 1)
        .max_evaluations(EvaluationKind::Cost, 3);
    assert_eq!(control.check(&state, &counts), None);
    control = control
        .max_evaluations(EvaluationKind::Gradient, 2)
        .max_time(Duration::ZERO)
        .target_objective(1.0);
    assert_eq!(
        control.check(&state, &counts),
        Some(TerminationReason::MaxEvaluations)
    );
    control = control.max_evaluations(EvaluationKind::Cost, 2);
    assert_eq!(
        control.check(&state, &counts),
        Some(TerminationReason::MaxEvaluations)
    );
    control = control.max_gradient_evals(1);
    assert_eq!(
        control.check(&state, &counts),
        Some(TerminationReason::MaxGradientEvals)
    );
    control = control.max_cost_evals(1);
    assert_eq!(
        control.check(&state, &counts),
        Some(TerminationReason::MaxCostEvals)
    );
    control = control.max_iter(0);
    assert_eq!(
        control.check(&state, &counts),
        Some(TerminationReason::MaxIter)
    );
}

#[test]
fn new_and_legacy_target_and_stall_setters_share_slots() {
    let mut state = point(1.0);
    let counts = EvalCounts::default();
    let mut control = RunControl::new().target_objective(1.0).target_cost(0.0);
    assert_eq!(control.check(&state, &counts), None);
    control = control.target_objective(1.0);
    assert_eq!(
        control.check(&state, &counts),
        Some(TerminationReason::TargetCost)
    );
    let mut control = RunControl::new()
        .no_objective_improvement(1, 0.0)
        .no_improvement(3, 0.0);
    state.increment_iter();
    assert_eq!(control.check(&state, &counts), None);
    control = control.no_objective_improvement(1, 0.0);
    assert_eq!(
        control.check(&state, &counts),
        Some(TerminationReason::NoImprovement)
    );
}

#[test]
fn repeated_checks_and_mid_step_publications_do_not_invent_stall_age() {
    for delta in [0.0, 1.0] {
        let mut state = point(10.0);
        let counts = EvalCounts::default();
        let mut control = RunControl::new().no_objective_improvement(2, delta);
        for _ in 0..4 {
            assert_eq!(control.check(&state, &counts), None);
        }
        state.increment_iter();
        state.replace(8.0, 8.0);
        state.update_best();
        assert_eq!(control.check(&state, &counts), None);
        state.increment_iter();
        for _ in 0..4 {
            assert_eq!(control.check(&state, &counts), None);
        }
        // A clean mid-step improvement retains the iteration number but
        // still resets a positive-delta anchor before further checks.
        state.replace(6.0, 6.0);
        state.update_best();
        assert_eq!(control.check(&state, &counts), None);
        state.increment_iter();
        assert_eq!(control.check(&state, &counts), None);
        state.increment_iter();
        assert_eq!(
            control.check(&state, &counts),
            Some(TerminationReason::NoImprovement)
        );
    }
}

#[test]
fn objective_anchor_handles_large_decreases_and_negative_infinity() {
    for initial in [f64::MAX, -f64::MAX] {
        let mut state = point(initial);
        let counts = EvalCounts::default();
        let mut control =
            RunControl::new().no_objective_improvement(2, f64::MAX);
        assert_eq!(control.check(&state, &counts), None);
        state.increment_iter();
        let improved = if initial > 0.0 {
            -f64::MAX
        } else {
            f64::NEG_INFINITY
        };
        state.replace(improved, improved);
        state.update_best();
        assert_eq!(control.check(&state, &counts), None);
        state.increment_iter();
        assert_eq!(control.check(&state, &counts), None);
        state.increment_iter();
        assert_eq!(
            control.check(&state, &counts),
            Some(TerminationReason::NoImprovement)
        );
    }
}

#[test]
fn invalid_objective_control_settings_are_rejected() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            std::panic::catch_unwind(
                || RunControl::<PointState<f64>>::new().target_objective(value)
            )
            .is_err()
        );
    }
    for value in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            std::panic::catch_unwind(|| RunControl::<PointState<f64>>::new()
                .no_objective_improvement(1, value))
            .is_err()
        );
    }
    assert!(
        std::panic::catch_unwind(|| RunControl::<PointState<f64>>::new()
            .no_objective_improvement(0, 0.0))
        .is_err()
    );
}
