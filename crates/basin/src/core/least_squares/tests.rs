use super::*;

struct NoCallbacks;
impl Residual for NoCallbacks {
    type Param = ();
    type Output = ();
    type Error = std::convert::Infallible;

    fn residual(&self, _: &()) -> Result<(), Self::Error> {
        unreachable!("loss tests supply their own residuals")
    }
}

#[test]
fn model_preserves_the_gradient_and_clips_negative_curvature() {
    let problem = RobustLeastSquares::from_residuals(NoCallbacks, CauchyLoss);
    let raw: Vec<f64> = vec![0.0, 0.5, 2.0];
    let (model, scale) = problem.model_residual(&raw).unwrap();
    for (i, &r) in raw.iter().enumerate() {
        let expected_gradient = r / (1.0 + r * r);
        let expected_curvature =
            ((1.0 - r * r) / (1.0 + r * r).powi(2)).max(f64::EPSILON);
        assert!((model[i] * scale[i] - expected_gradient).abs() < 1e-15);
        assert!((scale[i] * scale[i] - expected_curvature).abs() < 1e-15);
    }
    assert_eq!(scale[2], f64::EPSILON.sqrt());
    let true_cost = problem.residual_cost(&raw);
    let surrogate = model.iter().map(|r| r * r / 2.0).sum::<f64>();
    assert!(surrogate > 1e10 * true_cost);
}

#[test]
fn positive_model_curvature_matches_an_affine_objective_hessian() {
    let objective = RobustLeastSquares::from_residuals(NoCallbacks, SoftL1Loss)
        .with_scale(0.7);
    let residuals = |x: f64| vec![2.0 * x + 0.1, -x + 0.2];
    let r = residuals(0.3);
    let (_, scale) = objective.model_residual(&r).unwrap();
    let curvature = 4.0 * scale[0] * scale[0] + scale[1] * scale[1];
    let h = 1e-4;
    let difference = (objective.residual_cost(&residuals(0.3 + h))
        - 2.0 * objective.residual_cost(&r)
        + objective.residual_cost(&residuals(0.3 - h)))
        / (h * h);
    assert!((difference - curvature).abs() < 1e-6);
}

fn scalar_limits<F: Scalar>() {
    let zero = F::zero();
    let one = F::one();
    for loss in [
        &SquaredLoss as &dyn LossFunction<F>,
        &HuberLoss,
        &SoftL1Loss,
        &CauchyLoss,
        &ArctanLoss,
    ] {
        for z in [
            zero,
            F::epsilon() * F::epsilon(),
            one,
            F::max_value() / F::from_f64(4.0).unwrap(),
        ] {
            let result = loss.evaluate(z);
            assert!(result.value.is_finite() && result.value >= zero);
            assert!(
                result.first_derivative.is_finite()
                    && result.first_derivative >= zero
            );
            assert!(result.second_derivative.is_finite());
        }
        let adapter = RobustLeastSquares::from_residuals(NoCallbacks, loss);
        for r in [F::nan(), F::infinity(), F::neg_infinity()] {
            assert!(adapter.model_residual(&vec![r]).is_none());
            assert_eq!(adapter.residual_cost(&vec![r]), F::infinity());
        }
        assert_eq!(adapter.residual_cost(&vec![zero]), zero);
    }
    let huber: LossEvaluation<F> = HuberLoss.evaluate(one);
    assert_eq!(huber.value, one);
    assert_eq!(huber.first_derivative, one);
    assert_eq!(huber.second_derivative, zero);
}

#[test]
fn finite_and_non_finite_limits_in_both_scalar_types() {
    scalar_limits::<f64>();
    scalar_limits::<f32>();
}

fn scaled_derivatives<F: Scalar>() {
    let half = F::from_f64(0.5).unwrap();
    let two = F::from_f64(2.0).unwrap();
    let tolerance = F::from_f64(32.0).unwrap() * F::epsilon();
    for loss in [
        &SquaredLoss as &dyn LossFunction<F>,
        &HuberLoss,
        &SoftL1Loss,
        &CauchyLoss,
        &ArctanLoss,
    ] {
        for scale in [half, F::one(), two] {
            for residual in [-3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0] {
                let r = F::from_f64(residual).unwrap();
                let z = (r / scale) * (r / scale);
                let normalized = loss.evaluate(z);
                let scaled = loss.evaluate_scaled(r, scale);
                for (actual, expected) in [
                    (scaled.value, half * scale * scale * normalized.value),
                    (scaled.first_derivative, r * normalized.first_derivative),
                    (
                        scaled.second_derivative,
                        normalized.first_derivative
                            + two * z * normalized.second_derivative,
                    ),
                ] {
                    assert!(
                        (actual - expected).abs()
                            <= tolerance * F::one().max(expected.abs()),
                        "r={r:?}, scale={scale:?}: {actual:?} != {expected:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn scaled_losses_match_the_chain_rule() {
    scaled_derivatives::<f32>();
    scaled_derivatives::<f64>();
}

fn extreme_scales<F: Scalar>(large: F, small: F) {
    let zero = F::zero();
    let one = F::one();
    let half = F::from_f64(0.5).unwrap();
    let tolerance = F::from_f64(8.0).unwrap() * F::epsilon();
    for loss in [
        &SquaredLoss as &dyn LossFunction<F>,
        &HuberLoss,
        &SoftL1Loss,
        &CauchyLoss,
        &ArctanLoss,
    ] {
        for scale in [large, F::max_value()] {
            let adapter = RobustLeastSquares::from_residuals(NoCallbacks, loss)
                .with_scale(scale);
            for r in [-one, zero, one] {
                let term = adapter.term(r).expect("representable scaled loss");
                assert!((term.cost - half * r * r).abs() <= tolerance);
                assert!((term.derivative - r).abs() <= tolerance);
                assert!((term.curvature - one).abs() <= tolerance);
            }
        }
    }
    for scale in [
        small,
        F::min_positive_value(),
        F::min_positive_value() * F::epsilon(),
    ] {
        let adapter =
            RobustLeastSquares::from_residuals(NoCallbacks, &SquaredLoss)
                .with_scale(scale);
        for r in [-one, zero, one] {
            let term =
                adapter.term(r).expect("scale cancels out of squared loss");
            assert_eq!(term.cost, half * r * r);
            assert_eq!(term.derivative, r);
            assert_eq!(term.curvature, one);
        }
        for loss in [&HuberLoss as &dyn LossFunction<F>, &SoftL1Loss] {
            let adapter = RobustLeastSquares::from_residuals(NoCallbacks, loss)
                .with_scale(scale);
            for r in [-one, one] {
                let term = adapter.term(r).expect("representable linear tail");
                assert!((term.cost / scale - one).abs() <= tolerance);
                assert!((term.derivative / scale - r).abs() <= tolerance);
            }
        }
    }
}

#[test]
fn extreme_residual_scales_preserve_representable_terms() {
    extreme_scales::<f32>(1e23, 1e-23);
    extreme_scales::<f64>(1e200, 1e-200);
}

fn large_residuals<F: Scalar>() {
    let r = F::max_value() / F::from_f64(4.0).unwrap();
    let one = F::one();
    let tolerance = F::from_f64(8.0).unwrap() * F::epsilon();
    for loss in [&HuberLoss as &dyn LossFunction<F>, &SoftL1Loss] {
        let adapter = RobustLeastSquares::from_residuals(NoCallbacks, loss);
        let term = adapter.term(r).expect("representable linear tail");
        assert!((term.cost / r - one).abs() <= tolerance);
        assert!((term.derivative - one).abs() <= tolerance);
    }
    let cauchy = RobustLeastSquares::from_residuals(NoCallbacks, CauchyLoss)
        .term(r)
        .unwrap();
    assert!((cauchy.cost / r.ln() - one).abs() <= tolerance);
    assert!((cauchy.derivative * r - one).abs() <= tolerance);
    let arctan = RobustLeastSquares::from_residuals(NoCallbacks, ArctanLoss)
        .term(r)
        .unwrap();
    let expected = F::from_f64(std::f64::consts::FRAC_PI_4).unwrap();
    assert!((arctan.cost - expected).abs() <= tolerance);
}

#[test]
fn overflowing_normalized_squares_preserve_robust_tails() {
    large_residuals::<f32>();
    large_residuals::<f64>();
}

struct Invalid;
impl LossFunction for Invalid {
    fn evaluate(&self, _: f64) -> LossEvaluation {
        LossEvaluation {
            value: 1.0,
            first_derivative: 1.0,
            second_derivative: f64::NAN,
        }
    }
}

#[test]
fn invalid_curvature_is_not_masked_by_the_floor() {
    let adapter = RobustLeastSquares::from_residuals(NoCallbacks, Invalid);
    assert!(adapter.model_residual(&vec![0.5]).is_none());
    assert_eq!(adapter.residual_cost(&vec![0.5]), f64::INFINITY);
}

#[test]
fn scale_must_be_finite_and_positive() {
    for scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            std::panic::catch_unwind(|| RobustLeastSquares::from_residuals(
                NoCallbacks,
                HuberLoss
            )
            .with_scale(scale))
            .is_err()
        );
    }
}

#[test]
fn builtin_reductions_match_wider_cost_differences() {
    fn check(
        loss: impl LossFunction<f32> + LossFunction<f64>,
        rho: impl Fn(f64) -> f64,
    ) {
        for (a, b, scale) in [
            (0.5_f32, 0.49999997, 1.0),
            (1.0000001, 0.99999994, 1.0),
            (2.0, 1.9999999, 1.0),
            (300_000.0, 299_999.97, 1000.0),
            (300_000.0, 1.0, 1000.0),
        ] {
            for (before, after) in [(a, -b), (b, -a)] {
                let cost = |r| {
                    <_ as LossFunction<f32>>::evaluate_scaled(&loss, r, scale)
                        .value
                };
                let reduction = loss.reduction_scaled(
                    before,
                    after,
                    scale,
                    cost(before),
                    cost(after),
                );
                let scale = f64::from(scale);
                let reference = |r: f32| {
                    0.5 * scale * scale * rho((f64::from(r) / scale).powi(2))
                };
                let expected = reference(before) - reference(after);
                assert!(
                    (f64::from(reduction) / expected - 1.0).abs() < 0.002,
                    "reduction {reduction}, expected {expected}, residuals {before}, {after}"
                );
            }
        }
    }
    check(SquaredLoss, |z| z);
    check(
        HuberLoss,
        |z| if z <= 1.0 { z } else { 2.0 * z.sqrt() - 1.0 },
    );
    check(SoftL1Loss, |z| 2.0 * ((1.0 + z).sqrt() - 1.0));
    check(CauchyLoss, f64::ln_1p);
    check(ArctanLoss, f64::atan);
}

#[test]
fn arctan_reduction_stays_finite_near_the_float_limit() {
    let a = 2.8230258e19_f32;
    let b = 1.4115128e15_f32;
    let scale = 3.230007e19_f32;
    let cost = |r| ArctanLoss.evaluate_scaled(r, scale).value;
    let cost_a = cost(a);
    let cost_b = cost(b);
    assert!(cost_a.is_finite() && cost_b.is_finite());

    let reference_cost = |r| {
        let scale = f64::from(scale);
        (0.5 * scale * scale) * (f64::from(r) / scale).powi(2).atan()
    };
    let expected = reference_cost(a) - reference_cost(b);
    assert!(expected < f64::from(f32::MAX));
    let reduction = ArctanLoss.reduction_scaled(a, b, scale, cost_a, cost_b);
    assert!(
        (f64::from(reduction) / expected - 1.0).abs()
            <= f64::from(f32::EPSILON),
        "reduction {reduction}, expected {expected}"
    );
    assert_eq!(
        ArctanLoss.reduction_scaled(b, a, scale, cost_b, cost_a),
        -reduction
    );
}
