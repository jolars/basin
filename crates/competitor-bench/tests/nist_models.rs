//! Model reproduction checks, independent of solver stopping policies.

use basin::{CostFunction, Gradient, Jacobian, Residual, Scalar};
use competitor_bench::convergence::nist::{Nist, PrintedValue, datasets};
#[cfg(not(feature = "basin-latest"))]
use nalgebra::DVector;
#[cfg(feature = "basin-latest")]
use nalgebra_latest::DVector;

#[test]
fn snapshots_have_all_starts_observations_and_printed_intervals() {
    let cases = datasets().unwrap();
    assert_eq!(cases.len(), 27);
    assert_eq!(
        cases.iter().map(|d| d.observations.len()).sum::<usize>(),
        2176
    );
    for d in cases {
        for start in &d.starts {
            assert_eq!(start.len(), d.reference.len());
        }
        assert!(d.rss.half_width > 0.0);
        assert!(matches!(d.partition, "development" | "validation"));
    }
    let v = PrintedValue::parse("-2.5235058043E+03").unwrap();
    assert_eq!(v.midpoint, -2523.5058043);
    assert!((v.half_width - 5e-8).abs() < 1e-22);
    assert_eq!(PrintedValue::parse("0.00001").unwrap().half_width, 5e-6);
    assert!(PrintedValue::parse("NaN").is_err());
}

fn check_derivatives<F: Scalar>(step: f64, tolerance: f64) {
    for dataset in datasets().unwrap() {
        let model = Nist::<F>::new(dataset);
        for point in [
            model.dataset.start::<F>(0),
            model.dataset.start::<F>(1),
            model.dataset.reference_point::<F>(),
        ] {
            for observation in &model.dataset.observations {
                let x: Vec<F> = observation[1..]
                    .iter()
                    .map(|v| F::from_f64(v.midpoint).unwrap())
                    .collect();
                let (y, g) = model.response(&point, &x).unwrap();
                assert!(
                    y.is_finite() && g.iter().all(|v| v.is_finite()),
                    "{}",
                    model.dataset.id
                );
                for j in 0..point.len() {
                    let scale = point[j].abs().max(F::from_f64(1e-12).unwrap());
                    // Multiple stencil sizes distinguish cancellation from truncation error.
                    let mut error = f64::INFINITY;
                    for factor in [0.01, 0.1, 1.0, 10.0] {
                        let h = F::from_f64(step * factor).unwrap() * scale;
                        let mut plus = point.clone();
                        let mut minus = point.clone();
                        plus[j] = plus[j] + h;
                        minus[j] = minus[j] - h;
                        if plus[j] == minus[j] {
                            continue;
                        }
                        let yp = model.response(&plus, &x).unwrap().0;
                        let ym = model.response(&minus, &x).unwrap().0;
                        let numerical = (yp - ym) / (plus[j] - minus[j]);
                        error = error.min(
                            ((numerical - g[j]) * scale)
                                .abs()
                                .to_f64()
                                .unwrap(),
                        );
                    }
                    let allowance = tolerance
                        * (y.abs()
                            .to_f64()
                            .unwrap()
                            .max((g[j] * scale).abs().to_f64().unwrap())
                            .max(1e-12));
                    assert!(
                        error <= allowance,
                        "{} column {j}: error {error:e}, allowance {allowance:e}",
                        model.dataset.id
                    );
                }
            }
        }
    }
}

#[test]
fn analytic_jacobians_f64_every_row_and_both_starts() {
    check_derivatives::<f64>(1e-6, 2e-7);
}
#[test]
fn analytic_jacobians_native_f32_every_row_and_both_starts() {
    check_derivatives::<f32>(1e-3, 2e-3);
}

fn check_traits<F: Scalar>(tolerance: f64) {
    for d in datasets().unwrap() {
        let point = DVector::from_vec(d.start::<F>(1));
        let model = Nist::<F>::new(d);
        let r = model.residual(&point).unwrap();
        let j = model.jacobian(&point).unwrap();
        let g = model.gradient(&point).unwrap();
        assert_eq!(j.nrows(), r.len());
        assert_eq!(j.ncols(), point.len());
        let cost = model.cost(&point).unwrap();
        assert_eq!(cost + cost, model.rss(point.as_slice()).unwrap());
        for col in 0..point.len() {
            let expected: F =
                (0..r.len()).map(|row| j[(row, col)] * r[row]).sum();
            assert!(
                (expected - g[col]).abs().to_f64().unwrap()
                    <= tolerance * expected.abs().to_f64().unwrap().max(1.0)
            );
        }
        assert!(model.residuals(&[]).is_err());
        assert!(model.response(point.as_slice(), &[]).is_err());
    }
}
#[test]
fn adapters_agree_f64() {
    check_traits::<f64>(1e-12);
}
#[test]
fn adapters_agree_f32() {
    check_traits::<f32>(1e-5);
}

#[test]
fn nelson_uses_log_response_and_roszman_uses_principal_atan() {
    let cases = datasets().unwrap();
    let nelson = Nist::<f64>::new(
        cases.iter().find(|d| d.id == "Nelson").unwrap().clone(),
    );
    let b = nelson.dataset.start::<f64>(0);
    let row = &nelson.dataset.observations[0];
    let predicted =
        b[0] - b[1] * row[1].midpoint * (-b[2] * row[2].midpoint).exp();
    assert_eq!(
        nelson.residuals(&b).unwrap()[0],
        predicted - row[0].midpoint.ln()
    );
    let roszman = Nist::<f64>::new(
        cases.iter().find(|d| d.id == "Roszman1").unwrap().clone(),
    );
    let (y, _) = roszman.response(&[0.0, 0.0, 1.0, 0.0], &[-1.0]).unwrap();
    assert!((y - 0.25).abs() < 1e-15);
    let (y, _) = roszman.response(&[0.0, 0.0, 1.0, 0.0], &[1.0]).unwrap();
    assert!((y + 0.25).abs() < 1e-15);
}

#[test]
fn invalid_model_domains_are_visible_without_clamping() {
    let cases = datasets().unwrap();
    let model = Nist::<f64>::new(
        cases.iter().find(|d| d.id == "Bennett5").unwrap().clone(),
    );
    assert!(
        model
            .response(&[1.0, -2.0, 0.8], &[1.0])
            .unwrap()
            .0
            .is_nan()
    );
    let model = Nist::<f64>::new(
        cases.iter().find(|d| d.id == "Chwirut1").unwrap().clone(),
    );
    assert!(
        !model
            .response(&[1.0, 0.0, 0.0], &[1.0])
            .unwrap()
            .0
            .is_finite()
    );
    let model = Nist::<f64>::new(
        cases.iter().find(|d| d.id == "Rat43").unwrap().clone(),
    );
    let (value, g) = model.response(&[2.0, 1000.0, 1.0, 1.0], &[1.0]).unwrap();
    assert_eq!(value, 0.0);
    assert!(g.iter().all(|v| v.is_finite()));
}

#[test]
fn published_rss_agrees_with_printed_parameter_rounding_screen() {
    for dataset in datasets().unwrap() {
        let model = Nist::<f64>::new(dataset);
        let point = model.dataset.reference_point::<f64>();
        let residuals = model.residuals(&point).unwrap();
        let rows = model.jacobian_rows(&point).unwrap();
        let rss = model.rss(&point).unwrap();
        let mut allowance = model.dataset.rss.half_width;
        for (residual, row) in residuals.iter().zip(rows) {
            let delta: f64 = row
                .iter()
                .zip(&model.dataset.reference)
                .map(|(derivative, value)| derivative.abs() * value.half_width)
                .sum();
            // This local sensitivity screen is not a rigorous reference certificate.
            allowance += 8.0 * residual.abs() * delta + 16.0 * delta * delta;
        }
        assert!(
            (rss - model.dataset.rss.midpoint).abs() <= allowance,
            "{}: RSS {rss:e}, rounding allowance {allowance:e}",
            model.dataset.id
        );
    }
}
