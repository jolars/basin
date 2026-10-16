//! The adapters must preserve fixture arithmetic and diagnostic accounting.

use std::marker::PhantomData;

use basin::{Jacobian, MatTransposeVec, Residual, Scalar};
use competitor_bench::{
    lm_defaults::*,
    nist::{Nist, datasets},
};

fn value<F: Scalar>(x: F) -> f64 {
    x.to_f64().unwrap()
}

fn adapters<B: Backend<Float = F>, F: Scalar>()
where
    B::Matrix: MatTransposeVec<B::Vector>,
{
    for dataset in datasets().unwrap() {
        let model = Nist::<B::Float>::new(dataset);
        for start in 0..2 {
            let parameters = model.dataset.start::<B::Float>(start);
            let x = B::vector(parameters.clone());
            let adapter = Adapter::<B, _> {
                model: &model,
                backend: PhantomData,
            };
            assert_eq!(
                B::values(&adapter.residual(&x).unwrap()),
                model.residuals(&parameters).unwrap()
            );
            let rows = model.jacobian_rows(&parameters).unwrap();
            let weights: Vec<_> = (0..rows.len())
                .map(|i| {
                    F::from_f64((i + 1) as f64 / rows.len() as f64).unwrap()
                })
                .collect();
            let actual = B::values(
                &adapter
                    .jacobian(&x)
                    .unwrap()
                    .mat_transpose_vec(&B::vector(weights.clone())),
            );
            assert_eq!(actual.len(), parameters.len());
            for j in 0..parameters.len() {
                let expected = rows
                    .iter()
                    .zip(&weights)
                    .fold(F::zero(), |sum, (row, w)| sum + row[j] * *w);
                let magnitude: f64 = rows
                    .iter()
                    .zip(&weights)
                    .map(|(row, w)| value(row[j] * *w).abs())
                    .sum();
                let allowance =
                    64.0 * value(F::epsilon()) * rows.len() as f64 * magnitude;
                assert!(
                    (value(actual[j]) - value(expected)).abs() <= allowance,
                    "{} {} {} column {j}",
                    B::NAME,
                    B::PRECISION,
                    model.dataset.id
                );
            }
        }
    }
}

#[test]
fn all_backends_preserve_residuals_and_rectangular_jacobians() {
    adapters::<VecF64, _>();
    adapters::<NalgebraF64, _>();
    adapters::<NdarrayF64, _>();
    adapters::<FaerF64, _>();
    adapters::<VecF32, _>();
    adapters::<NalgebraF32, _>();
    adapters::<NdarrayF32, _>();
    adapters::<FaerF32, _>();
}

fn diagnostics<B: Backend>() {
    let dataset = datasets()
        .unwrap()
        .into_iter()
        .find(|d| d.id == "Misra1a")
        .unwrap();
    let model = Nist::<B::Float>::new(dataset);
    for configuration in Configuration::ALL {
        let plain =
            B::solve(&model, model.dataset.start(0), configuration, false)
                .unwrap();
        let traced =
            B::solve(&model, model.dataset.start(0), configuration, true)
                .unwrap();
        assert_eq!(
            B::values(plain.result.param()),
            B::values(traced.result.param())
        );
        assert_eq!(plain.result.counts, traced.result.counts);
        assert_eq!(plain.result.report.code(), traced.result.report.code());
        assert_eq!(plain.result.iter(), traced.result.iter());
        let trace = traced.trace.unwrap();
        assert!(trace.model_solves >= trace.accepted + trace.rejected);
        assert_eq!(
            trace.accepted + trace.rejected + 1,
            traced.result.counts.residual_evals
        );
        assert!(plain.trace.is_none());
    }
}

#[test]
fn tracing_preserves_trajectories_and_does_not_double_count_trials() {
    diagnostics::<VecF64>();
    diagnostics::<NalgebraF64>();
    diagnostics::<NdarrayF64>();
    diagnostics::<FaerF64>();
    diagnostics::<VecF32>();
    diagnostics::<NalgebraF32>();
    diagnostics::<NdarrayF32>();
    diagnostics::<FaerF32>();
}

#[test]
fn recovery_checks_detect_parameter_error_and_nonfinite_results() {
    let dataset = datasets()
        .unwrap()
        .into_iter()
        .find(|d| d.id == "Misra1a")
        .unwrap();
    let point = dataset.reference_point::<f64>();
    assert!(Quality::at(&dataset, &point).unwrap().recovered(1e-6));
    assert!(
        !Quality::at(&dataset, &dataset.start::<f64>(0))
            .unwrap()
            .recovered(1e-4)
    );
    let mut invalid = point;
    invalid[0] = f64::NAN;
    assert!(!Quality::at(&dataset, &invalid).unwrap().recovered(1e-3));
    assert_eq!(relative_error(0.25, 0.0), 0.25);
    assert_eq!(relative_error(f64::NAN, 0.0), f64::INFINITY);
}
