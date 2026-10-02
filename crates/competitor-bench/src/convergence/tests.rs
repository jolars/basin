use super::*;

#[test]
fn all_nist_files_and_certificates() {
    let cases = nist::cases();
    assert_eq!(cases.len(), 27);
    for nist in cases {
        let case = cases_for(&nist);
        assert_eq!(nist.starts[0].len(), nist.certified.len());
        assert_eq!(nist.starts[1].len(), nist.certified.len());
        let (r, j) = case.derivatives(&nist.certified);
        let rss = r.iter().map(|r| r * r).sum::<f64>();
        let error_norm = j
            .chunks_exact(nist.certified.len())
            .zip(&nist.data)
            .map(|(row, data)| {
                let uncertainty = row
                    .iter()
                    .zip(&nist.parameter_rounding)
                    .map(|(a, b)| a.abs() * b)
                    .sum::<f64>()
                    + 128.0 * f64::EPSILON * data[0].abs().max(1.0);
                uncertainty * uncertainty
            })
            .sum::<f64>()
            .sqrt();
        let tolerance = 2.0 * nist.rss.sqrt() * error_norm
            + error_norm * error_norm
            + 1e-9 * nist.rss;
        assert!(
            (rss - nist.rss).abs() <= tolerance,
            "{}: rss={rss:e}, certificate={:e}, allowance={tolerance:e}",
            nist.name,
            nist.rss
        );
    }
}

fn cases_for(n: &nist::Nist) -> Case {
    Case {
        name: n.name.into(),
        family: n.family,
        validation: n.validation,
        starts: n.starts.clone(),
        reference: n.certified.clone(),
        reference_cost: n.rss / 2.0,
        identifiable: false,
        model: Model::Nist(n.clone()),
    }
}

#[test]
fn derivatives_match_independent_central_differences() {
    for case in cases() {
        for p in case.starts.iter().chain(std::iter::once(&case.reference)) {
            let (_, j) = case.derivatives(p);
            for col in 0..p.len() {
                let h = 2e-5 * p[col].abs().max(1e-10);
                let mut plus = p.clone();
                plus[col] += h;
                let mut minus = p.clone();
                minus[col] -= h;
                let rp = case.residuals(&plus);
                let rm = case.residuals(&minus);
                for (row, (&a, &b)) in rp.iter().zip(&rm).enumerate() {
                    let numeric = (a - b) / (2.0 * h);
                    let exact = j[row * p.len() + col];
                    let allowance = 2e-4 * exact.abs().max(1e-5)
                        + 1e-8 * (a.abs() + b.abs()) / h;
                    assert!(
                        (exact - numeric).abs() <= allowance,
                        "{} col={col} row={row} exact={exact:e} numeric={numeric:e}",
                        case.name
                    );
                }
            }
            let p32: Vec<_> = p.iter().map(|&x| x as f32).collect();
            let (r32, j32) = case.derivatives(&p32);
            let rounded: Vec<_> = p32.iter().map(|&x| x as f64).collect();
            let (r64, j64) = case.derivatives(&rounded);
            // Subtracting a large observation can lose absolute accuracy even
            // when the model evaluation has small relative error.
            let response_scale = match &case.model {
                Model::Nist(m) => {
                    m.data.iter().map(|r| r[0].abs()).fold(1.0, f64::max)
                }
                Model::Quadratic(q) => {
                    q.condition.sqrt()
                        * p.iter().copied().map(f64::abs).fold(1.0, f64::max)
                }
            };
            for (a, b) in r32.iter().zip(&r64) {
                assert!(
                    (*a as f64 - b).abs()
                        < 0.003 * b.abs() + 3e-6 * response_scale,
                    "{}: f32 {a}, f64 {b}",
                    case.name
                );
            }
            for (a, b) in j32.iter().zip(&j64) {
                assert!(
                    (*a as f64 - b).abs() < 0.003 * (1.0 + b.abs()),
                    "{}: f32 Jacobian {a}, f64 {b}",
                    case.name
                );
            }
        }
    }
}

#[test]
fn analytic_controls_have_known_optima_and_whole_family_partitions() {
    let mut partitions = std::collections::BTreeMap::new();
    for case in cases() {
        if let Some(previous) = partitions.insert(case.family, case.validation)
        {
            assert_eq!(previous, case.validation);
        }
        if matches!(case.model, Model::Quadratic(_)) {
            assert!(
                (case.cost(&case.reference) - case.reference_cost).abs()
                    < 1e-14
            );
            assert!(
                case.gradient(&case.reference)
                    .iter()
                    .all(|g| g.abs() < 1e-14)
            );
            assert!(
                case.starts
                    .iter()
                    .all(|p| case.cost(p) > case.reference_cost)
            );
        }
    }
}

#[test]
fn fused_calls_charge_one_pass_and_diagnostics_do_not_charge() {
    let case = cases().into_iter().find(|c| c.name == "Misra1a").unwrap();
    let fixture = Fixture::<f64>::new(case.clone());
    let mut problem = basin::Problem::new(fixture.clone());
    let (cost, _) = problem.cost_and_gradient(&case.starts[0]).unwrap();
    assert_eq!(fixture.work.lock().unwrap().passes(), 1);
    assert_eq!(problem.counts().cost_evals, 1);
    assert_eq!(problem.counts().gradient_evals, 1);
    problem.residual_and_jacobian(&case.starts[0]).unwrap();
    assert_eq!(fixture.work.lock().unwrap().passes(), 2);
    assert_eq!(problem.counts().residual_evals, 1);
    assert_eq!(problem.counts().jacobian_evals, 1);
    assert!((case.cost(&case.starts[0]) - cost).abs() < 1e-10);
    assert_eq!(fixture.work.lock().unwrap().passes(), 2);
}

#[test]
fn batch_gradients_are_unbiased_and_seeds_are_reproducible() {
    use basin::{PointState, Sgd};
    let case = cases()
        .into_iter()
        .find(|c| c.name == "quadratic_n5_k1_rot1_rank5_noise0.1")
        .unwrap();
    let fixture = Fixture::<f64>::new(case.clone());
    let p = &case.starts[0];
    let gradient = fixture.gradient(p).unwrap();
    let indices: Vec<_> = (0..fixture.n_samples()).collect();
    let full_batch = fixture.batch_gradient(p, &indices).unwrap();
    let mut average = vec![0.0; p.len()];
    for i in indices {
        for (mean, sample) in average
            .iter_mut()
            .zip(fixture.batch_gradient(p, &[i]).unwrap())
        {
            *mean += sample / fixture.n_samples() as f64;
        }
    }
    for ((a, b), c) in gradient.iter().zip(&full_batch).zip(&average) {
        assert!((a - b).abs() < 1e-12 && (a - c).abs() < 1e-12);
    }
    let solve = |seed| {
        basin::Executor::new(
            Fixture::<f64>::new(case.clone()),
            Sgd::new(0.02, 1, seed),
            PointState::new(p.clone()),
        )
        .max_iter(20)
        .run()
        .unwrap()
    };
    let a = solve(42);
    let b = solve(42);
    let c = solve(43);
    assert_eq!(a.param(), b.param());
    assert_eq!(a.counts, b.counts);
    assert_ne!(a.param(), c.param());
}

#[test]
fn all_suites_keep_related_cases_in_one_partition() {
    let mut partitions = std::collections::BTreeMap::new();
    let families = cases()
        .into_iter()
        .map(|c| (c.family, c.validation))
        .chain(
            constraints::cases()
                .into_iter()
                .map(|c| (c.family, c.validation)),
        )
        .chain(
            scalar::cases()
                .into_iter()
                .map(|c| (c.family, c.validation)),
        );
    for (family, validation) in families {
        if let Some(old) = partitions.insert(family, validation) {
            assert_eq!(old, validation, "{family}");
        }
    }
    assert!(
        partitions
            .values()
            .filter(|&&validation| validation)
            .count()
            * 3
            >= partitions.len()
    );
}

#[test]
fn invalid_model_points_remain_nonfinite_in_diagnostics() {
    let case = cases().into_iter().find(|c| c.name == "Bennett5").unwrap();
    let mut p = case.reference.clone();
    p[1] = -1e6;
    let quality = trace::vector_quality(&case, &p);
    assert!(quality.cost.is_nan());
    assert!(quality.gradient_inf.is_nan());
    p[0] = f64::NAN;
    assert!(trace::vector_quality(&case, &p).parameter_error.is_nan());
    assert!(trace::vector_quality(&case, &p).violation.is_nan());
}
