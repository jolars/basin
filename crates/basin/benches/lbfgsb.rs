//! Fresh solves of the L-BFGS-B 3.0 reference drivers, including initialization.

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
#[path = "support/lbfgsb.rs"]
mod support;

// Compile the private production kernels directly, avoiding a separate copy
// of their implementation or a public API solely for benchmarking.
pub use basin::core;
mod backend {
    pub use basin::solver::lbfgs::AsFloatSlice;
}
#[allow(dead_code, unused_imports)]
#[path = "../src/solver/lbfgs/compact.rs"]
mod compact;
#[path = "support/lbfgsb_dimension.rs"]
mod dimension;
#[allow(dead_code, unused_imports)]
#[path = "../src/solver/lbfgs/formk.rs"]
mod formk;
#[allow(dead_code, unused_imports)]
#[path = "../src/solver/lbfgs/history.rs"]
mod history;
#[allow(dead_code, unused_imports)]
#[path = "../src/solver/lbfgs/subsm.rs"]
mod subsm;
#[allow(dead_code, unused_imports)]
#[path = "../src/solver/lbfgs/subspace_direction.rs"]
mod subspace_direction;
#[allow(dead_code, unused_imports)]
#[path = "../src/solver/lbfgs/workspace.rs"]
mod workspace;

use basin::State;
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn benchmark(c: &mut Criterion) {
    let mut alternatives = c.benchmark_group("lbfgsb_dimension");
    for case in dimension::Case::ALL {
        let problem =
            dimension::Quadratic::<Vec<f64>>::new(case, |x| x.to_vec());
        for alternate in [false, true] {
            let result = problem.solve(alternate);
            problem.verify(&result);
            eprintln!(
                "{} alternate={alternate}: cost={} iterations={} counts={:?} termination={:?}",
                case.name(),
                result.cost(),
                result.state.iter(),
                result.state.counts(),
                result.report.code()
            );
            alternatives.bench_function(
                format!(
                    "{}/{}",
                    case.name(),
                    if alternate { "adaptive" } else { "reference" }
                ),
                |b| b.iter(|| black_box(problem.solve(alternate))),
            );
        }
    }
    alternatives.finish();

    let mut models = c.benchmark_group("lbfgsb_models");
    for case in dimension::Case::ALL {
        let problem =
            dimension::Quadratic::<Vec<f64>>::new(case, |x| x.to_vec());
        for policy in [
            basin::solver::lbfgs::model_strategy::ModelDirection::Bounded,
            basin::solver::lbfgs::model_strategy::ModelDirection::InteriorFirst,
            basin::solver::lbfgs::model_strategy::ModelDirection::Projected,
        ] {
            let solve = || {
                basin::Executor::new(
                    &problem,
                    problem
                        .solver()
                        .with_compact_model()
                        .with_direction_policy(policy),
                    basin::FirstOrderState::new(problem.start()),
                )
                .max_iter(500)
                .run()
                .unwrap()
            };
            problem.verify(&solve());
            models.bench_function(format!("{}/{policy:?}", case.name()), |b| {
                b.iter(|| black_box(solve()))
            });
        }
    }
    models.finish();

    let mut projected = c.benchmark_group("lbfgsb_projected");
    for case in dimension::Case::ALL {
        use basin::line_search::projected_armijo::ProjectedArmijo;
        use basin::solver::lbfgs::model_strategy::ProjectedSearch;
        let problem =
            dimension::Quadratic::<Vec<f64>>::new(case, |x| x.to_vec());
        for (name, search) in [
            ("ray", ProjectedSearch::FeasibleRay),
            (
                "armijo",
                ProjectedSearch::Armijo(
                    ProjectedArmijo::new().with_fused_evaluations(),
                ),
            ),
            ("chord", ProjectedSearch::Chord),
        ] {
            for retain in [true, false] {
                let solve = || {
                    basin::Executor::from_start(
                        &problem,
                        problem
                            .solver()
                            .with_compact_model()
                            .with_projected_search(search.clone())
                            .with_curvature_retention(retain),
                        problem.start(),
                    )
                    .max_iter(1000)
                    .run()
                    .unwrap()
                };
                problem.verify(&solve());
                projected.bench_function(
                    format!("{}/{name}/retain{retain}", case.name()),
                    |b| b.iter(|| black_box(solve())),
                );
            }
        }
    }
    projected.finish();

    let mut directions = c.benchmark_group("lbfgsb_subspace");
    for (n, nf) in [(2, 2), (8, 8), (50, 50), (50, 1)] {
        for m in [3, 10] {
            let ws: Vec<Vec<f64>> = (0..m)
                .map(|a| {
                    (0..n)
                        .map(|j| ((a * 13 + j * 7) % 31) as f64 / 31.0 - 0.5)
                        .collect()
                })
                .collect();
            let wy: Vec<Vec<f64>> = ws
                .iter()
                .map(|s| {
                    let sum = s.iter().sum::<f64>();
                    s.iter()
                        .enumerate()
                        .map(|(j, &s)| (j + 1) as f64 * s + 0.125 * sum)
                        .collect()
                })
                .collect();
            let mut sy = vec![0.0; m * m];
            let mut ss = vec![0.0; m * m];
            for i in 0..m {
                for j in 0..=i {
                    sy[i * m + j] =
                        ws[i].iter().zip(&wy[j]).map(|(a, b)| a * b).sum();
                    ss[i * m + j] =
                        ws[i].iter().zip(&ws[j]).map(|(a, b)| a * b).sum();
                    ss[j * m + i] = ss[i * m + j];
                }
            }
            let theta = wy[m - 1].iter().map(|v| v * v).sum::<f64>()
                / sy[(m - 1) * m + m - 1];
            let mut wt = vec![0.0; m * m];
            compact::formt(theta, &sy, &ss, m, m, &mut wt).unwrap();
            let index: Vec<_> = (0..n).rev().collect();
            let mut wn = vec![0.0; 4 * m * m];
            let mut wn1 = wn.clone();
            for col in 1..=m {
                formk::formk(
                    &mut wn,
                    &mut wn1,
                    m,
                    col,
                    theta,
                    &sy,
                    &ws[..col],
                    &wy[..col],
                    nf,
                    &index,
                    0,
                    n,
                    &index,
                    col as u32,
                    true,
                )
                .unwrap();
            }
            let r: Vec<_> = (0..nf).map(|j| 0.02 * (j + 1) as f64).collect();
            let lower = vec![-100.0; n];
            let upper = vec![100.0; n];
            let base = vec![0.5; n];
            let gradient = vec![0.1; n];
            let mut point = base.clone();
            let mut d = r.clone();
            let mut previous = point.clone();
            let mut v = vec![0.0; 2 * m];
            subsm::subsm(
                &mut point,
                &mut d,
                &mut previous,
                &base,
                &gradient,
                &index[..nf],
                &lower,
                &upper,
                &ws,
                &wy,
                &wn,
                &mut v,
                m,
                m,
                theta,
            )
            .unwrap();
            let expected = point.clone();
            let mut scratch = subspace_direction::Workspace::default();
            assert!(scratch.try_direction(
                n,
                &index[..nf],
                &r,
                &ws,
                &wy,
                &sy,
                &wt,
                m,
                theta
            ));
            point.copy_from_slice(&base);
            subsm::project(
                &mut point,
                &mut scratch.direction,
                &mut previous,
                &base,
                &gradient,
                &index[..nf],
                &lower,
                &upper,
            );
            assert!(
                point
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| (a - b).abs() < 1e-8 * (1.0 + b.abs())),
                "n={n} nf={nf} m={m}: actual={point:?} expected={expected:?}"
            );
            directions.bench_function(
                format!("n{n}_free{nf}_m{m}/reference"),
                |b| {
                    b.iter(|| {
                        point.copy_from_slice(&base);
                        d.copy_from_slice(&r);
                        black_box(
                            subsm::subsm(
                                &mut point,
                                &mut d,
                                &mut previous,
                                &base,
                                &gradient,
                                &index[..nf],
                                &lower,
                                &upper,
                                &ws,
                                &wy,
                                &wn,
                                &mut v,
                                m,
                                m,
                                theta,
                            )
                            .unwrap(),
                        );
                        black_box(&point);
                    })
                },
            );
            directions.bench_function(
                format!("n{n}_free{nf}_m{m}/adaptive"),
                |b| {
                    b.iter(|| {
                        point.copy_from_slice(&base);
                        assert!(scratch.try_direction(
                            n,
                            &index[..nf],
                            &r,
                            &ws,
                            &wy,
                            &sy,
                            &wt,
                            m,
                            theta
                        ));
                        black_box(subsm::project(
                            &mut point,
                            &mut scratch.direction,
                            &mut previous,
                            &base,
                            &gradient,
                            &index[..nf],
                            &lower,
                            &upper,
                        ));
                        black_box(&point);
                    })
                },
            );
        }
    }
    directions.finish();
    let mut kernels = c.benchmark_group("lbfgsb_kernels");
    for n in [1, 2, 5, 10, 20, 40] {
        let stride = 2 * n + 4;
        let source: Vec<f64> = (0..n * stride)
            .map(|i| ((i * 17 + 3) % 31) as f64 / 16.0 - 1.0)
            .collect();
        let initial = vec![0.125_f64; n * stride];
        kernels.bench_function(format!("gram/n{n}"), |b| {
            b.iter_batched(
                || initial.clone(),
                |mut target| {
                    compact::add_gram_upper(
                        black_box(&source),
                        &mut target,
                        n,
                        stride,
                    );
                    black_box(target)
                },
                BatchSize::SmallInput,
            );
        });
        let mut positive = vec![-77.0_f64; n * stride];
        for i in 0..n {
            for j in i..n {
                positive[i * stride + j] = if i == j {
                    n as f64 + 1.0
                } else {
                    0.125 / (j - i + 1) as f64
                };
            }
        }
        kernels.bench_function(format!("cholesky/n{n}"), |b| {
            b.iter_batched(
                || positive.clone(),
                |mut target| {
                    let succeeded = compact::cholesky_upper_in_place(
                        &mut target,
                        n,
                        stride,
                    );
                    black_box((succeeded, target))
                },
                BatchSize::SmallInput,
            );
        });
    }
    kernels.finish();

    let mut group = c.benchmark_group("lbfgsb_reference");
    for number in 1..=3 {
        let mut problem = support::Driver::new(number);
        problem.check_feasibility = true;
        problem.verify(&problem.solve());
        problem.check_feasibility = false;
        group.bench_function(format!("driver{number}/vec"), |b| {
            b.iter(|| black_box(problem.solve()));
        });
        #[cfg(feature = "faer_all")]
        {
            let mut problem = support::Driver::<backend_aliases::faer::Col<f64>>::with_backend(number);
            problem.check_feasibility = true;
            problem.verify(&problem.solve());
            problem.check_feasibility = false;
            group.bench_function(format!("driver{number}/faer"), |b| {
                b.iter(|| black_box(problem.solve()));
            });
        }
    }
    for case in [support::short::Case::Mixed, support::short::Case::Rollover] {
        let mut problem = support::short::Short::<Vec<f64>>::new(case);
        for adapter in [
            support::short::Adapter::Reference,
            support::short::Adapter::Trimmed,
            support::short::Adapter::RequiredStops,
        ] {
            problem.check_feasibility = true;
            problem.verify(&problem.solve(adapter));
            problem.check_feasibility = false;
            let name = format!("{}/vec/{}", case.name(), adapter.name());
            group.bench_function(format!("{name}/solve"), |b| {
                b.iter(|| {
                    let result = problem.solve(adapter);
                    black_box(problem.extract(&result));
                });
            });
            // Initialization includes its first fused evaluation and teardown.
            group.bench_function(format!("{name}/initialize_and_drop"), |b| {
                b.iter(|| {
                    black_box(problem.initialize(adapter));
                });
            });
            // Setup is excluded; iteration, extraction, and teardown are timed.
            // Batched phase measurements need not sum to the full-solve timing.
            group.bench_function(format!("{name}/iterations_and_drop"), |b| {
                b.iter_batched(
                    || problem.initialize(adapter),
                    |stepper| {
                        let result = stepper.run_to_end().unwrap();
                        black_box(problem.extract(&result));
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
