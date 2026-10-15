use basin::{
    ClampInPlace, ConvergenceEvidence, ConvergenceTest, CostFunction, Executor,
    NelderMead, Scalar, ScaleInPlace, ScaledAdd, SimplexProgress, Termination,
    TerminationCode, VectorIndex, VectorLen,
};
use std::convert::Infallible;

#[derive(Clone, Copy)]
enum Landscape {
    Quadratic,
    Sphere,
    Rosenbrock,
    Flat,
}

struct Objective<V, F> {
    scale: F,
    magnitude: F,
    offset: F,
    landscape: Landscape,
    bounds: (V, V),
}

impl<V, F> basin::ProblemConstraints for Objective<V, F>
where
    F: Scalar,
    V: VectorIndex<F> + VectorLen,
{
    type Constraints = basin::Constraints<basin::BoxBounds<V>>;
    fn constraint_set(
        &self,
    ) -> <Self::Constraints as basin::ConstraintDescription>::Borrowed<'_> {
        basin::Constraints::new().with_bounds(basin::BoxBounds::new(
            Some(&self.bounds.0),
            Some(&self.bounds.1),
        ))
    }
}
impl<V, F> CostFunction for Objective<V, F>
where
    F: Scalar,
    V: VectorIndex<F> + VectorLen,
{
    type Param = V;
    type Output = F;
    type Error = Infallible;

    fn cost(&self, x: &V) -> Result<F, Infallible> {
        let u = x.get_scalar(0) / self.scale;
        let v = x.get_scalar(1) / self.scale;
        let value = match self.landscape {
            Landscape::Quadratic => {
                (u - F::one()).powi(2) + (v - F::one()).powi(2)
            }
            Landscape::Sphere => (0..x.vec_len())
                .map(|j| (x.get_scalar(j) / self.scale).powi(2))
                .sum(),
            Landscape::Rosenbrock => (0..x.vec_len() - 1)
                .map(|j| {
                    let u = x.get_scalar(j) / self.scale;
                    let v = x.get_scalar(j + 1) / self.scale;
                    F::from_f64(100.0).unwrap() * (v - u * u).powi(2)
                        + (F::one() - u).powi(2)
                })
                .sum(),
            Landscape::Flat => F::one(),
        };
        Ok(self.offset + self.magnitude * value)
    }
}

pub fn check<V, F>(make: impl Fn(&[F]) -> V, expected_relative: F)
where
    F: Scalar,
    V: Clone
        + VectorLen
        + VectorIndex<F>
        + ScaleInPlace<F>
        + ScaledAdd<F>
        + ClampInPlace,
{
    let f = |x| F::from_f64(x).unwrap();
    for landscape in [
        Landscape::Quadratic,
        Landscape::Sphere,
        Landscape::Rosenbrock,
    ] {
        for scale in [1e-6, 1.0, 1e6] {
            for magnitude in [1e-12, 1.0, 1e12] {
                let scale = f(scale);
                let magnitude = f(magnitude);
                let start = if matches!(landscape, Landscape::Rosenbrock) {
                    [f(-1.2) * scale, scale]
                } else {
                    [f(2.0) * scale, -scale]
                };
                let problem = Objective {
                    scale,
                    magnitude,
                    offset: F::zero(),
                    landscape,
                    bounds: (
                        make(&[-f(5.0) * scale; 2]),
                        make(&[f(5.0) * scale; 2]),
                    ),
                };
                let seed = SimplexProgress::new(make(&start));
                let initial_costs: Vec<F> = seed
                    .vertices()
                    .iter()
                    .map(|x| problem.cost(x).unwrap())
                    .collect();
                let best_initial =
                    initial_costs.iter().copied().fold(F::infinity(), F::min);
                let initial_spread =
                    initial_costs.iter().fold(F::zero(), |spread, &cost| {
                        spread.max((cost - best_initial).abs())
                    });
                let result = Executor::new(
                    basin::SamplingDomain::new(problem),
                    NelderMead::new(),
                    seed,
                )
                .max_iter(2_000)
                .run()
                .unwrap();
                assert_eq!(
                    result.report.code(),
                    TerminationCode::SimplexTolerance
                );
                assert!(result.iter() > 0 && result.iter() < 2_000);
                let Termination::Converged(convergence) =
                    &result.report.termination
                else {
                    panic!("expected simplex convergence");
                };
                assert_eq!(convergence.criteria().len(), 1);
                assert_eq!(
                    convergence.criteria()[0].test,
                    ConvergenceTest::Simplex
                );
                let ConvergenceEvidence::Simplex {
                    size: Some(size),
                    cost_spread: Some(cost),
                } = &convergence.criteria()[0].evidence
                else {
                    panic!("expected paired simplex evidence");
                };
                let best = &result.state.vertices()[0];
                let norm = (0..2).fold(F::zero(), |norm, j| {
                    norm.max(best.get_scalar(j).abs())
                });
                let distance = result.state.vertices().iter().fold(
                    F::zero(),
                    |distance, x| {
                        (0..2).fold(distance, |distance, j| {
                            distance.max(
                                (x.get_scalar(j) - best.get_scalar(j)).abs(),
                            )
                        })
                    },
                );
                let spread = result.state.costs().iter().fold(
                    F::zero(),
                    |spread, &value| {
                        spread.max((value - result.state.costs()[0]).abs())
                    },
                );
                assert_eq!(size.value, distance);
                assert_eq!(cost.value, spread);
                assert_eq!(size.reference, Some(norm));
                assert_eq!(
                    cost.reference,
                    Some(result.state.costs()[0].abs().max(initial_spread))
                );
                for threshold in [size, cost] {
                    assert_eq!(threshold.absolute_tolerance, Some(f(1e-8)));
                    assert_eq!(
                        threshold.relative_tolerance,
                        Some(expected_relative)
                    );
                    assert_eq!(
                        threshold.tolerance,
                        f(1e-8).max(
                            expected_relative * threshold.reference.unwrap()
                        )
                    );
                    assert!(threshold.value <= threshold.tolerance);
                }
                let target = if matches!(landscape, Landscape::Sphere) {
                    F::zero()
                } else {
                    F::one()
                };
                let error = (0..2).fold(F::zero(), |error, j| {
                    error.max((best.get_scalar(j) / scale - target).abs())
                });
                let accuracy = f(8.0) * (f(1e-8) / scale + expected_relative);
                assert!(
                    error <= accuracy,
                    "error {error:?}, accuracy {accuracy:?}"
                );
            }
        }
    }

    for landscape in [Landscape::Quadratic, Landscape::Flat] {
        let problem = Objective {
            scale: F::one(),
            magnitude: F::one(),
            offset: F::zero(),
            landscape,
            bounds: (make(&[-f(5.0); 2]), make(&[f(5.0); 2])),
        };
        let result = Executor::new(
            basin::SamplingDomain::new(problem),
            NelderMead::adaptive(),
            SimplexProgress::new(make(&[F::zero(); 2])),
        )
        .max_iter(1_000)
        .run()
        .unwrap();
        assert_eq!(result.report.code(), TerminationCode::SimplexTolerance);
        assert!(result.iter() > 0 && result.iter() < 1_000);
        if matches!(landscape, Landscape::Quadratic) {
            for j in 0..2 {
                assert!(
                    (result.param().get_scalar(j) - F::one()).abs() < f(1e-3)
                );
            }
        }
    }

    for offset in [-1e4, 1e4] {
        let problem = Objective {
            scale: F::one(),
            magnitude: F::one(),
            offset: f(offset),
            landscape: Landscape::Quadratic,
            bounds: (make(&[-f(5.0); 2]), make(&[f(5.0); 2])),
        };
        let result = Executor::new(
            basin::SamplingDomain::new(problem),
            NelderMead::default(),
            SimplexProgress::new(make(&[f(2.0), -F::one()])),
        )
        .max_iter(1_000)
        .run()
        .unwrap();
        assert_eq!(result.report.code(), TerminationCode::SimplexTolerance);
        let accuracy = f(8.0) * (F::epsilon() * f(offset).abs()).sqrt();
        for j in 0..2 {
            assert!(
                (result.param().get_scalar(j) - F::one()).abs() <= accuracy
            );
        }
    }

    // This SplitMix64 seed-2 corpus start previously stopped near cost 0.638
    // in f32 with standard coefficients and sqrt(epsilon) relative tolerances.
    let start = [
        0.36475893679231763,
        0.9965987354952985,
        0.38255232560002117,
        1.061676616780118,
        -0.7536452512755436,
        -0.6135109183532039,
        0.905414458066991,
        0.9563492973903163,
        -0.9987505112347921,
        0.9104638583355604,
    ]
    .map(f);
    for adaptive in [false, true] {
        for projected in [false, true] {
            let problem = Objective {
                scale: F::one(),
                magnitude: F::one(),
                offset: F::zero(),
                landscape: Landscape::Rosenbrock,
                bounds: (make(&[-f(2.0); 10]), make(&[f(2.0); 10])),
            };
            let solver = if adaptive {
                NelderMead::adaptive()
            } else {
                NelderMead::new()
            };
            let state = SimplexProgress::new(make(&start));
            let result = if projected {
                Executor::new(problem, solver.projected(), state)
                    .max_iter(10_000)
                    .max_cost_evals(20_000)
                    .run()
                    .unwrap()
            } else {
                Executor::new(
                    basin::SamplingDomain::new(problem),
                    solver,
                    state,
                )
                .max_iter(10_000)
                .max_cost_evals(20_000)
                .run()
                .unwrap()
            };
            assert_eq!(result.report.code(), TerminationCode::SimplexTolerance);
            assert!(result.cost() < f(1e-8), "cost {:?}", result.cost());
            for j in 0..10 {
                assert!(
                    (result.param().get_scalar(j) - F::one()).abs() < f(1e-4),
                    "coordinate {j}: {:?}",
                    result.param().get_scalar(j)
                );
            }
        }
    }

    for fixed in [false, true] {
        let upper = if fixed { F::zero() } else { f(2.0) };
        let problem = Objective {
            scale: F::one(),
            magnitude: F::one(),
            offset: F::zero(),
            landscape: Landscape::Quadratic,
            bounds: (make(&[F::zero(); 2]), make(&[upper; 2])),
        };
        let result = Executor::new(
            problem,
            NelderMead::with_params(F::one(), f(2.0), f(0.5), f(0.5))
                .projected(),
            SimplexProgress::from_simplex(vec![
                make(&[f(2.0); 2]),
                make(&[f(3.0), f(2.0)]),
                make(&[f(2.0), f(3.0)]),
            ]),
        )
        .max_iter(10)
        .run()
        .unwrap();
        assert_eq!(result.report.code(), TerminationCode::SimplexTolerance);
        assert_eq!(result.iter(), 0);
        assert_eq!(result.cost_evals(), 3);
        assert!(result.cost() > F::zero());
        for vertex in result.state.vertices() {
            for j in 0..2 {
                assert_eq!(vertex.get_scalar(j), upper);
            }
        }
    }
}
