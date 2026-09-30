use std::convert::Infallible;

#[path = "support/de_backend.rs"]
mod backend;

use basin::{
    BoxConstraints, CostFunction, De, DeCrossover, DeInject, DeMutation,
    ExactCheckpoint, Executor, NelderMead, PopulationProgress, State,
};

#[derive(Clone)]
struct Quadratic {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

const MUTATIONS: [DeMutation; 6] = [
    DeMutation::Rand1,
    DeMutation::Best1,
    DeMutation::Rand2,
    DeMutation::Best2,
    DeMutation::RandToBest1,
    DeMutation::CurrentToBest1,
];
const CROSSOVERS: [DeCrossover; 2] =
    [DeCrossover::Binomial, DeCrossover::Exponential];

fn configured(
    mutation: DeMutation,
    crossover: DeCrossover,
    dither: bool,
) -> De {
    let solver = De::new(42)
        .with_pop_size(24)
        .with_mutation(mutation)
        .with_crossover(crossover);
    if dither {
        solver.with_dither(0.5, 1.0)
    } else {
        solver
    }
}

type Checkpoint = ExactCheckpoint<De, PopulationProgress<Vec<f64>>>;

fn solve(solver: De, generations: u64) -> Checkpoint {
    Executor::new(Quadratic::new(), solver, PopulationProgress::empty())
        .max_iter(generations)
        .run_with_solver()
        .unwrap()
        .into_checkpoint()
}

fn assert_same_population(
    a: &PopulationProgress<Vec<f64>>,
    b: &PopulationProgress<Vec<f64>>,
) {
    assert_eq!(a.candidates(), b.candidates());
    assert_eq!(a.costs(), b.costs());
    assert_eq!(a.cost_evals(), b.cost_evals());
    assert_eq!(a.iter(), b.iter());
}

#[test]
fn every_combination_repeats_resets_and_resumes_exactly() {
    for mutation in MUTATIONS {
        for crossover in CROSSOVERS {
            for dither in [false, true] {
                let make = || configured(mutation, crossover, dither);
                let reference = solve(make(), 20);
                let repeated = solve(make(), 20);
                assert_same_population(reference.state(), repeated.state());
                assert_eq!(reference.counts().cost_evals, 24 * 21);

                let split = solve(make(), 7);
                #[cfg(feature = "serde")]
                let split: Checkpoint = postcard::from_bytes(
                    &postcard::to_allocvec(&split).unwrap(),
                )
                .unwrap();
                let resumed =
                    Executor::resume_from_checkpoint(Quadratic::new(), split)
                        .max_iter(20)
                        .run_with_solver()
                        .unwrap()
                        .into_checkpoint();
                assert_same_population(reference.state(), resumed.state());
                assert_eq!(reference.counts(), resumed.counts());
                #[cfg(feature = "serde")]
                assert_eq!(
                    postcard::to_allocvec(&reference).unwrap(),
                    postcard::to_allocvec(&resumed).unwrap()
                );

                let (used_solver, _, _) = resumed.into_parts();
                let fresh = solve(used_solver, 20);
                assert_same_population(reference.state(), fresh.state());
            }
        }
    }
}

#[test]
fn solutions_agree_with_scipy_1_16_2() {
    let rows: Vec<_> = include_str!("fixtures/de_solutions.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect();
    assert_eq!(rows.len(), 12);
    for (i, mutation) in MUTATIONS.into_iter().enumerate() {
        for (j, crossover) in CROSSOVERS.into_iter().enumerate() {
            let fields: Vec<_> = rows[2 * i + j].split_whitespace().collect();
            let cost: f64 = fields[1].parse().unwrap();
            let point: Vec<f64> =
                fields[3..].iter().map(|x| x.parse().unwrap()).collect();
            assert_eq!(Quadratic::new().cost(&point).unwrap(), cost);
            let result = solve(configured(mutation, crossover, true), 200);
            assert!(
                (result.state().cost() - cost).abs() < 1e-12,
                "{mutation:?} {crossover:?}: {}",
                result.state().cost()
            );
            for (actual, expected) in result.state().param().iter().zip(point) {
                assert!((actual - expected).abs() < 1e-6);
            }
        }
    }
}

#[test]
fn dithering_builder_precedence_and_convergence_forwarding() {
    let make = || De::new(42).with_pop_size(24);
    assert_same_population(
        solve(make().with_f(0.7), 10).state(),
        solve(make().with_dither(0.5, 1.0).with_f(0.7), 10).state(),
    );
    assert_same_population(
        solve(make().with_dither(0.5, 1.0), 10).state(),
        solve(make().with_f(0.7).with_dither(0.5, 1.0), 10).state(),
    );
    let result = Executor::new(
        Quadratic::new(),
        De::new(42)
            .with_absolute_cost_change_tolerance(None)
            .with_mutation(DeMutation::Best2)
            .with_crossover(DeCrossover::Exponential)
            .with_pop_size(24)
            .with_dither(0.5, 1.0),
        PopulationProgress::empty(),
    )
    .max_iter(10)
    .run()
    .unwrap();
    assert_same_population(
        &result.state,
        solve(
            configured(DeMutation::Best2, DeCrossover::Exponential, true),
            10,
        )
        .state(),
    );
}

#[test]
fn mutation_population_requirements_are_independent_of_builder_order() {
    for mutation in MUTATIONS {
        let minimum = match mutation {
            DeMutation::Rand2 => 6,
            DeMutation::Best2 => 5,
            _ => 4,
        };
        for population_first in [false, true] {
            let make = |n| {
                if population_first {
                    De::new(42).with_pop_size(n).with_mutation(mutation)
                } else {
                    De::new(42).with_mutation(mutation).with_pop_size(n)
                }
            };
            assert_eq!(
                solve(make(minimum), 2).state().candidates().len(),
                minimum
            );
            assert!(
                std::panic::catch_unwind(|| solve(make(minimum - 1), 0))
                    .is_err()
            );
        }
    }
}

#[test]
fn invalid_configuration_and_empty_bounds_panic() {
    for (min, max) in [
        (0.0, 1.0),
        (-1.0, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (f64::NAN, 1.0),
        (0.5, f64::NAN),
        (0.5, f64::INFINITY),
    ] {
        assert!(
            std::panic::catch_unwind(|| De::new(1).with_dither(min, max))
                .is_err()
        );
    }
    for cr in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
        assert!(
            std::panic::catch_unwind(|| De::<f64>::new(1).with_cr(cr)).is_err()
        );
    }
    for f in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(std::panic::catch_unwind(|| De::new(1).with_f(f)).is_err());
    }
    assert!(
        std::panic::catch_unwind(|| Executor::new(
            Quadratic {
                lower: vec![],
                upper: vec![]
            },
            De::new(1),
            PopulationProgress::empty(),
        )
        .max_iter(0)
        .run())
        .is_err()
    );
}

#[test]
fn fixed_coordinates_one_dimension_and_large_scales_remain_feasible() {
    for mutation in MUTATIONS {
        for crossover in CROSSOVERS {
            for (lower, upper) in [
                (vec![0.25], vec![0.25]),
                (vec![-1.0], vec![1.0]),
                (vec![-1.0, 0.25], vec![1.0, 0.25]),
            ] {
                let problem = Quadratic { lower, upper };
                let mut stepper = Executor::new(
                    problem.clone(),
                    configured(mutation, crossover, false).with_f(f64::MAX),
                    PopulationProgress::empty(),
                )
                .max_iter(4)
                .into_stepper()
                .unwrap();
                for _ in 0..4 {
                    stepper.step().unwrap();
                    for x in stepper.state().candidates() {
                        for (i, &value) in x.iter().enumerate() {
                            assert!(
                                value.is_finite()
                                    && value >= problem.lower[i]
                                    && value <= problem.upper[i]
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn configured_de_composes_with_local_search() {
    for mutation in MUTATIONS {
        for crossover in CROSSOVERS {
            let solver = DeInject::with_inner_solver(
                configured(mutation, crossover, true),
                NelderMead::adaptive(),
            )
            .with_inner_max_iter(20)
            .with_refine_every(2);
            let result = Executor::new(
                Quadratic::new(),
                solver,
                PopulationProgress::empty(),
            )
            .max_iter(12)
            .run()
            .unwrap();
            assert!(result.cost() < 1e-5);
            assert!(result.cost_evals() > 24 * 13);
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn thread_count_does_not_change_generations() {
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let many = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for mutation in MUTATIONS {
        for crossover in CROSSOVERS {
            let run = || solve(configured(mutation, crossover, true), 20);
            let a = one.install(run);
            let b = many.install(run);
            assert_same_population(a.state(), b.state());
        }
    }
}

#[test]
fn serial_reference_is_feature_independent() {
    // Captured without `parallel`; the same constants also check parallel builds.
    let expected = [
        [0xfa4fe27c9743bf67, 0xee72bc9e59ef17cb],
        [0x2c004f7581401f8f, 0x56844f737deef9bb],
        [0x112ac0cac55df8fd, 0x6fc9b230b267df7b],
        [0x45d703290ab98b7b, 0xdbbf78bffe50036c],
        [0x94983281ecdd9b7b, 0xdc1914d1035b7697],
        [0xd7aa73f81526e1c4, 0xd9da3182aea568c5],
    ];
    // Three coordinates distinguish the exponential segment from a binomial mask.
    for (i, mutation) in MUTATIONS.into_iter().enumerate() {
        for (j, crossover) in CROSSOVERS.into_iter().enumerate() {
            let result = Executor::new(
                Quadratic {
                    lower: vec![-2.0, -3.0, -4.0],
                    upper: vec![3.0, 4.0, 5.0],
                },
                configured(mutation, crossover, true),
                PopulationProgress::empty(),
            )
            .max_iter(20)
            .run()
            .unwrap();
            assert_eq!(
                population_fingerprint(&result.state),
                expected[i][j],
                "{mutation:?} {crossover:?}"
            );
        }
    }
}

impl Quadratic {
    fn new() -> Self {
        Self {
            lower: vec![-2.0, -3.0],
            upper: vec![3.0, 4.0],
        }
    }
}

impl CostFunction for Quadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(x.iter()
            .enumerate()
            .map(|(i, x)| (i + 1) as f64 * (x - 0.25).powi(2))
            .sum())
    }
}

impl BoxConstraints for Quadratic {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

fn population_fingerprint(state: &PopulationProgress<Vec<f64>>) -> u64 {
    state
        .candidates()
        .iter()
        .flatten()
        .chain(state.costs())
        .fold(0xcbf29ce484222325, |hash, x| {
            (hash ^ x.to_bits()).wrapping_mul(0x100000001b3)
        })
}

#[test]
fn legacy_default_trajectory() {
    // Captured from Basin's original rand/1/bin implementation before expansion.
    let expected = [
        0xa08e372be239ce1e,
        0x205b15f436e88878,
        0x2735be7516a9575e,
        0xafb9c0afb84ef594,
        0x275c0be219b0425b,
        0x31aff2f34fbc9e49,
    ];
    let mut stepper = Executor::new(
        Quadratic::new(),
        De::new(0xdecade).with_pop_size(6),
        PopulationProgress::empty(),
    )
    .max_iter(5)
    .into_stepper()
    .unwrap();
    for (generation, expected) in expected.into_iter().enumerate() {
        assert_eq!(population_fingerprint(stepper.state()), expected);
        assert_eq!(stepper.state().cost_evals(), 6 * (generation as u64 + 1));
        if generation < 5 {
            stepper.step().unwrap();
        }
    }
}

#[test]
fn vec_backends() {
    backend::check::<_, f32>(|x| x.to_vec());
    backend::check::<_, f64>(|x| x.to_vec());
}

macro_rules! nalgebra_test {
    ($name:ident, $feature:literal, $backend:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        fn $name() {
            backend::check::<_, f32>($backend::DVector::from_column_slice);
            backend::check::<_, f64>($backend::DVector::from_column_slice);
        }
    };
}
macro_rules! ndarray_test {
    ($name:ident, $feature:literal, $backend:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        fn $name() {
            backend::check::<_, f32>(|x| {
                $backend::Array1::from_vec(x.to_vec())
            });
            backend::check::<_, f64>(|x| {
                $backend::Array1::from_vec(x.to_vec())
            });
        }
    };
}
macro_rules! faer_test {
    ($name:ident, $feature:literal, $backend:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        fn $name() {
            backend::check::<_, f32>(|x| {
                $backend::Col::from_fn(x.len(), |i| x[i])
            });
            backend::check::<_, f64>(|x| {
                $backend::Col::from_fn(x.len(), |i| x[i])
            });
        }
    };
}

nalgebra_test!(nalgebra_032, "nalgebra_v0_32", nalgebra_0_32);
nalgebra_test!(nalgebra_033, "nalgebra_v0_33", nalgebra_0_33);
nalgebra_test!(nalgebra_034, "nalgebra_v0_34", nalgebra_0_34);
nalgebra_test!(nalgebra_035, "nalgebra_v0_35", nalgebra);
ndarray_test!(ndarray_015, "ndarray_v0_15", ndarray_0_15);
ndarray_test!(ndarray_016, "ndarray_v0_16", ndarray_0_16);
ndarray_test!(ndarray_017, "ndarray_v0_17", ndarray);
faer_test!(faer_022, "faer_v0_22", faer_0_22);
faer_test!(faer_023, "faer_v0_23", faer_0_23);
faer_test!(faer_024, "faer_v0_24", faer);
