//! Serial, matched-case comparison of Basin covariance refresh schedules.

use std::convert::Infallible;
use std::marker::PhantomData;
use std::ops::Index;
use std::sync::atomic::Ordering;
use std::time::Instant;

use basin::{
    BoundedCmaEs, BoundedCmaInject, BoxBounds, CmaEs, CmaInject, Constrained,
    ConstraintSet, CostFunction, DenseMatrix, EvaluationKind, Executor,
    NelderMead, PopulationProgress, Scalar, Solver,
};

use super::{BasinMatrix, BasinVector, Case, Workload};

const MAX_DIMENSION: usize = 48;

struct Objective<'a, V, F> {
    workload: &'a Workload,
    bounded: bool,
    param: PhantomData<(V, F)>,
}

impl<V, F> CostFunction for Objective<'_, V, F>
where
    V: Index<usize, Output = F>,
    F: Scalar,
{
    type Param = V;
    type Output = F;
    type Error = Infallible;

    fn cost(&self, x: &V) -> Result<F, Infallible> {
        self.workload.calls.fetch_add(1, Ordering::Relaxed);
        let mut storage = [0.0; MAX_DIMENSION];
        let point = &mut storage[..self.workload.weights.len()];
        for (i, value) in point.iter_mut().enumerate() {
            *value = x[i].to_f64().unwrap();
        }
        Ok(F::from_f64(objective(self.workload, point, self.bounded)).unwrap())
    }
}

fn objective(workload: &Workload, point: &[f64], bounded: bool) -> f64 {
    let cost = workload.cost(point);
    // With x[0] <= 0, the added term is nonnegative and vanishes at the known
    // minimizer. It pushes the solution against the upper face.
    if bounded { cost - point[0] } else { cost }
}

fn box_violation(point: &[f64]) -> f64 {
    point
        .iter()
        .enumerate()
        .fold(0.0_f64, |violation, (i, &x)| {
            let upper = if i == 0 { 0.0 } else { 5.0 };
            violation.max(x - upper).max(-5.0 - x)
        })
}

fn automatic_period(n: usize, lambda: usize) -> u64 {
    // Basin's Hansen Table 1 learning rates, evaluated consistently in f64.
    let weights = (1..=lambda / 2)
        .map(|i| ((lambda as f64 + 1.0) / 2.0).ln() - (i as f64).ln());
    let mu_eff = weights.clone().sum::<f64>().powi(2)
        / weights.map(|w| w * w).sum::<f64>();
    let n = n as f64;
    let c1 = 2.0 / ((n + 1.3).powi(2) + mu_eff);
    let cmu = (1.0 - c1).min(
        2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0).powi(2) + mu_eff),
    );
    // Hansen (2016), Appendix C uses a strict gap in population evaluations.
    1 + (1.0 / (10.0 * n * (c1 + cmu))).floor() as u64
}

#[derive(Clone, Copy)]
struct Scenario {
    case: Case,
    n: usize,
    lambda: usize,
    condition: f64,
}

#[derive(Clone, Copy)]
enum Variant {
    Plain,
    Bounded,
    Injected,
    BoundedInjected,
}

impl Variant {
    fn name(self) -> &'static str {
        match self {
            Self::Plain => "cma_es",
            Self::Bounded => "bounded_cma_es",
            Self::Injected => "cma_inject",
            Self::BoundedInjected => "bounded_cma_inject",
        }
    }

    fn bounded(self) -> bool {
        matches!(self, Self::Bounded | Self::BoundedInjected)
    }
}

#[derive(Debug, PartialEq)]
struct Outcome {
    reached: bool,
    calls: u64,
    generations: u64,
    point: Vec<f64>,
    cost: f64,
    violation: f64,
    status: basin::TerminationCode,
}

fn reached(cost: f64, violation: f64, calls: u64) -> bool {
    cost.is_finite()
        && cost <= super::TARGET
        && violation == 0.0
        && calls <= super::MAX_EVALUATIONS as u64
}

fn solve<P, S, V, F>(
    problem: P,
    solver: S,
    start: V,
    workload: &Workload,
    bounded: bool,
) -> (u128, Outcome)
where
    S: Solver<P, PopulationProgress<V, F>>,
    S::Error: std::fmt::Debug,
    V: Clone + Index<usize, Output = F> + basin::VectorLen,
    F: Scalar,
{
    let executor =
        Executor::new(problem, solver, PopulationProgress::from_point(start))
            .max_iter((super::MAX_GENERATIONS - 1) as u64)
            .max_evaluations(
                EvaluationKind::Cost,
                super::MAX_EVALUATIONS as u64,
            )
            .target_objective(F::from_f64(super::TARGET).unwrap());
    let clock = Instant::now();
    let result = executor.run().unwrap();
    let elapsed = clock.elapsed().as_nanos();
    let calls = result.counts.cost_evals;
    assert_eq!(calls as usize, workload.calls());
    let (point, cost, violation) = result.state.best().map_or_else(
        || (Vec::new(), f64::INFINITY, f64::INFINITY),
        |(param, reported)| {
            let point: Vec<_> = (0..workload.weights.len())
                .map(|i| param[i].to_f64().unwrap())
                .collect();
            if point.iter().any(|x| !x.is_finite()) {
                return (Vec::new(), f64::INFINITY, f64::INFINITY);
            }
            let cost = objective(workload, &point, bounded);
            assert_eq!(F::from_f64(cost).unwrap(), reported);
            let violation = if bounded { box_violation(&point) } else { 0.0 };
            (point, cost, violation)
        },
    );
    (
        elapsed,
        Outcome {
            reached: reached(cost, violation, calls),
            calls,
            generations: result.iter() + 1,
            point,
            cost,
            violation,
            status: result.report.code(),
        },
    )
}

macro_rules! backend {
    ($name:ident, $f:ty, $v:ty, $m:ty, $build:expr) => {
        fn $name(
            scenario: Scenario,
            variant: Variant,
            seed: u64,
            period: u64,
        ) -> (u128, Outcome) {
            let workload = Workload::conditioned(
                scenario.case,
                scenario.n,
                scenario.condition,
            );
            let bounded = variant.bounded();
            let make: fn(&[f64]) -> $v = $build;
            let mut initial = vec![3.0; scenario.n];
            if bounded {
                initial[0] = -2.0;
            }
            let start = make(&initial);
            let problem = Objective::<$v, $f> {
                workload: &workload,
                bounded,
                param: PhantomData,
            };
            let lambda = scenario.lambda;
            let sigma = super::SIGMA as $f;
            match variant {
                Variant::Plain | Variant::Injected => {
                    let cma = CmaEs::<$v, $m, $f>::new(seed, sigma)
                        .with_lambda(lambda)
                        .with_eigen_update_period(period);
                    match variant {
                        Variant::Plain => {
                            solve(problem, cma, start, &workload, bounded)
                        }
                        _ => solve(
                            problem,
                            CmaInject::with_inner_solver(
                                cma,
                                NelderMead::new(),
                            )
                            .with_inner_max_iter(5),
                            start,
                            &workload,
                            bounded,
                        ),
                    }
                }
                Variant::Bounded | Variant::BoundedInjected => {
                    let mut hi = vec![5.0; scenario.n];
                    hi[0] = 0.0;
                    let bounds = BoxBounds::new(
                        Some(make(&vec![-5.0; scenario.n])),
                        Some(make(&hi)),
                    );
                    let problem = Constrained::new(
                        problem,
                        ConstraintSet::new().with_bounds(bounds),
                    );
                    let cma = BoundedCmaEs::<$v, $m, $f>::new(seed, sigma)
                        .with_lambda(lambda);
                    match variant {
                        Variant::Bounded => solve(
                            problem,
                            cma.with_eigen_update_period(period),
                            start,
                            &workload,
                            bounded,
                        ),
                        _ => solve(
                            problem,
                            BoundedCmaInject::with_inner_solver(
                                cma,
                                NelderMead::new().projected(),
                            )
                            .with_eigen_update_period(period)
                            .with_inner_max_iter(5),
                            start,
                            &workload,
                            bounded,
                        ),
                    }
                }
            }
        }
    };
}

backend!(vec_f32, f32, Vec<f32>, DenseMatrix<f32>, |x| x
    .iter()
    .map(|&v| v as f32)
    .collect());
backend!(vec_f64, f64, Vec<f64>, DenseMatrix<f64>, <[f64]>::to_vec);
backend!(nalgebra_f32, f32, BasinVector<f32>, BasinMatrix<f32>, |x| {
    BasinVector::from_iterator(x.len(), x.iter().map(|&v| v as f32))
});
backend!(
    nalgebra_f64,
    f64,
    BasinVector<f64>,
    BasinMatrix<f64>,
    BasinVector::from_column_slice
);
backend!(
    ndarray_f32,
    f32,
    ndarray::Array1<f32>,
    ndarray::Array2<f32>,
    |x| ndarray::Array1::from_iter(x.iter().map(|&v| v as f32))
);
backend!(
    ndarray_f64,
    f64,
    ndarray::Array1<f64>,
    ndarray::Array2<f64>,
    |x| ndarray::Array1::from_vec(x.to_vec())
);
backend!(faer_f32, f32, faer::Col<f32>, faer::Mat<f32>, |x| {
    faer::Col::from_fn(x.len(), |i| x[i] as f32)
});
backend!(faer_f64, f64, faer::Col<f64>, faer::Mat<f64>, |x| {
    faer::Col::from_fn(x.len(), |i| x[i])
});

type Run = fn(Scenario, Variant, u64, u64) -> (u128, Outcome);

const BACKENDS: [(&str, &str, Run); 8] = [
    ("vec", "f32", vec_f32),
    ("vec", "f64", vec_f64),
    ("nalgebra", "f32", nalgebra_f32),
    ("nalgebra", "f64", nalgebra_f64),
    ("ndarray", "f32", ndarray_f32),
    ("ndarray", "f64", ndarray_f64),
    ("faer", "f32", faer_f32),
    ("faer", "f64", faer_f64),
];

const VARIANTS: [Variant; 4] = [
    Variant::Plain,
    Variant::Bounded,
    Variant::Injected,
    Variant::BoundedInjected,
];

/// Compare policies on identical Basin workloads and write one CSV row per policy.
pub fn sweep(repetitions: usize) {
    faer::set_global_parallelism(faer::Par::Seq);
    assert!(repetitions > 0 && repetitions % 2 == 1);
    println!(
        "problem,dimension,population,condition,seed,solver,backend,precision,policy,period,reached,median_ns,min_ns,max_ns,objective_calls,generations,cost,box_violation,status"
    );
    let scenarios = [
        (Case::RotatedEllipsoid, 8, 8, 1e2),
        (Case::RotatedEllipsoid, 8, 8, 1e6),
        (Case::RotatedEllipsoid, 48, 8, 1e2),
        (Case::RotatedEllipsoid, 48, 16, 1e2),
        (Case::Ackley, 8, 8, 1.0),
    ];
    for (case, n, lambda, condition) in scenarios {
        let scenario = Scenario {
            case,
            n,
            lambda,
            condition,
        };
        let policies = [
            ("fixed_1", 1),
            ("fixed_3", 3),
            ("automatic", automatic_period(n, lambda)),
        ];
        for seed in [0, 1, 42] {
            for variant in VARIANTS {
                for (backend, precision, run) in BACKENDS {
                    let mut runs = Vec::new();
                    for (_, period) in policies {
                        if !runs.iter().any(|(seen, _, _)| *seen == period) {
                            let (_, warmup) =
                                run(scenario, variant, seed, period);
                            runs.push((
                                period,
                                warmup,
                                Vec::with_capacity(repetitions),
                            ));
                        }
                    }
                    for repetition in 0..repetitions {
                        let count = runs.len();
                        for offset in 0..count {
                            let (period, expected, samples) =
                                &mut runs[(repetition + offset) % count];
                            let (elapsed, outcome) =
                                run(scenario, variant, seed, *period);
                            assert_eq!(
                                outcome, *expected,
                                "numerical result changed between repetitions"
                            );
                            samples.push(elapsed);
                        }
                    }
                    for (_, _, samples) in &mut runs {
                        samples.sort_unstable();
                    }
                    for (policy, period) in policies {
                        let (_, result, samples) = runs
                            .iter()
                            .find(|(seen, _, _)| *seen == period)
                            .unwrap();
                        println!(
                            "{},{n},{lambda},{condition},{seed},{},{backend},{precision},{policy},{period},{},{},{},{},{},{},{:.16e},{:.16e},{:?}",
                            case.name(),
                            variant.name(),
                            result.reached,
                            samples[repetitions / 2],
                            samples[0],
                            samples[repetitions - 1],
                            result.calls,
                            result.generations,
                            result.cost,
                            result.violation,
                            result.status,
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_uses_dimension_and_population() {
        assert_eq!(automatic_period(8, 8), 1);
        assert_eq!(automatic_period(48, 8), 2);
        assert_eq!(automatic_period(48, 16), 1);
    }

    #[test]
    fn target_requires_finite_objective_feasibility_and_budget() {
        assert!(reached(
            super::super::TARGET,
            0.0,
            super::super::MAX_EVALUATIONS as u64
        ));
        assert!(!reached(0.0, 1e-12, 1));
        assert!(!reached(f64::NAN, 0.0, 1));
        assert!(!reached(0.0, 0.0, super::super::MAX_EVALUATIONS as u64 + 1));
    }
}
