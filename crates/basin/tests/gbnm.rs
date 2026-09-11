//! Public contract tests for Luersen and Le Riche's Globalized Bounded
//! Nelder-Mead algorithm.

use std::convert::Infallible;

use basin::{
    BoxConstraints, CostFunction, Executor, Gbnm, GbnmState, State,
    TerminationReason,
};

#[derive(Clone)]
struct BoundedRosenbrock {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl BoundedRosenbrock {
    fn new(lower: Vec<f64>, upper: Vec<f64>) -> Self {
        Self { lower, upper }
    }
}

impl CostFunction for BoundedRosenbrock {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
        Ok((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2))
    }
}

impl BoxConstraints for BoundedRosenbrock {
    fn lower(&self) -> &Self::Param {
        &self.lower
    }

    fn upper(&self) -> &Self::Param {
        &self.upper
    }
}

#[derive(Clone)]
struct FlatBox {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl CostFunction for FlatBox {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, _x: &Self::Param) -> Result<Self::Output, Self::Error> {
        Ok(1.0)
    }
}

impl BoxConstraints for FlatBox {
    fn lower(&self) -> &Self::Param {
        &self.lower
    }

    fn upper(&self) -> &Self::Param {
        &self.upper
    }
}

#[test]
fn starts_from_the_callers_point_and_builds_a_feasible_regular_simplex() {
    let problem = BoundedRosenbrock::new(vec![0.0, 0.0], vec![20.0, 20.0]);
    let result =
        Executor::new(problem, Gbnm::new(42), GbnmState::new(vec![10.0, 10.0]))
            .max_iter(0)
            .run()
            .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxIter);
    assert_eq!(result.state.search_starts(), &[vec![10.0, 10.0]]);
    assert_eq!(result.state.restart_count(), 0);
    assert_eq!(result.state.vertices().len(), 3);
    assert_eq!(result.state.costs().len(), 3);
    assert_eq!(result.state.cost_evals(), 3);
    for vertex in result.state.vertices() {
        assert!(vertex.iter().all(|&x| (0.0..=20.0).contains(&x)));
    }
}

#[test]
fn upper_bound_start_explores_the_feasible_inward_direction() {
    #[derive(Clone)]
    struct Sphere(FlatBox);

    impl CostFunction for Sphere {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = Infallible;

        fn cost(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
            Ok(x.iter().map(|value| value * value).sum())
        }
    }

    impl BoxConstraints for Sphere {
        fn lower(&self) -> &Self::Param {
            &self.0.lower
        }

        fn upper(&self) -> &Self::Param {
            &self.0.upper
        }
    }

    let initial_problem = Sphere(FlatBox {
        lower: vec![-1.0, -1.0],
        upper: vec![1.0, 1.0],
    });
    let initial = Executor::new(
        initial_problem,
        Gbnm::new(5),
        GbnmState::new(vec![1.0, 0.0]),
    )
    .max_iter(0)
    .run()
    .unwrap();
    assert!(
        initial
            .state
            .vertices()
            .iter()
            .any(|vertex| vertex[0] < 1.0)
    );

    let problem = Sphere(FlatBox {
        lower: vec![-1.0, -1.0],
        upper: vec![1.0, 1.0],
    });
    let result =
        Executor::new(problem, Gbnm::new(5), GbnmState::new(vec![1.0, 0.0]))
            .max_iter(2_000)
            .max_cost_evals(500)
            .run()
            .unwrap();

    assert!(result.best_cost() < 1.0e-8, "cost = {}", result.best_cost());
}

#[test]
fn f32_extreme_box_starts_with_a_working_local_simplex() {
    #[derive(Clone)]
    struct ScaledSphere {
        lower: Vec<f32>,
        upper: Vec<f32>,
    }

    impl CostFunction for ScaledSphere {
        type Param = Vec<f32>;
        type Output = f32;
        type Error = Infallible;

        fn cost(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
            Ok(x.iter().map(|value| (value / 3.0e38_f32).powi(2)).sum())
        }
    }

    impl BoxConstraints for ScaledSphere {
        fn lower(&self) -> &Self::Param {
            &self.lower
        }

        fn upper(&self) -> &Self::Param {
            &self.upper
        }
    }

    let problem = ScaledSphere {
        lower: vec![-3.0e38, -3.0e38],
        upper: vec![3.0e38, 3.0e38],
    };
    let result = Executor::new(
        problem,
        Gbnm::<f32>::new(17),
        GbnmState::new(vec![0.0_f32, 0.0]),
    )
    .max_iter(1)
    .run()
    .unwrap();

    assert_eq!(result.state.restart_count(), 0);
    assert!(
        result
            .state
            .vertices()
            .iter()
            .flatten()
            .all(|value| value.is_finite())
    );
    assert!(result.state.costs().iter().all(|cost| cost.is_finite()));
}

#[test]
fn f32_extreme_box_supports_probabilistic_restarts() {
    #[derive(Clone)]
    struct FlatF32Box {
        lower: Vec<f32>,
        upper: Vec<f32>,
    }

    impl CostFunction for FlatF32Box {
        type Param = Vec<f32>;
        type Output = f32;
        type Error = Infallible;

        fn cost(&self, _x: &Self::Param) -> Result<Self::Output, Self::Error> {
            Ok(1.0)
        }
    }

    impl BoxConstraints for FlatF32Box {
        fn lower(&self) -> &Self::Param {
            &self.lower
        }

        fn upper(&self) -> &Self::Param {
            &self.upper
        }
    }

    let problem = FlatF32Box {
        lower: vec![-3.0e38, -3.0e38],
        upper: vec![3.0e38, 3.0e38],
    };
    let result = Executor::new(
        problem,
        Gbnm::<f32>::new(23),
        GbnmState::new(vec![0.0_f32, 0.0]),
    )
    .max_iter(1)
    .run()
    .unwrap();

    assert_eq!(result.state.restart_count(), 1);
    assert_eq!(result.state.search_starts().len(), 2);
    assert!(
        result
            .state
            .search_starts()
            .iter()
            .flatten()
            .all(
                |value| value.is_finite() && (-3.0e38..=3.0e38).contains(value)
            )
    );
}

#[test]
fn paper_rosenbrock_start_reaches_the_global_minimum() {
    let problem = BoundedRosenbrock::new(vec![0.0, 0.0], vec![20.0, 20.0]);
    let result =
        Executor::new(problem, Gbnm::new(7), GbnmState::new(vec![10.0, 10.0]))
            .max_iter(20_000)
            .max_cost_evals(5_000)
            .run()
            .unwrap();

    assert!(result.best_cost() < 1e-8, "cost = {}", result.best_cost());
    assert!((result.best_param()[0] - 1.0).abs() < 1e-3);
    assert!((result.best_param()[1] - 1.0).abs() < 1e-3);
    assert!(!result.state.local_optima().is_empty());
}

#[test]
fn seeded_restart_history_is_reproducible() {
    fn run(seed: u64) -> (Vec<Vec<f64>>, u64) {
        let problem = FlatBox {
            lower: vec![-5.0, -5.0],
            upper: vec![5.0, 5.0],
        };
        let result = Executor::new(
            problem,
            Gbnm::new(seed),
            GbnmState::new(vec![0.0, 0.0]),
        )
        .max_iter(12)
        .run()
        .unwrap();
        (
            result.state.search_starts().to_vec(),
            result.state.cost_evals(),
        )
    }

    let first = run(91);
    assert_eq!(first, run(91));
    assert_ne!(first.0, run(92).0);
    assert_eq!(first.1, 3 * 13);
}

#[test]
fn from_start_uses_the_gbnm_state() {
    let problem = BoundedRosenbrock::new(vec![0.0, 0.0], vec![20.0, 20.0]);
    let result = Executor::from_start(problem, Gbnm::new(11), vec![10.0, 10.0])
        .max_iter(0)
        .run()
        .unwrap();

    assert_eq!(result.state.search_starts(), &[vec![10.0, 10.0]]);
}

#[test]
fn pinned_coordinates_are_excluded_from_the_local_simplex() {
    let problem = BoundedRosenbrock::new(vec![-5.0, 2.0], vec![5.0, 2.0]);
    let result =
        Executor::new(problem, Gbnm::new(3), GbnmState::new(vec![4.0, -100.0]))
            .max_iter(0)
            .run()
            .unwrap();

    assert_eq!(result.state.vertices().len(), 2);
    assert!(
        result
            .state
            .vertices()
            .iter()
            .all(|vertex| vertex[1] == 2.0)
    );
}

#[test]
#[should_panic(expected = "GBNM requires at least one free coordinate")]
fn an_all_pinned_box_is_rejected() {
    let problem = FlatBox {
        lower: vec![1.0, 2.0],
        upper: vec![1.0, 2.0],
    };
    let _ =
        Executor::new(problem, Gbnm::new(1), GbnmState::new(vec![1.0, 2.0]))
            .max_iter(0)
            .run();
}

#[test]
#[should_panic(expected = "GBNM requires lower <= upper")]
fn reversed_bounds_are_rejected() {
    let problem = FlatBox {
        lower: vec![1.0, -1.0],
        upper: vec![-1.0, 1.0],
    };
    let _ =
        Executor::new(problem, Gbnm::new(1), GbnmState::new(vec![0.0, 0.0]))
            .max_iter(0)
            .run();
}

#[test]
#[should_panic(expected = "GBNM requires at least one restart candidate")]
fn zero_restart_candidates_are_rejected() {
    let _ = Gbnm::<f64>::new(1).with_restart_candidates(0);
}

#[test]
fn nan_costs_sort_after_finite_vertices() {
    struct NanAtOrigin(FlatBox);

    impl CostFunction for NanAtOrigin {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = Infallible;

        fn cost(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
            if x.iter().all(|value| *value == 0.0) {
                Ok(f64::NAN)
            } else {
                Ok(x.iter().map(|value| value * value).sum())
            }
        }
    }

    impl BoxConstraints for NanAtOrigin {
        fn lower(&self) -> &Self::Param {
            &self.0.lower
        }

        fn upper(&self) -> &Self::Param {
            &self.0.upper
        }
    }

    let result = Executor::new(
        NanAtOrigin(FlatBox {
            lower: vec![-1.0, -1.0],
            upper: vec![1.0, 1.0],
        }),
        Gbnm::new(2),
        GbnmState::new(vec![0.0, 0.0]),
    )
    .max_iter(0)
    .run()
    .unwrap();

    assert!(result.cost().is_finite());
    assert!(result.state.costs().last().unwrap().is_nan());
}

#[test]
fn probabilistic_restarts_escape_the_initial_rastrigin_basin() {
    struct Rastrigin(FlatBox);

    impl CostFunction for Rastrigin {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = Infallible;

        fn cost(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
            Ok(10.0 * x.len() as f64
                + x.iter()
                    .map(|value| {
                        value * value
                            - 10.0 * (std::f64::consts::TAU * value).cos()
                    })
                    .sum::<f64>())
        }
    }

    impl BoxConstraints for Rastrigin {
        fn lower(&self) -> &Self::Param {
            &self.0.lower
        }

        fn upper(&self) -> &Self::Param {
            &self.0.upper
        }
    }

    let problem = Rastrigin(FlatBox {
        lower: vec![-5.12, -5.12],
        upper: vec![5.12, 5.12],
    });
    let result =
        Executor::new(problem, Gbnm::new(7), GbnmState::new(vec![3.0, 3.0]))
            .max_iter(20_000)
            .max_cost_evals(5_000)
            .run()
            .unwrap();

    assert!(result.best_cost() < 1.0, "cost = {}", result.best_cost());
    assert!(result.state.restart_count() > 1);
}
