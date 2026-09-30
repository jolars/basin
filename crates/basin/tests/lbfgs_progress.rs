//! Limited-memory models belong to the solver, including during exact continuation.

use basin::{
    BoxConstraints, CostFunction, Executor, FirstOrderState, Gradient, Lbfgs,
    State,
};
use std::convert::Infallible;

struct Quadratic {
    lower: Vec<f64>,
    upper: Vec<f64>,
}
impl Quadratic {
    fn new(n: usize) -> Self {
        Self {
            lower: vec![-5.0; n],
            upper: vec![5.0; n],
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
            .map(|(i, x)| (i + 1) as f64 * x * x)
            .sum())
    }
}
impl Gradient for Quadratic {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(x.iter()
            .enumerate()
            .map(|(i, x)| 2.0 * (i + 1) as f64 * x)
            .collect())
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

macro_rules! lifecycle {
    ($name:ident, $solver:expr) => {
        #[test]
        fn $name() {
            let make = || {
                Executor::new(
                    Quadratic::new(2),
                    $solver,
                    FirstOrderState::new(vec![3.0, 2.0]),
                )
                .require_evaluated_state()
            };
            let complete = make().max_iter(5).run_with_solver().unwrap();
            for split in [0, 1, 3] {
                let initial = make().max_iter(split).run_with_solver().unwrap();
                assert!(initial.solver.history_len() <= 2);
                assert_eq!(initial.state.counts(), &initial.counts);
                let continued = Executor::resume_from_checkpoint(
                    Quadratic::new(2),
                    initial.into_checkpoint(),
                )
                .require_evaluated_state()
                .max_iter(5)
                .run_with_solver()
                .unwrap();
                assert_eq!(continued.state, complete.state);
                assert_eq!(continued.counts, complete.counts);
                assert_eq!(
                    continued.solver.history_len(),
                    complete.solver.history_len()
                );
            }
            let mut state = complete.state;
            state
                .replace(vec![3.0, 2.0, 1.0], 20.0, vec![6.0, 8.0, 6.0])
                .unwrap();
            let fresh =
                Executor::new(Quadratic::new(3), complete.solver, state)
                    .require_evaluated_state()
                    .max_iter(0)
                    .run_with_solver()
                    .unwrap();
            assert_eq!(fresh.state.iter(), 0);
            assert_eq!(fresh.state.best_iter(), 0);
            assert_eq!(fresh.counts.cost_evals, 1);
            assert_eq!(fresh.counts.gradient_evals, 1);
            assert_eq!(fresh.solver.history_len(), 0);
            assert_eq!(fresh.solver.history_capacity(), 2);
            let continued = Executor::resume_from_checkpoint(
                Quadratic::new(3),
                fresh.into_checkpoint(),
            )
            .max_iter(3)
            .run_with_solver()
            .unwrap();
            let rebuilt = Executor::from_start(
                Quadratic::new(3),
                $solver,
                vec![3.0, 2.0, 1.0],
            )
            .max_iter(3)
            .run_with_solver()
            .unwrap();
            assert_eq!(continued.state, rebuilt.state);
            assert_eq!(continued.counts, rebuilt.counts);
        }
    };
}

lifecycle!(
    bounded_history_and_exact_continuation,
    Lbfgs::new().with_m_capacity(2)
);
lifecycle!(
    unbounded_history_and_exact_continuation,
    Lbfgs::new().unbounded().with_m_capacity(2)
);

// Deliberately not Clone: exact continuation retains components by ownership.
struct StatefulSearch {
    calls: u64,
}

impl basin::LineSearch<Quadratic, Vec<f64>> for StatefulSearch {
    type Error = Infallible;

    fn reset(&mut self) {
        self.calls = 0;
    }

    fn next(
        &mut self,
        _: &mut basin::Problem<Quadratic>,
        _: &Vec<f64>,
        _: f64,
        _: &Vec<f64>,
        _: &Vec<f64>,
    ) -> Result<f64, Infallible> {
        self.calls += 1;
        Ok(0.1 / self.calls as f64)
    }

    fn next_with_bounds(
        &mut self,
        problem: &mut basin::Problem<Quadratic>,
        x: &Vec<f64>,
        cost: f64,
        gradient: &Vec<f64>,
        direction: &Vec<f64>,
        bounds: basin::LineSearchBounds,
    ) -> Result<basin::LineSearchResult<Vec<f64>, f64>, Infallible> {
        let step = self.next(problem, x, cost, gradient, direction)?;
        Ok(basin::LineSearchResult::new(
            basin::LineSearchOutcome::Step(step.min(bounds.max())),
        ))
    }
}

lifecycle!(
    bounded_search_resets_fresh_history_and_retains_exact_history,
    Lbfgs::with_line_search(StatefulSearch { calls: 99 }).with_m_capacity(2)
);
lifecycle!(
    unbounded_search_resets_fresh_history_and_retains_exact_history,
    Lbfgs::with_line_search(StatefulSearch { calls: 99 })
        .unbounded()
        .with_m_capacity(2)
);

#[cfg(feature = "serde")]
fn round_trip<T: serde::Serialize + serde::de::DeserializeOwned>(
    value: &T,
) -> T {
    postcard::from_bytes(&postcard::to_allocvec(value).unwrap()).unwrap()
}

#[cfg(feature = "serde")]
#[test]
fn serialized_checkpoints_preserve_models_after_rollover() {
    macro_rules! check {
        ($solver:expr) => {{
            let make = || {
                Executor::from_start(Quadratic::new(8), $solver, vec![3.0; 8])
            };
            let expected = make().max_iter(7).run_with_solver().unwrap();
            let first = make().max_iter(4).run_with_solver().unwrap();
            assert_eq!(first.solver.history_len(), 2);
            let checkpoint = round_trip(&first.into_checkpoint());
            let resumed =
                Executor::resume_from_checkpoint(Quadratic::new(8), checkpoint)
                    .require_evaluated_state()
                    .max_iter(7)
                    .run_with_solver()
                    .unwrap();
            assert_eq!(resumed.state, expected.state);
            assert_eq!(resumed.counts, expected.counts);
            assert_eq!(
                postcard::to_allocvec(&resumed.solver).unwrap(),
                postcard::to_allocvec(&expected.solver).unwrap()
            );
        }};
    }
    check!(Lbfgs::new().with_m_capacity(2));
    check!(Lbfgs::new().unbounded().with_m_capacity(2));
}

#[test]
fn custom_contiguous_backend_needs_no_matrix_capabilities() {
    use basin::solver::lbfgs::{AsFloatSlice, AsFloatSliceMut};
    use basin::{Dot, ScaledAdd, VectorLen};

    #[derive(Clone)]
    struct Vector(Vec<f64>);
    impl AsFloatSlice for Vector {
        fn as_float_slice(&self) -> &[f64] {
            &self.0
        }
    }
    impl AsFloatSliceMut for Vector {
        fn as_float_slice_mut(&mut self) -> &mut [f64] {
            &mut self.0
        }
    }
    impl VectorLen for Vector {
        fn vec_len(&self) -> usize {
            self.0.len()
        }
    }
    impl Dot for Vector {
        fn dot(&self, other: &Self) -> f64 {
            self.0.dot(&other.0)
        }
    }
    impl ScaledAdd for Vector {
        fn scaled_add(&mut self, scale: f64, other: &Self) {
            self.0.scaled_add(scale, &other.0);
        }
    }
    struct Sphere {
        lower: Vector,
        upper: Vector,
    }
    impl CostFunction for Sphere {
        type Param = Vector;
        type Output = f64;
        type Error = Infallible;
        fn cost(&self, x: &Vector) -> Result<f64, Infallible> {
            Ok(x.dot(x))
        }
    }
    impl Gradient for Sphere {
        type Gradient = Vector;
        fn gradient(&self, x: &Vector) -> Result<Vector, Infallible> {
            Ok(Vector(x.0.iter().map(|x| 2.0 * x).collect()))
        }
    }
    impl BoxConstraints for Sphere {
        fn lower(&self) -> &Vector {
            &self.lower
        }
        fn upper(&self) -> &Vector {
            &self.upper
        }
    }
    let problem = || Sphere {
        lower: Vector(vec![-5.0; 2]),
        upper: Vector(vec![5.0; 2]),
    };
    let bounded =
        Executor::from_start(problem(), Lbfgs::new(), Vector(vec![3.0, 2.0]))
            .require_evaluated_state()
            .max_iter(10)
            .run()
            .unwrap();
    assert!(bounded.cost() < 1e-12);
    let unbounded = Executor::from_start(
        problem(),
        Lbfgs::new().unbounded(),
        Vector(vec![3.0, 2.0]),
    )
    .require_evaluated_state()
    .max_iter(10)
    .run()
    .unwrap();
    assert!(unbounded.cost() < 1e-12);
}
