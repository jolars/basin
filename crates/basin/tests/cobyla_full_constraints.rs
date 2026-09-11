//! Public full-form constraint coverage for COBYLA and every supported backend.

use std::{cell::Cell, convert::Infallible, ops::Index};

use basin::core::math::{MatVec, VectorLen};
use basin::{
    Cobyla, CobylaState, CostFunction, DenseMatrix, Executor,
    FoldedConstraints, NonlinearConstraints, NonlinearInequalityConstraints,
    Problem, Solver, State, TerminationReason,
};

#[path = "support/backend_aliases.rs"]
mod backend_aliases;

struct Mixed<V, M> {
    inequality: M,
    inequality_rhs: V,
    equality: M,
    equality_rhs: V,
    lower: V,
    upper: V,
    vector: fn(Vec<f64>) -> V,
}

impl<V: Index<usize, Output = f64>, M> CostFunction for Mixed<V, M> {
    type Param = V;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &V) -> Result<f64, Infallible> {
        Ok((0..5).map(|i| (x[i] - 2.0).powi(2)).sum())
    }
}

impl<V: Index<usize, Output = f64>, M> NonlinearConstraints for Mixed<V, M> {
    type Matrix = M;

    fn nonlinear_constraints(&self, x: &V) -> Result<V, Infallible> {
        Ok((self.vector)(vec![x[0] * x[0] - 1.0]))
    }

    fn num_nonlinear_constraints(&self) -> usize {
        1
    }

    fn inequalities(&self) -> Option<(&M, &V)> {
        Some((&self.inequality, &self.inequality_rhs))
    }

    fn equalities(&self) -> Option<(&M, &V)> {
        Some((&self.equality, &self.equality_rhs))
    }

    fn lower(&self) -> Option<&V> {
        Some(&self.lower)
    }

    fn upper(&self) -> Option<&V> {
        Some(&self.upper)
    }
}

fn mixed<V, M>(
    vector: fn(Vec<f64>) -> V,
    matrix: impl Fn(&[f64]) -> M,
) -> Mixed<V, M> {
    Mixed {
        inequality: matrix(&[0.0, 1.0, 0.0, 0.0, 0.0]),
        inequality_rhs: vector(vec![0.5]),
        equality: matrix(&[0.0, 0.0, 1.0, 0.0, 0.0]),
        equality_rhs: vector(vec![0.25]),
        // Every non-finite entry denotes an absent bound, including NaN
        // and an infinity with the opposite sign from the usual sentinel.
        lower: vector(vec![
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NAN,
            f64::NEG_INFINITY,
            3.0,
        ]),
        upper: vector(vec![
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            0.75,
            f64::INFINITY,
        ]),
        vector,
    }
}

fn mixed_vec() -> Mixed<Vec<f64>, DenseMatrix<f64>> {
    mixed(
        |values| values,
        |row| DenseMatrix::from_row_slice(1, 5, row),
    )
}

fn solver() -> Cobyla {
    Cobyla::new()
        .with_initial_radius(0.5)
        .with_final_radius(1e-7)
}

fn assert_mixed_solution<V, M>(problem: Mixed<V, M>)
where
    V: Clone
        + VectorLen
        + Index<usize, Output = f64>
        + std::ops::IndexMut<usize, Output = f64>,
    M: MatVec<V>,
{
    let start = (problem.vector)(vec![0.0; 5]);
    let result =
        Executor::from_start(FoldedConstraints::new(problem), solver(), start)
            .max_cost_evals(3000)
            .run()
            .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    let x = result.best_param();
    let expected = [1.0, 0.5, 0.25, 0.75, 3.0];
    for (i, target) in expected.into_iter().enumerate() {
        assert!((x[i] - target).abs() < 1e-3, "x[{i}] = {}", x[i]);
    }
    assert!(x[0] * x[0] <= 1.0 + 1e-6);
    assert!(x[1] <= 0.5 + 1e-6);
    assert!((x[2] - 0.25).abs() < 1e-6);
    assert!(x[3] <= 0.75 + 1e-6);
    assert!(x[4] >= 3.0 - 1e-6);
    assert!((result.best_cost() - 8.875).abs() < 1e-3);
    assert_eq!(
        result.best_cost(),
        (0..5).map(|i| (x[i] - 2.0).powi(2)).sum::<f64>()
    );
}

#[test]
fn mixed_constraints_vec() {
    assert_mixed_solution(mixed_vec());
}

#[cfg(feature = "nalgebra_all")]
#[test]
fn mixed_constraints_nalgebra() {
    use backend_aliases::nalgebra::{DMatrix, DVector};

    assert_mixed_solution(mixed(DVector::from_vec, |row| {
        DMatrix::from_row_slice(1, 5, row)
    }));
}

#[cfg(feature = "ndarray_all")]
#[test]
fn mixed_constraints_ndarray() {
    use backend_aliases::ndarray::{Array1, Array2};

    assert_mixed_solution(mixed(Array1::from_vec, |row| {
        Array2::from_shape_vec((1, 5), row.to_vec()).unwrap()
    }));
}

#[cfg(feature = "faer_all")]
#[test]
fn mixed_constraints_faer() {
    use backend_aliases::faer::{Col, Mat};

    assert_mixed_solution(mixed(
        |values| Col::from_fn(values.len(), |i| values[i]),
        |row| Mat::from_fn(1, 5, |_, j| row[j]),
    ));
}

impl NonlinearInequalityConstraints for Mixed<Vec<f64>, DenseMatrix<f64>> {
    fn constraints(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        self.nonlinear_constraints(x)
    }

    fn num_constraints(&self) -> usize {
        1
    }
}

#[test]
fn explicit_adapter_selects_all_blocks_for_a_problem_with_both_traits() {
    let legacy = Executor::from_start(mixed_vec(), solver(), vec![0.0; 5])
        .max_cost_evals(3000)
        .run()
        .unwrap();
    assert!((legacy.best_cost() - 1.0).abs() < 1e-3);
    for i in 1..5 {
        assert!((legacy.best_param()[i] - 2.0).abs() < 1e-3);
    }

    assert_mixed_solution(mixed_vec());
}

struct Expanded(Mixed<Vec<f64>, DenseMatrix<f64>>);

impl CostFunction for Expanded {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        self.0.cost(x)
    }
}

impl NonlinearInequalityConstraints for Expanded {
    fn constraints(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(vec![
            3.0 - x[4],
            x[3] - 0.75,
            0.25 - x[2],
            x[2] - 0.25,
            x[1] - 0.5,
            x[0] * x[0] - 1.0,
        ])
    }

    fn num_constraints(&self) -> usize {
        6
    }
}

#[test]
fn agrees_with_hand_expanded_legacy_constraints() {
    let folded = Executor::from_start(
        FoldedConstraints::new(mixed_vec()),
        solver(),
        vec![0.0; 5],
    )
    .max_cost_evals(3000)
    .run()
    .unwrap();
    let expanded =
        Executor::from_start(Expanded(mixed_vec()), solver(), vec![0.0; 5])
            .max_cost_evals(3000)
            .run()
            .unwrap();

    assert_eq!(folded.reason, expanded.reason);
    assert_eq!(folded.cost_evals(), expanded.cost_evals());
    assert_eq!(folded.best_cost(), expanded.best_cost());
    assert_eq!(folded.best_param(), expanded.best_param());
}

#[derive(Debug, PartialEq)]
struct CallbackError {
    constraint: bool,
    call: usize,
}

struct FallibleDisk<'a> {
    costs: &'a Cell<usize>,
    constraints: &'a Cell<usize>,
    fail_constraint: bool,
    fail_at: usize,
}

impl CostFunction for FallibleDisk<'_> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = CallbackError;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        let call = self.costs.get();
        self.costs.set(call + 1);
        assert_eq!(self.constraints.get(), call);
        if !self.fail_constraint && call == self.fail_at {
            Err(CallbackError {
                constraint: false,
                call,
            })
        } else {
            Ok(x[0] * x[1])
        }
    }
}

impl NonlinearConstraints for FallibleDisk<'_> {
    type Matrix = DenseMatrix<f64>;

    fn nonlinear_constraints(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, Self::Error> {
        let call = self.constraints.get();
        self.constraints.set(call + 1);
        assert_eq!(self.costs.get(), call + 1);
        if self.fail_constraint && call == self.fail_at {
            Err(CallbackError {
                constraint: true,
                call,
            })
        } else {
            Ok(vec![x[0] * x[0] + x[1] * x[1] - 1.0])
        }
    }

    fn num_nonlinear_constraints(&self) -> usize {
        1
    }
}

#[test]
fn typed_callback_errors_abort_initialization_and_later_iterations() {
    for fail_constraint in [false, true] {
        for fail_at in [0, 1, 2, 3, 8, 15] {
            let costs = Cell::new(0);
            let constraints = Cell::new(0);
            let result = Executor::from_start(
                FoldedConstraints::new(FallibleDisk {
                    costs: &costs,
                    constraints: &constraints,
                    fail_constraint,
                    fail_at,
                }),
                solver(),
                vec![1.0, 1.0],
            )
            .max_iter(1000)
            .run();

            assert!(matches!(result, Err(CallbackError { constraint, call })
                if constraint == fail_constraint && call == fail_at));
            assert_eq!(costs.get(), fail_at + 1);
            assert_eq!(
                constraints.get(),
                fail_at + usize::from(fail_constraint)
            );
        }
    }
}

#[test]
fn counts_objective_evaluations_once_and_preserves_callback_order() {
    let costs = Cell::new(0);
    let constraints = Cell::new(0);
    let result = Executor::from_start(
        FoldedConstraints::new(FallibleDisk {
            costs: &costs,
            constraints: &constraints,
            fail_constraint: false,
            fail_at: usize::MAX,
        }),
        solver(),
        vec![1.0, 1.0],
    )
    .max_cost_evals(15)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxCostEvals);
    assert_eq!(result.cost_evals(), costs.get() as u64);
    assert_eq!(costs.get(), constraints.get());
    assert!(costs.get() >= 15);
}

#[test]
fn constraint_count_changes_panic_during_initialization_and_later_steps() {
    struct GrowingDisk {
        costs: Cell<usize>,
        grow_at: usize,
    }

    impl CostFunction for GrowingDisk {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = Infallible;

        fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
            self.costs.set(self.costs.get() + 1);
            Ok(x[0] * x[1])
        }
    }

    impl NonlinearConstraints for GrowingDisk {
        type Matrix = DenseMatrix<f64>;

        fn nonlinear_constraints(
            &self,
            x: &Vec<f64>,
        ) -> Result<Vec<f64>, Infallible> {
            let mut values = vec![x[0] * x[0] + x[1] * x[1] - 1.0];
            values.resize(self.num_nonlinear_constraints(), 0.0);
            Ok(values)
        }

        fn num_nonlinear_constraints(&self) -> usize {
            1 + usize::from(self.costs.get() >= self.grow_at)
        }
    }

    for grow_at in [2, 8] {
        let panic = std::panic::catch_unwind(|| {
            Executor::from_start(
                FoldedConstraints::new(GrowingDisk {
                    costs: Cell::new(0),
                    grow_at,
                }),
                solver(),
                vec![1.0, 1.0],
            )
            .max_iter(1000)
            .run()
            .unwrap();
        })
        .expect_err("changing the constraint count must panic");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .expect("the assertion must provide a panic message");
        assert!(
            message.contains(
                "Cobyla constraint count must remain fixed during a solve"
            ),
            "unexpected panic on callback {grow_at}: {message}"
        );
    }
}

#[test]
fn reused_solver_handles_changing_dimensions_and_empty_constraint_blocks() {
    struct Sphere {
        constraints: usize,
        lower: Option<Vec<f64>>,
    }

    impl CostFunction for Sphere {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = Infallible;

        fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
            Ok(x.iter().map(|v| v * v).sum())
        }
    }

    impl NonlinearConstraints for Sphere {
        type Matrix = DenseMatrix<f64>;

        fn nonlinear_constraints(
            &self,
            x: &Vec<f64>,
        ) -> Result<Vec<f64>, Infallible> {
            Ok((0..self.constraints)
                .map(|i| x[i % x.len()] - 4.0)
                .collect())
        }

        fn num_nonlinear_constraints(&self) -> usize {
            self.constraints
        }

        fn lower(&self) -> Option<&Vec<f64>> {
            self.lower.as_ref()
        }
    }

    let mut solver = solver();
    for (n, m, bounds) in
        [(2, 3, true), (5, 0, false), (1, 0, true), (3, 1, false)]
    {
        let mut problem = Problem::new(FoldedConstraints::new(Sphere {
            constraints: m,
            lower: bounds.then(|| vec![-3.0; n]),
        }));
        let mut state = solver
            .init(&mut problem, CobylaState::new(vec![1.0; n]))
            .unwrap();
        state.update_best();
        let snapshot = state.best_param().clone();
        let snapshot_cost = state.best_cost();
        let (mut state, _) = solver.next_iter(&mut problem, state).unwrap();

        assert_eq!(state.param().len(), n);
        assert_eq!(state.best_param(), &snapshot);
        assert_eq!(state.best_cost(), snapshot_cost);
        assert_eq!(state.cost(), state.param().iter().map(|v| v * v).sum());
        state.update_best();
        assert_eq!(state.best_param(), state.param());
        assert_eq!(state.best_cost(), state.cost());
    }
}
