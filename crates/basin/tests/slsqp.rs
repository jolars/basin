use basin::{
    ConstraintJacobian, CostFunction, DenseMatrix, Executor, Gradient,
    GradientState, NonlinearConstraints, RawEvaluationState, Slsqp, SlsqpState,
    State, TerminationReason,
};
use std::convert::Infallible;

#[derive(Clone)]
struct EqualityQuadratic;

impl CostFunction for EqualityQuadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok((x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2))
    }
}
impl Gradient for EqualityQuadratic {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(vec![2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)])
    }
}
impl NonlinearConstraints for EqualityQuadratic {
    type Matrix = DenseMatrix;
    fn nonlinear_constraints(
        &self,
        _: &Vec<f64>,
    ) -> Result<Vec<f64>, Infallible> {
        Ok(vec![])
    }
    fn num_nonlinear_constraints(&self) -> usize {
        0
    }
    fn num_nonlinear_equalities(&self) -> usize {
        1
    }
    fn nonlinear_equalities(
        &self,
        x: &Vec<f64>,
    ) -> Result<Option<Vec<f64>>, Infallible> {
        Ok(Some(vec![x[0] + x[1] - 1.0]))
    }
}
impl ConstraintJacobian for EqualityQuadratic {
    fn constraint_jacobian(
        &self,
        _: &Vec<f64>,
    ) -> Result<DenseMatrix, Infallible> {
        Ok(DenseMatrix::from_row_slice(1, 2, &[1.0, 1.0]))
    }
}

#[test]
fn analytic_equality_solution_and_records() {
    let result = Executor::new(
        EqualityQuadratic,
        Slsqp::new().with_absolute_accuracy_tolerance(1e-10),
        SlsqpState::new(vec![3.0, -1.0]),
    )
    .require_evaluated_state()
    .max_iter(100)
    .run()
    .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverConverged);
    let x = result.state.param();
    assert!(x[0].abs() < 1e-7, "{x:?}");
    assert!((x[1] - 1.0).abs() < 1e-7, "{x:?}");
    assert!((result.state.cost() - 2.0).abs() < 1e-10);
    assert_eq!(
        result.state.gradient().unwrap(),
        &EqualityQuadratic.gradient(x).unwrap()
    );
    assert!(result.state.constraint_violation().unwrap() < 1e-10);
    assert!(result.state.stationarity().unwrap() < 1e-6);
    assert!(result.state.raw_counts().residual_evals > 0);
    assert!(result.state.raw_counts().jacobian_evals > 0);
}

#[test]
fn optimal_seed_stops_at_iteration_zero() {
    let result = Executor::new(
        EqualityQuadratic,
        Slsqp::new(),
        SlsqpState::new(vec![0.0, 1.0]),
    )
    .max_iter(10)
    .run()
    .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert_eq!(result.state.iter(), 0);
    assert_eq!(result.state.raw_counts().cost_evals, 1);
    assert_eq!(result.state.raw_counts().gradient_evals, 1);
}

use basin::{DenseMatrixFromFn, MatrixIndex, Scalar, VectorIndex, VectorLen};
#[path = "support/backend_aliases.rs"]
mod backend_aliases;

#[derive(Clone)]
struct Hs71<V, F> {
    make: fn(&[F]) -> V,
    lower: V,
    upper: V,
}
fn num<F: Scalar>(v: f64) -> F {
    F::from_f64(v).unwrap()
}
impl<V, F: Scalar> Hs71<V, F> {
    fn new(make: fn(&[F]) -> V) -> Self {
        Self {
            make,
            lower: make(&[F::one(); 4]),
            upper: make(&[num(5.); 4]),
        }
    }
}
impl<V: VectorIndex<F>, F: Scalar> CostFunction for Hs71<V, F> {
    type Param = V;
    type Output = F;
    type Error = Infallible;
    fn cost(&self, x: &V) -> Result<F, Infallible> {
        let [a, b, c, d] = std::array::from_fn(|i| x.get_scalar(i));
        Ok(a * d * (a + b + c) + c)
    }
}
impl<V: VectorIndex<F>, F: Scalar> Gradient for Hs71<V, F> {
    type Gradient = V;
    fn gradient(&self, x: &V) -> Result<V, Infallible> {
        let [a, b, c, d] = std::array::from_fn(|i| x.get_scalar(i));
        Ok((self.make)(&[
            d * (num::<F>(2.) * a + b + c),
            a * d,
            a * d + F::one(),
            a * (a + b + c),
        ]))
    }
}
impl<V: VectorIndex<F> + DenseMatrixFromFn<F>, F: Scalar> NonlinearConstraints
    for Hs71<V, F>
{
    type Matrix = V::Matrix;
    fn lower(&self) -> Option<&V> {
        Some(&self.lower)
    }
    fn upper(&self) -> Option<&V> {
        Some(&self.upper)
    }
    fn num_nonlinear_constraints(&self) -> usize {
        1
    }
    fn nonlinear_constraints(&self, x: &V) -> Result<V, Infallible> {
        let product =
            (0..4).map(|i| x.get_scalar(i)).fold(F::one(), |a, b| a * b);
        Ok((self.make)(&[num::<F>(25.) - product]))
    }
    fn num_nonlinear_equalities(&self) -> usize {
        1
    }
    fn nonlinear_equalities(&self, x: &V) -> Result<Option<V>, Infallible> {
        let squares = (0..4).map(|i| x.get_scalar(i).powi(2)).sum::<F>();
        Ok(Some((self.make)(&[squares - num::<F>(40.)])))
    }
}
impl<V: VectorIndex<F> + DenseMatrixFromFn<F>, F: Scalar> ConstraintJacobian
    for Hs71<V, F>
{
    fn constraint_jacobian(&self, x: &V) -> Result<V::Matrix, Infallible> {
        Ok(V::dense_from_fn(2, 4, |i, j| {
            if i == 0 {
                num::<F>(2.) * x.get_scalar(j)
            } else {
                -(0..4)
                    .filter(|&k| k != j)
                    .map(|k| x.get_scalar(k))
                    .fold(F::one(), |a, b| a * b)
            }
        }))
    }
}

fn check_hs71<V, F>(make: fn(&[F]) -> V)
where
    V: Clone + VectorIndex<F> + VectorLen + DenseMatrixFromFn<F>,
    V::Matrix: MatrixIndex<F>,
    F: Scalar,
{
    let p = Hs71::new(make);
    let accuracy = if F::epsilon() > num(1e-10) {
        1e-3
    } else {
        1e-10
    };
    let result = Executor::new(
        p,
        Slsqp::new().with_absolute_accuracy_tolerance(num::<F>(accuracy)),
        SlsqpState::new(make(&[num(1.), num(5.), num(5.), num(1.)])),
    )
    .max_iter(100)
    .run_with_solver()
    .unwrap();
    assert_eq!(
        result.reason,
        TerminationReason::SolverConverged,
        "failure={:?}, x={:?}, violation={:?}, stationarity={:?}",
        result.state.failure(),
        (0..4)
            .map(|i| result.state.param().get_scalar(i).to_f64())
            .collect::<Vec<_>>(),
        result.state.constraint_violation().map(|v| v.to_f64()),
        result.state.stationarity().map(|v| v.to_f64())
    );
    for (i, target) in [
        1.,
        4.742999642848332,
        3.8211499768953514,
        1.3794082941785437,
    ]
    .into_iter()
    .enumerate()
    {
        assert!(
            (result.state.param().get_scalar(i) - num::<F>(target)).abs()
                < num(2e-3)
        );
    }
    assert!(result.state.constraint_violation().unwrap() < num(accuracy * 2.));
    assert!(result.state.stationarity().unwrap() < num(2e-3));
    assert!(result.state.complementarity().unwrap() < num(accuracy * 2.));
    assert!(result.solver.inequality_multipliers().unwrap()[0] > F::zero());
}
#[test]
fn hs71_vec_f64_and_f32() {
    check_hs71::<_, f64>(|v| v.to_vec());
    check_hs71::<_, f32>(|v| v.to_vec());
}
#[cfg(feature = "nalgebra_all")]
#[test]
fn hs71_nalgebra() {
    check_hs71::<_, f64>(backend_aliases::nalgebra::DVector::from_column_slice);
    check_hs71::<_, f32>(backend_aliases::nalgebra::DVector::from_column_slice);
}
#[cfg(feature = "ndarray_all")]
#[test]
fn hs71_ndarray() {
    check_hs71::<_, f64>(|v| {
        backend_aliases::ndarray::Array1::from_vec(v.to_vec())
    });
    check_hs71::<_, f32>(|v| {
        backend_aliases::ndarray::Array1::from_vec(v.to_vec())
    });
}
#[cfg(feature = "faer_all")]
#[test]
fn hs71_faer() {
    check_hs71::<_, f64>(|v| {
        backend_aliases::faer::Col::from_fn(v.len(), |i| v[i])
    });
    check_hs71::<_, f32>(|v| {
        backend_aliases::faer::Col::from_fn(v.len(), |i| v[i])
    });
}

#[test]
fn hs71_reference_accepted_trajectory_and_exact_continuation() {
    let p = Hs71::new(|v: &[f64]| v.to_vec());
    let mut stepper = Executor::new(
        p.clone(),
        Slsqp::new().with_absolute_accuracy_tolerance(1e-10),
        SlsqpState::new(vec![1., 5., 5., 1.]),
    )
    .max_iter(100)
    .into_stepper()
    .unwrap();
    // The reference prints its final point twice when the next QP finds convergence.
    let rows: Vec<Vec<f64>> = include_str!("fixtures/slsqp_hs71.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
        .map(|s| s.split_whitespace().map(|v| v.parse().unwrap()).collect())
        .collect();
    for (i, row) in rows[..rows.len() - 1].iter().enumerate() {
        if i > 0 {
            stepper.step().unwrap();
        }
        let state = stepper.state();
        assert_eq!(state.iter(), i as u64);
        assert!(
            (state.cost() - row[1]).abs() < 2e-8,
            "step {i}: {} != {}",
            state.cost(),
            row[1]
        );
        for (a, b) in state.param().iter().zip(&row[2..]) {
            assert!((a - b).abs() < 2e-8, "step {i}: {:?}", state.param());
        }
        assert_eq!(
            state.gradient().unwrap(),
            &p.gradient(state.param()).unwrap()
        );
        assert_eq!(state.best_param(), state.param());
        assert_eq!(state.best_cost(), state.cost());
        assert_eq!(state.raw_counts().gradient_evals, i as u64 + 1);
        if i == 2 {
            let checkpoint = stepper.into_checkpoint().unwrap();
            #[cfg(feature = "serde")]
            let checkpoint = bincode::serde::decode_from_slice(
                &bincode::serde::encode_to_vec(
                    &checkpoint,
                    bincode::config::standard(),
                )
                .unwrap(),
                bincode::config::standard(),
            )
            .unwrap()
            .0;
            stepper = Executor::resume_from_checkpoint(p.clone(), checkpoint)
                .max_iter(100)
                .into_stepper()
                .unwrap();
        }
    }
    let result = stepper.run_to_end().unwrap();
    assert_eq!(result.reason, TerminationReason::SolverConverged);
}

#[test]
#[cfg(feature = "serde")]
fn checkpoint_rebuilds_scratch_for_exact_continuation() {
    let problem = Hs71::new(|v: &[f64]| v.to_vec());
    let run = || {
        Executor::from_start(
            problem.clone(),
            Slsqp::new().with_absolute_accuracy_tolerance(1e-10),
            vec![1.0, 5.0, 5.0, 1.0],
        )
        .max_iter(100)
    };
    let expected = run().run_with_solver().unwrap();
    for split in [0, 1, 3, 5] {
        let mut stepper = run().into_stepper().unwrap();
        for _ in 0..split {
            stepper.step().unwrap();
        }
        let checkpoint = stepper.into_checkpoint().unwrap();
        let bytes = bincode::serde::encode_to_vec(
            checkpoint,
            bincode::config::standard(),
        )
        .unwrap();
        let (checkpoint, consumed): (
            basin::ExactCheckpoint<Slsqp, SlsqpState<Vec<f64>>>,
            usize,
        ) = bincode::serde::decode_from_slice(
            &bytes,
            bincode::config::standard(),
        )
        .unwrap();
        assert_eq!(consumed, bytes.len());
        let resumed =
            Executor::resume_from_checkpoint(problem.clone(), checkpoint)
                .max_iter(100)
                .run_with_solver()
                .unwrap();
        assert_eq!(resumed.reason, expected.reason);
        assert_eq!(resumed.state.param(), expected.state.param());
        assert_eq!(resumed.state.cost(), expected.state.cost());
        assert_eq!(resumed.state.gradient(), expected.state.gradient());
        assert_eq!(resumed.counts, expected.counts);
        assert_eq!(resumed.state.iter(), expected.state.iter());
        assert_eq!(
            resumed.solver.equality_multipliers(),
            expected.solver.equality_multipliers()
        );
        assert_eq!(
            resumed.solver.inequality_multipliers(),
            expected.solver.inequality_multipliers()
        );
    }
}

#[derive(Clone)]
struct LinearQuadratic {
    target: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    eq: DenseMatrix,
    eq_rhs: Vec<f64>,
    iq: DenseMatrix,
    iq_rhs: Vec<f64>,
}
impl LinearQuadratic {
    fn new(n: usize) -> Self {
        Self {
            target: vec![2.; n],
            lower: vec![f64::NEG_INFINITY; n],
            upper: vec![f64::INFINITY; n],
            eq: DenseMatrix::from_fn(0, n, |_, _| 0.),
            eq_rhs: vec![],
            iq: DenseMatrix::from_fn(0, n, |_, _| 0.),
            iq_rhs: vec![],
        }
    }
}
impl CostFunction for LinearQuadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(x.iter()
            .zip(&self.target)
            .map(|(a, b)| (a - b).powi(2))
            .sum())
    }
}
impl Gradient for LinearQuadratic {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(x.iter()
            .zip(&self.target)
            .map(|(a, b)| 2. * (a - b))
            .collect())
    }
}
impl NonlinearConstraints for LinearQuadratic {
    type Matrix = DenseMatrix;
    fn lower(&self) -> Option<&Vec<f64>> {
        Some(&self.lower)
    }
    fn upper(&self) -> Option<&Vec<f64>> {
        Some(&self.upper)
    }
    fn equalities(&self) -> Option<(&DenseMatrix, &Vec<f64>)> {
        Some((&self.eq, &self.eq_rhs))
    }
    fn inequalities(&self) -> Option<(&DenseMatrix, &Vec<f64>)> {
        Some((&self.iq, &self.iq_rhs))
    }
    fn num_nonlinear_constraints(&self) -> usize {
        0
    }
    fn nonlinear_constraints(
        &self,
        _: &Vec<f64>,
    ) -> Result<Vec<f64>, Infallible> {
        panic!("zero nonlinear rows must not be evaluated")
    }
}
impl ConstraintJacobian for LinearQuadratic {
    fn constraint_jacobian(
        &self,
        _: &Vec<f64>,
    ) -> Result<DenseMatrix, Infallible> {
        panic!("linear derivatives are exact")
    }
}
#[test]
fn linear_blocks_fixed_coordinates_active_bounds_and_unconstrained() {
    for mode in 0..3 {
        let mut p = LinearQuadratic::new(4);
        let expected = if mode == 0 {
            vec![2.; 4]
        } else {
            vec![0.25, 0.5, 0.75, 3.]
        };
        if mode > 0 {
            p.eq = DenseMatrix::from_row_slice(1, 4, &[1., 0., 0., 0.]);
            p.eq_rhs = vec![0.25];
            p.iq = DenseMatrix::from_row_slice(1, 4, &[0., 1., 0., 0.]);
            p.iq_rhs = vec![0.5];
            p.upper[2] = 0.75;
            p.lower[3] = 3.;
            if mode == 2 {
                p.upper[3] = 3.;
            }
        }
        let r = Executor::new(
            p,
            Slsqp::new().with_absolute_accuracy_tolerance(1e-12),
            SlsqpState::new(vec![-10.; 4]),
        )
        .max_iter(50)
        .run_with_solver()
        .unwrap();
        assert_eq!(r.reason, TerminationReason::SolverConverged);
        for (a, b) in r.state.param().iter().zip(expected) {
            assert!((a - b).abs() < 1e-7, "{:?}", r.state.param());
        }
        assert!(r.state.stationarity().unwrap() < 1e-7);
        assert_eq!(r.state.raw_counts().jacobian_evals, 0);
        assert_eq!(r.state.raw_counts().residual_evals, 0);
        if mode > 0 {
            assert!(
                (r.solver.equality_multipliers().unwrap()[0] - 3.5).abs()
                    < 1e-7
            );
            assert!(
                (r.solver.inequality_multipliers().unwrap()[0] - 3.).abs()
                    < 1e-7
            );
        }
    }
}
#[test]
fn all_fixed_feasible_and_infeasible() {
    for accuracy in [Some(1e-6), Some(0.), None] {
        for feasible in [true, false] {
            let mut p = LinearQuadratic::new(1);
            p.lower = vec![1.];
            p.upper = vec![1.];
            p.eq = DenseMatrix::from_row_slice(1, 1, &[1.]);
            p.eq_rhs = vec![if feasible { 1. } else { 2. }];
            let r = Executor::new(
                p,
                Slsqp::new().with_absolute_accuracy_tolerance(accuracy),
                SlsqpState::new(vec![0.]),
            )
            .require_evaluated_state()
            .max_iter(10)
            .run()
            .unwrap();
            assert_eq!(
                r.reason,
                if !feasible {
                    TerminationReason::SolverFailed
                } else if accuracy.is_some() {
                    TerminationReason::SolverConverged
                } else {
                    TerminationReason::MaxIter
                },
                "accuracy={accuracy:?}, feasible={feasible}"
            );
            assert_eq!(
                r.state.failure(),
                if feasible {
                    None
                } else {
                    Some(basin::SlsqpFailure::IncompatibleConstraints)
                }
            );
            assert_eq!(r.state.param(), &vec![1.]);
            assert_eq!(
                r.state.iter(),
                if feasible && accuracy.is_none() {
                    10
                } else {
                    0
                }
            );
        }
    }
}

#[test]
fn all_fixed_bounds_with_disabled_accuracy_reach_iteration_limit() {
    let mut p = LinearQuadratic::new(1);
    p.target = vec![0.];
    p.lower = vec![1.];
    p.upper = vec![1.];
    let r = Executor::new(
        p,
        Slsqp::new().with_absolute_accuracy_tolerance(None),
        SlsqpState::new(vec![0.]),
    )
    .require_evaluated_state()
    .max_iter(10)
    .run()
    .unwrap();
    assert_eq!(r.reason, TerminationReason::MaxIter);
    assert_eq!(r.state.failure(), None);
    assert_eq!(r.state.iter(), 10);
    assert_eq!(r.state.param(), &vec![1.]);
    assert_eq!(r.state.cost(), 1.);
    assert_eq!(r.state.gradient(), Some(&vec![2.]));
    assert_eq!(r.state.constraint_violation(), Some(0.));
    assert_eq!(r.state.raw_counts().cost_evals, 1);
    assert_eq!(r.state.raw_counts().gradient_evals, 1);
}

#[test]
fn dependent_and_excess_equalities_fail_cleanly() {
    use basin::SlsqpFailure;
    for (n, expected) in [
        (3, SlsqpFailure::RankDeficientEqualities),
        (1, SlsqpFailure::TooManyEqualities),
    ] {
        let mut p = LinearQuadratic::new(n);
        p.eq = DenseMatrix::from_fn(2, n, |i, j| {
            if j == 0 { (i + 1) as f64 } else { 0. }
        });
        p.eq_rhs = vec![1., 2.];
        let r = Executor::new(p, Slsqp::new(), SlsqpState::new(vec![0.; n]))
            .max_iter(20)
            .run()
            .unwrap();
        assert_eq!(r.reason, TerminationReason::SolverFailed);
        assert_eq!(r.state.failure(), Some(expected));
        assert_eq!(r.state.iter(), 0);
    }
}
#[test]
fn incompatible_inequalities_never_report_convergence() {
    let mut p = LinearQuadratic::new(2);
    p.iq = DenseMatrix::from_row_slice(2, 2, &[1., 0., -1., 0.]);
    p.iq_rhs = vec![0., -1.];
    let r = Executor::new(p, Slsqp::new(), SlsqpState::new(vec![0.; 2]))
        .max_iter(100)
        .run()
        .unwrap();
    assert_eq!(r.reason, TerminationReason::SolverFailed);
    assert!(r.state.constraint_violation().unwrap() > 0.9);
}
#[test]
fn numerical_constraint_derivatives_solve_hs71() {
    let p = basin::FiniteDiff::new(Hs71::new(|v: &[f64]| v.to_vec()))
        .with_bounds(vec![1.; 4], vec![5.; 4]);
    let r = Executor::new(
        p,
        Slsqp::new().with_absolute_accuracy_tolerance(1e-9),
        SlsqpState::new(vec![1., 5., 5., 1.]),
    )
    .max_iter(100)
    .run()
    .unwrap();
    assert_eq!(r.reason, TerminationReason::SolverConverged);
    assert!((r.state.cost() - 17.014017289134).abs() < 1e-7);
    assert!(r.state.constraint_violation().unwrap() < 1e-8);
}
#[test]
fn scipy_and_nlopt_final_output_agreement() {
    let p = Hs71::new(|v: &[f64]| v.to_vec());
    let r = Executor::new(
        p,
        Slsqp::new().with_absolute_accuracy_tolerance(1e-10),
        SlsqpState::new(vec![1., 5., 5., 1.]),
    )
    .max_iter(100)
    .run()
    .unwrap();
    for line in include_str!("fixtures/slsqp_hs71_libraries.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let row: Vec<f64> = line
            .split_whitespace()
            .skip(2)
            .map(|v| v.parse().unwrap())
            .collect();
        assert!((r.state.cost() - row[0]).abs() < 1e-8);
        for (a, b) in r.state.param().iter().zip(&row[1..5]) {
            assert!((a - b).abs() < 1e-7);
        }
        assert!(row[5].abs() < 1e-9 && row[6] < 1e-9 && row[7] < 1e-6);
        assert!(r.state.stationarity().unwrap() < 1e-6);
    }
}

#[derive(Debug, PartialEq, Eq)]
struct CallbackError(u8);
struct Faulty {
    callback: u8,
    after_initial: bool,
    nonfinite: bool,
}
impl Faulty {
    fn fails(&self, callback: u8, x: &[f64]) -> bool {
        self.callback == callback && (!self.after_initial || x != [3., -1.])
    }
}
impl CostFunction for Faulty {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = CallbackError;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, CallbackError> {
        if self.fails(0, x) {
            if self.nonfinite {
                return Ok(f64::NAN);
            }
            return Err(CallbackError(0));
        }
        Ok(EqualityQuadratic.cost(x).unwrap())
    }
}
impl Gradient for Faulty {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, CallbackError> {
        if self.fails(1, x) {
            if self.nonfinite {
                return Ok(vec![f64::NAN; 2]);
            }
            return Err(CallbackError(1));
        }
        Ok(EqualityQuadratic.gradient(x).unwrap())
    }
}
impl NonlinearConstraints for Faulty {
    type Matrix = DenseMatrix;
    fn num_nonlinear_constraints(&self) -> usize {
        0
    }
    fn nonlinear_constraints(
        &self,
        _: &Vec<f64>,
    ) -> Result<Vec<f64>, CallbackError> {
        Ok(vec![])
    }
    fn num_nonlinear_equalities(&self) -> usize {
        1
    }
    fn nonlinear_equalities(
        &self,
        x: &Vec<f64>,
    ) -> Result<Option<Vec<f64>>, CallbackError> {
        if self.fails(2, x) {
            if self.nonfinite {
                return Ok(Some(vec![f64::NAN]));
            }
            return Err(CallbackError(2));
        }
        Ok(EqualityQuadratic.nonlinear_equalities(x).unwrap())
    }
}
impl ConstraintJacobian for Faulty {
    fn constraint_jacobian(
        &self,
        x: &Vec<f64>,
    ) -> Result<DenseMatrix, CallbackError> {
        if self.fails(3, x) {
            if self.nonfinite {
                return Ok(DenseMatrix::from_row_slice(1, 2, &[f64::NAN, 1.]));
            }
            return Err(CallbackError(3));
        }
        Ok(EqualityQuadratic.constraint_jacobian(x).unwrap())
    }
}
#[test]
fn callback_errors_propagate_at_initialization_and_during_steps() {
    for callback in 0..4 {
        for after_initial in [false, true] {
            let p = Faulty {
                callback,
                after_initial,
                nonfinite: false,
            };
            let r =
                Executor::new(p, Slsqp::new(), SlsqpState::new(vec![3., -1.]))
                    .max_iter(20)
                    .run();
            assert_eq!(r.err().unwrap(), CallbackError(callback));
        }
    }
}
#[test]
fn nonfinite_callbacks_fail_without_publishing_a_partial_trial() {
    for callback in 0..4 {
        let p = Faulty {
            callback,
            after_initial: true,
            nonfinite: true,
        };
        let r = Executor::new(p, Slsqp::new(), SlsqpState::new(vec![3., -1.]))
            .max_iter(20)
            .run()
            .unwrap();
        assert_eq!(r.reason, TerminationReason::SolverFailed);
        assert_eq!(r.state.param(), &vec![3., -1.]);
        assert_eq!(r.state.best_param(), &vec![3., -1.]);
        assert_eq!(r.state.cost(), 13.);
        assert_eq!(r.state.gradient(), Some(&vec![4., -6.]));
        assert_eq!(r.state.iter(), 0);
        assert_eq!(r.state.best_cost_evals(), 2);
        assert!(r.state.cost_evals() > r.state.best_cost_evals());
    }
}
#[test]
fn nonfinite_initial_values_report_numerical_failure() {
    for callback in 0..4 {
        let p = Faulty {
            callback,
            after_initial: false,
            nonfinite: true,
        };
        let r = Executor::new(p, Slsqp::new(), SlsqpState::new(vec![3., -1.]))
            .max_iter(20)
            .run()
            .unwrap();
        assert_eq!(r.reason, TerminationReason::SolverFailed);
        assert_eq!(
            r.state.failure(),
            Some(basin::SlsqpFailure::NonFiniteEvaluation)
        );
        assert_eq!(r.state.iter(), 0);
    }
}
#[test]
fn inconsistent_linearization_uses_slack_recovery() {
    struct OutsideCircle {
        lo: Vec<f64>,
        hi: Vec<f64>,
    }
    impl CostFunction for OutsideCircle {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
            Ok(x[0] * x[0])
        }
    }
    impl NonlinearConstraints for OutsideCircle {
        type Matrix = DenseMatrix;
        fn lower(&self) -> Option<&Vec<f64>> {
            Some(&self.lo)
        }
        fn upper(&self) -> Option<&Vec<f64>> {
            Some(&self.hi)
        }
        fn num_nonlinear_constraints(&self) -> usize {
            1
        }
        fn nonlinear_constraints(
            &self,
            x: &Vec<f64>,
        ) -> Result<Vec<f64>, Infallible> {
            Ok(vec![1. - x[0] * x[0]])
        }
    }
    let p = basin::BoundedFiniteDiff::new(
        OutsideCircle {
            lo: vec![0.],
            hi: vec![2.],
        },
        vec![0.],
        vec![2.],
    );
    let r = Executor::new(
        p,
        Slsqp::new().with_absolute_accuracy_tolerance(1e-10),
        SlsqpState::new(vec![0.1]),
    )
    .max_iter(100)
    .run()
    .unwrap();
    assert_eq!(
        r.reason,
        TerminationReason::SolverConverged,
        "{:?}",
        r.state.failure()
    );
    assert!((r.state.param()[0] - 1.).abs() < 1e-7);
    assert!(r.state.constraint_violation().unwrap() < 1e-10);
}
#[test]
fn folded_constraints_include_both_signs_of_nonlinear_equalities() {
    let r = Executor::from_start(
        basin::FoldedConstraints::new(EqualityQuadratic),
        basin::Cobyla::new(),
        vec![3., -1.],
    )
    .max_iter(1000)
    .run()
    .unwrap();
    assert_eq!(r.reason, TerminationReason::SolverConverged);
    assert!((r.state.param()[0] + r.state.param()[1] - 1.).abs() < 1e-5);
    assert!(r.state.param()[0].abs() < 1e-3);
}
#[test]
#[should_panic(expected = "positive")]
fn zero_subproblem_limit_is_invalid() {
    let _ = Slsqp::<f64>::new().with_max_subproblem_iterations(0);
}
#[test]
#[should_panic]
fn nonfinite_accuracy_is_invalid() {
    let _ = Slsqp::new().with_absolute_accuracy_tolerance(f64::NAN);
}

#[test]
fn nnls_iteration_limit_reports_a_subproblem_failure() {
    let mut p = LinearQuadratic::new(2);
    p.iq = DenseMatrix::from_row_slice(2, 2, &[1., 0., 0., 1.]);
    p.iq_rhs = vec![0., 0.];
    let r = Executor::from_start(
        p,
        Slsqp::new().with_max_subproblem_iterations(1),
        vec![-1.; 2],
    )
    .max_iter(10)
    .run()
    .unwrap();
    assert_eq!(r.reason, TerminationReason::SolverFailed);
    assert_eq!(
        r.state.failure(),
        Some(basin::SlsqpFailure::SubproblemIterationLimit)
    );
    assert_eq!(r.state.iter(), 0);
}

#[test]
fn nonfinite_trials_can_backtrack_to_a_finite_solution() {
    struct HalfLine;
    impl CostFunction for HalfLine {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
            Ok(if x[0] > 0. {
                (x[0] - 0.1).powi(2)
            } else {
                f64::INFINITY
            })
        }
    }
    impl Gradient for HalfLine {
        type Gradient = Vec<f64>;
        fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
            assert!(x[0] > 0.);
            Ok(vec![2. * (x[0] - 0.1)])
        }
    }
    impl NonlinearConstraints for HalfLine {
        type Matrix = DenseMatrix;
        fn num_nonlinear_constraints(&self) -> usize {
            0
        }
        fn nonlinear_constraints(
            &self,
            _: &Vec<f64>,
        ) -> Result<Vec<f64>, Infallible> {
            Ok(vec![])
        }
    }
    impl ConstraintJacobian for HalfLine {
        fn constraint_jacobian(
            &self,
            _: &Vec<f64>,
        ) -> Result<DenseMatrix, Infallible> {
            unreachable!()
        }
    }
    let r = Executor::from_start(HalfLine, Slsqp::new(), vec![1.])
        .max_iter(10)
        .run()
        .unwrap();
    assert_eq!(r.reason, TerminationReason::SolverConverged);
    assert!((r.state.param()[0] - 0.1).abs() < 1e-12);
    assert_eq!(r.state.raw_counts().cost_evals, 3);
    assert_eq!(r.state.raw_counts().gradient_evals, 2);
}

#[test]
fn unconstrained_rosenbrock_exercises_repeated_bfgs_and_backtracking() {
    struct Rosenbrock;
    impl CostFunction for Rosenbrock {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
            Ok(100. * (x[1] - x[0] * x[0]).powi(2) + (1. - x[0]).powi(2))
        }
    }
    impl NonlinearConstraints for Rosenbrock {
        type Matrix = DenseMatrix;
        fn num_nonlinear_constraints(&self) -> usize {
            0
        }
        fn nonlinear_constraints(
            &self,
            _: &Vec<f64>,
        ) -> Result<Vec<f64>, Infallible> {
            Ok(vec![])
        }
    }
    let r = Executor::from_start(
        basin::FiniteDiff::new(Rosenbrock),
        Slsqp::new().with_absolute_accuracy_tolerance(1e-10),
        vec![-1.2, 1.],
    )
    .max_iter(100)
    .run()
    .unwrap();
    assert_eq!(
        r.reason,
        TerminationReason::SolverConverged,
        "{:?}",
        r.state.failure()
    );
    assert!(r.state.param().iter().all(|x| (x - 1.).abs() < 1e-5));
    assert!(r.state.cost() < 1e-10);
}

#[test]
fn disabled_accuracy_leaves_iteration_budget_in_control() {
    let r = Executor::from_start(
        EqualityQuadratic,
        Slsqp::new().with_absolute_accuracy_tolerance(None),
        vec![3., -1.],
    )
    .max_iter(1)
    .run()
    .unwrap();
    assert_eq!(r.reason, TerminationReason::MaxIter);
    assert_eq!(r.state.iter(), 1);
}

#[test]
fn repeated_zero_step_bfgs_updates_exhaust_the_reset_limit() {
    // An inconsistent derivative forces line-search exhaustion. At this
    // scale its final displacement rounds to zero, making BFGS singular.
    struct FlatWithBadGradient;
    impl CostFunction for FlatWithBadGradient {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = Infallible;
        fn cost(&self, _: &Vec<f64>) -> Result<f64, Infallible> {
            Ok(0.)
        }
    }
    impl Gradient for FlatWithBadGradient {
        type Gradient = Vec<f64>;
        fn gradient(&self, _: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
            Ok(vec![1.])
        }
    }
    impl NonlinearConstraints for FlatWithBadGradient {
        type Matrix = DenseMatrix;
        fn num_nonlinear_constraints(&self) -> usize {
            0
        }
        fn nonlinear_constraints(
            &self,
            _: &Vec<f64>,
        ) -> Result<Vec<f64>, Infallible> {
            Ok(vec![])
        }
    }
    impl ConstraintJacobian for FlatWithBadGradient {
        fn constraint_jacobian(
            &self,
            _: &Vec<f64>,
        ) -> Result<DenseMatrix, Infallible> {
            unreachable!()
        }
    }
    let r = Executor::from_start(
        FlatWithBadGradient,
        Slsqp::new().with_absolute_accuracy_tolerance(None),
        vec![1e100],
    )
    .max_iter(10)
    .run()
    .unwrap();
    assert_eq!(r.reason, TerminationReason::SolverFailed);
    assert_eq!(
        r.state.failure(),
        Some(basin::SlsqpFailure::NonDescentDirection)
    );
    assert_eq!(r.state.iter(), 5);
    assert_eq!(r.state.stationarity(), Some(1.));
}

#[test]
fn overflowing_directional_model_is_a_numerical_failure() {
    struct HugeLinear;
    impl CostFunction for HugeLinear {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
            Ok(1e200 * x[0])
        }
    }
    impl Gradient for HugeLinear {
        type Gradient = Vec<f64>;
        fn gradient(&self, _: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
            Ok(vec![1e200])
        }
    }
    impl NonlinearConstraints for HugeLinear {
        type Matrix = DenseMatrix;
        fn num_nonlinear_constraints(&self) -> usize {
            0
        }
        fn nonlinear_constraints(
            &self,
            _: &Vec<f64>,
        ) -> Result<Vec<f64>, Infallible> {
            Ok(vec![])
        }
    }
    impl ConstraintJacobian for HugeLinear {
        fn constraint_jacobian(
            &self,
            _: &Vec<f64>,
        ) -> Result<DenseMatrix, Infallible> {
            unreachable!()
        }
    }
    let r = Executor::from_start(HugeLinear, Slsqp::new(), vec![0.])
        .max_iter(10)
        .run()
        .unwrap();
    assert_eq!(r.reason, TerminationReason::SolverFailed);
    assert_eq!(
        r.state.failure(),
        Some(basin::SlsqpFailure::NonFiniteEvaluation)
    );
    assert_eq!(r.state.raw_counts().cost_evals, 1);
    assert_eq!(r.state.param(), &vec![0.]);
}
