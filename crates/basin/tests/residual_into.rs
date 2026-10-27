use std::fmt::Debug;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed},
};

use basin::{
    BoundedFiniteDiff, BoundedLevenbergMarquardt, CostFunction, DenseMatrix,
    DenseMatrixFromFn, Executor, FiniteDiff, GaussNewton, Jacobian,
    LevenbergMarquardt, LevenbergMarquardtQr, MatrixIndex, MatrixShape, Method,
    PointState, Problem, Residual, RobustLeastSquares, Scalar, Solver,
    SquaredLoss, TerminationCode, TrustRegionReflective, VectorIndex,
    VectorLen,
};

#[path = "support/backend_aliases.rs"]
mod backend_aliases;

#[derive(Default)]
struct Calls {
    owned: AtomicUsize,
    in_place: AtomicUsize,
    fused: AtomicUsize,
    jacobians: AtomicUsize,
    reject_once: AtomicBool,
    fail: AtomicBool,
}

#[derive(Clone)]
struct Fit<F: Scalar> {
    calls: Arc<Calls>,
    lower: Vec<F>,
    upper: Vec<F>,
}

impl<F: Scalar> Fit<F> {
    fn new(calls: Arc<Calls>) -> Self {
        Self {
            calls,
            lower: vec![F::zero(); 2],
            upper: vec![F::one(); 2],
        }
    }

    fn fill(x: &[F], out: &mut Vec<F>) {
        out.resize(3, F::zero());
        out[0] = x[0] - F::from_f64(0.3).unwrap();
        out[1] = x[1] - F::from_f64(0.7).unwrap();
        out[2] = out[0] + out[1];
    }
}

impl<F: Scalar> Residual for Fit<F> {
    type Param = Vec<F>;
    type Output = Vec<F>;
    type Error = &'static str;

    fn residual(&self, x: &Vec<F>) -> Result<Vec<F>, Self::Error> {
        self.calls.owned.fetch_add(1, Relaxed);
        let mut out = Vec::new();
        Self::fill(x, &mut out);
        Ok(out)
    }

    fn residual_into(
        &self,
        x: &Vec<F>,
        out: &mut Vec<F>,
    ) -> Result<(), Self::Error> {
        self.calls.in_place.fetch_add(1, Relaxed);
        Self::fill(x, out);
        if self.calls.fail.load(Relaxed) {
            return Err("residual failed");
        }
        if self.calls.reject_once.swap(false, Relaxed) {
            out.fill(F::nan());
        }
        Ok(())
    }
}

impl<F: Scalar> Jacobian for Fit<F> {
    type Jacobian = DenseMatrix<F>;
    fn jacobian(&self, _: &Vec<F>) -> Result<Self::Jacobian, Self::Error> {
        self.calls.jacobians.fetch_add(1, Relaxed);
        Ok(DenseMatrix::from_row_slice(
            3,
            2,
            &[F::one(), F::zero(), F::zero(), F::one(), F::one(), F::one()],
        ))
    }
    fn residual_and_jacobian(
        &self,
        x: &Vec<F>,
    ) -> Result<(Vec<F>, Self::Jacobian), Self::Error> {
        self.calls.fused.fetch_add(1, Relaxed);
        Ok((self.residual(x)?, self.jacobian(x)?))
    }
}

impl<F: Scalar> CostFunction for Fit<F> {
    type Param = Vec<F>;
    type Output = F;
    type Error = &'static str;
    fn cost(&self, _: &Vec<F>) -> Result<F, Self::Error> {
        panic!("least-squares solvers must derive cost from the residual");
    }
}

impl<F: Scalar> basin::Constraints for Fit<F> {
    type Set = basin::ConstraintSet<basin::BoxBounds<Vec<F>>>;
    fn constraints(
        &self,
    ) -> <Self::Set as basin::ConstraintDescription>::Borrowed<'_> {
        basin::ConstraintSet::new().with_bounds(basin::BoxBounds::new(
            Some(&self.lower),
            Some(&self.upper),
        ))
    }
}

fn check_solver<F: Scalar, S>(solver: S, reject_once: bool)
where
    S: Solver<Fit<F>, PointState<Vec<F>, F>>,
    S::Error: Debug,
{
    let calls = Arc::new(Calls::default());
    calls.reject_once.store(reject_once, Relaxed);
    let result = Executor::new(
        Fit::new(Arc::clone(&calls)),
        solver,
        PointState::new(vec![
            F::from_f64(0.8).unwrap(),
            F::from_f64(0.2).unwrap(),
        ]),
    )
    .max_iter(200)
    .run()
    .unwrap();
    assert_eq!(result.report.code(), TerminationCode::SolverConverged);
    assert!(result.cost() < F::from_f64(1e-10).unwrap());
    assert_eq!(calls.fused.load(Relaxed), 1);
    assert_eq!(calls.owned.load(Relaxed), 1);
    assert!(calls.in_place.load(Relaxed) > usize::from(reject_once));
    assert_eq!(
        result.counts.residual_evals as usize,
        1 + calls.in_place.load(Relaxed)
    );
    assert_eq!(
        result.counts.jacobian_evals as usize,
        calls.jacobians.load(Relaxed)
    );
}

fn check_solver_error<S>(solver: S)
where
    S: Solver<Fit<f64>, PointState<Vec<f64>>>,
    S::Error: Debug,
{
    let calls = Arc::new(Calls::default());
    calls.fail.store(true, Relaxed);
    let error = Executor::new(
        Fit::new(Arc::clone(&calls)),
        solver,
        PointState::new(vec![0.8, 0.2]),
    )
    .max_iter(10)
    .run()
    .err()
    .expect("in-place callback errors must abort the solver");
    assert!(format!("{error:?}").contains("residual failed"));
    assert_eq!(calls.owned.load(Relaxed), 1);
    assert_eq!(calls.in_place.load(Relaxed), 1);
}

macro_rules! solver_test {
    ($name:ident, $solver:ident, $reject:expr) => {
        #[test]
        fn $name() {
            check_solver(
                $solver::<Vec<f64>, DenseMatrix<f64>, f64>::default(),
                $reject,
            );
            check_solver(
                $solver::<Vec<f32>, DenseMatrix<f32>, f32>::default(),
                $reject,
            );
            check_solver_error($solver::default());
        }
    };
}

solver_test!(gauss_newton_reuses_residuals, GaussNewton, false);
solver_test!(
    lm_reuses_residuals_after_rejection,
    LevenbergMarquardt,
    true
);
solver_test!(
    qr_reuses_residuals_after_rejection,
    LevenbergMarquardtQr,
    true
);
solver_test!(
    bounded_lm_reuses_residuals_after_rejection,
    BoundedLevenbergMarquardt,
    true
);

#[test]
fn reflective_reuses_residuals_after_rejection() {
    check_solver(TrustRegionReflective::<f64>::new(), true);
    check_solver(TrustRegionReflective::<f32>::new(), true);
    check_solver_error(TrustRegionReflective::new());
}

#[test]
fn robust_evaluation_forwards_the_override_and_counts_requests() {
    let calls = Arc::new(Calls::default());
    let problem = RobustLeastSquares::new(
        Fit::<f64>::new(Arc::clone(&calls)),
        SquaredLoss,
    );
    let result = Executor::from_start(
        problem,
        LevenbergMarquardtQr::new(),
        vec![0.8, 0.2],
    )
    .max_iter(100)
    .run()
    .unwrap();
    assert_eq!(result.report.code(), TerminationCode::SolverConverged);
    assert_eq!(calls.owned.load(Relaxed), 1);
    assert!(calls.in_place.load(Relaxed) > 0);
    assert_eq!(
        result.counts.residual_evals as usize,
        1 + calls.in_place.load(Relaxed)
    );
}

#[test]
fn counted_failures_propagate_and_wrappers_forward_the_override() {
    let calls = Arc::new(Calls::default());
    let fit = Fit::<f64>::new(Arc::clone(&calls));
    let mut out = Vec::with_capacity(8);
    let allocation = out.as_ptr();
    let x = vec![0.8, 0.2];
    let mut problem = Problem::new(FiniteDiff::new(fit.clone()));
    problem.residual_into(&x, &mut out).unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(out.as_ptr(), allocation);
    let bounded =
        BoundedFiniteDiff::new(fit.clone(), vec![0.0; 2], vec![1.0; 2]);
    bounded.residual_into(&x, &mut out).unwrap();
    let constrained =
        basin::Constrained::new(fit.clone(), basin::NoConstraints::EMPTY);
    constrained.residual_into(&x, &mut out).unwrap();
    basin::SamplingDomain::new(fit)
        .residual_into(&x, &mut out)
        .unwrap();
    assert_eq!(calls.owned.load(Relaxed), 0);
    calls.fail.store(true, Relaxed);
    assert_eq!(problem.residual_into(&x, &mut out), Err("residual failed"));
    assert_eq!(problem.counts().residual_evals, 2);
    assert_eq!(calls.in_place.load(Relaxed), 5);
}

#[test]
fn default_method_preserves_owned_implementations_and_errors() {
    struct Owned;
    impl Residual for Owned {
        type Param = bool;
        type Output = Vec<f64>;
        type Error = &'static str;
        fn residual(&self, fail: &bool) -> Result<Vec<f64>, Self::Error> {
            if *fail {
                Err("failed")
            } else {
                Ok(vec![1.0, 2.0])
            }
        }
    }
    let mut problem = Problem::new(Owned);
    let mut out = vec![9.0];
    problem.residual_into(&false, &mut out).unwrap();
    assert_eq!(out, vec![1.0, 2.0]);
    assert_eq!(problem.residual_into(&true, &mut out), Err("failed"));
    assert_eq!(problem.counts().residual_evals, 2);
}

struct Probes<V, F> {
    make: fn(&[F]) -> V,
    calls: Arc<Calls>,
}

impl<V: VectorLen + VectorIndex<F>, F: Scalar> Residual for Probes<V, F> {
    type Param = V;
    type Output = V;
    type Error = &'static str;
    fn residual(&self, _: &V) -> Result<V, Self::Error> {
        panic!("finite-difference probes must use residual_into");
    }
    fn residual_into(&self, x: &V, out: &mut V) -> Result<(), Self::Error> {
        self.calls.in_place.fetch_add(1, Relaxed);
        if self.calls.fail.load(Relaxed) {
            return Err("probe failed");
        }
        if out.vec_len() != 3 {
            *out = (self.make)(&[F::zero(); 3]);
        }
        out.set_scalar(0, x.get_scalar(0));
        out.set_scalar(1, x.get_scalar(1));
        out.set_scalar(2, x.get_scalar(0) + x.get_scalar(1));
        Ok(())
    }
}

fn bounded_probes<V, F>(make: fn(&[F]) -> V)
where
    V: Clone + VectorLen + VectorIndex<F> + DenseMatrixFromFn<F> + Send + Sync,
    V::Matrix: MatrixIndex<F>,
    F: Scalar + Send + Sync,
{
    for method in [Method::Forward, Method::Central] {
        let calls = Arc::new(Calls::default());
        let fd = BoundedFiniteDiff::new(
            Probes {
                make,
                calls: Arc::clone(&calls),
            },
            make(&[F::zero(); 2]),
            make(&[F::one(); 2]),
        )
        .jacobian_method(method);
        let x = make(&[F::zero(), F::one()]);
        let j = fd.jacobian(&x).unwrap();
        assert_eq!((j.matrix_rows(), j.matrix_cols()), (3, 2));
        let tolerance = F::epsilon().sqrt() * F::from_f64(10.0).unwrap();
        for row in 0..3 {
            for col in 0..2 {
                let expected = if row == col || row == 2 {
                    F::one()
                } else {
                    F::zero()
                };
                assert!(
                    (j.matrix_entry(row, col) - expected).abs() < tolerance
                );
            }
        }
        assert_eq!(
            calls.in_place.load(Relaxed),
            if matches!(method, Method::Forward) {
                3
            } else {
                5
            }
        );
        calls.fail.store(true, Relaxed);
        assert!(matches!(fd.jacobian(&x), Err("probe failed")));
    }
}

#[test]
fn bounded_probes_vec() {
    bounded_probes::<_, f64>(|x| x.to_vec());
    bounded_probes::<_, f32>(|x| x.to_vec());
}

#[cfg(feature = "nalgebra_all")]
#[test]
fn bounded_probes_nalgebra() {
    bounded_probes::<_, f64>(
        backend_aliases::nalgebra::DVector::from_column_slice,
    );
    bounded_probes::<_, f32>(
        backend_aliases::nalgebra::DVector::from_column_slice,
    );
}

#[cfg(feature = "ndarray_all")]
#[test]
fn bounded_probes_ndarray() {
    bounded_probes::<_, f64>(|x| {
        backend_aliases::ndarray::Array1::from_vec(x.to_vec())
    });
    bounded_probes::<_, f32>(|x| {
        backend_aliases::ndarray::Array1::from_vec(x.to_vec())
    });
}

#[cfg(feature = "faer_all")]
#[test]
fn bounded_probes_faer() {
    bounded_probes::<_, f64>(|x| {
        backend_aliases::faer::Col::from_fn(x.len(), |i| x[i])
    });
    bounded_probes::<_, f32>(|x| {
        backend_aliases::faer::Col::from_fn(x.len(), |i| x[i])
    });
}

#[test]
fn unbounded_probes_use_in_place_evaluation() {
    for method in [Method::Forward, Method::Central] {
        let calls = Arc::new(Calls::default());
        let fd = FiniteDiff::new(Probes {
            make: |x: &[f64]| x.to_vec(),
            calls: Arc::clone(&calls),
        })
        .jacobian_method(method);
        let j = fd.jacobian(&vec![0.25, 0.75]).unwrap();
        for row in 0..3 {
            for col in 0..2 {
                let expected = if row == col || row == 2 { 1.0 } else { 0.0 };
                assert!((j.matrix_entry(row, col) - expected).abs() < 1e-8);
            }
        }
        assert_eq!(
            calls.in_place.load(Relaxed),
            if matches!(method, Method::Forward) {
                3
            } else {
                5
            }
        );
        calls.fail.store(true, Relaxed);
        assert!(matches!(
            fd.jacobian(&vec![0.25, 0.75]),
            Err("probe failed")
        ));
    }
}
