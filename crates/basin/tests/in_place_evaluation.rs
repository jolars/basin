//! Counted in-place dispatch, adapter forwarding, and reusable driver storage.

use std::sync::{Arc, Mutex};

use basin::{
    Bfgs, BoxBounds, Constrained, ConstraintSet, CostFunction,
    DenseMatrixFromFn, Executor, FirstOrderState, Gradient, GradientDescent,
    Hessian, HessianProduct, Jacobian, MatrixIndex, MatrixShape,
    MiniBatchGradient, Problem, Residual, SamplingDomain, Scalar, Sgd, State,
    VectorIndex, VectorLen,
};

#[path = "support/backend_aliases.rs"]
mod backend_aliases;

type Calls = Arc<Mutex<Vec<(&'static str, usize)>>>;

struct Probe<V, F> {
    vector: fn(&[F]) -> V,
    address: fn(&V) -> usize,
    calls: Calls,
    fail: bool,
    nonfinite: bool,
}
impl<V, F: Scalar> Probe<V, F>
where
    V: VectorLen + VectorIndex<F>,
{
    fn record(&self, name: &'static str, out: &V) -> Result<(), &'static str> {
        self.calls.lock().unwrap().push((name, (self.address)(out)));
        if self.fail {
            Err("callback failed")
        } else {
            Ok(())
        }
    }
    fn fill(&self, x: &V, out: &mut V, factor: F) {
        if out.vec_len() != x.vec_len() {
            *out = (self.vector)(&vec![F::nan(); x.vec_len()]);
        }
        for i in 0..x.vec_len() {
            out.set_scalar(
                i,
                if self.nonfinite {
                    F::nan()
                } else {
                    factor * x.get_scalar(i)
                },
            );
        }
    }
    fn value(&self, x: &V) -> F {
        if self.nonfinite {
            return F::infinity();
        }
        (0..x.vec_len())
            .map(|i| x.get_scalar(i) * x.get_scalar(i))
            .sum()
    }
}
impl<V, F: Scalar> CostFunction for Probe<V, F>
where
    V: VectorLen + VectorIndex<F>,
{
    type Param = V;
    type Output = F;
    type Error = &'static str;
    fn cost(&self, x: &V) -> Result<F, Self::Error> {
        if self.fail {
            Err("callback failed")
        } else {
            Ok(self.value(x))
        }
    }
}
impl<V, F: Scalar> Gradient for Probe<V, F>
where
    V: VectorLen + VectorIndex<F>,
{
    type Gradient = V;
    fn gradient(&self, x: &V) -> Result<V, Self::Error> {
        self.record("owned_gradient_seed", x)?;
        let mut out = (self.vector)(&[]);
        self.fill(x, &mut out, F::one() + F::one());
        Ok(out)
    }
    fn gradient_into(&self, x: &V, out: &mut V) -> Result<(), Self::Error> {
        self.record("gradient", out)?;
        self.fill(x, out, F::one() + F::one());
        Ok(())
    }
    fn cost_and_gradient(&self, x: &V) -> Result<(F, V), Self::Error> {
        let gradient = self.gradient(x)?;
        Ok((self.value(x), gradient))
    }
    fn cost_and_gradient_into(
        &self,
        x: &V,
        out: &mut V,
    ) -> Result<F, Self::Error> {
        self.record("fused_gradient", out)?;
        self.fill(x, out, F::one() + F::one());
        Ok(self.value(x))
    }
}
impl<V, F: Scalar> MiniBatchGradient for Probe<V, F>
where
    V: VectorLen + VectorIndex<F>,
{
    type Gradient = V;
    fn n_samples(&self) -> usize {
        5
    }
    fn batch_gradient(&self, x: &V, _: &[usize]) -> Result<V, Self::Error> {
        self.record("owned_batch_seed", x)?;
        let mut out = (self.vector)(&[]);
        self.fill(x, &mut out, F::one() + F::one());
        Ok(out)
    }
    fn batch_gradient_into(
        &self,
        x: &V,
        _: &[usize],
        out: &mut V,
    ) -> Result<(), Self::Error> {
        self.record("batch_gradient", out)?;
        self.fill(x, out, F::one() + F::one());
        Ok(())
    }
}
impl<V, F: Scalar> Residual for Probe<V, F>
where
    V: VectorLen + VectorIndex<F>,
{
    type Param = V;
    type Output = V;
    type Error = &'static str;
    fn residual(&self, x: &V) -> Result<V, Self::Error> {
        self.record("owned_residual_seed", x)?;
        let mut out = (self.vector)(&[]);
        self.fill(x, &mut out, F::one());
        Ok(out)
    }
    fn residual_into(&self, x: &V, out: &mut V) -> Result<(), Self::Error> {
        self.record("residual", out)?;
        self.fill(x, out, F::one());
        Ok(())
    }
}
impl<V, F: Scalar> Jacobian for Probe<V, F>
where
    V: VectorLen + VectorIndex<F> + DenseMatrixFromFn<F>,
{
    type Jacobian = V::Matrix;
    // A generic matrix has no construction capability. NLLS drivers need one
    // owned fused seed, then retain the resulting matrix for in-place refreshes.
    fn jacobian(&self, _: &V) -> Result<V::Matrix, Self::Error> {
        panic!("unexpected owned Jacobian refresh")
    }
    fn jacobian_into(
        &self,
        x: &V,
        out: &mut V::Matrix,
    ) -> Result<(), Self::Error> {
        self.record("jacobian", x)?;
        V::dense_from_fn_into(
            x.vec_len(),
            x.vec_len(),
            |i, j| if i == j { F::one() } else { F::zero() },
            out,
        );
        Ok(())
    }
    fn residual_and_jacobian(
        &self,
        x: &V,
    ) -> Result<(V, V::Matrix), Self::Error> {
        self.record("owned_fused_jacobian", x)?;
        let r = (self.vector)(
            &(0..x.vec_len())
                .map(|i| x.get_scalar(i))
                .collect::<Vec<_>>(),
        );
        let j = V::dense_from_fn(x.vec_len(), x.vec_len(), |i, j| {
            if i == j { F::one() } else { F::zero() }
        });
        Ok((r, j))
    }
    fn residual_and_jacobian_into(
        &self,
        x: &V,
        r: &mut V,
        j: &mut V::Matrix,
    ) -> Result<(), Self::Error> {
        self.record("fused_jacobian", r)?;
        self.fill(x, r, F::one());
        V::dense_from_fn_into(
            x.vec_len(),
            x.vec_len(),
            |i, j| if i == j { F::one() } else { F::zero() },
            j,
        );
        Ok(())
    }
}
impl<V, F: Scalar> Hessian for Probe<V, F>
where
    V: VectorLen + VectorIndex<F> + DenseMatrixFromFn<F>,
{
    type Hessian = V::Matrix;
    fn hessian(&self, x: &V) -> Result<V::Matrix, Self::Error> {
        self.record("owned_hessian_seed", x)?;
        Ok(V::dense_from_fn(x.vec_len(), x.vec_len(), |i, j| {
            if i == j {
                F::one() + F::one()
            } else {
                F::zero()
            }
        }))
    }
    fn hessian_into(
        &self,
        x: &V,
        out: &mut V::Matrix,
    ) -> Result<(), Self::Error> {
        self.record("hessian", x)?;
        V::dense_from_fn_into(
            x.vec_len(),
            x.vec_len(),
            |i, j| {
                if i == j {
                    F::one() + F::one()
                } else {
                    F::zero()
                }
            },
            out,
        );
        Ok(())
    }
    fn cost_and_gradient_and_hessian_into(
        &self,
        x: &V,
        g: &mut V,
        h: &mut V::Matrix,
    ) -> Result<F, Self::Error> {
        self.record("fused_hessian", g)?;
        self.fill(x, g, F::one() + F::one());
        V::dense_from_fn_into(
            x.vec_len(),
            x.vec_len(),
            |i, j| {
                if i == j {
                    F::one() + F::one()
                } else {
                    F::zero()
                }
            },
            h,
        );
        Ok(self.value(x))
    }
}
impl<V, F: Scalar> HessianProduct for Probe<V, F>
where
    V: VectorLen + VectorIndex<F>,
{
    fn hessian_product(&self, _: &V, _: &V) -> Result<V, Self::Error> {
        panic!("unexpected owned Hessian product")
    }
    fn hessian_product_into(
        &self,
        _: &V,
        v: &V,
        out: &mut V,
    ) -> Result<(), Self::Error> {
        self.record("hessian_product", out)?;
        self.fill(v, out, F::one() + F::one());
        Ok(())
    }
}

fn check_dispatch<V, M, F>(
    mut p: Problem<
        impl MiniBatchGradient<Gradient = V>
        + Jacobian<Param = V, Output = V, Error = &'static str, Jacobian = M>
        + Hessian<
            Param = V,
            Output = F,
            Gradient = V,
            Error = &'static str,
            Hessian = M,
        > + HessianProduct,
    >,
    x: &V,
    v: &mut V,
    m: &mut M,
    fail: bool,
) where
    V: VectorLen + VectorIndex<F>,
    M: MatrixIndex<F>,
    F: Scalar,
{
    let expected = if fail { Err("callback failed") } else { Ok(()) };
    assert_eq!(p.gradient_into(x, v), expected);
    assert_eq!(p.batch_gradient_into(x, &[4, 1], v), expected);
    assert_eq!(p.residual_into(x, v), expected);
    assert_eq!(p.jacobian_into(x, m), expected);
    assert_eq!(p.hessian_into(x, m), expected);
    assert_eq!(p.hessian_product_into(x, x, v), expected);
    assert_eq!(p.cost_and_gradient_into(x, v).map(|_| ()), expected);
    assert_eq!(p.residual_and_jacobian_into(x, v, m), expected);
    assert_eq!(
        p.cost_and_gradient_and_hessian_into(x, v, m).map(|_| ()),
        expected
    );
    let mut expected_counts = basin::EvalCounts::default();
    expected_counts.cost_evals = 2;
    expected_counts.gradient_evals = 4;
    expected_counts.residual_evals = 2;
    expected_counts.jacobian_evals = 2;
    expected_counts.hessian_evals = 2;
    expected_counts.hessian_product_evals = 1;
    assert_eq!(*p.counts(), expected_counts);
    if !fail {
        assert_eq!(v.vec_len(), x.vec_len());
        for i in 0..x.vec_len() {
            assert_eq!(
                v.get_scalar(i),
                (F::one() + F::one()) * x.get_scalar(i)
            );
        }
        assert_eq!(
            (m.matrix_rows(), m.matrix_cols()),
            (x.vec_len(), x.vec_len())
        );
    }
}

fn check_backend<V, F>(vector: fn(&[F]) -> V, address: fn(&V) -> usize)
where
    F: Scalar,
    V: Clone
        + VectorLen
        + VectorIndex<F>
        + DenseMatrixFromFn<F>
        + basin::DenseBackend<F>
        + basin::ScaledAdd<F>
        + basin::NegInPlace
        + basin::ScaleInPlace<F>
        + basin::Dot<F>
        + basin::NormSquared<F>,
    <V as DenseMatrixFromFn<F>>::Matrix: MatrixIndex<F>,
    <V as basin::DenseBackend<F>>::Matrix: basin::MatVec<V>
        + basin::MatrixIdentity
        + basin::ScaleInPlace<F>
        + basin::GeneralRankOneUpdate<V, F>,
{
    let f = |v| F::from_f64(v).unwrap();
    let calls = Calls::default();
    let make = |fail| Probe {
        vector,
        address,
        calls: calls.clone(),
        fail,
        nonfinite: false,
    };
    let x = vector(&[f(2.), f(-1.)]);
    for fail in [false, true] {
        for len in [0, 1, 2, 4] {
            let mut v = vector(&vec![F::nan(); len]);
            let mut m = V::dense_from_fn(len, len, |_, _| F::nan());
            check_dispatch(Problem::new(make(fail)), &x, &mut v, &mut m, fail);
            let p = Constrained::new(
                make(fail),
                ConstraintSet::new().with_bounds(BoxBounds::new(
                    Some(vector(&[f(-5.); 2])),
                    Some(vector(&[f(5.); 2])),
                )),
            );
            check_dispatch(
                Problem::new(SamplingDomain::new(p)),
                &x,
                &mut v,
                &mut m,
                fail,
            );
        }
    }
    calls.lock().unwrap().clear();
    let result = Executor::new(
        make(false),
        GradientDescent::new(f(0.1)),
        FirstOrderState::new(x.clone()),
    )
    .max_iter(5)
    .run_with_solver()
    .unwrap();
    assert_eq!(result.counts.cost_evals, 6);
    assert_eq!(result.counts.gradient_evals, 6);
    let allocations: std::collections::HashSet<_> = calls
        .lock()
        .unwrap()
        .iter()
        .filter(|(name, _)| *name == "fused_gradient")
        .map(|(_, ptr)| *ptr)
        .collect();
    assert_eq!(
        allocations.len(),
        1,
        "descent must retain its gradient allocation"
    );
    let resumed =
        Executor::resume_from_checkpoint(make(false), result.into_checkpoint())
            .max_iter(8)
            .run_with_solver()
            .unwrap();
    let full = Executor::new(
        make(false),
        GradientDescent::new(f(0.1)),
        FirstOrderState::new(x.clone()),
    )
    .max_iter(8)
    .run_with_solver()
    .unwrap();
    assert_eq!(resumed.counts, full.counts);
    for i in 0..x.vec_len() {
        assert_eq!(
            resumed.state.param().get_scalar(i),
            full.state.param().get_scalar(i)
        );
    }
    let fresh = Executor::new(
        make(false),
        resumed.solver,
        FirstOrderState::new(vector(&[f(3.), f(2.), f(1.)])),
    )
    .max_iter(2)
    .run_with_solver()
    .unwrap();
    assert_eq!(fresh.counts.gradient_evals, 3);
    assert_eq!(fresh.state.param().vec_len(), 3);
    calls.lock().unwrap().clear();
    Executor::from_start(
        make(false),
        Sgd::new(f(0.1), 2, 17).with_momentum(f(0.3)),
        x.clone(),
    )
    .max_iter(5)
    .run()
    .unwrap();
    let allocations: std::collections::HashSet<_> = calls
        .lock()
        .unwrap()
        .iter()
        .filter(|(name, _)| *name == "batch_gradient")
        .map(|(_, ptr)| *ptr)
        .collect();
    assert_eq!(
        allocations.len(),
        1,
        "SGD must retain its batch-gradient allocation"
    );
    Executor::from_start(
        make(false),
        Bfgs::with_line_search(basin::line_search::Constant(f(0.1))),
        x,
    )
    .max_iter(5)
    .run()
    .unwrap();
}

fn empty_matrix<V: DenseMatrixFromFn<F>, F: Scalar>(_: &V) -> V::Matrix {
    V::dense_from_fn(0, 0, |_, _| F::zero())
}

fn check_continuation<V, F, So>(
    make: impl Fn() -> Probe<V, F>,
    solver: impl Fn() -> So,
    start: &V,
) where
    F: Scalar,
    V: Clone + VectorLen + VectorIndex<F>,
    So: basin::Solver<Probe<V, F>, FirstOrderState<V, F>, Error = &'static str>,
{
    let expected =
        Executor::new(make(), solver(), FirstOrderState::new(start.clone()))
            .max_iter(5)
            .run_with_solver()
            .unwrap();
    for split in [0, 1, 3] {
        let initial = Executor::new(
            make(),
            solver(),
            FirstOrderState::new(start.clone()),
        )
        .max_iter(split)
        .run_with_solver()
        .unwrap();
        let resumed =
            Executor::resume_from_checkpoint(make(), initial.into_checkpoint())
                .max_iter(5)
                .run_with_solver()
                .unwrap();
        assert_eq!(resumed.counts, expected.counts);
        assert_eq!(resumed.state.cost(), expected.state.cost());
        for i in 0..start.vec_len() {
            assert_eq!(
                resumed.state.param().get_scalar(i),
                expected.state.param().get_scalar(i)
            );
        }
    }
}

fn check_unbounded_fd<V>(vector: fn(&[f64]) -> V, address: fn(&V) -> usize)
where
    V: Clone + VectorLen + VectorIndex + DenseMatrixFromFn + Send + Sync,
    V::Matrix: MatrixIndex,
{
    let make = || Probe {
        vector,
        address,
        calls: Calls::default(),
        fail: false,
        nonfinite: false,
    };
    for method in [basin::Method::Forward, basin::Method::Central] {
        let fd = basin::FiniteDiff::new(make())
            .gradient_method(method)
            .jacobian_method(method)
            .hessian_method(method)
            .with_step(1e-3);
        let mut gradient = vector(&[f64::NAN]);
        let mut jacobian = empty_matrix(&gradient);
        let mut hessian = empty_matrix(&gradient);
        for values in [&[2.0, -1.0][..], &[3.0, -2.0, 1.0][..]] {
            let x = vector(values);
            fd.gradient_into(&x, &mut gradient).unwrap();
            let allocation = address(&gradient);
            fd.cost_and_gradient_into(&x, &mut gradient).unwrap();
            assert_eq!(address(&gradient), allocation);
            for (i, value) in values.iter().enumerate() {
                assert!((gradient.get_scalar(i) - 2.0 * value).abs() < 0.002);
            }
            fd.jacobian_into(&x, &mut jacobian).unwrap();
            fd.cost_and_gradient_and_hessian_into(
                &x,
                &mut gradient,
                &mut hessian,
            )
            .unwrap();
            assert_eq!(
                (jacobian.matrix_rows(), jacobian.matrix_cols()),
                (values.len(), values.len())
            );
            assert_eq!(
                (hessian.matrix_rows(), hessian.matrix_cols()),
                (values.len(), values.len())
            );
            for i in 0..values.len() {
                for j in 0..values.len() {
                    assert!(
                        (jacobian.matrix_entry(i, j)
                            - if i == j { 1.0 } else { 0.0 })
                        .abs()
                            < 1e-10
                    );
                    assert!(
                        (hessian.matrix_entry(i, j)
                            - if i == j { 2.0 } else { 0.0 })
                        .abs()
                            < 1e-6
                    );
                }
            }
            let hv = if matches!(method, basin::Method::Forward) {
                basin::core::numdiff::forward_difference_hessian_product(
                    &make(),
                    &x,
                    &x,
                    f64::EPSILON,
                    Some(1e-3),
                )
                .unwrap()
            } else {
                basin::core::numdiff::central_difference_hessian_product(
                    &make(),
                    &x,
                    &x,
                    f64::EPSILON,
                    Some(1e-3),
                )
                .unwrap()
            };
            for (i, value) in values.iter().enumerate() {
                assert!((hv.get_scalar(i) - 2.0 * value).abs() < 1e-10);
            }
        }
        let mut counted = Problem::new(fd);
        counted
            .gradient_into(&vector(&[2.0, -1.0]), &mut gradient)
            .unwrap();
        assert_eq!(counted.counts().gradient_evals, 1);
        assert_eq!(counted.counts().cost_evals, 0);
    }
}
macro_rules! unbounded_fd_test {
    (f64,$vector:expr,$address:expr) => {
        check_unbounded_fd($vector, $address);
    };
    (f32,$vector:expr,$address:expr) => {};
}

macro_rules! backend_test {
    ($name:ident,$scalar:ident,$vector:expr,$address:expr) => {
        #[test]
        fn $name() {
            check_backend::<_, $scalar>($vector, $address);
            let vector: fn(&[$scalar]) -> _ = $vector;
            let address: fn(&_) -> usize = $address;
            let calls = Calls::default();
            let make = || Probe {
                vector,
                address,
                calls: calls.clone(),
                fail: false,
                nonfinite: false,
            };
            let start = vector(&[2.0, -1.0]);
            let mut poisoned = make();
            poisoned.nonfinite = true;
            let invalid = Executor::from_start(
                poisoned,
                basin::NonlinearCg::new(),
                start.clone(),
            )
            .max_iter(2)
            .run_with_solver()
            .unwrap();
            assert_eq!(invalid.counts.cost_evals, 1);
            assert_eq!(invalid.counts.gradient_evals, 1);
            assert_eq!(
                invalid.report.termination.code(),
                basin::TerminationCode::SolverFailed
            );
            check_continuation(
                make,
                || {
                    basin::NonlinearCg::with_line_search(
                        basin::line_search::Constant(0.1),
                    )
                },
                &start,
            );
            check_continuation(
                make,
                || Bfgs::with_line_search(basin::line_search::Constant(0.1)),
                &start,
            );
            check_continuation(
                make,
                || {
                    basin::Lbfgs::with_line_search(
                        basin::line_search::Constant(0.1),
                    )
                    .unbounded()
                },
                &start,
            );
            unbounded_fd_test!($scalar, vector, address);

            Executor::from_start(
                make(),
                basin::NonlinearCg::new(),
                start.clone(),
            )
            .max_iter(4)
            .run()
            .unwrap();
            Executor::from_start(
                make(),
                basin::Lbfgs::new().unbounded(),
                start.clone(),
            )
            .max_iter(4)
            .run()
            .unwrap();
            Executor::from_start(
                make(),
                basin::TrustRegion::<
                    _,
                    $scalar,
                    basin::solver::trust_region::MatrixFree,
                >::matrix_free_with(basin::Steihaug::new()),
                start.clone(),
            )
            .max_iter(4)
            .run()
            .unwrap();
            let bounds = || {
                ConstraintSet::new().with_bounds(BoxBounds::new(
                    Some(vector(&[-5.0; 2])),
                    Some(vector(&[5.0; 2])),
                ))
            };
            Executor::from_start(
                Constrained::new(make(), bounds()),
                basin::Lbfgs::new(),
                start.clone(),
            )
            .max_iter(4)
            .run()
            .unwrap();
            Executor::from_start(
                Constrained::new(make(), bounds()),
                basin::ProjectedGradientDescent::new(0.1),
                start.clone(),
            )
            .max_iter(4)
            .run()
            .unwrap();

            calls.lock().unwrap().clear();
            let gn = Executor::from_start(
                make(),
                basin::GaussNewton::<_, _, $scalar>::default(),
                start.clone(),
            )
            .max_iter(3)
            .run_with_solver()
            .unwrap();
            assert_eq!(
                calls
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|(name, _)| *name == "owned_fused_jacobian")
                    .count(),
                1
            );
            assert!(
                calls
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|(name, _)| *name == "jacobian")
            );
            calls.lock().unwrap().clear();
            let fresh = Executor::from_start(
                make(),
                gn.solver,
                vector(&[3.0, 2.0, 1.0]),
            )
            .max_iter(0)
            .run_with_solver()
            .unwrap();
            assert_eq!(fresh.counts.residual_evals, 1);
            assert_eq!(fresh.counts.jacobian_evals, 1);
            assert_eq!(fresh.state.param().vec_len(), 3);
            assert_eq!(
                calls.lock().unwrap().as_slice().first().unwrap().0,
                "fused_jacobian"
            );
            calls.lock().unwrap().clear();
            Executor::from_start(
                make(),
                basin::LevenbergMarquardt::<_, _, $scalar>::default(),
                start.clone(),
            )
            .max_iter(3)
            .run()
            .unwrap();
            assert_eq!(
                calls
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|(name, _)| *name == "owned_fused_jacobian")
                    .count(),
                1
            );
            assert!(
                calls
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|(name, _)| *name == "jacobian")
            );
            calls.lock().unwrap().clear();
            Executor::from_start(
                basin::RobustLeastSquares::new(make(), basin::SquaredLoss),
                basin::LevenbergMarquardt::<_, _, $scalar>::default(),
                start.clone(),
            )
            .max_iter(3)
            .run()
            .unwrap();
            assert_eq!(
                calls
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|(name, _)| *name == "owned_fused_jacobian")
                    .count(),
                1
            );
            assert!(
                calls
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|(name, _)| *name == "jacobian")
            );
            Executor::from_start(
                Constrained::new(make(), bounds()),
                basin::BoundedLevenbergMarquardt::<_, _, $scalar>::default(),
                start.clone(),
            )
            .max_iter(3)
            .run()
            .unwrap();

            calls.lock().unwrap().clear();
            Executor::from_start(
                make(),
                basin::LevenbergMarquardt::<_, _, $scalar>::default()
                    .with_pivoted_qr()
                    .with_evaluation_buffers(),
                start.clone(),
            )
            .max_iter(3)
            .run()
            .unwrap();
            assert_eq!(
                calls
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|(name, _)| *name == "owned_fused_jacobian")
                    .count(),
                1
            );
            assert!(
                calls
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|(name, _)| *name == "jacobian")
            );

            calls.lock().unwrap().clear();
            let tr = Executor::from_start(
                make(),
                basin::TrustRegion::<_, $scalar>::with_subproblem(
                    basin::Steihaug::new(),
                )
                .with_radius(0.1)
                .with_evaluation_buffers(),
                start.clone(),
            )
            .max_iter(3)
            .run_with_solver()
            .unwrap();
            assert_eq!(
                calls
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|(name, _)| *name == "owned_hessian_seed")
                    .count(),
                1
            );
            assert!(
                calls
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|(name, _)| *name == "hessian")
            );
            calls.lock().unwrap().clear();
            Executor::from_start(make(), tr.solver, vector(&[3.0, 2.0, 1.0]))
                .max_iter(1)
                .run()
                .unwrap();
            assert!(
                calls
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|(name, _)| *name == "hessian")
            );
            assert!(
                !calls
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|(name, _)| *name == "owned_hessian_seed")
            );

            calls.lock().unwrap().clear();
            let trf = Executor::from_start(
                Constrained::new(make(), bounds()),
                basin::TrustRegionReflective::<$scalar>::new()
                    .with_evaluation_buffers(),
                start.clone(),
            )
            .max_iter(3)
            .run_with_solver()
            .unwrap();
            assert_eq!(
                calls
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|(name, _)| *name == "owned_fused_jacobian")
                    .count(),
                1
            );
            assert!(
                calls
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|(name, _)| *name == "jacobian")
            );
            calls.lock().unwrap().clear();
            let resized = Executor::from_start(
                Constrained::new(
                    make(),
                    ConstraintSet::new().with_bounds(BoxBounds::new(
                        Some(vector(&[-5.0; 3])),
                        Some(vector(&[5.0; 3])),
                    )),
                ),
                trf.solver,
                vector(&[3.0, 2.0, 1.0]),
            )
            .max_iter(0)
            .run()
            .unwrap();
            assert_eq!(resized.counts.residual_evals, 1);
            assert_eq!(resized.counts.jacobian_evals, 1);
            assert_eq!(calls.lock().unwrap()[0].0, "fused_jacobian");

            Executor::from_start(
                Constrained::new(make(), bounds()),
                basin::Slsqp::<$scalar>::new().with_evaluation_buffers(),
                start.clone(),
            )
            .max_iter(4)
            .run()
            .unwrap();

            calls.lock().unwrap().clear();
            let objective = basin::RobustLeastSquares::<_, _, $scalar>::new(
                make(),
                basin::SquaredLoss,
            )
            .with_evaluation_buffers();
            let mut objective = Problem::new(objective);
            let mut gradient = vector(&[<$scalar>::NAN]);
            objective
                .cost_and_gradient_into(&start, &mut gradient)
                .unwrap();
            let pointer = address(&gradient);
            let value = objective
                .cost_and_gradient_into(&start, &mut gradient)
                .unwrap();
            assert_eq!(address(&gradient), pointer);
            assert_eq!(value, 2.5);
            assert_eq!(gradient.get_scalar(0), 2.0);
            assert_eq!(gradient.get_scalar(1), -1.0);
            objective
                .gradient_into(&vector(&[3.0, 2.0, 1.0]), &mut gradient)
                .unwrap();
            assert_eq!(gradient.vec_len(), 3);
            assert_eq!(objective.counts().cost_evals, 2);
            assert_eq!(objective.counts().gradient_evals, 3);
            assert_eq!(
                calls
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|(name, _)| *name == "owned_fused_jacobian")
                    .count(),
                1
            );
            assert_eq!(
                calls
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|(name, _)| *name == "fused_jacobian")
                    .count(),
                2
            );
            objective.cost(&start).unwrap();
            assert_eq!(calls.lock().unwrap().last().unwrap().0, "residual");

            // The bounded adapter is scalar-generic and must write into
            // caller storage rather than allocating an owned derivative.
            let fd = basin::BoundedFiniteDiff::new(
                make(),
                vector(&[-5.0; 2]),
                vector(&[5.0; 2]),
            )
            .with_step(0.01);
            let mut gradient = vector(&[<$scalar>::NAN]);
            fd.gradient_into(&start, &mut gradient).unwrap();
            assert_eq!(gradient.vec_len(), 2);
            assert!((gradient.get_scalar(0) - 4.0).abs() < 0.002);
            assert!((gradient.get_scalar(1) + 2.0).abs() < 0.002);
            let allocation = address(&gradient);
            fd.gradient_into(&start, &mut gradient).unwrap();
            assert_eq!(address(&gradient), allocation);
            let mut jacobian = empty_matrix(&start);
            fd.jacobian_into(&start, &mut jacobian).unwrap();
            assert!((jacobian.matrix_entry(0, 0) - 1.0).abs() < 0.002);
            assert!(jacobian.matrix_entry(0, 1).abs() < 0.002);
        }
    };
}
backend_test!(vec_f32, f32, |x| x.to_vec(), |x: &Vec<f32>| x.as_ptr()
    as usize);
backend_test!(vec_f64, f64, |x| x.to_vec(), |x: &Vec<f64>| x.as_ptr()
    as usize);
#[cfg(feature = "nalgebra_all")]
mod nalgebra {
    use super::*;
    use backend_aliases::nalgebra::DVector;
    backend_test!(f32, f32, DVector::from_column_slice, |x: &DVector<f32>| x
        .as_ptr()
        as usize);
    backend_test!(f64, f64, DVector::from_column_slice, |x: &DVector<f64>| x
        .as_ptr()
        as usize);
}
#[cfg(feature = "ndarray_all")]
mod ndarray {
    use super::*;
    use backend_aliases::ndarray::Array1;
    backend_test!(f32, f32, |x| Array1::from_vec(x.to_vec()), |x: &Array1<
        f32,
    >| x.as_ptr()
        as usize);
    backend_test!(f64, f64, |x| Array1::from_vec(x.to_vec()), |x: &Array1<
        f64,
    >| x.as_ptr()
        as usize);
}
#[cfg(feature = "faer_all")]
mod faer {
    use super::*;
    use backend_aliases::faer::Col;
    backend_test!(f32, f32, |x| Col::from_fn(x.len(), |i| x[i]), |x: &Col<
        f32,
    >| x
        .as_ptr()
        as usize);
    backend_test!(f64, f64, |x| Col::from_fn(x.len(), |i| x[i]), |x: &Col<
        f64,
    >| x
        .as_ptr()
        as usize);
}

fn vec_probe() -> Probe<Vec<f64>, f64> {
    Probe {
        vector: |x| x.to_vec(),
        address: |x| x.as_ptr() as usize,
        calls: Calls::default(),
        fail: false,
        nonfinite: false,
    }
}

fn check_buffer_checkpoint<P, So, S>(
    problem: impl Fn() -> P,
    solver: impl Fn() -> So,
) where
    S: State<Param = Vec<f64>, Float = f64> + basin::CountsMirror,
    So: basin::Solver<P, S> + basin::InitialState<Vec<f64>, State = S>,
    So::Error: std::fmt::Debug,
{
    let expected = Executor::from_start(problem(), solver(), vec![2.0, -1.0])
        .max_iter(5)
        .run_with_solver()
        .unwrap();
    for split in [0, 1, 3] {
        let part = Executor::from_start(problem(), solver(), vec![2.0, -1.0])
            .max_iter(split)
            .run_with_solver()
            .unwrap();
        let resumed =
            Executor::resume_from_checkpoint(problem(), part.into_checkpoint())
                .max_iter(5)
                .run_with_solver()
                .unwrap();
        assert_eq!(resumed.counts, expected.counts);
        assert_eq!(resumed.state.param(), expected.state.param());
        assert_eq!(resumed.state.cost(), expected.state.cost());
        assert_eq!(resumed.report.code(), expected.report.code());
    }
}

#[test]
fn opt_in_storage_preserves_exact_continuation() {
    check_buffer_checkpoint(vec_probe, || {
        basin::TrustRegion::new()
            .with_radius(0.1)
            .with_evaluation_buffers()
    });
    check_buffer_checkpoint(vec_probe, || {
        basin::LevenbergMarquardt::new()
            .with_initial_step_bound(0.01)
            .with_pivoted_qr()
            .with_evaluation_buffers()
    });
    let bounded = || {
        Constrained::new(
            vec_probe(),
            ConstraintSet::new().with_bounds(BoxBounds::new(
                Some(vec![-5.0; 2]),
                Some(vec![5.0; 2]),
            )),
        )
    };
    check_buffer_checkpoint(bounded, || {
        basin::TrustRegionReflective::new()
            .with_initial_radius(0.01)
            .with_evaluation_buffers()
    });
    check_buffer_checkpoint(bounded, || {
        basin::Slsqp::new().with_evaluation_buffers()
    });
    check_buffer_checkpoint(
        || {
            basin::RobustLeastSquares::new(vec_probe(), basin::SoftL1Loss)
                .with_evaluation_buffers()
        },
        || basin::GradientDescent::new(0.1),
    );
    check_buffer_checkpoint(
        || {
            basin::RobustLeastSquares::new(bounded(), basin::SoftL1Loss)
                .with_evaluation_buffers()
        },
        || {
            basin::TrustRegionReflective::new()
                .with_initial_radius(0.01)
                .with_evaluation_buffers()
        },
    );
}

#[test]
fn default_types_keep_original_auto_traits() {
    struct NonSendMatrix(std::marker::PhantomData<std::rc::Rc<()>>);
    impl basin::FactorizePivotedQr<Vec<f64>> for NonSendMatrix {
        type Factorization = basin::QrFactorization;
        fn factorize_pivoted_qr(
            &self,
            _: &Vec<f64>,
        ) -> Result<Self::Factorization, basin::QrSolveError> {
            unreachable!()
        }
    }
    fn send_sync<T: Send + Sync>() {}
    fn copy<T: Copy>() {}
    send_sync::<basin::LevenbergMarquardtQr<Vec<f64>, NonSendMatrix>>();
    send_sync::<basin::TrustRegion>();
    send_sync::<basin::TrustRegionReflective>();
    send_sync::<basin::Slsqp>();
    copy::<basin::RobustLeastSquares<(), basin::SquaredLoss>>();
    send_sync::<basin::RobustLeastSquares<(), basin::SquaredLoss>>();
    send_sync::<basin::SharedEvaluationBuffers<Vec<f64>, basin::DenseMatrix>>();
}

#[test]
fn buffered_robust_preserves_errors_and_nonfinite_gradients() {
    let mut probe = vec_probe();
    probe.fail = true;
    let objective = basin::RobustLeastSquares::new(probe, basin::SquaredLoss)
        .with_evaluation_buffers();
    let mut problem = Problem::new(objective);
    let mut out = vec![0.0; 2];
    assert_eq!(
        problem.cost_and_gradient_into(&vec![2.0, -1.0], &mut out),
        Err("callback failed")
    );
    assert_eq!(problem.counts().cost_evals, 1);
    assert_eq!(problem.counts().gradient_evals, 1);

    let mut probe = vec_probe();
    probe.nonfinite = true;
    let objective = basin::RobustLeastSquares::new(probe, basin::SquaredLoss)
        .with_evaluation_buffers();
    let value = objective
        .cost_and_gradient_into(&vec![2.0, -1.0], &mut out)
        .unwrap();
    // The owned fused seed is finite; the override produces non-finite residuals.
    assert!(value.is_finite());
    let value = objective
        .cost_and_gradient_into(&vec![2.0, -1.0], &mut out)
        .unwrap();
    assert_eq!(value, f64::INFINITY);
    assert!(out.iter().all(|v| v.is_nan()));
}

#[cfg(feature = "serde")]
#[test]
fn typed_hessian_buffer_survives_serialized_checkpoint() {
    type Solver = basin::TrustRegion<
        basin::Steihaug,
        f64,
        basin::solver::trust_region::ExactHessian,
        basin::EvaluationBuffers<(), basin::DenseMatrix>,
    >;
    type Checkpoint = basin::ExactCheckpoint<Solver, FirstOrderState<Vec<f64>>>;
    let solver = || {
        basin::TrustRegion::new()
            .with_radius(0.1)
            .with_evaluation_buffers()
    };
    let part = Executor::from_start(vec_probe(), solver(), vec![2.0, -1.0])
        .max_iter(2)
        .run_with_solver()
        .unwrap();
    let bytes = postcard::to_allocvec(&part.into_checkpoint()).unwrap();
    let checkpoint: Checkpoint = postcard::from_bytes(&bytes).unwrap();
    let resumed = Executor::resume_from_checkpoint(vec_probe(), checkpoint)
        .max_iter(5)
        .run_with_solver()
        .unwrap();
    let full = Executor::from_start(vec_probe(), solver(), vec![2.0, -1.0])
        .max_iter(5)
        .run_with_solver()
        .unwrap();
    assert_eq!(resumed.state.param(), full.state.param());
    assert_eq!(resumed.counts, full.counts);
}
