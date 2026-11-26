use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};

use basin::{
    CostFunction, DenseBackend, DenseMatrixFromFn, FiniteDiff, Gradient,
    Hessian, HessianProduct, HessianProductScratch, Jacobian, MatrixIndex,
    MatrixShape, Method, Residual, SquareMatrixFromFn, VectorIndex, VectorLen,
};

use basin::{ConstraintSolveError, NonlinearInequalityValues};

struct Quadratic<V> {
    residual: V,
    costs: AtomicUsize,
    residuals: AtomicUsize,
}

impl<V: VectorIndex<f32>> CostFunction for Quadratic<V> {
    type Param = V;
    type Output = f32;
    type Error = Infallible;

    fn cost(&self, x: &V) -> Result<f32, Infallible> {
        self.costs.fetch_add(1, Ordering::Relaxed);
        let (a, b) = (x.get_scalar(0), x.get_scalar(1));
        Ok(a * a + a * b + 2.0 * b * b)
    }
}

impl<V: Clone + VectorIndex<f32>> Residual for Quadratic<V> {
    type Param = V;
    type Output = V;
    type Error = Infallible;

    fn residual(&self, x: &V) -> Result<V, Infallible> {
        self.residuals.fetch_add(1, Ordering::Relaxed);
        let (a, b) = (x.get_scalar(0), x.get_scalar(1));
        let mut out = self.residual.clone();
        out.set_scalar(0, a * a + b);
        out.set_scalar(1, a - 2.0 * b);
        Ok(out)
    }
}

fn check_backend<V>(mut x: V)
where
    V: Clone
        + VectorLen
        + VectorIndex<f32>
        + DenseMatrixFromFn<f32, Matrix: MatrixIndex<f32>>
        + DenseBackend<f32, Matrix: MatrixIndex<f32> + SquareMatrixFromFn<f32>>
        + Send
        + Sync,
{
    x.set_scalar(0, 0.0);
    x.set_scalar(1, 0.5);
    for (method, gradient_probes, jacobian_probes, hessian_probes) in
        [(Method::Forward, 3, 3, 6), (Method::Central, 4, 4, 9)]
    {
        let problem = FiniteDiff::new_with_scalar(Quadratic {
            residual: x.clone(),
            costs: AtomicUsize::new(0),
            residuals: AtomicUsize::new(0),
        })
        .gradient_method(method)
        .jacobian_method(method)
        .hessian_method(method);
        let inner = problem.get_ref();
        let gradient = problem.gradient(&x).unwrap();
        assert_eq!(inner.costs.swap(0, Ordering::Relaxed), gradient_probes);
        for (i, expected) in [0.5, 2.0].into_iter().enumerate() {
            assert!((gradient.get_scalar(i) - expected).abs() < 0.002);
        }

        let jac = problem.jacobian(&x).unwrap();
        assert_eq!(inner.residuals.swap(0, Ordering::Relaxed), jacobian_probes);
        let (mut residual, mut jac_into) =
            problem.residual_and_jacobian(&x).unwrap();
        let fused_jacobian_probes =
            if method == Method::Forward { 3 } else { 5 };
        assert_eq!(
            inner.residuals.swap(0, Ordering::Relaxed),
            fused_jacobian_probes
        );
        problem
            .residual_and_jacobian_into(&x, &mut residual, &mut jac_into)
            .unwrap();
        assert_eq!(
            inner.residuals.swap(0, Ordering::Relaxed),
            fused_jacobian_probes
        );
        assert_eq!((jac.matrix_rows(), jac.matrix_cols()), (2, 2));
        for (i, row) in [[0.0, 1.0], [1.0, -2.0]].into_iter().enumerate() {
            for (j, expected) in row.into_iter().enumerate() {
                assert!((jac.matrix_entry(i, j) - expected).abs() < 0.002);
                assert_eq!(jac.matrix_entry(i, j), jac_into.matrix_entry(i, j));
            }
        }

        let mut hessian = problem.hessian(&x).unwrap();
        assert_eq!(inner.costs.swap(0, Ordering::Relaxed), hessian_probes);
        let mut fused_gradient = x.clone();
        let cost = problem
            .cost_and_gradient_and_hessian_into(
                &x,
                &mut fused_gradient,
                &mut hessian,
            )
            .unwrap();
        let fused_probes = if method == Method::Forward { 8 } else { 13 };
        assert_eq!(inner.costs.swap(0, Ordering::Relaxed), fused_probes);
        assert_eq!(cost, 0.5);
        for (i, row) in [[2.0, 1.0], [1.0, 4.0]].into_iter().enumerate() {
            assert_eq!(fused_gradient.get_scalar(i), gradient.get_scalar(i));
            for (j, expected) in row.into_iter().enumerate() {
                assert!((hessian.matrix_entry(i, j) - expected).abs() < 0.002);
            }
        }

        // Use a fixed step that is exactly representable for both levels of
        // differencing, so cancellation does not dominate the product check.
        let problem = problem.with_step(0.03125);
        let inner = problem.get_ref();
        let mut direction = x.clone();
        direction.set_scalar(0, 1.0);
        direction.set_scalar(1, -1.0);
        let product = problem.hessian_product(&x, &direction).unwrap();
        assert_eq!(inner.costs.swap(0, Ordering::Relaxed), 2 * gradient_probes);
        let mut out = x.clone();
        let mut scratch = HessianProductScratch::new();
        problem
            .hessian_product_with_scratch_into(
                &x,
                &direction,
                &mut out,
                &mut scratch,
            )
            .unwrap();
        assert_eq!(inner.costs.swap(0, Ordering::Relaxed), 2 * gradient_probes);
        for (i, expected) in [1.0, -3.0].into_iter().enumerate() {
            assert_eq!(product.get_scalar(i), expected);
            assert_eq!(out.get_scalar(i), expected);
        }
        direction.set_scalar(0, 0.0);
        direction.set_scalar(1, 0.0);
        problem
            .hessian_product_into(&x, &direction, &mut out)
            .unwrap();
        assert_eq!(inner.costs.load(Ordering::Relaxed), 0);
        assert_eq!(out.get_scalar(0), 0.0);
        assert_eq!(out.get_scalar(1), 0.0);
    }
}

#[test]
fn vec_f32() {
    check_backend(vec![0.0_f32; 2]);
}

#[test]
fn custom_dense_hessian_infers_scalar() {
    #[derive(Clone)]
    struct CustomVector([f32; 2]);

    impl VectorLen for CustomVector {
        fn vec_len(&self) -> usize {
            self.0.len()
        }
    }

    impl VectorIndex<f32> for CustomVector {
        fn get_scalar(&self, i: usize) -> f32 {
            self.0[i]
        }

        fn set_scalar(&mut self, i: usize, value: f32) {
            self.0[i] = value;
        }
    }

    impl DenseMatrixFromFn<f32> for CustomVector {
        type Matrix = Vec<Vec<f32>>;

        fn dense_from_fn<G: FnMut(usize, usize) -> f32>(
            rows: usize,
            cols: usize,
            mut entry: G,
        ) -> Self::Matrix {
            (0..rows)
                .map(|i| (0..cols).map(|j| entry(i, j)).collect())
                .collect()
        }
    }

    let x = CustomVector([0.0, 0.5]);
    let problem = || Quadratic {
        residual: x.clone(),
        costs: AtomicUsize::new(0),
        residuals: AtomicUsize::new(0),
    };
    let explicit = FiniteDiff::new_with_scalar(problem())
        .with_step(0.125)
        .with_dense_hessian();
    let expected = vec![vec![2.0, 1.0], vec![1.0, 4.0]];
    assert_eq!(explicit.hessian(&x).unwrap(), expected);
    let mut out = vec![];
    explicit.hessian_into(&x, &mut out).unwrap();
    assert_eq!(out, expected);
}

#[cfg(feature = "nalgebra")]
#[test]
fn nalgebra_f32() {
    use nalgebra::{DVector, SMatrix, SVector};

    check_backend(DVector::<f32>::zeros(2));
    check_backend(SVector::<f32, 2>::zeros());
    let x = SVector::<f32, 2>::new(0.0, 0.5);
    let fd = FiniteDiff::new_with_scalar(Quadratic {
        residual: x,
        costs: AtomicUsize::new(0),
        residuals: AtomicUsize::new(0),
    });
    let hessian: SMatrix<f32, 2, 2> = fd.hessian(&x).unwrap();
    assert!((hessian[(0, 1)] - 1.0).abs() < 0.002);
}

#[cfg(feature = "ndarray")]
#[test]
fn ndarray_f32() {
    check_backend(ndarray::Array1::<f32>::zeros(2));
}

#[cfg(feature = "faer")]
#[test]
fn faer_f32() {
    check_backend(faer::Col::<f32>::zeros(2));
}

#[test]
fn f32_precision_sets_probe_steps() {
    use std::sync::Mutex;

    #[derive(Default)]
    struct ProbeLog(Mutex<Vec<f32>>);

    impl CostFunction for ProbeLog {
        type Param = Vec<f32>;
        type Output = f32;
        type Error = Infallible;

        fn cost(&self, x: &Vec<f32>) -> Result<f32, Infallible> {
            self.0.lock().unwrap().push(x[0]);
            Ok(x[0] * x[0])
        }
    }

    impl Residual for ProbeLog {
        type Param = Vec<f32>;
        type Output = Vec<f32>;
        type Error = Infallible;

        fn residual(&self, x: &Vec<f32>) -> Result<Vec<f32>, Infallible> {
            self.0.lock().unwrap().push(x[0]);
            Ok(x.clone())
        }
    }

    let fd = FiniteDiff::<_, true, f32>::new_with_scalar(ProbeLog::default());
    let x = vec![0.0];
    let probes = |log: &ProbeLog, expected: &[f32]| {
        let mut actual = std::mem::take(&mut *log.0.lock().unwrap());
        actual.sort_by(f32::total_cmp);
        assert_eq!(actual, expected);
    };
    let gradient_step = f32::EPSILON.cbrt();
    fd.gradient(&x).unwrap();
    probes(fd.get_ref(), &[-gradient_step, gradient_step]);
    fd.jacobian(&x).unwrap();
    probes(fd.get_ref(), &[0.0, f32::EPSILON.sqrt()]);
    let hessian_step = f32::EPSILON.powf(0.25);
    fd.hessian(&x).unwrap();
    probes(fd.get_ref(), &[-hessian_step, 0.0, hessian_step]);

    let fd = fd.function_precision(0.0).gradient_method(Method::Forward);
    fd.gradient(&x).unwrap();
    probes(fd.get_ref(), &[0.0, f32::EPSILON.sqrt()]);

    let fd = fd.function_precision(0.015625);
    fd.jacobian(&x).unwrap();
    probes(fd.get_ref(), &[0.0, 0.125]);

    let fd = fd.with_step(0.25);
    fd.gradient(&x).unwrap();
    probes(fd.get_ref(), &[0.0, 0.25]);

    let bounded = fd.with_bounds(vec![0.0], vec![1.0]);
    bounded.gradient(&x).unwrap();
    probes(bounded.get_ref(), &[0.0, 0.25]);
}

impl NonlinearInequalityValues for Quadratic<Vec<f32>> {
    type Param = Vec<f32>;
    type Values = Vec<f32>;
    type Error = Infallible;

    fn nonlinear_inequalities(
        &self,
        x: &Vec<f32>,
    ) -> Result<Vec<f32>, ConstraintSolveError<Infallible>> {
        self.residuals.fetch_add(1, Ordering::Relaxed);
        Ok(vec![x[0] + 2.0 * x[1]])
    }
}

#[test]
fn f32_nonlinear_constraint_jacobian() {
    use basin::{Constrained, ConstraintJacobian, ConstraintSet};

    for (method, expected_probes) in
        [(Method::Forward, 3), (Method::Central, 5)]
    {
        let problem = Constrained::new(
            Quadratic {
                residual: vec![0.0_f32; 2],
                costs: AtomicUsize::new(0),
                residuals: AtomicUsize::new(0),
            },
            ConstraintSet::new().with_nonlinear_inequalities(1),
        );
        let fd = FiniteDiff::new_with_scalar(problem).jacobian_method(method);
        let jac = fd.constraint_jacobian(&vec![0.0, 0.5]).unwrap();
        assert_eq!((jac.matrix_rows(), jac.matrix_cols()), (1, 2));
        assert!((jac.get(0, 0) - 1.0).abs() < 0.002);
        assert!((jac.get(0, 1) - 2.0).abs() < 0.002);
        assert_eq!(
            fd.get_ref().inner().residuals.load(Ordering::Relaxed),
            expected_probes
        );
    }
}
