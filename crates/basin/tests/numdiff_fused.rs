use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use basin::{
    BoundedFiniteDiff, CostFunction, DenseMatrixFromFn, FiniteDiff, Gradient,
    Hessian, MatrixIndex, MatrixShape, Method, Problem, Scalar, VectorIndex,
    VectorLen,
};

struct Quadratic<V, F> {
    base: V,
    bounds: Option<(Vec<F>, Vec<F>)>,
    calls: AtomicUsize,
    base_calls: AtomicUsize,
}

impl<V, F> Quadratic<V, F> {
    fn new(base: V, bounds: Option<(Vec<F>, Vec<F>)>) -> Self {
        Self {
            base,
            bounds,
            calls: AtomicUsize::new(0),
            base_calls: AtomicUsize::new(0),
        }
    }

    fn reset(&self) {
        self.calls.store(0, Relaxed);
        self.base_calls.store(0, Relaxed);
    }

    fn check_calls(&self, expected: usize) {
        assert_eq!(self.calls.load(Relaxed), expected);
        assert_eq!(self.base_calls.load(Relaxed), 1);
    }
}

impl<V: VectorLen + VectorIndex<F>, F: Scalar> CostFunction
    for Quadratic<V, F>
{
    type Param = V;
    type Output = F;
    type Error = &'static str;

    fn cost(&self, x: &V) -> Result<F, Self::Error> {
        self.calls.fetch_add(1, Relaxed);
        if (0..x.vec_len()).all(|j| x.get_scalar(j) == self.base.get_scalar(j))
        {
            self.base_calls.fetch_add(1, Relaxed);
        }
        if let Some((lo, hi)) = &self.bounds {
            for j in 0..x.vec_len() {
                if x.get_scalar(j) < lo[j] || x.get_scalar(j) > hi[j] {
                    return Err("outside domain");
                }
            }
        }
        let sum: F = (0..x.vec_len()).map(|j| x.get_scalar(j)).sum();
        Ok(sum * sum
            + (0..x.vec_len())
                .map(|j| {
                    let value = x.get_scalar(j);
                    F::from_usize(j + 1).unwrap() * value * value
                })
                .sum::<F>())
    }
}

fn same_scalar<F: Scalar>(actual: F, expected: F) {
    assert_eq!(
        actual.to_f64().unwrap().to_bits(),
        expected.to_f64().unwrap().to_bits(),
    );
}

fn same_vector<V: VectorLen + VectorIndex<F>, F: Scalar>(a: &V, b: &V) {
    assert_eq!(a.vec_len(), b.vec_len());
    for j in 0..a.vec_len() {
        same_scalar(a.get_scalar(j), b.get_scalar(j));
    }
}

fn check_unbounded<V>(make: fn(&[f64]) -> V)
where
    V: Clone + VectorLen + VectorIndex + DenseMatrixFromFn + Send + Sync,
    V::Matrix: MatrixIndex,
{
    for values in [&[][..], &[0.25][..], &[-0.75, 0.25, 0.0, 1.25][..]] {
        let x = make(values);
        let n = values.len();
        for step in [None, Some(1e-3)] {
            for gradient_method in [Method::Forward, Method::Central] {
                for hessian_method in [Method::Forward, Method::Central] {
                    let mut fd =
                        FiniteDiff::new(Quadratic::new(x.clone(), None))
                            .gradient_method(gradient_method)
                            .hessian_method(hessian_method);
                    if let Some(step) = step {
                        fd = fd.with_step(step);
                    }
                    let cost = fd.cost(&x).unwrap();
                    let grad = fd.gradient(&x).unwrap();
                    let hess = fd.hessian(&x).unwrap();
                    let gradient_probes = match gradient_method {
                        Method::Forward => n,
                        Method::Central => 2 * n,
                        _ => unreachable!(),
                    };
                    let hessian_probes = match hessian_method {
                        Method::Forward => n + n * (n + 1) / 2,
                        Method::Central => 2 * n * n,
                        _ => unreachable!(),
                    };
                    for j in 0..n {
                        let expected = 2.0 * (j + 1) as f64 * values[j]
                            + 2.0 * values.iter().sum::<f64>();
                        assert!((grad.get_scalar(j) - expected).abs() < 0.02);
                        for i in 0..n {
                            let expected = if i == j {
                                2.0 * (j + 1) as f64 + 2.0
                            } else {
                                2.0
                            };
                            assert!(
                                (hess.matrix_entry(i, j) - expected).abs()
                                    < 1e-6
                            );
                        }
                    }
                    fd.get_ref().reset();
                    let (f, g) = fd.cost_and_gradient(&x).unwrap();
                    fd.get_ref().check_calls(1 + gradient_probes);
                    same_scalar(f, cost);
                    same_vector(&g, &grad);

                    fd.get_ref().reset();
                    let (f, g, h) =
                        fd.cost_and_gradient_and_hessian(&x).unwrap();
                    fd.get_ref()
                        .check_calls(1 + gradient_probes + hessian_probes);
                    same_scalar(f, cost);
                    same_vector(&g, &grad);
                    check_hessian::<V>(&h, &hess, n);

                    // Exercise replacement of wrong shapes and reuse the resized outputs.
                    for size in [0, n + 1, n] {
                        let mut g = make(&vec![99.0; size]);
                        let mut h = V::dense_from_fn(size, size, |_, _| 99.0);
                        fd.get_ref().reset();
                        let f = fd.cost_and_gradient_into(&x, &mut g).unwrap();
                        fd.get_ref().check_calls(1 + gradient_probes);
                        same_scalar(f, cost);
                        same_vector(&g, &grad);

                        fd.get_ref().reset();
                        let f = fd
                            .cost_and_gradient_and_hessian_into(
                                &x, &mut g, &mut h,
                            )
                            .unwrap();
                        fd.get_ref()
                            .check_calls(1 + gradient_probes + hessian_probes);
                        same_scalar(f, cost);
                        same_vector(&g, &grad);
                        check_hessian::<V>(&h, &hess, n);

                        fd.get_ref().reset();
                        fd.cost_and_gradient_and_hessian_into(
                            &x, &mut g, &mut h,
                        )
                        .unwrap();
                        fd.get_ref()
                            .check_calls(1 + gradient_probes + hessian_probes);
                        same_vector(&g, &grad);
                        check_hessian::<V>(&h, &hess, n);
                    }
                }
            }
        }
    }
}

fn check_hessian<V: DenseMatrixFromFn>(
    actual: &V::Matrix,
    expected: &V::Matrix,
    n: usize,
) where
    V::Matrix: MatrixIndex,
{
    assert_eq!((actual.matrix_rows(), actual.matrix_cols()), (n, n));
    for i in 0..n {
        for j in 0..n {
            same_scalar(actual.matrix_entry(i, j), expected.matrix_entry(i, j));
            same_scalar(actual.matrix_entry(i, j), actual.matrix_entry(j, i));
        }
    }
}

fn check_bounded<V, F>(make: fn(&[F]) -> V)
where
    V: Clone + VectorLen + VectorIndex<F> + Send + Sync,
    F: Scalar + Send + Sync,
{
    let zero = F::zero();
    let one = F::one();
    let half = F::from_f64(0.5).unwrap();
    // An adjacent-float interval forces the central stencil to fall back to one probe.
    let adjacent = one + F::epsilon();
    for (values, lower, upper) in [
        (vec![], vec![], vec![]),
        (vec![one], vec![one], vec![one]),
        (vec![one], vec![one], vec![adjacent]),
        (vec![adjacent], vec![one], vec![adjacent]),
        (
            vec![zero, one, half, half],
            vec![zero; 4],
            vec![one, one, half, one],
        ),
    ] {
        let x = make(&values);
        for method in [Method::Forward, Method::Central] {
            for step in [None, Some(F::from_f64(1e-3).unwrap())] {
                let mut fd = BoundedFiniteDiff::new(
                    Quadratic::new(
                        x.clone(),
                        Some((lower.clone(), upper.clone())),
                    ),
                    make(&lower),
                    make(&upper),
                )
                .gradient_method(method);
                if let Some(step) = step {
                    fd = fd.with_step(step);
                }
                let cost = fd.cost(&x).unwrap();
                fd.get_ref().reset();
                let expected = fd.gradient(&x).unwrap();
                let calls = fd.get_ref().calls.load(Relaxed);
                let free = lower
                    .iter()
                    .zip(&upper)
                    .filter(|(lo, hi)| lo != hi)
                    .count();
                let probes = if method == Method::Forward || values.len() == 1 {
                    free
                } else {
                    2 * free
                };
                assert_eq!(calls, 1 + probes);
                fd.get_ref().reset();
                let (f, g) = fd.cost_and_gradient(&x).unwrap();
                fd.get_ref().check_calls(calls);
                same_scalar(f, cost);
                same_vector(&g, &expected);

                let mut out = make(&vec![one; values.len() + 1]);
                for _ in 0..2 {
                    fd.get_ref().reset();
                    let f = fd.cost_and_gradient_into(&x, &mut out).unwrap();
                    fd.get_ref().check_calls(calls);
                    same_scalar(f, cost);
                    same_vector(&out, &expected);
                }
                for j in 0..values.len() {
                    if lower[j] == upper[j] {
                        same_scalar(expected.get_scalar(j), zero);
                    } else {
                        let analytic = F::from_usize(2 * (j + 1)).unwrap()
                            * values[j]
                            + F::from_f64(2.0).unwrap()
                                * values.iter().copied().sum::<F>();
                        assert!(
                            (expected.get_scalar(j) - analytic)
                                .abs()
                                .to_f64()
                                .unwrap()
                                < 0.03
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn vec_fused_derivatives() {
    check_unbounded::<Vec<f64>>(|x| x.to_vec());
    check_bounded::<Vec<f64>, f64>(|x| x.to_vec());
    check_bounded::<Vec<f32>, f32>(|x| x.to_vec());
}

fn check_gradient_staging<V, F, P>(
    make: fn(&[F]) -> V,
    wrap: impl Fn(Quadratic<V, F>, usize, Method) -> P,
) where
    V: Clone + VectorLen + VectorIndex<F>,
    F: Scalar,
    P: Gradient<Param = V, Output = F, Gradient = V, Error = &'static str>,
{
    let quarter = F::from_f64(0.25).unwrap();
    let step = F::from_f64(0.125).unwrap();
    for n in [0, 1, 4, 31, 32, 33, 256, 257] {
        let x = make(&vec![quarter; n]);
        for method in [Method::Forward, Method::Central] {
            for fail in [false, true] {
                let mut upper = vec![F::one(); n];
                if fail && n > 0 {
                    // Earlier coordinates succeed before the last probe fails.
                    upper[n - 1] = quarter;
                }
                let problem = Quadratic::new(
                    x.clone(),
                    Some((vec![F::zero(); n], upper)),
                );
                let p = wrap(problem, n, method);
                for size in [0, n, n + 1] {
                    let initial = make(&vec![F::from_f64(99.0).unwrap(); size]);
                    let mut out = initial.clone();
                    for _ in 0..2 {
                        for fused in [false, true] {
                            let result = if fused {
                                p.cost_and_gradient_into(&x, &mut out)
                                    .map(|_| ())
                            } else {
                                p.gradient_into(&x, &mut out)
                            };
                            if fail && n > 0 {
                                assert_eq!(result, Err("outside domain"));
                                same_vector(&out, &initial);
                            } else {
                                result.unwrap();
                                assert_eq!(out.vec_len(), n);
                                for j in 0..n {
                                    let mut expected = F::from_usize(n + j + 1)
                                        .unwrap()
                                        / (F::one() + F::one());
                                    if method == Method::Forward {
                                        expected = expected
                                            + F::from_usize(j + 2).unwrap()
                                                * step;
                                    }
                                    same_scalar(out.get_scalar(j), expected);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn check_unbounded_gradient_staging<V>(make: fn(&[f64]) -> V)
where
    V: Clone + VectorLen + VectorIndex + Send + Sync,
{
    check_gradient_staging(make, |problem, _, method| {
        FiniteDiff::new(problem)
            .gradient_method(method)
            .with_step(0.125)
    });
}

fn check_bounded_gradient_staging<V, F>(make: fn(&[F]) -> V)
where
    V: Clone + VectorLen + VectorIndex<F> + Send + Sync,
    F: Scalar + Send + Sync,
{
    check_gradient_staging(make, |problem, n, method| {
        BoundedFiniteDiff::new(
            problem,
            make(&vec![F::zero(); n]),
            make(&vec![F::one(); n]),
        )
        .gradient_method(method)
        .with_step(F::from_f64(0.125).unwrap())
    });
}

#[test]
fn vec_gradient_staging() {
    check_unbounded_gradient_staging(|x| x.to_vec());
    check_bounded_gradient_staging(|x: &[f64]| x.to_vec());
    check_bounded_gradient_staging(|x: &[f32]| x.to_vec());
}

macro_rules! backend_tests {
    ($name:ident, $feature:literal, $make64:expr, $make32:expr) => {
        #[cfg(feature = $feature)]
        #[test]
        fn $name() {
            let make64: fn(&[f64]) -> _ = $make64;
            check_unbounded(make64);
            check_bounded(make64);
            check_bounded($make32);
            check_unbounded_gradient_staging(make64);
            check_bounded_gradient_staging(make64);
            check_bounded_gradient_staging($make32);
        }
    };
}

backend_tests!(
    nalgebra_032,
    "nalgebra_v0_32",
    nalgebra_0_32::DVector::<f64>::from_column_slice,
    nalgebra_0_32::DVector::<f32>::from_column_slice
);
backend_tests!(
    nalgebra_033,
    "nalgebra_v0_33",
    nalgebra_0_33::DVector::<f64>::from_column_slice,
    nalgebra_0_33::DVector::<f32>::from_column_slice
);
backend_tests!(
    nalgebra_034,
    "nalgebra_v0_34",
    nalgebra_0_34::DVector::<f64>::from_column_slice,
    nalgebra_0_34::DVector::<f32>::from_column_slice
);
backend_tests!(
    nalgebra_035,
    "nalgebra_v0_35",
    nalgebra::DVector::<f64>::from_column_slice,
    nalgebra::DVector::<f32>::from_column_slice
);
backend_tests!(
    ndarray_015,
    "ndarray_v0_15",
    |x| ndarray_0_15::Array1::from_vec(x.to_vec()),
    |x| ndarray_0_15::Array1::<f32>::from_vec(x.to_vec())
);
backend_tests!(
    ndarray_016,
    "ndarray_v0_16",
    |x| ndarray_0_16::Array1::from_vec(x.to_vec()),
    |x| ndarray_0_16::Array1::<f32>::from_vec(x.to_vec())
);
backend_tests!(
    ndarray_017,
    "ndarray_v0_17",
    |x| ndarray::Array1::from_vec(x.to_vec()),
    |x| ndarray::Array1::<f32>::from_vec(x.to_vec())
);
backend_tests!(
    faer_022,
    "faer_v0_22",
    |x| faer_0_22::Col::from_fn(x.len(), |i| x[i]),
    |x| faer_0_22::Col::<f32>::from_fn(x.len(), |i| x[i])
);
backend_tests!(
    faer_023,
    "faer_v0_23",
    |x| faer_0_23::Col::from_fn(x.len(), |i| x[i]),
    |x| faer_0_23::Col::<f32>::from_fn(x.len(), |i| x[i])
);
backend_tests!(
    faer_024,
    "faer_v0_24",
    |x| faer::Col::from_fn(x.len(), |i| x[i]),
    |x| faer::Col::<f32>::from_fn(x.len(), |i| x[i])
);

struct Failing {
    at: f64,
    calls: AtomicUsize,
}

impl CostFunction for Failing {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = &'static str;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        self.calls.fetch_add(1, Relaxed);
        if x.iter().any(|v| v.abs() >= self.at) || self.at == 0.0 {
            Err("probe failed")
        } else {
            Ok(x.iter().map(|v| v * v).sum())
        }
    }
}

#[test]
fn fused_errors_preserve_request_counts() {
    for method in [Method::Forward, Method::Central] {
        for at in [0.0, 1e-10, 1e-4, f64::INFINITY] {
            let failing = || Failing {
                at,
                calls: AtomicUsize::new(0),
            };
            for owned in [false, true] {
                let mut p = Problem::new(
                    FiniteDiff::new(failing())
                        .gradient_method(method)
                        .hessian_method(method),
                );
                let x = vec![0.0];
                let result = if owned {
                    p.cost_and_gradient_and_hessian(&x).map(|_| ())
                } else {
                    p.cost_and_gradient_and_hessian_into(
                        &x,
                        &mut vec![],
                        &mut Vec::<f64>::dense_from_fn(
                            0,
                            0,
                            |_, _| unreachable!(),
                        ),
                    )
                    .map(|_| ())
                };
                assert_eq!(
                    result,
                    if at.is_finite() {
                        Err("probe failed")
                    } else {
                        Ok(())
                    }
                );
                assert_eq!(p.counts().cost_evals, 1);
                assert_eq!(p.counts().gradient_evals, 1);
                assert_eq!(p.counts().hessian_evals, 1);
                if at == 0.0 {
                    assert_eq!(p.inner().get_ref().calls.load(Relaxed), 1);
                }

                let mut p = Problem::new(
                    BoundedFiniteDiff::new(failing(), vec![0.0], vec![1.0])
                        .gradient_method(method),
                );
                let mut out = vec![123.0, 456.0];
                let result = if owned {
                    p.cost_and_gradient(&x).map(|_| ())
                } else {
                    p.cost_and_gradient_into(&x, &mut out).map(|_| ())
                };
                assert_eq!(
                    result,
                    if at <= 1e-10 {
                        Err("probe failed")
                    } else {
                        Ok(())
                    }
                );
                assert_eq!(p.counts().cost_evals, 1);
                assert_eq!(p.counts().gradient_evals, 1);
                if at == 0.0 {
                    assert_eq!(p.inner().get_ref().calls.load(Relaxed), 1);
                }
                if !owned && result.is_err() {
                    assert_eq!(out, vec![123.0, 456.0]);
                    assert_eq!(
                        p.inner().gradient_into(&x, &mut out),
                        Err("probe failed")
                    );
                    assert_eq!(out, vec![123.0, 456.0]);
                }
            }
        }
    }
}

#[test]
fn bounded_probe_errors_leave_outputs_untouched() {
    struct FailingLast(f64);
    impl CostFunction for FailingLast {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = &'static str;

        fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
            if x[1] >= self.0 {
                Err("probe failed")
            } else {
                Ok(x.iter().map(|v| v * v).sum())
            }
        }
    }

    for method in [Method::Forward, Method::Central] {
        for at in [0.0, 0.1, 0.2] {
            if at == 0.2 && method == Method::Forward {
                continue;
            }
            let p = BoundedFiniteDiff::new(
                FailingLast(at),
                vec![0.0; 2],
                vec![1.0; 2],
            )
            .gradient_method(method)
            .with_step(0.125);
            for initial in [vec![123.0], vec![123.0, 456.0]] {
                let mut out = initial.clone();
                let x = vec![0.0; 2];
                assert_eq!(p.gradient_into(&x, &mut out), Err("probe failed"));
                assert_eq!(out, initial);
                assert_eq!(
                    p.cost_and_gradient_into(&x, &mut out),
                    Err("probe failed")
                );
                assert_eq!(out, initial);
            }
        }
    }
}

#[test]
fn fused_calls_preserve_nonfinite_base_values() {
    struct Nonfinite(f64);
    impl CostFunction for Nonfinite {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = &'static str;

        fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
            Ok(if x[0] == 0.0 { self.0 } else { x[0] * x[0] })
        }
    }

    fn same_nonfinite(actual: f64, expected: f64) {
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            same_scalar(actual, expected);
        }
    }

    let x = vec![0.0];
    for base in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        for method in [Method::Forward, Method::Central] {
            let fd = FiniteDiff::new(Nonfinite(base))
                .gradient_method(method)
                .hessian_method(method);
            let expected_g = fd.gradient(&x).unwrap();
            let expected_h = fd.hessian(&x).unwrap();
            let (f, g) = fd.cost_and_gradient(&x).unwrap();
            same_nonfinite(f, base);
            same_nonfinite(g[0], expected_g[0]);
            let (f, mut g, mut h) =
                fd.cost_and_gradient_and_hessian(&x).unwrap();
            same_nonfinite(f, base);
            same_nonfinite(g[0], expected_g[0]);
            same_nonfinite(h.matrix_entry(0, 0), expected_h.matrix_entry(0, 0));
            let f = fd.cost_and_gradient_into(&x, &mut g).unwrap();
            same_nonfinite(f, base);
            same_nonfinite(g[0], expected_g[0]);
            let f = fd
                .cost_and_gradient_and_hessian_into(&x, &mut g, &mut h)
                .unwrap();
            same_nonfinite(f, base);
            same_nonfinite(g[0], expected_g[0]);
            same_nonfinite(h.matrix_entry(0, 0), expected_h.matrix_entry(0, 0));

            let fd =
                BoundedFiniteDiff::new(Nonfinite(base), vec![0.0], vec![1.0])
                    .gradient_method(method);
            let expected = fd.gradient(&x).unwrap();
            let (f, mut g) = fd.cost_and_gradient(&x).unwrap();
            same_nonfinite(f, base);
            same_nonfinite(g[0], expected[0]);
            let f = fd.cost_and_gradient_into(&x, &mut g).unwrap();
            same_nonfinite(f, base);
            same_nonfinite(g[0], expected[0]);
        }
    }
}
