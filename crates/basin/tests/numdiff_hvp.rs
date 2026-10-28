//! Repeated Hessian-vector products reuse storage without caching evaluations.

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use basin::{
    CostFunction, FiniteDiff, Gradient, HessianProduct, HessianProductScratch,
    Method, VectorIndex, VectorLen, central_difference_hessian_product,
    forward_difference_hessian_product,
};

struct Cubic<V> {
    calls: AtomicUsize,
    make: fn(&[f64]) -> V,
}

impl<V: VectorLen + VectorIndex> CostFunction for Cubic<V> {
    type Param = V;
    type Output = f64;
    type Error = &'static str;

    fn cost(&self, x: &V) -> Result<f64, Self::Error> {
        let sum: f64 = (0..x.vec_len()).map(|j| x.get_scalar(j)).sum();
        Ok(0.5 * sum * sum
            + (0..x.vec_len())
                .map(|j| (j + 1) as f64 * x.get_scalar(j).powi(3) / 3.0)
                .sum::<f64>())
    }
}

impl<V: VectorLen + VectorIndex> Gradient for Cubic<V> {
    type Gradient = V;

    fn gradient(&self, _: &V) -> Result<V, Self::Error> {
        panic!("the product should use gradient_into")
    }

    fn gradient_into(&self, x: &V, out: &mut V) -> Result<(), Self::Error> {
        self.calls.fetch_add(1, Relaxed);
        if out.vec_len() != x.vec_len() {
            *out = (self.make)(&vec![0.0; x.vec_len()]);
        }
        let sum: f64 = (0..x.vec_len()).map(|j| x.get_scalar(j)).sum();
        for j in 0..x.vec_len() {
            out.set_scalar(j, (j + 1) as f64 * x.get_scalar(j).powi(2) + sum);
        }
        Ok(())
    }
}

fn check<V>(make: fn(&[f64]) -> V)
where
    V: Clone + VectorLen + VectorIndex + Send + Sync,
{
    let problem = Cubic {
        calls: AtomicUsize::new(0),
        make,
    };
    let mut scratch = HessianProductScratch::new();
    let mut out = make(&[f64::NAN; 3]);
    for n in [0, 1, 4, 33, 2, 0, 7] {
        for shift in [-0.25, 0.5] {
            let x = make(
                &(0..n).map(|j| shift + j as f64 / 16.0).collect::<Vec<_>>(),
            );
            for direction in [-1.0, 0.0, 0.75] {
                let v = make(
                    &(0..n)
                        .map(|j| direction / (j + 1) as f64)
                        .collect::<Vec<_>>(),
                );
                for method in [Method::Forward, Method::Central] {
                    for step in [None, Some(0.001)] {
                        problem.calls.store(0, Relaxed);
                        match method {
                            Method::Forward => scratch.forward_into(
                                &problem,
                                &x,
                                &v,
                                f64::EPSILON,
                                step,
                                &mut out,
                            ),
                            Method::Central => scratch.central_into(
                                &problem,
                                &x,
                                &v,
                                f64::EPSILON,
                                step,
                                &mut out,
                            ),
                            _ => unreachable!(),
                        }
                        .unwrap();
                        assert_eq!(
                            problem.calls.load(Relaxed),
                            if n == 0 || direction == 0.0 { 0 } else { 2 }
                        );
                        assert_eq!(out.vec_len(), n);
                        let sum_v: f64 = (0..n).map(|j| v.get_scalar(j)).sum();
                        for j in 0..n {
                            let mut expected = 2.0
                                * (j + 1) as f64
                                * x.get_scalar(j)
                                * v.get_scalar(j)
                                + sum_v;
                            if let (Method::Forward, Some(h)) = (method, step) {
                                expected += h
                                    * (j + 1) as f64
                                    * v.get_scalar(j).powi(2);
                            }
                            assert!(
                                (out.get_scalar(j) - expected).abs()
                                    < 2e-5 * (1.0 + expected.abs())
                            );
                        }
                        let owned = match method {
                            Method::Forward => {
                                forward_difference_hessian_product(
                                    &problem,
                                    &x,
                                    &v,
                                    f64::EPSILON,
                                    step,
                                )
                            }
                            Method::Central => {
                                central_difference_hessian_product(
                                    &problem,
                                    &x,
                                    &v,
                                    f64::EPSILON,
                                    step,
                                )
                            }
                            _ => unreachable!(),
                        }
                        .unwrap();
                        for j in 0..n {
                            assert_eq!(
                                out.get_scalar(j).to_bits(),
                                owned.get_scalar(j).to_bits()
                            );
                        }

                        let mut fd = FiniteDiff::new(Cubic {
                            calls: AtomicUsize::new(0),
                            make,
                        })
                        .gradient_method(method)
                        .hessian_method(method)
                        .function_precision(1e-12);
                        if let Some(h) = step {
                            fd = fd.with_step(h);
                        }
                        fd.hessian_product_with_scratch_into(
                            &x,
                            &v,
                            &mut out,
                            &mut scratch,
                        )
                        .unwrap();
                        let expected = fd.hessian_product(&x, &v).unwrap();
                        for j in 0..n {
                            assert_eq!(
                                out.get_scalar(j).to_bits(),
                                expected.get_scalar(j).to_bits()
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn vec_repeated_products() {
    check::<Vec<f64>>(|x| x.to_vec());
}

macro_rules! backend_test {
    ($name:ident, $feature:literal, $make:expr) => {
        #[cfg(feature = $feature)]
        #[test]
        fn $name() {
            check($make);
        }
    };
}

backend_test!(
    nalgebra_032,
    "nalgebra_v0_32",
    nalgebra_0_32::DVector::<f64>::from_column_slice
);
backend_test!(
    nalgebra_033,
    "nalgebra_v0_33",
    nalgebra_0_33::DVector::<f64>::from_column_slice
);
backend_test!(
    nalgebra_034,
    "nalgebra_v0_34",
    nalgebra_0_34::DVector::<f64>::from_column_slice
);
backend_test!(
    nalgebra_035,
    "nalgebra_v0_35",
    nalgebra::DVector::<f64>::from_column_slice
);
backend_test!(ndarray_015, "ndarray_v0_15", |x| {
    ndarray_0_15::Array1::from_vec(x.to_vec())
});
backend_test!(ndarray_016, "ndarray_v0_16", |x| {
    ndarray_0_16::Array1::from_vec(x.to_vec())
});
backend_test!(ndarray_017, "ndarray_v0_17", |x| ndarray::Array1::from_vec(
    x.to_vec()
));
backend_test!(faer_022, "faer_v0_22", |x| faer_0_22::Col::from_fn(
    x.len(),
    |j| x[j]
));
backend_test!(faer_023, "faer_v0_23", |x| faer_0_23::Col::from_fn(
    x.len(),
    |j| x[j]
));
backend_test!(faer_024, "faer_v0_24", |x| faer::Col::from_fn(
    x.len(),
    |j| x[j]
));

struct Failing {
    reject: f64,
    calls: AtomicUsize,
}

impl CostFunction for Failing {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = &'static str;
    fn cost(&self, _: &Vec<f64>) -> Result<f64, Self::Error> {
        unreachable!()
    }
}

impl Gradient for Failing {
    type Gradient = Vec<f64>;
    fn gradient(&self, _: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        unreachable!()
    }
    fn gradient_into(
        &self,
        x: &Vec<f64>,
        out: &mut Vec<f64>,
    ) -> Result<(), Self::Error> {
        self.calls.fetch_add(1, Relaxed);
        out.clear();
        if x[0] == self.reject {
            return Err("gradient rejected");
        }
        out.extend(x.iter().map(|value| 2.0 * value));
        Ok(())
    }
}

#[test]
fn errors_propagate_and_scratch_can_be_reused() {
    let mut scratch = HessianProductScratch::default();
    let x = vec![1.0, 2.0];
    let v = vec![1.0, -0.5];
    let mut out = vec![];
    for method in [Method::Forward, Method::Central] {
        let probes = if method == Method::Forward {
            [1.0, 1.25]
        } else {
            [1.25, 0.75]
        };
        for (i, reject) in probes.into_iter().enumerate() {
            let p = Failing {
                reject,
                calls: AtomicUsize::new(0),
            };
            let result = if method == Method::Forward {
                scratch.forward_into(
                    &p,
                    &x,
                    &v,
                    f64::EPSILON,
                    Some(0.25),
                    &mut out,
                )
            } else {
                scratch.central_into(
                    &p,
                    &x,
                    &v,
                    f64::EPSILON,
                    Some(0.25),
                    &mut out,
                )
            };
            assert_eq!(result, Err("gradient rejected"));
            assert_eq!(p.calls.load(Relaxed), i + 1);
            let p = Failing {
                reject: f64::NAN,
                calls: AtomicUsize::new(0),
            };
            scratch
                .central_into(&p, &x, &v, f64::EPSILON, Some(0.25), &mut out)
                .unwrap();
            assert_eq!(out, vec![2.0, -1.0]);
        }
    }
}

#[test]
fn scratch_does_not_add_thread_safety_bounds() {
    struct Local(std::rc::Rc<std::cell::Cell<usize>>);
    impl CostFunction for Local {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, _: &Vec<f64>) -> Result<f64, Self::Error> {
            unreachable!()
        }
    }
    impl Gradient for Local {
        type Gradient = Vec<f64>;
        fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
            self.0.set(self.0.get() + 1);
            Ok(x.iter().map(|value| 2.0 * value).collect())
        }
    }
    let p = Local(std::rc::Rc::new(std::cell::Cell::new(0)));
    let mut scratch = HessianProductScratch::new();
    let mut out = vec![];
    scratch
        .central_into(&p, &vec![1.0], &vec![2.0], f64::EPSILON, None, &mut out)
        .unwrap();
    assert!((out[0] - 4.0).abs() < 1e-8);
    assert_eq!(p.0.get(), 2);
}

#[test]
fn wrapper_remains_copy_and_scratch_is_send_sync() {
    #[derive(Clone, Copy)]
    struct CopyProblem;
    fn copy<T: Copy>(_: T) {}
    fn send_sync<T: Send + Sync>() {}
    copy(FiniteDiff::new(CopyProblem));
    send_sync::<FiniteDiff<CopyProblem>>();
    send_sync::<HessianProductScratch<Vec<f64>>>();
}

#[test]
fn nonfinite_gradients_and_zero_directions_keep_their_behavior() {
    struct Nonfinite(f64);
    impl CostFunction for Nonfinite {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, _: &Vec<f64>) -> Result<f64, Self::Error> {
            unreachable!()
        }
    }
    impl Gradient for Nonfinite {
        type Gradient = Vec<f64>;
        fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
            Ok(vec![self.0; x.len()])
        }
    }
    let mut scratch = HessianProductScratch::new();
    let mut out = vec![123.0; 3];
    for value in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let p = Nonfinite(value);
        for method in [Method::Forward, Method::Central] {
            let run = |scratch: &mut HessianProductScratch<_>,
                       v: &Vec<f64>,
                       out: &mut _| {
                if method == Method::Forward {
                    scratch.forward_into(
                        &p,
                        &vec![1.0],
                        v,
                        f64::EPSILON,
                        None,
                        out,
                    )
                } else {
                    scratch.central_into(
                        &p,
                        &vec![1.0],
                        v,
                        f64::EPSILON,
                        None,
                        out,
                    )
                }
                .unwrap();
            };
            run(&mut scratch, &vec![1.0], &mut out);
            assert_eq!(out.len(), 1);
            assert!(out[0].is_nan());
            run(&mut scratch, &vec![0.0], &mut out);
            assert_eq!(out, vec![0.0]);
        }
    }
}

#[test]
fn independent_scratch_buffers_can_share_a_wrapper() {
    let fd = FiniteDiff::new(Cubic {
        calls: AtomicUsize::new(0),
        make: |x| x.to_vec(),
    })
    .with_step(0.001);
    std::thread::scope(|scope| {
        for i in 1..=4 {
            let fd = &fd;
            scope.spawn(move || {
                let x = vec![0.5, -0.25];
                let v = vec![i as f64, -0.5];
                let mut scratch = HessianProductScratch::new();
                let mut out = vec![];
                for _ in 0..5 {
                    fd.hessian_product_with_scratch_into(
                        &x,
                        &v,
                        &mut out,
                        &mut scratch,
                    )
                    .unwrap();
                    let expected = fd.hessian_product(&x, &v).unwrap();
                    assert_eq!(out, expected);
                }
            });
        }
    });
}
