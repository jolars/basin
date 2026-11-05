//! Staging must preserve the original stencils, callback work, and atomic
//! publication of Jacobians, including across column-block boundaries.

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use basin::{
    DenseMatrixFromFn, FiniteDiff, Jacobian, MatrixIndex, MatrixShape, Method,
    Residual, VectorIndex, VectorLen,
};

struct Function<V> {
    make: fn(&[f64]) -> V,
    rows: usize,
    calls: AtomicUsize,
    fail: bool,
}

fn residual<V: VectorLen + VectorIndex>(x: &V, i: usize) -> f64 {
    (0..x.vec_len())
        .map(|j| {
            let xj = x.get_scalar(j);
            (i + j + 1) as f64 * xj + xj * xj
        })
        .sum()
}

impl<V: VectorLen + VectorIndex> Residual for Function<V> {
    type Param = V;
    type Output = V;
    type Error = &'static str;

    fn residual(&self, _: &V) -> Result<V, Self::Error> {
        panic!("probes must use residual_into")
    }

    fn residual_into(&self, x: &V, out: &mut V) -> Result<(), Self::Error> {
        self.calls.fetch_add(1, Relaxed);
        if self.fail && x.get_scalar(x.vec_len() - 1) != 0.5 {
            return Err("last-coordinate probe failed");
        }
        if out.vec_len() != self.rows {
            *out = (self.make)(&vec![0.0; self.rows]);
        }
        for i in 0..self.rows {
            out.set_scalar(i, residual(x, i));
        }
        Ok(())
    }
}

fn check<V>(make: fn(&[f64]) -> V)
where
    V: Clone + VectorLen + VectorIndex + DenseMatrixFromFn + Send + Sync,
    V::Matrix: MatrixIndex,
{
    for step in [None, Some(0.125)] {
        for (n, m) in [(0, 3), (3, 0), (4, 17), (17, 4), (65, 3)] {
            for method in [Method::Forward, Method::Central] {
                let mut fd = FiniteDiff::new(Function {
                    make,
                    rows: m,
                    calls: AtomicUsize::new(0),
                    fail: false,
                })
                .jacobian_method(method);
                if let Some(step) = step {
                    fd = fd.with_step(step);
                }
                let x = make(
                    &(0..n)
                        .map(|j| [-0.75, 0.0, 0.5, 1.25][j % 4])
                        .collect::<Vec<_>>(),
                );
                // Reference probes evaluate the original formulas independently of
                // the wrapper's staging and output matrix storage order.
                let expected = V::dense_from_fn(m, n, |i, j| {
                    let mut probe = x.clone();
                    let xj = x.get_scalar(j);
                    let h = step.unwrap_or_else(|| {
                        if matches!(method, Method::Central) {
                            let h = f64::EPSILON.cbrt() * xj.abs().max(1.0);
                            (xj + h) - xj
                        } else {
                            let h = f64::EPSILON.sqrt() * xj.abs();
                            if h == 0.0 { f64::EPSILON.sqrt() } else { h }
                        }
                    });
                    probe.set_scalar(j, xj + h);
                    let plus = residual(&probe, i);
                    probe.set_scalar(j, xj - h);
                    if matches!(method, Method::Central) {
                        (plus - residual(&probe, i)) / (2.0 * h)
                    } else {
                        (plus - residual(&x, i)) / h
                    }
                });
                let verify = |actual: &V::Matrix, fused: bool| {
                    assert_eq!(
                        (actual.matrix_rows(), actual.matrix_cols()),
                        (m, n)
                    );
                    for i in 0..m {
                        for j in 0..n {
                            assert_eq!(
                                actual.matrix_entry(i, j).to_bits(),
                                expected.matrix_entry(i, j).to_bits()
                            );
                        }
                    }
                    let count = if matches!(method, Method::Central) {
                        2 * n + usize::from(fused || n == 0)
                    } else {
                        n + 1
                    };
                    assert_eq!(fd.get_ref().calls.swap(0, Relaxed), count);
                };
                verify(&fd.jacobian(&x).unwrap(), false);
                let mut jac = V::dense_from_fn(1, 1, |_, _| 99.0);
                for _ in 0..2 {
                    fd.jacobian_into(&x, &mut jac).unwrap();
                    verify(&jac, false);
                    let mut res = make(&[99.0]);
                    fd.residual_and_jacobian_into(&x, &mut res, &mut jac)
                        .unwrap();
                    verify(&jac, true);
                    assert_eq!(res.vec_len(), m);
                }
            }
        }
    }
    for method in [Method::Forward, Method::Central] {
        let fd = FiniteDiff::new(Function {
            make,
            rows: 3,
            calls: AtomicUsize::new(0),
            fail: true,
        })
        .jacobian_method(method)
        .with_step(0.125);
        let x = make(&[0.5; 65]);
        let mut jac = V::dense_from_fn(2, 1, |_, _| 99.0);
        assert_eq!(
            fd.jacobian_into(&x, &mut jac),
            Err("last-coordinate probe failed")
        );
        assert_eq!((jac.matrix_rows(), jac.matrix_cols()), (2, 1));
        assert_eq!(jac.matrix_entry(0, 0), 99.0);
        assert_eq!(jac.matrix_entry(1, 0), 99.0);
        let mut res = make(&[]);
        assert_eq!(
            fd.residual_and_jacobian_into(&x, &mut res, &mut jac),
            Err("last-coordinate probe failed")
        );
        assert_eq!((jac.matrix_rows(), jac.matrix_cols()), (2, 1));
        assert_eq!(jac.matrix_entry(0, 0), 99.0);
        assert_eq!(jac.matrix_entry(1, 0), 99.0);
    }
}

#[test]
fn vec_stencils_and_publication() {
    check(|x| x.to_vec());
}

#[test]
fn inconsistent_probe_shapes_do_not_publish_a_jacobian() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    struct ChangingShape {
        coordinate: usize,
        both_probes: bool,
    }
    impl Residual for ChangingShape {
        type Param = Vec<f64>;
        type Output = Vec<f64>;
        type Error = &'static str;

        fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
            // Exercise both within-column and across-column shape guards.
            let changed = if self.both_probes {
                x[self.coordinate] != 0.5
            } else {
                x[self.coordinate] > 0.5
            };
            Ok(vec![1.0; if changed { 2 } else { 3 }])
        }
    }
    for method in [Method::Forward, Method::Central] {
        for coordinate in [0, 64] {
            for both_probes in [false, true] {
                let fd = FiniteDiff::new(ChangingShape {
                    coordinate,
                    both_probes,
                })
                .jacobian_method(method)
                .with_step(0.125);
                let mut jac = basin::DenseMatrix::from_fn(2, 1, |_, _| 99.0);
                assert!(
                    catch_unwind(AssertUnwindSafe(|| {
                        fd.jacobian_into(&vec![0.5; 65], &mut jac).unwrap();
                    }))
                    .is_err()
                );
                assert_eq!((jac.nrows(), jac.ncols()), (2, 1));
                assert_eq!(jac.get(0, 0), 99.0);
                assert_eq!(jac.get(1, 0), 99.0);
            }
        }
    }
}

macro_rules! backend_tests {
    ($name:ident, $feature:literal, $make:expr) => {
        #[cfg(feature = $feature)]
        #[test]
        fn $name() {
            check($make);
        }
    };
}

backend_tests!(
    nalgebra_032,
    "nalgebra_v0_32",
    nalgebra_0_32::DVector::from_column_slice
);
backend_tests!(
    nalgebra_033,
    "nalgebra_v0_33",
    nalgebra_0_33::DVector::from_column_slice
);
backend_tests!(
    nalgebra_034,
    "nalgebra_v0_34",
    nalgebra_0_34::DVector::from_column_slice
);
backend_tests!(
    nalgebra_035,
    "nalgebra_v0_35",
    nalgebra::DVector::from_column_slice
);
backend_tests!(ndarray_015, "ndarray_v0_15", |x| {
    ndarray_0_15::Array1::from_vec(x.to_vec())
});
backend_tests!(ndarray_016, "ndarray_v0_16", |x| {
    ndarray_0_16::Array1::from_vec(x.to_vec())
});
backend_tests!(ndarray_017, "ndarray_v0_17", |x| ndarray::Array1::from_vec(
    x.to_vec()
));
backend_tests!(faer_022, "faer_v0_22", |x| faer_0_22::Col::from_fn(
    x.len(),
    |i| x[i]
));
backend_tests!(faer_023, "faer_v0_23", |x| faer_0_23::Col::from_fn(
    x.len(),
    |i| x[i]
));
backend_tests!(faer_024, "faer_v0_24", |x| faer::Col::from_fn(
    x.len(),
    |i| x[i]
));
