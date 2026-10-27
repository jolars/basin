//! Preserve the original Hessian stencils while changing their assembly storage.

use std::marker::PhantomData;

use basin::{
    CostFunction, DenseMatrixFromFn, FiniteDiff, Hessian, MatrixIndex,
    MatrixShape, Method, VectorIndex, VectorLen, central_difference_hessian,
    forward_difference_hessian,
};

struct Function<V> {
    evaluate: fn(&[f64]) -> Result<f64, &'static str>,
    vector: PhantomData<V>,
}

impl<V: VectorLen + VectorIndex> CostFunction for Function<V> {
    type Param = V;
    type Output = f64;
    type Error = &'static str;

    fn cost(&self, x: &V) -> Result<f64, Self::Error> {
        (self.evaluate)(
            &(0..x.vec_len())
                .map(|i| x.get_scalar(i))
                .collect::<Vec<_>>(),
        )
    }
}

fn nonlinear(x: &[f64]) -> Result<f64, &'static str> {
    let sum = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| (i + 1) as f64 * xi)
        .sum::<f64>();
    Ok(sum.sin()
        + sum * sum * sum / 7.0
        + x.iter()
            .enumerate()
            .map(|(i, &xi)| (i + 1) as f64 * xi.cos())
            .sum::<f64>())
}

// This reference keeps the pre-optimization probe and arithmetic order, assembling
// a full matrix with nested loops independently of the production packed indexing.
fn reference(
    evaluate: fn(&[f64]) -> Result<f64, &'static str>,
    x: &[f64],
    method: Method,
    step: Option<f64>,
) -> Vec<Vec<f64>> {
    let n = x.len();
    let f0 = evaluate(x).unwrap();
    let scale = f64::EPSILON.powf(0.25);
    let h: Vec<_> = x
        .iter()
        .map(|&xi| {
            step.unwrap_or_else(|| {
                let h = scale * xi.abs().max(1.0);
                let reround = (xi + h) - xi;
                if reround == 0.0 { h } else { reround }
            })
        })
        .collect();
    let mut probe = x.to_vec();
    let fi: Vec<_> = if method == Method::Forward {
        (0..n)
            .map(|i| {
                probe[i] = x[i] + h[i];
                let value = evaluate(&probe).unwrap();
                probe[i] = x[i];
                value
            })
            .collect()
    } else {
        Vec::new()
    };
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in i..n {
            let value = if method == Method::Forward {
                if i == j {
                    probe[i] = x[i] + 2.0 * h[i];
                } else {
                    probe[i] = x[i] + h[i];
                    probe[j] = x[j] + h[j];
                }
                let cross = evaluate(&probe).unwrap();
                probe[i] = x[i];
                probe[j] = x[j];
                (cross - fi[i] - fi[j] + f0) / (h[i] * h[j])
            } else if i == j {
                probe[i] = x[i] + h[i];
                let fp = evaluate(&probe).unwrap();
                probe[i] = x[i] - h[i];
                let fm = evaluate(&probe).unwrap();
                probe[i] = x[i];
                (fp - 2.0 * f0 + fm) / (h[i] * h[i])
            } else {
                probe[i] = x[i] + h[i];
                probe[j] = x[j] + h[j];
                let fpp = evaluate(&probe).unwrap();
                probe[j] = x[j] - h[j];
                let fpm = evaluate(&probe).unwrap();
                probe[i] = x[i] - h[i];
                let fmm = evaluate(&probe).unwrap();
                probe[j] = x[j] + h[j];
                let fmp = evaluate(&probe).unwrap();
                probe[i] = x[i];
                probe[j] = x[j];
                (fpp - fpm - fmp + fmm) / (4.0 * h[i] * h[j])
            };
            out[i][j] = value;
            out[j][i] = value;
        }
    }
    out
}

fn check_matrix<M: MatrixIndex>(actual: &M, expected: &[Vec<f64>]) {
    let n = expected.len();
    assert_eq!((actual.matrix_rows(), actual.matrix_cols()), (n, n));
    for (i, row) in expected.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            assert_eq!(
                actual.matrix_entry(i, j).to_bits(),
                value.to_bits(),
                "entry ({i}, {j})"
            );
            assert_eq!(
                actual.matrix_entry(i, j).to_bits(),
                actual.matrix_entry(j, i).to_bits(),
                "symmetry at ({i}, {j})"
            );
        }
    }
}

fn check<V>(make: fn(&[f64]) -> V)
where
    V: Clone + VectorLen + VectorIndex + DenseMatrixFromFn + Send + Sync,
    V::Matrix: MatrixIndex,
{
    for n in [0, 1, 2, 3, 9, 33] {
        let values: Vec<_> =
            (0..n).map(|i| ((i * 7 % 11) as f64 - 5.0) / 3.0).collect();
        let x = make(&values);
        for method in [Method::Forward, Method::Central] {
            for step in [None, Some(0.013)] {
                let mut fd = FiniteDiff::new(Function::<V> {
                    evaluate: nonlinear,
                    vector: PhantomData,
                })
                .hessian_method(method);
                if let Some(step) = step {
                    fd = fd.with_step(step);
                }
                let expected = reference(nonlinear, &values, method, step);
                check_matrix(&fd.hessian(&x).unwrap(), &expected);
                let free = if method == Method::Forward {
                    forward_difference_hessian(
                        fd.get_ref(),
                        &x,
                        f64::EPSILON,
                        step,
                    )
                } else {
                    central_difference_hessian(
                        fd.get_ref(),
                        &x,
                        f64::EPSILON,
                        step,
                    )
                };
                check_matrix(&free.unwrap(), &expected);
                for (rows, cols) in [(0, 0), (n + 1, n), (n, n)] {
                    let mut out = V::dense_from_fn(rows, cols, |_, _| 99.0);
                    for _ in 0..2 {
                        fd.hessian_into(&x, &mut out).unwrap();
                        check_matrix(&out, &expected);
                    }
                }
            }
        }
    }

    // Cover base, single-coordinate, and mixed probes. A mixed-probe failure
    // follows successful entries, so assembly cannot publish partial data.
    for evaluate in [
        (|_: &[f64]| Err("base")) as fn(&[f64]) -> _,
        |x: &[f64]| {
            if x[1] > 0.0 {
                Err("probe")
            } else {
                nonlinear(x)
            }
        },
        |x: &[f64]| {
            if x[0] > 0.0 && x[1] > 0.0 {
                Err("mixed probe")
            } else {
                nonlinear(x)
            }
        },
    ] {
        for method in [Method::Forward, Method::Central] {
            let fd = FiniteDiff::new(Function::<V> {
                evaluate,
                vector: PhantomData,
            })
            .hessian_method(method)
            .with_step(0.013);
            let x = make(&[0.0; 4]);
            assert!(fd.hessian(&x).is_err());
            for (rows, cols) in [(4, 4), (2, 3)] {
                let mut out = V::dense_from_fn(rows, cols, |i, j| {
                    (1 + i * cols + j) as f64
                });
                let error = fd.hessian_into(&x, &mut out).unwrap_err();
                assert_eq!(error, evaluate(&[0.013, 0.013]).unwrap_err());
                assert_eq!(
                    (out.matrix_rows(), out.matrix_cols()),
                    (rows, cols)
                );
                for i in 0..rows {
                    for j in 0..cols {
                        assert_eq!(
                            out.matrix_entry(i, j),
                            (1 + i * cols + j) as f64
                        );
                    }
                }
            }
        }
    }

    for evaluate in [
        (|_: &[f64]| Ok(f64::INFINITY)) as fn(&[f64]) -> _,
        |_: &[f64]| Ok(f64::NAN),
    ] {
        for method in [Method::Forward, Method::Central] {
            let fd = FiniteDiff::new(Function::<V> {
                evaluate,
                vector: PhantomData,
            })
            .hessian_method(method);
            let out = fd.hessian(&make(&[0.0; 3])).unwrap();
            for i in 0..3 {
                for j in 0..3 {
                    assert!(out.matrix_entry(i, j).is_nan());
                    assert_eq!(
                        out.matrix_entry(i, j).to_bits(),
                        out.matrix_entry(j, i).to_bits()
                    );
                }
            }
        }
    }
}

#[test]
fn vec_hessian_matches_original_stencils() {
    check::<Vec<f64>>(|x| x.to_vec());
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
    nalgebra_0_32::DVector::<f64>::from_column_slice
);
backend_tests!(
    nalgebra_033,
    "nalgebra_v0_33",
    nalgebra_0_33::DVector::<f64>::from_column_slice
);
backend_tests!(
    nalgebra_034,
    "nalgebra_v0_34",
    nalgebra_0_34::DVector::<f64>::from_column_slice
);
backend_tests!(
    nalgebra_035,
    "nalgebra_v0_35",
    nalgebra::DVector::<f64>::from_column_slice
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
