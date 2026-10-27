//! Dense finite-difference outputs retain allocations when their shape matches.

#[path = "support/backend_aliases.rs"]
mod backend_aliases;

#[cfg(any(
    feature = "nalgebra_all",
    feature = "ndarray_all",
    feature = "faer_all"
))]
fn check<V, F>(address: impl Fn(&V::Matrix) -> usize)
where
    V: basin::DenseMatrixFromFn<F>,
    V::Matrix: basin::MatrixIndex<F>,
    F: basin::Scalar,
{
    use basin::{MatrixIndex, MatrixShape};
    let mut out = V::dense_from_fn(1, 1, |_, _| F::nan());
    for (rows, cols) in [(2, 3), (3, 2), (0, 2), (4, 1)] {
        V::dense_from_fn_into(rows, cols, |_, _| F::nan(), &mut out);
        let allocation = address(&out);
        for offset in [1, 2] {
            let mut calls = 0;
            V::dense_from_fn_into(
                rows,
                cols,
                |i, j| {
                    calls += 1;
                    F::from_usize(i * cols + j + offset).unwrap()
                },
                &mut out,
            );
            assert_eq!(calls, rows * cols);
            assert_eq!((out.matrix_rows(), out.matrix_cols()), (rows, cols));
            assert_eq!(address(&out), allocation);
            for i in 0..rows {
                for j in 0..cols {
                    assert_eq!(
                        out.matrix_entry(i, j),
                        F::from_usize(i * cols + j + offset).unwrap()
                    );
                }
            }
        }
    }
}

#[cfg(feature = "nalgebra_all")]
#[test]
fn nalgebra_f32_and_f64() {
    use backend_aliases::nalgebra::DVector;
    check::<DVector<f32>, f32>(|m| m.as_ptr() as usize);
    check::<DVector<f64>, f64>(|m| m.as_ptr() as usize);
}

#[cfg(feature = "ndarray_all")]
#[test]
fn ndarray_f32_and_f64() {
    use backend_aliases::ndarray::Array1;
    check::<Array1<f32>, f32>(|m| m.as_ptr() as usize);
    check::<Array1<f64>, f64>(|m| m.as_ptr() as usize);
}

#[cfg(feature = "faer_all")]
#[test]
fn faer_f32_and_f64() {
    use backend_aliases::faer::Col;
    check::<Col<f32>, f32>(|m| m.as_ptr() as usize);
    check::<Col<f64>, f64>(|m| m.as_ptr() as usize);
}
