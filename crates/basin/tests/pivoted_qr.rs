//! The same regularized least-squares contract across all dense backends.

use basin::{
    DenseMatrix, FactorizePivotedQr, QrSolveError, RegularizedQrSolve,
};

#[path = "support/backend_aliases.rs"]
mod backend_aliases;

fn check_backend<M, V>(
    matrix: impl Fn(usize, usize, &[f64]) -> M,
    vector: impl Fn(&[f64]) -> V,
    entries: impl Fn(V) -> Vec<f64>,
) where
    M: FactorizePivotedQr<V>,
{
    type Case = (usize, usize, &'static [f64], &'static [f64], &'static [f64]);
    let cases: &[Case] = &[
        (
            3,
            3,
            &[0., 0., 100., 2., 0., 0., 0., 10., 0.],
            &[300., 2., 20.],
            &[1., 2., 3.],
        ),
        (
            4,
            2,
            &[1., 0., 0., 2., 1., 2., 1., -2.],
            &[1., 4., 5., -3.],
            &[1., 2.],
        ),
        (
            3,
            3,
            &[1e-8, 0., 0., 0., 1., 0., 0., 0., 1e8],
            &[1e-8, 1., 1e8],
            &[1., 1., 1.],
        ),
    ];
    for &(m, n, a, b, expected) in cases {
        let qr = matrix(m, n, a).factorize_pivoted_qr(&vector(b)).unwrap();
        let d = qr.column_norms_squared();
        for mu in [0., 1e-3, 1., 1e-20] {
            let x = entries(qr.solve_regularized(mu, &d, None).unwrap());
            if mu == 0. {
                for (x, target) in x.iter().zip(expected) {
                    assert!((x - target).abs() < 1e-12, "{x} != {target}");
                }
            }
            // Stationarity checks the regularization as well as the RHS and permutation.
            let diagonal = entries(qr.column_norms_squared());
            for j in 0..n {
                let g: f64 = (0..m)
                    .map(|i| {
                        a[i * n + j]
                            * ((0..n).map(|k| a[i * n + k] * x[k]).sum::<f64>()
                                - b[i])
                    })
                    .sum();
                let penalty = mu * diagonal[j] * x[j];
                let scale = (0..m)
                    .map(|i| (a[i * n + j] * b[i]).abs())
                    .sum::<f64>()
                    .max(penalty.abs());
                assert!(
                    (g + penalty).abs() <= 1e-12 * scale,
                    "normal residual {} at column {j}",
                    g + penalty
                );
            }
        }
    }
    // Positive damping resolves deficient, wide, and completely insensitive systems.
    for (m, n, a, b, expected) in [
        (2, 2, vec![1., 1., 1., 1.], vec![2., 2.], vec![0.8, 0.8]),
        (1, 2, vec![1., 1.], vec![2.], vec![2. / 3., 2. / 3.]),
        (2, 2, vec![1., 0., 0., 0.], vec![2., 1.], vec![1., 0.]),
        (1, 2, vec![0., 0.], vec![2.], vec![0., 0.]),
    ] {
        let qr = matrix(m, n, &a)
            .factorize_pivoted_qr(&vector(&b))
            .unwrap_or_else(|e| panic!("matrix {a:?}, rhs {b:?}: {e}"));
        let d = vector(&vec![1.; n]);
        assert!(matches!(
            qr.solve_regularized(0., &d, None),
            Err(QrSolveError::RankDeficient)
        ));
        let x = entries(qr.solve_regularized(1., &d, None).unwrap());
        for (x, target) in x.iter().zip(expected) {
            assert!((x - target).abs() < 1e-12, "{x} != {target}");
        }
    }
    let a = matrix(3, 2, &[1., 1., 1., 1. + 1e-8, 1., 1. - 1e-8]);
    let qr = a.factorize_pivoted_qr(&vector(&[0., -1e-8, 1e-8])).unwrap();
    let d = qr.column_norms_squared();
    let x = entries(qr.solve_regularized(1e-20, &d, None).unwrap());
    assert!((x[0] - 1.).abs() < 4e-4 && (x[1] + 1.).abs() < 4e-4);
    assert!(matches!(
        qr.solve_regularized(0., &d, Some(1e-6)),
        Err(QrSolveError::RankDeficient)
    ));
    assert!(qr.solve_regularized(1., &d, Some(1e-6)).is_ok());
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            matrix(1, 1, &[invalid]).factorize_pivoted_qr(&vector(&[1.])),
            Err(QrSolveError::NonFinite)
        ));
        assert!(matches!(
            matrix(1, 1, &[1.]).factorize_pivoted_qr(&vector(&[invalid])),
            Err(QrSolveError::NonFinite)
        ));
        assert!(qr.solve_regularized(invalid, &d, None).is_err());
    }
    assert!(matches!(
        qr.solve_regularized(-1., &d, None),
        Err(QrSolveError::InvalidRegularization)
    ));
    assert!(qr.solve_regularized(1., &vector(&[-1., 1.]), None).is_err());
    // This reference uses a cancellation-free analytic formula. Forming the
    // Gram matrix rounds its (1,1) entry to one and loses positive definiteness.
    let eps = 1e-8;
    let mu = 1e-20;
    let qr = matrix(2, 2, &[1., 1., 0., eps])
        .factorize_pivoted_qr(&vector(&[0., eps]))
        .unwrap();
    let x =
        entries(qr.solve_regularized(mu, &vector(&[1., 1.]), None).unwrap());
    let denominator = eps * eps * (1. + mu) + mu * (2. + mu);
    assert!((x[0] + eps * eps / denominator).abs() < 1e-10);
    assert!((x[1] - (1. + mu) * eps * eps / denominator).abs() < 1e-10);

    #[cfg(feature = "nalgebra_all")]
    {
        use backend_aliases::nalgebra::{DMatrix, DVector};
        for (m, n) in [(1, 3), (3, 5), (7, 3), (3, 3), (12, 5)] {
            let a: Vec<f64> =
                (0..m * n).map(|i| ((i * 17 + 3) as f64).sin()).collect();
            let b: Vec<f64> = (0..m).map(|i| (i as f64 + 0.5).cos()).collect();
            let d: Vec<f64> = (0..n).map(|j| (j + 1) as f64).collect();
            let qr =
                matrix(m, n, &a).factorize_pivoted_qr(&vector(&b)).unwrap();
            for mu in [1e-3_f64, 1.] {
                let augmented = DMatrix::from_fn(m + n, n, |i, j| {
                    if i < m {
                        a[i * n + j]
                    } else if i - m == j {
                        (mu * d[j]).sqrt()
                    } else {
                        0.
                    }
                });
                let rhs =
                    DVector::from_fn(
                        m + n,
                        |i, _| if i < m { b[i] } else { 0. },
                    );
                let reference =
                    augmented.svd(true, true).solve(&rhs, 1e-14).unwrap();
                let x = entries(
                    qr.solve_regularized(mu, &vector(&d), None).unwrap(),
                );
                for (x, target) in x.iter().zip(reference.iter()) {
                    assert!(
                        (x - target).abs() < 1e-10,
                        "{m}x{n}: {x} != {target}"
                    );
                }
            }
        }
    }
}

#[test]
fn vec_contract() {
    check_backend(DenseMatrix::from_row_slice, |x| x.to_vec(), |x| x);
}

#[test]
fn f32_contract() {
    let a = DenseMatrix::<f32>::from_row_slice(2, 2, &[0., 10., 1., 0.]);
    let qr = a.factorize_pivoted_qr(&vec![20., 1.]).unwrap();
    let h = qr.solve_regularized(0., &vec![1., 1.], None).unwrap();
    assert!((h[0] - 1.).abs() < 1e-6 && (h[1] - 2.).abs() < 1e-6);
}

#[cfg(feature = "nalgebra_all")]
#[test]
fn nalgebra_contract() {
    use backend_aliases::nalgebra::{DMatrix, DVector};
    check_backend(DMatrix::from_row_slice, DVector::from_column_slice, |x| {
        x.as_slice().to_vec()
    });
}

#[cfg(feature = "ndarray_all")]
#[test]
fn ndarray_contract() {
    use backend_aliases::ndarray::{Array1, Array2, ShapeBuilder};
    for column_major in [false, true] {
        check_backend(
            |m, n, data| {
                let mut a = if column_major {
                    Array2::zeros((m, n).f())
                } else {
                    Array2::zeros((m, n))
                };
                for i in 0..m {
                    for j in 0..n {
                        a[(i, j)] = data[i * n + j];
                    }
                }
                a
            },
            |x| Array1::from_vec(x.to_vec()),
            |x| x.to_vec(),
        );
    }
}

#[cfg(feature = "faer_all")]
#[test]
fn faer_contract() {
    use backend_aliases::faer::{Col, Mat};
    check_backend(
        |m, n, a| Mat::from_fn(m, n, |i, j| a[i * n + j]),
        |x| Col::from_fn(x.len(), |i| x[i]),
        |x| x.iter().copied().collect(),
    );
}
