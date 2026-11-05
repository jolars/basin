use super::{backend::AsFloatSlice, compact, subspace_direction::Workspace};
use crate::Scalar;
use crate::{Dot, solver::lbfgs::workspace::History};

// Independent pivoted elimination on an explicitly updated dense BFGS
// matrix; no compact middle-system routines are used by this oracle.
fn solve<F: Scalar>(mut a: Vec<F>, mut b: Vec<F>) -> Vec<F> {
    let n = b.len();
    for k in 0..n {
        let pivot = (k..n)
            .max_by(|&i, &j| {
                a[i * n + k].abs().partial_cmp(&a[j * n + k].abs()).unwrap()
            })
            .unwrap();
        for j in 0..n {
            a.swap(k * n + j, pivot * n + j);
        }
        b.swap(k, pivot);
        for i in k + 1..n {
            let c = a[i * n + k] / a[k * n + k];
            for j in k + 1..n {
                a[i * n + j] = a[i * n + j] - c * a[k * n + j];
            }
            b[i] = b[i] - c * b[k];
        }
    }
    for i in (0..n).rev() {
        for j in i + 1..n {
            b[i] = b[i] - a[i * n + j] * b[j];
        }
        b[i] = b[i] / a[i * n + i];
    }
    b
}

fn check<V, F>(make: fn(&[F]) -> V)
where
    V: AsFloatSlice<F> + Dot<F>,
    F: Scalar,
{
    let scalar = |x| F::from_f64(x).unwrap();
    let mut scratch = Workspace::default();
    for n in [2, 7] {
        let m = 3;
        let mut history = History::new(m);
        let mut wt = vec![F::zero(); m * m];
        for update in 0..10 {
            if update == 6 {
                history = History::new(m);
                scratch.reset();
            }
            let s: Vec<_> = (0..n)
                .map(|j| {
                    scalar(((j * 17 + update * 7) % 31) as f64 / 31.0 - 0.5)
                })
                .collect();
            let sum: F = s.iter().copied().sum();
            let y: Vec<_> = s
                .iter()
                .enumerate()
                .map(|(j, &s)| scalar((j + 1) as f64) * s + scalar(0.125) * sum)
                .collect();
            assert!(history.append_pair(make(&s), make(&y)));
            let col = history.ws.len();
            compact::formt(
                history.theta,
                &history.sy,
                &history.ss,
                col,
                m,
                &mut wt,
            )
            .unwrap();
            let mut dense = vec![F::zero(); n * n];
            for i in 0..n {
                dense[i * n + i] = history.theta;
            }
            for a in 0..col {
                let s = history.ws[a].as_float_slice();
                let y = history.wy[a].as_float_slice();
                let bs: Vec<F> = (0..n)
                    .map(|i| (0..n).map(|j| dense[i * n + j] * s[j]).sum())
                    .collect();
                let sbs: F = s.iter().zip(&bs).map(|(s, b)| *s * *b).sum();
                let sy = history.sy[a * m + a];
                for i in 0..n {
                    for j in 0..n {
                        dense[i * n + j] = dense[i * n + j] + y[i] * y[j] / sy
                            - bs[i] * bs[j] / sbs;
                    }
                }
            }
            for nf in [1, 2, n] {
                if nf != n && nf != 1 {
                    continue;
                }
                let ind: Vec<_> = (0..nf).map(|j| n - 1 - j).collect();
                // Include a Cauchy displacement with active coordinates;
                // applying -H*g alone is not a solution of this RHS.
                let r: Vec<F> = ind
                    .iter()
                    .map(|&i| {
                        -scalar(0.03 * (i + 1) as f64)
                            - (0..n)
                                .map(|j| {
                                    dense[i * n + j]
                                        * scalar(0.02 * (j as f64 - 2.0))
                                })
                                .sum::<F>()
                    })
                    .collect();
                let reduced: Vec<_> = ind
                    .iter()
                    .flat_map(|&i| {
                        ind.iter()
                            .map(|&j| dense[i * n + j])
                            .collect::<Vec<_>>()
                    })
                    .collect();
                let expected = solve(reduced, r.clone());
                assert!(
                    scratch.try_direction(
                        n,
                        &ind,
                        &r,
                        &history.ws,
                        &history.wy,
                        &history.sy,
                        &wt,
                        m,
                        history.theta
                    ),
                    "n={n} update={update} nf={nf} rhs={r:?} theta={:?} epsilon={:?}",
                    history.theta,
                    F::epsilon()
                );
                for (actual, expected) in scratch.direction.iter().zip(expected)
                {
                    let tolerance = scalar(2048.0)
                        * F::epsilon()
                        * (F::one() + expected.abs());
                    assert!(
                        (*actual - expected).abs() <= tolerance,
                        "n={n} update={update} nf={nf}: {actual:?} vs {expected:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn vec_dense_oracle() {
    check::<_, f64>(|x| x.to_vec());
    check::<_, f32>(|x| x.to_vec());
}
macro_rules! backend {
    ($name:ident,$feature:literal,$make:expr) => {
        #[cfg(feature=$feature)]
        #[test]
        fn $name() {
            check::<_, f64>($make);
            check::<_, f32>($make);
        }
    };
}
backend!(
    nalgebra_032,
    "nalgebra_v0_32",
    nalgebra_0_32::DVector::from_column_slice
);
backend!(
    nalgebra_033,
    "nalgebra_v0_33",
    nalgebra_0_33::DVector::from_column_slice
);
backend!(
    nalgebra_034,
    "nalgebra_v0_34",
    nalgebra_0_34::DVector::from_column_slice
);
backend!(
    nalgebra_035,
    "nalgebra_v0_35",
    nalgebra::DVector::from_column_slice
);
backend!(ndarray_015, "ndarray_v0_15", |x| {
    ndarray_0_15::Array1::from_vec(x.to_vec())
});
backend!(ndarray_016, "ndarray_v0_16", |x| {
    ndarray_0_16::Array1::from_vec(x.to_vec())
});
backend!(ndarray_017, "ndarray_v0_17", |x| ndarray::Array1::from_vec(
    x.to_vec()
));
backend!(faer_022, "faer_v0_22", |x| faer_0_22::Col::from_fn(
    x.len(),
    |i| x[i]
));
backend!(faer_023, "faer_v0_23", |x| faer_0_23::Col::from_fn(
    x.len(),
    |i| x[i]
));
backend!(faer_024, "faer_v0_24", |x| faer::Col::from_fn(
    x.len(),
    |i| x[i]
));

#[test]
fn unsupported_and_nonfinite_models_decline() {
    let mut work = Workspace::<f64>::default();
    let ws = vec![vec![1.0; 4]];
    let wy = vec![vec![2.0; 4]];
    for (r, sy, theta) in [
        (vec![f64::NAN; 4], 8.0, 1.0),
        (vec![1.0; 4], 0.0, 1.0),
        (vec![1.0; 4], 8.0, f64::INFINITY),
    ] {
        assert!(!work.try_direction(
            4,
            &[0, 1, 2, 3],
            &r,
            &ws,
            &wy,
            &[sy],
            &[2.0],
            1,
            theta
        ));
    }
    for (ind, r, sy, wt, theta) in [
        (vec![], vec![], 8.0, 2.0, 1.0),
        (vec![0, 1, 2], vec![1.0; 3], 8.0, 2.0, 1.0),
        (vec![0], vec![f64::NAN], 8.0, 2.0, 1.0),
        (vec![0], vec![1.0], 0.0, 2.0, 1.0),
        (vec![0], vec![1.0], 8.0, 0.0, 1.0),
        (vec![0], vec![1.0], 8.0, 2.0, f64::INFINITY),
        (vec![0, 1, 2, 3], vec![0.0; 4], 8.0, 2.0, 1.0),
    ] {
        assert!(!work.try_direction(
            4,
            &ind,
            &r,
            &ws,
            &wy,
            &[sy],
            &[wt],
            1,
            theta
        ));
    }
    assert!(!work.try_direction(
        4,
        &[0],
        &[1.0],
        &Vec::<Vec<f64>>::new(),
        &[],
        &[],
        &[],
        1,
        1.0
    ));
    // A finite but inconsistent compact model can have negative curvature.
    assert!(!work.try_direction(
        2,
        &[0],
        &[1.0],
        &[vec![10.0, 0.0], vec![10.0, 0.0], vec![10.0, 0.0]],
        &[vec![0.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        3,
        1.0
    ));
}
