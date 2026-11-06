use super::{
    backend::AsFloatSlice,
    integrated_model::{Model, Workspace},
    workspace::History,
};
use crate::{Dot, Scalar};

#[test]
fn bounded_model_rebuilds_after_delayed_inverse_steps_evictions_and_reset() {
    fn check<F: Scalar>() {
        let f = |v| F::from_f64(v).unwrap();
        for n in [2, 11] {
            let mut h = History::new(3);
            let mut delayed = Model::new(n, 3);
            for update in 0..19 {
                if update == 8 {
                    h = History::new(3);
                }
                let s: Vec<_> = (0..n)
                    .map(|i| f(0.1 + ((i * 7 + update * 3) % 17) as f64 / 17.0))
                    .collect();
                let y: Vec<_> = s
                    .iter()
                    .enumerate()
                    .map(|(i, s)| *s * f((i + 1) as f64))
                    .collect();
                assert!(h.append_pair(s, y));
                assert!(delayed.prepare_inverse(&h.ws, &h.wy, h.theta));
                let g = vec![f(0.3); n];
                let mut out = vec![F::zero(); n];
                delayed.inverse(&g, None, &mut out);
                if update % 4 == 3 {
                    assert!(delayed.rebuild(
                        &h.ws,
                        &h.wy,
                        &h.sy,
                        &h.ss,
                        h.theta,
                        update as u32
                    ));
                    let mut fresh = Model::new(n, 3);
                    assert!(fresh.rebuild(
                        &h.ws,
                        &h.wy,
                        &h.sy,
                        &h.ss,
                        h.theta,
                        update as u32
                    ));
                    let mut expected = out.clone();
                    delayed.apply(&g, &mut out);
                    fresh.apply(&g, &mut expected);
                    assert_eq!(out, expected);
                }
            }
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn restricted_pairs_survive_mask_changes_and_unsafe_restrictions_are_skipped() {
    let mut h = History::new(3);
    assert!(h.append_pair(vec![1.0, 1.0], vec![-1.0, 3.0]));
    let mut model = Model::new(2, 3);
    assert!(model.rebuild(&h.ws, &h.wy, &h.sy, &h.ss, h.theta, 1));
    let mut out = vec![0.0; 2];
    for _ in 0..3 {
        model.inverse(&[2.0, 6.0], Some(&[true, false]), &mut out);
        assert_eq!(out, vec![2.0, 0.0]);
        model.inverse(&[2.0, 6.0], Some(&[false, true]), &mut out);
        close(out[1], 2.0);
        assert_eq!(out[0], 0.0);
    }
    assert_eq!(h.ws.len(), 1);
}

#[test]
fn empty_model_handles_unbounded_fixed_and_empty_boxes_and_rejects_nonfinite_data()
 {
    let mut model = Model::new(2, 3);
    assert!(model.rebuild::<Vec<f64>>(&[], &[], &[], &[], 2.0, 0));
    let mut work = Workspace::new(2, 3);
    let mut z = vec![0.0; 2];
    model.cauchy(
        &[0.2, 0.5],
        &[0.4, -1.0],
        &mut work,
        &mut z,
        &[f64::NEG_INFINITY, 0.5],
        &[f64::INFINITY, 0.5],
    );
    close(z[0], 0.0);
    assert_eq!(z[1], 0.5);
    assert!(!model.rebuild::<Vec<f64>>(&[], &[], &[], &[], f64::NAN, 1));
    assert!(!model.rebuild(
        &[vec![f64::INFINITY; 2]],
        &[vec![1.0; 2]],
        &[1.0; 9],
        &[1.0; 9],
        1.0,
        2
    ));
    let mut empty = Model::new(0, 3);
    assert!(empty.rebuild::<Vec<f64>>(&[], &[], &[], &[], 1.0, 0));
    empty.cauchy(&[], &[], &mut Workspace::new(0, 3), &mut [], &[], &[]);
}
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

fn close<F: Scalar>(a: F, b: F) {
    let tol =
        F::from_f64(8192.0).unwrap() * F::epsilon() * (F::one() + b.abs());
    assert!((a - b).abs() <= tol, "{a:?} != {b:?}, tol={tol:?}");
}
fn check<V: AsFloatSlice<F> + Dot<F>, F: Scalar>(make: fn(&[F]) -> V) {
    let f = |v| F::from_f64(v).unwrap();
    for n in [1, 2, 7, 25] {
        for m in [1, 3, 10] {
            let mut h = History::new(m);
            let mut model = Model::new(n, m);
            let mut work = Workspace::new(n, m);
            for update in 0..2 * m + 4 {
                if update == m + 1 {
                    h = History::new(m);
                }
                let s: Vec<_> = (0..n)
                    .map(
                        |i| f(((i * 17 + update * 7) % 31) as f64 / 31.0 - 0.5),
                    )
                    .collect();
                let total: F = s.iter().copied().sum();
                let y: Vec<_> = s
                    .iter()
                    .enumerate()
                    .map(|(i, s)| f((i + 1) as f64) * *s + f(0.125) * total)
                    .collect();
                assert!(h.append_pair(make(&s), make(&y)));
                assert!(model.rebuild(
                    &h.ws,
                    &h.wy,
                    &h.sy,
                    &h.ss,
                    h.theta,
                    update as u32
                ));
                let mut b = vec![F::zero(); n * n];
                for i in 0..n {
                    b[i * n + i] = h.theta;
                }
                for a in 0..h.ws.len() {
                    let s = h.ws[a].as_float_slice();
                    let y = h.wy[a].as_float_slice();
                    let bs: Vec<F> = (0..n)
                        .map(|i| (0..n).map(|j| b[i * n + j] * s[j]).sum())
                        .collect();
                    let sbs: F = s.iter().zip(&bs).map(|(s, b)| *s * *b).sum();
                    let sy = h.sy[a * m + a];
                    for i in 0..n {
                        for j in 0..n {
                            b[i * n + j] = b[i * n + j] + y[i] * y[j] / sy
                                - bs[i] * bs[j] / sbs;
                        }
                    }
                }
                let g: Vec<_> =
                    (0..n).map(|i| f(0.03 * (i + 1) as f64)).collect();
                let mut product = vec![F::zero(); n];
                model.apply(&g, &mut product);
                for i in 0..n {
                    close(
                        product[i],
                        (0..n).map(|j| b[i * n + j] * g[j]).sum(),
                    );
                }
                let inverse = solve(b.clone(), g.clone());
                model.inverse(&g, None, &mut product);
                for i in 0..n {
                    close(product[i], inverse[i]);
                }
                // Force free dimensions on both sides of the direct/middle-system
                // crossover, plus all-free recursion. Bounds are arbitrary, not [0,1].
                for nf in [1, 2, n / 2, n] {
                    if nf == 0 || nf > n {
                        continue;
                    }
                    let mut x = vec![F::zero(); n];
                    let mut z = vec![F::zero(); n];
                    let l = vec![f(-100.0); n];
                    let u = vec![f(100.0); n];
                    for i in 0..n {
                        x[i] = if i < nf { f(0.02) } else { l[i] };
                        z[i] = if i < nf { f(0.04) } else { l[i] };
                    }
                    let rhs: Vec<_> = (0..nf)
                        .map(|i| {
                            -g[i]
                                - (0..n)
                                    .map(|j| b[i * n + j] * (z[j] - x[j]))
                                    .sum::<F>()
                        })
                        .collect();
                    let reduced: Vec<_> = (0..nf)
                        .flat_map(|i| {
                            (0..nf).map(|j| b[i * n + j]).collect::<Vec<_>>()
                        })
                        .collect();
                    let expected = solve(reduced, rhs);
                    let before = z.clone();
                    model.subspace(&x, &g, &mut z, &mut work, &l, &u);
                    for i in 0..nf {
                        close(z[i], before[i] + expected[i]);
                    }
                    for i in nf..n {
                        assert_eq!(z[i], l[i]);
                    }
                }
                // Cauchy point against an independent dense segment minimization.
                let x: Vec<_> =
                    (0..n).map(|i| f(0.1 * (i % 3) as f64)).collect();
                let g: Vec<_> = (0..n)
                    .map(|i| f(if i % 2 == 0 { 1.3 } else { -0.9 }))
                    .collect();
                let l = vec![f(-0.2); n];
                let mut u = vec![f(0.4); n];
                if n > 1 {
                    u[1] = x[1];
                }
                let mut z = vec![F::zero(); n];
                model.cauchy(&x, &g, &mut work, &mut z, &l, &u);
                let mut expected = x.clone();
                let mut d: Vec<_> = (0..n)
                    .map(|i| {
                        if l[i] == u[i] || (x[i] >= u[i] && g[i] < F::zero()) {
                            F::zero()
                        } else {
                            -g[i]
                        }
                    })
                    .collect();
                for _ in 0..=n {
                    let bd: Vec<F> = (0..n)
                        .map(|i| (0..n).map(|j| b[i * n + j] * d[j]).sum())
                        .collect();
                    let curvature: F =
                        d.iter().zip(&bd).map(|(d, b)| *d * *b).sum();
                    let derivative: F = (0..n)
                        .map(|i| {
                            d[i] * (g[i]
                                + (0..n)
                                    .map(|j| {
                                        b[i * n + j] * (expected[j] - x[j])
                                    })
                                    .sum::<F>())
                        })
                        .sum();
                    if derivative >= F::zero() || curvature <= F::zero() {
                        break;
                    }
                    let mut limit = F::infinity();
                    for i in 0..n {
                        if d[i] != F::zero() {
                            let dist = if d[i] > F::zero() {
                                u[i] - expected[i]
                            } else {
                                l[i] - expected[i]
                            };
                            limit = limit.min(dist / d[i]);
                        }
                    }
                    let t = (-derivative / curvature).min(limit);
                    for i in 0..n {
                        expected[i] =
                            (expected[i] + t * d[i]).max(l[i]).min(u[i]);
                    }
                    if t < limit {
                        break;
                    }
                    for i in 0..n {
                        if expected[i] <= l[i] || expected[i] >= u[i] {
                            d[i] = F::zero();
                        }
                    }
                }
                for i in 0..n {
                    close(z[i], expected[i]);
                }
            }
        }
    }
}
#[test]
fn vec_dense_models() {
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
