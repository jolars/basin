//! Conditioning probes and synthetic calibration comparisons for TODO.md.
//! The prototype isolates QR from damping-policy changes. Its diagnostic rank
//! handling is deliberately incomplete; it is not a reusable solver.

use basin::{Executor, Jacobian, LevenbergMarquardt, NllsState, Residual};
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{DMatrix, DVector, Dyn, Owned};
use std::{cell::Cell, convert::Infallible};

#[derive(Clone)]
enum Model {
    Linear(DMatrix<f64>),
    Svi(Vec<f64>),
    Ssvi(Vec<(f64, f64)>),
}
impl Model {
    fn evaluate(&self, p: &DVector<f64>) -> (DVector<f64>, DMatrix<f64>) {
        match self {
            Self::Linear(a) => (a * p, a.clone()),
            Self::Svi(ks) => {
                let mut y = DVector::zeros(ks.len());
                let mut j = DMatrix::zeros(ks.len(), 5);
                for (i, &k) in ks.iter().enumerate() {
                    let dk = k - p[3];
                    let s = (dk * dk + p[4] * p[4]).sqrt();
                    y[i] = p[0] + p[1] * (p[2] * dk + s);
                    j[(i, 0)] = 1.0;
                    j[(i, 1)] = p[2] * dk + s;
                    j[(i, 2)] = p[1] * dk;
                    j[(i, 3)] = p[1] * (-p[2] - dk / s);
                    j[(i, 4)] = p[1] * p[4] / s;
                }
                (y, j)
            }
            Self::Ssvi(data) => {
                let mut y = DVector::zeros(data.len());
                let mut j = DMatrix::zeros(data.len(), 3);
                for (i, &(k, theta)) in data.iter().enumerate() {
                    let phi = p[1]
                        / (theta.powf(p[2]) * (1.0 + theta).powf(1.0 - p[2]));
                    let u = phi * k + p[0];
                    let s = (u * u + 1.0 - p[0] * p[0]).sqrt();
                    y[i] = 0.5 * theta * (1.0 + p[0] * phi * k + s);
                    let dphi = 0.5 * theta * k * (p[0] + u / s);
                    j[(i, 0)] = 0.5 * theta * (phi * k + phi * k / s);
                    j[(i, 1)] = dphi * phi / p[1];
                    j[(i, 2)] = dphi * phi * ((1.0 + theta).ln() - theta.ln());
                }
                (y, j)
            }
        }
    }
}
struct Problem<'a> {
    model: &'a Model,
    y: &'a DVector<f64>,
    nr: &'a Cell<u64>,
    nj: &'a Cell<u64>,
    x: DVector<f64>,
}
impl Problem<'_> {
    fn r(&self, x: &DVector<f64>) -> DVector<f64> {
        self.nr.set(self.nr.get() + 1);
        self.model.evaluate(x).0 - self.y
    }
    fn j(&self, x: &DVector<f64>) -> DMatrix<f64> {
        self.nj.set(self.nj.get() + 1);
        self.model.evaluate(x).1
    }
}
impl Residual for Problem<'_> {
    type Param = DVector<f64>;
    type Output = DVector<f64>;
    type Error = Infallible;
    fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, Infallible> {
        Ok(self.r(x))
    }
}
impl Jacobian for Problem<'_> {
    type Jacobian = DMatrix<f64>;
    fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, Infallible> {
        Ok(self.j(x))
    }
}
impl LeastSquaresProblem<f64, Dyn, Dyn> for Problem<'_> {
    type ParameterStorage = Owned<f64, Dyn>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;
    fn params(&self) -> DVector<f64> {
        self.x.clone()
    }
    fn set_params(&mut self, x: &DVector<f64>) {
        self.x.copy_from(x);
    }
    fn residuals(&self) -> Option<DVector<f64>> {
        Some(self.r(&self.x))
    }
    fn jacobian(&self) -> Option<DMatrix<f64>> {
        Some(self.j(&self.x))
    }
}
fn check_jacobian(model: &Model, x: &DVector<f64>) {
    let (_, j) = model.evaluate(x);
    for k in 0..x.len() {
        let step = 1e-6 * x[k].abs().max(0.01);
        let mut xp = x.clone();
        xp[k] += step;
        let mut xm = x.clone();
        xm[k] -= step;
        let numeric =
            (model.evaluate(&xp).0 - model.evaluate(&xm).0) / (2.0 * step);
        let error = (&numeric - j.column(k)).amax();
        assert!(
            error < 1e-7 * j.column(k).amax().max(1.0),
            "jacobian column {k}: {error}"
        );
    }
}
// A diagnostic Nielsen loop isolates the factorization choice from MINPACK's
// different damping and trust-region updates. It is not a production solver.
fn prototype(
    p: &Problem<'_>,
    mut x: DVector<f64>,
    qr: bool,
) -> (DVector<f64>, String) {
    let mut r = p.r(&x);
    let mut j = p.j(&x);
    let mut d = DVector::from_fn(x.len(), |k, _| j.column(k).norm_squared());
    for v in d.iter_mut() {
        if *v == 0.0 {
            *v = 1.0;
        }
    }
    let mut mu = 1e-3;
    let mut nu = 2.0;
    let max_iters = 200 * (x.len() + 1) - 1;
    for iteration in 0..max_iters {
        let g = j.transpose() * &r;
        let diag = DVector::from_fn(x.len(), |k, _| j.column(k).norm_squared());
        let relative = g
            .iter()
            .zip(diag.iter())
            .map(|(&g, &d)| g * g / if d == 0.0 { 1.0 } else { d })
            .fold(0.0, f64::max);
        if relative <= 1e-24 * r.norm_squared() {
            return (x, "gtol".into());
        }
        for k in 0..d.len() {
            d[k] = d[k].max(diag[k]);
        }
        let mut h = None;
        for _ in 0..50 {
            h =
                if qr {
                    let m = j.nrows();
                    let n = j.ncols();
                    let a = DMatrix::from_fn(m + n, n, |i, k| {
                        if i < m {
                            j[(i, k)]
                        } else if i - m == k {
                            (mu * d[k]).sqrt()
                        } else {
                            0.0
                        }
                    });
                    let qr = a.col_piv_qr();
                    let mut rhs = DVector::from_fn(m + n, |i, _| {
                        if i < m { -r[i] } else { 0.0 }
                    });
                    qr.q_tr_mul(&mut rhs);
                    qr.r()
                        .solve_upper_triangular(&rhs.rows(0, n).into_owned())
                        .map(|mut x| {
                            qr.p().inv_permute_rows(&mut x);
                            x
                        })
                } else {
                    let mut a = j.transpose() * &j;
                    for k in 0..x.len() {
                        a[(k, k)] += mu * d[k];
                    }
                    a.cholesky().map(|c| c.solve(&(-&g)))
                };
            if h.is_some() {
                break;
            }
            mu *= nu;
            nu *= 2.0;
        }
        let Some(h) = h else {
            return (x, "factorization-failed".into());
        };
        let predicted = 0.5 * (mu * h.dot(&d.component_mul(&h)) - h.dot(&g));
        let trial = &x + &h;
        let rt = p.r(&trial);
        let cost = 0.5 * r.norm_squared();
        let actual = cost - 0.5 * rt.norm_squared();
        let rho = if predicted > 0.0 {
            actual / predicted
        } else {
            0.0
        };
        if rho > 0.0 {
            x = trial;
            r = rt;
            mu *= (1.0 - (2.0 * rho - 1.0).powi(3)).max(1.0 / 3.0);
            nu = 2.0;
        } else {
            mu *= nu;
            nu *= 2.0;
        }
        if actual.abs() <= 1e-12 * cost
            && predicted <= 1e-12 * cost
            && rho <= 2.0
        {
            return (x, "ftol".into());
        }
        if h.norm_squared() <= 1e-24 * x.norm_squared() {
            return (x, "xtol".into());
        }
        if rho > 0.0 && iteration + 1 < max_iters {
            j = p.j(&x);
        }
    }
    (x, "MaxIter".into())
}
fn compare(name: &str, model: Model, truth: Vec<f64>, starts: Vec<Vec<f64>>) {
    let truth = DVector::from_vec(truth);
    let (y, j) = model.evaluate(&truth);
    let sv = j.svd(false, false).singular_values;
    println!("condition,{name},{:.5e}", sv.max() / sv.min());
    check_jacobian(&model, &truth);
    for (start, x) in starts.into_iter().enumerate() {
        let x = DVector::from_vec(x);
        check_jacobian(&model, &x);
        for kind in ["basin", "minpack", "prototype-cholesky", "prototype-qr"] {
            let nr = Cell::new(0);
            let nj = Cell::new(0);
            let p = Problem {
                model: &model,
                y: &y,
                nr: &nr,
                nj: &nj,
                x: x.clone(),
            };
            let (solution, reason) = if kind == "basin" {
                let solver = LevenbergMarquardt::new()
                    .with_tol_grad(0.0)
                    .with_tol_grad_rel(1e-12)
                    .with_tol_cost_rel(1e-12)
                    .with_tol_step_rel(1e-12);
                let out = Executor::new(p, solver, NllsState::new(x.clone()))
                    .max_iter((200 * (x.len() + 1) - 1) as u64)
                    .run()
                    .unwrap();
                (out.param().clone(), format!("{:?}", out.reason))
            } else if kind == "minpack" {
                let (p, report) =
                    levenberg_marquardt::LevenbergMarquardt::new()
                        .with_tol(1e-12)
                        .with_patience(200)
                        .minimize(p);
                (p.x, format!("{:?}", report.termination))
            } else {
                prototype(&p, x.clone(), kind == "prototype-qr")
            };
            let residual = (model.evaluate(&solution).0 - &y).norm();
            let error = (&solution - &truth).amax();
            println!(
                "solve,{name},{start},{kind},{},{},{residual:.6e},{error:.6e},{reason},{:?}",
                nr.get(),
                nj.get(),
                solution.as_slice()
            );
        }
    }
}
fn step_probe() {
    for eps in [1e-4_f64, 1e-6, 1e-8, 1e-10, 0.0] {
        let a = DMatrix::from_row_slice(
            3,
            2,
            &[1.0, 1.0, 1.0, 1.0 + eps, 1.0, 1.0 - eps],
        );
        let truth = DVector::from_vec(vec![1.0, -1.0]);
        let r = -&a * &truth;
        let gram = a.transpose() * &a;
        let g = a.transpose() * &r;
        for mu in [1e-3, 1e-10, 1e-16, 1e-20, 0.0] {
            let mut damped = gram.clone();
            for k in 0..2 {
                damped[(k, k)] += mu * gram[(k, k)];
            }
            let ch = damped.cholesky().map(|c| c.solve(&(-&g)));
            let stacked = DMatrix::from_fn(5, 2, |i, k| {
                if i < 3 {
                    a[(i, k)]
                } else if i - 3 == k {
                    (mu * gram[(k, k)]).sqrt()
                } else {
                    0.0
                }
            });
            let rhs =
                DVector::from_fn(5, |i, _| if i < 3 { -r[i] } else { 0.0 });
            let svd =
                stacked.clone().svd(true, true).solve(&rhs, 1e-30).unwrap();
            let qr = stacked.col_piv_qr();
            let mut qtb = rhs;
            qr.q_tr_mul(&mut qtb);
            let h = qr
                .r()
                .solve_upper_triangular(&qtb.rows(0, 2).into_owned())
                .map(|mut x| {
                    qr.p().inv_permute_rows(&mut x);
                    x
                });
            let error = |h: Option<DVector<f64>>| {
                h.map(|h| (h - &svd).norm()).unwrap_or(f64::NAN)
            };
            println!("step,{eps:e},{mu:e},{:.6e},{:.6e}", error(ch), error(h));
        }
    }
}
fn main() {
    step_probe();
    for eps in [1e-4, 1e-8, 0.0] {
        compare(
            &format!("collinear-{eps:e}"),
            Model::Linear(DMatrix::from_row_slice(
                3,
                2,
                &[1.0, 1.0, 1.0, 1.0 + eps, 1.0, 1.0 - eps],
            )),
            vec![1.0, -1.0],
            vec![vec![0.0, 0.0], vec![2.0, 1.0]],
        );
    }
    compare(
        "scaled",
        Model::Linear(DMatrix::from_diagonal(&DVector::from_vec(vec![
            1e-8, 1.0, 1e8,
        ]))),
        vec![1.0, 1.0, 1.0],
        vec![vec![0.0; 3], vec![2.0; 3]],
    );
    for width in [0.5, 0.01] {
        let ks = (0..41).map(|i| width * (i as f64 / 20.0 - 1.0)).collect();
        compare(
            &format!("svi-width-{width}"),
            Model::Svi(ks),
            vec![0.04, 0.1, -0.3, 0.0, 0.2],
            vec![
                vec![0.02, 0.2, -0.5, 0.05, 0.1],
                vec![0.1, 0.05, 0.2, -0.1, 0.5],
                vec![0.035, 0.12, -0.2, 0.01, 0.25],
            ],
        );
    }
    for narrow in [false, true] {
        let theta = if narrow {
            vec![0.039, 0.04, 0.041]
        } else {
            vec![0.01, 0.04, 0.1, 0.25]
        };
        let data = theta
            .into_iter()
            .flat_map(|t| {
                (0..21).map(move |i| (0.5 * (i as f64 / 10.0 - 1.0), t))
            })
            .collect();
        compare(
            &format!("ssvi-narrow-{narrow}"),
            Model::Ssvi(data),
            vec![-0.3, 0.5, 0.5],
            vec![
                vec![-0.5, 0.3, 0.7],
                vec![0.2, 0.8, 0.3],
                vec![-0.2, 0.6, 0.55],
            ],
        );
    }
}
