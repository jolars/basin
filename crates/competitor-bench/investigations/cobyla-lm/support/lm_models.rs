//! Shared deterministic calibration formulas, Jacobian checks, and starts.

use nalgebra::{DMatrix, DVector};

#[derive(Clone)]
pub(crate) enum Model {
    Linear(DMatrix<f64>),
    Svi(Vec<f64>),
    Ssvi(Vec<(f64, f64)>),
}
impl Model {
    pub(crate) fn evaluate(
        &self,
        p: &DVector<f64>,
    ) -> (DVector<f64>, DMatrix<f64>) {
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
pub(crate) fn check_jacobian(model: &Model, x: &DVector<f64>) {
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
pub(crate) fn for_each_case(
    mut compare: impl FnMut(&str, Model, Vec<f64>, Vec<Vec<f64>>),
) {
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
