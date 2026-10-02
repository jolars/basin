//! Focused convergence pilot on NIST StRD Misra1a, using both supplied starts.
//! Run `cargo run -p competitor-bench --release --bin convergence_pilot`.
//!
//! Source: https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Misra1a.dat
//! The published residual sum of squares is 0.12455138894. Basin's least
//! squares cost is half that sum. This pilot is not a default-selection sweep.

use basin::{
    BoxConstraints, CostFunction, DenseMatrix, Executor, FirstOrderState,
    Gradient, Jacobian, Lbfgsb, LevenbergMarquardt, NelderMead, PointState,
    Residual, SimplexProgress, TrustRegionReflective,
};
use std::convert::Infallible;

const DATA: [(f64, f64); 14] = [
    (10.07, 77.6),
    (14.73, 114.9),
    (17.94, 141.1),
    (23.93, 190.8),
    (29.61, 239.9),
    (35.18, 289.0),
    (40.02, 332.8),
    (44.82, 378.4),
    (50.76, 434.8),
    (55.05, 477.3),
    (61.01, 536.8),
    (66.40, 593.1),
    (75.47, 689.1),
    (81.78, 760.0),
];
const STARTS: [[f64; 2]; 2] = [[500.0, 0.0001], [250.0, 0.0005]];
const CERTIFIED: [f64; 2] = [238.94212918, 0.00055015643181];
const CERTIFIED_COST: f64 = 0.12455138894 / 2.0;

#[derive(Clone)]
struct Misra1a {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl Misra1a {
    fn new() -> Self {
        Self {
            lower: vec![0.0, 0.0],
            upper: vec![1000.0, 0.01],
        }
    }

    fn residuals(x: &[f64]) -> impl Iterator<Item = f64> + '_ {
        DATA.iter()
            .map(|&(y, t)| x[0] * (1.0 - (-x[1] * t).exp()) - y)
    }

    fn jacobian_rows(x: &[f64]) -> impl Iterator<Item = [f64; 2]> + '_ {
        DATA.iter().map(|&(_, t)| {
            let e = (-x[1] * t).exp();
            [1.0 - e, x[0] * t * e]
        })
    }

    fn diagnostics(x: &[f64]) -> (f64, f64, f64) {
        let cost = 0.5 * Self::residuals(x).map(|r| r * r).sum::<f64>();
        let mut gradient = [0.0; 2];
        for (r, row) in Self::residuals(x).zip(Self::jacobian_rows(x)) {
            gradient[0] += row[0] * r;
            gradient[1] += row[1] * r;
        }
        let scaled_gradient = (gradient[0].abs() * CERTIFIED[0])
            .max(gradient[1].abs() * CERTIFIED[1]);
        let parameter_error = x
            .iter()
            .zip(CERTIFIED)
            .map(|(&a, b)| ((a - b) / b).abs())
            .fold(0.0, f64::max);
        (cost, scaled_gradient, parameter_error)
    }
}

impl CostFunction for Misra1a {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(0.5 * Self::residuals(x).map(|r| r * r).sum::<f64>())
    }
}

impl Gradient for Misra1a {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        let mut gradient = vec![0.0; 2];
        for (r, row) in Self::residuals(x).zip(Self::jacobian_rows(x)) {
            gradient[0] += row[0] * r;
            gradient[1] += row[1] * r;
        }
        Ok(gradient)
    }
}

impl Residual for Misra1a {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = Infallible;

    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(Self::residuals(x).collect())
    }
}

impl Jacobian for Misra1a {
    type Jacobian = DenseMatrix;

    fn jacobian(&self, x: &Vec<f64>) -> Result<DenseMatrix, Infallible> {
        let rows: Vec<f64> = Self::jacobian_rows(x).flatten().collect();
        Ok(DenseMatrix::from_row_slice(DATA.len(), 2, &rows))
    }
}

impl BoxConstraints for Misra1a {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

fn main() {
    println!(
        "start,solver,termination,iterations,cost_gap,scaled_gradient,relative_parameter_error,cost_evals,gradient_evals,residual_evals,jacobian_evals"
    );
    for (start_index, start) in STARTS.iter().enumerate() {
        let start = start.to_vec();
        let nm = Executor::new(
            Misra1a::new(),
            NelderMead::new(),
            SimplexProgress::new(start.clone()),
        )
        .max_iter(500)
        .run()
        .unwrap();
        let lbfgsb = Executor::new(
            Misra1a::new(),
            Lbfgsb::new(),
            FirstOrderState::new(start.clone()),
        )
        .max_iter(500)
        .run()
        .unwrap();
        let lm = Executor::new(
            Misra1a::new(),
            LevenbergMarquardt::new(),
            PointState::new(start.clone()),
        )
        .max_iter(500)
        .run()
        .unwrap();
        let trf = Executor::new(
            Misra1a::new(),
            TrustRegionReflective::new(),
            PointState::new(start),
        )
        .max_iter(500)
        .run()
        .unwrap();
        for (solver, cost, param, iter, code, counts) in [
            (
                "nelder_mead",
                nm.cost(),
                nm.param().clone(),
                nm.iter(),
                nm.report.code(),
                *nm.state.counts(),
            ),
            (
                "lbfgsb",
                lbfgsb.cost(),
                lbfgsb.param().clone(),
                lbfgsb.iter(),
                lbfgsb.report.code(),
                *lbfgsb.state.counts(),
            ),
            (
                "lm",
                lm.cost(),
                lm.param().clone(),
                lm.iter(),
                lm.report.code(),
                *lm.state.counts(),
            ),
            (
                "trf",
                trf.cost(),
                trf.param().clone(),
                trf.iter(),
                trf.report.code(),
                *trf.state.counts(),
            ),
        ] {
            let (checked_cost, gradient, parameter_error) =
                Misra1a::diagnostics(&param);
            assert!((cost - checked_cost).abs() < 1e-9);
            println!(
                "{},{},{:?},{},{:.9e},{:.9e},{:.9e},{},{},{},{}",
                start_index + 1,
                solver,
                code,
                iter,
                checked_cost - CERTIFIED_COST,
                gradient,
                parameter_error,
                counts.cost_evals,
                counts.gradient_evals,
                counts.residual_evals,
                counts.jacobian_evals
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn misra1a_matches_certificate_and_analytic_derivatives() {
        let (cost, _, _) = Misra1a::diagnostics(&CERTIFIED);
        assert!((cost - CERTIFIED_COST).abs() < 1e-10);
        for x in STARTS.iter().chain(std::iter::once(&CERTIFIED)) {
            for column in 0..2 {
                let step = 1e-6 * x[column];
                let mut plus = *x;
                let mut minus = *x;
                plus[column] += step;
                minus[column] -= step;
                for ((analytic, positive), negative) in
                    Misra1a::jacobian_rows(x)
                        .zip(Misra1a::residuals(&plus))
                        .zip(Misra1a::residuals(&minus))
                {
                    let numeric = (positive - negative) / (2.0 * step);
                    assert!(
                        (analytic[column] - numeric).abs()
                            < 1e-8 * analytic[column].abs().max(1.0),
                        "column {column}: analytic={}, numeric={numeric}",
                        analytic[column]
                    );
                }
            }
        }
    }
}
