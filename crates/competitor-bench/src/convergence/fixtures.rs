//! Native-scalar analytic fixtures for validating the measurement layer.

use basin::{BoxConstraints, CostFunction, Gradient, Scalar};

use super::ledger::{OracleError, OracleKind, WorkLedger};
use super::quality::{Interval, SmoothCertificate};

/// `a/2 * ((x0-1)^2 + (x1+2)^2) + b` has an exact dyadic witness.
/// Its unshifted verifier uses independent widened arithmetic, never the
/// cached solver objective. These are development fixtures, not corpus entries.
#[derive(Clone, Debug)]
pub struct Quadratic<F: Scalar = f64> {
    pub ledger: WorkLedger,
    pub multiplier: F,
    pub offset: F,
    pub fused: bool,
    lower: Vec<F>,
    upper: Vec<F>,
}

impl<F: Scalar> Quadratic<F> {
    pub fn new(ledger: WorkLedger) -> Self {
        Self {
            ledger,
            multiplier: F::one(),
            offset: F::zero(),
            fused: true,
            lower: vec![F::from_f64(-8.0).unwrap(); 2],
            upper: vec![F::from_f64(8.0).unwrap(); 2],
        }
    }

    fn values(&self, x: &[F]) -> (F, Vec<F>) {
        let d = [x[0] - F::one(), x[1] + F::from_f64(2.0).unwrap()];
        let cost = self.multiplier
            * F::from_f64(0.5).unwrap()
            * (d[0] * d[0] + d[1] * d[1])
            + self.offset;
        (cost, d.iter().map(|d| self.multiplier * *d).collect())
    }

    fn point(x: &[F]) -> Vec<f64> {
        x.iter().map(|v| v.to_f64().unwrap()).collect()
    }

    pub fn certificate(&self, start: &[F]) -> SmoothCertificate {
        let a = self.multiplier.to_f64().unwrap();
        assert!(a.is_finite() && a > 0.0);
        let x = Self::point(start);
        let gap = 0.5 * ((x[0] - 1.0).powi(2) + (x[1] + 2.0).powi(2));
        SmoothCertificate {
            reference: Interval::exact(0.0),
            objective_scale: a * gap.max(1.0),
            coordinate_scales: vec![1.0; 2],
            solution_set: vec![vec![1.0, -2.0]],
            require_parameter: true,
            bounds: Some((Self::point(&self.lower), Self::point(&self.upper))),
        }
    }

    /// The exact sum-of-squares reference is zero. Widening f32 coordinates is
    /// exact; this nonnegative expression has no subtractive cancellation after
    /// forming the dyadic differences. The interval includes a rounding screen.
    pub fn verify(&self, x: &[f64]) -> (Interval, Vec<f64>) {
        let a = self.multiplier.to_f64().unwrap();
        let d = [x[0] - 1.0, x[1] + 2.0];
        let value = a * 0.5 * (d[0] * d[0] + d[1] * d[1]);
        let error = 8.0 * f64::EPSILON * value.abs();
        (
            Interval {
                lower: (value - error).max(0.0),
                upper: value + error,
            },
            d.iter().map(|v| a * v).collect(),
        )
    }
}

impl<F: Scalar> CostFunction for Quadratic<F> {
    type Param = Vec<F>;
    type Output = F;
    type Error = OracleError;

    fn cost(&self, x: &Vec<F>) -> Result<F, OracleError> {
        self.ledger.evaluate(OracleKind::Cost, &Self::point(x), || {
            if x.len() != 2 {
                return Err(OracleError::Callback("wrong dimension"));
            }
            let value = self.values(x).0;
            Ok((value, Some(value.to_f64().unwrap())))
        })
    }
}

impl<F: Scalar> Gradient for Quadratic<F> {
    type Gradient = Vec<F>;

    fn gradient(&self, x: &Vec<F>) -> Result<Vec<F>, OracleError> {
        self.ledger
            .evaluate(OracleKind::Gradient, &Self::point(x), || {
                if x.len() != 2 {
                    return Err(OracleError::Callback("wrong dimension"));
                }
                Ok((self.values(x).1, None))
            })
    }

    fn cost_and_gradient(
        &self,
        x: &Vec<F>,
    ) -> Result<(F, Vec<F>), OracleError> {
        if !self.fused {
            return Ok((self.cost(x)?, self.gradient(x)?));
        }
        self.ledger
            .evaluate(OracleKind::CostGradient, &Self::point(x), || {
                if x.len() != 2 {
                    return Err(OracleError::Callback("wrong dimension"));
                }
                let values = self.values(x);
                let cost = values.0.to_f64().unwrap();
                Ok((values, Some(cost)))
            })
    }
}

impl<F: Scalar> BoxConstraints for Quadratic<F> {
    fn lower(&self) -> &Vec<F> {
        &self.lower
    }
    fn upper(&self) -> &Vec<F> {
        &self.upper
    }
}
