use super::{LossEvaluation, LossFunction};
use crate::Scalar;

/// Ordinary squared loss: `ρ(z) = z`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SquaredLoss;
/// Huber loss: `ρ(z) = z` for `z <= 1`, otherwise `2√z - 1`.
#[derive(Clone, Copy, Debug, Default)]
pub struct HuberLoss;
/// Smooth absolute-value loss: `ρ(z) = 2(√(1 + z) - 1)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SoftL1Loss;
/// Cauchy loss: `ρ(z) = ln(1 + z)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct CauchyLoss;
/// Bounded arctangent loss: `ρ(z) = atan(z)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ArctanLoss;

impl<F: Scalar> LossFunction<F> for SquaredLoss {
    fn evaluate(&self, z: F) -> LossEvaluation<F> {
        LossEvaluation {
            value: z,
            first_derivative: F::one(),
            second_derivative: F::zero(),
        }
    }

    fn evaluate_scaled(&self, residual: F, _: F) -> LossEvaluation<F> {
        // The residual scale cancels exactly for squared loss.
        LossEvaluation {
            value: (F::from_f64(0.5).unwrap() * residual) * residual,
            first_derivative: residual,
            second_derivative: F::one(),
        }
    }
}
impl<F: Scalar> LossFunction<F> for HuberLoss {
    fn evaluate(&self, z: F) -> LossEvaluation<F> {
        if z <= F::one() {
            return SquaredLoss.evaluate(z);
        }
        let root = z.sqrt();
        let first_derivative = root.recip();
        LossEvaluation {
            value: F::from_f64(2.0).unwrap() * root - F::one(),
            first_derivative,
            second_derivative: -F::from_f64(0.5).unwrap()
                * (first_derivative / z),
        }
    }

    fn evaluate_scaled(&self, residual: F, scale: F) -> LossEvaluation<F> {
        if residual.abs() <= scale {
            return SquaredLoss.evaluate_scaled(residual, scale);
        }
        LossEvaluation {
            value: scale * (residual.abs() - F::from_f64(0.5).unwrap() * scale),
            first_derivative: scale.copysign(residual),
            second_derivative: F::zero(),
        }
    }
}
impl<F: Scalar> LossFunction<F> for SoftL1Loss {
    fn evaluate(&self, z: F) -> LossEvaluation<F> {
        let t = F::one() + z;
        let root = t.sqrt();
        let first_derivative = root.recip();
        LossEvaluation {
            // Rationalization preserves the value near zero.
            value: F::from_f64(2.0).unwrap() * (z / (root + F::one())),
            first_derivative,
            second_derivative: -F::from_f64(0.5).unwrap()
                * (first_derivative / t),
        }
    }

    fn evaluate_scaled(&self, residual: F, scale: F) -> LossEvaluation<F> {
        if residual.abs() <= scale {
            let t = residual / scale;
            let root = (F::one() + t * t).sqrt();
            let inverse = root.recip();
            LossEvaluation {
                value: (residual / (root + F::one())) * residual,
                first_derivative: residual / root,
                second_derivative: inverse * inverse * inverse,
            }
        } else {
            // Reciprocal coordinates keep the square bounded, and
            // rationalization avoids subtracting nearly equal values.
            let t = scale / residual.abs();
            let root = (F::one() + t * t).sqrt();
            let inverse = t / root;
            LossEvaluation {
                value: scale * (residual.abs() / (root + t)),
                first_derivative: scale.copysign(residual) / root,
                second_derivative: inverse * inverse * inverse,
            }
        }
    }
}
impl<F: Scalar> LossFunction<F> for CauchyLoss {
    fn evaluate(&self, z: F) -> LossEvaluation<F> {
        let first_derivative = (F::one() + z).recip();
        LossEvaluation {
            value: z.ln_1p(),
            first_derivative,
            second_derivative: -first_derivative * first_derivative,
        }
    }

    fn evaluate_scaled(&self, residual: F, scale: F) -> LossEvaluation<F> {
        let half = F::from_f64(0.5).unwrap();
        if residual.abs() <= scale {
            let t = residual / scale;
            let z = t * t;
            let ratio = if z == F::zero() {
                F::one()
            } else {
                z.ln_1p() / z
            };
            let inverse = (F::one() + z).recip();
            LossEvaluation {
                value: (residual * (half * ratio)) * residual,
                first_derivative: residual * inverse,
                second_derivative: (F::one() - z) * inverse * inverse,
            }
        } else {
            let t = scale / residual.abs();
            let z = t * t;
            let inverse = (F::one() + z).recip();
            let normalized = residual.abs() / scale;
            let log = if normalized.is_finite() {
                normalized.ln()
            } else {
                residual.abs().ln() - scale.ln()
            };
            LossEvaluation {
                value: (scale * (log + half * z.ln_1p())) * scale,
                first_derivative: (scale * inverse) * t.copysign(residual),
                second_derivative: (t * inverse)
                    * ((z - F::one()) * t * inverse),
            }
        }
    }
}
impl<F: Scalar> LossFunction<F> for ArctanLoss {
    fn evaluate(&self, z: F) -> LossEvaluation<F> {
        let two = F::from_f64(2.0).unwrap();
        let (first_derivative, second_derivative) = if z > F::one() {
            // Reciprocal coordinates avoid overflow when z² is unrepresentable.
            let inverse = z.recip();
            let denominator = F::one() + inverse * inverse;
            let first = inverse * inverse / denominator;
            (first, -two * first * (inverse / denominator))
        } else {
            let first = (F::one() + z * z).recip();
            (first, -two * z * first * first)
        };
        LossEvaluation {
            value: z.atan(),
            first_derivative,
            second_derivative,
        }
    }

    fn evaluate_scaled(&self, residual: F, scale: F) -> LossEvaluation<F> {
        let half = F::from_f64(0.5).unwrap();
        let three = F::from_f64(3.0).unwrap();
        if residual.abs() <= scale {
            let t = residual / scale;
            let z = t * t;
            let ratio = if z == F::zero() {
                F::one()
            } else {
                z.atan() / z
            };
            let inverse = (F::one() + z * z).recip();
            LossEvaluation {
                value: (residual * (half * ratio)) * residual,
                first_derivative: residual * inverse,
                second_derivative: (F::one() - three * z * z)
                    * inverse
                    * inverse,
            }
        } else {
            let t = scale / residual.abs();
            let z = t * t;
            let inverse = (F::one() + z * z).recip();
            let angle =
                F::from_f64(std::f64::consts::FRAC_PI_2).unwrap() - z.atan();
            LossEvaluation {
                value: (scale * (half * angle)) * scale,
                first_derivative: (((scale * t.copysign(residual)) * inverse)
                    * t)
                    * t,
                second_derivative: (z * inverse)
                    * ((z * z - three) * z * inverse),
            }
        }
    }
}
