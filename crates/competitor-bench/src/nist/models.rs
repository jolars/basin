//! Published response formulas and hand-derived response derivatives.

use basin::Scalar;

pub(super) fn response<F: Scalar>(id: &str, b: &[F], x: &[F]) -> (F, Vec<F>) {
    let c = |v| F::from_f64(v).unwrap();
    let one = F::one();
    let two = c(2.0);
    let t = x[0];
    let mut g = vec![F::zero(); b.len()];
    let y = match id {
        "BoxBOD" | "Misra1a" => {
            let e = (-b[1] * t).exp();
            g[0] = -(-b[1] * t).exp_m1();
            g[1] = b[0] * t * e;
            b[0] * g[0]
        }
        "Misra1b" => {
            let q = one + b[1] * t / two;
            g[0] = one - q.powi(-2);
            g[1] = b[0] * t * q.powi(-3);
            b[0] * g[0]
        }
        "Misra1c" => {
            let q = one + two * b[1] * t;
            g[0] = one - q.powf(c(-0.5));
            g[1] = b[0] * t * q.powf(c(-1.5));
            b[0] * g[0]
        }
        "Misra1d" => {
            let q = one + b[1] * t;
            g[0] = b[1] * t / q;
            g[1] = b[0] * t / (q * q);
            b[0] * g[0]
        }
        "Chwirut1" | "Chwirut2" => {
            let q = b[1] + b[2] * t;
            let y = (-b[0] * t).exp() / q;
            g[0] = -t * y;
            g[1] = -y / q;
            g[2] = t * g[1];
            y
        }
        "DanWood" => {
            g[0] = t.powf(b[1]);
            let y = b[0] * g[0];
            g[1] = y * t.ln();
            y
        }
        "Bennett5" => {
            let q = b[1] + t;
            g[0] = q.powf(-one / b[2]);
            let y = b[0] * g[0];
            g[1] = -y / (b[2] * q);
            g[2] = y * q.ln() / (b[2] * b[2]);
            y
        }
        "Lanczos1" | "Lanczos2" | "Lanczos3" => {
            let mut y = F::zero();
            for i in [0, 2, 4] {
                g[i] = (-b[i + 1] * t).exp();
                let term = b[i] * g[i];
                g[i + 1] = -t * term;
                y = y + term;
            }
            y
        }
        "MGH17" => {
            g[0] = one;
            g[1] = (-t * b[3]).exp();
            g[2] = (-t * b[4]).exp();
            g[3] = -t * b[1] * g[1];
            g[4] = -t * b[2] * g[2];
            b[0] + b[1] * g[1] + b[2] * g[2]
        }
        "MGH10" => {
            let q = t + b[2];
            g[0] = (b[1] / q).exp();
            let y = b[0] * g[0];
            g[1] = y / q;
            g[2] = -y * b[1] / (q * q);
            y
        }
        "Gauss1" | "Gauss2" | "Gauss3" => {
            g[0] = (-b[1] * t).exp();
            let mut y = b[0] * g[0];
            g[1] = -t * y;
            for i in [2, 5] {
                let d = t - b[i + 1];
                let w = b[i + 2];
                g[i] = (-(d / w).powi(2)).exp();
                let term = b[i] * g[i];
                g[i + 1] = two * term * d / (w * w);
                g[i + 2] = two * term * d * d / w.powi(3);
                y = y + term;
            }
            y
        }
        "Eckerle4" => {
            let d = (t - b[2]) / b[1];
            g[0] = (-d * d / two).exp() / b[1];
            let y = b[0] * g[0];
            g[1] = y * (d * d - one) / b[1];
            g[2] = y * d / b[1];
            y
        }
        "Kirby2" | "Hahn1" | "Thurber" => {
            let degree = (b.len() - 1) / 2;
            let mut numerator = b[degree];
            let mut denominator = b[2 * degree];
            for i in (0..degree).rev() {
                numerator = numerator * t + b[i];
                denominator =
                    denominator * t + if i == 0 { one } else { b[degree + i] };
            }
            let y = numerator / denominator;
            let mut power = one;
            for i in 0..=degree {
                g[i] = power / denominator;
                if i > 0 {
                    g[degree + i] = -y * power / denominator;
                }
                power = power * t;
            }
            y
        }
        "MGH09" => {
            let numerator = t * t + t * b[1];
            let denominator = t * t + t * b[2] + b[3];
            g[0] = numerator / denominator;
            let y = b[0] * g[0];
            g[1] = b[0] * t / denominator;
            g[2] = -y * t / denominator;
            g[3] = -y / denominator;
            y
        }
        "Nelson" => {
            let e = (-b[2] * x[1]).exp();
            g[0] = one;
            g[1] = -t * e;
            g[2] = b[1] * t * x[1] * e;
            b[0] - b[1] * t * e
        }
        "Roszman1" => {
            let pi = c(std::f64::consts::PI);
            let d = t - b[3];
            let q = b[2] / d;
            g[0] = one;
            g[1] = -t;
            g[2] = -one / (pi * d * (one + q * q));
            g[3] = -q / (pi * d * (one + q * q));
            // atan2 changes the published principal branch for negative d.
            b[0] - b[1] * t - q.atan() / pi
        }
        "ENSO" => {
            let tau_t = c(std::f64::consts::TAU) * t;
            g[0] = one;
            g[1] = (tau_t / c(12.0)).cos();
            g[2] = (tau_t / c(12.0)).sin();
            let mut y = b[0] + b[1] * g[1] + b[2] * g[2];
            for i in [3, 6] {
                let a = tau_t / b[i];
                g[i + 1] = a.cos();
                g[i + 2] = a.sin();
                g[i] = a / b[i] * (b[i + 1] * a.sin() - b[i + 2] * a.cos());
                y = y + b[i + 1] * g[i + 1] + b[i + 2] * g[i + 2];
            }
            y
        }
        "Rat42" | "Rat43" => {
            let z = b[1] - b[2] * t;
            // Softplus avoids overflow without changing the logistic model.
            let log_q = z.max(F::zero()) + (-z.abs()).exp().ln_1p();
            let sigmoid = if z >= F::zero() {
                one / (one + (-z).exp())
            } else {
                z.exp() / (one + z.exp())
            };
            let power = if id == "Rat43" { one / b[3] } else { one };
            g[0] = (-power * log_q).exp();
            let y = b[0] * g[0];
            g[1] = -y * power * sigmoid;
            g[2] = -t * g[1];
            if id == "Rat43" {
                g[3] = y * log_q / (b[3] * b[3]);
            }
            y
        }
        _ => unreachable!("validated NIST dataset ID"),
    };
    (y, g)
}
