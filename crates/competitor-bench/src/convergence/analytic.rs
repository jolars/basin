//! Analytic least-squares controls with a known solution and spectrum.

use super::dual::Real;

#[derive(Clone, Debug)]
pub struct Quadratic {
    pub dimension: usize,
    /// Condition number of the objective Hessian, before optional rank loss.
    pub condition: f64,
    pub rotated: bool,
    pub rank_deficient: bool,
    /// Paired observation errors leave the exact minimizer unchanged.
    pub noise: f64,
}

impl Quadratic {
    pub fn reference(&self) -> Vec<f64> {
        (0..self.dimension)
            .map(|i| (i + 1) as f64 / self.dimension as f64)
            .collect()
    }

    pub fn residuals<T: Real>(&self, p: &[T]) -> Vec<T> {
        let n = self.dimension;
        let mut z: Vec<_> = p
            .iter()
            .zip(self.reference())
            .map(|(&p, r)| p - T::number(r))
            .collect();
        if self.rotated {
            // A Householder reflection preserves the spectrum. Unequal
            // components avoid a mere coordinate swap in two dimensions.
            let dot =
                z.iter().enumerate().fold(T::number(0.0), |a, (i, &b)| {
                    a + T::number((i + 1) as f64) * b
                });
            let norm2 = (1..=n).map(|i| (i * i) as f64).sum::<f64>();
            for (i, zi) in z.iter_mut().enumerate() {
                *zi = *zi - T::number(2.0 * (i + 1) as f64 / norm2) * dot;
            }
        }
        z.into_iter()
            .enumerate()
            .flat_map(|(i, zi)| {
                let eigenvalue = if self.rank_deficient && i == n - 1 {
                    0.0
                } else {
                    self.condition.powf(i as f64 / (n - 1) as f64)
                };
                let r = T::number((eigenvalue / 2.0).sqrt()) * zi;
                [r + T::number(self.noise), r - T::number(self.noise)]
            })
            .collect()
    }
}
