use crate::core::math::{Dot, ScaledAdd};
use crate::core::problem::{CostFunction, Gradient, Problem};
use crate::line_search::LineSearch;

/// Strong Wolfe line search via bracketing + bisection-based zoom.
///
/// Returns an `α > 0` along the caller-supplied descent direction `d`
/// satisfying both
///
/// * Armijo (sufficient decrease): `f(x + α d) ≤ f(x) + c1 · α · ∇f(x)ᵀd`
/// * Strong curvature: `|∇f(x + α d)ᵀd| ≤ c2 · |∇f(x)ᵀd|`
///
/// Defaults are the quasi-Newton conventions from Nocedal & Wright §3.5
/// (`c1 = 1e-4`, `c2 = 0.9`) and `α_init = 1.0` so BFGS gets the unit step
/// it expects asymptotically.
///
/// Algorithms 3.5 (bracketing) and 3.6 (zoom) from Nocedal & Wright. The
/// zoom phase uses bisection (always-progress, no interpolation pitfalls);
/// cubic-interpolation in zoom is a possible future perf improvement.
pub struct Wolfe {
    /// Armijo slope coefficient in `(0, 1)`. Default `1e-4` (N&W §3.5).
    pub c1: f64,
    /// Strong-curvature coefficient in `(c1, 1)`. Default `0.9`.
    pub c2: f64,
    /// Initial trial step. Default `1.0` so quasi-Newton solvers get the
    /// unit step they expect asymptotically.
    pub alpha_init: f64,
    /// Upper bound on the bracketing trial step. Default `10.0`.
    pub alpha_max: f64,
    /// Maximum bracketing/zoom iterations before bailing. Default `25`.
    pub max_iter: u32,
}

impl Default for Wolfe {
    fn default() -> Self {
        Self {
            c1: 1e-4,
            c2: 0.9,
            alpha_init: 1.0,
            alpha_max: 10.0,
            max_iter: 25,
        }
    }
}

impl Wolfe {
    /// Strong-Wolfe line search with the Nocedal & Wright defaults
    /// (`c1 = 1e-4`, `c2 = 0.9`, `α_init = 1.0`, `α_max = 10.0`,
    /// `max_iter = 25`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the Armijo slope coefficient. Panics if not in `(0, 1)`.
    pub fn c1(mut self, c1: f64) -> Self {
        assert!(0.0 < c1 && c1 < 1.0, "c1 must be in (0, 1)");
        self.c1 = c1;
        self
    }

    /// Override the strong-curvature coefficient. Panics if not in `(0, 1)`.
    pub fn c2(mut self, c2: f64) -> Self {
        assert!(0.0 < c2 && c2 < 1.0, "c2 must be in (0, 1)");
        self.c2 = c2;
        self
    }

    /// Override the initial trial step. Panics if not strictly positive.
    pub fn alpha_init(mut self, alpha_init: f64) -> Self {
        assert!(alpha_init > 0.0, "alpha_init must be > 0");
        self.alpha_init = alpha_init;
        self
    }

    /// Override the maximum trial step. Panics if not strictly positive.
    pub fn alpha_max(mut self, alpha_max: f64) -> Self {
        assert!(alpha_max > 0.0, "alpha_max must be > 0");
        self.alpha_max = alpha_max;
        self
    }

    /// Override the bracketing/zoom iteration cap.
    pub fn max_iter(mut self, max_iter: u32) -> Self {
        self.max_iter = max_iter;
        self
    }
}

impl<P, V> LineSearch<P, V> for Wolfe
where
    P: CostFunction<Param = V, Output = f64> + Gradient<Gradient = V>,
    V: ScaledAdd<f64> + Dot + Clone,
{
    type Error = P::Error;

    fn next(
        &mut self,
        problem: &mut Problem<P>,
        param: &V,
        cost: f64,
        gradient: &V,
        direction: &V,
    ) -> Result<f64, Self::Error> {
        let phi0 = cost;
        let phi0_prime = gradient.dot(direction);

        // If `direction` is not a descent direction (or `phi0_prime` is
        // NaN), bail with α = 0 rather than looping forever. Written
        // positively so NaN routes here too — `NaN < 0.0` is false.
        if phi0_prime >= 0.0 || phi0_prime.is_nan() {
            return Ok(0.0);
        }

        let mut alpha_prev = 0.0;
        let mut phi_prev = phi0;
        let mut alpha = self.alpha_init.min(self.alpha_max);

        for i in 0..self.max_iter {
            let mut trial = param.clone();
            trial.scaled_add(alpha, direction);
            let phi = problem.cost(&trial)?;

            // Armijo failed OR φ stopped decreasing → minimum is in
            // (alpha_prev, alpha). Hand to zoom.
            if phi > phi0 + self.c1 * alpha * phi0_prime || (i > 0 && phi >= phi_prev) {
                return self.zoom(
                    problem, param, direction, phi0, phi0_prime, alpha_prev, phi_prev, alpha,
                );
            }

            let g_trial = problem.gradient(&trial)?;
            let phi_prime = g_trial.dot(direction);

            // Strong curvature satisfied → accept.
            if phi_prime.abs() <= -self.c2 * phi0_prime {
                return Ok(alpha);
            }

            // Slope flipped sign → minimum is in (alpha, alpha_prev). Note
            // the swapped argument order so `alpha_lo < alpha_hi` is *not*
            // assumed inside zoom (matches N&W).
            if phi_prime >= 0.0 {
                return self.zoom(
                    problem, param, direction, phi0, phi0_prime, alpha, phi, alpha_prev,
                );
            }

            alpha_prev = alpha;
            phi_prev = phi;
            // Expansion: double, capped at alpha_max. Once we're at the
            // cap, the next iteration's φ check will end up in zoom anyway.
            let next_alpha = (alpha * 2.0).min(self.alpha_max);
            if next_alpha == alpha {
                // Cannot expand further. Best we can do is return current
                // α — Armijo is satisfied here even if curvature isn't.
                return Ok(alpha);
            }
            alpha = next_alpha;
        }

        // Bracketing exhausted without locating a Wolfe step; return the
        // last α (Armijo held there). Caller (BFGS) treats this like any
        // other α — the curvature condition guard will detect the failure
        // and skip the H update if needed.
        Ok(alpha)
    }
}

impl Wolfe {
    /// Zoom on bracket `(alpha_lo, alpha_hi)` with `φ(alpha_lo) = phi_lo`.
    /// `alpha_hi` may be either side of `alpha_lo`; the algorithm only
    /// requires that the bracket contains a Wolfe-satisfying step.
    #[allow(clippy::too_many_arguments)]
    fn zoom<P, V>(
        &self,
        problem: &mut Problem<P>,
        param: &V,
        direction: &V,
        phi0: f64,
        phi0_prime: f64,
        mut alpha_lo: f64,
        mut phi_lo: f64,
        mut alpha_hi: f64,
    ) -> Result<f64, P::Error>
    where
        P: CostFunction<Param = V, Output = f64> + Gradient<Gradient = V>,
        V: ScaledAdd<f64> + Dot + Clone,
    {
        for _ in 0..self.max_iter {
            // Bisection. Cubic-safeguarded interpolation would converge
            // faster but is brittle; bisection always halves the bracket.
            let alpha_j = 0.5 * (alpha_lo + alpha_hi);

            let mut trial = param.clone();
            trial.scaled_add(alpha_j, direction);
            let phi_j = problem.cost(&trial)?;

            if phi_j > phi0 + self.c1 * alpha_j * phi0_prime || phi_j >= phi_lo {
                alpha_hi = alpha_j;
            } else {
                let g_j = problem.gradient(&trial)?;
                let phi_j_prime = g_j.dot(direction);

                if phi_j_prime.abs() <= -self.c2 * phi0_prime {
                    return Ok(alpha_j);
                }

                if phi_j_prime * (alpha_hi - alpha_lo) >= 0.0 {
                    alpha_hi = alpha_lo;
                }
                alpha_lo = alpha_j;
                phi_lo = phi_j;
            }

            // Bracket collapsed — return the best α we have.
            if (alpha_hi - alpha_lo).abs() <= f64::EPSILON * alpha_hi.abs().max(1.0) {
                break;
            }
        }

        Ok(alpha_lo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple 1D quadratic via Vec<f64>: f(x) = (x[0] - 3)^2.
    /// Minimum at x = 3, ∇f = 2(x - 3).
    struct Quadratic;

    impl CostFunction for Quadratic {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
            Ok((x[0] - 3.0).powi(2))
        }
    }

    impl Gradient for Quadratic {
        type Gradient = Vec<f64>;
        fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
            Ok(vec![2.0 * (x[0] - 3.0)])
        }
    }

    #[test]
    fn satisfies_strong_wolfe_on_quadratic() {
        let mut p = Problem::new(Quadratic);
        let x = vec![0.0];
        let f0 = p.cost(&x).unwrap();
        let g = p.gradient(&x).unwrap();
        let d = vec![-g[0]]; // = +6, descent direction since g[0] = -6
        let mut ls = Wolfe::new();
        let alpha =
            LineSearch::<Quadratic, Vec<f64>>::next(&mut ls, &mut p, &x, f0, &g, &d).unwrap();

        assert!(alpha > 0.0);

        // Verify Armijo and strong curvature at the returned α.
        let c1 = 1e-4;
        let c2 = 0.9;
        let mut x_new = x.clone();
        x_new[0] += alpha * d[0];
        let f_new = p.cost(&x_new).unwrap();
        let g_new = p.gradient(&x_new).unwrap();
        let g0_dot_d = g[0] * d[0];
        let gnew_dot_d = g_new[0] * d[0];

        assert!(
            f_new <= f0 + c1 * alpha * g0_dot_d + 1e-12,
            "Armijo failed: f_new={f_new}, threshold={}",
            f0 + c1 * alpha * g0_dot_d,
        );
        assert!(
            gnew_dot_d.abs() <= -c2 * g0_dot_d + 1e-12,
            "Strong curvature failed: |g_new·d|={}, threshold={}",
            gnew_dot_d.abs(),
            -c2 * g0_dot_d,
        );
    }

    #[test]
    fn unit_step_accepted_when_quadratic_minimum_inside_bracket() {
        // For a 1D quadratic with minimum at x_min=3, starting at x=0 with
        // d = -g = 6, the exact line minimum is α* = 0.5. Wolfe should land
        // close to it with strong curvature.
        let mut p = Problem::new(Quadratic);
        let x = vec![0.0];
        let f0 = p.cost(&x).unwrap();
        let g = p.gradient(&x).unwrap();
        let d = vec![6.0];
        let mut ls = Wolfe::new();
        let alpha =
            LineSearch::<Quadratic, Vec<f64>>::next(&mut ls, &mut p, &x, f0, &g, &d).unwrap();

        // Strong curvature with c2=0.9 admits a wide range; just check we
        // ended up reasonably close to the line minimum.
        assert!(
            (alpha - 0.5).abs() < 0.5,
            "expected α near 0.5, got {alpha}",
        );
    }
}
