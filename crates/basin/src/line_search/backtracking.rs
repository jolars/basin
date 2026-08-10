use crate::core::math::{Dot, Scalar, ScaledAdd};
use crate::core::problem::{CostFunction, Problem};
use crate::line_search::LineSearch;

/// Backtracking line search satisfying the Armijo condition only
/// (Nocedal & Wright §3.1). Halves the trial step until
/// `f(x + α d) ≤ f(x) + c · α · ∇f(x)ᵀd`.
pub struct Backtracking<F = f64> {
    /// Initial trial step. Default `1.0`.
    pub alpha_init: F,
    /// Backtracking factor in `(0, 1)`. Default `0.5`.
    pub rho: F,
    /// Armijo slope coefficient in `(0, 1)`. Default `1e-4`.
    pub c: F,
    /// Maximum number of backtracks before giving up. Default `50`.
    pub max_iter: u32,
}

impl<F: Scalar> Default for Backtracking<F> {
    fn default() -> Self {
        Self {
            alpha_init: F::one(),
            rho: F::from_f64(0.5).unwrap(),
            c: F::from_f64(1e-4).unwrap(),
            max_iter: 50,
        }
    }
}

impl<F: Scalar> Backtracking<F> {
    /// Backtracking line search with default parameters
    /// (`α_init = 1.0`, `ρ = 0.5`, `c = 1e-4`, `max_iter = 50`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the initial trial step.
    pub fn alpha_init(mut self, alpha_init: F) -> Self {
        self.alpha_init = alpha_init;
        self
    }

    /// Override the backtracking factor.
    pub fn rho(mut self, rho: F) -> Self {
        self.rho = rho;
        self
    }

    /// Override the Armijo slope coefficient.
    pub fn c(mut self, c: F) -> Self {
        self.c = c;
        self
    }

    /// Override the maximum number of backtracks.
    pub fn max_iter(mut self, max_iter: u32) -> Self {
        self.max_iter = max_iter;
        self
    }
}

impl<P, V, F> LineSearch<P, V, F> for Backtracking<F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>,
    V: ScaledAdd<F> + Dot<F> + Clone,
{
    type Error = P::Error;

    fn next(
        &mut self,
        problem: &mut Problem<P>,
        param: &V,
        cost: F,
        gradient: &V,
        direction: &V,
    ) -> Result<F, Self::Error> {
        // Armijo: f(x + α d) ≤ f(x) + c α (∇f · d). For a descent direction,
        // `g_dot_d` is negative, so the threshold drops with α.
        let g_dot_d = gradient.dot(direction);
        let mut alpha = self.alpha_init;
        for _ in 0..self.max_iter {
            let mut trial = param.clone();
            trial.scaled_add(alpha, direction);
            let trial_cost = problem.cost(&trial)?;
            if trial_cost <= cost + self.c * alpha * g_dot_d {
                return Ok(alpha);
            }
            alpha = alpha * self.rho;
        }
        Ok(alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1D quadratic via Vec<f64>: f(x) = (x[0] − 3)². Min at x = 3,
    /// ∇f = 2(x − 3).
    struct Quadratic;

    impl CostFunction for Quadratic {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
            Ok((x[0] - 3.0).powi(2))
        }
    }

    /// (alpha, cost_evals consumed by the line search).
    fn run(
        ls: &mut Backtracking,
        x: &[f64],
        grad: &[f64],
        dir: &[f64],
    ) -> (f64, u64) {
        let mut p = Problem::new(Quadratic);
        let x = x.to_vec();
        let f0 = p.cost(&x).unwrap();
        // The probing cost above is not part of the line search; reset.
        let baseline = p.counts().cost_evals;
        let g = grad.to_vec();
        let d = dir.to_vec();
        let alpha = ls.next(&mut p, &x, f0, &g, &d).unwrap();
        (alpha, p.counts().cost_evals - baseline)
    }

    #[test]
    fn accepts_alpha_init_when_armijo_holds() {
        // Start at x=2, d=+1 (descent: g=−2, gᵀd=−2 < 0). α_init=0.5
        // → x=2.5, f=0.25, threshold 1 − 1e-4·0.5·2 = 0.9999. 0.25 ≤ 0.9999 ✓.
        let mut ls = Backtracking::new().alpha_init(0.5);
        let (alpha, cost_evals) = run(&mut ls, &[2.0], &[-2.0], &[1.0]);
        assert_eq!(alpha, 0.5, "expected α_init accepted on first try");
        assert_eq!(cost_evals, 1);
    }

    #[test]
    fn backtracks_when_initial_alpha_overshoots() {
        // From x=0, g=−6, direction d=+6. α_init=1.0 lands at x=6, f=9
        // (way past minimum at x=3). Backtrack until Armijo holds.
        let mut ls = Backtracking::new(); // ρ=0.5, c=1e-4, max_iter=50
        let (alpha, cost_evals) = run(&mut ls, &[0.0], &[-6.0], &[6.0]);
        let f0 = 9.0; // (0-3)^2
        let f_new = (alpha * 6.0 - 3.0).powi(2);
        let g_dot_d = (-6.0_f64) * 6.0;
        assert!(
            f_new <= f0 + 1e-4 * alpha * g_dot_d,
            "Armijo violated: f_new={f_new}, threshold={}",
            f0 + 1e-4 * alpha * g_dot_d,
        );
        assert!(alpha < 1.0, "expected backtrack, got α={alpha}");
        assert!(cost_evals > 1);
    }

    #[test]
    fn reports_cost_eval_count() {
        let mut ls = Backtracking::new().rho(0.5);
        let (_, cost_evals) = run(&mut ls, &[0.0], &[-6.0], &[6.0]);
        assert!(cost_evals >= 1);
        assert!(
            cost_evals <= ls.max_iter as u64,
            "cost_evals={cost_evals} exceeds max_iter={}",
            ls.max_iter
        );
    }

    #[test]
    fn caps_at_max_iter_when_armijo_never_holds() {
        // Wrong-sign direction: gᵀd > 0, so f increases with α and Armijo
        // (with descent-direction assumption) is unsatisfiable. Backtrack
        // burns max_iter cost evals and returns the smallest α tried.
        let mut ls = Backtracking::new().max_iter(5);
        let (alpha, cost_evals) = run(&mut ls, &[0.0], &[-6.0], &[-6.0]);
        assert_eq!(cost_evals, 5);
        // α reduced 5 times by ρ=0.5 from 1.0 → 1/32.
        assert!(
            (alpha - 1.0 / 32.0).abs() < 1e-12,
            "expected α=1/32, got {alpha}",
        );
    }
}
