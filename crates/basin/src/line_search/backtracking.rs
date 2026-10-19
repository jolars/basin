use crate::core::math::{Dot, Scalar, ScaledAdd};
use crate::core::problem::{CostFunction, Problem};
use crate::line_search::{
    LineSearch, LineSearchBounds, LineSearchOutcome, LineSearchResult,
};

/// Backtracking line search satisfying the Armijo condition only
/// (Nocedal & Wright §3.1). Halves the trial step until
/// `f(x + α d) ≤ f(x) + c · α · ∇f(x)ᵀd`.
/// Exhaustion reports [`LineSearchOutcome::Failed`]; the legacy
/// [`LineSearch::next`] method returns zero on failure.
///
/// # Configuration
///
/// Construct with [`Self::new`] or [`Self::default`], then use the builders or
/// assign public fields. Destructuring requires `..`. Future controls must
/// default to behavior compatible with existing constructors.
///
/// Struct literals, including update syntax, are unavailable outside Basin:
///
/// ```compile_fail,E0639
/// use basin::Backtracking;
///
/// let search = Backtracking::<f64> {
///     max_iter: 100,
///     ..Backtracking::new()
/// };
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
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

    /// Configure the sufficient decrease coefficient.
    /// Uses the same mathematical condition and validation as the original setting.
    pub fn with_sufficient_decrease_coefficient(mut self, c: F) -> Self {
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
        self.next_with_outcome(problem, param, cost, gradient, direction)
            .map(|outcome| match outcome {
                LineSearchOutcome::Step(alpha) => alpha,
                LineSearchOutcome::Failed => F::zero(),
            })
    }

    fn next_with_outcome(
        &mut self,
        problem: &mut Problem<P>,
        param: &V,
        cost: F,
        gradient: &V,
        direction: &V,
    ) -> Result<LineSearchOutcome<F>, Self::Error> {
        // Armijo: f(x + α d) ≤ f(x) + c α (∇f · d). For a descent direction,
        // `g_dot_d` is negative, so the threshold drops with α.
        let g_dot_d = gradient.dot(direction);
        let mut alpha = self.alpha_init;
        let valid = cost.is_finite()
            && g_dot_d.is_finite()
            && g_dot_d <= F::zero()
            && alpha.is_finite()
            && alpha > F::zero()
            && F::zero() < self.rho
            && self.rho < F::one()
            && F::zero() < self.c
            && self.c < F::one();
        if !valid {
            return Ok(LineSearchOutcome::Failed);
        }
        for _ in 0..self.max_iter {
            if alpha <= F::zero() {
                break;
            }
            let mut trial = param.clone();
            trial.scaled_add(alpha, direction);
            let trial_cost = problem.cost(&trial)?;
            if trial_cost.is_finite()
                && trial_cost <= cost + self.c * alpha * g_dot_d
            {
                return Ok(LineSearchOutcome::Step(alpha));
            }
            alpha = alpha * self.rho;
        }
        Ok(LineSearchOutcome::Failed)
    }

    fn next_with_bounds(
        &mut self,
        problem: &mut Problem<P>,
        param: &V,
        cost: F,
        gradient: &V,
        direction: &V,
        bounds: LineSearchBounds<F>,
    ) -> Result<LineSearchResult<V, F>, Self::Error> {
        if !(F::zero() < self.rho && self.rho < F::one()) {
            return Ok(LineSearchResult::new(LineSearchOutcome::Failed));
        }
        let mut search = Self {
            alpha_init: bounds.initial,
            ..*self
        };
        search.next_with_evaluation(problem, param, cost, gradient, direction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct RejectEveryTrial<F>(F);

    impl<F: Scalar> crate::ProblemConstraints for RejectEveryTrial<F> {
        type Constraints = crate::NoConstraints;
        fn constraint_set(
            &self,
        ) -> <Self::Constraints as crate::ConstraintDescription>::Borrowed<'_>
        {
            crate::Constraints::new()
        }
    }
    impl<F: Scalar> CostFunction for RejectEveryTrial<F> {
        type Param = Vec<F>;
        type Output = F;
        type Error = std::convert::Infallible;

        fn cost(&self, _: &Vec<F>) -> Result<F, Self::Error> {
            Ok(self.0)
        }
    }

    fn exhausted_outcomes<F: Scalar>() {
        let x = vec![F::zero()];
        let gradient = vec![-F::one()];
        let direction = vec![F::one()];
        for cap in [0, 5] {
            let mut problem = Problem::new(RejectEveryTrial(F::one()));
            let mut search = Backtracking::<F>::new().max_iter(cap);
            assert_eq!(
                search
                    .next_with_outcome(
                        &mut problem,
                        &x,
                        F::zero(),
                        &gradient,
                        &direction,
                    )
                    .unwrap(),
                LineSearchOutcome::Failed,
            );
            assert_eq!(problem.counts().cost_evals, u64::from(cap));

            let mut problem = Problem::new(RejectEveryTrial(F::infinity()));
            assert_eq!(
                search
                    .next_with_bounds(
                        &mut problem,
                        &x,
                        F::zero(),
                        &gradient,
                        &direction,
                        LineSearchBounds::new(F::one(), F::one()),
                    )
                    .unwrap()
                    .outcome,
                LineSearchOutcome::Failed,
            );
            assert_eq!(problem.counts().cost_evals, u64::from(cap));
        }
    }

    #[test]
    fn exhaustion_reports_failure_in_both_search_modes() {
        exhausted_outcomes::<f64>();
        exhausted_outcomes::<f32>();
    }

    /// 1D quadratic via Vec<f64>: f(x) = (x[0] − 3)². Min at x = 3,
    /// ∇f = 2(x − 3).
    struct Quadratic;

    impl crate::ProblemConstraints for Quadratic {
        type Constraints = crate::NoConstraints;
        fn constraint_set(
            &self,
        ) -> <Self::Constraints as crate::ConstraintDescription>::Borrowed<'_>
        {
            crate::Constraints::new()
        }
    }
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
    fn legacy_next_rejects_an_ascent_direction_without_trials() {
        let mut ls = Backtracking::new().max_iter(5);
        let (alpha, cost_evals) = run(&mut ls, &[0.0], &[-6.0], &[-6.0]);
        assert_eq!(cost_evals, 0);
        assert_eq!(alpha, 0.0);
    }
}
