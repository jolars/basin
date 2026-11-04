use crate::core::math::{Dot, Scalar, ScaledAdd};
use crate::core::problem::{CostFunction, Gradient, Problem};
use crate::line_search::{
    LineSearch, LineSearchBounds, LineSearchOutcome, LineSearchResult,
};

/// Strong Wolfe with fused cost-and-gradient evaluation at every trial.
///
/// Construct by configuring [`Wolfe`] and calling
/// [`Wolfe::with_fused_evaluations`]. This uses the same Nocedal & Wright
/// Algorithms 3.5/3.6 (bracketing and bisection zoom) and accepted-point reuse.
/// Unlike ordinary [`Wolfe`], Armijo-rejected trials also request gradients.
/// Choose this strategy when a fused callback shares expensive intermediate
/// work; separate inexpensive costs and expensive gradients can favor `Wolfe`.
/// Hard evaluation errors propagate without retrying a separate callback.
///
/// # Backends
///
/// Supports the same `ScaledAdd<F> + Dot<F> + Clone` parameter types as
/// [`Wolfe`]: `Vec<F>`, nalgebra `DVector<F>`, ndarray `Array1<F>`, and faer
/// `Col<F>`, with both `f32` and `f64`.
///
/// # Examples
///
/// ```
/// use basin::{Tnc, Wolfe};
/// let search = Wolfe::<f64>::new().max_iter(60).with_fused_evaluations();
/// let solver = Tnc::<_, Vec<f64>>::with_line_search(search);
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FusedWolfe<F: Scalar = f64> {
    search: Wolfe<F>,
}

impl<F: Scalar> Default for FusedWolfe<F> {
    fn default() -> Self {
        Wolfe::new().with_fused_evaluations()
    }
}

impl<P, V, F> LineSearch<P, V, F> for FusedWolfe<F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V>,
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
        let result = self
            .next_with_evaluation(problem, param, cost, gradient, direction)?;
        Ok(match result.outcome {
            LineSearchOutcome::Step(alpha) => alpha,
            LineSearchOutcome::Failed => F::zero(),
        })
    }

    fn next_with_evaluation(
        &mut self,
        problem: &mut Problem<P>,
        param: &V,
        cost: F,
        gradient: &V,
        direction: &V,
    ) -> Result<LineSearchResult<V, F>, Self::Error> {
        self.search
            .search(problem, param, cost, gradient, direction, true)
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
        let max = self.search.alpha_max.min(bounds.max);
        if !(max.is_finite() && max > F::zero()) {
            return Ok(LineSearchResult::new(LineSearchOutcome::Failed));
        }
        let search = Wolfe {
            alpha_init: bounds.initial.min(max),
            alpha_max: max,
            ..self.search
        };
        search.search(problem, param, cost, gradient, direction, true)
    }
}

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
///
/// The evaluation-returning methods retain the objective and gradient when
/// accepting an evaluated trial, so callers can reuse them. By default, rejected
/// Armijo trials request only the objective. [`Self::with_fused_evaluations`]
/// instead evaluates cost and gradient together at every trial. Exhaustion may return a step without
/// an evaluation; callers must evaluate that point themselves.
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
/// use basin::Wolfe;
///
/// let search = Wolfe::<f64> {
///     max_iter: 50,
///     ..Wolfe::new()
/// };
/// ```
///
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct Wolfe<F = f64> {
    /// Armijo slope coefficient in `(0, 1)`. Default `1e-4` (N&W §3.5).
    pub c1: F,
    /// Strong-curvature coefficient in `(c1, 1)`. Default `0.9`.
    pub c2: F,
    /// Initial trial step. Default `1.0` so quasi-Newton solvers get the
    /// unit step they expect asymptotically.
    pub alpha_init: F,
    /// Upper bound on the bracketing trial step. Default `10.0`.
    pub alpha_max: F,
    /// Maximum bracketing or zoom iterations before bailing. Default `25`.
    pub max_iter: u32,
}

impl<F: Scalar> Default for Wolfe<F> {
    fn default() -> Self {
        Self {
            c1: F::from_f64(1e-4).unwrap(),
            c2: F::from_f64(0.9).unwrap(),
            alpha_init: F::one(),
            alpha_max: F::from_f64(10.0).unwrap(),
            max_iter: 25,
        }
    }
}

impl<F: Scalar> Wolfe<F> {
    /// Strong-Wolfe line search with the Nocedal & Wright defaults
    /// (`c1 = 1e-4`, `c2 = 0.9`, `α_init = 1.0`, `α_max = 10.0`,
    /// `max_iter = 25`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure the sufficient decrease coefficient.
    /// Uses the same mathematical condition and validation as the original setting.
    pub fn with_sufficient_decrease_coefficient(mut self, c1: F) -> Self {
        assert!(F::zero() < c1 && c1 < F::one(), "c1 must be in (0, 1)");
        self.c1 = c1;
        self
    }

    /// Configure the curvature coefficient.
    /// Uses the same mathematical condition and validation as the original setting.
    pub fn with_curvature_coefficient(mut self, c2: F) -> Self {
        assert!(F::zero() < c2 && c2 < F::one(), "c2 must be in (0, 1)");
        self.c2 = c2;
        self
    }

    /// Override the initial trial step. Panics if not strictly positive.
    pub fn alpha_init(mut self, alpha_init: F) -> Self {
        assert!(alpha_init > F::zero(), "alpha_init must be > 0");
        self.alpha_init = alpha_init;
        self
    }

    /// Override the maximum trial step. Panics if not strictly positive.
    pub fn alpha_max(mut self, alpha_max: F) -> Self {
        assert!(alpha_max > F::zero(), "alpha_max must be > 0");
        self.alpha_max = alpha_max;
        self
    }

    /// Override the bracketing or zoom iteration cap.
    pub fn max_iter(mut self, max_iter: u32) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Evaluate cost and gradient together at every trial, including rejected
    /// trials. Configure the search before selecting this strategy.
    ///
    /// Returns a separate strategy so the original Wolfe configuration and
    /// serialized checkpoint layout remain unchanged.
    pub fn with_fused_evaluations(self) -> FusedWolfe<F> {
        FusedWolfe { search: self }
    }
}

impl<P, V, F> LineSearch<P, V, F> for Wolfe<F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V>,
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
        let result = self
            .next_with_evaluation(problem, param, cost, gradient, direction)?;
        Ok(match result.outcome {
            LineSearchOutcome::Step(alpha) => alpha,
            LineSearchOutcome::Failed => F::zero(),
        })
    }

    fn next_with_evaluation(
        &mut self,
        problem: &mut Problem<P>,
        param: &V,
        cost: F,
        gradient: &V,
        direction: &V,
    ) -> Result<LineSearchResult<V, F>, Self::Error> {
        self.search(problem, param, cost, gradient, direction, false)
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
        let max = self.alpha_max.min(bounds.max);
        if !(max.is_finite() && max > F::zero()) {
            return Ok(LineSearchResult::new(LineSearchOutcome::Failed));
        }
        let mut search = Self {
            alpha_init: bounds.initial.min(max),
            alpha_max: max,
            ..*self
        };
        search.next_with_evaluation(problem, param, cost, gradient, direction)
    }
}

impl<F: Scalar> Wolfe<F> {
    fn search<P, V>(
        &self,
        problem: &mut Problem<P>,
        param: &V,
        cost: F,
        gradient: &V,
        direction: &V,
        fused: bool,
    ) -> Result<LineSearchResult<V, F>, P::Error>
    where
        P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V>,
        V: ScaledAdd<F> + Dot<F> + Clone,
    {
        let phi0 = cost;
        let phi0_prime = gradient.dot(direction);

        // If `direction` is not a descent direction (or `phi0_prime` is
        // NaN), bail with α = 0 rather than looping forever. Written
        // positively so NaN routes here too: `NaN < 0.0` is false.
        if phi0_prime >= F::zero() || phi0_prime.is_nan() {
            return Ok(LineSearchResult::new(LineSearchOutcome::Step(
                F::zero(),
            )));
        }

        let two = F::from_f64(2.0).unwrap();
        let mut alpha_prev = F::zero();
        let mut phi_prev = phi0;
        let mut alpha = self.alpha_init.min(self.alpha_max);

        let mut gradient_buffer = None;
        let mut trial = param.clone();
        for i in 0..self.max_iter {
            if i > 0 {
                trial.clone_from(param);
            }
            trial.scaled_add(alpha, direction);
            let (phi, fused_gradient) = if fused {
                let (phi, g) = problem.cost_and_gradient_reusing(
                    &trial,
                    gradient_buffer.take(),
                )?;
                (phi, Some(g))
            } else {
                (problem.cost(&trial)?, None)
            };

            // Armijo failed OR φ stopped decreasing → minimum is in
            // (alpha_prev, alpha). Hand to zoom.
            if phi > phi0 + self.c1 * alpha * phi0_prime
                || (i > 0 && phi >= phi_prev)
            {
                if fused_gradient.is_some() {
                    gradient_buffer = fused_gradient;
                }
                return self.zoom(
                    problem,
                    param,
                    direction,
                    phi0,
                    phi0_prime,
                    alpha_prev,
                    phi_prev,
                    alpha,
                    &mut gradient_buffer,
                    trial,
                    fused,
                );
            }

            let g_trial = if let Some(g) = fused_gradient {
                g
            } else {
                problem.gradient_reusing(&trial, gradient_buffer.take())?
            };
            let phi_prime = g_trial.dot(direction);

            // Strong curvature satisfied → accept.
            if phi_prime.abs() <= -self.c2 * phi0_prime {
                return Ok(LineSearchResult::with_evaluation(
                    alpha, trial, phi, g_trial,
                ));
            }

            // Slope flipped sign → minimum is in (alpha, alpha_prev). Note
            // the swapped argument order so `alpha_lo < alpha_hi` is *not*
            // assumed inside zoom (matches N&W).
            if phi_prime >= F::zero() {
                gradient_buffer = Some(g_trial);
                return self.zoom(
                    problem,
                    param,
                    direction,
                    phi0,
                    phi0_prime,
                    alpha,
                    phi,
                    alpha_prev,
                    &mut gradient_buffer,
                    trial,
                    fused,
                );
            }

            alpha_prev = alpha;
            phi_prev = phi;
            // Expansion: double, capped at alpha_max. Once we're at the
            // cap, the next iteration's φ check will end up in zoom anyway.
            let next_alpha = (alpha * two).min(self.alpha_max);
            if next_alpha == alpha {
                // Cannot expand further. Best we can do is return the current
                // α: Armijo is satisfied here even if curvature isn't.
                return Ok(LineSearchResult::with_evaluation(
                    alpha, trial, phi, g_trial,
                ));
            }
            gradient_buffer = Some(g_trial);
            alpha = next_alpha;
        }

        // Bracketing exhausted without locating a Wolfe step; return the
        // last α (Armijo held there). Caller (BFGS) treats this like any
        // other α: the curvature condition guard will detect the failure
        // and skip the H update if needed.
        // Expansion may have advanced alpha past the last evaluated trial.
        // Do not attach that trial's values to the returned step.
        Ok(LineSearchResult::new(LineSearchOutcome::Step(alpha)))
    }

    /// Zoom on bracket `(alpha_lo, alpha_hi)` with `φ(alpha_lo) = phi_lo`.
    /// `alpha_hi` may be either side of `alpha_lo`; the algorithm only
    /// requires that the bracket contains a Wolfe-satisfying step.
    #[allow(clippy::too_many_arguments)]
    fn zoom<P, V>(
        &self,
        problem: &mut Problem<P>,
        param: &V,
        direction: &V,
        phi0: F,
        phi0_prime: F,
        mut alpha_lo: F,
        mut phi_lo: F,
        mut alpha_hi: F,
        gradient_buffer: &mut Option<V>,
        mut trial: V,
        fused: bool,
    ) -> Result<LineSearchResult<V, F>, P::Error>
    where
        P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V>,
        V: ScaledAdd<F> + Dot<F> + Clone,
    {
        let half = F::from_f64(0.5).unwrap();
        for _ in 0..self.max_iter {
            // Bisection. Cubic-safeguarded interpolation would converge
            // faster but is brittle; bisection always halves the bracket.
            let alpha_j = half * (alpha_lo + alpha_hi);

            trial.clone_from(param);
            trial.scaled_add(alpha_j, direction);
            let (phi_j, fused_gradient) = if fused {
                let (phi, g) = problem.cost_and_gradient_reusing(
                    &trial,
                    gradient_buffer.take(),
                )?;
                (phi, Some(g))
            } else {
                (problem.cost(&trial)?, None)
            };

            if phi_j > phi0 + self.c1 * alpha_j * phi0_prime || phi_j >= phi_lo
            {
                alpha_hi = alpha_j;
                if fused_gradient.is_some() {
                    *gradient_buffer = fused_gradient;
                }
            } else {
                let g_j = if let Some(g) = fused_gradient {
                    g
                } else {
                    problem.gradient_reusing(&trial, gradient_buffer.take())?
                };
                let phi_j_prime = g_j.dot(direction);

                if phi_j_prime.abs() <= -self.c2 * phi0_prime {
                    return Ok(LineSearchResult::with_evaluation(
                        alpha_j, trial, phi_j, g_j,
                    ));
                }
                *gradient_buffer = Some(g_j);

                if phi_j_prime * (alpha_hi - alpha_lo) >= F::zero() {
                    alpha_hi = alpha_lo;
                }
                alpha_lo = alpha_j;
                phi_lo = phi_j;
            }

            // Bracket collapsed: return the best α we have.
            if (alpha_hi - alpha_lo).abs()
                <= F::epsilon() * alpha_hi.abs().max(F::one())
            {
                break;
            }
        }

        // alpha_lo may be an earlier trial whose gradient buffer was reused,
        // or the unevaluated starting point. Preserve the step-only fallback.
        Ok(LineSearchResult::new(LineSearchOutcome::Step(alpha_lo)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple 1D quadratic via Vec<f64>: f(x) = (x[0] - 3)^2.
    /// Minimum at x = 3, ∇f = 2(x - 3).
    struct Quadratic;

    impl crate::Constraints for Quadratic {
        type Set = crate::NoConstraints;
        fn constraints(
            &self,
        ) -> <Self::Set as crate::ConstraintDescription>::Borrowed<'_> {
            crate::ConstraintSet::new()
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

    impl Gradient for Quadratic {
        type Gradient = Vec<f64>;
        fn gradient(
            &self,
            x: &Vec<f64>,
        ) -> Result<Vec<f64>, std::convert::Infallible> {
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
        let alpha = LineSearch::<Quadratic, Vec<f64>>::next(
            &mut ls, &mut p, &x, f0, &g, &d,
        )
        .unwrap();

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
        let alpha = LineSearch::<Quadratic, Vec<f64>>::next(
            &mut ls, &mut p, &x, f0, &g, &d,
        )
        .unwrap();

        // Strong curvature with c2=0.9 admits a wide range; just check we
        // ended up reasonably close to the line minimum.
        assert!(
            (alpha - 0.5).abs() < 0.5,
            "expected α near 0.5, got {alpha}",
        );
    }
}
