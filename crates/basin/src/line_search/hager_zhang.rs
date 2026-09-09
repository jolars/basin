use crate::core::math::{Dot, Scalar, ScaledAdd};
use crate::core::problem::{CostFunction, Gradient, Problem};
use crate::line_search::{
    LineSearch, LineSearchBounds, LineSearchOutcome, LineSearchResult,
};

/// Hager–Zhang line search with approximate-Wolfe safeguards.
///
/// The search finds an `α > 0` along the caller-supplied descent direction
/// `d`. It accepts a step satisfying either the ordinary Wolfe conditions
///
/// * `f(x + αd) − f(x) ≤ δ α ∇f(x)ᵀd`, and
/// * `∇f(x + αd)ᵀd ≥ σ ∇f(x)ᵀd`,
///
/// or the approximate-Wolfe conditions from Hager and Zhang. The latter
/// replace the cancellation-prone sufficient-decrease test near a minimizer
/// with a function-value tolerance and a two-sided derivative test.
///
/// Bracketing follows the expansion and contraction procedure in the paper;
/// refinement uses the published interval update and double-secant step. A
/// midpoint step is forced whenever the secant steps fail to reduce the
/// bracket by [`gamma`](Self::gamma).
///
/// Every trial point is evaluated once through
/// [`Gradient::cost_and_gradient`]. Non-finite trial data are soft rejections
/// and cause the search to contract. A problem `Err` remains a hard abort. If
/// the step bounds, floating-point resolution, or evaluation budget prevent a
/// Wolfe step from being found, [`LineSearch::next_with_outcome`] reports
/// [`LineSearchOutcome::Failed`]. The legacy [`LineSearch::next`] method maps
/// that outcome to `0`.
/// When invoked through [`LineSearch::next_with_bounds`], the search also
/// accepts an Armijo-decreasing upper endpoint with a negative slope, since
/// the constrained interval may contain no Wolfe step.
///
/// The `η` parameter sometimes exposed alongside this line search belongs to
/// the Hager–Zhang conjugate-gradient update, not to the line-search algorithm,
/// and is therefore intentionally absent here.
///
/// # Backends
///
/// Supported parameter types: `Vec<F>`, nalgebra vectors, ndarray vectors, and
/// faer vectors for every backend version supported by Basin.
///
/// # Reference
///
/// William W. Hager and Hongchao Zhang, “A new conjugate gradient method with
/// guaranteed descent and an efficient line search,” *SIAM Journal on
/// Optimization* 16(1), 2005, pp. 170–192.
/// [doi:10.1137/030601880](https://doi.org/10.1137/030601880).
pub struct HagerZhang<F = f64> {
    /// Sufficient-decrease coefficient in `(0, 0.5)`. Default `0.1`.
    pub delta: F,
    /// Curvature coefficient in `[delta, 1)`. Default `0.9`.
    pub sigma: F,
    /// Relative function-value tolerance for approximate Wolfe. Default
    /// `1e-6`.
    pub epsilon: F,
    /// Contraction fraction in `(0, 1)`. Default `0.5`.
    pub theta: F,
    /// Required bracket-width reduction in `(0, 1)`. Default `0.66`.
    pub gamma: F,
    /// Initial trial step. Default `1.0`.
    pub alpha_init: F,
    /// Smallest permitted trial step. Default is the machine epsilon for `F`.
    pub step_min: F,
    /// Largest permitted trial step. Default `1e5`.
    pub step_max: F,
    /// Expansion factor, which must exceed one. Default `5.0`.
    pub rho: F,
    /// Maximum number of fused cost-and-gradient evaluations. Default `100`.
    pub maxfev: u32,
}

impl<F: Scalar> Default for HagerZhang<F> {
    fn default() -> Self {
        Self {
            delta: F::from_f64(0.1).unwrap(),
            sigma: F::from_f64(0.9).unwrap(),
            epsilon: F::from_f64(1e-6).unwrap(),
            theta: F::from_f64(0.5).unwrap(),
            gamma: F::from_f64(0.66).unwrap(),
            alpha_init: F::one(),
            step_min: F::epsilon(),
            step_max: F::from_f64(1e5).unwrap(),
            rho: F::from_f64(5.0).unwrap(),
            maxfev: 100,
        }
    }
}

impl<F: Scalar> HagerZhang<F> {
    /// Construct a Hager–Zhang search with the parameters from the reference
    /// implementation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the ordinary and approximate-Wolfe coefficients.
    ///
    /// Panics unless `0 < delta < 0.5` and `delta <= sigma < 1`.
    pub fn delta_sigma(mut self, delta: F, sigma: F) -> Self {
        let half = F::from_f64(0.5).unwrap();
        assert!(
            delta.is_finite()
                && sigma.is_finite()
                && F::zero() < delta
                && delta < half
                && delta <= sigma
                && sigma < F::one(),
            "delta and sigma must satisfy 0 < delta < 0.5 and delta <= sigma < 1"
        );
        self.delta = delta;
        self.sigma = sigma;
        self
    }

    /// Set the non-negative relative function-value tolerance.
    pub fn epsilon(mut self, epsilon: F) -> Self {
        assert!(
            epsilon.is_finite() && epsilon >= F::zero(),
            "epsilon must be finite and >= 0"
        );
        self.epsilon = epsilon;
        self
    }

    /// Set the contraction fraction. Panics unless it is in `(0, 1)`.
    pub fn theta(mut self, theta: F) -> Self {
        assert!(
            theta.is_finite() && F::zero() < theta && theta < F::one(),
            "theta must be in (0, 1)"
        );
        self.theta = theta;
        self
    }

    /// Set the bracket-width safeguard. Panics unless it is in `(0, 1)`.
    pub fn gamma(mut self, gamma: F) -> Self {
        assert!(
            gamma.is_finite() && F::zero() < gamma && gamma < F::one(),
            "gamma must be in (0, 1)"
        );
        self.gamma = gamma;
        self
    }

    /// Set the initial trial step. Panics unless it is finite and positive.
    pub fn alpha_init(mut self, alpha_init: F) -> Self {
        assert!(
            alpha_init.is_finite() && alpha_init > F::zero(),
            "alpha_init must be finite and > 0"
        );
        self.alpha_init = alpha_init;
        self
    }

    /// Set the inclusive trial-step bounds.
    ///
    /// Panics unless both bounds are finite and `0 <= step_min < step_max`.
    pub fn bounds(mut self, step_min: F, step_max: F) -> Self {
        assert!(
            step_min.is_finite()
                && step_max.is_finite()
                && F::zero() <= step_min
                && step_min < step_max,
            "step bounds must satisfy 0 <= step_min < step_max"
        );
        self.step_min = step_min;
        self.step_max = step_max;
        self
    }

    /// Set the expansion factor. Panics unless it is finite and greater than
    /// one.
    pub fn rho(mut self, rho: F) -> Self {
        assert!(
            rho.is_finite() && rho > F::one(),
            "rho must be finite and > 1"
        );
        self.rho = rho;
        self
    }

    /// Set the fused-evaluation cap.
    pub fn maxfev(mut self, maxfev: u32) -> Self {
        self.maxfev = maxfev;
        self
    }

    fn valid_config(&self) -> bool {
        let zero = F::zero();
        let one = F::one();
        let half = F::from_f64(0.5).unwrap();
        self.delta.is_finite()
            && self.sigma.is_finite()
            && self.epsilon.is_finite()
            && self.theta.is_finite()
            && self.gamma.is_finite()
            && self.alpha_init.is_finite()
            && self.step_min.is_finite()
            && self.step_max.is_finite()
            && self.rho.is_finite()
            && zero < self.delta
            && self.delta < half
            && self.delta <= self.sigma
            && self.sigma < one
            && self.epsilon >= zero
            && zero < self.theta
            && self.theta < one
            && zero < self.gamma
            && self.gamma < one
            && self.alpha_init > zero
            && zero <= self.step_min
            && self.step_min < self.step_max
            && self.rho > one
    }

    fn acceptable(
        &self,
        trial: Trial<F>,
        phi0: F,
        dphi0: F,
        epsilon_k: F,
    ) -> bool {
        if !trial.phi.is_finite() || !trial.dphi.is_finite() {
            return false;
        }

        let ordinary = trial.phi - phi0 <= self.delta * trial.alpha * dphi0
            && trial.dphi >= self.sigma * dphi0;
        let two = F::from_f64(2.0).unwrap();
        let approximate = (two * self.delta - F::one()) * dphi0 >= trial.dphi
            && trial.dphi >= self.sigma * dphi0
            && trial.phi <= phi0 + epsilon_k;
        ordinary || approximate
    }

    fn secant(&self, a: Trial<F>, b: Trial<F>) -> Option<F> {
        let denominator = b.dphi - a.dphi;
        if !denominator.is_finite() || denominator == F::zero() {
            return None;
        }
        let alpha = a.alpha - a.dphi * (b.alpha - a.alpha) / denominator;
        if alpha.is_finite() && self.step_min <= alpha {
            Some(alpha)
        } else {
            None
        }
    }

    fn midpoint(&self, a: F, b: F) -> Option<F> {
        let half = F::from_f64(0.5).unwrap();
        let midpoint = (a + half * (b - a)).max(self.step_min);
        if midpoint.is_finite() && a < midpoint && midpoint < b {
            Some(midpoint)
        } else {
            None
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate<P, V>(
        &self,
        problem: &mut Problem<P>,
        param: &V,
        direction: &V,
        alpha: F,
        evaluations: &mut u32,
    ) -> Result<Option<Trial<F>>, P::Error>
    where
        P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V>,
        V: ScaledAdd<F> + Dot<F> + Clone,
    {
        if *evaluations >= self.maxfev {
            return Ok(None);
        }
        let mut point = param.clone();
        point.scaled_add(alpha, direction);
        *evaluations += 1;
        let (phi, gradient) = problem.cost_and_gradient(&point)?;
        Ok(Some(Trial {
            alpha,
            phi,
            dphi: gradient.dot(direction),
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn contract<P, V>(
        &self,
        problem: &mut Problem<P>,
        param: &V,
        direction: &V,
        mut a: Trial<F>,
        mut upper: F,
        phi0: F,
        dphi0: F,
        epsilon_k: F,
        evaluations: &mut u32,
    ) -> Result<SearchResult<F>, P::Error>
    where
        P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V>,
        V: ScaledAdd<F> + Dot<F> + Clone,
    {
        loop {
            let alpha = ((F::one() - self.theta) * a.alpha
                + self.theta * upper)
                .max(self.step_min);
            if !(alpha.is_finite() && a.alpha < alpha && alpha < upper) {
                return Ok(SearchResult::Failed);
            }
            let Some(trial) =
                self.evaluate(problem, param, direction, alpha, evaluations)?
            else {
                return Ok(SearchResult::Failed);
            };
            if self.acceptable(trial, phi0, dphi0, epsilon_k) {
                return Ok(SearchResult::Accepted(alpha));
            }
            if !trial.is_finite() {
                upper = alpha;
            } else if trial.dphi >= F::zero() {
                return Ok(SearchResult::Bracket(a, trial));
            } else if trial.phi > phi0 + epsilon_k {
                upper = alpha;
            } else {
                a = trial;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn update<P, V>(
        &self,
        problem: &mut Problem<P>,
        param: &V,
        direction: &V,
        a: Trial<F>,
        b: Trial<F>,
        trial: Trial<F>,
        phi0: F,
        dphi0: F,
        epsilon_k: F,
        evaluations: &mut u32,
    ) -> Result<SearchResult<F>, P::Error>
    where
        P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V>,
        V: ScaledAdd<F> + Dot<F> + Clone,
    {
        // U0: only an interior trial can alter the bracket.
        if !(a.alpha < trial.alpha && trial.alpha < b.alpha) {
            return Ok(SearchResult::Bracket(a, b));
        }
        // A rejected trial is still a safe upper bound. The synthetic
        // positive slope makes the next secant fall back to a midpoint.
        if !trial.is_finite() {
            return Ok(SearchResult::Bracket(a, Trial::rejected(trial.alpha)));
        }
        // U1: the opposite-slope condition closes the upper bracket.
        if trial.dphi >= F::zero() {
            return Ok(SearchResult::Bracket(a, trial));
        }
        // U2: a low point with negative slope advances the lower bracket.
        if trial.phi <= phi0 + epsilon_k {
            return Ok(SearchResult::Bracket(trial, b));
        }
        // U3: a high point with negative slope needs an inner contraction
        // before either endpoint invariant can be restored.
        self.contract(
            problem,
            param,
            direction,
            a,
            trial.alpha,
            phi0,
            dphi0,
            epsilon_k,
            evaluations,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn refine<P, V>(
        &self,
        problem: &mut Problem<P>,
        param: &V,
        direction: &V,
        mut a: Trial<F>,
        mut b: Trial<F>,
        phi0: F,
        dphi0: F,
        epsilon_k: F,
        evaluations: &mut u32,
    ) -> Result<LineSearchOutcome<F>, P::Error>
    where
        P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V>,
        V: ScaledAdd<F> + Dot<F> + Clone,
    {
        loop {
            let old_a = a;
            let old_b = b;
            let old_width = b.alpha - a.alpha;
            // S1: take a safeguarded secant and apply U0–U3.
            let Some(first_alpha) = self
                .secant(a, b)
                .filter(|alpha| a.alpha < *alpha && *alpha < b.alpha)
                .or_else(|| self.midpoint(a.alpha, b.alpha))
            else {
                return Ok(LineSearchOutcome::Failed);
            };
            let Some(first) = self.evaluate(
                problem,
                param,
                direction,
                first_alpha,
                evaluations,
            )?
            else {
                return Ok(LineSearchOutcome::Failed);
            };
            if self.acceptable(first, phi0, dphi0, epsilon_k) {
                return Ok(LineSearchOutcome::Step(first.alpha));
            }

            match self.update(
                problem,
                param,
                direction,
                a,
                b,
                first,
                phi0,
                dphi0,
                epsilon_k,
                evaluations,
            )? {
                SearchResult::Accepted(alpha) => {
                    return Ok(LineSearchOutcome::Step(alpha));
                }
                SearchResult::Bracket(next_a, next_b) => {
                    a = next_a;
                    b = next_b;
                }
                SearchResult::Failed => return Ok(LineSearchOutcome::Failed),
            }

            // S2/S3 choose the second secant from whichever endpoint the
            // first secant replaced. The S2 pair is intentionally reversed.
            let second_alpha = if first.alpha == b.alpha {
                self.secant(old_b, b)
            } else if first.alpha == a.alpha {
                self.secant(old_a, a)
            } else {
                None
            };
            if let Some(alpha) = second_alpha {
                if a.alpha < alpha && alpha < b.alpha {
                    // S4: apply U0–U3 once more at the double-secant point.
                    let Some(second) = self.evaluate(
                        problem,
                        param,
                        direction,
                        alpha,
                        evaluations,
                    )?
                    else {
                        return Ok(LineSearchOutcome::Failed);
                    };
                    if self.acceptable(second, phi0, dphi0, epsilon_k) {
                        return Ok(LineSearchOutcome::Step(second.alpha));
                    }
                    match self.update(
                        problem,
                        param,
                        direction,
                        a,
                        b,
                        second,
                        phi0,
                        dphi0,
                        epsilon_k,
                        evaluations,
                    )? {
                        SearchResult::Accepted(alpha) => {
                            return Ok(LineSearchOutcome::Step(alpha));
                        }
                        SearchResult::Bracket(next_a, next_b) => {
                            a = next_a;
                            b = next_b;
                        }
                        SearchResult::Failed => {
                            return Ok(LineSearchOutcome::Failed);
                        }
                    }
                }
            }

            // Hager–Zhang's L2 safeguard guarantees a fixed fractional
            // reduction when the double-secant step was too conservative.
            if b.alpha - a.alpha > self.gamma * old_width {
                let Some(alpha) = self.midpoint(a.alpha, b.alpha) else {
                    return Ok(LineSearchOutcome::Failed);
                };
                let Some(middle) = self.evaluate(
                    problem,
                    param,
                    direction,
                    alpha,
                    evaluations,
                )?
                else {
                    return Ok(LineSearchOutcome::Failed);
                };
                if self.acceptable(middle, phi0, dphi0, epsilon_k) {
                    return Ok(LineSearchOutcome::Step(middle.alpha));
                }
                match self.update(
                    problem,
                    param,
                    direction,
                    a,
                    b,
                    middle,
                    phi0,
                    dphi0,
                    epsilon_k,
                    evaluations,
                )? {
                    SearchResult::Accepted(alpha) => {
                        return Ok(LineSearchOutcome::Step(alpha));
                    }
                    SearchResult::Bracket(next_a, next_b) => {
                        a = next_a;
                        b = next_b;
                    }
                    SearchResult::Failed => {
                        return Ok(LineSearchOutcome::Failed);
                    }
                }
            }

            if a.alpha >= b.alpha
                || b.alpha - a.alpha >= old_width
                || *evaluations >= self.maxfev
            {
                return Ok(LineSearchOutcome::Failed);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn search<P, V>(
        &self,
        problem: &mut Problem<P>,
        param: &V,
        cost: F,
        gradient: &V,
        direction: &V,
        accept_boundary: bool,
    ) -> Result<LineSearchOutcome<F>, P::Error>
    where
        P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V>,
        V: ScaledAdd<F> + Dot<F> + Clone,
    {
        let zero = F::zero();
        let dphi0 = gradient.dot(direction);
        if !self.valid_config()
            || self.maxfev == 0
            || !cost.is_finite()
            || !dphi0.is_finite()
        {
            return Ok(LineSearchOutcome::Failed);
        }
        if dphi0 == zero {
            return Ok(LineSearchOutcome::Step(zero));
        }
        if dphi0 > zero {
            return Ok(LineSearchOutcome::Failed);
        }

        let epsilon_k = self.epsilon * cost.abs();
        if !epsilon_k.is_finite() || !(cost + epsilon_k).is_finite() {
            return Ok(LineSearchOutcome::Failed);
        }
        let mut evaluations = 0;
        let mut a = Trial {
            alpha: zero,
            phi: cost,
            dphi: dphi0,
        };
        let mut alpha = self.alpha_init.max(self.step_min).min(self.step_max);

        let (a, b) = loop {
            let Some(trial) = self.evaluate(
                problem,
                param,
                direction,
                alpha,
                &mut evaluations,
            )?
            else {
                return Ok(LineSearchOutcome::Failed);
            };
            if self.acceptable(trial, cost, dphi0, epsilon_k)
                || (accept_boundary
                    && alpha == self.step_max
                    && trial.is_finite()
                    && trial.phi <= cost + self.delta * alpha * dphi0
                    && trial.dphi < zero)
            {
                return Ok(LineSearchOutcome::Step(alpha));
            }

            if trial.is_finite() && trial.dphi >= zero {
                break (a, trial);
            }

            if !trial.is_finite() || trial.phi > cost + epsilon_k {
                match self.contract(
                    problem,
                    param,
                    direction,
                    a,
                    alpha,
                    cost,
                    dphi0,
                    epsilon_k,
                    &mut evaluations,
                )? {
                    SearchResult::Accepted(alpha) => {
                        return Ok(LineSearchOutcome::Step(alpha));
                    }
                    SearchResult::Bracket(left, right) => break (left, right),
                    SearchResult::Failed => {
                        return Ok(LineSearchOutcome::Failed);
                    }
                }
            }

            a = trial;
            let expanded = (alpha * self.rho).min(self.step_max);
            if !expanded.is_finite() || expanded <= alpha {
                return Ok(LineSearchOutcome::Failed);
            }
            alpha = expanded;
        };

        self.refine(
            problem,
            param,
            direction,
            a,
            b,
            cost,
            dphi0,
            epsilon_k,
            &mut evaluations,
        )
    }
}

impl<P, V, F> LineSearch<P, V, F> for HagerZhang<F>
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
        match self.search(problem, param, cost, gradient, direction, false)? {
            LineSearchOutcome::Step(alpha) => Ok(alpha),
            LineSearchOutcome::Failed => Ok(F::zero()),
        }
    }

    fn next_with_outcome(
        &mut self,
        problem: &mut Problem<P>,
        param: &V,
        cost: F,
        gradient: &V,
        direction: &V,
    ) -> Result<LineSearchOutcome<F>, Self::Error> {
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
        let search = Self {
            alpha_init: bounds.initial,
            step_max: self.step_max.min(bounds.max),
            ..*self
        };
        search
            .search(problem, param, cost, gradient, direction, true)
            .map(LineSearchResult::new)
    }
}

#[derive(Clone, Copy)]
struct Trial<F> {
    alpha: F,
    phi: F,
    dphi: F,
}

impl<F: Scalar> Trial<F> {
    fn is_finite(self) -> bool {
        self.phi.is_finite() && self.dphi.is_finite()
    }

    fn rejected(alpha: F) -> Self {
        Self {
            alpha,
            phi: F::infinity(),
            dphi: F::infinity(),
        }
    }
}

enum SearchResult<F> {
    Accepted(F),
    Bracket(Trial<F>, Trial<F>),
    Failed,
}
