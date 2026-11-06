//! Armijo backtracking along a box-projected path.
use crate::line_search::LineSearchEvaluation;
use crate::solver::lbfgs::AsFloatSliceMut;
use crate::{CostFunction, Gradient, Problem, Scalar};
type AcceptedTrial<V, F> = (F, LineSearchEvaluation<V, F>);

/// Backtracking on `project(x + alpha * direction)` for a box-constrained solver.
///
/// Used by [`ProjectedSearch::Armijo`](crate::solver::lbfgs::model_strategy::ProjectedSearch::Armijo).
/// Sufficient decrease uses the actual displacement:
/// `f(trial) <= f(x) + c1 * g(x)'(trial-x)`, with strict cost decrease and a
/// negative finite predicted reduction. Starts at one; by default contracts by
/// one half, with `c1 = 1e-4` and at most 40 trials.
/// Duplicate projected points are not evaluated twice; an unchanged point,
/// exhausted search or unusable arithmetic never implies convergence.
///
/// Ordinary mode requests only costs on rejected trials and the gradient on
/// acceptance. [`Self::with_fused_evaluations`] requests both at each trial and
/// returns the accepted evaluation without another callback. All requests use
/// [`Problem`] accounting and propagate the original error.
/// Override [`Gradient::cost_and_gradient`] to share intermediate work and
/// [`Gradient::cost_and_gradient_into`] to reuse gradient storage. The default
/// combined callback calls the separate cost and gradient methods; the default
/// buffer-writing callback delegates to the combined callback. Fused mode also
/// requests gradients at rejected trials, which can cost more when costs are cheap
/// and gradients are expensive.
///
/// This strategy deliberately does not implement the straight-ray
/// [`LineSearch`](crate::LineSearch) contract. Its accepted point need not equal
/// `x + alpha * direction`.
///
/// # Backends
///
/// Contiguous `Vec<F>`, nalgebra `DVector<F>`, ndarray `Array1<F>` and faer
/// `Col<F>`, for `f32` and `f64`, through the projected model wrapper.
///
/// # References
///
/// Adapted from sisyfuzz 0.1.2 `src/line_search.rs` projected Armijo search
/// (Apache-2.0); see the crate COPYRIGHT notice. For the Armijo condition, see
/// the references on [`Backtracking`](crate::Backtracking).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ProjectedArmijo<F: Scalar = f64> {
    sufficient_decrease: F,
    contraction: F,
    max_steps: usize,
    fused: bool,
}
impl<F: Scalar> Default for ProjectedArmijo<F> {
    fn default() -> Self {
        Self::new()
    }
}
impl<F: Scalar> ProjectedArmijo<F> {
    /// Cost-first Armijo search, with c1=1e-4, contraction=0.5 and 40 trials.
    pub fn new() -> Self {
        Self {
            sufficient_decrease: F::from_f64(1e-4).unwrap(),
            contraction: F::from_f64(0.5).unwrap(),
            max_steps: 40,
            fused: false,
        }
    }
    /// Set c1 in (0,1). Panics for an invalid coefficient.
    pub fn with_sufficient_decrease_coefficient(mut self, value: F) -> Self {
        assert!(value.is_finite() && value > F::zero() && value < F::one());
        self.sufficient_decrease = value;
        self
    }
    /// Set the contraction multiplier in (0,1). Panics for invalid values.
    pub fn with_contraction_factor(mut self, value: F) -> Self {
        assert!(value.is_finite() && value > F::zero() && value < F::one());
        self.contraction = value;
        self
    }
    /// Set a positive trial limit. Panics for zero.
    pub fn max_iter(mut self, value: usize) -> Self {
        assert!(value > 0);
        self.max_steps = value;
        self
    }
    /// Request combined cost/gradient evaluations, including rejected trials.
    pub fn with_fused_evaluations(mut self) -> Self {
        self.fused = true;
        self
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn search<P, V>(
        &mut self,
        problem: &mut Problem<P>,
        x: &V,
        cost: F,
        g: &V,
        direction: &V,
        lower: &[F],
        upper: &[F],
        buffer: &mut Option<V>,
    ) -> Result<Option<AcceptedTrial<V, F>>, P::Error>
    where
        P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V>,
        V: AsFloatSliceMut<F> + Clone,
    {
        let mut alpha = F::one();
        let mut trial = x.clone();
        let mut previous: Option<V> = None;
        for _ in 0..self.max_steps {
            let mut slope = F::zero();
            let mut finite = true;
            for i in 0..x.as_float_slice().len() {
                let raw = x.as_float_slice()[i]
                    + alpha * direction.as_float_slice()[i];
                let value = raw.max(lower[i]).min(upper[i]);
                finite &= !raw.is_nan() && value.is_finite();
                trial.as_float_slice_mut()[i] = value;
                slope = slope
                    + g.as_float_slice()[i] * (value - x.as_float_slice()[i]);
            }
            if trial.as_float_slice() == x.as_float_slice() {
                return Ok(None);
            }
            let duplicate = previous
                .as_ref()
                .is_some_and(|p| p.as_float_slice() == trial.as_float_slice());
            if finite && !duplicate && slope.is_finite() && slope < F::zero() {
                let (value, gradient) = if self.fused {
                    let (value, gradient) = problem
                        .cost_and_gradient_reusing(&trial, buffer.take())?;
                    (value, Some(gradient))
                } else {
                    (problem.cost(&trial)?, None)
                };
                if value.is_finite()
                    && value < cost
                    && value <= cost + self.sufficient_decrease * slope
                {
                    let gradient = if let Some(gradient) = gradient {
                        gradient
                    } else {
                        problem.gradient_reusing(&trial, buffer.take())?
                    };
                    if gradient.as_float_slice().len()
                        == x.as_float_slice().len()
                        && gradient
                            .as_float_slice()
                            .iter()
                            .all(|v| v.is_finite())
                    {
                        return Ok(Some((
                            alpha,
                            LineSearchEvaluation::new(trial, value, gradient),
                        )));
                    }
                    *buffer = Some(gradient);
                } else if let Some(gradient) = gradient {
                    *buffer = Some(gradient);
                }
                match previous.as_mut() {
                    Some(p) => p.clone_from(&trial),
                    None => previous = Some(trial.clone()),
                }
            }
            alpha = alpha * self.contraction;
            if !alpha.is_finite() || alpha <= F::zero() {
                break;
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    struct Objective {
        points: RefCell<Vec<Vec<f64>>>,
        constant: bool,
        fail: bool,
    }
    impl CostFunction for Objective {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = &'static str;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
            assert!(x.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));
            self.points.borrow_mut().push(x.clone());
            if self.fail {
                return Err("original error");
            }
            Ok(if self.constant {
                0.0
            } else {
                -x[0] + 20.0 * (x[1] - 0.5).powi(2)
            })
        }
    }
    impl Gradient for Objective {
        type Gradient = Vec<f64>;
        fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
            Ok(vec![-1.0, 40.0 * (x[1] - 0.5)])
        }
    }
    fn objective(constant: bool, fail: bool) -> Problem<Objective> {
        Problem::new(Objective {
            points: RefCell::new(Vec::new()),
            constant,
            fail,
        })
    }
    #[test]
    fn projects_each_trial_and_reuses_the_accepted_evaluation() {
        for fused in [false, true] {
            let mut p = objective(false, false);
            let mut search = ProjectedArmijo::new();
            if fused {
                search = search.with_fused_evaluations();
            }
            let (alpha, accepted) = search
                .search(
                    &mut p,
                    &vec![0.5, 0.5],
                    -0.5,
                    &vec![-1.0, 0.0],
                    &vec![2.0, 0.2],
                    &[0.0; 2],
                    &[1.0; 2],
                    &mut None,
                )
                .unwrap()
                .unwrap();
            assert_eq!(alpha, 0.5);
            assert_eq!(accepted.param, vec![1.0, 0.6]);
            assert!((accepted.cost + 0.8).abs() < 1e-14);
            assert_eq!(
                *p.inner().points.borrow(),
                vec![vec![1.0, 0.7], vec![1.0, 0.6]]
            );
            assert_eq!(p.counts().cost_evals, 2);
            assert_eq!(p.counts().gradient_evals, if fused { 2 } else { 1 });
        }
    }
    #[test]
    fn duplicate_and_unchanged_points_are_not_evaluated_and_errors_propagate() {
        let mut p = objective(true, false);
        let result = ProjectedArmijo::new()
            .search(
                &mut p,
                &vec![0.5, 0.5],
                0.0,
                &vec![-1.0, 0.0],
                &vec![1e16, 0.0],
                &[0.0; 2],
                &[1.0; 2],
                &mut None,
            )
            .unwrap();
        assert!(result.is_none());
        assert_eq!(p.counts().cost_evals, 1);
        let result = ProjectedArmijo::new()
            .search(
                &mut p,
                &vec![0.5, 0.5],
                0.0,
                &vec![-1.0, 0.0],
                &vec![0.0; 2],
                &[0.0; 2],
                &[1.0; 2],
                &mut None,
            )
            .unwrap();
        assert!(result.is_none());
        assert_eq!(p.counts().cost_evals, 1);
        let mut p = objective(false, true);
        let result = ProjectedArmijo::new().with_fused_evaluations().search(
            &mut p,
            &vec![0.5, 0.5],
            -0.5,
            &vec![-1.0, 0.0],
            &vec![2.0, 0.2],
            &[0.0; 2],
            &[1.0; 2],
            &mut None,
        );
        assert_eq!(result.unwrap_err(), "original error");
        assert_eq!(p.counts().cost_evals, 1);
        assert_eq!(p.counts().gradient_evals, 1);
    }
    #[test]
    fn nonfinite_costs_and_gradients_are_never_selected() {
        struct Invalid(bool);
        impl CostFunction for Invalid {
            type Param = Vec<f64>;
            type Output = f64;
            type Error = std::convert::Infallible;
            fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
                assert!(x[0].is_finite() && x[0] >= 0.0 && x[0] <= 1.0);
                Ok(if self.0 { f64::NAN } else { -x[0] })
            }
        }
        impl Gradient for Invalid {
            type Gradient = Vec<f64>;
            fn gradient(&self, _: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
                Ok(vec![f64::NAN])
            }
        }
        for invalid_cost in [true, false] {
            for fused in [true, false] {
                let mut search = ProjectedArmijo::new().max_iter(4);
                if fused {
                    search = search.with_fused_evaluations();
                }
                let mut problem = Problem::new(Invalid(invalid_cost));
                assert!(
                    search
                        .search(
                            &mut problem,
                            &vec![0.5],
                            -0.5,
                            &vec![-1.0],
                            &vec![1.0],
                            &[0.0],
                            &[1.0],
                            &mut None
                        )
                        .unwrap()
                        .is_none()
                );
            }
        }
    }
}
