//! Box-constrained, preconditioned truncated Newton.
// Adapted from sisyfuzz 0.1.0, Copyright 2026 Frans Henskens (Apache-2.0).
// Contributed to Basin under MIT OR Apache-2.0; see COPYRIGHT for provenance.
use std::marker::PhantomData;

use crate::core::constraint::evaluation::{
    BoxProblem, invalid, validate_coordinate_box_problem,
};
use crate::core::inner::{InitialState, WarmStart};
use crate::core::math::{Dot, Scalar, ScaledAdd, VectorIndex, VectorLen};
use crate::core::problem::{CostFunction, Gradient, HessianProduct, Problem};
use crate::core::state::{FirstOrderState, State};
use crate::line_search::{
    LineSearch, LineSearchBounds, LineSearchOutcome, Wolfe,
};
use crate::{ConstraintSolveError, Solver, SolverStep, Termination};

mod history;
use history::History;

/// Curvature obtained by feasible directional differences of gradients.
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GradientDifferences;

/// Curvature obtained from the problem's exact [`HessianProduct`] callback.
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExactProducts;

/// Inner conjugate-gradient termination, available through [`Tnc::cg_termination`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum TncCgTermination {
    /// No inner iteration has run.
    #[default]
    Initial,
    /// Relative residual tolerance was met.
    Residual,
    /// Inner iteration limit was reached.
    IterationLimit,
    /// Curvature was nonpositive or arithmetic was unusable.
    Breakdown,
    /// No representable directional stencil fits the box.
    UnavailableStencil,
}

/// Preconditioned truncated Newton for smooth objectives with finite box bounds.
///
/// Solves the Newton equations approximately on free coordinates with linear CG,
/// truncating on nonpositive curvature and falling back to projected steepest
/// descent when the direction is unusable. A limited-memory BFGS inverse model
/// preconditions CG (memory 10 by default); memory zero selects identity.
/// Changes to the free-coordinate set reset that model.
///
/// Adapted from the independent Rust implementation in
/// [`sisyfuzz` 0.1.0](https://docs.rs/crate/sisyfuzz/0.1.0/source/src/tnc.rs)
/// to Basin's [`Solver`] lifecycle and shared line searches.
/// Nash's survey below describes the algorithm family;
/// SciPy TNC provided practical context and a benchmark reference for the
/// original implementation. This port does not promise matching trajectories.
///
/// The default uses counted, fused gradient evaluations at feasible directional
/// probes and Basin's [`Wolfe`] search with bounded trial steps. The search aims
/// for the strong-Wolfe conditions but can return a step without satisfying
/// them. TNC checks accepted evaluations for feasibility, finite data and
/// nonincreasing cost; it does not guarantee strong curvature at every step. Use
/// [`exact_products`](Self::exact_products) to require exact [`HessianProduct`]
/// callbacks instead. Callback errors propagate as [`ConstraintSolveError`];
/// unavailable stencils, failed searches and non-finite iterates never imply
/// convergence. Accepted line-search evaluations are reused when supplied.
/// Execution limits retain Basin's between-step semantics.
/// Custom searches must implement [`LineSearch::next_with_bounds`]; its default
/// implementation fails without evaluating the objective.
///
/// The inner solve starts with zero direction and residual `r = -g_free`.
/// Preconditioned CG approximately solves `H_free d = -g_free`, stopping when
/// `||r||_2 <= eta * ||g_free||_2`, with `eta = 0.5` by default and at most
/// `min(50, dimension)` iterations. Gradient differences use a maximum component
/// displacement of `sqrt(F::epsilon())`, shortened or reversed to fit the box.
///
/// Fresh initialization validates the complete box-only declaration, projects
/// the seed, resets history and reevaluates it. Solver-and-state checkpoints
/// preserve curvature and diagnostics. Projected-gradient stationarity uses
/// the infinity norm of the gradient with outward bound components removed,
/// measured in the caller's coordinates (normalized when using
/// [`ParameterTransform`](crate::ParameterTransform)). The default tolerance
/// is `1e-6`; `None` disables the test, including exact-zero stopping.
///
/// # Backends
///
/// `Vec<F>`, nalgebra `DVector<F>`, ndarray `Array1<F>`, and faer `Col<F>`,
/// for all supported versions and both `f32` and `f64`. Requires [`VectorIndex`],
/// [`VectorLen`], [`Dot`], [`ScaledAdd`] and `Clone`; no matrices or BLAS.
///
/// # References
///
/// Stephen G. Nash, "A survey of truncated-Newton methods," *Journal of
/// Computational and Applied Mathematics* 124 (2000), 45-59.
/// <https://doi.org/10.1016/S0377-0427(00)00426-X>.
///
/// SciPy 1.17.0, [`minimize(method="TNC")`](https://docs.scipy.org/doc/scipy-1.17.0/reference/optimize.minimize-tnc.html),
/// for practical context and benchmark comparison of the original independent
/// implementation. This port is not a translation of SciPy's C implementation.
///
/// # Example
///
/// Minimize a shifted quadratic subject to a finite box. The unconstrained
/// minimum is outside the box, so the solution lies on its upper bound.
///
/// ```
/// use basin::{BoxBounds, Constrained, ConstraintSet, CostFunction, Executor,
///             Gradient, Tnc};
/// use std::convert::Infallible;
///
/// struct Quadratic;
/// impl CostFunction for Quadratic {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
///         Ok((x[0] - 2.0).powi(2))
///     }
/// }
/// impl Gradient for Quadratic {
///     type Gradient = Vec<f64>;
///     fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
///         Ok(vec![2.0 * (x[0] - 2.0)])
///     }
/// }
/// let problem = Constrained::new(Quadratic, ConstraintSet::new()
///     .with_bounds(BoxBounds::new(Some(vec![0.0]), Some(vec![1.0]))));
/// let result = Executor::from_start(problem, Tnc::new(), vec![0.5])
///     .max_iter(100).run().unwrap();
/// assert_eq!(result.param(), &vec![1.0]);
/// assert!(matches!(result.report.termination, basin::Termination::Converged(_)));
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Tnc<L, V, F: Scalar = f64, M = GradientDifferences> {
    line_search: L,
    tolerance: Option<F>,
    cg_tolerance: F,
    max_cg_iterations: usize,
    difference_step: F,
    memory: usize,
    history: History<F>,
    lower: Vec<F>,
    upper: Vec<F>,
    cg_iterations: usize,
    cg_termination: TncCgTermination,
    marker: PhantomData<(V, M)>,
}

impl<V, F: Scalar> Default for Tnc<Wolfe<F>, V, F> {
    fn default() -> Self {
        Self::new()
    }
}
impl<V, F: Scalar> Tnc<Wolfe<F>, V, F> {
    /// Construct with Basin's strong-Wolfe line search (60 inner iterations).
    pub fn new() -> Self {
        let mut search = Wolfe::new();
        search.max_iter = 60;
        Self::with_line_search(search)
    }
}
impl<L, V, F: Scalar> Tnc<L, V, F> {
    /// Construct with a line search that enforces [`LineSearchBounds`].
    pub fn with_line_search(line_search: L) -> Self {
        Self {
            line_search,
            tolerance: Some(F::from_f64(1e-6).unwrap()),
            cg_tolerance: F::from_f64(0.5).unwrap(),
            max_cg_iterations: 50,
            difference_step: F::epsilon().sqrt(),
            memory: 10,
            history: History::default(),
            lower: Vec::new(),
            upper: Vec::new(),
            cg_iterations: 0,
            cg_termination: TncCgTermination::Initial,
            marker: PhantomData,
        }
    }
    /// Require exact products; no numerical fallback is performed on failure.
    pub fn exact_products(self) -> Tnc<L, V, F, ExactProducts> {
        Tnc {
            line_search: self.line_search,
            tolerance: self.tolerance,
            cg_tolerance: self.cg_tolerance,
            max_cg_iterations: self.max_cg_iterations,
            difference_step: self.difference_step,
            memory: self.memory,
            history: self.history,
            lower: self.lower,
            upper: self.upper,
            cg_iterations: self.cg_iterations,
            cg_termination: self.cg_termination,
            marker: PhantomData,
        }
    }
}
impl<L, V, F: Scalar, M> Tnc<L, V, F, M> {
    /// Set nonnegative finite projected-gradient tolerance; `None` disables.
    ///
    /// # Panics
    ///
    /// Panics if an enabled tolerance is negative or non-finite.
    pub fn with_absolute_projected_gradient_tolerance(
        mut self,
        tolerance: impl Into<Option<F>>,
    ) -> Self {
        self.tolerance = tolerance.into();
        assert!(
            self.tolerance
                .is_none_or(|v| v.is_finite() && v >= F::zero())
        );
        self
    }
    /// Relative CG residual tolerance in `(0, 1)`.
    ///
    /// # Panics
    ///
    /// Panics if `tolerance` is non-finite or outside `(0, 1)`.
    pub fn with_relative_cg_residual_tolerance(mut self, tolerance: F) -> Self {
        assert!(
            tolerance.is_finite()
                && tolerance > F::zero()
                && tolerance < F::one()
        );
        self.cg_tolerance = tolerance;
        self
    }
    /// Positive inner iteration limit, also capped by dimension.
    ///
    /// # Panics
    ///
    /// Panics if `limit` is zero.
    pub fn with_max_cg_iterations(mut self, limit: usize) -> Self {
        assert!(limit > 0);
        self.max_cg_iterations = limit;
        self
    }
    /// Maximum component displacement for gradient differences.
    ///
    /// Defaults to `sqrt(F::epsilon())`. Feasible probes may use a smaller
    /// displacement near a bound.
    ///
    /// # Panics
    ///
    /// Panics if `step` is non-finite or outside `[sqrt(F::epsilon()), 0.1]`.
    pub fn with_difference_step(mut self, step: F) -> Self {
        assert!(
            step.is_finite()
                && step >= F::epsilon().sqrt()
                && step <= F::from_f64(0.1).unwrap()
        );
        self.difference_step = step;
        self
    }
    /// Preconditioner history capacity; zero selects identity.
    pub fn with_memory(mut self, memory: usize) -> Self {
        self.memory = memory;
        self.history = History::default();
        self
    }
    /// Inner iterations in the most recent CG solve, including failed steps.
    ///
    /// Zero before the first CG solve and after fresh initialization. A step
    /// that stops before CG starts leaves this diagnostic unchanged.
    pub fn cg_iterations(&self) -> usize {
        self.cg_iterations
    }
    /// Termination of the most recent CG solve, including failed steps.
    ///
    /// [`TncCgTermination::Initial`] before the first CG solve and after fresh
    /// initialization. A step that stops before CG starts leaves this unchanged.
    pub fn cg_termination(&self) -> TncCgTermination {
        self.cg_termination
    }
}

pub(crate) trait Products<P, V, F>
where
    P: CostFunction,
{
    #[allow(clippy::too_many_arguments)]
    fn product(
        problem: &mut Problem<P>,
        x: &V,
        g: &V,
        v: &V,
        out: &mut V,
        lower: &[F],
        upper: &[F],
        step: F,
    ) -> Result<bool, P::Error>;
}
impl<P, V, F> Products<P, V, F> for ExactProducts
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>
        + Gradient<Gradient = V>
        + HessianProduct,
{
    fn product(
        problem: &mut Problem<P>,
        x: &V,
        _: &V,
        v: &V,
        out: &mut V,
        _: &[F],
        _: &[F],
        _: F,
    ) -> Result<bool, P::Error> {
        problem.hessian_product_into(x, v, out)?;
        Ok(true)
    }
}
impl<P, V, F> Products<P, V, F> for GradientDifferences
where
    F: Scalar,
    V: Clone + VectorLen + VectorIndex<F>,
    P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V>,
{
    fn product(
        problem: &mut Problem<P>,
        x: &V,
        g: &V,
        v: &V,
        out: &mut V,
        lower: &[F],
        upper: &[F],
        step: F,
    ) -> Result<bool, P::Error> {
        let n = x.vec_len();
        let norm = (0..n).fold(F::zero(), |a, i| a.max(v.get_scalar(i).abs()));
        if norm == F::zero() {
            fill(out, F::zero());
            return Ok(true);
        }
        let requested = step / norm;
        let forward = ray(x, v, lower, upper, F::one());
        let backward = ray(x, v, lower, upper, -F::one());
        let half = F::from_f64(0.5).unwrap();
        let h = if forward >= requested {
            requested
        } else if backward > forward {
            -requested.min(backward * half)
        } else {
            requested.min(forward * half)
        };
        if !h.is_finite() || h == F::zero() {
            return Ok(false);
        }
        let mut point = x.clone();
        let mut changed = false;
        for i in 0..n {
            let value = (x.get_scalar(i) + h * v.get_scalar(i))
                .max(lower[i])
                .min(upper[i]);
            changed |= value != x.get_scalar(i);
            point.set_scalar(i, value);
        }
        if !changed {
            return Ok(false);
        }
        problem.cost_and_gradient_into(&point, out)?;
        if out.vec_len() != n {
            return Ok(false);
        }
        for i in 0..n {
            out.set_scalar(i, (out.get_scalar(i) - g.get_scalar(i)) / h);
        }
        Ok(true)
    }
}
fn fill<V: VectorLen + VectorIndex<F>, F: Scalar>(v: &mut V, value: F) {
    for i in 0..v.vec_len() {
        v.set_scalar(i, value);
    }
}
fn finite<V: VectorLen + VectorIndex<F>, F: Scalar>(v: &V) -> bool {
    (0..v.vec_len()).all(|i| v.get_scalar(i).is_finite())
}
fn blocked<F: Scalar>(x: F, d: F, lo: F, hi: F) -> bool {
    (x <= lo && d < F::zero()) || (x >= hi && d > F::zero()) || lo == hi
}
fn ray<V: VectorLen + VectorIndex<F>, F: Scalar>(
    x: &V,
    v: &V,
    lo: &[F],
    hi: &[F],
    sign: F,
) -> F {
    let mut limit = F::infinity();
    for i in 0..x.vec_len() {
        let d = sign * v.get_scalar(i);
        if d > F::zero() {
            limit = limit.min((hi[i] - x.get_scalar(i)) / d);
        }
        if d < F::zero() {
            limit = limit.min((lo[i] - x.get_scalar(i)) / d);
        }
    }
    limit.max(F::zero())
}
impl<L, V, F, M> InitialState<V> for Tnc<L, V, F, M>
where
    V: Clone,
    F: Scalar,
{
    type State = FirstOrderState<V, F>;
    fn seed(&self, x: &V) -> Self::State {
        FirstOrderState::new(x.clone())
    }
}
impl<L, V: Clone, F: Scalar, M> WarmStart<V> for Tnc<L, V, F, M> {}

impl<P, L, V, F, M> Solver<P, FirstOrderState<V, F>> for Tnc<L, V, F, M>
where
    F: Scalar,
    V: Clone + VectorLen + VectorIndex<F> + Dot<F> + ScaledAdd<F>,
    P: CostFunction<Param = V, Output = F>
        + Gradient<Gradient = V>
        + BoxProblem<Param = V>,
    L: LineSearch<P, V, F, Error = P::Error>,
    M: Products<P, V, F>,
{
    type Error = ConstraintSolveError<P::Error>;
    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: FirstOrderState<V, F>,
    ) -> Result<FirstOrderState<V, F>, Self::Error> {
        let n = state.param().vec_len();
        validate_coordinate_box_problem(problem.inner(), n, true)
            .map_err(invalid)?;
        self.lower = (0..n)
            .map(|i| problem.inner().lower().get_scalar(i))
            .collect();
        self.upper = (0..n)
            .map(|i| problem.inner().upper().get_scalar(i))
            .collect();
        state.reset();
        self.line_search.reset();
        self.history = History::default();
        self.cg_iterations = 0;
        self.cg_termination = TncCgTermination::Initial;
        for i in 0..n {
            let value = state.param().get_scalar(i);
            if value.is_finite() {
                state
                    .seed_param_mut()
                    .set_scalar(i, value.max(self.lower[i]).min(self.upper[i]));
            }
        }
        let (cost, g) = problem.cost_and_gradient(state.param())?;
        state
            .set_evaluation(cost, g)
            .map_err(|_| invalid(crate::ConstraintError::LayoutChanged))?;
        Ok(state)
    }
    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: FirstOrderState<V, F>,
    ) -> Result<SolverStep<FirstOrderState<V, F>>, Self::Error> {
        let (x, cost, g) =
            state.current().expect("TNC requires initialization");
        if !cost.is_finite() || !finite(x) || !finite(g) {
            return Ok(SolverStep::stopped(
                state,
                Termination::numerical_failure(
                    "TNC requires finite iterate data.",
                ),
            ));
        }
        let n = x.vec_len();
        let free: Vec<_> = (0..n)
            .map(|i| {
                !blocked(
                    x.get_scalar(i),
                    -g.get_scalar(i),
                    self.lower[i],
                    self.upper[i],
                )
            })
            .collect();
        let projected: Vec<_> = (0..n)
            .map(|i| if free[i] { g.get_scalar(i) } else { F::zero() })
            .collect();
        if let Some(termination) = projected_gradient_termination(
            x,
            g,
            &self.lower,
            &self.upper,
            self.tolerance,
        ) {
            return Ok(SolverStep::stopped(state, termination));
        }
        let mut direction = x.clone();
        fill(&mut direction, F::zero());
        let mut residual: Vec<_> = projected.iter().map(|&v| -v).collect();
        let mut z = self.history.apply(&residual);
        for i in 0..n {
            if !free[i] {
                z[i] = F::zero();
            }
        }
        let mut vector = x.clone();
        for (i, &v) in z.iter().enumerate() {
            vector.set_scalar(i, v);
        }
        let mut product = g.clone();
        let mut rz = dot(&residual, &z);
        let squared = dot(&projected, &projected);
        let target = self.cg_tolerance * self.cg_tolerance * squared;
        self.cg_iterations = 0;
        self.cg_termination = TncCgTermination::IterationLimit;
        for _ in 0..self.max_cg_iterations.min(n) {
            if !rz.is_finite() || rz <= F::zero() {
                self.cg_termination = TncCgTermination::Breakdown;
                break;
            }
            self.cg_iterations += 1;
            if !M::product(
                problem,
                x,
                g,
                &vector,
                &mut product,
                &self.lower,
                &self.upper,
                self.difference_step,
            )? {
                self.cg_termination = TncCgTermination::UnavailableStencil;
                break;
            }
            if product.vec_len() != n {
                return Ok(SolverStep::stopped(
                    state,
                    Termination::numerical_failure(
                        "TNC product shape changed.",
                    ),
                ));
            }
            for (i, &is_free) in free.iter().enumerate() {
                if !is_free {
                    product.set_scalar(i, F::zero());
                }
            }
            let curvature = vector.dot(&product);
            if !curvature.is_finite() || curvature <= F::zero() {
                self.cg_termination = TncCgTermination::Breakdown;
                break;
            }
            let alpha = rz / curvature;
            if !alpha.is_finite() {
                self.cg_termination = TncCgTermination::Breakdown;
                break;
            }
            direction.scaled_add(alpha, &vector);
            for (i, r) in residual.iter_mut().enumerate() {
                *r = *r - alpha * product.get_scalar(i);
            }
            if dot(&residual, &residual) <= target {
                self.cg_termination = TncCgTermination::Residual;
                break;
            }
            z = self.history.apply(&residual);
            for i in 0..n {
                if !free[i] {
                    z[i] = F::zero();
                }
            }
            let next_rz = dot(&residual, &z);
            let beta = next_rz / rz;
            for (i, &value) in z.iter().enumerate() {
                vector.set_scalar(i, value + beta * vector.get_scalar(i));
            }
            rz = next_rz;
        }
        for i in 0..n {
            if blocked(
                x.get_scalar(i),
                direction.get_scalar(i),
                self.lower[i],
                self.upper[i],
            ) {
                direction.set_scalar(i, F::zero());
            }
        }
        if !finite(&direction)
            || !g.dot(&direction).is_finite()
            || g.dot(&direction) >= F::zero()
        {
            for (i, &v) in projected.iter().enumerate() {
                direction.set_scalar(i, -v);
            }
        }
        let mathematical_maximum =
            ray(x, &direction, &self.lower, &self.upper, F::one());
        if !mathematical_maximum.is_finite()
            || mathematical_maximum <= F::zero()
            || g.dot(&direction) >= F::zero()
        {
            return Ok(SolverStep::stopped(
                state,
                Termination::numerical_failure(
                    "TNC has no feasible descent ray.",
                ),
            ));
        }
        // An exact feasible quotient can round to an infeasible reconstructed
        // point. Check the backend's actual arithmetic before any callback.
        let Some(maximum) = representable_limit(
            x,
            &direction,
            &self.lower,
            &self.upper,
            mathematical_maximum,
        ) else {
            return Ok(SolverStep::stopped(
                state,
                Termination::numerical_failure(
                    "TNC has no representable feasible line-search interval.",
                ),
            ));
        };
        let result = self.line_search.next_with_bounds(
            problem,
            x,
            cost,
            g,
            &direction,
            LineSearchBounds::new(F::one().min(maximum), maximum),
        )?;
        let LineSearchOutcome::Step(alpha) = result.outcome else {
            return Ok(SolverStep::stopped(
                state,
                Termination::numerical_failure("TNC line search failed."),
            ));
        };
        if !alpha.is_finite() || alpha <= F::zero() || alpha > maximum {
            return Ok(SolverStep::stopped(
                state,
                Termination::numerical_failure(
                    "TNC line search returned an invalid step.",
                ),
            ));
        }
        let mut next = x.clone();
        next.scaled_add(alpha, &direction);
        if alpha == maximum {
            // Adopt the mathematical endpoint, not a positive residual left by
            // multiply/add. If this differs from a retained probe, evaluate it
            // again so accepted cost/gradient always belong to the actual point.
            next = endpoint(
                x,
                &direction,
                &self.lower,
                &self.upper,
                mathematical_maximum,
            );
        }
        let retained = result.evaluation.filter(|e| {
            e.param.vec_len() == n
                && (0..n).all(|i| e.param.get_scalar(i) == next.get_scalar(i))
        });
        let (next_cost, next_g) = if let Some(e) = retained {
            (e.cost, e.gradient)
        } else {
            let (f, g) = problem.cost_and_gradient(&next)?;
            (f, g)
        };
        if next.vec_len() != n
            || next_g.vec_len() != n
            || !next_cost.is_finite()
            || !finite(&next)
            || !finite(&next_g)
            || (0..n).any(|i| {
                next.get_scalar(i) < self.lower[i]
                    || next.get_scalar(i) > self.upper[i]
            })
            || (0..n).all(|i| next.get_scalar(i) == x.get_scalar(i))
            || next_cost > cost
        {
            return Ok(SolverStep::stopped(
                state,
                Termination::numerical_failure(
                    "TNC received an unusable accepted evaluation.",
                ),
            ));
        }
        let changed = (0..n).any(|i| {
            free[i]
                == blocked(
                    next.get_scalar(i),
                    -next_g.get_scalar(i),
                    self.lower[i],
                    self.upper[i],
                )
        });
        if changed {
            self.history = History::default();
        } else {
            let s = (0..n)
                .map(|i| next.get_scalar(i) - x.get_scalar(i))
                .collect();
            let y = (0..n)
                .map(|i| next_g.get_scalar(i) - g.get_scalar(i))
                .collect();
            self.history.update(s, y, self.memory);
        }
        state
            .replace(next, next_cost, next_g)
            .expect("shape checked");
        Ok(SolverStep::completed(state))
    }

    fn terminate(
        &self,
        state: &FirstOrderState<V, F>,
    ) -> Option<Termination<F>> {
        let (x, cost, g) = state.current()?;
        if !cost.is_finite()
            || !finite(x)
            || !finite(g)
            || self.lower.len() != x.vec_len()
        {
            return None;
        }
        projected_gradient_termination(
            x,
            g,
            &self.lower,
            &self.upper,
            self.tolerance,
        )
    }
}

fn dot<F: Scalar>(a: &[F], b: &[F]) -> F {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

fn projected_gradient_termination<V, F>(
    x: &V,
    g: &V,
    lower: &[F],
    upper: &[F],
    tolerance: Option<F>,
) -> Option<Termination<F>>
where
    F: Scalar,
    V: VectorLen + VectorIndex<F>,
{
    let tolerance = tolerance?;
    let norm = (0..x.vec_len()).fold(F::zero(), |norm, i| {
        if blocked(x.get_scalar(i), -g.get_scalar(i), lower[i], upper[i]) {
            norm
        } else {
            norm.max(g.get_scalar(i).abs())
        }
    });
    (norm <= tolerance).then(|| {
        Termination::upper_bound(
            stationarity_test(),
            norm,
            tolerance,
            tolerance,
            None,
        )
    })
}

fn stationarity_test() -> crate::ConvergenceTest {
    crate::ConvergenceTest::Custom {
        key: "tnc.projected_free_gradient_infinity".into(),
        definition: "max |g_i| over coordinates not fixed or blocked by an outward negative gradient <= tolerance".into(),
    }
}

fn representable_limit<V, F>(
    x: &V,
    direction: &V,
    lower: &[F],
    upper: &[F],
    mut limit: F,
) -> Option<F>
where
    F: Scalar,
    V: Clone + VectorIndex<F> + VectorLen + ScaledAdd<F>,
{
    let contraction = F::one() - F::from_f64(4.0).unwrap() * F::epsilon();
    for _ in 0..32 {
        let mut trial = x.clone();
        trial.scaled_add(limit, direction);
        if (0..x.vec_len()).all(|i| {
            trial.get_scalar(i).is_finite()
                && trial.get_scalar(i) >= lower[i]
                && trial.get_scalar(i) <= upper[i]
        }) {
            return Some(limit);
        }
        let next = limit * contraction;
        if next == limit || next <= F::zero() {
            return None;
        }
        limit = next;
    }
    None
}

fn endpoint<V, F>(x: &V, direction: &V, lower: &[F], upper: &[F], step: F) -> V
where
    F: Scalar,
    V: Clone + VectorIndex<F> + VectorLen + ScaledAdd<F>,
{
    let mut out = x.clone();
    out.scaled_add(step, direction);
    for i in 0..x.vec_len() {
        let d = direction.get_scalar(i);
        let value = if d > F::zero() && step == (upper[i] - x.get_scalar(i)) / d
        {
            upper[i]
        } else if d < F::zero() && step == (lower[i] - x.get_scalar(i)) / d {
            lower[i]
        } else {
            out.get_scalar(i).max(lower[i]).min(upper[i])
        };
        out.set_scalar(i, value);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundoff_endpoints_are_feasible_and_exact() {
        for (x, d) in [
            (0.3, -1.1),
            (0.00030968749959447983, -1.1),
            (0.763774618976614, -0.033916744080123774),
        ] {
            let point = vec![x];
            let direction = vec![d];
            let limit = ray(&point, &direction, &[0.0], &[1.0], 1.0);
            let safe =
                representable_limit(&point, &direction, &[0.0], &[1.0], limit)
                    .unwrap();
            let mut trial = point.clone();
            trial.scaled_add(safe, &direction);
            assert!((0.0..=1.0).contains(&trial[0]));
            assert_eq!(
                endpoint(&point, &direction, &[0.0], &[1.0], limit),
                vec![0.0]
            );
        }
    }
}
