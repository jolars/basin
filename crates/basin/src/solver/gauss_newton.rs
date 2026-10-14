use super::least_squares_diagnostics::{
    LeastSquaresCheck, LeastSquaresDiagnostics, LeastSquaresObservation, Trace,
};
use crate::core::inner::InitialState;
use crate::core::least_squares::evaluation::{Evaluation, NllsStep};
use crate::core::math::{
    GramMatrix, LinearSolveSpd, MatTransposeVec, NegInPlace, NormInfinity,
    NormSquared, Scalar, ScaledAdd,
};
use crate::core::problem::{Jacobian, Problem, Residual};
use crate::core::solver::Solver;
use crate::core::state::{PointState, State};
use crate::core::termination::Termination;
use crate::{
    LossFunction, RobustLeastSquares, ScaleRowsInPlace, VectorIndex, VectorLen,
};

/// Pure Gauss-Newton solver for nonlinear least-squares problems
/// `min ½‖r(x)‖²`.
///
/// Wrap the raw problem in [`RobustLeastSquares`] to minimize a robust loss.
/// The adapter supplies the safeguarded Gauss–Newton model, and reported
/// costs and convergence use the robust objective and its gradient.
///
/// Each iteration solves the normal equations `(JᵀJ) δ = −Jᵀr` via
/// Cholesky on the Gram matrix `JᵀJ` and takes the full step
/// `x ← x + δ`. No damping, no line search; that's what
/// Levenberg-Marquardt is for. See Madsen, Nielsen, Tingleff (2004),
/// *Methods for Non-Linear Least Squares Problems*, §3.1.
///
/// **Cholesky-on-`JᵀJ` vs QR-on-`J`.** Cholesky on the Gram is the
/// simple path and the only one the
/// [`linalg`](crate::core::math) tier exposes today. It squares the
/// condition number of `J` and fails noisily on rank-deficient `J`;
/// see the [`solve_spd` failure path](#failure-modes) below. QR-on-`J`
/// is more numerically robust but adds a second factorization to the
/// linalg surface; deferred until a solver actually needs it. Pure GN
/// is the right vehicle for Cholesky: when `J` is so ill-conditioned
/// that QR matters in practice, you wanted LM (or TRF) anyway.
///
/// # Failure modes
///
/// - **Rank-deficient `J`** (`JᵀJ` not positive definite) → the
///   Cholesky inside [`LinearSolveSpd`] returns
///   [`NotPositiveDefinite`](crate::core::math::LinearSolveError::NotPositiveDefinite),
///   and the solver returns [`crate::TerminationCode::SolverFailed`]. This
///   is the *correct* behavior for pure GN; Powell's singular
///   function is the canonical example. Reach for Levenberg-Marquardt
///   when this fires.
/// - **Divergence on highly nonlinear or poorly initialized problems.**
///   No safeguard here either; pure GN trusts the linear model. Catch
///   this with a finite [`max_iter`](crate::Executor::max_iter)
///   on the executor and inspect the returned state and stopping reason.
///
/// # Convergence
///
/// The native first-order test is `‖Jᵀr‖_∞ ≤ tolerance`, configured by
/// [`with_absolute_gradient_tolerance`](Self::with_absolute_gradient_tolerance).
/// Its default is `1e-8`; `None` disables it, and zero tests exact stationarity.
/// It reports [`crate::TerminationCode::SolverConverged`] before computing a step.
/// Optional observed cost and step checks are disabled by default and combine
/// with this test using OR. Execution budgets belong on the executor.
///
/// The least-squares gradient is `Jᵀr`. It is computed inside the solver;
/// [`PointState`] does not expose a [`GradientState`](crate::GradientState).
///
/// # Backends
///
/// LA-heavy: the default `Vec<f64>` backend (over the hand-rolled
/// [`DenseMatrix<f64>`](crate::DenseMatrix), via a pure-Rust Cholesky),
/// nalgebra (`DVector<f64>`/`DMatrix<f64>`), faer (`Col<f64>` /
/// `Mat<f64>`), and ndarray (`Array1<f64>`/`Array2<f64>`, the latter over
/// the same pure-Rust Cholesky), plus the nalgebra-sparse / faer-sparse
/// matrices.
/// All support `f32` and `f64`; use `GaussNewton::default()` for `f32`.
/// Robust objectives additionally require [`ScaleRowsInPlace`] on custom
/// Jacobian types.
///
/// # State convention
///
/// For an ordinary residual problem, `state.cost` carries the LM convention
/// `½‖r‖²`, derived from the
/// residual the solver computes itself. The bound on `P` is
/// [`Residual`] + [`Jacobian`], not [`CostFunction`](crate::core::problem::CostFunction);
/// problems whose user-facing `cost()` uses an unscaled `Σ rᵢ²` form
/// (e.g. Rosenbrock-as-residuals) will see `state.cost()` differ from
/// `problem.cost(state.param())` by a factor of two. Both go to zero
/// at the optimum, so cost-based termination criteria are unaffected.
///
/// # Examples
///
/// Identical setup to [`LevenbergMarquardt`](crate::LevenbergMarquardt):
/// implement `Residual` + `Jacobian`, then drive a `PointState` through
/// the `Executor`, swapping `LevenbergMarquardt::new()` for
/// `GaussNewton::new()`.
pub struct GaussNewton<V, M, F: Scalar = f64> {
    tol_grad: Option<F>,

    // Residual and Jacobian caches across iterations. `r_cache` is set
    // to `r(x_new)` after the full GN step and reused at the top of the
    // next iter. `j_cache` is set only by `init` (init's `J(x₀)` is
    // reused for iter 0); after a step `J` is at the old iterate and
    // is dropped.
    r_cache: Option<V>,
    j_cache: Option<M>,
    model_r_cache: Option<V>,
    failed: bool,
    trace: Option<Trace<F>>,
    trace_vector: Option<fn(&V) -> Vec<F>>,
}

impl<V, M, F: Scalar> LeastSquaresDiagnostics<F> for GaussNewton<V, M, F> {
    fn least_squares_observations(&self) -> &[LeastSquaresObservation<F>] {
        self.trace.as_ref().map_or(&[], |trace| &trace.observations)
    }
}

impl<V, M, F: Scalar> Default for GaussNewton<V, M, F> {
    fn default() -> Self {
        Self {
            tol_grad: Some(F::from_f64(1e-8).unwrap()),
            r_cache: None,
            j_cache: None,
            model_r_cache: None,
            failed: false,
            trace: None,
            trace_vector: None,
        }
    }
}

impl<V, M> GaussNewton<V, M> {
    /// Pure Gauss-Newton with the default first-order optimality
    /// tolerance (`tol_grad = 1e-8`).
    pub fn new() -> Self {
        Self::default()
    }
}

impl<V, M, F: Scalar> GaussNewton<V, M, F> {
    /// Enable native model, Cholesky, and full-step observations.
    ///
    /// Recording is disabled by default and performs no problem callbacks.
    /// Fresh initialization resets the sequence; exact continuation preserves it.
    /// A full step has no decrease-based acceptance test. Its publication and
    /// independent quality must be checked separately from its dispatch.
    pub fn with_trial_diagnostics(mut self, enabled: bool) -> Self
    where
        V: VectorLen + VectorIndex<F>,
    {
        self.trace = enabled.then(Trace::default);
        self.trace_vector = enabled.then_some(|v: &V| {
            (0..v.vec_len()).map(|i| v.get_scalar(i)).collect()
        });
        self
    }

    /// Configure the infinity norm of J-transpose times residual.
    ///
    /// `None` disables the test; zero requests an exact-zero threshold.
    /// Values must be finite and nonnegative. Enabled tests combine with OR;
    /// each model-based test retains its internal conjunction and observation stage.
    /// Repeated calls replace this setting. Existing solver defaults are retained.
    pub fn with_absolute_gradient_tolerance(
        mut self,
        value: impl Into<Option<F>>,
    ) -> Self {
        self.tol_grad = crate::core::convergence::optional_tolerance(value);
        self
    }
}

impl<V, M, F> InitialState<V> for GaussNewton<V, M, F>
where
    F: Scalar,
    V: Clone,
{
    type State = PointState<V, F>;
    fn seed(&self, x: &V) -> Self::State {
        PointState::new(x.clone())
    }
}

impl<P, V, M, F> Solver<P, PointState<V, F>> for GaussNewton<V, M, F>
where
    F: Scalar,
    P: Residual<Param = V, Output = V> + Jacobian<Jacobian = M>,
    V: ScaledAdd<F> + NormSquared<F> + NormInfinity<F> + NegInPlace + Clone,
    M: GramMatrix + MatTransposeVec<V> + LinearSolveSpd<V>,
{
    type Error = <P as Residual>::Error;
    fn init(
        &mut self,
        problem: &mut Problem<P>,
        state: PointState<V, F>,
    ) -> Result<PointState<V, F>, Self::Error> {
        self.init_evaluated(problem, state)
    }
    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        state: PointState<V, F>,
    ) -> Result<crate::SolverStep<PointState<V, F>>, Self::Error> {
        self.next_evaluated(problem, state)
    }
}

impl<P, L, V, M, F> Solver<RobustLeastSquares<P, L, F>, PointState<V, F>>
    for GaussNewton<V, M, F>
where
    F: Scalar,
    P: Residual<Param = V, Output = V> + Jacobian<Jacobian = M>,
    V: ScaledAdd<F> + NormSquared<F> + NormInfinity<F> + NegInPlace + Clone,
    M: GramMatrix + MatTransposeVec<V> + LinearSolveSpd<V>,
    L: LossFunction<F>,
    V: VectorLen + VectorIndex<F>,
    M: ScaleRowsInPlace<F>,
{
    type Error = <P as Residual>::Error;
    fn init(
        &mut self,
        problem: &mut Problem<RobustLeastSquares<P, L, F>>,
        state: PointState<V, F>,
    ) -> Result<PointState<V, F>, Self::Error> {
        self.init_evaluated(problem, state)
    }
    fn next_iter(
        &mut self,
        problem: &mut Problem<RobustLeastSquares<P, L, F>>,
        state: PointState<V, F>,
    ) -> Result<crate::SolverStep<PointState<V, F>>, Self::Error> {
        self.next_evaluated(problem, state)
    }
}

impl<V, M, F: Scalar> GaussNewton<V, M, F>
where
    V: ScaledAdd<F> + NormSquared<F> + NormInfinity<F> + NegInPlace + Clone,
    M: GramMatrix + MatTransposeVec<V> + LinearSolveSpd<V>,
{
    fn init_evaluated<E: Evaluation<V, M, F>>(
        &mut self,
        problem: &mut E,
        mut state: PointState<V, F>,
    ) -> Result<PointState<V, F>, E::Error> {
        state.reset();
        if let Some(trace) = &mut self.trace {
            trace.reset();
        }
        // Seed cost so iter-0 termination criteria see a populated
        // state. Both `r(x₀)` and `J(x₀)` are stashed so the first
        // `next_iter` doesn't re-evaluate them at the same point.
        self.failed = false;
        self.r_cache = None;
        self.j_cache = None;
        self.model_r_cache = None;
        let (r, j) = problem.residual_and_jacobian(state.param())?;
        state.replace(
            state.param().clone(),
            problem.cost(&r, |r| F::from_f64(0.5).unwrap() * r.norm_squared()),
        );
        let Some((model_r, j)) = problem.model(&r, j) else {
            self.failed = true;
            if let Some(trace) = &mut self.trace {
                trace.event(state.cost(), "construction", "invalid_model");
            }
            return Ok(state);
        };
        self.r_cache = Some(r);
        self.j_cache = Some(j);
        self.model_r_cache = model_r;
        Ok(state)
    }

    fn next_evaluated<E: Evaluation<V, M, F>>(
        &mut self,
        problem: &mut E,
        mut state: PointState<V, F>,
    ) -> NllsStep<V, F, E::Error> {
        if let Some(trace) = &mut self.trace {
            trace.clear();
        }
        if self.failed {
            if let Some(trace) = &mut self.trace {
                trace.event(state.cost(), "construction", "invalid_model");
            }
            return Ok(crate::SolverStep::from((
                state,
                Some(Termination::numerical_failure(
                    "Gauss-Newton could not initialize a finite residual model.",
                )),
            )));
        }
        let r = match self.r_cache.take() {
            Some(r) => r,
            None => problem.residual(state.param())?,
        };
        let (model_r, j) = match self.j_cache.take() {
            Some(j) => (self.model_r_cache.take(), j),
            None => {
                let j = problem.jacobian(state.param())?;
                let Some(model) = problem.model(&r, j) else {
                    self.failed = true;
                    if let Some(trace) = &mut self.trace {
                        trace.event(
                            state.cost(),
                            "construction",
                            "invalid_model",
                        );
                    }
                    return Ok(crate::SolverStep::from((
                        state,
                        Some(Termination::numerical_failure(
                            "Gauss-Newton could not construct the residual model at the current point.",
                        )),
                    )));
                };
                model
            }
        };

        // The corrected model preserves the objective gradient, so the same
        // first-order optimality test applies to ordinary and robust losses.
        let g = j.mat_transpose_vec(model_r.as_ref().unwrap_or(&r));
        if !problem.valid_vector(&g) {
            self.failed = true;
            if let Some(trace) = &mut self.trace {
                let index = trace.event(
                    state.cost(),
                    "construction",
                    "invalid_gradient",
                );
                trace.observations[index].gradient =
                    self.trace_vector.unwrap()(&g);
            }
            return Ok(crate::SolverStep::from((
                state,
                Some(Termination::numerical_failure(
                    "The Gauss-Newton model gradient is invalid.",
                )),
            )));
        }
        let gradient_norm = g.norm_infinity();
        if let Some(trace) = &mut self.trace {
            let mut row = LeastSquaresObservation::model(state.cost());
            row.gradient = self.trace_vector.unwrap()(&g);
            row.checks.push(LeastSquaresCheck {
                name: "absolute_gradient",
                tolerance: self.tol_grad,
                passed: self.tol_grad.map(|tol| gradient_norm <= tol),
                evidence: self.tol_grad.map(|tolerance| {
                    crate::ConvergenceEvidence::UpperBound {
                        value: gradient_norm,
                        bound: tolerance,
                        tolerance,
                        reference: None,
                    }
                }),
            });
            trace.push(row);
        }
        if self.tol_grad.is_some_and(|tol| gradient_norm <= tol) {
            self.r_cache = Some(r);
            self.j_cache = Some(j);
            self.model_r_cache = model_r;
            return Ok(crate::SolverStep::from((
                state,
                Some(Termination::upper_bound(
                    crate::ConvergenceTest::AbsoluteGradientInfinity,
                    gradient_norm,
                    self.tol_grad.unwrap(),
                    self.tol_grad.unwrap(),
                    None,
                )),
            )));
        }

        // Solve (JᵀJ) δ = −Jᵀr. Cholesky failure means JᵀJ is not
        // positive definite (rank-deficient J); pure GN can't recover,
        // LM in S4 will.
        let gram = j.gram();
        let mut neg_g = g;
        neg_g.neg_in_place();
        let solved = gram.solve_spd(&neg_g);
        if let Some(trace) = &mut self.trace {
            let mut row = LeastSquaresObservation::model(state.cost());
            row.event = "subproblem";
            row.outcome = Some(if solved.is_ok() {
                "solved"
            } else {
                "factorization_failed"
            });
            row.model_solves = 1;
            row.gradient = self.trace_vector.unwrap()(&neg_g)
                .into_iter()
                .map(|v| -v)
                .collect();
            if let Ok(delta) = &solved {
                row.step = self.trace_vector.unwrap()(delta);
                row.step_norm = Some(delta.norm_squared().sqrt());
            }
            trace.push(row);
        }
        let delta = match solved {
            Ok(d) => d,
            Err(_) => {
                // State unchanged on Cholesky failure; restore caches
                // for any subsequent reuse (e.g. via `InnerExecutor`).
                self.r_cache = Some(r);
                self.j_cache = Some(j);
                self.model_r_cache = model_r;
                return Ok(crate::SolverStep::from((
                    state,
                    Some(Termination::numerical_failure(
                        "The Gauss-Newton normal-equation Cholesky solve failed.",
                    )),
                )));
            }
        };

        // Full GN step. Refresh state.cost from a fresh residual so the
        // post-iteration state is consistent (Solver contract); stash
        // that residual so the next iter reuses it without re-eval.
        // `J(x_new)` is not computed, so `j_cache` stays empty.
        let mut param = state.param().clone();
        param.scaled_add(F::one(), &delta);
        let trial = self.trace.as_mut().map(|trace| {
            let mut row = LeastSquaresObservation::model(state.cost());
            row.event = "trial";
            row.outcome = Some("callback_pending");
            row.trial = true;
            row.step = self.trace_vector.unwrap()(&delta);
            row.gradient = self.trace_vector.unwrap()(&neg_g)
                .into_iter()
                .map(|v| -v)
                .collect();
            row.step_norm = Some(delta.norm_squared().sqrt());
            row.predicted_reduction = Some(
                -row.gradient
                    .iter()
                    .zip(&row.step)
                    .fold(F::zero(), |sum, (g, d)| sum + *g * *d)
                    / F::from_f64(2.).unwrap(),
            );
            trace.push(row)
        });
        let r_new = problem.residual(&param).inspect_err(|_| {
            if let Some(index) = trial {
                self.trace.as_mut().unwrap().observations[index].outcome =
                    Some("callback_failed");
            }
        })?;
        let trial_cost = problem
            .cost(&r_new, |r| F::from_f64(0.5).unwrap() * r.norm_squared());
        if let Some(index) = trial {
            let row = &mut self.trace.as_mut().unwrap().observations[index];
            row.outcome = Some("unconditional_full_step");
            row.trial_cost = Some(trial_cost);
            row.actual_reduction = Some(row.base_cost - trial_cost);
            row.accepted = Some(true);
        }
        state.replace(param, trial_cost);
        self.r_cache = Some(r_new);
        self.j_cache = None;

        let failed = E::ROBUST && !state.cost().is_finite();
        self.failed = failed;
        Ok(crate::SolverStep::from((
            state,
            failed.then(|| Termination::numerical_failure("Gauss-Newton trial produced a non-finite robust objective.")),
        )))
    }
}
