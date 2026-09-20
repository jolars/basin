//! Dense trust-region-reflective nonlinear least squares.

mod step;

use crate::core::constraint::BoxConstraints;
use crate::core::inner::InitialState;
use crate::core::least_squares::evaluation::{BoundedEvaluation, NllsStep};
use crate::core::math::dense_svd::{DenseSvd, norm};
use crate::core::math::{MatrixIndex, Scalar, VectorIndex, VectorLen};
use crate::core::problem::{Jacobian, Problem, Residual};
use crate::core::solver::Solver;
use crate::core::state::NllsState;
use crate::core::termination::TerminationReason;
use crate::{LossFunction, RobustLeastSquares, ScaleRowsInPlace};
use step::{Model, interior, number, update_radius};

/// Trust-region-reflective minimization of `½‖r(x)‖²` with box bounds.
///
/// This dense variant uses Coleman–Li scaling, an explicit trust-region
/// radius, a rank-aware SVD subproblem solve, and selection among a truncated
/// trust-region step, a first-boundary reflection, and a scaled-gradient step.
/// [`Trf`](crate::Trf) retains the earlier bounded-LM algorithm, including its
/// sparse and downstream-backend support.
///
/// Wrap the raw problem in [`RobustLeastSquares`] to minimize a robust loss.
/// The adapter supplies the safeguarded Gauss–Newton model. Reported costs,
/// actual reductions, and Coleman–Li scaling use the robust objective and
/// its gradient.
///
/// # Algorithm
///
/// For `g = Jᵀr`, let `v[i]` be the distance to the upper bound when `g[i] < 0`,
/// to the lower bound when `g[i] > 0`, and one when that side is unbounded or
/// the gradient is zero. With `d = sqrt(v)` and `c = g ⊙ dv`, solve
/// `min_h g_hᵀh + ½‖J diag(d) h‖² + ½∑ c[i] h[i]²`, `‖h‖ ≤ radius`,
/// where `g_h = d ⊙ g` and the original step is `d ⊙ h`.
///
/// A pure-Rust one-sided Jacobi SVD factors `[J diag(d); diag(sqrt(c))]`
/// directly. Singular directions below the relative rank cutoff are omitted.
/// The minimum-norm Gauss–Newton step is used when it fits; otherwise a
/// safeguarded secular solve places the step on the trust-region boundary
/// (relative radius accuracy `0.01`). The factorization is reused on rejection.
/// Storage is `O(mn + n²)`; this is intended for small and medium dense problems.
/// No sparse, LSMR, or large-scale subspace path is provided.
///
/// Candidate selection and radius updates follow SciPy's dense TRF variant.
/// Finite objective decreases are accepted. The ratio uses ordinary actual
/// objective reduction and the Coleman–Li model prediction, unlike the
/// curvature-corrected actual reduction in the original paper and Basin's
/// legacy `Trf`. Ratios below `0.25` shrink the radius to one quarter of the
/// scaled step norm; ratios above `0.75` double it when the step exceeds `0.95`
/// of the old radius. The interior factor is `max(0.995, 1 - ‖v ⊙ g‖∞)`.
///
/// # Bounds and failure behavior
///
/// Supply [`Residual`], [`Jacobian`], and [`BoxConstraints`]. The solver never
/// calls the raw problem's `CostFunction::cost`; state cost is `½‖r‖²` for an
/// ordinary residual problem, or the chosen robust objective. Use
/// [`crate::BoundedFiniteDiff`] when numerical Jacobians must respect bounds.
/// Residual dimensions and bounds must remain fixed during a solve.
///
/// Finite initial parameters are projected strictly inside each free interval.
/// Finite equal bounds fix a coordinate, which is eliminated internally.
/// All-fixed problems evaluate the residual once and finish without a Jacobian.
/// Signed infinite bounds are supported. Invalid shapes, reversed or NaN bounds,
/// non-finite initial parameters, and invalid settings panic.
///
/// Non-finite trial residuals cause rejection and radius contraction. Invalid
/// initial evaluations or derivatives, failed factorizations (100 Jacobi sweeps),
/// exhausted inner attempts, or numerical inability to make progress report
/// [`TerminationReason::SolverFailed`]. A free interval without a representable
/// interior fails before any callback. User errors propagate unchanged.
///
/// # Convergence and lifecycle
///
/// The native test is `max |v ⊙ Jᵀr| ≤ tolerance` over free coordinates,
/// evaluated at the initial point and after accepted steps. The default is
/// `1e-8`; `None` disables it and zero requests exact stationarity. Optional
/// observed cost and step tests are disabled by default and combine with OR.
/// Execution budgets belong on [`crate::Executor`]. All-fixed successful
/// termination is structural and does not depend on a tolerance.
///
/// [`NllsState`] publishes matching parameters and cost. Residuals, derivatives,
/// and the radius are solver-owned. Fresh runs reset them; exact solver/state
/// continuation retains them. Rejected trials reuse the current Jacobian, and
/// every accepted point has a freshly evaluated Jacobian before publication.
///
/// # Backends
///
/// `Vec<F>`/`DenseMatrix<F>`, nalgebra `DVector<F>`/`DMatrix<F>`, ndarray
/// `Array1<F>`/`Array2<F>`, and faer `Col<F>`/`Mat<F>`, for `f32` and `f64`.
/// All share the pure-Rust dense kernel through [`MatrixIndex`]; BLAS/LAPACK
/// and optional backend features are unnecessary for the default `Vec` path.
/// Sparse Jacobians are not supported. Custom dense types can implement
/// `MatrixIndex`, `VectorIndex`, and `VectorLen`.
/// Robust objectives additionally require [`ScaleRowsInPlace`] on custom
/// Jacobian types.
///
/// # References
///
/// - M. A. Branch, T. F. Coleman, and Y. Li (1999), "A Subspace, Interior, and
///   Conjugate Gradient Method for Large-Scale Bound-Constrained Minimization
///   Problems", *SIAM Journal on Scientific Computing* 21(1), 1–23.
///   <https://doi.org/10.1137/S1064827595289108>.
/// - J. J. Moré (1978), "The Levenberg-Marquardt Algorithm: Implementation and
///   Theory", *Numerical Analysis*, LN Mathematics 630, 105–116.
///   <https://doi.org/10.1007/BFb0067700>.
/// - Reference comparisons use [SciPy 1.16.2's dense TRF](https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_lsq/trf.py)
///   (BSD-3-Clause). Fixed-coordinate elimination and minimum-norm interior
///   steps for deficient rank are Basin extensions; trajectories need not agree.
///
/// # Examples
///
/// ```
/// use basin::{BoxConstraints, CostFunction, DenseMatrix, Executor, Jacobian,
///     Residual, TrustRegionReflective};
/// use std::convert::Infallible;
/// struct Fit { lower: Vec<f64>, upper: Vec<f64> }
/// impl CostFunction for Fit {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
///         Ok(0.5 * (x[0] - 2.0).powi(2))
///     }
/// }
/// impl Residual for Fit {
///     type Param = Vec<f64>;
///     type Output = Vec<f64>;
///     type Error = Infallible;
///     fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
///         Ok(vec![x[0] - 2.0])
///     }
/// }
/// impl Jacobian for Fit {
///     type Jacobian = DenseMatrix;
///     fn jacobian(&self, _: &Vec<f64>) -> Result<DenseMatrix, Infallible> {
///         Ok(DenseMatrix::from_row_slice(1, 1, &[1.0]))
///     }
/// }
/// impl BoxConstraints for Fit {
///     fn lower(&self) -> &Vec<f64> { &self.lower }
///     fn upper(&self) -> &Vec<f64> { &self.upper }
/// }
/// let problem = Fit { lower: vec![0.0], upper: vec![1.0] };
/// let result = Executor::from_start(problem, TrustRegionReflective::new(),
///     vec![0.5]).max_iter(100).run().unwrap();
/// assert!((result.param()[0] - 1.0).abs() < 1e-7);
/// ```
#[derive(Clone, Debug)]
pub struct TrustRegionReflective<F: Scalar = f64> {
    gradient_tolerance: Option<F>,
    initial_radius: Option<F>,
    rank_tolerance: Option<F>,
    max_inner_attempts: usize,
    max_subproblem_iterations: usize,
    work: Option<Work<F>>,
}

impl<F: Scalar> Default for TrustRegionReflective<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Scalar> TrustRegionReflective<F> {
    /// Construct the dense reflective solver with automatic initial radius,
    /// automatic rank cutoff, gradient tolerance `1e-8`, and both limits 50.
    pub fn new() -> Self {
        Self {
            gradient_tolerance: Some(number(1e-8)),
            initial_radius: None,
            rank_tolerance: None,
            max_inner_attempts: 50,
            max_subproblem_iterations: 50,
            work: None,
        }
    }

    /// Set `max |v ⊙ Jᵀr|` tolerance over free coordinates (default `1e-8`).
    /// `None` disables the test; zero means exact zero. Values must be finite
    /// and nonnegative. Enabled convergence checks combine with OR.
    pub fn with_absolute_scaled_gradient_tolerance(
        mut self,
        value: impl Into<Option<F>>,
    ) -> Self {
        self.gradient_tolerance =
            crate::core::convergence::optional_tolerance(value);
        self
    }

    /// Set a finite positive initial scaled radius. `None` restores automatic
    /// `‖x / sqrt(v)‖₂` initialization, with one substituted when it is zero.
    pub fn with_initial_radius(mut self, value: impl Into<Option<F>>) -> Self {
        let value = value.into();
        assert!(
            value.is_none_or(|r| r.is_finite() && r > F::zero()),
            "initial radius must be finite and positive"
        );
        self.initial_radius = value;
        self
    }

    /// Set the relative singular-value cutoff in `[0, 1)`. Zero discards only
    /// exact zeros. `None` uses `epsilon * max(rows, columns)` of the augmented
    /// free-coordinate matrix. This is a rank safeguard, not a stopping test.
    pub fn with_rank_tolerance(mut self, value: impl Into<Option<F>>) -> Self {
        let value = value.into();
        assert!(
            value.is_none_or(|r| r.is_finite()
                && r >= F::zero()
                && r < F::one()),
            "rank tolerance must be finite and in [0, 1)"
        );
        self.rank_tolerance = value;
        self
    }

    /// Set the positive maximum number of trial attempts per accepted step
    /// (default 50). Exhaustion reports `SolverFailed` at the unchanged point.
    pub fn with_max_inner_attempts(mut self, value: usize) -> Self {
        assert!(value > 0, "inner attempt limit must be positive");
        self.max_inner_attempts = value;
        self
    }

    /// Set the positive secular-equation iteration limit (default 50).
    /// Failure to reach relative radius accuracy `0.01` reports `SolverFailed`.
    pub fn with_max_subproblem_iterations(mut self, value: usize) -> Self {
        assert!(value > 0, "subproblem iteration limit must be positive");
        self.max_subproblem_iterations = value;
        self
    }
}

impl<V: Clone, F: Scalar> InitialState<V> for TrustRegionReflective<F> {
    type State = NllsState<V, F>;
    fn seed(&self, x: &V) -> Self::State {
        NllsState::new(x.clone())
    }
}

#[derive(Clone, Debug)]
struct Work<F> {
    free: Vec<usize>,
    lower: Vec<F>,
    upper: Vec<F>,
    residual: Vec<F>,
    jacobian: Vec<F>,
    gradient: Vec<F>,
    optimality: F,
    radius: F,
    failed: bool,
}

fn vector<V: VectorIndex<F> + VectorLen, F: Scalar>(v: &V) -> Vec<F> {
    (0..v.vec_len()).map(|i| v.get_scalar(i)).collect()
}

fn jacobian<M: MatrixIndex<F>, F: Scalar>(
    j: &M,
    rows: usize,
    cols: usize,
    free: &[usize],
) -> Option<Vec<F>> {
    assert_eq!(
        j.matrix_rows(),
        rows,
        "Jacobian row count must equal residual count"
    );
    assert_eq!(
        j.matrix_cols(),
        cols,
        "Jacobian column count must equal parameter count"
    );
    if (0..rows).any(|i| (0..cols).any(|k| !j.matrix_entry(i, k).is_finite())) {
        return None;
    }
    Some(
        (0..rows)
            .flat_map(|i| free.iter().map(move |&k| j.matrix_entry(i, k)))
            .collect(),
    )
}

fn cost<F: Scalar>(r: &[F]) -> F {
    let length = norm(r);
    number::<F>(0.5) * length * length
}

impl<F: Scalar> Work<F> {
    fn model<V: VectorIndex<F>>(&mut self, x: &V) -> Option<Model<F>> {
        let n = self.free.len();
        let mut model = Model {
            j: self.jacobian.clone(),
            g: vec![F::zero(); n],
            c: vec![F::zero(); n],
            d: vec![F::one(); n],
        };
        self.optimality = F::zero();
        for i in 0..n {
            let g = self.gradient[i];
            let x = x.get_scalar(self.free[i]);
            let v = if g < F::zero() && self.upper[i].is_finite() {
                self.upper[i] - x
            } else if g > F::zero() && self.lower[i].is_finite() {
                x - self.lower[i]
            } else {
                F::one()
            };
            model.c[i] = if (g < F::zero() && self.upper[i].is_finite())
                || (g > F::zero() && self.lower[i].is_finite())
            {
                g.abs()
            } else {
                F::zero()
            };
            model.d[i] = v.sqrt();
            model.g[i] = model.d[i] * g;
            let optimality = v * g.abs();
            if !v.is_finite() || v <= F::zero() || !optimality.is_finite() {
                return None;
            }
            self.optimality = self.optimality.max(optimality);
            for row in model.j.chunks_mut(n) {
                row[i] = row[i] * model.d[i];
            }
        }
        model
            .j
            .iter()
            .chain(&model.g)
            .all(|x| x.is_finite())
            .then_some(model)
    }

    fn update_gradient(&mut self) -> bool {
        let n = self.free.len();
        self.gradient = (0..n)
            .map(|i| {
                self.residual
                    .iter()
                    .enumerate()
                    .map(|(row, &r)| self.jacobian[row * n + i] * r)
                    .sum()
            })
            .collect();
        self.gradient.iter().all(|x| x.is_finite())
    }
}

impl<P, V, F> Solver<P, NllsState<V, F>> for TrustRegionReflective<F>
where
    F: Scalar,
    P: Residual<Param = V, Output = V> + Jacobian + BoxConstraints<Param = V>,
    P::Jacobian: MatrixIndex<F>,
    V: Clone + VectorLen + VectorIndex<F>,
{
    type Error = <P as Residual>::Error;
    fn init(
        &mut self,
        problem: &mut Problem<P>,
        state: NllsState<V, F>,
    ) -> Result<NllsState<V, F>, Self::Error> {
        self.init_evaluated(problem, state)
    }
    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        state: NllsState<V, F>,
    ) -> Result<(NllsState<V, F>, Option<TerminationReason>), Self::Error> {
        self.next_evaluated(problem, state)
    }
    fn terminate(&self, state: &NllsState<V, F>) -> Option<TerminationReason> {
        self.terminate_evaluated(state)
    }
}

impl<P, L, V, F> Solver<RobustLeastSquares<P, L, F>, NllsState<V, F>>
    for TrustRegionReflective<F>
where
    F: Scalar,
    P: Residual<Param = V, Output = V> + Jacobian + BoxConstraints<Param = V>,
    P::Jacobian: MatrixIndex<F>,
    V: Clone + VectorLen + VectorIndex<F>,
    L: LossFunction<F>,
    P::Jacobian: ScaleRowsInPlace<F>,
{
    type Error = <P as Residual>::Error;
    fn init(
        &mut self,
        problem: &mut Problem<RobustLeastSquares<P, L, F>>,
        state: NllsState<V, F>,
    ) -> Result<NllsState<V, F>, Self::Error> {
        self.init_evaluated(problem, state)
    }
    fn next_iter(
        &mut self,
        problem: &mut Problem<RobustLeastSquares<P, L, F>>,
        state: NllsState<V, F>,
    ) -> Result<(NllsState<V, F>, Option<TerminationReason>), Self::Error> {
        self.next_evaluated(problem, state)
    }
    fn terminate(&self, state: &NllsState<V, F>) -> Option<TerminationReason> {
        self.terminate_evaluated(state)
    }
}

impl<F: Scalar> TrustRegionReflective<F> {
    fn init_evaluated<V, M, E>(
        &mut self,
        problem: &mut E,
        state: NllsState<V, F>,
    ) -> Result<NllsState<V, F>, E::Error>
    where
        V: Clone + VectorLen + VectorIndex<F>,
        M: MatrixIndex<F>,
        E: BoundedEvaluation<V, M, F>,
    {
        self.work = None;
        let mut state = NllsState::new(state.param);
        let n = state.param.vec_len();
        assert!(n > 0, "TRF requires at least one parameter");
        let (lo, hi) = (problem.lower(), problem.upper());
        assert_eq!(lo.vec_len(), n, "lower bound shape mismatch");
        assert_eq!(hi.vec_len(), n, "upper bound shape mismatch");
        let mut work = Work {
            free: Vec::new(),
            lower: Vec::new(),
            upper: Vec::new(),
            residual: Vec::new(),
            jacobian: Vec::new(),
            gradient: Vec::new(),
            optimality: F::infinity(),
            radius: F::one(),
            failed: false,
        };
        for i in 0..n {
            let (x, l, u) = (
                state.param.get_scalar(i),
                lo.get_scalar(i),
                hi.get_scalar(i),
            );
            assert!(x.is_finite(), "initial parameters must be finite");
            assert!(
                !l.is_nan()
                    && !u.is_nan()
                    && l <= u
                    && l < F::infinity()
                    && u > F::neg_infinity(),
                "invalid box bounds"
            );
            if let Some(x) = interior(x, l, u, true) {
                state.param.set_scalar(i, x);
            } else {
                work.failed = true;
            }
            if l < u {
                work.free.push(i);
                work.lower.push(l);
                work.upper.push(u);
            }
        }
        state.cost = Some(F::infinity());
        if !work.failed {
            if work.free.is_empty() {
                let r = problem.residual(&state.param)?;
                work.residual = vector(&r);
                state.cost = Some(problem.cost(&r, |_| cost(&work.residual)));
                work.optimality = F::zero();
            } else {
                let (r, j) = problem.residual_and_jacobian(&state.param)?;
                work.residual = vector(&r);
                state.cost = Some(problem.cost(&r, |_| cost(&work.residual)));
                if let Some((model_r, j)) = problem.model(&r, j) {
                    if let Some(model_r) = model_r {
                        work.residual = vector(&model_r);
                    }
                    if let Some(j) =
                        jacobian(&j, work.residual.len(), n, &work.free)
                    {
                        work.jacobian = j;
                    } else {
                        work.failed = true;
                    }
                } else {
                    work.failed = true;
                }
            }
            assert!(
                !work.residual.is_empty(),
                "TRF requires at least one residual"
            );
            let f = state.cost.expect("evaluated cost");
            state.cost = Some(if f.is_finite() { f } else { F::infinity() });
            work.failed |=
                !f.is_finite() || work.residual.iter().any(|x| !x.is_finite());
            if !work.failed && !work.free.is_empty() {
                work.failed = !work.update_gradient();
                if !work.failed {
                    if let Some(model) = work.model(&state.param) {
                        let scaled: Vec<F> = work
                            .free
                            .iter()
                            .zip(&model.d)
                            .map(|(&i, &d)| state.param.get_scalar(i) / d)
                            .collect();
                        let auto = norm(&scaled);
                        work.radius = self.initial_radius.unwrap_or(
                            if auto > F::zero() { auto } else { F::one() },
                        );
                        work.failed = !work.radius.is_finite();
                    } else {
                        work.failed = true;
                    }
                }
            }
        }
        self.work = Some(work);
        Ok(state)
    }

    fn next_evaluated<V, M, E>(
        &mut self,
        problem: &mut E,
        mut state: NllsState<V, F>,
    ) -> NllsStep<V, F, E::Error>
    where
        V: Clone + VectorLen + VectorIndex<F>,
        M: MatrixIndex<F>,
        E: BoundedEvaluation<V, M, F>,
    {
        let work = self.work.as_mut().expect("TRF must be initialized");
        let failed = Some(TerminationReason::SolverFailed);
        if work.failed {
            return Ok((state, failed));
        }
        let Some(model) = work.model(&state.param) else {
            work.failed = true;
            return Ok((state, failed));
        };
        let n = work.free.len();
        let m = work.residual.len();
        let mut augmented = model.j.clone();
        augmented.resize((m + n) * n, F::zero());
        for i in 0..n {
            augmented[(m + i) * n + i] = model.c[i].sqrt();
        }
        let Some(svd) = DenseSvd::factor(m + n, n, augmented) else {
            work.failed = true;
            return Ok((state, failed));
        };
        let mut rhs = work.residual.clone();
        rhs.resize(m + n, F::zero());
        let x: Vec<F> = work
            .free
            .iter()
            .map(|&i| state.param.get_scalar(i))
            .collect();
        let theta = number::<F>(0.995).max(F::one() - work.optimality);
        let old_cost = state.cost.expect("TRF must be initialized");
        for _ in 0..self.max_inner_attempts {
            let Some(p) = svd.trust_step(
                &rhs,
                work.radius,
                self.rank_tolerance,
                self.max_subproblem_iterations,
            ) else {
                break;
            };
            let (mut h, _) = model.select(
                &x,
                &work.lower,
                &work.upper,
                &p,
                work.radius,
                theta,
            );
            let mut trial = state.param.clone();
            let mut feasible = true;
            let mut roundoff = F::zero();
            for i in 0..n {
                if let Some(value) = interior(
                    x[i] + model.d[i] * h[i],
                    work.lower[i],
                    work.upper[i],
                    false,
                ) {
                    trial.set_scalar(work.free[i], value);
                    h[i] = (value - x[i]) / model.d[i];
                    // Reconstructing a small step loses accuracy at the
                    // parameter scale, measured here in scaled coordinates.
                    roundoff = roundoff.hypot(
                        (F::epsilon() * x[i].abs().max(value.abs()))
                            / model.d[i],
                    );
                } else {
                    feasible = false;
                    break;
                }
            }
            let length = norm(&h);
            let predicted = -model.value(&h);
            if !feasible
                || !length.is_finite()
                || length == F::zero()
                || !predicted.is_finite()
                || predicted <= F::zero()
            {
                break;
            }
            if length - work.radius
                > number::<F>(64.0) * F::epsilon() * work.radius
                    + number::<F>(4.0) * roundoff
            {
                break;
            }
            let raw_r = problem.residual(&trial)?;
            let r = vector(&raw_r);
            assert_eq!(r.len(), m, "residual shape changed during solve");
            let new_cost = problem.cost(&raw_r, |_| cost(&r));
            if !new_cost.is_finite() || r.iter().any(|x| !x.is_finite()) {
                work.radius = number::<F>(0.25) * length;
                continue;
            }
            let actual = old_cost - new_cost;
            let ratio = actual / predicted;
            work.radius =
                update_radius(work.radius, length, ratio).min(F::max_value());
            if actual > F::zero() {
                let j = problem.jacobian(&trial)?;
                let Some((model_r, j)) = problem.model(&raw_r, j) else {
                    break;
                };
                let Some(j) =
                    jacobian(&j, m, state.param.vec_len(), &work.free)
                else {
                    break;
                };
                work.residual = model_r.map_or(r, |r| vector(&r));
                work.jacobian = j;
                state.param = trial;
                state.cost = Some(new_cost);
                work.failed = !work.update_gradient()
                    || work.model(&state.param).is_none();
                return Ok((state, if work.failed { failed } else { None }));
            }
        }
        work.failed = true;
        Ok((state, failed))
    }

    fn terminate_evaluated<V>(
        &self,
        _: &NllsState<V, F>,
    ) -> Option<TerminationReason> {
        self.work
            .as_ref()
            .filter(|w| {
                !w.failed
                    && (w.free.is_empty()
                        || self
                            .gradient_tolerance
                            .is_some_and(|t| w.optimality <= t))
            })
            .map(|_| TerminationReason::SolverConverged)
    }
}
