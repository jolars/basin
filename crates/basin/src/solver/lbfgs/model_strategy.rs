//! Alternative bounded quasi-Newton model and direction policies.
use super::integrated_model::{Model, Workspace};
use super::*;
/// Projected searches with independent path and curvature-retention controls.
pub mod projected;
pub use projected::{ProjectedLbfgs, ProjectedSearch};

/// Direction selection for [`ModelLbfgsb`]. These policies can change trajectories.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum ModelDirection {
    /// Generalized Cauchy point and a dimension-dependent reduced solve.
    #[default]
    Bounded,
    /// Try an inverse step when its start and endpoint are strictly interior.
    /// A failed line search retries the bounded direction with history intact.
    InteriorFirst,
    /// Apply retained curvature on the current free coordinates, then project
    /// the endpoint and search along the feasible chord. Unsafe restricted
    /// curvature pairs are skipped without deleting the original pairs.
    /// Pairs can become usable again when coordinates are released from bounds.
    /// Capacity eviction and numerical restarts still limit their lifetime.
    /// This is projected L-BFGS, rather than the L-BFGS-B Cauchy algorithm.
    Projected,
    /// Explicit name for the projected-endpoint chord policy. Equivalent to
    /// [`Self::Projected`], whose existing meaning and serialized tag are retained.
    ProjectedChord,
}

/// Opt-in L-BFGS-B model with small-dimensional signed columns.
///
/// Construct with [`Lbfgs::with_compact_model`] after configuring the base solver.
/// For `n <= m` it uses signed BFGS columns; otherwise it factors a scaled
/// `m`-dimensional Schur complement. Reduced solves choose inverse recursion,
/// direct free-coordinate elimination, or a symmetric history-space system.
/// Basin's projected Newton/backtracking safeguard is retained. Alternative
/// [`ModelDirection`] policies are explicit and leave the reference default alone.
///
/// This path validates its own model and does not construct the reference K/T
/// factors. Unusable models/directions use the existing one-restart budget;
/// a restart clears curvature before retrying. Reference compact-phase
/// diagnostics are unavailable on this path; selected-point diagnostics remain.
/// Rounding, callback sequences, and convergence outcomes may differ from the
/// reference solver. No general speed or convergence improvement is guaranteed.
///
/// # Curvature retention
///
/// Reference L-BFGS-B already retains accepted full pairs across free-set changes.
/// [`ModelDirection::Projected`] additionally applies inverse BFGS updates on
/// the current free coordinates, skipping unsafe restricted pairs without
/// deleting them. Subsequent directions can reuse them after coordinates become
/// free again. Normal eviction and numerical restarts can still remove pairs.
/// The base solver's curvature acceptance and stopping settings remain in effect.
/// [`Self::with_projected_search`] selects a separate wrapper with explicit ray,
/// projected Armijo or chord geometry and its own global curvature safeguard;
/// see [`ProjectedLbfgs`].
///
/// Inverse policies defer derived model construction until a bounded fallback
/// needs it. A failed InteriorFirst line search retries the bounded direction
/// with history intact; failed probes are counted and hard errors propagate.
///
/// # Lifecycle
///
/// Fresh initialization resets the model. Exact continuation retains the
/// underlying history and policy, rebuilding only derived scratch after serde.
/// [`Self::into_inner`] discards curvature and returns configuration for a fresh
/// reference run; it is not an exact-continuation conversion.
///
/// ```
/// use basin::solver::lbfgs::model_strategy::ModelDirection;
/// let solver = basin::Lbfgsb::<Vec<f64>>::new()
///     .with_m_capacity(10)
///     .with_compact_model()
///     .with_direction_policy(ModelDirection::InteriorFirst);
/// ```
///
/// # Backends
///
/// Contiguous `Vec<F>`, nalgebra `DVector<F>`, ndarray `Array1<F>`, and faer
/// `Col<F>`, for `f32` and `f64`, with the same custom slice and bounded line
/// search requirements as [`Lbfgs`].
///
/// # References
///
/// BFGS and compact-model identities: see [`Lbfgs`]. Model representation,
/// reduced solves and direction policies adapted from sisyfuzz 0.1.2,
/// `lbfgsb/model.rs`, `lbfgsb.rs`, and `quasi_newton.rs` (Apache-2.0).
/// Projected mode searches a feasible chord to the projected endpoint.
/// Sisyfuzz's default projected Armijo search instead projects each trial;
/// its More-Thuente and Wolfe searches cap the original direction at the first
/// bound. The chord policy differs from both choices.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ModelLbfgsb<V, F: Scalar = f64, S = MoreThuente<F>> {
    solver: Lbfgsb<V, F, S>,
    direction: ModelDirection,
    #[cfg_attr(feature = "serde", serde(skip))]
    work: RunWorkspace<F>,
}

impl<V, F: Scalar, S> ModelLbfgsb<V, F, S> {
    pub(super) fn new(solver: Lbfgsb<V, F, S>) -> Self {
        Self {
            solver,
            direction: ModelDirection::Bounded,
            work: RunWorkspace::default(),
        }
    }
    /// Select a numerical direction policy. Configure before starting a run.
    pub fn with_direction_policy(mut self, policy: ModelDirection) -> Self {
        self.direction = policy;
        self
    }
    /// Read the selected direction policy.
    pub fn direction_policy(&self) -> ModelDirection {
        self.direction
    }
    /// Select a projected search separately from the retained inverse model.
    /// See [`ProjectedLbfgs`] for stopping, curvature and checkpoint settings.
    /// This selects free-coordinate inverse updates regardless of the preceding
    /// [`ModelDirection`]. Existing model-direction calls keep their behavior.
    pub fn with_projected_search(
        self,
        search: ProjectedSearch<F>,
    ) -> ProjectedLbfgs<V, F, S> {
        ProjectedLbfgs::new(self, search)
    }
    /// Recover the base solver configuration, discarding curvature and derived
    /// model caches. A fresh executor run initializes the reference model.
    pub fn into_inner(mut self) -> Lbfgsb<V, F, S> {
        self.solver.history = None;
        self.solver
    }
}

pub(super) struct RunWorkspace<F: Scalar> {
    model: Option<Model<F>>,
    work: Option<Workspace<F>>,
    shape: (usize, usize),
    policy: ModelDirection,
    free: Vec<bool>,
    pub(super) raw_direction: bool,
    pub(super) projected_trials: bool,
    pub(super) lower: Vec<F>,
    pub(super) upper: Vec<F>,
}
impl<F: Scalar> Default for RunWorkspace<F> {
    fn default() -> Self {
        Self {
            model: None,
            work: None,
            shape: (0, 0),
            policy: ModelDirection::Bounded,
            free: Vec::new(),
            raw_direction: false,
            projected_trials: false,
            lower: Vec::new(),
            upper: Vec::new(),
        }
    }
}
impl<F: Scalar> RunWorkspace<F> {
    pub(super) fn direction<V: AsFloatSlice<F>>(
        &mut self,
        ws: &[V],
        wy: &[V],
        sy: &[F],
        ss: &[F],
        theta: F,
        m: usize,
        update: u32,
        x: &[F],
        g: &[F],
        lower: &[F],
        upper: &[F],
        allow_interior: bool,
        z: &mut [F],
    ) -> Result<bool, ()> {
        let n = x.len();
        if self.projected_trials {
            self.lower.resize(n, F::zero());
            self.lower.copy_from_slice(lower);
            self.upper.resize(n, F::zero());
            self.upper.copy_from_slice(upper);
        }
        if self.model.is_none() || self.shape != (n, m) {
            self.model = Some(Model::new(n, m));
            self.work = Some(Workspace::new(n, m));
            self.shape = (n, m);
        }
        let model = self.model.as_mut().unwrap();
        if self.policy == ModelDirection::InteriorFirst
            && allow_interior
            && x.iter()
                .enumerate()
                .all(|(i, v)| *v > lower[i] && *v < upper[i])
        {
            if !model.prepare_inverse(ws, wy, theta) {
                return Err(());
            }
            model.inverse(g, None, z);
            for i in 0..n {
                z[i] = x[i] - z[i];
            }
            if z.iter()
                .enumerate()
                .all(|(i, v)| v.is_finite() && *v > lower[i] && *v < upper[i])
                && descending(x, g, z)
            {
                return Ok(true);
            }
        }
        if matches!(
            self.policy,
            ModelDirection::Projected | ModelDirection::ProjectedChord
        ) {
            if !model.prepare_inverse(ws, wy, theta) {
                return Err(());
            }
            self.free.resize(n, false);
            for i in 0..n {
                self.free[i] = lower[i] != upper[i]
                    && !((x[i] <= lower[i] && g[i] > F::zero())
                        || (x[i] >= upper[i] && g[i] < F::zero()));
            }
            model.inverse(g, Some(&self.free), z);
            if self.raw_direction {
                let mut slope = F::zero();
                for i in 0..n {
                    z[i] = -z[i];
                    if (x[i] <= lower[i] && z[i] < F::zero())
                        || (x[i] >= upper[i] && z[i] > F::zero())
                        || lower[i] == upper[i]
                    {
                        z[i] = F::zero();
                    }
                    slope = slope + g[i] * z[i];
                }
                return if z.iter().all(|v| v.is_finite())
                    && slope.is_finite()
                    && slope < F::zero()
                {
                    Ok(false)
                } else {
                    Err(())
                };
            }
            for i in 0..n {
                z[i] = (x[i] - z[i]).max(lower[i]).min(upper[i]);
            }
        } else {
            if !model.rebuild(ws, wy, sy, ss, theta, update) {
                return Err(());
            }
            let work = self.work.as_mut().unwrap();
            model.cauchy(x, g, work, z, lower, upper);
            model.subspace(x, g, z, work, lower, upper);
        }
        if z.iter().all(|v| v.is_finite()) && descending(x, g, z) {
            Ok(false)
        } else {
            Err(())
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(super) struct ProjectedSettings<F: Scalar> {
    pub search: ProjectedSearch<F>,
    pub curvature_tolerance: F,
}

pub(super) fn clear_history<V, F: Scalar>(
    history: &mut super::workspace::History<V, F>,
) {
    history.ws.clear();
    history.wy.clear();
    history.sy.fill(F::zero());
    history.ss.fill(F::zero());
    history.theta = F::one();
    if let Some(work) = history.work.as_mut() {
        work.reset_history();
    }
}
fn descending<F: Scalar>(x: &[F], g: &[F], z: &[F]) -> bool {
    let slope = x
        .iter()
        .zip(g)
        .zip(z)
        .fold(F::zero(), |s, ((x, g), z)| s + (*z - *x) * *g);
    slope.is_finite() && slope < F::zero()
}

impl<P, V, S, F> Solver<P, FirstOrderState<V, F>> for ModelLbfgsb<V, F, S>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>
        + Gradient<Gradient = V>
        + BoxProblem<Param = V>,
    V: AsFloatSliceMut<F> + Clone + Dot<F> + ScaledAdd<F> + VectorLen,
    S: LineSearch<P, V, F, Error = P::Error>,
{
    type Error = crate::ConstraintSolveError<P::Error>;
    fn init(
        &mut self,
        p: &mut Problem<P>,
        s: FirstOrderState<V, F>,
    ) -> Result<FirstOrderState<V, F>, Self::Error> {
        self.work = RunWorkspace::default();
        self.solver.init(p, s)
    }
    fn next_iter(
        &mut self,
        p: &mut Problem<P>,
        s: FirstOrderState<V, F>,
    ) -> Result<crate::SolverStep<FirstOrderState<V, F>>, Self::Error> {
        self.work.policy = self.direction;
        self.solver
            .next_iter_bounded_model(p, s, None, Some(&mut self.work))
    }
    fn reset_convergence(&mut self) {
        <Lbfgsb<V,F,S> as Solver<P,FirstOrderState<V,F>>>::reset_convergence(&mut self.solver);
    }
    fn should_check_iterate_change(&self) -> bool {
        <Lbfgsb<V,F,S> as Solver<P,FirstOrderState<V,F>>>::should_check_iterate_change(&self.solver)
    }
    fn check_convergence(
        &mut self,
        p: &Problem<P>,
        s: &FirstOrderState<V, F>,
    ) -> Option<Termination<F>> {
        self.solver.check_convergence(p, s)
    }
    fn terminate(&self, s: &FirstOrderState<V, F>) -> Option<Termination<F>> {
        <Lbfgsb<V, F, S> as Solver<P, FirstOrderState<V, F>>>::terminate(
            &self.solver,
            s,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    struct Toy {
        lower: Vec<f64>,
        upper: Vec<f64>,
    }
    impl CostFunction for Toy {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
            Ok(0.5 * (x[0] - 0.3).powi(2))
        }
    }
    impl Gradient for Toy {
        type Gradient = Vec<f64>;
        fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
            Ok(vec![x[0] - 0.3])
        }
    }
    impl crate::Constraints for Toy {
        type Set = crate::ConstraintSet<crate::BoxBounds<Vec<f64>>>;
        fn constraints(
            &self,
        ) -> <Self::Set as crate::ConstraintDescription>::Borrowed<'_> {
            crate::ConstraintSet::new().with_bounds(crate::BoxBounds::new(
                Some(&self.lower),
                Some(&self.upper),
            ))
        }
    }
    struct FailSecond(usize);
    impl LineSearch<Toy, Vec<f64>> for FailSecond {
        type Error = Infallible;
        fn next(
            &mut self,
            _: &mut Problem<Toy>,
            _: &Vec<f64>,
            _: f64,
            _: &Vec<f64>,
            _: &Vec<f64>,
        ) -> Result<f64, Infallible> {
            Ok(0.25)
        }
        fn next_with_bounds(
            &mut self,
            p: &mut Problem<Toy>,
            x: &Vec<f64>,
            _: f64,
            _: &Vec<f64>,
            _: &Vec<f64>,
            bounds: LineSearchBounds<f64>,
        ) -> Result<LineSearchResult<Vec<f64>, f64>, Infallible> {
            self.0 += 1;
            if self.0 == 2 {
                p.cost(x)?;
                Ok(LineSearchResult::new(LineSearchOutcome::Failed))
            } else {
                Ok(LineSearchResult::new(LineSearchOutcome::Step(
                    0.25_f64.min(bounds.max()),
                )))
            }
        }
    }
    #[test]
    fn interior_failure_retries_bounded_with_history_and_charges_failed_probe()
    {
        let mut solver = Lbfgsb::with_line_search(FailSecond(0))
            .with_compact_model()
            .with_direction_policy(ModelDirection::InteriorFirst);
        let mut problem = Problem::new(Toy {
            lower: vec![-1.0],
            upper: vec![1.0],
        });
        let state = solver
            .init(&mut problem, FirstOrderState::new(vec![0.4]))
            .unwrap();
        let state = solver.next_iter(&mut problem, state).unwrap().state;
        assert_eq!(solver.solver.history.as_ref().unwrap().ws.len(), 1);
        let _ = solver.next_iter(&mut problem, state).unwrap();
        assert_eq!(solver.solver.line_search.0, 3);
        assert_eq!(solver.solver.history.as_ref().unwrap().ws.len(), 2);
        assert_eq!(problem.counts().cost_evals, 4);
        assert_eq!(problem.counts().gradient_evals, 3);
    }
}
