use crate::core::constraint::BoxConstraints;
use crate::core::convergence::optional_tolerance;
use crate::core::math::{Scalar, VectorIndex, VectorLen};
use crate::core::parallel::{MaybeSend, MaybeSync};
use crate::core::problem::{CostFunction, Problem};
use crate::core::solver::Solver;
use crate::core::state::{PointState, State};
use crate::core::termination::TerminationReason;

mod rectangle;
use rectangle::{Rectangle, Work};

/// Original DIRECT (DIviding RECTangles) for deterministic global optimization
/// over finite box bounds, following Jones, Perttunen, and Stuckman (1993).
///
/// # Algorithm
///
/// The free coordinates are normalized to the unit hypercube. Initialization
/// evaluates its midpoint once. Each iteration selects all potentially optimal
/// rectangles using their center costs `f` and Euclidean half-diagonals `d`:
/// a rectangle `j` qualifies when some `K > 0` satisfies both
/// `f_j - K d_j <= f_i - K d_i` for every `i` and
/// `f_j - K d_j <= f_min - epsilon * abs(f_min)`.
/// The implementation groups equal sizes and constructs their lower convex
/// hull, retaining collinear eligible points and all exact cost ties.
///
/// The selected set is frozen for the iteration and processed from largest
/// to smallest, with creation order breaking ties. Each rectangle is sampled
/// at `c ± delta e_i` along every longest axis, where `delta` is one-third of
/// that side's length. Probes are ordered by axis, negative then positive.
/// Trisection proceeds along axes in ascending order of their best probe
/// cost, with axis index breaking ties. Each split divides the remaining
/// central rectangle, so the better samples occupy larger children. Integer
/// trisection depths identify equal sizes without floating-point comparisons.
///
/// This is original DIRECT, not the locally biased DIRECT-L variant. SciPy's
/// corresponding setting is `locally_biased=False`. Basin uses exact cost
/// ties; SciPy 1.16.2 groups costs within an absolute `1e-13`, so complete
/// trajectories and evaluation counts need not coincide.
///
/// # Initialization and continuation
///
/// Supply a [`PointState`] with the same dimension as the bounds. Its parameter
/// provides a backend template; its values do not select the starting point.
/// Every fresh run starts at the box midpoint and clears the partition. This
/// solver does not implement point warm starts. An entirely fixed box requires
/// just one evaluation. Exact continuation uses a solver-and-state
/// [`ExactCheckpoint`](crate::ExactCheckpoint), preserving the partition and
/// evaluation counts. With `serde`, the solver's evolving machinery is also
/// serializable. Bounds and the objective must remain unchanged during a run
/// and across exact continuation.
///
/// # Termination
///
/// By default the solver stops when the best rectangle's half-diagonal in
/// normalized coordinates is at most `1e-6`. Relative volume stopping is
/// disabled by default because volumes shrink quickly with dimension. The
/// enabled geometry checks compose with OR and return
/// [`SolverConverged`](TerminationReason::SolverConverged). They measure
/// search resolution and do not certify global optimality.
///
/// Use [`Executor`](crate::Executor) for budgets, objective targets,
/// cancellation, and stagnation. Budgets and cancellation are checked between
/// complete subdivision sweeps, so the last sweep can exceed an evaluation
/// budget. Storage grows as `O(n * evaluations)`; cost ties can make sweeps
/// large. Independent probes use [`Problem::cost_batch`], preserving numerical
/// results with the optional `parallel` feature. Actual callback scheduling
/// may differ under parallel execution.
///
/// # Numerical safeguards
///
/// Bounds must be finite and ordered. Fixed coordinates are removed from the
/// normalized geometry and restored in every callback. Interpolation avoids
/// overflowing the width of a box spanning extreme finite values. If a split
/// cannot produce distinct probes in both normalized and original coordinates,
/// the solver returns [`NumericalNoProgress`](TerminationReason::NumericalNoProgress)
/// without evaluating those probes.
///
/// NaN and positive infinity reject a sample. As an extension for rejected
/// samples, each sweep additionally explores the oldest largest rejected
/// rectangle; its cost is excluded from the finite selection diagram. This
/// permits recovery from a rejected midpoint. Geometry stopping requires a
/// usable incumbent. An all-rejected run stops through executor controls with
/// no incumbent, or reports [`SolverFailed`](TerminationReason::SolverFailed)
/// for an entirely fixed box. Negative infinity is retained and stops with
/// `SolverConverged`. Typed objective errors propagate unchanged.
///
/// # Backends
///
/// `Vec<F>`, `nalgebra::DVector<F>`, `ndarray::Array1<F>`, and `faer::Col<F>`
/// support both `F = f64` (default) and `f32`, with their corresponding backend
/// features. Only [`VectorLen`], [`VectorIndex<F>`], and `Clone` are required.
/// With `parallel`, the problem and parameter must be `Sync`, and the output
/// and problem error must be `Send`.
///
/// # Example
///
/// ```
/// use basin::{BoxConstraints, CostFunction, Direct, Executor, PointState};
/// struct BoundedQuartic { lower: Vec<f64>, upper: Vec<f64> }
/// impl CostFunction for BoundedQuartic {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
///         Ok(x.iter().map(|x| 0.5 * (x.powi(4) - 16.0*x*x + 5.0*x)).sum())
///     }
/// }
/// impl BoxConstraints for BoundedQuartic {
///     fn lower(&self) -> &Vec<f64> { &self.lower }
///     fn upper(&self) -> &Vec<f64> { &self.upper }
/// }
/// let result = Executor::new(
///     BoundedQuartic { lower: vec![-5.0; 2], upper: vec![5.0; 2] },
///     Direct::new(),
///     PointState::new(vec![0.0; 2]),
/// ).target_objective(-78.33).max_cost_evals(2000).run().unwrap();
/// assert!(result.cost() < -78.33);
/// ```
///
/// # References
///
/// Donald R. Jones, Cary D. Perttunen, and Bruce E. Stuckman,
/// “Lipschitzian optimization without the Lipschitz constant,”
/// *Journal of Optimization Theory and Applications* 79, 157–181 (1993),
/// especially pp. 169–170, the subdivision procedure and Definition 4.1.
/// <https://doi.org/10.1007/BF00941892>.
///
/// Solution quality and evaluation counts are cross-checked with
/// [SciPy 1.16.2](https://github.com/scipy/scipy/tree/v1.16.2/scipy/optimize/_direct)
/// (BSD-3-Clause), whose DIRECT kernel derives from Gablonsky's DIRECT 2.0.4
/// (MIT). Basin implements the published algorithm independently.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct Direct<F: Scalar = f64> {
    epsilon: F,
    radius_tolerance: Option<F>,
    volume_tolerance: Option<F>,
    work: Option<Work<F>>,
}

impl<F: Scalar> Default for Direct<F> {
    fn default() -> Self {
        Self {
            epsilon: F::from_f64(1e-4).unwrap(),
            radius_tolerance: Some(F::from_f64(1e-6).unwrap()),
            volume_tolerance: None,
            work: None,
        }
    }
}

impl<F: Scalar> Direct<F> {
    /// Construct original DIRECT with epsilon `1e-4`, normalized radius
    /// tolerance `1e-6`, and no volume tolerance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the potential-improvement factor (default `1e-4`).
    ///
    /// This is a rectangle-selection control, not a stopping tolerance.
    /// Smaller values allow more local refinement. Zero removes the required
    /// improvement margin. Panics unless `epsilon` is finite and nonnegative.
    pub fn with_epsilon(mut self, epsilon: F) -> Self {
        assert!(
            epsilon.is_finite() && epsilon >= F::zero(),
            "DIRECT epsilon must be finite and nonnegative"
        );
        self.epsilon = epsilon;
        self
    }

    /// Stop at a best-rectangle half-diagonal at most `value` in normalized
    /// unit-box coordinates (default `Some(1e-6)`).
    ///
    /// This is an absolute Euclidean radius after coordinate normalization,
    /// not a radius in the original units. Fixed coordinates do not contribute.
    /// `None` disables the check and zero requests exact zero. Panics for a
    /// non-finite or negative tolerance. Enabled geometry checks compose with OR.
    pub fn with_absolute_radius_tolerance(
        mut self,
        value: impl Into<Option<F>>,
    ) -> Self {
        self.radius_tolerance = optional_tolerance(value);
        self
    }

    /// Stop when the best rectangle's volume, relative to the original box
    /// over free coordinates, is at most `value` (default `None`).
    ///
    /// `None` disables the check and zero requests exact zero. The comparison
    /// uses logarithmic volumes to avoid treating underflow as convergence.
    /// Panics for a non-finite or negative tolerance. Enabled geometry checks
    /// compose with OR.
    pub fn with_relative_volume_tolerance(
        mut self,
        value: impl Into<Option<F>>,
    ) -> Self {
        self.volume_tolerance = optional_tolerance(value);
        self
    }
}

impl<P, V, F> Solver<P, PointState<V, F>> for Direct<F>
where
    F: Scalar + MaybeSend,
    P: CostFunction<Param = V, Output = F> + BoxConstraints + MaybeSync,
    P::Error: MaybeSend,
    V: Clone + VectorLen + VectorIndex<F> + MaybeSync,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PointState<V, F>,
    ) -> Result<PointState<V, F>, Self::Error> {
        self.work = None;
        let lower = problem.inner().lower();
        let upper = problem.inner().upper();
        let n = lower.vec_len();
        assert!(n > 0, "DIRECT requires a nonempty box");
        assert_eq!(upper.vec_len(), n, "DIRECT bound dimensions differ");
        assert_eq!(
            state.param().vec_len(),
            n,
            "DIRECT state and bound dimensions differ"
        );
        let mut bounds = Vec::with_capacity(n);
        let mut active = Vec::new();
        for i in 0..n {
            let lo = lower.get_scalar(i);
            let hi = upper.get_scalar(i);
            assert!(
                lo.is_finite() && hi.is_finite() && lo <= hi,
                "DIRECT requires finite ordered bounds at coordinate {i}"
            );
            bounds.push((lo, hi));
            if lo < hi {
                active.push(i);
            }
        }
        let root = Rectangle::new(
            vec![F::from_f64(0.5).unwrap(); active.len()],
            F::infinity(),
        );
        let mut work = Work::new(bounds, active, root);
        let point = work.point(state.param(), &work.rectangles[0].center);
        let cost = problem.cost(&point)?;
        work.rectangles[0].cost = cost;
        work.consider(0);
        state.reset();
        state.replace(point, cost);
        self.work = Some(work);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PointState<V, F>,
    ) -> Result<(PointState<V, F>, Option<TerminationReason>), Self::Error>
    {
        let work = self.work.as_mut().expect("DIRECT must be initialized");
        let selected = work.select(self.epsilon);
        if selected.is_empty() {
            return Ok((state, Some(TerminationReason::NumericalNoProgress)));
        }
        for id in selected {
            let Some(probes) = work.probes(state.param(), id) else {
                return Ok((
                    state,
                    Some(TerminationReason::NumericalNoProgress),
                ));
            };
            let costs = problem.cost_batch(&probes.params)?;
            work.divide(id, probes.axes, probes.centers, costs);
            if let Some(best) = work.best {
                let rectangle = &work.rectangles[best];
                state.replace(
                    work.point(state.param(), &rectangle.center),
                    rectangle.cost,
                );
                if rectangle.cost == F::neg_infinity() {
                    return Ok((
                        state,
                        Some(TerminationReason::SolverConverged),
                    ));
                }
            }
        }
        Ok((state, None))
    }

    fn terminate(
        &self,
        _state: &PointState<V, F>,
    ) -> Option<TerminationReason> {
        let work = self.work.as_ref()?;
        let Some(best) = work.best else {
            return work
                .active
                .is_empty()
                .then_some(TerminationReason::SolverFailed);
        };
        let rectangle = &work.rectangles[best];
        if rectangle.cost == F::neg_infinity() || work.active.is_empty() {
            return Some(TerminationReason::SolverConverged);
        }
        let radius = work.radius(rectangle);
        let radius_small = self.radius_tolerance.is_some_and(|t| {
            t > F::zero() && radius > F::zero() && radius <= t
        });
        let volume_small = self.volume_tolerance.is_some_and(|t| {
            t > F::zero()
                && F::from_u64(rectangle.level()).unwrap()
                    * F::from_f64(3.0).unwrap().ln()
                    >= -t.ln()
        });
        (radius_small || volume_small)
            .then_some(TerminationReason::SolverConverged)
    }
}
