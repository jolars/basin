use crate::core::executor::OptimizationResult;
use crate::core::inner::{InitialState, InnerExecutor, WarmStart};
use crate::core::math::{
    ComponentMulAssign, MatTransposeVec, MatVec, MatrixFromDiagonal,
    MatrixIdentity, NormSquared, RankOneUpdate, SampleStandardNormal, Scalar,
    ScaleInPlace, ScaledAdd, SymmetricEigen, VectorLen,
};
use crate::core::problem::{CostFunction, Problem};
use crate::core::solver::Solver;
use crate::core::state::{
    CountsMirror, FirstOrderState, IntoInitialSimplex, PointState,
    PopulationProgress, SimplexProgress, State,
};
use crate::core::termination::TerminationReason;
use crate::solver::cma_es::{CmaEs, sort_population_ascending};
use crate::solver::lbfgs::{Bounded, Lbfgs};
use crate::solver::levenberg_marquardt::LevenbergMarquardt;
use crate::solver::nelder_mead::NelderMead;

/// An inner solver eligible to plug into a CMA-ES injection wrapper
/// ([`CmaInject`]/[`BoundedCmaInject`](crate::solver::BoundedCmaInject)).
///
/// Extends [`WarmStart`] (and thus [`InitialState`]), which supplies the
/// associated [`State`](InitialState::State) shape and the σ-free
/// [`seed`](InitialState::seed). `MemeticInner` adds the step-size-scaled
/// seed CMA-ES injection needs: given a candidate `x` and the current
/// CMA step-size `σ`, configure the inner solver and build a fresh progress
/// seed whose search scale tracks the outer distribution's spread.
///
/// # Implementations
///
/// Shipped impls for [`NelderMead`], [`LevenbergMarquardt`], and
/// [`Lbfgsb`](crate::Lbfgsb). To plug in something else, either impl this trait (plus
/// [`WarmStart`] and [`InitialState`]) on your solver, or wrap a `Solver<P, S>` in
/// [`ClosureInner`] with an inline seeder closure (escape hatch for
/// one-off experiments and the `AlwaysFails`-style failure-bubbling
/// tests).
///
/// # Why an associated state type
///
/// Each inner publishes its progress shape: NM uses a simplex (`n + 1`
/// vertices), LM uses a point and cost, and L-BFGS-B also publishes a gradient.
/// The inner solver owns its evolving model. Tying
/// [`State`](InitialState::State) to [`InitialState`] lets the memetic factory
/// write `BoundedCmaInject::with_inner_solver(cma, Lbfgsb::new())`
/// without the caller having to spell out `FirstOrderState<V>` in turbofish;
/// `I` determines it.
///
/// # Eval aggregation
///
/// No per-trait hook: same-problem composition shares the outer's
/// [`Problem`] wrapper, so inner evals flow through automatically.
/// [`PopulationProgress`] preserves every raw evaluation category; inner
/// derivative work remains separate from outer cost calls.
pub trait MemeticInner<V, F = f64>: WarmStart<V>
where
    F: Scalar,
{
    /// Configure the inner search scale and supply a progress seed at `x`.
    /// The caller must then initialize a fresh solve with the returned state.
    /// The default ignores `sigma` and uses [`InitialState::seed`].
    fn seed_scaled(&mut self, x: &V, _sigma: F) -> Self::State {
        self.seed(x)
    }
}

/// Closure type for `ClosureInner`'s state seeder.
type ClosureSeedFn<V, S, F> = Box<dyn Fn(&V, F) -> S>;

/// Closure-based [`MemeticInner`] wrapper for custom inner solvers.
///
/// Intended use is one-off experiments and contract tests (e.g. the
/// `AlwaysFails` harness verifying `SolverFailed` bubbling). For
/// shipping configurations, prefer impl-ing `MemeticInner` on your
/// solver type; it's a three-line trait.
pub struct ClosureInner<I, S, V, F = f64> {
    inner: I,
    seed_fn: ClosureSeedFn<V, S, F>,
}

impl<I, S, V, F> ClosureInner<I, S, V, F> {
    /// Wrap `inner` with an explicit seeder closure.
    pub fn new(inner: I, seed_fn: impl Fn(&V, F) -> S + 'static) -> Self {
        Self {
            inner,
            seed_fn: Box::new(seed_fn),
        }
    }
}

impl<P, I, S, V, F> Solver<P, S> for ClosureInner<I, S, V, F>
where
    I: Solver<P, S>,
    S: State,
{
    type Error = I::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        state: S,
    ) -> Result<S, Self::Error> {
        self.inner.init(problem, state)
    }
    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        state: S,
    ) -> Result<(S, Option<TerminationReason>), Self::Error> {
        self.inner.next_iter(problem, state)
    }
    fn terminate(&self, state: &S) -> Option<TerminationReason> {
        self.inner.terminate(state)
    }
}

impl<I, S, V, F> InitialState<V> for ClosureInner<I, S, V, F>
where
    F: Scalar,
    S: State<Param = V>,
{
    type State = S;
    fn seed(&self, x: &V) -> S {
        // σ-free seed: the closure receives σ = 0. `ClosureInner` is an
        // experiment and contract-test escape hatch, so a documented dummy
        // is acceptable here where it is not in the native impls.
        (self.seed_fn)(x, F::zero())
    }
}

impl<I, S, V, F> WarmStart<V> for ClosureInner<I, S, V, F>
where
    F: Scalar,
    S: State<Param = V>,
{
}

impl<I, S, V, F> MemeticInner<V, F> for ClosureInner<I, S, V, F>
where
    F: Scalar,
    S: State<Param = V>,
{
    fn seed_scaled(&mut self, x: &V, sigma: F) -> S {
        (self.seed_fn)(x, sigma)
    }
}

// WarmStart + MemeticInner impls for the three shipped inners.

impl<Mode, V, F> InitialState<V> for NelderMead<V, F, Mode>
where
    F: Scalar,
    V: VectorLen + Clone + IntoInitialSimplex<V, F>,
{
    type State = SimplexProgress<V, F>;
    fn seed(&self, x: &V) -> SimplexProgress<V, F> {
        // σ-free seed: Nelder-Mead's own default relative-step simplex
        // (FMINSEARCH and SciPy 5%), used when there is no outer step-size to
        // track (e.g. a barrier or AL inner).
        SimplexProgress::new(x.clone())
    }
}

impl<Mode, V, F> WarmStart<V> for NelderMead<V, F, Mode>
where
    F: Scalar,
    V: VectorLen + Clone + IntoInitialSimplex<V, F>,
{
}

impl<Mode, V, F> MemeticInner<V, F> for NelderMead<V, F, Mode>
where
    F: Scalar,
    V: VectorLen + Clone + IntoInitialSimplex<V, F> + crate::VectorIndex<F>,
{
    fn seed_scaled(&mut self, x: &V, sigma: F) -> SimplexProgress<V, F> {
        // σ-scaled axis-aligned simplex: edge = current CMA step-size,
        // so the inner's exploration tracks the outer distribution's
        // spread and shrinks with σ. Hansen 2011 doesn't prescribe a
        // specific simplex; this matches the S11 default that the
        // existing tests validate against.
        let n = x.vec_len();
        let mut vertices = Vec::with_capacity(n + 1);
        vertices.push(x.clone());
        for j in 0..n {
            let mut v = x.clone();
            v.set_scalar(j, v.get_scalar(j) + sigma);
            vertices.push(v);
        }
        SimplexProgress::from_simplex(vertices)
    }
}

impl<V, M, F> InitialState<V> for LevenbergMarquardt<V, M, F>
where
    F: Scalar,
    V: Clone,
{
    type State = PointState<V, F>;
    fn seed(&self, x: &V) -> PointState<V, F> {
        PointState::new(x.clone())
    }
}

impl<V, M, F> WarmStart<V> for LevenbergMarquardt<V, M, F>
where
    F: Scalar,
    V: Clone,
{
}

impl<V, M, F> MemeticInner<V, F> for LevenbergMarquardt<V, M, F>
where
    F: Scalar,
    V: Clone,
{
    // `seed_scaled` defaults to `seed`; LM ignores σ.
}

// `WarmStart` is generic over the mode marker so both `Lbfgsb` (bounded,
// used as a CMA inner) and `Lbfgs<V, F, Unbounded>` (used as a barrier or AL
// inner) seed the same `FirstOrderState`. `MemeticInner` stays on the bounded
// alias only; CMA injection pairs with the bounded variant.
impl<Mode, S, V, F> InitialState<V> for Lbfgs<V, F, Mode, S>
where
    F: Scalar,
    V: Clone,
{
    type State = FirstOrderState<V, F>;
    fn seed(&self, x: &V) -> FirstOrderState<V, F> {
        FirstOrderState::new(x.clone())
    }
}

impl<Mode, S, V, F> WarmStart<V> for Lbfgs<V, F, Mode, S>
where
    F: Scalar,
    V: Clone,
{
}

impl<S, V, F> MemeticInner<V, F> for Lbfgs<V, F, Bounded, S>
where
    F: Scalar,
    V: Clone,
{
    // `seed_scaled` defaults to `seed`; L-BFGS-B ignores σ.
}

// CmaInject: memetic CMA-ES with Hansen-2011 injection.

/// Memetic CMA-ES with Hansen (2011) injection: outer CMA-ES proposes
/// `λ` candidates per generation, an inner local solver
/// ([`MemeticInner`]) refines the best `k`, and the refined points are
/// Mahalanobis-clipped and injected back into the population for the
/// next CMA update.
///
/// The only departure from the standard
/// [`CmaEs`] update is clipping each
/// injected point's normalized step in Mahalanobis distance:
///
/// ```text
///   y_i ← min(1, c_y / ‖C^{-1/2} y_i‖) · y_i        (Hansen 2011 eq. 4)
///   c_y = √n + 2n/(n+2)                              (Table 1 default)
/// ```
///
/// with `y_i = (x_i − m)/σ` and `C^{-1/2} = B D^{-1} Bᵀ` from the
/// post-update eigendecomposition CMA-ES already maintains. After
/// clipping, replaced candidates re-enter the population on equal
/// footing with regular samples; all subsequent CMA updates
/// (m, p_σ, p_c, C, σ) run the standard equations unchanged. Lamarckian
/// by construction; no Baldwinian mode in the paper.
///
/// # Inner solver
///
/// Generic over any `I: MemeticInner<V>`. The associated `I::State`
/// determines the inner state shape. Shipped impls cover
/// [`NelderMead`], [`LevenbergMarquardt`], and [`Lbfgsb`](crate::Lbfgsb). For
/// L-BFGS-B inner with consistent bound flow, use the bounded sibling
/// [`BoundedCmaInject`](crate::solver::BoundedCmaInject) over
/// [`BoundedCmaEs`](crate::solver::BoundedCmaEs).
///
/// # Eval aggregation
///
/// Same-problem composition: the inner shares the outer's
/// [`Problem`] wrapper, so every inner cost/gradient/Jacobian/
/// Hessian call bumps the same
/// [`EvalCounts`](crate::core::problem::EvalCounts) as the outer's own
/// evaluations. [`PopulationProgress`] preserves all six categories without
/// folding derivative work into `cost_evals`. Use `counts().total_work()` for
/// an explicit aggregate. Exact checkpoints retain the outer distribution,
/// inner solver, progress, and counts together. With `serde`, these serialize
/// when their constituent types do; application hooks cannot be serialized.
///
/// # Backends
///
/// Same coverage as [`CmaEs`]: `Vec<F>` for `F = f32` or `f64` (via
/// [`DenseMatrix`](crate::DenseMatrix)), nalgebra, ndarray, and faer. The
/// matrix capabilities are those of [`CmaEs`], and the
/// shipped [`MemeticInner`] inners are
/// backend-generic.
///
/// # Examples
///
/// See [`CmaEs`] for the base population-based `Executor` pattern;
/// `CmaInject` adds a local-search inner via Hansen-2011 injection.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "I: serde::Serialize, V: serde::Serialize, M: serde::Serialize, F: serde::Serialize",
        deserialize = "I: serde::Deserialize<'de>, V: serde::Deserialize<'de>, M: serde::Deserialize<'de>, F: serde::Deserialize<'de>"
    ))
)]
pub struct CmaInject<I, V, M, F = f64>
where
    F: Scalar,
    I: MemeticInner<V, F>,
{
    cma: CmaEs<V, M, F>,
    inner: InnerExecutor<I::State, I>,
    k: usize,
    c_y_override: Option<F>,
}

impl<I, V, M, F> CmaInject<I, V, M, F>
where
    F: Scalar,
    I: MemeticInner<V, F>,
    I::State: CountsMirror,
{
    /// Wrap a configured [`CmaEs`] with `inner` as the local
    /// refinement step. Defaults: `k = 1` refinement per generation,
    /// inner `max_iter = 50`, `c_y` = Hansen-2011 Table 1 default.
    pub fn with_inner_solver(cma: CmaEs<V, M, F>, inner: I) -> Self {
        Self {
            cma,
            inner: InnerExecutor::new(inner).max_iter(50),
            k: 1,
            c_y_override: None,
        }
    }

    /// Inspect the outer distribution model without transferring its ownership.
    pub fn cma(&self) -> &CmaEs<V, M, F> {
        &self.cma
    }

    /// Number of best-ranked candidates to refine and inject each
    /// generation. Default `1`.
    ///
    /// # Panics
    ///
    /// Panics if `k == 0`. `k > λ` is silently clamped at runtime.
    pub fn with_k(mut self, k: usize) -> Self {
        assert!(k >= 1, "CmaInject requires k >= 1, got {}", k);
        self.k = k;
        self
    }

    /// Override the Hansen-2011 clipping threshold `c_y` (default
    /// `√n + 2n/(n+2)`).
    ///
    /// # Panics
    ///
    /// Panics if `c_y <= 0`.
    pub fn with_c_y(mut self, c_y: F) -> Self {
        assert!(c_y > F::zero(), "CmaInject requires c_y > 0, got {:?}", c_y);
        self.c_y_override = Some(c_y);
        self
    }

    /// Inner solver iteration budget per outer generation (default `50`).
    pub fn with_inner_max_iter(self, n: u64) -> Self {
        let Self {
            cma,
            inner,
            k,
            c_y_override,
        } = self;
        Self {
            cma,
            inner: inner.max_iter(n),
            k,
            c_y_override,
        }
    }

    /// Configure the outer CMA distribution-size tolerance; `None` disables it.
    pub fn with_absolute_distribution_size_tolerance(
        mut self,
        value: impl Into<Option<F>>,
    ) -> Self {
        self.cma = self.cma.with_absolute_distribution_size_tolerance(value);
        self
    }

    /// Add a factory creating fresh application-stop history for each inner run.
    pub fn inner_stop_when_factory<Mk, CheckFn>(self, make: Mk) -> Self
    where
        Mk: FnMut() -> CheckFn + 'static,
        CheckFn: FnMut(&I::State) -> Option<TerminationReason> + 'static,
    {
        let Self {
            cma,
            inner,
            k,
            c_y_override,
        } = self;
        Self {
            cma,
            inner: inner.stop_when_factory(make),
            k,
            c_y_override,
        }
    }
}

/// Hansen 2011 Table 1: `c_y = √n + 2n/(n+2)`, chosen so <10% of
/// regular `y_i` would be clipped at typical `n` and <1% for `n > 10`.
///
/// `pub(crate)` so the sibling
/// [`BoundedCmaInject`](crate::solver::BoundedCmaInject) can share
/// this default without re-deriving it.
pub(crate) fn default_c_y<F: Scalar>(n: usize) -> F {
    let n = F::from_usize(n).unwrap();
    let two = F::from_f64(2.0).unwrap();
    n.sqrt() + two * n / (n + two)
}

impl<P, I, V, M, F> Solver<P, PopulationProgress<V, F>>
    for CmaInject<I, V, M, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>,
    I: MemeticInner<V, F>
        + Solver<P, <I as InitialState<V>>::State, Error = P::Error>,
    I::State: State<Param = V, Float = F> + CountsMirror,
    V: VectorLen
        + Clone
        + ScaledAdd<F>
        + ScaleInPlace<F>
        + ComponentMulAssign
        + NormSquared<F>
        + SampleStandardNormal
        + std::ops::Index<usize, Output = F>
        + std::ops::IndexMut<usize, Output = F>,
    M: MatrixIdentity
        + MatrixFromDiagonal<V>
        + MatVec<V>
        + MatTransposeVec<V>
        + ScaleInPlace<F>
        + RankOneUpdate<V, F>
        + SymmetricEigen<V>
        + Clone,
    CmaEs<V, M, F>: Solver<P, PopulationProgress<V, F>, Error = P::Error>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        state: PopulationProgress<V, F>,
    ) -> Result<PopulationProgress<V, F>, Self::Error> {
        // Hansen's preliminary experiments inject from iter 1 onward,
        // so we delegate the initial population to vanilla CMA-ES.
        self.cma.init(problem, state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        state: PopulationProgress<V, F>,
    ) -> Result<
        (PopulationProgress<V, F>, Option<TerminationReason>),
        Self::Error,
    > {
        // 1. Vanilla CMA-ES iteration: update m, σ, C from the
        //    previous generation, sample λ fresh candidates sorted by
        //    cost ascending.
        let (mut state, reason) = self.cma.next_iter(problem, state)?;
        if let Some(r) = reason {
            return Ok((state, Some(r)));
        }

        let work = self.cma.work.as_mut().expect("CMA init precedes injection");
        let n = work.m.vec_len();
        let m = work.m.clone();
        let sigma = work.sigma;
        let c_y = self.c_y_override.unwrap_or_else(|| default_c_y::<F>(n));
        let refine = self.k.min(state.candidates.len());

        for i in 0..refine {
            // 2. Seed the inner state via the trait. The σ argument
            //    lets seeders that scale with the CMA distribution
            //    (NM's σ-scaled simplex) track the current spread.
            let inner_state = self
                .inner
                .solver_mut()
                .seed_scaled(&state.candidates[i], sigma);

            // 3. Drive the inner. Same-problem composition: inner shares
            //    the outer wrapper, so its evals flow into the outer's
            //    EvalCounts transparently and the PopulationProgress mirror picks
            //    them up in their original categories.
            let inner_result: OptimizationResult<I::State> =
                self.inner.run(problem, inner_state)?;

            // 4. Failure routing: bubble SolverFailed only (composition
            //    contract).
            if inner_result.reason.is_failure() {
                sort_population_ascending(
                    &mut state.candidates,
                    &mut state.costs,
                );
                return Ok((state, Some(inner_result.reason)));
            }

            // 5. Extract refined point.
            let x_refined = inner_result.state.param().clone();

            // 6. y = (x_refined − m) / σ.
            let mut y = x_refined;
            y.scaled_add(-F::one(), &m);
            y.scale_in_place(F::one() / sigma);

            // 7. ‖C^{-1/2} y‖ = ‖D^{-1} ⊙ Bᵀ y‖, with B, D⁻¹ from the solver.
            let inv_sqrt_norm = {
                let mut bt_y = work.b.mat_transpose_vec(&y);
                bt_y.component_mul_assign(&work.d_inv);
                bt_y.norm_squared().sqrt()
            };

            // 8. Clipping factor α (Hansen 2011 eq. 4 + eq. 10).
            if inv_sqrt_norm > F::zero() {
                let alpha = (c_y / inv_sqrt_norm).min(F::one());
                if alpha < F::one() {
                    y.scale_in_place(alpha);
                }
            }

            // 9. x_inj = m + σ · y_clipped.
            let mut x_inj = m.clone();
            x_inj.scaled_add(sigma, &y);

            // 10. Re-evaluate: clipping moves the point in original
            //     space, so the cost field has to match.
            let cost_new = problem.cost(&x_inj)?;

            state.candidates[i] = x_inj;
            state.costs[i] = cost_new;
        }

        // 12. Re-sort: rank-µ update depends on the order.
        if refine > 0 {
            sort_population_ascending(&mut state.candidates, &mut state.costs);
        }

        Ok((state, None))
    }
    fn terminate(
        &self,
        state: &PopulationProgress<V, F>,
    ) -> Option<TerminationReason> {
        <_ as Solver<P, PopulationProgress<V, F>>>::terminate(&self.cma, state)
    }
}
