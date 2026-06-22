use crate::core::executor::OptimizationResult;
use crate::core::inner::{InitialState, InnerExecutor, WarmStart};
use crate::core::math::{
    ComponentMulAssign, MatTransposeVec, MatVec, MatrixFromDiagonal, MatrixIdentity, NormSquared,
    RankOneUpdate, SampleStandardNormal, Scalar, ScaleInPlace, ScaledAdd, SymmetricEigen,
    VectorLen,
};
use crate::core::problem::{CostFunction, Problem};
use crate::core::solver::Solver;
use crate::core::state::{
    BasicSimplexState, CmaEsState, CountsMirror, IntoInitialSimplex, LbfgsState, NllsState, State,
};
use crate::core::termination::{TerminationCriterion, TerminationReason};
use crate::solver::cma_es::{CmaEs, sort_population_ascending};
use crate::solver::lbfgs::{Bounded, Lbfgs};
use crate::solver::levenberg_marquardt::LevenbergMarquardt;
use crate::solver::nelder_mead::NelderMead;

/// An inner solver eligible to plug into a CMA-ES injection wrapper
/// ([`CmaInject`] / [`BoundedCmaInject`](crate::solver::BoundedCmaInject)).
///
/// Extends [`WarmStart`] (and thus [`InitialState`]), which supplies the
/// associated [`State`](InitialState::State) shape and the σ-free
/// [`seed`](InitialState::seed). `MemeticInner` adds the step-size-scaled
/// seed CMA-ES injection needs: given a candidate `x` and the current
/// CMA step-size `σ`, build a fresh inner state whose default scale
/// tracks the outer distribution's spread.
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
/// Each inner has a natural state shape: NM wants a simplex (`n + 1`
/// vertices), LM wants a single iterate with cached residual / Jacobian,
/// L-BFGS-B wants the limited-memory history. Tying
/// [`State`](InitialState::State) to [`InitialState`] lets the memetic factory
/// write `BoundedCmaInject::with_inner_solver(cma, Lbfgsb::new())`
/// without the caller having to spell out `LbfgsState<V>` in turbofish —
/// `I` determines it.
///
/// # Eval aggregation
///
/// No per-trait hook: same-problem composition shares the outer's
/// [`Problem`] wrapper, so inner evals flow through automatically.
/// [`CmaEsState`]'s [`CountsMirror`] folds every kind of work
/// (`cost + gradient + residual + jacobian + hessian`) into the outer's
/// `cost_evals` via `delta.total_work()`, so a derivative-based inner
/// (LM, L-BFGS-B) has its gradient work honestly collapse into the
/// outer's single `cost_evals` counter. See CONTRIBUTING.md "Solver
/// composition" rule 1.
pub trait MemeticInner<V, F = f64>: WarmStart<V>
where
    F: Scalar,
{
    /// Build a fresh inner state seeded at CMA-ES candidate `x`, scaled
    /// by the current step-size `sigma`. Called once per refined
    /// candidate per outer generation.
    ///
    /// Defaults to the σ-free [`seed`](InitialState::seed); only inners whose
    /// state scales with σ (Nelder-Mead's simplex edge) override it.
    fn seed_scaled(&self, x: &V, _sigma: F) -> Self::State {
        self.seed(x)
    }
}

/// Closure type for `ClosureInner`'s state seeder.
type ClosureSeedFn<V, S, F> = Box<dyn Fn(&V, F) -> S>;

/// Closure-based [`MemeticInner`] wrapper for custom inners that don't
/// have a native impl. Holds an inner solver plus the seeder closure
/// `MemeticInner` would otherwise express directly.
///
/// Intended use is one-off experiments and contract tests (e.g. the
/// `AlwaysFails` harness verifying `SolverFailed` bubbling). For
/// shipping configurations, prefer impl-ing `MemeticInner` on your
/// solver type — it's a three-line trait.
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

    fn init(&mut self, problem: &mut Problem<P>, state: S) -> Result<S, Self::Error> {
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
        // experiment / contract-test escape hatch, so a documented dummy
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
    fn seed_scaled(&self, x: &V, sigma: F) -> S {
        (self.seed_fn)(x, sigma)
    }
}

// -----------------------------------------------------------------------
// WarmStart + MemeticInner impls for the three shipped inners.
// -----------------------------------------------------------------------

impl<Mode, V, F> InitialState<V> for NelderMead<Mode, F>
where
    F: Scalar,
    V: VectorLen + Clone + IntoInitialSimplex<V> + std::ops::IndexMut<usize, Output = F>,
{
    type State = BasicSimplexState<V, F>;
    fn seed(&self, x: &V) -> BasicSimplexState<V, F> {
        // σ-free seed: Nelder-Mead's own default relative-step simplex
        // (FMINSEARCH/SciPy 5%), used when there is no outer step-size to
        // track (e.g. a barrier / AL inner).
        BasicSimplexState::new(x.clone())
    }
}

impl<Mode, V, F> WarmStart<V> for NelderMead<Mode, F>
where
    F: Scalar,
    V: VectorLen + Clone + IntoInitialSimplex<V> + std::ops::IndexMut<usize, Output = F>,
{
}

impl<Mode, V, F> MemeticInner<V, F> for NelderMead<Mode, F>
where
    F: Scalar,
    V: VectorLen + Clone + IntoInitialSimplex<V> + std::ops::IndexMut<usize, Output = F>,
{
    fn seed_scaled(&self, x: &V, sigma: F) -> BasicSimplexState<V, F> {
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
            v[j] = v[j] + sigma;
            vertices.push(v);
        }
        BasicSimplexState::from_simplex(vertices)
    }
}

impl<V, M, F> InitialState<V> for LevenbergMarquardt<V, M, F>
where
    F: Scalar,
    V: Clone,
{
    type State = NllsState<V, F>;
    fn seed(&self, x: &V) -> NllsState<V, F> {
        NllsState::new(x.clone())
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
    // `seed_scaled` defaults to `seed` — LM ignores σ.
}

// `WarmStart` is generic over the mode marker so both `Lbfgsb` (bounded,
// used as a CMA inner) and `Lbfgs<Unbounded>` (used as a barrier / AL
// inner) seed the same `LbfgsState`. `MemeticInner` stays on the bounded
// alias only — CMA injection pairs with the bounded variant.
impl<Mode, S, V, F> InitialState<V> for Lbfgs<Mode, S, F>
where
    F: Scalar,
    V: Clone,
{
    type State = LbfgsState<V, F>;
    fn seed(&self, x: &V) -> LbfgsState<V, F> {
        LbfgsState::new(x.clone(), self.m_capacity)
    }
}

impl<Mode, S, V, F> WarmStart<V> for Lbfgs<Mode, S, F>
where
    F: Scalar,
    V: Clone,
{
}

impl<S, V, F> MemeticInner<V, F> for Lbfgs<Bounded, S, F>
where
    F: Scalar,
    V: Clone,
{
    // `seed_scaled` defaults to `seed` — L-BFGS-B ignores σ.
}

// -----------------------------------------------------------------------
// CmaInject — memetic CMA-ES with Hansen-2011 injection.
// -----------------------------------------------------------------------

/// Memetic CMA-ES with Hansen (2011) injection: outer CMA-ES proposes
/// `λ` candidates per generation, an inner local solver
/// ([`MemeticInner`]) refines the best `k`, and the refined points are
/// Mahalanobis-clipped and injected back into the population for the
/// next CMA update.
///
/// The only departure from the standard
/// [`CmaEs`] update is clipping each
/// injected point's normalised step in Mahalanobis distance:
///
/// ```text
///   y_i ← min(1, c_y / ‖C^{-1/2} y_i‖) · y_i        (Hansen 2011 eq. 4)
///   c_y = √n + 2n/(n+2)                              (Table 1 default)
/// ```
///
/// with `y_i = (x_i − m)/σ` and `C^{-1/2} = B D^{-1} Bᵀ` from the
/// post-update eigendecomposition CMA-ES already maintains. After
/// clipping, replaced candidates re-enter the population on equal
/// footing with regular samples — all subsequent CMA updates
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
/// [`Problem`] wrapper, so every inner cost / gradient / Jacobian /
/// Hessian call bumps the same
/// [`EvalCounts`](crate::core::problem::EvalCounts) as the outer's own
/// evaluations. [`CmaEsState`]'s [`CountsMirror`] folds every
/// kind of work into the outer's single `cost_evals` via
/// `delta.total_work()` — CMA-ES outer state has no `gradient_evals`
/// field, so a derivative-based inner (LM, L-BFGS-B) has its gradient
/// work honestly collapse into `cost_evals` with no per-trait cross-type
/// fold. See CONTRIBUTING.md "Solver composition" rule 1.
///
/// # Backends
///
/// Same coverage as [`CmaEs`]: nalgebra (`DVector` / `DMatrix`) and
/// faer (`Col` / `Mat`). `Vec<f64>` and `ndarray` produce a
/// compile-time error per tenet 5.
///
/// # Examples
///
/// See [`CmaEs`] for the base population-based `Executor` pattern;
/// `CmaInject` adds a local-search inner via Hansen-2011 injection.
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

    /// Register a termination criterion on the inner loop.
    /// Criteria are reused across every outer iteration's inner run, but
    /// each is reset at the start of every run, so stateful criteria —
    /// including [`MaxTime`](crate::core::termination::MaxTime) — are safe.
    /// See CONTRIBUTING.md "Solver composition" rule 2.
    pub fn inner_terminate_on<C>(self, criterion: C) -> Self
    where
        C: TerminationCriterion<I::State> + 'static,
    {
        let Self {
            cma,
            inner,
            k,
            c_y_override,
        } = self;
        Self {
            cma,
            inner: inner.terminate_on(criterion),
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

impl<P, I, V, M, F> Solver<P, CmaEsState<V, M, F>> for CmaInject<I, V, M, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>,
    I: MemeticInner<V, F> + Solver<P, <I as InitialState<V>>::State, Error = P::Error>,
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
    CmaEs<V, M, F>: Solver<P, CmaEsState<V, M, F>, Error = P::Error>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        state: CmaEsState<V, M, F>,
    ) -> Result<CmaEsState<V, M, F>, Self::Error> {
        // Hansen's preliminary experiments inject from iter 1 onward,
        // so we delegate the initial population to vanilla CMA-ES.
        self.cma.init(problem, state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        state: CmaEsState<V, M, F>,
    ) -> Result<(CmaEsState<V, M, F>, Option<TerminationReason>), Self::Error> {
        // 1. Vanilla CMA-ES iteration: update m, σ, C from the
        //    previous generation, sample λ fresh candidates sorted by
        //    cost ascending.
        let (mut state, reason) = self.cma.next_iter(problem, state)?;
        if let Some(r) = reason {
            return Ok((state, Some(r)));
        }

        // Snapshot the post-update distribution from the state for
        // clipping (it lives on `CmaEsState` now, not the solver).
        let n = state.m.vec_len();
        let m = state.m.clone();
        let sigma = state.sigma;
        let c_y = self.c_y_override.unwrap_or_else(|| default_c_y::<F>(n));
        let refine = self.k.min(state.candidates.len());

        for i in 0..refine {
            // 2. Seed the inner state via the trait. The σ argument
            //    lets seeders that scale with the CMA distribution
            //    (NM's σ-scaled simplex) track the current spread.
            let inner_state = self.inner.solver().seed_scaled(&state.candidates[i], sigma);

            // 3. Drive the inner. Same-problem composition: inner shares
            //    the outer wrapper, so its evals flow into the outer's
            //    EvalCounts transparently and the CmaEsState mirror picks
            //    them up via `total_work()`.
            let inner_result: OptimizationResult<I::State> =
                self.inner.run(problem, inner_state)?;

            // 4. Failure routing: bubble SolverFailed only (composition
            //    contract).
            if inner_result.reason.is_failure() {
                return Ok((state, Some(inner_result.reason)));
            }

            // 5. Extract refined point.
            let x_refined = inner_result.state.param().clone();

            // 6. y = (x_refined − m) / σ.
            let mut y = x_refined;
            y.scaled_add(-F::one(), &m);
            y.scale_in_place(F::one() / sigma);

            // 7. ‖C^{-1/2} y‖ = ‖D^{-1} ⊙ Bᵀ y‖ — B, D⁻¹ from the state.
            let inv_sqrt_norm = {
                let mut bt_y = state.b.mat_transpose_vec(&y);
                bt_y.component_mul_assign(&state.d_inv);
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
}
