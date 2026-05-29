use crate::core::constraint::BoxConstraints;
use crate::core::executor::OptimizationResult;
use crate::core::inner::{InnerExecutor, WarmStart};
use crate::core::math::{
    ClampInPlace, ComponentMulAssign, MatDiagonal, MatTransposeVec, MatVec, MatrixFromDiagonal,
    MatrixIdentity, NormSquared, RankOneUpdate, SampleStandardNormal, ScaleInPlace, ScaledAdd,
    SymmetricEigen, VectorLen,
};
use crate::core::problem::{CostFunction, Problem};
use crate::core::solver::Solver;
use crate::core::state::{BasicPopulationState, CountsMirror, State};
use crate::core::termination::{TerminationCriterion, TerminationReason};
use crate::solver::bounded_cma_es::{BoundedCmaEs, evaluate_with_penalty};
use crate::solver::cma_es::sort_population_ascending;
use crate::solver::cma_inject::{MemeticInner, default_c_y};

/// Memetic [`BoundedCmaEs`] with Hansen (2011) injection. Sibling of
/// [`CmaInject`](super::CmaInject); the outer is the bounded variant
/// of CMA-ES, and bound-respecting inners (notably
/// [`Lbfgsb`](crate::Lbfgsb)) see the same bounds the outer
/// enforces via BoundPenalty.
///
/// The injection mechanism (Hansen 2011 eq. 4 — Mahalanobis clip,
/// plug back into the standard CMA update) is unchanged from
/// `CmaInject`; only the outer solver is replaced. Adaptive
/// boundary-penalty bookkeeping inside `BoundedCmaEs` is orthogonal
/// to injection — `BoundedCmaInject` reads the same post-update
/// `m, σ, B, D^{-1}` for clipping, and re-evaluates injected
/// candidates with the same γ-weighted penalty regular samples
/// receive (otherwise an out-of-box LM/L-BFGS-B refinement would
/// dominate the population at a spuriously low raw cost — bug
/// noticed when writing the LM example).
///
/// # Inner solver
///
/// Generic over any `I: MemeticInner<V>`. The associated `I::State`
/// determines the inner state shape. Shipped impls cover
/// [`NelderMead`](crate::solver::NelderMead),
/// [`LevenbergMarquardt`](crate::solver::LevenbergMarquardt), and
/// [`Lbfgsb`](crate::Lbfgsb). L-BFGS-B is the natural inner
/// here — its `P: BoxConstraints` bound matches the outer's, and the
/// same box flows through both ends of the composition.
///
/// Note an asymmetry: NM and LM are unconstrained solvers, so even
/// when the outer enforces bounds, their inner polish may step
/// outside the box. The outer's BoundPenalty re-pulls injected
/// candidates back into feasibility through the penalized
/// re-evaluation below.
///
/// # Backends
///
/// Same coverage as [`BoundedCmaEs`]: nalgebra (`DVector` / `DMatrix`)
/// and faer (`Col` / `Mat`). `Vec<f64>` and `ndarray` produce a
/// compile-time error per tenet 5.
///
/// # Examples
///
/// See [`BoundedCmaEs`] for the bounded population-based `Executor`
/// pattern; `BoundedCmaInject` adds a bound-respecting local-search inner.
pub struct BoundedCmaInject<I, V, M>
where
    I: MemeticInner<V>,
{
    cma: BoundedCmaEs<V, M>,
    inner: InnerExecutor<I::State, I>,
    k: usize,
    c_y_override: Option<f64>,
}

impl<I, V, M> BoundedCmaInject<I, V, M>
where
    I: MemeticInner<V>,
    I::State: CountsMirror,
{
    /// Wrap a configured [`BoundedCmaEs`] with `inner` as the local
    /// refinement step. Defaults: `k = 1`, inner `max_iter = 50`,
    /// `c_y` = Hansen-2011 Table 1 default.
    pub fn with_inner_solver(cma: BoundedCmaEs<V, M>, inner: I) -> Self {
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
    /// Panics if `k == 0`.
    pub fn with_k(mut self, k: usize) -> Self {
        assert!(k >= 1, "BoundedCmaInject requires k >= 1, got {}", k);
        self.k = k;
        self
    }

    /// Override the Hansen-2011 clipping threshold `c_y` (default
    /// `√n + 2n/(n+2)`).
    ///
    /// # Panics
    ///
    /// Panics if `c_y <= 0`.
    pub fn with_c_y(mut self, c_y: f64) -> Self {
        assert!(c_y > 0.0, "BoundedCmaInject requires c_y > 0, got {}", c_y);
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

    /// Register a stateless termination criterion on the inner loop.
    /// See [`CmaInject::inner_terminate_on`](super::CmaInject::inner_terminate_on)
    /// for the statelessness contract.
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

impl<P, I, V, M> Solver<P, BasicPopulationState<V>> for BoundedCmaInject<I, V, M>
where
    P: CostFunction<Param = V, Output = f64> + BoxConstraints,
    I: MemeticInner<V> + Solver<P, <I as WarmStart<V>>::State, Error = P::Error>,
    I::State: State<Param = V, Float = f64> + CountsMirror,
    V: VectorLen
        + Clone
        + ScaledAdd<f64>
        + ScaleInPlace
        + ComponentMulAssign
        + ClampInPlace
        + NormSquared
        + SampleStandardNormal
        + std::ops::Index<usize, Output = f64>
        + std::ops::IndexMut<usize, Output = f64>,
    M: MatrixIdentity
        + MatrixFromDiagonal<V>
        + MatVec<V>
        + MatTransposeVec<V>
        + MatDiagonal<V>
        + ScaleInPlace
        + RankOneUpdate<V>
        + SymmetricEigen<V>
        + Clone,
    BoundedCmaEs<V, M>: Solver<P, BasicPopulationState<V>, Error = P::Error>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        state: BasicPopulationState<V>,
    ) -> Result<BasicPopulationState<V>, Self::Error> {
        self.cma.init(problem, state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        state: BasicPopulationState<V>,
    ) -> Result<(BasicPopulationState<V>, Option<TerminationReason>), Self::Error> {
        // 1. Standard BoundedCmaEs iteration first.
        let (mut state, reason) = self.cma.next_iter(problem, state)?;
        if let Some(r) = reason {
            return Ok((state, Some(r)));
        }

        // Snapshot CMA-ES internals for clipping.
        let (n, m, sigma) = {
            let w = self
                .cma
                .working()
                .expect("BoundedCmaEs::init must run before BoundedCmaInject::next_iter");
            (w.n, w.m.clone(), w.sigma)
        };
        let c_y = self.c_y_override.unwrap_or_else(|| default_c_y(n));
        let refine = self.k.min(state.candidates.len());

        for i in 0..refine {
            // 2. Seed inner state via the trait.
            let inner_state = self.inner.solver().seed_scaled(&state.candidates[i], sigma);

            // 3. Drive the inner solver. Same-problem composition: inner
            //    shares the outer wrapper, so its evals flow into the
            //    outer's EvalCounts transparently.
            let inner_result: OptimizationResult<I::State> =
                self.inner.run(problem, inner_state)?;

            // 4. Failure routing: bubble SolverFailed only.
            if inner_result.reason.is_failure() {
                return Ok((state, Some(inner_result.reason)));
            }

            // 5. Extract refined point.
            let x_refined = inner_result.state.param().clone();

            // 6. y = (x_refined − m) / σ.
            let mut y = x_refined;
            y.scaled_add(-1.0, &m);
            y.scale_in_place(1.0 / sigma);

            // 7. ‖C^{-1/2} y‖ = ‖D^{-1} ⊙ Bᵀ y‖.
            let inv_sqrt_norm = {
                let w = self.cma.working().expect("working still populated");
                let mut bt_y = w.b.mat_transpose_vec(&y);
                bt_y.component_mul_assign(&w.d_inv);
                bt_y.norm_squared().sqrt()
            };

            // 8. Clipping (Hansen 2011 eq. 4 + eq. 10).
            if inv_sqrt_norm > 0.0 {
                let alpha = (c_y / inv_sqrt_norm).min(1.0);
                if alpha < 1.0 {
                    y.scale_in_place(alpha);
                }
            }

            // 9. Reconstruct x_inj = m + σ · y_clipped.
            let mut x_inj = m.clone();
            x_inj.scaled_add(sigma, &y);

            // 10. Re-evaluate using the BoundPenalty so the injected
            //     candidate ranks consistently with regular samples in
            //     `state.costs`. Raw cost would let an LM/L-BFGS-B
            //     refinement that landed outside the box skip the
            //     penalty and dominate the population (bug surfaces
            //     dramatically on bound-active problems). Mirrors
            //     `bounded_cma_es.rs:next_iter`'s `evaluate_with_penalty`
            //     for regular samples; uses the same γ via
            //     `working().gamma`.
            let lo = problem.inner().lower().clone();
            let hi = problem.inner().upper().clone();
            let gamma = self
                .cma
                .working()
                .expect("working still populated")
                .gamma
                .clone();
            let cost_new = {
                let (_raw, pen) = evaluate_with_penalty(problem, &x_inj, &lo, &hi, &gamma, n)?;
                pen
            };

            state.candidates[i] = x_inj;
            state.costs[i] = cost_new;
        }

        if refine > 0 {
            sort_population_ascending(&mut state.candidates, &mut state.costs);
        }

        Ok((state, None))
    }

    fn terminate(&self, state: &BasicPopulationState<V>) -> Option<TerminationReason> {
        // TolX inherits from BoundedCmaEs unchanged.
        <BoundedCmaEs<V, M> as Solver<P, BasicPopulationState<V>>>::terminate(&self.cma, state)
    }
}
