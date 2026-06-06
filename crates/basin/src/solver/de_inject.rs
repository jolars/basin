use std::marker::PhantomData;

use rand_distr::uniform::SampleUniform;

use crate::core::constraint::BoxConstraints;
use crate::core::executor::OptimizationResult;
use crate::core::inner::{InnerExecutor, WarmStart};
use crate::core::math::{SampleUniformBox, Scalar, ScaleInPlace, ScaledAdd, VectorLen};
use crate::core::problem::{CostFunction, Problem};
use crate::core::solver::Solver;
use crate::core::state::{BasicPopulationState, CountsMirror, State};
use crate::core::termination::{MaxCostEvals, TerminationCriterion, TerminationReason};
use crate::solver::cma_es::sort_population_ascending;
use crate::solver::cma_inject::MemeticInner;
use crate::solver::de::De;

/// Memetic Differential Evolution with per-generation injection: outer
/// [`De`] runs a full DE/rand/1/bin generation, an inner local solver
/// ([`MemeticInner`]) refines the best `k` candidates, and the refined
/// points are box-clipped and written back if they improve the
/// candidates they came from.
///
/// `DeInject` is the DE-flavored sibling of
/// [`CmaInject`](crate::solver::CmaInject): same composition shape (top-k
/// refinement after the outer step, no per-individual chain state) on
/// top of DE's bounded-search loop instead of CMA-ES's covariance model.
/// For the persistent-chain analogue with a steady-state GA outer, see
/// [`MaLsChCma`](crate::solver::MaLsChCma).
///
/// # Algorithm
///
/// One [`next_iter`](Solver::next_iter) =
///
/// 1. **DE generation.** Delegate to [`De::next_iter`]: mutation
///    (DE/rand/1), reinit-per-coord bound repair, binomial crossover,
///    greedy (`≤`) selection, sort ascending.
/// 2. **Inject best `k`.** For each of the top-`k` candidates (the
///    population is sorted ascending after step 1):
///    - Seed the inner via
///      [`MemeticInner::seed_scaled`] at the candidate. DE has no σ
///      analogue, so the seeder receives `F::one()` — only Nelder-Mead's
///      seeder consults `σ` (as the initial simplex edge); LM and
///      L-BFGS-B ignore it. See "Open questions" below.
///    - Drive the inner via [`InnerExecutor::run`] against the outer's
///      shared [`Problem`] wrapper. Eval aggregation happens
///      transparently (same-problem composition; see CONTRIBUTING.md
///      "Solver composition" rule 1).
///    - Saturating-clip the refined point to the box. *Not*
///      reinit-per-coord — that's a diversity-preserving repair for
///      donor vectors; for a post-LS polish it would throw away the
///      inner's work.
///    - Re-evaluate the clipped point. **Conditional write-back**: only
///      adopt if `cost_refined < cost_origin` (strictly less). DE's main
///      selection rule uses `≤`; here we use `<` because the refined
///      point is a polish, not a fresh trial — no-cost-change swaps
///      would diversify without improving the elite.
/// 3. **Re-sort.** Restore the ascending invariant on
///    [`PopulationState`](crate::core::state::PopulationState).
///
/// One refinement schedule is supported via
/// [`with_refine_every`](Self::with_refine_every) — refine on outer iters
/// `0, n, 2n, …` for the configured `n`. Adaptive triggering (super-fit
/// control, Caponio/Neri/Tirronen 2009) is intentionally deferred.
///
/// # Inner solver
///
/// Generic over any `I: MemeticInner<V, F>`. Shipped impls cover
/// [`NelderMead`](crate::solver::NelderMead),
/// [`LevenbergMarquardt`](crate::solver::LevenbergMarquardt), and
/// [`Lbfgsb`](crate::Lbfgsb).
///
/// # Bounds
///
/// Unlike CMA-ES, [`De`] is bounded by construction (requires
/// [`BoxConstraints`]). One struct covers both unconstrained and bounded
/// use — there is no `BoundedDeInject` because DE is always bounded.
///
/// # Eval aggregation
///
/// Same-problem composition: the inner shares the outer's
/// [`Problem`] wrapper, so every inner cost / gradient / Jacobian /
/// Hessian call bumps the same
/// [`EvalCounts`](crate::core::problem::EvalCounts) as the outer's own
/// evaluations. [`BasicPopulationState`]'s [`CountsMirror`] folds every
/// kind of work into the outer's single `cost_evals` via
/// `delta.total_work()`, so a derivative-based inner (LM, L-BFGS-B) has
/// its gradient work honestly collapse into `cost_evals`. See
/// CONTRIBUTING.md "Solver composition" rule 1.
///
/// # Termination
///
/// No solver-internal optimality test — inherits [`De`]'s "pair with
/// framework criteria" model
/// ([`MaxIter`](crate::core::termination::MaxIter),
/// [`MaxCostEvals`],
/// [`MaxTime`](crate::core::termination::MaxTime),
/// [`CostTolerance`](crate::core::termination::CostTolerance), or
/// [`ParamTolerance`](crate::core::termination::ParamTolerance)). The
/// strict-less write-back preserves DE's non-increasing
/// `state.cost()` contract.
///
/// # Caveats
///
/// Highly multimodal landscapes (Rastrigin, Schwefel) reward
/// *exploration*, not the *exploitation* a local-search polish adds —
/// memetic DE measurably underperforms pure DE in this regime
/// (Neri & Tirronen 2010 §4.2, where DEahcSPX ranks 4th-of-5 on
/// Rastrigin at D=50). The survey's overall finding is that "limited
/// employment of these alternative moves appears to be the best
/// option" — refine *sparingly*, not every generation. Practical
/// mitigations: sparser schedule
/// ([`with_refine_every`](Self::with_refine_every) ≥ 5), smaller inner
/// budget ([`with_inner_max_iter`](Self::with_inner_max_iter)), or a
/// gradient-using inner ([`Lbfgsb`](crate::Lbfgsb)) that can escape
/// shallow local minima. Ackley, Levy, Booth, and Rosenbrock-style
/// basins exhibit the helpful side of the combination instead.
///
/// # Backends
///
/// Same vector-tier coverage as [`De`]: `Vec<f64>`,
/// `nalgebra::DVector<f64>` (feature `nalgebra`),
/// `ndarray::Array1<f64>` (feature `ndarray`), and `faer::Col<f64>`
/// (feature `faer`). No matrix operations required. The effective
/// coverage is the intersection of [`De`]'s and the chosen inner's —
/// e.g. an [`Lbfgsb`](crate::Lbfgsb) inner narrows to nalgebra / faer.
///
/// # References
///
/// - Noman & Iba, *Accelerating Differential Evolution Using an
///   Adaptive Local Search* (IEEE TEC, 2008) — establishes the design
///   slot `DeInject` fills (DE outer + fittest-individual LS each
///   generation, Lamarckian write-back). The paper's specific LS is
///   simplex crossover with other population members as parents (SPX),
///   not a black-box inner; `DeInject` generalises by accepting any
///   [`MemeticInner`]. The paper's *adaptive LS length* (hill-climb
///   until no improvement) is not built in here — to approximate it,
///   pair a large [`with_inner_max_iter`](Self::with_inner_max_iter)
///   with the inner's own convergence tolerance.
/// - Neri & Tirronen, *Recent advances in differential evolution: a
///   survey and experimental analysis* (Artificial Intelligence Review,
///   2010) — survey + 30D/50D benchmarks. §4.2 covers DEahcSPX; the
///   overall message ("limited employment of these alternative moves
///   appears to be the best option") motivates the
///   [`with_refine_every`](Self::with_refine_every) knob.
/// - Caponio, Neri & Tirronen, *Super-fit control adaptation in memetic
///   differential evolution frameworks* (Soft Computing, 2009) — §2.4
///   defines an adaptive trigger χ = (f_best − f_avg) /
///   max_k(f_best,k − f_avg,k), firing LS only when the best
///   individual genuinely dominates the population. Not implemented —
///   would require tracking the historical max and a probabilistic
///   schedule; `with_refine_every` is the v1 stand-in.
pub struct DeInject<I, V, F = f64>
where
    F: Scalar,
    I: MemeticInner<V, F>,
{
    de: De<F>,
    inner: InnerExecutor<I::State, I>,
    k: usize,
    refine_every: u64,
    _phantom: PhantomData<V>,
}

impl<I, V, F> DeInject<I, V, F>
where
    F: Scalar,
    I: MemeticInner<V, F>,
    I::State: CountsMirror,
{
    /// Wrap a configured [`De`] with `inner` as the local refinement
    /// step. Defaults: `k = 1` refinement per generation, refine every
    /// outer iteration, inner `max_iter = 50`.
    pub fn with_inner_solver(de: De<F>, inner: I) -> Self {
        Self {
            de,
            inner: InnerExecutor::new(inner).max_iter(50),
            k: 1,
            refine_every: 1,
            _phantom: PhantomData,
        }
    }

    /// Number of best-ranked candidates to refine each generation.
    /// Default `1`.
    ///
    /// # Panics
    ///
    /// Panics if `k == 0`. `k > pop_size` is silently clamped at
    /// runtime.
    pub fn with_k(mut self, k: usize) -> Self {
        assert!(k >= 1, "DeInject requires k >= 1, got {}", k);
        self.k = k;
        self
    }

    /// Refine every `n`-th outer iteration. With `n = 1` (default), every
    /// generation triggers a refinement pass; with `n = 5`, refinement
    /// fires on outer iters `0, 5, 10, …`.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    pub fn with_refine_every(mut self, n: u64) -> Self {
        assert!(n >= 1, "DeInject requires refine_every >= 1, got {}", n);
        self.refine_every = n;
        self
    }

    /// Inner solver iteration budget per refinement (default `50`).
    pub fn with_inner_max_iter(self, n: u64) -> Self {
        let Self {
            de,
            inner,
            k,
            refine_every,
            _phantom,
        } = self;
        Self {
            de,
            inner: inner.max_iter(n),
            k,
            refine_every,
            _phantom,
        }
    }

    /// Register a stateless termination criterion on the inner loop.
    /// Criteria are reused across every outer iteration's inner run,
    /// so they MUST be stateless across calls — `MaxIter`, the
    /// `*Tolerance` family, and `MaxCostEvals` are safe;
    /// [`MaxTime`](crate::core::termination::MaxTime) is **not**.
    /// See CONTRIBUTING.md "Solver composition" rule 2.
    pub fn inner_terminate_on<C>(self, criterion: C) -> Self
    where
        C: TerminationCriterion<I::State> + 'static,
    {
        let Self {
            de,
            inner,
            k,
            refine_every,
            _phantom,
        } = self;
        Self {
            de,
            inner: inner.terminate_on(criterion),
            k,
            refine_every,
            _phantom,
        }
    }
}

impl<I, V, F> DeInject<I, V, F>
where
    F: Scalar,
    I: MemeticInner<V, F>,
    I::State: CountsMirror,
    MaxCostEvals: TerminationCriterion<I::State> + 'static,
{
    /// Cap the inner refinement at `evals` cost-evaluations via
    /// [`MaxCostEvals`] — the local-search-intensity idiom
    /// [`MaLsChCma`](crate::solver::MaLsChCma) uses. Composes with
    /// [`with_inner_max_iter`](Self::with_inner_max_iter): whichever
    /// budget fires first stops the inner.
    ///
    /// Stateless across calls — safe to reuse for the lifetime of the
    /// outer solver (composition contract 2).
    pub fn with_ls_intensity(self, evals: u64) -> Self {
        self.inner_terminate_on(MaxCostEvals(evals))
    }
}

impl<P, I, V, F> Solver<P, BasicPopulationState<V, F>> for DeInject<I, V, F>
where
    F: Scalar + SampleUniform,
    P: CostFunction<Param = V, Output = F> + BoxConstraints<Param = V>,
    I: MemeticInner<V, F> + Solver<P, <I as WarmStart<V>>::State, Error = P::Error>,
    I::State: State<Param = V, Float = F> + CountsMirror,
    V: VectorLen
        + Clone
        + SampleUniformBox
        + ScaledAdd<F>
        + ScaleInPlace<F>
        + std::ops::Index<usize, Output = F>
        + std::ops::IndexMut<usize, Output = F>,
    De<F>: Solver<P, BasicPopulationState<V, F>, Error = P::Error>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        state: BasicPopulationState<V, F>,
    ) -> Result<BasicPopulationState<V, F>, Self::Error> {
        // Delegate the initial population sampling to vanilla DE; the
        // injection layer kicks in from `next_iter` onward.
        self.de.init(problem, state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        state: BasicPopulationState<V, F>,
    ) -> Result<(BasicPopulationState<V, F>, Option<TerminationReason>), Self::Error> {
        // 1. Vanilla DE generation: mutation, repair, crossover,
        //    selection, ascending sort.
        let (mut state, reason) = self.de.next_iter(problem, state)?;
        if let Some(r) = reason {
            return Ok((state, Some(r)));
        }

        // 2. Schedule gate. `state.iter()` is pre-increment here
        //    (executor bumps it after `next_iter` returns), so iter == 0
        //    on the first call. With `refine_every = 1` (default) this
        //    is always true.
        if state.iter() % self.refine_every != 0 {
            return Ok((state, None));
        }

        // Snapshot the box once; reused for the clip on every refined
        // candidate.
        let lo = problem.inner().lower().clone();
        let hi = problem.inner().upper().clone();
        let refine = self.k.min(state.candidates.len());

        for i in 0..refine {
            // 3. Seed the inner state. DE has no σ analogue; pass
            //    `F::one()` as the neutral default. NM's seeder reads
            //    `σ` as the initial simplex edge (1.0 is a reasonable
            //    default for the [-5, 5]-style boxes in the test suite);
            //    LM and L-BFGS-B ignore it.
            let x_seed = state.candidates[i].clone();
            let c_orig = state.costs[i];
            let inner_state = self.inner.solver().seed_scaled(&x_seed, F::one());

            // 4. Drive the inner. Same-problem composition: inner shares
            //    the outer wrapper, so its evals flow into the outer's
            //    EvalCounts transparently and the BasicPopulationState
            //    mirror picks them up via `total_work()`.
            let inner_result: OptimizationResult<I::State> =
                self.inner.run(problem, inner_state)?;

            // 5. Failure routing: bubble SolverFailed only (composition
            //    contract 3). MaxIter / tolerances / SolverConverged are
            //    clean stops — consume the inner's final iterate.
            if inner_result.reason.is_failure() {
                return Ok((state, Some(inner_result.reason)));
            }

            // 6. Saturating clip to the box. The inner may have stepped
            //    outside `[lo, hi]` (NM, LM, and L-BFGS-B are
            //    unconstrained on their seed — L-BFGS-B respects bounds
            //    only when paired with a `BoxConstraints` problem,
            //    which is this case, but we re-clip for safety).
            let mut x_refined = inner_result.state.param().clone();
            let n = x_refined.vec_len();
            for j in 0..n {
                if x_refined[j] < lo[j] {
                    x_refined[j] = lo[j];
                } else if x_refined[j] > hi[j] {
                    x_refined[j] = hi[j];
                }
            }

            // 7. Re-evaluate after the clip (counted by the shared
            //    wrapper). Conditional write-back: strictly less, so the
            //    population-best slot is monotone non-increasing under
            //    the same contract as vanilla DE's selection.
            let c_refined = problem.cost(&x_refined)?;
            if c_refined < c_orig {
                state.candidates[i] = x_refined;
                state.costs[i] = c_refined;
            }
        }

        // 8. Re-sort. Cheap; mirrors `CmaInject`.
        if refine > 0 {
            sort_population_ascending(&mut state.candidates, &mut state.costs);
        }

        Ok((state, None))
    }

    fn terminate(&self, state: &BasicPopulationState<V, F>) -> Option<TerminationReason> {
        <De<F> as Solver<P, BasicPopulationState<V, F>>>::terminate(&self.de, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::nelder_mead::NelderMead;

    #[test]
    #[should_panic(expected = "DeInject requires k >= 1")]
    fn with_k_zero_panics() {
        let _ = DeInject::<NelderMead<crate::solver::nelder_mead::Unbounded, f64>, Vec<f64>, f64>::with_inner_solver(
            De::new(0),
            NelderMead::new(),
        )
        .with_k(0);
    }

    #[test]
    #[should_panic(expected = "DeInject requires refine_every >= 1")]
    fn with_refine_every_zero_panics() {
        let _ = DeInject::<NelderMead<crate::solver::nelder_mead::Unbounded, f64>, Vec<f64>, f64>::with_inner_solver(
            De::new(0),
            NelderMead::new(),
        )
        .with_refine_every(0);
    }
}
