//! CMA-ES distribution state.
//!
//! Carries the λ sampled candidates (the [`PopulationState`] surface)
//! **and** the search distribution that defines them: the mean `m`,
//! step-size `σ`, covariance `C`, its eigenpair `B`/`D` (with `D⁻¹`),
//! and the evolution paths `p_σ`/`p_c`. Holding the distribution here lets
//! [`CmaEsTolerance`](crate::core::termination::CmaEsTolerance) bind on
//! it like [`SimplexTolerance`](crate::core::termination::SimplexTolerance)
//! binds on a simplex, and lets both [`CmaEs`](crate::solver::CmaEs) and
//! [`BoundedCmaEs`](crate::solver::BoundedCmaEs) be configuration-only.
//!
//! # Result semantics: `xfavorite` vs `xbest`
//!
//! CMA-ES's recommended solution is the distribution mean (pycma's
//! `xfavorite`), not the best evaluated sample. So [`State::param`]/
//! [`State::cost`] return `m` and `f(m)` (the solver evaluates `f(m)`
//! once per generation), while [`State::best_param`]/
//! [`State::best_cost`] track the best evaluated point ever seen
//! (pycma's `xbest`). An
//! [`OptimizationResult`](crate::core::executor::OptimizationResult)
//! therefore surfaces both: `result.param()` is the recommended mean,
//! `result.best_param()` the safest evaluated point.
//!
//! The single shared state is used by the bounded variant too: the
//! adaptive boundary-penalty bookkeeping rides along as
//! [`penalty`](CmaEsState): an `Option<BoundPenalty>` that is `None`
//! for plain [`CmaEs`](crate::solver::CmaEs) and installed by
//! [`BoundedCmaEs`](crate::solver::BoundedCmaEs)'s `init`. This mirrors
//! [`LbfgsState`](crate::LbfgsState)'s `work: Option<LbfgsbWork>` field,
//! where one state serves both the bounded and unbounded solver.

use std::collections::VecDeque;

use crate::core::math::{
    ComponentMulAssign, MatrixFromDiagonal, MatrixIdentity, Scalar,
    ScaleInPlace, VectorLen,
};
use crate::core::problem::EvalCounts;
use crate::core::state::{CountsMirror, PopulationState, State};

/// Solver state for CMA-ES and bounded CMA-ES.
///
/// Construct with [`new`](Self::new) (isotropic `C = I`) or
/// `CmaEsState::new(mean, sigma).with_stds(stds)` (anisotropic
/// `C = diag(stds²)`). The solver fills the first generation's
/// `candidates`/`costs` and the cost of the mean in
/// [`Solver::init`](crate::core::solver::Solver::init).
///
/// The scalar `F` defaults to `f64` so call sites resolve unchanged.
pub struct CmaEsState<V, M, F = f64> {
    // PopulationState data.
    /// λ sampled candidates, sorted by ascending cost after every
    /// `init`/`next_iter`.
    pub(crate) candidates: Vec<V>,
    /// Costs in parallel with [`candidates`](Self::candidates) (the
    /// *penalized* cost for the bounded variant).
    pub(crate) costs: Vec<F>,

    // The mean is the recommended iterate.
    /// Distribution mean `m`, the recommended solution (`xfavorite`).
    pub(crate) m: V,
    /// `f(m)`, evaluated once per generation by the solver. `None`
    /// before [`Solver::init`](crate::core::solver::Solver::init).
    pub(crate) m_cost: Option<F>,

    // Search distribution.
    pub(crate) sigma: F,
    pub(crate) p_sigma: V,
    pub(crate) p_c: V,
    pub(crate) c: M,
    /// Eigenvectors of `C` from the most recent eigendecomposition.
    pub(crate) b: M,
    /// Square roots of `C`'s eigenvalues (Hansen's `D`).
    pub(crate) d: V,
    /// Reciprocals of [`d`](Self::d), for `C^{−1/2} = B D⁻¹ Bᵀ`.
    pub(crate) d_inv: V,
    /// Generation counter for the `h_σ` formula.
    pub(crate) generation: u64,

    // --- best evaluated point (xbest/fbest) ---
    pub(crate) best_param: Option<V>,
    pub(crate) best_cost: F,
    pub(crate) best_iter: u64,
    pub(crate) best_cost_evals: u64,

    // --- counters ---
    pub(crate) iter: u64,
    pub(crate) cost_evals: u64,

    // --- bounded-only; None for plain CmaEs (mirrors LbfgsState::work) ---
    pub(crate) penalty: Option<BoundPenalty<V, F>>,
}

/// Mutable adaptive boundary-penalty bookkeeping for
/// [`BoundedCmaEs`](crate::solver::BoundedCmaEs). `None` on the
/// unconstrained path; installed by `BoundedCmaEs::init`. The
/// derived penalty *constants* (`damp`, `edist_threshold`,
/// `hist_cap`) stay on the solver; only the per-iteration state lives
/// here.
pub(crate) struct BoundPenalty<V, F = f64> {
    /// Per-coordinate quadratic-penalty weights `γ ∈ Rⁿ`.
    pub(crate) gamma: V,
    /// True once `gamma` has been calibrated from the fitness history.
    pub(crate) weights_initialized: bool,
    /// Recent normalized fitness-IQR estimates (newest first).
    pub(crate) hist: VecDeque<F>,
    /// Raw (un-penalized) f-values of the most recent generation, in
    /// sample order, read by the γ-update only as a flat bag for the
    /// IQR.
    pub(crate) raw_costs: Vec<F>,
}

impl<V, M, F> CmaEsState<V, M, F>
where
    V: VectorLen
        + Clone
        + ScaleInPlace<F>
        + std::ops::IndexMut<usize, Output = F>,
    M: MatrixIdentity,
    F: Scalar,
{
    /// Build an initial CMA-ES state at `mean` with step-size `sigma`
    /// and an isotropic covariance `C = I`. The solver populates the
    /// first generation in [`Solver::init`](crate::core::solver::Solver::init).
    ///
    /// # Panics
    ///
    /// Panics if `sigma ≤ 0` or `mean` is empty.
    pub fn new(mean: V, sigma: F) -> Self {
        assert!(
            sigma > F::zero(),
            "CmaEsState requires sigma > 0, got {:?}",
            sigma
        );
        let n = mean.vec_len();
        assert!(n >= 1, "CmaEsState requires a non-empty mean");

        let mut p_sigma = mean.clone();
        p_sigma.scale_in_place(F::zero());
        let p_c = p_sigma.clone();

        let mut d = mean.clone();
        let mut d_inv = mean.clone();
        for i in 0..n {
            d[i] = F::one();
            d_inv[i] = F::one();
        }

        Self {
            candidates: Vec::new(),
            costs: Vec::new(),
            m: mean,
            m_cost: None,
            sigma,
            p_sigma,
            p_c,
            c: M::identity(n),
            b: M::identity(n),
            d,
            d_inv,
            generation: 0,
            best_param: None,
            best_cost: F::infinity(),
            best_iter: 0,
            best_cost_evals: 0,
            iter: 0,
            cost_evals: 0,
            penalty: None,
        }
    }
}

impl<V, M, F> CmaEsState<V, M, F>
where
    V: VectorLen
        + Clone
        + ComponentMulAssign
        + std::ops::IndexMut<usize, Output = F>,
    M: MatrixFromDiagonal<V>,
    F: Scalar,
{
    /// Seed an anisotropic initial covariance `C = diag(stds²)` instead
    /// of the isotropic default. The first generation then samples
    /// `m + σ · diag(stds) · N(0, I)`, i.e. optimizing in coordinates
    /// rescaled by `1/stds`. `σ` remains the scalar overall step-size;
    /// `stds` only sets the *shape*. For a diagonal `C` the
    /// eigendecomposition is exactly `B = I`, `D = diag(stds)`, so
    /// `(d, d_inv)` are seeded directly without an eigensolve.
    ///
    /// # Panics
    ///
    /// Panics if `stds.len() != mean.len()` or any entry is not
    /// strictly positive (a non-positive std makes `1/stds` non-finite
    /// in the `C^{−1/2}` factor).
    pub fn with_stds(mut self, stds: V) -> Self {
        let n = self.m.vec_len();
        assert_eq!(
            stds.vec_len(),
            n,
            "CmaEsState::with_stds requires stds.len() == mean.len(), got {} vs {}",
            stds.vec_len(),
            n
        );
        for i in 0..n {
            assert!(
                stds[i] > F::zero(),
                "CmaEsState::with_stds requires every std > 0, got stds[{}] = {:?}",
                i,
                stds[i]
            );
        }
        let mut sq = stds.clone();
        sq.component_mul_assign(&stds);
        self.c = M::from_diagonal(&sq);
        for i in 0..n {
            self.d[i] = stds[i];
            self.d_inv[i] = F::one() / stds[i];
        }
        self
    }
}

impl<V, M, F> CmaEsState<V, M, F>
where
    V: VectorLen + std::ops::Index<usize, Output = F>,
    F: Scalar,
{
    /// The distribution mean `m`, CMA-ES's recommended solution
    /// (`xfavorite`). Same value [`State::param`] returns.
    pub fn mean(&self) -> &V {
        &self.m
    }

    /// The current step-size `σ`.
    pub fn sigma(&self) -> F {
        self.sigma
    }

    /// Largest single-axis standard deviation of the search
    /// distribution, `maxᵢ dᵢ` (`dᵢ` are the square roots of `C`'s
    /// eigenvalues). The TolX test reads `σ · max_axis_std`.
    pub(crate) fn max_axis_std(&self) -> F {
        let mut m = F::zero();
        for i in 0..self.d.vec_len() {
            let v = self.d[i];
            if v > m {
                m = v;
            }
        }
        m
    }
}

impl<V: Clone, M, F: Scalar> State for CmaEsState<V, M, F> {
    type Param = V;
    type Float = F;

    fn iter(&self) -> u64 {
        self.iter
    }

    fn increment_iter(&mut self) {
        self.iter += 1;
    }

    fn cost_evals(&self) -> u64 {
        self.cost_evals
    }

    fn param(&self) -> &V {
        &self.m
    }

    /// Cost at the mean, `f(m)`.
    ///
    /// # Panics
    ///
    /// Panics if read before
    /// [`Solver::init`](crate::core::solver::Solver::init) has
    /// evaluated the mean. By contract the executor calls `init` before
    /// any termination check, so reads from criteria and from
    /// [`OptimizationResult`](crate::core::executor::OptimizationResult)
    /// are safe.
    fn cost(&self) -> F {
        self.m_cost.expect(
            "CmaEsState::cost read before Solver::init evaluated the mean",
        )
    }

    fn best_param(&self) -> &V {
        self.best_param.as_ref().expect(
            "CmaEsState::best_param read before Solver::init populated it",
        )
    }

    fn best_cost(&self) -> F {
        self.best_cost
    }

    fn best_iter(&self) -> u64 {
        self.best_iter
    }

    fn best_cost_evals(&self) -> u64 {
        self.best_cost_evals
    }

    /// Refresh the best evaluated point from this generation. Considers
    /// both the best sample (`candidates[0]`, sorted ascending) and the
    /// mean (`m`, with cost `m_cost`), keeping whichever is lowest over
    /// all history, so `best_param()` is the true `xbest` even when the
    /// mean is worse than a sample (or vice versa).
    fn update_best(&mut self) {
        if let Some(&best_sample_cost) = self.costs.first() {
            if self.best_param.is_none() || best_sample_cost < self.best_cost {
                self.best_param = Some(self.candidates[0].clone());
                self.best_cost = best_sample_cost;
                self.best_iter = self.iter;
                self.best_cost_evals = self.cost_evals;
            }
        }
        if let Some(mc) = self.m_cost {
            if self.best_param.is_none() || mc < self.best_cost {
                self.best_param = Some(self.m.clone());
                self.best_cost = mc;
                self.best_iter = self.iter;
                self.best_cost_evals = self.cost_evals;
            }
        }
    }

    fn reset_best(&mut self) {
        self.best_param = None;
        self.best_cost = F::infinity();
        self.best_iter = 0;
        self.best_cost_evals = 0;
    }
}

impl<V, M, F> CountsMirror for CmaEsState<V, M, F>
where
    CmaEsState<V, M, F>: State,
{
    fn mirror(&mut self, delta: &EvalCounts) {
        // Derivative-free outer: any kind of work (e.g. an L-BFGS inner's
        // gradient calls inside a CMA-injection wrapper) folds into the
        // single `cost_evals` counter, matching `BasicPopulationState`.
        self.cost_evals = delta.total_work();
    }
}

impl<V: Clone, M, F: Scalar> PopulationState for CmaEsState<V, M, F> {
    fn candidates(&self) -> &[V] {
        &self.candidates
    }

    fn costs(&self) -> &[F] {
        &self.costs
    }
}
