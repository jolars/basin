//! Solver-owned CMA distribution and boundary-penalty history.
use crate::core::math::{
    ComponentMulAssign, MatrixFromDiagonal, MatrixIdentity, Scalar,
    ScaleInPlace, VectorLen,
};
use std::collections::VecDeque;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct Distribution<V, M, F: Scalar> {
    // Unbounded members move back to progress at publication. Bounded genotypes
    // remain here because recombination needs the points before box repair.
    pub(crate) candidates: Vec<V>,
    pub(crate) costs: Vec<F>,
    pub(crate) objective_costs: Vec<F>,
    pub(crate) m: V,
    pub(crate) m_cost: Option<F>,
    pub(crate) sigma: F,
    pub(crate) p_sigma: V,
    pub(crate) p_c: V,
    pub(crate) c: M,
    pub(crate) b: M,
    pub(crate) d: V,
    pub(crate) d_inv: V,
    pub(crate) generation: u64,
    pub(crate) penalty: Option<BoundPenalty<V, F>>,
}
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct BoundPenalty<V, F> {
    pub(crate) gamma: V,
    pub(crate) weights_initialized: bool,
    pub(crate) hist: VecDeque<F>,
    // The gamma update uses the sampled generation, before local injection.
    pub(crate) raw_costs: Vec<F>,
}
impl<V, M, F> Distribution<V, M, F>
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
    pub(crate) fn new(mean: V, sigma: F) -> Self {
        assert!(
            sigma.is_finite() && sigma > F::zero(),
            "Distribution requires sigma > 0, got {:?}",
            sigma
        );
        let n = mean.vec_len();
        assert!(n >= 1, "Distribution requires a non-empty mean");

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
            objective_costs: Vec::new(),
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
            penalty: None,
        }
    }
}

impl<V, M, F> Distribution<V, M, F>
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
    pub(crate) fn with_stds(mut self, stds: V) -> Self {
        let n = self.m.vec_len();
        assert_eq!(
            stds.vec_len(),
            n,
            "Distribution::with_stds requires stds.len() == mean.len(), got {} vs {}",
            stds.vec_len(),
            n
        );
        for i in 0..n {
            assert!(
                stds[i].is_finite() && stds[i] > F::zero(),
                "Distribution::with_stds requires every std > 0, got stds[{}] = {:?}",
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

impl<V, M, F> Distribution<V, M, F>
where
    V: VectorLen + std::ops::Index<usize, Output = F>,
    F: Scalar,
{
    pub(crate) fn max_axis_std(&self) -> F {
        (0..self.d.vec_len())
            .fold(F::zero(), |largest, i| largest.max(self.d[i]))
    }
}
