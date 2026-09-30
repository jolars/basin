//! `MA-LSCh-CMA`: the CMA-ES-chain configuration of the generic
//! [`MaLsCh`] memetic solver (Molina et al. 2010 §4.4).
//!
//! Everything algorithmic lives in [`ma_ls_ch`](crate::solver::ma_ls_ch)
//! (the SSGA framework and chain bookkeeping) and in [`CmaEs`]'s
//! [`ResumableInner`](crate::core::inner::ResumableInner) impl (fresh
//! chains at `σ = ½ ·` nearest-neighbor distance, resume via a local
//! iter reset, per-segment TolX at `1e-12 ·` the starting σ). This
//! module provides the [`MaLsChCma`] alias, constructor, and builder.
//! The outer solver owns the saved chains and publishes [`PopulationProgress`](crate::PopulationProgress).

use crate::core::math::{MatrixIdentity, Scalar, ScaleInPlace, VectorLen};
use crate::solver::cma_es::CmaEs;
use crate::solver::ma_ls_ch::MaLsCh;

/// `MA-LSCh-CMA`: [`MaLsCh`] with CMA-ES as the chain operator, per
/// Molina et al. 2010 §4.4.
///
/// The outer solver retains each individual's complete CMA-ES
/// distribution (`m`, `σ`, `C`, `p_σ`, `p_c`, eigendecomposition
/// `B/D`) in its chain slot, so re-selecting it resumes the same CMA-ES
/// run. CMA-ES adapts a per-basin search distribution; the chain
/// mechanism rewards basins that keep improving by extending their LS
/// time. See [`MaLsCh`] for the algorithm, default parameters,
/// contract, and termination notes.
///
/// # Backends
///
/// Same dense coverage and matrix capabilities as [`CmaEs`]: `Vec<F>` with
/// [`DenseMatrix`](crate::DenseMatrix), nalgebra `DVector<F>`, ndarray `Array1<F>`,
/// and faer `Col<F>`, for `F = f32` or `f64`. Publishes shared
/// [`PopulationProgress`](crate::PopulationProgress); chains live on the solver.
///
/// # Examples
///
/// A memetic algorithm pairing a steady-state GA with CMA-ES local-search
/// chains. See [`RandomSearch`](crate::RandomSearch) for the population-
/// based `Executor` pattern.
pub type MaLsChCma<V, M, F = f64> = MaLsCh<V, CmaEs<V, M, F>, F>;

impl<V, M, F: Scalar> MaLsCh<V, CmaEs<V, M, F>, F>
where
    V: VectorLen
        + Clone
        + ScaleInPlace<F>
        + std::ops::IndexMut<usize, Output = F>,
    M: MatrixIdentity,
{
    /// Build a new `MaLsChCma` with the Molina 2010 §4.4.7 defaults
    /// and a PRNG seeded from `seed`.
    ///
    /// The CMA prototype held internally is `CmaEs::new(0, F::one())`; its RNG is
    /// never drawn (each fresh chain reseeds from the outer RNG per the
    /// [`ResumableInner`](crate::core::inner::ResumableInner) purity
    /// contract), so the dummy seed is inert.
    pub fn new(seed: u64) -> Self {
        Self::with_inner(seed, CmaEs::new(0, F::one()))
    }

    /// Override the inner CMA-ES population size `λ_inner` (default is
    /// [`CmaEs::default_lambda(D)`](CmaEs::default_lambda) computed at
    /// init time from the problem's dimension).
    ///
    /// # Panics
    ///
    /// Panics if `lambda < 4` (Hansen 2016's lower bound on CMA-ES λ).
    pub fn with_inner_lambda(mut self, lambda: usize) -> Self {
        assert!(lambda >= 4, "inner_lambda must be >= 4, got {}", lambda);
        self.ls = self.ls.with_lambda(lambda);
        self
    }

    /// Renamed alias of
    /// [`with_initial_scale_fallback`](MaLsCh::with_initial_scale_fallback):
    /// σ is CMA-ES's name for the chain scale, but the knob is
    /// operator-agnostic, so the generic builder uses the neutral name.
    #[deprecated(
        since = "1.5.0",
        note = "renamed to `with_initial_scale_fallback`"
    )]
    pub fn with_initial_sigma_fallback(self, sigma: F) -> Self {
        self.with_initial_scale_fallback(sigma)
    }
}
