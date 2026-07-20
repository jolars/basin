//! `MA-LSCh-CMA`: the CMA-ES-chain configuration of the generic
//! [`MaLsCh`] memetic solver (Molina et al. 2010 §4.4).
//!
//! Everything algorithmic lives in [`ma_ls_ch`](crate::solver::ma_ls_ch)
//! (the SSGA framework and chain bookkeeping) and in [`CmaEs`]'s
//! [`ResumableInner`](crate::core::inner::ResumableInner) impl (fresh
//! chains at `σ = ½ ·` nearest-neighbor distance, resume via a local
//! iter reset, per-segment TolX at `1e-12 ·` the starting σ). This
//! module is the concrete public face: the [`MaLsChCma`]/[`MaLsChState`]
//! aliases plus the CMA-specific constructor and builder.

use crate::core::state::CmaEsState;
use crate::solver::cma_es::CmaEs;
use crate::solver::ma_ls_ch::{MaLsCh, MaLsChGenericState};

/// `MA-LSCh-CMA`: [`MaLsCh`] with CMA-ES as the chain operator, per
/// Molina et al. 2010 §4.4.
///
/// Each individual that has undergone LS keeps the *full CMA-ES
/// evolution state* (`m`, `σ`, `C`, `p_σ`, `p_c`, eigendecomposition
/// `B/D`) in its chain slot, so re-selecting it resumes the same CMA-ES
/// run. CMA-ES adapts a per-basin search distribution; the chain
/// mechanism rewards basins that keep improving by extending their LS
/// time. See [`MaLsCh`] for the algorithm, default parameters,
/// contract, and termination notes.
///
/// # Backends
///
/// Same coverage as [`CmaEs`]: the default `Vec<f64>` (via
/// [`DenseMatrix`](crate::DenseMatrix)), nalgebra, ndarray, and faer. The only
/// linear-algebra requirement is the matrix bound
/// [`SymmetricEigen`](crate::core::math::SymmetricEigen), which
/// every backend satisfies.
///
/// # Examples
///
/// A memetic algorithm pairing a steady-state GA with CMA-ES local-search
/// chains. See [`RandomSearch`](crate::RandomSearch) for the population-
/// based `Executor` pattern.
pub type MaLsChCma<V, M> = MaLsCh<V, CmaEs<V, M>>;

/// State carried by [`MaLsChCma`]: the [`MaLsChGenericState`] whose
/// chain slots hold saved `(CmaEs, CmaEsState)` pairs — the [`CmaEs`]
/// carries the derived constants + RNG; the [`CmaEsState`] carries the
/// evolution state (mean, sigma, covariance, paths) and the previous
/// generation's λ candidates the next CMA `next_iter` needs as the
/// recombination basis.
pub type MaLsChState<V, M> = MaLsChGenericState<V, (CmaEs<V, M>, CmaEsState<V, M>)>;

impl<V, M> MaLsCh<V, CmaEs<V, M>> {
    /// Build a new `MaLsChCma` with the Molina 2010 §4.4.7 defaults
    /// and a PRNG seeded from `seed`.
    ///
    /// The CMA prototype held internally is `CmaEs::new(0)`; its RNG is
    /// never drawn (each fresh chain reseeds from the outer RNG per the
    /// [`ResumableInner`](crate::core::inner::ResumableInner) purity
    /// contract), so the dummy seed is inert.
    pub fn new(seed: u64) -> Self {
        Self::with_inner(seed, CmaEs::new(0))
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
    #[deprecated(since = "1.5.0", note = "renamed to `with_initial_scale_fallback`")]
    pub fn with_initial_sigma_fallback(self, sigma: f64) -> Self {
        self.with_initial_scale_fallback(sigma)
    }
}
