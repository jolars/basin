//! `MA-SW-Chains`: the Solis-Wets-chain configuration of the generic
//! [`MaLsCh`] memetic solver (Molina et al., CEC 2010).
//!
//! Everything algorithmic lives in [`ma_ls_ch`](crate::solver::ma_ls_ch)
//! (the SSGA framework and chain bookkeeping) and in [`SolisWets`]'s
//! [`ResumableInner`](crate::core::inner::ResumableInner) impl (fresh
//! chains at `ρ = ½ ·` nearest-neighbor distance with the cost slot
//! primed, resume via a local iter reset, no per-segment tolerance —
//! segments are purely budget-driven). This module is the concrete
//! public face: the [`MaLsChSw`]/[`MaLsChSwState`] aliases plus the
//! constructor.

use crate::core::state::SolisWetsState;
use crate::solver::ma_ls_ch::{MaLsCh, MaLsChGenericState};
use crate::solver::solis_wets::SolisWets;

/// `MA-SW-Chains`: [`MaLsCh`] with Solis-Wets as the chain operator,
/// per Molina, Lozano & Herrera (CEC 2010) — the winner of the CEC'2010
/// large-scale global optimization competition.
///
/// The high-dimensional counterpart of
/// [`MaLsChCma`](crate::solver::MaLsChCma): where a CMA-ES chain stores
/// an O(n²) covariance per individual, a Solis-Wets chain snapshot is
/// just `(#s, #f, bias, ρ)` — O(n) per individual and O(n) per
/// evaluation — so the chain-memetic approach stays viable when the
/// dimension grows. The trade-off is isotropic (plus bias) mutations:
/// on strongly ill-conditioned basins at moderate dimension the CMA
/// variant typically refines deeper.
///
/// See [`MaLsCh`] for the algorithm, shared default parameters,
/// contract, and termination notes. The CEC'2010 benchmark setting at
/// `n = 1000` used `I_str = 500`
/// ([`with_ls_intensity`](MaLsCh::with_ls_intensity)); basin keeps the
/// family-wide default of `300`.
///
/// # Backends
///
/// The outer SSGA and the Solis-Wets inner need only the vector tier,
/// so all four backends work — `Vec<f64>`, `nalgebra::DVector<f64>`
/// (feature `nalgebra`), `ndarray::Array1<f64>` (feature `ndarray`),
/// and `faer::Col<f64>` (feature `faer`) — with **no matrix type and no
/// `linalg` tier involved**, unlike the CMA variant.
///
/// # References
///
/// - Molina, D., Lozano, M., and Herrera, F. (2010). "MA-SW-Chains:
///   Memetic algorithm based on local search chains for large scale
///   continuous global optimization." *IEEE Congress on Evolutionary
///   Computation (CEC 2010)*, 3153-3160.
///   <https://doi.org/10.1109/CEC.2010.5586034>
///
/// # Examples
///
/// ```
/// use basin::{
///     BoxConstraints, CostFunction, Executor, MaLsChSw, MaLsChSwState, MaxCostEvals,
/// };
///
/// struct BoundedSphere {
///     lower: Vec<f64>,
///     upper: Vec<f64>,
/// }
/// impl CostFunction for BoundedSphere {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
///         Ok(x.iter().map(|xi| xi * xi).sum())
///     }
/// }
/// impl BoxConstraints for BoundedSphere {
///     fn lower(&self) -> &Vec<f64> { &self.lower }
///     fn upper(&self) -> &Vec<f64> { &self.upper }
/// }
///
/// let problem = BoundedSphere { lower: vec![-5.0; 5], upper: vec![5.0; 5] };
/// let result = Executor::new(
///     problem,
///     MaLsChSw::<Vec<f64>>::new(42).with_pop_size(20),
///     MaLsChSwState::new(),
/// )
/// .max_iter(u64::MAX)
/// .terminate_on(MaxCostEvals(10_000))
/// .run()
/// .unwrap();
/// assert!(result.cost() < 1e-6);
/// ```
pub type MaLsChSw<V> = MaLsCh<V, SolisWets>;

/// State carried by [`MaLsChSw`]: the [`MaLsChGenericState`] whose
/// chain slots hold saved `(SolisWets, SolisWetsState)` pairs — the
/// [`SolisWets`] carries the hyperparameters + RNG stream; the
/// [`SolisWetsState`] carries the iterate, bias, `ρ`, and streak
/// counters (the MA-SW-Chains §II.C snapshot).
pub type MaLsChSwState<V> = MaLsChGenericState<V, (SolisWets, SolisWetsState<V>)>;

impl<V> MaLsCh<V, SolisWets> {
    /// Build a new `MaLsChSw` with the Molina 2010 §4.4.7 framework
    /// defaults, a default [`SolisWets`] prototype (1981 paper
    /// constants), and a PRNG seeded from `seed`.
    ///
    /// The prototype's RNG is never drawn (each fresh chain reseeds from
    /// the outer RNG per the
    /// [`ResumableInner`](crate::core::inner::ResumableInner) purity
    /// contract). To customize the Solis-Wets constants, construct via
    /// [`MaLsCh::with_inner`] instead:
    /// `MaLsCh::with_inner(seed, SolisWets::new(0).with_bias_gain(0.3))`.
    pub fn new(seed: u64) -> Self {
        Self::with_inner(seed, SolisWets::new(0))
    }
}
