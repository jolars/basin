//! `MA-SW-Chains`: the Solis-Wets-chain configuration of the generic
//! [`MaLsCh`] memetic solver (Molina et al., CEC 2010).
//!
//! Everything algorithmic lives in [`ma_ls_ch`](crate::solver::ma_ls_ch)
//! (the SSGA framework and chain bookkeeping) and in [`SolisWets`]'s
//! [`ResumableInner`](crate::core::inner::ResumableInner) impl (fresh
//! chains at `ρ = ½ ·` nearest-neighbor distance with the cost slot
//! primed, resume via a local iter reset, no per-segment
//! tolerance—segments are purely budget-driven). This module is the
//! concrete [`MaLsChSw`] alias and constructor. The outer solver owns the chains
//! and publishes [`PopulationProgress`](crate::PopulationProgress).

use crate::core::math::{Scalar, ScaleInPlace, VectorLen};
use crate::solver::ma_ls_ch::MaLsCh;
use crate::solver::solis_wets::SolisWets;

/// `MA-SW-Chains`: [`MaLsCh`] with Solis-Wets as the chain operator,
/// per Molina, Lozano, and Herrera (CEC 2010)—the winner of the CEC'2010
/// large-scale global optimization competition.
///
/// The high-dimensional counterpart of
/// [`MaLsChCma`](crate::solver::MaLsChCma): where a CMA-ES chain stores
/// an O(n²) covariance per individual, a Solis-Wets chain snapshot is
/// just `(#s, #f, bias, ρ)`—O(n) per individual and O(n) per
/// evaluation—so the chain-memetic approach stays viable when the
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
/// so all four backends work—`Vec<F>`, `nalgebra::DVector<F>`
/// (feature `nalgebra`), `ndarray::Array1<F>` (feature `ndarray`),
/// and `faer::Col<F>` (feature `faer`)—with **no matrix type and no
/// `linalg` tier involved**. Both `f32` and `f64` are supported.
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
///     BoxConstraints, CostFunction, Executor, MaLsChSw, PopulationProgress,
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
///     fn lower(&self) -> &Vec<f64> {
///         &self.lower
///     }
///     fn upper(&self) -> &Vec<f64> {
///         &self.upper
///     }
/// }
///
/// let problem = BoundedSphere {
///     lower: vec![-5.0; 5],
///     upper: vec![5.0; 5],
/// };
/// let result = Executor::new(
///     problem,
///     MaLsChSw::<Vec<f64>>::new(42).with_pop_size(20),
///     PopulationProgress::empty(),
/// )
/// .max_iter(u64::MAX)
/// .max_cost_evals(10_000)
/// .run()
/// .unwrap();
/// assert!(result.cost() < 1e-6);
/// ```
pub type MaLsChSw<V, F = f64> = MaLsCh<V, SolisWets<V, F>, F>;

impl<V, F: Scalar> MaLsCh<V, SolisWets<V, F>, F>
where
    V: Clone + VectorLen + ScaleInPlace<F>,
{
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
