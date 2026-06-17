//! The swappable trust-region subproblem seam of the Powell DFO family.
//!
//! Every solver in the family minimizes the shared [`QuadraticModel`] over a
//! trust region `‖d‖ ≤ Δ`, but each intersects that ball with its own feasible
//! region: nothing for NEWUOA, a box for BOBYQA, linear constraints for LINCOA.
//! [`TrustRegionSubproblem`] captures exactly that one degree of freedom — the
//! model core, the `H` update, and the driver loop are otherwise identical.
//!
//! The associated [`Region`](TrustRegionSubproblem::Region) type carries the
//! per-solver feasibility data (`()`, a shifted box, …) rather than a shared
//! constraint supertrait, keeping the box / linear / unconstrained cases
//! distinct — consistent with basin's constraint design (`.claude/rules/`).

use super::model::QuadraticModel;
use crate::core::math::Scalar;

/// The outcome of one trust-region subproblem solve.
///
/// Shared by every [`TrustRegionSubproblem`] implementation (NEWUOA's TRSAPP,
/// BOBYQA's TRSBOX, …) so the driver consumes a uniform result regardless of
/// which subproblem produced the step.
pub(crate) struct TrustRegionStep<F = f64> {
    /// The step `d`, length `n`, expressed **relative to `x_opt`**. The driver
    /// maps it to absolute coordinates as `x_opt + d = x0 + (xⱼ − x0) + d` with
    /// `j = kopt`.
    pub(crate) d: Vec<F>,
    /// CRVMIN — the least curvature `sᵢᵀ∇²Q sᵢ / ‖sᵢ‖²` seen on the *interior*
    /// path, or `0` if the path ever reached the trust-region boundary. The
    /// driver's ρ-reduction test reads this.
    pub(crate) crvmin: F,
    /// The predicted reduction `Q(x_opt) − Q(x_opt + d) ≥ 0`. The driver's RATIO
    /// accept/reject test divides the actual reduction by this.
    pub(crate) predicted_reduction: F,
}

/// A strategy that approximately minimizes the model over a trust region
/// intersected with a feasible [`Region`](Self::Region).
///
/// NEWUOA supplies TRSAPP (`Region = ()`), BOBYQA supplies TRSBOX
/// (`Region` = the shifted box), LINCOA a projected-Krylov solver. The
/// implementor is a lightweight strategy value; the model is borrowed.
pub(crate) trait TrustRegionSubproblem<F: Scalar> {
    /// The feasibility data the strategy needs beyond the trust radius.
    type Region;

    /// Approximately minimize `Q(x_opt + d)` over `‖d‖ ≤ delta` and `region`.
    fn solve(
        &self,
        model: &QuadraticModel<F>,
        delta: F,
        region: &Self::Region,
    ) -> TrustRegionStep<F>;
}
