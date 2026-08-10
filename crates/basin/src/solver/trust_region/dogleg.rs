//! Dogleg trust-region subproblem.

use super::{CauchyPoint, Step, Subproblem, model_decrease, tau_to_boundary};
use crate::core::math::{
    Dot, LinearSolveSpd, MatVec, NegInPlace, NormSquared, Scalar, ScaleInPlace,
    ScaledAdd,
};

/// Dogleg subproblem solver (Nocedal & Wright, *Numerical Optimization*, 2e,
/// eq. 4.16; Powell 1970).
///
/// Approximates the trust-region step by a two-segment path: from the origin
/// to the unconstrained Cauchy minimizer `p_U = −(gᵀg / gᵀBg) g`, then on to
/// the full Newton step `p_B = −B⁻¹g`. The returned step is the point where
/// this path first reaches radius `Δ` (or `p_B` itself when it lies inside
/// the region).
///
/// The Newton step requires a Cholesky solve, so the Hessian type must
/// implement [`LinearSolveSpd`]; when the Hessian is **not** positive
/// definite (factorization fails) the geometry of the dogleg path breaks
/// down and this strategy falls back to the [`CauchyPoint`] step. For
/// problems with frequently indefinite Hessians prefer [`Steihaug`], which
/// handles negative curvature directly.
///
/// [`Steihaug`]: super::Steihaug
#[derive(Debug, Clone, Copy, Default)]
pub struct Dogleg;

impl<V, M, F> Subproblem<V, M, F> for Dogleg
where
    F: Scalar,
    V: Clone
        + Dot<F>
        + NormSquared<F>
        + ScaledAdd<F>
        + ScaleInPlace<F>
        + NegInPlace,
    M: MatVec<V> + LinearSolveSpd<V>,
{
    fn solve(&self, g: &V, b: &M, radius: F) -> Step<V, F> {
        // Full Newton step p_B = −B⁻¹ g. A non-SPD Hessian makes the dogleg
        // geometry invalid, so defer to the Cauchy point there.
        let p_b = match b.solve_spd(g) {
            Ok(mut p) => {
                p.neg_in_place();
                p
            }
            Err(_) => return CauchyPoint.solve(g, b, radius),
        };
        let p_b_norm = p_b.norm_squared().sqrt();
        if p_b_norm <= radius {
            // Newton step is inside the region: take it whole.
            let predicted_reduction = model_decrease(g, b, &p_b);
            return Step {
                d: p_b,
                predicted_reduction,
                hit_boundary: p_b_norm >= radius,
            };
        }

        // Cauchy minimizer along −g: p_U = −(gᵀg / gᵀBg) g.
        let bg = b.matvec(g);
        let gbg = g.dot(&bg);
        let gg = g.dot(g);
        // With an SPD B the Newton solve above succeeded and gᵀBg > 0; this
        // guard covers only pathological roundoff, walking to the boundary
        // along steepest descent.
        if gbg <= F::zero() {
            let mut d = g.clone();
            d.scale_in_place(-(radius / gg.sqrt()));
            let predicted_reduction = model_decrease(g, b, &d);
            return Step {
                d,
                predicted_reduction,
                hit_boundary: true,
            };
        }

        let mut p_u = g.clone();
        p_u.scale_in_place(-(gg / gbg));
        let p_u_norm = p_u.norm_squared().sqrt();
        if p_u_norm >= radius {
            // The Cauchy point is already outside the region: clip the
            // steepest-descent step to the boundary.
            let mut d = g.clone();
            d.scale_in_place(-(radius / gg.sqrt()));
            let predicted_reduction = model_decrease(g, b, &d);
            return Step {
                d,
                predicted_reduction,
                hit_boundary: true,
            };
        }

        // Second leg: walk from p_U toward p_B until ‖·‖ = Δ. With p_U inside
        // and p_B outside the region, the crossing parameter τ lies in (0, 1).
        let mut diff = p_b;
        diff.scaled_add(-F::one(), &p_u); // diff = p_B − p_U
        let tau = tau_to_boundary(&p_u, &diff, radius);
        let mut d = p_u;
        d.scaled_add(tau, &diff);
        let predicted_reduction = model_decrease(g, b, &d);
        Step {
            d,
            predicted_reduction,
            hit_boundary: true,
        }
    }
}
