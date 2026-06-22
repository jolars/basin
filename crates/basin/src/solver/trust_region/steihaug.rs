//! Steihaug truncated conjugate-gradient trust-region subproblem.

use super::{Step, Subproblem, model_decrease, tau_to_boundary};
use crate::core::math::{
    Dot, MatVec, NegInPlace, NormSquared, Scalar, ScaleInPlace, ScaledAdd, VectorLen,
};

/// Steihaug truncated conjugate-gradient subproblem solver (Nocedal &
/// Wright, *Numerical Optimization*, 2e, Algorithm 7.2; Steihaug 1983).
///
/// Runs conjugate gradient on the model `m(p) = gᵀp + ½ pᵀ B p`, truncating
/// as soon as it (a) generates a direction of non-positive curvature
/// `dᵀBd ≤ 0`, or (b) would step outside the trust region. In either case it
/// follows the current direction to the boundary and stops; otherwise it
/// converges to the unconstrained model minimizer inside the region, with a
/// residual tolerance `ε = min(½, √‖g‖) · ‖g‖` (the standard forcing
/// sequence). Matrix-free — it touches the Hessian only through
/// [`MatVec`] (Hessian-vector products) — so it runs on every backend and
/// scales to large, possibly indefinite problems.
///
/// This is [`TrustRegion`](super::TrustRegion)'s default subproblem.
#[derive(Debug, Clone, Copy)]
pub struct Steihaug {
    /// CG iteration cap; `None` uses the problem dimension `n` (CG's natural
    /// bound for exact arithmetic).
    max_iter: Option<usize>,
}

impl Steihaug {
    /// Steihaug truncated CG with the default iteration cap (the problem
    /// dimension `n`).
    pub fn new() -> Self {
        Self { max_iter: None }
    }

    /// Cap the number of CG iterations per subproblem solve. By default the
    /// cap is the problem dimension `n`; lowering it yields a cheaper, looser
    /// step. Must be `≥ 1`.
    pub fn with_max_iter(mut self, n: usize) -> Self {
        assert!(n >= 1, "max_iter must be ≥ 1");
        self.max_iter = Some(n);
        self
    }
}

impl Default for Steihaug {
    fn default() -> Self {
        Self::new()
    }
}

impl<V, M, F> Subproblem<V, M, F> for Steihaug
where
    F: Scalar,
    V: Clone + Dot<F> + NormSquared<F> + ScaledAdd<F> + ScaleInPlace<F> + NegInPlace + VectorLen,
    M: MatVec<V>,
{
    fn solve(&self, g: &V, b: &M, radius: F) -> Step<V, F> {
        let n = g.vec_len();
        let max_iter = self.max_iter.unwrap_or(n.max(1));

        // z₀ = 0, r₀ = g, d₀ = −r₀.
        let mut z = g.clone();
        z.scale_in_place(F::zero());
        let mut r = g.clone();
        let mut r_dot = r.dot(&r);

        // Forcing tolerance ε = min(½, √‖g‖) · ‖g‖.
        let g_norm = r_dot.sqrt();
        let half = F::from_f64(0.5).unwrap();
        let s = g_norm.sqrt();
        let tol = (if s < half { s } else { half }) * g_norm;

        // Only triggers for g ≈ 0 (ε ≥ ‖g‖ is impossible otherwise): the
        // zero step, which the driver reads as convergence.
        if g_norm <= tol {
            let predicted_reduction = model_decrease(g, b, &z);
            return Step {
                d: z,
                predicted_reduction,
                hit_boundary: false,
            };
        }

        let mut d = r.clone();
        d.neg_in_place();

        for _ in 0..max_iter {
            let bd = b.matvec(&d);
            let dbd = d.dot(&bd);

            // Non-positive curvature: the model is unbounded along ±d, so
            // walk to the trust-region boundary and stop.
            if dbd <= F::zero() {
                let tau = tau_to_boundary(&z, &d, radius);
                z.scaled_add(tau, &d);
                let predicted_reduction = model_decrease(g, b, &z);
                return Step {
                    d: z,
                    predicted_reduction,
                    hit_boundary: true,
                };
            }

            let alpha = r_dot / dbd;

            // Tentative CG iterate z + α d. If it leaves the region, clip to
            // the boundary along d instead.
            let mut z_next = z.clone();
            z_next.scaled_add(alpha, &d);
            if z_next.norm_squared().sqrt() >= radius {
                let tau = tau_to_boundary(&z, &d, radius);
                z.scaled_add(tau, &d);
                let predicted_reduction = model_decrease(g, b, &z);
                return Step {
                    d: z,
                    predicted_reduction,
                    hit_boundary: true,
                };
            }
            z = z_next;

            // r ← r + α B d; converged inside the region once it is small.
            r.scaled_add(alpha, &bd);
            let r_dot_next = r.dot(&r);
            if r_dot_next.sqrt() < tol {
                let predicted_reduction = model_decrease(g, b, &z);
                return Step {
                    d: z,
                    predicted_reduction,
                    hit_boundary: false,
                };
            }

            // d ← −r + β d.
            let beta = r_dot_next / r_dot;
            let mut d_next = r.clone();
            d_next.neg_in_place();
            d_next.scaled_add(beta, &d);
            d = d_next;
            r_dot = r_dot_next;
        }

        // Iteration cap reached: return the best interior iterate so far.
        let predicted_reduction = model_decrease(g, b, &z);
        Step {
            d: z,
            predicted_reduction,
            hit_boundary: false,
        }
    }
}
