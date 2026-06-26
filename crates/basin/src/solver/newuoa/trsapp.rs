//! TRSAPP: the NEWUOA trust-region subproblem (Powell 2006, §5).
//!
//! Given the current [`QuadraticModel`] and a trust radius `Δ`, TRSAPP finds an
//! approximate solution of
//!
//! ```text
//! minimize  Q(x_opt + d)   subject to   ‖d‖ ≤ Δ ,
//! ```
//!
//! where `x_opt` is the best interpolation point. It is the *unconstrained*
//! member of the swappable subproblem family (BOBYQA → TRSBOX, LINCOA →
//! projected Krylov); the model core is untouched.
//!
//! The method (Powell 2006, §5) is a Fletcher–Reeves truncated conjugate
//! gradient walk that produces a piecewise-linear path of monotonically
//! increasing `‖d‖` (eqs. 5.4–5.14). If the path reaches the trust-region
//! boundary, a 2-D subspace refinement loop (eqs. 5.15–5.19) sweeps the
//! boundary circle. The implicit-Hessian product `∇²Q u` is formed in `O(mn)`
//! by [`QuadraticModel::hessian_matvec`] (eq. 5.3), so no `n × n` matrix is ever
//! materialized.

use crate::core::math::Scalar;
use crate::solver::powell::{QuadraticModel, TrustRegionStep, TrustRegionSubproblem};

/// TRSAPP: NEWUOA's unconstrained trust-region subproblem strategy (Powell
/// 2006, §5). The unit value carries no state; the model is borrowed per solve.
pub(crate) struct Trsapp;

impl<F: Scalar> TrustRegionSubproblem<F> for Trsapp {
    /// TRSAPP is unconstrained—the trust ball is its only feasible region.
    type Region = ();

    fn solve(&self, model: &QuadraticModel<F>, delta: F, _region: &()) -> TrustRegionStep<F> {
        model.trust_region_step(delta)
    }
}

impl<F: Scalar> QuadraticModel<F> {
    /// Approximately minimize `Q(x_opt + d)` over `‖d‖ ≤ delta` by truncated
    /// conjugate gradients (Powell 2006, §5: subroutine TRSAPP).
    ///
    /// Returns the step `d` (relative to `x_opt`), [`CRVMIN`](TrustRegionStep::crvmin),
    /// and the [predicted reduction](TrustRegionStep::predicted_reduction).
    ///
    /// # Panics
    ///
    /// Panics unless `delta > 0`.
    pub(crate) fn trust_region_step(&self, delta: F) -> TrustRegionStep<F> {
        assert!(
            delta > F::zero(),
            "trust_region_step: delta must be positive"
        );

        let n = self.n();
        let half = F::from_f64(0.5).expect("0.5 representable");
        let two = F::from_f64(2.0).expect("2.0 representable");
        // The two §5.13 / §5.15 empirical tolerances: 1e-2 on gradient/decrease
        // ratios, used squared where it compares squared norms.
        let tol = F::from_f64(1e-2).expect("1e-2 representable");
        let tol_sq = tol * tol;
        let delta_sq = delta * delta;

        // x_opt − x0 = xpt_row(kopt); gradient at x_opt (eq. 5.7):
        //   ∇Q(x_opt) = ∇Q(x0) + ∇²Q (x_opt − x0).
        let xopt_disp = self.xpt_row(self.kopt()).to_vec();
        let hxopt = self.hessian_matvec(&xopt_disp);
        let g0: Vec<F> = self
            .gradient()
            .iter()
            .zip(&hxopt)
            .map(|(a, b)| *a + *b)
            .collect();

        let mut d = vec![F::zero(); n]; // d_0 = 0
        let mut predicted_reduction = F::zero();
        let mut crvmin = F::zero();
        let mut crvmin_set = false;

        let gnorm0_sq = dot(&g0, &g0);
        // Stationary point of Q at x_opt: nothing to do (eq. 5.5, j = 1 with a
        // zero initial gradient truncates the path at its start).
        if gnorm0_sq <= F::zero() {
            return TrustRegionStep {
                d,
                crvmin,
                predicted_reduction,
            };
        }
        let tol_gsq = tol_sq * gnorm0_sq; // (1e-2 ‖∇Q(x_opt)‖)²  (eq. 5.13)

        // ---- Phase A: interior truncated CG (eqs. 5.5–5.14) ----
        let mut g = g0.clone(); // g_{j-1} = ∇Q(x_opt + d_{j-1})
        let mut s = vec![F::zero(); n];
        let mut gnorm_sq = gnorm0_sq; // ‖g_{j-1}‖²
        let mut prev_gnorm_sq = F::zero(); // ‖g_{j-2}‖²
        let mut first = true;
        let mut hit_boundary = false;

        // Theoretical CG upper bound on segments is n (eq. 5.13).
        for _ in 0..n {
            // Direction (eq. 5.5): s_1 = −g; s_j = −g + β s_{j-1}, with the
            // Fletcher–Reeves ratio β = ‖g_{j-1}‖² / ‖g_{j-2}‖².
            if first {
                for i in 0..n {
                    s[i] = -g[i];
                }
                first = false;
            } else {
                let beta = gnorm_sq / prev_gnorm_sq;
                for i in 0..n {
                    s[i] = -g[i] + beta * s[i];
                }
            }

            let hs = self.hessian_matvec(&s);
            let kappa = dot(&s, &hs); // sᵀ ∇²Q s
            let s_sq = dot(&s, &s);
            if s_sq <= F::zero() {
                break; // degenerate direction; keep the current d
            }

            // Boundary steplength α̂: positive root of ‖d + α s‖ = Δ, i.e.
            //   s_sq α² + 2 (dᵀs) α + (‖d‖² − Δ²) = 0.
            let ds = dot(&d, &s);
            let dd = dot(&d, &d);
            let disc = ds * ds + s_sq * (delta_sq - dd);
            let alpha_bd = (-ds + disc.max(F::zero()).sqrt()) / s_sq;

            // Interior-vs-boundary branch (eq. 5.9): Q decreases monotonically to
            // the boundary iff −‖g‖² + α̂ κ ≤ 0. That covers κ ≤ 0 and the case
            // where the unconstrained CG step ‖g‖²/κ overshoots α̂.
            let alpha;
            if kappa <= F::zero() || alpha_bd * kappa <= gnorm_sq {
                alpha = alpha_bd;
                hit_boundary = true;
            } else {
                alpha = gnorm_sq / kappa; // interior CG step (eq. 5.10)
                let curv = kappa / s_sq; // sᵀ∇²Q s / ‖s‖²  (eq. 5.14)
                crvmin = if crvmin_set { crvmin.min(curv) } else { curv };
                crvmin_set = true;
            }

            // Per-segment model decrease (eq. 5.8), using the CG property
            //   g_{j-1}ᵀ s_j = −‖g_{j-1}‖²  ⇒  ΔQ = α‖g‖² − ½ α² κ ≥ 0.
            let seg_drop = alpha * gnorm_sq - half * alpha * alpha * kappa;
            predicted_reduction = predicted_reduction + seg_drop;

            // Advance d_j and g_j (eqs. 5.4, 5.11).
            for i in 0..n {
                d[i] = d[i] + alpha * s[i];
            }
            for i in 0..n {
                g[i] = g[i] + alpha * hs[i];
            }

            if hit_boundary {
                crvmin = F::zero(); // §5: CRVMIN meaningless once the boundary is met
                break; // continue with the Phase B refinement
            }

            // Truncation tests (eq. 5.13): gradient small relative to the start,
            // or this segment's decrease small relative to the total so far.
            let new_gnorm_sq = dot(&g, &g);
            if new_gnorm_sq <= tol_gsq || seg_drop <= tol * predicted_reduction {
                return TrustRegionStep {
                    d,
                    crvmin,
                    predicted_reduction,
                };
            }

            prev_gnorm_sq = gnorm_sq;
            gnorm_sq = new_gnorm_sq;
        }

        if !hit_boundary {
            // Fell out via the j = n cap while still interior.
            return TrustRegionStep {
                d,
                crvmin,
                predicted_reduction,
            };
        }

        // ---- Phase B: 2-D boundary refinement (eqs. 5.15–5.19) ----
        // Invariant ‖d‖ = Δ; `g` holds ∇Q(x_opt + d), `g0` holds ∇Q(x_opt).
        for _ in 0..n {
            let gd = g.clone(); // ∇Q(x_opt + d_{j-1}); needed unmodified by eq. 5.19
            let gd_norm_sq = dot(&gd, &gd);
            let dd = dot(&d, &d); // ≈ Δ²

            // KKT acceptance (eq. 5.15): at the constrained optimum
            // ∇Q(x_opt + d) = −λ d with λ ≥ 0, so the component of gd orthogonal
            // to d vanishes. Build that component and accept d_{j-1} if it is
            // negligible relative to ‖gd‖.
            let coef = dot(&d, &gd) / dd;
            let mut s: Vec<F> = (0..n).map(|i| gd[i] - coef * d[i]).collect();
            let s_raw_norm_sq = dot(&s, &s);
            if s_raw_norm_sq <= tol_sq * gd_norm_sq {
                break; // accept d = d_{j-1}
            }

            // Rescale to s ⟂ d, ‖s‖ = Δ (eq. 5.17).
            let scale = delta / s_raw_norm_sq.sqrt();
            for i in 0..n {
                s[i] = s[i] * scale;
            }
            let hs = self.hessian_matvec(&s);

            // d(θ) = cosθ d + sinθ s (eq. 5.16). The model change relative to
            // x_opt (eq. 5.18), with ∇²Q d = gd − g0:
            //   q(θ) = cosθ (dᵀg0) + sinθ (sᵀg0)
            //        + ½[ cos²θ (dᵀHd) + 2 sinθcosθ (sᵀHd) + sin²θ (sᵀHs) ].
            let hd: Vec<F> = (0..n).map(|i| gd[i] - g0[i]).collect();
            let d_g0 = dot(&d, &g0);
            let s_g0 = dot(&s, &g0);
            let d_hd = dot(&d, &hd);
            let s_hd = dot(&s, &hd); // = dᵀ∇²Q s by symmetry
            let s_hs = dot(&s, &hs);
            let q_of = |theta: F| -> F {
                let c = theta.cos();
                let sn = theta.sin();
                c * d_g0 + sn * s_g0 + half * (c * c * d_hd + two * sn * c * s_hd + sn * sn * s_hs)
            };

            // Powell minimizes q(θ) over θ ∈ (0, 2π] by sampling IU = 49 equally
            // spaced angles, then a 3-point parabolic refinement (Fortran IU).
            const NTHETA: usize = 49;
            let pi = F::from_f64(core::f64::consts::PI).expect("π representable");
            let two_pi = pi + pi;
            let step = two_pi / F::from_f64(NTHETA as f64).expect("NTHETA representable");

            let q_at_zero = q_of(F::zero()); // q at d_{j-1}
            let mut best_theta = F::zero();
            let mut best_q = q_at_zero;
            for k in 1..=NTHETA {
                let theta = step * F::from_f64(k as f64).expect("k representable");
                let qv = q_of(theta);
                if qv < best_q {
                    best_q = qv;
                    best_theta = theta;
                }
            }
            // Parabolic refinement around the best grid node (concave-up guard).
            let qm = q_of(best_theta - step);
            let qp = q_of(best_theta + step);
            let denom = qm - (best_q + best_q) + qp;
            let theta_star = if denom > F::zero() {
                best_theta + half * (qm - qp) / denom * step
            } else {
                best_theta
            };
            let q_new = q_of(theta_star).min(best_q);

            // Segment decrease Q(x_opt + d_{j-1}) − Q(x_opt + d_j).
            let seg_drop = q_at_zero - q_new;
            if seg_drop <= F::zero() {
                break; // no boundary point improves on d_{j-1}
            }
            predicted_reduction = predicted_reduction + seg_drop;

            // Update d_j (eq. 5.16) and ∇Q(x_opt + d_j) (eq. 5.19).
            let c = theta_star.cos();
            let sn = theta_star.sin();
            for i in 0..n {
                d[i] = c * d[i] + sn * s[i];
            }
            for i in 0..n {
                g[i] = (F::one() - c) * g0[i] + c * gd[i] + sn * hs[i];
            }

            // Second truncation test of eq. 5.13.
            if seg_drop <= tol * predicted_reduction {
                break;
            }
        }

        TrustRegionStep {
            d,
            crvmin,
            predicted_reduction,
        }
    }
}

/// Plain dot product of two equal-length slices.
fn dot<F: Scalar>(a: &[F], b: &[F]) -> F {
    a.iter().zip(b).map(|(x, y)| *x * *y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::math::{DenseMatrix, SymmetricEigen};

    /// Build a `QuadraticModel` representing the pure quadratic
    /// `Q(x) = gᵀx + ½ xᵀ G x` exactly, with `x_opt = x0 = 0`.
    ///
    /// All implicit Hessian coefficients `γⱼ` are zero and every interpolation
    /// displacement is the origin, so `∇Q(x_opt) = g` and `∇²Q v = G v` exactly,
    /// and `eval_change(d) = Q(d) − Q(0)`. The factored-`H` blocks are unused by
    /// TRSAPP and are filled with zeros. `m = n + 2` is the minimal legal npt.
    fn pure_quadratic(g: &[f64], gmat: &DenseMatrix<f64>) -> QuadraticModel<f64> {
        let n = g.len();
        assert_eq!(gmat.nrows(), n);
        assert_eq!(gmat.ncols(), n);
        let m = n + 2;
        QuadraticModel::from_parts(
            n,
            m,
            vec![0.0; n],                                   // x0
            DenseMatrix::from_fn(m, n, |_, _| 0.0),         // xpt (all origin)
            vec![0.0; m],                                   // fval
            0,                                              // kopt → x_opt = x0
            g.to_vec(),                                     // gq = ∇Q(x0)
            gmat.clone(),                                   // Γ = G
            vec![0.0; m],                                   // γⱼ ≡ 0
            DenseMatrix::from_fn(n, m, |_, _| 0.0),         // Ξ (unused)
            DenseMatrix::from_fn(n, n, |_, _| 0.0),         // Υ (unused)
            DenseMatrix::from_fn(m, m - n - 1, |_, _| 0.0), // Z (unused)
            vec![1.0; m - n - 1],                           // signs (unused)
        )
    }

    fn diag(d: &[f64]) -> DenseMatrix<f64> {
        let n = d.len();
        DenseMatrix::from_fn(n, n, |i, j| if i == j { d[i] } else { 0.0 })
    }

    fn norm(v: &[f64]) -> f64 {
        dot(v, v).sqrt()
    }

    /// `B y`, with `B = self` (eigenvector matrix) and `y` length `ncols`.
    fn mat_vec(b: &DenseMatrix<f64>, y: &[f64]) -> Vec<f64> {
        (0..b.nrows())
            .map(|i| (0..b.ncols()).map(|j| b.get(i, j) * y[j]).sum())
            .collect()
    }

    /// `Bᵀ v`, projecting `v` onto the eigenbasis.
    fn mat_t_vec(b: &DenseMatrix<f64>, v: &[f64]) -> Vec<f64> {
        (0..b.ncols())
            .map(|j| (0..b.nrows()).map(|i| b.get(i, j) * v[i]).sum())
            .collect()
    }

    /// Exact trust-region solution for an SPD `G`: the Newton step `−G⁻¹g` if it
    /// is interior, else the Moré–Sorensen boundary point found by bisecting the
    /// secular equation `Σ ĝᵢ²/(λᵢ+μ)² = Δ²` in the eigenbasis. The reference
    /// oracle for tests (a), (b), (d).
    fn exact_tr_solve_spd(g: &[f64], gmat: &DenseMatrix<f64>, delta: f64) -> Vec<f64> {
        let (b, lambda) = gmat.try_eigh().expect("SPD eigendecomposition");
        let ghat = mat_t_vec(&b, g);
        // Newton step in the eigenbasis: yᵢ = −ĝᵢ/λᵢ.
        let y_newton: Vec<f64> = ghat.iter().zip(&lambda).map(|(gh, l)| -gh / l).collect();
        if norm(&y_newton) <= delta {
            return mat_vec(&b, &y_newton);
        }
        // Bisect μ ≥ 0 so that ‖y(μ)‖ = Δ, where yᵢ(μ) = −ĝᵢ/(λᵢ+μ).
        let phi = |mu: f64| -> f64 {
            let s: f64 = ghat
                .iter()
                .zip(&lambda)
                .map(|(gh, l)| {
                    let yi = gh / (l + mu);
                    yi * yi
                })
                .sum();
            s.sqrt() - delta
        };
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        while phi(hi) > 0.0 {
            hi *= 2.0;
        }
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if phi(mid) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let mu = 0.5 * (lo + hi);
        let y: Vec<f64> = ghat
            .iter()
            .zip(&lambda)
            .map(|(gh, l)| -gh / (l + mu))
            .collect();
        mat_vec(&b, &y)
    }

    /// (a) Strictly convex quadratic, large Δ → TRSAPP returns the interior
    /// Newton step `−G⁻¹g`, with CRVMIN a positive curvature estimate inside the
    /// spectrum and the predicted reduction matching the model.
    #[test]
    fn interior_newton_step_on_spd() {
        let gmat = DenseMatrix::from_row_slice(2, 2, &[3.0, 1.0, 1.0, 4.0]);
        let g = [2.0, -3.0];
        let model = pure_quadratic(&g, &gmat);
        let trs = model.trust_region_step(100.0);

        let dstar = exact_tr_solve_spd(&g, &gmat, 100.0); // = (-1, 1)
        assert!(norm(&[trs.d[0] - dstar[0], trs.d[1] - dstar[1]]) < 1e-9);
        assert!(norm(&trs.d) < 100.0);

        // CRVMIN is a Rayleigh quotient over the CG directions → strictly inside
        // the spectrum [λ_min, λ_max] (eq. 5.14); positive for SPD.
        let (_b, lambda) = gmat.try_eigh().unwrap();
        let lam_min = lambda.iter().cloned().fold(f64::INFINITY, f64::min);
        let lam_max = lambda.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(trs.crvmin > 0.0);
        assert!(trs.crvmin >= lam_min - 1e-9 && trs.crvmin <= lam_max + 1e-9);

        // predicted_reduction = Q(x_opt) − Q(x_opt + d) = −eval_change(d).
        assert!((trs.predicted_reduction + model.eval_change(&trs.d)).abs() < 1e-9);
        assert!(trs.predicted_reduction > 0.0);
    }

    /// (b) Same SPD quadratic with Δ smaller than the Newton-step length: the
    /// step lands exactly on the boundary, CRVMIN is reset to zero, and the step
    /// reaches the same model value as the exact constrained minimizer.
    #[test]
    fn boundary_step_on_spd() {
        let gmat = DenseMatrix::from_row_slice(2, 2, &[3.0, 1.0, 1.0, 4.0]);
        let g = [2.0, -3.0];
        let model = pure_quadratic(&g, &gmat);
        let delta = 0.5; // < ‖Newton step‖ = √2
        let trs = model.trust_region_step(delta);

        assert!((norm(&trs.d) - delta).abs() < 1e-9);
        assert_eq!(trs.crvmin, 0.0);
        assert!(trs.predicted_reduction > 0.0);
        assert!((trs.predicted_reduction + model.eval_change(&trs.d)).abs() < 1e-9);
        assert!(model.eval_change(&trs.d) < 0.0);

        // Matches the exact boundary minimizer's model value (SPD → TRSAPP is
        // optimal here up to the boundary sweep).
        let dstar = exact_tr_solve_spd(&g, &gmat, delta);
        assert!((model.eval_change(&trs.d) - model.eval_change(&dstar)).abs() < 1e-6);
    }

    /// (c) Indefinite `G` with a negative-curvature direction the CG path is
    /// steered into: the step runs to the boundary, CRVMIN is zero, and the
    /// model is strongly reduced.
    #[test]
    fn negative_curvature_goes_to_boundary() {
        let gmat = diag(&[2.0, -1.0]);
        let g = [0.1, 0.1];
        let model = pure_quadratic(&g, &gmat);
        let delta = 1.0;
        let trs = model.trust_region_step(delta);

        assert!((norm(&trs.d) - delta).abs() < 1e-9);
        assert_eq!(trs.crvmin, 0.0);
        assert!(trs.predicted_reduction > 0.0);
        assert!(model.eval_change(&trs.d) < 0.0);
        assert!((trs.predicted_reduction + model.eval_change(&trs.d)).abs() < 1e-9);
    }

    /// (d) Random-but-fixed SPD `G` (LCG-seeded), several Δ straddling the
    /// Newton-step length: TRSAPP feasible, achieves essentially the same model
    /// reduction as the exact dense trust-region solve, and never beats it.
    #[test]
    fn spd_matches_dense_trust_region() {
        let n = 4;
        // Deterministic SPD G = AᵀA + I via an LCG (mirrors update.rs's long_run).
        let mut seed: u64 = 0x2545F4914F6CDD1D;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / ((1u64 << 31) as f64) - 1.0 // in (−1, 1)
        };
        let a = DenseMatrix::from_fn(n, n, |_, _| next());
        let gmat = DenseMatrix::from_fn(n, n, |i, j| {
            let mut s = if i == j { 1.0 } else { 0.0 };
            for k in 0..n {
                s += a.get(k, i) * a.get(k, j);
            }
            s
        });
        let g: Vec<f64> = (0..n).map(|_| next()).collect();
        let model = pure_quadratic(&g, &gmat);

        for &delta in &[0.05, 0.2, 0.5, 1.0, 5.0] {
            let trs = model.trust_region_step(delta);
            let dstar = exact_tr_solve_spd(&g, &gmat, delta);

            assert!(norm(&trs.d) <= delta + 1e-9, "delta={delta}: feasible");
            let trs_red = -model.eval_change(&trs.d);
            let opt_red = -model.eval_change(&dstar);
            assert!(trs_red > 0.0, "delta={delta}: positive reduction");
            // Can't beat the exact minimizer (tiny numerical slack).
            assert!(
                trs_red <= opt_red + 1e-6,
                "delta={delta}: trs {trs_red} > opt {opt_red}"
            );
            // Near-optimal: truncated CG + boundary sweep recovers ≥ 99%.
            assert!(
                trs_red >= 0.99 * opt_red,
                "delta={delta}: trs {trs_red} < 0.99·opt {opt_red}"
            );
        }
    }

    /// (e) Degenerate guards: a zero gradient yields the zero step, and the
    /// 1-D case returns the exact Newton step (interior) or the boundary clamp.
    #[test]
    fn degenerate_and_one_dimensional() {
        // Zero gradient at x_opt → d = 0, no reduction.
        let model0 = pure_quadratic(&[0.0, 0.0], &diag(&[2.0, 3.0]));
        let trs0 = model0.trust_region_step(1.0);
        assert_eq!(trs0.d, vec![0.0, 0.0]);
        assert_eq!(trs0.crvmin, 0.0);
        assert_eq!(trs0.predicted_reduction, 0.0);

        // 1-D, interior: Q(x) = −4x + ½·2·x² ⇒ min at x = 2, inside Δ = 10.
        let model1 = pure_quadratic(&[-4.0], &diag(&[2.0]));
        let trs1 = model1.trust_region_step(10.0);
        assert!((trs1.d[0] - 2.0).abs() < 1e-12);
        assert!((trs1.crvmin - 2.0).abs() < 1e-12); // single curvature = G[0,0]

        // 1-D, boundary: same model, Δ = 0.5 < 2 ⇒ clamp to x = 0.5.
        let trs2 = model1.trust_region_step(0.5);
        assert!((trs2.d[0] - 0.5).abs() < 1e-12);
        assert_eq!(trs2.crvmin, 0.0);
    }

    /// (f) The accumulated predicted reduction (eq. 5.8) equals the model
    /// evaluator `−eval_change(d)` across interior, boundary, and indefinite runs.
    #[test]
    fn predicted_reduction_matches_model() {
        let cases: &[(Vec<f64>, DenseMatrix<f64>, f64)] = &[
            (
                vec![2.0, -3.0],
                DenseMatrix::from_row_slice(2, 2, &[3.0, 1.0, 1.0, 4.0]),
                100.0,
            ),
            (
                vec![2.0, -3.0],
                DenseMatrix::from_row_slice(2, 2, &[3.0, 1.0, 1.0, 4.0]),
                0.5,
            ),
            (vec![0.1, 0.1], diag(&[2.0, -1.0]), 1.0),
        ];
        for (g, gmat, delta) in cases {
            let model = pure_quadratic(g, gmat);
            let trs = model.trust_region_step(*delta);
            assert!((trs.predicted_reduction + model.eval_change(&trs.d)).abs() < 1e-9);
        }
    }
}
