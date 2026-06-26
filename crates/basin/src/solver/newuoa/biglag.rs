//! BIGLAG: the NEWUOA geometry-improving step (Powell 2006, §6).
//!
//! When the trust-region iterations leave an interpolation point `x_t` far from
//! `x_opt`, the model degrades. BIGLAG repairs it by choosing a replacement
//! `x⁺ = x_opt + d` that approximately solves
//!
//! ```text
//! maximize  |ℓ_t(x_opt + d)|   subject to  ‖d‖ ≤ Δ̄ ,
//! ```
//!
//! where `ℓ_t` is the `t`-th Lagrange function of the current model (eqs.
//! 6.1–6.6). Because `τ = ℓ_t(x_opt + d)` enters the update denominator
//! `σ = αβ + τ²` (eq. 6.5), a large `|ℓ_t|` keeps the least-Frobenius `H`-update
//! well-conditioned, which is the whole point of the geometry step.
//!
//! The method (eqs. 6.7–6.16) is the same 2-D circular search that TRSAPP uses
//! on the trust-region boundary ([`super::trsapp`], eqs. 5.16–5.19): each
//! iteration maximizes `|ℓ_t(x_opt + d(θ))|` over `d(θ) = cosθ d_{j-1} + sinθ
//! s_j` with `s_j ⟂ d_{j-1}`, `‖d_{j-1}‖ = ‖s_j‖ = Δ̄`. The Lagrange function's
//! gradient and Hessian come from its coefficients
//! ([`QuadraticModel::lagrange_coeffs`]) without ever forming `∇²ℓ_t` (eq. 6.15).
//!
//! # BIGDEN fallback
//!
//! Powell rejects BIGLAG's `d` when `|σ| < 0.8 τ²` (eq. 6.17) and falls back to
//! [`bigden`](QuadraticModel::bigden), which maximizes the denominator `|σ|`
//! directly. The driver ([`super::driver`]) applies this test on the
//! recomputed update scalars after each BIGLAG step and substitutes BIGDEN's `d`
//! when it fires: a rare event in practice (Powell 2006, §6).

use crate::core::math::Scalar;
use crate::solver::powell::QuadraticModel;

/// Result of one BIGLAG solve (Powell 2006, §6).
pub(crate) struct BiglagResult<F = f64> {
    /// The geometry step `d`, length `n`, expressed **relative to `x_opt`** (the
    /// new interpolation point is `x_opt + d`). Same convention as
    /// [`super::trsapp::TrsappResult::d`].
    pub(crate) d: Vec<F>,
    /// `τ = ℓ_t(x_opt + d)`: the Lagrange-function value at the new point. The
    /// driver does not screen it directly: it recomputes `τ`/`σ` from the §4
    /// update (`prepare_update`/`update_params`) and applies the eq. 6.17
    /// BIGDEN test there. This field is informational.
    pub(crate) tau: F,
}

impl<F: Scalar> QuadraticModel<F> {
    /// Approximately maximize `|ℓ_t(x_opt + d)|` over `‖d‖ ≤ delta_bar` by the
    /// 2-D circular search of Powell 2006, §6 (subroutine BIGLAG).
    ///
    /// `t` is the interpolation index to be replaced (chosen by the driver's
    /// Box-7 "furthest point" rule); `delta_bar` is the bound `Δ̄` of eq. 6.16.
    ///
    /// # Panics
    ///
    /// Panics unless `delta_bar > 0` and `t < m`.
    pub(crate) fn biglag(&self, t: usize, delta_bar: F) -> BiglagResult<F> {
        assert!(delta_bar > F::zero(), "biglag: delta_bar must be positive");
        assert!(t < self.m(), "biglag: t must index an interpolation point");

        let n = self.n();
        let half = F::from_f64(0.5).expect("0.5 representable");
        let two = F::from_f64(2.0).expect("2.0 representable");

        // ℓ_t coefficients: g = ∇ℓ_t(x0), λ the implicit-Hessian coefficients
        // (Powell 2006, eqs. 6.2, 6.15). ∇²ℓ_t·u = lagrange_hessian_matvec(λ, u).
        let (g_x0, lambda) = self.lagrange_coeffs(t);
        let xopt = self.xpt_row(self.kopt()).to_vec();
        // ℓ_t(x_opt) = δ_(opt)t by the Lagrange conditions (eq. 6.1).
        let base = if t == self.kopt() {
            F::one()
        } else {
            F::zero()
        };
        // ∇ℓ_t(x_opt) = ∇ℓ_t(x0) + ∇²ℓ_t (x_opt − x0) (eq. 6.11 differentiated).
        let h_xopt = self.lagrange_hessian_matvec(&lambda, &xopt);
        let grad_opt: Vec<F> = (0..n).map(|i| g_x0[i] + h_xopt[i]).collect();

        // ℓ_t(x_opt + d) via eq. 6.11, shifted by the constant ℓ_t(x_opt).
        let ell = |d: &[F]| -> F {
            let hd = self.lagrange_hessian_matvec(&lambda, d);
            base + dot(d, &grad_opt) + half * dot(d, &hd)
        };

        // ---- d_0 (eq. 6.9): ±Δ̄ (x_t − x_opt)/‖x_t − x_opt‖, sign for larger |ℓ|.
        let xt = self.xpt_row(t).to_vec();
        let mut u0: Vec<F> = (0..n).map(|i| xt[i] - xopt[i]).collect();
        let mut u0_norm = norm(&u0);
        if u0_norm <= F::zero() {
            // x_t coincides with x_opt (the driver never picks t = opt, but guard
            // anyway): fall back to the Lagrange gradient as the search ray.
            u0 = grad_opt.clone();
            u0_norm = norm(&u0);
            if u0_norm <= F::zero() {
                return BiglagResult {
                    d: vec![F::zero(); n],
                    tau: base,
                };
            }
        }
        let scale0 = delta_bar / u0_norm;
        let dpos: Vec<F> = (0..n).map(|i| scale0 * u0[i]).collect();
        let ell_pos = ell(&dpos);
        let ell_neg = ell(&dpos.iter().map(|x| -*x).collect::<Vec<_>>());
        let mut d: Vec<F> = if ell_pos.abs() >= ell_neg.abs() {
            dpos
        } else {
            dpos.iter().map(|x| -*x).collect()
        };
        let mut ell_d = ell(&d); // ℓ_t(x_opt + d), signed
        // ∇ℓ_t(x_opt + d) = ∇ℓ_t(x_opt) + ∇²ℓ_t d.
        let hd0 = self.lagrange_hessian_matvec(&lambda, &d);
        let mut grad_cur: Vec<F> = (0..n).map(|i| grad_opt[i] + hd0[i]).collect();

        let delta_sq = delta_bar * delta_bar;
        // Near-degenerate cutoff for s_j ‖orthogonal component‖² (eq. 6.12):
        // the orthogonal part vanishes ⇔ ∇ℓ_t ∥ d, and BIGLAG returns d_{j-1}.
        let degen = F::from_f64(1e-8).expect("1e-8 representable");

        // ---- Iterations j = 1..n (eqs. 6.7–6.14). ----
        for j in 0..n {
            let dd = dot(&d, &d); // ≈ Δ̄²

            // Choose the gradient that spans the 2-D subspace with d_{j-1}.
            // j = 0 (the first iteration) may use ∇ℓ_t(x_opt) instead of
            // ∇ℓ_t(x_opt + d_0) when both eq. 6.10 conditions hold.
            let gsrc: &[F] = if j == 0 {
                let dg = dot(&d, &grad_opt);
                let gg = dot(&grad_opt, &grad_opt);
                let cond_i = dg * dg <= from::<F>(0.99) * delta_sq * gg; // (6.10a)
                let cond_ii = gg.sqrt() * delta_bar >= from::<F>(0.1) * ell_d.abs(); // (6.10b)
                if cond_i && cond_ii {
                    &grad_opt
                } else {
                    &grad_cur
                }
            } else {
                &grad_cur
            };

            // s_raw = gsrc − (dᵀgsrc/dd) d  (the component of gsrc ⟂ d, eq. 6.8).
            let coef = dot(&d, gsrc) / dd;
            let s_raw: Vec<F> = (0..n).map(|i| gsrc[i] - coef * d[i]).collect();
            let s_raw_sq = dot(&s_raw, &s_raw);
            let gsrc_sq = dot(gsrc, gsrc);
            if s_raw_sq <= degen * gsrc_sq {
                break; // (6.12): ∇ℓ_t ∥ d, no useful search direction.
            }

            // Rescale s ⟂ d, ‖s‖ = Δ̄.
            let s_scale = delta_bar / s_raw_sq.sqrt();
            let s: Vec<F> = (0..n).map(|i| s_scale * s_raw[i]).collect();
            let hs = self.lagrange_hessian_matvec(&lambda, &s);

            // ℓ_t(x_opt + d(θ)) with d(θ) = cosθ d + sinθ s. Using
            // ∇²ℓ_t d = ∇ℓ_t(x_opt + d) − ∇ℓ_t(x_opt) = grad_cur − grad_opt:
            let hd: Vec<F> = (0..n).map(|i| grad_cur[i] - grad_opt[i]).collect();
            let d_g0 = dot(&d, &grad_opt);
            let s_g0 = dot(&s, &grad_opt);
            let d_hd = dot(&d, &hd);
            let s_hd = dot(&s, &hd); // = dᵀ∇²ℓ_t s by symmetry
            let s_hs = dot(&s, &hs);
            let ell_of = |theta: F| -> F {
                let c = theta.cos();
                let sn = theta.sin();
                base + c * d_g0
                    + sn * s_g0
                    + half * (c * c * d_hd + two * sn * c * s_hd + sn * sn * s_hs)
            };

            // Maximize |ℓ_t(x_opt + d(θ))| over θ ∈ [0, 2π): the θ = 0 incumbent
            // plus NTHETA = 49 interior nodes on a 2π/(NTHETA+1) grid (50 distinct
            // angles, none at 2π ≡ 0, matching PRIMA's biglag.f `TWOPI/(IU+1)`),
            // then a 3-point parabolic refinement, mirroring TRSAPP's boundary
            // sweep (trsapp.rs) but on the modulus rather than the model value.
            const NTHETA: usize = 49;
            let pi = F::from_f64(core::f64::consts::PI).expect("π representable");
            let step =
                (pi + pi) / F::from_f64((NTHETA + 1) as f64).expect("NTHETA+1 representable");
            let mut best_theta = F::zero();
            let mut best_abs = ell_of(F::zero()).abs(); // θ = 0 ⇒ d_{j-1}
            for k in 1..=NTHETA {
                let theta = step * F::from_f64(k as f64).expect("k representable");
                let av = ell_of(theta).abs();
                if av > best_abs {
                    best_abs = av;
                    best_theta = theta;
                }
            }
            // Parabolic refinement around the best node (concave-down guard).
            let am = ell_of(best_theta - step).abs();
            let ap = ell_of(best_theta + step).abs();
            let denom = am - (best_abs + best_abs) + ap;
            let theta_star = if denom < F::zero() {
                best_theta + half * (am - ap) / denom * step
            } else {
                best_theta
            };

            // Commit d_j = cosθ* d + sinθ* s and its Lagrange value. The refined
            // angle is committed unconditionally (no `max(best_abs)` guard, unlike
            // TRSAPP's defensive sweep), faithful to PRIMA's biglag.f, where the
            // |ℓ| no-decrease guarantee rests on the θ = 0 seed being reachable and
            // the parabola being seeded at the grid maximum.
            let c = theta_star.cos();
            let sn = theta_star.sin();
            let d_new: Vec<F> = (0..n).map(|i| c * d[i] + sn * s[i]).collect();
            let ell_new = ell_of(theta_star);
            let ell_prev_abs = ell_d.abs();
            d = d_new;
            ell_d = ell_new;

            // ∇ℓ_t(x_opt + d_j) via the recurrence (eq. 6.14).
            grad_cur = (0..n)
                .map(|i| (F::one() - c) * grad_opt[i] + c * grad_cur[i] + sn * hs[i])
                .collect();

            // Termination (6.13): the iteration barely improved |ℓ_t|.
            let onep1 = from::<F>(1.1);
            if ell_d.abs() <= onep1 * ell_prev_abs {
                break;
            }
        }

        BiglagResult { d, tau: ell_d }
    }
}

/// Build an `F` from an `f64` literal (the constants are exactly representable).
fn from<F: Scalar>(x: f64) -> F {
    F::from_f64(x).expect("constant representable")
}

/// Plain dot product of two equal-length slices.
fn dot<F: Scalar>(a: &[F], b: &[F]) -> F {
    a.iter().zip(b).map(|(x, y)| *x * *y).sum()
}

/// Euclidean norm of `v`.
fn norm<F: Scalar>(v: &[F]) -> F {
    dot(v, v).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::powell::QuadraticModel;

    /// A smooth test objective whose model `initialize` can interpolate.
    fn quad(x: &[f64]) -> f64 {
        // Q(x) = (x0−1)² + 2(x1+2)² − x0 x1, a generic indefinite-free quadratic.
        (x[0] - 1.0).powi(2) + 2.0 * (x[1] + 2.0).powi(2) - x[0] * x[1]
    }

    fn rosen(x: &[f64]) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
    }

    /// Evaluate ℓ_t at an absolute interpolation point x_i using the
    /// coefficients, to check the Lagrange conditions ℓ_t(x_i) = δ_it (eq. 6.1).
    fn ell_at_point<F: crate::core::math::Scalar>(
        model: &QuadraticModel<F>,
        t: usize,
        i: usize,
    ) -> F {
        let (g_x0, lambda) = model.lagrange_coeffs(t);
        // ℓ_t(x0 + s) − ℓ_t(x0) = sᵀg + ½ sᵀ∇²ℓ_t s with s = x_i − x0.
        let s = model.xpt_row(i).to_vec();
        let hs = model.lagrange_hessian_matvec(&lambda, &s);
        let half = F::from_f64(0.5).unwrap();
        // Reference to point opt: ℓ_t(x_i) − ℓ_t(x_opt). With ℓ_t(x_opt)=δ_(opt)t.
        let base = if t == model.kopt() {
            F::one()
        } else {
            F::zero()
        };
        let xopt = model.xpt_row(model.kopt()).to_vec();
        let hxopt = model.lagrange_hessian_matvec(&lambda, &xopt);
        let val_i = dot(&s, &g_x0) + half * dot(&s, &hs);
        let val_opt = dot(&xopt, &g_x0) + half * dot(&xopt, &hxopt);
        base + val_i - val_opt
    }

    /// The coefficients from `lagrange_coeffs` must reproduce the defining
    /// Lagrange conditions `ℓ_t(x_i) = δ_it` at every interpolation point.
    #[test]
    fn lagrange_conditions_hold() {
        let model = QuadraticModel::initialize(vec![0.3, -0.7], 0.5, 5, &quad);
        let m = model.m();
        for t in 0..m {
            for i in 0..m {
                let want = if i == t { 1.0 } else { 0.0 };
                let got = ell_at_point(&model, t, i);
                assert!(
                    (got - want).abs() < 1e-9,
                    "ℓ_{t}(x_{i}) = {got}, want {want}"
                );
            }
        }
    }

    /// BIGLAG must (a) respect the bound ‖d‖ ≤ Δ̄ with the bound essentially
    /// active, and (b) not decrease |ℓ_t| below the initial ray d_0 (eq. 6.9).
    #[test]
    fn improves_over_initial_ray_and_respects_bound() {
        let model = QuadraticModel::initialize(vec![-1.2, 1.0], 0.5, 5, &rosen);
        let kopt = model.kopt();
        let delta_bar = 0.4;
        // Pick the interpolation point furthest from x_opt (the driver's Box 7).
        let xopt = model.xpt_row(kopt).to_vec();
        let (t, _) = (0..model.m())
            .map(|j| {
                let r = model.xpt_row(j);
                let dist: f64 = (0..2).map(|i| (r[i] - xopt[i]).powi(2)).sum::<f64>().sqrt();
                (j, dist)
            })
            .filter(|(j, _)| *j != kopt)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        // |ℓ_t| on the raw initial ray, for the improvement comparison.
        let (g_x0, lambda) = model.lagrange_coeffs(t);
        let h_xopt = model.lagrange_hessian_matvec(&lambda, &xopt);
        let grad_opt: Vec<f64> = (0..2).map(|i| g_x0[i] + h_xopt[i]).collect();
        let xt = model.xpt_row(t).to_vec();
        let mut u0: Vec<f64> = (0..2).map(|i| xt[i] - xopt[i]).collect();
        let nrm: f64 = u0.iter().map(|x| x * x).sum::<f64>().sqrt();
        for v in &mut u0 {
            *v *= delta_bar / nrm;
        }
        let ell_d0 = {
            let hd = model.lagrange_hessian_matvec(&lambda, &u0);
            dot(&u0, &grad_opt) + 0.5 * dot(&u0, &hd)
        };

        let res = model.biglag(t, delta_bar);
        let dnorm: f64 = res.d.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(dnorm <= delta_bar * (1.0 + 1e-9), "‖d‖ = {dnorm} > Δ̄");
        assert!(
            dnorm >= 0.99 * delta_bar,
            "‖d‖ = {dnorm} ≪ Δ̄ (bound inactive)"
        );
        assert!(
            res.tau.abs() >= ell_d0.abs() - 1e-12,
            "|ℓ_t| not improved: {} < {}",
            res.tau.abs(),
            ell_d0.abs()
        );
    }

    /// BIGLAG must find a `|ℓ_t|` close to the brute-force maximum over the
    /// trust-region disk (it should not leave a much better point on the table).
    #[test]
    fn near_optimal_vs_brute_force() {
        let model = QuadraticModel::initialize(vec![0.4, -0.6, 0.2], 0.5, 7, &|x: &[f64]| {
            (0..3)
                .map(|i| (x[i] - 0.5 * (i as f64)).powi(2))
                .sum::<f64>()
                + 0.3 * x[0] * x[1]
                - 0.2 * x[1] * x[2]
        });
        let n = 3;
        let kopt = model.kopt();
        let xopt = model.xpt_row(kopt).to_vec();
        let delta_bar = 0.3;
        for t in 0..model.m() {
            if t == kopt {
                continue;
            }
            let (g_x0, lambda) = model.lagrange_coeffs(t);
            let h_xopt = model.lagrange_hessian_matvec(&lambda, &xopt);
            let grad_opt: Vec<f64> = (0..n).map(|i| g_x0[i] + h_xopt[i]).collect();
            let ell = |d: &[f64]| -> f64 {
                let hd = model.lagrange_hessian_matvec(&lambda, d);
                dot(d, &grad_opt) + 0.5 * dot(d, &hd)
            };
            // Brute force over a spherical grid of radius delta_bar.
            let mut brute = 0.0_f64;
            let steps = 40;
            for a in 0..steps {
                let theta = std::f64::consts::PI * (a as f64) / (steps as f64);
                for b in 0..(2 * steps) {
                    let phi = std::f64::consts::PI * (b as f64) / (steps as f64);
                    let d = [
                        delta_bar * theta.sin() * phi.cos(),
                        delta_bar * theta.sin() * phi.sin(),
                        delta_bar * theta.cos(),
                    ];
                    brute = brute.max(ell(&d).abs());
                }
            }
            let res = model.biglag(t, delta_bar);
            assert!(
                res.tau.abs() >= 0.9 * brute,
                "t={t}: biglag |ℓ_t|={} < 0.9 * brute {brute}",
                res.tau.abs()
            );
        }
    }

    /// After BIGLAG, swapping x_t for x_opt + d gives a non-negligible update
    /// denominator σ (the geometry step's purpose: keep the H-update well-posed).
    #[test]
    fn produces_well_conditioned_replacement() {
        let model = QuadraticModel::initialize(vec![-1.2, 1.0], 0.5, 5, &rosen);
        let kopt = model.kopt();
        let xopt = model.xpt_row(kopt).to_vec();
        let (t, _) = (0..model.m())
            .filter(|j| *j != kopt)
            .map(|j| {
                let r = model.xpt_row(j);
                let dist: f64 = (0..2).map(|i| (r[i] - xopt[i]).powi(2)).sum::<f64>().sqrt();
                (j, dist)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        let res = model.biglag(t, 0.4);
        let xnew: Vec<f64> = (0..2).map(|i| xopt[i] + res.d[i]).collect();
        let ctx = model.prepare_update(&xnew);
        let sc = model.update_params(t, &ctx);
        assert!(sc.sigma.abs() > 1e-6, "σ = {} too small", sc.sigma);
    }
}
