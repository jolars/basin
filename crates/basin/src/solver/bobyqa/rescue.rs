//! The method of RESCUE (BOBYQA, Powell 2009, §5).
//!
//! When rounding errors damage the denominator `σ` of the least-Frobenius-norm
//! `H`-update (eq. 4.9), the interpolation set's geometry has degraded.
//! RESCUE replaces a few interpolation points by fresh ones to restore the
//! conditioning of the interpolation system. It:
//!
//! 1. recentres the origin on `x_opt` ([`QuadraticModel::shift_origin`]);
//! 2. lays down a set of *provisional* interpolation points around `x_opt`
//!    (coordinate steps of length `δ`, clipped to the bounds), represented
//!    implicitly by `ptsid` + `ptsaux`, and rebuilds `H` (`Ξ`/`Υ`/`Ω`) for that
//!    provisional set with all `Ω`-signs `+1`;
//! 3. reinstates as many *original* points as can safely replace a provisional
//!    point (the denominator test of eq. 5.7), updating `H` via [`updateh_rsc`];
//! 4. evaluates `F` at the genuinely-new provisional points that remain and
//!    rebuilds the quadratic model `∇Q` / `Γ` / `γ` by Lagrange corrections.
//!
//! Ported from PRIMA v0.7.2 `fortran/bobyqa/rescue.f90` (the `rescue` and
//! `updateh_rsc` subroutines), translated into basin's [`QuadraticModel`]
//! layout. Two deliberate departures from the literal Fortran, both forced by
//! basin's representation rather than by choice:
//!
//! - **The gradient `gq` stays anchored at `x0`.** PRIMA's `gopt` is `∇Q(x_opt)`
//!   and its final step (rescue.f90:559–561) re-anchors `gopt` when `kopt`
//!   moves. basin stores `∇Q(x0)` and derives `gradient_at_opt()` on demand, so
//!   that re-anchoring is omitted; after the §5 origin shift `x0 = x_opt`, so the
//!   incremental `gq += moderr·Ξ_kpt` updates are exactly right for `∇Q(x0)`.
//! - **No `maxfun` gate.** PRIMA aborts RESCUE when the evaluation budget is
//!   exhausted; basin enforces budgets framework-side (the `Executor`'s
//!   `MaxCostEvals`), so RESCUE evaluates the handful of new points it needs and
//!   lets the executor stop afterward, like [`super::init`] does at start-up.

use crate::core::math::Scalar;
use crate::solver::powell::QuadraticModel;

use super::init::{extra_pair, xinbd};

/// Decode PRIMA's packed `ptsid` value into the 1-based coordinate indices
/// `(ip, iq)`; `0` means "absent" (rescue.f90:358–359). With `ip = ⌊ptsid⌋` and
/// `iq = ⌊(n+1)·ptsid − (n+1)·ip⌋`, the provisional step is
/// `ptsaux(1,ip)·e_ip + ptsaux(1,iq)·e_iq` (both present), or a single
/// `ptsaux(1,ip)·e_ip` / `ptsaux(2,iq)·e_iq` (only `ip` / only `iq`).
fn decode_ptsid<F: Scalar>(p: F, n: usize) -> (usize, usize) {
    let np1 = F::from_usize(n + 1).expect("n+1 representable");
    let ip_f = p.floor();
    let ip = ip_f.to_usize().unwrap_or(0);
    let iq_f = (np1 * p - np1 * ip_f).floor();
    let iq = iq_f.to_usize().unwrap_or(0);
    (ip, iq)
}

/// Plain dot product of two equal-length slices.
fn dot<F: Scalar>(a: &[F], b: &[F]) -> F {
    a.iter().zip(b).map(|(x, y)| *x * *y).sum()
}

/// Run RESCUE (Powell 2009, §5) on `model`, mutating it (interpolation set,
/// `H`-factorization, quadratic model, `kopt`, `x0`) and the shifted bounds
/// `sl`/`su` in place. `lower`/`upper` are the absolute box `[a, b]`, `delta`
/// the current trust-region radius; `eval` evaluates `F` at the genuinely-new
/// provisional points (its first `Err` bubbles), and each `(x_abs, F)` it
/// produces is appended to `evaluated` for the driver to fold into its best.
// The shifted bounds (`sl`/`su`), absolute bounds (`lower`/`upper`), `delta`,
// `eval`, and `evaluated` are distinct disjoint borrows of `BobyqaWork` fields
// at the call site; bundling them into a struct would fight the borrow checker.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rescue<F: Scalar, E>(
    model: &mut QuadraticModel<F>,
    sl: &mut [F],
    su: &mut [F],
    lower: &[F],
    upper: &[F],
    delta: F,
    eval: &mut impl FnMut(&[F]) -> Result<F, E>,
    evaluated: &mut Vec<(Vec<F>, F)>,
) -> Result<(), E> {
    let n = model.n();
    let m = model.m();
    let rank = m - n - 1;
    let zero = F::zero();
    let one = F::one();
    let half = F::from_f64(0.5).expect("0.5 representable");
    let two = F::from_f64(2.0).expect("2.0 representable");
    let np1 = F::from_usize(n + 1).expect("n+1 representable");

    // --- Prologue: shift the origin onto x_opt (rescue.f90:233–244). ---
    let kopt0 = model.kopt();
    let xopt = model.xpt_row(kopt0).to_vec();
    // The shifted bounds move with the new origin (= old x_opt).
    for i in 0..n {
        sl[i] = (sl[i] - xopt[i]).min(zero);
        su[i] = (su[i] - xopt[i]).max(zero);
    }
    // shift_origin recentres x0/xpt/gq/Γ (γ unchanged); its Ξ/Υ work is
    // overwritten below by the provisional H. Equivalent to PRIMA's HQ r2update
    // prologue, and additionally re-anchors gq to the new x0 (= x_opt).
    model.shift_origin();
    // Clamp x0 into the box (PRIMA: xbase = min(max(xl, xbase+xopt), xu)); a
    // no-op in exact arithmetic since x0 + xopt stays feasible.
    for i in 0..n {
        model.x0[i] = lower[i].max(upper[i].min(model.x0[i]));
    }
    // x_opt is now the origin; force its displacement to exactly zero.
    for i in 0..n {
        model.xpt.set(kopt0, i, zero);
    }

    // --- PTSAUX: the two provisional step lengths per coordinate (5.4–5.5;
    // rescue.f90:247–252). ptsaux[j] = [ptsaux(1,j+1), ptsaux(2,j+1)]. ---
    let ptsaux: Vec<[F; 2]> = (0..n)
        .map(|j| {
            let mut a1 = delta.min(su[j]);
            let mut a2 = (-delta).max(sl[j]);
            if a1 + a2 < zero {
                core::mem::swap(&mut a1, &mut a2);
            }
            if a2.abs() < half * a1.abs() {
                a2 = half * a1;
            }
            [a1, a2]
        })
        .collect();

    // --- Provisional H from the coordinate-direction points (rescue.f90:254–276).
    // Identifiers go into `ptsid`; the nonzero Ξ/Υ/Ω entries are written below,
    // with all Ω-signs forced to +1. ---
    let sfrac = half / np1;
    let mut ptsid = vec![zero; m];
    ptsid[0] = sfrac;
    for i in 0..n {
        for j in 0..m {
            model.bmat_xi.set(i, j, zero);
        }
        for j in 0..n {
            model.bmat_ups.set(i, j, zero);
        }
    }
    for i in 0..m {
        for j in 0..rank {
            model.zmat.set(i, j, zero);
        }
    }
    for k in 0..rank {
        model.zsign[k] = one;
    }
    for k in 1..=n {
        let c = k - 1; // 0-based coordinate
        let kf = F::from_usize(k).expect("k representable");
        ptsid[k] = kf + sfrac;
        if k < m - n {
            ptsid[k + n] = kf / np1 + sfrac;
            let temp = one / (ptsaux[c][0] - ptsaux[c][1]);
            model.bmat_xi.set(c, k, -temp + one / ptsaux[c][0]);
            model.bmat_xi.set(c, k + n, temp + one / ptsaux[c][1]);
            let b0 = -model.bmat_xi.get(c, k) - model.bmat_xi.get(c, k + n);
            model.bmat_xi.set(c, 0, b0);
            let z0 = two.sqrt() / (ptsaux[c][0] * ptsaux[c][1]).abs();
            model.zmat.set(0, c, z0);
            model.zmat.set(k, c, z0 * ptsaux[c][1] * temp);
            model.zmat.set(k + n, c, -z0 * ptsaux[c][0] * temp);
        } else {
            // Unreachable for m >= 2n+1 (basin's only supported regime); ported
            // from rescue.f90:271–275 for completeness.
            model.bmat_xi.set(c, 0, -one / ptsaux[c][0]);
            model.bmat_xi.set(c, k, one / ptsaux[c][0]);
            model
                .bmat_ups
                .set(c, c, -half * ptsaux[c][0] * ptsaux[c][0]);
        }
    }

    // --- Provisional H from the two-coordinate points (rescue.f90:278–287),
    // reusing `extra_pair` as PRIMA's `setij`. ---
    for t in 0..(m - 2 * n - 1) {
        let slot = 2 * n + 1 + t; // 0-based provisional point
        let (p, q) = extra_pair(t, n); // 0-based coords; PRIMA ip=p+1, iq=q+1
        let ipf = F::from_usize(p + 1).expect("ip representable");
        let iqf = F::from_usize(q + 1).expect("iq representable");
        ptsid[slot] = ipf + iqf / np1 + sfrac;
        let temp = one / (ptsaux[p][0] * ptsaux[q][0]);
        let col = slot - n - 1; // 0-based Ω column for this extra point
        model.zmat.set(0, col, temp);
        model.zmat.set(slot, col, temp);
        model.zmat.set(p + 1, col, -temp);
        model.zmat.set(q + 1, col, -temp);
    }

    // --- Reinstate x_opt at provisional slot kopt by exchanging slots 1 and
    // kopt; mark slot kopt as taken (rescue.f90:289–297). ---
    if kopt0 != 0 {
        for i in 0..n {
            let a = model.bmat_xi.get(i, 0);
            let b = model.bmat_xi.get(i, kopt0);
            model.bmat_xi.set(i, 0, b);
            model.bmat_xi.set(i, kopt0, a);
        }
        for j in 0..rank {
            let a = model.zmat.get(0, j);
            let b = model.zmat.get(kopt0, j);
            model.zmat.set(0, j, b);
            model.zmat.set(kopt0, j, a);
        }
    }
    ptsid[0] = ptsid[kopt0];
    ptsid[kopt0] = zero;

    // --- Scores: distance of each original point from x_opt (rescue.f90:299–312).
    // The paper uses the distance (not its square); SCORE(KOPT)=0 skips x_opt. ---
    let mut score: Vec<F> = (0..m)
        .map(|k| model.xpt_row(k).iter().fold(zero, |a, &v| a + v * v).sqrt())
        .collect();
    score[kopt0] = zero;
    let scoreinc = score.iter().fold(zero, |a, &v| a.max(v));
    let mut nprov = m - 1;

    // --- The reinstatement loop (rescue.f90:326–447). Bounded by m² iterations;
    // retains at least one provisional point. ---
    let c005 = F::from_f64(5e-2).expect("0.05 representable");
    let max_iter = m * m;
    for _ in 0..max_iter {
        if score.iter().all(|&s| s <= zero) || nprov <= 1 {
            break;
        }

        // KORIG: the closest original point not yet processed (smallest positive
        // score).
        let mut korig = 0;
        let mut best = F::infinity();
        for (k, &s) in score.iter().enumerate() {
            if s > zero && s < best {
                best = s;
                korig = k;
            }
        }

        // WMV = w(XNEW) − w(XOPT) for XNEW = XPT(:,KORIG), XOPT = 0
        // (rescue.f90:352–373). Provisional points are read through ptsid/ptsaux.
        let xkorig = model.xpt_row(korig).to_vec();
        let mut wmv = vec![zero; m + n];
        for (k, w) in wmv.iter_mut().enumerate().take(m) {
            let val = if k == kopt0 {
                zero
            } else if ptsid[k] <= zero {
                dot(&xkorig, model.xpt_row(k))
            } else {
                let (ip, iq) = decode_ptsid(ptsid[k], n);
                if ip > 0 && iq > 0 {
                    xkorig[ip - 1] * ptsaux[ip - 1][0] + xkorig[iq - 1] * ptsaux[iq - 1][0]
                } else if ip > 0 {
                    xkorig[ip - 1] * ptsaux[ip - 1][0]
                } else if iq > 0 {
                    xkorig[iq - 1] * ptsaux[iq - 1][1]
                } else {
                    zero
                }
            };
            *w = half * val * val;
        }
        wmv[m..(m + n)].copy_from_slice(&xkorig);

        // VLAG = H·WMV + e_KOPT (rescue.f90:375–388); β (4.10) of BOBYQA paper.
        let (mut vlag_lambda, vlag_g) = model.apply_h(&wmv[..m], &wmv[m..]);
        vlag_lambda[kopt0] = vlag_lambda[kopt0] + one;
        // bsum = wmv(1:n) · (Ξ·wmv_λ + vlag_g), with vlag_g = Ξ·wmv_λ + Υ·wmv_g.
        let mut xi_wmv = vec![zero; n];
        for (i, x) in xi_wmv.iter_mut().enumerate() {
            let mut acc = zero;
            for j in 0..m {
                acc = acc + model.bmat_xi.get(i, j) * wmv[j];
            }
            *x = acc;
        }
        // wᵀHw = ‖Z'wmv_λ‖² + wmv_g·(2·Ξ·wmv_λ + Υ·wmv_g), the cross/Υ terms
        // being `bsum`. N.B. basin contracts against the g-part `wmv_g`
        // (`wmv[m..]`), the exact form per BOBYQA eq. 4.12 and the PRIMA comment's
        // own derivation; PRIMA's rescue.f90:384 contracts against `wmv(1:n)`
        // (the first n λ-entries), which only meets its loose internal 1e-2
        // H-error check. The g-part form makes the rebuilt H equal inv(W) to
        // working precision (asserted by `assert_h_matches_inverse` in tests).
        let mut bsum = zero;
        for i in 0..n {
            bsum = bsum + wmv[m + i] * (xi_wmv[i] + vlag_g[i]);
        }
        let mut zsum = zero;
        for j in 0..rank {
            let mut zw = zero;
            for i in 0..m {
                zw = zw + model.zmat.get(i, j) * wmv[i];
            }
            zsum = zsum + zw * zw;
        }
        let norm_sq = xkorig.iter().fold(zero, |a, &v| a + v * v);
        let beta = half * norm_sq * norm_sq - zsum - bsum;

        // DEN(k) = σ for replacing provisional point k (rescue.f90:392–394).
        let mut den = vec![zero; m];
        for (k, d) in den.iter_mut().enumerate() {
            if ptsid[k] > zero {
                let mut hdiag = zero;
                for j in 0..rank {
                    let z = model.zmat.get(k, j);
                    hdiag = hdiag + z * z;
                }
                *d = hdiag * beta + vlag_lambda[k] * vlag_lambda[k];
            }
        }

        // Denominator test (rescue.f90:414–419). If no provisional point can be
        // safely replaced, lower KORIG's priority and retry later.
        let sum_abs = vlag_lambda
            .iter()
            .chain(vlag_g.iter())
            .fold(zero, |a, &v| a + v.abs());
        let maxsq = vlag_lambda.iter().fold(zero, |a, &v| a.max(v * v));
        let any_good = den.iter().any(|&d| d > c005 * maxsq);
        if !(sum_abs.is_finite() && any_good) {
            score[korig] = -score[korig] - scoreinc;
            continue;
        }

        // KPROV: the provisional point with the largest (finite) denominator.
        let mut kprov = 0;
        let mut bestden = F::neg_infinity();
        for (k, &d) in den.iter().enumerate() {
            if !d.is_nan() && d > bestden {
                bestden = d;
                kprov = k;
            }
        }

        // Exchange provisional points KPROV and KORIG, then replace the (now
        // KORIG-th) provisional point by the KORIG-th original point
        // (rescue.f90:423–446).
        if kprov != korig {
            for i in 0..n {
                let a = model.bmat_xi.get(i, kprov);
                let b = model.bmat_xi.get(i, korig);
                model.bmat_xi.set(i, kprov, b);
                model.bmat_xi.set(i, korig, a);
            }
            for j in 0..rank {
                let a = model.zmat.get(kprov, j);
                let b = model.zmat.get(korig, j);
                model.zmat.set(kprov, j, b);
                model.zmat.set(korig, j, a);
            }
            vlag_lambda.swap(kprov, korig);
        }
        ptsid[kprov] = ptsid[korig];
        ptsid[korig] = zero;
        score[korig] = zero;
        for s in score.iter_mut() {
            *s = s.abs();
        }

        let mut vlag_full = vlag_lambda;
        vlag_full.extend_from_slice(&vlag_g);
        updateh_rsc(model, korig, beta, &mut vlag_full);
        nprov -= 1;
    }

    // --- Re-evaluate the remaining provisional (genuinely new) points and
    // rebuild the quadratic model (rescue.f90:455–561). ---
    let kbase = model.kopt();
    let fbase = model.fval[kbase];
    if nprov > 0 {
        for kpt in 0..m {
            if ptsid[kpt] <= zero {
                continue;
            }

            // Absorb γ_kpt into the explicit Hessian, then zero it (Phase B).
            let g_kpt = model.gamma[kpt];
            if g_kpt != zero {
                let row = model.xpt_row(kpt).to_vec();
                for i in 0..n {
                    for j in 0..n {
                        let add = g_kpt * row[i] * row[j];
                        model
                            .gamma_explicit
                            .set(i, j, model.gamma_explicit.get(i, j) + add);
                    }
                }
            }
            model.gamma[kpt] = zero;

            // Materialize the new provisional point (≤ 2 nonzeros).
            let (ip, iq) = decode_ptsid(ptsid[kpt], n);
            for i in 0..n {
                model.xpt.set(kpt, i, zero);
            }
            let mut xp = zero;
            let mut xq = zero;
            if ip > 0 && iq > 0 {
                xp = ptsaux[ip - 1][0];
                model.xpt.set(kpt, ip - 1, xp);
                xq = ptsaux[iq - 1][0];
                model.xpt.set(kpt, iq - 1, xq);
            } else if ip > 0 {
                xp = ptsaux[ip - 1][0];
                model.xpt.set(kpt, ip - 1, xp);
            } else if iq > 0 {
                xq = ptsaux[iq - 1][1];
                model.xpt.set(kpt, iq - 1, xq);
            }

            // Evaluate F at the new point.
            let disp = model.xpt_row(kpt).to_vec();
            let xabs = xinbd(&model.x0, &disp, lower, upper, sl, su);
            let f = eval(&xabs)?;
            evaluated.push((xabs, f));
            model.fval[kpt] = f;
            if f < model.fval[model.kopt] {
                model.kopt = kpt;
            }

            // Model value VQUAD at the new point (its ≤ 2 nonzeros) + implicit part.
            let mut vquad = fbase;
            if ip > 0 && iq > 0 {
                let (p, qq) = (ip - 1, iq - 1);
                vquad = vquad + xp * (model.gq[p] + half * xp * model.gamma_explicit.get(p, p));
                vquad = vquad + xq * (model.gq[qq] + half * xq * model.gamma_explicit.get(qq, qq));
                vquad = vquad + xp * xq * model.gamma_explicit.get(p, qq);
            } else if ip > 0 {
                let p = ip - 1;
                vquad = vquad + xp * (model.gq[p] + half * xp * model.gamma_explicit.get(p, p));
            } else if iq > 0 {
                let qq = iq - 1;
                vquad = vquad + xq * (model.gq[qq] + half * xq * model.gamma_explicit.get(qq, qq));
            }
            let xnew = model.xpt_row(kpt).to_vec();
            let mut implicit = zero;
            for k in 0..m {
                let xxpt = dot(&xnew, model.xpt_row(k));
                implicit = implicit + model.gamma[k] * xxpt * xxpt;
            }
            vquad = vquad + half * implicit;

            // Model update (rescue.f90:532–554).
            let moderr = f - vquad;
            for i in 0..n {
                model.gq[i] = model.gq[i] + moderr * model.bmat_xi.get(i, kpt);
            }
            let mut pqinc = vec![zero; m];
            for (k, pk) in pqinc.iter_mut().enumerate() {
                let mut om = zero;
                for j in 0..rank {
                    om = om + model.zmat.get(k, j) * model.zmat.get(kpt, j);
                }
                *pk = moderr * om;
            }
            for k in 0..m {
                if ptsid[k] <= zero {
                    model.gamma[k] = model.gamma[k] + pqinc[k];
                }
            }
            for k in 0..m {
                if ptsid[k] <= zero {
                    continue;
                }
                let (ipk, iqk) = decode_ptsid(ptsid[k], n);
                if ipk > 0 && iqk > 0 {
                    let (p, qq) = (ipk - 1, iqk - 1);
                    let (ap, aq) = (ptsaux[p][0], ptsaux[qq][0]);
                    let dpp = model.gamma_explicit.get(p, p) + pqinc[k] * ap * ap;
                    model.gamma_explicit.set(p, p, dpp);
                    let dqq = model.gamma_explicit.get(qq, qq) + pqinc[k] * aq * aq;
                    model.gamma_explicit.set(qq, qq, dqq);
                    let dpq = model.gamma_explicit.get(p, qq) + pqinc[k] * ap * aq;
                    model.gamma_explicit.set(p, qq, dpq);
                    model.gamma_explicit.set(qq, p, dpq);
                } else if ipk > 0 {
                    let p = ipk - 1;
                    let ap = ptsaux[p][0];
                    let dpp = model.gamma_explicit.get(p, p) + pqinc[k] * ap * ap;
                    model.gamma_explicit.set(p, p, dpp);
                } else if iqk > 0 {
                    let qq = iqk - 1;
                    let aq = ptsaux[qq][1];
                    let dqq = model.gamma_explicit.get(qq, qq) + pqinc[k] * aq * aq;
                    model.gamma_explicit.set(qq, qq, dqq);
                }
            }
            ptsid[kpt] = zero;
        }
    }
    // Phase D (rescue.f90:559–561) is intentionally omitted: basin's gq is
    // anchored at x0, and after the prologue shift x0 = x_opt, so the
    // incremental gq updates above already give ∇Q(x0). gradient_at_opt()
    // recovers ∇Q(x_kopt) on demand for any moved kopt.

    Ok(())
}

/// Update `Ξ`/`Υ`/`Ω` to reinstate the `knew`-th original point in place of the
/// `knew`-th provisional point (BOBYQA eq. 4.9/4.14; PRIMA `updateh_rsc`,
/// rescue.f90:612–758). Operates on the all-`+1`-sign provisional `Ω`
/// (`zsign` untouched); `fval` and the model are not touched here. `vlag` is the
/// (length `m+n`) `H·w` in `[λ; g]` layout; it is mutated locally. On
/// `DAMAGING_ROUNDING` (non-finite `vlag`/`β` or `denom ≤ 0`) the update is
/// silently skipped to protect `H` — matching PRIMA, which calls this without
/// capturing its optional `info` here (the entry gate at the call site already
/// rejects bad denominators, so the skip is not expected to trigger).
fn updateh_rsc<F: Scalar>(model: &mut QuadraticModel<F>, knew: usize, beta: F, vlag: &mut [F]) {
    let n = model.n();
    let m = model.m();
    let rank = m - n - 1;
    let zero = F::zero();
    let one = F::one();

    let tau = vlag[knew];
    let mut denom = zero;
    for j in 0..rank {
        let z = model.zmat.get(knew, j);
        denom = denom + z * z;
    }
    denom = denom * beta + tau * tau;

    // Rounding can leave VLAG/β non-finite or DENOM ≤ 0; skip rather than wreck H.
    let sum_abs = vlag.iter().fold(zero, |a, &v| a + v.abs());
    if !((sum_abs + beta.abs()).is_finite() && denom > zero) {
        return;
    }

    // After this, VLAG = H·w − e_KNEW.
    vlag[knew] = vlag[knew] - one;

    // Collapse the KNEW-th row of Z onto column 0 with Givens rotations.
    let mut zmax = zero;
    for i in 0..m {
        for j in 0..rank {
            zmax = zmax.max(model.zmat.get(i, j).abs());
        }
    }
    let ztest = F::from_f64(1e-20).expect("1e-20 representable") * zmax;
    for j in 1..rank {
        if model.zmat.get(knew, j).abs() > ztest {
            let a = model.zmat.get(knew, 0);
            let b = model.zmat.get(knew, j);
            let d = (a * a + b * b).sqrt();
            let cos = a / d;
            let sin = b / d;
            model.rotate_zmat_cols(0, j, cos, sin);
        }
        model.zmat.set(knew, j, zero);
    }

    // HCOL = the KNEW-th column of the un-updated H (minus its suppressed entry).
    let zk0 = model.zmat.get(knew, 0);
    let mut hcol = vec![zero; m + n];
    for (i, h) in hcol.iter_mut().enumerate().take(m) {
        *h = zk0 * model.zmat.get(i, 0);
    }
    for r in 0..n {
        hcol[m + r] = model.bmat_xi.get(r, knew);
    }

    // Complete ZMAT (BOBYQA eq. 4.14) and BMAT (last N rows of eq. 4.9).
    let sqrtdn = denom.sqrt();
    for i in 0..m {
        let z = model.zmat.get(i, 0);
        model
            .zmat
            .set(i, 0, (tau / sqrtdn) * z - (zk0 / sqrtdn) * vlag[i]);
    }
    let alpha = hcol[knew];
    let mut v1 = vec![zero; n];
    let mut v2 = vec![zero; n];
    for r in 0..n {
        let vg = vlag[m + r];
        let hg = hcol[m + r];
        v1[r] = (alpha * vg - tau * hg) / denom;
        v2[r] = (-beta * hg - tau * vg) / denom;
    }
    for r in 0..n {
        for j in 0..m {
            let add = v1[r] * vlag[j] + v2[r] * hcol[j];
            model.bmat_xi.set(r, j, model.bmat_xi.get(r, j) + add);
        }
        for j in 0..n {
            let add = v1[r] * vlag[m + j] + v2[r] * hcol[m + j];
            model.bmat_ups.set(r, j, model.bmat_ups.get(r, j) + add);
        }
    }
    // The rank-2 update need not keep Υ exactly symmetric; restore it.
    let half = F::from_f64(0.5).expect("0.5 representable");
    for r in 0..n {
        for c in (r + 1)..n {
            let avg = half * (model.bmat_ups.get(r, c) + model.bmat_ups.get(c, r));
            model.bmat_ups.set(r, c, avg);
            model.bmat_ups.set(c, r, avg);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::powell::kkt::assert_h_matches_inverse;

    /// `decode_ptsid` must invert the `ptsid` packing for the origin, the
    /// single-coordinate points, and the two-coordinate points, at several `n`.
    #[test]
    fn ptsid_round_trips() {
        for n in [1usize, 2, 3, 5] {
            let np1 = (n + 1) as f64;
            let sfrac = 0.5 / np1;
            // Origin: ip = iq = 0.
            assert_eq!(decode_ptsid(sfrac, n), (0, 0));
            // Single-coordinate "+" point along e_k: ip = k, iq = 0.
            for k in 1..=n {
                let pid = k as f64 + sfrac;
                assert_eq!(decode_ptsid(pid, n), (k, 0), "n={n} ip-only k={k}");
            }
            // Single-coordinate "−" point along e_k: ip = 0, iq = k.
            for k in 1..=n {
                let pid = k as f64 / np1 + sfrac;
                assert_eq!(decode_ptsid(pid, n), (0, k), "n={n} iq-only k={k}");
            }
            // Two-coordinate point (ip, iq), both in 1..=n.
            for ip in 1..=n {
                for iq in 1..=n {
                    let pid = ip as f64 + iq as f64 / np1 + sfrac;
                    assert_eq!(decode_ptsid(pid, n), (ip, iq), "n={n} pair ({ip},{iq})");
                }
            }
        }
    }

    /// End-to-end: after RESCUE rebuilds the interpolation set and model, the
    /// stored `H` must equal `inv(W)` for the new geometry (KKT identity, with
    /// `Ω`-signs all +1), and the model must still interpolate `F` at every
    /// point. This single check validates the provisional-H construction, the
    /// reinstatement loop, `updateh_rsc`, and the model rebuild together.
    fn rescue_rebuilds_and_interpolates(n: usize, m: usize, rho: f64, f: impl Fn(&[f64]) -> f64) {
        let lower = vec![-5.0; n];
        let upper = vec![5.0; n];
        let x0 = vec![0.5; n];
        let mut init = QuadraticModel::try_initialize_bounded::<core::convert::Infallible>(
            x0,
            &lower,
            &upper,
            rho,
            m,
            &mut |x| Ok(f(x)),
        )
        .expect("infallible objective");

        let mut evaluated = Vec::new();
        rescue::<f64, core::convert::Infallible>(
            &mut init.model,
            &mut init.sl,
            &mut init.su,
            &lower,
            &upper,
            rho,
            &mut |x| Ok(f(x)),
            &mut evaluated,
        )
        .expect("infallible objective");

        // KKT identity for the rebuilt geometry.
        assert_h_matches_inverse(&init.model, 1e-8);

        // Interpolation: Q(x_j) − Q(x_opt) == F(x_j) − F(x_opt) for all j.
        let model = &init.model;
        let kopt = model.kopt();
        let q_opt = model.eval_change(model.xpt_row(kopt));
        let f_opt = model.fval(kopt);
        for j in 0..model.m() {
            let qj = model.eval_change(model.xpt_row(j));
            let lhs = qj - q_opt;
            let rhs = model.fval(j) - f_opt;
            assert!(
                (lhs - rhs).abs() < 1e-7,
                "interp j={j}: {lhs} vs {rhs} (Δ={:e})",
                (lhs - rhs).abs()
            );
        }
        // KOPT is the best point.
        for j in 0..model.m() {
            assert!(model.fval(j) >= f_opt - 1e-12, "fval[{j}] below fopt");
        }
    }

    #[test]
    fn rescue_quadratic_2d() {
        // A convex quadratic with a cross term.
        let f = |x: &[f64]| {
            2.0 * x[0] * x[0] + 1.5 * x[0] * x[1] + 3.0 * x[1] * x[1] + x[0] - 2.0 * x[1] + 0.7
        };
        rescue_rebuilds_and_interpolates(2, 5, 0.3, f);
    }

    #[test]
    fn rescue_quadratic_2d_full_npt() {
        let f = |x: &[f64]| (x[0] - 0.3).powi(2) + 2.0 * (x[1] + 0.4).powi(2) + 0.5 * x[0] * x[1];
        rescue_rebuilds_and_interpolates(2, 6, 0.25, f);
    }

    #[test]
    fn rescue_nonquadratic_3d() {
        // Non-quadratic: the model becomes approximate, but the H-algebra and the
        // interpolation conditions are exact regardless of F.
        let f = |x: &[f64]| {
            (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2) + (x[2] - 0.5).powi(4)
        };
        rescue_rebuilds_and_interpolates(3, 7, 0.4, f);
    }
}
