//! BOBYQA geometry: point-to-drop selection and the ALTMOV geometry step
//! (Powell 2009, §3, the discussion from eq. 3.7).
//!
//! - [`setdrop_tr`] picks which interpolation point to replace after a
//!   trust-region step, scoring each candidate by a distance weight times the
//!   update denominator (eq. 6.1 / PRIMA `setdrop_tr`).
//! - [`geostep`] (ALTMOV) chooses a geometry-improving step for a chosen point:
//!   a search along the `m−1` lines through `x_opt` and another interpolation
//!   point (eq. 3.11), optionally replaced by a constrained Cauchy step of the
//!   Lagrange function (eqs. 3.12–3.15) when that yields a larger denominator.
//!
//! Ported from PRIMA v0.7.2 `fortran/bobyqa/geometry.f90`. The update
//! denominator `den(k)` (PRIMA `calden`) is obtained from the shared model's
//! [`update_params`](QuadraticModel::update_params) `σ`; the `k`-th Lagrange
//! function's gradient and Hessian come from
//! [`lagrange_coeffs`](QuadraticModel::lagrange_coeffs) /
//! [`lagrange_hessian_matvec`](QuadraticModel::lagrange_hessian_matvec).

use crate::core::math::Scalar;
use crate::solver::powell::QuadraticModel;

/// The update denominator `σ` for replacing each interpolation point `k` with
/// `x_opt + d` (PRIMA `calden`), via one `prepare_update` and a per-`k`
/// `update_params`. `d` is relative to `x_opt`.
fn denominators<F: Scalar>(model: &QuadraticModel<F>, d: &[F]) -> Vec<F> {
    let n = model.n();
    let kopt = model.kopt();
    let xopt = model.xpt_row(kopt).to_vec();
    let xnew_disp: Vec<F> = (0..n).map(|i| xopt[i] + d[i]).collect();
    let ctx = model.prepare_update(&xnew_disp);
    (0..model.m())
        .map(|k| model.update_params(k, &ctx).sigma)
        .collect()
}

/// `‖x_t − center‖²` for every interpolation point.
fn distsq_to<F: Scalar>(model: &QuadraticModel<F>, center: &[F]) -> Vec<F> {
    let n = model.n();
    (0..model.m())
        .map(|k| {
            let row = model.xpt_row(k);
            (0..n).fold(F::zero(), |a, i| {
                a + (row[i] - center[i]) * (row[i] - center[i])
            })
        })
        .collect()
}

/// Choose the interpolation point to drop after a trust-region step (PRIMA
/// `setdrop_tr`; Powell 2009, eq. 6.1). `d` is the step relative to `x_opt`;
/// `ximproved` indicates `F(x_opt + d) < F(x_opt)`. Returns `None` (PRIMA
/// `KNEW = 0`) when no point should be replaced.
pub(crate) fn setdrop_tr<F: Scalar>(
    model: &QuadraticModel<F>,
    ximproved: bool,
    d: &[F],
    rho: F,
) -> Option<usize> {
    let n = model.n();
    let kopt = model.kopt();
    let one = F::one();
    let zero = F::zero();

    // DISTSQ around the point that will be optimal next iteration.
    let center: Vec<F> = if ximproved {
        let xopt = model.xpt_row(kopt);
        (0..n).map(|i| xopt[i] + d[i]).collect()
    } else {
        model.xpt_row(kopt).to_vec()
    };
    let distsq = distsq_to(model, &center);
    let rho2 = rho * rho;
    let den = denominators(model, d);

    let mut score = vec![zero; model.m()];
    for k in 0..model.m() {
        let w = (distsq[k] / rho2).max(one);
        let w4 = w * w * w * w;
        score[k] = w4 * den[k];
    }
    if !ximproved {
        score[kopt] = -one;
    }

    let any_gt1 = score.iter().any(|&s| s > one);
    let any_gt0 = score.iter().any(|&s| s > zero);
    if any_gt1 || (ximproved && any_gt0) {
        let mut knew = 0;
        let mut best = F::neg_infinity();
        for k in 0..model.m() {
            if !score[k].is_nan() && score[k] > best {
                best = score[k];
                knew = k;
            }
        }
        Some(knew)
    } else if ximproved {
        // Ensure the improved trial point still enters the set.
        let mut knew = 0;
        let mut best = F::neg_infinity();
        for k in 0..model.m() {
            if !distsq[k].is_nan() && distsq[k] > best {
                best = distsq[k];
                knew = k;
            }
        }
        Some(knew)
    } else {
        None
    }
}

/// `|a|` carrying the sign of `b` (Fortran `sign(a, b)`; `sign(a, 0) = |a|`).
fn sign_of<F: Scalar>(a: F, b: F) -> F {
    if b < F::zero() { -a.abs() } else { a.abs() }
}

/// ALTMOV: a geometry-improving step `d` (relative to `x_opt`) for replacing
/// `XPT(knew)` (Powell 2009, §3 from eq. 3.7). `delbar` is the geometry trust
/// radius; `sl`/`su` the shifted bounds. The returned `d` satisfies
/// `sl ≤ x_opt + d ≤ su` and `‖d‖ ≲ delbar`.
pub(crate) fn geostep<F: Scalar>(
    model: &QuadraticModel<F>,
    knew: usize,
    delbar: F,
    sl: &[F],
    su: &[F],
) -> Vec<F> {
    let n = model.n();
    let m = model.m();
    let kopt = model.kopt();
    let zero = F::zero();
    let one = F::one();
    let two = F::from_f64(2.0).unwrap();
    let half = F::from_f64(0.5).unwrap();

    let xopt = model.xpt_row(kopt).to_vec();
    // KNEW-th Lagrange function: implicit coeffs `pqlag` (= Ω e_knew), its
    // diagonal H element `alpha`, and the gradient `glag` at x_opt.
    let (g0, pqlag) = model.lagrange_coeffs(knew);
    let alpha = pqlag[knew];
    let hxopt = model.lagrange_hessian_matvec(&pqlag, &xopt);
    let mut glag: Vec<F> = (0..n).map(|i| g0[i] + hxopt[i]).collect();

    // Fallback: a finite displacement toward XPT(knew) if the gradient is NaN.
    if glag.iter().any(|v| v.is_nan()) {
        let d: Vec<F> =
            (0..n).map(|i| model.xpt_row(knew)[i] - xopt[i]).collect();
        let dn = (0..n).fold(zero, |a, i| a + d[i] * d[i]).sqrt();
        let scale = half.min(delbar / dn);
        return (0..n).map(|i| scale * d[i]).collect();
    }

    let distsq = distsq_to(model, &xopt);
    // PHI_k'(0) = glag · (x_k − x_opt).
    let dderiv: Vec<F> = (0..m)
        .map(|k| {
            let row = model.xpt_row(k);
            (0..n).fold(zero, |a, i| a + glag[i] * (row[i] - xopt[i]))
        })
        .collect();

    // Per line k: step lengths [slbd, subd, stpm] and bound codes [ilbd, iubd, 0].
    let mut stplen = vec![[zero; 3]; m];
    let mut isbd = vec![[0isize; 3]; m];
    let mut dderiv = dderiv;
    for k in 0..m {
        if k == kopt || dderiv[k].is_nan() {
            dderiv[k] = zero;
            continue;
        }
        let dk = distsq[k].sqrt();
        let mut subd = delbar / dk;
        let mut slbd = -subd;
        let mut ilbd = 0isize;
        let mut iubd = 0isize;
        let sumin = one.min(subd);

        let row = model.xpt_row(k);
        let xdiff: Vec<F> = (0..n).map(|i| row[i] - xopt[i]).collect();
        let mut lfrac = vec![zero; n];
        let mut ufrac = vec![zero; n];
        for i in 0..n {
            lfrac[i] = sign_of(subd, -xdiff[i]);
            if sl[i] - xopt[i] > -xdiff[i].abs() * subd {
                lfrac[i] = (sl[i] - xopt[i]) / xdiff[i];
            }
            ufrac[i] = sign_of(subd, xdiff[i]);
            if su[i] - xopt[i] < xdiff[i].abs() * subd {
                ufrac[i] = (su[i] - xopt[i]) / xdiff[i];
            }
        }

        // Revise SLBD by the bounds.
        let mut best = slbd;
        let mut iarg: Option<usize> = None;
        for i in 0..n {
            let cand = if xdiff[i] > zero {
                lfrac[i]
            } else if xdiff[i] < zero {
                ufrac[i]
            } else {
                slbd
            };
            if !cand.is_nan() && cand > best {
                best = cand;
                iarg = Some(i);
            }
        }
        if let Some(i) = iarg {
            slbd = best;
            ilbd = -((i as isize) + 1) * (if xdiff[i] < zero { -1 } else { 1 });
        }

        // Revise SUBD by the bounds.
        let mut best = subd;
        let mut iarg: Option<usize> = None;
        for i in 0..n {
            let cand = if xdiff[i] > zero {
                ufrac[i]
            } else if xdiff[i] < zero {
                lfrac[i]
            } else {
                subd
            };
            if !cand.is_nan() && cand < best {
                best = cand;
                iarg = Some(i);
            }
        }
        if let Some(i) = iarg {
            subd = sumin.max(best);
            iubd = ((i as isize) + 1) * (if xdiff[i] < zero { -1 } else { 1 });
        }

        // Critical point STPM of PHI_k on [slbd, subd].
        let mut stpm = half;
        if k == knew {
            stpm = slbd;
            if (one - dderiv[k]).abs() > zero {
                stpm = -half * dderiv[k] / (one - dderiv[k]);
            }
        }
        stpm = slbd.max(subd.min(stpm));

        stplen[k] = [slbd, subd, stpm];
        isbd[k] = [ilbd, iubd, 0];
    }

    // PREDSQ over all 3·(m−1) trial points (eq. 3.11); pick the global best.
    let mut best_pred = F::neg_infinity();
    let mut best_isq = 0usize;
    let mut best_ksq = kopt;
    for k in 0..m {
        if k == kopt {
            continue;
        }
        for isq in 0..3 {
            let t = stplen[k][isq];
            let vlag = if k == knew {
                t * (t * (one - dderiv[k]) + dderiv[k])
            } else {
                t * (one - t) * dderiv[k]
            };
            let vlag = if vlag.is_nan() { zero } else { vlag };
            let bb = half * (t * (one - t) * distsq[k]);
            let betabd = bb * bb;
            let mut predsq = vlag * vlag * (vlag * vlag + alpha * betabd);
            if predsq.is_nan() {
                predsq = zero;
            }
            if predsq > best_pred {
                best_pred = predsq;
                best_isq = isq;
                best_ksq = k;
            }
        }
    }

    // XLINE: the line step, snapped exactly to the bounds it reaches.
    let stpsiz = stplen[best_ksq][best_isq];
    let ibd = isbd[best_ksq][best_isq];
    let row = model.xpt_row(best_ksq);
    let mut xline: Vec<F> = (0..n)
        .map(|i| sl[i].max(su[i].min(xopt[i] + stpsiz * (row[i] - xopt[i]))))
        .collect();
    if ibd < 0 {
        xline[(-ibd) as usize - 1] = sl[(-ibd) as usize - 1];
    } else if ibd > 0 {
        xline[ibd as usize - 1] = su[ibd as usize - 1];
    }
    let d_line: Vec<F> = (0..n).map(|i| xline[i] - xopt[i]).collect();

    // For larger radii the line step alone is used (PRIMA: only try the Cauchy
    // alternative when delbar is small: it helps bound-constrained runs).
    if delbar > F::from_f64(1e-2).unwrap() {
        return d_line;
    }
    let den_line_knew = denominators(model, &d_line)[knew];

    // Constrained Cauchy step of |Lagrange| (eqs. 3.13–3.15), downhill then uphill.
    let bigstp = delbar + delbar;
    let mut xcauchy = xopt.clone();
    let mut vlagsq_cauchy = zero;
    for uphill in 0..2 {
        if uphill == 1 {
            for v in glag.iter_mut() {
                *v = -*v;
            }
        }
        let mut s = vec![zero; n];
        let mut mask_free = vec![false; n];
        for i in 0..n {
            mask_free[i] = (xopt[i] - sl[i]).min(glag[i]) > zero
                || (xopt[i] - su[i]).max(glag[i]) < zero;
            if mask_free[i] {
                s[i] = bigstp;
            }
        }
        let mut ggfree = (0..n).fold(zero, |a, i| {
            if mask_free[i] {
                a + glag[i] * glag[i]
            } else {
                a
            }
        });
        if ggfree <= zero {
            continue;
        }

        // Fix more components iteratively (eq. 3.15).
        let mut sfixsq = zero;
        let mut grdstp = zero;
        for _ in 0..n {
            let resis = delbar * delbar - sfixsq;
            if resis <= zero {
                break;
            }
            let ssqsav = sfixsq;
            grdstp = (resis / ggfree).sqrt();
            for i in 0..n {
                if s[i] >= bigstp {
                    let xtemp = xopt[i] - grdstp * glag[i];
                    if xtemp <= sl[i] {
                        s[i] = sl[i] - xopt[i];
                        sfixsq = sfixsq + s[i] * s[i];
                    } else if xtemp >= su[i] {
                        s[i] = su[i] - xopt[i];
                        sfixsq = sfixsq + s[i] * s[i];
                    }
                }
            }
            ggfree = (0..n).fold(zero, |a, i| {
                if s[i] >= bigstp {
                    a + glag[i] * glag[i]
                } else {
                    a
                }
            });
            // Loop continues only while a component was newly fixed (sfixsq grew)
            // and free gradient mass remains: PRIMA geometry.f90 stop test.
            if !(sfixsq > ssqsav && ggfree > zero) {
                break;
            }
        }

        // Assemble the candidate point x and step s.
        let mut x = vec![zero; n];
        for i in 0..n {
            x[i] = if glag[i] > zero { sl[i] } else { su[i] };
            if s[i].abs() <= zero {
                x[i] = xopt[i];
            }
            if s[i] >= bigstp {
                x[i] = sl[i].max(su[i].min(xopt[i] - grdstp * glag[i]));
                s[i] = -grdstp * glag[i];
            }
        }
        let gs = (0..n).fold(zero, |a, i| a + glag[i] * s[i]);

        // Curvature of the Lagrange function along S; rescale S if helpful.
        let mut curv = {
            let hs = model.lagrange_hessian_matvec(&pqlag, &s);
            (0..n).fold(zero, |a, i| a + s[i] * hs[i])
        };
        if uphill == 1 {
            curv = -curv;
        }
        let vlagsq;
        if curv > -gs && curv < -(one + two.sqrt()) * gs {
            let scaling = -gs / curv;
            for i in 0..n {
                x[i] = sl[i].max(su[i].min(xopt[i] + scaling * s[i]));
            }
            let v = half * gs * scaling;
            vlagsq = v * v;
        } else {
            let v = gs + half * curv;
            vlagsq = v * v;
        }

        if vlagsq > vlagsq_cauchy {
            xcauchy = x;
            vlagsq_cauchy = vlagsq;
        }
    }

    let d_cauchy: Vec<F> = (0..n).map(|i| xcauchy[i] - xopt[i]).collect();
    let den_cauchy_knew = denominators(model, &d_cauchy)[knew];

    if den_cauchy_knew > den_line_knew.max(zero) || den_line_knew.is_nan() {
        d_cauchy
    } else {
        d_line
    }
}
