//! TRSBOX: BOBYQA's box-constrained trust-region subproblem (Powell 2009, §3).
//!
//! Approximately solves
//!
//! ```text
//! minimize  Q(x_opt + d)   subject to   ‖d‖ ≤ Δ  and  sl ≤ x_opt + d ≤ su,
//! ```
//!
//! (`sl = a − x0`, `su = b − x0` are the shifted bounds, `x_opt` the best
//! interpolation point relative to `x0`). It is the box-aware member of the
//! shared [`TrustRegionSubproblem`] family—the unconstrained TRSAPP (NEWUOA)
//! is the special case with no active bounds.
//!
//! The method (Powell 2009, §3) is an **active-set truncated conjugate
//! gradient**: from `d = 0`, each line search is capped by the least of the
//! bound step `α_B`, the trust-boundary step `α_Δ`, and the
//! monotonic-decrease step `α_Q`; hitting a bound fixes that variable and
//! restarts the CG. If `d` reaches the trust boundary, a sequence of
//! 2-D rotations on the boundary sphere (eq. 3.6) further reduces `Q` over the
//! free variables. The implicit-Hessian product `∇²Q s` is formed in `O(mn)` by
//! [`QuadraticModel::hessian_matvec`], so the work per change is `O(n)`.
//!
//! Ported from PRIMA v0.7.2 (`fortran/bobyqa/trustregion.f90`, subroutine
//! `trsbox`, with `interval_fun_trsbox` and `common/univar.f90:interval_max`).

use crate::core::math::Scalar;
use crate::solver::powell::{QuadraticModel, TrustRegionStep, TrustRegionSubproblem};

/// The shifted box `sl ≤ x_opt + d ≤ su` that TRSBOX confines its step to
/// (`sl = a − x0 ≤ 0`, `su = b − x0 ≥ 0`).
pub(crate) struct ShiftedBox<F = f64> {
    /// Shifted lower bounds, length `n`.
    pub(crate) sl: Vec<F>,
    /// Shifted upper bounds, length `n`.
    pub(crate) su: Vec<F>,
}

/// TRSBOX: BOBYQA's box-constrained trust-region subproblem strategy.
pub(crate) struct Trsbox;

impl<F: Scalar> TrustRegionSubproblem<F> for Trsbox {
    type Region = ShiftedBox<F>;

    fn solve(
        &self,
        model: &QuadraticModel<F>,
        delta: F,
        region: &ShiftedBox<F>,
    ) -> TrustRegionStep<F> {
        trsbox(model, delta, &region.sl, &region.su)
    }
}

/// `Σ_{i: xbdi[i]==0} a[i]·b[i]`—inner product over the *free* variables.
fn dot_free<F: Scalar>(a: &[F], b: &[F], xbdi: &[i8]) -> F {
    let mut s = F::zero();
    for i in 0..a.len() {
        if xbdi[i] == 0 {
            s = s + a[i] * b[i];
        }
    }
    s
}

/// The §3 objective of the boundary half-angle search: the reduction in `Q`
/// achieved by rotating through `hangt = tan(θ/2)`, with
/// `args = [shs, dhd, dhs, dredg, sredg]` (PRIMA `interval_fun_trsbox`).
fn interval_fun<F: Scalar>(hangt: F, args: &[F; 5]) -> F {
    if hangt.abs() > F::zero() {
        let one = F::one();
        let half = F::from_f64(0.5).unwrap();
        let sth = (hangt + hangt) / (one + hangt * hangt);
        let mut f = args[0] + hangt * (hangt * args[1] - args[2] - args[2]);
        f = sth * (hangt * args[3] - args[4] - half * sth * f);
        f
    } else {
        F::zero()
    }
}

/// Approximate maximizer of `interval_fun(·, args)` over `[lb, ub]` by a grid
/// of `grid_size` points plus a 3-point parabolic refinement (PRIMA
/// `interval_max`).
fn interval_max<F: Scalar>(lb: F, ub: F, args: &[F; 5], grid_size: usize) -> F {
    if ub <= lb {
        return lb;
    }
    let gm1 = F::from_usize(grid_size - 1).unwrap();
    let xgrid = |k: usize| lb + (ub - lb) * F::from_usize(k).unwrap() / gm1;
    let fgrid: Vec<F> = (0..grid_size)
        .map(|k| interval_fun(xgrid(k), args))
        .collect();

    if fgrid.iter().all(|f| f.is_nan()) {
        return lb;
    }
    let mut kopt = 0;
    let mut fopt = F::neg_infinity();
    for (k, &f) in fgrid.iter().enumerate() {
        if !f.is_nan() && f > fopt {
            fopt = f;
            kopt = k;
        }
    }
    if kopt == 0 {
        return lb;
    }
    if kopt == grid_size - 1 {
        return ub;
    }
    let fprev = fgrid[kopt - 1];
    let fnext = fgrid[kopt + 1];
    let mut step = F::zero();
    if (fprev - fnext).abs() > F::zero() {
        let half = F::from_f64(0.5).unwrap();
        step = half * ((fnext - fprev) / (fopt + fopt - fprev - fnext));
    }
    if step.is_finite() && step.abs() > F::zero() {
        lb + (ub - lb) * (F::from_usize(kopt).unwrap() + step) / gm1
    } else {
        xgrid(kopt)
    }
}

/// TRSBOX proper (Powell 2009, §3). `sl`/`su` are the shifted bounds; the
/// returned step `d` is relative to `x_opt` and satisfies `sl ≤ x_opt + d ≤ su`,
/// `‖d‖ ≲ delta`.
pub(crate) fn trsbox<F: Scalar>(
    model: &QuadraticModel<F>,
    delta: F,
    sl: &[F],
    su: &[F],
) -> TrustRegionStep<F> {
    assert!(delta > F::zero(), "trsbox: delta must be positive");
    let n = model.n();
    let zero = F::zero();
    let one = F::one();
    let half = F::from_f64(0.5).unwrap();
    let ctest = F::from_f64(0.01).unwrap();
    let eps = F::epsilon();

    let xopt = model.xpt_row(model.kopt()).to_vec();
    let gopt_raw = model.gradient_at_opt();

    // Scale if the gradient is huge, to avoid overflow (PRIMA). The step is
    // scale-invariant; CRVMIN and the predicted reduction are scaled back.
    let maxg = gopt_raw.iter().fold(zero, |m, g| m.max(g.abs()));
    let scaled = maxg > F::from_f64(1e12).unwrap();
    let modscal = if scaled { one / maxg } else { one };
    let gopt: Vec<F> = gopt_raw.iter().map(|g| *g * modscal).collect();
    // `∇²Q s`, scaled to match `gopt`.
    let hess = |s: &[F]| -> Vec<F> {
        let mut hs = model.hessian_matvec(s);
        if scaled {
            for v in hs.iter_mut() {
                *v = *v * modscal;
            }
        }
        hs
    };

    // XBDI(i): -1/0/+1 = fixed at lower / free / fixed at upper.
    let mut xbdi = vec![0i8; n];
    for i in 0..n {
        if xopt[i] >= su[i] && gopt[i] <= zero {
            xbdi[i] = 1;
        } else if xopt[i] <= sl[i] && gopt[i] >= zero {
            xbdi[i] = -1;
        }
    }
    let mut nact = xbdi.iter().filter(|&&b| b != 0).count();

    let mut d = vec![zero; n];
    let mut crvmin: Option<F> = None;
    let mut gnew = gopt.clone();
    let mut gredsq = dot_free(&gnew, &gnew, &xbdi);
    let mut delsq = delta * delta;
    let mut qred = zero;
    let mut beta = zero;
    let mut itercg = 0usize;
    let mut twod_search = false;
    let mut s = vec![zero; n];
    let mut ggsav = zero;

    let maxiter = (n - nact).saturating_mul(n - nact).min(10_000);
    for _ in 0..maxiter {
        let resid = delsq - {
            let mut acc = zero;
            for i in 0..n {
                if xbdi[i] == 0 {
                    acc = acc + d[i] * d[i];
                }
            }
            acc
        };
        if resid <= zero {
            twod_search = true;
            break;
        }

        // CG direction: steepest descent on restart, else conjugate. Fixed
        // components forced to zero.
        for i in 0..n {
            s[i] = if itercg == 0 {
                -gnew[i]
            } else {
                beta * s[i] - gnew[i]
            };
        }
        for i in 0..n {
            if xbdi[i] != 0 {
                s[i] = zero;
            }
        }
        let stepsq: F = s.iter().fold(zero, |a, v| a + *v * *v);
        let ds = dot_free(&d, &s, &xbdi);
        if !(stepsq > eps * delsq
            && gredsq * delsq > (ctest * qred) * (ctest * qred)
            && ds.is_finite())
        {
            break;
        }

        // Step to the trust boundary, ignoring bounds (α_Δ via the discriminant).
        let disc = stepsq * resid + ds * ds;
        let sqrtd = (disc.max(zero))
            .sqrt()
            .max((stepsq * resid).max(zero).sqrt())
            .max(ds.abs());
        let bstep = if ds >= zero {
            resid / (sqrtd + ds)
        } else {
            (sqrtd - ds) / stepsq
        };
        if bstep <= zero || !bstep.is_finite() {
            break;
        }

        let hs = hess(&s);
        let shs = dot_free(&s, &hs, &xbdi);
        let mut stplen = bstep;
        if shs > zero {
            stplen = bstep.min(gredsq / shs);
        }

        // Cap the step at the nearest bound (α_B), recording its index in IACT.
        let mut iact: Option<usize> = None;
        {
            let mut sbound = vec![stplen; n];
            for i in 0..n {
                let xnew_i = xopt[i] + d[i];
                let xtest = xnew_i + stplen * s[i];
                if s[i] > zero && xtest > su[i] {
                    sbound[i] = (su[i] - xnew_i) / s[i];
                } else if s[i] < zero && xtest < sl[i] {
                    sbound[i] = (sl[i] - xnew_i) / s[i];
                }
                if sbound[i].is_nan() {
                    sbound[i] = stplen;
                }
            }
            let mut best = stplen;
            for i in 0..n {
                if sbound[i] < best {
                    best = sbound[i];
                    iact = Some(i);
                }
            }
            if let Some(_ia) = iact {
                stplen = best;
            }
        }

        // Apply the step; accumulate the reduction and (if interior) CRVMIN.
        let mut sdec = zero;
        if stplen > zero {
            itercg += 1;
            let rayleigh = shs / stepsq;
            if iact.is_none() && rayleigh > zero {
                crvmin = Some(match crvmin {
                    Some(c) => c.min(rayleigh),
                    None => rayleigh,
                });
            }
            ggsav = gredsq;
            for i in 0..n {
                gnew[i] = gnew[i] + stplen * hs[i];
            }
            gredsq = dot_free(&gnew, &gnew, &xbdi);
            for i in 0..n {
                d[i] = d[i] + stplen * s[i];
            }
            sdec = (stplen * (ggsav - half * stplen * shs)).max(zero);
            qred = qred + sdec;
        }

        if let Some(ia) = iact {
            // Hit a new bound: fix it and restart the CG.
            nact += 1;
            xbdi[ia] = if s[ia] > zero { 1 } else { -1 };
            if nact >= n {
                break;
            }
            delsq = delsq - d[ia] * d[ia];
            if delsq <= zero {
                twod_search = true;
                break;
            }
            beta = zero;
            itercg = 0;
            gredsq = dot_free(&gnew, &gnew, &xbdi);
        } else if stplen < bstep {
            // Interior CG step: continue or stop on the §3 tolerance (eq. 3.4).
            if itercg >= n - nact || sdec <= ctest * qred || sdec.is_nan() || qred.is_nan() {
                break;
            }
            beta = gredsq / ggsav;
        } else {
            // Reached the trust boundary.
            twod_search = true;
            break;
        }
    }

    // --- 2-D boundary search (Powell 2009, eq. 3.6). ---
    let maxiter2 = if twod_search {
        crvmin = Some(zero);
        10 * (n - nact)
    } else {
        0
    };
    let mut nactsav = nact.wrapping_sub(1);
    // `hdred` (= ∇²Q · reduced-D) and `dredsq` persist across boundary
    // iterations; both are recomputed whenever the active set grows.
    let mut hdred = vec![zero; n];
    let mut dredsq = zero;
    for iter in 0..maxiter2 {
        let xnew: Vec<F> = (0..n).map(|i| xopt[i] + d[i]).collect();
        for i in 0..n {
            if xbdi[i] == 0 && xnew[i] >= su[i] {
                xbdi[i] = 1;
            } else if xbdi[i] == 0 && xnew[i] <= sl[i] {
                xbdi[i] = -1;
            }
        }
        nact = xbdi.iter().filter(|&&b| b != 0).count();
        if nact >= n - 1 {
            break;
        }
        gredsq = dot_free(&gnew, &gnew, &xbdi);
        let dredg = dot_free(&d, &gnew, &xbdi);
        if iter == 0 || nact > nactsav {
            dredsq = {
                let mut a = zero;
                for i in 0..n {
                    if xbdi[i] == 0 {
                        a = a + d[i] * d[i];
                    }
                }
                a
            };
            let mut dred = d.clone();
            for i in 0..n {
                if xbdi[i] != 0 {
                    dred[i] = zero;
                }
            }
            hdred = hess(&dred);
            nactsav = nact;
        }

        // S: in span{P d, P g}, orthogonal to P d, descent on g.
        let temp0 = gredsq * dredsq - dredg * dredg;
        if temp0 <= ctest * ctest * qred * qred || temp0.is_nan() || qred.is_nan() {
            break;
        }
        let temp = temp0.sqrt();
        for i in 0..n {
            s[i] = (dredg * d[i] - dredsq * gnew[i]) / temp;
            if xbdi[i] != 0 {
                s[i] = zero;
            }
        }
        let sredg = -temp;

        // Bound on tan(θ/2) from the free-variable simple bounds (eq. 3.6).
        let mut hangt_bd = one;
        let mut iact: Option<usize> = None;
        let mut tanbd = vec![one; n];
        for i in 0..n {
            if xbdi[i] != 0 {
                continue;
            }
            let ssq = d[i] * d[i] + s[i] * s[i];
            let ssqrt = ssq.sqrt();
            if xopt[i] - sl[i] < ssqrt {
                let disc = (ssq - (xopt[i] - sl[i]) * (xopt[i] - sl[i]))
                    .max(zero)
                    .sqrt();
                if disc - s[i] > zero {
                    tanbd[i] = tanbd[i].min((xnew[i] - sl[i]) / (disc - s[i]));
                }
            }
            if su[i] - xopt[i] < ssqrt {
                let disc = (ssq - (su[i] - xopt[i]) * (su[i] - xopt[i]))
                    .max(zero)
                    .sqrt();
                if disc + s[i] > zero {
                    tanbd[i] = tanbd[i].min((su[i] - xnew[i]) / (disc + s[i]));
                }
            }
            if tanbd[i].is_nan() {
                tanbd[i] = zero;
            }
        }
        for i in 0..n {
            if xbdi[i] == 0 && tanbd[i] < hangt_bd {
                hangt_bd = tanbd[i];
                iact = Some(i);
            }
        }
        // `iact` is only meaningful when some tanbd dipped below 1.
        if hangt_bd >= one {
            iact = None;
        }
        if hangt_bd <= zero {
            break;
        }

        let hs = hess(&s);
        let shs = dot_free(&s, &hs, &xbdi);
        let dhs = dot_free(&d, &hs, &xbdi);
        let dhd = dot_free(&d, &hdred, &xbdi);
        let args = [shs, dhd, dhs, dredg, sredg];
        if args.iter().any(|a| a.is_nan()) {
            break;
        }
        let gs_f = F::from_f64(17.0).unwrap() * hangt_bd + F::from_f64(4.1).unwrap();
        let grid_size = 2 * (gs_f.to_f64().unwrap().round() as usize).max(2);
        let hangt = interval_max(zero, hangt_bd, &args, grid_size);
        let sdec = interval_fun(hangt, &args);
        // `.not. sdec > 0` in PRIMA—also exits on NaN.
        if sdec <= zero || sdec.is_nan() {
            break;
        }

        let cth = (one - hangt * hangt) / (one + hangt * hangt);
        let sth = (hangt + hangt) / (one + hangt * hangt);
        for i in 0..n {
            gnew[i] = gnew[i] + (cth - one) * hdred[i] + sth * hs[i];
        }
        for i in 0..n {
            if xbdi[i] == 0 {
                d[i] = cth * d[i] + sth * s[i];
            }
        }
        for i in 0..n {
            hdred[i] = cth * hdred[i] + sth * hs[i];
        }
        qred = qred + sdec;
        if let Some(ia) = iact {
            if hangt >= hangt_bd {
                // The rotation hit a bound: fix that variable.
                let mid = half * (sl[ia] + su[ia]);
                xbdi[ia] = if xopt[ia] + d[ia] - mid >= zero {
                    1
                } else {
                    -1
                };
                continue;
            }
        }
        if sdec <= ctest * qred || sdec.is_nan() {
            break;
        }
    }

    // Snap to the bounds, then form d relative to x_opt.
    let mut xnew: Vec<F> = (0..n)
        .map(|i| sl[i].max(su[i].min(xopt[i] + d[i])))
        .collect();
    for i in 0..n {
        if xbdi[i] == -1 {
            xnew[i] = sl[i];
        } else if xbdi[i] == 1 {
            xnew[i] = su[i];
        }
    }
    for i in 0..n {
        d[i] = xnew[i] - xopt[i];
    }

    let mut crvmin_out = match crvmin {
        Some(c) if c.is_finite() => c,
        _ => zero,
    };
    if scaled && crvmin_out > zero {
        crvmin_out = crvmin_out / modscal;
    }
    let predicted_reduction = if scaled { qred / modscal } else { qred };

    TrustRegionStep {
        d,
        crvmin: crvmin_out,
        predicted_reduction,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::powell::QuadraticModel;

    /// Build a bound-aware model for a quadratic `f`.
    fn model_for(
        x0: Vec<f64>,
        lower: &[f64],
        upper: &[f64],
        rho: f64,
        m: usize,
        f: impl Fn(&[f64]) -> f64,
    ) -> (QuadraticModel<f64>, Vec<f64>, Vec<f64>) {
        let out = QuadraticModel::try_initialize_bounded::<core::convert::Infallible>(
            x0,
            lower,
            upper,
            rho,
            m,
            &mut |x| Ok(f(x)),
        )
        .unwrap();
        (out.model, out.sl, out.su)
    }

    fn norm(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// The predicted reduction equals the model's own change at `x_opt`:
    /// `qred = Q(x_opt) − Q(x_opt + d)`.
    fn assert_qred_matches(model: &QuadraticModel<f64>, d: &[f64], qred: f64) {
        let xopt = model.xpt_row(model.kopt()).to_vec();
        let xopt_d: Vec<f64> = xopt.iter().zip(d).map(|(a, b)| a + b).collect();
        let want = model.eval_change(&xopt) - model.eval_change(&xopt_d);
        assert!(
            (qred - want).abs() < 1e-9,
            "qred={qred}, Q(xopt)-Q(xopt+d)={want}"
        );
    }

    /// Slack bounds, large Δ: TRSBOX behaves like the unconstrained solve—it
    /// reaches the interior minimizer (gradient ≈ 0 at x_opt + d) with a
    /// positive predicted reduction that matches the model, and no bound active.
    #[test]
    fn interior_unconstrained_equivalent() {
        // Convex quadratic with min at (1, -2), well inside a slack box.
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + 2.0 * (x[1] + 2.0).powi(2);
        let (model, sl, su) = model_for(vec![0.0, 0.0], &[-10.0, -10.0], &[10.0, 10.0], 0.5, 5, f);
        let step = trsbox(&model, 100.0, &sl, &su);

        assert!(step.predicted_reduction > 0.0);
        assert_qred_matches(&model, &step.d, step.predicted_reduction);
        // Feasible.
        let xopt = model.xpt_row(model.kopt()).to_vec();
        for i in 0..2 {
            let xi = xopt[i] + step.d[i];
            assert!(xi >= sl[i] - 1e-9 && xi <= su[i] + 1e-9);
        }
        // Interior of a large region ⇒ CRVMIN set (> 0 for SPD).
        assert!(step.crvmin > 0.0, "crvmin={}", step.crvmin);
    }

    /// Small Δ: the step rides the trust boundary, CRVMIN resets to zero.
    #[test]
    fn boundary_step_resets_crvmin() {
        let f = |x: &[f64]| (x[0] - 5.0).powi(2) + (x[1] - 5.0).powi(2);
        let (model, sl, su) = model_for(vec![0.0, 0.0], &[-10.0, -10.0], &[10.0, 10.0], 0.5, 5, f);
        let delta = 0.5;
        let step = trsbox(&model, delta, &sl, &su);
        assert!(
            (norm(&step.d) - delta).abs() < 1e-6,
            "‖d‖={}",
            norm(&step.d)
        );
        assert_eq!(step.crvmin, 0.0);
        assert!(step.predicted_reduction > 0.0);
        assert_qred_matches(&model, &step.d, step.predicted_reduction);
    }

    /// The unconstrained minimizer lies outside the box: TRSBOX keeps the step
    /// feasible (some coordinate pinned to a bound) and still reduces Q.
    #[test]
    fn active_bound_keeps_feasible() {
        // Min at (5, 5) but the box caps x at 2.
        let f = |x: &[f64]| (x[0] - 5.0).powi(2) + (x[1] - 5.0).powi(2);
        let (model, sl, su) = model_for(vec![0.0, 0.0], &[-2.0, -2.0], &[2.0, 2.0], 0.5, 5, f);
        let xopt = model.xpt_row(model.kopt()).to_vec();
        // Big Δ so only the box (not the trust region) limits the step.
        let step = trsbox(&model, 100.0, &sl, &su);

        assert!(step.predicted_reduction > 0.0);
        assert_qred_matches(&model, &step.d, step.predicted_reduction);
        for i in 0..2 {
            let xi = xopt[i] + step.d[i];
            assert!(
                xi >= sl[i] - 1e-9 && xi <= su[i] + 1e-9,
                "coord {i} infeasible: {xi} not in [{}, {}]",
                sl[i],
                su[i]
            );
        }
        // At least one coordinate should be driven to its upper bound.
        let at_bound = (0..2).any(|i| (xopt[i] + step.d[i] - su[i]).abs() < 1e-7);
        assert!(at_bound, "expected a coordinate pinned to the upper bound");
    }
}
