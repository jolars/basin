//! TRSTEP: LINCOA's linearly-constrained trust-region subproblem (Powell 2015,
//! §3–§5; PRIMA `trustregion.f90::trstep`).
//!
//! Approximately solves
//!
//! ```text
//! minimize  Q(x_opt + d)   subject to   ‖d‖ ≤ Δ  and  A (x_opt + d) ≤ b,
//! ```
//!
//! by a **projected truncated conjugate gradient**: [`getact`](super::getact)
//! selects the nearly-active constraints, maintains their QR factorization, and
//! returns the projected steepest-descent direction `psd`; the CG search
//! directions are projected onto the null space of the active normals and
//! truncated at the trust boundary or when a new constraint becomes active. It is
//! the linearly-constrained analogue of the shared
//! [`TrustRegionSubproblem`](crate::solver::powell::TrustRegionSubproblem)
//! strategies: TRSAPP (NEWUOA) and TRSBOX (BOBYQA) are the no-constraint and
//! box-only special cases.
//!
//! Unlike NEWUOA and BOBYQA, LINCOA does **not** implement the `TrustRegionSubproblem`
//! seam: its active-set QR ([`ActiveSetQr`]) is a *warm start* that must persist
//! across driver iterations, so the driver calls the free [`trstep`] directly
//! with its own `&mut ActiveSetQr` rather than through the seam's stateless
//! `&self` `solve`. The seam stays the right shape for the stateless members.
//!
//! D = 0 is assumed feasible (`b ≥ 0` up to rounding), which LINCOA's init
//! guarantees by keeping `x_opt` feasible.

use crate::core::math::Scalar;
use crate::solver::powell::{QuadraticModel, TrustRegionStep};

use super::getact::{ActiveSetQr, getact};

/// Plain dot product.
fn dot<F: Scalar>(a: &[F], b: &[F]) -> F {
    a.iter().zip(b).fold(F::zero(), |acc, (x, y)| acc + *x * *y)
}

/// `vᵀ a_j` where `a_j` is column `j` of the `n × m` column-major `amat`.
fn amat_col_dot<F: Scalar>(v: &[F], amat: &[F], j: usize, n: usize) -> F {
    (0..n).fold(F::zero(), |acc, r| acc + v[r] * amat[r + j * n])
}

/// Solve the lower-triangular system `Rᵀ z = b` (forward substitution), where `R`
/// is the `n × n` column-major upper-triangular factor and only its leading
/// `k × k` block is used. PRIMA `solve(transpose(rfac), ·)`.
fn solve_rt<F: Scalar>(rfac: &[F], n: usize, k: usize, b: &[F]) -> Vec<F> {
    let mut z = vec![F::zero(); k];
    for i in 0..k {
        let mut s = b[i];
        for j in 0..i {
            s = s - rfac[j + i * n] * z[j]; // Rᵀ(i,j) = R(j,i) = rfac[j + i*n]
        }
        z[i] = s / rfac[i + i * n];
    }
    z
}

/// Largest of three values (PRIMA's `maxval([...])` guard against `sqrt`
/// underflow in a discriminant).
fn max3<F: Scalar>(a: F, b: F, c: F) -> F {
    a.max(b).max(c)
}

/// Solve LINCOA's linearly-constrained trust-region subproblem, warm-starting
/// (and updating) the active-set QR in `warm` (Powell 2015, §3–§5; PRIMA
/// `trstep`).
///
/// `amat` is the `n × m` column-major matrix of unit constraint normals and
/// `rescon` the length-`m` residuals (`b − A x_opt`). Returns the step `d = s`
/// (relative to `x_opt`), `crvmin = 0` (LINCOA tracks no interior curvature:
/// the driver's accurate-model test does not use it), and the predicted
/// reduction `Q(x_opt) − Q(x_opt + s) ≥ 0`, paired with `ngetact`, the number of
/// [`getact`] invocations (PRIMA's `ngetact`; the driver's short-step test uses
/// it: `ngetact ≥ 2` means the active set changed during the CG).
pub(crate) fn trstep<F: Scalar>(
    model: &QuadraticModel<F>,
    delta: F,
    amat: &[F],
    rescon: &[F],
    warm: &mut ActiveSetQr<F>,
) -> (TrustRegionStep<F>, usize) {
    let n = model.n();
    let m = rescon.len();
    let (zero, one) = (F::zero(), F::one());
    let two = F::from_f64(2.0).expect("2.0 representable");
    let half = F::from_f64(0.5).expect("0.5 representable");
    let eps = F::epsilon();
    let realmin = F::min_positive_value();
    let tinycv = two.powi(-200);
    let ctest = F::from_f64(0.01).expect("0.01 representable");
    let pt2 = F::from_f64(0.2).expect("0.2 representable");
    let c1em4 = F::from_f64(1e-4).expect("1e-4 representable");

    let make_step = |s: Vec<F>| -> TrustRegionStep<F> {
        let hs = model.hessian_matvec(&s);
        let gopt = model.gradient_at_opt();
        let pred = -(dot(&gopt, &s) + half * dot(&s, &hs));
        TrustRegionStep {
            d: s,
            crvmin: zero,
            predicted_reduction: pred,
        }
    };

    // Scale the gradient and Hessian if GOPT is huge, to avoid FP exceptions. The
    // step is scale-invariant; the predicted reduction is recomputed unscaled.
    let gopt0 = model.gradient_at_opt();
    let gmax = gopt0.iter().fold(zero, |acc, v| acc.max(v.abs()));
    let modscal = if gmax > F::from_f64(1e12).expect("1e12 representable") {
        (two * realmin).max(one / gmax)
    } else {
        one
    };
    let mut g: Vec<F> = gopt0.iter().map(|&v| v * modscal).collect();
    if !g.iter().fold(zero, |a, v| a + v.abs()).is_finite() {
        return (make_step(vec![zero; n]), 0);
    }

    // Initialize RESNEW / RESACT (PRIMA trstep init).
    let mut resnew = vec![zero; m];
    for j in 0..m {
        let mut r = rescon[j];
        if rescon[j] >= zero {
            r = tinycv.max(rescon[j]);
        }
        if rescon[j] >= delta {
            r = -one;
        }
        resnew[j] = r;
    }
    for ic in 0..warm.nact {
        resnew[warm.iact[ic]] = zero;
    }
    let mut resact = vec![zero; m];
    for ic in 0..warm.nact {
        resact[ic] = rescon[warm.iact[ic]];
    }

    let delsq = delta * delta;
    let mut s = vec![zero; n];
    let mut ss = zero;
    let mut reduct = zero;
    let mut newact = true;
    let mut ngetact: usize = 0;
    let mut itercg: isize = -1;
    let mut d = vec![zero; n];
    let mut gamma = zero;
    // Assigned (and read) within each iteration before any use.
    let mut hd: Vec<F>;

    let maxiter = (10 * (m + n)).min(10_000);
    for _iter in 0..maxiter {
        if newact {
            let mut psd = vec![zero; n];
            getact(
                amat,
                n,
                m,
                delta,
                &g,
                warm,
                &mut resact,
                &mut resnew,
                &mut psd,
            );
            ngetact += 1;
            let dd = dot(&psd, &psd);
            if dd <= eps * delsq || dd.is_nan() {
                break;
            }
            let scale_psd = (pt2 * delta) / dd.sqrt();
            for p in psd.iter_mut() {
                *p = *p * scale_psd;
            }

            // Optionally modify PSD by a projection step toward the boundaries of
            // the active constraints when their residuals are substantial.
            gamma = zero;
            let mut dproj = vec![zero; n];
            if (0..warm.nact).any(|ic| resact[ic] > c1em4 * delta) {
                let z = solve_rt(&warm.rfac, n, warm.nact, &resact[..warm.nact]);
                for k in 0..warm.nact {
                    for r in 0..n {
                        dproj[r] = dproj[r] + warm.qfac[r + k * n] * z[k];
                    }
                }
                let spsd: Vec<F> = (0..n).map(|i| s[i] + psd[i]).collect();
                let ds = dot(&dproj, &spsd);
                let dd2 = dot(&dproj, &dproj);
                let resid = delsq - dot(&spsd, &spsd);
                if resid > zero && dd2 > eps * delsq && !ds.is_nan() {
                    let sqrtd = max3(
                        (ds * ds + dd2 * resid).sqrt(),
                        ds.abs(),
                        (dd2 * resid).sqrt(),
                    );
                    gamma = if ds <= zero {
                        (sqrtd - ds) / dd2
                    } else {
                        resid / (sqrtd + ds)
                    };
                    if gamma < zero || !gamma.is_finite() {
                        gamma = zero;
                    }
                    // Reduce GAMMA so the move also keeps the inactive constraints feasible.
                    let mut gmin = gamma.min(one);
                    for j in 0..m {
                        if resnew[j] > zero {
                            let adj = amat_col_dot(&dproj, amat, j, n);
                            if adj > zero {
                                let psd_aj = amat_col_dot(&psd, amat, j, n);
                                let fr = (resnew[j] - psd_aj) / adj;
                                if fr < gmin {
                                    gmin = fr;
                                }
                            }
                        }
                    }
                    gamma = gmin;
                }
            }

            if gamma > zero {
                for i in 0..n {
                    d[i] = psd[i] + gamma * dproj[i];
                }
                itercg = -1;
            } else {
                d.copy_from_slice(&psd);
                itercg = 0;
            }
        }
        itercg += 1;

        // ALPHA: steplength along D to the trust boundary.
        let resid = delsq - ss;
        let dg = dot(&d, &g);
        let ds = dot(&d, &s);
        let dd = dot(&d, &d);
        if resid <= zero || dd <= eps * delsq || ds.is_nan() {
            break;
        }
        let sqrtd = max3((ds * ds + dd * resid).sqrt(), ds.abs(), (dd * resid).sqrt());
        let mut alpha = if ds <= zero {
            (sqrtd - ds) / dd
        } else {
            resid / (sqrtd + ds)
        };
        if alpha <= zero || !alpha.is_finite() {
            break;
        }
        if -alpha * dg <= ctest * reduct || (alpha * dg).is_nan() {
            break;
        }

        // Curvature along D; reduce ALPHA to the model minimizer if needed.
        hd = model.hessian_matvec(&d);
        for h in hd.iter_mut() {
            *h = *h * modscal;
        }
        let dhd = dot(&d, &hd);
        let alpht = alpha;
        if dg + alpha * dhd > zero {
            alpha = -dg / dhd;
        }

        // Reduce ALPHA further to preserve feasibility; record the binding
        // constraint JSAV (the first global argmin of FRAC, if it is < ALPHA).
        let alphm = alpha;
        let mut ad = vec![-one; m];
        for j in 0..m {
            if resnew[j] > zero {
                ad[j] = amat_col_dot(&d, amat, j, n);
            }
        }
        let mut jsav: Option<usize> = None;
        for j in 0..m {
            if ad[j] > zero {
                let mut fr = resnew[j] / ad[j];
                if fr.is_nan() {
                    fr = alpha;
                }
                if fr < alpha {
                    alpha = fr;
                    jsav = Some(j);
                }
            }
        }

        // Post-process ALPHA per ITERCG.
        if itercg == 0 {
            alpha = one;
        } else if itercg == 1 && gamma <= zero {
            alpha = alpha.max(one);
        } else {
            alpha = alpha.max(zero);
        }
        alpha = alpha.min(alphm);

        // Update S, G.
        let sold = s.clone();
        for i in 0..n {
            s[i] = s[i] + alpha * d[i];
        }
        ss = dot(&s, &s);
        if !ss.is_finite() {
            s = sold;
            break;
        }
        for i in 0..n {
            g[i] = g[i] + alpha * hd[i];
        }
        if !g.iter().fold(zero, |a, v| a + v.abs()).is_finite() {
            break;
        }

        // Update RESNEW for the inactive constraints.
        for j in 0..m {
            if resnew[j] > zero {
                resnew[j] = tinycv.max(resnew[j] - alpha * ad[j]);
            }
        }
        // Update RESACT iff this was the modified post-GETACT step.
        if itercg == 0 {
            let factor = one - alpha * gamma;
            for ic in 0..warm.nact {
                resact[ic] = factor * resact[ic];
            }
        }

        // Update REDUCT.
        reduct = reduct - alpha * (dg + half * alpha * dhd);
        if reduct <= zero || reduct.is_nan() {
            s = sold;
            break;
        }

        // Termination tests.
        if alpha >= alpht || -alphm * (dg + half * alphm * dhd) <= ctest * reduct {
            break;
        }

        // New active constraint → recompute the active set next iteration.
        newact = jsav.is_some();
        if newact {
            continue;
        }
        // A stationary point of the current null space is reached after N-NACT
        // CG iterations.
        if itercg as usize >= n - warm.nact {
            break;
        }

        // Next (conjugate) search direction, projected onto the inactive subspace.
        let pg: Vec<F> = if warm.nact == 0 {
            g.clone()
        } else {
            let mut p = vec![zero; n];
            for col in warm.nact..n {
                let coeff = (0..n).fold(zero, |a, r| a + g[r] * warm.qfac[r + col * n]);
                for r in 0..n {
                    p[r] = p[r] + coeff * warm.qfac[r + col * n];
                }
            }
            p
        };
        let beta = if itercg == 0 {
            zero
        } else {
            dot(&pg, &hd) / dhd
        };
        for i in 0..n {
            d[i] = -pg[i] + beta * d[i];
        }
    }

    (make_step(s), ngetact)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A diagonal convex quadratic `0.5 Σ hᵢ (xᵢ − cᵢ)²`. With `npt = 2n+1` the
    /// least-Frobenius model interpolates it exactly (no cross terms).
    fn diag_quadratic(h: &'static [f64], c: &'static [f64]) -> impl Fn(&[f64]) -> f64 {
        move |x: &[f64]| {
            0.5 * (0..x.len())
                .map(|i| h[i] * (x[i] - c[i]).powi(2))
                .sum::<f64>()
        }
    }

    /// Unconstrained (m = 0): TRSTEP reaches the interior minimizer, so the
    /// gradient at `x_opt + s` is ≈ 0 and the predicted reduction is positive.
    #[test]
    fn unconstrained_reaches_interior_min() {
        let h = &[2.0, 4.0];
        let c = &[0.4, -0.3];
        let model = QuadraticModel::initialize(vec![0.0, 0.0], 0.3, 5, &diag_quadratic(h, c));
        let n = 2;
        let mut warm = ActiveSetQr::<f64>::new(n, 0);
        let (step, _) = trstep(&model, 5.0, &[], &[], &mut warm);
        // Gradient at x_opt + s ≈ 0 (interior stationary point).
        let gopt = model.gradient_at_opt();
        let hs = model.hessian_matvec(&step.d);
        for i in 0..n {
            assert!(
                (gopt[i] + hs[i]).abs() < 1e-7,
                "g_final[{i}] = {}",
                gopt[i] + hs[i]
            );
        }
        assert!(step.predicted_reduction > 0.0);
    }

    /// A single linear constraint that cuts off the unconstrained minimizer:
    /// TRSTEP returns a feasible step that activates the constraint, with the
    /// projected gradient (KKT residual) ≈ 0 on the constraint's null space.
    #[test]
    fn constrained_step_is_feasible_and_kkt() {
        let h = &[2.0, 2.0];
        let c = &[1.0, 0.0]; // unconstrained min at (1, 0)
        let model = QuadraticModel::initialize(vec![0.0, 0.0], 0.3, 5, &diag_quadratic(h, c));
        let n = 2;
        let m = 1;
        // Constraint x0 ≤ 0.5 (normal (1,0)). x_opt is near 0, so rescon = 0.5 - x_opt0.
        let amat = vec![1.0, 0.0];
        let xopt: Vec<f64> = (0..n)
            .map(|i| model.x0()[i] + model.xpt_row(model.kopt())[i])
            .collect();
        let rescon = vec![0.5 - xopt[0]];
        let mut warm = ActiveSetQr::<f64>::new(n, m);
        // Δ ≈ ρ (realistic): a huge Δ would make the constraint "nearly active"
        // immediately, so GETACT's projected gradient would be exactly 0 (the
        // correct LCPP solution) and the step would be empty.
        let (step, _) = trstep(&model, 0.3, &amat, &rescon, &mut warm);

        // Feasibility: a·(x_opt + s) ≤ b  ⟺  a·s ≤ rescon.
        let a_dot_s = amat_col_dot(&step.d, &amat, 0, n);
        assert!(
            a_dot_s <= rescon[0] + 1e-9,
            "a·s = {a_dot_s}, rescon = {}",
            rescon[0]
        );
        assert!(step.predicted_reduction > 0.0);
        // The minimizer (1,0) is infeasible (x0=1 > 0.5), so the constraint binds.
        assert_eq!(warm.nact, 1);
        // KKT: gradient at x_opt+s is parallel to the active normal (its
        // projection onto the null space of (1,0), i.e. the y-component, ≈ 0).
        let gopt = model.gradient_at_opt();
        let hs = model.hessian_matvec(&step.d);
        assert!(
            (gopt[1] + hs[1]).abs() < 1e-7,
            "null-space grad = {}",
            gopt[1] + hs[1]
        );
        // The active constraint is essentially tight at the solution.
        assert!(
            (a_dot_s - rescon[0]).abs() < 1e-6,
            "constraint not tight: {a_dot_s}"
        );
    }

    /// The step stays within (a small multiple of) the trust radius.
    #[test]
    fn step_respects_trust_radius() {
        let h = &[1.0, 1.0];
        let c = &[10.0, 10.0]; // far min → boundary solution
        let model = QuadraticModel::initialize(vec![0.0, 0.0], 0.3, 5, &diag_quadratic(h, c));
        let n = 2;
        let delta = 1.0;
        let mut warm = ActiveSetQr::<f64>::new(n, 0);
        let (step, _) = trstep(&model, delta, &[], &[], &mut warm);
        let snorm = dot(&step.d, &step.d).sqrt();
        assert!(snorm <= 2.0 * delta, "‖s‖ = {snorm}");
        assert!(
            snorm > 0.5 * delta,
            "‖s‖ = {snorm} (expected near boundary)"
        );
        assert!(step.predicted_reduction > 0.0);
    }
}
