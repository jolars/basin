//! COBYLA trust-region linear-programming step (`trstlp`) and radius update
//! (`trrad`).
//!
//! Direct port of PRIMA's `trustregion.f90` (`trstlp`/`trstlp_sub`/`trrad`).
//! `trstlp` finds the step `d` by two stages: stage 1 minimizes the L-infinity
//! violation of the linearized constraints `Aᵀd ≤ b` subject to `‖d‖ ≤ Δ`; stage
//! 2 spends any remaining freedom minimizing the linearized objective `gᵀd`
//! without increasing the greatest constraint violation. It maintains a QR
//! factorization of the active-constraint gradients (`z` = Q, `zdota` = diag R),
//! updated by Givens rotations as constraints enter (`qradd`) or leave / reorder
//! (`qrexc`), with Lagrange multipliers from a least-squares solve (`lsqr`).
//!
//! N.B. (sign convention): like PRIMA, the constraints handed here are `Aᵀd ≤ b`
//! (the negative of Powell's original `Aᵀd ≥ b`). The COBYLA driver passes
//! `A = [gradients of cᵢ]` and `b = −conmat(:, pole)`.

use crate::core::math::Scalar;

use super::linalg::{
    col, dot, eye, hypotenuse, isminor, planerot, row_times_mat,
};

/// Solve the COBYLA trust-region LP. `a` is the `n × m` column-major matrix of
/// constraint gradients, `b` the length-`m` right-hand side, `g` the length-`n`
/// objective gradient, `delta` the trust-region radius. Returns the step `d`.
pub(crate) fn trstlp<F: Scalar>(
    a: &[F],
    n: usize,
    m: usize,
    b: &[F],
    delta: F,
    g: &[F],
) -> Vec<F> {
    let mcon = m + 1;
    // A_aug = [A, g] (n × (m+1)); b_aug = [b, 0].
    let mut a_aug = vec![F::zero(); n * mcon];
    a_aug[..n * m].copy_from_slice(&a[..n * m]);
    a_aug[n * m..n * mcon].copy_from_slice(&g[..n]);
    let mut b_aug = vec![F::zero(); mcon];
    b_aug[..m].copy_from_slice(&b[..m]);

    // Scale any column with huge entries to avoid floating-point exceptions; the
    // trust-region step is scale invariant.
    let big = F::from_f64(1.0e12).unwrap();
    let realmin = F::min_positive_value();
    for i in 0..mcon {
        let mx = (0..n)
            .map(|r| a_aug[r + i * n].abs())
            .fold(F::zero(), F::max);
        if mx > big {
            let scal = ((F::one() + F::one()) * realmin).max(F::one() / mx);
            for r in 0..n {
                a_aug[r + i * n] = a_aug[r + i * n] * scal;
            }
            b_aug[i] = b_aug[i] * scal;
        }
    }

    let mut iact = vec![0usize; mcon];
    let mut vmultc = vec![F::zero(); mcon];
    let mut z = eye::<F>(n);
    let mut d = vec![F::zero(); n];
    let mut nact = 0usize;

    // Stage 1: first m columns. Stage 2: full mcon columns.
    trstlp_sub(
        1,
        &a_aug,
        n,
        m,
        &b_aug,
        delta,
        &mut d,
        &mut iact,
        &mut vmultc,
        &mut z,
        &mut nact,
    );
    trstlp_sub(
        2,
        &a_aug,
        n,
        mcon,
        &b_aug,
        delta,
        &mut d,
        &mut iact,
        &mut vmultc,
        &mut z,
        &mut nact,
    );
    d
}

/// `zdota(0..nact)` recomputed from `z` and the active columns of `a`.
fn zdota_active<F: Scalar>(
    a: &[F],
    n: usize,
    iact: &[usize],
    nact: usize,
    z: &[F],
) -> Vec<F> {
    (0..nact)
        .map(|k| dot(col(z, n, k), col(a, n, iact[k])))
        .collect()
}

/// The active-column matrix `A(:, iact(0..nact))` (n × nact, column-major).
fn active_cols<F: Scalar>(
    a: &[F],
    n: usize,
    iact: &[usize],
    nact: usize,
) -> Vec<F> {
    let mut out = vec![F::zero(); n * nact];
    for k in 0..nact {
        out[k * n..(k + 1) * n].copy_from_slice(col(a, n, iact[k]));
    }
    out
}

/// QR rank-one add (PRIMA `qradd_Rdiag`): attempt to append column `c` to the
/// active set, updating `z` (Q) and `zdota` (diag R). `nact` may grow by one.
fn qradd<F: Scalar>(
    c: &[F],
    z: &mut [F],
    zdota: &mut [F],
    nact: &mut usize,
    n: usize,
) {
    let mut cq = row_times_mat(c, z, n, n);
    let cabs: Vec<F> = c.iter().map(|&x| x.abs()).collect();
    let zabs: Vec<F> = z.iter().map(|&x| x.abs()).collect();
    let cqa = row_times_mat(&cabs, &zabs, n, n);
    for k in 0..n {
        if isminor(cq[k], cqa[k]) {
            cq[k] = F::zero();
        }
    }
    // Givens-zero cq[k+1] into cq[k] for k = n-2 .. nact (0-based cols (k, k+1)).
    let mut k = n as isize - 2;
    while k >= *nact as isize {
        let ku = k as usize;
        if cq[ku + 1].abs() > F::zero() {
            let (cc, ss) = planerot(cq[ku], cq[ku + 1]);
            for r in 0..n {
                let a0 = z[r + ku * n];
                let a1 = z[r + (ku + 1) * n];
                z[r + ku * n] = cc * a0 + ss * a1;
                z[r + (ku + 1) * n] = -ss * a0 + cc * a1;
            }
            cq[ku] = hypotenuse(cq[ku], cq[ku + 1]);
        }
        k -= 1;
    }
    let eps2 = F::epsilon() * F::epsilon();
    if *nact < n && cq[*nact].abs() > eps2 && !isminor(cq[*nact], cqa[*nact]) {
        *nact += 1;
    }
    if *nact >= 1 && *nact <= n {
        zdota[*nact - 1] = cq[*nact - 1];
    }
}

/// QR column-exchange (PRIMA `qrexc_Rdiag`): rearrange active columns
/// `[i, i+1, …, nact-1]` to `[i+1, …, nact-1, i]` (0-based `i`), updating `z`
/// and `zdota`. `aact` is the active-column matrix (n × nact).
fn qrexc<F: Scalar>(
    aact: &[F],
    z: &mut [F],
    zdota: &mut [F],
    n: usize,
    nact: usize,
    i: usize,
) {
    if i + 1 >= nact {
        return;
    }
    for k in i..(nact - 1) {
        let (cc, ss) =
            planerot(zdota[k + 1], dot(col(z, n, k), col(aact, n, k + 1)));
        // Q(:, [k, k+1]) = [Q(:, k+1), Q(:, k)] * G^T.
        for r in 0..n {
            let p1 = z[r + (k + 1) * n]; // Q(:, k+1)
            let p2 = z[r + k * n]; // Q(:, k)
            z[r + k * n] = cc * p1 + ss * p2;
            z[r + (k + 1) * n] = -ss * p1 + cc * p2;
        }
    }
    // Recompute the affected diagonal of R from scratch.
    for k in i..(nact - 1) {
        zdota[k] = dot(col(z, n, k), col(aact, n, k + 1));
    }
    zdota[nact - 1] = dot(col(z, n, nact - 1), col(aact, n, i));
}

/// Least-squares multipliers against the stored QR (PRIMA `lsqr_Rdiag`, the
/// `Q`+`Rdiag`-present branch). `aact` is the active-column matrix (n × nact),
/// `target` the length-`n` right-hand side. Returns length-`nact`.
fn lsqr<F: Scalar>(
    aact: &[F],
    target: &[F],
    z: &[F],
    zdota: &[F],
    n: usize,
    nact: usize,
) -> Vec<F> {
    let mut x = vec![F::zero(); nact];
    let mut y = target.to_vec();
    for i in (0..nact).rev() {
        let zi = col(z, n, i);
        let yq = dot(&y, zi);
        let ya: Vec<F> = y.iter().map(|&v| v.abs()).collect();
        let za: Vec<F> = zi.iter().map(|&v| v.abs()).collect();
        let yqa = dot(&ya, &za);
        if isminor(yq, yqa) {
            x[i] = F::zero();
        } else {
            x[i] = yq / zdota[i];
            let aci = col(aact, n, i);
            for r in 0..n {
                y[r] = y[r] - x[i] * aci[r];
            }
        }
    }
    x
}

#[allow(clippy::too_many_arguments)]
fn trstlp_sub<F: Scalar>(
    stage: u8,
    a: &[F],
    n: usize,
    mcon: usize,
    b: &[F],
    delta: F,
    d: &mut [F],
    iact: &mut [usize],
    vmultc: &mut [F],
    z: &mut [F],
    nact: &mut usize,
) {
    let zero = F::zero();
    let one = F::one();
    let eps = F::epsilon();
    let realmax = F::max_value();
    // `m` (1-based in PRIMA): number of "real" constraints, all of them in
    // stage 1, all-but-the-objective in stage 2.
    let m_real = if stage == 1 { mcon } else { mcon - 1 };

    let mut icon: usize;
    let mut sdirn = vec![zero; n];

    if stage == 1 {
        for k in 0..mcon {
            iact[k] = k;
        }
        *nact = 0;
        for v in d.iter_mut() {
            *v = zero;
        }
        let mut cviol = zero;
        for k in 0..mcon {
            cviol = cviol.max(-b[k]);
        }
        for k in 0..mcon {
            vmultc[k] = cviol + b[k];
        }
        z.copy_from_slice(&eye::<F>(n));
        if mcon == 0 || cviol <= zero {
            return;
        }
        if b[..mcon].iter().all(|x| x.is_nan()) {
            return;
        }
        // icon = argmax of -b over non-NaN entries.
        let mut best = F::neg_infinity();
        icon = 0;
        for k in 0..mcon {
            if !b[k].is_nan() && -b[k] > best {
                best = -b[k];
                icon = k;
            }
        }
    } else {
        if dot(&d[..n], &d[..n]) >= delta * delta {
            return;
        }
        iact[mcon - 1] = mcon - 1;
        vmultc[mcon - 1] = zero;
        icon = mcon - 1;
    }

    let mut cviol = if stage == 1 {
        let mut c = zero;
        for k in 0..mcon {
            c = c.max(-b[k]);
        }
        c
    } else {
        // max over the m real columns of (A(:,k)·d − b[k]) and 0.
        let mut c = zero;
        for k in 0..m_real {
            c = c.max(dot(col(a, n, k), &d[..n]) - b[k]);
        }
        c
    };

    let mut zdota = vec![zero; n];
    {
        let zd = zdota_active(a, n, iact, *nact, z);
        zdota[..*nact].copy_from_slice(&zd);
    }

    let mut optold = realmax;
    let mut nactold = *nact;
    let mut nfail = 0i32;

    let maxiter = 10_000.min(100 * m_real.max(n));
    for _iter in 0..maxiter {
        let optnew = if stage == 1 {
            cviol
        } else {
            dot(col(a, n, mcon - 1), &d[..n])
        };

        if optnew < optold || *nact > nactold {
            nactold = *nact;
            nfail = 0;
        } else {
            nfail += 1;
        }
        optold = optold.min(optnew);
        if nfail == 3 {
            break;
        }

        if icon >= *nact {
            // --- Add constraint iact[icon] to the active set. ---
            let zdasav: Vec<F> = zdota[..*nact].to_vec();
            let nactsav = *nact;
            let ccol = col(a, n, iact[icon]).to_vec();
            qradd(&ccol, z, &mut zdota, nact, n);

            if *nact == nactsav + 1 {
                if *nact != icon + 1 {
                    // vmultc([icon, nact]) = [vmultc(nact), 0]; iact swap.
                    let nm1 = *nact - 1;
                    vmultc[icon] = vmultc[nm1];
                    vmultc[nm1] = zero;
                    iact.swap(icon, nm1);
                } else {
                    vmultc[*nact - 1] = zero;
                }
            } else {
                // C was in range(active): revise multipliers via lsqr against the
                // UN-updated active set (use zdasav), then drop a constraint.
                let aact = active_cols(a, n, iact, *nact);
                let target = col(a, n, iact[icon]).to_vec();
                let mut vmultd = vec![zero; mcon];
                {
                    let vd = lsqr(&aact, &target, z, &zdasav, n, *nact);
                    vmultd[..*nact].copy_from_slice(&vd);
                }
                let any_pos =
                    (0..*nact).any(|k| vmultd[k] > zero && iact[k] < m_real);
                if !any_pos {
                    break;
                }
                let mut frac = realmax;
                for k in 0..*nact {
                    if vmultd[k] > zero && iact[k] < m_real {
                        frac = frac.min(vmultc[k] / vmultd[k]);
                    }
                }
                for k in 0..*nact {
                    vmultc[k] = zero.max(vmultc[k] - frac * vmultd[k]);
                }
                if zdota[*nact - 1].is_nan()
                    || zdota[*nact - 1].abs() <= eps * eps
                {
                    break;
                }
                let nm1 = *nact - 1;
                vmultc[icon] = zero;
                vmultc[nm1] = frac;
                iact.swap(icon, nm1);
            }

            // Stage 2: keep the objective (column mcon-1) as the last active.
            if stage == 2 && iact[*nact - 1] != mcon - 1 {
                if *nact <= 1 {
                    break;
                }
                let aact = active_cols(a, n, iact, *nact);
                qrexc(&aact, z, &mut zdota, n, *nact, *nact - 2);
                iact.swap(*nact - 2, *nact - 1);
                vmultc.swap(*nact - 2, *nact - 1);
            }

            if zdota[*nact - 1].is_nan() || zdota[*nact - 1].abs() <= eps * eps
            {
                break;
            }

            // Set SDIRN.
            if stage == 1 {
                let sa = dot(&sdirn, col(a, n, iact[*nact - 1]));
                let zc = col(z, n, *nact - 1);
                let coef = (sa + one) / zdota[*nact - 1];
                for r in 0..n {
                    sdirn[r] = sdirn[r] - coef * zc[r];
                }
            } else {
                let zc = col(z, n, *nact - 1);
                let coef = one / zdota[*nact - 1];
                for r in 0..n {
                    sdirn[r] = -coef * zc[r];
                }
            }
        } else {
            // --- Delete constraint iact[icon] from the active set. ---
            let aact = active_cols(a, n, iact, *nact);
            qrexc(&aact, z, &mut zdota, n, *nact, icon);
            // iact(icon:nact) <- [iact(icon+1:nact), iact(icon)]
            let saved = iact[icon];
            for k in icon..(*nact - 1) {
                iact[k] = iact[k + 1];
            }
            iact[*nact - 1] = saved;
            let savedc = vmultc[icon];
            for k in icon..(*nact - 1) {
                vmultc[k] = vmultc[k + 1];
            }
            vmultc[*nact - 1] = savedc;
            *nact -= 1;

            if stage == 2 && *nact == 0 {
                break;
            }
            if *nact > 0
                && (zdota[*nact - 1].is_nan()
                    || zdota[*nact - 1].abs() <= eps * eps)
            {
                break;
            }

            if stage == 1 {
                let zc = col(z, n, *nact).to_vec(); // Z(:, nact+1) in 1-based = index nact
                let coef = dot(&sdirn, &zc);
                for r in 0..n {
                    sdirn[r] = sdirn[r] - coef * zc[r];
                }
            } else {
                let zc = col(z, n, *nact - 1);
                let coef = one / zdota[*nact - 1];
                for r in 0..n {
                    sdirn[r] = -coef * zc[r];
                }
            }
        }

        // --- Step to the trust-region boundary / to zero out CVIOL. ---
        let dd = delta * delta - dot(&d[..n], &d[..n]);
        let ss = dot(&sdirn, &sdirn);
        let sd = dot(&sdirn, &d[..n]);
        if dd <= zero || ss <= eps * delta * delta || sd.is_nan() {
            break;
        }
        let sqrtd = (ss * dd + sd * sd)
            .sqrt()
            .max(sd.abs())
            .max((ss * dd).sqrt());
        let mut step = if sd > zero {
            dd / (sqrtd + sd)
        } else {
            (sqrtd - sd) / ss
        };
        if step <= zero || !step.is_finite() {
            break;
        }
        if stage == 1 {
            if isminor(cviol, step) {
                break;
            }
            step = step.min(cviol);
        }

        // DNEW = D + step·SDIRN; reduce CVIOL in stage 1.
        let mut dnew = vec![zero; n];
        for r in 0..n {
            dnew[r] = d[r] + step * sdirn[r];
        }
        if stage == 1 {
            let mut c = zero;
            for k in 0..*nact {
                c = c.max(dot(col(a, n, iact[k]), &dnew) - b[iact[k]]);
            }
            cviol = c;
        }

        // VMULTD: multipliers for DNEW (active), residuals (inactive).
        let mut vmultd = vec![zero; mcon];
        {
            let aact = active_cols(a, n, iact, *nact);
            let vd = lsqr(&aact, &dnew, z, &zdota, n, *nact);
            for k in 0..*nact {
                vmultd[k] = -vd[k];
            }
        }
        if stage == 2 && *nact >= 1 {
            vmultd[*nact - 1] = zero.max(vmultd[*nact - 1]);
        }
        // Inactive residuals: cvshift = cviol − (A(:,iact)·dnew − b(iact)).
        for k in *nact..mcon {
            let j = iact[k];
            let adn = dot(col(a, n, j), &dnew);
            let cvshift = cviol - (adn - b[j]);
            let dabs: Vec<F> = dnew.iter().map(|&v| v.abs()).collect();
            let aabs: Vec<F> = col(a, n, j).iter().map(|&v| v.abs()).collect();
            let cvsabs = dot(&dabs, &aabs) + b[j].abs() + cviol;
            vmultd[k] = if isminor(cvshift, cvsabs) {
                zero
            } else {
                cvshift
            };
        }

        // Fraction of the step from D to DNEW.
        let mut frac = one;
        icon = usize::MAX; // sentinel for "0" (Fortran icon = 0 → exit)
        let mut have = false;
        for k in 0..mcon {
            if vmultd[k] < zero {
                let f = vmultc[k] / (vmultc[k] - vmultd[k]);
                if !have || f < frac {
                    frac = f;
                    icon = k;
                    have = true;
                }
            }
        }
        if !have {
            frac = one;
        }

        // Update D, VMULTC, CVIOL.
        let dold = d[..n].to_vec();
        for r in 0..n {
            d[r] = (one - frac) * d[r] + frac * dnew[r];
        }
        if !d[..n].iter().map(|v| v.abs()).sum::<F>().is_finite() {
            d[..n].copy_from_slice(&dold);
            break;
        }
        for k in 0..mcon {
            vmultc[k] = zero.max((one - frac) * vmultc[k] + frac * vmultd[k]);
        }
        if stage == 1 {
            let mut c = zero;
            for k in 0..mcon {
                c = c.max(dot(col(a, n, k), &d[..n]) - b[k]);
            }
            cviol = c;
        }

        if !have {
            break;
        }
    }
}

/// Trust-region radius update (PRIMA `trrad`, COBYLA path). `ratio` is the
/// actual-to-predicted merit reduction; `dnorm` the step length.
pub(crate) fn trrad<F: Scalar>(
    delta_in: F,
    dnorm: F,
    eta1: F,
    eta2: F,
    gamma1: F,
    gamma2: F,
    ratio: F,
) -> F {
    if ratio <= eta1 {
        gamma1 * dnorm
    } else if ratio <= eta2 {
        (gamma1 * delta_in).max(dnorm)
    } else {
        (gamma1 * delta_in).max(gamma2 * dnorm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(d: &[f64]) -> f64 {
        dot(d, d).sqrt()
    }

    #[test]
    fn unconstrained_step_is_minus_delta_gradient_direction() {
        // m = 0: stage 1 leaves d = 0; stage 2 minimizes gᵀd s.t. ‖d‖ ≤ Δ,
        // so d = −Δ g/‖g‖.
        let n = 2;
        let g = vec![3.0_f64, 4.0];
        let delta = 2.0;
        let d = trstlp::<f64>(&[], n, 0, &[], delta, &g);
        assert!((norm(&d) - delta).abs() < 1e-9, "‖d‖ = {}", norm(&d));
        // direction parallel to −g
        let gn = norm(&g);
        assert!((d[0] - (-delta * g[0] / gn)).abs() < 1e-9);
        assert!((d[1] - (-delta * g[1] / gn)).abs() < 1e-9);
    }

    #[test]
    fn inactive_constraint_does_not_bind() {
        // One constraint e0·d ≤ 5 (far away); unconstrained optimum has ‖d‖=Δ=1
        // pointing along −g = (−1, 0), which satisfies d0 ≤ 5. So same as
        // unconstrained.
        let n = 2;
        let a = vec![1.0_f64, 0.0]; // column 0 = (1, 0)
        let b = vec![5.0_f64];
        let g = vec![1.0_f64, 0.0];
        let d = trstlp::<f64>(&a, n, 1, &b, 1.0, &g);
        assert!((d[0] - (-1.0)).abs() < 1e-9, "d = {:?}", d);
        assert!(d[1].abs() < 1e-9);
    }

    #[test]
    fn active_constraint_binds_the_step() {
        // Minimize gᵀd, g = (−1, 0) (wants d in +x), s.t. d0 ≤ 0.3, ‖d‖ ≤ 1.
        // Optimum: d0 = 0.3, d1 = 0 (push x as far as allowed; no y incentive).
        let n = 2;
        let a = vec![1.0_f64, 0.0]; // d0 ≤ 0.3
        let b = vec![0.3_f64];
        let g = vec![-1.0_f64, 0.0];
        let d = trstlp::<f64>(&a, n, 1, &b, 1.0, &g);
        assert!((d[0] - 0.3).abs() < 1e-9, "d = {:?}", d);
        assert!(d[1].abs() < 1e-9, "d = {:?}", d);
    }

    #[test]
    fn infeasible_center_reduced_in_stage1() {
        // Constraint d0 ≤ −0.5 with d starting at 0 is violated at the center
        // (0 ≤ −0.5 false). Stage 1 should move d to satisfy it within ‖d‖≤1.
        let n = 2;
        let a = vec![1.0_f64, 0.0];
        let b = vec![-0.5_f64];
        let g = vec![0.0_f64, 0.0];
        let d = trstlp::<f64>(&a, n, 1, &b, 1.0, &g);
        assert!(d[0] <= -0.5 + 1e-9, "d = {:?}", d);
    }
}
