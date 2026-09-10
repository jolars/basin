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

use super::linalg::{col, dot, dot_pair, hypotenuse, isminor, planerot};

/// Scratch reused by both LP stages and successive driver iterations.
pub(crate) struct TrstlpWork<F> {
    a_aug: Vec<F>,
    b_aug: Vec<F>,
    iact: Vec<usize>,
    vmultc: Vec<F>,
    z: Vec<F>,
    d: Vec<F>,
    scratch: TrstlpScratch<F>,
}

struct TrstlpScratch<F> {
    sdirn: Vec<F>,
    zdota: Vec<F>,
    zdasav: Vec<F>,
    cq: Vec<F>,
    cqa: Vec<F>,
    y: Vec<F>,
    dnew: Vec<F>,
    vmultd: Vec<F>,
    dold: Vec<F>,
}

impl<F: Scalar> TrstlpWork<F> {
    pub(crate) fn new(n: usize, m: usize) -> Self {
        let zero = F::zero();
        Self {
            a_aug: vec![zero; n * (m + 1)],
            b_aug: vec![zero; m + 1],
            iact: vec![0; m + 1],
            vmultc: vec![zero; m + 1],
            z: vec![zero; n * n],
            d: vec![zero; n],
            scratch: TrstlpScratch {
                sdirn: vec![zero; n],
                zdota: vec![zero; n],
                zdasav: vec![zero; n],
                cq: vec![zero; n],
                cqa: vec![zero; n],
                y: vec![zero; n],
                dnew: vec![zero; n],
                vmultd: vec![zero; m + 1],
                dold: vec![zero; n],
            },
        }
    }

    /// Solve `min gᵀd` with linearized constraints `Aᵀd ≤ b` and `‖d‖ ≤ Δ`.
    /// The returned step is valid until the next solve on this workspace.
    pub(crate) fn solve(
        &mut self,
        a: &[F],
        b: &[F],
        delta: F,
        g: &[F],
    ) -> &[F] {
        let n = self.d.len();
        // Keep small vector lengths visible through both LP stages so
        // the compiler can remove dynamic loop and slice bounds machinery.
        match n {
            1 => self.solve_with_dimension::<1>(a, b, delta, g),
            2 => self.solve_with_dimension::<2>(a, b, delta, g),
            3 => self.solve_with_dimension::<3>(a, b, delta, g),
            _ => self.solve_with_dimension::<0>(a, b, delta, g),
        }
    }

    fn solve_with_dimension<const N: usize>(
        &mut self,
        a: &[F],
        b: &[F],
        delta: F,
        g: &[F],
    ) -> &[F] {
        // Zero selects the dynamic dimension for the general path.
        let n = if N == 0 { self.d.len() } else { N };
        let m = self.b_aug.len() - 1;
        let mcon = m + 1;
        let Self {
            a_aug,
            b_aug,
            iact,
            vmultc,
            z,
            d,
            scratch,
        } = self;
        a_aug[..n * m].copy_from_slice(a);
        a_aug[n * m..].copy_from_slice(g);
        b_aug[..m].copy_from_slice(b);
        b_aug[m] = F::zero();

        // Huge columns are scaled to avoid floating-point exceptions.
        let big = F::from_f64(1.0e12).unwrap();
        let realmin = F::min_positive_value();
        for i in 0..mcon {
            let mx = col(a_aug, n, i)
                .iter()
                .map(|v| v.abs())
                .fold(F::zero(), F::max);
            if mx > big {
                let scal = ((F::one() + F::one()) * realmin).max(F::one() / mx);
                for r in 0..n {
                    a_aug[r + i * n] = a_aug[r + i * n] * scal;
                }
                b_aug[i] = b_aug[i] * scal;
            }
        }
        let mut nact = 0;
        for (stage, columns) in [(1, m), (2, mcon)] {
            trstlp_sub::<F, N>(
                stage, a_aug, n, columns, b_aug, delta, d, iact, vmultc, z,
                &mut nact, scratch,
            );
        }
        d
    }
}

#[cfg(test)]
fn trstlp<F: Scalar>(
    a: &[F],
    n: usize,
    m: usize,
    b: &[F],
    delta: F,
    g: &[F],
) -> Vec<F> {
    TrstlpWork::new(n, m).solve(a, b, delta, g).to_vec()
}

/// QR rank-one add (PRIMA `qradd_Rdiag`): attempt to append column `c` to the
/// active set, updating `z` (Q) and `zdota` (diag R). `nact` may grow by one.
fn qradd<F: Scalar>(
    c: &[F],
    z: &mut [F],
    zdota: &mut [F],
    nact: &mut usize,
    n: usize,
    cq: &mut [F],
    cqa: &mut [F],
) {
    for k in 0..n {
        (cq[k], cqa[k]) = dot_pair(c, col(z, n, k));
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
/// and `zdota`. Active columns are read through `iact` without packing them.
fn qrexc<F: Scalar>(
    a: &[F],
    iact: &[usize],
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
            planerot(zdota[k + 1], dot(col(z, n, k), col(a, n, iact[k + 1])));
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
        zdota[k] = dot(col(z, n, k), col(a, n, iact[k + 1]));
    }
    zdota[nact - 1] = dot(col(z, n, nact - 1), col(a, n, iact[i]));
}

/// Least-squares multipliers against the stored QR (PRIMA `lsqr_Rdiag`, the
/// `Q`+`Rdiag`-present branch). Active columns are indexed by `iact`;
/// `target` is the length-`n` right-hand side. Writes `nact` multipliers.
#[allow(clippy::too_many_arguments)]
fn lsqr<F: Scalar>(
    a: &[F],
    iact: &[usize],
    target: &[F],
    z: &[F],
    zdota: &[F],
    n: usize,
    nact: usize,
    x: &mut [F],
    y: &mut [F],
) {
    y.copy_from_slice(target);
    for i in (0..nact).rev() {
        let zi = col(z, n, i);
        let (yq, yqa) = dot_pair(y, zi);
        if isminor(yq, yqa) {
            x[i] = F::zero();
        } else {
            x[i] = yq / zdota[i];
            let aci = col(a, n, iact[i]);
            for r in 0..n {
                y[r] = y[r] - x[i] * aci[r];
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn trstlp_sub<F: Scalar, const N: usize>(
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
    scratch: &mut TrstlpScratch<F>,
) {
    let n = if N == 0 { n } else { N };
    let TrstlpScratch {
        sdirn,
        zdota,
        zdasav,
        cq,
        cqa,
        y,
        dnew,
        vmultd,
        dold,
    } = scratch;
    let zero = F::zero();
    let one = F::one();
    let eps = F::epsilon();
    let realmax = F::max_value();
    // `m` (1-based in PRIMA): number of "real" constraints, all of them in
    // stage 1, all-but-the-objective in stage 2.
    let m_real = if stage == 1 { mcon } else { mcon - 1 };

    let mut icon: usize;
    sdirn.fill(zero);

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
        z.fill(zero);
        for k in 0..n {
            z[k + k * n] = one;
        }
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

    zdota.fill(zero);
    for k in 0..*nact {
        zdota[k] = dot(col(z, n, k), col(a, n, iact[k]));
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
            zdasav[..*nact].copy_from_slice(&zdota[..*nact]);
            let nactsav = *nact;
            qradd(col(a, n, iact[icon]), z, zdota, nact, n, cq, cqa);

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
                vmultd.fill(zero);
                lsqr(
                    a,
                    iact,
                    col(a, n, iact[icon]),
                    z,
                    zdasav,
                    n,
                    *nact,
                    vmultd,
                    y,
                );
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
                qrexc(a, iact, z, zdota, n, *nact, *nact - 2);
                iact.swap(*nact - 2, *nact - 1);
                vmultc.swap(*nact - 2, *nact - 1);
            }

            if zdota[*nact - 1].is_nan() || zdota[*nact - 1].abs() <= eps * eps
            {
                break;
            }

            // Set SDIRN.
            if stage == 1 {
                let sa = dot(sdirn, col(a, n, iact[*nact - 1]));
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
            qrexc(a, iact, z, zdota, n, *nact, icon);
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
                let zc = col(z, n, *nact);
                let coef = dot(sdirn, zc);
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
        let ss = dot(sdirn, sdirn);
        let sd = dot(sdirn, &d[..n]);
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
        for r in 0..n {
            dnew[r] = d[r] + step * sdirn[r];
        }
        if stage == 1 {
            let mut c = zero;
            for k in 0..*nact {
                c = c.max(dot(col(a, n, iact[k]), dnew) - b[iact[k]]);
            }
            cviol = c;
        }

        // VMULTD: multipliers for DNEW (active), residuals (inactive).
        vmultd.fill(zero);
        lsqr(a, iact, dnew, z, zdota, n, *nact, vmultd, y);
        for v in &mut vmultd[..*nact] {
            *v = -*v;
        }
        if stage == 2 && *nact >= 1 {
            vmultd[*nact - 1] = zero.max(vmultd[*nact - 1]);
        }
        // Inactive residuals: cvshift = cviol − (A(:,iact)·dnew − b(iact)).
        for k in *nact..mcon {
            let j = iact[k];
            let (adn, adn_abs) = dot_pair(col(a, n, j), dnew);
            let cvshift = cviol - (adn - b[j]);
            let cvsabs = adn_abs + b[j].abs() + cviol;
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
        dold.copy_from_slice(&d[..n]);
        for r in 0..n {
            d[r] = (one - frac) * d[r] + frac * dnew[r];
        }
        if !d[..n].iter().map(|v| v.abs()).sum::<F>().is_finite() {
            d[..n].copy_from_slice(dold);
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

    #[test]
    fn reused_lp_handles_scaled_dependent_constraints() {
        fn check<F: Scalar>() {
            let f = |x| F::from_f64(x).unwrap();
            let mut work = TrstlpWork::new(2, 3);
            for scale in [1.0, 1e14, 1e-8, 1.0] {
                // Duplicate rows exercise rank rejection. The zero row leaves
                // the same feasible cap, including after a huge-column solve.
                let a =
                    [f(scale), f(0.0), f(2.0 * scale), f(0.0), f(0.0), f(0.0)];
                let b = [f(-0.25 * scale), f(-0.5 * scale), f(1.0)];
                let g = [f(0.0), f(scale)];
                let d = work.solve(&a, &b, f(1.0), &g);
                assert!((d[0].to_f64().unwrap() + 0.25).abs() < 2e-6);
                assert!(
                    (d[1].to_f64().unwrap() + 0.9375_f64.sqrt()).abs() < 2e-6
                );
            }
        }
        check::<f64>();
        check::<f32>();
    }

    #[test]
    fn reused_lp_recovers_after_nonfinite_models() {
        let mut work = TrstlpWork::new(2, 1);
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0] {
            let d = work.solve(&[value, 0.0], &[0.3], 1.0, &[-1.0, 0.0]);
            assert!(d.iter().all(|x| x.is_finite()));
            assert!(norm(d) <= 1.0 + 1e-12);
            // A valid solve must not inherit an active set or scaled model
            // from a failed or non-finite predecessor.
            let d = work.solve(&[1.0, 0.0], &[0.3], 1.0, &[-1.0, 0.0]);
            assert!((d[0] - 0.3).abs() < 1e-12);
            assert!(d[1].abs() < 1e-12);
        }
    }

    #[test]
    fn small_dimensional_specializations_match_dynamic_path() {
        fn check<F: Scalar>() {
            let f = |x| F::from_f64(x).unwrap();
            for (n, m) in (1..=3).flat_map(|n| [0, 1, 4, 9].map(|m| (n, m))) {
                let mut specialized = TrstlpWork::new(n, m);
                let mut dynamic = TrstlpWork::new(n, m);
                for sample in 0..48 {
                    let scale = [1e-8, 1.0, 1e14][sample % 3];
                    let value = |i: usize| {
                        scale * (((i * 11 + sample * 7) % 23) as f64 - 11.0)
                            / 8.0
                    };
                    let a: Vec<_> = (0..n * m).map(|i| f(value(i))).collect();
                    let b: Vec<_> = (0..m).map(|i| f(value(i + 3))).collect();
                    let g: Vec<_> =
                        (0..n).map(|i| f(value(5 + 4 * i))).collect();
                    let delta = f([1e-9, 0.5, 2.0][sample % 3]);
                    let expected =
                        dynamic.solve_with_dimension::<0>(&a, &b, delta, &g);
                    let actual = specialized.solve(&a, &b, delta, &g);
                    for (x, y) in actual.iter().zip(expected) {
                        assert_eq!(
                            x.to_f64().unwrap().to_bits(),
                            y.to_f64().unwrap().to_bits()
                        );
                    }
                }
            }
        }
        check::<f64>();
        check::<f32>();
    }
}
