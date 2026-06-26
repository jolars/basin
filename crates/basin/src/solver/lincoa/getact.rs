//! GETACT: LINCOA's active-set step and the incremental QR factorization of the
//! active linear-constraint normals (Powell 2015, §3; PRIMA `getact.f90`).
//!
//! GETACT solves the linearly-constrained projected problem (Powell 2015,
//! eqs. 3.3–3.5)
//!
//! ```text
//! min ‖d + g‖   subject to   a_jᵀ d ≤ 0   for j in the nearly-active set,
//! ```
//!
//! whose solution `psd` is the projected steepest-descent direction that drives
//! the truncated-CG of [`trstep`](super::trstep). It uses the Goldfarb–Idnani
//! (1983) active-set method and maintains a QR factorization of the active
//! constraint normals.
//!
//! The active set and its `Q R` factorization are **warm-started** (GETACT
//! reuses them within a TRSTEP solve and across driver iterations), so they live
//! in [`ActiveSetQr`], owned by the driver ([`LincoaWork`](super::driver)) and
//! threaded by `&mut` into the free `trstep`/`getact` functions rather than
//! hidden behind the immutable [`TrustRegionSubproblem`](crate::solver::powell::TrustRegionSubproblem)
//! seam (which stays unchanged for NEWUOA and BOBYQA). This mirrors PRIMA, where
//! `iact`/`nact`/`qfac`/`rfac` are `lincob` locals threaded as inout.
//!
//! The QR is held as bespoke `Vec<F>` scratch (column-major `n × n` `Q` and
//! `R`), maintained by Givens rotations on add or drop of an active normal
//! (`qradd_Rfull`/`qrexc_Rfull` in PRIMA `powalg.f90`): there is no QR in
//! basin's `linalg` tier and this active-set factorization is solver-internal,
//! exactly the precedent set by the model's `H`-algebra.

use crate::core::math::Scalar;

/// Incremental QR factorization of the active linear-constraint normals,
/// warm-started across TRSTEP and driver iterations.
///
/// LINCOA normalizes every constraint normal to unit length at init, so the
/// columns of `A` referenced by `iact[..nact]` are unit vectors; `qfac`/`rfac`
/// factor the `n × nact` matrix of those active normals: column-major `n × n`
/// `Q` (orthogonal) and `R` (upper-triangular, positive diagonal on the leading
/// `nact × nact` block), with `Q[:, ..nact] · R[..nact, ..nact] = A[:, iact]`.
pub(crate) struct ActiveSetQr<F = f64> {
    /// Number of variables `n`.
    pub(crate) n: usize,
    /// Indices of the active constraints; the first [`nact`](Self::nact) entries
    /// are the active set, the rest scratch. Length `m`.
    pub(crate) iact: Vec<usize>,
    /// Size of the active set (`≤ n`).
    pub(crate) nact: usize,
    /// Orthogonal factor `Q` (`n × n`, column-major: `Q[r,c] = qfac[r + c*n]`).
    pub(crate) qfac: Vec<F>,
    /// Upper-triangular factor `R` (`n × n`, column-major); only the leading
    /// `nact × nact` block is meaningful.
    pub(crate) rfac: Vec<F>,
}

impl<F: Scalar> ActiveSetQr<F> {
    /// Cold start: empty active set, `Q = I`, `R = 0` (PRIMA `lincob.f90`
    /// initialization of `nact`/`qfac`/`rfac`/`iact`).
    pub(crate) fn new(n: usize, m: usize) -> Self {
        let mut qfac = vec![F::zero(); n * n];
        for i in 0..n {
            qfac[i + i * n] = F::one();
        }
        Self {
            n,
            iact: (0..m).collect(),
            nact: 0,
            qfac,
            rfac: vec![F::zero(); n * n],
        }
    }

    /// Overwrite `qfac` with the identity (PRIMA `qfac = eye(n)`).
    fn set_qfac_identity(&mut self) {
        let n = self.n;
        for x in self.qfac.iter_mut() {
            *x = F::zero();
        }
        for i in 0..n {
            self.qfac[i + i * n] = F::one();
        }
    }
}

/// `TINYCV` (PRIMA `consts.F90`): `2^-200`, the tiny constraint value a freshly
/// deactivated constraint's residual is floored to.
fn tinycv<F: Scalar>() -> F {
    F::from_f64(2.0).expect("2.0 representable").powi(-200)
}

/// Fortran `sign(a, b)`: `|a|` if `b ≥ 0` (including `+0`), else `-|a|`.
fn fsign<F: Scalar>(a: F, b: F) -> F {
    if b >= F::zero() { a.abs() } else { -a.abs() }
}

/// Givens rotation `(c, s)` such that `G·[x0, x1]ᵀ = [r, 0]ᵀ` with
/// `G = [[c, s], [-s, c]]` (PRIMA `linalg.f90::planerot`). Kept orthogonal for
/// the NaN/Inf/zero edge cases.
fn planerot<F: Scalar>(x0: F, x1: F) -> (F, F) {
    let (zero, one) = (F::zero(), F::one());
    if x0.is_nan() || x1.is_nan() {
        return (one, zero);
    }
    if x0.is_infinite() && x1.is_infinite() {
        let inv_sqrt2 = F::from_f64(2.0).expect("2.0 representable").sqrt().recip();
        return (fsign(inv_sqrt2, x0), fsign(inv_sqrt2, x1));
    }
    if x0.abs() <= zero && x1.abs() <= zero {
        return (one, zero);
    }
    let eps = F::epsilon();
    if x1.abs() <= eps * x0.abs() {
        return (fsign(one, x0), zero);
    }
    if x0.abs() <= eps * x1.abs() {
        return (zero, fsign(one, x1));
    }
    let r = (x0 * x0 + x1 * x1).sqrt();
    (x0 / r, x1 / r)
}

/// Append the unit column `c` (length `n`) as the new last active column,
/// updating the QR and incrementing `nact` (PRIMA `qradd_Rfull`). `Q` is `n × n`
/// column-major, `R` is `n × n` column-major. It is assumed `c` is not in the
/// span of the existing active columns (always true in LINCOA).
fn qradd<F: Scalar>(c: &[F], qfac: &mut [F], rfac: &mut [F], nact: &mut usize, n: usize) {
    let na = *nact;
    // cq = Qᵀ c.
    let mut cq = vec![F::zero(); n];
    for (k, cqk) in cq.iter_mut().enumerate() {
        let mut s = F::zero();
        for r in 0..n {
            s = s + c[r] * qfac[r + k * n];
        }
        *cqk = s;
    }
    // Rotate columns [na .. n-1] of Q so they stay orthogonal to c, zeroing
    // cq[k+1] into cq[k] for k = n-2 down to na.
    let mut kk = n as isize - 2;
    while kk >= na as isize {
        let k = kk as usize;
        if cq[k + 1].abs() > F::zero() {
            let (cc, ss) = planerot(cq[k], cq[k + 1]);
            for r in 0..n {
                let a = qfac[r + k * n];
                let b = qfac[r + (k + 1) * n];
                qfac[r + k * n] = cc * a + ss * b;
                qfac[r + (k + 1) * n] = cc * b - ss * a;
            }
            cq[k] = (cq[k] * cq[k] + cq[k + 1] * cq[k + 1]).sqrt();
        }
        kk -= 1;
    }
    // New R column: R[0..na, na] = Qᵀ c over the (unchanged) first na columns.
    for i in 0..na {
        let mut s = F::zero();
        for r in 0..n {
            s = s + c[r] * qfac[r + i * n];
        }
        rfac[i + na * n] = s;
    }
    // Keep the new diagonal entry positive.
    if cq[na] < F::zero() {
        for r in 0..n {
            qfac[r + na * n] = -qfac[r + na * n];
        }
    }
    rfac[na + na * n] = cq[na].abs();
    *nact = na + 1;
}

/// Cyclically move active column `icon` to the last (`ncols-1`) position,
/// re-triangularizing the QR by Givens rotations (PRIMA `qrexc_Rfull`).
/// `ncols` is the current active count; `n` the matrix dimension.
fn qrexc<F: Scalar>(qfac: &mut [F], rfac: &mut [F], n: usize, ncols: usize, icon: usize) {
    if icon + 1 >= ncols {
        return;
    }
    for kk in icon..(ncols - 1) {
        let r_kp1_kp1 = rfac[(kk + 1) + (kk + 1) * n];
        let r_k_kp1 = rfac[kk + (kk + 1) * n];
        let (cc, ss) = planerot(r_kp1_kp1, r_k_kp1);
        let hypt = r_kp1_kp1.hypot(r_k_kp1);
        // Q(:, [kk, kk+1]) = Q(:, [kk+1, kk]) · Gᵀ.
        for r in 0..n {
            let qk = qfac[r + kk * n];
            let qk1 = qfac[r + (kk + 1) * n];
            qfac[r + kk * n] = cc * qk1 + ss * qk;
            qfac[r + (kk + 1) * n] = cc * qk - ss * qk1;
        }
        // R([kk, kk+1], kk..ncols) = G · R([kk+1, kk], kk..ncols).
        for j in kk..ncols {
            let rkj = rfac[kk + j * n];
            let rk1j = rfac[(kk + 1) + j * n];
            rfac[kk + j * n] = cc * rk1j + ss * rkj;
            rfac[(kk + 1) + j * n] = cc * rkj - ss * rk1j;
        }
        // Swap columns kk, kk+1 of R for rows 0..=kk+1.
        for r in 0..=(kk + 1) {
            rfac.swap(r + kk * n, r + (kk + 1) * n);
        }
        rfac[kk + kk * n] = hypt;
        rfac[(kk + 1) + kk * n] = F::zero();
    }
}

/// Solve the upper-triangular system `R[..k, ..k] · x = b` (PRIMA `solve` on the
/// `istriu` branch). `R` is `n × n` column-major; `k = b.len()`.
fn rsolve_upper<F: Scalar>(rfac: &[F], n: usize, b: &[F]) -> Vec<F> {
    let k = b.len();
    let mut x = vec![F::zero(); k];
    for i in (0..k).rev() {
        let mut s = b[i];
        for j in (i + 1)..k {
            s = s - rfac[i + j * n] * x[j];
        }
        x[i] = s / rfac[i + i * n];
    }
    x
}

/// Least-squares multipliers `vlam = R[..nact,..nact]⁻¹ (Qᵀ g)` over the active
/// columns (PRIMA `lsqr_Rfull`).
fn lsqr_mult<F: Scalar>(g: &[F], qfac: &[F], rfac: &[F], n: usize, nact: usize) -> Vec<F> {
    let mut x = vec![F::zero(); nact];
    for (i, xi) in x.iter_mut().enumerate() {
        let mut s = F::zero();
        for r in 0..n {
            s = s + g[r] * qfac[r + i * n];
        }
        *xi = s;
    }
    for i in (0..nact).rev() {
        for j in (i + 1)..nact {
            x[i] = x[i] - rfac[i + j * n] * x[j];
        }
        x[i] = x[i] / rfac[i + i * n];
    }
    x
}

/// Add constraint `l` (gradient `c`) to the active set (PRIMA `addact`).
fn addact<F: Scalar>(
    l: usize,
    c: &[F],
    warm: &mut ActiveSetQr<F>,
    resact: &mut [F],
    resnew: &mut [F],
    vlam: &mut [F],
) {
    let n = warm.n;
    qradd(c, &mut warm.qfac, &mut warm.rfac, &mut warm.nact, n);
    let na = warm.nact; // already incremented
    warm.iact[na - 1] = l;
    resact[na - 1] = resnew[l];
    resnew[l] = F::zero();
    vlam[na - 1] = F::zero();
}

/// Delete the `icon`-th active constraint (PRIMA `delact`), reducing `nact`.
fn delact<F: Scalar>(
    icon: usize,
    warm: &mut ActiveSetQr<F>,
    resact: &mut [F],
    resnew: &mut [F],
    vlam: &mut [F],
) {
    let n = warm.n;
    let nact = warm.nact;
    qrexc(&mut warm.qfac, &mut warm.rfac, n, nact, icon);
    // Rotate iact/resact/vlam: position icon goes to the (now last) slot.
    let li = warm.iact[icon];
    let ri = resact[icon];
    let vi = vlam[icon];
    for t in icon..(nact - 1) {
        warm.iact[t] = warm.iact[t + 1];
        resact[t] = resact[t + 1];
        vlam[t] = vlam[t + 1];
    }
    warm.iact[nact - 1] = li;
    resact[nact - 1] = ri;
    vlam[nact - 1] = vi;
    resnew[warm.iact[nact - 1]] = resact[nact - 1].max(tinycv());
    warm.nact = nact - 1;
}

/// `psd ← −Q[:, nact..n] (Q[:, nact..n]ᵀ g)`: minus the projection of `g` onto
/// the range of the inactive `Q` columns (PRIMA's projection of `-G`).
fn project_neg_g<F: Scalar>(g: &[F], qfac: &[F], n: usize, nact: usize, psd: &mut [F]) {
    let zero = F::zero();
    for p in psd.iter_mut() {
        *p = zero;
    }
    for col in nact..n {
        let mut coeff = zero;
        for r in 0..n {
            coeff = coeff + g[r] * qfac[r + col * n];
        }
        for r in 0..n {
            psd[r] = psd[r] - coeff * qfac[r + col * n];
        }
    }
}

/// GETACT: compute the projected steepest-descent direction `psd` for the
/// linearly-constrained trust-region subproblem and (warm-)update the active set
/// and its QR factorization (Powell 2015, §3; PRIMA `getact`).
///
/// - `amat`: `n × m` column-major unit constraint normals (`a_j = amat[.. , j]`).
/// - `g`: `∇Q` at the current CG iterate (length `n`).
/// - `delta`: trust-region radius (defines the nearly-active band `0.2·Δ`).
/// - `resact`/`resnew`: TRSTEP-level residual scratch (length `m`); kept up to
///   date here (only permuted, values preserved across add or drop).
/// - `psd`: output direction (length `n`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn getact<F: Scalar>(
    amat: &[F],
    n: usize,
    m: usize,
    delta: F,
    g: &[F],
    warm: &mut ActiveSetQr<F>,
    resact: &mut [F],
    resnew: &mut [F],
    psd: &mut [F],
) {
    let (zero, one) = (F::zero(), F::one());
    let two = F::from_f64(2.0).expect("2.0 representable");
    let ten = F::from_f64(10.0).expect("10.0 representable");
    let eps = F::epsilon();
    let realmax = F::max_value();

    // Quick return when there are no constraints.
    if m == 0 {
        warm.nact = 0;
        warm.set_qfac_identity();
        for i in 0..n {
            psd[i] = -g[i];
        }
        return;
    }

    let gg: F = (0..n).fold(zero, |a, i| a + g[i] * g[i]);
    let tdel = F::from_f64(0.2).expect("0.2 representable") * delta;
    if warm.nact == 0 {
        warm.set_qfac_identity();
    }

    let mut vlam = vec![zero; n];

    // Remove active constraints whose residuals exceed TDEL.
    let mut icon = warm.nact;
    while icon >= 1 {
        if resact[icon - 1] > tdel {
            delact(icon - 1, warm, resact, resnew, &mut vlam);
        }
        icon -= 1;
    }

    // Remove active constraints with nonnegative Lagrange multipliers; set the
    // surviving multipliers.
    while warm.nact > 0 {
        let v = lsqr_mult(g, &warm.qfac, &warm.rfac, n, warm.nact);
        vlam[..warm.nact].copy_from_slice(&v);
        if !(0..warm.nact).any(|i| vlam[i] >= zero) {
            break;
        }
        // icon = last index with vlam >= 0.
        let icon = (0..warm.nact)
            .filter(|&i| vlam[i] >= zero)
            .next_back()
            .unwrap();
        delact(icon, warm, resact, resnew, &mut vlam);
    }

    // Main Goldfarb–Idnani iteration.
    let mut psdsav = vec![zero; n];
    let mut ddsav = two * gg;
    let maxiter = (2 * (m + n)).max(m);

    for _iter in 0..maxiter {
        if warm.nact >= n {
            for p in psd.iter_mut() {
                *p = zero;
            }
            break;
        }

        project_neg_g(g, &warm.qfac, n, warm.nact, psd);
        let dd: F = (0..n).fold(zero, |a, i| a + psd[i] * psd[i]);
        let dnorm = dd.sqrt();

        if dnorm <= eps || dnorm.is_nan() {
            break;
        }
        if dd >= ddsav {
            for p in psd.iter_mut() {
                *p = zero;
            }
            break;
        }
        let psd_dot_g: F = (0..n).fold(zero, |a, i| a + psd[i] * g[i]);
        let abs_sum: F = (0..n).fold(zero, |a, i| a + psd[i].abs());
        if psd_dot_g > zero || !abs_sum.is_finite() {
            psd.copy_from_slice(&psdsav);
            break;
        }

        psdsav.copy_from_slice(psd);
        ddsav = dd;

        // Pick the most violated nearly-active constraint, if any.
        // apsd[j] = psdᵀ a_j.
        let mut l: Option<usize> = None;
        let mut violmx = -realmax;
        let scale = dnorm / delta;
        for j in 0..m {
            let apsd_j = (0..n).fold(zero, |a, r| a + psd[r] * amat[r + j * n]);
            if resnew[j] > zero
                && resnew[j] <= tdel
                && apsd_j > scale * resnew[j]
                && apsd_j > violmx
            {
                violmx = apsd_j;
                l = Some(j);
            }
        }

        // Terminate if no violation, or a positive VIOLMX is at the rounding
        // level relative to the active residuals.
        let act_inf = (0..warm.nact).fold(zero, |a, ic| {
            let j = warm.iact[ic];
            let apsd_j = (0..n).fold(zero, |b, r| b + psd[r] * amat[r + j * n]);
            a.max(apsd_j.abs())
        });
        let l = match l {
            Some(l) if violmx > (eps * dnorm).max(ten * act_inf) => l,
            _ => break,
        };

        // Add constraint L; column = a_L.
        let c: Vec<F> = (0..n).map(|r| amat[r + l * n]).collect();
        addact(l, &c, warm, resact, resnew, &mut vlam);

        // Drag VIOLMX down to zero, deactivating constraints as multipliers turn
        // nonnegative (PRIMA's VMU loop).
        while violmx > zero && warm.nact > 0 {
            let na = warm.nact;
            let mut rhs = vec![zero; na];
            rhs[na - 1] = one / warm.rfac[(na - 1) + (na - 1) * n];
            let vmu = rsolve_upper(&warm.rfac, n, &rhs);

            let mut vmult = violmx;
            for ic in 0..na {
                if vmu[ic] < zero && vlam[ic] < zero {
                    let f = vlam[ic] / vmu[ic];
                    if f < vmult {
                        vmult = f;
                    }
                }
            }
            // icon = last 1-based index with frac <= vmult (0 = none).
            let mut icon1 = 0usize;
            for ic in 0..na {
                let f = if vmu[ic] < zero && vlam[ic] < zero {
                    vlam[ic] / vmu[ic]
                } else {
                    realmax
                };
                if f <= vmult {
                    icon1 = ic + 1;
                }
            }

            violmx = (violmx - vmult).max(zero);
            for ic in 0..na {
                vlam[ic] = vlam[ic] - vmult * vmu[ic];
            }
            if icon1 >= 1 && icon1 <= na {
                vlam[icon1 - 1] = zero;
            }

            // Reduce the active set so all surviving multipliers are negative.
            let mut ic = warm.nact;
            while ic >= 1 {
                if vlam[ic - 1] >= zero {
                    delact(ic - 1, warm, resact, resnew, &mut vlam);
                }
                ic -= 1;
            }
        }

        if warm.nact == 0 {
            break;
        }
    }

    // NACT may be 0 here; reset to the unconstrained projection.
    if warm.nact == 0 {
        warm.set_qfac_identity();
        for i in 0..n {
            psd[i] = -g[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at<F: Copy>(a: &[F], r: usize, c: usize, n: usize) -> F {
        a[r + c * n]
    }

    /// `QᵀQ == I` to tolerance.
    fn assert_orthonormal(qfac: &[f64], n: usize) {
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for r in 0..n {
                    s += at(qfac, r, i, n) * at(qfac, r, j, n);
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((s - want).abs() < 1e-12, "QᵀQ[{i},{j}] = {s}");
            }
        }
    }

    /// `R` upper-triangular with positive diagonal on the leading block.
    fn assert_upper_tri_pos(rfac: &[f64], n: usize, nact: usize) {
        for c in 0..nact {
            for r in 0..n {
                if r > c {
                    assert!(at(rfac, r, c, n).abs() < 1e-12, "R[{r},{c}] not zero");
                }
            }
            assert!(at(rfac, c, c, n) > 0.0, "R[{c},{c}] not positive");
        }
    }

    /// `Q[:, ..nact] · R[..nact, ..nact]` reproduces the active normals.
    fn assert_qr_reconstructs(warm: &ActiveSetQr<f64>, amat: &[f64], n: usize) {
        for (ic, &j) in warm.iact[..warm.nact].iter().enumerate() {
            for r in 0..n {
                let mut s = 0.0;
                for k in 0..warm.nact {
                    s += at(&warm.qfac, r, k, n) * at(&warm.rfac, k, ic, n);
                }
                assert!(
                    (s - at(amat, r, j, n)).abs() < 1e-10,
                    "QR col {ic} (constraint {j}) row {r}: {s} vs {}",
                    at(amat, r, j, n)
                );
            }
        }
    }

    /// `qradd` builds a correct QR of a sequence of independent unit columns.
    #[test]
    fn qradd_builds_valid_qr() {
        let n = 3;
        // Three independent (non-orthogonal) unit normals.
        let cols = [[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.6, 0.8]];
        let mut warm = ActiveSetQr::<f64>::new(n, 3);
        let mut amat = vec![0.0; n * 3];
        for (j, c) in cols.iter().enumerate() {
            for r in 0..n {
                amat[r + j * n] = c[r];
            }
            qradd(c, &mut warm.qfac, &mut warm.rfac, &mut warm.nact, n);
            warm.iact[warm.nact - 1] = j;
        }
        assert_eq!(warm.nact, 3);
        assert_orthonormal(&warm.qfac, n);
        assert_upper_tri_pos(&warm.rfac, n, 3);
        assert_qr_reconstructs(&warm, &amat, n);
    }

    /// `qrexc` preserves a valid QR of the remaining columns after a deletion.
    #[test]
    fn qrexc_preserves_qr_after_deletion() {
        let n = 3;
        let cols = [[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.6, 0.8]];
        let mut warm = ActiveSetQr::<f64>::new(n, 3);
        let mut amat = vec![0.0; n * 3];
        for (j, c) in cols.iter().enumerate() {
            for r in 0..n {
                amat[r + j * n] = c[r];
            }
            qradd(c, &mut warm.qfac, &mut warm.rfac, &mut warm.nact, n);
            warm.iact[warm.nact - 1] = j;
        }
        // Delete the middle active column (icon = 1): it cycles to the end and nact drops.
        let mut resact = vec![0.0; 3];
        let mut resnew = vec![1.0; 3];
        let mut vlam = vec![0.0; n];
        delact(1, &mut warm, &mut resact, &mut resnew, &mut vlam);
        assert_eq!(warm.nact, 2);
        assert_eq!(&warm.iact[..2], &[0, 2]); // constraint 1 was dropped
        assert_orthonormal(&warm.qfac, n);
        assert_upper_tri_pos(&warm.rfac, n, 2);
        assert_qr_reconstructs(&warm, &amat, n);
    }

    /// With no constraints, GETACT returns the steepest-descent direction `-g`.
    #[test]
    fn getact_unconstrained_is_neg_g() {
        let n = 2;
        let g = [1.0, -2.0];
        let mut warm = ActiveSetQr::<f64>::new(n, 0);
        let mut psd = vec![0.0; n];
        getact(&[], n, 0, 1.0, &g, &mut warm, &mut [], &mut [], &mut psd);
        assert_eq!(psd, vec![-1.0, 2.0]);
        assert_eq!(warm.nact, 0);
    }

    /// A gradient pushing into a single nearly-active constraint activates it,
    /// and `psd` becomes the projection onto that constraint's null space.
    #[test]
    fn getact_activates_single_constraint() {
        let n = 2;
        let m = 1;
        // Constraint x ≤ b, normal a = (1, 0). Gradient g = (-1, 0): -g = (1, 0)
        // points straight at the constraint, so it must become active and psd → 0.
        let amat = vec![1.0, 0.0];
        let g = [-1.0, 0.0];
        let mut warm = ActiveSetQr::<f64>::new(n, m);
        let mut resact = vec![0.0; m];
        let mut resnew = vec![1e-6; m]; // nearly active (small positive residual)
        let mut psd = vec![0.0; n];
        getact(
            &amat,
            n,
            m,
            1.0,
            &g,
            &mut warm,
            &mut resact,
            &mut resnew,
            &mut psd,
        );
        assert_eq!(warm.nact, 1);
        assert_eq!(warm.iact[0], 0);
        assert!(
            psd[0].abs() < 1e-12 && psd[1].abs() < 1e-12,
            "psd = {psd:?}"
        );
    }

    /// GETACT postconditions on a 3-D problem with two nearly-active constraints:
    /// `psd` is orthogonal to the active normals, a descent direction, and
    /// `‖psd‖² ≤ 2‖g‖²`; the QR stays consistent.
    #[test]
    fn getact_postconditions_hold() {
        let n = 3;
        let m = 2;
        // a1 = (1,0,0), a2 = (0,1,0), both nearly active; g pushes into both.
        let amat = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let g = [-1.0, -1.0, -0.5];
        let mut warm = ActiveSetQr::<f64>::new(n, m);
        let mut resact = vec![0.0; m];
        let mut resnew = vec![1e-4; m];
        let mut psd = vec![0.0; n];
        getact(
            &amat,
            n,
            m,
            1.0,
            &g,
            &mut warm,
            &mut resact,
            &mut resnew,
            &mut psd,
        );

        let gg: f64 = g.iter().map(|x| x * x).sum();
        let dd: f64 = psd.iter().map(|x| x * x).sum();
        let psd_dot_g: f64 = psd.iter().zip(&g).map(|(a, b)| a * b).sum();
        assert!(dd <= 2.0 * gg + 1e-12, "‖psd‖² = {dd}, 2gg = {}", 2.0 * gg);
        assert!(psd_dot_g <= 1e-10, "psdᵀg = {psd_dot_g}");
        // psd ⟂ each active normal.
        for ic in 0..warm.nact {
            let j = warm.iact[ic];
            let ap: f64 = (0..n).map(|r| psd[r] * amat[r + j * n]).sum();
            assert!(ap.abs() < 1e-10, "psdᵀa_{j} = {ap}");
        }
        assert_orthonormal(&warm.qfac, n);
        // Expected: both constraints active, psd ≈ (0, 0, 0.5).
        assert_eq!(warm.nact, 2);
        assert!((psd[2] - 0.5).abs() < 1e-10, "psd = {psd:?}");
    }
}
