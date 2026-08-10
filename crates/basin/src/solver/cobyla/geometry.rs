//! COBYLA geometry management (PRIMA `geometry.f90`): `assess_geo`,
//! `setdrop_geo`, `setdrop_tr`, `geostep`.
//!
//! `assess_geo` tests the acceptability of the simplex (Powell 1994 eq. 14) via
//! the per-vertex face distances `σ` (`vsig`) and edge lengths `η` (`veta`).
//! `setdrop_tr`/`setdrop_geo` pick which vertex to replace after a
//! trust-region or geometry step; `geostep` builds the geometry-improving step
//! (eq. 15–17), choosing its sign by the linear merit model.

use crate::core::math::Scalar;

use super::model::{build_a, build_g};
use super::update::NO_DROP;

/// Per-vertex face distances `σ[j] = 1/‖simi(j, :)‖` (j = 0..n).
fn vsig<F: Scalar>(simi: &[F], n: usize) -> Vec<F> {
    (0..n)
        .map(|j| {
            let s: F = (0..n).map(|l| simi[j + l * n] * simi[j + l * n]).sum();
            F::one() / s.sqrt()
        })
        .collect()
}

/// Per-vertex edge lengths `η[j] = ‖sim(:, j)‖` (j = 0..n, the displacements).
fn veta<F: Scalar>(sim: &[F], n: usize) -> Vec<F> {
    (0..n)
        .map(|j| {
            let s: F = (0..n).map(|r| sim[r + j * n] * sim[r + j * n]).sum();
            s.sqrt()
        })
        .collect()
}

/// Does the simplex have acceptable geometry (eq. 14)? PRIMA `assess_geo`.
pub(crate) fn assess_geo<F: Scalar>(
    delta: F,
    factor_alpha: F,
    factor_beta: F,
    sim: &[F],
    simi: &[F],
    n: usize,
) -> bool {
    let vs = vsig(simi, n);
    let ve = veta(sim, n);
    vs.iter().all(|&s| s >= factor_alpha * delta)
        && ve.iter().all(|&e| e <= factor_beta * delta)
}

/// Pick the vertex to drop for a geometry-improving step (eq. 15–16). Returns a
/// 0-based index in `0..n`, or [`NO_DROP`] on all-`NaN` (a bug). PRIMA
/// `setdrop_geo`.
pub(crate) fn setdrop_geo<F: Scalar>(
    delta: F,
    factor_alpha: F,
    factor_beta: F,
    sim: &[F],
    simi: &[F],
    n: usize,
) -> usize {
    let vs = vsig(simi, n);
    let ve = veta(sim, n);
    if ve.iter().any(|&e| e > factor_beta * delta) {
        argmax_nonnan(&ve).unwrap_or(NO_DROP)
    } else if vs.iter().any(|&s| s < factor_alpha * delta) {
        argmin_nonnan(&vs).unwrap_or(NO_DROP)
    } else {
        NO_DROP
    }
}

/// Pick the vertex to replace with the trust-region trial point (eq. 19–22),
/// using the combined distance and poisedness score. Returns a 0-based index in
/// `0..=n` (`n` = the pole) or [`NO_DROP`]. PRIMA `setdrop_tr`.
pub(crate) fn setdrop_tr<F: Scalar>(
    ximproved: bool,
    d: &[F],
    delta: F,
    rho: F,
    sim: &[F],
    simi: &[F],
    n: usize,
) -> usize {
    let zero = F::zero();
    let one = F::one();
    let tenth = F::from_f64(0.1).unwrap();
    let np = n + 1;
    let mut distsq = vec![zero; np];
    if ximproved {
        for j in 0..n {
            let s: F = (0..n)
                .map(|r| (sim[r + j * n] - d[r]) * (sim[r + j * n] - d[r]))
                .sum();
            distsq[j] = s;
        }
        distsq[n] = d.iter().map(|&v| v * v).sum();
    } else {
        for j in 0..n {
            distsq[j] = (0..n).map(|r| sim[r + j * n] * sim[r + j * n]).sum();
        }
        distsq[n] = zero;
    }
    let den = rho.max(tenth * delta);
    let den2 = den * den;
    let weight: Vec<F> = distsq.iter().map(|&ds| one.max(ds / den2)).collect();
    // simid = simi · d (length n); the (n+1)-th Lagrange value is 1 − Σ simid.
    let simid: Vec<F> = (0..n)
        .map(|i| (0..n).map(|l| simi[i + l * n] * d[l]).sum::<F>())
        .collect();
    let sum_simid: F = simid.iter().cloned().sum();
    let mut score = vec![zero; np];
    for k in 0..n {
        score[k] = weight[k] * simid[k].abs();
    }
    score[n] = weight[n] * (one - sum_simid).abs();
    if !ximproved {
        score[n] = -one;
    }
    if score.iter().any(|&s| s > zero) {
        argmax_nonnan(&score).unwrap_or(NO_DROP)
    } else if ximproved {
        argmax_nonnan(&distsq).unwrap_or(NO_DROP)
    } else {
        NO_DROP
    }
}

/// Build the geometry-improving step replacing vertex `jdrop` (eq. 17), with its
/// sign chosen to minimize the predicted merit. PRIMA `geostep`. `m_lcon = 0`,
/// so the step uses only the nonlinear-constraint model.
#[allow(clippy::too_many_arguments)]
pub(crate) fn geostep<F: Scalar>(
    jdrop: usize,
    conmat: &[F],
    cpen: F,
    delta: F,
    fval: &[F],
    factor_gamma: F,
    simi: &[F],
    n: usize,
    m: usize,
) -> Vec<F> {
    let zero = F::zero();
    let row_norm: F = (0..n)
        .map(|l| simi[jdrop + l * n] * simi[jdrop + l * n])
        .sum();
    let vsigj = F::one() / row_norm.sqrt();
    let scale = factor_gamma * delta * vsigj;
    let mut d: Vec<F> = (0..n).map(|l| scale * simi[jdrop + l * n]).collect();

    // Choose the sign by the linear merit model.
    let g = build_g(fval, simi, n);
    let a = build_a(conmat, simi, n, m);
    // ad[i] = (Aᵀ d)[i] = Σ_l d[l] · A[l, i].
    let ad: Vec<F> = (0..m)
        .map(|i| (0..n).map(|l| d[l] * a[l + i * n]).sum::<F>())
        .collect();
    let mut cvpd = zero;
    let mut cvnd = zero;
    for i in 0..m {
        let pole = conmat[i + n * m];
        cvpd = cvpd.max(pole + ad[i]);
        cvnd = cvnd.max(pole - ad[i]);
    }
    cvpd = cvpd.max(zero);
    cvnd = cvnd.max(zero);
    let dg: F = (0..n).map(|l| d[l] * g[l]).sum();
    if -dg + cpen * cvnd < dg + cpen * cvpd {
        for v in d.iter_mut() {
            *v = -*v;
        }
    }
    d
}

fn argmax_nonnan<F: Scalar>(v: &[F]) -> Option<usize> {
    let mut best: Option<(usize, F)> = None;
    for (i, &x) in v.iter().enumerate() {
        if x.is_nan() {
            continue;
        }
        match best {
            Some((_, b)) if x <= b => {}
            _ => best = Some((i, x)),
        }
    }
    best.map(|(i, _)| i)
}

fn argmin_nonnan<F: Scalar>(v: &[F]) -> Option<usize> {
    let mut best: Option<(usize, F)> = None;
    for (i, &x) in v.iter().enumerate() {
        if x.is_nan() {
            continue;
        }
        match best {
            Some((_, b)) if x >= b => {}
            _ => best = Some((i, x)),
        }
    }
    best.map(|(i, _)| i)
}
