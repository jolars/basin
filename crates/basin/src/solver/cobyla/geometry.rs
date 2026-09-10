//! COBYLA geometry management (PRIMA `geometry.f90`): `assess_geo`,
//! `setdrop_geo`, `setdrop_tr`, `geostep`.
//!
//! `assess_geo` tests the acceptability of the simplex (Powell 1994 eq. 14) via
//! the per-vertex face distances `σ` (`vsig`) and edge lengths `η` (`veta`).
//! `setdrop_tr`/`setdrop_geo` pick which vertex to replace after a
//! trust-region or geometry step; `geostep` builds the geometry-improving step
//! (eq. 15–17), choosing its sign by the linear merit model.

use crate::core::math::Scalar;

use super::model::ModelWork;
use super::update::NO_DROP;

/// Per-vertex face distances `σ[j] = 1/‖simi(j, :)‖` (j = 0..n).
fn vsig<F: Scalar>(
    simi: &[F],
    n: usize,
) -> impl Iterator<Item = F> + Clone + '_ {
    (0..n).map(move |j| {
        let s: F = (0..n).map(|l| simi[j + l * n] * simi[j + l * n]).sum();
        F::one() / s.sqrt()
    })
}

/// Per-vertex edge lengths `η[j] = ‖sim(:, j)‖` (j = 0..n, the displacements).
fn veta<F: Scalar>(
    sim: &[F],
    n: usize,
) -> impl Iterator<Item = F> + Clone + '_ {
    (0..n).map(move |j| {
        let s: F = (0..n).map(|r| sim[r + j * n] * sim[r + j * n]).sum();
        s.sqrt()
    })
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
    vsig(simi, n).all(|s| s >= factor_alpha * delta)
        && veta(sim, n).all(|e| e <= factor_beta * delta)
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
    if ve.clone().any(|e| e > factor_beta * delta) {
        argmax_nonnan(ve).unwrap_or(NO_DROP)
    } else if vs.clone().any(|s| s < factor_alpha * delta) {
        argmin_nonnan(vs).unwrap_or(NO_DROP)
    } else {
        NO_DROP
    }
}

/// Geometry scores reused when choosing a replacement vertex.
pub(crate) struct GeometryWork<F> {
    distsq: Vec<F>,
    simid: Vec<F>,
    score: Vec<F>,
}

impl<F: Scalar> GeometryWork<F> {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            distsq: vec![F::zero(); n + 1],
            simid: vec![F::zero(); n],
            score: vec![F::zero(); n + 1],
        }
    }
}

/// Pick the vertex to replace with the trust-region trial point (eq. 19–22),
/// using the combined distance and poisedness score. Returns a 0-based index in
/// `0..=n` (`n` = the pole) or [`NO_DROP`]. PRIMA `setdrop_tr`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn setdrop_tr<F: Scalar>(
    ximproved: bool,
    d: &[F],
    delta: F,
    rho: F,
    sim: &[F],
    simi: &[F],
    n: usize,
    work: &mut GeometryWork<F>,
) -> usize {
    let GeometryWork {
        distsq,
        simid,
        score,
    } = work;
    let zero = F::zero();
    let one = F::one();
    let tenth = F::from_f64(0.1).unwrap();
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
    // simid = simi · d (length n); the (n+1)-th Lagrange value is 1 − Σ simid.
    for i in 0..n {
        simid[i] = (0..n).map(|l| simi[i + l * n] * d[l]).sum::<F>();
    }
    let sum_simid: F = simid.iter().cloned().sum();
    for k in 0..n {
        score[k] = one.max(distsq[k] / den2) * simid[k].abs();
    }
    score[n] = one.max(distsq[n] / den2) * (one - sum_simid).abs();
    if !ximproved {
        score[n] = -one;
    }
    if score.iter().any(|&s| s > zero) {
        argmax_nonnan(score.iter().copied()).unwrap_or(NO_DROP)
    } else if ximproved {
        argmax_nonnan(distsq.iter().copied()).unwrap_or(NO_DROP)
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
    model: &mut ModelWork<F>,
    d: &mut [F],
) {
    let zero = F::zero();
    let row_norm: F = (0..n)
        .map(|l| simi[jdrop + l * n] * simi[jdrop + l * n])
        .sum();
    let vsigj = F::one() / row_norm.sqrt();
    let scale = factor_gamma * delta * vsigj;
    for l in 0..n {
        d[l] = scale * simi[jdrop + l * n];
    }

    // Choose the sign by the linear merit model.
    model.build(fval, conmat, simi);
    let g = &model.g;
    let a = &model.a;
    let mut cvpd = zero;
    let mut cvnd = zero;
    for i in 0..m {
        let pole = conmat[i + n * m];
        let ad: F = (0..n).map(|l| d[l] * a[l + i * n]).sum();
        cvpd = cvpd.max(pole + ad);
        cvnd = cvnd.max(pole - ad);
    }
    cvpd = cvpd.max(zero);
    cvnd = cvnd.max(zero);
    let dg: F = (0..n).map(|l| d[l] * g[l]).sum();
    if -dg + cpen * cvnd < dg + cpen * cvpd {
        for v in d.iter_mut() {
            *v = -*v;
        }
    }
}

fn argmax_nonnan<F: Scalar>(v: impl IntoIterator<Item = F>) -> Option<usize> {
    let mut best: Option<(usize, F)> = None;
    for (i, x) in v.into_iter().enumerate() {
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

fn argmin_nonnan<F: Scalar>(v: impl IntoIterator<Item = F>) -> Option<usize> {
    let mut best: Option<(usize, F)> = None;
    for (i, x) in v.into_iter().enumerate() {
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
