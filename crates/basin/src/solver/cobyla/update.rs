// `!(cval[k] > cmin)` in findpole is a deliberate NaN-preserving port of PRIMA's
// `~(cval > cmin)` mask (`<=` would exclude NaN, changing the tie-break); keep it.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

//! COBYLA simplex update (PRIMA `update.f90`): `findpole`/`updatepole`/
//! `updatexfc`.
//!
//! The simplex is stored as `sim` (n × (n+1) column-major: columns `0..n` are
//! displacements from the pole, column `n` is the pole—the current best vertex
//! in absolute coordinates) and its companion `simi = inv(sim[:, 0..n])`. `fval`,
//! `conmat` (m × (n+1)), and `cval` hold the objective, constraints, and
//! constraint violations at the vertices. Vertex replacement updates `simi` by a
//! rank-one (Sherman-Morrison) formula, recomputing it from scratch if rounding
//! has damaged it.

use crate::core::math::Scalar;

use super::linalg::{col, eye, inv, maxabs};

/// `usize::MAX` marks "no vertex to drop" (PRIMA's `jdrop == 0`).
pub(crate) const NO_DROP: usize = usize::MAX;

/// `n × n` times `n × n`, column-major.
fn matmul_nn<F: Scalar>(a: &[F], b: &[F], n: usize) -> Vec<F> {
    let mut c = vec![F::zero(); n * n];
    for j in 0..n {
        for k in 0..n {
            let bkj = b[k + j * n];
            if bkj == F::zero() {
                continue;
            }
            for i in 0..n {
                c[i + j * n] = c[i + j * n] + a[i + k * n] * bkj;
            }
        }
    }
    c
}

/// `max |A·simi[:,0..n] − I|`; `NaN` if the product contains `NaN`.
fn inv_error<F: Scalar>(simi: &[F], sim: &[F], n: usize) -> F {
    let prod = matmul_nn(simi, &sim[..n * n], n);
    let ident = eye::<F>(n);
    let diff: Vec<F> = prod.iter().zip(&ident).map(|(&a, &b)| a - b).collect();
    maxabs(&diff)
}

/// Identify the best vertex of the simplex w.r.t. the merit `φ = f + cpen·cstrv`,
/// preferring smaller `cstrv` on ties; returns its 0-based index in `0..=n`
/// (default `n`, the pole). PRIMA `findpole`.
pub(crate) fn findpole<F: Scalar>(cpen: F, cval: &[F], fval: &[F], n: usize) -> usize {
    let np = n + 1;
    let phi: Vec<F> = (0..np).map(|k| fval[k] + cpen * cval[k]).collect();
    let phimin = phi.iter().cloned().fold(F::infinity(), F::min);
    let mut jopt = n;
    if phimin < phi[jopt] || (0..np).any(|k| cval[k] < cval[jopt] && phi[k] <= phi[jopt]) {
        // argmin cval over {phi <= phimin}, first such index.
        let cmin = (0..np)
            .filter(|&k| phi[k] <= phimin)
            .map(|k| cval[k])
            .fold(F::infinity(), F::min);
        jopt = (0..np)
            .find(|&k| phi[k] <= phimin && !(cval[k] > cmin))
            .unwrap_or(n);
    }
    jopt
}

/// Switch the best vertex to the pole position `sim[:, n]`, updating `simi`,
/// `fval`, `conmat`, `cval`. Returns `false` on damaging rounding (state
/// restored). PRIMA `updatepole`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn updatepole<F: Scalar>(
    cpen: F,
    conmat: &mut [F],
    cval: &mut [F],
    fval: &mut [F],
    sim: &mut [F],
    simi: &mut [F],
    n: usize,
    m: usize,
) -> bool {
    let zero = F::zero();
    let itol = F::one();
    let jopt = findpole(cpen, cval, fval, n);

    let sim_old = sim.to_vec();
    let simi_old = simi.to_vec();

    if jopt < n {
        // sim(:, n) += sim(:, jopt); save sim(:, jopt); zero it; sim(:, 0..n) -= sim_jopt.
        let sim_jopt: Vec<F> = col(sim, n, jopt).to_vec();
        for r in 0..n {
            sim[r + n * n] = sim[r + n * n] + sim_jopt[r];
            sim[r + jopt * n] = zero;
        }
        for j in 0..n {
            for r in 0..n {
                sim[r + j * n] = sim[r + j * n] - sim_jopt[r];
            }
        }
        // simi(jopt, :) = -sum over rows of simi (column sums).
        for l in 0..n {
            let mut s = zero;
            for i in 0..n {
                s = s + simi[i + l * n];
            }
            simi[jopt + l * n] = -s;
        }
    }

    let mut erri = inv_error(simi, sim, n);
    if erri > F::from_f64(0.1).unwrap() * itol || erri.is_nan() {
        if let Some(fresh) = inv(&sim[..n * n], n) {
            let erri_test = inv_error(&fresh, sim, n);
            if erri_test < erri || (erri.is_nan() && !erri_test.is_nan()) {
                simi.copy_from_slice(&fresh);
                erri = erri_test;
            }
        }
    }

    if erri <= itol {
        if jopt < n {
            fval.swap(jopt, n);
            cval.swap(jopt, n);
            for r in 0..m {
                conmat.swap(r + jopt * m, r + n * m);
            }
        }
        true
    } else {
        sim.copy_from_slice(&sim_old);
        simi.copy_from_slice(&simi_old);
        false
    }
}

/// Replace vertex `jdrop` with the new point `pole + d` (objective `f`,
/// constraints `constr`, violation `cstrv`), updating `simi` by a rank-one
/// formula, then re-pole. `jdrop == NO_DROP` does nothing. `jdrop < n` replaces
/// a displacement vertex; `jdrop == n` moves the pole. Returns `false` on
/// damaging rounding. PRIMA `updatexfc`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn updatexfc<F: Scalar>(
    jdrop: usize,
    constr: &[F],
    cpen: F,
    cstrv: F,
    d: &[F],
    f: F,
    conmat: &mut [F],
    cval: &mut [F],
    fval: &mut [F],
    sim: &mut [F],
    simi: &mut [F],
    n: usize,
    m: usize,
) -> bool {
    if jdrop == NO_DROP {
        return true;
    }
    let zero = F::zero();
    let one = F::one();
    let itol = F::one();
    let sim_old = sim.to_vec();
    let simi_old = simi.to_vec();

    if jdrop < n {
        // sim(:, jdrop) = d.
        for r in 0..n {
            sim[r + jdrop * n] = d[r];
        }
        // simi_jdrop = simi(jdrop, :) / (simi(jdrop, :) · d).
        let mut denom = zero;
        for l in 0..n {
            denom = denom + simi[jdrop + l * n] * d[l];
        }
        let simi_jdrop: Vec<F> = (0..n).map(|l| simi[jdrop + l * n] / denom).collect();
        // md = simi · d.
        let md: Vec<F> = (0..n)
            .map(|i| (0..n).map(|l| simi[i + l * n] * d[l]).sum::<F>())
            .collect();
        // simi -= outer(md, simi_jdrop); then simi(jdrop, :) = simi_jdrop.
        for l in 0..n {
            for i in 0..n {
                simi[i + l * n] = simi[i + l * n] - md[i] * simi_jdrop[l];
            }
        }
        for l in 0..n {
            simi[jdrop + l * n] = simi_jdrop[l];
        }
    } else {
        // jdrop == n: move the pole by d.
        for r in 0..n {
            sim[r + n * n] = sim[r + n * n] + d[r];
        }
        for j in 0..n {
            for r in 0..n {
                sim[r + j * n] = sim[r + j * n] - d[r];
            }
        }
        let simid: Vec<F> = (0..n)
            .map(|i| (0..n).map(|l| simi[i + l * n] * d[l]).sum::<F>())
            .collect();
        let sum_simid: F = simid.iter().cloned().sum();
        let denom = one - sum_simid;
        for l in 0..n {
            let mut col_sum = zero;
            for i in 0..n {
                col_sum = col_sum + simi[i + l * n];
            }
            let factor = col_sum / denom;
            for i in 0..n {
                simi[i + l * n] = simi[i + l * n] + simid[i] * factor;
            }
        }
    }

    let mut erri = inv_error(simi, sim, n);
    if erri > F::from_f64(0.1).unwrap() * itol || erri.is_nan() {
        if let Some(fresh) = inv(&sim[..n * n], n) {
            let erri_test = inv_error(&fresh, sim, n);
            if erri_test < erri || (erri.is_nan() && !erri_test.is_nan()) {
                simi.copy_from_slice(&fresh);
                erri = erri_test;
            }
        }
    }

    if erri <= itol {
        fval[jdrop] = f;
        for r in 0..m {
            conmat[r + jdrop * m] = constr[r];
        }
        cval[jdrop] = cstrv;
        updatepole(cpen, conmat, cval, fval, sim, simi, n, m)
    } else {
        sim.copy_from_slice(&sim_old);
        simi.copy_from_slice(&simi_old);
        false
    }
}
