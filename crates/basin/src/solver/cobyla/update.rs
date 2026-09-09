// `!(cval[k] > cmin)` in findpole is a deliberate NaN-preserving port of PRIMA's
// `~(cval > cmin)` mask (`<=` would exclude NaN, changing the tie-break); keep it.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

//! COBYLA simplex update (PRIMA `update.f90`): `findpole`/`updatepole`/
//! `updatexfc`.
//!
//! The simplex is stored as `sim` (n × (n+1) column-major: columns `0..n` are
//! displacements from the pole, column `n` is the pole: the current best vertex
//! in absolute coordinates) and its companion `simi = inv(sim[:, 0..n])`. `fval`,
//! `conmat` (m × (n+1)), and `cval` hold the objective, constraints, and
//! constraint violations at the vertices. Vertex replacement updates `simi` by a
//! rank-one (Sherman-Morrison) formula, recomputing it from scratch if rounding
//! has damaged it.

use crate::core::math::Scalar;

use super::linalg::{col, inv};

/// `usize::MAX` marks "no vertex to drop" (PRIMA's `jdrop == 0`).
pub(crate) const NO_DROP: usize = usize::MAX;

/// Rollback and product storage reused across simplex updates.
pub(crate) struct UpdateWork<F> {
    sim_old: Vec<F>,
    simi_old: Vec<F>,
    product: Vec<F>,
    row: Vec<F>,
    md: Vec<F>,
}

impl<F: Scalar> UpdateWork<F> {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            sim_old: vec![F::zero(); n * (n + 1)],
            simi_old: vec![F::zero(); n * n],
            product: vec![F::zero(); n],
            row: vec![F::zero(); n],
            md: vec![F::zero(); n],
        }
    }
}

/// Maximum entrywise inverse residual, reduced in column-major order.
fn inv_error<F: Scalar>(
    simi: &[F],
    sim: &[F],
    n: usize,
    product: &mut [F],
) -> F {
    let mut error = F::zero();
    for j in 0..n {
        product.fill(F::zero());
        for k in 0..n {
            let bkj = sim[k + j * n];
            // Keep the original product's zero skipping and accumulation order,
            // including its behavior for non-finite entries.
            if bkj == F::zero() {
                continue;
            }
            for (value, &aik) in product.iter_mut().zip(col(simi, n, k)) {
                *value = *value + aik * bkj;
            }
        }
        for (i, &value) in product.iter().enumerate() {
            let diff = value - if i == j { F::one() } else { F::zero() };
            error = if diff.is_nan() {
                F::nan()
            } else {
                error.max(diff.abs())
            };
        }
    }
    error
}

/// Identify the best vertex of the simplex w.r.t. the merit `φ = f + cpen·cstrv`,
/// preferring smaller `cstrv` on ties; returns its 0-based index in `0..=n`
/// (default `n`, the pole). PRIMA `findpole`.
pub(crate) fn findpole<F: Scalar>(
    cpen: F,
    cval: &[F],
    fval: &[F],
    n: usize,
) -> usize {
    let np = n + 1;
    let phi = |k: usize| fval[k] + cpen * cval[k];
    let phimin = (0..np).map(phi).fold(F::infinity(), F::min);
    let mut jopt = n;
    if phimin < phi(jopt)
        || (0..np).any(|k| cval[k] < cval[jopt] && phi(k) <= phi(jopt))
    {
        // argmin cval over {phi <= phimin}, first such index.
        let cmin = (0..np)
            .filter(|&k| phi(k) <= phimin)
            .map(|k| cval[k])
            .fold(F::infinity(), F::min);
        jopt = (0..np)
            .find(|&k| phi(k) <= phimin && !(cval[k] > cmin))
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
    work: &mut UpdateWork<F>,
) -> bool {
    let zero = F::zero();
    let itol = F::one();
    let jopt = findpole(cpen, cval, fval, n);

    if jopt < n {
        work.sim_old.copy_from_slice(sim);
        work.simi_old.copy_from_slice(simi);
        // sim(:, n) += sim(:, jopt); save sim(:, jopt); zero it; sim(:, 0..n) -= sim_jopt.
        let sim_jopt = &mut work.row;
        sim_jopt.copy_from_slice(col(sim, n, jopt));
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

    let mut erri = inv_error(simi, sim, n, &mut work.product);
    if erri > F::from_f64(0.1).unwrap() * itol || erri.is_nan() {
        if let Some(fresh) = inv(&sim[..n * n], n) {
            let erri_test = inv_error(&fresh, sim, n, &mut work.product);
            if erri_test < erri || (erri.is_nan() && !erri_test.is_nan()) {
                if erri_test <= itol {
                    simi.copy_from_slice(&fresh);
                }
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
        if jopt < n {
            sim.copy_from_slice(&work.sim_old);
            simi.copy_from_slice(&work.simi_old);
        }
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
    work: &mut UpdateWork<F>,
) -> bool {
    if jdrop == NO_DROP {
        return true;
    }
    let zero = F::zero();
    let one = F::one();
    let itol = F::one();
    work.sim_old.copy_from_slice(sim);
    work.simi_old.copy_from_slice(simi);

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
        let simi_jdrop = &mut work.row;
        for l in 0..n {
            simi_jdrop[l] = simi[jdrop + l * n] / denom;
        }
        // md = simi · d.
        let md = &mut work.md;
        for i in 0..n {
            md[i] = (0..n).map(|l| simi[i + l * n] * d[l]).sum::<F>();
        }
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
        let simid = &mut work.md;
        for i in 0..n {
            simid[i] = (0..n).map(|l| simi[i + l * n] * d[l]).sum::<F>();
        }
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

    let mut erri = inv_error(simi, sim, n, &mut work.product);
    if erri > F::from_f64(0.1).unwrap() * itol || erri.is_nan() {
        if let Some(fresh) = inv(&sim[..n * n], n) {
            let erri_test = inv_error(&fresh, sim, n, &mut work.product);
            if erri_test < erri || (erri.is_nan() && !erri_test.is_nan()) {
                if erri_test <= itol {
                    simi.copy_from_slice(&fresh);
                }
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
        updatepole(cpen, conmat, cval, fval, sim, simi, n, m, work)
    } else {
        sim.copy_from_slice(&work.sim_old);
        simi.copy_from_slice(&work.simi_old);
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_check_matches_a_dense_residual() {
        let a = [2.0_f64, 1.0, -1.0, 0.0, 3.0, 2.0, 1.0, 0.0, 4.0];
        let mut inverse = inv(&a, 3).unwrap();
        let mut product = [0.0; 3];
        assert!(inv_error(&inverse, &a, 3, &mut product) < 1e-14);
        inverse[1] += 0.25;
        assert!((inv_error(&inverse, &a, 3, &mut product) - 0.5).abs() < 1e-14);
        inverse.fill(f64::NAN);
        assert!(inv_error(&inverse, &a, 3, &mut product).is_nan());
    }

    #[test]
    fn unchanged_pole_still_repairs_a_damaged_inverse() {
        let mut sim = [1.0, 0.0, 0.0, 1.0, 2.0, 3.0];
        let mut simi = [0.0; 4];
        let mut work = UpdateWork::new(2);
        assert!(updatepole(
            1.0,
            &mut [],
            &mut [0.0; 3],
            &mut [2.0, 1.0, 0.0],
            &mut sim,
            &mut simi,
            2,
            0,
            &mut work
        ));
        assert_eq!(simi, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(sim, [1.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn singular_repoling_restores_the_original_simplex() {
        let original = [10.0, 0.0, 20.0, 0.0, 4.0, 5.0];
        let mut sim = original;
        let mut simi = [1.0, 0.0, 0.0, 1.0];
        let mut fval = [-1.0, 1.0, 0.0];
        let mut conmat = [-1.0, -2.0, -3.0];
        let mut work = UpdateWork::new(2);
        assert!(!updatepole(
            1.0,
            &mut conmat,
            &mut [0.0; 3],
            &mut fval,
            &mut sim,
            &mut simi,
            2,
            1,
            &mut work
        ));
        assert_eq!(sim, original);
        assert_eq!(simi, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(fval, [-1.0, 1.0, 0.0]);
        assert_eq!(conmat, [-1.0, -2.0, -3.0]);
    }

    #[test]
    fn rejected_vertex_updates_restore_reused_rollback_storage() {
        let mut work = UpdateWork::new(2);
        for d in [[0.0, 0.0], [f64::NAN, 1.0], [f64::INFINITY, 0.0]] {
            let original = [1.0, 0.0, 0.0, 1.0, 2.0, 3.0];
            let mut sim = original;
            let mut simi = [1.0, 0.0, 0.0, 1.0];
            let mut fval = [2.0, 1.0, 0.0];
            assert!(!updatexfc(
                0,
                &[],
                1.0,
                0.0,
                &d,
                -1.0,
                &mut [],
                &mut [0.0; 3],
                &mut fval,
                &mut sim,
                &mut simi,
                2,
                0,
                &mut work
            ));
            assert_eq!(sim, original);
            assert_eq!(simi, [1.0, 0.0, 0.0, 1.0]);
            assert_eq!(fval, [2.0, 1.0, 0.0]);
        }
    }
}
