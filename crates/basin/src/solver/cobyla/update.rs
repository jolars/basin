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
    checked_sim: Vec<F>,
    checked_simi: Vec<F>,
    checked_error: Option<F>,
}

impl<F: Scalar> UpdateWork<F> {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            sim_old: vec![F::zero(); n * (n + 1)],
            simi_old: vec![F::zero(); n * n],
            product: vec![F::zero(); n],
            row: vec![F::zero(); n],
            md: vec![F::zero(); n],
            checked_sim: vec![F::zero(); if n > 4 { n * n } else { 0 }],
            checked_simi: vec![F::zero(); if n > 4 { n * n } else { 0 }],
            checked_error: None,
        }
    }

    fn inverse_error(&mut self, simi: &[F], sim: &[F], n: usize) -> F {
        // Comparing two n-by-n inputs is cheaper than repeating their cubic
        // product, except for the tiny kernels. The pole column is not an input
        // to the residual. Signed zeros must match, and NaNs prevent reuse.
        if n <= 4 {
            return inv_error(simi, sim, n, &mut self.product);
        }
        let sim = &sim[..n * n];
        let same = |a: &[F], b: &[F]| {
            a.iter().zip(b).all(|(&x, &y)| {
                x == y && x.is_sign_negative() == y.is_sign_negative()
            })
        };
        if let Some(error) = self.checked_error {
            if same(sim, &self.checked_sim) && same(simi, &self.checked_simi) {
                return error;
            }
        }
        let error = inv_error(simi, sim, n, &mut self.product);
        self.checked_sim.copy_from_slice(sim);
        self.checked_simi.copy_from_slice(simi);
        self.checked_error = Some(error);
        error
    }
}

/// Maximum entrywise inverse residual, reduced in column-major order.
fn inv_error<F: Scalar>(
    simi: &[F],
    sim: &[F],
    n: usize,
    product: &mut [F],
) -> F {
    // Fixed-size scratch lets the compiler keep tiny products in registers.
    match n {
        1 => inv_error_impl(simi, sim, 1, &mut [F::zero(); 1]),
        2 => inv_error_impl(simi, sim, 2, &mut [F::zero(); 2]),
        3 => inv_error_impl(simi, sim, 3, &mut [F::zero(); 3]),
        4 => inv_error_impl(simi, sim, 4, &mut [F::zero(); 4]),
        _ => inv_error_impl(simi, sim, n, product),
    }
}

#[inline(always)]
fn inv_error_impl<F: Scalar>(
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
    updatepole_with_residual(
        cpen, conmat, cval, fval, sim, simi, n, m, None, work,
    )
}

/// A vertex update has already checked this exact inverse. Reuse its residual
/// if repoling leaves the matrices unchanged; recovery still uses the same
/// thresholds and runs again when the residual warrants it.
#[allow(clippy::too_many_arguments)]
fn updatepole_with_residual<F: Scalar>(
    cpen: F,
    conmat: &mut [F],
    cval: &mut [F],
    fval: &mut [F],
    sim: &mut [F],
    simi: &mut [F],
    n: usize,
    m: usize,
    mut residual: Option<F>,
    work: &mut UpdateWork<F>,
) -> bool {
    let zero = F::zero();
    let itol = F::one();
    let jopt = findpole(cpen, cval, fval, n);

    if jopt < n {
        residual = None;
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

    let mut erri = residual.unwrap_or_else(|| work.inverse_error(simi, sim, n));
    if erri > F::from_f64(0.1).unwrap() * itol || erri.is_nan() {
        if let Some(fresh) = inv(&sim[..n * n], n) {
            let erri_test = work.inverse_error(&fresh, sim, n);
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

    let mut erri = work.inverse_error(simi, sim, n);
    if erri > F::from_f64(0.1).unwrap() * itol || erri.is_nan() {
        if let Some(fresh) = inv(&sim[..n * n], n) {
            let erri_test = work.inverse_error(&fresh, sim, n);
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
        updatepole_with_residual(
            cpen,
            conmat,
            cval,
            fval,
            sim,
            simi,
            n,
            m,
            Some(erri),
            work,
        )
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
    fn inverse_residual_preserves_zero_skipping_and_nonfinite_reduction() {
        for n in 1..=17 {
            let sim: Vec<f64> = (0..n * n)
                .map(|k| ((k * 13 + 7) % 19) as f64 - 9.0)
                .collect();
            let inverse: Vec<f64> = (0..n * n)
                .map(|k| ((k * 7 + 3) % 17) as f64 / 8.0 - 1.0)
                .collect();
            for exceptional in [None, Some(f64::NAN), Some(f64::INFINITY)] {
                let mut inverse = inverse.clone();
                if let Some(value) = exceptional {
                    inverse[n - 1] = value;
                }
                let mut expected = 0.0_f64;
                for j in 0..n {
                    for i in 0..n {
                        let mut entry = 0.0;
                        for k in 0..n {
                            if sim[k + j * n] != 0.0 {
                                entry += inverse[i + k * n] * sim[k + j * n];
                            }
                        }
                        let diff = entry - f64::from(i == j);
                        expected = if diff.is_nan() {
                            f64::NAN
                        } else {
                            expected.max(diff.abs())
                        };
                    }
                }
                let actual = inv_error(&inverse, &sim, n, &mut vec![0.0; n]);
                assert!(
                    actual.to_bits() == expected.to_bits()
                        || (actual.is_nan() && expected.is_nan()),
                    "n={n}: {actual} != {expected}"
                );
            }
        }
        // Zero coefficients must suppress even infinite entries in the inverse.
        assert_eq!(inv_error(&[f64::INFINITY], &[-0.0], 1, &mut [0.0]), 1.0);
    }

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

    #[test]
    fn cached_inverse_residual_tracks_both_matrices() {
        let n = 5;
        let mut sim = super::super::linalg::eye::<f64>(n);
        let mut simi = sim.clone();
        let mut work = UpdateWork::new(n);
        for (matrix, index, value) in [
            (false, 0, 1.0),
            (false, 24, 2.0),
            (true, 24, 0.5),
            (true, 24, f64::NAN),
            (false, 24, -0.0),
            (true, 24, f64::INFINITY),
            (false, 24, 1.0),
            (true, 24, 1.0),
        ] {
            if matrix {
                simi[index] = value;
            } else {
                sim[index] = value;
            }
            let expected = inv_error(&simi, &sim, n, &mut vec![0.0; n]);
            for _ in 0..2 {
                let actual = work.inverse_error(&simi, &sim, n);
                assert!(
                    actual.to_bits() == expected.to_bits()
                        || (actual.is_nan() && expected.is_nan())
                );
            }
        }
    }
}
