// Index-based loops mirror the Fortran source for parity.
#![allow(clippy::needless_range_loop)]

//! `L·E·Lᵀ` factorization of the indefinite middle matrix `K`.
//!
//! Port of `formk` (`references/lbfgsb-v3.0/lbfgsb.f:1848`). Given the
//! current free-set / active-set partition (`ind`, `nsub`), the
//! variables that entered or left the free set since the previous
//! iteration (`indx2`, `nenter`, `ileave`), the limited-memory history
//! (`ws_cols`, `wy_cols`, `sy`, `theta`), and the persistent
//! lower-triangular Gram cache `wn1`, this routine produces the
//! `L·E·Lᵀ` factorization of
//!
//! ```text
//!     K = [ -D − YᵀZZᵀY/θ    L_aᵀ − R_zᵀ ]
//!         [ L_a − R_z        θ SᵀAAᵀS    ]
//! ```
//!
//! with `E = diag(−I, I)`. The factor is stored in the upper triangle
//! of `wn`; downstream `subsm` consumes it via the paired
//! `solve_upper_tri{,_transposed}` helpers in [`super::compact`].
//!
//! `wn1` is the lower-triangular `N`-matrix Gram cache:
//!
//! ```text
//!     N = [ Y' Z Z' Y     L_a' + R_z' ]
//!         [ L_a + R_z     S' A A' S   ]
//! ```
//!
//! maintained incrementally across outer iterations. On each call we:
//!
//! 1. **Update `wn1`** with the new history pair (`updatd == true`) and
//!    with the entering / leaving variables since the previous
//!    free-set.
//! 2. **Build the upper triangle of `wn`** from `wn1` with the sign /
//!    `θ` rescaling that produces `−K(1,1)` (positive) and `K(2,1)`,
//!    `K(2,2)` in their natural signs.
//! 3. **Cholesky-factor the (1,1) block** of `wn`. This gives `L_S`
//!    such that `L_S L_Sᵀ = D + YᵀZZᵀY/θ` (the `−E·K(1,1)` block).
//! 4. **Forward-substitute** the (1,2) block through `L_Sᵀ`.
//! 5. **Schur-update the (2,2) block** with the inner product of the
//!    transformed (1,2) block. The update is a `+` (not the usual
//!    `−`), which mirrors Fortran's structured handling of the
//!    indefinite middle and yields a positive-definite Schur
//!    complement.
//! 6. **Cholesky-factor the (2,2) block**.
//!
//! Together steps 3–6 are the `L·E·Lᵀ` decomposition; the upper
//! triangle of `wn` ends up storing the upper-triangular `L^T` factor
//! used by [`super::subsm`].

use crate::core::math::Scalar;

/// Reasons [`formk`] can fail. Matches Fortran's `info ≠ 0` exits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FormkError {
    /// The first Cholesky factorization (of the (1,1) block) failed—
    /// the limited-memory matrix has degenerated. Mirrors Fortran
    /// `info = −1` from `formk`.
    NotPositiveDefiniteFirst,
    /// The second Cholesky factorization (of the (2,2) Schur block)
    /// failed. Mirrors Fortran `info = −2` from `formk`.
    NotPositiveDefiniteSecond,
}

/// Build the `L·E·Lᵀ` factorization of `K` and store its upper-
/// triangular factor in `wn`. Updates the lower-triangular `wn1` Gram
/// cache in place.
///
/// # Parameters
///
/// - `wn`: output. `2m × 2m` row-major buffer, stride `2m`; only the
///   leading `2*col × 2*col` upper triangle is touched.
/// - `wn1`: input + output. `2m × 2m` row-major buffer carrying the
///   `N` Gram cache between iterations.
/// - `m`, `col`, `theta`: compact-form parameters matching the
///   history in `ws_cols`/`wy_cols`.
/// - `sy`: `m × m` row-major; only the `col × col` leading block is
///   read for its diagonal `D = diag(sᵢ·yᵢ)`.
/// - `ws_cols`, `wy_cols`: `col` history columns, oldest first.
/// - `nsub`: number of free variables; `ind[0..nsub]` are free
///   indices, `ind[nsub..n]` active indices.
/// - `ind`: length `n`. Free + active partition (free first).
/// - `nenter`: number of entering variables;
///   `indx2[0..nenter]` are their indices.
/// - `ileave`: `indx2[ileave..n]` are the leaving variables.
/// - `indx2`: length `n`. Entering + leaving partition.
/// - `iupdat`: total accepted BFGS updates so far (Fortran `iupdat`,
///   for the `> m` shift trigger).
/// - `updatd`: was history updated since the previous formk call?
#[allow(clippy::too_many_arguments)]
pub(crate) fn formk<F: Scalar>(
    wn: &mut [F],
    wn1: &mut [F],
    m: usize,
    col: usize,
    theta: F,
    sy: &[F],
    ws_cols: &[&[F]],
    wy_cols: &[&[F]],
    nsub: usize,
    ind: &[usize],
    nenter: usize,
    ileave: usize,
    indx2: &[usize],
    iupdat: u32,
    updatd: bool,
) -> Result<(), FormkError> {
    debug_assert!(col <= m);
    debug_assert!(col >= 1, "formk requires col ≥ 1");
    let n = ind.len();
    debug_assert!(nsub <= n);
    debug_assert!(ileave <= n);
    debug_assert!(nenter <= n);
    let two_m = 2 * m;
    debug_assert!(wn.len() >= two_m * two_m);
    debug_assert!(wn1.len() >= two_m * two_m);
    debug_assert_eq!(ws_cols.len(), col);
    debug_assert_eq!(wy_cols.len(), col);

    // -----------------------------------------------------------------
    // Phase 1—update `wn1` for the new pair (when `updatd`).
    // -----------------------------------------------------------------
    let upcl = if updatd {
        if iupdat as usize > m {
            // Shift each of (1,1), (2,2), (2,1) up-and-left by one row+col.
            // This drops the oldest column's contribution; matches the
            // Fortran `dcopy` loops at `lbfgsb.f:1996-2001`.
            for jy in 0..m - 1 {
                let len = m - 1 - jy;
                // (1,1) block: lower triangle, stride 2m, leading m × m.
                for k in 0..len {
                    let src = (jy + 1 + k) * two_m + (jy + 1);
                    let dst = (jy + k) * two_m + jy;
                    wn1[dst] = wn1[src];
                }
                // (2,2) block: same shape, offset by (m, m).
                let js = m + jy;
                for k in 0..len {
                    let src = (js + 1 + k) * two_m + (js + 1);
                    let dst = (js + k) * two_m + js;
                    wn1[dst] = wn1[src];
                }
                // (2,1) block: full rectangular m × m, rows [m, 2m),
                // shifted left-and-up by one (column `jy + 1` → column
                // `jy`, rows `m+1..2m-1` → rows `m..2m-2`).
                for k in 0..m - 1 {
                    let src = (m + 1 + k) * two_m + (jy + 1);
                    let dst = (m + k) * two_m + jy;
                    wn1[dst] = wn1[src];
                }
            }
        }

        // Put new rows in (1,1), (2,1), (2,2)—the *last* history
        // column contributes its row of `Y'ZZ'Y`, `S'AA'Y`, `S'AA'S`.
        // `last = col − 1` is the 0-indexed slot of the newest pair.
        let last = col - 1;
        for jy in 0..col {
            let mut temp1 = F::zero(); // (Y'ZZ'Y)[last, jy] over free indices
            let mut temp2 = F::zero(); // (S'AA'S)[last, jy] over active indices
            let mut temp3 = F::zero(); // (L_a)[last, jy] over active indices
            for k in 0..nsub {
                let k1 = ind[k];
                temp1 = temp1 + wy_cols[last][k1] * wy_cols[jy][k1];
            }
            for k in nsub..n {
                let k1 = ind[k];
                temp2 = temp2 + ws_cols[last][k1] * ws_cols[jy][k1];
                temp3 = temp3 + ws_cols[last][k1] * wy_cols[jy][k1];
            }
            wn1[last * two_m + jy] = temp1;
            wn1[(m + last) * two_m + (m + jy)] = temp2;
            wn1[(m + last) * two_m + jy] = temp3;
        }

        // Last column of (2,1): R_z column for the new pair. Walks the
        // *free* index set with `last` on the `wy` side.
        for i in 0..col {
            let mut temp3 = F::zero();
            for k in 0..nsub {
                let k1 = ind[k];
                temp3 = temp3 + ws_cols[i][k1] * wy_cols[last][k1];
            }
            wn1[(m + i) * two_m + last] = temp3;
        }
        col - 1
    } else {
        col
    };

    // -----------------------------------------------------------------
    // Phase 2—modify old parts of `wn1` for entering / leaving vars.
    // (1,1) and (2,2) symmetric / lower-tri updates.
    // -----------------------------------------------------------------
    for iy in 0..upcl {
        for jy in 0..=iy {
            let mut temp1 = F::zero();
            let mut temp2 = F::zero();
            let mut temp3 = F::zero();
            let mut temp4 = F::zero();
            for k in 0..nenter {
                let k1 = indx2[k];
                temp1 = temp1 + wy_cols[iy][k1] * wy_cols[jy][k1];
                temp2 = temp2 + ws_cols[iy][k1] * ws_cols[jy][k1];
            }
            for k in ileave..n {
                let k1 = indx2[k];
                temp3 = temp3 + wy_cols[iy][k1] * wy_cols[jy][k1];
                temp4 = temp4 + ws_cols[iy][k1] * ws_cols[jy][k1];
            }
            wn1[iy * two_m + jy] = wn1[iy * two_m + jy] + temp1 - temp3;
            wn1[(m + iy) * two_m + (m + jy)] = wn1[(m + iy) * two_m + (m + jy)] + (-temp2 + temp4);
        }
    }

    // (2,1) block update—full rectangle, with sign flip across the
    // (block) diagonal (`is ≤ jy + m` in Fortran ⇔ `iy ≤ jy` here).
    for iy in 0..upcl {
        for jy in 0..upcl {
            let mut temp1 = F::zero();
            let mut temp3 = F::zero();
            for k in 0..nenter {
                let k1 = indx2[k];
                temp1 = temp1 + ws_cols[iy][k1] * wy_cols[jy][k1];
            }
            for k in ileave..n {
                let k1 = indx2[k];
                temp3 = temp3 + ws_cols[iy][k1] * wy_cols[jy][k1];
            }
            let delta = if iy <= jy {
                temp1 - temp3
            } else {
                -temp1 + temp3
            };
            wn1[(m + iy) * two_m + jy] = wn1[(m + iy) * two_m + jy] + delta;
        }
    }

    // -----------------------------------------------------------------
    // Phase 3—form the upper triangle of `wn` from `wn1`.
    //   (1,1) ← (Y'ZZ'Y)/θ + D
    //   (1,2) ← −L_a' + R_z'   (from wn1's (2,1) block, with sign
    //                          flipped on the strict lower part)
    //   (2,2) ← θ · S'AA'S
    // -----------------------------------------------------------------
    let col2 = 2 * col;
    for iy in 0..col {
        // (1,1) and (2,2) upper triangles, from wn1's lower triangles.
        for jy in 0..=iy {
            wn[jy * two_m + iy] = wn1[iy * two_m + jy] / theta;
            wn[(col + jy) * two_m + (col + iy)] = wn1[(m + iy) * two_m + (m + jy)] * theta;
        }
        // (1,2) block, rows 0..col, col `col + iy`. Strict-lower part
        // of wn1's (2,1) goes in with a `−` sign (L_a'); the
        // upper-tri-plus-diagonal goes in with `+` sign (R_z').
        let is = col + iy;
        let is1 = m + iy;
        for jy in 0..iy {
            wn[jy * two_m + is] = -wn1[is1 * two_m + jy];
        }
        for jy in iy..col {
            wn[jy * two_m + is] = wn1[is1 * two_m + jy];
        }
        // Add D's diagonal back into (1,1).
        wn[iy * two_m + iy] = wn[iy * two_m + iy] + sy[iy * m + iy];
    }

    // -----------------------------------------------------------------
    // Phase 4—Cholesky of (1,1) in place. Upper triangle gets `L^T`.
    // -----------------------------------------------------------------
    for j in 0..col {
        let mut s = wn[j * two_m + j];
        for k in 0..j {
            let jkj = wn[k * two_m + j];
            s = s - jkj * jkj;
        }
        if !s.is_finite() || s <= F::zero() {
            return Err(FormkError::NotPositiveDefiniteFirst);
        }
        let djj = s.sqrt();
        wn[j * two_m + j] = djj;
        for i in (j + 1)..col {
            let mut s = wn[j * two_m + i];
            for k in 0..j {
                s = s - wn[k * two_m + j] * wn[k * two_m + i];
            }
            wn[j * two_m + i] = s / djj;
        }
    }

    // -----------------------------------------------------------------
    // Phase 5—solve `L^T · X = wn(0..col, col..2col)` in place. One
    // forward solve per column of the (1,2) block.
    // -----------------------------------------------------------------
    for js in col..col2 {
        for i in 0..col {
            let mut s = wn[i * two_m + js];
            for k in 0..i {
                s = s - wn[k * two_m + i] * wn[k * two_m + js];
            }
            wn[i * two_m + js] = s / wn[i * two_m + i];
        }
    }

    // -----------------------------------------------------------------
    // Phase 6—Schur-update (2,2): wn(2,2) += X^T X (upper triangle).
    // -----------------------------------------------------------------
    for is in col..col2 {
        for js in is..col2 {
            let mut acc = F::zero();
            for k in 0..col {
                acc = acc + wn[k * two_m + is] * wn[k * two_m + js];
            }
            wn[is * two_m + js] = wn[is * two_m + js] + acc;
        }
    }

    // -----------------------------------------------------------------
    // Phase 7—Cholesky of (2,2) in place. Same algorithm as Phase 4,
    // offset by `col`.
    // -----------------------------------------------------------------
    for j in 0..col {
        let mut s = wn[(col + j) * two_m + (col + j)];
        for k in 0..j {
            let jkj = wn[(col + k) * two_m + (col + j)];
            s = s - jkj * jkj;
        }
        if !s.is_finite() || s <= F::zero() {
            return Err(FormkError::NotPositiveDefiniteSecond);
        }
        let djj = s.sqrt();
        wn[(col + j) * two_m + (col + j)] = djj;
        for i in (j + 1)..col {
            let mut s = wn[(col + j) * two_m + (col + i)];
            for k in 0..j {
                s = s - wn[(col + k) * two_m + (col + j)] * wn[(col + k) * two_m + (col + i)];
            }
            wn[(col + j) * two_m + (col + i)] = s / djj;
        }
    }

    Ok(())
}

#[cfg(test)]
// Explicit `i * two_m + j` indexing (including `0 * two_m + 0`)
// mirrors the Fortran source's 2-D layout—load-bearing for
// readability when cross-checking against `lbfgsb.f`.
#[allow(clippy::identity_op, clippy::erasing_op)]
mod tests {
    use super::*;

    /// First-iteration formk: `col == 1`, one history pair
    /// `s = (1, 2), y = (1, 1)`, `θ = 1`, var 0 free, var 1 active.
    /// Verifies the resulting `wn` matches the hand-built fixture in
    /// `subsm`'s `col_one_one_free_one_active_matches_closed_form_newton`
    /// (`wn[0,0] = 2`, `wn[0,1] = 0.5`, `wn[1,1] = √4.25`).
    #[test]
    fn col_one_two_vars_one_free_one_active_matches_hand_fixture() {
        let s = [1.0_f64, 2.0];
        let y = [1.0_f64, 1.0];
        let theta = 1.0;
        let m = 1;
        let two_m = 2 * m;

        // sy is m × m row-major; diagonal holds s·y.
        let mut sy = vec![0.0_f64; m * m];
        sy[0] = s[0] * y[0] + s[1] * y[1]; // 3

        let ws_cols: Vec<&[f64]> = vec![&s];
        let wy_cols: Vec<&[f64]> = vec![&y];

        // var 0 free, var 1 active ⇒ ind = [0, 1], nsub = 1.
        let ind = [0_usize, 1];
        // No entering / leaving since this is the first formk call.
        let indx2 = [0_usize; 2];
        let nenter = 0;
        let ileave = 2;

        let mut wn = vec![0.0_f64; two_m * two_m];
        let mut wn1 = vec![0.0_f64; two_m * two_m];

        formk(
            &mut wn, &mut wn1, m, 1, theta, &sy, &ws_cols, &wy_cols, 1, &ind, nenter, ileave,
            &indx2, 1, true,
        )
        .unwrap();

        // Phase 1: wn1[0,0] = wy[0][0]*wy[0][0] = 1 (Y'ZZ'Y free part).
        assert!((wn1[0 * two_m + 0] - 1.0).abs() < 1e-12);
        // wn1[1,1] = ws[0][1]^2 = 4 (S'AA'S active part).
        assert!((wn1[1 * two_m + 1] - 4.0).abs() < 1e-12);
        // wn1[1,0] gets written twice: once with the active dot (=2),
        // then overwritten by the free dot (=1) from the column loop.
        assert!((wn1[1 * two_m + 0] - 1.0).abs() < 1e-12);

        // wn[0,0] = 2 (= √4, the Cholesky of D + Y'ZZ'Y/θ = 3 + 1).
        assert!((wn[0 * two_m + 0] - 2.0).abs() < 1e-12);
        // wn[0,1] = 0.5 (from solving J^T x = wn1[1,0]/theta + sy off-diag = 1).
        assert!((wn[0 * two_m + 1] - 0.5).abs() < 1e-12);
        // wn[1,1] = √4.25 (Schur-updated and Cholesky-factored (2,2)).
        assert!((wn[1 * two_m + 1] - 4.25_f64.sqrt()).abs() < 1e-12);
    }

    /// `col == 1`, both variables free. No active set ⇒ `temp2 = 0`,
    /// `temp3 = 0` in Phase 1. (2,2) block of wn1 stays zero;
    /// (1,1) Cholesky gets `D + ‖y‖²`; (2,2) becomes a pure Schur term.
    #[test]
    fn col_one_both_free_zeroes_active_blocks() {
        let s = [1.0_f64, 2.0];
        let y = [1.0_f64, 1.0];
        let theta = 1.0;
        let m = 1;
        let two_m = 2 * m;

        let mut sy = vec![0.0_f64; m * m];
        sy[0] = s[0] * y[0] + s[1] * y[1]; // 3

        let ws_cols: Vec<&[f64]> = vec![&s];
        let wy_cols: Vec<&[f64]> = vec![&y];

        let ind = [0_usize, 1];
        let indx2 = [0_usize; 2];

        let mut wn = vec![0.0_f64; two_m * two_m];
        let mut wn1 = vec![0.0_f64; two_m * two_m];

        formk(
            &mut wn, &mut wn1, m, 1, theta, &sy, &ws_cols, &wy_cols, 2, // both free
            &ind, 0, 2, // no entering / leaving
            &indx2, 1, true,
        )
        .unwrap();

        // Y'ZZ'Y[0,0] = y·y = 2 (free part covers both vars).
        assert!((wn1[0 * two_m + 0] - 2.0).abs() < 1e-12);
        // S'AA'S[0,0] = 0 (no active variables).
        assert!((wn1[1 * two_m + 1] - 0.0).abs() < 1e-12);
        // (2,1)[0,0] is written twice: first 0 (no active set), then
        // overwritten by R_z = s·y over the free set = 1·1 + 2·1 = 3.
        assert!((wn1[1 * two_m + 0] - 3.0).abs() < 1e-12);

        // wn[0,0] = √(D + Y'ZZ'Y/θ) = √(3 + 2) = √5.
        assert!((wn[0 * two_m + 0] - 5.0_f64.sqrt()).abs() < 1e-12);
        // wn[0,1] = 3 / √5.
        assert!((wn[0 * two_m + 1] - 3.0 / 5.0_f64.sqrt()).abs() < 1e-12);
        // wn[1,1]: (2,2)_init = θ·S'AA'S = 0, Schur += (3/√5)² = 9/5;
        // sqrt = √(9/5).
        assert!((wn[1 * two_m + 1] - (9.0_f64 / 5.0).sqrt()).abs() < 1e-12);
    }

    /// Singular (1,1) block returns `NotPositiveDefiniteFirst`.
    /// Construct a degenerate case: `s = y = 0`, `θ = 1`. Then
    /// `D + Y'ZZ'Y/θ = 0`, Cholesky fails on first pivot.
    #[test]
    fn singular_first_block_returns_error() {
        let s = [0.0_f64, 0.0];
        let y = [0.0_f64, 0.0];
        let m = 1;
        let two_m = 2 * m;

        let sy = vec![0.0_f64; m * m];
        let ws_cols: Vec<&[f64]> = vec![&s];
        let wy_cols: Vec<&[f64]> = vec![&y];
        let ind = [0_usize, 1];
        let indx2 = [0_usize; 2];

        let mut wn = vec![0.0_f64; two_m * two_m];
        let mut wn1 = vec![0.0_f64; two_m * two_m];

        let res = formk(
            &mut wn, &mut wn1, m, 1, 1.0, &sy, &ws_cols, &wy_cols, 2, &ind, 0, 2, &indx2, 1, true,
        );
        assert_eq!(res, Err(FormkError::NotPositiveDefiniteFirst));
    }
}
