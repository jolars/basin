//! Subspace minimization for L-BFGS-B.
//!
//! Port of `subsm` (`references/lbfgsb-v3.0/lbfgsb.f:2991`). Given
//! the Cauchy point `xcp`, the index set of free variables, the
//! reduced gradient at `xcp`, and the compact-form L-BFGS data, this
//! routine produces an approximate solution of
//!
//! ```text
//!     (P)   min Q(x) = rᵀ(x − xcp) + (1/2)(x − xcp)ᵀ B (x − xcp)
//!           s.t. l ≤ x ≤ u  and  x_i = xcp_i for all i ∈ A(xcp)
//! ```
//!
//! along the subspace unconstrained Newton direction
//! `d = −(ZᵀBZ)⁻¹ r`, where `Z` is the projection onto the free set.
//! Per Nocedal–Morales 2011 (the v3.0 remark), the Newton direction is
//! formed via the Sherman–Morrison–Woodbury identity:
//!
//! ```text
//!     d = (1/θ) r + (1/θ²) ZᵀW K⁻¹ WᵀZ r
//! ```
//!
//! with `K` the `2col × 2col` indefinite middle matrix
//!
//! ```text
//!     K = [ −D − YᵀZZᵀY/θ    L_aᵀ − R_zᵀ ]
//!         [ L_a − R_z         θ SᵀAAᵀS    ]
//! ```
//!
//! whose `L E Lᵀ` factorization (with `E = diag(−I, I)`) is precomputed
//! by `formk` (Stage 6) and supplied to this routine in `wn`.
//!
//! After forming `d` in the subspace coordinates, [`subsm`] projects
//! `xcp + d` back into the box. If any component hits a bound, the
//! `iword = 1` path is taken and a directional-derivative check at the
//! *original* iterate decides between accepting the projected step or
//! falling back to a uniform-α bound-backtracking step
//! (`lbfgsb.f:3273-3329`, the v3.0 deviation from Algorithm 778).

use super::compact::{solve_upper_tri, solve_upper_tri_transposed};
use crate::core::math::Scalar;

/// Status of the subspace solution returned from [`subsm`]. Mirrors
/// Fortran `iword`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SubsmStatus {
    /// `iword == 0`: the Newton step stayed inside the feasible box.
    InteriorStep,
    /// `iword == 1`: at least one bound was encountered (either by
    /// projection or by the bound-backtracking fallback).
    BoundEncountered,
}

/// Reasons [`subsm`] can fail. Matches Fortran `info ≠ 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SubsmError {
    /// `K` is ill-conditioned (zero/NaN/Inf pivot in the LEL^T
    /// factor stored in `wn`). Fortran's `info ≠ 0` from `dtrsl`.
    SingularK,
}

/// Compute the subspace minimizer.
///
/// # Parameters
///
/// - `x`: length `n`. On entry: the Cauchy point `xcp` (output of
///   [`super::cauchy::cauchy`]). On exit: the subspace minimizer.
/// - `d`: length ≥ `nsub` (the algorithm uses `d[0..nsub]`). On
///   entry: the reduced gradient `r` at `xcp` (produced by Fortran
///   `cmprlb`, indexed by subspace position). On exit: the Newton
///   direction in the same subspace ordering. `d[nsub..]` is left
///   untouched.
/// - `xp`: length `n`. Scratch slot for the projected-Newton
///   safeguard (holds `x` before projection so it can be restored if
///   the backtracking branch fires).
/// - `xx`: length `n`. The current iterate (the outer-loop `x`,
///   *not* `xcp`). Used by the directional-derivative check.
/// - `gg`: length `n`. The gradient at `xx`.
/// - `ind`: length `nsub`. Coordinate indices of free variables.
/// - `l`, `u`: length `n`. Bounds with `±∞` for missing sides
///   (basin's `BoxConstraints` convention). Fortran's `nbd(k)` code
///   is recovered per-component from `l[k].is_finite()` /
///   `u[k].is_finite()`.
/// - `ws_cols`, `wy_cols`: `col` history columns, oldest first;
///   each inner slice has length `n`.
/// - `wn`: `2m × 2m` row-major, leading `2*col × 2*col` upper triangle
///   stores the `L E Lᵀ` factor of `K` (built by `formk` in Stage 6;
///   tests may construct it directly for small fixtures).
/// - `wv`: length ≥ `2*m` scratch for `WᵀZ d` and the middle-system
///   solve.
/// - `m`, `col`, `theta`: compact-form parameters matching the data
///   stored in `wn`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn subsm<F: Scalar>(
    x: &mut [F],
    d: &mut [F],
    xp: &mut [F],
    xx: &[F],
    gg: &[F],
    ind: &[usize],
    l: &[F],
    u: &[F],
    ws_cols: &[&[F]],
    wy_cols: &[&[F]],
    wn: &[F],
    wv: &mut [F],
    m: usize,
    col: usize,
    theta: F,
) -> Result<SubsmStatus, SubsmError> {
    let n = x.len();
    let nsub = ind.len();
    debug_assert_eq!(xp.len(), n);
    debug_assert_eq!(xx.len(), n);
    debug_assert_eq!(gg.len(), n);
    debug_assert_eq!(l.len(), n);
    debug_assert_eq!(u.len(), n);
    debug_assert!(d.len() >= nsub);
    debug_assert_eq!(ws_cols.len(), col);
    debug_assert_eq!(wy_cols.len(), col);
    debug_assert!(col <= m);
    debug_assert!(wv.len() >= 2 * m);

    if nsub == 0 {
        return Ok(SubsmStatus::InteriorStep);
    }

    let zero = F::zero();
    let one = F::one();
    let m2 = 2 * m;
    let col2 = 2 * col;

    // Step 1: wv = WᵀZ d.
    //   wv[i]       = Σ_j wy[k, i] · d[j]            k = ind[j]
    //   wv[col + i] = θ · Σ_j ws[k, i] · d[j]
    for i in 0..col {
        let mut temp1 = zero;
        let mut temp2 = zero;
        for j in 0..nsub {
            let k = ind[j];
            temp1 = temp1 + wy_cols[i][k] * d[j];
            temp2 = temp2 + ws_cols[i][k] * d[j];
        }
        wv[i] = temp1;
        wv[col + i] = theta * temp2;
    }

    // Step 2: wv := K⁻¹ wv via the L E Lᵀ factorization in `wn`.
    //   Forward solve (`dtrsl(..., job=11)`):  L wv = wv.
    //   Apply E = diag(−I, I):                negate top half.
    //   Backward solve (`dtrsl(..., job=01)`): Lᵀ wv = wv.
    if col > 0 {
        // Check K's diagonal upfront (singular pivot ⇒ SubsmError).
        for i in 0..col2 {
            let pivot = wn[i * m2 + i];
            if pivot == zero || !pivot.is_finite() {
                return Err(SubsmError::SingularK);
            }
        }
        solve_upper_tri_transposed(wn, col2, m2, &mut wv[0..col2]);
        for slot in wv.iter_mut().take(col) {
            *slot = -*slot;
        }
        solve_upper_tri(wn, col2, m2, &mut wv[0..col2]);
    }

    // Step 3: d ← (1/θ) d + (1/θ²) ZᵀW wv.
    //   Increment: d[i] += wy[k, jy] · wv[jy] / θ + ws[k, jy] · wv[col + jy].
    //   Then d ← d / θ to absorb the remaining 1/θ.
    for jy in 0..col {
        let js = col + jy;
        for i in 0..nsub {
            let k = ind[i];
            d[i] = d[i] + wy_cols[jy][k] * wv[jy] / theta + ws_cols[jy][k] * wv[js];
        }
    }
    for slot in d.iter_mut().take(nsub) {
        *slot = *slot / theta;
    }

    // Step 4: projected Newton step. Save x first so the backtracking
    // branch can restore it.
    xp.copy_from_slice(x);
    let mut iword = SubsmStatus::InteriorStep;
    for i in 0..nsub {
        let k = ind[i];
        let dk = d[i];
        let xk = x[k];
        let l_finite = l[k].is_finite();
        let u_finite = u[k].is_finite();
        if !l_finite && !u_finite {
            x[k] = xk + dk;
        } else if l_finite && !u_finite {
            let new_x = (xk + dk).max(l[k]);
            x[k] = new_x;
            if new_x == l[k] {
                iword = SubsmStatus::BoundEncountered;
            }
        } else if l_finite && u_finite {
            let tmp = (xk + dk).max(l[k]);
            x[k] = tmp.min(u[k]);
            if x[k] == l[k] || x[k] == u[k] {
                iword = SubsmStatus::BoundEncountered;
            }
        } else {
            // upper only
            let new_x = (xk + dk).min(u[k]);
            x[k] = new_x;
            if new_x == u[k] {
                iword = SubsmStatus::BoundEncountered;
            }
        }
    }

    if iword == SubsmStatus::InteriorStep {
        return Ok(iword);
    }

    // Step 5: sign of the directional derivative at the original
    // iterate `xx`. If `(x − xx)ᵀ gg > 0` the projected step ascends
    // at `xx`; fall through to the bound-backtracking branch
    // (`lbfgsb.f:3273-3286`).
    let mut dd_p = zero;
    for i in 0..n {
        dd_p = dd_p + (x[i] - xx[i]) * gg[i];
    }
    if dd_p <= zero {
        return Ok(iword);
    }

    // Step 6: restore x = xp and find the largest uniform α ∈ [0, 1]
    // keeping `xp + α d` feasible. Mirrors Fortran's loop 60 + branch
    // at 3319 (`lbfgsb.f:3290-3329`). Note: `temp1` is NOT reset per
    // iteration: when a variable doesn't bind, temp1 retains its
    // previous value (which equals `alpha` after the prior `if` flush),
    // so the `temp1 < alpha` check correctly skips non-binders.
    x.copy_from_slice(xp);
    let mut alpha = one;
    let mut temp1 = alpha;
    let mut ibd: Option<usize> = None;

    for i in 0..nsub {
        let k = ind[i];
        let dk = d[i];
        let l_finite = l[k].is_finite();
        let u_finite = u[k].is_finite();
        if !l_finite && !u_finite {
            continue;
        }
        if dk < zero && l_finite {
            let temp2 = l[k] - x[k];
            if temp2 >= zero {
                temp1 = zero;
            } else if dk * alpha < temp2 {
                temp1 = temp2 / dk;
            }
        } else if dk > zero && u_finite {
            let temp2 = u[k] - x[k];
            if temp2 <= zero {
                temp1 = zero;
            } else if dk * alpha > temp2 {
                temp1 = temp2 / dk;
            }
        }
        if temp1 < alpha {
            alpha = temp1;
            ibd = Some(i);
        }
    }

    if alpha < one {
        if let Some(ibd) = ibd {
            let dk = d[ibd];
            let k = ind[ibd];
            if dk > zero {
                x[k] = u[k];
                d[ibd] = zero;
            } else if dk < zero {
                x[k] = l[k];
                d[ibd] = zero;
            }
        }
    }
    for i in 0..nsub {
        let k = ind[i];
        x[k] = x[k] + alpha * d[i];
    }

    Ok(iword)
}

#[cfg(test)]
// Explicit `i * m2 + j` indexing (including `0 * m2 + 0`) mirrors the
// Fortran source's 2-D layout: load-bearing for readability when
// cross-checking against `lbfgsb.f`.
#[allow(clippy::identity_op, clippy::erasing_op)]
mod tests {
    use super::*;

    /// `nsub == 0` ⇒ no free variables ⇒ subsm returns immediately
    /// with `iword = 0` and `x` unchanged.
    #[test]
    fn empty_subspace_is_no_op() {
        let mut x = vec![0.5_f64, 1.5];
        let x_in = x.clone();
        let mut d = Vec::<f64>::new();
        let mut xp = vec![0.0_f64; 2];
        let xx = vec![0.0_f64; 2];
        let gg = vec![0.0_f64; 2];
        let ind: Vec<usize> = Vec::new();
        let l = vec![0.0, 0.0];
        let u = vec![2.0, 2.0];
        let ws_cols: Vec<&[f64]> = Vec::new();
        let wy_cols: Vec<&[f64]> = Vec::new();
        let wn = Vec::<f64>::new();
        let mut wv = vec![0.0; 2];

        let status = subsm(
            &mut x, &mut d, &mut xp, &xx, &gg, &ind, &l, &u, &ws_cols, &wy_cols, &wn, &mut wv, 1,
            0, 1.0,
        )
        .unwrap();
        assert_eq!(status, SubsmStatus::InteriorStep);
        assert_eq!(x, x_in);
    }

    /// `col == 0` ⇒ no compact-form correction ⇒ `d ← d/θ` followed by
    /// componentwise feasible projection. With θ = 1 and `d_in = −g`,
    /// the result is a steepest-descent step from `xcp` clipped to the
    /// box.
    #[test]
    fn col_zero_unbounded_uses_scaled_reduced_gradient() {
        // n = 2, both vars free, both unbounded.
        // Suppose Cauchy left xcp = (1.0, 2.0) and the reduced
        // gradient at xcp is r = (3.0, −1.0). Newton direction is
        // (1/θ)·r = r (θ = 1). Subspace minimizer = xcp + r = (4, 1).
        let mut x = vec![1.0, 2.0];
        let mut d = vec![3.0, -1.0];
        let mut xp = vec![0.0; 2];
        let xx = vec![0.0, 0.0];
        let gg = vec![0.0, 0.0];
        let ind = vec![0usize, 1];
        let l = vec![f64::NEG_INFINITY, f64::NEG_INFINITY];
        let u = vec![f64::INFINITY, f64::INFINITY];
        let ws_cols: Vec<&[f64]> = Vec::new();
        let wy_cols: Vec<&[f64]> = Vec::new();
        let wn = Vec::<f64>::new();
        let mut wv = vec![0.0; 2];

        let status = subsm(
            &mut x, &mut d, &mut xp, &xx, &gg, &ind, &l, &u, &ws_cols, &wy_cols, &wn, &mut wv, 1,
            0, 1.0,
        )
        .unwrap();
        assert_eq!(status, SubsmStatus::InteriorStep);
        assert!((x[0] - 4.0).abs() < 1e-12);
        assert!((x[1] - 1.0).abs() < 1e-12);
        // d_out = d_in / θ = (3, −1).
        assert!((d[0] - 3.0).abs() < 1e-12);
        assert!((d[1] - (-1.0)).abs() < 1e-12);
    }

    /// `col == 0` with `θ = 2` scales the Newton direction
    /// proportionally.
    #[test]
    fn col_zero_theta_two_halves_the_step() {
        let mut x = vec![1.0];
        let mut d = vec![4.0];
        let mut xp = vec![0.0; 1];
        let xx = vec![0.0];
        let gg = vec![0.0];
        let ind = vec![0usize];
        let l = vec![f64::NEG_INFINITY];
        let u = vec![f64::INFINITY];
        let ws_cols: Vec<&[f64]> = Vec::new();
        let wy_cols: Vec<&[f64]> = Vec::new();
        let wn = Vec::<f64>::new();
        let mut wv = vec![0.0; 2];
        subsm(
            &mut x, &mut d, &mut xp, &xx, &gg, &ind, &l, &u, &ws_cols, &wy_cols, &wn, &mut wv, 1,
            0, 2.0,
        )
        .unwrap();
        // d_in = 4, θ = 2 ⇒ d_out = 2, x_new = 1 + 2 = 3.
        assert!((d[0] - 2.0).abs() < 1e-12);
        assert!((x[0] - 3.0).abs() < 1e-12);
    }

    /// `col == 0` with a binding upper bound. xcp = 0.5, d_in = 1.0,
    /// θ = 1 ⇒ Newton step lands at 1.5 but u = 1.0 clips. iword
    /// becomes `BoundEncountered`.
    #[test]
    fn col_zero_step_clips_at_upper_bound() {
        let mut x = vec![0.5];
        let mut d = vec![1.0];
        let mut xp = vec![0.0; 1];
        // Setting xx = x so the directional-derivative check at the
        // current iterate is zero (gradient irrelevant); we'll get
        // iword=1 from the projection itself but not trigger
        // backtracking.
        let xx = vec![0.5];
        let gg = vec![0.0];
        let ind = vec![0usize];
        let l = vec![f64::NEG_INFINITY];
        let u = vec![1.0];
        let ws_cols: Vec<&[f64]> = Vec::new();
        let wy_cols: Vec<&[f64]> = Vec::new();
        let wn = Vec::<f64>::new();
        let mut wv = vec![0.0; 2];

        let status = subsm(
            &mut x, &mut d, &mut xp, &xx, &gg, &ind, &l, &u, &ws_cols, &wy_cols, &wn, &mut wv, 1,
            0, 1.0,
        )
        .unwrap();
        assert_eq!(status, SubsmStatus::BoundEncountered);
        assert_eq!(x[0], 1.0);
        // d is updated to d_in / θ before the projection runs.
        assert!((d[0] - 1.0).abs() < 1e-12);
    }

    /// `col == 0` with a binding lower bound, both-bounded variable.
    /// xcp = 0.5, d_in = −1.0, θ = 1 ⇒ projected step lands at l = 0.
    #[test]
    fn col_zero_step_clips_at_lower_bound_both_bounded() {
        let mut x = vec![0.5];
        let mut d = vec![-1.0];
        let mut xp = vec![0.0; 1];
        let xx = vec![0.5];
        let gg = vec![0.0];
        let ind = vec![0usize];
        let l = vec![0.0];
        let u = vec![1.0];
        let ws_cols: Vec<&[f64]> = Vec::new();
        let wy_cols: Vec<&[f64]> = Vec::new();
        let wn = Vec::<f64>::new();
        let mut wv = vec![0.0; 2];

        let status = subsm(
            &mut x, &mut d, &mut xp, &xx, &gg, &ind, &l, &u, &ws_cols, &wy_cols, &wn, &mut wv, 1,
            0, 1.0,
        )
        .unwrap();
        assert_eq!(status, SubsmStatus::BoundEncountered);
        assert_eq!(x[0], 0.0);
    }

    /// `col == 1` with one free + one active variable. Construct `wn`
    /// by hand following `formk`'s block-Cholesky (Cholesky of the
    /// (1,1) block `D + YᵀZZᵀY/θ`, then Schur-update the (2,2) block
    /// with the `+B·A⁻¹·Bᵀ` correction that produces the L·E·Lᵀ
    /// factorization of the indefinite `K`, then Cholesky of the
    /// updated (2,2)).
    ///
    /// Setup: `n = 2`, `s = (1, 2)`, `y = (1, 1)`, θ = 1, var 0 free,
    /// var 1 active at lower bound `l[1] = 0`. With this history, the
    /// compact-form Hessian is
    /// `B = [[17/15, −1/15]; [−1/15, 8/15]]`, and the subspace Newton
    /// step from `xcp[0] = 0` is `(Z'BZ)⁻¹ · d_in = (15/17) · 1`. The
    /// test verifies `d_out` and the resulting `x[0]` against that
    /// closed form.
    #[test]
    fn col_one_one_free_one_active_matches_closed_form_newton() {
        // History pair.
        let s = [1.0_f64, 2.0];
        let y = [1.0_f64, 1.0];
        let theta = 1.0_f64;

        // Hand-build `wn` (2m × 2m, stride m2 = 2, leading 2×2 used).
        // Per formk for col=1, theta=1, active = {var 1}:
        //   wn(1,1) = √(D + Y'ZZ'Y/θ) = √(s·y + y0²) = √(3 + 1) = 2.
        //   wn(1,2) = (forward sub of L_A on −L_a'+R_z' = +s0·y0) / L_A
        //           = s0·y0 / L_A = 1 / 2 = 0.5.
        //   Schur update: (2,2) += wn(1,2)² = 0.25, then (2,2) base
        //   = θ·s1² = 4 ⇒ becomes 4.25 ⇒ wn(2,2) = √4.25.
        let m = 1;
        let m2 = 2 * m;
        let mut wn = vec![0.0_f64; m2 * m2];
        wn[0 * m2 + 0] = 2.0;
        wn[0 * m2 + 1] = 0.5;
        wn[1 * m2 + 1] = 4.25_f64.sqrt();

        // Run subsm with d_in[0] = 1, xcp = (0, 0).
        let mut x = vec![0.0_f64, 0.0];
        let mut d = vec![1.0_f64];
        let mut xp = vec![0.0_f64; 2];
        let xx = vec![0.0_f64, 0.0];
        let gg = vec![0.0_f64, 0.0];
        let ind = vec![0usize];
        let l = vec![f64::NEG_INFINITY, 0.0];
        let u = vec![f64::INFINITY, f64::INFINITY];
        let ws_cols: Vec<&[f64]> = vec![&s];
        let wy_cols: Vec<&[f64]> = vec![&y];
        let mut wv = vec![0.0_f64; 2 * m];

        let status = subsm(
            &mut x, &mut d, &mut xp, &xx, &gg, &ind, &l, &u, &ws_cols, &wy_cols, &wn, &mut wv, m,
            1, theta,
        )
        .unwrap();
        assert_eq!(status, SubsmStatus::InteriorStep);
        // Closed form: (Z'BZ)⁻¹ · d_in = 15/17.
        let expected = 15.0 / 17.0;
        assert!(
            (d[0] - expected).abs() < 1e-12,
            "d_out = {} vs {}",
            d[0],
            expected
        );
        assert!(
            (x[0] - expected).abs() < 1e-12,
            "x[0] = {} vs {}",
            x[0],
            expected
        );
        // Active variable untouched.
        assert_eq!(x[1], 0.0);
    }

    /// `col == 0` backtracking branch: project gives a positive
    /// directional derivative at `xx`, so subsm restores `x = xp` then
    /// applies the uniform-α step. Construct a 2-D case where:
    ///
    /// - Variable 0 has bounds [0, 1]; xcp[0] = 0.5; d_in[0] points
    ///   toward upper bound (d_in[0] = 2 ⇒ Newton tries 0.5 + 2 = 2.5,
    ///   clamped to 1.0).
    /// - Variable 1 unbounded; xcp[1] = 0.0; d_in[1] = −0.1 (a tiny
    ///   descent in component 1).
    /// - xx = (0, 0), gg = (+10, −0.1) (so moving x[0] up is uphill).
    ///
    /// Then `(x_proj − xx) · gg = (1.0 − 0)·10 + (−0.1 − 0)·(−0.1)
    /// = 10.01 > 0` ⇒ backtracking branch fires. Bound-backtracking
    /// caps α at `(u[0] − xp[0]) / d_in[0] = (1 − 0.5) / 2 = 0.25`.
    /// Bound applied: x[0] = u[0] = 1.0, d[0] = 0. Then
    /// `x[1] = xp[1] + 0.25 · d_in[1] = 0 − 0.025`.
    #[test]
    fn col_zero_backtracking_caps_step_at_binding_variable() {
        let mut x = vec![0.5, 0.0];
        let mut d = vec![2.0, -0.1];
        let mut xp = vec![0.0; 2];
        let xx = vec![0.0, 0.0];
        let gg = vec![10.0, -0.1];
        let ind = vec![0usize, 1];
        let l = vec![0.0, f64::NEG_INFINITY];
        let u = vec![1.0, f64::INFINITY];
        let ws_cols: Vec<&[f64]> = Vec::new();
        let wy_cols: Vec<&[f64]> = Vec::new();
        let wn = Vec::<f64>::new();
        let mut wv = vec![0.0; 2];

        let status = subsm(
            &mut x, &mut d, &mut xp, &xx, &gg, &ind, &l, &u, &ws_cols, &wy_cols, &wn, &mut wv, 1,
            0, 1.0,
        )
        .unwrap();
        assert_eq!(status, SubsmStatus::BoundEncountered);
        // Variable 0 pinned to upper bound, d[0] zeroed.
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert_eq!(d[0], 0.0);
        // Variable 1 stepped by alpha = 0.25 along d_in[1] = −0.1.
        assert!((x[1] - (-0.025)).abs() < 1e-12);
    }
}
