//! Generalized Cauchy point for L-BFGS-B.
//!
//! Port of `cauchy` (`references/lbfgsb-v3.0/lbfgsb.f:1222`) — given
//! the current iterate `x`, gradient `g`, bounds `(l, u)`, and the
//! compact-form L-BFGS data `(ws, wy, sy, wt, theta)`, compute the
//! first local minimizer of
//!
//! ```text
//!     Q(x + s) = gᵀs + (1/2) sᵀ B s
//! ```
//!
//! along the projected gradient path `P(x − t g, l, u)`. The result
//! `x_c` (the GCP) is what subspace minimization (`subsm`) then
//! refines.
//!
//! All array arguments are `&[f64]` / `&mut [f64]`; the surrounding
//! solver supplies them by viewing the active backend's storage as a
//! slice (nalgebra `DVector::as_slice`, faer
//! `Col::try_as_col_major().unwrap().as_slice()`, or plain
//! `Vec::as_slice()`).

use super::compact::{bmv, BmvError};

/// `iwhere` classification per Fortran v3.0 (`lbfgsb.f:1287-1296`).
/// Stored as `i8` so `−3` round-trips correctly while staying compact.
pub(crate) mod iwhere {
    /// Variable is free, has bounds, has not been moved by Cauchy.
    pub(crate) const FREE_NOT_MOVED: i8 = -3;
    /// Variable is always free (no bounds; `l = −∞ ∧ u = +∞`).
    pub(crate) const ALWAYS_FREE: i8 = -1;
    /// Variable is free, has bounds, will be moved.
    pub(crate) const FREE_MOVED: i8 = 0;
    /// Variable is fixed at its lower bound.
    pub(crate) const AT_LOWER: i8 = 1;
    /// Variable is fixed at its upper bound.
    pub(crate) const AT_UPPER: i8 = 2;
    /// Variable is always fixed (degenerate `l = u`).
    pub(crate) const ALWAYS_FIXED: i8 = 3;
}

/// Diagnostic / return values from [`cauchy`]. Mirrors what Fortran
/// `cauchy` leaves in its working arrays on exit (the `nseg` counter
/// and a flag indicating the entire ray was bounded). The solver
/// doesn't read these fields — they exist for unit-test
/// introspection and to keep the function signature self-
/// documenting.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct CauchyResult {
    /// Number of quadratic segments explored (Fortran `nseg`).
    pub(crate) nseg: usize,
    /// True iff every nonzero component of the search direction `d`
    /// is bounded — Fortran's `bnded` flag, which gates the
    /// `f1 = f2 = dtm = 0` post-loop branch.
    pub(crate) bounded: bool,
}

/// Reasons [`cauchy`] can fail. Matches Fortran's nonzero `info`
/// exit, which only fires via `bmv`'s singular-`J` path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CauchyError {
    /// The middle-matrix solve in `bmv` failed (zero pivot in `J`).
    /// Mirrors Fortran `info ≠ 0` returned by `cauchy`.
    SingularMiddleMatrix,
}

impl From<BmvError> for CauchyError {
    fn from(_: BmvError) -> Self {
        Self::SingularMiddleMatrix
    }
}

/// Compute the generalized Cauchy point. See module docs for parameter
/// roles. Bounds use `±∞` for missing sides (basin's `BoxConstraints`
/// convention); the Fortran `nbd(i)` code is recovered per-component
/// from `l[i].is_finite() / u[i].is_finite()`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cauchy(
    x: &[f64],
    l: &[f64],
    u: &[f64],
    g: &[f64],
    ws_cols: &[&[f64]],
    wy_cols: &[&[f64]],
    sy: &[f64],
    wt: &[f64],
    m: usize,
    theta: f64,
    sbgnrm: f64,
    xcp: &mut [f64],
    d: &mut [f64],
    t: &mut [f64],
    iwhere: &mut [i8],
    iorder: &mut [usize],
    p_buf: &mut [f64],
    c_buf: &mut [f64],
    wbp_buf: &mut [f64],
    v_buf: &mut [f64],
) -> Result<CauchyResult, CauchyError> {
    let n = x.len();
    let col = ws_cols.len();
    debug_assert_eq!(col, wy_cols.len());
    debug_assert!(col <= m);
    debug_assert_eq!(l.len(), n);
    debug_assert_eq!(u.len(), n);
    debug_assert_eq!(g.len(), n);
    debug_assert_eq!(xcp.len(), n);
    debug_assert_eq!(d.len(), n);
    debug_assert_eq!(t.len(), n);
    debug_assert_eq!(iwhere.len(), n);
    debug_assert_eq!(iorder.len(), n);
    debug_assert!(p_buf.len() >= 2 * m && c_buf.len() >= 2 * m);
    debug_assert!(wbp_buf.len() >= 2 * m && v_buf.len() >= 2 * m);

    // Early exit when the projected gradient is zero (`lbfgsb.f:1423`).
    if sbgnrm <= 0.0 {
        xcp.copy_from_slice(x);
        return Ok(CauchyResult {
            nseg: 0,
            bounded: true,
        });
    }

    let col2 = 2 * col;
    let mut bnded = true;
    let mut nfree = n;
    let mut nbreak: usize = 0;
    let mut ibkmin: usize = 0;
    let mut bkmin: f64 = 0.0;
    let mut f1: f64 = 0.0;

    // Zero the leading 2*col of p_buf (Fortran loop 20).
    for slot in p_buf.iter_mut().take(col2) {
        *slot = 0.0;
    }

    // Per-variable classify / direction / breakpoint (Fortran loop 50).
    for i in 0..n {
        let neggi = -g[i];
        if iwhere[i] != iwhere::ALWAYS_FIXED && iwhere[i] != iwhere::ALWAYS_FREE {
            let lower_finite = l[i].is_finite();
            let upper_finite = u[i].is_finite();
            let tl = if lower_finite { x[i] - l[i] } else { 0.0 };
            let tu = if upper_finite { u[i] - x[i] } else { 0.0 };
            let xlower = lower_finite && tl <= 0.0;
            let xupper = upper_finite && tu <= 0.0;
            iwhere[i] = iwhere::FREE_MOVED;
            if xlower {
                if neggi <= 0.0 {
                    iwhere[i] = iwhere::AT_LOWER;
                }
            } else if xupper {
                if neggi >= 0.0 {
                    iwhere[i] = iwhere::AT_UPPER;
                }
            } else if neggi.abs() <= 0.0 {
                iwhere[i] = iwhere::FREE_NOT_MOVED;
            }
        }

        if iwhere[i] != iwhere::FREE_MOVED && iwhere[i] != iwhere::ALWAYS_FREE {
            d[i] = 0.0;
        } else {
            d[i] = neggi;
            f1 -= neggi * neggi;
            // p ← p − Wᵀ eᵢ · gᵢ.
            for j in 0..col {
                p_buf[j] += wy_cols[j][i] * neggi;
                p_buf[col + j] += ws_cols[j][i] * neggi;
            }
            let lower_finite = l[i].is_finite();
            let upper_finite = u[i].is_finite();
            if lower_finite && neggi < 0.0 {
                let tnew = (x[i] - l[i]) / -neggi;
                iorder[nbreak] = i;
                t[nbreak] = tnew;
                if nbreak == 0 || tnew < bkmin {
                    bkmin = tnew;
                    ibkmin = nbreak;
                }
                nbreak += 1;
            } else if upper_finite && neggi > 0.0 {
                let tnew = (u[i] - x[i]) / neggi;
                iorder[nbreak] = i;
                t[nbreak] = tnew;
                if nbreak == 0 || tnew < bkmin {
                    bkmin = tnew;
                    ibkmin = nbreak;
                }
                nbreak += 1;
            } else {
                // No bound in the search direction — park index at
                // the tail of iorder (Fortran's `nfree` walks down).
                nfree -= 1;
                iorder[nfree] = i;
                if neggi.abs() > 0.0 {
                    bnded = false;
                }
            }
        }
    }

    // θ-scale the second half of p (`lbfgsb.f:1514`).
    if theta != 1.0 {
        for slot in p_buf.iter_mut().skip(col).take(col) {
            *slot *= theta;
        }
    }

    // Initialize xcp = x.
    xcp.copy_from_slice(x);

    if nbreak == 0 && nfree == n {
        // d = 0 — already feasible-stationary.
        return Ok(CauchyResult {
            nseg: 0,
            bounded: true,
        });
    }

    // Zero c = Wᵀ(xcp − x).
    for slot in c_buf.iter_mut().take(col2) {
        *slot = 0.0;
    }

    // Initial f2 = −θ f1 − pᵀ M p.
    let mut f2 = -theta * f1;
    let f2_org = f2;
    if col > 0 {
        bmv(sy, wt, col, m, p_buf, v_buf)?;
        for j in 0..col2 {
            f2 -= v_buf[j] * p_buf[j];
        }
    }

    let mut dtm = -f1 / f2;
    let mut tsum = 0.0;
    let mut nseg: usize = 1;

    if nbreak == 0 {
        return Ok(finalize_with_free_move(
            d, xcp, dtm, tsum, c_buf, p_buf, col2, bnded, nseg,
        ));
    }

    let mut nleft = nbreak;
    let mut iter: usize = 1;
    let mut tj: f64 = 0.0;
    let mut all_pinned_early = false;

    loop {
        let tj0 = tj;
        let ibp;
        if iter == 1 {
            tj = bkmin;
            ibp = iorder[ibkmin];
        } else {
            if iter == 2 && ibkmin != nbreak - 1 {
                // Overwrite the now-empty slot (where bkmin used to
                // live) with the entry at position nbreak−1 so that
                // the active heap range [0..nleft) holds exactly the
                // remaining (nleft) breakpoints (`lbfgsb.f:1578-1581`).
                t[ibkmin] = t[nbreak - 1];
                iorder[ibkmin] = iorder[nbreak - 1];
            }
            hpsolb(nleft, t, iorder, iter == 2);
            tj = t[nleft - 1];
            ibp = iorder[nleft - 1];
        }

        let dt = tj - tj0;
        if dtm < dt {
            // Stationary minimum is inside the current segment.
            break;
        }

        tsum += dt;
        nleft -= 1;
        iter += 1;
        let dibp = d[ibp];
        d[ibp] = 0.0;
        let zibp;
        if dibp > 0.0 {
            zibp = u[ibp] - x[ibp];
            xcp[ibp] = u[ibp];
            iwhere[ibp] = iwhere::AT_UPPER;
        } else {
            zibp = l[ibp] - x[ibp];
            xcp[ibp] = l[ibp];
            iwhere[ibp] = iwhere::AT_LOWER;
        }

        if nleft == 0 && nbreak == n {
            // Every variable now pinned to a bound; no free move left.
            dtm = dt;
            all_pinned_early = true;
            break;
        }

        nseg += 1;
        let dibp2 = dibp * dibp;
        f1 = f1 + dt * f2 + dibp2 - theta * dibp * zibp;
        f2 -= theta * dibp2;

        if col > 0 {
            // c ← c + dt · p.
            for j in 0..col2 {
                c_buf[j] += dt * p_buf[j];
            }
            // wbp = row of W at variable `ibp`.
            for j in 0..col {
                wbp_buf[j] = wy_cols[j][ibp];
                wbp_buf[col + j] = theta * ws_cols[j][ibp];
            }
            // v = M · wbp.
            bmv(sy, wt, col, m, wbp_buf, v_buf)?;
            let mut wmc = 0.0;
            let mut wmp = 0.0;
            let mut wmw = 0.0;
            for j in 0..col2 {
                wmc += c_buf[j] * v_buf[j];
                wmp += p_buf[j] * v_buf[j];
                wmw += wbp_buf[j] * v_buf[j];
            }
            // p ← p − dibp · wbp.
            for j in 0..col2 {
                p_buf[j] -= dibp * wbp_buf[j];
            }
            f1 += dibp * wmc;
            f2 += 2.0 * dibp * wmp - dibp2 * wmw;
        }

        // Floor f2 at `eps · f2_org` (`lbfgsb.f:1666`).
        let floor = f64::EPSILON * f2_org;
        if f2 < floor {
            f2 = floor;
        }

        if nleft > 0 {
            dtm = -f1 / f2;
        } else if bnded {
            // Entire ray was bounded — pin to current xcp.
            f1 = 0.0;
            let _ = f1; // explicit zero matches Fortran but unused below
            dtm = 0.0;
            all_pinned_early = true;
            break;
        } else {
            dtm = -f1 / f2;
            break;
        }
    }

    if all_pinned_early {
        // Fortran exit 999: only update c, no further free move.
        if dtm <= 0.0 {
            dtm = 0.0;
        }
        for j in 0..col2 {
            c_buf[j] += dtm * p_buf[j];
        }
        return Ok(CauchyResult {
            nseg,
            bounded: bnded,
        });
    }

    Ok(finalize_with_free_move(
        d, xcp, dtm, tsum, c_buf, p_buf, col2, bnded, nseg,
    ))
}

/// Apply `xcp ← xcp + (tsum + dtm) · d` and `c ← c + dtm · p`.
/// Matches Fortran's exit path through label `888` then `999`.
#[allow(clippy::too_many_arguments)]
fn finalize_with_free_move(
    d: &[f64],
    xcp: &mut [f64],
    mut dtm: f64,
    tsum: f64,
    c_buf: &mut [f64],
    p_buf: &[f64],
    col2: usize,
    bnded: bool,
    nseg: usize,
) -> CauchyResult {
    if dtm <= 0.0 {
        dtm = 0.0;
    }
    let total = tsum + dtm;
    for i in 0..xcp.len() {
        xcp[i] += total * d[i];
    }
    for j in 0..col2 {
        c_buf[j] += dtm * p_buf[j];
    }
    CauchyResult {
        nseg,
        bounded: bnded,
    }
}

/// Heap-based "extract min" over `t[0..nleft]` with parallel index
/// permutation in `iorder`. Port of Fortran `hpsolb` (`lbfgsb.f:2341`).
///
/// On entry:
///
/// - `nleft` is the current heap size.
/// - `t[0..nleft]` and `iorder[0..nleft]` hold the heap entries.
/// - `first == true` triggers a one-time min-heap build (Fortran
///   `iheap == 0`).
///
/// On exit:
///
/// - `t[nleft − 1]` and `iorder[nleft − 1]` hold the smallest entry.
/// - `t[0..nleft − 1]` and `iorder[0..nleft − 1]` remain a valid
///   min-heap.
///
/// Callers decrement `nleft` themselves after consuming the popped min.
fn hpsolb(nleft: usize, t: &mut [f64], iorder: &mut [usize], first: bool) {
    if first {
        // Build a min-heap on `t[0..nleft]` by sift-up insertion of
        // positions 1..nleft.
        for k in 1..nleft {
            let ddum = t[k];
            let indxin = iorder[k];
            let mut i = k;
            while i > 0 {
                let parent = (i - 1) / 2;
                if ddum < t[parent] {
                    t[i] = t[parent];
                    iorder[i] = iorder[parent];
                    i = parent;
                } else {
                    break;
                }
            }
            t[i] = ddum;
            iorder[i] = indxin;
        }
    }
    if nleft <= 1 {
        return;
    }
    // Extract min: pop root, sift down the last entry from position 0,
    // place the saved root at the freed tail.
    let out = t[0];
    let out_i = iorder[0];
    let ddum = t[nleft - 1];
    let ddum_i = iorder[nleft - 1];

    let active = nleft - 1;
    let mut i: usize = 0;
    loop {
        let left = 2 * i + 1;
        if left >= active {
            break;
        }
        let right = left + 1;
        let child = if right < active && t[right] < t[left] {
            right
        } else {
            left
        };
        if t[child] < ddum {
            t[i] = t[child];
            iorder[i] = iorder[child];
            i = child;
        } else {
            break;
        }
    }
    t[i] = ddum;
    iorder[i] = ddum_i;
    t[nleft - 1] = out;
    iorder[nleft - 1] = out_i;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to allocate the working buffers cauchy expects.
    struct Buffers {
        xcp: Vec<f64>,
        d: Vec<f64>,
        t: Vec<f64>,
        iwhere: Vec<i8>,
        iorder: Vec<usize>,
        p: Vec<f64>,
        c: Vec<f64>,
        wbp: Vec<f64>,
        v: Vec<f64>,
    }

    impl Buffers {
        fn new(n: usize, m: usize) -> Self {
            Self {
                xcp: vec![0.0; n],
                d: vec![0.0; n],
                t: vec![0.0; n],
                iwhere: vec![iwhere::FREE_MOVED; n],
                iorder: vec![0; n],
                p: vec![0.0; 2 * m],
                c: vec![0.0; 2 * m],
                wbp: vec![0.0; 2 * m],
                v: vec![0.0; 2 * m],
            }
        }
    }

    #[test]
    fn zero_projected_gradient_returns_x() {
        // sbgnrm = 0 → early exit with xcp = x and no segments.
        let x = vec![0.5, 1.5];
        let l = vec![0.0, 0.0];
        let u = vec![2.0, 2.0];
        let g = vec![0.0, 0.0];
        let ws_cols: Vec<&[f64]> = Vec::new();
        let wy_cols: Vec<&[f64]> = Vec::new();
        let sy = Vec::<f64>::new();
        let wt = Vec::<f64>::new();
        let mut b = Buffers::new(2, 1);
        let res = cauchy(
            &x,
            &l,
            &u,
            &g,
            &ws_cols,
            &wy_cols,
            &sy,
            &wt,
            1,
            1.0,
            0.0,
            &mut b.xcp,
            &mut b.d,
            &mut b.t,
            &mut b.iwhere,
            &mut b.iorder,
            &mut b.p,
            &mut b.c,
            &mut b.wbp,
            &mut b.v,
        )
        .unwrap();
        assert_eq!(b.xcp, x);
        assert_eq!(res.nseg, 0);
    }

    #[test]
    fn col_zero_unconstrained_step_is_steepest_descent() {
        // n = 2, no bounds, no history (col = 0). Quadratic
        // q(α) = gᵀ(−αg) + (1/2)(−αg)ᵀ I (−αg) = −α‖g‖² + (α²/2)‖g‖²,
        // minimum at α = 1 ⇒ xcp = x − g.
        let x = vec![3.0, 4.0];
        let l = vec![f64::NEG_INFINITY; 2];
        let u = vec![f64::INFINITY; 2];
        let g = vec![1.0, 2.0];
        let ws_cols: Vec<&[f64]> = Vec::new();
        let wy_cols: Vec<&[f64]> = Vec::new();
        let sy = Vec::<f64>::new();
        let wt = Vec::<f64>::new();
        let mut b = Buffers::new(2, 1);
        // Pre-fill iwhere as ALWAYS_FREE for the two unbounded vars
        // (the solver's init does this; here we set it directly).
        for slot in b.iwhere.iter_mut() {
            *slot = iwhere::ALWAYS_FREE;
        }
        let sbgnrm = g.iter().cloned().map(f64::abs).fold(0.0, f64::max);
        let res = cauchy(
            &x,
            &l,
            &u,
            &g,
            &ws_cols,
            &wy_cols,
            &sy,
            &wt,
            1,
            1.0,
            sbgnrm,
            &mut b.xcp,
            &mut b.d,
            &mut b.t,
            &mut b.iwhere,
            &mut b.iorder,
            &mut b.p,
            &mut b.c,
            &mut b.wbp,
            &mut b.v,
        )
        .unwrap();
        assert!((b.xcp[0] - 2.0).abs() < 1e-12);
        assert!((b.xcp[1] - 2.0).abs() < 1e-12);
        // No segments crossed — single piece, no breakpoints.
        assert_eq!(res.nseg, 1);
    }

    #[test]
    fn col_zero_with_active_upper_bound_pins_at_breakpoint() {
        // 1-D, x = 0.5, g = −1 (so direction +1), upper bound u = 1.
        // q(α) = −α + α²/2 (theta = 1, B0 = I). Unconstrained min is
        // at α* = 1, but the breakpoint at α = (u − x)/(-g) = 0.5
        // arrives first. After hitting the breakpoint, all variables
        // are pinned ⇒ exit via "all pinned" branch with xcp = u.
        let x = vec![0.5];
        let l = vec![f64::NEG_INFINITY];
        let u = vec![1.0];
        let g = vec![-1.0];
        let ws_cols: Vec<&[f64]> = Vec::new();
        let wy_cols: Vec<&[f64]> = Vec::new();
        let sy = Vec::<f64>::new();
        let wt = Vec::<f64>::new();
        let mut b = Buffers::new(1, 1);
        let sbgnrm = 1.0;
        let res = cauchy(
            &x,
            &l,
            &u,
            &g,
            &ws_cols,
            &wy_cols,
            &sy,
            &wt,
            1,
            1.0,
            sbgnrm,
            &mut b.xcp,
            &mut b.d,
            &mut b.t,
            &mut b.iwhere,
            &mut b.iorder,
            &mut b.p,
            &mut b.c,
            &mut b.wbp,
            &mut b.v,
        )
        .unwrap();
        assert!((b.xcp[0] - 1.0).abs() < 1e-12);
        assert_eq!(b.iwhere[0], iwhere::AT_UPPER);
        assert_eq!(res.nseg, 1);
    }

    #[test]
    fn already_at_lower_with_outward_gradient_pins_immediately() {
        // x = 0 = l, g = +1 (so −g = −1 points outward, would step
        // below l). Fortran's `xlower && neggi ≤ 0` ⇒ iwhere = 1
        // ⇒ d = 0 ⇒ xcp = x.
        let x = vec![0.0];
        let l = vec![0.0];
        let u = vec![1.0];
        let g = vec![1.0];
        let ws_cols: Vec<&[f64]> = Vec::new();
        let wy_cols: Vec<&[f64]> = Vec::new();
        let sy = Vec::<f64>::new();
        let wt = Vec::<f64>::new();
        let mut b = Buffers::new(1, 1);
        let sbgnrm = 1.0;
        let res = cauchy(
            &x,
            &l,
            &u,
            &g,
            &ws_cols,
            &wy_cols,
            &sy,
            &wt,
            1,
            1.0,
            sbgnrm,
            &mut b.xcp,
            &mut b.d,
            &mut b.t,
            &mut b.iwhere,
            &mut b.iorder,
            &mut b.p,
            &mut b.c,
            &mut b.wbp,
            &mut b.v,
        )
        .unwrap();
        assert_eq!(b.xcp, x);
        assert_eq!(b.iwhere[0], iwhere::AT_LOWER);
        // No breakpoints classified, no free move ⇒ nseg = 0.
        assert_eq!(res.nseg, 0);
    }

    #[test]
    fn col_one_unconstrained_uses_compact_quadratic() {
        // 1-D, n = 1, no bounds, history of one pair (s = 1, y = 2)
        // ⇒ sy[0,0] = 2, ss[0,0] = 1, theta = y·y / s·y = 4/2 = 2.
        // The compact B has B = θ − Wᵀ M⁻¹ W with W = [y; θ s] and
        // M = [−d; θ ss]⁻¹ (col = 1 simplifies fully).
        //
        // For 1-D with one history pair, B reduces to a scalar `b`.
        // Cauchy then takes α = ‖g‖² / (θ ‖g‖² + ...) — and since the
        // problem is unbounded, the GCP is simply x − α g for that α.
        //
        // Rather than work out the closed-form `b`, we check that:
        //   (a) the function succeeds,
        //   (b) xcp moves *opposite* the gradient direction,
        //   (c) nseg == 1 (no breakpoints crossed).
        let x = vec![5.0];
        let l = vec![f64::NEG_INFINITY];
        let u = vec![f64::INFINITY];
        let g = vec![1.0];
        let s_hist = [1.0_f64];
        let y_hist = [2.0_f64];
        let ws_cols: Vec<&[f64]> = vec![&s_hist];
        let wy_cols: Vec<&[f64]> = vec![&y_hist];

        let m = 1;
        let mut sy = vec![0.0; m * m];
        let mut ss = vec![0.0; m * m];
        sy[0] = 2.0; // s·y
        ss[0] = 1.0; // s·s
        let theta = 4.0 / 2.0;
        let mut wt = vec![0.0; m * m];
        super::super::compact::formt(theta, &sy, &ss, 1, m, &mut wt).unwrap();

        let mut b = Buffers::new(1, m);
        for slot in b.iwhere.iter_mut() {
            *slot = iwhere::ALWAYS_FREE;
        }
        let res = cauchy(
            &x,
            &l,
            &u,
            &g,
            &ws_cols,
            &wy_cols,
            &sy,
            &wt,
            m,
            theta,
            1.0,
            &mut b.xcp,
            &mut b.d,
            &mut b.t,
            &mut b.iwhere,
            &mut b.iorder,
            &mut b.p,
            &mut b.c,
            &mut b.wbp,
            &mut b.v,
        )
        .unwrap();
        assert!(b.xcp[0] < x[0], "expected xcp < x for g > 0");
        assert_eq!(res.nseg, 1);
    }

    #[test]
    fn col_one_with_breakpoint_pins_at_bound() {
        // 1-D, x = 0.5, g = −1 (so −g moves toward upper bound), u = 1,
        // l = −∞. With a history pair, the curvature is θ − wᵀ M⁻¹ w
        // (positive); the unconstrained min sits at some α* > 0. The
        // bound is at α_b = (u − x) / 1 = 0.5. If α* > α_b the GCP is
        // exactly u; we choose the (s, y) pair to make this hold.
        //
        // For col = 1, B = θ ‖d‖² − p_top² / d − p_bot² / (θ ss),
        // (the bmv-derived denominator). Pick theta = 1, s = 1, y = 1
        // ⇒ sy = 1, ss = 1. Then the curvature in the d = +1
        // direction is 1 − (−1)²/1 + (−1·1)²/(1·1) = ... but easier:
        // numerically check `xcp = u`.
        let x = vec![0.5];
        let l = vec![f64::NEG_INFINITY];
        let u = vec![1.0];
        let g = vec![-1.0];
        let s_hist = [1.0_f64];
        let y_hist = [1.0_f64];
        let ws_cols: Vec<&[f64]> = vec![&s_hist];
        let wy_cols: Vec<&[f64]> = vec![&y_hist];

        let m = 1;
        let mut sy = vec![0.0; m * m];
        let mut ss = vec![0.0; m * m];
        sy[0] = 1.0;
        ss[0] = 1.0;
        let theta = 1.0_f64;
        let mut wt = vec![0.0; m * m];
        super::super::compact::formt(theta, &sy, &ss, 1, m, &mut wt).unwrap();

        let mut b = Buffers::new(1, m);
        let res = cauchy(
            &x,
            &l,
            &u,
            &g,
            &ws_cols,
            &wy_cols,
            &sy,
            &wt,
            m,
            theta,
            1.0,
            &mut b.xcp,
            &mut b.d,
            &mut b.t,
            &mut b.iwhere,
            &mut b.iorder,
            &mut b.p,
            &mut b.c,
            &mut b.wbp,
            &mut b.v,
        )
        .unwrap();
        // The unconstrained min sits at α* = 1 (since b = θ − wᵀ M⁻¹ w
        // works out to 1 here for the (1, 1) pair), but the breakpoint
        // at α_b = 0.5 binds first. Either way xcp should land at the
        // bound u = 1 since the test problem is 1-D with one breakpoint.
        assert!((b.xcp[0] - 1.0).abs() < 1e-12);
        assert_eq!(b.iwhere[0], iwhere::AT_UPPER);
        assert_eq!(res.nseg, 1);
    }

    #[test]
    fn hpsolb_extracts_smallest_and_leaves_heap() {
        // Build a heap over [3.0, 1.0, 4.0, 1.5, 9.0, 2.6] with
        // index permutation [10, 11, 12, 13, 14, 15]. Iterate pops.
        let mut t = vec![3.0, 1.0, 4.0, 1.5, 9.0, 2.6];
        let mut iorder = vec![10usize, 11, 12, 13, 14, 15];
        let mut nleft = t.len();
        let mut popped = Vec::new();
        let mut first = true;
        while nleft > 0 {
            hpsolb(nleft, &mut t, &mut iorder, first);
            first = false;
            popped.push((t[nleft - 1], iorder[nleft - 1]));
            nleft -= 1;
        }
        // Expected sorted: 1.0, 1.5, 2.6, 3.0, 4.0, 9.0.
        let mut expected = vec![
            (3.0, 10),
            (1.0, 11),
            (4.0, 12),
            (1.5, 13),
            (9.0, 14),
            (2.6, 15),
        ];
        expected.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        assert_eq!(popped, expected);
    }
}
