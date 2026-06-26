//! Bound-aware initialization of the shared [`QuadraticModel`] (BOBYQA, Powell
//! 2009, §2).
//!
//! BOBYQA seeds the same coordinate-cross interpolation set as NEWUOA (§3), but
//! clips the base point into `[a, b]` and, where a coordinate of `x0` sits on a
//! bound, places **both** of that coordinate's two points on the feasible side
//! (`+Δ` and `+2Δ`, or `−Δ` and `−2Δ`) instead of the symmetric `±Δ`. The
//! gradient, the diagonal of `∇²Q`, and the closed-form `H` factorization are
//! therefore written for the general step multipliers `α_i`, `β_i` of Powell
//! 2009, eq. 2.10 (the symmetric `α=Δ, β=−Δ` case recovers NEWUOA).
//!
//! Ported from PRIMA v0.7.2 (`fortran/bobyqa/initialize.f90`, subroutines
//! `initxf`/`initq`/`inith`, plus `common/xinbd.f90`), translated into
//! basin's [`QuadraticModel`] layout: `xpt` is `m × n` (row per point), the
//! gradient `gq` is stored at `x0` (not shifted to `x_opt`), the explicit
//! Hessian part is `Γ` with implicit coefficients `γ ≡ 0`, and the `Ω` block of
//! `H` is held as `Z Z^T` with all signs `+1`. Correctness is validated against
//! the dense KKT inverse (`assert_h_matches_inverse`) for both interior and
//! bound-active starts, so the convenient (PRIMA) choice of `Z` need not match
//! the paper's eq. 2.14 column for column.
//!
//! Only `m ≥ 2n+1` is implemented (the recommended case, as for NEWUOA); the
//! `n+2 ≤ m ≤ 2n` partial-model regime is deferred.

use crate::core::math::{DenseMatrix, Scalar};
use crate::solver::powell::QuadraticModel;

/// The product of [`try_initialize_bounded`](QuadraticModel::try_initialize_bounded):
/// the seeded model (whose `x0()` is the §2-adjusted base point) together with
/// the shifted bounds the driver/TRSBOX work in.
pub(crate) struct BoundedInit<F = f64> {
    /// The seeded quadratic model (interpolation set relative to `x0`).
    pub(crate) model: QuadraticModel<F>,
    /// Shifted lower bounds `sl = a − x0` (all `≤ 0`).
    pub(crate) sl: Vec<F>,
    /// Shifted upper bounds `su = b − x0` (all `≥ 0`).
    pub(crate) su: Vec<F>,
}

/// Adjust `x0` into the box leaving room for the coordinate steps (Powell 2009,
/// the paragraph around eq. 2.1). Mirrors PRIMA's default (`honour_x0 = .false.`)
/// revision in `preproc.f90`: clip to `[a, b]`, then apply the half-`ρ` split per
/// coordinate: within `½ρ` of a bound snaps *onto* the bound (asymmetric cross),
/// but between `½ρ` and `ρ` of a bound is pushed `ρ` *into the interior* (so a
/// near-bound start that has room keeps the symmetric cross). Requires
/// `b_i ≥ a_i + 2ρ`, which makes the lower and upper cases mutually exclusive.
fn adjust_x0<F: Scalar>(x0: &mut [F], lower: &[F], upper: &[F], rho: F) {
    let half = F::from_f64(0.5).expect("0.5 representable");
    for i in 0..x0.len() {
        let (a, b) = (lower[i], upper[i]);
        // Moderate into [a, b] first (PRIMA moderates x0 before this revision).
        let mut xi = x0[i].max(a).min(b);
        // Lower side: onto the bound within ½ρ, else ρ into the interior.
        if xi <= a + half * rho {
            xi = a;
        } else if xi < a + rho {
            xi = a + rho;
        }
        // Upper side (mutually exclusive with the above given b ≥ a + 2ρ).
        if xi >= b - half * rho {
            xi = b;
        } else if xi > b - rho {
            xi = b - rho;
        }
        x0[i] = xi;
    }
}

/// `x0 + step`, clamped to `[lower, upper]` and snapped exactly onto a bound when
/// the (possibly rounded) step reaches the shifted bound (PRIMA `xinbd`).
///
/// `pub(crate)` so RESCUE (the `super::rescue` module) can reuse it for its
/// re-evaluation of provisional points.
pub(crate) fn xinbd<F: Scalar>(
    x0: &[F],
    step: &[F],
    lower: &[F],
    upper: &[F],
    sl: &[F],
    su: &[F],
) -> Vec<F> {
    (0..x0.len())
        .map(|i| {
            let s = sl[i].max(su[i].min(step[i]));
            if s <= sl[i] {
                lower[i]
            } else if s >= su[i] {
                upper[i]
            } else {
                lower[i].max(upper[i].min(x0[i] + s))
            }
        })
        .collect()
}

/// The `t`-th extra interpolation point's coordinate pair `{p(j), q(j)}` for
/// `j = 2n+1+t` (0-based), cycling through `{1..n}` per Powell 2009, eq. 2.4
/// (identical to NEWUOA §3). Returned 0-based.
///
/// `pub(crate)` so RESCUE (the `super::rescue` module) can reuse it as PRIMA's
/// `setij` when rebuilding the provisional interpolation set.
pub(crate) fn extra_pair(t: usize, n: usize) -> (usize, usize) {
    let ip = 2 * n + 1 + t; // 0-based extra point index
    let jdiv = (ip - n - 1) / n; // ≥ 1
    let p1 = ip - n - jdiv * n; // 1-based coordinate in [1, n]
    let mut q1 = p1 + jdiv;
    if q1 > n {
        q1 -= n;
    }
    (p1 - 1, q1 - 1)
}

impl<F: Scalar> QuadraticModel<F> {
    /// Build BOBYQA's first model under box bounds (Powell 2009, §2).
    ///
    /// `eval` is called once per interpolation point (`m` evaluations) at an
    /// absolute point already clipped into `[lower, upper]`; the first `Err`
    /// aborts and bubbles to the caller. Returns the model plus the shifted
    /// bounds `sl`/`su` and the adjusted base point.
    ///
    /// # Panics
    ///
    /// Panics unless `n ≥ 1`, `2n+1 ≤ m ≤ ½(n+1)(n+2)`, and every coordinate has
    /// room `b_i ≥ a_i + 2ρ` for the construction (Powell 2009, eq. 2.1).
    pub(crate) fn try_initialize_bounded<E>(
        mut x0: Vec<F>,
        lower: &[F],
        upper: &[F],
        rho_beg: F,
        m: usize,
        eval: &mut impl FnMut(&[F]) -> Result<F, E>,
    ) -> Result<BoundedInit<F>, E> {
        let n = x0.len();
        assert!(n >= 1, "BOBYQA model needs n >= 1");
        assert!(
            m > 2 * n,
            "BOBYQA model init currently requires m >= 2n+1 (got m={m}, n={n}); \
             the n+2 <= m <= 2n partial-model case is not yet implemented"
        );
        assert!(
            m <= (n + 1) * (n + 2) / 2,
            "BOBYQA model requires m <= (n+1)(n+2)/2 (got m={m}, n={n})"
        );
        let two = F::from_f64(2.0).expect("2.0 representable");
        let rho = rho_beg;
        for i in 0..n {
            assert!(
                upper[i] >= lower[i] + two * rho,
                "BOBYQA requires b_i >= a_i + 2*rho_beg for room to sample \
                 (coordinate {i}: a={:?}, b={:?}, rho_beg={:?})",
                lower[i],
                upper[i],
                rho
            );
        }

        // --- 1. Feasibility adjustment of x0, then the shifted bounds. ---
        adjust_x0(&mut x0, lower, upper, rho);
        let sl: Vec<F> = (0..n).map(|i| lower[i] - x0[i]).collect();
        let su: Vec<F> = (0..n).map(|i| upper[i] - x0[i]).collect();
        let zero = F::zero();

        // --- 2. Interpolation displacements (Powell 2009, eq. 2.2 / eq. 2.10). ---
        let mut xpt = DenseMatrix::from_fn(m, n, |_, _| zero);
        let mut fval = vec![zero; m];

        for k in 0..n {
            // α point (k+1): +ρ, or −ρ when x0_k sits on the upper bound.
            let alpha = if su[k] <= zero { -rho } else { rho };
            xpt.set(k + 1, k, alpha);
            // β point (n+k+1): −ρ interior; +min(2ρ, su) at lower bound;
            // max(−2ρ, sl) at upper bound.
            let beta = if sl[k] >= zero {
                (two * rho).min(su[k])
            } else if su[k] <= zero {
                (-two * rho).max(sl[k])
            } else {
                -rho
            };
            xpt.set(k + 1 + n, k, beta);
        }

        // --- 3. Evaluate the first 2n+1 points. ---
        for k in 0..(2 * n + 1) {
            let disp = xpt.row(k).to_vec();
            fval[k] = eval(&xinbd(&x0, &disp, lower, upper, &sl, &su))?;
        }

        // --- 4. Reorder each interior coordinate's pair toward the lower F, ---
        // biasing the off-diagonal construction (PRIMA initxf; Powell 2009, the
        // paragraph after eq. 2.2). Only interior coordinates (α·β < 0).
        for k in 0..n {
            let a = xpt.get(k + 1, k);
            let b = xpt.get(k + 1 + n, k);
            if a * b < zero && fval[k + 1 + n] < fval[k + 1] {
                for c in 0..n {
                    let tmp = xpt.get(k + 1, c);
                    xpt.set(k + 1, c, xpt.get(k + 1 + n, c));
                    xpt.set(k + 1 + n, c, tmp);
                }
                fval.swap(k + 1, k + 1 + n);
            }
        }

        // --- 5. Extra points for the off-diagonal Hessian (m > 2n+1, eq. 2.3). ---
        let mut extras: Vec<(usize, usize, usize)> = Vec::new(); // (t, p, q)
        for t in 0..(m - 2 * n - 1) {
            let ip = 2 * n + 1 + t;
            let (p, q) = extra_pair(t, n);
            let dp = xpt.get(p + 1, p);
            let dq = xpt.get(q + 1, q);
            xpt.set(ip, p, dp);
            xpt.set(ip, q, dq);
            let disp = xpt.row(ip).to_vec();
            fval[ip] = eval(&xinbd(&x0, &disp, lower, upper, &sl, &su))?;
            extras.push((t, p, q));
        }

        // --- 6. Model gradient at x0, diagonal Γ, off-diagonal Γ (γ ≡ 0). ---
        let fbase = fval[0];
        let xa: Vec<F> = (0..n).map(|i| xpt.get(i + 1, i)).collect();
        let xb: Vec<F> = (0..n).map(|i| xpt.get(i + 1 + n, i)).collect();
        let mut gq = vec![zero; n];
        let mut gamma_explicit = DenseMatrix::from_fn(n, n, |_, _| zero);
        for i in 0..n {
            let fa = fval[i + 1] - fbase;
            let fb = fval[i + 1 + n] - fbase;
            // Two-point (general α/β) gradient and three-point diagonal Hessian.
            gq[i] = (fa / xa[i] * xb[i] - fb / xb[i] * xa[i]) / (xb[i] - xa[i]);
            gamma_explicit.set(i, i, two * (fa / xa[i] - fb / xb[i]) / (xa[i] - xb[i]));
        }
        for &(t, p, q) in &extras {
            let ip = 2 * n + 1 + t;
            let val =
                (fbase - fval[p + 1] - fval[q + 1] + fval[ip]) / (xpt.get(ip, p) * xpt.get(ip, q));
            gamma_explicit.set(p, q, val);
            gamma_explicit.set(q, p, val);
        }
        let gamma = vec![zero; m];

        // --- 7. Factored inverse-KKT H (PRIMA inith; Υ ≡ 0 for m ≥ 2n+1). ---
        let npt_minus = m - n - 1;
        let mut bmat_xi = DenseMatrix::from_fn(n, m, |_, _| zero);
        let bmat_ups = DenseMatrix::from_fn(n, n, |_, _| zero);
        let mut zmat = DenseMatrix::from_fn(m, npt_minus, |_, _| zero);
        let zsign = vec![F::one(); npt_minus];

        let half = F::from_f64(0.5).expect("0.5 representable");
        let rhosq = rho * rho;
        let sqrt2 = two.sqrt();
        let sqrt_half = half.sqrt();
        for i in 0..n {
            bmat_xi.set(i, 0, -(xa[i] + xb[i]) / (xa[i] * xb[i]));
            bmat_xi.set(i, i + n + 1, -half / xa[i]);
            bmat_xi.set(i, i + 1, -bmat_xi.get(i, 0) - bmat_xi.get(i, i + n + 1));

            zmat.set(0, i, sqrt2 / (xa[i] * xb[i]));
            zmat.set(i + 1, i, -zmat.get(0, i) - sqrt_half / rhosq);
            zmat.set(i + 1 + n, i, sqrt_half / rhosq);
        }
        for &(t, p, q) in &extras {
            let kk = n + t; // column for this extra point
            let ip = 2 * n + 1 + t;
            zmat.set(0, kk, F::one() / rhosq);
            zmat.set(ip, kk, F::one() / rhosq);
            zmat.set(p + 1, kk, -F::one() / rhosq);
            zmat.set(q + 1, kk, -F::one() / rhosq);
        }

        // --- 8. Best interpolation point. ---
        let mut kopt = 0;
        for j in 1..m {
            if fval[j] < fval[kopt] {
                kopt = j;
            }
        }

        let model = QuadraticModel {
            n,
            m,
            x0,
            xpt,
            fval,
            kopt,
            gq,
            gamma_explicit,
            gamma,
            bmat_xi,
            bmat_ups,
            zmat,
            zsign,
        };
        Ok(BoundedInit { model, sl, su })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::powell::kkt::assert_h_matches_inverse;

    /// Infallible wrapper for tests.
    fn init(
        x0: Vec<f64>,
        lower: &[f64],
        upper: &[f64],
        rho: f64,
        m: usize,
        f: impl Fn(&[f64]) -> f64,
    ) -> BoundedInit<f64> {
        QuadraticModel::try_initialize_bounded::<core::convert::Infallible>(
            x0,
            lower,
            upper,
            rho,
            m,
            &mut |x| Ok(f(x)),
        )
        .expect("infallible objective")
    }

    /// With a slack box and an interior start, BOBYQA's init reduces to NEWUOA's
    /// symmetric cross: gradient, diagonal, and off-diagonal Hessian are exact
    /// for a quadratic, and the KKT identity holds.
    #[test]
    fn interior_start_recovers_quadratic_and_kkt() {
        let g = [[4.0, 1.5], [1.5, 3.0]];
        let b = [1.0, 2.0];
        let f = |x: &[f64]| {
            0.5 * (g[0][0] * x[0] * x[0] + 2.0 * g[0][1] * x[0] * x[1] + g[1][1] * x[1] * x[1])
                + b[0] * x[0]
                + b[1] * x[1]
        };
        let out = init(vec![0.0, 0.0], &[-10.0, -10.0], &[10.0, 10.0], 0.3, 6, f);
        let model = &out.model;

        assert!((model.gamma_explicit.get(0, 0) - g[0][0]).abs() < 1e-12);
        assert!((model.gamma_explicit.get(1, 1) - g[1][1]).abs() < 1e-12);
        assert!((model.gamma_explicit.get(0, 1) - g[0][1]).abs() < 1e-12);
        assert!((model.gamma_explicit.get(1, 0) - g[0][1]).abs() < 1e-12);
        // gradient at x0 = 0 is b.
        assert!((model.gradient()[0] - b[0]).abs() < 1e-12);
        assert!((model.gradient()[1] - b[1]).abs() < 1e-12);
        assert_h_matches_inverse(model, 1e-10);
    }

    /// A start on the lower bound in one coordinate forces the asymmetric
    /// `+Δ, +2Δ` placement there. The model must still interpolate the quadratic
    /// exactly and satisfy the KKT identity (the general-`α/β` formulas).
    #[test]
    fn lower_bound_start_is_exact_and_kkt() {
        let f = |x: &[f64]| {
            2.0 * x[0] * x[0] + 1.5 * x[0] * x[1] + 3.0 * x[1] * x[1] + x[0] - 2.0 * x[1] + 0.7
        };
        // x0_0 sits exactly on the lower bound; x0_1 interior.
        let out = init(vec![0.0, 0.5], &[0.0, -5.0], &[5.0, 5.0], 0.25, 6, f);
        let model = &out.model;

        assert!(out.sl[0] == 0.0, "coord 0 pinned to lower bound");
        assert!(out.su[0] > 0.0);
        // Interpolates F at every point: eval_change(xj - x0) = F(xj) - F(x0).
        let x0 = model.x0();
        let f0 = f(x0);
        for j in 0..model.m() {
            let disp = model.xpt_row(j).to_vec();
            let xj: Vec<f64> = x0.iter().zip(&disp).map(|(a, d)| a + d).collect();
            assert!(
                (model.eval_change(&disp) - (f(&xj) - f0)).abs() < 1e-9,
                "interpolation at point {j}"
            );
        }
        assert_h_matches_inverse(model, 1e-9);
    }

    /// A start on the upper bound forces `−Δ, −2Δ`. KKT identity must hold, and
    /// x0 must be left on the bound (su = 0).
    #[test]
    fn upper_bound_start_is_kkt() {
        let f = |x: &[f64]| (x[0] - 3.0).powi(2) + 2.0 * (x[1] + 1.0).powi(2);
        let out = init(vec![2.0, 0.0], &[-5.0, -5.0], &[2.0, 5.0], 0.5, 5, f);
        assert!(out.su[0] == 0.0, "coord 0 pinned to upper bound");
        assert!(out.sl[0] < 0.0);
        assert_h_matches_inverse(&out.model, 1e-9);
    }

    /// PRIMA's half-`ρ` split (`preproc.f90`, `honour_x0 = .false.`): a start
    /// within `½ρ` of a bound snaps *onto* it, but one between `½ρ` and `ρ` is
    /// pushed `ρ` *into the interior* (so a near-bound start that has room keeps
    /// the symmetric cross rather than going bound-active).
    #[test]
    fn x0_half_rho_split() {
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let rho = 0.5; // half*rho = 0.25.
        // coord 0: 0.2 from the lower bound (< ½ρ) → snap to a = 0.
        // coord 1: 0.4 from the lower bound (½ρ < 0.4 < ρ) → pushed to a + ρ = 0.5.
        let out = init(vec![0.2, 0.4], &[0.0, 0.0], &[10.0, 10.0], rho, 5, f);
        assert_eq!(out.model.x0()[0], 0.0, "within ½ρ snaps to the bound");
        assert_eq!(out.model.x0()[1], 0.5, "between ½ρ and ρ pushed ρ interior");
        // coord 0 is bound-active (sl = 0); coord 1 is interior (sl < 0 < su).
        assert_eq!(out.sl[0], 0.0);
        assert!(out.sl[1] < 0.0 && out.su[1] > 0.0);
        assert_h_matches_inverse(&out.model, 1e-9);
    }

    /// The §2 feasibility adjustment moves an out-of-box or too-close start.
    #[test]
    fn x0_adjusted_into_box() {
        // x0_0 below lower bound; x0_1 within rho of the upper bound.
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let out = init(vec![-9.0, 4.9], &[-2.0, -5.0], &[5.0, 5.0], 0.5, 5, f);
        assert!(out.model.x0()[0] == -2.0, "x0_0 clipped to lower bound");
        assert!(out.model.x0()[1] == 5.0, "x0_1 pushed onto upper bound");
        assert_h_matches_inverse(&out.model, 1e-9);
    }
}
