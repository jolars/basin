//! LINCOA geometry: point-to-drop selection, the geometry-improving step, and
//! the constraint-residual update (Powell 2015; PRIMA `geometry.f90` +
//! `update.f90::updateres`).
//!
//! - [`setdrop_tr`] picks which interpolation point to replace after a
//!   trust-region step, scoring each candidate by a distance weight times the
//!   update denominator (PRIMA `setdrop_tr`).
//! - [`geostep`] chooses a geometry-improving step for a chosen point by
//!   approximately maximizing the modulus of the update denominator `σ`. It
//!   tries three candidates: a line search through `x_opt` and another
//!   interpolation point, a steepest-ascent step of the `knew`-th Lagrange
//!   function, and (when constraints are active) a **projected** gradient step
//!   onto the null space of the active normals, and selects by `|σ|` and
//!   feasibility. The projected step is what makes LINCOA effective on
//!   constrained problems.
//! - [`update_rescon`] maintains `rescon` after `x_opt` moves (PRIMA
//!   `updateres`): near-active residuals are recomputed, far ones decay, and
//!   any residual `≥ delta` is stored **negated** (the far-constraint encoding
//!   [`trstep`](super::trstep) consumes).
//!
//! The update denominator `den(k)` (PRIMA `calden`) comes from the shared
//! model's [`prepare_update`](QuadraticModel::prepare_update) /
//! [`update_params`](QuadraticModel::update_params) `σ`; the `knew`-th Lagrange
//! function's gradient and Hessian from
//! [`lagrange_coeffs`](QuadraticModel::lagrange_coeffs) /
//! [`lagrange_hessian_matvec`](QuadraticModel::lagrange_hessian_matvec), exactly
//! as BOBYQA's geometry does.

use crate::core::math::Scalar;
use crate::solver::powell::QuadraticModel;

use super::getact::ActiveSetQr;

/// `a_jᵀ v` where `a_j` is column `j` of the `n × m` column-major `amat`.
fn col_dot<F: Scalar>(v: &[F], amat: &[F], j: usize, n: usize) -> F {
    (0..n).fold(F::zero(), |acc, r| acc + v[r] * amat[r + j * n])
}

/// Plain dot product.
fn dot<F: Scalar>(a: &[F], b: &[F]) -> F {
    a.iter().zip(b).fold(F::zero(), |acc, (x, y)| acc + *x * *y)
}

/// Euclidean norm.
fn norm<F: Scalar>(v: &[F]) -> F {
    dot(v, v).sqrt()
}

/// The update denominator `σ` for replacing each interpolation point `k` with
/// `x_opt + d` (PRIMA `calden`), via one `prepare_update` and a per-`k`
/// `update_params`. `d` is relative to `x_opt`.
fn denominators<F: Scalar>(model: &QuadraticModel<F>, d: &[F]) -> Vec<F> {
    let n = model.n();
    let kopt = model.kopt();
    let xopt = model.xpt_row(kopt).to_vec();
    let xnew_disp: Vec<F> = (0..n).map(|i| xopt[i] + d[i]).collect();
    let ctx = model.prepare_update(&xnew_disp);
    (0..model.m())
        .map(|k| model.update_params(k, &ctx).sigma)
        .collect()
}

/// `‖x_t − center‖²` for every interpolation point.
fn distsq_to<F: Scalar>(model: &QuadraticModel<F>, center: &[F]) -> Vec<F> {
    let n = model.n();
    (0..model.m())
        .map(|k| {
            let row = model.xpt_row(k);
            (0..n).fold(F::zero(), |a, i| {
                a + (row[i] - center[i]) * (row[i] - center[i])
            })
        })
        .collect()
}

/// Project `g` onto the null space of the active constraint normals, i.e. onto
/// the span of columns `nact..n` of the orthogonal factor `qfac` (`n × n`,
/// column-major). Matches the conjugate-gradient projection in
/// [`trstep`](super::trstep).
fn project_null<F: Scalar>(
    qfac: &[F],
    n: usize,
    nact: usize,
    g: &[F],
) -> Vec<F> {
    let mut p = vec![F::zero(); n];
    for col in nact..n {
        let coeff = (0..n).fold(F::zero(), |a, r| a + g[r] * qfac[r + col * n]);
        for r in 0..n {
            p[r] = p[r] + coeff * qfac[r + col * n];
        }
    }
    p
}

/// Choose the interpolation point to drop after a trust-region step (PRIMA
/// LINCOA `setdrop_tr`). `d` is the step relative to `x_opt`; `ximproved`
/// indicates `F(x_opt + d) < F(x_opt)`. Returns `None` (PRIMA `KNEW = 0`) when no
/// point should be replaced.
///
/// The distance weight uses power 3 and scale `max(0.1·delta, rho)²` (LINCOA's
/// choice, distinct from BOBYQA's), and the acceptance test is simply
/// `any(score > 0)`.
pub(crate) fn setdrop_tr<F: Scalar>(
    model: &QuadraticModel<F>,
    ximproved: bool,
    d: &[F],
    delta: F,
    rho: F,
) -> Option<usize> {
    let n = model.n();
    let kopt = model.kopt();
    let zero = F::zero();
    let one = F::one();
    let tenth = F::from_f64(0.1).expect("0.1 representable");

    let center: Vec<F> = if ximproved {
        let xopt = model.xpt_row(kopt);
        (0..n).map(|i| xopt[i] + d[i]).collect()
    } else {
        model.xpt_row(kopt).to_vec()
    };
    let distsq = distsq_to(model, &center);
    let scale = (tenth * delta).max(rho);
    let scale2 = scale * scale;
    let den = denominators(model, d);

    let mut score = vec![zero; model.m()];
    for k in 0..model.m() {
        let w = (distsq[k] / scale2).max(one);
        let w3 = w * w * w; // LINCOA weight power 3
        score[k] = w3 * den[k].abs();
    }
    if !ximproved {
        score[kopt] = -one;
    }

    if score.iter().any(|&s| s > zero) {
        let mut knew = 0;
        let mut best = F::neg_infinity();
        for k in 0..model.m() {
            if !score[k].is_nan() && score[k] > best {
                best = score[k];
                knew = k;
            }
        }
        Some(knew)
    } else if ximproved {
        let mut knew = 0;
        let mut best = F::neg_infinity();
        for k in 0..model.m() {
            if !distsq[k].is_nan() && distsq[k] > best {
                best = distsq[k];
                knew = k;
            }
        }
        Some(knew)
    } else {
        None
    }
}

/// A geometry-improving step `s` (relative to `x_opt`) for replacing `XPT(knew)`
/// (Powell 2015; PRIMA LINCOA `geostep`). `delbar` is the geometry trust radius;
/// `amat`/`rescon` and the active-set QR `warm` describe the linear feasible
/// region. Returns `(s, feasible)` with `‖s‖ ≈ delbar`; `feasible` reports
/// whether `x_opt + s` satisfies the relevant inequality constraints.
pub(crate) fn geostep<F: Scalar>(
    model: &QuadraticModel<F>,
    knew: usize,
    delbar: F,
    amat: &[F],
    rescon: &[F],
    warm: &ActiveSetQr<F>,
) -> (Vec<F>, bool) {
    let n = model.n();
    let m = rescon.len();
    let kopt = model.kopt();
    let zero = F::zero();
    let one = F::one();
    let half = F::from_f64(0.5).expect("0.5 representable");
    let ten = F::from_f64(10.0).expect("10 representable");
    let eps = F::epsilon();

    let xopt = model.xpt_row(kopt).to_vec();

    // KNEW-th Lagrange function: implicit coeffs `pqlag`, gradient `glag` at x_opt.
    let (g0, pqlag) = model.lagrange_coeffs(knew);
    let hxopt = model.lagrange_hessian_matvec(&pqlag, &xopt);
    let glag: Vec<F> = (0..n).map(|i| g0[i] + hxopt[i]).collect();

    // --- Candidate 1: line search through XOPT and another interpolation point. ---
    let mut distsq = distsq_to(model, &xopt);
    distsq[kopt] = one; // artificial positive (avoids div-by-zero; unused)
    let dderiv: Vec<F> = (0..model.m())
        .map(|k| {
            let row = model.xpt_row(k);
            (0..n).fold(zero, |a, i| a + glag[i] * (row[i] - xopt[i]))
        })
        .collect();

    let mut stplen: Vec<F> =
        (0..model.m()).map(|k| -delbar / distsq[k].sqrt()).collect();
    let mut vlagabs: Vec<F> = (0..model.m())
        .map(|k| (stplen[k] * (one - stplen[k]) * dderiv[k]).abs())
        .collect();
    // KNEW gets the maximizer of |PHI_knew(t)| (PHI(0)=0, PHI(1)=1).
    if dderiv[knew] * (dderiv[knew] - one) < zero {
        stplen[knew] = -stplen[knew];
    }
    vlagabs[knew] = (stplen[knew] * dderiv[knew]).abs()
        + stplen[knew] * stplen[knew] * (dderiv[knew] - one).abs();
    vlagabs[kopt] = -one;

    // Pick the line K (default KNEW unless another strictly exceeds it).
    let mut kline = knew;
    if (0..model.m())
        .any(|k| vlagabs[k] > vlagabs[knew] && !vlagabs[k].is_nan())
    {
        let mut best = F::neg_infinity();
        for k in 0..model.m() {
            if !vlagabs[k].is_nan() && vlagabs[k] > best {
                best = vlagabs[k];
                kline = k;
            }
        }
    }
    let row = model.xpt_row(kline);
    let mut s: Vec<F> =
        (0..n).map(|i| stplen[kline] * (row[i] - xopt[i])).collect();
    let mut denabs = denominators(model, &s)[knew].abs();

    // --- Candidate 2: steepest-ascent step of the Lagrange function. ---
    let gnorm = norm(&glag);
    if gnorm > eps && gnorm.is_finite() {
        let mut gstp: Vec<F> =
            (0..n).map(|i| (delbar / gnorm) * glag[i]).collect();
        let hg = model.lagrange_hessian_matvec(&pqlag, &gstp);
        if dot(&gstp, &hg) < zero {
            for v in gstp.iter_mut() {
                *v = -*v;
            }
        }
        let den_knew = denominators(model, &gstp)[knew].abs();
        if den_knew > denabs || denabs.is_nan() {
            denabs = den_knew;
            s = gstp;
        }
    }

    // RSTAT: -1 irrelevant (|rescon| >= delbar), 0 active, 1 inactive & relevant.
    let mut rstat = vec![1i8; m];
    for j in 0..m {
        if rescon[j].abs() >= delbar {
            rstat[j] = -1;
        }
    }
    for ic in 0..warm.nact {
        rstat[warm.iact[ic]] = 0;
    }

    // Feasibility of the current S (relevant constraints: rstat >= 0).
    let cstrv_of = |step: &[F], pred: &dyn Fn(i8) -> bool| -> F {
        let mut cv = zero;
        for j in 0..m {
            if pred(rstat[j]) {
                let viol = col_dot(step, amat, j, n) - rescon[j];
                if viol > cv {
                    cv = viol;
                }
            }
        }
        cv
    };
    let mut feasible = cstrv_of(&s, &|r| r >= 0) <= zero;

    // --- Candidate 3: projected gradient step (only when constraints active). ---
    if warm.nact > 0 {
        let pglag = project_null(&warm.qfac, n, warm.nact, &glag);
        let pgnorm = norm(&pglag);
        if pgnorm > eps && pgnorm.is_finite() {
            let mut pgstp: Vec<F> =
                (0..n).map(|i| (delbar / pgnorm) * pglag[i]).collect();
            let hpg = model.lagrange_hessian_matvec(&pqlag, &pgstp);
            if dot(&pgstp, &hpg) < zero {
                for v in pgstp.iter_mut() {
                    *v = -*v;
                }
            }
            // Constraint violation over inactive & relevant constraints only.
            let cstrv = cstrv_of(&pgstp, &|r| r == 1);
            // Rounding tolerance: active normals are orthogonal to PGSTP in theory.
            let mut active_inf = zero;
            for ic in 0..warm.nact {
                let v = col_dot(&pgstp, amat, warm.iact[ic], n).abs();
                if v > active_inf {
                    active_inf = v;
                }
            }
            let cvtol = (eps * norm(&pgstp)).max(ten * active_inf);
            let mut take = false;
            if cstrv <= cvtol {
                let tenth = F::from_f64(0.1).expect("0.1 representable");
                take = denominators(model, &pgstp)[knew].abs() > tenth * denabs;
            }
            if take || denabs.is_nan() {
                s = pgstp;
                feasible = cstrv <= cvtol;
            }
        }
    }

    // --- Fallback: a finite displacement toward XPT(knew) if S is NaN. ---
    if s.iter().any(|v| v.is_nan()) {
        let raw: Vec<F> =
            (0..n).map(|i| model.xpt_row(knew)[i] - xopt[i]).collect();
        let scaling = delbar / norm(&raw);
        let factor = (F::from_f64(0.6).expect("0.6 representable") * scaling)
            .max(half.min(scaling));
        s = raw.iter().map(|&v| factor * v).collect();
        feasible = cstrv_of(&s, &|r| r >= 0) <= zero;
    }

    (s, feasible)
}

/// Update `rescon` after `x_opt` has moved by a step of norm `dnorm` (PRIMA
/// `updateres`). Only runs when `ximproved` (Powell's code does not refresh
/// otherwise). Near-active residuals (`|rescon| < dnorm + delta`) are recomputed
/// from `b − A x_opt`; far ones decay toward `−delta`; any residual `≥ delta` is
/// stored **negated** so [`trstep`](super::trstep) treats it as far or inactive.
///
/// `bvec`/`xopt` are in the model's `x0`-relative coordinates (so
/// `b − a_jᵀ x_opt` is the true residual), and `amat` is the `n × m` column-major
/// normal matrix.
#[allow(clippy::too_many_arguments)]
pub(crate) fn update_rescon<F: Scalar>(
    ximproved: bool,
    amat: &[F],
    bvec: &[F],
    delta: F,
    dnorm: F,
    xopt: &[F],
    rescon: &mut [F],
    n: usize,
) {
    if !ximproved {
        return;
    }
    let zero = F::zero();
    let m = bvec.len();
    for j in 0..m {
        if rescon[j].abs() < dnorm + delta {
            let ax = col_dot(xopt, amat, j, n);
            rescon[j] = (bvec[j] - ax).max(zero);
        } else {
            rescon[j] = (-rescon[j].abs() + dnorm).min(-delta);
        }
        if rescon[j] >= delta {
            rescon[j] = -rescon[j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diag_quadratic(
        h: &'static [f64],
        c: &'static [f64],
    ) -> impl Fn(&[f64]) -> f64 {
        move |x: &[f64]| {
            0.5 * (0..x.len())
                .map(|i| h[i] * (x[i] - c[i]).powi(2))
                .sum::<f64>()
        }
    }

    /// `update_rescon` recomputes a near-active residual and negates a far one.
    #[test]
    fn rescon_encoding_near_and_far() {
        // m = 2: constraint 0 near-active, constraint 1 far.
        // amat columns (n=1): [1.0], [1.0]; bvec = [0.5, 10.0]; xopt = [0.0].
        let amat = [1.0f64, 1.0];
        let bvec = [0.5f64, 10.0];
        let xopt = [0.0f64];
        let mut rescon = [0.4f64, 9.0];
        update_rescon(true, &amat, &bvec, 1.0, 0.3, &xopt, &mut rescon, 1);
        // Constraint 0: |0.4| < 0.3 + 1.0 → recompute b - a·xopt = 0.5; < delta=1 → stays.
        assert!((rescon[0] - 0.5).abs() < 1e-12);
        // Constraint 1: |9.0| >= 1.3 → min(-9 + 0.3, -1) = -8.7 (far, negative).
        assert!((rescon[1] - (-8.7)).abs() < 1e-12);
    }

    /// A near residual recomputed to `≥ delta` is stored negated.
    #[test]
    fn rescon_negates_when_ge_delta() {
        let amat = [1.0f64];
        let bvec = [2.0f64];
        let xopt = [0.0f64];
        let mut rescon = [0.5f64];
        update_rescon(true, &amat, &bvec, 1.0, 0.3, &xopt, &mut rescon, 1);
        // |0.5| < 1.3 → recompute 2.0; 2.0 >= delta=1 → negate → -2.0.
        assert!((rescon[0] - (-2.0)).abs() < 1e-12);
    }

    /// `update_rescon` is a no-op when the step did not improve `x_opt`.
    #[test]
    fn rescon_no_update_when_not_improved() {
        let amat = [1.0f64];
        let bvec = [2.0f64];
        let xopt = [0.0f64];
        let mut rescon = [0.4f64];
        update_rescon(false, &amat, &bvec, 1.0, 0.3, &xopt, &mut rescon, 1);
        assert_eq!(rescon[0], 0.4);
    }

    /// Unconstrained `setdrop_tr` returns a droppable point for an improving step.
    #[test]
    fn setdrop_returns_point_for_improving_step() {
        let h = &[2.0, 4.0];
        let c = &[0.4, -0.3];
        let model = QuadraticModel::initialize(
            vec![0.0, 0.0],
            0.3,
            5,
            &diag_quadratic(h, c),
        );
        let d = [0.1, 0.05];
        let knew = setdrop_tr(&model, true, &d, 0.3, 0.3);
        assert!(knew.is_some());
    }

    /// `geostep` returns a step of length ≈ delbar that improves geometry, with a
    /// non-zero update denominator at `knew`.
    #[test]
    fn geostep_unconstrained_has_radius_and_denominator() {
        let h = &[2.0, 4.0];
        let c = &[0.4, -0.3];
        let model = QuadraticModel::initialize(
            vec![0.0, 0.0],
            0.3,
            5,
            &diag_quadratic(h, c),
        );
        let n = 2;
        // knew != kopt.
        let knew = (0..model.m()).find(|&k| k != model.kopt()).unwrap();
        let warm = ActiveSetQr::<f64>::new(n, 0);
        let delbar = 0.2;
        let (s, feasible) = geostep(&model, knew, delbar, &[], &[], &warm);
        let sn = norm(&s);
        assert!(sn > 0.5 * delbar && sn < 2.0 * delbar, "‖s‖ = {sn}");
        assert!(feasible, "unconstrained step is always feasible");
        let den = denominators(&model, &s)[knew];
        assert!(den.abs() > 0.0);
    }
}
