use crate::core::math::{Dot, Scalar, ScaledAdd};
use crate::core::problem::{CostFunction, Gradient, Problem};
use crate::line_search::LineSearch;

/// Moré–Thuente line search — port of MINPACK-2's `dcsrch` + `dcstep`.
///
/// Finds an `α > 0` along the caller-supplied descent direction `d`
/// satisfying both the strong-Wolfe conditions:
///
/// * Armijo (sufficient decrease): `f(x + α d) ≤ f(x) + ftol · α · ∇f(x)ᵀd`
/// * Strong curvature: `|∇f(x + α d)ᵀd| ≤ gtol · |∇f(x)ᵀd|`
///
/// Same exit criteria as the strong-Wolfe `Wolfe` line search, but the
/// stepping strategy is different: rather than bisection, this uses
/// the safeguarded cubic/quadratic interpolation of Moré & Thuente
/// 1994 (ACM TOMS 20(3)). The algorithm maintains a bracketing
/// interval `[stx, sty]` containing a Wolfe-satisfying step, computes
/// a cubic-interpolation trial step from the function and derivative
/// values at the bracket endpoints, and safeguards against bad
/// extrapolations.
///
/// **Why this exists alongside `Wolfe`.** L-BFGS-B v3.0's iteration
/// trajectory is locked to this specific line search. Strong-Wolfe via
/// bisection (basin's `Wolfe`) selects different valid Wolfe steps, so
/// iterates diverge from the Fortran reference after the first line
/// search. `MoreThuente` is what unlocks bit-for-bit comparison with
/// `references/lbfgsb-v3.0/`.
///
/// **Conventions.** Parameter names (`ftol`, `gtol`, `xtol`) match the
/// Moré–Thuente paper and the Fortran source. The strong-Wolfe `c1`
/// and `c2` correspond to `ftol` and `gtol` respectively.
///
/// **Defaults match L-BFGS-B's `lnsrlb` caller** (`ftol = 1e-3`,
/// `gtol = 0.9`, `xtol = 0.1`, `stpmin = 0`, `stpmax = 1e10`) rather
/// than the looser MINPACK-2 standalone defaults. This is the
/// load-bearing choice for L-BFGS-B parity.
///
/// **Reference.** `references/lbfgsb-v3.0/lbfgsb.f` lines 3347–3948
/// (`dcsrch` and `dcstep`). MINPACK-1 1983, MINPACK-2 1993; J. J. Moré
/// and D. J. Thuente, *Line search algorithms with guaranteed
/// sufficient decrease*, ACM TOMS 20(3), 1994.
pub struct MoreThuente<F = f64> {
    /// Sufficient-decrease (Armijo) coefficient. Default `1e-3`
    /// (Fortran `lnsrlb` constant). Strong-Wolfe `c1` analog.
    pub ftol: F,
    /// Curvature coefficient. Default `0.9` (Fortran `lnsrlb`
    /// constant). Strong-Wolfe `c2` analog.
    pub gtol: F,
    /// Relative tolerance on the bracket width — exits with
    /// `xtol`-warning if the bracket has collapsed to relative size
    /// `≤ xtol`. Default `0.1` (Fortran `lnsrlb` constant).
    pub xtol: F,
    /// Initial trial step. Default `1.0` (the quasi-Newton unit step).
    pub alpha_init: F,
    /// Hard lower bound on the step. Default `0.0`.
    pub stpmin: F,
    /// Hard upper bound on the step. Default `1e10` (Fortran `big`).
    /// L-BFGS-B overrides this per-iteration via direct field
    /// mutation, with `stpmax = max α s.t. x + α·d ∈ [l, u]`.
    pub stpmax: F,
    /// Safety cap on function evaluations. Default `20`. The Moré–
    /// Thuente warning conditions normally terminate well before
    /// this; the cap exists to bound pathological inputs.
    pub maxfev: u32,
}

impl<F: Scalar> Default for MoreThuente<F> {
    fn default() -> Self {
        Self {
            ftol: F::from_f64(1.0e-3).unwrap(),
            gtol: F::from_f64(0.9).unwrap(),
            xtol: F::from_f64(0.1).unwrap(),
            alpha_init: F::one(),
            stpmin: F::zero(),
            stpmax: F::from_f64(1.0e10).unwrap(),
            maxfev: 20,
        }
    }
}

impl<F: Scalar> MoreThuente<F> {
    /// Moré–Thuente with L-BFGS-B's `lnsrlb` defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the Armijo coefficient. Panics if not in `(0, 1)`.
    pub fn ftol(mut self, ftol: F) -> Self {
        assert!(
            F::zero() < ftol && ftol < F::one(),
            "ftol must be in (0, 1)"
        );
        self.ftol = ftol;
        self
    }

    /// Override the curvature coefficient. Panics if not in `(0, 1)`.
    pub fn gtol(mut self, gtol: F) -> Self {
        assert!(
            F::zero() < gtol && gtol < F::one(),
            "gtol must be in (0, 1)"
        );
        self.gtol = gtol;
        self
    }

    /// Override the bracket-width relative tolerance. Panics if `< 0`.
    pub fn xtol(mut self, xtol: F) -> Self {
        assert!(xtol >= F::zero(), "xtol must be ≥ 0");
        self.xtol = xtol;
        self
    }

    /// Override the initial trial step. Panics if not strictly positive.
    pub fn alpha_init(mut self, alpha_init: F) -> Self {
        assert!(alpha_init > F::zero(), "alpha_init must be > 0");
        self.alpha_init = alpha_init;
        self
    }

    /// Override the step lower bound. Panics if `< 0`.
    pub fn stpmin(mut self, stpmin: F) -> Self {
        assert!(stpmin >= F::zero(), "stpmin must be ≥ 0");
        self.stpmin = stpmin;
        self
    }

    /// Override the step upper bound. Panics if `< stpmin`.
    pub fn stpmax(mut self, stpmax: F) -> Self {
        assert!(stpmax > F::zero(), "stpmax must be > 0");
        self.stpmax = stpmax;
        self
    }

    /// Override the function-evaluation cap.
    pub fn maxfev(mut self, maxfev: u32) -> Self {
        self.maxfev = maxfev;
        self
    }
}

impl<P, V, F> LineSearch<P, V, F> for MoreThuente<F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V>,
    V: ScaledAdd<F> + Dot<F> + Clone,
{
    type Error = P::Error;

    fn next(
        &mut self,
        problem: &mut Problem<P>,
        param: &V,
        cost: F,
        gradient: &V,
        direction: &V,
    ) -> Result<F, Self::Error> {
        let zero = F::zero();
        let p5 = F::from_f64(0.5).unwrap();
        let p66 = F::from_f64(0.66).unwrap();
        let xtrapl = F::from_f64(1.1).unwrap();
        let xtrapu = F::from_f64(4.0).unwrap();

        let finit = cost;
        let ginit = gradient.dot(direction);

        // Initial-derivative checks (Fortran START block, lines 3497–3512).
        // Defensive: catch ascent direction or non-finite slope. The
        // `!is_finite()` guard routes NaN here too.
        if !ginit.is_finite() || ginit >= zero {
            return Ok(zero);
        }
        if !(self.alpha_init >= self.stpmin && self.alpha_init <= self.stpmax) {
            return Ok(zero);
        }

        // Initialization (Fortran lines 3514–3541).
        let mut brackt = false;
        let mut stage: u8 = 1;
        let gtest = self.ftol * ginit;
        let mut width = self.stpmax - self.stpmin;
        let mut width1 = width / p5;

        let mut stx = zero;
        let mut fx = finit;
        let mut gx = ginit;
        let mut sty = zero;
        let mut fy = finit;
        let mut gy = ginit;
        let mut stmin = zero;
        let mut stmax = self.alpha_init + xtrapu * self.alpha_init;

        let mut stp = self.alpha_init;

        for _ in 0..self.maxfev {
            // Evaluate f(stp), g(stp) — Fortran `task = 'FG'` callback.
            let mut trial = param.clone();
            trial.scaled_add(stp, direction);
            let f = problem.cost(&trial)?;
            let g_full = problem.gradient(&trial)?;
            let g = g_full.dot(direction);

            // Stage transition (Fortran lines 3572–3574).
            let ftest = finit + stp * gtest;
            if stage == 1 && f <= ftest && g >= zero {
                stage = 2;
            }

            // Warning / convergence tests (Fortran lines 3578–3594).
            //
            // All four warning conditions and the convergence
            // condition terminate the search. Treated identically
            // here: return the current `stp` as the chosen step.
            // L-BFGS-B's `lnsrlb` routes `WARN`/`CONV` to its
            // `NEW_X` path, so a warning is not a failure.
            let warn_rounding = brackt && (stp <= stmin || stp >= stmax);
            let warn_xtol = brackt && stmax - stmin <= self.xtol * stmax;
            let warn_stpmax = stp == self.stpmax && f <= ftest && g <= gtest;
            let warn_stpmin = stp == self.stpmin && (f > ftest || g >= gtest);
            let converged = f <= ftest && g.abs() <= self.gtol * (-ginit);

            if warn_rounding || warn_xtol || warn_stpmax || warn_stpmin || converged {
                return Ok(stp);
            }

            // Step update via `dcstep`, with optional modified-function
            // path in stage 1 (Fortran lines 3600–3630).
            if stage == 1 && f <= fx && f > ftest {
                let fm = f - stp * gtest;
                let mut fxm = fx - stx * gtest;
                let mut fym = fy - sty * gtest;
                let gm = g - gtest;
                let mut gxm = gx - gtest;
                let mut gym = gy - gtest;

                dcstep::<F>(
                    &mut stx,
                    &mut fxm,
                    &mut gxm,
                    &mut sty,
                    &mut fym,
                    &mut gym,
                    &mut stp,
                    fm,
                    gm,
                    &mut brackt,
                    stmin,
                    stmax,
                );

                fx = fxm + stx * gtest;
                fy = fym + sty * gtest;
                gx = gxm + gtest;
                gy = gym + gtest;
            } else {
                dcstep::<F>(
                    &mut stx,
                    &mut fx,
                    &mut gx,
                    &mut sty,
                    &mut fy,
                    &mut gy,
                    &mut stp,
                    f,
                    g,
                    &mut brackt,
                    stmin,
                    stmax,
                );
            }

            // Bisection if bracket width isn't shrinking (Fortran 3634–3638).
            if brackt {
                if (sty - stx).abs() >= p66 * width1 {
                    stp = stx + p5 * (sty - stx);
                }
                width1 = width;
                width = (sty - stx).abs();
            }

            // Recompute (stmin, stmax) bounds for the next trial step
            // (Fortran 3642–3648).
            if brackt {
                stmin = stx.min(sty);
                stmax = stx.max(sty);
            } else {
                stmin = stp + xtrapl * (stp - stx);
                stmax = stp + xtrapu * (stp - stx);
            }

            // Clamp stp to user-supplied (stpmin, stpmax) (Fortran 3652–3653).
            stp = stp.max(self.stpmin).min(self.stpmax);

            // If further progress impossible, fall back to stx
            // (Fortran 3658–3659). The Fortran is
            // `(brackt ∧ (A ∨ B)) ∨ (brackt ∧ C)` ≡
            // `brackt ∧ (A ∨ B ∨ C)`.
            if brackt && (stp <= stmin || stp >= stmax || stmax - stmin <= self.xtol * stmax) {
                stp = stx;
            }
        }

        // maxfev exhausted: return current best step (Armijo holds
        // at stx by invariant of dcstep, so this is a usable step).
        Ok(stx)
    }
}

/// Safeguarded cubic/quadratic step interpolation.
///
/// Direct port of `dcstep.f` (Fortran lines 3694–3948). Updates
/// `(stx, fx, dx)` and `(sty, fy, dy)` — the bracketing interval
/// endpoints — and computes the next trial step `stp`. The
/// four-case structure of Moré & Thuente §3:
///
/// 1. `fp > fx`: minimum bracketed. Take cubic step or average of
///    cubic and quadratic, whichever is closer to stx.
/// 2. `sgn(dp) ≠ sgn(dx)`: minimum bracketed on the other side.
///    Take cubic or secant, whichever is farther from stp.
/// 3. `|dp| < |dx|`: same-sign derivative shrinking. Cubic with
///    safeguards.
/// 4. Otherwise: cubic if bracketed, else hit a step bound.
#[allow(clippy::too_many_arguments)]
fn dcstep<F: Scalar>(
    stx: &mut F,
    fx: &mut F,
    dx: &mut F,
    sty: &mut F,
    fy: &mut F,
    dy: &mut F,
    stp: &mut F,
    fp: F,
    dp: F,
    brackt: &mut bool,
    stpmin: F,
    stpmax: F,
) {
    let zero = F::zero();
    let two = F::from_f64(2.0).unwrap();
    let three = F::from_f64(3.0).unwrap();
    let p66 = F::from_f64(0.66).unwrap();

    let sgnd = dp * (*dx / dx.abs());
    let stpf;

    if fp > *fx {
        // Case 1.
        let theta = three * (*fx - fp) / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max(dx.abs()).max(dp.abs());
        let mut gamma = s * ((theta / s).powi(2) - (*dx / s) * (dp / s)).sqrt();
        if *stp < *stx {
            gamma = -gamma;
        }
        let p = (gamma - *dx) + theta;
        let q = ((gamma - *dx) + gamma) + dp;
        let r = p / q;
        let stpc = *stx + r * (*stp - *stx);
        let stpq = *stx + ((*dx / ((*fx - fp) / (*stp - *stx) + *dx)) / two) * (*stp - *stx);
        stpf = if (stpc - *stx).abs() < (stpq - *stx).abs() {
            stpc
        } else {
            stpc + (stpq - stpc) / two
        };
        *brackt = true;
    } else if sgnd < zero {
        // Case 2.
        let theta = three * (*fx - fp) / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max(dx.abs()).max(dp.abs());
        let mut gamma = s * ((theta / s).powi(2) - (*dx / s) * (dp / s)).sqrt();
        if *stp > *stx {
            gamma = -gamma;
        }
        let p = (gamma - dp) + theta;
        let q = ((gamma - dp) + gamma) + *dx;
        let r = p / q;
        let stpc = *stp + r * (*stx - *stp);
        let stpq = *stp + (dp / (dp - *dx)) * (*stx - *stp);
        stpf = if (stpc - *stp).abs() > (stpq - *stp).abs() {
            stpc
        } else {
            stpq
        };
        *brackt = true;
    } else if dp.abs() < dx.abs() {
        // Case 3.
        let theta = three * (*fx - fp) / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max(dx.abs()).max(dp.abs());
        // `gamma = 0` only arises if the cubic does not tend to infinity
        // in the direction of the step. The `max(0, ·)` guards a
        // negative argument to sqrt from rounding.
        let mut gamma = s * (zero.max((theta / s).powi(2) - (*dx / s) * (dp / s))).sqrt();
        if *stp > *stx {
            gamma = -gamma;
        }
        let p = (gamma - dp) + theta;
        let q = (gamma + (*dx - dp)) + gamma;
        let r = p / q;
        let stpc = if r < zero && gamma != zero {
            *stp + r * (*stx - *stp)
        } else if *stp > *stx {
            stpmax
        } else {
            stpmin
        };
        let stpq = *stp + (dp / (dp - *dx)) * (*stx - *stp);

        stpf = if *brackt {
            let cand = if (stpc - *stp).abs() < (stpq - *stp).abs() {
                stpc
            } else {
                stpq
            };
            if *stp > *stx {
                (*stp + p66 * (*sty - *stp)).min(cand)
            } else {
                (*stp + p66 * (*sty - *stp)).max(cand)
            }
        } else {
            let cand = if (stpc - *stp).abs() > (stpq - *stp).abs() {
                stpc
            } else {
                stpq
            };
            cand.min(stpmax).max(stpmin)
        };
    } else {
        // Case 4.
        stpf = if *brackt {
            let theta = three * (fp - *fy) / (*sty - *stp) + *dy + dp;
            let s = theta.abs().max(dy.abs()).max(dp.abs());
            let mut gamma = s * ((theta / s).powi(2) - (*dy / s) * (dp / s)).sqrt();
            if *stp > *sty {
                gamma = -gamma;
            }
            let p = (gamma - dp) + theta;
            let q = ((gamma - dp) + gamma) + *dy;
            let r = p / q;
            *stp + r * (*sty - *stp)
        } else if *stp > *stx {
            stpmax
        } else {
            stpmin
        };
    }

    // Update the bracket (Fortran 3928–3941).
    if fp > *fx {
        *sty = *stp;
        *fy = fp;
        *dy = dp;
    } else {
        if sgnd < zero {
            *sty = *stx;
            *fy = *fx;
            *dy = *dx;
        }
        *stx = *stp;
        *fx = fp;
        *dx = dp;
    }

    *stp = stpf;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1D quadratic via Vec<f64>: f(x) = (x[0] - 3)^2. Min at x = 3.
    struct Quadratic;

    impl CostFunction for Quadratic {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
            Ok((x[0] - 3.0).powi(2))
        }
    }

    impl Gradient for Quadratic {
        type Gradient = Vec<f64>;
        fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
            Ok(vec![2.0 * (x[0] - 3.0)])
        }
    }

    /// 1D cubic with non-trivial bracketing: f(x) = (x-2)^3 - 3(x-2),
    /// has a local min at x = 3 (f = -2) and a local max at x = 1
    /// (f = 2). Starting at x = 0 with d = +1, the initial slope is
    /// f'(0) = 3·(0-2)^2 - 3 = 9, NOT a descent direction. Going the
    /// other way: d = -1 at x = 5, f'(5) = 3·9 - 3 = 24, f'·d = -24 < 0,
    /// descends toward x = 3.
    struct Cubic;

    impl CostFunction for Cubic {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
            let t = x[0] - 2.0;
            Ok(t.powi(3) - 3.0 * t)
        }
    }

    impl Gradient for Cubic {
        type Gradient = Vec<f64>;
        fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
            let t = x[0] - 2.0;
            Ok(vec![3.0 * t.powi(2) - 3.0])
        }
    }

    #[test]
    fn satisfies_strong_wolfe_on_quadratic() {
        let mut p = Problem::new(Quadratic);
        let x = vec![0.0];
        let f0 = p.cost(&x).unwrap();
        let g = p.gradient(&x).unwrap();
        let d = vec![-g[0]]; // = +6
        let mut ls = MoreThuente::new();
        let alpha =
            LineSearch::<Quadratic, Vec<f64>>::next(&mut ls, &mut p, &x, f0, &g, &d).unwrap();

        assert!(alpha > 0.0);

        let mut x_new = x.clone();
        x_new[0] += alpha * d[0];
        let f_new = p.cost(&x_new).unwrap();
        let g_new = p.gradient(&x_new).unwrap();
        let g0_dot_d = g[0] * d[0];
        let gnew_dot_d = g_new[0] * d[0];

        assert!(
            f_new <= f0 + ls.ftol * alpha * g0_dot_d + 1e-12,
            "Armijo failed",
        );
        assert!(
            gnew_dot_d.abs() <= -ls.gtol * g0_dot_d + 1e-12,
            "Strong curvature failed",
        );
    }

    #[test]
    fn unit_step_accepted_when_quadratic_minimum_within_initial_step() {
        let mut p = Problem::new(Quadratic);
        let x = vec![0.0];
        let f0 = p.cost(&x).unwrap();
        let g = p.gradient(&x).unwrap();
        let d = vec![6.0];
        let mut ls = MoreThuente::new();
        let alpha =
            LineSearch::<Quadratic, Vec<f64>>::next(&mut ls, &mut p, &x, f0, &g, &d).unwrap();

        assert!(
            (alpha - 0.5).abs() < 0.5,
            "expected α near 0.5, got {alpha}",
        );
        let mut x_new = x.clone();
        x_new[0] += alpha * d[0];
        let f_new = p.cost(&x_new).unwrap();
        assert!(f_new <= f0 + ls.ftol * alpha * (g[0] * d[0]) + 1e-12);
    }

    #[test]
    fn ascent_direction_returns_zero_step() {
        let mut p = Problem::new(Quadratic);
        let x = vec![0.0];
        let f0 = p.cost(&x).unwrap();
        let g = p.gradient(&x).unwrap(); // g = -6 at x=0
        let baseline = *p.counts();
        let d = vec![g[0]]; // d = -6 → gᵀd = +36 > 0 (ascent)
        let mut ls = MoreThuente::new();
        let alpha =
            LineSearch::<Quadratic, Vec<f64>>::next(&mut ls, &mut p, &x, f0, &g, &d).unwrap();

        assert_eq!(alpha, 0.0);
        // Early-bail path makes no probes.
        assert_eq!(p.counts().cost_evals, baseline.cost_evals);
        assert_eq!(p.counts().gradient_evals, baseline.gradient_evals);
    }

    #[test]
    fn cubic_satisfies_wolfe_on_nontrivial_function() {
        let mut p = Problem::new(Cubic);
        let x = vec![5.0];
        let f0 = p.cost(&x).unwrap();
        let g = p.gradient(&x).unwrap();
        let d = vec![-1.0];
        let mut ls = MoreThuente::new().alpha_init(3.0);
        let alpha = LineSearch::<Cubic, Vec<f64>>::next(&mut ls, &mut p, &x, f0, &g, &d).unwrap();

        assert!(alpha > 0.0);
        let mut x_new = x.clone();
        x_new[0] += alpha * d[0];
        let f_new = p.cost(&x_new).unwrap();
        let g_new = p.gradient(&x_new).unwrap();
        let g0_dot_d = g[0] * d[0];
        let gnew_dot_d = g_new[0] * d[0];

        assert!(
            f_new <= f0 + ls.ftol * alpha * g0_dot_d + 1e-12,
            "Armijo failed at α={alpha}: f_new={f_new}, threshold={}",
            f0 + ls.ftol * alpha * g0_dot_d,
        );
        assert!(
            gnew_dot_d.abs() <= -ls.gtol * g0_dot_d + 1e-12,
            "Strong curvature failed at α={alpha}: |g·d|={}, threshold={}",
            gnew_dot_d.abs(),
            -ls.gtol * g0_dot_d,
        );
    }

    #[test]
    fn respects_stpmax_when_minimum_is_beyond() {
        let mut p = Problem::new(Quadratic);
        let x = vec![0.0];
        let f0 = p.cost(&x).unwrap();
        let g = p.gradient(&x).unwrap();
        let d = vec![6.0];
        let mut ls = MoreThuente::new().stpmax(0.1).alpha_init(0.1);
        let alpha =
            LineSearch::<Quadratic, Vec<f64>>::next(&mut ls, &mut p, &x, f0, &g, &d).unwrap();

        assert!((alpha - 0.1).abs() < 1e-12, "expected α=0.1, got {alpha}",);
    }
}
