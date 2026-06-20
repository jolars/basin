//! LINCOA driver loop (Powell 2015; PRIMA `lincob.f90`) — resumable working
//! state.
//!
//! `LincoaWork` carries the shared [`QuadraticModel`], the folded constraint
//! system (`amat`/`bvec`/`rescon`), the warm-started active-set QR
//! ([`ActiveSetQr`]), and the ρ/Δ schedule across iterations. The public
//! [`Lincoa`](crate::solver::Lincoa) solver builds it in `init` and drives one
//! [`step`](LincoaWork::step) per `next_iter`; each step is one trust-region
//! iteration that may also take a geometry step or reduce ρ.
//!
//! Reuses basin's shared model machinery (`prepare_update` / `update_params` /
//! `commit_update`, `shift_origin`, and the `alt_model_change` /
//! `adopt_alt_model` alternative-model pair for LINCOA's `tryqalt`) and the
//! LINCOA-specific [`trstep`](super::trstep::trstep) (with its warm-started
//! `ActiveSetQr`), [`geostep`](super::geometry::geostep),
//! [`setdrop_tr`](super::geometry::setdrop_tr), and
//! [`update_rescon`](super::geometry::update_rescon).

use crate::core::math::Scalar;
use crate::solver::powell::QuadraticModel;

use super::geometry::{geostep, setdrop_tr, update_rescon};
use super::getact::ActiveSetQr;
use super::init::LinearSetup;
use super::trstep::trstep;

/// What one [`step`](LincoaWork::step) decided about the outer loop.
pub(crate) enum Transition {
    /// Keep iterating.
    Continue,
    /// ρ was reduced this step; keep iterating at the finer radius.
    RhoReduced,
    /// ρ reached ρ_end — LINCOA's natural convergence.
    Converged,
}

/// The result of one [`step`](LincoaWork::step): the loop decision. Evaluated
/// points are not returned — the public solver reports the model's best feasible
/// point ([`best`](LincoaWork::best)) and the `Problem` wrapper counts evals.
pub(crate) struct StepOutcome {
    pub(crate) transition: Transition,
}

/// The resumable working state of a LINCOA run.
pub(crate) struct LincoaWork<F = f64> {
    pub(crate) model: QuadraticModel<F>,
    /// Unit-normalized constraint normals, `n × m` column-major.
    amat: Vec<F>,
    /// Right-hand sides in `x0`-relative coordinates (PRIMA `b`).
    bvec: Vec<F>,
    /// Constraint residuals at `x_opt` with the far-constraint negative encoding
    /// (PRIMA `rescon`).
    rescon: Vec<F>,
    /// Warm-started active-set QR (PRIMA `iact`/`nact`/`qfac`/`rfac`).
    qr: ActiveSetQr<F>,
    /// Final radius `ρ_end` — drives the schedule and the convergence stop.
    rho_end: F,
    /// Current trust-region radius `ρ`.
    rho: F,
    /// Current trust-region radius `Δ`.
    delta: F,
    /// `‖d‖` of the latest 4 trust-region steps at the current ρ (PRIMA
    /// `dnorm_rec`; reset to a huge value to prefer geometry over ρ-reduction).
    dnorm_rec: [F; 4],
    /// Whether the alternative model beat the regular one on the latest 3
    /// evaluations (PRIMA `qalt_better`); all-true triggers `tryqalt`.
    qalt_better: [bool; 3],
    n: usize,
    m: usize,
}

/// Update Δ from the reduction ratio (PRIMA `trrad`; eta1=0.1, eta2=0.7,
/// gamma1=0.5, gamma2=2.0).
fn trrad<F: Scalar>(delta: F, dnorm: F, ratio: F) -> F {
    let g1 = F::from_f64(0.5).expect("0.5 representable");
    let g2 = F::from_f64(2.0).expect("2.0 representable");
    let eta1 = F::from_f64(0.1).expect("0.1 representable");
    let eta2 = F::from_f64(0.7).expect("0.7 representable");
    if ratio <= eta1 {
        (g1 * delta).min(dnorm)
    } else if ratio <= eta2 {
        (g1 * delta).max(dnorm)
    } else {
        (g1 * delta).max(g2 * dnorm)
    }
}

/// Reduce ρ toward ρ_end (PRIMA `redrho`).
fn redrho<F: Scalar>(rho: F, rho_end: F) -> F {
    let ratio = rho / rho_end;
    if ratio > F::from_f64(250.0).expect("250.0 representable") {
        F::from_f64(0.1).expect("0.1 representable") * rho
    } else if ratio <= F::from_f64(16.0).expect("16.0 representable") {
        rho_end
    } else {
        ratio.sqrt() * rho_end
    }
}

/// Reduction ratio with careful Inf/NaN handling (PRIMA `redrat`).
fn redrat<F: Scalar>(ared: F, pred: F) -> F {
    let eta1 = F::from_f64(0.1).expect("0.1 representable");
    let half = F::from_f64(0.5).expect("0.5 representable");
    if ared.is_nan() {
        F::neg_infinity()
    } else if pred.is_nan() || pred <= F::zero() {
        if ared > F::zero() {
            half * eta1
        } else {
            F::neg_infinity()
        }
    } else {
        ared / pred
    }
}

/// Euclidean norm.
fn norm<F: Scalar>(v: &[F]) -> F {
    v.iter().fold(F::zero(), |a, x| a + *x * *x).sqrt()
}

/// `a_jᵀ v` where `a_j` is column `j` of the `n × m` column-major `amat`.
fn col_dot<F: Scalar>(v: &[F], amat: &[F], j: usize, n: usize) -> F {
    (0..n).fold(F::zero(), |acc, r| acc + v[r] * amat[r + j * n])
}

impl<F: Scalar> LincoaWork<F> {
    /// Build the initial model under the folded linear-constraint system (Powell
    /// 2015) and seed the ρ/Δ schedule. `amat`/`bvec_abs` are the unit-normalized
    /// system from [`fold_constraints`](super::init::fold_constraints) in absolute
    /// coordinates. Returns the work plus the initial best feasible point/value.
    ///
    /// # Panics
    ///
    /// Panics unless `ρ_beg > ρ_end > 0` and, via
    /// [`try_initialize_linear`](QuadraticModel::try_initialize_linear), `n ≥ 1`
    /// and `2n+1 ≤ npt ≤ ½(n+1)(n+2)`.
    pub(crate) fn try_init<E>(
        x0: Vec<F>,
        amat: Vec<F>,
        bvec_abs: Vec<F>,
        rho_beg: F,
        rho_end: F,
        npt: usize,
        eval: &mut impl FnMut(&[F]) -> Result<F, E>,
    ) -> Result<(Self, Vec<F>, F), E> {
        assert!(
            rho_beg > rho_end && rho_end > F::zero(),
            "LINCOA needs rho_beg > rho_end > 0"
        );
        let n = x0.len();
        let m = bvec_abs.len();
        let LinearSetup { model, amat, bvec } =
            QuadraticModel::try_initialize_linear(x0, amat, bvec_abs, rho_beg, npt, eval)?;

        // Initial RESCON: true residual b − A x_opt, with the far-constraint
        // negative encoding (≥ rhobeg ⟹ store negated). PRIMA lincob:305-306.
        let zero = F::zero();
        let xopt = model.xpt_row(model.kopt()).to_vec();
        let mut rescon = vec![zero; m];
        for j in 0..m {
            let r = (bvec[j] - col_dot(&xopt, &amat, j, n)).max(zero);
            rescon[j] = if r >= rho_beg { -r } else { r };
        }

        let best_x = model.best_point();
        let best_f = model.fopt();
        let big = F::from_f64(1e30).expect("1e30 representable");
        let work = Self {
            model,
            amat,
            bvec,
            rescon,
            qr: ActiveSetQr::new(n, m),
            rho_end,
            rho: rho_beg,
            delta: rho_beg,
            dnorm_rec: [big; 4],
            qalt_better: [false; 3],
            n,
            m,
        };
        Ok((work, best_x, best_f))
    }

    /// The current trust-region radius `ρ`.
    pub(crate) fn rho(&self) -> F {
        self.rho
    }

    /// The current best feasible point (absolute) and its objective value.
    pub(crate) fn best(&self) -> (Vec<F>, F) {
        (self.model.best_point(), self.model.fopt())
    }

    /// Absolute point `x0 + disp` for an objective evaluation. LINCOA does not
    /// clip — trust-region steps stay feasible and geometry steps may legitimately
    /// leave the feasible region.
    fn abs_point(&self, disp: &[F]) -> Vec<F> {
        let x0 = self.model.x0();
        (0..self.n).map(|i| x0[i] + disp[i]).collect()
    }

    /// The interpolation point furthest from `x_opt`, and its squared distance.
    fn far_point(&self) -> (usize, F) {
        let xopt = self.model.xpt_row(self.model.kopt());
        let mut best_k = 0;
        let mut best = F::zero();
        for k in 0..self.model.m() {
            let row = self.model.xpt_row(k);
            let d2 = (0..self.n).fold(F::zero(), |a, i| {
                a + (row[i] - xopt[i]) * (row[i] - xopt[i])
            });
            if d2 > best {
                best = d2;
                best_k = k;
            }
        }
        (best_k, best)
    }

    /// Regular and alternative model prediction errors at `x_opt + d`
    /// (`moderr = f − f_opt − ΔQ`, `moderr_alt = f − f_opt − ΔQ_int`), and push the
    /// `|moderr_alt| < 0.1·|moderr|` flag into `qalt_better` (PRIMA).
    fn record_moderr(&mut self, f_new: F, f_opt: F, xopt: &[F], d: &[F]) {
        let n = self.n;
        let xnew: Vec<F> = (0..n).map(|i| xopt[i] + d[i]).collect();
        let q_change = self.model.eval_change(&xnew) - self.model.eval_change(xopt);
        let moderr = f_new - f_opt - q_change;
        let moderr_alt = f_new - f_opt - self.model.alt_model_change(d);
        let tenth = F::from_f64(0.1).expect("0.1 representable");
        let flag = moderr_alt.abs() < tenth * moderr.abs();
        self.qalt_better = [self.qalt_better[1], self.qalt_better[2], flag];
    }

    /// Adopt the alternative model if it has been better on the latest 3
    /// evaluations (PRIMA `tryqalt`).
    fn try_qalt(&mut self) {
        if self.qalt_better.iter().all(|&b| b) {
            self.model.adopt_alt_model();
            self.qalt_better = [false; 3];
        }
    }

    /// Shift the origin onto `x_opt` when it has drifted far (PRIMA's
    /// `sum(xopt²) ≥ 1e3·Δ²`), shifting `bvec` consistently. `rescon` is the true
    /// residual `b − A x_opt`, which is invariant under this shift, so it is left
    /// untouched (PRIMA lincob:628-635).
    fn maybe_shift_origin(&mut self) {
        let xopt = self.model.xpt_row(self.model.kopt()).to_vec();
        let xopt_sq = xopt.iter().fold(F::zero(), |a, v| a + *v * *v);
        let thr = F::from_f64(1e3).expect("1e3 representable") * self.delta * self.delta;
        if xopt_sq >= thr {
            for j in 0..self.m {
                self.bvec[j] = self.bvec[j] - col_dot(&xopt, &self.amat, j, self.n);
            }
            self.model.shift_origin();
        }
    }

    /// One LINCOA trust-region iteration (PRIMA `lincob` loop body).
    pub(crate) fn step<E>(
        &mut self,
        eval: &mut impl FnMut(&[F]) -> Result<F, E>,
    ) -> Result<StepOutcome, E> {
        let n = self.n;
        let zero = F::zero();
        let half = F::from_f64(0.5).expect("0.5 representable");
        let tenth = F::from_f64(0.1).expect("0.1 representable");
        let pt2 = F::from_f64(0.2).expect("0.2 representable");
        let pt1999 = F::from_f64(0.1999).expect("0.1999 representable");
        let gamma3 = F::from_f64(1.5).expect("1.5 representable");
        let eta1 = F::from_f64(0.1).expect("0.1 representable");
        let big = F::from_f64(1e30).expect("1e30 representable");

        // --- Trust-region subproblem (Powell 2015, §3) with the warm QR. ---
        let (trs, ngetact) = trstep(
            &self.model,
            self.delta,
            &self.amat,
            &self.rescon,
            &mut self.qr,
        );
        let d = trs.d;
        let qred = trs.predicted_reduction;
        let dnorm = self.delta.min(norm(&d));
        let shortd = (dnorm < half * self.delta && ngetact < 2) || dnorm < pt1999 * self.delta;

        // dnorm_rec ring of the last 4 trust-region DNORMs (PRIMA lincob:373-381).
        self.dnorm_rec = [
            self.dnorm_rec[1],
            self.dnorm_rec[2],
            self.dnorm_rec[3],
            dnorm,
        ];
        if self.delta > self.rho || !shortd {
            self.dnorm_rec = [big; 4];
        }

        let qred_thr = F::from_f64(1e-5).expect("1e-5 representable") * self.rho * self.rho;
        let trfail = qred <= qred_thr || qred.is_nan(); // tiny/negative qred, or NaN

        let mut ratio = -F::one();
        let mut knew_tr: Option<usize> = None;

        if shortd || trfail {
            self.delta = half * self.delta;
            if self.delta <= gamma3 * self.rho {
                self.delta = self.rho;
            }
        } else {
            let f_opt = self.model.fopt();
            let kopt = self.model.kopt();
            let xopt = self.model.xpt_row(kopt).to_vec();
            let xnew_disp: Vec<F> = (0..n).map(|i| xopt[i] + d[i]).collect();
            let xabs = self.abs_point(&xnew_disp);
            let f_new = eval(&xabs)?;

            self.record_moderr(f_new, f_opt, &xopt, &d);
            ratio = redrat(f_opt - f_new, qred);
            self.delta = trrad(self.delta, dnorm, ratio);
            if self.delta <= gamma3 * self.rho {
                self.delta = self.rho;
            }
            let ximproved = f_new < f_opt;

            knew_tr = setdrop_tr(&self.model, ximproved, &d, self.delta, self.rho);
            if let Some(knew) = knew_tr {
                let old_kopt = self.model.kopt();
                let ctx = self.model.prepare_update(&xnew_disp);
                let sc = self.model.update_params(knew, &ctx);
                if sc.sigma != zero {
                    self.model.commit_update(knew, &ctx, &sc, f_new);
                    // PRIMA `updatexf`: kopt = knew iff the point improved (and is
                    // feasible). The shared `commit_update` instead refreshes kopt
                    // to argmin fval, which would wrongly pick an infeasible
                    // low-`f` point; override it. (TR steps are feasible, so this
                    // matches argmin here, but the geometry path needs the guard.)
                    self.model.kopt = if ximproved { knew } else { old_kopt };
                    self.try_qalt();
                    let new_xopt = self.model.xpt_row(self.model.kopt()).to_vec();
                    update_rescon(
                        ximproved,
                        &self.amat,
                        &self.bvec,
                        self.delta,
                        norm(&d),
                        &new_xopt,
                        &mut self.rescon,
                        n,
                    );
                }
            }
        }

        // --- Decide: improve geometry, reduce ρ, or continue. ---
        let accurate_mod = self.dnorm_rec.iter().all(|&x| x <= half * self.rho)
            || self.dnorm_rec[2..].iter().all(|&x| x <= pt2 * self.rho);
        let (far_k, far_d2) = self.far_point();
        let close_itpset = {
            let four = F::from_f64(4.0).expect("4.0 representable");
            far_d2 <= four * self.delta * self.delta
        };
        let adequate_geo = (shortd && accurate_mod) || close_itpset;
        let small_trrad = self.delta.max(dnorm) <= self.rho;

        let bad_geo = shortd || trfail || ratio <= eta1 || knew_tr.is_none();
        let improve_geo = bad_geo && !adequate_geo;
        let bad_rho = shortd || trfail || ratio <= zero || knew_tr.is_none();
        let reduce_rho = bad_rho && adequate_geo && small_trrad;

        if improve_geo {
            let knew_geo = far_k;
            let delbar = (tenth * self.delta).max(self.rho);
            let (dgeo, feasible) = geostep(
                &self.model,
                knew_geo,
                delbar,
                &self.amat,
                &self.rescon,
                &self.qr,
            );
            let f_opt = self.model.fopt();
            let kopt = self.model.kopt();
            let xopt = self.model.xpt_row(kopt).to_vec();
            let xnew_disp: Vec<F> = (0..n).map(|i| xopt[i] + dgeo[i]).collect();
            let xabs = self.abs_point(&xnew_disp);
            let f_new = eval(&xabs)?;

            self.record_moderr(f_new, f_opt, &xopt, &dgeo);
            let ximproved = f_new < f_opt && feasible;

            let old_kopt = self.model.kopt();
            let ctx = self.model.prepare_update(&xnew_disp);
            let sc = self.model.update_params(knew_geo, &ctx);
            if sc.sigma != zero {
                self.model.commit_update(knew_geo, &ctx, &sc, f_new);
                // Feasibility-aware kopt (PRIMA `updatexf`): a geometry point may
                // be infeasible, so never let argmin-fval promote it to x_opt.
                self.model.kopt = if ximproved { knew_geo } else { old_kopt };
                self.try_qalt();
                let new_xopt = self.model.xpt_row(self.model.kopt()).to_vec();
                update_rescon(
                    ximproved,
                    &self.amat,
                    &self.bvec,
                    self.delta,
                    norm(&dgeo),
                    &new_xopt,
                    &mut self.rescon,
                    n,
                );
            }
        }

        let mut transition = Transition::Continue;
        if reduce_rho {
            if self.rho <= self.rho_end {
                // Final short trust step (PRIMA lincob:639-653): if the converging
                // iteration's trust step was short, evaluate F at x_opt + d before
                // stopping.
                if shortd {
                    let xopt = self.model.xpt_row(self.model.kopt()).to_vec();
                    let disp: Vec<F> = (0..n).map(|i| xopt[i] + d[i]).collect();
                    let xabs = self.abs_point(&disp);
                    // The eval bumps the count (PRIMA parity); x_opt stays the
                    // reported answer, so the value itself is not retained.
                    eval(&xabs)?;
                }
                return Ok(StepOutcome {
                    transition: Transition::Converged,
                });
            }
            let rho_new = redrho(self.rho, self.rho_end);
            self.delta = (half * self.rho).max(rho_new);
            self.rho = rho_new;
            self.dnorm_rec = [big; 4];
            transition = Transition::RhoReduced;
        }

        self.maybe_shift_origin();
        Ok(StepOutcome { transition })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::lincoa::init::fold_constraints;

    /// Drive a full LINCOA run on a closure objective, returning `(x, f, evals)`.
    #[allow(clippy::too_many_arguments)]
    fn run(
        x0: Vec<f64>,
        amat: Vec<f64>,
        bvec: Vec<f64>,
        rho_beg: f64,
        rho_end: f64,
        npt: usize,
        max_evals: usize,
        f: impl Fn(&[f64]) -> f64,
    ) -> (Vec<f64>, f64, usize) {
        let evals = core::cell::Cell::new(0usize);
        let mut eval = |x: &[f64]| -> Result<f64, core::convert::Infallible> {
            evals.set(evals.get() + 1);
            Ok(f(x))
        };
        let (mut work, _x0_best, _f0_best) =
            LincoaWork::try_init(x0, amat, bvec, rho_beg, rho_end, npt, &mut eval).unwrap();
        for _ in 0..max_evals {
            let out = work.step(&mut eval).unwrap();
            if matches!(out.transition, Transition::Converged) || evals.get() >= max_evals {
                break;
            }
        }
        // The authoritative best is the model's best feasible iterate (x_opt).
        let (mx, mf) = work.best();
        (mx, mf, evals.get())
    }

    /// Unconstrained 2D sphere: LINCOA reaches the minimizer.
    #[test]
    fn unconstrained_sphere_converges() {
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] + 2.0).powi(2);
        let (x, fx, _evals) = run(vec![0.0, 0.0], vec![], vec![], 0.5, 1e-6, 5, 200, f);
        assert!((x[0] - 1.0).abs() < 1e-4, "x0 = {}", x[0]);
        assert!((x[1] + 2.0).abs() < 1e-4, "x1 = {}", x[1]);
        assert!(fx < 1e-8, "f = {fx}");
    }

    /// `min ‖x − c‖²` with `c = (2, 2)` subject to `x0 + x1 ≤ 2`. The constrained
    /// optimum is the projection `(1, 1)` with f = 2.
    #[test]
    fn linearly_constrained_quadratic_hits_projection() {
        let c = [2.0, 2.0];
        let f = move |x: &[f64]| (x[0] - c[0]).powi(2) + (x[1] - c[1]).powi(2);
        let x0 = vec![0.0, 0.0];
        let (amat, bvec) =
            fold_constraints::<f64>(2, &x0, None, None, &[], &[(vec![1.0, 1.0], 2.0)]);
        let (x, fx, _evals) = run(x0, amat, bvec, 0.5, 1e-7, 5, 300, f);
        assert!((x[0] - 1.0).abs() < 1e-3, "x0 = {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-3, "x1 = {}", x[1]);
        assert!((fx - 2.0).abs() < 1e-3, "f = {fx}");
        // Feasibility: x0 + x1 <= 2.
        assert!(x[0] + x[1] <= 2.0 + 1e-6, "infeasible: {} + {}", x[0], x[1]);
    }

    /// Box bounds folded in: `min ‖x − c‖²`, `c = (5, 5)`, box `[-1, 1]²`.
    /// Optimum is the corner `(1, 1)`.
    #[test]
    fn box_constrained_quadratic_hits_corner() {
        let c = [5.0, 5.0];
        let f = move |x: &[f64]| (x[0] - c[0]).powi(2) + (x[1] - c[1]).powi(2);
        let x0 = vec![0.0, 0.0];
        let (amat, bvec) =
            fold_constraints::<f64>(2, &x0, Some(&[-1.0, -1.0]), Some(&[1.0, 1.0]), &[], &[]);
        let (x, _fx, _evals) = run(x0, amat, bvec, 0.3, 1e-7, 5, 300, f);
        assert!((x[0] - 1.0).abs() < 1e-3, "x0 = {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-3, "x1 = {}", x[1]);
    }
}
