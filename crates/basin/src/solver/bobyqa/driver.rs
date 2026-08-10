//! BOBYQA driver loop (Powell 2009, §6): resumable working state.
//!
//! `BobyqaWork` carries the shared [`QuadraticModel`], the shifted bounds
//! `sl`/`su`, and the ρ/Δ schedule across iterations. The public
//! [`Bobyqa`](crate::solver::Bobyqa) solver builds it in `init` and drives one
//! [`step`](BobyqaWork::step) per `next_iter`; each step is one trust-region
//! iteration that may also take a geometry step or reduce ρ.
//!
//! Ported from PRIMA v0.7.2 `fortran/bobyqa/bobyqb.f90`, reusing basin's shared
//! model machinery (`prepare_update`/`update_params`/`commit_update`,
//! `shift_origin`, the §8 `tryqalt` alternative model) and the BOBYQA-specific
//! [`trsbox`](super::trsbox::trsbox)/[`geostep`](super::geometry::geostep) /
//! [`setdrop_tr`](super::geometry::setdrop_tr), and [`rescue`](super::rescue) for
//! the §5 restoration procedure (invoked at the two denominator-failure sites,
//! after a trust-region step and before a geometry step).

use crate::core::math::Scalar;
use crate::solver::powell::{QuadraticModel, TrustRegionSubproblem};

use super::geometry::{geostep, setdrop_tr};
use super::rescue::rescue;
use super::trsbox::{ShiftedBox, Trsbox};

/// What one [`step`](BobyqaWork::step) decided about the outer loop.
pub(crate) enum Transition {
    /// Keep iterating.
    Continue,
    /// ρ was reduced this step; keep iterating at the finer radius.
    RhoReduced,
    /// ρ reached ρ_end: BOBYQA's natural convergence.
    Converged,
}

/// The result of one [`step`](BobyqaWork::step): the loop decision plus every
/// `(absolute point, F)` evaluated, for the public solver to fold into its best.
pub(crate) struct StepOutcome<F = f64> {
    pub(crate) transition: Transition,
    pub(crate) evaluated: Vec<(Vec<F>, F)>,
}

/// The resumable working state of a BOBYQA run.
pub(crate) struct BobyqaWork<F = f64> {
    pub(crate) model: QuadraticModel<F>,
    /// Shifted lower bounds `sl = a − x0` (`≤ 0`); the TRSBOX/ALTMOV region.
    pub(crate) sl: Vec<F>,
    /// Shifted upper bounds `su = b − x0` (`≥ 0`).
    pub(crate) su: Vec<F>,
    /// Absolute lower bounds `a` (PRIMA `xl`). Kept because the origin `x0`
    /// drifts, so `sl`/`su` alone can no longer recover the box: RESCUE's
    /// re-evaluation needs the absolute bounds for `xinbd`.
    pub(crate) lower: Vec<F>,
    /// Absolute upper bounds `b` (PRIMA `xu`).
    pub(crate) upper: Vec<F>,
    /// Final radius `ρ_end`: drives the schedule and the convergence stop.
    rho_end: F,
    /// Current trust-region radius `ρ`.
    rho: F,
    /// Current trust-region radius `Δ`.
    delta: F,
    /// `‖d‖` of the latest 3 evaluations at the current ρ (PRIMA `dnorm_rec`).
    dnorm_rec: [F; 3],
    /// Model prediction errors of the latest 3 evaluations (PRIMA `moderr_rec`).
    moderr_rec: [F; 3],
    /// §8 Qint counter (consecutive `tryqalt` flags); reuses the shared model.
    itest: usize,
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

/// Reduce ρ toward ρ_end (PRIMA `redrho`; eq. 7.6 analogue).
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

/// Reduction ratio with careful Inf/NaN handling (PRIMA `redrat`; `rshrink` is
/// `eta1`, used only when the predicted reduction is non-positive).
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

impl<F: Scalar> BobyqaWork<F> {
    /// Build the initial model under box bounds (Powell 2009, §2) and seed the
    /// ρ/Δ schedule. Returns the work plus the initial best feasible
    /// point and value. `eval` may fail; the first `Err` aborts and bubbles.
    ///
    /// `ρ_beg` is reduced for a narrow box (see below), so the effective
    /// `ρ_beg`/`ρ_end` may differ from the requested ones.
    ///
    /// # Panics
    ///
    /// Panics unless `ρ_beg > ρ_end > 0` (after the narrow-box revision) and,
    /// via [`QuadraticModel::try_initialize_bounded`], `n ≥ 1` and
    /// `2n+1 ≤ npt ≤ ½(n+1)(n+2)`.
    pub(crate) fn try_init<E>(
        x0: Vec<F>,
        lower: &[F],
        upper: &[F],
        mut rho_beg: F,
        mut rho_end: F,
        npt: usize,
        eval: &mut impl FnMut(&[F]) -> Result<F, E>,
    ) -> Result<(Self, Vec<F>, F), E> {
        // Narrow-box revision (PRIMA `preproc.f90`): BOBYQA needs room
        // `ρ_beg ≤ min(b_i − a_i)/2` to seed the cross. If the requested `ρ_beg`
        // is too large for the box, shrink it to `min(b_i − a_i)/4` (and pull
        // `ρ_end` down to keep `ρ_beg > ρ_end`) rather than rejecting the solve.
        // A degenerate coordinate (`b_i == a_i`) leaves no room and still trips
        // the assert below.
        let two = F::from_f64(2.0).expect("2.0 representable");
        let min_room = (0..x0.len())
            .map(|i| upper[i] - lower[i])
            .fold(F::infinity(), |acc, w| acc.min(w));
        if min_room.is_finite() && rho_beg > min_room / two {
            let four = F::from_f64(4.0).expect("4.0 representable");
            let tenth = F::from_f64(0.1).expect("0.1 representable");
            rho_beg = min_room / four;
            rho_end = rho_end.min(tenth * rho_beg);
        }
        assert!(
            rho_beg > rho_end && rho_end > F::zero(),
            "BOBYQA needs rho_beg > rho_end > 0"
        );
        let init = QuadraticModel::try_initialize_bounded(
            x0, lower, upper, rho_beg, npt, eval,
        )?;
        let best_x = init.model.best_point();
        let best_f = init.model.fopt();
        let big = F::from_f64(1e30).expect("1e30 representable");
        let work = Self {
            model: init.model,
            sl: init.sl,
            su: init.su,
            lower: lower.to_vec(),
            upper: upper.to_vec(),
            rho_end,
            rho: rho_beg,
            delta: rho_beg,
            dnorm_rec: [big; 3],
            moderr_rec: [big; 3],
            itest: 0,
        };
        Ok((work, best_x, best_f))
    }

    /// The current trust-region radius `ρ`.
    pub(crate) fn rho(&self) -> F {
        self.rho
    }

    /// Absolute point `x0 + clip(disp, sl, su)` for an objective evaluation.
    fn abs_point(&self, disp: &[F]) -> Vec<F> {
        let x0 = self.model.x0();
        (0..self.model.n())
            .map(|i| x0[i] + self.sl[i].max(self.su[i].min(disp[i])))
            .collect()
    }

    /// Push a value into a 3-entry "latest" ring (drop the oldest).
    fn push_ring(ring: &mut [F; 3], v: F) {
        ring[0] = ring[1];
        ring[1] = ring[2];
        ring[2] = v;
    }

    /// EBOUND for the accurate-model test (PRIMA `errbd`; eqs. 6.8–6.11). `d` is
    /// the short trust step relative to `x_opt`, `crvmin` from TRSBOX.
    fn errbd(&self, d: &[F], crvmin: F) -> F {
        let n = self.model.n();
        let half = F::from_f64(0.5).expect("0.5 representable");
        let eighth = F::from_f64(0.125).expect("0.125 representable");
        let rho = self.rho;
        let rho2 = rho * rho;
        let xopt = self.model.xpt_row(self.model.kopt()).to_vec();
        let gopt = self.model.gradient_at_opt();
        let hd = self.model.hessian_matvec(d);
        let max_moderr = self
            .moderr_rec
            .iter()
            .fold(F::zero(), |a, &v| a.max(v.abs()));
        // diag(∇²Q)_i = Γ_ii + Σ_k γ_k (x_k − x0)_i².
        let mut ebound = F::infinity();
        for i in 0..n {
            let xnew_i = xopt[i] + d[i];
            let gnew_i = gopt[i] + hd[i];
            let bfirst = if xnew_i <= self.sl[i] {
                gnew_i * rho
            } else if xnew_i >= self.su[i] {
                -gnew_i * rho
            } else {
                max_moderr
            };
            let mut diag_h = self.model.gamma_explicit.get(i, i);
            for k in 0..self.model.m() {
                let xki = self.model.xpt_row(k)[i];
                diag_h = diag_h + self.model.gamma[k] * xki * xki;
            }
            let bsecond = half * diag_h * rho2;
            ebound = ebound.min(bfirst.max(bfirst + bsecond));
        }
        if crvmin > F::zero() {
            ebound = ebound.min(eighth * crvmin * rho2);
        }
        ebound
    }

    /// The interpolation point furthest from `x_opt`, and its squared distance.
    fn far_point(&self) -> (usize, F) {
        let kopt = self.model.kopt();
        let xopt = self.model.xpt_row(kopt);
        let n = self.model.n();
        let mut best_k = 0;
        let mut best = F::zero();
        for k in 0..self.model.m() {
            let row = self.model.xpt_row(k);
            let d2 = (0..n).fold(F::zero(), |a, i| {
                a + (row[i] - xopt[i]) * (row[i] - xopt[i])
            });
            if d2 > best {
                best = d2;
                best_k = k;
            }
        }
        (best_k, best)
    }

    /// §8 Qint alternative-model test (PRIMA `tryqalt`): after three consecutive
    /// iterations where the regular model is poor but the least-Frobenius
    /// interpolant has a much smaller gradient at `x_opt`, adopt `Q_int`.
    fn try_qalt(&mut self, ratio: F) {
        let c01 = F::from_f64(0.1).expect("0.1 representable");
        let ten = F::from_f64(10.0).expect("10.0 representable");
        let galt = self.model.alt_gradient_at_opt();
        let gisq: F = galt.iter().fold(F::zero(), |a, v| a + *v * *v);
        let gopt = self.model.gradient_at_opt();
        let gqsq: F = gopt.iter().fold(F::zero(), |a, v| a + *v * *v);
        let flag = ratio <= c01 && gqsq >= ten * gisq;
        self.itest = if flag { self.itest + 1 } else { 0 };
        if self.itest >= 3 {
            self.model.adopt_alt_model();
            self.itest = 0;
        }
    }

    /// Shift the origin onto `x_opt` when it has drifted far (PRIMA's
    /// `sum(xopt²) ≥ 1e3·Δ²`), updating the shifted bounds accordingly.
    fn maybe_shift_origin(&mut self) {
        let n = self.model.n();
        let xopt = self.model.xpt_row(self.model.kopt()).to_vec();
        let xopt_sq = xopt.iter().fold(F::zero(), |a, v| a + *v * *v);
        let thr = F::from_f64(1e3).expect("1e3 representable")
            * self.delta
            * self.delta;
        if xopt_sq >= thr {
            for i in 0..n {
                self.sl[i] = (self.sl[i] - xopt[i]).min(F::zero());
                self.su[i] = (self.su[i] - xopt[i]).max(F::zero());
            }
            self.model.shift_origin();
        }
    }

    /// One BOBYQA trust-region iteration (PRIMA `bobyqb` loop body).
    pub(crate) fn step<E>(
        &mut self,
        eval: &mut impl FnMut(&[F]) -> Result<F, E>,
    ) -> Result<StepOutcome<F>, E> {
        let n = self.model.n();
        let zero = F::zero();
        let half = F::from_f64(0.5).expect("0.5 representable");
        let tenth = F::from_f64(0.1).expect("0.1 representable");
        let ten = F::from_f64(10.0).expect("10.0 representable");
        let gamma3 = F::from_f64(1.5).expect("1.5 representable");
        let eta1 = F::from_f64(0.1).expect("0.1 representable");
        let big = F::from_f64(1e30).expect("1e30 representable");
        let mut evaluated: Vec<(Vec<F>, F)> = Vec::new();

        // --- Trust-region subproblem (§3), through the shared seam. ---
        let region = ShiftedBox {
            sl: self.sl.clone(),
            su: self.su.clone(),
        };
        let trs = Trsbox.solve(&self.model, self.delta, &region);
        let mut d = trs.d;
        let crvmin = trs.crvmin;
        let qred = trs.predicted_reduction;
        let dnorm = self.delta.min(norm(&d));
        let shortd = dnorm < half * self.rho;
        // PRIMA `.not. (qred > 1e-5·ρ²)`: tiny or negative qred, or NaN.
        let qred_thr = F::from_f64(1e-5).expect("1e-5 representable")
            * self.rho
            * self.rho;
        let trfail = qred <= qred_thr || qred.is_nan();

        let mut ratio = -F::one();
        let mut knew_tr: Option<usize> = None;
        let mut ebound = zero;

        if shortd || trfail {
            self.delta = tenth * self.delta;
            if self.delta <= gamma3 * self.rho {
                self.delta = self.rho;
            }
            ebound = self.errbd(&d, crvmin);
        } else {
            let f_opt = self.model.fopt();
            let kopt = self.model.kopt();
            let xopt = self.model.xpt_row(kopt).to_vec();
            let (xabs, f_new, moderr) =
                self.commit_eval_only(&xopt, &d, eval)?;
            let mut ximproved = f_new < f_opt;
            evaluated.push((xabs, f_new));

            Self::push_ring(&mut self.dnorm_rec, dnorm);
            Self::push_ring(&mut self.moderr_rec, moderr);
            ratio = redrat(f_opt - f_new, qred);
            self.delta = trrad(self.delta, dnorm, ratio);
            if self.delta <= gamma3 * self.rho {
                self.delta = self.rho;
            }

            // Displacement of the trust-region candidate from x0 (= x_opt + d).
            let mut xnew_disp: Vec<F> =
                (0..n).map(|i| xopt[i] + d[i]).collect();

            // RESCUE if rounding has damaged the denominator for d (§5;
            // PRIMA bobyqb:348-375). The test uses the same `vlag = H·w + e_opt`
            // (= `ctx.hw`) and per-point denominators `σ` the update would use.
            let m = self.model.m();
            let need_rescue = ximproved && {
                let ctx = self.model.prepare_update(&xnew_disp);
                let sum_abs = ctx.hw.iter().fold(zero, |a, v| a + v.abs());
                let maxsq =
                    (0..m).fold(zero, |a, k| a.max(ctx.hw[k] * ctx.hw[k]));
                let any_good = (0..m)
                    .any(|k| self.model.update_params(k, &ctx).sigma > maxsq);
                !(sum_abs.is_finite() && any_good)
            };
            if need_rescue {
                rescue(
                    &mut self.model,
                    &mut self.sl,
                    &mut self.su,
                    &self.lower,
                    &self.upper,
                    self.delta,
                    eval,
                    &mut evaluated,
                )?;
                self.dnorm_rec = [big; 3];
                self.moderr_rec = [big; 3];
                // RESCUE shifted x0 to the pre-rescue best; re-express d relative
                // to the new x_opt and refresh ximproved (bobyqb:372-374). qred is
                // intentionally *not* recomputed (it is not a real TR step now).
                let xopt2 = self.model.xpt_row(self.model.kopt()).to_vec();
                let clamp_d: Vec<F> = (0..n)
                    .map(|i| self.sl[i].max(self.su[i].min(d[i])))
                    .collect();
                d = (0..n).map(|i| clamp_d[i] - xopt2[i]).collect();
                xnew_disp = clamp_d;
                ximproved = f_new < self.model.fopt();
            }

            // Choose the point to drop and commit the model update.
            knew_tr = setdrop_tr(&self.model, ximproved, &d, self.rho);
            if let Some(knew) = knew_tr {
                let ctx = self.model.prepare_update(&xnew_disp);
                let sc = self.model.update_params(knew, &ctx);
                if sc.sigma != zero {
                    self.model.commit_update(knew, &ctx, &sc, f_new);
                    self.try_qalt(ratio);
                }
            }
        }

        // --- Decide: improve geometry, reduce ρ, or continue. ---
        let accurate_mod = self.moderr_rec.iter().all(|e| e.abs() <= ebound)
            && self.dnorm_rec.iter().all(|&dn| dn <= self.rho);
        let (far_k, far_d2) = self.far_point();
        let close_itpset = far_d2
            <= (self.delta * self.delta)
                .max((ten * self.rho) * (ten * self.rho));
        let adequate_geo = (shortd && accurate_mod) || close_itpset;
        let small_trrad = self.delta.max(dnorm) <= self.rho;

        let bad_geo = shortd || trfail || ratio <= eta1 || knew_tr.is_none();
        let improve_geo = bad_geo && !adequate_geo;
        let bad_rho = shortd || trfail || ratio <= zero || knew_tr.is_none();
        let reduce_rho = bad_rho && adequate_geo && small_trrad;

        if improve_geo {
            let knew_geo = far_k;
            let delbar =
                ((tenth * far_d2.sqrt()).min(self.delta)).max(self.rho);
            let dgeo =
                geostep(&self.model, knew_geo, delbar, &self.sl, &self.su);
            let kopt = self.model.kopt();
            let xopt = self.model.xpt_row(kopt).to_vec();
            let xnew_disp: Vec<F> = (0..n).map(|i| xopt[i] + dgeo[i]).collect();

            // RESCUE if rounding has damaged the denominator for the geometry
            // step (§5; PRIMA bobyqb:479-541). The test is the stricter
            // `den[knew_geo] > ½·vlag[knew_geo]²`.
            let ctx = self.model.prepare_update(&xnew_disp);
            let sum_abs = ctx.hw.iter().fold(zero, |a, v| a + v.abs());
            let vlag_knew = ctx.hw[knew_geo];
            let den_knew = self.model.update_params(knew_geo, &ctx).sigma;
            let need_rescue = !(sum_abs.is_finite()
                && den_knew > half * vlag_knew * vlag_knew);

            if need_rescue {
                rescue(
                    &mut self.model,
                    &mut self.sl,
                    &mut self.su,
                    &self.lower,
                    &self.upper,
                    self.delta,
                    eval,
                    &mut evaluated,
                )?;
                self.dnorm_rec = [big; 3];
                self.moderr_rec = [big; 3];
                // The geometry step `dgeo` is discarded; the next iteration
                // recomputes a trust-region step on the rescued model.
            } else {
                let (xabs, f_new, moderr) =
                    self.commit_eval_only(&xopt, &dgeo, eval)?;
                Self::push_ring(
                    &mut self.dnorm_rec,
                    self.delta.min(norm(&dgeo)),
                );
                Self::push_ring(&mut self.moderr_rec, moderr);
                let sc = self.model.update_params(knew_geo, &ctx);
                if sc.sigma != zero {
                    self.model.commit_update(knew_geo, &ctx, &sc, f_new);
                }
                evaluated.push((xabs, f_new));
            }
        }

        let mut transition = Transition::Continue;
        if reduce_rho {
            if self.rho <= self.rho_end {
                self.maybe_shift_origin();
                // Final short trust step (PRIMA bobyqb:576-584): when the
                // converging iteration's trust step was short, evaluate F at
                // x_opt + d before stopping: it is "often a good move in
                // variable space". Mirrors the NEWUOA sibling's Box-13 step. `d`
                // is unchanged in the shortd branch, so it is the TRSBOX step;
                // it is gated on `shortd` exactly as PRIMA gates on `.and. shortd`.
                if shortd {
                    let xopt = self.model.xpt_row(self.model.kopt()).to_vec();
                    let disp: Vec<F> = (0..n).map(|i| xopt[i] + d[i]).collect();
                    let xabs = self.abs_point(&disp);
                    let f_new = eval(&xabs)?;
                    evaluated.push((xabs, f_new));
                }
                return Ok(StepOutcome {
                    transition: Transition::Converged,
                    evaluated,
                });
            }
            let rho_new = redrho(self.rho, self.rho_end);
            self.delta = (half * self.rho).max(rho_new);
            self.rho = rho_new;
            self.dnorm_rec =
                [F::from_f64(1e30).expect("1e30 representable"); 3];
            self.moderr_rec =
                [F::from_f64(1e30).expect("1e30 representable"); 3];
            transition = Transition::RhoReduced;
        }

        self.maybe_shift_origin();
        Ok(StepOutcome {
            transition,
            evaluated,
        })
    }

    /// Evaluate `F` at `x_opt + d` and the model error, **without** committing
    /// the model update (the caller commits separately, choosing `knew`).
    fn commit_eval_only<E>(
        &mut self,
        xopt: &[F],
        d: &[F],
        eval: &mut impl FnMut(&[F]) -> Result<F, E>,
    ) -> Result<(Vec<F>, F, F), E> {
        let n = self.model.n();
        let f_opt = self.model.fopt();
        let xnew_disp: Vec<F> = (0..n).map(|i| xopt[i] + d[i]).collect();
        let xabs = self.abs_point(&xnew_disp);
        let f_new = eval(&xabs)?;
        let q_change =
            self.model.eval_change(&xnew_disp) - self.model.eval_change(xopt);
        let moderr = f_new - f_opt - q_change;
        Ok((xabs, f_new, moderr))
    }
}

/// Euclidean norm.
fn norm<F: Scalar>(v: &[F]) -> F {
    v.iter().fold(F::zero(), |a, x| a + *x * *x).sqrt()
}
