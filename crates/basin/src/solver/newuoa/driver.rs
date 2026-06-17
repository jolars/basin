//! The NEWUOA driver loop (Powell 2006, §2 Figure 1, with the §7 details).
//!
//! This drives the [`QuadraticModel`] through trust-region iterations: at each
//! step TRSAPP (§5) proposes `d`, `F` is sampled at `x_opt + d`, the gain ratio
//! RATIO (eq. 2.2) decides whether to accept and how to revise `Δ` (eq. 7.1),
//! the MOVE rule (eqs. 7.2–7.5) picks the interpolation point to drop, and the
//! least-Frobenius update (§4) folds the new point into the model. The
//! two-radius `ρ/Δ` schedule (eq. 7.6) shrinks `ρ` from `ρ_beg` to `ρ_end`.
//!
//! # Geometry-improving branch
//!
//! When a trust-region step is unproductive (`RATIO < 0.1`, Box 6 → 7) or a
//! short step suggests the model is convex (`‖d‖ < ½ρ`, Box 3 → 14), Figure 1
//! may replace the interpolation point furthest from `x_opt` with a
//! geometry-improving step. This version implements that branch (Boxes 7–9):
//! Box 7/8 picks the furthest point `x_t` and tests `DIST = ‖x_t − x_opt‖ ≥ 2Δ`;
//! Box 9 calls [BIGLAG](super::biglag) to maximize `|ℓ_t(x_opt + d)|` over
//! `‖d‖ ≤ Δ̄` (eq. 6.16) — falling back to [BIGDEN](QuadraticModel::bigden) when
//! the BIGLAG step would leave the update denominator ill-conditioned
//! (`|σ| < 0.8 τ²`, eq. 6.17) — and folds the new point in with `MOVE = t`
//! fixed. The short-step Box-14 test (eq. 7.7) gates whether the work at the
//! current `ρ` is complete, using `CRVMIN` and the model errors `|F − Q|` of the
//! 3 most recent updates.
//!
//! Before each Box-5 update the §7 origin shift (eq. 7.10) re-centres `x0` on
//! `x_opt` when the step is small relative to the drift `‖x_opt − x0‖`, keeping
//! the `H` algebra accurate over long far-drifting runs (see [`crate::solver::powell::origin`]).
//!
//! # Qint robustness modification (§8)
//!
//! The least-Frobenius update makes the *minimal* change to `∇²Q` per new
//! interpolation condition, which on problems like VARDIM (eq. 8.1) leaves the
//! model Hessian far too large to shrink in time. Powell's §8 fix replaces `Q`
//! wholesale with the alternative interpolant `Q_int` (the quadratic minimizing
//! `‖∇²Q‖_F` subject to the current conditions; eq. 8.3) once it is persistently
//! the better model. Each Box-5 trust-region update that drops a point sets a
//! flag — here in PRIMA v0.7.2's tuned form of eq. 8.4 (`tryqalt`): `RATIO ≤ 0.1`
//! **and** `‖∇Q_int(x_opt)‖² ≤ 0.1·‖∇Q(x_opt)‖²` (gradients at `x_opt`; Powell's
//! paper states `0.01` / `100×` at `x0`, which PRIMA re-tuned). After three
//! consecutive YES flags `Q` is adopted from `Q_int` (see [`adopt_alt_model`]).
//! Geometry steps (Box 9, `RATIO := 1`) fail the flag, so they reset the counter.
//!
//! # Stepper structure
//!
//! The Figure-1 loop is factored into a resumable [`NewuoaWork`]: one
//! [`NewuoaWork::step`] call performs one trust-region iteration plus the outer
//! ρ-schedule decision. The standalone [`minimize`] entry owns a `NewuoaWork` on
//! the stack and loops `step` to completion; the public
//! [`Newuoa`](crate::solver::Newuoa) solver parks the same `NewuoaWork` on its
//! struct and calls `step` once per `next_iter`, so framework termination
//! (`MaxCostEvals`, `RhoTolerance`, …) composes around it. This mirrors how
//! Levenberg-Marquardt keeps its μ/ν working state on the solver struct.
//!
//! [`adopt_alt_model`]: QuadraticModel::adopt_alt_model

use super::trsapp::Trsapp;
use crate::core::math::Scalar;
use crate::solver::powell::{QuadraticModel, TrustRegionSubproblem};

/// Configuration for a standalone NEWUOA run via [`minimize`].
///
/// The public [`Newuoa`](crate::solver::Newuoa) solver does not use this — it
/// configures `ρ_beg` / `ρ_end` / `npt` on the solver and delegates the budget
/// to framework termination ([`MaxCostEvals`](crate::MaxCostEvals)).
pub(crate) struct NewuoaConfig<F = f64> {
    /// Initial trust-region radius `ρ_beg` (also the initial `Δ`).
    pub(crate) rho_beg: F,
    /// Final trust-region radius `ρ_end`; the run stops once `ρ` reaches it.
    pub(crate) rho_end: F,
    /// Soft cap on objective evaluations: checked once per outer iteration, so a
    /// single `step` may overshoot it by the ≤ 2 evaluations it samples (the
    /// budget is honored within that slack). The public [`Newuoa`] solver instead
    /// delegates to framework termination, which caps each evaluation exactly.
    ///
    /// [`Newuoa`]: crate::solver::Newuoa
    pub(crate) max_fun: usize,
    /// Number of interpolation points `npt = m`, in `[n+2, ½(n+1)(n+2)]`;
    /// `2n+1` is Powell's recommended default.
    pub(crate) npt: usize,
}

/// Why a NEWUOA run stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NewuoaStop {
    /// `ρ` reached `ρ_end` (the natural convergence stop).
    RhoReached,
    /// The evaluation budget `max_fun` was exhausted.
    MaxFun,
}

/// The result of a NEWUOA run.
pub(crate) struct NewuoaOutcome<F = f64> {
    /// Best point found, in absolute coordinates.
    pub(crate) x: Vec<F>,
    /// Objective value at [`x`](Self::x).
    pub(crate) f: F,
    /// Number of objective evaluations performed.
    pub(crate) nf: usize,
    /// Why the run stopped.
    pub(crate) stop: NewuoaStop,
}

/// What a [`NewuoaWork::step`] decided about the ρ/Δ schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Transition {
    /// More trust-region work remains at the current `ρ`.
    Continue,
    /// The work at this `ρ` finished and `ρ` was reduced (eq. 7.6).
    RhoReduced,
    /// `ρ` reached `ρ_end` and the work there is complete — converged.
    Converged,
}

/// The result of one [`NewuoaWork::step`]: the schedule transition and the
/// objective values sampled during the step.
///
/// `evaluated` holds 0, 1, or 2 `(x_absolute, F(x))` pairs in evaluation order
/// — a trust-region step (Box 4) can be followed by a geometry step (Box 9) in
/// the same iteration, so a step can sample `F` twice. The caller folds these
/// into its best-so-far tracking; in the framework path the [`Problem`] wrapper
/// also counts each call.
///
/// [`Problem`]: crate::core::problem::Problem
pub(crate) struct StepOutcome<F = f64> {
    pub(crate) transition: Transition,
    pub(crate) evaluated: Vec<(Vec<F>, F)>,
}

/// The resumable working state of a NEWUOA run: the [`QuadraticModel`] plus the
/// ρ/Δ schedule, the Box-14 model-error history, and the §8 Qint counter.
///
/// Shared by the standalone [`minimize`] driver and the public
/// [`Newuoa`](crate::solver::Newuoa) solver — see the module docs.
pub(crate) struct NewuoaWork<F = f64> {
    model: QuadraticModel<F>,
    /// Final radius `ρ_end` — drives the eq-7.6 schedule and the convergence
    /// stop. The model itself stays solver-side (not on the state).
    rho_end: F,
    /// Current trust-region radius `ρ`.
    rho: F,
    /// Current trust-region radius `Δ`.
    delta: F,
    /// Per-ρ ring of the most recent updates' `(‖d‖, |F − Q|)` for the Box-14
    /// model-accuracy test (eq. 7.7); reset on every ρ reduction (Box 12). Both
    /// trust-region and geometry updates push here (only the last 3 are read).
    history: Vec<(F, F)>,
    /// §8 Qint counter: consecutive Box-5 iterations whose flag (eq. 8.4) is
    /// YES. Persists across ρ reductions (as in NEWUOA's `ITEST`); reset by a
    /// Qint adoption and by geometry steps (their `RATIO := 1` fails the flag).
    itest: usize,
}

impl<F: Scalar> NewuoaWork<F> {
    /// Box 1: build the initial model (its `npt` evaluations of `F`) and seed
    /// the ρ/Δ schedule. Returns the work plus the initial best point/value
    /// (tracked independently of the model's `kopt`). `eval` may fail; the first
    /// `Err` aborts and bubbles to the caller.
    ///
    /// # Panics
    ///
    /// Panics unless `ρ_beg > ρ_end > 0` (and, via
    /// [`QuadraticModel::try_initialize`], `n ≥ 1`, `2n+1 ≤ npt ≤ ½(n+1)(n+2)`).
    pub(crate) fn try_init<E>(
        x0: Vec<F>,
        rho_beg: F,
        rho_end: F,
        npt: usize,
        eval: &mut impl FnMut(&[F]) -> Result<F, E>,
    ) -> Result<(Self, Vec<F>, F), E> {
        assert!(
            rho_beg > rho_end && rho_end > F::zero(),
            "NEWUOA needs rho_beg > rho_end > 0"
        );
        let model = QuadraticModel::try_initialize(x0, rho_beg, npt, eval)?;
        let best_x = model.best_point();
        let best_f = model.fopt();
        let work = Self {
            model,
            rho_end,
            rho: rho_beg,
            delta: rho_beg,
            history: Vec::new(),
            itest: 0,
        };
        Ok((work, best_x, best_f))
    }

    /// The current trust-region radius `ρ` (read by the public solver and by
    /// [`RhoTolerance`](crate::RhoTolerance) via the state).
    pub(crate) fn rho(&self) -> F {
        self.rho
    }

    /// One trust-region iteration of Figure 1 (Boxes 2–11/14–15) followed by the
    /// outer ρ-schedule decision (Box 12). `eval` may fail; the first `Err`
    /// aborts and bubbles.
    pub(crate) fn step<E>(
        &mut self,
        eval: &mut impl FnMut(&[F]) -> Result<F, E>,
    ) -> Result<StepOutcome<F>, E> {
        let n = self.model.n();
        let half = F::from_f64(0.5).expect("0.5 representable");
        let c01 = F::from_f64(0.1).expect("0.1 representable");
        let c07 = F::from_f64(0.7).expect("0.7 representable");
        let c15 = F::from_f64(1.5).expect("1.5 representable");
        let two = F::from_f64(2.0).expect("2.0 representable");
        let ten = F::from_f64(10.0).expect("10.0 representable");
        let eighth = F::from_f64(0.125).expect("0.125 representable");

        let mut evaluated: Vec<(Vec<F>, F)> = Vec::new();

        // Box 2: trust-region subproblem. NEWUOA's strategy is TRSAPP (§5); the
        // call goes through the shared `TrustRegionSubproblem` seam (BOBYQA swaps
        // in TRSBOX here).
        let trs = Trsapp.solve(&self.model, self.delta, &());
        let d = trs.d;
        let dnorm = norm(&d);
        let crvmin = trs.crvmin;
        let pred = trs.predicted_reduction;

        let kopt = self.model.kopt();
        let xopt_disp = self.model.xpt_row(kopt).to_vec();
        let f_opt = self.model.fopt();

        // Box 3: a short step (‖d‖ < ½ρ) does not warrant an F evaluation.
        if dnorm < half * self.rho {
            // Box 14: the work at this ρ is complete if at least 3 updates have
            // happened here and the 3 most recent each had ‖d‖ ≤ ρ and a model
            // error ≤ ⅛ρ²·CRVMIN. This is PRIMA v0.7.2's `accurate_mod`
            // (newuob.f90); Powell's paper eq. 7.7 additionally gates on the
            // predicted reduction, a term PRIMA drops — we track PRIMA here.
            let thr = eighth * self.rho * self.rho * crvmin;
            let box14_y = self.history.len() >= 3
                && self
                    .history
                    .iter()
                    .rev()
                    .take(3)
                    .all(|(dn, err)| *dn <= self.rho && *err <= thr);
            if box14_y {
                if self.rho <= self.rho_end {
                    // Box 13: evaluate the final short step; it is often a good
                    // move in variable space.
                    let (xabs, f_new) = eval_step(&self.model, &xopt_disp, &d, eval)?;
                    evaluated.push((xabs, f_new));
                    return Ok(StepOutcome {
                        transition: Transition::Converged,
                        evaluated,
                    });
                }
                return Ok(StepOutcome {
                    transition: self.finish_rho(half),
                    evaluated,
                });
            }
            // Box 15: a big reduction in Δ (floored at ρ), then the Box-7/8
            // geometry decision.
            self.delta = (half * self.delta).max(self.rho);
            let (t, dist) = far_point(&self.model);
            if dist >= two * self.delta {
                let ev = do_geometry(
                    &mut self.model,
                    t,
                    dist,
                    self.delta,
                    self.rho,
                    eval,
                    &mut self.history,
                )?;
                evaluated.push(ev);
                self.itest = 0; // geometry's RATIO := 1 fails the §8 flag (eq. 8.4).
                return Ok(StepOutcome {
                    transition: Transition::Continue,
                    evaluated,
                });
            }
            // Box 10: continue a TR iteration only if Δ has not reached ρ.
            if self.delta > self.rho {
                return Ok(StepOutcome {
                    transition: Transition::Continue,
                    evaluated,
                });
            }
            // Box 11.
            return Ok(StepOutcome {
                transition: self.finish_rho(half),
                evaluated,
            });
        }

        // Box 4: evaluate F at x⁺ = x_opt + d.
        let xnew_disp: Vec<F> = (0..n).map(|i| xopt_disp[i] + d[i]).collect();
        let xabs: Vec<F> = (0..n).map(|i| self.model.x0()[i] + xnew_disp[i]).collect();
        let f_new = eval(&xabs)?;
        evaluated.push((xabs, f_new));

        // RATIO = actual / predicted reduction (eq. 2.2), then revise Δ.
        let ratio = if pred > F::zero() {
            (f_opt - f_new) / pred
        } else {
            F::zero()
        };
        self.delta = revise_delta(self.delta, dnorm, ratio, self.rho, half, c01, c07, c15, two);

        // §7 origin shift (eq. 7.10): if x_opt has drifted far from x0, re-centre
        // before the update so the §4 algebra stays accurate. The shift moves x0
        // onto x_opt, so x_opt's displacement becomes 0 and x⁺ = x_opt + d has
        // displacement d (the absolute point is unchanged, so f_new / pred still
        // apply).
        let (xnew_disp, xopt_disp) = if maybe_shift_origin(&mut self.model, dnorm) {
            (d.clone(), vec![F::zero(); n])
        } else {
            (xnew_disp, xopt_disp)
        };

        // Record the model error |F(x⁺) − Q(x⁺)| = |f_new − f_opt + pred|
        // (eq. 7.7), then Box 5: σ-weighted MOVE update (eqs. 7.2–7.5).
        self.history.push((dnorm, (f_new - f_opt + pred).abs()));
        let knew = apply_move_update(
            &mut self.model,
            &xnew_disp,
            &xopt_disp,
            f_new,
            f_opt,
            self.delta,
            self.rho,
        );

        // §8 Qint, in PRIMA v0.7.2's tuned form of eq. 8.4 (`tryqalt`): run only
        // when the update dropped a point (`knew_tr > 0`); an unchanged set
        // leaves the model unchanged. Flag this iteration if the regular model is
        // poor (RATIO ≤ 0.1) and the alternative interpolant has a much smaller
        // gradient at x_opt (`‖∇Q‖² < 10·‖∇Q_int‖²` fails it); after three
        // consecutive flags, replace Q by Q_int. (Powell's paper uses 0.01 /
        // 100×; PRIMA re-tuned to 0.1 / 10× and compares at x_opt.)
        if knew {
            let galt = self.model.alt_gradient_at_opt();
            let gisq: F = galt.iter().map(|x| *x * *x).sum();
            let gopt = self.model.gradient_at_opt();
            let gqsq: F = gopt.iter().map(|x| *x * *x).sum();
            let flag = ratio <= c01 && gqsq >= ten * gisq;
            self.itest = if flag { self.itest + 1 } else { 0 };
            if self.itest >= 3 {
                self.model.adopt_alt_model();
                self.itest = 0;
            }
        }

        // Box 6: a good ratio earns another trust-region iteration.
        if ratio >= c01 {
            return Ok(StepOutcome {
                transition: Transition::Continue,
                evaluated,
            });
        }
        // Box 7/8: improve the model by replacing the furthest point when it is
        // unacceptably far (DIST ≥ 2Δ).
        let (t, dist) = far_point(&self.model);
        if dist >= two * self.delta {
            let ev = do_geometry(
                &mut self.model,
                t,
                dist,
                self.delta,
                self.rho,
                eval,
                &mut self.history,
            )?;
            evaluated.push(ev);
            return Ok(StepOutcome {
                transition: Transition::Continue,
                evaluated,
            });
        }
        // Box 10: keep working at this ρ unless the step and Δ are at the ρ floor
        // and the step did not reduce F.
        if dnorm > self.rho || self.delta > self.rho || ratio > F::zero() {
            return Ok(StepOutcome {
                transition: Transition::Continue,
                evaluated,
            });
        }
        // Box 11: work at this ρ is complete.
        Ok(StepOutcome {
            transition: self.finish_rho(half),
            evaluated,
        })
    }

    /// Box 12: the work at this ρ finished. If ρ is already at ρ_end the run has
    /// converged; otherwise reduce ρ (eq. 7.6), reset Δ (preserving Δ ≥ ½ρ_old),
    /// and clear the per-ρ history.
    fn finish_rho(&mut self, half: F) -> Transition {
        if self.rho <= self.rho_end {
            return Transition::Converged;
        }
        let rho_new = shrink_rho(self.rho, self.rho_end);
        self.delta = (half * self.rho).max(rho_new);
        self.rho = rho_new;
        self.history.clear();
        Transition::RhoReduced
    }
}

/// Minimize `f` by NEWUOA, starting from `x0` (Powell 2006).
///
/// `f` is the objective; it is evaluated `npt` times during initialization and
/// once per accepted trust-region step thereafter, up to `cfg.max_fun`. This is
/// the standalone entry (used by the tests and the PRIMA parity fixtures); the
/// public solver is [`Newuoa`](crate::solver::Newuoa).
///
/// # Panics
///
/// Panics (via [`NewuoaWork::try_init`]) unless `n ≥ 1`,
/// `2n+1 ≤ npt ≤ ½(n+1)(n+2)`, and `rho_beg > rho_end > 0`.
pub(crate) fn minimize<F: Scalar>(
    x0: Vec<F>,
    cfg: &NewuoaConfig<F>,
    f: impl Fn(&[F]) -> F,
) -> NewuoaOutcome<F> {
    let mut eval = |x: &[F]| Ok::<F, core::convert::Infallible>(f(x));
    let (mut work, mut best_x, mut best_f) =
        NewuoaWork::try_init(x0, cfg.rho_beg, cfg.rho_end, cfg.npt, &mut eval)
            .expect("infallible objective");
    let mut nf = cfg.npt;

    loop {
        if nf >= cfg.max_fun {
            return NewuoaOutcome {
                x: best_x,
                f: best_f,
                nf,
                stop: NewuoaStop::MaxFun,
            };
        }
        let out = work.step(&mut eval).expect("infallible objective");
        for (xabs, f_new) in out.evaluated {
            nf += 1;
            if f_new < best_f {
                best_f = f_new;
                best_x = xabs;
            }
        }
        match out.transition {
            Transition::Converged => {
                return NewuoaOutcome {
                    x: best_x,
                    f: best_f,
                    nf,
                    stop: NewuoaStop::RhoReached,
                };
            }
            Transition::Continue | Transition::RhoReduced => {}
        }
    }
}

/// Evaluate `F` at `x_opt + d`, returning the absolute point and its value.
fn eval_step<F: Scalar, E>(
    model: &QuadraticModel<F>,
    xopt_disp: &[F],
    d: &[F],
    eval: &mut impl FnMut(&[F]) -> Result<F, E>,
) -> Result<(Vec<F>, F), E> {
    let n = model.n();
    let xabs: Vec<F> = (0..n)
        .map(|i| model.x0()[i] + xopt_disp[i] + d[i])
        .collect();
    let val = eval(&xabs)?;
    Ok((xabs, val))
}

/// Box 7: the interpolation point furthest from `x_opt`, and that distance
/// `DIST = ‖x_t − x_opt‖`. `x_opt` itself is excluded.
fn far_point<F: Scalar>(model: &QuadraticModel<F>) -> (usize, F) {
    let kopt = model.kopt();
    let xopt = model.xpt_row(kopt);
    let mut best_t = kopt;
    let mut best_d2 = F::zero();
    for t in 0..model.m() {
        if t == kopt {
            continue;
        }
        let row = model.xpt_row(t);
        let d2: F = (0..model.n())
            .map(|i| (row[i] - xopt[i]) * (row[i] - xopt[i]))
            .sum();
        if d2 > best_d2 {
            best_d2 = d2;
            best_t = t;
        }
    }
    (best_t, best_d2.sqrt())
}

/// Box 9: the geometry-improving step. BIGLAG (§6) chooses `d` to maximize
/// `|ℓ_t(x_opt + d)|` over `‖d‖ ≤ Δ̄` (eq. 6.16); `F` is sampled at `x_opt + d`,
/// and the §4 update folds it in with `MOVE = t` fixed (the furthest point from
/// Box 7). Returns the evaluated absolute point and its value (the caller folds
/// it into best-so-far tracking) and pushes the Box-14 history entry.
///
/// If the BIGLAG step would leave the update denominator ill-conditioned —
/// `|σ| < 0.8 τ²` (eq. 6.17) — the step is replaced by [`bigden`], which
/// maximizes `|σ|` directly (eq. 6.18). This is rare in practice (Powell 2006,
/// §6), so the quartic BIGDEN search runs almost never.
///
/// [`bigden`]: QuadraticModel::bigden
#[allow(clippy::too_many_arguments)]
fn do_geometry<F: Scalar, E>(
    model: &mut QuadraticModel<F>,
    t: usize,
    dist: F,
    delta: F,
    rho: F,
    eval: &mut impl FnMut(&[F]) -> Result<F, E>,
    history: &mut Vec<(F, F)>,
) -> Result<(Vec<F>, F), E> {
    let n = model.n();
    let half = F::from_f64(0.5).expect("0.5 representable");
    let c01 = F::from_f64(0.1).expect("0.1 representable");

    // Δ̄ = max[min{0.1·DIST, 0.5·Δ}, ρ] (eq. 6.16).
    let delta_bar = ((c01 * dist).min(half * delta)).max(rho);
    let mut d = model.biglag(t, delta_bar).d;

    // §7 origin shift (eq. 7.10) before the Box-5 update, as in the TR path.
    // `t` and `kopt` are preserved by the shift; x_opt's displacement becomes 0.
    // `d` is relative to x_opt, so the re-centring leaves it unchanged.
    maybe_shift_origin(model, norm(&d));

    let kopt = model.kopt();
    let xopt_disp = model.xpt_row(kopt).to_vec();
    let f_opt = model.fopt();

    // Box 5 update quantities for the BIGLAG step, and the well-conditioning test
    // (eq. 6.17). σ is invariant under the origin shift, so testing it here
    // (post-shift) is consistent with the committed update below.
    let mut xnew_disp: Vec<F> = (0..n).map(|i| xopt_disp[i] + d[i]).collect();
    let mut ctx = model.prepare_update(&xnew_disp);
    let mut sc = model.update_params(t, &ctx);
    let c08 = F::from_f64(0.8).expect("0.8 representable");
    if sc.sigma.abs() < c08 * sc.tau * sc.tau {
        // BIGDEN fallback (eq. 6.18): re-solve maximizing |σ| directly.
        d = model.bigden(t, &d);
        xnew_disp = (0..n).map(|i| xopt_disp[i] + d[i]).collect();
        ctx = model.prepare_update(&xnew_disp);
        sc = model.update_params(t, &ctx);
    }
    let dnorm = norm(&d);

    let xabs: Vec<F> = (0..n).map(|i| model.x0()[i] + xnew_disp[i]).collect();
    let f_new = eval(&xabs)?;

    // Model error |F(x⁺) − Q(x⁺)| on the OLD model, for the eq. 7.7 history. The
    // recorded step length is `min(Δ̄, ‖d‖)`, as in PRIMA.
    let q_new = model.eval_change(&xnew_disp);
    let q_opt = model.eval_change(&xopt_disp);
    let model_err = (f_new - f_opt - (q_new - q_opt)).abs();
    history.push((dnorm.min(delta_bar), model_err));

    // Box 5 update with the fixed index `t`.
    if sc.sigma != F::zero() {
        model.commit_update(t, &ctx, &sc, f_new);
    }
    Ok((xabs, f_new))
}

/// §7 origin shift (Powell 2006, eq. 7.10): re-centre `x0` on `x_opt` when the
/// step is small relative to the drift `‖x_opt − x0‖`, keeping the `H` algebra
/// accurate. Returns whether the model was shifted (in which case `x_opt` now
/// sits at `x0`, so its displacement is the zero vector).
fn maybe_shift_origin<F: Scalar>(model: &mut QuadraticModel<F>, dnorm: F) -> bool {
    let c1em3 = F::from_f64(1e-3).expect("1e-3 representable");
    let xopt = model.xpt_row(model.kopt());
    let xopt_sq: F = xopt.iter().map(|x| *x * *x).sum();
    if xopt_sq > F::zero() && dnorm * dnorm < c1em3 * xopt_sq {
        model.shift_origin();
        true
    } else {
        false
    }
}

/// Box 5: choose MOVE (the interpolation point to drop, eqs. 7.2–7.5) and apply
/// the least-Frobenius update, unless the MOVE = 0 rule preserves the points.
fn apply_move_update<F: Scalar>(
    model: &mut QuadraticModel<F>,
    xnew_disp: &[F],
    xopt_disp: &[F],
    f_new: F,
    f_opt: F,
    delta: F,
    rho: F,
) -> bool {
    let m = model.m();
    let kopt = model.kopt();
    let one = F::one();
    let c01 = F::from_f64(0.1).expect("0.1 representable");

    let ctx = model.prepare_update(xnew_disp);
    // x* is the x_opt that will be current after this step (eq. 7.5): the new
    // point if it improves, else the incumbent.
    let xstar: &[F] = if f_new < f_opt { xnew_disp } else { xopt_disp };
    let denom = (c01 * delta).max(rho);

    // t* = argmax_t w_t |σ_t| over T (eq. 7.4). T excludes kopt when the step
    // does not strictly improve, to keep the best point (eq. 7.2 preamble).
    let mut chosen: Option<(usize, crate::solver::powell::update::UpdateScalars<F>, F)> = None;
    for t in 0..m {
        if f_new >= f_opt && t == kopt {
            continue;
        }
        let sc = model.update_params(t, &ctx);
        let dist = norm_diff(model.xpt_row(t), xstar);
        let r = dist / denom;
        // w_t = max{1, (‖x_t − x*‖ / max[0.1Δ, ρ])^6} (eq. 7.5).
        let w = (r * r * r * r * r * r).max(one);
        let w_sigma = w * sc.sigma.abs();
        if chosen.as_ref().is_none_or(|c| w_sigma > c.2) {
            chosen = Some((t, sc, w_sigma));
        }
    }

    if let Some((t_star, scalars, w_sigma)) = chosen {
        // MOVE = 0 (preserve points) only if the step did not improve and the
        // best weighted denominator is still small (eq. 7.5 tail).
        let move_zero = f_new >= f_opt && w_sigma < one;
        if !move_zero && scalars.sigma != F::zero() {
            model.commit_update(t_star, &ctx, &scalars, f_new);
            return true; // a point was dropped (`knew_tr > 0`).
        }
    }
    false
}

/// Δ revision (Powell 2006, eq. 7.1): the RATIO-dependent intermediate value,
/// then snapped to `ρ` when it would fall within `1.5ρ`.
#[allow(clippy::too_many_arguments)]
fn revise_delta<F: Scalar>(
    delta_old: F,
    dnorm: F,
    ratio: F,
    rho: F,
    half: F,
    c01: F,
    c07: F,
    c15: F,
    two: F,
) -> F {
    let delta_int = if ratio <= c01 {
        half * dnorm
    } else if ratio <= c07 {
        dnorm.max(half * delta_old)
    } else {
        (two * dnorm).max(half * delta_old)
    };
    if delta_int <= c15 * rho {
        rho
    } else {
        delta_int
    }
}

/// ρ reduction (Powell 2006, eq. 7.6).
fn shrink_rho<F: Scalar>(rho: F, rho_end: F) -> F {
    let c16 = F::from_f64(16.0).expect("16 representable");
    let c250 = F::from_f64(250.0).expect("250 representable");
    let c01 = F::from_f64(0.1).expect("0.1 representable");
    if rho <= c16 * rho_end {
        rho_end
    } else if rho <= c250 * rho_end {
        (rho * rho_end).sqrt()
    } else {
        c01 * rho
    }
}

/// Euclidean norm of `v`.
fn norm<F: Scalar>(v: &[F]) -> F {
    v.iter().map(|x| *x * *x).sum::<F>().sqrt()
}

/// Euclidean distance `‖a − b‖`.
fn norm_diff<F: Scalar>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x - *y) * (*x - *y))
        .sum::<F>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(n: usize, rho_beg: f64, rho_end: f64) -> NewuoaConfig<f64> {
        NewuoaConfig {
            rho_beg,
            rho_end,
            max_fun: 500,
            npt: 2 * n + 1,
        }
    }

    fn norm2(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// The initial model of a quadratic is exact (§3 init recovers it), so one
    /// trust-region step lands near the minimizer and the run converges to it.
    #[test]
    fn convex_quadratic_2d() {
        // Q(x) = (x0 − 1)² + 2(x1 + 2)²  →  min at (1, −2), value 0.
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + 2.0 * (x[1] + 2.0).powi(2);
        let out = minimize(vec![0.0, 0.0], &cfg(2, 0.5, 1e-8), f);
        assert!(norm2(&out.x, &[1.0, -2.0]) < 1e-6, "x = {:?}", out.x);
        assert!(out.f < 1e-10, "f = {}", out.f);
        assert_eq!(out.stop, NewuoaStop::RhoReached);
    }

    /// Ill-conditioned but still convex quadratic in 4-D.
    #[test]
    fn convex_quadratic_4d() {
        let diag = [1.0, 10.0, 100.0, 0.5];
        let xstar = [3.0, -1.0, 2.0, -4.0];
        let f = move |x: &[f64]| {
            (0..4)
                .map(|i| diag[i] * (x[i] - xstar[i]).powi(2))
                .sum::<f64>()
        };
        let out = minimize(vec![0.0; 4], &cfg(4, 1.0, 1e-8), f);
        assert!(norm2(&out.x, &xstar) < 1e-5, "x = {:?}", out.x);
        assert!(out.f < 1e-8, "f = {}", out.f);
    }

    /// The classic non-quadratic test: Rosenbrock, min at (1, 1), value 0.
    ///
    /// With the BIGLAG geometry step (Boxes 7–9) the interpolation set stays
    /// well-poised through Rosenbrock's curved valley, so the run converges to
    /// the minimizer (~8 digits) rather than stalling at ~3 digits as it did
    /// before geometry landed.
    #[test]
    fn rosenbrock_2d() {
        let f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
        let out = minimize(vec![-1.2, 1.0], &cfg(2, 0.5, 1e-8), f);
        assert!(out.f < 1e-7, "f = {}, x = {:?}", out.f, out.x);
    }

    /// The extended Rosenbrock chain in 6-D, started far from the minimizer.
    ///
    /// The iterate drifts several units from the base point `x0` along the
    /// valley, and the geometry steps do many extra `H`-updates. Without the §7
    /// origin shifts (eq. 7.10) the `H` matrix accumulates rounding error there
    /// and the run stalls at ~3 digits; with them re-centring `x0` on `x_opt`,
    /// the run reaches the minimizer to ~6 digits.
    #[test]
    fn chained_rosenbrock_6d() {
        let n = 6;
        let f = |x: &[f64]| {
            (0..x.len() - 1)
                .map(|i| (1.0 - x[i]).powi(2) + 100.0 * (x[i + 1] - x[i] * x[i]).powi(2))
                .sum::<f64>()
        };
        let out = minimize(vec![-1.0; n], &cfg(n, 0.5, 1e-7), f);
        assert!(out.f < 1e-6, "f = {}, x = {:?}", out.f, out.x);
    }

    /// VARDIM (Powell 2006, eq. 8.1), the problem that motivated the §8 Qint
    /// modification.
    ///
    /// `F(x) = Σ(xₗ−1)² + (Σ ℓ(xₗ−1))² + (Σ ℓ(xₗ−1))⁴`, min at `x = 1`, value 0.
    /// The quartic term makes the initial diagonal model Hessian far too large;
    /// the least-Frobenius update cannot shrink it fast enough and the run stalls
    /// without the Qint robustness step (eqs. 8.3–8.4), which periodically
    /// replaces `Q` by the least-Frobenius-Hessian interpolant. The test asserts
    /// both that the run converges and that the Qint adoption path actually
    /// fired (a positive `QINT_ADOPTIONS` delta).
    #[test]
    fn vardim_8d() {
        use crate::solver::powell::update::QINT_ADOPTIONS;
        use std::sync::atomic::Ordering;

        let n = 8;
        let f = |x: &[f64]| {
            let s: f64 = (0..x.len()).map(|l| (l as f64 + 1.0) * (x[l] - 1.0)).sum();
            let sq: f64 = (0..x.len()).map(|l| (x[l] - 1.0).powi(2)).sum();
            sq + s * s + s.powi(4)
        };
        let x_start: Vec<f64> = (0..n).map(|l| 1.0 - (l as f64 + 1.0) / n as f64).collect();

        let adopt0 = QINT_ADOPTIONS.load(Ordering::Relaxed);
        let out = minimize(
            x_start,
            &NewuoaConfig {
                rho_beg: 0.5,
                rho_end: 1e-8,
                max_fun: 2000,
                npt: 2 * n + 1,
            },
            f,
        );
        assert!(out.f < 1e-6, "f = {}, x = {:?}", out.f, out.x);
        assert!(norm2(&out.x, &vec![1.0; n]) < 1e-3, "x = {:?}", out.x);
        assert!(
            QINT_ADOPTIONS.load(Ordering::Relaxed) > adopt0,
            "expected the §8 Qint adoption path to fire"
        );
    }

    /// The budget stop fires when `max_fun` is too small to converge.
    #[test]
    fn respects_max_fun() {
        let f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
        let out = minimize(
            vec![-1.2, 1.0],
            &NewuoaConfig {
                rho_beg: 0.5,
                rho_end: 1e-10,
                max_fun: 15,
                npt: 5,
            },
            f,
        );
        assert_eq!(out.stop, NewuoaStop::MaxFun);
        assert!(out.nf <= 16); // 15 budget, checked at loop top
    }
}
