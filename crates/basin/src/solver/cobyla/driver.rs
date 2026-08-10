// `!(a > b)` comparisons below are deliberate NaN-preserving ports of PRIMA's
// `~(a > b)` masks (`a <= b` would exclude NaN, changing tie-breaking); keep them.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

//! COBYLA driver (PRIMA `cobylb`): the trust-region / ρ schedule loop.
//!
//! [`CobylaWork`] owns the simplex, the penalty parameter `cpen`, the
//! trust-region radius `delta`, the resolution `rho`, and the return filter.
//! [`CobylaWork::step`] runs **one** iteration of PRIMA's `do tr` loop (0, 1, or
//! 2 function evaluations) and reports a [`Transition`]; the basin
//! [`Executor`](crate::Executor) drives it and owns budget / `ρ`-floor
//! termination.

use crate::core::math::Scalar;

use super::filter::{moderatef, savefilt, selectx};
use super::geometry::{assess_geo, geostep, setdrop_geo, setdrop_tr};
use super::init::initxfc;
use super::linalg::dot;
use super::model::{build_a, build_g};
use super::trstlp::{trrad, trstlp};
use super::update::{NO_DROP, updatepole, updatexfc};

/// A point evaluator: maps `x` to its (raw) objective value and constraint
/// vector, propagating the problem's error type `E`.
pub(crate) type EvalFn<'a, F, E> =
    dyn FnMut(&[F]) -> Result<(F, Vec<F>), E> + 'a;

/// Outcome of one [`CobylaWork::step`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Transition {
    /// Keep iterating.
    Continue,
    /// `ρ` was reduced this step (resolution refined).
    RhoReduced,
    /// `ρ` reached `ρ_end`: natural convergence.
    Converged,
    /// Damaging rounding broke the simplex bookkeeping; abort.
    Failed,
}

/// Resumable COBYLA working state.
pub(crate) struct CobylaWork<F = f64> {
    n: usize,
    m: usize,
    sim: Vec<F>,    // n × (n+1)
    simi: Vec<F>,   // n × n
    fval: Vec<F>,   // n+1
    conmat: Vec<F>, // m × (n+1)
    cval: Vec<F>,   // n+1
    cpen: F,
    delta: F,
    rho: F,
    rho_end: F,
    eta1: F,
    eta2: F,
    gamma1: F,
    gamma2: F,
    gamma3: F,
    factor_alpha: F,
    factor_beta: F,
    factor_gamma: F,
    ctol: F,
    cweight: F,
    // Return filter.
    maxfilt: usize,
    nfilt: usize,
    xfilt: Vec<F>,
    ffilt: Vec<F>,
    cfilt: Vec<F>,
    confilt: Vec<F>,
}

impl<F: Scalar> CobylaWork<F> {
    /// Build and seed the simplex from the start point `x0`, returning the work
    /// plus the initial reported `(x, f)`. `eval(x) -> (f, constr)` evaluates
    /// the (raw) objective and constraint vector.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_init<E>(
        x0: Vec<F>,
        m: usize,
        rho_beg: F,
        rho_end: F,
        eval: &mut EvalFn<F, E>,
    ) -> Result<(Self, Vec<F>, F), E> {
        let n = x0.len();
        let zero = F::zero();

        // Evaluate the start point first (moderated).
        let (f0_raw, c0_raw) = eval(&x0)?;
        let f0 = moderatef(f0_raw);
        let constr0: Vec<F> = c0_raw.iter().map(|&v| moderatef(v)).collect();

        let out = initxfc(n, m, &x0, f0, &constr0, rho_beg, eval)?;

        let eta1 = F::from_f64(0.1).unwrap();
        let eta2 = F::from_f64(0.7).unwrap();
        let gamma1 = F::from_f64(0.5).unwrap();
        let gamma2 = F::from_f64(2.0).unwrap();
        let gamma3 = F::one().max(
            (F::from_f64(0.75).unwrap() * gamma2)
                .min(F::from_f64(1.5).unwrap()),
        );
        let ctol = F::epsilon().sqrt();
        let cweight = F::from_f64(1.0e8).unwrap();
        let cpenmin = F::epsilon();

        let cpen = cpenmin.max(F::from_f64(1.0e3).unwrap().min(fcratio(
            &out.conmat,
            &out.fval,
            n,
            m,
        )));

        let maxfilt = 2000usize;
        let mut work = Self {
            n,
            m,
            sim: out.sim,
            simi: out.simi,
            fval: out.fval,
            conmat: out.conmat,
            cval: out.cval,
            cpen,
            delta: rho_beg,
            rho: rho_beg,
            rho_end,
            eta1,
            eta2,
            gamma1,
            gamma2,
            gamma3,
            factor_alpha: F::from_f64(0.25).unwrap(),
            factor_beta: F::from_f64(2.1).unwrap(),
            factor_gamma: F::from_f64(0.5).unwrap(),
            ctol,
            cweight,
            maxfilt,
            nfilt: 0,
            xfilt: vec![zero; n * maxfilt],
            ffilt: vec![zero; maxfilt],
            cfilt: vec![zero; maxfilt],
            confilt: vec![zero; m * maxfilt],
        };

        // Seed the filter from the initial simplex vertices.
        for j in 0..=n {
            let mut x = vec![zero; n];
            for r in 0..n {
                x[r] = if j < n {
                    work.sim[r + j * n] + work.sim[r + n * n]
                } else {
                    work.sim[r + n * n]
                };
            }
            let constr: Vec<F> =
                (0..m).map(|i| work.conmat[i + j * m]).collect();
            work.save_to_filter(&x, work.fval[j], work.cval[j], &constr);
        }

        let (bx, bf) = work.best();
        Ok((work, bx, bf))
    }

    /// Current trust-region resolution `ρ`.
    pub(crate) fn rho(&self) -> F {
        self.rho
    }

    fn pole(&self) -> Vec<F> {
        (0..self.n).map(|r| self.sim[r + self.n * self.n]).collect()
    }

    fn save_to_filter(&mut self, x: &[F], f: F, cstrv: F, constr: &[F]) {
        savefilt(
            x,
            f,
            cstrv,
            constr,
            self.n,
            self.m,
            self.ctol,
            self.cweight,
            self.maxfilt,
            &mut self.nfilt,
            &mut self.xfilt,
            &mut self.ffilt,
            &mut self.cfilt,
            &mut self.confilt,
        );
    }

    /// The filter-selected incumbent `(x, f)` COBYLA would return now.
    pub(crate) fn best(&self) -> (Vec<F>, F) {
        let kopt = selectx(
            &self.ffilt[..self.nfilt],
            &self.cfilt[..self.nfilt],
            self.cpen.max(self.cweight),
            self.ctol,
        );
        let x = (0..self.n).map(|r| self.xfilt[r + kopt * self.n]).collect();
        (x, self.ffilt[kopt])
    }

    /// Evaluate `(f, constr, cstrv)` at `x`, moderated.
    fn eval_moderated<E>(
        eval: &mut EvalFn<F, E>,
        x: &[F],
    ) -> Result<(F, Vec<F>, F), E> {
        let (f_raw, c_raw) = eval(x)?;
        let f = moderatef(f_raw);
        let constr: Vec<F> = c_raw.iter().map(|&v| moderatef(v)).collect();
        let cstrv = constr
            .iter()
            .cloned()
            .fold(F::zero(), F::max)
            .max(F::zero());
        Ok((f, constr, cstrv))
    }

    /// One iteration of PRIMA's `do tr` loop.
    pub(crate) fn step<E>(
        &mut self,
        eval: &mut EvalFn<F, E>,
    ) -> Result<Transition, E> {
        let n = self.n;
        let m = self.m;
        let zero = F::zero();
        let one = F::one();

        // (1) Increase CPEN if needed so PREREM > 0, then re-pole.
        self.cpen = self.get_cpen();
        if !updatepole(
            self.cpen,
            &mut self.conmat,
            &mut self.cval,
            &mut self.fval,
            &mut self.sim,
            &mut self.simi,
            n,
            m,
        ) {
            return Ok(Transition::Failed);
        }

        let adequate_geo = assess_geo(
            self.delta,
            self.factor_alpha,
            self.factor_beta,
            &self.sim,
            &self.simi,
            n,
        );

        // (2) Linear models and the trust-region step.
        let g = build_g(&self.fval, &self.simi, n);
        let a = build_a(&self.conmat, &self.simi, n, m);
        let b: Vec<F> = (0..m).map(|i| -self.conmat[i + n * m]).collect();
        let d = trstlp(&a, n, m, &b, self.delta, &g);
        let dnorm = self.delta.min(dot(&d, &d).sqrt());
        let shortd = dnorm < F::from_f64(0.1).unwrap() * self.rho;

        let preref = -dot(&d, &g);
        // prerec = cval[pole] − max(0, maxᵢ(conmat[i,pole] + (Aᵀd)[i])).
        let mut lin_cv = zero;
        for i in 0..m {
            let adi: F = (0..n).map(|l| d[l] * a[l + i * n]).sum();
            lin_cv = lin_cv.max(self.conmat[i + n * m] + adi);
        }
        lin_cv = lin_cv.max(zero);
        let prerec = self.cval[n] - lin_cv;
        let prerem = preref + self.cpen * prerec;
        let trfail = !(prerem
            > F::from_f64(1.0e-5).unwrap()
                * self.cpen.min(one)
                * self.rho
                * self.rho);

        let mut ratio = -one;
        let mut jdrop_tr = NO_DROP;

        if shortd || trfail {
            self.delta = F::from_f64(0.1).unwrap() * self.delta;
            if self.delta <= self.gamma3 * self.rho {
                self.delta = self.rho;
            }
        } else {
            let pole = self.pole();
            let x: Vec<F> = (0..n).map(|r| pole[r] + d[r]).collect();
            let (f, constr, cstrv) = Self::eval_moderated(eval, &x)?;
            self.save_to_filter(&x, f, cstrv, &constr);

            let actrem = (self.fval[n] + self.cpen * self.cval[n])
                - (f + self.cpen * cstrv);
            ratio = redrat(actrem, prerem, self.eta1);
            self.delta = trrad(
                self.delta,
                dnorm,
                self.eta1,
                self.eta2,
                self.gamma1,
                self.gamma2,
                ratio,
            );
            if self.delta <= self.gamma3 * self.rho {
                self.delta = self.rho;
            }
            let ximproved = actrem > zero;
            jdrop_tr = setdrop_tr(
                ximproved, &d, self.delta, self.rho, &self.sim, &self.simi, n,
            );
            if !updatexfc(
                jdrop_tr,
                &constr,
                self.cpen,
                cstrv,
                &d,
                f,
                &mut self.conmat,
                &mut self.cval,
                &mut self.fval,
                &mut self.sim,
                &mut self.simi,
                n,
                m,
            ) {
                return Ok(Transition::Failed);
            }
        }

        // (3) Decide geometry / ρ-reduction.
        let bad_trstep =
            shortd || trfail || ratio <= zero || jdrop_tr == NO_DROP;
        let improve_geo = bad_trstep && !adequate_geo;
        let reduce_rho =
            bad_trstep && adequate_geo && self.delta.max(dnorm) <= self.rho;

        if improve_geo
            && !assess_geo(
                self.delta,
                self.factor_alpha,
                self.factor_beta,
                &self.sim,
                &self.simi,
                n,
            )
        {
            let jdrop_geo = setdrop_geo(
                self.delta,
                self.factor_alpha,
                self.factor_beta,
                &self.sim,
                &self.simi,
                n,
            );
            if jdrop_geo == NO_DROP {
                return Ok(Transition::Failed);
            }
            let d = geostep(
                jdrop_geo,
                &self.conmat,
                self.cpen,
                self.delta,
                &self.fval,
                self.factor_gamma,
                &self.simi,
                n,
                m,
            );
            let pole = self.pole();
            let x: Vec<F> = (0..n).map(|r| pole[r] + d[r]).collect();
            let (f, constr, cstrv) = Self::eval_moderated(eval, &x)?;
            self.save_to_filter(&x, f, cstrv, &constr);
            if !updatexfc(
                jdrop_geo,
                &constr,
                self.cpen,
                cstrv,
                &d,
                f,
                &mut self.conmat,
                &mut self.cval,
                &mut self.fval,
                &mut self.sim,
                &mut self.simi,
                n,
                m,
            ) {
                return Ok(Transition::Failed);
            }
        }

        if reduce_rho {
            if self.rho <= self.rho_end {
                // Converged. Refine with the pending short step if it was never
                // evaluated (PRIMA's end-of-run trust-region tail).
                if shortd {
                    let pole = self.pole();
                    let x: Vec<F> = (0..n).map(|r| pole[r] + d[r]).collect();
                    let (f, constr, cstrv) = Self::eval_moderated(eval, &x)?;
                    self.save_to_filter(&x, f, cstrv, &constr);
                }
                return Ok(Transition::Converged);
            }
            self.delta = (F::from_f64(0.5).unwrap() * self.rho)
                .max(redrho(self.rho, self.rho_end));
            self.rho = redrho(self.rho, self.rho_end);
            self.cpen = F::epsilon().max(self.cpen.min(fcratio(
                &self.conmat,
                &self.fval,
                n,
                m,
            )));
            if !updatepole(
                self.cpen,
                &mut self.conmat,
                &mut self.cval,
                &mut self.fval,
                &mut self.sim,
                &mut self.simi,
                n,
                m,
            ) {
                return Ok(Transition::Failed);
            }
            return Ok(Transition::RhoReduced);
        }

        Ok(Transition::Continue)
    }

    /// Increase `cpen` so the predicted merit reduction is positive (PRIMA
    /// `getcpen`). Works on local copies of the simplex bookkeeping.
    fn get_cpen(&self) -> F {
        let n = self.n;
        let m = self.m;
        let zero = F::zero();
        let realmax = F::max_value();
        let two = F::from_f64(2.0).unwrap();

        let mut cpen = self.cpen;
        let mut conmat = self.conmat.clone();
        let mut cval = self.cval.clone();
        let mut fval = self.fval.clone();
        let mut sim = self.sim.clone();
        let mut simi = self.simi.clone();

        for _ in 0..(n + 1) {
            if !updatepole(
                cpen,
                &mut conmat,
                &mut cval,
                &mut fval,
                &mut sim,
                &mut simi,
                n,
                m,
            ) {
                break;
            }
            let g = build_g(&fval, &simi, n);
            let a = build_a(&conmat, &simi, n, m);
            let b: Vec<F> = (0..m).map(|i| -conmat[i + n * m]).collect();
            let d = trstlp(&a, n, m, &b, self.delta, &g);
            let preref = -dot(&d, &g);
            let mut lin_cv = zero;
            for i in 0..m {
                let adi: F = (0..n).map(|l| d[l] * a[l + i * n]).sum();
                lin_cv = lin_cv.max(conmat[i + n * m] + adi);
            }
            lin_cv = lin_cv.max(zero);
            let prerec = cval[n] - lin_cv;
            if !(prerec > zero && preref < zero) {
                break;
            }
            cpen = cpen.max((-two * (preref / prerec)).min(realmax));
            if super::update::findpole(cpen, &cval, &fval, n) == n {
                break;
            }
        }
        cpen
    }
}

/// Ratio between the typical change of `F` and that of the constraints
/// (Powell 1994 eq. 12–13). PRIMA `fcratio`.
fn fcratio<F: Scalar>(conmat: &[F], fval: &[F], n: usize, m: usize) -> F {
    let zero = F::zero();
    let half = F::from_f64(0.5).unwrap();
    let np = n + 1;
    let mut r = zero;
    if m == 0 {
        return r;
    }
    // cmin/cmax over -conmat (convert to Powell's c ≥ 0 form).
    let cmin: Vec<F> = (0..m)
        .map(|i| {
            (0..np)
                .map(|j| -conmat[i + j * m])
                .fold(F::infinity(), F::min)
        })
        .collect();
    let cmax: Vec<F> = (0..m)
        .map(|i| {
            (0..np)
                .map(|j| -conmat[i + j * m])
                .fold(F::neg_infinity(), F::max)
        })
        .collect();
    let fmin = fval.iter().cloned().fold(F::infinity(), F::min);
    let fmax = fval.iter().cloned().fold(F::neg_infinity(), F::max);
    let any = (0..m).any(|i| cmin[i] < half * cmax[i]);
    if any && fmin < fmax {
        let denom = (0..m)
            .filter(|&i| cmin[i] < half * cmax[i])
            .map(|i| cmax[i].max(zero) - cmin[i])
            .fold(F::infinity(), F::min);
        r = (fmax - fmin) / denom;
    }
    r
}

/// Trust-region reduction ratio with Inf/NaN handling (PRIMA `redrat`).
fn redrat<F: Scalar>(ared: F, pred: F, rshrink: F) -> F {
    let realmax = F::max_value();
    let half = F::from_f64(0.5).unwrap();
    if ared.is_nan() {
        -realmax
    } else if pred.is_nan() || pred <= F::zero() {
        if ared > F::zero() {
            half * rshrink
        } else {
            -realmax
        }
    } else if pred.is_infinite()
        && pred > F::zero()
        && ared.is_infinite()
        && ared > F::zero()
    {
        F::one()
    } else if pred.is_infinite()
        && pred > F::zero()
        && ared.is_infinite()
        && ared < F::zero()
    {
        -realmax
    } else {
        ared / pred
    }
}

/// Reduce `ρ` toward `ρ_end` (PRIMA `redrho`, shared schedule).
fn redrho<F: Scalar>(rho_in: F, rhoend: F) -> F {
    let tenth = F::from_f64(0.1).unwrap();
    let ratio = rho_in / rhoend;
    if ratio > F::from_f64(250.0).unwrap() {
        tenth * rho_in
    } else if ratio <= F::from_f64(16.0).unwrap() {
        rhoend
    } else {
        ratio.sqrt() * rhoend
    }
}
