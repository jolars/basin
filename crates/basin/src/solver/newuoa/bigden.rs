//! BIGDEN: the denominator-maximizing geometry step (Powell 2006, §6).
//!
//! [`biglag`](super::biglag) chooses the geometry step `d` to maximize the
//! Lagrange value `|ℓ_t(x_opt + d)| = |τ|`. A large `|τ|` *usually* keeps the
//! least-Frobenius `H`-update denominator `σ = αβ + τ²` (eq. 4.12) well away
//! from zero, but computer rounding can leave
//!
//! ```text
//! |σ| = |αβ + τ²| < 0.8 τ²          (eq. 6.17)
//! ```
//!
//! in which case the update is ill-conditioned. BIGDEN is Powell's remedy: it
//! re-solves the geometry subproblem maximizing `|σ|` *directly*,
//!
//! ```text
//! maximize  |σ(x_opt + d)|   subject to  ‖d‖ ≤ Δ̄ ,          (eq. 6.18)
//! ```
//!
//! seeded with BIGLAG's step (`|σ|` is expected large where `|τ|` is large; eq.
//! 6.21). Because `σ(x)` is a *quartic* polynomial, this is much more laborious
//! than BIGLAG, but, as Powell notes, the situation (6.17) "is very rare in
//! practice", so the cost is incurred almost never.
//!
//! # Method (eqs. 6.22–6.34)
//!
//! Like BIGLAG, BIGDEN is an iterative 2-D circular search: each iteration
//! rotates `d(θ) = cosθ d_{j-1} + sinθ s_j` with `s_j ⟂ d_{j-1}`,
//! `‖d_{j-1}‖ = ‖s_j‖ = Δ̄`. Along the circle, `σ̂(θ) = σ(x_opt + d(θ))` is a
//! trigonometric polynomial of degree 4,
//!
//! ```text
//! σ̂(θ) = σ̌₁ + Σ_{k=1}^4 { σ̌_{2k} cos(kθ) + σ̌_{2k+1} sin(kθ) } ,   (eq. 6.22)
//! ```
//!
//! whose nine coefficients are assembled from the `(m+n)×5` matrices `U`
//! (coefficients of `w − v`, eqs. 6.24–6.26) and `V = H_red U` (eq. 6.27), plus
//! the square-bracket terms of eq. 6.29. The maximizer of `|σ̂|` is found by a
//! 49-point sweep with a 3-point parabolic refinement, and the next search
//! direction `s_{j+1}` is set to (half) the denominator gradient (eqs.
//! 6.31–6.34). This module is a direct port of PRIMA's classical `bigden.f`.
//!
//! BIGDEN returns only the refined `d`; the driver
//! ([`super::driver`]) recomputes `α`/`β`/`τ`/`σ` for the committed step through
//! the ordinary [`prepare_update`](QuadraticModel::prepare_update) /
//! [`update_params`](QuadraticModel::update_params) path.

use crate::core::math::Scalar;
use crate::solver::powell::QuadraticModel;

/// Test-only counter of how many times [`bigden`](QuadraticModel::bigden) has
/// run, so a test can assert the eq. 6.17 fallback is actually exercised.
#[cfg(test)]
pub(crate) static BIGDEN_CALLS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

impl<F: Scalar> QuadraticModel<F> {
    /// The `t`-th column of the `Ω` block of `H` (= the `Ω`-part of `H eₜ`),
    /// `(Ω eₜ)_i = Σ_k sₖ z_{k,t} z_{k,i}`, together with its `t`-th entry
    /// `α = Ω_{tt} = eₜᵀ H eₜ`. (PRIMA stores this column in `W(N+1..N+NPT)`.)
    fn omega_column(&self, t: usize) -> (F, Vec<F>) {
        let m = self.m;
        let rank = m - self.n - 1;
        let mut col = vec![F::zero(); m];
        for k in 0..rank {
            let temp = self.zsign[k] * self.zmat.get(t, k);
            for i in 0..m {
                col[i] = col[i] + temp * self.zmat.get(i, k);
            }
        }
        let alpha = col[t];
        (alpha, col)
    }

    /// One BIGDEN iteration's coefficient assembly (Powell 2006, eqs. 6.22–6.30).
    ///
    /// Given the current step `d` and a direction `s` already orthogonal to `d`
    /// with `‖s‖ = ‖d‖`, build the `(m+n)×5` matrix `prod = H·U` (the `θ`-Fourier
    /// rows of `θ·Wcheck`, eq. 6.27) and the nine coefficients `denex` of
    /// `σ̂(θ)` (eq. 6.22). Returns `(prod, denex)`. The driver gradient recurrence
    /// reads `prod`; the angle sweep reads `denex`.
    fn bigden_iter_coeffs(&self, t: usize, alpha: F, d: &[F], s: &[F]) -> (Vec<[F; 5]>, [F; 9]) {
        let n = self.n;
        let m = self.m;
        let rank = m - n - 1;
        let half = F::from_f64(0.5).expect("0.5 representable");
        let quart = F::from_f64(0.25).expect("0.25 representable");
        let two = F::from_f64(2.0).expect("2.0 representable");
        let zero = F::zero();

        let xopt = self.xpt_row(self.kopt).to_vec();
        let dd = dot(d, d);
        let xoptsq = dot(&xopt, &xopt);
        let xoptd = dot(&xopt, d);
        let xopts = dot(&xopt, s);

        // First two terms of β (the part that does not pass through H), eq. 6.30.
        let mut den = [zero; 9];
        let tempa = half * xoptd * xoptd;
        let tempb = half * xopts * xopts;
        den[0] = dd * (xoptsq + half * dd) + tempa + tempb;
        den[1] = two * xoptd * dd;
        den[2] = two * xopts * dd;
        den[3] = tempa - tempb;
        den[4] = xoptd * xopts;

        // wvec: the θ-Fourier coefficients of Wcheck (`w − v`), eqs. 6.24–6.26.
        // Rows 0..m are the λ-part; rows m..m+n are the g-part.
        let mut wvec = vec![[zero; 5]; m + n];
        for k in 0..m {
            let row = self.xpt_row(k);
            let ta = dot(row, d);
            let tb = dot(row, s);
            let tc = dot(row, &xopt);
            wvec[k][0] = quart * (ta * ta + tb * tb);
            wvec[k][1] = ta * tc;
            wvec[k][2] = tb * tc;
            wvec[k][3] = quart * (ta * ta - tb * tb);
            wvec[k][4] = half * ta * tb;
        }
        for i in 0..n {
            let ip = m + i;
            wvec[ip][1] = d[i];
            wvec[ip][2] = s[i];
        }

        // prod = H · wvec, column by column (eq. 6.27). Column jc spans either the
        // λ-part only (NW = m) or the full vector (NW = m+n) for jc ∈ {1, 2}
        // (PRIMA's `JC == 2 .OR. JC == 3`), where the g-part of `w − v` is nonzero.
        let mut prod = vec![[zero; 5]; m + n];
        for jc in 0..5 {
            let full = jc == 1 || jc == 2;
            // λ-part: Ω · wvec_λ (via the factorization).
            for j in 0..rank {
                let mut sum = zero;
                for k in 0..m {
                    sum = sum + self.zmat.get(k, j) * wvec[k][jc];
                }
                sum = self.zsign[j] * sum;
                for k in 0..m {
                    prod[k][jc] = prod[k][jc] + sum * self.zmat.get(k, j);
                }
            }
            if full {
                // λ-part += Ξᵀ · wvec_g.
                for k in 0..m {
                    let mut sum = zero;
                    for r in 0..n {
                        sum = sum + self.bmat_xi.get(r, k) * wvec[m + r][jc];
                    }
                    prod[k][jc] = prod[k][jc] + sum;
                }
            }
            // g-part: Ξ · wvec_λ (+ Υ · wvec_g when `full`).
            let nw = if full { m + n } else { m };
            for r in 0..n {
                let mut sum = zero;
                for i in 0..nw {
                    let bij = if i < m {
                        self.bmat_xi.get(r, i)
                    } else {
                        self.bmat_ups.get(i - m, r)
                    };
                    sum = sum + bij * wvec[i][jc];
                }
                prod[m + r][jc] = sum;
            }
        }

        // Subtract the (w−v)ᵀ H (w−v) contribution from β (eq. 6.28), accumulating
        // into all nine coefficients of `den`.
        let mut par = [zero; 5];
        for k in 0..(m + n) {
            let mut sum = zero;
            for i in 0..5 {
                par[i] = half * prod[k][i] * wvec[k][i];
                sum = sum + par[i];
            }
            den[0] = den[0] - par[0] - sum;
            let ta = prod[k][0] * wvec[k][1] + prod[k][1] * wvec[k][0];
            let tb = prod[k][1] * wvec[k][3] + prod[k][3] * wvec[k][1];
            let tc = prod[k][2] * wvec[k][4] + prod[k][4] * wvec[k][2];
            den[1] = den[1] - ta - half * (tb + tc);
            den[5] = den[5] - half * (tb - tc);
            let ta = prod[k][0] * wvec[k][2] + prod[k][2] * wvec[k][0];
            let tb = prod[k][1] * wvec[k][4] + prod[k][4] * wvec[k][1];
            let tc = prod[k][2] * wvec[k][3] + prod[k][3] * wvec[k][2];
            den[2] = den[2] - ta - half * (tb - tc);
            den[6] = den[6] - half * (tb + tc);
            let ta = prod[k][0] * wvec[k][3] + prod[k][3] * wvec[k][0];
            den[3] = den[3] - ta - par[1] + par[2];
            let ta = prod[k][0] * wvec[k][4] + prod[k][4] * wvec[k][0];
            let tb = prod[k][1] * wvec[k][2] + prod[k][2] * wvec[k][1];
            den[4] = den[4] - ta - half * tb;
            den[7] = den[7] - par[3] + par[4];
            let ta = prod[k][3] * wvec[k][4] + prod[k][4] * wvec[k][3];
            den[8] = den[8] - half * ta;
        }

        // denex: the coefficients of the full denominator σ̂ = -α·den + (eₜᵀ H
        // (w−v))² (eq. 6.29). The square of the degree-2 function `prod[t]` adds
        // degree-4 harmonics.
        let mut denex = [zero; 9];
        let mut sum = zero;
        for i in 0..5 {
            par[i] = half * prod[t][i] * prod[t][i];
            sum = sum + par[i];
        }
        denex[0] = alpha * den[0] + par[0] + sum;
        let ta = two * prod[t][0] * prod[t][1];
        let tb = prod[t][1] * prod[t][3];
        let tc = prod[t][2] * prod[t][4];
        denex[1] = alpha * den[1] + ta + tb + tc;
        denex[5] = alpha * den[5] + tb - tc;
        let ta = two * prod[t][0] * prod[t][2];
        let tb = prod[t][1] * prod[t][4];
        let tc = prod[t][2] * prod[t][3];
        denex[2] = alpha * den[2] + ta + tb - tc;
        denex[6] = alpha * den[6] + tb + tc;
        let ta = two * prod[t][0] * prod[t][3];
        denex[3] = alpha * den[3] + ta + par[1] - par[2];
        let ta = two * prod[t][0] * prod[t][4];
        denex[4] = alpha * den[4] + ta + prod[t][1] * prod[t][2];
        denex[7] = alpha * den[7] + par[3] - par[4];
        denex[8] = alpha * den[8] + prod[t][3] * prod[t][4];

        (prod, denex)
    }

    /// Approximately maximize `|σ(x_opt + d)|` over `‖d‖ ≤ Δ̄` by the 2-D circular
    /// search of Powell 2006, §6 (subroutine BIGDEN). `t` is the interpolation
    /// index to be replaced; `d0` is the step from
    /// [`biglag`](Self::biglag), whose length is taken as the trust-region bound
    /// `Δ̄` (the iteration preserves `‖d‖ = ‖d0‖`).
    ///
    /// Returns the refined step `d`, relative to `x_opt` (the new interpolation
    /// point is `x_opt + d`), the same convention as [`biglag`](Self::biglag). In
    /// degenerate situations (no usable orthogonal direction, e.g. `n = 1`) the
    /// seed `d0` is returned unchanged.
    ///
    /// # Panics
    ///
    /// Panics unless `t < m` and `d0.len() == n`.
    pub(crate) fn bigden(&self, t: usize, d0: &[F]) -> Vec<F> {
        let n = self.n;
        let m = self.m;
        assert!(t < m, "bigden: t must index an interpolation point");
        assert_eq!(d0.len(), n, "bigden: d0 must have length n");

        #[cfg(test)]
        BIGDEN_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let half = F::from_f64(0.5).expect("0.5 representable");
        let one = F::one();
        let zero = F::zero();
        let twopi = F::from_f64(2.0 * core::f64::consts::PI).expect("2π representable");
        let onep1 = from::<F>(1.1);
        let c099 = from::<F>(0.99);
        let degen = from::<F>(1e-8);

        let kopt = self.kopt;
        let xopt = self.xpt_row(kopt).to_vec();
        let (alpha, w_npt) = self.omega_column(t);

        // Seed d from BIGLAG; s usually the direction x_t − x_opt (eq. 6.21).
        let mut d = d0.to_vec();
        let mut s: Vec<F> = (0..n).map(|i| self.xpt_row(t)[i] - xopt[i]).collect();
        let mut dd = dot(&d, &d);
        let mut ds = dot(&d, &s);
        let mut ss = dot(&s, &s);
        if dd <= zero {
            return d0.to_vec();
        }

        // If s is nearly parallel to d, replace it with the interpolation
        // direction least parallel to d (the ω-ratio rule, eq. 6.21).
        if ds * ds > c099 * dd * ss {
            let mut ksav = t;
            let mut dtest = ds * ds / ss;
            for k in 0..m {
                if k == kopt {
                    continue;
                }
                let mut dstemp = zero;
                let mut sstemp = zero;
                for i in 0..n {
                    let diff = self.xpt_row(k)[i] - xopt[i];
                    dstemp = dstemp + d[i] * diff;
                    sstemp = sstemp + diff * diff;
                }
                if dstemp * dstemp / sstemp < dtest {
                    ksav = k;
                    dtest = dstemp * dstemp / sstemp;
                    ds = dstemp;
                    ss = sstemp;
                }
            }
            for i in 0..n {
                s[i] = self.xpt_row(ksav)[i] - xopt[i];
            }
        }
        let mut ssden = dd * ss - ds * ds;
        // No usable 2-D subspace (e.g. n = 1, or s ∥ d): keep BIGLAG's step.
        if ssden <= degen * dd * ss {
            return d0.to_vec();
        }

        let mut densav = zero;
        let mut par = [zero; 9];

        for iterc in 1..=n {
            // Overwrite s with the unit-length-scaled direction ⟂ d, ‖s‖ = ‖d‖.
            let scale = one / ssden.sqrt();
            for i in 0..n {
                s[i] = scale * (dd * s[i] - ds * d[i]);
            }

            let (prod, denex) = self.bigden_iter_coeffs(t, alpha, &d, &s);

            // Seek the angle maximizing |σ̂(θ)|: 49-point sweep (θ = 0 included as
            // the incumbent) with a 3-point parabolic refinement (eq. 6.22).
            let mut sum = denex[0] + denex[1] + denex[3] + denex[5] + denex[7];
            let denold = sum;
            let mut denmax = sum;
            let mut isave: isize = 0;
            let iu = 49usize;
            let step_t = twopi / F::from_f64((iu + 1) as f64).expect("iu+1 representable");
            let mut tempa = zero;
            let mut tempb = zero;
            for i in 1..=iu {
                let angle = F::from_f64(i as f64).expect("i representable") * step_t;
                fill_harmonics(&mut par, angle);
                let sumold = sum;
                sum = zero;
                for j in 0..9 {
                    sum = sum + denex[j] * par[j];
                }
                if sum.abs() > denmax.abs() {
                    denmax = sum;
                    isave = i as isize;
                    tempa = sumold;
                } else if i as isize == isave + 1 {
                    tempb = sum;
                }
            }
            if isave == 0 {
                tempa = sum;
            }
            if isave == iu as isize {
                tempb = denold;
            }
            let mut step = zero;
            if tempa != tempb {
                let ta = tempa - denmax;
                let tb = tempb - denmax;
                if ta + tb != zero {
                    step = half * (ta - tb) / (ta + tb);
                }
            }
            let angle = step_t * (F::from_f64(isave as f64).expect("isave representable") + step);

            // Commit d_j = cosθ* d + sinθ* s; recompute σ̂ and the Lagrange vector.
            fill_harmonics(&mut par, angle);
            let mut denmax_new = zero;
            for j in 0..9 {
                denmax_new = denmax_new + denex[j] * par[j];
            }
            let mut vlag = vec![zero; m + n];
            for k in 0..(m + n) {
                let mut v = zero;
                for j in 0..5 {
                    v = v + prod[k][j] * par[j];
                }
                vlag[k] = v;
            }
            let tau = vlag[t];

            let cos = par[1];
            let sin = par[2];
            let mut wpt = vec![zero; n];
            let mut sum_dw = zero;
            let mut sum_ww = zero;
            dd = zero;
            for i in 0..n {
                d[i] = cos * d[i] + sin * s[i];
                wpt[i] = xopt[i] + d[i];
                dd = dd + d[i] * d[i];
                sum_dw = sum_dw + d[i] * wpt[i];
                sum_ww = sum_ww + wpt[i] * wpt[i];
            }

            // Convergence (eq. 6.19), or the iteration cap j = n.
            if iterc >= n {
                break;
            }
            if iterc > 1 && denold > densav {
                densav = denold;
            }
            if denmax_new.abs() <= onep1 * densav.abs() {
                break;
            }
            densav = denmax_new;

            // s ← ½ ∇σ(x_opt + d_j) for the next subspace (eqs. 6.31–6.34).
            for i in 0..n {
                let temp = sum_dw * xopt[i] + sum_ww * d[i] - vlag[m + i];
                s[i] = tau * self.bmat_xi.get(i, t) + alpha * temp;
            }
            for k in 0..m {
                let row = self.xpt_row(k);
                let sumk = dot(row, &wpt);
                let temp = (tau * w_npt[k] - alpha * vlag[k]) * sumk;
                for i in 0..n {
                    s[i] = s[i] + temp * row[i];
                }
            }
            ss = dot(&s, &s);
            ds = dot(&d, &s);
            ssden = dd * ss - ds * ds;
            if ssden < degen * dd * ss {
                break;
            }
        }

        d
    }
}

/// Build an `F` from an `f64` literal (the constants are exactly representable).
fn from<F: Scalar>(x: f64) -> F {
    F::from_f64(x).expect("constant representable")
}

/// Plain dot product of two equal-length slices.
fn dot<F: Scalar>(a: &[F], b: &[F]) -> F {
    a.iter().zip(b).map(|(x, y)| *x * *y).sum()
}

/// Fill `par` with the harmonic basis `(1, cosθ, sinθ, cos2θ, sin2θ, …, cos4θ,
/// sin4θ)` used to evaluate the degree-4 trig polynomials of eq. 6.22, via the
/// Chebyshev-style recurrence `cos(kθ)`, `sin(kθ)` from `cos((k-1)θ)`,
/// `sin((k-1)θ)`.
fn fill_harmonics<F: Scalar>(par: &mut [F; 9], angle: F) {
    par[0] = F::one();
    par[1] = angle.cos();
    par[2] = angle.sin();
    let mut j = 3;
    while j <= 7 {
        par[j] = par[1] * par[j - 2] - par[2] * par[j - 1];
        par[j + 1] = par[1] * par[j - 1] + par[2] * par[j - 2];
        j += 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::powell::QuadraticModel;

    fn rosen(x: &[f64]) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
    }

    /// `σ` at the absolute point `x_opt + d` (displacement `d` relative to
    /// `x_opt`), computed independently through the tested update path.
    fn sigma_at<F: Scalar>(model: &QuadraticModel<F>, t: usize, d: &[F]) -> F {
        let xopt = model.xpt_row(model.kopt()).to_vec();
        let n = model.n();
        let xnew: Vec<F> = (0..n).map(|i| xopt[i] + d[i]).collect();
        let ctx = model.prepare_update(&xnew);
        model.update_params(t, &ctx).sigma
    }

    /// Evaluate `σ̂(θ) = Σ_j denex_j par_j(θ)` from the assembled coefficients.
    fn sigma_hat(denex: &[f64; 9], theta: f64) -> f64 {
        let mut par = [0.0_f64; 9];
        fill_harmonics(&mut par, theta);
        (0..9).map(|j| denex[j] * par[j]).sum()
    }

    /// The furthest interpolation point from `x_opt` (the driver's Box-7 rule).
    fn far_t<F: Scalar>(model: &QuadraticModel<F>) -> usize {
        let kopt = model.kopt();
        let xopt = model.xpt_row(kopt).to_vec();
        (0..model.m())
            .filter(|j| *j != kopt)
            .map(|j| {
                let r = model.xpt_row(j);
                let d2: F = (0..model.n())
                    .map(|i| (r[i] - xopt[i]) * (r[i] - xopt[i]))
                    .sum();
                (j, d2)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
    }

    /// The strongest internal check: the BIGDEN-assembled `σ̂(θ)` coefficients
    /// must reproduce, over a grid of angles, the `σ` that the already-tested
    /// `prepare_update`/`update_params` path computes at the corresponding
    /// absolute point. This validates the dense eq. 6.24–6.30 assembly directly.
    #[test]
    fn sigma_coefficients_match_update_params() {
        let model = QuadraticModel::initialize(vec![-1.2, 1.0], 0.5, 5, &rosen);
        let n = model.n();
        let t = far_t(&model);
        let (alpha, _) = model.omega_column(t);

        // A seed d and an orthogonal s of equal length (n = 2: the unique ⟂).
        let d = model.biglag(t, 0.4).d;
        let dnorm = dot(&d, &d).sqrt();
        let mut s = vec![-d[1], d[0]];
        let snorm = dot(&s, &s).sqrt();
        for v in &mut s {
            *v *= dnorm / snorm;
        }

        let (_prod, denex) = model.bigden_iter_coeffs(t, alpha, &d, &s);

        for k in 0..16 {
            let theta = std::f64::consts::TAU * (k as f64) / 16.0;
            let dq: Vec<f64> = (0..n)
                .map(|i| theta.cos() * d[i] + theta.sin() * s[i])
                .collect();
            let want = sigma_at(&model, t, &dq);
            let got = sigma_hat(&denex, theta);
            assert!(
                (got - want).abs() <= 1e-9 * (1.0 + want.abs()),
                "θ={theta}: σ̂={got} vs update_params σ={want}"
            );
        }
    }

    /// BIGDEN must keep `‖d‖ ≈ ‖d0‖` (the trust-region bound) and must not make
    /// the denominator worse than its BIGLAG seed.
    #[test]
    fn preserves_radius_and_improves_denominator() {
        let model = QuadraticModel::initialize(vec![-1.2, 1.0], 0.5, 5, &rosen);
        let t = far_t(&model);
        let d0 = model.biglag(t, 0.4).d;
        let r0 = dot(&d0, &d0).sqrt();
        let sig0 = sigma_at(&model, t, &d0).abs();

        let d = model.bigden(t, &d0);
        let r = dot(&d, &d).sqrt();
        assert!((r - r0).abs() <= 1e-9 * (1.0 + r0), "‖d‖={r} vs ‖d0‖={r0}");
        let sig = sigma_at(&model, t, &d).abs();
        assert!(sig >= sig0 - 1e-12, "|σ| not improved: {sig} < seed {sig0}");
    }

    /// BIGDEN's `|σ|` must be close to the brute-force maximum of `|σ(x_opt+d)|`
    /// over the radius-`‖d0‖` disk (it should not leave a far better point).
    #[test]
    fn near_optimal_vs_brute_force() {
        let model = QuadraticModel::initialize(vec![0.4, -0.6, 0.2], 0.5, 7, &|x: &[f64]| {
            (0..3)
                .map(|i| (x[i] - 0.5 * (i as f64)).powi(2))
                .sum::<f64>()
                + 0.3 * x[0] * x[1]
                - 0.2 * x[1] * x[2]
        });
        let t = far_t(&model);
        let d0 = model.biglag(t, 0.3).d;
        let radius = dot(&d0, &d0).sqrt();

        let d = model.bigden(t, &d0);
        let sig = sigma_at(&model, t, &d).abs();

        // Brute force over a spherical grid of the given radius.
        let mut brute = 0.0_f64;
        let steps = 40;
        for a in 0..steps {
            let theta = std::f64::consts::PI * (a as f64) / (steps as f64);
            for b in 0..(2 * steps) {
                let phi = std::f64::consts::PI * (b as f64) / (steps as f64);
                let dir = [
                    radius * theta.sin() * phi.cos(),
                    radius * theta.sin() * phi.sin(),
                    radius * theta.cos(),
                ];
                brute = brute.max(sigma_at(&model, t, &dir).abs());
            }
        }
        assert!(sig >= 0.9 * brute, "bigden |σ|={sig} < 0.9·brute {brute}");
    }

    /// With `n = 1` there is no orthogonal search direction, so BIGDEN returns
    /// the seed unchanged without panicking or producing NaNs.
    #[test]
    fn degenerate_n1_returns_seed() {
        let model =
            QuadraticModel::initialize(vec![0.3], 0.5, 3, &|x: &[f64]| (x[0] - 1.0).powi(2));
        let kopt = model.kopt();
        let t = (0..model.m()).find(|j| *j != kopt).unwrap();
        let d0 = vec![0.2_f64];
        let d = model.bigden(t, &d0);
        assert_eq!(d, d0, "n=1 must return the seed unchanged");
    }
}
