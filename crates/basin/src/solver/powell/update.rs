//! The least-Frobenius-norm model update (Powell 2006, §4; H-update derivation
//! Powell 2004a).
//!
//! One interpolation point `x_t` is replaced by `x⁺ = x_opt + d` per iteration.
//! The update is split into three phases so the §7 MOVE/`t`-selection logic
//! (later step) can choose `t` cheaply:
//!
//! 1. [`prepare_update`](QuadraticModel::prepare_update)—the `t`-independent
//!    work: the vector `w` (eq. 4.10), `H w` (via eq. 4.25), and `β`
//!    (eq. 4.12).
//! 2. [`update_params`](QuadraticModel::update_params)—the per-`t` scalars
//!    `α`, `τ`, `σ` (eq. 4.12). `σ` is the denominator MOVE keeps large.
//! 3. [`commit_update`](QuadraticModel::commit_update)—apply the rank-2 `H`
//!    update (eq. 4.11) to `Ξ`/`Υ` and the `Ω`-factorization (eqs. 4.18–4.20),
//!    then the model update `Γ⁺`/`γ⁺`/`∇Q⁺` (eqs. 4.29–4.30).
//!
//! `β` uses the unclamped definition (eq. 4.12), the form Powell states NEWUOA
//! prefers (§4, the discussion after eq. 4.22): the `Ω`-factorization update
//! absorbs occasional negative `α`/`β`/`σ`/`sₖ` from rounding rather than
//! masking them with the eq. 4.22 safeguard.

use crate::core::math::Scalar;

use super::model::QuadraticModel;

/// Test-only counter of how many times the `Ω`-factorization cancellation
/// branch (eqs. 4.19/4.20) has run, so a stress test can assert that branch is
/// actually exercised rather than silently skipped.
#[cfg(test)]
pub(crate) static CANCELLATION_HITS: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Test-only counter of how many times the §8 Qint replacement
/// ([`adopt_alt_model`](QuadraticModel::adopt_alt_model)) has run, so the VARDIM
/// driver test can assert the robustness path actually fires.
#[cfg(test)]
pub(crate) static QINT_ADOPTIONS: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// The `t`-independent quantities of one update, shared across candidate `t`
/// values (Powell 2006, §4). Built by
/// [`prepare_update`](QuadraticModel::prepare_update).
pub(crate) struct UpdateContext<F = f64> {
    /// `H w` in the suppressed `(m+n)` layout `[λ-part (m); g-part (n)]`
    /// (eq. 4.25). The suppressed constant-term entry is absent.
    pub(crate) hw: Vec<F>,
    /// `β = ½‖x⁺−x0‖⁴ − wᵀ H w` (eq. 4.12, unclamped).
    pub(crate) beta: F,
    /// The new displacement `x⁺ − x0`, length `n`.
    pub(crate) xnew: Vec<F>,
}

/// The per-`t` denominator scalars (Powell 2006, eq. 4.12). Built by
/// [`update_params`](QuadraticModel::update_params).
pub(crate) struct UpdateScalars<F = f64> {
    /// `α = eₜᵀ H eₜ = Ω_{tt}`.
    pub(crate) alpha: F,
    /// `β` (copied from the [`UpdateContext`]).
    pub(crate) beta: F,
    /// `τ = eₜᵀ H w`.
    pub(crate) tau: F,
    /// `σ = αβ + τ²`—the update denominator.
    pub(crate) sigma: F,
}

impl<F: Scalar> QuadraticModel<F> {
    /// Apply the suppressed inverse-KKT operator `H` to a vector given by its
    /// `λ`-part (length `m`) and `g`-part (length `n`), returning the
    /// `(λ-part, g-part)` of `H·v`:
    ///
    /// ```text
    /// (H v)_λ = Ω v_λ + Ξᵀ v_g ,   (H v)_g = Ξ v_λ + Υ v_g ,
    /// ```
    ///
    /// with `Ω v_λ = Σ_k sₖ zₖ (zₖ·v_λ)` from the stored factorization.
    ///
    /// `pub(crate)` so RESCUE (the `bobyqa::rescue` module) can reuse it to
    /// form `vlag = H·(w−v)` against the rebuilt provisional `H`.
    pub(crate) fn apply_h(&self, v_lambda: &[F], v_g: &[F]) -> (Vec<F>, Vec<F>) {
        let n = self.n;
        let m = self.m;
        let rank = m - n - 1;

        // λ-part: Ω v_λ (via factorization) + Ξᵀ v_g.
        let mut out_lambda = vec![F::zero(); m];
        for k in 0..rank {
            let mut zdot = F::zero();
            for i in 0..m {
                zdot = zdot + self.zmat.get(i, k) * v_lambda[i];
            }
            let coef = self.zsign[k] * zdot;
            for i in 0..m {
                out_lambda[i] = out_lambda[i] + coef * self.zmat.get(i, k);
            }
        }
        for r in 0..n {
            let vr = v_g[r];
            for i in 0..m {
                out_lambda[i] = out_lambda[i] + self.bmat_xi.get(r, i) * vr;
            }
        }

        // g-part: Ξ v_λ + Υ v_g.
        let mut out_g = vec![F::zero(); n];
        for r in 0..n {
            let mut acc = F::zero();
            for i in 0..m {
                acc = acc + self.bmat_xi.get(r, i) * v_lambda[i];
            }
            for s in 0..n {
                acc = acc + self.bmat_ups.get(r, s) * v_g[s];
            }
            out_g[r] = acc;
        }

        (out_lambda, out_g)
    }

    /// The coefficients of the `t`-th Lagrange function `ℓ_t` of the current
    /// model (Powell 2006, eqs. 6.1–6.4): the gradient `g = ∇ℓ_t(x0)` (length
    /// `n`) and the implicit-Hessian coefficients `λ` (length `m`), with
    /// `∇²ℓ_t = Σ_k λ_k (x_k − x0)(x_k − x0)ᵀ`.
    ///
    /// These are exactly the `t`-th column of `H`: `(λ, g) = H eₜ` in the
    /// suppressed `[λ; g]` layout, i.e. `λ = Ω eₜ` and `g = Ξ eₜ`. BIGLAG
    /// (Powell 2006, §6) reads them to maximize `|ℓ_t(x_opt + d)|`. Pairs with
    /// [`lagrange_hessian_matvec`](Self::lagrange_hessian_matvec).
    pub(crate) fn lagrange_coeffs(&self, t: usize) -> (Vec<F>, Vec<F>) {
        assert!(
            t < self.m,
            "lagrange_coeffs: t must index an interpolation point"
        );
        let mut e_t = vec![F::zero(); self.m];
        e_t[t] = F::one();
        let v_g = vec![F::zero(); self.n];
        let (lambda, g) = self.apply_h(&e_t, &v_g);
        (g, lambda)
    }

    /// The right-hand side `r` of the §8 alternative-model interpolation system:
    /// `rᵢ = F(xᵢ) − F(x_opt)` (Powell 2006, §8, the modified RHS of system 3.10).
    ///
    /// Subtracting `F(x_opt)` rather than using the raw `F(xᵢ)` reduces rounding
    /// damage and leaves `∇Q_int`/`∇²Q_int` unchanged, since the Lagrange
    /// functions form a partition of unity (so a constant shift moves only the
    /// dropped constant term).
    fn alt_rhs(&self) -> Vec<F> {
        let fopt = self.fval[self.kopt];
        (0..self.m).map(|i| self.fval[i] - fopt).collect()
    }

    /// The current model's gradient at the best point,
    /// `∇Q(x_opt) = ∇Q(x0) + ∇²Q·(x_opt − x0)`.
    ///
    /// NEWUOA (and PRIMA's `GOPT`) maintains the gradient at `x_opt`, and the §8
    /// test (eq. 8.4) compares it against the alternative model's gradient there
    /// (see [`alt_gradient_at_opt`](Self::alt_gradient_at_opt)).
    pub(crate) fn gradient_at_opt(&self) -> Vec<F> {
        let xopt = self.xpt.row(self.kopt).to_vec();
        let hd = self.hessian_matvec(&xopt);
        (0..self.n).map(|i| self.gq[i] + hd[i]).collect()
    }

    /// The §8 alternative model's gradient at the best point, `∇Q_int(x_opt)`.
    ///
    /// `Q_int` minimizes `‖∇²Q‖_F` subject to the current interpolation
    /// conditions (Powell 2006, §8, eq. 8.3): its gradient at `x0` is `Ξ·r` and
    /// its implicit-Hessian coefficients are `Ω·r` (with `rᵢ = F(xᵢ) − F(x_opt)`,
    /// [`alt_rhs`](Self::alt_rhs)), both read off `H·[r; 0]`. The gradient is then
    /// shifted to `x_opt` by `∇²Q_int·(x_opt − x0)`. This mirrors PRIMA's
    /// `galt = Ξ·r + hess_mul(x_opt, …, Ω·r)` in `tryqalt`. The full model is only
    /// formed on the rare adoption ([`adopt_alt_model`](Self::adopt_alt_model)).
    pub(crate) fn alt_gradient_at_opt(&self) -> Vec<F> {
        let r = self.alt_rhs();
        let zeros = vec![F::zero(); self.n];
        // (Ω·r, Ξ·r): the alternative implicit coefficients and the x0-gradient.
        let (pqalt, galt0) = self.apply_h(&r, &zeros);
        let xopt = self.xpt.row(self.kopt).to_vec();
        let hd = self.lagrange_hessian_matvec(&pqalt, &xopt);
        (0..self.n).map(|i| galt0[i] + hd[i]).collect()
    }

    /// The §8 alternative model's *change* `Q_int(x_opt + d) − Q_int(x_opt)`.
    ///
    /// LINCOA's `tryqalt` compares the alternative model's prediction error
    /// against the current model's (PRIMA `quadinc(d, xpt, galt, pqalt)`), unlike
    /// NEWUOA/BOBYQA which compare gradients. `Q_int` carries no explicit `Γ`
    /// part, so its Hessian is the rank-one sum keyed by `γ_alt = Ω·r` (with
    /// `rᵢ = F(xᵢ) − F(x_opt)`, [`alt_rhs`](Self::alt_rhs)); its gradient at
    /// `x_opt` is `Ξ·r + ∇²Q_int·(x_opt − x0)`.
    pub(crate) fn alt_model_change(&self, d: &[F]) -> F {
        let r = self.alt_rhs();
        let zeros = vec![F::zero(); self.n];
        let (pqalt, galt0) = self.apply_h(&r, &zeros);
        let xopt = self.xpt.row(self.kopt).to_vec();
        let hxopt = self.lagrange_hessian_matvec(&pqalt, &xopt);
        let gopt: Vec<F> = (0..self.n).map(|i| galt0[i] + hxopt[i]).collect();
        let hd = self.lagrange_hessian_matvec(&pqalt, d);
        let half = F::from_f64(0.5).expect("0.5 representable");
        dot(&gopt, d) + half * dot(d, &hd)
    }

    /// Replace `Q` by the §8 alternative model `Q_int` (Powell 2006, eq. 8.3):
    /// the least-Frobenius-Hessian interpolant of the current function values.
    ///
    /// Solving the KKT system with right-hand side `r` (= [`alt_rhs`]) is exactly
    /// `H · [r; 0]`, whose `g`-part is `∇Q_int(x0) = Ξ r` and whose `λ`-part is
    /// the implicit-Hessian coefficients `γ = Ω r`. `Q_int` carries **no explicit
    /// `Γ` part** (eq. 8.3), so the explicit Hessian block is zeroed. The
    /// interpolation set, `H`, `fval`, and `kopt` are untouched.
    pub(crate) fn adopt_alt_model(&mut self) {
        let r = self.alt_rhs();
        let zeros = vec![F::zero(); self.n];
        let (gamma_new, gq_new) = self.apply_h(&r, &zeros);
        self.gq = gq_new;
        self.gamma = gamma_new;
        for i in 0..self.n {
            for j in 0..self.n {
                self.gamma_explicit.set(i, j, F::zero());
            }
        }
        #[cfg(test)]
        QINT_ADOPTIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Compute the `t`-independent update quantities for the proposed new point
    /// `x⁺` with displacement `xnew = x⁺ − x0` (Powell 2006, eqs. 4.10, 4.25,
    /// 4.26, 4.12). Call once per iteration, then [`update_params`] per `t`.
    ///
    /// [`update_params`]: QuadraticModel::update_params
    pub(crate) fn prepare_update(&self, xnew: &[F]) -> UpdateContext<F> {
        let n = self.n;
        let m = self.m;
        assert_eq!(xnew.len(), n, "prepare_update: xnew must have length n");
        let half = F::from_f64(0.5).expect("0.5 representable");
        let two = F::from_f64(2.0).expect("2.0 representable");
        let opt = self.kopt;
        let xopt = self.xpt.row(opt).to_vec();

        // λ-parts of w (eq. 4.10) and v (eq. 4.24): w_i = ½((xᵢ−x0)·xnew)²,
        // v_i = ½((xᵢ−x0)·xopt)². The g-parts are xnew and xopt themselves;
        // the suppressed constant-term entry (= 1 for both) is dropped.
        let mut w_lambda = vec![F::zero(); m];
        let mut v_lambda = vec![F::zero(); m];
        let mut wmv_lambda = vec![F::zero(); m];
        for i in 0..m {
            let xi = self.xpt.row(i);
            let dw = dot(xi, xnew);
            let dv = dot(xi, &xopt);
            w_lambda[i] = half * dw * dw;
            v_lambda[i] = half * dv * dv;
            wmv_lambda[i] = w_lambda[i] - v_lambda[i];
        }
        let mut wmv_g = vec![F::zero(); n];
        for k in 0..n {
            wmv_g[k] = xnew[k] - xopt[k];
        }

        // H(w − v); then H w = H(w − v) + e_opt (eq. 4.25), exploiting H v = e_opt.
        let (hwmv_lambda, hwmv_g) = self.apply_h(&wmv_lambda, &wmv_g);
        let mut hw = vec![F::zero(); m + n];
        hw[..m].copy_from_slice(&hwmv_lambda);
        hw[m..(m + n)].copy_from_slice(&hwmv_g);
        hw[opt] = hw[opt] + F::one();

        // wᵀ H w via eq. 4.26 (the suppressed (w−v) has zero constant entry).
        let wmv_h_wmv = dot(&wmv_lambda, &hwmv_lambda) + dot(&wmv_g, &hwmv_g);
        let w_h_w = wmv_h_wmv + two * w_lambda[opt] - v_lambda[opt];

        // β = ½‖xnew‖⁴ − wᵀ H w (eq. 4.12, unclamped).
        let nrm2 = dot(xnew, xnew);
        let quartic = half * nrm2 * nrm2;
        let beta = quartic - w_h_w;

        UpdateContext {
            hw,
            beta,
            xnew: xnew.to_vec(),
        }
    }

    /// The per-`t` denominator scalars `α = Ω_{tt}`, `τ = (H w)_t`, and
    /// `σ = αβ + τ²` (Powell 2006, eq. 4.12). Cheap (`O(m)`), so MOVE can scan
    /// every candidate `t` before committing.
    pub(crate) fn update_params(&self, t: usize, ctx: &UpdateContext<F>) -> UpdateScalars<F> {
        let m = self.m;
        let n = self.n;
        let rank = m - n - 1;
        assert!(t < m, "update_params: t must index an interpolation point");

        // α = Ω_{tt} = Σ_k sₖ z_{k,t}².
        let mut alpha = F::zero();
        for k in 0..rank {
            let zt = self.zmat.get(t, k);
            alpha = alpha + self.zsign[k] * zt * zt;
        }
        let tau = ctx.hw[t];
        let sigma = alpha * ctx.beta + tau * tau;
        UpdateScalars {
            alpha,
            beta: ctx.beta,
            tau,
            sigma,
        }
    }

    /// Commit the swap `x_t ← x⁺ = x_opt + d`: apply the rank-2 `H` update
    /// (Powell 2006, eq. 4.11) to `Ξ`/`Υ` and the `Ω`-factorization (eqs.
    /// 4.18–4.20), then update the model `Γ`/`γ`/`∇Q` (eqs. 4.29–4.30).
    ///
    /// `ctx` and `scalars` are the outputs of [`prepare_update`] /
    /// [`update_params`] for this `t`; `f_new = F(x⁺)` is the objective value at
    /// the new point. The model residual `df = F(x⁺) − Q_old(x⁺)` (eq. 4.23) is
    /// formed internally without the (unstored) constant term.
    ///
    /// # Panics
    ///
    /// Panics if `σ = 0` (the update is undefined; the caller's MOVE logic must
    /// keep `|σ|` away from zero). The Ω-factorization's cancellation branch
    /// (eqs. 4.19–4.20) likewise assumes its denominator `ζ = τ² ± β·z²` stays
    /// nonzero—the same MOVE-maintained precondition, not separately guarded.
    ///
    /// [`prepare_update`]: QuadraticModel::prepare_update
    /// [`update_params`]: QuadraticModel::update_params
    pub(crate) fn commit_update(
        &mut self,
        t: usize,
        ctx: &UpdateContext<F>,
        scalars: &UpdateScalars<F>,
        f_new: F,
    ) {
        let n = self.n;
        let m = self.m;
        let rank = m - n - 1;
        assert!(t < m, "commit_update: t must index an interpolation point");
        assert!(
            scalars.sigma != F::zero(),
            "commit_update: σ = 0 (degenerate update; MOVE must keep |σ| large)"
        );
        let alpha = scalars.alpha;
        let beta = scalars.beta;
        let tau = scalars.tau;
        let sigma = scalars.sigma;

        // --- Phase 0: model residual df (eq. 4.23), using the OLD model. ---
        // df = (F(x⁺) − F(x_opt)) − (Q_old(x⁺) − Q_old(x_opt)), with
        // Q_old(x_opt) = F(x_opt) ⇒ df = F(x⁺) − Q_old(x⁺), no constant needed.
        let xopt_disp = self.xpt.row(self.kopt).to_vec();
        let q_new = self.eval_change(&ctx.xnew);
        let q_opt = self.eval_change(&xopt_disp);
        let df = (f_new - self.fval[self.kopt]) - (q_new - q_opt);

        // --- Phase 1: capture OLD quantities for the H update. ---
        // ehw = e_t − H w (suppressed [λ; g]).
        let mut ehw_lambda = vec![F::zero(); m];
        for i in 0..m {
            let e = if i == t { F::one() } else { F::zero() };
            ehw_lambda[i] = e - ctx.hw[i];
        }
        let mut ehw_g = vec![F::zero(); n];
        for r in 0..n {
            ehw_g[r] = -ctx.hw[m + r];
        }
        // het = H e_t: λ-part is Ω column t, g-part is Ξ column t.
        let mut het_lambda = vec![F::zero(); m];
        for i in 0..m {
            let mut acc = F::zero();
            for k in 0..rank {
                acc = acc + self.zsign[k] * self.zmat.get(i, k) * self.zmat.get(t, k);
            }
            het_lambda[i] = acc;
        }
        let mut het_g = vec![F::zero(); n];
        for r in 0..n {
            het_g[r] = self.bmat_xi.get(r, t);
        }
        // OLD point t and its implicit-Hessian coefficient (eq. 4.29).
        let old_xt = self.xpt.row(t).to_vec();
        let old_gamma_t = self.gamma[t];

        let inv_sigma = F::one() / sigma;

        // --- Phase 2: Ξ⁺ and Υ⁺ from the eq. 4.11 sub-blocks. ---
        // bottom-left (g × λ) block ⇒ Ξ; bottom-right (g × g) ⇒ Υ.
        for r in 0..n {
            for j in 0..m {
                let term = alpha * ehw_g[r] * ehw_lambda[j] - beta * het_g[r] * het_lambda[j]
                    + tau * (het_g[r] * ehw_lambda[j] + ehw_g[r] * het_lambda[j]);
                let updated = self.bmat_xi.get(r, j) + inv_sigma * term;
                self.bmat_xi.set(r, j, updated);
            }
        }
        for r in 0..n {
            for s in 0..n {
                let term = alpha * ehw_g[r] * ehw_g[s] - beta * het_g[r] * het_g[s]
                    + tau * (het_g[r] * ehw_g[s] + ehw_g[r] * het_g[s]);
                let updated = self.bmat_ups.get(r, s) + inv_sigma * term;
                self.bmat_ups.set(r, s, updated);
            }
        }

        // --- Phase 3: Ω-factorization update (eqs. 4.17–4.20). ---
        self.update_omega_factorization(t, &ehw_lambda, tau, sigma, beta);

        // --- Phase 4: install the new interpolation point. ---
        for k in 0..n {
            self.xpt.set(t, k, ctx.xnew[k]);
        }
        self.fval[t] = f_new;

        // --- Phase 5: H⁺ e_t = (Ω⁺ column t, Ξ⁺ column t). ---
        let mut hcol_lambda = vec![F::zero(); m];
        for i in 0..m {
            let mut acc = F::zero();
            for k in 0..rank {
                acc = acc + self.zsign[k] * self.zmat.get(i, k) * self.zmat.get(t, k);
            }
            hcol_lambda[i] = acc;
        }
        let mut hcol_g = vec![F::zero(); n];
        for r in 0..n {
            hcol_g[r] = self.bmat_xi.get(r, t);
        }

        // --- Phase 6: model update (eqs. 4.29, 4.30). ---
        // Γ⁺ = Γ + γ_t (x_t^old − x0)(x_t^old − x0)ᵀ.
        for i in 0..n {
            for j in 0..n {
                let add = old_gamma_t * old_xt[i] * old_xt[j];
                self.gamma_explicit
                    .set(i, j, self.gamma_explicit.get(i, j) + add);
            }
        }
        // λ⁺ = df · (H⁺ e_t)_λ ; γ⁺_t = λ⁺_t, γ⁺_j = γ_j + λ⁺_j (j ≠ t).
        for j in 0..m {
            let lam = df * hcol_lambda[j];
            if j == t {
                self.gamma[j] = lam;
            } else {
                self.gamma[j] = self.gamma[j] + lam;
            }
        }
        // ∇Q⁺(x0) = ∇Q_old(x0) + g⁺, g⁺ = df · (H⁺ e_t)_g.
        for k in 0..n {
            self.gq[k] = self.gq[k] + df * hcol_g[k];
        }

        // --- Phase 7: refresh kopt. ---
        let mut kopt = 0;
        for j in 1..m {
            if self.fval[j] < self.fval[kopt] {
                kopt = j;
            }
        }
        self.kopt = kopt;
    }

    /// Update the factorization `Ω = Σ_k sₖ zₖ zₖᵀ` for the rank-2 `H` change
    /// (Powell 2006, eqs. 4.17–4.20). First collapse the `t`-th row of `Z`
    /// within each sign group to a single nonzero (eq. 4.17 rotations), then
    /// apply eq. 4.18 (one representative, `|K| = m−n−2`) or eqs. 4.19/4.20
    /// (two representatives, `|K| = m−n−3`).
    ///
    /// `chop` is `ehw_lambda = (e_t − H w)` restricted to its first `m`
    /// components (eq. 4.18).
    fn update_omega_factorization(&mut self, t: usize, chop: &[F], tau: F, sigma: F, beta: F) {
        let m = self.m;
        let n = self.n;
        let rank = m - n - 1;

        // Collapse: within each sign group, rotate (eq. 4.17) so at most one
        // column has a nonzero t-th entry.
        let mut pos_rep: Option<usize> = None;
        let mut neg_rep: Option<usize> = None;
        for k in 0..rank {
            if self.zmat.get(t, k) == F::zero() {
                continue;
            }
            let positive = self.zsign[k] > F::zero();
            let cur = if positive { pos_rep } else { neg_rep };
            match cur {
                None => {
                    if positive {
                        pos_rep = Some(k);
                    } else {
                        neg_rep = Some(k);
                    }
                }
                Some(r) => {
                    let a = self.zmat.get(t, r);
                    let b = self.zmat.get(t, k);
                    let denom = (a * a + b * b).sqrt();
                    let c = a / denom;
                    let s = b / denom;
                    self.rotate_zmat_cols(r, k, c, s);
                    // z[k][t] is now mathematically zero; force it exactly so the
                    // membership test above stays robust to rounding.
                    self.zmat.set(t, k, F::zero());
                }
            }
        }

        let sign_sigma = if sigma >= F::zero() {
            F::one()
        } else {
            -F::one()
        };

        match (pos_rep, neg_rep) {
            // Normal case |K| = m−n−2 (eq. 4.18): one representative column.
            (Some(kk), None) | (None, Some(kk)) => {
                let ztkk = self.zmat.get(t, kk);
                let inv_sqrt = F::one() / sigma.abs().sqrt();
                for i in 0..m {
                    let zi = self.zmat.get(i, kk);
                    self.zmat.set(i, kk, inv_sqrt * (tau * zi + ztkk * chop[i]));
                }
                self.zsign[kk] = sign_sigma * self.zsign[kk];
            }
            // Cancellation case |K| = m−n−3 (eqs. 4.19/4.20): one positive and
            // one negative representative remain. Two new columns are formed to
            // avoid cancellation, the branch depending on the sign of β.
            (Some(kp), Some(kn)) => {
                #[cfg(test)]
                CANCELLATION_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                // Capture the OLD columns z_1 (positive) and z_2 (negative);
                // both new columns are linear combinations of them.
                let z1: Vec<F> = (0..m).map(|i| self.zmat.get(i, kp)).collect();
                let z2: Vec<F> = (0..m).map(|i| self.zmat.get(i, kn)).collect();
                let zt1 = z1[t];
                let zt2 = z2[t];

                if beta >= F::zero() {
                    // eq. 4.19: ζ = τ² + β Z_{t,1}².
                    let zeta = tau * tau + beta * zt1 * zt1;
                    let inv_sqrt_zeta = F::one() / zeta.abs().sqrt();
                    let inv_sqrt_zs = F::one() / (zeta * sigma).abs().sqrt();
                    for i in 0..m {
                        let new1 = inv_sqrt_zeta * (tau * z1[i] + zt1 * chop[i]);
                        let new2 = inv_sqrt_zs
                            * (-beta * zt1 * zt2 * z1[i] + zeta * z2[i] + tau * zt2 * chop[i]);
                        self.zmat.set(i, kp, new1);
                        self.zmat.set(i, kn, new2);
                    }
                    self.zsign[kp] = F::one(); // s⁺_1 = +1
                    self.zsign[kn] = -sign_sigma; // s⁺_2 = −sign(σ)
                } else {
                    // eq. 4.20: ζ = τ² − β Z_{t,2}².
                    let zeta = tau * tau - beta * zt2 * zt2;
                    let inv_sqrt_zeta = F::one() / zeta.abs().sqrt();
                    let inv_sqrt_zs = F::one() / (zeta * sigma).abs().sqrt();
                    for i in 0..m {
                        let new1 = inv_sqrt_zs
                            * (zeta * z1[i] + beta * zt1 * zt2 * z2[i] + tau * zt1 * chop[i]);
                        let new2 = inv_sqrt_zeta * (tau * z2[i] + zt2 * chop[i]);
                        self.zmat.set(i, kp, new1);
                        self.zmat.set(i, kn, new2);
                    }
                    self.zsign[kp] = sign_sigma; // s⁺_1 = sign(σ)
                    self.zsign[kn] = -F::one(); // s⁺_2 = −1
                }
            }
            (None, None) => {
                panic!("commit_update: Ω_tt = 0 (no factor column couples to t)");
            }
        }
    }

    /// Givens-style rotation of two `Z` columns `c1`, `c2`:
    /// `z_{c1} ← c·z_{c1} + s·z_{c2}`, `z_{c2} ← −s·z_{c1} + c·z_{c2}`
    /// (Powell 2006, eq. 4.17). Valid only when the two columns share a sign,
    /// which preserves `Ω`. `pub(crate)` so RESCUE's `updateh_rsc` can reuse it
    /// for the `planerot` collapse of the `knew`-th row of `Z`.
    pub(crate) fn rotate_zmat_cols(&mut self, c1: usize, c2: usize, cos: F, sin: F) {
        for i in 0..self.m {
            let z1 = self.zmat.get(i, c1);
            let z2 = self.zmat.get(i, c2);
            self.zmat.set(i, c1, cos * z1 + sin * z2);
            self.zmat.set(i, c2, -sin * z1 + cos * z2);
        }
    }
}

/// Plain dot product of two equal-length slices.
fn dot<F: Scalar>(a: &[F], b: &[F]) -> F {
    a.iter().zip(b).map(|(x, y)| *x * *y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::math::DenseMatrix;
    use crate::solver::powell::kkt::{assert_h_matches_inverse, build_w_dense, invert_dense};

    /// Build the full `(m+n+1)` vector `w` (Powell 2006, eq. 4.10) for a new
    /// displacement `xnew`, in the *unsuppressed* index order
    /// `[λ (m); c (1); g (n)]`—the order [`build_w_dense`] uses.
    fn full_w(model: &QuadraticModel<f64>, xnew: &[f64]) -> Vec<f64> {
        let n = model.n();
        let m = model.m();
        let mut w = vec![0.0; m + n + 1];
        for i in 0..m {
            let xi = model.xpt_row(i);
            let d: f64 = xi.iter().zip(xnew).map(|(a, b)| a * b).sum();
            w[i] = 0.5 * d * d;
        }
        w[m] = 1.0; // constant-term entry
        for k in 0..n {
            w[m + 1 + k] = xnew[k];
        }
        w
    }

    /// T8: `update_params` must reproduce `α`, `β`, `τ`, `σ` computed
    /// independently from the *full* dense `H = inv(W)`. This validates the
    /// suppressed-representation arithmetic and the eq. 4.25/4.26 trick against
    /// brute-force linear algebra, before any mutation.
    #[test]
    fn update_params_match_dense_inverse() {
        let model = QuadraticModel::initialize(vec![0.4, -0.3], 0.25, 6, |x: &[f64]| {
            2.0 * x[0] * x[0] + 1.5 * x[0] * x[1] + 3.0 * x[1] * x[1] + x[0] - 2.0 * x[1]
        });
        let n = model.n();
        let m = model.m();
        let w_dense = build_w_dense(&model);
        let h = invert_dense(&w_dense).unwrap();

        // A generic new point.
        let xnew = vec![0.15, 0.35];
        let ctx = model.prepare_update(&xnew);

        // Dense reference quantities.
        let wf = full_w(&model, &xnew);
        let dim = m + n + 1;
        // H wf.
        let mut hwf = vec![0.0; dim];
        for i in 0..dim {
            let mut acc = 0.0;
            for j in 0..dim {
                acc += h.get(i, j) * wf[j];
            }
            hwf[i] = acc;
        }
        // wᵀ H w.
        let w_h_w: f64 = wf.iter().zip(&hwf).map(|(a, b)| a * b).sum();
        let nrm2: f64 = xnew.iter().map(|z| z * z).sum();
        let beta_dense = 0.5 * nrm2 * nrm2 - w_h_w;
        assert!(
            (ctx.beta - beta_dense).abs() < 1e-9,
            "β: suppressed {} vs dense {}",
            ctx.beta,
            beta_dense
        );

        for t in 0..m {
            let s = model.update_params(t, &ctx);
            let alpha_dense = h.get(t, t);
            let tau_dense = hwf[t];
            let sigma_dense = alpha_dense * beta_dense + tau_dense * tau_dense;
            assert!(
                (s.alpha - alpha_dense).abs() < 1e-9,
                "α[t={t}]: {} vs {}",
                s.alpha,
                alpha_dense
            );
            assert!(
                (s.tau - tau_dense).abs() < 1e-9,
                "τ[t={t}]: {} vs {}",
                s.tau,
                tau_dense
            );
            assert!(
                (s.sigma - sigma_dense).abs() < 1e-9,
                "σ[t={t}]: {} vs {}",
                s.sigma,
                sigma_dense
            );
        }
    }

    /// `apply_h` must agree with the dense `H = inv(W)` on an arbitrary vector,
    /// in both the `λ` and `g` blocks (the suppressed `c` entry excluded).
    #[test]
    fn apply_h_matches_dense_inverse() {
        let model = QuadraticModel::initialize(vec![0.0, 0.0], 0.3, 5, |x: &[f64]| {
            x[0] * x[0] + 2.0 * x[1] * x[1] + x[0] - x[1]
        });
        let n = model.n();
        let m = model.m();
        let h = invert_dense(&build_w_dense(&model)).unwrap();

        // Arbitrary suppressed vector (λ then g); the c entry is taken as 0.
        let v_lambda: Vec<f64> = (0..m).map(|i| 0.5 + 0.3 * i as f64).collect();
        let v_g: Vec<f64> = (0..n).map(|k| -1.0 + 0.7 * k as f64).collect();
        let (out_lambda, out_g) = {
            let vl: Vec<f64> = v_lambda.clone();
            let vg: Vec<f64> = v_g.clone();
            // Borrow-free call.
            model_apply_h(&model, &vl, &vg)
        };

        // Dense reference: build full v with c entry 0, multiply by H, read off
        // the λ and g blocks.
        let dim = m + n + 1;
        let mut vfull = vec![0.0; dim];
        vfull[..m].copy_from_slice(&v_lambda);
        for k in 0..n {
            vfull[m + 1 + k] = v_g[k];
        }
        for i in 0..m {
            let mut acc = 0.0;
            for j in 0..dim {
                acc += h.get(i, j) * vfull[j];
            }
            assert!((out_lambda[i] - acc).abs() < 1e-9, "λ[{i}]");
        }
        for r in 0..n {
            let mut acc = 0.0;
            for j in 0..dim {
                acc += h.get(m + 1 + r, j) * vfull[j];
            }
            assert!((out_g[r] - acc).abs() < 1e-9, "g[{r}]");
        }
    }

    /// §8: after [`QuadraticModel::adopt_alt_model`] the model is `Q_int`, the
    /// least-Frobenius-Hessian interpolant, so it must reproduce every function
    /// value: `Q_int(xⱼ) − Q_int(x_opt) == F(xⱼ) − F(x_opt)` for all `j`. With
    /// `npt = ½(n+1)(n+2)` the quadratic is fully determined, so the §3 model is
    /// already exact and both [`gradient_at_opt`] and [`alt_gradient_at_opt`]
    /// equal the analytic `∇F(x_opt)`.
    ///
    /// [`gradient_at_opt`]: QuadraticModel::gradient_at_opt
    /// [`alt_gradient_at_opt`]: QuadraticModel::alt_gradient_at_opt
    #[test]
    fn adopt_alt_model_interpolates_and_matches_gradient() {
        // F(x) = 2x0² + 1.5 x0 x1 + 3 x1² + x0 − 2 x1;
        // ∇F = (4x0 + 1.5x1 + 1, 1.5x0 + 6x1 − 2).
        let f = |x: &[f64]| {
            2.0 * x[0] * x[0] + 1.5 * x[0] * x[1] + 3.0 * x[1] * x[1] + x[0] - 2.0 * x[1]
        };
        let x0 = [0.4, -0.3];
        let mut model = QuadraticModel::initialize(x0.to_vec(), 0.25, 6, f);

        // For an exact quadratic the model is exact, so the regular and the
        // alternative gradient at x_opt both equal the analytic ∇F(x_opt).
        let xopt = model.best_point();
        let gx = [
            4.0 * xopt[0] + 1.5 * xopt[1] + 1.0,
            1.5 * xopt[0] + 6.0 * xopt[1] - 2.0,
        ];
        let gopt = model.gradient_at_opt();
        let galt = model.alt_gradient_at_opt();
        assert!(
            (gopt[0] - gx[0]).abs() < 1e-9 && (gopt[1] - gx[1]).abs() < 1e-9,
            "gradient_at_opt {gopt:?} vs analytic {gx:?}"
        );
        assert!(
            (galt[0] - gx[0]).abs() < 1e-9 && (galt[1] - gx[1]).abs() < 1e-9,
            "alt_gradient_at_opt {galt:?} vs analytic {gx:?}"
        );

        model.adopt_alt_model();

        // Interpolation conditions hold for the adopted model.
        let kopt = model.kopt();
        let q_opt = model.eval_change(model.xpt_row(kopt));
        let f_opt = model.fval[kopt];
        for j in 0..model.m() {
            let q_j = model.eval_change(model.xpt_row(j));
            let lhs = q_j - q_opt;
            let rhs = model.fval[j] - f_opt;
            assert!((lhs - rhs).abs() < 1e-9, "interp j={j}: {lhs} vs {rhs}");
        }
    }

    /// `apply_h` is private; this test-only shim exposes it.
    fn model_apply_h(
        model: &QuadraticModel<f64>,
        v_lambda: &[f64],
        v_g: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        model.apply_h(v_lambda, v_g)
    }

    /// Pick the non-`opt` interpolation index with the largest `|σ|` for the
    /// proposed update—a stand-in for the §7 MOVE rule, keeping the test's
    /// chosen `t` well away from a zero denominator.
    fn best_t(model: &QuadraticModel<f64>, ctx: &UpdateContext<f64>) -> usize {
        let mut best = None;
        let mut best_abs = -1.0;
        for t in 0..model.m() {
            if t == model.kopt() {
                continue;
            }
            let s = model.update_params(t, ctx).sigma.abs();
            if s > best_abs {
                best_abs = s;
                best = Some(t);
            }
        }
        best.expect("at least one candidate t")
    }

    /// T6: a model initialized on an exact quadratic is exact everywhere
    /// (n=1, m=3 captures the full 1-D quadratic). One `commit_update` to a new
    /// point must keep it exact: the residual `df` is zero, so `Γ`/`γ`/`∇Q` are
    /// unchanged, the new point is interpolated, and the KKT identity holds.
    #[test]
    fn update_preserves_exact_quadratic_n1() {
        let (a, b, c) = (0.5, 1.3, 2.0);
        let f = |x: &[f64]| a + b * x[0] + 0.5 * c * x[0] * x[0];
        let x0v = 0.4;
        let mut model = QuadraticModel::initialize(vec![x0v], 0.15, 3, f);

        let xnew = vec![0.9];
        let ctx = model.prepare_update(&xnew);
        let t = best_t(&model, &ctx);
        let scalars = model.update_params(t, &ctx);
        let f_new = f(&[x0v + xnew[0]]);
        model.commit_update(t, &ctx, &scalars, f_new);

        // df = 0 ⇒ model coefficients unchanged and γ still zero.
        assert!((model.gradient()[0] - (b + c * x0v)).abs() < 1e-9);
        assert!((model.gamma_explicit.get(0, 0) - c).abs() < 1e-9);
        assert!(model.gamma.iter().all(|&g| g.abs() < 1e-9));
        // Exactness preserved (includes the freshly inserted point at d = 0.9).
        for d in [0.3, -1.1, 0.9, 2.0] {
            let expected = f(&[x0v + d]) - f(&[x0v]);
            assert!((model.eval_change(&[d]) - expected).abs() < 1e-9, "d={d}");
        }
        assert_h_matches_inverse(&model, 1e-9);
    }

    /// T7: after one update with a generic displacement, the stored
    /// `Ω`/`Ξ`/`Υ` must still equal the blocks of `inv(W⁺)` built from the new
    /// interpolation geometry—validating eq. 4.11 (Ξ/Υ) and eq. 4.18 (the
    /// Ω-factorization normal branch) together (n=2, m=5).
    #[test]
    fn update_preserves_kkt_identity() {
        let f = |x: &[f64]| x[0] * x[0] + 2.0 * x[1] * x[1] + 0.5 * x[0] * x[1] + x[0] - x[1];
        let x0 = vec![0.3, -0.2];
        let mut model = QuadraticModel::initialize(x0.clone(), 0.2, 5, f);

        let xnew = vec![0.5, 0.4];
        let ctx = model.prepare_update(&xnew);
        let t = best_t(&model, &ctx);
        let scalars = model.update_params(t, &ctx);
        let xabs: Vec<f64> = x0.iter().zip(&xnew).map(|(a, b)| a + b).collect();
        model.commit_update(t, &ctx, &scalars, f(&xabs));

        assert_h_matches_inverse(&model, 1e-9);
    }

    /// A long, well-conditioned sequence of updates: the KKT identity must hold
    /// after *every* update, guarding against rounding-error accumulation in the
    /// `H`-factorization over many iterations (the normal-branch path that the
    /// future driver loop will hammer).
    #[test]
    fn long_run_preserves_kkt_identity() {
        let f = |x: &[f64]| {
            100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2) + 0.5 * x[2] * x[2]
        };
        let n = 3;
        let m = 7;
        let x0 = vec![-1.0, 1.0, 0.5];
        let mut model = QuadraticModel::initialize(x0.clone(), 0.3, m, f);

        // Deterministic LCG → pseudo-random steps in [−0.5, 0.5).
        let mut state: u64 = 0x1234_5678;
        let mut rand = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as f64) / (u32::MAX as f64) - 0.5
        };

        for _ in 0..120 {
            let xopt: Vec<f64> = (0..n).map(|k| model.xpt_row(model.kopt())[k]).collect();
            let xnew: Vec<f64> = (0..n).map(|k| xopt[k] + 0.25 * rand()).collect();

            let ctx = model.prepare_update(&xnew);
            let t = best_t(&model, &ctx);
            let scalars = model.update_params(t, &ctx);
            // MOVE keeps |σ| comfortably away from zero; skip otherwise.
            if scalars.sigma.abs() < 1e-10 {
                continue;
            }
            let xabs: Vec<f64> = x0.iter().zip(&xnew).map(|(a, b)| a + b).collect();
            model.commit_update(t, &ctx, &scalars, f(&xabs));

            assert_h_matches_inverse(&model, 1e-7);
        }
    }

    /// T9: the `Ω`-factorization cancellation branch (eqs. 4.19/4.20) is a
    /// rounding-recovery path—in exact arithmetic `α, β ≥ 0 ⇒ σ ≥ 0`, so it is
    /// only reached once rounding has driven a sign negative (Powell 2006, §4),
    /// which cannot be triggered deterministically by well-conditioned updates.
    /// Instead we validate the branch's *algebra* directly: it must realize the
    /// rank-2 update of `Ω` exactly, i.e.
    ///
    /// ```text
    /// s⁺_1 z⁺_1 z⁺_1ᵀ + s⁺_2 z⁺_2 z⁺_2ᵀ
    ///     = s_1 z_1 z_1ᵀ + s_2 z_2 z_2ᵀ
    ///       + σ⁻¹[ α·a aᵀ − β·b bᵀ + τ·(b aᵀ + a bᵀ) ] ,
    /// ```
    ///
    /// with `a = chop(e_t − H w)`, `b = Ω column t = s_1 z_{1,t} z_1 + s_2 z_{2,t} z_2`,
    /// `α = Ω_{tt}`, `σ = αβ + τ²`. This is a pure identity in the inputs, so we
    /// check it on a hand-built indefinite factorization (`s_1 = +1`, `s_2 = −1`)
    /// for both the `β ≥ 0` (eq. 4.19) and `β < 0` (eq. 4.20) sub-branches.
    #[test]
    fn cancellation_branch_satisfies_rank2_identity() {
        use crate::solver::powell::kkt::omega_from_factorization;
        use std::sync::atomic::Ordering;

        let n = 1;
        let m = 4;
        let rank = m - n - 1; // = 2
        let z1 = [1.0, 0.5, -0.3, 0.7];
        let z2 = [0.4, -0.6, 0.2, 0.9];
        let t = 0; // both columns have a nonzero t-th entry
        let chop = [0.2, -0.5, 0.8, -0.1];
        let tau = 0.37;

        for &beta in &[0.85_f64, -0.85] {
            let hits0 = CANCELLATION_HITS.load(Ordering::Relaxed);

            // Hand-built indefinite factorization: column 0 (s=+1), 1 (s=−1).
            let mut zdata = Vec::with_capacity(m * rank);
            for i in 0..m {
                zdata.push(z1[i]);
                zdata.push(z2[i]);
            }
            let mut model = QuadraticModel::from_parts(
                n,
                m,
                vec![0.0; n],
                DenseMatrix::from_fn(m, n, |_, _| 0.0),
                vec![0.0; m],
                0,
                vec![0.0; n],
                DenseMatrix::from_fn(n, n, |_, _| 0.0),
                vec![0.0; m],
                DenseMatrix::from_fn(n, m, |_, _| 0.0),
                DenseMatrix::from_fn(n, n, |_, _| 0.0),
                DenseMatrix::from_row_slice(m, rank, &zdata),
                vec![1.0, -1.0],
            );

            let zt1 = z1[t];
            let zt2 = z2[t];
            let alpha = zt1 * zt1 - zt2 * zt2; // s_1 z_{1,t}² + s_2 z_{2,t}²
            let sigma = alpha * beta + tau * tau;
            // b = Ω column t = s_1 z_{1,t} z_1 + s_2 z_{2,t} z_2.
            let b: Vec<f64> = (0..m).map(|i| zt1 * z1[i] - zt2 * z2[i]).collect();

            let old_omega = omega_from_factorization(&model);
            model.update_omega_factorization(t, &chop, tau, sigma, beta);
            let new_omega = omega_from_factorization(&model);

            assert!(
                CANCELLATION_HITS.load(Ordering::Relaxed) > hits0,
                "expected the cancellation branch to run (β={beta})"
            );

            let inv = 1.0 / sigma;
            for i in 0..m {
                for j in 0..m {
                    let delta = inv
                        * (alpha * chop[i] * chop[j] - beta * b[i] * b[j]
                            + tau * (b[i] * chop[j] + chop[i] * b[j]));
                    let want = old_omega.get(i, j) + delta;
                    let got = new_omega.get(i, j);
                    assert!(
                        (got - want).abs() < 1e-9,
                        "β={beta} Ω⁺[{i},{j}]: got {got} want {want} (Δ={:e})",
                        (got - want).abs()
                    );
                }
            }
        }
    }

    /// A non-quadratic objective: one update must still leave `H` consistent
    /// with `inv(W⁺)` (the model becomes only approximate, but the H-algebra is
    /// exact regardless of `F`).
    #[test]
    fn update_kkt_identity_nonquadratic() {
        let f = |x: &[f64]| (x[0] * x[0] + x[1]).exp() + x[1] * x[1] * x[1];
        let x0 = vec![0.1, 0.2];
        let mut model = QuadraticModel::initialize(x0.clone(), 0.15, 5, f);

        let xnew = vec![0.3, -0.25];
        let ctx = model.prepare_update(&xnew);
        let t = best_t(&model, &ctx);
        let scalars = model.update_params(t, &ctx);
        let xabs: Vec<f64> = x0.iter().zip(&xnew).map(|(a, b)| a + b).collect();
        model.commit_update(t, &ctx, &scalars, f(&xabs));

        assert_h_matches_inverse(&model, 1e-9);
    }
}
