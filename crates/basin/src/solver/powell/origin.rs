//! Origin shift (Powell 2006, §7, eqs. 7.10–7.14; derivation Powell 2004a §5).
//!
//! As the iterates move away from the base point `x0`, the first `m` components
//! of the update vector `w` (eq. 4.10) grow like `½‖x_opt−x0‖⁴`, so the `β`
//! formula (7.8) suffers catastrophic cancellation: an error `ε` in `H₁₁` is
//! amplified by `‖x_opt−x0‖⁶`. NEWUOA therefore re-centres `x0` on the current
//! best point `x_opt` whenever the step is small relative to that drift
//! (eq. 7.10), re-expressing the model and the factored `H` in the new
//! coordinates **without changing the quadratic `Q`**.
//!
//! The transform (eqs. 7.11–7.14):
//! - `Ω` (and its factorization) is untouched;
//! - `Ξ_red += Y Ω` and `Υ += Y Ξ_redᵀ + Ξ_red Yᵀ + Y Ω Yᵀ` (the (7.12)
//!   congruence), with `Y` the `n × m` matrix of columns
//!   `y_j = {sᵀ(x_j − x_av)}(x_j − x_av) + ¼‖s‖² s` (eq. 7.11);
//! - `∇Q(x0) += ∇²Q s` (eq. 7.13);
//! - `Γ += v sᵀ + s vᵀ` with `v = Σ_j γ_j (x_j − x_av)` (eq. 7.14, `γ` unchanged);
//! - every displacement `x_j − x0` loses `s`, and `x0` gains `s`,
//!
//! where `s = x_opt − x0` and `x_av = ½(x0 + x_opt)`. The work is `O(m²n)`,
//! dominated by the `Y Ω` / `Y Ω Yᵀ` products, so the driver only shifts on the
//! small fraction of iterations that trip eq. 7.10.

use crate::core::math::Scalar;
use crate::solver::powell::model::QuadraticModel;

impl<F: Scalar> QuadraticModel<F> {
    /// Re-centre `x0` on the current best point `x_opt` (Powell 2006, §7). The
    /// quadratic `Q` and the interpolation values `F(x_j)` are unchanged; only
    /// the coordinate representation (`x0`, the displacements `x_j − x0`, the
    /// gradient `∇Q(x0)`, `Γ`, and the `Ξ`/`Υ` blocks of `H`) is revised.
    ///
    /// A no-op when `x_opt = x0` (the best point is already the base point).
    pub(crate) fn shift_origin(&mut self) {
        let n = self.n;
        let m = self.m;
        let rank = m - n - 1;
        let half = F::from_f64(0.5).expect("0.5 representable");
        let quarter = F::from_f64(0.25).expect("0.25 representable");

        // s = x_opt − x0 (the displacement of the current best point).
        let s = self.xpt.row(self.kopt).to_vec();
        let s_sq = dot(&s, &s);
        if s_sq <= F::zero() {
            return; // x_opt already at x0.
        }

        // Gradient (eq. 7.13): ∇Q(x0_new) = ∇Q(x0) + ∇²Q s. Computed on the OLD
        // representation; ∇²Q is x0-invariant, so this must precede the Γ update.
        let hs = self.hessian_matvec(&s);
        for i in 0..n {
            self.gq[i] = self.gq[i] + hs[i];
        }

        // Γ update (eq. 7.14): Γ += v sᵀ + s vᵀ with v = Σ_j γ_j (x_j − x_av),
        // x_av = ½(x0 + x_opt) ⇒ x_j − x_av = (x_j − x0) − ½ s. γ is unchanged.
        let mut v = vec![F::zero(); n];
        for j in 0..m {
            let gj = self.gamma[j];
            let row = self.xpt.row(j);
            for i in 0..n {
                v[i] = v[i] + gj * (row[i] - half * s[i]);
            }
        }
        for a in 0..n {
            for b in 0..n {
                let add = v[a] * s[b] + s[a] * v[b];
                self.gamma_explicit
                    .set(a, b, self.gamma_explicit.get(a, b) + add);
            }
        }

        // Y (n × m): columns y_j = {sᵀ(x_j − x_av)}(x_j − x_av) + ¼‖s‖² s (7.11).
        let mut y: Vec<Vec<F>> = Vec::with_capacity(m);
        for j in 0..m {
            let row = self.xpt.row(j);
            let wj: Vec<F> = (0..n).map(|i| row[i] - half * s[i]).collect();
            let s_dot_w = dot(&s, &wj);
            let yj: Vec<F> = (0..n)
                .map(|i| s_dot_w * wj[i] + quarter * s_sq * s[i])
                .collect();
            y.push(yj);
        }

        // Yz_k = Y z_k (n-vector) for each Ω-factorization column k.
        let mut yz: Vec<Vec<F>> = Vec::with_capacity(rank);
        for k in 0..rank {
            let mut col = vec![F::zero(); n];
            for j in 0..m {
                let zjk = self.zmat.get(j, k);
                for i in 0..n {
                    col[i] = col[i] + y[j][i] * zjk;
                }
            }
            yz.push(col);
        }

        // Υ += Y Ξ_redᵀ + Ξ_red Yᵀ + Y Ω Yᵀ, using the OLD Ξ_red (`bmat_xi`):
        //   (Y Ξ_redᵀ)[a,b] = Σ_i y_i[a] Ξ_red[b,i],
        //   (Y Ω Yᵀ)[a,b]   = Σ_k s_k Yz_k[a] Yz_k[b].
        let mut ups_add = vec![vec![F::zero(); n]; n];
        for a in 0..n {
            for b in 0..n {
                let mut yxt = F::zero(); // (Y Ξ_redᵀ)[a,b]
                let mut xyt = F::zero(); // (Ξ_red Yᵀ)[a,b] = (Y Ξ_redᵀ)[b,a]
                for i in 0..m {
                    yxt = yxt + y[i][a] * self.bmat_xi.get(b, i);
                    xyt = xyt + y[i][b] * self.bmat_xi.get(a, i);
                }
                let mut yomyt = F::zero();
                for k in 0..rank {
                    yomyt = yomyt + self.zsign[k] * yz[k][a] * yz[k][b];
                }
                ups_add[a][b] = yxt + xyt + yomyt;
            }
        }
        for a in 0..n {
            for b in 0..n {
                self.bmat_ups
                    .set(a, b, self.bmat_ups.get(a, b) + ups_add[a][b]);
            }
        }

        // Ξ_red += Y Ω = Σ_k s_k Yz_k z_kᵀ ⇒ (Y Ω)[a,i] = Σ_k s_k Yz_k[a] z_k[i].
        for a in 0..n {
            for i in 0..m {
                let mut yom = F::zero();
                for k in 0..rank {
                    yom = yom + self.zsign[k] * yz[k][a] * self.zmat.get(i, k);
                }
                self.bmat_xi.set(a, i, self.bmat_xi.get(a, i) + yom);
            }
        }

        // Displacements: x_j − x0_new = (x_j − x0) − s; then x0_new = x0 + s.
        for j in 0..m {
            for i in 0..n {
                let val = self.xpt.get(j, i) - s[i];
                self.xpt.set(j, i, val);
            }
        }
        for i in 0..n {
            self.x0[i] = self.x0[i] + s[i];
        }
    }
}

/// Plain dot product of two equal-length slices.
fn dot<F: Scalar>(a: &[F], b: &[F]) -> F {
    a.iter().zip(b).map(|(x, y)| *x * *y).sum()
}

#[cfg(test)]
mod tests {
    use crate::solver::powell::kkt::assert_h_matches_inverse;
    use crate::solver::powell::model::QuadraticModel;

    /// A quadratic whose minimizer sits well away from the sampling origin, so
    /// the initial best interpolation point `x_opt` differs from `x0`.
    fn offset_quadratic(x: &[f64]) -> f64 {
        (x[0] - 2.0).powi(2) + 3.0 * (x[1] + 1.5).powi(2) + 0.5 * x[0] * x[1]
    }

    /// After a shift, the stored `Ξ`/`Υ`/`Ω` blocks must still equal `inv(W)`
    /// for the `W` built from the *new* geometry (the master KKT identity).
    #[test]
    fn shift_preserves_kkt_identity() {
        let mut model = QuadraticModel::initialize(vec![0.0, 0.0], 0.5, 5, &offset_quadratic);
        // The best initial sample is off-centre, so s = x_opt − x0 ≠ 0.
        assert_ne!(model.kopt(), usize::MAX);
        assert!(model.xpt_row(model.kopt()).iter().any(|v| v.abs() > 0.0));
        model.shift_origin();
        assert_h_matches_inverse(&model, 1e-9);
        // x_opt now coincides with x0 (its displacement is zero).
        assert!(model.xpt_row(model.kopt()).iter().all(|v| v.abs() < 1e-12));
    }

    /// The shift is a pure change of representation: `Q` is unchanged, so the
    /// model still interpolates `F` at every point, and differences of the model
    /// value between any two absolute points are invariant.
    #[test]
    fn shift_preserves_the_quadratic() {
        let mut model = QuadraticModel::initialize(vec![0.0, 0.0], 0.5, 5, &offset_quadratic);
        let m = model.m();
        let kopt = model.kopt();

        // Q-differences at a few absolute probe points, measured before the shift.
        let probes = [[0.3, -0.4], [-0.2, 0.1], [0.5, 0.5]];
        let x0_old = model.x0().to_vec();
        let before: Vec<f64> = probes
            .iter()
            .map(|p| {
                let d: Vec<f64> = (0..2).map(|i| p[i] - x0_old[i]).collect();
                model.eval_change(&d)
            })
            .collect();
        // Interpolation residuals Q(x_j) − Q(x_opt) − (F(x_j) − F(x_opt)).
        let interp_before: Vec<f64> = (0..m)
            .map(|j| {
                model.eval_change(model.xpt_row(j).to_vec().as_slice())
                    - model.eval_change(model.xpt_row(kopt).to_vec().as_slice())
                    - (model.fval(j) - model.fopt())
            })
            .collect();

        model.shift_origin();

        let x0_new = model.x0().to_vec();
        for (k, p) in probes.iter().enumerate() {
            let d: Vec<f64> = (0..2).map(|i| p[i] - x0_new[i]).collect();
            // Q(p) − Q(x0) shifts by the constant Q(x0_new) − Q(x0_old); compare
            // a *difference* of probes, which cancels that constant.
            let after = model.eval_change(&d);
            let q0 = {
                let d0: Vec<f64> = (0..2).map(|i| probes[0][i] - x0_new[i]).collect();
                model.eval_change(&d0)
            };
            let want = before[k] - before[0];
            assert!(
                (after - q0 - want).abs() < 1e-9,
                "probe {k}: Δ {} vs {want}",
                after - q0
            );
        }
        // Interpolation is preserved (residuals stay ~0).
        for j in 0..m {
            let res = model.eval_change(model.xpt_row(j).to_vec().as_slice())
                - model.eval_change(model.xpt_row(kopt).to_vec().as_slice())
                - (model.fval(j) - model.fopt());
            assert!(
                (res - interp_before[j]).abs() < 1e-9 && res.abs() < 1e-8,
                "interp residual {j}: {res}"
            );
        }
    }

    /// Stress: after many model updates have moved `x_opt` far from `x0`, a shift
    /// still restores the KKT identity (the `O(m²n)` congruence is exact).
    #[test]
    fn shift_after_updates_preserves_kkt() {
        // n = 3, m = 7; drive a handful of updates with a non-quadratic F so the
        // implicit part γ is populated, then shift.
        let f = |x: &[f64]| {
            (1.0 - x[0]).powi(2)
                + 100.0 * (x[1] - x[0] * x[0]).powi(2)
                + (x[2] - 0.5).powi(2) * (x[2] - 0.5).powi(2)
        };
        let mut model = QuadraticModel::initialize(vec![-1.0, 1.0, 0.0], 0.4, 7, &f);
        // A few accepted "steps": move the best point around to grow ‖x_opt−x0‖.
        for step in 0..6 {
            let kopt = model.kopt();
            let xopt = model.xpt_row(kopt).to_vec();
            let d = [0.15 * (step as f64 + 1.0) * 0.3, -0.1, 0.05];
            let xnew_disp: Vec<f64> = (0..3).map(|i| xopt[i] + d[i]).collect();
            let xabs: Vec<f64> = (0..3).map(|i| model.x0()[i] + xnew_disp[i]).collect();
            let f_new = f(&xabs);
            let ctx = model.prepare_update(&xnew_disp);
            // Drop the point furthest from the new candidate (any valid t).
            let t = (0..model.m())
                .filter(|j| *j != kopt)
                .max_by(|a, b| {
                    let da = model.update_params(*a, &ctx).sigma.abs();
                    let db = model.update_params(*b, &ctx).sigma.abs();
                    da.partial_cmp(&db).unwrap()
                })
                .unwrap();
            let sc = model.update_params(t, &ctx);
            if sc.sigma != 0.0 {
                model.commit_update(t, &ctx, &sc, f_new);
            }
        }
        assert_h_matches_inverse(&model, 1e-7);
        model.shift_origin();
        assert_h_matches_inverse(&model, 1e-7);
    }
}
