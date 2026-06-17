//! Closed-form initialization of the [`QuadraticModel`] (Powell 2006, §3).
//!
//! Given the base point `x0`, the initial trust radius `ρ_beg`, the number of
//! interpolation points `m`, and the objective `F`, this builds the first
//! quadratic model and the factored inverse-KKT matrix `H` in `O(m²)`
//! operations using the convenient geometry of the coordinate-cross sample set.
//!
//! Only `m ≥ 2n+1` is implemented (the recommended case). The `n+2 ≤ m ≤ 2n`
//! partial-model regime (Powell 2006, §3, second paragraph) is deferred.

use crate::core::math::{DenseMatrix, Scalar};

use super::model::QuadraticModel;

impl<F: Scalar> QuadraticModel<F> {
    /// Build the initial model from the coordinate-offset sample set (Powell
    /// 2006, eqs. 3.2–3.5 for the geometry and the model's `Γ`/`∇Q`; eqs.
    /// 3.13–3.20 for the closed-form `Ξ`/`Υ`/`Ω`-factorization).
    ///
    /// `eval` evaluates the objective at an absolute point; it is called exactly
    /// once per interpolation point (`m` evaluations total) and may fail — the
    /// first `Err` aborts initialization and bubbles to the caller (the public
    /// [`Newuoa`](crate::solver::Newuoa) solver forwards the problem's typed
    /// cost error). Keeping it a closure lets initialization be unit-tested
    /// without a `Problem`/driver; the infallible [`initialize`](Self::initialize)
    /// wrapper serves the standalone driver and the tests.
    ///
    /// # Panics
    ///
    /// Panics unless `n ≥ 1` and `2n+1 ≤ m ≤ ½(n+1)(n+2)`.
    pub(crate) fn try_initialize<E>(
        x0: Vec<F>,
        rho_beg: F,
        m: usize,
        eval: &mut impl FnMut(&[F]) -> Result<F, E>,
    ) -> Result<Self, E> {
        let n = x0.len();
        assert!(n >= 1, "NEWUOA model needs n >= 1");
        assert!(
            m > 2 * n,
            "NEWUOA model init currently requires m >= 2n+1 (got m={m}, n={n}); \
             the n+2 <= m <= 2n partial-model case is not yet implemented"
        );
        assert!(
            m <= (n + 1) * (n + 2) / 2,
            "NEWUOA model requires m <= (n+1)(n+2)/2 (got m={m}, n={n})"
        );

        let two = F::from_f64(2.0).expect("2.0 representable");
        let half = F::from_f64(0.5).expect("0.5 representable");
        let rho = rho_beg;
        let rho2 = rho * rho;

        // --- 1. Interpolation points (displacements from x0) + objective. ---
        // (eq 3.2): point 0 = x0; points 1..=n = x0 + ρ e_k; n+1..=2n = x0 − ρ e_k.
        let mut xpt = DenseMatrix::from_fn(m, n, |_, _| F::zero());
        let mut fval = vec![F::zero(); m];

        fval[0] = eval(x0.as_slice())?;
        for k in 0..n {
            // +ρ e_k → point (k+1).
            xpt.set(k + 1, k, rho);
            let mut xp = x0.clone();
            xp[k] = xp[k] + rho;
            fval[k + 1] = eval(xp.as_slice())?;
            // −ρ e_k → point (k+1+n).
            xpt.set(k + 1 + n, k, -rho);
            let mut xm = x0.clone();
            xm[k] = xm[k] - rho;
            fval[k + 1 + n] = eval(xm.as_slice())?;
        }

        // (eq 3.4): σ_k = −1 if F(x0 − ρ e_k) < F(x0 + ρ e_k), else +1. Biases
        // the paired off-diagonal points towards smaller objective values.
        let mut sigma = vec![F::one(); n];
        for k in 0..n {
            if fval[k + 1 + n] < fval[k + 1] {
                sigma[k] = -F::one();
            }
        }

        // --- 2. Paired points for the off-diagonal Hessian (eq 3.3), m>2n+1. ---
        // For each extra point store (ip, p, q, p̂, q̂) (all 0-based), where p,q
        // are coordinates and p̂,q̂ the point indices of x0+σ_p ρ e_p and
        // x0+σ_q ρ e_q (eq 3.19) — reused by both the Γ off-diagonal and the
        // Ω-factorization column (eq 3.20).
        let mut extras: Vec<(usize, usize, usize, usize, usize)> = Vec::new();
        for ip in (2 * n + 1)..m {
            // Powell's p,q selection rule (§3), converted to 0-based point index.
            let jdiv = (ip - n - 1) / n; // ≥ 1 since ip ≥ 2n+1
            let p1 = ip - n - jdiv * n; // 1-based coordinate in [1, n]
            let mut q1 = p1 + jdiv;
            if q1 > n {
                q1 -= n;
            }
            let p_idx = p1 - 1;
            let q_idx = q1 - 1;
            // p̂: point index of x0 + σ_p ρ e_p (eq 3.19); +ρ ⇒ point p1, −ρ ⇒ p1+n.
            let phat = if sigma[p_idx] > F::zero() { p1 } else { p1 + n };
            let qhat = if sigma[q_idx] > F::zero() { q1 } else { q1 + n };

            xpt.set(ip, p_idx, sigma[p_idx] * rho);
            xpt.set(ip, q_idx, sigma[q_idx] * rho);
            let mut x = x0.clone();
            x[p_idx] = x[p_idx] + sigma[p_idx] * rho;
            x[q_idx] = x[q_idx] + sigma[q_idx] * rho;
            fval[ip] = eval(x.as_slice())?;

            extras.push((ip, p_idx, q_idx, phat, qhat));
        }

        // --- 3. Model gradient and Hessian (the explicit part Γ; γⱼ ≡ 0). ---
        let f0 = fval[0];
        let mut gq = vec![F::zero(); n];
        let mut gamma_explicit = DenseMatrix::from_fn(n, n, |_, _| F::zero());
        // Diagonal + gradient from the central differences of the coordinate
        // cross (the first 2n+1 points determine these uniquely, §3).
        for k in 0..n {
            let fp = fval[k + 1];
            let fm = fval[k + 1 + n];
            gq[k] = (fp - fm) / (two * rho);
            gamma_explicit.set(k, k, (fp + fm - two * f0) / rho2);
        }
        // Off-diagonals from the paired points (eq 3.5):
        // (∇²Q)_{pq} = σ_p σ_q ρ⁻² (F(x0) − F(p̂) − F(q̂) + F(paired)).
        for &(ip, p_idx, q_idx, phat, qhat) in &extras {
            let val =
                sigma[p_idx] * sigma[q_idx] * (f0 - fval[phat] - fval[qhat] + fval[ip]) / rho2;
            gamma_explicit.set(p_idx, q_idx, val);
            gamma_explicit.set(q_idx, p_idx, val);
        }
        let gamma = vec![F::zero(); m];

        // --- 4. Factored inverse-KKT H (suppressed (m+1)-th row/col). ---
        let npt_minus = m - n - 1;
        let mut bmat_xi = DenseMatrix::from_fn(n, m, |_, _| F::zero());
        // Υ is identically zero for m ≥ 2n+1 (eq 3.16).
        let bmat_ups = DenseMatrix::from_fn(n, n, |_, _| F::zero());
        let mut zmat = DenseMatrix::from_fn(m, npt_minus, |_, _| F::zero());
        let zsign = vec![F::one(); npt_minus]; // all +1 initially (eq 3.17)

        // Ξ rows 2..n+1 of the full (n+1)×m matrix become rows 0..n of the
        // suppressed bmat_xi (eq 3.14): Ξ_{r, r+1} = 1/(2ρ), Ξ_{r, r+1+n} = −1/(2ρ).
        let inv_2rho = F::one() / (two * rho);
        for r in 0..n {
            bmat_xi.set(r, r + 1, inv_2rho);
            bmat_xi.set(r, r + 1 + n, -inv_2rho);
        }

        // Ω-factorization diagonal columns (eq 3.18), columns 0..n−1:
        // Z_{0,k} = −√2 ρ⁻², Z_{k+1,k} = Z_{k+1+n,k} = ½√2 ρ⁻².
        let inv_rho2 = F::one() / rho2;
        let sqrt2_over_rho2 = two.sqrt() * inv_rho2;
        let half_sqrt2_over_rho2 = half * sqrt2_over_rho2;
        for kk in 0..n {
            zmat.set(0, kk, -sqrt2_over_rho2);
            zmat.set(kk + 1, kk, half_sqrt2_over_rho2);
            zmat.set(kk + 1 + n, kk, half_sqrt2_over_rho2);
        }
        // Off-diagonal columns (eq 3.20), one per extra point, column kk = ip−n−1:
        // Z_{0,k} = ρ⁻², Z_{p̂,k} = Z_{q̂,k} = −ρ⁻², Z_{ip,k} = ρ⁻².
        for &(ip, _p_idx, _q_idx, phat, qhat) in &extras {
            let kk = ip - n - 1;
            zmat.set(0, kk, inv_rho2);
            zmat.set(phat, kk, -inv_rho2);
            zmat.set(qhat, kk, -inv_rho2);
            zmat.set(ip, kk, inv_rho2);
        }

        // --- 5. Best interpolation point (eq "opt"). ---
        let mut kopt = 0;
        for j in 1..m {
            if fval[j] < fval[kopt] {
                kopt = j;
            }
        }

        Ok(QuadraticModel {
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
        })
    }

    /// Infallible convenience wrapper around [`try_initialize`](Self::try_initialize)
    /// for an objective that cannot fail (the standalone driver and the tests).
    pub(crate) fn initialize(x0: Vec<F>, rho_beg: F, m: usize, f: impl Fn(&[F]) -> F) -> Self {
        Self::try_initialize::<core::convert::Infallible>(x0, rho_beg, m, &mut |x| Ok(f(x)))
            .expect("infallible objective")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::newuoa::kkt::assert_h_matches_inverse;

    /// T1: n=1, m=3 (=2n+1). An exact 1-D quadratic `F = a + b·x + ½c·x²` must
    /// be recovered: `∇Q(x0) = b + c·x0`, `Γ = [c]`, `γ ≡ 0`, and
    /// `eval_change(d)` reproduces `F(x0+d) − F(x0)` exactly.
    #[test]
    fn n1_recovers_exact_quadratic() {
        let (a, b, c) = (1.5, -0.75, 2.25);
        let x0v = 0.5;
        let f = |x: &[f64]| a + b * x[0] + 0.5 * c * x[0] * x[0];
        let model = QuadraticModel::initialize(vec![x0v], 0.1, 3, f);

        assert!((model.gradient()[0] - (b + c * x0v)).abs() < 1e-12);
        assert!((model.gamma_explicit.get(0, 0) - c).abs() < 1e-12);
        assert!(model.gamma.iter().all(|&g| g == 0.0));

        for d in [0.3, -1.7, 0.0, 2.0] {
            let expected = f(&[x0v + d]) - f(&[x0v]);
            assert!(
                (model.eval_change(&[d]) - expected).abs() < 1e-12,
                "eval_change({d})"
            );
        }
    }

    /// T2: n=2, m=5 (=2n+1). A separable quadratic must give `Γ = diag(c1,c2)`,
    /// zero off-diagonals, `γ ≡ 0`, and `∇Q(x0) = b + diag(c)·x0`.
    #[test]
    fn n2_separable_quadratic_is_diagonal() {
        let (c1, c2) = (3.0, 5.0);
        let b = [1.0, -2.0];
        let x0v = [0.25, -0.5];
        let f = |x: &[f64]| 0.5 * (c1 * x[0] * x[0] + c2 * x[1] * x[1]) + b[0] * x[0] + b[1] * x[1];
        let model = QuadraticModel::initialize(x0v.to_vec(), 0.2, 5, f);

        assert!((model.gamma_explicit.get(0, 0) - c1).abs() < 1e-12);
        assert!((model.gamma_explicit.get(1, 1) - c2).abs() < 1e-12);
        assert!(model.gamma_explicit.get(0, 1).abs() < 1e-12);
        assert!(model.gamma_explicit.get(1, 0).abs() < 1e-12);
        assert!(model.gamma.iter().all(|&g| g == 0.0));
        assert!((model.gradient()[0] - (b[0] + c1 * x0v[0])).abs() < 1e-12);
        assert!((model.gradient()[1] - (b[1] + c2 * x0v[1])).abs() < 1e-12);
    }

    /// T3: n=2, m=6 (>2n+1). The single off-diagonal pair must recover the cross
    /// term `G12` via the paired point (eq 3.5), with the σ sign rule exercised:
    /// `x0 = 0`, `b = (1, 2)` makes the gradient positive in both coordinates,
    /// so `σ_1 = σ_2 = −1` (the `−ρ e_k` points are lower).
    #[test]
    fn n2_recovers_off_diagonal_cross_term() {
        let g = [[4.0, 1.5], [1.5, 3.0]];
        let b = [1.0, 2.0];
        let f = |x: &[f64]| {
            0.5 * (g[0][0] * x[0] * x[0] + 2.0 * g[0][1] * x[0] * x[1] + g[1][1] * x[1] * x[1])
                + b[0] * x[0]
                + b[1] * x[1]
        };
        let model = QuadraticModel::initialize(vec![0.0, 0.0], 0.3, 6, f);

        assert!(
            (model.gamma_explicit.get(0, 1) - g[0][1]).abs() < 1e-12,
            "off-diag = {}",
            model.gamma_explicit.get(0, 1)
        );
        assert!((model.gamma_explicit.get(1, 0) - g[0][1]).abs() < 1e-12);
        // Diagonals still exact.
        assert!((model.gamma_explicit.get(0, 0) - g[0][0]).abs() < 1e-12);
        assert!((model.gamma_explicit.get(1, 1) - g[1][1]).abs() < 1e-12);
    }

    /// T4: the master KKT identity — the stored `Ω`/`Ξ`/`Υ` blocks must equal
    /// the corresponding blocks of `inv(W)` after INIT, validating eqs 3.13–3.20
    /// across `m = 2n+1` (n=1, n=2) and `m > 2n+1` (n=2).
    #[test]
    fn init_satisfies_kkt_identity() {
        // n=1, m=3.
        let m1 = QuadraticModel::initialize(vec![0.5], 0.1, 3, |x: &[f64]| {
            1.0 + 0.3 * x[0] + 0.5 * x[0] * x[0]
        });
        assert_h_matches_inverse(&m1, 1e-10);

        // n=2, m=5.
        let m2 = QuadraticModel::initialize(vec![0.25, -0.5], 0.2, 5, |x: &[f64]| {
            x[0] * x[0] + 2.0 * x[1] * x[1] + x[0] - x[1]
        });
        assert_h_matches_inverse(&m2, 1e-10);

        // n=2, m=6 (one off-diagonal pair).
        let m3 = QuadraticModel::initialize(vec![0.0, 0.0], 0.3, 6, |x: &[f64]| {
            2.0 * x[0] * x[0] + 1.5 * x[0] * x[1] + 1.5 * x[1] * x[1] + x[0] + 2.0 * x[1]
        });
        assert_h_matches_inverse(&m3, 1e-10);
    }

    /// T5: the model must interpolate `F` at every interpolation point. With no
    /// stored constant, this is `eval_change(xⱼ − x0) = F(xⱼ) − F(x0)`.
    #[test]
    fn init_interpolates_at_all_points() {
        let f = |x: &[f64]| {
            2.0 * x[0] * x[0] + 1.5 * x[0] * x[1] + 3.0 * x[1] * x[1] + x[0] - 2.0 * x[1] + 0.7
        };
        let x0v = vec![0.4, -0.3];
        let model = QuadraticModel::initialize(x0v.clone(), 0.25, 6, f);
        let f0 = f(x0v.as_slice());
        for j in 0..model.m() {
            let disp = model.xpt_row(j).to_vec();
            let xj: Vec<f64> = x0v.iter().zip(&disp).map(|(a, b)| a + b).collect();
            let expected = f(xj.as_slice()) - f0;
            assert!(
                (model.eval_change(&disp) - expected).abs() < 1e-10,
                "interpolation at point {j}: got {}, want {expected}",
                model.eval_change(&disp)
            );
        }
    }
}
