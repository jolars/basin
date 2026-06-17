//! The [`QuadraticModel`] struct and its read surface (model evaluation, the
//! implicit-Hessian matvec, accessors). Initialization lives in
//! [`super::init`]; the least-Frobenius-norm update in [`super::update`].

use crate::core::math::{DenseMatrix, Scalar};

/// NEWUOA's interpolation-based quadratic model together with the factored
/// inverse-KKT matrix `H = W⁻¹` that maintains it (Powell 2006, §§3–4; the
/// H-update derivation is Powell 2004a).
///
/// The model is the quadratic
///
/// ```text
/// Q(x0 + d) = Q(x0) + dᵀ ∇Q(x0) + ½ dᵀ ∇²Q d ,
/// ```
///
/// whose Hessian is stored *implicitly* (Powell 2006, eq. 4.28) as
///
/// ```text
/// ∇²Q = Γ + Σⱼ γⱼ (xⱼ − x0)(xⱼ − x0)ᵀ ,
/// ```
///
/// i.e. an explicit symmetric part `Γ` plus a sum of rank-one terms keyed to
/// the interpolation points. The constant term `Q(x0)` is **deliberately not
/// stored** (Powell 2006, eq. 4.23 — the model works in *changes* to `F` and
/// `x`), so [`eval_change`](Self::eval_change) returns `Q(x0 + d) − Q(x0)`.
///
/// `H` has block form `[[Ω, Ξᵀ], [Ξ, Υ]]` of order `m + n + 1`. Its `(m+1)`-th
/// row and column are suppressed (Powell 2006, §4), leaving `Ξ` of shape `n × m`
/// and `Υ` of shape `n × n`; `Ω` is held as the factorization
/// `Ω = Σ_{k=1}^{m−n−1} sₖ zₖ zₖᵀ` (columns `zₖ` are the columns of `Z`, signs
/// `sₖ ∈ {±1}`), which keeps the leading `m × m` block positive semidefinite
/// in the face of rounding error.
///
/// This is solver-internal scratch, not a `State`; see the module docs for why
/// it is bounded on `F: Scalar` and stores its matrices as
/// [`DenseMatrix`]/`Vec<F>` rather than the generic backend matrix.
pub(crate) struct QuadraticModel<F = f64> {
    /// Number of variables.
    pub(crate) n: usize,
    /// Number of interpolation points (npt), in `[n + 2, ½(n+1)(n+2)]`.
    pub(crate) m: usize,

    /// Base point `x0`, length `n`.
    pub(crate) x0: Vec<F>,
    /// Interpolation-point displacements: row `j` is `xⱼ − x0` (`m × n`).
    pub(crate) xpt: DenseMatrix<F>,
    /// Objective values `F(xⱼ)` at the interpolation points, length `m`.
    /// (NEWUOA's `FVAL`.) Maintained so the update can form the model residual
    /// `df = F(x⁺) − Q_old(x⁺)` (Powell 2006, eq. 4.23) and track `kopt`.
    pub(crate) fval: Vec<F>,
    /// Index in `[0, m)` of the current best interpolation point (the "opt" of
    /// Powell 2006, eqs. 4.24–4.26).
    pub(crate) kopt: usize,

    /// Model gradient `∇Q(x0)`, length `n`.
    pub(crate) gq: Vec<F>,
    /// Explicit symmetric part `Γ` of `∇²Q` (`n × n`).
    pub(crate) gamma_explicit: DenseMatrix<F>,
    /// Implicit Hessian coefficients `γⱼ`, length `m`.
    pub(crate) gamma: Vec<F>,

    /// `Ξ` block of `H` after suppression of its first row (`n × m`).
    pub(crate) bmat_xi: DenseMatrix<F>,
    /// `Υ` block of `H` after suppression of its first row and column (`n × n`).
    pub(crate) bmat_ups: DenseMatrix<F>,
    /// Columns `zₖ` of the `Ω`-factorization (`m × (m−n−1)`).
    pub(crate) zmat: DenseMatrix<F>,
    /// Signs `sₖ ∈ {±1}` of the `Ω`-factorization, length `m − n − 1`. All `+1`
    /// initially (Powell 2006, eq. 3.17).
    pub(crate) zsign: Vec<F>,
}

impl<F: Scalar> QuadraticModel<F> {
    /// Number of variables `n`.
    pub(crate) fn n(&self) -> usize {
        self.n
    }

    /// Number of interpolation points `m`.
    pub(crate) fn m(&self) -> usize {
        self.m
    }

    /// Index of the current best interpolation point.
    pub(crate) fn kopt(&self) -> usize {
        self.kopt
    }

    /// The displacement `xⱼ − x0` of interpolation point `j`, length `n`.
    pub(crate) fn xpt_row(&self, j: usize) -> &[F] {
        self.xpt.row(j)
    }

    /// The base point `x0`, length `n`.
    pub(crate) fn x0(&self) -> &[F] {
        &self.x0
    }

    /// The objective value `F(xⱼ)` at interpolation point `j`.
    pub(crate) fn fval(&self, j: usize) -> F {
        self.fval[j]
    }

    /// The objective value at the current best interpolation point,
    /// `F(x_opt) = F(x_{kopt})`. Equals `Q(x_opt)` since the model interpolates.
    pub(crate) fn fopt(&self) -> F {
        self.fval[self.kopt]
    }

    /// The current best point in absolute coordinates, `x0 + (x_{kopt} − x0)`,
    /// length `n`.
    pub(crate) fn best_point(&self) -> Vec<F> {
        let row = self.xpt.row(self.kopt);
        (0..self.n).map(|i| self.x0[i] + row[i]).collect()
    }

    /// The model gradient `∇Q(x0)`, length `n`.
    pub(crate) fn gradient(&self) -> &[F] {
        &self.gq
    }

    /// Apply the implicit Hessian: `∇²Q · v = Γ v + Σⱼ γⱼ (xⱼ − x0)((xⱼ − x0)·v)`.
    ///
    /// Computed without ever forming `∇²Q` (Powell 2006, eq. 4.28); this is the
    /// matvec the trust-region subproblem (TRSAPP) will call.
    pub(crate) fn hessian_matvec(&self, v: &[F]) -> Vec<F> {
        assert_eq!(v.len(), self.n, "hessian_matvec: v must have length n");
        // y = Γ v (Γ is symmetric n×n).
        let mut y = vec![F::zero(); self.n];
        for i in 0..self.n {
            let mut acc = F::zero();
            for j in 0..self.n {
                acc = acc + self.gamma_explicit.get(i, j) * v[j];
            }
            y[i] = acc;
        }
        // y += Σ_k γ_k (x_k − x0) · ((x_k − x0)·v).
        for k in 0..self.m {
            let row = self.xpt.row(k);
            let dot = dot(row, v);
            let coef = self.gamma[k] * dot;
            for i in 0..self.n {
                y[i] = y[i] + coef * row[i];
            }
        }
        y
    }

    /// Apply the implicit Hessian of a *Lagrange function* `ℓ_t`:
    /// `∇²ℓ_t · u = Σ_k λ_k (x_k − x0)((x_k − x0)·u)` (Powell 2006, eq. 6.15).
    ///
    /// Unlike `Q`, a Lagrange function has **no explicit `Γ` part** — its Hessian
    /// is purely the rank-one sum keyed by its implicit coefficients `λ`, which
    /// are the `Ω`-column of `H` returned by
    /// [`lagrange_coeffs`](Self::lagrange_coeffs). BIGLAG (Powell 2006, §6) uses
    /// this for the gradient recurrence (eq. 6.14).
    pub(crate) fn lagrange_hessian_matvec(&self, lambda: &[F], u: &[F]) -> Vec<F> {
        assert_eq!(
            lambda.len(),
            self.m,
            "lagrange_hessian_matvec: |λ| must be m"
        );
        assert_eq!(u.len(), self.n, "lagrange_hessian_matvec: |u| must be n");
        let mut y = vec![F::zero(); self.n];
        for k in 0..self.m {
            let row = self.xpt.row(k);
            let coef = lambda[k] * dot(row, u);
            for i in 0..self.n {
                y[i] = y[i] + coef * row[i];
            }
        }
        y
    }

    /// The model *change* `Q(x0 + d) − Q(x0) = dᵀ ∇Q(x0) + ½ dᵀ ∇²Q d`.
    ///
    /// The constant term is not stored, so this is the only model value the
    /// solver needs (predicted reductions, interpolation residuals).
    pub(crate) fn eval_change(&self, d: &[F]) -> F {
        assert_eq!(d.len(), self.n, "eval_change: d must have length n");
        let hd = self.hessian_matvec(d);
        let lin = dot(d, &self.gq);
        let quad = dot(d, &hd);
        let half = F::from_f64(0.5).expect("0.5 representable");
        lin + half * quad
    }
}

/// Plain dot product of two equal-length slices.
fn dot<F: Scalar>(a: &[F], b: &[F]) -> F {
    a.iter().zip(b).map(|(x, y)| *x * *y).sum()
}

#[cfg(test)]
impl<F: Scalar> QuadraticModel<F> {
    /// Test-only raw constructor: assemble a model directly from its parts,
    /// bypassing [`super::init`]. Lets the read surface (`eval_change`,
    /// `hessian_matvec`) be tested against hand-built `Γ`/`γ`/`∇Q` before the
    /// §3 initialization exists.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        n: usize,
        m: usize,
        x0: Vec<F>,
        xpt: DenseMatrix<F>,
        fval: Vec<F>,
        kopt: usize,
        gq: Vec<F>,
        gamma_explicit: DenseMatrix<F>,
        gamma: Vec<F>,
        bmat_xi: DenseMatrix<F>,
        bmat_ups: DenseMatrix<F>,
        zmat: DenseMatrix<F>,
        zsign: Vec<F>,
    ) -> Self {
        Self {
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
        }
    }

    /// Assemble the dense `∇²Q = Γ + Σⱼ γⱼ (xⱼ−x0)(xⱼ−x0)ᵀ` directly, for
    /// cross-checking [`hessian_matvec`](Self::hessian_matvec).
    pub(crate) fn dense_hessian(&self) -> DenseMatrix<F> {
        let mut h = self.gamma_explicit.clone();
        for k in 0..self.m {
            let row = self.xpt.row(k).to_vec();
            for i in 0..self.n {
                for j in 0..self.n {
                    let add = self.gamma[k] * row[i] * row[j];
                    h.set(i, j, h.get(i, j) + add);
                }
            }
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a model on `n` variables, `m` interpolation points, with the given
    /// `Γ`, `γ`, `∇Q` and interpolation displacements. The `H`-factorization
    /// blocks are filled with zeros — irrelevant to the read surface, which
    /// only touches `gq`, `gamma_explicit`, `gamma`, and `xpt`.
    fn model_with(
        n: usize,
        m: usize,
        gq: Vec<f64>,
        gamma_explicit: DenseMatrix,
        gamma: Vec<f64>,
        xpt_rows: &[&[f64]],
    ) -> QuadraticModel {
        let mut data = Vec::with_capacity(m * n);
        for row in xpt_rows {
            assert_eq!(row.len(), n);
            data.extend_from_slice(row);
        }
        let xpt = DenseMatrix::from_row_slice(m, n, &data);
        let npt_minus = m - n - 1;
        QuadraticModel::from_parts(
            n,
            m,
            vec![0.0; n],
            xpt,
            vec![0.0; m],
            0,
            gq,
            gamma_explicit,
            gamma,
            DenseMatrix::from_fn(n, m, |_, _| 0.0),
            DenseMatrix::from_fn(n, n, |_, _| 0.0),
            DenseMatrix::from_fn(m, npt_minus, |_, _| 0.0),
            vec![1.0; npt_minus],
        )
    }

    /// `hessian_matvec(eᵢ)` must reproduce, column by column, the dense
    /// `∇²Q = Γ + Σⱼ γⱼ (xⱼ−x0)(xⱼ−x0)ᵀ` assembled directly (T10).
    #[test]
    fn hessian_matvec_matches_dense_assembly() {
        let n = 2;
        let m = 5;
        // A non-trivial explicit part and two active implicit rank-one terms.
        let gamma_explicit = DenseMatrix::from_row_slice(2, 2, &[3.0, 1.0, 1.0, 4.0]);
        let gamma = vec![0.5, 0.0, -0.25, 0.0, 0.0];
        let xpt_rows: &[&[f64]] = &[
            &[1.0, 0.0],
            &[0.0, 1.0],
            &[1.0, 1.0],
            &[-1.0, 0.0],
            &[0.0, -1.0],
        ];
        let model = model_with(n, m, vec![0.0, 0.0], gamma_explicit, gamma, xpt_rows);

        let dense = model.dense_hessian();
        for j in 0..n {
            let mut e = vec![0.0; n];
            e[j] = 1.0;
            let col = model.hessian_matvec(&e);
            for i in 0..n {
                assert!(
                    (col[i] - dense.get(i, j)).abs() < 1e-12,
                    "∇²Q[{i},{j}]: matvec {} vs dense {}",
                    col[i],
                    dense.get(i, j)
                );
            }
        }
    }

    /// The implicit Hessian must stay symmetric: `eᵢᵀ ∇²Q eⱼ = eⱼᵀ ∇²Q eᵢ`.
    #[test]
    fn hessian_matvec_is_symmetric() {
        let n = 3;
        let m = 7;
        let gamma_explicit =
            DenseMatrix::from_row_slice(3, 3, &[2.0, -1.0, 0.5, -1.0, 3.0, 0.25, 0.5, 0.25, 1.0]);
        let gamma = vec![0.3, -0.1, 0.0, 0.2, 0.0, 0.0, 0.15];
        let xpt_rows: &[&[f64]] = &[
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0],
            &[1.0, -1.0, 0.0],
            &[0.5, 0.5, 0.5],
            &[-1.0, 0.0, 2.0],
            &[0.0, 1.0, -1.0],
        ];
        let model = model_with(n, m, vec![0.0; n], gamma_explicit, gamma, xpt_rows);
        let h = model.dense_hessian();
        for i in 0..n {
            for j in 0..n {
                assert!((h.get(i, j) - h.get(j, i)).abs() < 1e-12);
            }
        }
    }

    // NOTE: `from_parts` / `dense_hessian` live in a `#[cfg(test)]` inherent
    // impl above this module (clippy::items_after_test_module).

    /// `eval_change(d)` must equal `dᵀ∇Q + ½ dᵀ∇²Q d`, computed independently
    /// from the dense Hessian. With `∇Q = b` and `∇²Q = G`, this is exactly the
    /// change in the quadratic `F(x0+d) − F(x0)` an exact model would report.
    #[test]
    fn eval_change_matches_quadratic_form() {
        let n = 2;
        let m = 5;
        let b = vec![2.0, -3.0];
        let gamma_explicit = DenseMatrix::from_row_slice(2, 2, &[3.0, 1.0, 1.0, 4.0]);
        let gamma = vec![0.5, 0.0, -0.25, 0.0, 0.0];
        let xpt_rows: &[&[f64]] = &[
            &[1.0, 0.0],
            &[0.0, 1.0],
            &[1.0, 1.0],
            &[-1.0, 0.0],
            &[0.0, -1.0],
        ];
        let model = model_with(n, m, b.clone(), gamma_explicit, gamma, xpt_rows);
        let dense = model.dense_hessian();

        for d in [vec![0.7, -1.2], vec![-2.0, 0.3], vec![1.0, 1.0]] {
            // Reference: bᵀd + ½ dᵀ G d.
            let lin: f64 = b.iter().zip(&d).map(|(x, y)| x * y).sum();
            let mut quad = 0.0;
            for i in 0..n {
                for j in 0..n {
                    quad += d[i] * dense.get(i, j) * d[j];
                }
            }
            let expected = lin + 0.5 * quad;
            let got = model.eval_change(&d);
            assert!(
                (got - expected).abs() < 1e-12,
                "eval_change={got} expected={expected} for d={d:?}"
            );
        }
    }
}
