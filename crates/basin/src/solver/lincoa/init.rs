//! Linearly-constrained initialization of the shared [`QuadraticModel`] (LINCOA,
//! Powell 2015; PRIMA `lincoa.f90::get_lincon` + `initialize.f90::initxf`).
//!
//! LINCOA seeds the **same** coordinate-cross interpolation set and `H`
//! factorization as NEWUOA (so this reuses
//! [`try_initialize`](QuadraticModel::try_initialize) verbatim — PRIMA's `inith`
//! is the NEWUOA initialization), then layers the linear-constraint machinery on
//! top:
//!
//! 1. [`fold_constraints`] wraps box bounds, linear equalities, and linear
//!    inequalities into a single system `A x ≤ b` with **unit-norm columns**
//!    (PRIMA `get_lincon`). Each column of `amat` is one constraint normal
//!    (`n × m`, column-major), matching what [`trstep`](super::trstep) and
//!    [`getact`](super::getact) consume. The right-hand side is relaxed so the
//!    start point is feasible (Powell's code modifies `b` rather than projecting
//!    `x0`).
//! 2. [`try_initialize_linear`](QuadraticModel::try_initialize_linear) shifts `b`
//!    into the model's `x0`-relative coordinates and picks `kopt` among the
//!    **feasible** interpolation points only (PRIMA `initxf`).
//!
//! Equality constraints `a·x = β` are folded as the pair `a·x ≤ β`, `−a·x ≤ −β`
//! (PRIMA's deliberately naive treatment — no special permanently-active
//! handling), so the driver works on a pure inequality system.

use crate::core::math::Scalar;
use crate::solver::powell::QuadraticModel;

/// The product of [`try_initialize_linear`](QuadraticModel::try_initialize_linear):
/// the seeded model together with the folded constraint system the driver /
/// TRSTEP work in.
pub(crate) struct LinearSetup<F = f64> {
    /// The seeded quadratic model (interpolation set relative to `x0`).
    pub(crate) model: QuadraticModel<F>,
    /// Unit-normalized constraint normals, `n × m` column-major; column `j` is
    /// the normal `a_j` of the `j`-th inequality `a_jᵀ x ≤ b_j`.
    pub(crate) amat: Vec<F>,
    /// Right-hand sides in the model's `x0`-relative coordinates
    /// (`b_abs − A x0`), length `m`. The driver's residual is `b − A (x_opt−x0)`.
    pub(crate) bvec: Vec<F>,
}

/// Wrap box, equality, and inequality constraints into a single unit-normalized
/// inequality system `A x ≤ b` in **absolute** coordinates (PRIMA `get_lincon`).
///
/// Columns are laid out `[lower bounds, upper bounds, equalities (±), inequalities]`
/// — `n × m` column-major. The right-hand side is relaxed so `x0` is feasible
/// (`b_j ← max(b_j, a_jᵀ x0)`, computed with the *un-normalized* normals, exactly
/// as Powell's code does), then every column and its `b` entry are divided by the
/// normal's Euclidean norm. Constraints with a zero gradient are dropped (PRIMA
/// warns and ignores them).
///
/// - `lower`/`upper`: optional box bounds, length `n`; non-finite entries mean the
///   coordinate is unbounded on that side and contribute no constraint.
/// - `eq`: linear equalities as `(normal, β)` with the normal length `n`.
/// - `ineq`: linear inequalities `a·x ≤ β` as `(normal, β)`.
pub(crate) fn fold_constraints<F: Scalar>(
    n: usize,
    x0: &[F],
    lower: Option<&[F]>,
    upper: Option<&[F]>,
    eq: &[(Vec<F>, F)],
    ineq: &[(Vec<F>, F)],
) -> (Vec<F>, Vec<F>) {
    let zero = F::zero();
    let one = F::one();

    // Collect the raw (un-normalized) columns in PRIMA's order.
    let mut raw: Vec<(Vec<F>, F)> = Vec::new();
    if let Some(lo) = lower {
        for i in 0..n {
            if lo[i].is_finite() {
                let mut a = vec![zero; n];
                a[i] = -one; // −x_i ≤ −lower_i
                raw.push((a, -lo[i]));
            }
        }
    }
    if let Some(up) = upper {
        for i in 0..n {
            if up[i].is_finite() {
                let mut a = vec![zero; n];
                a[i] = one; // x_i ≤ upper_i
                raw.push((a, up[i]));
            }
        }
    }
    for (a, b) in eq {
        // a·x = β  ⟶  a·x ≤ β  and  −a·x ≤ −β.
        raw.push((a.iter().map(|v| -*v).collect(), -*b));
        raw.push((a.clone(), *b));
    }
    for (a, b) in ineq {
        raw.push((a.clone(), *b));
    }

    // Relax (with the un-normalized normals) then normalize.
    let mut amat = Vec::with_capacity(n * raw.len());
    let mut bvec = Vec::with_capacity(raw.len());
    for (a, b) in &raw {
        let norm = a.iter().fold(zero, |s, v| s + *v * *v).sqrt();
        if norm <= zero {
            continue; // zero-gradient constraint: ignore
        }
        let ax0 = (0..n).fold(zero, |s, i| s + a[i] * x0[i]);
        let b_relaxed = (*b).max(ax0);
        for i in 0..n {
            amat.push(a[i] / norm);
        }
        bvec.push(b_relaxed / norm);
    }
    (amat, bvec)
}

/// `a_jᵀ v` where `a_j` is column `j` of the `n × m` column-major `amat`.
fn col_dot<F: Scalar>(v: &[F], amat: &[F], j: usize, n: usize) -> F {
    (0..n).fold(F::zero(), |acc, r| acc + v[r] * amat[r + j * n])
}

/// The feasibility-aware best interpolation point (PRIMA `initxf`): `kopt` is the
/// least-`F` point among those satisfying `A x ≤ b`. If no sampled point is
/// feasible, the least-violation point is forced feasible so `kopt` is defined
/// (LINCOA assumes a feasible start, but rounding can leave every cross point a
/// hair outside).
fn feasible_kopt<F: Scalar>(model: &QuadraticModel<F>, amat: &[F], bvec: &[F], n: usize) -> usize {
    let zero = F::zero();
    let npt = model.m();
    let m = bvec.len();

    // Constraint violation of each point (in x0-relative coordinates): the model
    // displacements `xpt_k` satisfy `a_jᵀ xpt_k ≤ b_disp_j`.
    let cval: Vec<F> = (0..npt)
        .map(|k| {
            let xk = model.xpt_row(k);
            let mut cv = zero;
            for j in 0..m {
                let viol = col_dot(xk, amat, j, n) - bvec[j];
                if viol > cv {
                    cv = viol;
                }
            }
            cv
        })
        .collect();

    let mut feasible: Vec<bool> = cval.iter().map(|&c| c <= zero).collect();
    // Force the least-violation point feasible (PRIMA: minloc(cval)).
    let mut kmin = 0;
    for k in 1..npt {
        if cval[k] < cval[kmin] {
            kmin = k;
        }
    }
    feasible[kmin] = true;

    let mut kopt = kmin;
    let mut found = false;
    for k in 0..npt {
        if feasible[k] && (!found || model.fval(k) < model.fval(kopt)) {
            kopt = k;
            found = true;
        }
    }
    kopt
}

impl<F: Scalar> QuadraticModel<F> {
    /// Build LINCOA's first model under the folded linear-constraint system
    /// (Powell 2015; PRIMA `initxf`).
    ///
    /// `amat`/`bvec_abs` are the unit-normalized system from [`fold_constraints`]
    /// in **absolute** coordinates. The model is seeded exactly as NEWUOA's
    /// (coordinate cross + closed-form `H`), then `b` is shifted into `x0`-relative
    /// coordinates and `kopt` is chosen among the feasible cross points.
    ///
    /// `eval` is called once per interpolation point (`m_npt` evaluations); the
    /// first `Err` bubbles to the caller.
    ///
    /// # Panics
    ///
    /// Panics unless `n ≥ 1` and `2n+1 ≤ npt ≤ ½(n+1)(n+2)` (via
    /// [`try_initialize`](Self::try_initialize)).
    pub(crate) fn try_initialize_linear<E>(
        x0: Vec<F>,
        amat: Vec<F>,
        bvec_abs: Vec<F>,
        rho_beg: F,
        npt: usize,
        eval: &mut impl FnMut(&[F]) -> Result<F, E>,
    ) -> Result<LinearSetup<F>, E> {
        let n = x0.len();
        let m = bvec_abs.len();

        // Seed the coordinate cross, the model, and the H factorization.
        let mut model = Self::try_initialize(x0.clone(), rho_beg, npt, eval)?;

        // Shift the right-hand sides into x0-relative coordinates.
        let bvec: Vec<F> = (0..m)
            .map(|j| bvec_abs[j] - col_dot(&x0, &amat, j, n))
            .collect();

        model.kopt = feasible_kopt(&model, &amat, &bvec, n);

        Ok(LinearSetup { model, amat, bvec })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single inequality `x0 + x1 ≤ 2` folds to one unit-normal column
    /// `(1/√2, 1/√2)` with `b = 2/√2`.
    #[test]
    fn single_inequality_is_normalized() {
        let (amat, bvec) =
            fold_constraints::<f64>(2, &[0.0, 0.0], None, None, &[], &[(vec![1.0, 1.0], 2.0)]);
        let s = 1.0 / 2.0_f64.sqrt();
        assert!((amat[0] - s).abs() < 1e-12);
        assert!((amat[1] - s).abs() < 1e-12);
        assert!((bvec[0] - 2.0 * s).abs() < 1e-12);
    }

    /// Box bounds fold to `±e_i` columns (already unit norm, unchanged).
    #[test]
    fn box_bounds_fold_to_axis_normals() {
        let (amat, bvec) = fold_constraints::<f64>(
            2,
            &[0.0, 0.0],
            Some(&[-1.0, -3.0]),
            Some(&[2.0, 4.0]),
            &[],
            &[],
        );
        // 4 columns: -e0, -e1, e0, e1; column-major n=2.
        assert_eq!(bvec.len(), 4);
        // lower x0 >= -1  ->  -x0 <= 1
        assert_eq!(&amat[0..2], &[-1.0, 0.0]);
        assert_eq!(bvec[0], 1.0);
        // lower x1 >= -3  ->  -x1 <= 3
        assert_eq!(&amat[2..4], &[0.0, -1.0]);
        assert_eq!(bvec[1], 3.0);
        // upper x0 <= 2
        assert_eq!(&amat[4..6], &[1.0, 0.0]);
        assert_eq!(bvec[2], 2.0);
        // upper x1 <= 4
        assert_eq!(&amat[6..8], &[0.0, 1.0]);
        assert_eq!(bvec[3], 4.0);
    }

    /// An equality `x0 + x1 = 1` folds to two opposite columns (±) with matching
    /// signed right-hand sides.
    #[test]
    fn equality_folds_to_two_inequalities() {
        // x0 = (0.5, 0.5) satisfies x0 + x1 = 1 exactly (no relaxation).
        let (amat, bvec) =
            fold_constraints::<f64>(2, &[0.5, 0.5], None, None, &[(vec![1.0, 1.0], 1.0)], &[]);
        assert_eq!(bvec.len(), 2);
        let s = 1.0 / 2.0_f64.sqrt();
        // First column is -a (from -a·x <= -b), second is +a.
        assert!((amat[0] + s).abs() < 1e-12 && (amat[1] + s).abs() < 1e-12);
        assert!((bvec[0] + s).abs() < 1e-12); // -1/√2
        assert!((amat[2] - s).abs() < 1e-12 && (amat[3] - s).abs() < 1e-12);
        assert!((bvec[1] - s).abs() < 1e-12); // +1/√2
    }

    /// An infeasible start relaxes `b` up to the constraint value at `x0`.
    #[test]
    fn infeasible_start_relaxes_b() {
        // x0 = (3, 3) violates x0 + x1 <= 2: A x0 = 6 > 2.
        let (_amat, bvec) =
            fold_constraints::<f64>(2, &[3.0, 3.0], None, None, &[], &[(vec![1.0, 1.0], 2.0)]);
        let s = 1.0 / 2.0_f64.sqrt();
        // Relaxed (absolute, normalized): max(2, 6)/√2 = 6/√2.
        assert!((bvec[0] - 6.0 * s).abs() < 1e-12);
    }

    /// `try_initialize_linear` shifts `b` into `x0`-relative coordinates and
    /// picks `kopt` among feasible cross points only.
    #[test]
    fn init_shifts_b_and_picks_feasible_kopt() {
        // min ‖x − c‖² with c = (2, 2), constraint x0 + x1 <= 2 (optimum (1,1)).
        let c = [2.0, 2.0];
        let f = |x: &[f64]| (x[0] - c[0]).powi(2) + (x[1] - c[1]).powi(2);
        let x0 = vec![0.0, 0.0];
        let (amat, bvec_abs) =
            fold_constraints::<f64>(2, &x0, None, None, &[], &[(vec![1.0, 1.0], 2.0)]);

        let setup = QuadraticModel::try_initialize_linear::<core::convert::Infallible>(
            x0.clone(),
            amat,
            bvec_abs.clone(),
            0.3,
            5,
            &mut |x| Ok(f(x)),
        )
        .expect("infallible");

        // b shifted to x0-relative: since x0 = 0, no shift.
        assert!((setup.bvec[0] - bvec_abs[0]).abs() < 1e-12);

        // kopt must be a feasible point: a·(x_opt − x0) ≤ b_disp.
        let kopt = setup.model.kopt();
        let xk = setup.model.xpt_row(kopt);
        let ax = super::col_dot(xk, &setup.amat, 0, 2);
        assert!(ax <= setup.bvec[0] + 1e-12, "kopt point is infeasible");
    }
}
