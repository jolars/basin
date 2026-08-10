//! Vector-tier primitives for box-constrained NLLS: the Coleman-Li
//! affine scaling vectors and friends.
//!
//! Three element-wise operations the [`Trf`](crate::solver::Trf) solver
//! needs that aren't generic enough to live in `math.rs`:
//!
//! - **Coleman-Li affine scaling**: `d²` and the `C` diagonal from
//!   Branch-Coleman-Li 1999 eqs (i)–(iv) (`source.marker.md:54-59` in
//!   `references/branch-coleman-li-1999/`). Defines the diagonal
//!   trust-region scaling matrix `D = diag(|v|^{-1/2})` and the
//!   curvature correction `C = D·diag(g)·J^v·D` that's also diagonal.
//! - **Strict-interior step-back**: largest `τ` keeping `x + τ·s` in
//!   the box, used to cut the unconstrained Newton step back to the
//!   feasible region. The TRF caller multiplies by `θ < 1` to keep
//!   iterates in the *open* box (D is undefined where `v_i = 0`).
//! - **Scaled gradient ∞-norm**: the `‖D·g‖_∞` first-order optimality
//!   measure, computed from `g` and `d²` without materializing `D·g`.
//!
//! Per-backend implementations in `vec.rs`, `nalgebra_backend.rs`,
//! `ndarray_backend.rs`, `faer_backend.rs`. All four backends are
//! supported: the operations are pure element-wise, no LA dependency,
//! so the vector-tier coverage is honest across the corpus.
//!
//! The trait carries an `F = f64` scalar default so existing
//! `V: BoxAffineScaling` bounds resolve to the `f64` impl unchanged,
//! while individual backends now broaden into other `F: Scalar`.

use super::Scalar;

/// Box-constrained NLLS vector primitives. Implemented for every
/// vector backend basin ships (`Vec<f64>`, `nalgebra::DVector<f64>`,
/// `ndarray::Array1<f64>`, `faer::Col<f64>`).
///
/// Used exclusively by the [`Trf`](crate::solver::Trf) solver today.
/// The trait groups three otherwise-unrelated element-wise operations
/// because they share their caller and have no other users; splitting
/// would just inflate the solver's `where` clause.
///
/// `F` defaults to `f64`, matching the legacy scalar pin; per-backend
/// impls are generic in `F: Scalar` so [`Trf`](crate::solver::Trf) can
/// run with `F = f32` once the broader NLLS slice picks it up.
pub trait BoxAffineScaling<F = f64>: Sized {
    /// Coleman-Li affine scaling diagonals per Branch-Coleman-Li 1999
    /// eqs (i)–(iv). `self` is the current iterate `x`. After return:
    ///
    /// - `d_sq[i] = 1/|v_i|`, where `v_i` follows the four-case
    ///   definition based on `sign(g_i)` and finiteness of bounds:
    ///   `v_i = x_i − u_i` (case i: `g_i < 0` and finite upper),
    ///   `v_i = x_i − l_i` (case ii: `g_i ≥ 0` and finite lower),
    ///   `v_i = ±1` (cases iii/iv: relevant bound is infinite).
    /// - `c_diag[i] = |g_i|/|v_i|` for cases (i)/(ii), `0` for cases
    ///   (iii)/(iv). This is the diagonal of BCL's
    ///   `C = D·diag(g)·J^v·D` term, always non-negative.
    ///
    /// `d_sq` and `c_diag` are overwritten. All five arguments must
    /// be the same shape; backends panic on mismatch.
    ///
    /// # Contract
    ///
    /// - **Caller must:** pass `self` strictly inside `[lower, upper]`
    ///   (no equalities). Cases (i)/(ii) divide by `|x_i − bound_i|`,
    ///   which the caller must keep strictly positive. The TRF solver
    ///   enforces this with an `init`-time strict-interior projection.
    /// - **Implementor must:** match the BCL case dispatch exactly.
    ///   Tied cases (`g_i = 0` exactly) fall into case (ii) per the
    ///   `g_i ≥ 0` clause.
    fn compute_cl_scaling(
        &self,
        gradient: &Self,
        lower: &Self,
        upper: &Self,
        d_sq: &mut Self,
        c_diag: &mut Self,
    );

    /// Largest `τ ≥ 0` such that `self + τ·step` stays component-wise
    /// inside `[lower, upper]`. Returns `F::infinity()` when no
    /// component of `step` points at a finite bound from the current
    /// iterate (e.g. all bounds infinite, or `step` is zero where
    /// bounds are finite).
    ///
    /// # Contract
    ///
    /// - **Caller must:** pass `self`, `step`, `lower`, `upper` of the
    ///   same shape; backends panic on mismatch.
    /// - **Implementor must:** for each component `i` with `step[i] ≠ 0`,
    ///   compute the per-component limit
    ///   `(upper_i − self_i) / step_i` if `step_i > 0` (and `upper_i`
    ///   is finite) or `(lower_i − self_i) / step_i` if `step_i < 0`
    ///   (and `lower_i` is finite). Take the minimum over all such
    ///   limits. Components with `step_i = 0` or with the relevant
    ///   bound infinite contribute no limit.
    /// - The TRF caller multiplies `τ` by a strict-interior factor
    ///   `θ ∈ [θ_l, 1)` to keep iterates in the *open* box.
    fn max_feasible_step(&self, step: &Self, lower: &Self, upper: &Self) -> F;

    /// `max_i |self[i]| / d_sq[i]`, the BCL first-order optimality
    /// measure `‖v ⊙ g‖_∞` when `self = g` and `d_sq[i] = 1/|v_i|` is
    /// the [`compute_cl_scaling`](Self::compute_cl_scaling) output. Equals `max_i |g_i · v_i|`.
    ///
    /// This metric goes to zero at any KKT point of the box-constrained
    /// problem: interior *or* face-active. The unscaled `‖g‖_∞`
    /// doesn't vanish on a finite face, and the scaled `‖D·g‖_∞ =
    /// max |g_i| / √|v_i|` actually *blows up* on a face (denominator
    /// → 0), so neither is a usable termination measure for TRF.
    ///
    /// SciPy's `least_squares(method='trf')` uses the same metric (its
    /// `g_norm = max |g · v|`); see
    /// `references/branch-coleman-li-1999/NOTES.md` for the derivation.
    ///
    /// # Contract
    ///
    /// - **Caller must:** pass `d_sq` of the same shape as `self`,
    ///   with strictly positive entries. Backends panic on shape
    ///   mismatch; entries of `0` produce `inf`, which propagates.
    fn cl_kkt_inf_norm(&self, d_sq: &Self) -> F;

    /// `Σ self[i]² · weights[i]`, the squared D-norm `‖D · self‖²`
    /// when `weights = d_sq` is the [`compute_cl_scaling`](Self::compute_cl_scaling) output.
    /// Used in the BCL scaled trust-region predicted-reduction
    /// `½(μ · ‖D·h‖² − h^T g)`, the analogue of Nielsen's LM
    /// `½(μ‖h‖² − h^T g)` with the affine-scaling D folded in.
    ///
    /// # Contract
    ///
    /// - **Caller must:** pass `weights` of the same shape as `self`.
    ///   Backends panic on shape mismatch.
    /// - **Implementor must:** return `Σᵢ self[i]² · weights[i]`.
    fn weighted_norm_squared(&self, weights: &Self) -> F;

    /// Project `self` into the *open* box `(lower, upper)` element-wise.
    /// Components outside `[lower, upper]` are clamped to a strict-
    /// interior point; components already strictly inside are
    /// unchanged. The strict-interior offset is
    /// `rstep · max(1, |bound|)` from the relevant finite bound.
    ///
    /// Used at TRF `init` to bring an arbitrary starting point into
    /// the open feasible region: the affine scaling matrix `D` is
    /// undefined where `v_i = 0` (i.e. on a finite face).
    ///
    /// # Contract
    ///
    /// - **Caller must:** pass `lower`/`upper` of the same shape as
    ///   `self` with `lower[i] < upper[i]` for any component that has
    ///   both bounds finite. Backends panic on shape mismatch; equal
    ///   finite bounds produce an undefined post-projection iterate.
    /// - **Implementor must:** for each component `i`:
    ///   * If both bounds infinite: leave `self[i]` unchanged.
    ///   * Else compute `lo_inner = lower[i] + rstep · max(1, |lower[i]|)`
    ///     (treating infinite `lower[i]` as `-∞`, which makes
    ///     `lo_inner = -∞`).
    ///   * Compute `hi_inner = upper[i] - rstep · max(1, |upper[i]|)`
    ///     analogously.
    ///   * Set `self[i] ← clamp(self[i], lo_inner, hi_inner)`.
    fn project_strictly_inside(&mut self, lower: &Self, upper: &Self, rstep: F);
}

/// Helper for the case dispatch in `compute_cl_scaling`: writes
/// `(d_sq_i, c_diag_i)` for one component given the local `(x, g, l, u)`.
/// Per-backend impls call this on each index to share the arithmetic;
/// generic over `F: Scalar` so every backend can supply its own
/// element type.
#[inline]
pub(crate) fn cl_scaling_pair<F: Scalar>(x: F, g: F, l: F, u: F) -> (F, F) {
    let zero = F::zero();
    let one = F::one();
    if g < zero {
        if u.is_finite() {
            // Case (i): v_i = x − u < 0, |v_i| = u − x > 0 by strict feasibility.
            let abs_v = u - x;
            (one / abs_v, (-g) / abs_v)
        } else {
            // Case (iii): v_i = -1, J^v_ii = 0, c_i = 0.
            (one, zero)
        }
    } else if l.is_finite() {
        // Case (ii): v_i = x − l > 0 by strict feasibility.
        let abs_v = x - l;
        (one / abs_v, g / abs_v)
    } else {
        // Case (iv): v_i = 1, J^v_ii = 0, c_i = 0.
        (one, zero)
    }
}

/// Helper for `project_strictly_inside`: per-component clamp into the
/// *open* box. Returns the projected value.
#[inline]
pub(crate) fn project_strictly_inside_component<F: Scalar>(
    x: F,
    l: F,
    u: F,
    rstep: F,
) -> F {
    let one = F::one();
    let lo_inner = if l.is_finite() {
        l + rstep * l.abs().max(one)
    } else {
        F::neg_infinity()
    };
    let hi_inner = if u.is_finite() {
        u - rstep * u.abs().max(one)
    } else {
        F::infinity()
    };
    // num_traits::Float has no `clamp`; emulate with min/max.
    if x < lo_inner {
        lo_inner
    } else if x > hi_inner {
        hi_inner
    } else {
        x
    }
}

/// Helper for `max_feasible_step`: per-component limit, returning
/// `F::infinity()` when `step_i = 0` or the relevant bound is infinite.
#[inline]
pub(crate) fn max_feasible_step_component<F: Scalar>(
    x: F,
    step: F,
    l: F,
    u: F,
) -> F {
    let zero = F::zero();
    if step > zero {
        if u.is_finite() {
            (u - x) / step
        } else {
            F::infinity()
        }
    } else if step < zero {
        if l.is_finite() {
            (l - x) / step
        } else {
            F::infinity()
        }
    } else {
        F::infinity()
    }
}
