//! Math abstraction the solvers depend on.
//!
//! Two tiers per `CONTRIBUTING.md` tenet 5:
//!
//! - **Vector tier** (this module): small ops every backend can implement
//!   well: [`ScaledAdd`], [`NormSquared`], [`NormInfinity`], [`Dot`],
//!   [`NegInPlace`]. Backend-generic solvers (gradient descent,
//!   Nelder-Mead) bound on these.
//! - **`linalg` tier**: LA-heavy ops ([`MatVec`],
//!   [`MatTransposeVec`], [`GramMatrix`], [`LinearSolveSpd`],
//!   [`LinearSolveLstsq`]) that only the matrix-capable backends
//!   (nalgebra, faer; sparse counterparts in S2b) implement. LA-heavy
//!   solvers (Gauss-Newton, LM) bound on these so other backends
//!   produce compile-time errors instead of runtime surprises. The two
//!   matvec ops ([`MatVec`], [`MatTransposeVec`]) are the exception: they
//!   are *also* implemented for the `Vec<f64>` backend (via the
//!   hand-rolled [`DenseMatrix`]) and `ndarray::Array2<f64>`, so the
//!   linear-constraint solvers run on every backend. [`SymmetricEigen`] is
//!   a second exception: [`DenseMatrix`] implements it via a pure-Rust
//!   cyclic Jacobi eigensolver (`dense_eig`), so CMA-ES runs on the default
//!   backend too. The SPD solve [`LinearSolveSpd`] and its companion
//!   [`GramMatrix`] (plus [`AddDiagonalVectorInPlace`]) are a third:
//!   [`DenseMatrix`] implements them via a pure-Rust Cholesky (`dense_chol`),
//!   so Gauss-Newton, Levenberg-Marquardt, and TRF run on the default backend.
//!   [`AddDiagonalVectorInPlace`] and [`MaxDiagonal`] are public so downstream
//!   Jacobian types can implement TRF's damping path. The QR least-squares
//!   solve [`LinearSolveLstsq`] remains backend-specific.

/// Scalar element type for vectors and matrices in the math layer.
///
/// Bundles every bound the rest of the crate needs from a scalar so call sites
/// can write `F: Scalar` instead of repeating the trait list. `f64` and `f32`
/// both satisfy it; user code never needs to implement it directly (the
/// blanket impl picks up any type that meets the bounds).
///
/// The constituent bounds:
///
/// - [`num_traits::Float`]: arithmetic, `epsilon`, `infinity`, `is_finite`,
///   `sqrt`, `powf`, … plus `Copy` and `PartialOrd` transitively.
/// - [`num_traits::FromPrimitive`]: `from_f64(...)` so tolerance defaults
///   (`1e-4`, `1e-8`, …) can be expressed at any `F` without sprinkling
///   `as` casts. Use [`F::from_f64(lit).unwrap()`](num_traits::FromPrimitive::from_f64)
///   at the construction site of any literal that doesn't have a Float
///   constructor of its own (e.g. `F::epsilon()` is fine as-is).
/// - [`std::iter::Sum`]: for the natural `.iter().map(...).sum()` pattern
///   used in raw test-problem functions (Sphere, Rosenbrock, …).
/// - [`std::fmt::Debug`]: so solver/state structs can `#[derive(Debug)]`.
/// - [`Default`]: matches `f64`'s `Default = 0.0` so generic states can
///   `Option<F>::default()` cleanly.
/// - `'static`: matches `f64`'s implicit `'static` so the bound doesn't
///   force lifetime plumbing through every solver.
pub trait Scalar:
    num_traits::Float
    + num_traits::FromPrimitive
    + std::iter::Sum
    + std::fmt::Debug
    + Default
    + 'static
{
}

impl<F> Scalar for F where
    F: num_traits::Float
        + num_traits::FromPrimitive
        + std::iter::Sum
        + std::fmt::Debug
        + Default
        + 'static
{
}

/// In-place `self ← self + scalar · other`. Backend-generic vector update.
///
/// The scalar type defaults to `f64` so the legacy `ScaledAdd<f64>` bound
/// keeps working unchanged while individual backends start to broaden into
/// other `F: Scalar`.
pub trait ScaledAdd<S = f64> {
    /// Add `scalar · other` into `self` in place.
    fn scaled_add(&mut self, scalar: S, other: &Self);
}

/// `‖x‖₂² = Σ xᵢ²`. Avoids the `sqrt` cost when the squared form is
/// what's actually needed (most quadratic-cost convergence checks).
///
/// `F` defaults to `f64` so existing `V: NormSquared` bounds keep
/// resolving to the `f64` impl unchanged.
pub trait NormSquared<F = f64> {
    /// Compute `Σ xᵢ²` as `F`.
    fn norm_squared(&self) -> F;
}

/// `‖x‖_∞ = maxᵢ |xᵢ|`. Used by first-order optimality stopping rules
/// (e.g. `‖∇f‖_∞ ≤ tol`).
///
/// `F` defaults to `f64`; see [`NormSquared`] for the rationale.
pub trait NormInfinity<F = f64> {
    /// Compute `maxᵢ |xᵢ|` as `F`.
    fn norm_infinity(&self) -> F;
}

/// Inner product of two same-shaped values. Used by line searches that take
/// an explicit search direction (Armijo and curvature checks both need
/// `gᵀd`). Generalizes `NormSquared`: `x.norm_squared() == x.dot(x)`.
///
/// `F` defaults to `f64`; see [`NormSquared`] for the rationale.
pub trait Dot<F = f64> {
    /// Compute `Σᵢ self[i] · other[i]` as `F`.
    fn dot(&self, other: &Self) -> F;
}

/// In-place negation. Lets solvers compute `direction = -gradient` in a
/// backend-generic way without allocating per-iteration scratch types.
pub trait NegInPlace {
    /// Negate every component of `self` in place.
    fn neg_in_place(&mut self);
}

/// In-place scalar multiplication `self ← scalar · self`. Used by
/// CMA-ES to update the cumulation paths (`p_σ ← (1−c_σ) p_σ + …`,
/// Hansen 2016 eq. 31) and the covariance matrix
/// (`C ← (1 + c_1 δ_h − c_1 − c_µ Σ w_j) C + …`, eq. 47) without
/// allocating a clone per iteration.
///
/// `ScaledAdd<f64>` already covers `self ← self + s · other`; the
/// borrow checker forbids `self.scaled_add(s, &self)`, so an honest
/// in-place scale needs its own trait.
///
/// `F` defaults to `f64`; see [`NormSquared`] for the rationale.
pub trait ScaleInPlace<F = f64> {
    /// Multiply every component of `self` by `scalar` in place.
    fn scale_in_place(&mut self, scalar: F);
}

/// Number of components in a 1-D vector. Used by CMA-ES to derive the
/// search-space dimension `n` from a template vector at solver
/// construction time, so callers don't have to thread `n` separately
/// from the initial mean. Method named `vec_len` to avoid colliding
/// with the inherent `len()` methods on `Vec`, `DVector`, `Array1`,
/// `Col`.
pub trait VectorLen {
    /// Number of components in `self`.
    fn vec_len(&self) -> usize;
}

/// In-place componentwise multiplication `self[i] ← self[i] · other[i]`.
/// CMA-ES uses this to apply the diagonal `D` (sqrt-eigenvalue) factor:
/// the sampling step `y_k = B D z_k` is `z.component_mul_assign(&d);
/// y = B.matvec(&z)`, and the conjugate-path step `C^{−1/2} v =
/// B (1/d ⊙ Bᵀv)` is the same pattern with `1/d`.
pub trait ComponentMulAssign {
    /// Multiply `self[i]` by `other[i]` for every `i`, in place.
    fn component_mul_assign(&mut self, other: &Self);
}

/// In-place componentwise maximum `self[i] ← max(self[i], other[i])`.
/// Levenberg-Marquardt uses this to maintain the monotone running-max
/// scaling diagonal `D_k = max(D_{k−1}, diag(JᵀJ))` of MINPACK-style
/// Marquardt damping (Moré 1978): a parameter whose column curvature
/// momentarily drops doesn't lose the damping floor accumulated from
/// earlier iterations.
pub(crate) trait ComponentMaxAssign {
    /// Set `self[i]` to `max(self[i], other[i])` for every `i`, in place.
    fn component_max_assign(&mut self, other: &Self);
}

/// In-place componentwise division `self[i] ← self[i] / other[i]`. The
/// counterpart of [`ComponentMulAssign`]. Levenberg-Marquardt forms the
/// MINPACK `gtol` measure (the per-column cosine `g_j² / (JᵀJ)ⱼⱼ`)
/// with this.
///
/// # Contract
///
/// - **Caller must:** ensure `other[i] ≠ 0` for every `i`; division by a
///   zero divisor yields a non-finite value that propagates. (LM floors
///   the divisor away from zero with [`FloorZerosInPlace`] first.)
pub(crate) trait ComponentDivAssign {
    /// Divide `self[i]` by `other[i]` for every `i`, in place.
    fn component_div_assign(&mut self, other: &Self);
}

/// In-place floor of non-positive entries to a positive `value`,
/// leaving strictly-positive entries untouched
/// (`self[i] ← value` where `self[i] ≤ 0`, else unchanged).
///
/// This is *not* a blanket lower-clamp: a legitimately small positive
/// entry keeps its value. It exists for MINPACK's zero-column guard in
/// Marquardt-scaled Levenberg-Marquardt: a Jacobian column that is
/// entirely zero gives `diag(JᵀJ)ⱼ = 0`, which would make the damping
/// `μ·D` vanish on that coordinate and leave the normal-equations
/// matrix singular there. MINPACK sets such a column's scale to `1`
/// (lmder, `mode = 1`); flooring zeros to `1` reproduces that, so a
/// fully-insensitive parameter simply stays put instead of failing the
/// Cholesky.
///
/// `F` defaults to `f64`; see [`NormSquared`] for the rationale.
pub(crate) trait FloorZerosInPlace<F = f64> {
    /// Replace every entry `≤ 0` with `value`; leave positive entries
    /// unchanged.
    fn floor_zeros_in_place(&mut self, value: F);
}

/// Per-component scalar read and write on a 1-D vector backend. The minimal
/// access finite-difference differentiation needs: perturb one coordinate
/// of a parameter vector and read derivative values back out
/// ([`crate::core::numdiff`]).
///
/// Methods are named `get_scalar`/`set_scalar` rather than `get`/`set`
/// to dodge the inherent `slice::get -> Option<&T>` (which would shadow a
/// trait `get` at call sites on a concrete `Vec`) and the `Index`/
/// `IndexMut` traits, the same defensive convention as
/// [`VectorLen::vec_len`].
///
/// `F` defaults to `f64`; see [`NormSquared`] for the rationale.
///
/// # Contract
///
/// - **Caller must:** pass `i < self.vec_len()`. Backends index directly and
///   panic on out-of-bounds, matching the underlying `Index` impls.
pub trait VectorIndex<F = f64> {
    /// Read component `i` as `F`.
    fn get_scalar(&self, i: usize) -> F;
    /// Write `value` into component `i`.
    fn set_scalar(&mut self, i: usize, value: F);
}

mod cl_scaling;
mod clamp;
mod dense;
mod dense_chol;
mod dense_eig;
mod linalg;
mod sample;
mod scalar;
mod vec;

#[cfg(feature = "nalgebra_all")]
mod nalgebra_backend;

#[cfg(feature = "nalgebra_all")]
mod nalgebra_sparse_backend;

#[cfg(feature = "ndarray_all")]
mod ndarray_backend;

#[cfg(feature = "faer_all")]
mod faer_backend;

#[cfg(feature = "faer_all")]
mod faer_sparse_backend;

pub use clamp::ClampInPlace;
pub use dense::DenseMatrix;
pub use linalg::{
    AddDiagonalVectorInPlace, DenseMatrixFromFn, GramMatrix, LinearSolveError,
    LinearSolveLstsq, LinearSolveSpd, MatTransposeVec, MatVec,
    MatrixFromDiagonal, MatrixIdentity, MaxDiagonal, SymmetricEigen,
    SymmetricEigenError,
};
pub use sample::{SampleStandardNormal, SampleUniformBox};

// Remaining per-solver plumbing ops carry no meaning outside one shipped
// solver's internals, so they stay off the frozen public surface.
pub(crate) use cl_scaling::BoxAffineScaling;
pub(crate) use linalg::{GeneralRankOneUpdate, MatDiagonal, RankOneUpdate};
