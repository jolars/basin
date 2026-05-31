//! Constraint markers carried on the problem (tenet 4 in `CONTRIBUTING.md`).
//! [`BoxConstraints`] (interval bounds, consumed by projection-based
//! solvers), [`LinearInequalityConstraints`] (`A x ≤ b`, consumed by the
//! log-barrier method), and [`LinearEqualityConstraints`] (`A x = b`,
//! consumed by the augmented-Lagrangian method).
//!
//! These are deliberately *sibling* traits, not members of a `Constraint`
//! supertrait/hierarchy. Per tenet 4, a shared abstraction waits until ≥2
//! constrained solvers share more than data accessors — and they don't:
//! the box family keeps feasibility by *projection*, the linear-inequality
//! family by a *barrier*, and the linear-equality family by a *penalty plus
//! multipliers* (the augmented Lagrangian). Three feasibility mechanisms,
//! and the traits share no operation beyond `lower()` / `upper()` resp.
//! `a()` / `b()`, so a one-member hierarchy would still be pure overhead.

use crate::core::problem::CostFunction;

/// Box (interval) bounds on the parameter.
///
/// Lives on the *problem* side (tenet 4 in `CONTRIBUTING.md`): constraints
/// describe the problem, not the executor. Solvers that require box
/// bounds bind on this trait so handing them an unconstrained problem is
/// a compile error rather than a silent runtime issue.
///
/// `BoxConstraints` is a supertrait of `CostFunction` so the `Param` type
/// is shared automatically — solver bounds read
/// `P: BoxConstraints<Param = f64>` instead of repeating the parameter
/// type across two trait bounds.
///
/// For 1D problems `Param = f64` and bounds are scalars; for n-D box
/// constraints `Param` is a vector and bounds are vectors of the same
/// length.
///
/// # Examples
///
/// Attach box bounds to a problem so a bounded solver (e.g.
/// [`ProjectedGradientDescent`](crate::solver::ProjectedGradientDescent))
/// will accept it. The equality / inequality sibling traits
/// ([`LinearEqualityConstraints`], [`LinearInequalityConstraints`]) are
/// implemented the same way, exposing `a()` / `b()` instead:
///
/// ```
/// use basin::{BoxConstraints, CostFunction};
///
/// struct BoundedSphere {
///     lower: Vec<f64>,
///     upper: Vec<f64>,
/// }
/// impl CostFunction for BoundedSphere {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
///         Ok(x.iter().map(|xi| xi * xi).sum())
///     }
/// }
/// impl BoxConstraints for BoundedSphere {
///     fn lower(&self) -> &Vec<f64> { &self.lower }
///     fn upper(&self) -> &Vec<f64> { &self.upper }
/// }
///
/// let problem = BoundedSphere { lower: vec![-1.0, -1.0], upper: vec![1.0, 1.0] };
/// assert_eq!(problem.lower(), &vec![-1.0, -1.0]);
/// ```
pub trait BoxConstraints: CostFunction {
    /// Element-wise lower bound on `Param`. Same shape as `Param`.
    fn lower(&self) -> &Self::Param;
    /// Element-wise upper bound on `Param`. Same shape as `Param`.
    fn upper(&self) -> &Self::Param;
}

/// Linear inequality constraints `A x ≤ b` in standard form.
///
/// `A` is the `m × n` constraint matrix and `b ∈ ℝᵐ`; the feasible set is
/// `{ x ∈ ℝⁿ : A x ≤ b }`. Like [`BoxConstraints`], the constraint data
/// lives on the *problem* side (tenet 4 in `CONTRIBUTING.md`): solvers that
/// handle linear inequalities (currently the log-barrier
/// [`BarrierMethod`](crate::solver::BarrierMethod)) bind on this trait, so
/// handing them an unconstrained problem is a compile error.
///
/// `LinearInequalityConstraints` is a supertrait of [`CostFunction`] so the
/// `Param` type is shared automatically.
///
/// # Shapes
///
/// `b` shares the parameter's vector *type* ([`Param`](CostFunction::Param))
/// but lives in `ℝᵐ` — one entry per constraint row — whereas the iterate
/// lives in `ℝⁿ`. `m` and `n` need not match.
///
/// # Matrix type and consumers
///
/// The trait stays free of math bounds on [`Matrix`](Self::Matrix), the
/// same way [`BoxConstraints`] does not bound `Param` on
/// [`ClampInPlace`](crate::core::math::ClampInPlace). Consumers add the
/// operations they actually need: the barrier requires
/// `Matrix: MatVec<Param> + MatTransposeVec<Param>` (for `A x` and
/// `Aᵀ v`) — a strict subset of the LA tier that never includes a linear
/// solve. With those bounds the barrier is available on the matrix-capable
/// backends (nalgebra `DMatrix`/`DVector`, faer `Mat`/`Col`); `Vec<f64>`
/// and `ndarray` produce a compile-time error until they grow the two
/// matvec impls (tenet 5).
pub trait LinearInequalityConstraints: CostFunction {
    /// The `m × n` constraint-matrix type. Consumers bound this on
    /// [`MatVec<Param>`](crate::core::math::MatVec) +
    /// [`MatTransposeVec<Param>`](crate::core::math::MatTransposeVec).
    type Matrix;
    /// The constraint matrix `A` (`m` rows = number of inequalities).
    fn a(&self) -> &Self::Matrix;
    /// The right-hand side `b ∈ ℝᵐ`.
    fn b(&self) -> &Self::Param;
}

/// Linear equality constraints `A x = b` in standard form.
///
/// `A` is the `m × n` constraint matrix and `b ∈ ℝᵐ`; the feasible set is
/// the affine subspace `{ x ∈ ℝⁿ : A x = b }`. Like the sibling
/// constraint traits, the data lives on the *problem* side (tenet 4 in
/// `CONTRIBUTING.md`): solvers that handle linear equalities (currently the
/// [`AugmentedLagrangianMethod`](crate::solver::AugmentedLagrangianMethod))
/// bind on this trait, so handing them an unconstrained problem is a compile
/// error.
///
/// This is a **distinct trait** from [`LinearInequalityConstraints`] even
/// though the data shape (`A`, `b`) is identical: the semantics differ
/// (`= b` vs `≤ b`), so the type system must keep them apart — an
/// `A x ≤ b` problem must not be silently accepted by an equality solver,
/// and vice versa.
///
/// `LinearEqualityConstraints` is a supertrait of [`CostFunction`] so the
/// `Param` type is shared automatically.
///
/// # Shapes
///
/// `b` shares the parameter's vector *type* ([`Param`](CostFunction::Param))
/// but lives in `ℝᵐ` — one entry per constraint row — whereas the iterate
/// lives in `ℝⁿ`. `m` and `n` need not match.
///
/// # Matrix type and consumers
///
/// The trait stays free of math bounds on [`Matrix`](Self::Matrix), the
/// same way the sibling traits leave their carriers unbounded. Consumers add
/// the operations they actually need: the augmented Lagrangian requires
/// `Matrix: MatVec<Param> + MatTransposeVec<Param>` (for `A x` and `Aᵀ v`) —
/// a strict subset of the LA tier that never includes a linear solve. With
/// those bounds the method is available on the matrix-capable backends
/// (nalgebra `DMatrix`/`DVector`, faer `Mat`/`Col`); `Vec<f64>` and
/// `ndarray` produce a compile-time error until they grow the two matvec
/// impls (tenet 5).
pub trait LinearEqualityConstraints: CostFunction {
    /// The `m × n` constraint-matrix type. Consumers bound this on
    /// [`MatVec<Param>`](crate::core::math::MatVec) +
    /// [`MatTransposeVec<Param>`](crate::core::math::MatTransposeVec).
    type Matrix;
    /// The constraint matrix `A` (`m` rows = number of equalities).
    fn a(&self) -> &Self::Matrix;
    /// The right-hand side `b ∈ ℝᵐ`.
    fn b(&self) -> &Self::Param;
}
