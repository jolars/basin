//! Constraint markers carried on the problem (tenet 4 in `CONTRIBUTING.md`).
//! [`BoxConstraints`] (interval bounds, consumed by projection-based
//! solvers), [`LinearInequalityConstraints`] (`A x ≤ b`, consumed by the
//! log-barrier method), [`LinearEqualityConstraints`] (`A x = b`, consumed by
//! the augmented-Lagrangian method), and [`NonlinearInequalityConstraints`]
//! (`c(x) ≤ 0` for an arbitrary vector-valued `c`, consumed by COBYLA).
//!
//! These four are deliberately *sibling* traits, not members of a
//! `Constraint` supertrait/hierarchy. Per tenet 4, a shared *parent*
//! abstraction waits until ≥2 constrained solvers share more than data
//! accessors — and they don't: the box family keeps feasibility by
//! *projection*, the linear-inequality family by a *barrier*, the
//! linear-equality family by a *penalty plus multipliers* (the augmented
//! Lagrangian), and the nonlinear-inequality family by an *exact-penalty merit
//! function with a geometry/acceptance test* (COBYLA). Four feasibility
//! mechanisms, and the traits share no operation beyond `lower()` / `upper()`
//! resp. `a()` / `b()` resp. `constraints()`, so a one-member hierarchy would
//! still be pure overhead.
//!
//! [`LinearConstraints`] is a different beast: a standalone *aggregator* of
//! the general linearly-constrained problem (box bounds, linear equalities,
//! and linear inequalities at once), consumed by [`Lincoa`](crate::Lincoa),
//! which folds all three kinds into one `A x ≤ b` system handled by one
//! active-set feasibility mechanism. It is *not* a parent of the three
//! siblings above — it neither extends them nor is extended by them — so the
//! "no `Constraint` parent" decision still stands (see
//! [`LinearConstraints`] and `.claude/rules/constraints.md`).

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

/// Nonlinear inequality constraints `c(x) ≤ 0` for an arbitrary vector-valued
/// constraint function `c : ℝⁿ → ℝᵐ`.
///
/// `x` is feasible iff every component satisfies `cᵢ(x) ≤ 0`; the *constraint
/// violation* is `[ maxᵢ cᵢ(x) ]₊` (zero when feasible). Unlike the three
/// linear sibling traits, the constraint here is a **function** evaluated at the
/// iterate, not matrix/vector data — so this trait carries a
/// [`constraints`](Self::constraints) evaluator rather than `a()` / `b()`
/// accessors. Like the siblings, the constraints live on the *problem* side
/// (tenet 4 in `CONTRIBUTING.md`): the derivative-free solver that handles them
/// ([`Cobyla`](crate::Cobyla)) binds on this trait, so handing it an
/// unconstrained problem is a compile error.
///
/// `NonlinearInequalityConstraints` is a supertrait of [`CostFunction`] so the
/// `Param` / `Error` types are shared automatically.
///
/// # Sign convention
///
/// The feasible direction is `cᵢ(x) ≤ 0`, matching the linear-inequality
/// sibling's `A x ≤ b` and PRIMA's modern COBYLA interface
/// (`min F(x) s.t. constr(x) ≤ 0`). Powell's 1994 paper writes the constraints
/// as `cᵢ(x) ≥ 0`; negate to convert (`basin`'s `cᵢ` is Powell's `−cᵢ`). An
/// equality `g(x) = 0` is expressed as the pair `g(x) ≤ 0`, `−g(x) ≤ 0`.
///
/// # Shapes
///
/// The iterate lives in `ℝⁿ` ([`Param`](CostFunction::Param)); the value
/// returned by [`constraints`](Self::constraints) lives in `ℝᵐ` — one entry per
/// constraint, length [`num_constraints`](Self::num_constraints) — but shares
/// the parameter's vector *type*. `m` and `n` need not match.
///
/// # Future direction: a `NonlinearConstraints` aggregator (deferred)
///
/// This trait is the *single-kind* surface (nonlinear inequalities only),
/// matching Powell's 1994 paper. PRIMA's modern COBYLA additionally folds
/// linear inequalities, linear equalities, and box bounds into the same
/// `constr(x) ≤ 0` vector. A planned `NonlinearConstraints` *aggregator*
/// (analogous to [`LinearConstraints`], which does this for the linear kinds)
/// will expose the nonlinear block plus optional linear/box blocks and fold
/// them together — letting a problem hand COBYLA all four constraint kinds at
/// once. That is **deliberately deferred**, not foreclosed: like
/// [`LinearConstraints`] it will be *standalone* (not a parent of this trait,
/// no blanket bridge — a blanket impl could only forward the nonlinear block
/// and would silently drop the linear/box data), so adding it later is purely
/// additive and breaks nothing here. This single-kind trait remains the right
/// surface for the inequality-only consumer.
///
/// # Examples
///
/// A problem feasible inside the unit disk, `x₁² + x₂² ≤ 1`, written as the
/// single constraint `c(x) = x₁² + x₂² − 1 ≤ 0`:
///
/// ```
/// use basin::{CostFunction, NonlinearInequalityConstraints};
///
/// struct InDisk;
/// impl CostFunction for InDisk {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
///         Ok(x[0] * x[1]) // minimize x1*x2 over the disk
///     }
/// }
/// impl NonlinearInequalityConstraints for InDisk {
///     fn constraints(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
///         Ok(vec![x[0] * x[0] + x[1] * x[1] - 1.0])
///     }
///     fn num_constraints(&self) -> usize { 1 }
/// }
///
/// let p = InDisk;
/// assert_eq!(p.constraints(&vec![0.0, 0.0]).unwrap(), vec![-1.0]); // interior: feasible
/// assert_eq!(p.num_constraints(), 1);
/// ```
pub trait NonlinearInequalityConstraints: CostFunction {
    /// Evaluate the constraint function `c(x) ∈ ℝᵐ`. The point is feasible iff
    /// every returned component is `≤ 0`; the constraint violation is
    /// `[ maxᵢ cᵢ(x) ]₊`. The returned vector has length
    /// [`num_constraints`](Self::num_constraints) and shares the parameter's
    /// vector type.
    fn constraints(&self, x: &Self::Param) -> Result<Self::Param, Self::Error>;

    /// The number of constraints `m` (length of the
    /// [`constraints`](Self::constraints) vector). Lets a solver size its
    /// per-constraint model storage before the first evaluation.
    fn num_constraints(&self) -> usize;
}

/// The general linearly-constrained problem: box bounds, linear equalities,
/// and linear inequalities together.
///
/// This is the constraint surface of [`Lincoa`](crate::Lincoa), which
/// minimizes a smooth objective over the feasible set
///
/// ```text
/// { x ∈ ℝⁿ :  lower ≤ x ≤ upper,   A_eq x = b_eq,   A_ineq x ≤ b_ineq }
/// ```
///
/// using only objective values. Every accessor is **optional** (defaulting to
/// `None`), so a problem implements only the blocks it actually has — an
/// inequality-only problem overrides just [`inequalities`](Self::inequalities),
/// a box-bounded one just [`lower`](Self::lower) / [`upper`](Self::upper), and
/// so on. LINCOA folds whatever is present into a single unit-normalized
/// `A x ≤ b` system (bounds as `±eᵢ` rows, an equality `aᵀx = β` as the pair
/// `aᵀx ≤ β`, `−aᵀx ≤ −β`) and keeps every iterate feasible with one
/// active-set projection.
///
/// # Not a parent of the sibling constraint traits
///
/// Despite the family-resembling name, `LinearConstraints` is **standalone**:
/// it is *not* a supertrait of [`LinearInequalityConstraints`] /
/// [`LinearEqualityConstraints`] / [`BoxConstraints`], and there are no
/// blanket impls bridging from them. A problem that wants LINCOA implements
/// `LinearConstraints` directly. This is deliberate — a blanket
/// `impl<P: LinearInequalityConstraints> LinearConstraints for P` could only
/// forward the inequality block, silently dropping any box/equality data the
/// problem also carries, and would block a manual impl by coherence. Keeping
/// it standalone makes the full general form expressible. The sibling traits
/// remain the right surface for their single-kind consumers (the log-barrier
/// [`BarrierMethod`](crate::solver::BarrierMethod) on
/// [`LinearInequalityConstraints`], the
/// [`AugmentedLagrangianMethod`](crate::solver::AugmentedLagrangianMethod) on
/// [`LinearEqualityConstraints`]); see `.claude/rules/constraints.md`.
///
/// # Shapes
///
/// The iterate lives in `ℝⁿ`. [`lower`](Self::lower) / [`upper`](Self::upper)
/// share the parameter type and length `n`; non-finite entries mean that
/// coordinate is unbounded on that side. Each linear block returns its matrix
/// paired with its right-hand side `b ∈ ℝᵐ` (one entry per row), where `m` is
/// that block's own row count — `m` need not match `n` or the other block's
/// row count. The matrix and its `b` are returned **together** as a tuple so
/// they cannot fall out of sync.
///
/// # Matrix type and consumers
///
/// Like the sibling traits, this leaves [`Matrix`](Self::Matrix) free of math
/// bounds; the consumer adds what it needs. LINCOA reads each constraint
/// normal as `Aᵀ eⱼ`, so it requires `Matrix: MatTransposeVec<Param>` (never a
/// linear solve), which every backend's matrix type provides (`DenseMatrix`
/// for `Vec<f64>`, `Array2`, `DMatrix`, `Mat`) — so LINCOA works on all four.
///
/// # Examples
///
/// A box-bounded problem (no matrix blocks) handed to LINCOA:
///
/// ```
/// use basin::CostFunction;
/// use basin::core::constraint::LinearConstraints;
///
/// struct BoundedSphere { lower: Vec<f64>, upper: Vec<f64> }
/// impl CostFunction for BoundedSphere {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
///         Ok(x.iter().map(|xi| xi * xi).sum())
///     }
/// }
/// impl LinearConstraints for BoundedSphere {
///     type Matrix = basin::DenseMatrix<f64>; // unused: no equality/inequality blocks
///     fn lower(&self) -> Option<&Vec<f64>> { Some(&self.lower) }
///     fn upper(&self) -> Option<&Vec<f64>> { Some(&self.upper) }
/// }
///
/// let p = BoundedSphere { lower: vec![0.5, 0.5], upper: vec![2.0, 2.0] };
/// assert_eq!(p.lower(), Some(&vec![0.5, 0.5]));
/// assert!(p.inequalities().is_none());
/// ```
pub trait LinearConstraints: CostFunction {
    /// The constraint-matrix type for the equality / inequality blocks.
    /// LINCOA bounds it on
    /// [`MatTransposeVec<Param>`](crate::core::math::MatTransposeVec) (each
    /// constraint normal is `Aᵀ eⱼ`); never a linear solve.
    type Matrix;

    /// Linear inequalities `A_ineq x ≤ b_ineq` as `(A_ineq, b_ineq)`, or
    /// `None` (the default) when the problem has no inequality constraints.
    fn inequalities(&self) -> Option<(&Self::Matrix, &Self::Param)> {
        None
    }

    /// Linear equalities `A_eq x = b_eq` as `(A_eq, b_eq)`, or `None` (the
    /// default) when the problem has no equality constraints.
    ///
    /// Each equality folds to the inequality pair `aᵀx ≤ β`, `−aᵀx ≤ −β`. If the
    /// start point is *not* on the equality (`aᵀx0 ≠ β`), both folded rows are
    /// relaxed independently to keep `x0` feasible, so the equality is widened to
    /// the slab between `β` and `aᵀx0` rather than re-projected onto the line
    /// (Powell's behavior). Start on the equality for it to bind exactly.
    fn equalities(&self) -> Option<(&Self::Matrix, &Self::Param)> {
        None
    }

    /// Element-wise lower bound on the iterate (length `n`), or `None` (the
    /// default) when unbounded below. Non-finite entries leave that coordinate
    /// unbounded below.
    fn lower(&self) -> Option<&Self::Param> {
        None
    }

    /// Element-wise upper bound on the iterate (length `n`), or `None` (the
    /// default) when unbounded above. Non-finite entries leave that coordinate
    /// unbounded above.
    fn upper(&self) -> Option<&Self::Param> {
        None
    }
}
