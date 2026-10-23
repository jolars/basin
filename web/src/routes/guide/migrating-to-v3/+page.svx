# Migrating to Basin 3.0

Basin 3.0 makes constraint declarations optional. This guide describes the
development API on `main`; Basin 2.0 remains the published stable release.
For an upgrade from Basin 1.x, first follow the
[Basin 2.0 migration guide](../migrating-to-v2/), then apply these changes.

## Plain objectives and residuals

Implement `CostFunction` and any derivatives your solver needs. A problem used
with unconstrained solvers no longer needs an empty constraint declaration.
Remove implementations whose only purpose was returning `NoConstraints::EMPTY`.
`Residual`, `NonlinearEqualityValues`, `NonlinearInequalityValues`, and
`ConstraintJacobian` also stand alone.

Unconstrained solvers now ignore declared constraints. Passing a problem with
box bounds to `GradientDescent` optimizes its objective without enforcing those
bounds. Use `ProjectedGradientDescent` or another compatible constrained solver
to enforce them. The same rule applies to unbounded solver modes and least
squares: a declaration alone does not make an algorithm respect constraints.

## Declare constraints directly

The former `ProblemConstraints` trait is now `Constraints`. Its associated type
is `Set`, and its getter is `constraints`. The former `Constraints` data container
is now `ConstraintSet`.

Add this implementation alongside your objective and derivative implementations:

```rust
use basin::{BoxBounds, ConstraintDescription, ConstraintSet, Constraints};

struct ShiftedSphere {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl Constraints for ShiftedSphere {
    type Set = ConstraintSet<BoxBounds<Vec<f64>>>;

    fn constraints(&self) -> <Self::Set as ConstraintDescription>::Borrowed<'_> {
        ConstraintSet::new().with_bounds(BoxBounds::new(
            Some(&self.lower),
            Some(&self.upper),
        ))
    }
}
```

The borrowed view still supports bounds, matrices, and right-hand sides stored
in independent fields. If you store the whole set, return
`self.constraints.borrowed()` instead. Builders preserve existing blocks and
change the set's type. Constrained solvers still require the complete declaration
and reject unsupported blocks; a box-only solver cannot consume additional
nonlinear constraints. Validation and callback error handling are unchanged.

## Attach a set to an existing objective

`Constrained::new(objective, set)` remains a convenience adapter. Its objective
needs no empty declaration, and callbacks can be implemented independently.
The supplied set becomes the complete declaration, replacing any declaration
on the inner objective. Combine blocks in one `ConstraintSet` before wrapping
when they must all be enforced.

`FiniteDiff`, `BoundedFiniteDiff`, and `RobustLeastSquares` continue forwarding
constraint declarations when their inner problem implements `Constraints`.
Their unconstrained evaluation paths also work without a declaration.

## Update generic bounds

`P: CostFunction` and `P: Residual` no longer imply constraint metadata. Add
`P: Constraints` wherever generic code reads `P::Set` or calls `constraints()`.
For example, replace a bound such as
`P: CostFunction<Constraints = NoConstraints>` with
`P: CostFunction + Constraints<Set = NoConstraints>` only when your code needs
that explicit restriction. Unconstrained solvers no longer require it.

Replace qualified calls to `ProblemConstraints::constraint_set` with
`Constraints::constraints`, and `Constraints::new()` on the old data container
with `ConstraintSet::new()`.

## Sampling and composed solvers

`SamplingBounds` no longer requires a `NoConstraints` declaration. Its box
controls sampling rather than feasibility. `SamplingDomain` still explicitly
reinterprets a box-constrained problem as a sampling domain.

`RelaxedInner` remains available for box-constrained compositions that need its
problem transformation and conversion to `ConstraintSolveError`. Existing
uses remain valid. Execution budgets, evaluation accounting, and state ownership
are unchanged. Constraint sets retain their serialized fields and layout; formats
that embed Rust struct names may need the new `ConstraintSet` name.
