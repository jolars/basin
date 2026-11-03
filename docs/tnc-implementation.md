# TNC implementation notes

The public contract and current settings live in
[`Tnc`'s rustdoc](../crates/basin/src/solver/tnc.rs). This note records the
implementation history and the changes made for Basin.

## Original implementation

The original TNC implementation is published in
[`sisyfuzz` 0.1.0](https://docs.rs/crate/sisyfuzz/0.1.0/source/src/tnc.rs)
under Apache-2.0.

The original method was independently written in Rust using preconditioned
conjugate gradients, Hessian-vector products, negative-curvature truncation,
bound handling and shared line searches. It used `f64` and normalized unit-box
coordinates. The first version used diagonal BFGS preconditioning and a relative
CG residual tolerance of `0.1`.

Subsequent local experiments selected ten limited-memory BFGS pairs and residual
tolerance `0.5`. Exact Hessian-vector products and endpoint snapping were also
added before the Basin port. Those experiments motivated the defaults; they do
not establish that the choices are best for every problem.

## Changes for Basin

- **Solver lifecycle.** The port uses `Solver` and `FirstOrderState`, with typed
  callback errors and exact solver-and-state checkpoints. Fresh initialization
  resets history and reevaluates the projected seed.
- **Backend and scalar support.** Scalar and vector operations use Basin's
  capability traits, supporting both `f32` and `f64` across all four dense
  backends. No matrix backend or new dependency is required.
- **Coordinates and constraints.** TNC consumes a complete box-only problem
  declaration and works in the supplied finite box. Normalization is an
  explicit `ParameterTransform` adapter, including linear and logarithmic maps
  and the gradient and Hessian-vector chain rules. Stationarity is measured in
  the supplied coordinates.
- **Curvature.** The limited-memory defaults are retained; memory zero selects
  identity preconditioning. The prototype's diagonal and hybrid options were
  not ported. Gradient differences are the default; `.exact_products()` requires
  the problem's exact callback and propagates failures without a numerical
  fallback.
- **Line search.** The port uses Basin's bounded `LineSearch` interface and
  defaults to `Wolfe`. Feasible step limits are checked using backend arithmetic.
  A snapped endpoint is reevaluated if it differs from a retained probe. TNC
  checks accepted evaluations for feasibility, finite data and nonincreasing
  cost; strong-Wolfe conditions are not guaranteed at every accepted step.
- **Accounting and execution controls.** Evaluations pass through `Problem`.
  Internal transform callbacks follow Basin's adapter-level counting contract.
  Cancellation and evaluation limits use Basin's between-step semantics,
  replacing the prototype's checks inside the inner solve and callbacks.

## References and validation

Stephen G. Nash, "A survey of truncated-Newton methods," *Journal of Computational
and Applied Mathematics* 124 (2000), 45–59,
[doi:10.1016/S0377-0427(00)00426-X](https://doi.org/10.1016/S0377-0427(00)00426-X),
describes the algorithm family, rather than this exact combination of defaults
and safeguards.

SciPy's TNC documentation and implementation behavior provided practical
context and a benchmark reference for the original implementation. Comparisons
used [SciPy 1.17.0](https://docs.scipy.org/doc/scipy-1.17.0/reference/optimize.minimize-tnc.html),
including a matched CG iteration cap. Neither the original implementation nor
the Basin port translates SciPy's C code. Different line searches, scaling and
stopping rules preclude a claim of trajectory or callback-count parity.

The committed tests use analytic solutions and numerical invariants rather than
external trajectories. See the [TNC validation notes](../crates/basin/tests/fixtures/README.md#tnc-analytic-validation)
for coverage and oracle choices. Historical benchmark comparisons are not
performance measurements of the final Basin port.
