# TODO

Ordered by recommended sequence.

## General design

- [ ] **Add the full-form `NonlinearConstraints` aggregator (tenet 4).** Model
  PRIMA's full COBYLA input by folding nonlinear inequalities, optional linear
  inequalities and equalities, and optional box bounds into one `c(x) ≤ 0`
  vector. Keep the trait standalone like `LinearConstraints`: it must not be a
  parent of the sibling constraint traits, and blanket bridges must not silently
  discard constraint blocks. Preserve the existing
  `NonlinearInequalityConstraints` API, all four backends, and wasm support.

## Deferred design

- [ ] **Revisit a shared constraint-violation capability (tenet 3).** COBYLA and
  constrained MADS now provide multiple consumers, but they use different
  violation measures, and only `ConstrainedMadsState` exposes its measure.
  Define common state semantics before adding a reporting API or a composite
  feasibility-and-optimality stopping rule. Do not add a standalone
  `FeasibilityTolerance`: executor criteria are combined with OR, so it could
  stop at the first feasible but nonoptimal iterate.

- [ ] **Design nonlinear equality constraints when a solver needs their
  structure (tenet 4).** For now, represent `g(x) = 0` as the pair `g(x) ≤ 0`
  and `−g(x) ≤ 0`. Do not add a dedicated trait without a consumer that can
  validate equality-specific operations and semantics.

See `CONTRIBUTING.md` for the design tenets and constraints that shape these
decisions.
