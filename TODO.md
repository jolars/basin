# TODO

Ordered by recommended sequence: each item is easier or better-informed once the
previous lands.

## General design

- [ ] **Constraints (tenet 4).** Box bounds shipped (`BoxConstraints`, consumed
  by `ProjectedGradientDescent`/`LBFGSB`/`Trf`/`BoundedCmaEs`). Linear
  inequalities `A x ≤ b` shipped (`LinearInequalityConstraints` + the
  `LogBarrier` adapter + the log-barrier `BarrierMethod`, a
  `constrOptim`-style layer over an inner `GradientDescent`; all backends
  via `MatVec`/`MatTransposeVec`). Linear equalities `A x = b` shipped
  (`LinearEqualityConstraints` + the `AugmentedLagrangian` adapter +
  `AugmentedLagrangianMethod`, a penalty-plus-multiplier outer loop over an
  inner `GradientDescent`; tolerates infeasible starts, all backends via
  `MatVec`/`MatTransposeVec`). Both are now inner-solver-agnostic over
  gradient inners (`So: WarmStart<V>`, `So::State: GradientState`:
  `GradientDescent`/`BFGS`/unbounded `LBFGS`): see the completed
  inner-solver-agnostic item below. The backend gate is now lifted:
  `MatVec`/`MatTransposeVec` ship for every backend (`Vec<f64>` via the
  hand-rolled `DenseMatrix`, nalgebra, faer, and `ndarray` via `Array2`), so
  both methods run on the default backend with no external LA crate.
  Nonlinear inequalities `c(x) ≤ 0` shipped
  (`NonlinearInequalityConstraints`, consumed by `Cobyla` (the
  derivative-free COBYLA, Powell 1994, ported from PRIMA) via an L-infinity
  exact-penalty merit function, and by `Mads<Constrained>`
  (`Mads::constrained()`) via the *progressive barrier* (Audet & Dennis
  2009, an aggregate violation `h(x) = Σⱼ max(cⱼ, 0)²` and a threshold
  driven to zero around two incumbents; tolerates an infeasible start);
  function-valued so it needs only vector-tier ops, hence all backends +
  wasm). Remaining: phase-1 feasibility (the barrier needs a strictly
  feasible start today; the augmented Lagrangian, COBYLA, and the MADS
  progressive barrier do not); a framework-level `FeasibilityTolerance` once
  a 2nd nonlinear or equality-constrained solver justifies it (tenet 3);
  nonlinear *equality* constraints; and a `NonlinearConstraints` aggregator
  (nonlinear + linear + box, PRIMA's full COBYLA form): deferred-but-wanted,
  must be standalone like `LinearConstraints` (see
  `.claude/rules/constraints.md`). Keep deferring a `Constraint` supertrait:
  box (projection), linear-inequality (barrier), linear-equality
  (penalty+multipliers), and nonlinear-inequality (merit) still share no
  feasibility op beyond accessors (tenet 4).

See `CONTRIBUTING.md` for the design tenets and constraints that shape these
decisions.
