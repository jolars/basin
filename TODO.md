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

- [x] **General trust-region Newton solver (the `Hessian`-trait consumer).**
  BUILT (branch `trust-region`): public `TrustRegion<Sub, F>` over
  `BasicState`, consuming the `Hessian` trait: the second-order solver that
  trait + `FiniteDiff` were added ahead of. Modeled on argmin's
  `trustregion` structure (outer loop + pluggable subproblem), anchored to
  Nocedal & Wright Ch. 4 (Algorithm 4.1) and the in-repo TRSAPP truncated-CG
  (NEWUOA), **not** a port. (gomez's TrustRegion was rejected as a model:
  it's a nonlinear-systems `f(x)=0` root-finder, overlapping LM and `Trf`,
  not a general minimizer.) Trust radius δ lives in the solver struct (LM's
  `mu`/`nu` precedent), not on state; one Hessian per outer iteration is
  reused across an LM-style inner shrink loop (`with_max_inner_attempts`),
  so rejected steps re-solve with smaller δ at zero extra derivative evals.
  New `pub(crate)` `(g, B)` subproblem seam (`Subproblem` trait + `Step`),
  distinct from the Powell `QuadraticModel` `TrustRegionSubproblem` seam
  (different model type); shared `model_decrease`/`tau_to_boundary` helpers.
  **v1 subproblems (all shipped):** `Steihaug` (matrix-free, `MatVec` only,
  all backends, wasm-clean; the default), `Dogleg` (Cholesky Newton step via
  `LinearSolveSpd` + Cauchy fallback on indefinite B, all backends but
  ndarray), `CauchyPoint` (closed-form baseline, universal). v1 forms full B
  once per accepted iterate (Dogleg needs it anyway); DONE post-v1: the
  matrix-free `HessianProduct` problem trait + `TrustRegion` `MatrixFree`
  mode (`TrustRegion::matrix_free()`, mode marker type param,
  `SubproblemHvp` seam for Steihaug/CauchyPoint, counted
  `hessian_product_evals`, `FiniteDiff` synthesis per N&W eq. 8.20), so
  Steihaug needn't form B. Tests: Cauchy/Steihaug/Dogleg on quadratic +
  Rosenbrock (analytic `DenseMatrix` Hessian), f32 round-trip, `FiniteDiff`
  central-difference Hessian on nalgebra Rosenbrock. Web catalog + Backends
  notes updated.
  - **Deferred: Moré-Sorensen exact subproblem step.** The near-exact global
    solve (secular equation, hard case) via `SymmetricEigen` + Cholesky is out
    of v1 scope. It would require ingesting Moré & Sorensen (1983), "Computing a
    Trust Region Step" (paper-anchored, per the no-invention tenet). Add it as a
    fourth subproblem strategy once a consumer wants the extra robustness; until
    then v1's three iterative or closed-form strategies cover the common cases.

See `CONTRIBUTING.md` for the design tenets and constraints that shape these
decisions.
