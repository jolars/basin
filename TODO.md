# TODO

Ordered by recommended sequence: each item is easier or better-informed once
the previous lands.

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
  hand-rolled `DenseMatrix`, nalgebra, faer, and `ndarray` via `Array2`),
  so both methods run on the default backend with no external LA crate.
  Nonlinear inequalities `c(x) ≤ 0` shipped
  (`NonlinearInequalityConstraints`, consumed by `Cobyla` (the
  derivative-free COBYLA, Powell 1994, ported from PRIMA) via an
  L-infinity exact-penalty merit function, and by `Mads<Constrained>`
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
  `.claude/rules/constraints.md`). Keep deferring a `Constraint` supertrait: box
  (projection), linear-inequality (barrier), linear-equality
  (penalty+multipliers), and nonlinear-inequality (merit) still share no
  feasibility op beyond accessors (tenet 4).

- [ ] **Broaden backend coverage (tenet 5).** Ongoing: most solvers should run
  on most backends (`Vec<f64>`, nalgebra, ndarray, faer), gated only by
  honest implementability (`.claude/rules/backends.md`), not by which
  backend it is. The canonical per-solver record is the matrix in
  `web/src/routes/docs/solvers/+page.svx` plus each solver's "Backends" doc
  note: this entry is just the roadmap pointer. Recently landed: `BFGS`
  on `Vec<f64>` + faer + ndarray (now all four backends); `LBFGS`/`LBFGSB`
  on ndarray (now all four backends); `CmaEs`/`BoundedCmaEs` on `Vec<f64>`
  via the pure-Rust cyclic-Jacobi eigensolver (`dense_eig.rs`);
  `CmaEs`/`BoundedCmaEs` on ndarray (same Jacobi solver wired through
  `as_standard_layout()` on `Array2`); the least-squares family
  (`GaussNewton`/`LevenbergMarquardt`/`Trf`) on ndarray, plus `Trf` on
  `Vec<f64>` (now all four backends each): the whole family is the
  normal-equations path (`JᵀJ` via `GramMatrix` + a pure-Rust Cholesky
  `LinearSolveSpd`, the same `dense_chol` reused on `Array2` through
  `as_standard_layout()`, with `AddDiagonalVectorInPlace`/`MaxDiagonal`
  for the LM/Trf damping), *not* QR, so no `LinearSolveLstsq` was needed.
  Remaining honest (pure-Rust, no BLAS) gaps: the memetic family
  (`CmaInject`, `BoundedCmaInject`, `MaLsChCma`) compiles and runs on all four
  backends already (the `SymmetricEigen` matrix bound resolves everywhere and
  the shipped `MemeticInner` inners are backend-generic); only the `Vec<f64>`
  and ndarray integration tests are still missing. No permanent (BLAS-only)
  gaps recorded yet.

- [x] **General trust-region Newton solver (the `Hessian`-trait consumer).**
  BUILT (branch `trust-region`): public `TrustRegion<Sub, F>` over
  `BasicState`, consuming the `Hessian` trait: the second-order solver
  that trait + `FiniteDiff` were added ahead of. Modeled on argmin's
  `trustregion` structure (outer loop + pluggable subproblem), anchored to
  Nocedal & Wright Ch. 4 (Algorithm 4.1) and the in-repo TRSAPP truncated-CG
  (NEWUOA), **not** a port. (gomez's TrustRegion was rejected as a model:
  it's a nonlinear-systems `f(x)=0` root-finder, overlapping LM and `Trf`, not a
  general minimizer.) Trust radius δ lives in the solver struct (LM's
  `mu`/`nu` precedent), not on state; one Hessian per outer iteration is
  reused across an LM-style inner shrink loop (`with_max_inner_attempts`),
  so rejected steps re-solve with smaller δ at zero extra derivative evals.
  New `pub(crate)` `(g, B)` subproblem seam (`Subproblem` trait + `Step`),
  distinct from the Powell `QuadraticModel` `TrustRegionSubproblem` seam
  (different model type); shared `model_decrease`/`tau_to_boundary`
  helpers. **v1 subproblems (all shipped):** `Steihaug` (matrix-free,
  `MatVec` only, all backends, wasm-clean; the default), `Dogleg`
  (Cholesky Newton step via `LinearSolveSpd` + Cauchy fallback on indefinite
  B, all backends but ndarray), `CauchyPoint` (closed-form baseline,
  universal). v1 forms full B once per accepted iterate (Dogleg needs it
  anyway); DONE post-v1: the matrix-free `HessianProduct` problem trait +
  `TrustRegion` `MatrixFree` mode (`TrustRegion::matrix_free()`, mode marker
  type param, `SubproblemHvp` seam for Steihaug/CauchyPoint, counted
  `hessian_product_evals`, `FiniteDiff` synthesis per N&W eq. 8.20), so
  Steihaug needn't form B. Tests: Cauchy/Steihaug/Dogleg on quadratic +
  Rosenbrock (analytic `DenseMatrix` Hessian), f32 round-trip, `FiniteDiff`
  central-difference Hessian on nalgebra Rosenbrock. Web catalog +
  Backends notes updated.
  - **Deferred: Moré-Sorensen exact subproblem step.** The near-exact global
    solve (secular equation, hard case) via `SymmetricEigen` + Cholesky is out
    of v1 scope. It would require ingesting Moré & Sorensen (1983), "Computing a
    Trust Region Step" (paper-anchored, per the no-invention tenet). Add it as a
    fourth subproblem strategy once a consumer wants the extra robustness; until
    then v1's three iterative or closed-form strategies cover the common cases.

## Cleanup and design debt (review notes)

Surfaced while implementing the termination layer. Not blocking, but each gets
harder to fix as more code piles on.

- [ ] **Unified `Composed<Outer, Inner>` abstraction (or honest "no").** Two
  concrete memetic shapes exist: `CmaInject` and `BoundedCmaInject`
  (per-generation top-k polish via `MemeticInner`, S11 + S13) and `MaLsChCma`
  (per-individual persistent LS chains, S12). `MemeticInner` covers
  CMA-injection composition but not MA-LSCh's persistent-state shape. The
  state-seeding slice is already extracted as `WarmStart<V>` (`core/inner.rs`),
  shared by the barrier and AL family and (via `MemeticInner: WarmStart`)
  CMA-injection. It is not a `Composed` abstraction: it says nothing about the
  outer loop, eval routing, or failure bubbling (see the `WarmStart` note in
  `CONTRIBUTING.md` "Solver composition"). Open question: is there a shared
  `Composed` marker coarser than `WarmStart`, or do these shapes share nothing
  beyond the three composition contracts? Resolve by writing the trait or
  writing the "no" comment in `core/inner.rs`.

See `CONTRIBUTING.md` for the design tenets and constraints that shape these
decisions.
