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

- [x] **Made `BarrierMethod`/`AugmentedLagrangianMethod`
  inner-solver-agnostic.** Both now bound
  `So: WarmStart<V> + for<'a> Solver<Adapter<'a, P>, So::State>` with
  `So::State: GradientState<Param=V>` (was hard-wired to `BasicState<V>` ⇒
  `GradientDescent` only). The state-seeding primitive is the new
  `WarmStart<V>` trait (`src/core/inner.rs`):
  `type State: State<Param=V>; fn seed(&self, x: &V) -> State`. The σ-free
  `seed` resolved the "open design wrinkle": option (b):
  `MemeticInner<V>: WarmStart<V>` now extends it, adding the
  CMA-flavored `seed_scaled(x, σ)` (defaults to `seed`; only Nelder-Mead
  overrides) + `work_units`; `CmaInject`/`BoundedCmaInject` call sites
  switched `seed(x, σ)` → `seed_scaled(x, σ)`. The barrier and AL methods read
  `cost_evals`/`gradient_evals` off `So::State: GradientState` directly, so
  they need neither `seed_scaled` nor `work_units`. Shipped `WarmStart`
  impls: `GradientDescent`, `BFGS`, mode-generic `LBFGS` (covers `LBFGSB` +
  unbounded), plus the split-out `NelderMead`/`LevenbergMarquardt` impls.
  The two non-fixes held: **least-squares inners** (LM/Gauss-Newton/`Trf`)
  are excluded automatically: the adapters expose
  `CostFunction + Gradient`, not `Residual + Jacobian` (a barrier or Lagrangian
  is not a sum of squares); **derivative-free inners** (Nelder-Mead) are
  excluded by the `GradientState` bound, which is also exactly why the
  σ-free `seed` is the right thing (the only σ-sensitive inner can't reach
  the barrier and AL). Tests: `tests/barrier_method_nalgebra.rs` (`BFGS` +
  `Backtracking` inner, Armijo respects the `+∞` wall) and
  `tests/augmented_lagrangian_nalgebra.rs` (`BFGS` and unbounded `LBFGS`
  inners) prove a non-`BasicState` inner converges to the same optimum as
  the `GradientDescent` inner.

- [ ] **Observer layer.** First slice shipped: `Observe<S>` trait (three
  defaulted infallible methods, generic over the minimum state shape per
  tenet 3), `ObserverMode::{Never, Always, Every(n)}`,
  `Executor::observe_with` builder, wired into `Stepper` so `observe_init`
  fires after `Solver::init`, `observe_iter` after each successful step
  (mode-gated), and `observe_final` on clean stop. No concrete observers
  ship; the trait + wiring is the meat. Remaining: a `BestCostState`
  extension trait (only `BasicSimplexState`/`BasicPopulationState` track
  best-so-far today) plus an `ObserverMode::NewBest` variant bound on it; a
  zero-dep starter set in core (`StoreBest`, `Report`) once `BestCostState`
  lands; satellite crates for heavier integrations
  (`basin-observer-tracing`, `basin-observer-slog`, eventually a
  TUI or spectator-style crate) per the repo-structure rule (features on
  `basin` for light deps, separate crate only when heavy or
  platform-specific). A `CheckpointWriter` observer (serialize `state` every
  N iters via `serde` + `bincode`, gated on the `serde` feature and
  `not(target_arch = "wasm32")`) belongs in the same followup: argmin
  ships checkpointing as a first-class executor concern, but for basin "save
  the iterate periodically so a new run can warm-start" is exactly an
  observer's job; resume just deserializes into the initial state, no
  framework support needed. Keep observers strictly read-only: problem
  transformers (gradient clipping etc.) stay as problem-adapter wrappers,
  not observer hooks, mirroring how constraints attach problem-side.

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
  anyway); a matrix-free `HessianProduct` trait so Steihaug needn't form B
  is a future additive extension. Tests: Cauchy/Steihaug/Dogleg on quadratic +
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

- [x] **Rustdoc the load-bearing invariants on public traits.** Done in S0.
  `# Contract` heading + `**Caller must:**`/`**Implementor must:**`
  bullets are the established convention; `#![warn(missing_docs)]` and
  `#![warn(rustdoc::broken_intra_doc_links)]` are on at the crate root.
  Filling in docs on items that hold no contract (struct fields, trivial
  constructors) is the open follow-up: those are the \~100 `missing_docs`
  warnings still surfaced by the lint.

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

- [x] **Workspace wasm build broken by `competitor-bench` transitive dep.**
  Resolved (2026-06-10) by setting
  `default-members = ["crates/basin", "crates/basin-wasm"]` in the
  workspace manifest, so the bare `cargo build`/`cargo build --target wasm32-unknown-unknown`
  builds only the two
  shippable, wasm-clean crates. `competitor-bench` (which links
  `levenberg-marquardt` → `getrandom 0.3`, with no wasm backend unless its
  `wasm_js` feature is enabled) stays a `members` entry, so `--workspace`
  clippy and tests still cover it; it's just out of the *default* build set. A
  `getrandom_backend` cfg in `.cargo/config.toml` was ruled out: the cfg
  alone doesn't satisfy getrandom 0.3 (the `wasm_js` backend also needs its
  feature, which a bench crate has no business pulling). CI gained a
  regression guard running the bare workspace wasm build (ci.yml `wasm` job,
  `default` matrix entry), which now also exercises `basin-wasm` on wasm.

- [x] **State construction surface: hide the state zoo behind the solver (or
  honest "no").** Resolved: added `Executor::from_start(problem, solver, x0)`
  (`core/executor.rs`), which seeds the solver's natural state from a bare
  starting vector via the new public `InitialState<V>` trait (`core/inner.rs`).
  `WarmStart<V>: InitialState<V>` is now an empty marker for composition-safe
  inners, so non-inner solvers (TrustRegion, the Powell and MADS families,
  barrier and AL) are seedable without being treated as inners. The three-arg
  `Executor::new` stays as the explicit custom-seed form. CMA-ES (needs σ), the
  population GA, DE, and random search (sample the box), and the bracketing
  scalar solvers (Brent, golden-section) deliberately don't implement
  `InitialState`, so calling `from_start` on them is a compile error that
  directs callers to `new`. Also closed the BFGS `Vec` and faer seed gap
  (`WarmStart` was nalgebra-only), added the solver-to-state-to-`from_start`
  table to the crate-root rustdoc and a getting-started section in the web docs,
  and added round-trip, per-backend BFGS, and f32 tests in
  `tests/from_start_round_trip.rs`. The richer state taxonomy is kept on
  purpose: the capability traits (`GradientState`, `SimplexState`,
  `PopulationState`) are what termination criteria bind on (tenet 3), and the
  concrete state types carry real working memory (L-BFGS history, CMA
  covariance, NLLS counters, simplex vertices), so they are not collapsed into a
  single `IterState`. A further two-arg `Executor::new(problem, solver)` seeding
  from the problem's own `x0` was considered and not added: explicit
  construction is where `x0` lives and makes the solver-state pairing visible,
  so the explicit form stays the one the docs teach.

See `CONTRIBUTING.md` for the design tenets and constraints that shape these
decisions.
