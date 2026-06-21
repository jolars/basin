# TODO

Ordered by recommended sequence --- each item is easier or better-informed once
the previous lands.

## General design

- [ ] **Constraints (tenet 4).** Box bounds shipped (`BoxConstraints`, consumed
      by `ProjectedGradientDescent` / `LBFGSB` / `Trf` / `BoundedCmaEs`). Linear
      inequalities `A x ≤ b` shipped (`LinearInequalityConstraints` + the
      `LogBarrier` adapter + the log-barrier `BarrierMethod`, a
      `constrOptim`-style layer over an inner `GradientDescent`; all backends
      via `MatVec`/`MatTransposeVec`). Linear equalities `A x = b` shipped
      (`LinearEqualityConstraints` + the `AugmentedLagrangian` adapter +
      `AugmentedLagrangianMethod`, a penalty-plus-multiplier outer loop over an
      inner `GradientDescent`; tolerates infeasible starts, all backends via
      `MatVec`/`MatTransposeVec`). Both are now inner-solver-agnostic over
      gradient inners (`So:       WarmStart<V>`, `So::State: GradientState`:
      `GradientDescent`/`BFGS`/ unbounded `LBFGS`) --- see the completed
      inner-solver-agnostic item below. The backend gate is now lifted:
      `MatVec`/`MatTransposeVec` ship for every backend --- `Vec<f64>` (via the
      hand-rolled `DenseMatrix`), nalgebra, faer, and `ndarray` (`Array2`) ---
      so both methods run on the default backend with no external LA crate.
      Nonlinear inequalities `c(x) ≤ 0` shipped
      (`NonlinearInequalityConstraints`, consumed directly by `Cobyla` --- the
      derivative-free COBYLA, Powell 1994, ported from PRIMA --- via an
      L-infinity exact-penalty merit function; function-valued so it needs only
      vector-tier ops, hence all backends + wasm). Remaining: phase-1
      feasibility (the barrier needs a strictly feasible start today; the
      augmented Lagrangian and COBYLA do not); a framework-level
      `FeasibilityTolerance` once a 2nd nonlinear/equality-constrained solver
      justifies it (tenet 3); nonlinear *equality* constraints; and a
      `NonlinearConstraints` aggregator (nonlinear + linear + box, PRIMA's full
      COBYLA form) --- deferred-but-wanted, must be standalone like
      `LinearConstraints` (see `.claude/rules/constraints.md`). Keep deferring a
      `Constraint` supertrait --- box (projection), linear-inequality (barrier),
      linear-equality (penalty+multipliers), and nonlinear-inequality (merit)
      still share no feasibility op beyond accessors (tenet 4).
- [ ] **Broaden backend coverage (tenet 5).** Ongoing: most solvers should run
      on most backends (`Vec<f64>`, nalgebra, ndarray, faer), gated only by
      honest implementability (`.claude/rules/backends.md`), not by which
      backend it is. The canonical per-solver record is the matrix in
      `web/src/routes/docs/solvers/+page.svx` plus each solver's "Backends" doc
      note --- this entry is just the roadmap pointer. Recently landed: `BFGS`
      on `Vec<f64>` + faer; `LBFGS`/`LBFGSB` on ndarray (now all four backends);
      `CmaEs`/`BoundedCmaEs` on `Vec<f64>` via the pure-Rust cyclic-Jacobi
      eigensolver (`dense_eig.rs`); `CmaEs`/`BoundedCmaEs` on ndarray (same
      Jacobi solver wired through `as_standard_layout()` on `Array2`). Remaining
      honest (pure-Rust, no BLAS) gaps: `BFGS` on ndarray (rank-one update ops
      on `Array2` --- the last `✗` in its row); the least-squares family
      (`GaussNewton`/`LevenbergMarquardt`/`Trf`) on `Vec<f64>` + ndarray (a
      pure-Rust `LinearSolveLstsq`/QR on `DenseMatrix` + `Array2`, explicitly
      blessed by the backends rule); the memetic family
      (`CmaInject`/`BoundedCmaInject`/`MaLsChCma`) on `Vec<f64>` + ndarray ---
      now that the CMA family covers both backends, the matrix bounds resolve on
      `Array2<f64>`; ndarray coverage just needs wiring tests + a
      `MemeticInner<Array1<f64>>` inner choice. While there, fix the stale
      "Backends" notes in `cma_inject.rs` (\~L286) and `bounded_cma_inject.rs`
      (\~L52): both claim "`Vec<f64>` and `ndarray` produce a compile-time
      error" while also saying "Same coverage as `CmaEs`" --- the Vec<f64> half
      was already wrong, and ndarray now resolves too. No permanent (BLAS-only)
      gaps recorded yet.
- [x] **Made `BarrierMethod` / `AugmentedLagrangianMethod`
      inner-solver-agnostic.** Both now bound
      `So: WarmStart<V> + for<'a> Solver<Adapter<'a, P>,       So::State>` with
      `So::State: GradientState<Param=V>` (was hard-wired to `BasicState<V>` ⇒
      `GradientDescent` only). The state-seeding primitive is the new
      `WarmStart<V>` trait (`src/core/inner.rs`):
      `type State: State<Param=V>; fn seed(&self, x: &V) -> State`. The σ-free
      `seed` resolved the "open design wrinkle" --- option (b):
      `MemeticInner<V>:       WarmStart<V>` now extends it, adding the
      CMA-flavored `seed_scaled(x, σ)` (defaults to `seed`; only Nelder-Mead
      overrides) + `work_units`; `CmaInject`/`BoundedCmaInject` call sites
      switched `seed(x, σ)` → `seed_scaled(x, σ)`. The barrier/AL methods read
      `cost_evals`/ `gradient_evals` off `So::State: GradientState` directly, so
      they need neither `seed_scaled` nor `work_units`. Shipped `WarmStart`
      impls: `GradientDescent`, `BFGS`, mode-generic `LBFGS` (covers `LBFGSB` +
      unbounded), plus the split-out `NelderMead`/`LevenbergMarquardt` impls.
      The two non-fixes held: **least-squares inners** (LM/Gauss-Newton/`Trf`)
      are excluded automatically --- the adapters expose
      `CostFunction + Gradient`, not `Residual + Jacobian` (a barrier/Lagrangian
      is not a sum of squares); **derivative-free inners** (Nelder-Mead) are
      excluded by the `GradientState` bound, which is also exactly why the
      σ-free `seed` is the right thing (the only σ-sensitive inner can't reach
      the barrier/AL). Tests: `tests/barrier_method_nalgebra.rs` (`BFGS` +
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
      extension trait (only `BasicSimplexState` / `BasicPopulationState` track
      best-so-far today) plus an `ObserverMode::NewBest` variant bound on it; a
      zero-dep starter set in core (`StoreBest`, `Report`) once `BestCostState`
      lands; satellite crates for heavier integrations
      (`basin-observer-tracing`, `basin-observer-slog`, eventually a
      TUI/spectator-style crate) per the repo-structure rule (features on
      `basin` for light deps, separate crate only when heavy or
      platform-specific). A `CheckpointWriter` observer (serialize `state` every
      N iters via `serde` + `bincode`, gated on the `serde` feature and
      `not(target_arch = "wasm32")`) belongs in the same followup --- argmin
      ships checkpointing as a first-class executor concern, but for basin "save
      the iterate periodically so a new run can warm-start" is exactly an
      observer's job; resume just deserializes into the initial state, no
      framework support needed. Keep observers strictly read-only --- problem
      transformers (gradient clipping etc.) stay as problem-adapter wrappers,
      not observer hooks, mirroring how constraints attach problem-side.

- [ ] **Powell-family model-based DFO: NEWUOA → BOBYQA → LINCOA.** Add Powell's
      least-Frobenius-norm, trust-region, model-based derivative-free solvers.
      NEWUOA (unconstrained) first; BOBYQA (box) and LINCOA (linear) reuse a
      shared `QuadraticModel` core (interpolation set + `Γ`/`γ` model + factored
      inverse-KKT `H` via `Ξ`/`Υ`/`Ω` + the least-Frobenius `H`-update + ρ/Δ
      schedule) and only swap the trust-region subproblem (TRSAPP → TRSBOX →
      projected-Krylov) --- which maps onto tenet 4 (BOBYQA = `BoxConstraints`,
      LINCOA = `LinearInequalityConstraints`, both already modeled). All five
      Powell papers are ingested under `references/` (math in `source.marker.md`
      + per-paper `NOTES.md`). **Full plan, design seam, basin-infra reuse,
      build order, and PRIMA/NLopt validation strategy:
      `docs/newuoa-roadmap.md`.** Paper-anchored --- implement from the papers,
      cross-check against PRIMA.

## Cleanup / design debt (review notes)

Surfaced while implementing the termination layer. Not blocking, but each gets
harder to fix as more code piles on.

- [x] **Rustdoc the load-bearing invariants on public traits.** Done in S0.
      `# Contract` heading + `**Caller must:**` / `**Implementor must:**`
      bullets are the established convention; `#![warn(missing_docs)]` and
      `#![warn(rustdoc::broken_intra_doc_links)]` are on at the crate root.
      Filling in docs on items that hold no contract (struct fields, trivial
      constructors) is the open follow-up --- those are the \~100 `missing_docs`
      warnings still surfaced by the lint.
- [ ] **Unified `Composed<Outer, Inner>` abstraction (or honest "no").** Two
      concrete memetic shapes now exist: `CmaInject` / `BoundedCmaInject`
      (per-generation top-k polish via `MemeticInner`, S11 + S13) and
      `MaLsChCma` (per-individual persistent LS chains, S12). The `MemeticInner`
      trait covers CMA-injection-style composition but doesn't model MA-LSCh's
      persistent-state shape. Partial progress: the *state-seeding* slice was
      extracted as `WarmStart<V>` (`core/inner.rs`), now shared by the
      barrier/AL family and (via `MemeticInner: WarmStart`) CMA-injection ---
      but that is explicitly **not** a `Composed` abstraction (it says nothing
      about the outer loop, eval routing, or failure bubbling; see the
      `WarmStart` note in `CONTRIBUTING.md` "Solver composition"). Remaining
      question: is there a shared `Composed` abstraction (coarser than
      `WarmStart` --- a "composed solver" marker), or do these memetic shapes
      genuinely share nothing beyond the three CONTRIBUTING.md composition
      contracts (+ `WarmStart` for some)? Resolve by either writing the trait or
      writing the honest "no" comment in `core/inner.rs`.
- [x] **Workspace wasm build broken by `competitor-bench` transitive dep.**
      Resolved (2026-06-10) by setting `default-members = ["crates/basin",
      "crates/basin-wasm"]` in the workspace manifest, so the bare
      `cargo build` / `cargo build --target wasm32-unknown-unknown` builds only
      the two shippable, wasm-clean crates. `competitor-bench` (which links
      `levenberg-marquardt` → `getrandom 0.3`, with no wasm backend unless its
      `wasm_js` feature is enabled) stays a `members` entry, so `--workspace`
      clippy/tests still cover it; it's just out of the *default* build set. A
      `getrandom_backend` cfg in `.cargo/config.toml` was ruled out: the cfg
      alone doesn't satisfy getrandom 0.3 (the `wasm_js` backend also needs its
      feature, which a bench crate has no business pulling). CI gained a
      regression guard running the bare workspace wasm build (ci.yml `wasm` job,
      `default` matrix entry), which now also exercises `basin-wasm` on wasm.

See `CONTRIBUTING.md` for the design tenets and constraints that shape these
decisions.
