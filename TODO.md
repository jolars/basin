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
      Remaining: phase-1 feasibility (the barrier needs a strictly feasible
      start today; the augmented Lagrangian does not); a framework-level
      `FeasibilityTolerance` once a 2nd equality-constrained solver justifies it
      (tenet 3); nonlinear equality and nonlinear (in)equality constraints. Keep
      deferring a `Constraint` supertrait --- box (projection),
      linear-inequality (barrier), and linear-equality (penalty+multipliers)
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
- [ ] **Generalize over scalar (`f64` → `F: Float`).** Per the
      provisional-choices section in `AGENTS.md`. Neither the first stochastic
      solver (S7 `RandomSearch`) nor CMA-ES (S8) forced this --- both landed on
      `f64`, and the bound-boilerplate cost of preemptive generality still
      outweighs the refactor. Trigger now reads "a real f32 use case appears" or
      "a later stochastic solver needs it".

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
- [ ] **`ParamVec<F>`marker for solvers doing linear algebra on params.**
      Nelder-Mead needs `V: Clone + ScaledAdd<f64>`; gradient descent needs
      `V: ScaledAdd<f64>`; future solvers will repeat the bound pair. Add a
      blanket-impl marker like
      `trait ParamVec<F>: Clone + ScaledAdd<F> + NormSquared {}` once the third
      solver wants it --- premature with only two users.
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
      `WarmStart` note in `AGENTS.md` "Solver composition"). Remaining question:
      is there a shared `Composed` abstraction (coarser than `WarmStart` --- a
      "composed solver" marker), or do these memetic shapes genuinely share
      nothing beyond the three AGENTS.md composition contracts (+ `WarmStart`
      for some)? Resolve by either writing the trait or writing the honest "no"
      comment in `core/inner.rs`.
- [ ] **Retire `MemeticInner::work_units` (now dead).** No shipped solver calls
      it: `CmaInject` / `BoundedCmaInject` route inner evals through the shared
      `Problem<P>` wrapper, and `BasicPopulationState`'s `CountsMirror` folds
      inner work via `delta.total_work()`. The trait method + 4 impls in
      `solver/cma_inject.rs` (NM, LM, `LBFGSB`, `ClosureInner`) and the rustdoc
      selling it as the eval-aggregation channel (\~L53-58 / L70-75 / L283) can
      go; `ClosureInner::new`'s `work_fn` parameter likewise becomes vestigial.
      Cheap follow-up since the API is already `!`-tagged.
- [ ] **Workspace wasm build broken by `competitor-bench` transitive dep.**
      `cargo build --target wasm32-unknown-unknown` at the workspace level fails
      inside `levenberg-marquardt` (via `getrandom 0.3` needing
      `--cfg getrandom_backend="wasm_js"`). The library itself is clean ---
      `cargo build --target wasm32-unknown-unknown -p basin -p basin-wasm`
      succeeds --- so the wasm tenet holds for the two crates that need it.
      Options: exclude `competitor-bench` from the wasm CI step, gate its
      `levenberg-marquardt` dep behind a non-default feature, or set the
      `getrandom_backend` cfg in `.cargo/config.toml`.

See `AGENTS.md` for the design tenets and constraints that shape these
decisions.
