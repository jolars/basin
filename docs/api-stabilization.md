# Pre-1.0 API stabilization audit

Status: **landed in 1.0.0.** This was the decision checklist that shaped basin's
1.0 surface, produced under an explicit "no new solvers or features unless they
probe what the API should be" constraint. It is kept as a historical record of
why the 1.0 API looks the way it does; the `[DECIDE]` / `[RECOMMEND]` / `[DO]` /
`[REVIEW]` tags below reflect each item's status *at audit time*, not open
questions.

A 1.0 release freezes the public trait/struct/enum surface into a compatibility
promise. The highest-value work right now is settling the *shape* of that
surface --- especially choices that are cheap today but become
major-version-bump breaking changes once 1.0 ships. This doc catalogs those
choices, each tagged:

- `[DECIDE]` --- needs a maintainer call; a recommendation is given.
- `[RECOMMEND]` --- clear recommendation, confirm and schedule.
- `[DO]` --- unambiguous, just needs scheduling.
- `[REVIEW]` --- needs an audit pass to produce the answer.

It was produced by internal review (trait surface, examples, tests) and a
feature comparison against argmin (Rust) and ensmallen (C++). Most
argmin/ensmallen gaps (serialization/checkpointing, observer metadata, callback
hooks) are *features* and out of scope; what survives here is the subset that
forces a freeze-now API decision, plus internal consistency issues.

Up-front decisions already taken:

- **Serialization / checkpointing: door open, deferred.** Not built for 1.0. The
  deliberate stance is that public states/solvers *should* be serde-able
  eventually; this doc verifies no 1.0 choice forecloses adding
  `#[derive(Serialize, Deserialize)]` behind a `serde` feature later (additive,
  non-breaking). See A3.
- **Probing method: internal review** (no downstream consumer pulled in).

Execution note: each lettered group (A / B / C) is an independently shippable,
PR-sized chunk once the `[DECIDE]` items are ratified.

--------------------------------------------------------------------------------

## A. Freeze-now decisions --- breaking if deferred past 1.0

### A1. `#[non_exhaustive]` on public enums `[DONE]`

*Highest leverage, nearly free.* No public enum used `#[non_exhaustive]` before
this change:

  | Enum                   | Location                  | Grows after 1.0?                                    |
  | ---------------------- | ------------------------- | --------------------------------------------------- |
  | `TerminationReason`    | `core/termination.rs:16`  | **Yes** — every new solver convergence/failure mode |
  | `SymmetricEigenError`  | `core/math/linalg.rs:423` | Likely (new failure modes)                          |
  | `LinearSolveError`     | `core/math/linalg.rs:506` | Likely (new failure modes)                          |
  | `ObserverMode`         | `core/observer.rs:106`    | Maybe (`NewBest`, closure filter)                   |
  | `StepOutcome`          | `core/executor.rs:113`    | Unlikely (Continue / Stopped is complete)           |
  | `Method` (finite-diff) | `core/numdiff.rs:75`      | Maybe (higher-order stencils)                       |
  | `Dimensionality`       | `problems/spec.rs:28`     | Unlikely                                            |

Frozen exhaustively, every addition to `TerminationReason` is a major bump ---
unacceptable for an enum that *will* grow as solvers are added.

**Decided:** `#[non_exhaustive]` applied to `TerminationReason`,
`SymmetricEigenError`, `LinearSolveError`, **and** `ObserverMode` and `Method`
--- the two `[DECIDE]` enums were resolved *yes*: both are config-style enums
users construct rather than match on, both have plausible future variants
(`NewBest` / closure filter; higher-order stencils), so the cost is effectively
nil and the door stays open. `StepOutcome` and `Dimensionality` were left
exhaustive (genuinely closed; exhaustive matching is a feature for callers).

Cost was trivial: the attribute plus one wildcard arm in the only external
exhaustive match (`reason_str` in `crates/basin-wasm/src/lib.rs`). All matches
inside `crates/basin` are unaffected (`non_exhaustive` is a no-op within the
defining crate).

### A2. Observer metadata / KV channel `[DONE — deferred]`

argmin passes a `KV` key-value store to observers so solvers can surface
algorithm-specific metrics (step size, population diversity, barrier μ). basin's
`Observe` (`core/observer.rs:68`) passes only `&S`.

**Decision: no KV channel for 1.0.** basin's design already says "state shape is
the contract" (tenet 3 --- criteria and observers bind on the minimum `State`/
`GradientState`/`SimplexState`/`PopulationState` shape). Algorithm-specific
metrics belong on a richer state trait an observer binds on, not a
stringly-typed side channel that erases the compile-time shape guarantee.
Recorded as a deliberate non-tenet choice in `AGENTS.md`.

**Correction to this item's original framing: A2 is *not* a true freeze-now /
one-way door.** The original draft claimed "adding a metadata argument to
`observe_iter` after 1.0 is breaking, so the choice must be made now." That is
only true for the naïve route of mutating the existing method's signature. A KV
channel can be added post-1.0 *additively*: add a new default-bodied trait
method
(`fn observe_iter_with(&mut self, state: &S, kv: &Kv) { self.observe_iter(state) }`),
switch the executor's internal call site (`executor.rs:239`) to it, and every
existing `Observe` impl keeps compiling via the forwarding default; a concrete
`Kv` type keeps the trait object-safe for the `Box<dyn Observe<S>>` storage. So
the door is *deferred open*, not closed --- the 1.0 commitment is purely "don't
build it now."

The genuine future motivation, if it appears: solver-internal working scalars
that don't fit a state trait (CMA-ES σ / covariance / evolution paths, LM μ / ν
/ diag --- see A3) live in the *solver* struct, not the state, so "expose it on
a richer state trait" does not cover them. Until then, keep `Observe`
infallible, read-only, and state-only.

### A3. serde door-open verification `[DONE]`

Deferred-but-not-foreclosed (per the up-front decision). The "serde-readiness"
audit below confirms the concrete public state structs **and the solver
structs** can gain `#[derive(Serialize, Deserialize)]` behind a future `serde`
feature without a breaking change. **Verdict: the door is open.** Findings (and
corrections to this item's original scoping) follow the rationale.

**Why solvers, not just states (the observer ≠ checkpointing point).** The
observer layer already covers the *save / monitor* half of what a checkpointing
system would do --- an `Observe` impl sees `&S` each iteration and can serialize
it. What it does **not** cover is *resume*: (1) the hook receives only the
state, not the solver, and (2) `Executor::run` always calls `Solver::init`, so
there is no load-and-continue entry. Crucially, basin does **not** keep all
iteration-carrying state in the state object --- `CmaEs` holds its
`Working<V, M, F>` (covariance, σ, mean, evolution paths) inside the *solver*
struct (`solver/cma_es.rs`), and `LevenbergMarquardt` holds `mu` / `nu` / `diag`
/ the Gram caches there too (`solver/levenberg_marquardt.rs`). So a state-only
snapshot is not a faithful resume point for those solvers. Conclusion: observers
replace a separate checkpointing system **iff basin never needs to resume an
interrupted run**; if resume is ever in scope (crash recovery on long fits;
pause/resume in the wasm/browser target), it needs serde on *both* state and
solver plus an Executor load path. That work is purely additive, hence safely
deferrable past 1.0 --- but the door-open check below must include solvers or
the door is quietly half-closed.

- **States to check:** `BasicState`, `BasicSimplexState`,
  `BasicPopulationState`, `QuasiNewtonState`, `LbfgsState` (`core/state.rs`,
  `core/state/lbfgs.rs`).
- **Solver structs to check** (carry resumable working state): `CmaEs` /
  `BoundedCmaEs` and their `Working` (`solver/cma_es.rs`); `LevenbergMarquardt`
  and its caches (`solver/levenberg_marquardt.rs`); `Bfgs`, `Lbfgs`,
  `GradientDescent` (incl. its `velocity`), `NelderMead`, `GaussNewton`, `Trf`,
  and the line searches they embed (`MoreThuente` / `Wolfe` carry scratch). The
  memetic/outer solvers (`CmaInject`, `BarrierMethod`,
  `AugmentedLagrangianMethod`) embed an inner solver + an `InnerExecutor`, so
  they inherit the `InnerExecutor` blocker below --- flag, don't try to resolve.
- **Known blockers to document (not fix):** `InnerExecutor` holds
  `Vec<Box<dyn TerminationCriterion>>` (`core/inner.rs`) --- not serializable,
  but it's a driver, not persisted state; a resume design would rebuild it
  rather than deserialize it. A future closure-filter `ObserverMode` variant
  would also block; another reason to keep `ObserverMode` as plain data (ties to
  A1).
- `PhantomData<fn() -> Mode>` type-state markers (`NelderMead`, `Lbfgs`) derive
  serde fine --- note for completeness.

**Also `[DO]` --- done.** `CONTRIBUTING.md` listed a `serde` feature in its
features enumeration ("nalgebra, ndarray, faer, serde, parallel, problems") that
**does not exist** in `crates/basin/Cargo.toml` (`[features]` defines only
`nalgebra`, `ndarray`, `ndarray-blas`, `faer`, `parallel`, `problems`). The
stray `serde` mention has been removed.

#### Serde-readiness findings

Method: field-level audit of every public state and solver struct, hunting for
fields whose type can *never* implement `Deserialize` (trait objects, fn
pointers, closures, references) --- those are the only thing that could
foreclose adding a derive later, since a generic field is auto-bounded by
serde's derive (`impl<T: Serialize> Serialize for Foo<T>`) and a derive is
otherwise purely additive.

**Serde-clean (no field blocks a future derive; generics auto-bound):**

- **All five states** --- `BasicState`, `BasicSimplexState`,
  `BasicPopulationState`, `QuasiNewtonState`, `LbfgsState` (incl. nested
  `LbfgsbWork`: only `Vec<F>` / `Vec<usize>` / `Vec<i8>` / scalars). Scalars,
  `Vec`, `Option`, and generic `V` / `M` / `F` only.
- **All four line searches** --- `Constant`, `Backtracking`, `Wolfe`,
  `MoreThuente`. *Correction:* these are stateless scalar config; the original
  list claimed "`MoreThuente` / `Wolfe` carry scratch" --- they do not. Their
  scratch (`stx` / `fx` / `brackt` / ...) is function-local, not struct fields.
- **Core working-state solvers** --- `CmaEs` / `BoundedCmaEs` (+ their
  `Working`, incl. `ChaCha8Rng` and `VecDeque<F>`), `LevenbergMarquardt`,
  `GaussNewton`, `Trf`. Serde-able given backend serde bounds; `ChaCha8Rng`
  needs `rand_chacha`'s own `serde` feature --- an additive dep-feature, not a
  foreclosure.
- **Solvers holding a generic line search** --- `Bfgs`, `Lbfgs`,
  `GradientDescent` (incl. `velocity: Option<V>`). The `line_search: S` field is
  a *generic param*, auto-bounded by serde's derive; **not** a blocker.
- **`PhantomData<fn() -> Mode>` type-state markers** (`NelderMead`, `Lbfgs`) ---
  serde impls `PhantomData<T>` for all `T`; worst case the derive adds a
  spurious `Mode: Serialize` bound, fixed with a one-line `#[serde(bound = "")]`
  or by deriving serde on the zero-size marker types. Additive either way.
- **`BarrierMethod`, `AugmentedLagrangianMethod`** --- serde-clean.
  *Correction:* the original list said all three outer solvers "embed an inner
  solver + an `InnerExecutor` ... inherit the `InnerExecutor` blocker." That is
  **wrong** for these two: they store `inner_solver: So` + scalar config and
  build their termination criteria with a fresh `Vec` per call via `run_loop`
  (they explicitly *sidestep* `InnerExecutor`). No trait-object field.

**True non-serde fields --- and why none foreclose the door:**

- **`InnerExecutor.criteria: Vec<Box<dyn TerminationCriterion<S>>>`**
  (`core/inner.rs`) --- trait object, not serde-able. It is a *driver*, not
  persisted state; a resume design rebuilds it. *Correction to blast radius:*
  the structs that actually **store** an `InnerExecutor` are the three injection
  solvers --- `CmaInject`, `BoundedCmaInject`, `DeInject` --- not the barrier /
  AL pair. Adding serde to those later means `#[serde(skip)]` on the private
  `inner` field plus a rebuild on load; additive, so not foreclosed. The
  top-level `Executor` carries the same `Vec<Box<dyn …>>` criteria/observers and
  has the same driver status (never in the persisted-state set).
- **`ClosureInner.seed_fn: Box<dyn Fn(&V, F) -> S>`** (`solver/cma_inject.rs`)
  --- a closure. `ClosureInner` is publicly exported but documented as a one-off
  experiment / contract-test escape hatch that wraps user logic; it is
  intrinsically non-serializable (any closure-holding type is), and it carries
  no iteration state. Not in scope for resume, and nothing about it foreclosed
  by a 1.0 choice --- the non-serde-ness is inherent to "wraps a closure."
- A future **closure-filter `ObserverMode`** variant would also be non-serde;
  another reason to keep `ObserverMode` plain data (ties to A1, where it was
  kept a plain `#[non_exhaustive]` enum).

**Conclusion.** No 1.0 choice forecloses a later `serde` feature. The persisted
iteration state (states + the working-state solvers) is uniformly serde-clean
given backend bounds; the only non-serde types are runtime drivers a resume
design rebuilds (`InnerExecutor` / `Executor`) or an explicit closure escape
hatch (`ClosureInner`), none of which block adding derives additively. If resume
is ever scoped, the work is: serde on states + solver structs, `#[serde(skip)]`
+ rebuild for the `InnerExecutor`-holding injection solvers, plus an `Executor`
load path that skips `Solver::init`.

### A4. Error-type model `[DONE]`

The model: per-trait associated `type Error`; `LineSearch` constrains
`L::Error = P::Error` (`line_search.rs`); a two-channel convention of
soft-reject (`Ok(f64::INFINITY)` rejects a point) vs hard-abort (`Err(_)` ends
the solve). It is load-bearing and internally consistent.

**Decision: ratified as-is for 1.0** --- no code change to the model. The action
item was documentation, and it is done: a prominent, stable-guarantee
`# Error model` section now lives in the crate-level docs (`lib.rs`),
consolidating what was spread across per-trait rustdoc. It names *three*
outcomes rather than two --- soft-reject (`Ok(f64::INFINITY)`, one point),
**clean stop** (a `TerminationReason` via `Executor::run`'s
`Ok(OptimizationResult)`, which is *not* an error), and hard-abort (`Err(_)`,
bubbles out as `Result<_, P::Error>`) --- plus the "one error type, threaded
through" plumbing (`CostFunction::Error` / `Residual::Error` chosen once;
`Solver::Error` and `LineSearch::Error` mirror `P::Error`; `Infallible` for the
zero-cost happy path). The per-trait docs remain the detailed reference. (The
clean-stop channel was added explicitly because the original two-channel framing
omitted it, and it is exactly the distinction downstream is most likely to rely
on.)

### A5. `Solver::name()` introspection `[DONE — deferred]`

argmin requires `Solver::name() -> &str` for logging/observer display; basin's
`Solver` trait (`core/solver.rs`) has none. Adding it with a default impl is
*additive and non-breaking* even post-1.0, so there is no freeze-now pressure.
Recorded as an explicit deferred choice in `AGENTS.md` ("Provisional choices")
so the absence reads as deliberate, not an oversight. Revisit only if/when an
observer that prints the solver name is wanted.

--------------------------------------------------------------------------------

## B. Internal consistency --- "things that chafe", cheapest pre-1.0

### B1. Builder-method naming `[DONE]`

Post-construction setters were inconsistent --- most used `with_*`, but a
sizable minority were bare. The original audit undercounted the offenders (it
listed only `Bfgs` and `LevenbergMarquardt`); the actual set spanned seven
solvers:

  | Solver                      | Bare setters (renamed)                                                            |
  | --------------------------- | --------------------------------------------------------------------------------- |
  | `Bfgs`                      | `epsilon`                                                                         |
  | `Lbfgs`                     | `tol_pg`, `epsilon`, `m_capacity`                                                 |
  | `GaussNewton`               | `tol_grad`                                                                        |
  | `Trf`                       | `tol_grad`, `tau`, `rstep`, `theta`, `max_inner_attempts`                         |
  | `LevenbergMarquardt`        | `tol_grad`, `tol_grad_rel`, `ftol`, `xtol`, `tau`, `max_inner_attempts`           |
  | `BarrierMethod`             | `reduction`, `tol`, `inner_max_iter`, `inner_grad_tol`                            |
  | `AugmentedLagrangianMethod` | `rho_increase`, `feasibility_decrease`, `tol`, `inner_max_iter`, `inner_grad_tol` |

**Resolved (2026-06-04).** Standardized every chained *solver* setter on
`with_*` (`with_epsilon`, `with_tol_grad`, `with_tau`,
`with_max_inner_attempts`, `with_tol_pg`, `with_m_capacity`, `with_reduction`,
`with_tol`, `with_inner_max_iter`, `with_inner_grad_tol`, `with_rho_increase`,
`with_feasibility_decrease`, ...). Constructors that take required values stay
on `new()` / named constructors; this was purely the chained setters.

Every old bare name shipped in v0.9.0 (verified against the tag), so each is
kept as a `#[deprecated(since = "0.10.0")]` forwarding shim that delegates to
the new `with_*` method --- a non-breaking rename window for downstream
consumers (e.g. eunoia's NLLS migration). The shims live in a dedicated `impl`
block per solver, marked for removal at 1.0, joining the existing `BFGS` /
`LBFGS` / `LBFGSB` aliases tracked under [B3](#b3-drop-deprecated-aliases-do).

Two sub-decisions:

- **`ftol`/`xtol` were also renamed for clarity, not just prefixed.** The
  MINPACK abbreviations were the lone outliers against basin's native
  `tol_<thing>` vocabulary (`tol_grad`, `tol_grad_rel`, L-BFGS-B's `tol_pg`,
  CMA-ES's `tol_x`). They became `with_tol_cost_rel` (MINPACK `ftol`) and
  `with_tol_step_rel` (MINPACK `xtol`), giving LM a regular grid `tol_grad` /
  `tol_grad_rel` / `tol_cost_rel` / `tol_step_rel`. The MINPACK names are
  preserved verbatim in the rustdoc ("the MINPACK `ftol` test") so migrants from
  the `levenberg-marquardt` crate can still find them. `step` (not `param`) was
  chosen deliberately to *differentiate* from the framework's
  [`RelativeParamTolerance`], which is a subtly different control --- the
  solver-internal `tol_step_rel` is MINPACK-exact and can fire on attempted
  (rejected) steps, whereas the framework criterion reads accepted `‖xₖ − xₖ₋₁‖`
  off `State`. The struct fields and internal convergence locals were renamed to
  match.

- **Scope: solvers only.** The core builders ---
  `Executor::{max_iter,   terminate_on, run_to_end}`, `InnerExecutor::max_iter`,
  `FiniteDiff::{gradient_method, jacobian_method, hessian_method,   function_precision}`
  --- were deliberately left alone. They use an established *verb-style*
  driver/wrapper idiom; forcing `Executor::max_iter → with_max_iter` while
  `terminate_on` stays would make the Executor *less* consistent, not more.

[`RelativeParamTolerance`]: ../crates/basin/src/core/termination.rs

### B2. Constructor convention `[RESOLVED]`

**Decision:** `new()` is the canonical entry point and takes exactly the
parameters that have no sensible default. Where `new()` is nullary, also
`impl Default` so `Foo::default()` agrees. Named constructors (`adaptive`,
`standard`) stay as *additional* presets, not the sole entry.

Status across the solvers:

- **`NelderMead` now has `new()`** (the standard 1965 coefficients α=1, β=2,
  γ=0.5, δ=0.5 *are* the default), with `impl Default`. The old `standard()` is
  `#[deprecated]` (0.10.0) → `new()`, since once `new()` carries the standard
  params it was pure redundancy; `adaptive()` stays as the only
  meaningfully-named preset and `with_params(α, β, γ, δ)` as the explicit
  override (`solver/nelder_mead.rs`). This resolves the open "gain `new()` vs
  omit it" question in favour of consistency.
- **`CmaEs::new(seed)`** --- the required `mean`/`sigma` are gone; the initial
  mean now comes from the state (x0 in the right place) and σ has a default,
  fixed at the root in
  [B8](#b8-cma-es-distribution-dedicated-cmaesstate-vs-solver-parked-working-state-decide).
- **`GradientDescent::new(alpha)`** keeps a required arg --- a step has no
  universal default.
- **`Bfgs::new()` / `LevenbergMarquardt::new()`** stay nullary with `Default`.

Required-arg `new()` is fine where the parameter has no reasonable default.

### B3. Drop deprecated aliases `[DONE]`

Resolved (2026-06-10). Every `#[deprecated]` item was removed — the full set
turned out broader than the three originally itemized here, because B1's `with_*`
rename left a forwarding shim on *seven* solvers (LM, TRF, Gauss-Newton, BFGS,
L-BFGS, barrier, AL), all marked "remove at 1.0". Removed:

- The screaming-case type aliases `BFGS`, `LBFGS`, `LBFGSB` (and their
  `#[allow(deprecated)]` re-exports in `lib.rs` / `solver.rs`) — the frozen
  surface now carries only `Bfgs` / `Lbfgs` / `Lbfgsb`.
- `NelderMead::standard()` — was exactly `new()`.
- Every bare-name builder shim from the B1 rename: `epsilon` / `tol_pg` /
  `m_capacity` (BFGS/L-BFGS), `tol_grad` / `tol_grad_rel` / `ftol` / `xtol` /
  `tau` / `rstep` / `theta` / `max_inner_attempts` (NLLS family), and the
  barrier / AL `reduction` / `tol` / `inner_max_iter` / `inner_grad_tol` /
  `rho_increase` / `feasibility_decrease`.

No internal consumer (tests, benches, examples, `basin-wasm`) referenced any
removed item; the only non-source fixups were the web solver catalogue
(`web/.../docs/solvers/+page.svx`, names + docs.rs links) and two `.claude/rules`
files. Verified: build, clippy `-D warnings`, `cargo doc`, full test suite.

### B4. State generic ergonomics `[DONE]`

The richer states leaked their matrix type into call-site turbofish:
`QuasiNewtonState::<Vec<f64>, DenseMatrix>::new(x)`. `M` is hard for a new user
to guess (DenseMatrix vs nalgebra `DMatrix` vs faer `Mat`).

(Correction to the original framing: `LbfgsState` no longer carries an `M`
generic --- it is `LbfgsState<V, F = f64>`, so its only turbofish is `V`, the
user's own param type, not the unguessable `M`. The pain was `QuasiNewtonState`
alone.)

**Resolved (2026-06-07).** Shipped per-backend type aliases for
`QuasiNewtonState`, each `<F = f64>` and re-exported from the crate root
(`core/state.rs`, `lib.rs`):

  | Alias                      | Pins                                          | Feature    |
  | -------------------------- | --------------------------------------------- | ---------- |
  | `DenseQuasiNewtonState`    | `QuasiNewtonState<Vec<F>, DenseMatrix<F>, F>` | (always)   |
  | `NalgebraQuasiNewtonState` | `QuasiNewtonState<DVector<F>, DMatrix<F>, F>` | `nalgebra` |
  | `FaerQuasiNewtonState`     | `QuasiNewtonState<Col<F>, Mat<F>, F>`         | `faer`     |

The common path is now `DenseQuasiNewtonState::new(x)`. No ndarray alias: BFGS
rejects `ndarray` at compile time per tenet 5. Chose aliases over the "smarter
inference" alternative (a `PairedMatrix` V→M trait) because a transparent type
alias is the *lightest* frozen surface --- trivially deprecatable, not
downstream-implementable --- whereas a public trait freezes heavier and only
removes the turbofish at construction, not at type-annotation sites.
`LbfgsState` deliberately gets no parallel aliases (no `M` to hide). The shipped
tests/benches (`tests/bfgs*.rs`, `tests/lbfgsb_nalgebra.rs`,
`benches/solver_backends.rs`) were migrated to the aliases, exercising all three
backends. (Note: `BasicState<P, F = f64>`, `BasicSimplexState`,
`BasicPopulationState` already read cleanly.)

### B5. Pre-init `cost()` contract `[DONE]` (downgraded from "type inconsistency")

Correction to an earlier finding: the public `State::cost()` is **uniformly
`-> Self::Float`** across all states (`core/state.rs:127` trait; impls at
`state.rs:390/632/785/942`, `state/lbfgs.rs:322`) --- there is *no* return-type
inconsistency. The real nuance is the pre-init contract: gradient states store
`cost: Option<F>` internally and **panic** if `cost()` is read before
`Solver::init` populates it (`state.rs:390–392`,
`"BasicState::cost read before Solver::init populated it"`), whereas
simplex/population states are populated at construction and never panic. The
`Executor` guarantees `init` runs before any read, so this is safe in normal
use.

**Recommend (light):** no API change. Just ensure the panic-vs-always-populated
distinction is documented consistently on each state's `cost()` and confirm no
public path exposes a pre-init read. Decide whether that's worth a doc-only pass
now or is already adequately covered.

**Resolved (doc-only).** Per-impl `# Panics` sections now cover all four
lazily-populated states (`BasicState`, `QuasiNewtonState`, `LbfgsState`,
`CmaEsState`); the construction-populated states (`BasicSimplexState`,
`BasicPopulationState`) never panic. The trait-level `State::cost()` doc was
stale --- it listed `BasicSimplexState` as panicking (it doesn't), qualified
`QuasiNewtonState` as nalgebra-only (panics on every backend post-B4), and
omitted `LbfgsState` / `CmaEsState` --- now corrected to enumerate the lazy set
accurately and note the populated-at-construction states. No public path exposes
a pre-init read.

### B6. `InnerExecutor` criteria-reuse semantics `[RESOLVED]`

`InnerExecutor` (`core/inner.rs`) holds one
`Vec<Box<dyn TerminationCriterion<S>>>` for its whole lifetime and reuses it on
every `run()`. This was correct for stateless criteria (`MaxIter`,
`MaxCostEvals`) but a sharp edge for stateful ones. The original write-up
flagged only `MaxTime` (`start: Option<Instant>` set on first `check`, never
cleared → fires prematurely on a second `run()`); auditing the full criterion
set found the gap was broader --- `RelativeGradientTolerance` anchors `‖∇f_0‖`
on the first run via `get_or_insert` and never re-anchors (silently wrong, not
just early-stop), and `NoImprovement`'s `stalled` counter accumulates across
runs. The `*Tolerance` "last value" criteria self-heal (they re-seed on the
first check of each run), so they were correct only by luck.

**Resolution (shipped):** a defaulted `fn reset(&mut self) {}` lifecycle hook on
`TerminationCriterion`, called on each criterion at the top of `run_loop` (the
single choke point --- `InnerExecutor::run` delegates straight to it). Stateful
criteria override `reset` to clear per-run state (`MaxTime`,
`RelativeGradientTolerance`, `NoImprovement`, plus the self-healing `*Tolerance`
family for a clean "reset == freshly constructed" invariant). The hook is
additive and object-safe (defaulted body keeps `Box<dyn TerminationCriterion>`
working), so no existing call site changed. Contract 2 in `inner.rs` /
`.claude/rules/solver-composition.md` was rewritten from "criteria must be
stateless" to "criteria are reset per run, so stateful ones are safe to reuse; a
custom criterion holding cross-call state must override `reset`."

**Interaction with C1 (`run_loop` visibility):** this removes "reconstruct
criteria per call" as a *reason* to drop to `run_loop` (an `InnerExecutor` now
reuses stateful criteria safely), but `run_loop` stays public as the low-level
driver / adapter-problem entry point. The adapter-problem outer solvers
(`BarrierMethod`, `AugmentedLagrangianMethod`) still build a fresh criteria
vector per outer iteration --- but now for the *intrinsic* reason (each outer
iter minimizes a changing surrogate against a fresh `Problem::new(adapter)`), no
longer as a `MaxTime` dodge; their "Composition" notes were updated to say so.

### B7. Gradient criteria silently no-op on NLLS solvers `[RESOLVED]`

The NLLS solvers (`LevenbergMarquardt`, `Trf`, `GaussNewton`) deliberately leave
`state.gradient = None` --- the framework's L2-squared [`GradientTolerance`] is
the wrong metric for least squares, where the canonical first-order test is
`‖Jᵀr‖_∞` (documented on `levenberg_marquardt.rs` / `trf.rs`, and the reason
each solver carries its own `with_tol_grad`). The footgun was: those solvers ran
on `BasicState`, which *does* impl `GradientState`, so a user could still attach
`GradientTolerance` (or `RelativeGradientTolerance` /
`ProjectedGradientTolerance`) to the executor. It **type-checked and silently
never fired** --- `check` reads `state.gradient()?`, which short-circuits to
`None` on the permanent absence, so no termination, no panic, no warning. The
compile-time guard from tenet 3 (can't pair a gradient criterion with a
derivative-free solver) didn't catch this, because the NLLS state was nominally
a `GradientState`; it just never populated the gradient.

(The original audit claimed `MaxGradientEvals` was the analogous *inert* case
--- "LM makes Jacobian calls, so `gradient_evals` stays `0` and the budget never
trips". That was **wrong**: `BasicState`'s `CountsMirror` folded `jacobian_evals`
*into* `gradient_evals`, and the mirror runs every step, so `MaxGradientEvals`
on an LM run actually *did* fire, silently counting Jacobian work as gradient
work --- a second footgun pointing the opposite way.)

**Resolved (option B --- dedicated non-`GradientState` states).** The NLLS trio
now runs on a new `NllsState` and `Brent` on a new `ScalarState`; neither impls
`GradientState`. Attaching any gradient criterion (`GradientTolerance`,
`RelativeGradientTolerance`, `ProjectedGradientTolerance`, `MaxGradientEvals`) to
these solvers is now a **compile error** --- the same tenet-3 guard that already
keeps gradient criteria off derivative-free solvers. This closes both footguns
(the silent-no-op tolerance *and* the mislabeled `MaxGradientEvals` count). Notes:

- The asymmetry stays load-bearing: nothing populates a wrong-metric gradient.
  The fix is purely at the type level (drop the `GradientState` impl by using a
  state that never had it), so `MaxCostEvals` / `result.cost_evals()` are
  unchanged (`NllsState` preserves the `cost_evals = cost + residual` fold).
- `NllsState` exposes honest `residual_evals()` / `jacobian_evals()` accessors
  (MINPACK `nfev` / `njev`) in place of the old mislabeled `gradient_evals()`.
- Each NLLS / Brent solver's `# Termination` rustdoc now states the framework
  gradient criteria are a compile error and points at `with_tol_grad` /
  `with_tol_grad_rel` (the solver's own first-order tests).
- `BasicState` retains its `GradientState` impl --- it still serves the genuine
  gradient solvers (`GradientDescent`, `ProjectedGradientDescent`, `Sgd`).

[`GradientTolerance`]: ../crates/basin/src/core/termination.rs

### B8. CMA-ES distribution: dedicated `CmaEsState` vs solver-parked working state `[RESOLVED]`

**Resolved:** introduced `CmaEsState<V, M, F>` (`core/state/cma_es.rs`). The
distribution (`m`, `σ`, `C`, `B`/`D`, evolution paths) now lives in the state;
`CmaEs` and `BoundedCmaEs` are configuration-only (derived constants + RNG
cached on the solver). One shared state serves both solvers (bound-penalty
bookkeeping rides along as `Option<BoundPenalty>`, mirroring
`LbfgsState::work`). The canonical TolX test is now the composable
`CmaEsTolerance` criterion (binding the concrete state, firing
`TerminationReason::CmaEsTolerance`) instead of a hardcoded `terminate` hook,
restoring tenet 3. `CmaEs::new`/`BoundedCmaEs::new` take only the seed; the
mean/σ/stds go on `CmaEsState::new(mean, sigma).with_stds(stds)`. Per-generation
the solver evaluates `f(m)` so `State::param`/`cost` report the mean
(`xfavorite`) while `best_param`/`best_cost` report the best evaluated sample
(`xbest`). The original section is kept below for the rationale.

CMA-ES is the one solver whose iterate does not live in its state. Its
distribution parameters --- the mean `m`, step-size `σ`, covariance `C`,
evolution paths `p_σ` / `p_c`, and the `B`/`D` eigendecomposition --- live in a
`Working` struct *on the solver* (`solver/cma_es.rs`), while the state is a
plain `BasicPopulationState` holding only the λ sampled candidates. The initial
mean is threaded in through the constructor
(`CmaEs::new(initial_mean, initial_sigma, seed)`, `:213`) and seeds `m` at
`init` (`:516`); the state is built separately with
`BasicPopulationState::with_size(λ)` --- sized, but carrying no starting
iterate. Every other solver does the opposite: x0 lives in the state
(`BasicState::new(x0)`), uniformly.

This subsumes the CMA-ES half of [B2](#b2-constructor-convention-decide). The
constructor asymmetry there ("`CmaEs::new` takes required args, most `new()`
don't") is downstream of *this* choice: `mean` is in the constructor only
because there is nowhere in the state to put it.

**The "working state lives on the solver" rationale is already contradicted by
L-BFGS.** [AGENTS.md](../AGENTS.md) records (under the observer-KV non-tenet)
that "solver-internal working state (CMA-ES σ / covariance / evolution paths)
lives in the *solver* struct, not the state." But `LbfgsState`
(`state/lbfgs.rs:27`) carries the `(s,y)` history (`ws`, `wy`, `sy`, `ss`),
`theta`, *and* an entire `LbfgsbWork` scratch struct (`:78`, \~20 internal
buffers --- `z`, `r`, `d`, `wn`, `iwhere`, ...) --- unambiguous solver working
state, in a *dedicated state*. There is no principled line separating "L-BFGS
history earns a state" from "CMA-ES distribution doesn't." The recorded note is
descriptive of one solver's choice, not a rule the crate follows.

**The impurity already has a visible cost.** `CmaEs::terminate` (`:802`) takes
`_state` --- ignores it --- and reaches into `self.state` to compute the
canonical TolX test, `σ · maxᵢ dᵢ < tol_x` (Hansen 2016 App. B.3). So CMA-ES's
canonical convergence criterion is a hardcoded solver hook, *not* a composable
framework [`TerminationCriterion`], because σ and the covariance axes `D` aren't
in the state for one to bind on. The doc comment at `:112` lists Stagnation /
TolXUp / TolFun as out of scope; those are the other canonical CMA-ES stopping
rules, blocked the same way (they read σ/C/history the state can't see). That is
tenet 3 --- state shape is the contract --- being quietly routed around.

**What a dedicated `CmaEsState<V, M, F>` would buy.** Hold `m`, `σ`, `C`, `p_σ`,
`p_c`, and impl `PopulationState` so the λ candidates stay exposed --- exactly
as `QuasiNewtonState` impls `GradientState` while additionally carrying `H`.
Then:

- **The constructor question resolves cleanly.** `mean` becomes the state's
  initial iterate (`CmaEsState::new(mean, sigma)`), so x0 lives in the state
  like every other solver and `CmaEs::new()` drops to true hyperparameters ---
  B2's CMA-ES asymmetry disappears instead of needing a documented exception.
- **σ and `D` become visible**, so TolX / TolUpSigma / TolFun can be real
  framework criteria binding on a `CmaEsState`-style shape (configured on the
  Executor, like `SimplexTolerance` binds on `SimplexState`) rather than frozen
  into a solver hook.
- **The result can be the mean.** Canonical CMA-ES reports `m` as the
  recommended solution; with `m` on the solver, `OptimizationResult` (final
  state) can only surface a sampled candidate. *(Not fully traced --- what
  `BasicPopulationState` exposes as incumbent should be confirmed before relying
  on this point.)*

**Counter-costs (so the call is deliberate):**

- A criterion binding on `CmaEsState`'s σ/C is **CMA-ES-specific** --- it
  doesn't generalize across solvers the way `GradientTolerance` does. That's
  consistent with the existing model (`SimplexTolerance` is simplex-only), but
  it does mean the termination layer grows solver-family-specific criteria.
- Real **pre-1.0 surface to design and freeze**: a state generic over
  `<V, M, F>` with a covariance matrix plus its trait impls. L-BFGS proves it's
  tractable, but it isn't free, and it touches the memetic CMA composition sites
  (`ma_ls_ch_cma.rs:622`, which currently injects the mean via the constructor).

**Decide:** introduce `CmaEsState` (consistency with L-BFGS/BFGS, fixes the
constructor, unlocks σ/C termination as framework criteria) or keep the
`BasicPopulationState` + solver-`Working` split and instead **record the
asymmetry as a deliberate non-tenet** (CMA-ES optimizes a solver-owned sampling
distribution; its seed and spread are solver parameters; the population state
holds only the samples). If kept, update the AGENTS.md note so it reads as a
choice rather than a rule the L-BFGS state already breaks. **Recommend:**
introduce `CmaEsState` --- it's the move that makes CMA-ES consistent with the
rest of the crate, and the only argument for the status quo is scope, not
principle.

[`TerminationCriterion`]: ../crates/basin/src/core/termination.rs

--------------------------------------------------------------------------------

## C. Surface-minimization review

Every `pub` item is something 1.0 must keep. Trim what doesn't need to be
public.

### C1. `run_loop` visibility `[RESOLVED: keep public]`

Both the high-level `InnerExecutor` builder and the low-level `run_loop`
function are public (`lib.rs:105`). The original write-up framed `run_loop` as
the escape hatch for custom outer solvers needing *fresh per-call termination
criteria*, because `InnerExecutor` reused criteria and that misbehaved with
stateful ones like `MaxTime`. **B6 removed that motivation:** `run_loop` now
`reset`s every criterion at entry and `InnerExecutor` delegates straight to it,
so criteria reuse is safe in both. The criteria caveat is no longer a reason for
either choice.

**Resolution:** keep `run_loop` public, as the low-level driver and the
*adapter-problem* entry point — outer solvers that minimize a changing surrogate
against a fresh `Problem::new(adapter)` per iteration (`BarrierMethod`,
`AugmentedLagrangianMethod`, and future external constrained methods like a
nonlinear-constraint penalty solver), which `InnerExecutor`'s owned-reusable-
criteria model doesn't serve.

Reasons, in increasing weight:

1. **Near-zero frozen surface.** The usual "fewer public items" argument barely
   applies: the signature is already maximally permissive — `criteria: &mut
   [Box<dyn TerminationCriterion<S>>]` (a slice, not `&mut Vec`), and the element
   type is already implicitly public (the trait is public; `InnerExecutor::
   terminate_on` stores exactly `Box<dyn TerminationCriterion<S>>`). Hiding
   `run_loop` would free essentially nothing to refactor later.
2. **Genuine external use case.** The adapter-problem pattern (fresh
   `Problem::new(adapter)` per outer iter, fold counts back) is what the shipped
   barrier / AL methods do and what an external author writing a custom
   constrained method — the deferred "nonlinear constraints" story — replicates.
3. **Consistency with committed surface.** `WarmStart` is already `pub` and the
   three composition contracts are documented for "every outer solver" — phrased
   for external authors. Publishing the contract machinery while hiding the one
   primitive an adapter-problem author needs would be incoherent; the surrounding
   surface already answers "is composition user-extensible?" with yes.

The only thing that would have flipped this to `pub(crate)` is deciding
composition is basin-internal only — but that's a bigger call than `run_loop`
(it would also pull `WarmStart` and the documented contracts private), and the
shipped surface deliberately goes the other way.

### C2. Math-tier re-exports `[RESOLVED]`

`Scalar`, `ScaledAdd`, `Dot`, `NormSquared`, and the rest of the `core::math`
re-exports were public because user-implemented problems and custom solvers
reference the bounds. The audit asked for the *minimum* set that must stay
public — any op used only internally by shipped solvers should not enter the
frozen surface.

**Resolution (shipped, commit `9411824`, breaking):** the curation key is that
every math bound sits in *impl position* (`impl Solver for …`), so a trait is
forced `pub` only when named in a genuinely public declaration; everything else
is per-solver plumbing that can be `pub(crate)` without tripping
`private_bounds`.

- **Kept `pub` + re-exported (load-bearing):** the vector tier; the LA tier
  (`MatVec`, `MatTransposeVec`, `GramMatrix`, `LinearSolveSpd`,
  `LinearSolveLstsq`, `SymmetricEigen`); sampling; concrete types / errors; and
  the traits named in public signatures (numdiff's `VectorLen` / `VectorIndex` /
  `DenseMatrixFromFn`; `CmaEsState`'s `MatrixIdentity` / `MatrixFromDiagonal` /
  `ComponentMulAssign`). Six load-bearing items that were missing from the
  re-export list were added.
- **Demoted to `pub(crate)` (impl-only plumbing):** `AddDiagonalVectorInPlace`,
  `MaxDiagonal`, `MatDiagonal`, `RankOneUpdate`, `GeneralRankOneUpdate`,
  `BoxAffineScaling`, `ComponentMaxAssign`, `ComponentDivAssign`,
  `FloorZerosInPlace`.
- **Deleted entirely:** `AddDiagonalInPlace` — the scalar diagonal op had no
  consumer (LM / TRF use the vector form) and was only masked as "used" because
  `pub use` re-exports escape the dead-code lint.

The current re-export set lives at `lib.rs:101`. Verified across default +
all-features build, clippy, `cargo doc`, the wasm32 build (all `-D warnings`),
and the test suite.

--------------------------------------------------------------------------------

## Out of scope for this step (and for 1.0)

- Building checkpointing, the `Checkpoint` trait, or the `serde` feature itself
  (door kept open per A3; implementation is later).
- Observer event hooks beyond init/iter/final (`BeginEpoch`, `StepTaken`, ...).
- New solvers; backend-coverage gaps (BFGS-on-ndarray, least-squares on
  `Vec<f64>` / ndarray, memetic family on `Vec<f64>` / ndarray); phase-1
  feasibility. Tracked in `TODO.md` --- feature work, not API-freeze decisions.

--------------------------------------------------------------------------------

## Ratification checklist

The items that needed a maintainer call --- all now decided and reflected in
their sections above:

- [x] A1 --- enums getting `#[non_exhaustive]`: TerminationReason + both linalg
      error enums + ObserverMode / Method. Resolved *yes* for the config-style
      enums.
- [x] A2 --- no observer KV channel; recorded as a deliberate non-tenet
      (additively reopenable post-1.0).
- [x] B2 --- `NelderMead` gains `new()` (= standard params) + `Default`;
      `standard()` deprecated → `new()`. "Required iff no sensible default" rule
      recorded.
- [x] B5 --- pre-init `cost()` contract documented (downgraded from "type
      inconsistency").
- [x] B8 --- `CmaEsState` introduced (σ / covariance / evolution paths moved off
      `BasicPopulationState`), enabling the composable `CmaEsTolerance` criterion.
- [x] C1 --- `run_loop` stays **public** (low-level driver + adapter-problem
      entry point); criteria-reuse caveat retired by B6.
- [x] B3 --- all `#[deprecated]` aliases and `with_*`-rename shims dropped
      (2026-06-10). The frozen surface carries only the PascalCase names.

Every audit item is now resolved; the public surface is ready to freeze for 1.0.
