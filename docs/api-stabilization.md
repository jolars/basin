# Pre-1.0 API stabilization audit

Status: **draft for review.** This is a decision checklist, not an
implementation. basin is heading toward 1.0.0 with API stabilization as the
focus and an explicit "no new solvers or features unless they probe what the API
should be" constraint.

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

*Highest leverage, nearly free.* No public enum used `#[non_exhaustive]`
before this change:

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
method (`fn observe_iter_with(&mut self, state: &S, kv: &Kv) {
self.observe_iter(state) }`), switch the executor's internal call site
(`executor.rs:239`) to it, and every existing `Observe` impl keeps compiling via
the forwarding default; a concrete `Kv` type keeps the trait object-safe for the
`Box<dyn Observe<S>>` storage. So the door is *deferred open*, not closed --- the
1.0 commitment is purely "don't build it now."

The genuine future motivation, if it appears: solver-internal working scalars
that don't fit a state trait (CMA-ES σ / covariance / evolution paths, LM μ / ν
/ diag --- see A3) live in the *solver* struct, not the state, so "expose it on
a richer state trait" does not cover them. Until then, keep `Observe`
infallible, read-only, and state-only.

### A3. serde door-open verification `[REVIEW]`

Deferred-but-not-foreclosed (per the up-front decision). Produce a
"serde-readiness" subsection confirming the concrete public state structs **and
the solver structs** can gain `#[derive(Serialize, Deserialize)]` behind a future
`serde` feature without a breaking change.

**Why solvers, not just states (the observer ≠ checkpointing point).** The
observer layer already covers the *save / monitor* half of what a checkpointing
system would do — an `Observe` impl sees `&S` each iteration and can serialize it.
What it does **not** cover is *resume*: (1) the hook receives only the state, not
the solver, and (2) `Executor::run` always calls `Solver::init`, so there is no
load-and-continue entry. Crucially, basin does **not** keep all iteration-carrying
state in the state object — `CmaEs` holds its `Working<V, M, F>` (covariance, σ,
mean, evolution paths) inside the *solver* struct (`solver/cma_es.rs`), and
`LevenbergMarquardt` holds `mu` / `nu` / `diag` / the Gram caches there too
(`solver/levenberg_marquardt.rs`). So a state-only snapshot is not a faithful
resume point for those solvers. Conclusion: observers replace a separate
checkpointing system **iff basin never needs to resume an interrupted run**; if
resume is ever in scope (crash recovery on long fits; pause/resume in the
wasm/browser target), it needs serde on *both* state and solver plus an Executor
load path. That work is purely additive, hence safely deferrable past 1.0 — but
the door-open check below must include solvers or the door is quietly half-closed.

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
  they inherit the `InnerExecutor` blocker below — flag, don't try to resolve.
- **Known blockers to document (not fix):** `InnerExecutor` holds
  `Vec<Box<dyn TerminationCriterion>>` (`core/inner.rs`) — not serializable, but
  it's a driver, not persisted state; a resume design would rebuild it rather
  than deserialize it. A future closure-filter `ObserverMode` variant would also
  block; another reason to keep `ObserverMode` as plain data (ties to A1).
- `PhantomData<fn() -> Mode>` type-state markers (`NelderMead`, `Lbfgs`) derive
  serde fine — note for completeness.

**Also `[DO]`:** `CONTRIBUTING.md` lists a `serde` feature in its features
enumeration ("nalgebra, ndarray, faer, serde, parallel, problems") that **does
not exist** in `crates/basin/Cargo.toml` (`[features]`, lines 28--62, defines
only `nalgebra`, `ndarray`, `ndarray-blas`, `faer`, `parallel`, `problems`).
Remove the `serde` mention until the feature actually lands, or the doc
misleads.

### A4. Error-type model `[RECOMMEND]`

The model: per-trait associated `type Error`; `LineSearch` constrains
`L::Error = P::Error` (`line_search.rs`); a two-channel convention of
soft-reject (`Ok(f64::INFINITY)` rejects a point) vs hard-abort (`Err(_)` ends
the solve). It is load-bearing and internally consistent.

**Recommend:** ratify as-is for 1.0. Action item is documentation, not code:
elevate the soft-reject vs hard-abort contract to a prominent, stable-guarantee
section in the crate-level docs (it is currently spread across per-trait
rustdoc). This is the contract most likely to be relied on by downstream and
hardest to change later.

### A5. `Solver::name()` introspection `[DO: defer]`

argmin requires `Solver::name() -> &str` for logging/observer display; basin's
`Solver` trait (`core/solver.rs`) has none. Adding it with a default impl is
*additive and non-breaking* even post-1.0. Record as explicitly deferred so it
reads as a deliberate choice, not an oversight. Revisit only if/when an observer
that prints the solver name is wanted.

--------------------------------------------------------------------------------

## B. Internal consistency --- "things that chafe", cheapest pre-1.0

### B1. Builder-method naming `[RECOMMEND]`

Post-construction setters are inconsistent --- most use `with_*`, a few are
bare:

  | Solver               | Bare setters (offenders)                                                            | Location                                |
  | -------------------- | ----------------------------------------------------------------------------------- | --------------------------------------- |
  | `Bfgs`               | `epsilon()`                                                                         | `solver/bfgs.rs:116`                    |
  | `LevenbergMarquardt` | `tol_grad()`, `tol_grad_rel()`, `ftol()`, `xtol()`, `tau()`, `max_inner_attempts()` | `solver/levenberg_marquardt.rs:270–361` |

vs the `with_*` majority (`GradientDescent::with_momentum`,
`CmaEs::with_lambda`, `BarrierMethod::with_inner_max_iter`, ...).

**Recommend:** standardize on `with_*` for every post-construction setter.
Rename to `with_epsilon`, `with_tol_grad`, `with_ftol`, `with_xtol`, `with_tau`,
`with_max_inner_attempts`, etc. Keep deprecated forwarding shims only if
existing docs/tests depend on the bare names; drop them at the 1.0 tag. (Builder
methods that take a required value at construction stay on `new()` / named
constructors --- this is purely about the chained setters.)

### B2. Constructor convention `[DECIDE]`

`new()` is canonical for most solvers, but two things vary:

- **`NelderMead` has no `new()`** --- only `standard()`, `adaptive()`,
  `with_params(α, β, γ, δ)` (`solver/nelder_mead.rs:129–152`).
- **Some `new()` take required args** (`CmaEs::new(mean, sigma, seed)`,
  `GradientDescent::new(alpha)`) while most take none (`Bfgs::new()`,
  `LevenbergMarquardt::new()`).

**Recommend:** keep `new()` as the canonical entry point wherever a sensible
default exists; keep named constructors (`adaptive`, `standard`) as *additional*
presets, not the sole entry. **Decide** whether `NelderMead` gains `new()` =
`standard()` (consistency) or deliberately omits it (the choice between standard
and adaptive params is intentional and there's no neutral default --- a
defensible reason to keep named-only). Required-arg `new()` is fine where the
parameter has no reasonable default (CMA-ES needs a mean/sigma; GD needs a
step).

### B3. Drop deprecated aliases `[DO]`

`BFGS`, `LBFGS`, `LBFGSB` are still publicly re-exported behind
`#[allow(deprecated)]` (`lib.rs:98–102`). Remove before 1.0 so the frozen
surface carries only `Bfgs` / `Lbfgs` / `Lbfgsb`. A 1.0 is the natural place to
shed pre-1.0 deprecations.

### B4. State generic ergonomics `[RECOMMEND]`

The richer states leak their matrix type into call-site turbofish:
`QuasiNewtonState::<Vec<f64>, DenseMatrix>::new(x)`,
`LbfgsState::<DVector, DMatrix, f64>::new(...)`. `M` is hard for a new user to
guess (DenseMatrix vs nalgebra `DMatrix` vs faer `Mat`).

**Recommend:** ship per-backend type aliases (or smarter inference) so the
common path doesn't spell `M` --- e.g. a `QuasiNewtonState`/`LbfgsState` alias
pinned per backend behind the matching feature. Pure-additive, but the *alias
names* enter the frozen surface, so settle them before 1.0. (Note:
`BasicState<P, F = f64>`, `BasicSimplexState`, `BasicPopulationState` already
read cleanly --- this is only the BFGS/L-BFGS states.)

### B5. Pre-init `cost()` contract `[DECIDE]` (downgraded from "type inconsistency")

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

--------------------------------------------------------------------------------

## C. Surface-minimization review

Every `pub` item is something 1.0 must keep. Trim what doesn't need to be
public.

### C1. `run_loop` visibility `[DECIDE]`

Both the high-level `InnerExecutor` builder and the low-level `run_loop`
function are public (`lib.rs:68`). `run_loop` is the escape hatch for custom
outer solvers that need fresh per-call termination criteria (the `InnerExecutor`
reuses criteria, which misbehaves with stateful ones like `MaxTime`).

**Decide:** keep `run_loop` public (sanction the escape hatch, document the
criteria-reuse caveat) or make it `pub(crate)` and route everything through
`InnerExecutor`. Fewer public items = less frozen surface; but if any realistic
custom-solver author needs it, exposing it now avoids a later additive scramble.
Leaning: keep public *if* the composition story is meant to be user-extensible;
otherwise hide it.

### C2. Math-tier re-exports `[REVIEW]`

`Scalar`, `ScaledAdd`, `Dot`, `NormSquared`, and the rest of the `core::math`
re-exports (`lib.rs:70`) are public because user-implemented problems and custom
solvers reference the bounds. Audit for the *minimum* set that must be public:
any op only used internally by shipped solvers should not enter the frozen
surface. Deliverable: a list of math items to keep `pub` (load-bearing for the
user-facing generic bounds) vs demote to `pub(crate)`.

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

The `[DECIDE]` items needing a maintainer call:

- [ ] A1 --- which enums get `#[non_exhaustive]` (recommend: TerminationReason +
      both linalg error enums; decide ObserverMode / Method).
- [ ] A2 --- commit to no observer KV channel (recommend: yes, record as
      non-tenet).
- [ ] B2 --- does `NelderMead` gain `new()`?
- [ ] B5 --- doc-only pre-init `cost()` pass now, or defer?
- [ ] C1 --- `run_loop` stays public or becomes `pub(crate)`?

Once ratified, A / B / C become three follow-up PRs.
