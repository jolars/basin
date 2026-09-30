# Migrating to Basin 2.0

## Structured termination reports

Ordinary `Executor::run()` now returns `OptimizationResult<S>` with three
owned fields: `state`, authoritative `counts: EvalCounts`, and
`report: TerminationReport<S::Float>`. Use `run_with_solver()` when you also
need the final solver. The report retains the stopping decision and its
evidence even after the solver is dropped or resumed.

```rust
use basin::Termination;

match &result.report.termination {
    Termination::Converged(convergence) => {
        for criterion in convergence.criteria() {
            println!("{:?}: {:?}", criterion.test, criterion.evidence);
        }
    }
    Termination::Limit(limits) => println!("exhausted budgets: {limits:?}"),
    stop => println!("stopped: {stop:?}"),
}
```

`Termination` distinguishes `Converged`, `Limit`, `Target`, `Stalled`,
`Failed`, `Cancelled`, and `Application`. A numerical failure returns a
coherent state through `Ok`; a typed callback abort still returns `Err` and
consumes the state. A failed stepper exposes evaluation counts but no report
or recoverable checkpoint.

Convergence retains all sufficient predicates evaluated at the stopping
stage. `ConvergenceTest` identifies their formulas and norms;
`ConvergenceEvidence` records measured values, tolerances, and reference
values. Compound conditions stay grouped: LM's relative model-reduction test
records actual and predicted reductions, reference cost, gain ratio, and
tolerance. Extreme-scale LM comparisons retain significands and binary
exponents rather than overflowing a reconstructed norm. External solvers can
use `Termination::custom(key, definition, measurements)` with a namespaced
key and named scalar measurements. Reports do not certify global optimality
or parameter accuracy.

The main replacements are:

| Basin 1.x | Basin 2.0 |
| --- | --- |
| `result.reason` | `result.report.termination` |
| `TerminationReason` | `Termination<F>` for decisions; `TerminationReport<F>` for published events |
| `next_iter -> Result<(S, Option<TerminationReason>), E>` | `next_iter -> Result<SolverStep<S>, E>` |
| `terminate` / `check_convergence -> Option<TerminationReason>` | `Option<Termination<S::Float>>` |
| `stop_when -> Option<TerminationReason>` | `Option<ApplicationStop>` |
| `Observe::observe_final(..., &TerminationReason)` | `observe_final(..., &TerminationReport<S::Float>)` |
| `ObservationEvent::Final(reason)` | `ObservationEvent::Final(&report)` |
| `StepOutcome::Stopped(reason)` | `StepOutcome::Stopped(report)` |
| `Stepper::finished() -> Option<TerminationReason>` | `Option<&TerminationReport<S::Float>>` |
| Enum casts such as `reason as u8` | `result.report.code().as_u8()` |

`State::Float` now requires `Scalar`. `StepOutcome<F>` is `Clone`, rather than
`Copy`. Final observer events borrow the report; clone it explicitly when
retaining it beyond the callback. `TerminationCode` is a compact, lossy
projection with an explicit `as_u8()` mapping preserving the 24 Basin 1.x
numeric identifiers. Simultaneous criteria or budgets project to the first
entry. Inspect the structured report when the distinction matters.

`SolverStep` separates completion from termination. Use
`SolverStep::completed(state)` to continue, `stopped(state, decision)` to
publish a partial stopping step, or
`completed_with_termination(state, decision)` when the stopping iteration
finished. A completed stopping step increments the iteration count, fires
the gated iteration observer, and then fires the final observer. A partial
stopping step fires only the final observer. A continuing partial step
publishes progress without incrementing the count or firing iteration observers.
In all cases the executor
publishes counts and incumbent updates. `report.stage` distinguishes an
execution boundary from a step and records whether the step completed.

LM now counts a completed trial that passes a native stopping test as an
iteration, including a rejected trial. Initial gradient checks remain partial
step stops at iteration zero. Evaluation counts and numerical trajectories
are unchanged. BFGS reports a finite nonpositive line-search step as a
numerical stall; a non-finite step is a numerical failure. Bounded L-BFGS
reports its bound-clipped gradient predicate consistently, with the compact
code `ProjectedGradientTolerance` for both default and configured tolerances.

Execution controls still precede numerical checks. A limit report retains
all budgets exhausted at that boundary, including their observed counts and
thresholds. Application hooks return, for example,
`Some(ApplicationStop::new("feasible_target"))`; they cannot impersonate a
numerical convergence test. Targets and stalls retain their measurements.

Exact continuation preserves solver and convergence history, discards the
previous event, and produces a new report. Native identities are available
from `result.report.termination.native_convergence_tests()` without retaining
the solver. The compatibility accessor on `OptimizationResultWithSolver`
now derives its owned `Vec<NativeConvergenceTest>` from the report, so an
execution limit cannot expose an earlier convergence event.

Composed solvers can call
`report.into_outer_termination(PartialResultPolicy::Consume)` to accept inner
limits, targets, or stalls as partial progress, or use `RequireConvergence`.
An inner numerical failure retains its nested report. Cancellation and
application stops propagate. An algorithm may explicitly reject a failed
local candidate and continue, as basin hopping does; an inner convergence
report never establishes outer convergence.

Exact checkpoint files now use format version 3 because solver convergence
history includes report payloads. Older exact formats are rejected. Finish
those runs with the matching application and export parameters for a fresh
run. State-only checkpoint files retain their format.

## Solver and line-search settings

Basin 2.0 removes `BarrierMethod::new`, `AugmentedLagrangianMethod::new`,
and their `with_inner_grad_tol` setters. Construct either method with
`with_inner_solver(inner)` and configure convergence on `inner` before passing
it in. The outer methods retain their default 50-iteration inner budgets but
no longer install an implicit `1e-8` gradient test. For example:

```rust
let solver = BarrierMethod::with_inner_solver(
    GradientDescent::with_line_search(Backtracking::new())
        .with_absolute_gradient_tolerance(1e-8),
);
```

Use the same pattern with `AugmentedLagrangianMethod`. Its inner solver may
choose any convergence test. For `BarrierMethod`, configure gradient
convergence if Phase I must certify that a constraint system has no strict
interior; an inner iteration budget alone cannot establish that certificate.

Basin 2.0 removes the deprecated setter aliases below. Replace each call with
the named method on the same solver or line search. The replacement keeps the
algorithm's setting and default unless noted here.

| Solver or component | Removed method | Replacement |
| --- | --- | --- |
| `Brent`, `BrentDerivative`, `GoldenSection` | `with_tol(relative, absolute)` | `new().with_relative_position_tolerance(relative).with_absolute_position_tolerance(absolute)` |
| `BrentRoot` | `with_tol(relative, absolute)` | `with_relative_position_tolerance(relative).with_absolute_position_tolerance(absolute)` |
| `GaussNewton`, `Trf`, `LevenbergMarquardt`, `LevenbergMarquardtQr` | `with_tol_grad` | `with_absolute_gradient_tolerance` (`with_absolute_scaled_gradient_tolerance` for `Trf`) |
| `LevenbergMarquardt`, `LevenbergMarquardtQr` | `with_tol_grad_rel`, `with_tol_cost_rel`, `with_tol_step_rel` | `with_gradient_orthogonality_tolerance`, `with_relative_model_reduction_tolerance`, `with_relative_step_tolerance` |
| `LevenbergMarquardtQr` | `with_rank_tolerance` | `with_relative_rank_tolerance` |
| `Lbfgs` in bounded mode | `with_tol_pg` | `with_absolute_projected_gradient_tolerance` |
| `Bfgs`, `Lbfgs` | `with_epsilon` | `with_relative_curvature_tolerance` |
| `Gbnm` | `with_small_tolerance`, `with_flat_tolerance` | `with_normalized_simplex_size_tolerance`, `with_absolute_simplex_cost_tolerance` |
| `Gbnm` | `with_degeneracy_tolerances(edge, determinant)` | `with_edge_ratio_tolerance(edge).with_normalized_determinant_tolerance(determinant)` |
| `BarrierMethod`, `AugmentedLagrangianMethod` | `with_tol` | `with_absolute_duality_gap_tolerance`, `with_absolute_feasibility_tolerance`, respectively |
| `BarrierMethod` | `with_phase_one_tol` | `with_absolute_phase_one_gap_tolerance` |
| `Newuoa`, `Bobyqa`, `Lincoa`, `Cobyla` | `with_rho_beg`, `with_rho_end` | `with_initial_radius`, `with_final_radius` |
| `Mads` | `with_min_poll_size` | `with_minimum_poll_size` |
| `SolisWets` | `with_rho_init` | `with_initial_step_size` |
| `MaLsChCma` | `with_initial_sigma_fallback` | `with_initial_scale_fallback` |
| `Backtracking` | `c` | `with_sufficient_decrease_coefficient` |
| `Wolfe` | `c1`, `c2` | `with_sufficient_decrease_coefficient`, `with_curvature_coefficient` |
| `MoreThuente` | `ftol`, `gtol`, `xtol` | `with_sufficient_decrease_coefficient`, `with_curvature_coefficient`, `with_relative_bracket_tolerance` |
| `HagerZhang` | `delta_sigma`, `epsilon` | `with_wolfe_coefficients`, `with_relative_cost_relaxation_tolerance` |

The newer optional numerical tolerance setters accept `None` to disable a
check and zero to request an exact-zero threshold. Some old setters treated
zero as disabled, so use `None` when preserving that behavior. The new setters
also validate finite, nonnegative values. `with_absolute_duality_gap_tolerance`
and `with_absolute_feasibility_tolerance` accept `None` when a check should be
disabled; their former `with_tol` aliases required a positive value. The
relative position tolerance on `BrentRoot` still requires at least four times
the scalar machine epsilon.

## Backend features

The unversioned `nalgebra`, `ndarray`, and `faer` features now select Basin's
newest supported release of each backend. In particular, `nalgebra` moves from
0.34 to 0.35, and `nalgebra-lapack` moves with it. To keep nalgebra 0.34, replace
these features with `nalgebra_v0_34` and `nalgebra_v0_34-lapack`, respectively.
The current `ndarray` and `faer` targets remain 0.17 and 0.24.

The `*_latest` features, including `nalgebra_latest-lapack` and
`ndarray_latest-blas`, remain available as deprecated synonyms for the
unversioned features. Use an exact version feature when your application pins
its backend dependency; a moving alias may select a newer, incompatible
backend release in a later Basin version. The `nalgebra` and `nalgebra-lapack`
features now require Rust 1.89 because nalgebra 0.35 does. Basin's package MSRV
remains Rust 1.87 for features that do not select nalgebra 0.35.

## Test problems

Basin 2.0 enables no features by default. If your code imports
`basin::problems`, add `problems` to your dependency's feature list:

```toml
[dependencies]
basin = { version = "2", features = ["problems"] }
```

Keep any backend or other features you already enable. The corpus API is
unchanged, and `problems` adds no dependencies. Applications that implement
their own objectives need no changes for this feature switch. Basin 1.x keeps
`problems` enabled by default.

When working in this repository, use `cargo test -p basin --features problems`
to include corpus-dependent tests. Enable the same feature for corpus
benchmarks, for example
`cargo bench -p basin --features problems --bench rosenbrock`. The WASM
visualizer and competitor benchmark crate enable it explicitly in their
dependencies.

## Checkpoint files

Basin 2.0 removes bincode and the readers for the two legacy checkpoint formats.
The `serde` feature still enables serialization and, on native targets,
checkpoint file I/O. The checkpoint writers keep their postcard formats:

  | Checkpoint                      | Accepted format                                     | Removed format                            |
  | ------------------------------- | --------------------------------------------------- | ----------------------------------------- |
  | State (`read_checkpoint`)       | `BASINST\0`, version 1, postcard payload            | Unprefixed bincode payload                |
  | Exact (`read_exact_checkpoint`) | `BASINEX\0`, version 3, postcard header and payload | Versions 1 (bincode) and 2 (older solver layouts) |

Both readers return `std::io::ErrorKind::InvalidData` for removed formats,
unsupported versions, malformed data, or trailing bytes. Reading a checkpoint
does not modify it.

### State checkpoints

Before upgrading, use a compatible Basin 1.x release to load an unprefixed
checkpoint with `read_checkpoint`. Basin 1.15 introduced postcard writers while
retaining the legacy readers. With a compatible state type in that release, use
`CheckpointWriter` through `Observe::observe_iter` to save the loaded state to a
new file. Read the new file back to verify it before replacing the original; the
observer reports write failures to stderr.

Retaining the postcard format does not guarantee that a state's serialized
fields remain compatible across major releases. If the state type or its layout
changes, load it in the compatible 1.x application, export its current or best
parameters in an application-owned format, and construct a fresh 2.0 state from
those parameters. This starts a new run and does not preserve the solver's
history or trajectory.

### Exact checkpoints

Exact checkpoints require the exact Basin package version and concrete solver
and state type names recorded in the file. A 1.x exact checkpoint cannot be
resumed in 2.0, even if it already uses postcard version 2. Changing the format
version or the Basin version in the header is not a migration.

For exact continuation, keep the matching 1.x application and its dependencies
until the run finishes. To move the optimization to 2.0, load the checkpoint
with that application, export parameters from `checkpoint.state()`, and start a
fresh 2.0 run from those parameters. Reattach the problem and execution controls
in the new application.

## Observing solver diagnostics

Algorithm quantities that moved from state to solver remain available during
ordinary `Executor::run()` calls. Use `observe_solver` for a closure receiving
`(&state, &solver, ObservationEvent)`, or implement `ObserveSolver<S, So>` and
register it with `observe_solver_with`. Existing `Observe<S>` implementations
and `observe_with` calls continue to handle progress alone.

For an annealing executor, replace a state-temperature logger with:

```rust,ignore
executor.observe_solver(
    |state, solver, event| {
        println!("{event:?}: {} {}", state.iter(), solver.temperature());
    },
    ObserverMode::Every(10),
).run()?;
```

Import `State` and `ObserverMode` from `basin`; `ObservationEvent` and
`ObserveSolver` are also root exports. A reusable observer keeps the familiar
`observe_init`, `observe_iter`, and `observe_final` hooks, now with `&So` after
`&S`. The final hook also receives `&TerminationReport<S::Float>`. All hooks have default
no-op implementations. The closure receives `ObservationEvent::Init`, `Iter`,
or `Final(reason)` instead.

Both observer kinds fire in registration order. Modes filter completed
iterations only; initialization and clean termination always fire. Exact
continuation observes its restored boundary, potentially at a nonzero
iteration. A clean partial-step stop refreshes counts without incrementing the
iteration, and a hard error emits no final callback. Registered observers are
owned and require `'static` captures, but the state and solver may borrow local
data. A callback can cancel a cloned `CancellationToken` to request a clean stop.

The solver retains its models and workspace. Observation borrows existing
diagnostics without cloning or evaluating the problem. Preserve each getter's
meaning: annealing's `temperature()` describes the next proposal after cooling
and any restart; SLSQP stationarity and multiplier slices can be unavailable.
Callbacks must not evaluate the problem or mutate solver machinery through
interior mutability. Outer observers do not automatically receive inner-solver
iterations.

Applications that already drive a `Stepper` can keep reading
`Stepper::solver()` between steps. Use `run_with_solver()` when only final
diagnostics are needed. See the
[observer API](https://docs.rs/basin/latest/basin/core/observer/index.html)
for the full lifecycle and a runnable logging example.

## Shared progress states

The migration replaces algorithm-specific progress with shared states. The
following replacements are implemented on `main`:

  | Basin 1.x state                                               | Basin 2.0 state                 | Solver                                                                  |
  | ------------------------------------------------------------- | ------------------------------- | ----------------------------------------------------------------------- |
  | `BasicSimplexState<V, F>`                                     | `SimplexProgress<V, F>`         | `NelderMead`, both modes                                                |
  | `MaLsChGenericState<V, C>`, `MaLsChState<V, M>`, `MaLsChSwState<V>` | `PopulationProgress<V, F>` | `MaLsCh`, `MaLsChCma`, `MaLsChSw` |
  | `CmaEsState<V, M, F>` | `PopulationProgress<V, F>` | `CmaEs`, `BoundedCmaEs`, `CmaInject`, `BoundedCmaInject` |
  | `BasicPopulationState<V, F>`                                  | `PopulationProgress<V, F>`      | `RandomSearch`, `De`, `Ssga`, `DeInject`                                |
  | `SlsqpState<V, F>`                                            | `SelectedFirstOrderState<V, F>` | `Slsqp`                                                                 |
  | `MadsState<V, F>`                                             | `PointState<V, F>`              | `Mads`, unbounded and box-bounded modes                                 |
  | `ConstrainedMadsState<V, F>`                                  | `SelectedState<V, F>`           | `Mads`, progressive-barrier mode                                        |
  | `CobylaState<V, F>`                                           | `SelectedState<V, F>`           | `Cobyla`                                                                |
  | `NewuoaState<V, F>`, `BobyqaState<V, F>`, `LincoaState<V, F>` | `PointState<V, F>`              | `Newuoa`, `Bobyqa`, `Lincoa`                                            |
  | `GbnmState<V, F>`                                             | `PointState<V, F>`              | `Gbnm`                                                                  |
  | `GlobalBestPsoState<V, F, R>`                                 | `PopulationProgress<V, F>`      | `GlobalBestPso`                                                         |
  | `QuasiNewtonState<V, M, F>` and its backend aliases           | `FirstOrderState<V, F>`         | `Bfgs`                                                                  |
  | `LbfgsState<V, F>`                                            | `FirstOrderState<V, F>`         | `Lbfgs`, `Lbfgsb`                                                       |
  | `BasicState<V, F>`                                            | `FirstOrderState<V, F>`         | `GradientDescent`, `ProjectedGradientDescent`, both `TrustRegion` modes |
  | `BasicState<V, F>`                                            | `PointState<V, F>`              | `Sgd`, `BasinHopping`, `BarrierMethod`                                  |
  | `BasicState<V, F>`                                            | `SelectedState<V, F>`           | `AugmentedLagrangianMethod`                                             |
  | `NllsState<V, F>`                                             | `PointState<V, F>`              | `GaussNewton`, both LM variants, `Trf`, `TrustRegionReflective`         |
  | `SimulatedAnnealingState<V, N, F, R>`                         | `ProposalState<V, F>`           | `SimulatedAnnealing`                                                    |
  | `SolisWetsState<V, F>`                                        | `PointState<V, F>`              | `SolisWets`                                                             |
  | `ScalarState<F>`                                              | `PointState<F, F>`              | `Brent`, `GoldenSection`                                                |
  | `ScalarGradientState<F>`                                      | `FirstOrderState<F, F>`         | `BrentDerivative`                                                       |

Construct shared states with `PointState::new(x)` or `FirstOrderState::new(x)`.
Construction supplies a seed, not an evaluated record. `current()` and `best()`
return `None` until those records become available. First-order
`replace(x, cost, gradient)` requires matching dimensions and preserves the
previous record on a dimension error. Scalar `f32` and `f64` parameters each
have one coordinate. Scalar solver bounds remain problem-side through
`BoxConstraints`.

NEWUOA, BOBYQA, and LINCOA now publish `PointState<V, F>`. Replace their
algorithm-specific `*State::new(x)` constructors with `PointState::new(x)`;
`Executor::from_start` still constructs the appropriate state. The solver types
remain `Newuoa<F = f64>`, `Bobyqa<Mode = Bounded, F = f64>`, and
`Lincoa<F = f64>`, with an explicit `F: Scalar` bound. Vectors still require
`Clone`, `VectorLen`, and `Index`/`IndexMut`; LINCOA constraint matrices still
require `MatTransposeVec`. Every supported dense backend release works with both
`f32` and `f64`.

Read the radius through `result.solver.rho()` after `run_with_solver()`, or
through `stepper.solver().rho()`. This returns `None` before initialization.
`PointState` supplies no `RhoState` capability. Native final-radius stopping and
`with_absolute_radius_tolerance` continue to read the solver's schedule. The
shared state preserves coherent evaluated points and historical objective
incumbents with all six raw counts. LINCOA publishes only its model's feasible
incumbent, excluding infeasible geometry probes; an infeasible start still
relaxes the constraints as documented in its solver API.

Fresh runs reevaluate the interpolation set and rebuild every model, even if the
supplied solver or state has already run. Radius schedules, shifted bounds, and
LINCOA's active-set QR live on the solver. Keep an `ExactCheckpoint` for exact
continuation, including through `serde`; a serialized `PointState` alone
provides a fresh point warm start. The removed state types and their layouts
have no compatibility reader.

COBYLA uses `SelectedState<V, F>` and has the type `Cobyla<V, F = f64>`. Replace
`CobylaState::new(x)` with `SelectedState::new(x)` or keep using
`Executor::from_start`. An explicit `Cobyla<f32>` annotation becomes
`Cobyla<V, f32>`. Its backend requirements remain `Clone`, `VectorLen`, and
indexing, plus `MatVec` for a `FoldedConstraints` matrix. Both constraint paths
support every dense backend release with `f32` and `f64`.

`current()` and `best()` expose the filter-selected point, objective, and
maximum positive constraint violation. COBYLA can prefer a higher-cost point as
feasibility improves. It does not implement `ObjectiveIncumbentState`;
objective-only targets and stalls cannot assume its selection is an objective
minimum. An unchanged selection keeps its original iteration and evaluation
metadata. The driver still moderates extreme and non-finite values according to
PRIMA's rules, including in published objective and violation records.

The solver owns its callback buffer, simplex models, filter, and radius
schedule. Read `rho()` on the retained solver; the shared state has no radius
capability. Fresh runs rebuild these structures and reevaluate their seed. Exact
checkpoints retain them, and `serde` supports the complete solver and shared
state. Old `CobylaState` payloads have no compatibility reader.

COBYLA now counts inequality callbacks in `residual_evals`. With
`FoldedConstraints`, its nonlinear inequality and equality callbacks each count
once, including empty blocks and failing attempts; arithmetic for bounds and
linear constraints adds no callback count. `cost_evals()` reports only objective
calls. Constraint budgets use `max_evaluations(EvaluationKind::Residual, n)`,
and `state.counts()` preserves all six categories. The constraint traits and
`FoldedConstraints` adapter remain problem-side.

MADS keeps its `Mads<Mode = Unbounded, F = f64>` type, now explicitly bounded by
`F: Scalar`. Unbounded and box-bounded modes use `PointState::new(x)`;
progressive-barrier mode uses `SelectedState::new(x)`. `Executor::from_start`
chooses the matching state. `Clone`, `VectorLen`, and indexing remain the only
vector requirements, and all dense backend versions support both scalar types.

Read `poll_size()` and `mesh_index()` on the retained solver, where both return
`Option` values that are absent before initialization. The shared states have no
`MeshState` capability. In constrained mode, read the squared-sum violation from
`state.current()` or `state.best()` instead of `constraint_violation()`. The
progressive barrier still selects a feasible incumbent when available; its
objective can increase while feasibility improves. Unchanged selections keep
their original metadata. Constraint calls now count as residual work, while
rejected box probes consume no objective call.

Every fresh run rebuilds the mesh schedule, Halton history, and progressive
barrier, and reevaluates the starting point. Native poll-size stopping and
`with_absolute_poll_size_tolerance` remain solver-owned. All modes support owned
and serialized exact checkpoints; old MADS state types and payloads have no
compatibility reader.

SLSQP keeps the type `Slsqp<F = f64>` and now publishes
`SelectedFirstOrderState<V, F>`. Replace `SlsqpState::new(x)` with the shared
constructor or use `Executor::from_start`. `current()` returns a point,
objective, matching objective gradient, and sum of constraint violations;
`best()` returns the selected point, objective, and violation. The state
implements `GradientState` and `EvaluatedGradientState`, but not
`ObjectiveIncumbentState`. Each accepted iterate is an explicit selection,
including when its objective increases as feasibility improves. Failed trials
preserve the previous accepted record and its selection metadata.

Move `stationarity()`, `complementarity()`, and `failure()` calls from state to
the solver retained by `run_with_solver()` or `stepper.solver()`. The solver's
`constraint_violation()` mirrors the violation in the current shared record.
These diagnostics return `Option` values until available. Native accuracy and
safeguard behavior are unchanged. Cost and gradient readers now count only their
named categories; constraint and constraint-Jacobian calls remain separate in
`state.counts()` and in category-specific execution budgets.

Fresh runs rebuild the model and reevaluate their clipped seed. Owned and
serialized exact checkpoints retain accepted progress, models, and diagnostic
history, while numerical scratch is rebuilt as needed. The removed SLSQP state
and its serialized layout have no compatibility reader. Vector indexing and
`VectorLen`, along with matrix shape and entry access, remain the backend
requirements. Every supported dense backend release supports `f32` and `f64`.

External constrained first-order solvers can also use
`SelectedFirstOrderState::replace(x, cost, gradient, violation)`, which rejects
mismatched dimensions without changing progress, followed by `select_current()`
when they select an incumbent. The executor supplies publication counts and
iteration metadata. The state preserves a pending selection even if another
current record replaces it before that boundary.

Nelder-Mead now has the type `NelderMead<V, F = f64, Mode = Unbounded>` and uses
`SimplexProgress<V, F>`. Replace `BasicSimplexState::new(x)` and
`from_simplex(vertices)` with the corresponding `SimplexProgress` constructors,
or keep using `Executor::from_start`. An explicit scalar annotation becomes
`NelderMead::<V, f32>`; `.projected()` selects the third type parameter. The
solver owns its three scratch vectors and adaptive coefficients. The state owns
the authoritative simplex, a separate historical incumbent, and all raw counts.
Current vertices can change without overwriting an older best point; NaN and
positive infinity do not establish incumbents.

`SimplexProgress::replace(vertices, costs)` validates the complete record before
mutation and returns `SimplexShapeError` on a structural mismatch. It sorts
matching pairs with NaNs last and preserves the order of ties. `take_vertices()`
transfers the arrays for in-place work without copying them; restore them with
`replace` before publication. Checked `current()` and `evaluated_vertices()`
return `None` for an unevaluated seed or drained state. Initial simplex
construction rejects malformed shapes. A simplex has at least two same-length
vertices and at most `n + 1` vertices in `n` coordinates; embedded subspace
simplexes remain supported. Degenerate geometry is allowed.

Nelder-Mead's vector bounds now include `VectorLen`, in addition to `Clone`,
`ScaleInPlace<F>`, and `ScaledAdd<F>`; projection also requires `ClampInPlace`.
`IntoInitialSimplex<V, F = f64>` now takes a relative step of type `F` and has a
blanket implementation for `Clone + VectorLen + VectorIndex<F>`. It supports all
backend versions and both scalar types. Remove custom implementations that
overlap that blanket, and pass custom geometry through `from_simplex`. Fresh
runs reevaluate all vertices, reset progress, and rebuild the workspace and
coefficients, including after dimension changes. Exact checkpoints retain solver
and progress together; old state payloads containing scratch cannot be read as
the new state. Simplex-collapse convergence retains its existing semantics and
does not treat an unchanged best vertex as a collapsed simplex.

GBNM now has the type `Gbnm<V, F = f64>` because it owns a projected
`NelderMead<V, F, Projected>`. Replace explicit `Gbnm::<f32>` annotations with
`Gbnm::<V, f32>`; ordinary `Gbnm::new(seed)` calls still infer the vector. Its
outer state is now `PointState<V, F>`; replace `GbnmState::new(x)` with
`PointState::new(x)`, or use `Executor::from_start`. The solver owns its active
local simplex, workspace, search starts, retained local optima, and restart RNG.
Move `vertices()`, `costs()`, `search_starts()`, `local_optima()`, and
`restart_count()` calls from state to solver. Obtain the solver through
`run_with_solver()` or `Stepper::solver()`.

GBNM's current record is the best vertex of its active search and can worsen
after a restart. Its shared state retains a matching historical incumbent. Fresh
initialization resets the RNG and restart history and evaluates a regular
simplex around the current seed, including after dimension changes. Local
simplex collapse still triggers a restart, not convergence of the global
algorithm; `PointState` does not implement `SimplexState`. Raw counts retain
their categories, and NaN or positive infinity cannot establish an incumbent.
Exact checkpoints retain all local geometry and history with the solver. Old
`GbnmState` payloads are incompatible; a shared state alone is a warm-start
seed, not an exact continuation snapshot.

Replace `BasicPopulationState::with_size(n)` with `PopulationProgress::empty()`.
Population size has one owner: `RandomSearch::new(lambda, seed)`, or the
`with_pop_size(n)` setting on `De` and `Ssga`.
`PopulationProgress::from_population` supplies explicit members, which these
solvers now honor. Members must be finite, match the bounds' dimension, and have
the configured population size. Fresh initialization projects them into the
finite box and reevaluates every member. It resets iterations, incumbents,
counts, and the solver's configured RNG. Reusing a final population therefore
starts a fresh search from those members; supply an empty state to reproduce the
original seeded run. Exact continuation skips this reset and requires the
solver, population, and counts together.

`PopulationProgress::replace(members, costs)` validates the complete record
before mutation and preserves member order. It returns `PopulationShapeError`
for empty or inconsistent shapes. `take_members()` transfers storage without
copying; restore the arrays with `replace` before publication. The current
record is the lowest-cost member, or another evaluated point supplied by
`publish_representative`, such as a distribution mean or a swarm's global best.
Incumbent selection considers both samples and this representative and retains
matching historical records even when the whole population is replaced. Checked
`current()`, `best()`, and `evaluated_members()` report missing records
explicitly.

The `PopulationState` trait no longer requires ascending costs: generic readers
must inspect the costs instead of assuming member zero is best. Random search,
DE, SSGA, and DE injection still sort their records by cost. Their optional
cost-change and step tests compare generation representatives, including
generations whose elite remains unchanged. The shared state stores no gradient,
even when a DE-injection inner solver uses one. All six raw evaluation
categories survive composition, and `cost_evals()` counts only cost calls.

`RandomSearch` now requires `VectorLen`, `Index<usize, Output = F>`, and
`IndexMut<usize, Output = F>` in addition to `SampleUniformBox` and `Clone` so
it can validate and project explicit seeds. All four dense backends support
these bounds for both scalar types. `De<F>` and `Ssga<F>` now require
`F: Scalar` on the types themselves. Their constructor signatures and
DE-injection generics remain unchanged. With `serde`, all four solvers serialize
their live RNGs and settings; DE injection also requires a serializable inner
solver without application hooks. Old population state payloads are incompatible
with `PopulationProgress`.

`GlobalBestPso` now has the type `GlobalBestPso<V, F = f64, R = ChaCha8Rng>`.
Replace scalar annotations such as `GlobalBestPso::<f32>` with
`GlobalBestPso::<V, f32>` and custom-RNG annotations with
`GlobalBestPso<V, F, R>`. The vector bounds remain `Clone`, `VectorLen`,
`SampleUniformBox`, `Index<usize, Output = F>`, and
`IndexMut<usize, Output = F>`; the RNG still requires `Rng + Clone`. Its
boundary and velocity policy enums now require `F: Scalar` on their public
types.

Replace `GlobalBestPsoState::new()` with `PopulationProgress::empty()` and
`from_positions(xs)` with `PopulationProgress::from_population(xs)`. Replace
`from_positions_and_velocities(xs, vs)` with that population constructor and
`solver.with_initial_velocities(Some(vs))`. Velocities correspond to the input
member order and must match its count and dimensions. `None` restores sampled
half-displacement velocities. Fresh initialization repairs positions and
velocities according to the selected boundary policy, evaluates every member,
and resets personal bests, global best, RNG, counts, and working buffers.
Reusing final progress starts from its current members; it does not resume the
previous particle motions. Configured initial velocities are reapplied to the
new input order on every fresh run.

Move `velocities()`, `personal_best_positions()`, and `personal_best_costs()`
from state to the solver obtained through `run_with_solver()` or
`Stepper::solver()`. The solver's `global_best()` returns a checked point/cost
pair. Progress still reports the historical global best as its current
representative, while `candidates()` and `costs()` expose the current swarm. Its
cost-change and step checks therefore retain their global-best semantics. NaN
and positive infinity cannot establish a progress incumbent; an all-rejected
initial swarm still stops with `SolverFailed`, and negative infinity still
triggers the solver's native `SolverConverged` stop.

Use `ExactCheckpoint<GlobalBestPso<V, F, R>, PopulationProgress<V, F>>` to
continue particle motions and RNG draws exactly. `Executor::resume` with a
state-only snapshot is no longer supported. With `serde`, solver-aware
checkpoints include both the configured and live RNGs and all particle models.
Old state payloads are incompatible with the new progress type; deserialize them
with Basin 1.x and export positions and, if needed, velocities for an explicit
fresh initialization.

Replace `CmaEsState::<V, M, F>::new(mean, sigma).with_stds(stds)` with
`PopulationProgress::<V, F>::from_point(mean)` and
`CmaEs::<V, M, F>::new(seed, sigma).with_stds(stds)`. Apply the same change to
`BoundedCmaEs`; both injection wrappers accept the configured base solver and
shared population progress. Solver generic order remains `V, M, F`, with
`F: Scalar = f64`; progress no longer carries `M`. Specify the covariance
matrix type on the solver when inference previously obtained it from state.
The dense vector and matrix capability bounds remain unchanged, with all four
backends supporting `f32` and `f64`. Base CMA solvers also support
`Executor::from_start` because their initial scale is now configured explicitly.

CMA solvers own the distribution mean, covariance, paths, eigenpairs, step size,
derived constants, RNG, and boundary-penalty history. Move `mean()` and `sigma()`
from state to the solver returned by `run_with_solver()` or `Stepper::solver()`;
they return `None` before initialization. Progress holds the evaluated members
and an evaluated mean as its current representative. A fresh run reconstructs
all distribution machinery around the current progress point, reapplies the
configured scale and standard deviations, restarts the original RNG seed, and
reevaluates the first generation and mean. Empty progress cannot seed CMA.
Use `from_point` to supply a mean; explicit unevaluated populations use their
first member as the seed. Resetting shared progress preserves its representative
as an unevaluated seed and clears all evaluated records and counts.

Bounded CMA now publishes clipped points with their **raw objective costs**.
Previously, its state paired unrepaired genotypes with penalized fitness.
The solver retains genotypes and penalized ranking for the unchanged adaptation
rules; inspect `genotypes()` and `penalized_costs()` there. Published members
retain the model's rank order, so their raw costs need not be sorted. The current
record is the clipped mean, and the historical incumbent compares raw objective
costs at published feasible points. Injection uses the same distinction.
`cost_evals()` reports only cost calls; inner derivatives remain in their own
categories through `counts()`. Distribution-size convergence still checks
`σ * max_axis_std` on the solver, with the same strict threshold and stop reason.

Exact CMA continuation requires the complete solver, `PopulationProgress`, and
counts. All four CMA variants support owned checkpoints and, with `serde`,
serialize their models and RNGs when the vector, matrix, scalar, and chosen inner
solver do. Application hooks remain unserializable. Legacy CMA state files
cannot resume these solvers. Load them in Basin 1.x and export a point plus any
initialization settings needed for a fresh 2.0 run.

Replace `MaLsChState::new()`, `MaLsChSwState::new()`, and
`MaLsChGenericState::new()` with `PopulationProgress::empty()`. The solver
now owns every persistent `(inner solver, inner progress)` chain and its
eligibility history. Move `ls_application_count(i)` to the retained solver;
`chain(i)` also exposes a saved pair for inspection. The generic solver is
`MaLsCh<V, LS, F = f64>` with `LS: ResumableInner<V, F>` on the type itself.
The concrete aliases are `MaLsChCma<V, M, F = f64>` and
`MaLsChSw<V, F = f64>`. Scalar settings use `F`; both aliases support all four
dense backends with `f32` and `f64`. Custom operators keep the same resumable
chain contract and can impose narrower capabilities.

`PopulationProgress::from_population(members)` now supplies explicit MA-LS
seeds. Their count must match `with_pop_size`; fresh initialization projects
them into the finite sampling box, reevaluates every member, drops all chains,
and resets the RNG and progress bookkeeping. Reusing a populated result starts
from its members rather than sampling replacements. Use `empty()` to resample.
The unbounded local chain may still move outside the sampling box. MA-LS
reports those actual evaluated points and retains the historical objective
incumbent. Category readers no longer fold inner derivatives into cost calls.
`with_ls_intensity` and `with_nfrec` continue to budget raw cost evaluations;
outer objective and step checks observe the published best member.

MA-LS now supports exact owned checkpoints and `serde` checkpoints containing
the outer RNG, retained chains, eligibility history, population, and all counts.
Serialization requires the inner solver and its associated state to serialize.
Stored chains resume without initialization, while each local segment restarts
its own budgets and convergence controls. Legacy state payloads cannot preserve
that ownership layout: export their population using Basin 1.x and start a fresh
2.0 run, or finish exact continuation in the matching 1.x application.

BFGS now has the type `Bfgs<V, F, M, L>`, with defaults for the scalar, matrix,
and line search. `Bfgs::new()` and `Bfgs::with_line_search(search)` infer the
matrix from `V: DenseBackend<F>`. Every enabled backend version has its own
association. To override the matrix, use
`Bfgs::<V, F, M, L>::with_matrix_and_line_search(search)`. Custom vector
backends implement `DenseBackend`, and the selected matrix must implement
`MatVec`, `MatrixIdentity`, `ScaleInPlace`, and `GeneralRankOneUpdate`.
`Executor::from_start(problem, Bfgs::new(), x)` still infers both solver and
state. The inverse Hessian belongs to the solver; retain it with
`run_with_solver()` and read `result.solver.inverse_hessian()`.

L-BFGS now has the type `Lbfgs<V, F, Mode, L>`, with `Bounded` and
`MoreThuente<F>` defaults. `Lbfgsb<V, F, L>` aliases the bounded mode. Replace
`LbfgsState::new(x, m)` with `FirstOrderState::new(x)` and configure
`Lbfgs::new().with_m_capacity(m)`. This setting now controls every fresh solve,
including standalone execution. Replace `Lbfgs::<Unbounded>::new()` with
`Lbfgs::new().unbounded()` and the unbounded `with_line_search` constructor with
`Lbfgs::with_line_search(search).unbounded()`. Both modes require `VectorLen`,
`Clone`, `Dot`, `ScaledAdd`, and the public
`solver::lbfgs::{AsFloatSlice, AsFloatSliceMut}` traits. All four dense backends
support both scalar types; ndarray storage must be contiguous. Bounded mode
still requires problem-side `BoxConstraints` and a line search implementing
`next_with_bounds`. History and compact work buffers belong to the solver;
retain it with `run_with_solver()` and inspect `history_len()` and
`history_capacity()`. Mode transitions discard the initialized model and require
a fresh run. Changing capacity takes effect on fresh initialization; a
checkpoint retains the capacity with which its model was built. The
projected-gradient and line-search stopping rules are unchanged.

Gradient descent retains the type `GradientDescent<L, V, F>`, and trust regions
retain `TrustRegion<Sub, F, Mode>`. Replace their `BasicState::new(x)` arguments
with `FirstOrderState::new(x)`. All three migrated drivers now require
`V: VectorLen` to validate point/gradient dimensions. Projected gradient descent
has the type `ProjectedGradientDescent<L, F = f64>`; its constructors infer the
scalar from the step or problem, so `Executor::from_start` works with both `f32`
and `f64`. Projection, momentum updates, trust-region acceptance, and rejection
stopping semantics are unchanged. Fresh gradient-descent runs clear momentum,
fresh trust-region runs restore the configured radius, and fresh nonlinear-CG
runs reset conjugacy and line-search history. Exact checkpoints retain these
components. `TrustRegion::radius()` exposes the current radius when the solver
is retained.

Trust-region `gradient_evals()` now counts gradient calls alone. Budget
second-order work explicitly with separate `Gradient`, `Hessian`, and
`HessianProduct` limits, or use `max_evaluations(EvaluationKind::TotalWork, n)`
for the whole solve. Total work also includes objective calls, so adjust the
limit when replacing a budget that formerly folded derivatives together.

SGD retains `Sgd<V, F = f64>` and its batch sampling, momentum, and objective
refresh schedule, but uses `PointState<V, F>` without a gradient capability. A
periodic refresh now publishes the evaluated point and cost together. For
example, with seven samples and batches of two, iteration four reports the point
evaluated at iteration three; it no longer pairs iteration four's working point
with iteration three's cost. Use `with_cost_eval_every(1)` to report every
iterate, or retain the solver with `run_with_solver()` and read
`working_param()` to inspect the latest mini-batch iterate without evaluating
it. Ordinary results retain the last evaluated point. Cost- and step-change
checks run only at refreshes. Replace SGD's `max_gradient_evals(n)` with
`max_evaluations(EvaluationKind::Gradient, n)`, since `PointState` does not
implement `GradientState`. Iteration and raw gradient budgets still count every
mini-batch step; objective stall patience counts those steps too, so choose it
with the refresh period in mind. Exact checkpoints retain the working iterate,
refresh phase, momentum, batch order, and RNG. Fresh state-only runs restart
from the reported evaluated point. Custom vectors keep the existing
`Clone + ScaledAdd<F> + ScaleInPlace<F>` bounds.

Solis-Wets now has the type `SolisWets<V, F = f64>`. Replace
`SolisWetsState::new(x, rho)` with `PointState::new(x)` and configure
`SolisWets::new(seed).with_initial_step_size(rho)`. Replace explicit
`SolisWets::<F>` annotations with `SolisWets::<V, F>` or let the parameter type
infer `V`. The solver owns bias, adaptive step size, success/failure streaks,
and the RNG; retain it with `run_with_solver()` and use `bias()`, `step_size()`,
`success_count()`, and `failure_count()`. The state no longer implements
`RhoState`. Native step-size stopping reads the solver's current value, and
cost- and step-change checks now ignore rejected proposals. Custom vectors need
`Clone`, `VectorLen`, `SampleStandardNormal`, `ScaledAdd<F>`, and
`ScaleInPlace<F>`. Fresh initialization always reevaluates the point and resets
the model and RNG, even when the state already contains a cost. Resume the exact
trajectory with a solver-aware checkpoint.

Simulated annealing retains `SimulatedAnnealing<N, F = f64, R = ChaCha8Rng>` and
now uses `ProposalState<V, F>`. `TemperatureSchedule<F = f64>` and the solver
explicitly bound their scalar by `Scalar`. `Executor::from_start` remains the
usual entry point; explicit callers can use `ProposalState::new(x)`. The solver
owns the neighbor, RNG, cooling age, and reannealing progress. The state retains
the accepted point and its objective, objective-ordered best record, all raw
counts, and `AcceptanceState` counters for `no_acceptance`. Rejected proposals
cannot trigger cost- or step-change convergence. The existing non-finite
acceptance and native stopping rules remain in effect; NaN and positive infinity
do not establish incumbents.

Fresh annealing runs reevaluate the point and copy the configured neighbor and
RNG into fresh working components. The existing `Clone` requirements on these
components move from state seeding to the solver implementation; cloning must
produce independent evolution state. Arbitrary cloneable parameters, including
discrete structures, remain supported without vector math. `seed_chain(seed)`
creates an independent chain from the configured components, without consuming
live randomness. Read `temperature()`, `reannealings()`, and `neighbor()` from
the solver borrowed by `observe_solver`, the retained solver after
`run_with_solver()`, or through `Stepper::solver()` between steps. See
[observing solver diagnostics](#observing-solver-diagnostics) for custom logging.

State-only annealing snapshots no longer implement `ExactResumeState` or work
with `Executor::resume`. Pass them to `Executor::new` for a fresh chain, or
retain solver and state together with `run_with_solver().into_checkpoint()` and
continue with `Executor::resume_from_checkpoint`. Exact serialization includes
configured and live neighbor/RNG components and cooling history; old 1.x solver
and state payloads are incompatible.

External proposal-based solvers can reuse `ProposalState`. Initialize its record
with `replace(x, cost)`, publish accepted moves with `accept_proposal(x, cost)`,
and record rejections with `reject_proposal()`. The executor stamps acceptances
at the completed publication boundary, including mid-step stops; repeated
publication does not refresh their age.

Basin-hopping retains `BasinHopping<I, V, F, S, A>` and now uses
`PointState<V, F>`. Replace `BasicState::new(x)` with `PointState::new(x)`, or
use `Executor::from_start`. Fresh initialization restarts the RNG, hop counters,
perturbation strategy, and acceptance rule, then runs the initial local solve.
Stateful custom strategies implement the new default-bodied `StepTaker::reset`
and `AcceptanceTest::reset` hooks; they need not implement `Clone`. Exact
checkpoints preserve the adapted walk. With `run_with_solver()`, inspect
`solver.step_taker()` and the default strategy's `stepsize()`. Rejected hops no
longer trigger cost- or step-change convergence. The outer state has no gradient
capability; use raw category budgets for derivative work and
`EvaluationKind::TotalWork` for all inner work. `RandomDisplacement`
serialization now includes its configured initial scale as well as its current
adapted scale, so old strategy payloads must also be reconstructed.

Scaled and persistent composition follows the same ownership change.
`MemeticInner::seed_scaled` now takes `&mut self`, allowing the inner solver to
configure its initial model scale before a fresh run. Update custom
implementations and use `InnerExecutor::solver_mut()` at call sites. Solis-Wets
sets its configured initial step size to the supplied scale. Persistent
`ResumableInner` chains initialize once, then skip `Solver::init` on subsequent
segments. Implement `seeded_chain_is_initialized()` only when `seed_chain`
already constructs a complete model and evaluated progress, as Solis-Wets does
from the supplied `fx`. Its default remains `false`. Local-search segments
explicitly restart budgets, convergence history, and progress metadata while
retaining the model; this differs from exact checkpoint continuation, which
retains convergence history and cumulative counts. Shared states provide
`reset_progress()` for segment bookkeeping while preserving the evaluated
record. `MaLsChSwState<V>` now stores `(SolisWets<V>, PointState<V>)` chain
pairs.

The augmented-Lagrangian outer method now publishes `SelectedState<V, F>`. Its
public type remains `AugmentedLagrangianMethod<So, V, F = f64>`, and its
constructors now support both scalar types. Replace `BasicState::new(x)` with
`SelectedState::new(x)` or use `Executor::from_start`. `current()` and `best()`
return checked `(point, objective, constraint_norm)` records. Selection prefers
feasible points, then lower objectives among them; before feasibility, it
prefers lower constraint norms, breaking exact ties by objective. The configured
feasibility tolerance defines eligibility, with exact zero when disabled. The
selected objective can increase as feasibility improves. Cost- and step-change
tests observe only feasible outer iterates.

`SelectedState` provides `replace(x, cost, violation)` and the explicit
`select_current()` operation for external solvers. Call the latter only when
selection changes. The executor records the final charged counts and completed
iteration number at publication, including mid-step stops; repeated observation
does not refresh the incumbent's age. It has no gradient capability or
`ObjectiveIncumbentState` guarantee. Use a `stop_when` hook that checks both
violation and objective for constrained targets. `target_objective` and
`no_objective_improvement` deliberately reject this state. The Basin 1.x
`target_cost` and `no_improvement` controls did not check feasibility and must
not be used for these selections. Budget gradients with
`max_evaluations(EvaluationKind::Gradient, n)`. The outer no longer computes an
unused original-objective gradient; surrogate gradients remain charged inner
work. Fresh runs reset multipliers and penalty history, and exact checkpoints
preserve them. Adapter work now merges into the outer counts before propagating
a hard callback error. The vector and matrix capabilities are unchanged; custom
inner solvers can require additional ones.

`BarrierMethod<So, F = f64>` now uses `PointState<V, F>` and supports both
scalar types through its constructors. Replace `BasicState::new(x)` with
`PointState::new(x)` or use `Executor::from_start`. Its current record contains
the original objective inside the strict domain `A x < b`, and `+∞` outside it.
Phase I computes this rejection value without calling the original objective or
gradient. Only strictly feasible points with eligible objectives can become
incumbents, so objective-based controls remain compatible. Cost- and step-change
convergence runs only at accepted Phase II boundaries. The outer state no longer
advertises a gradient, and the outer loop no longer computes one. Inner gradient
calls remain charged in their own category; use
`max_evaluations(EvaluationKind::Gradient, n)` to budget them. Adapter counts
survive hard callback errors in either phase. Fresh runs reset the phase and
barrier schedule and reevaluate the seed; exact checkpoints retain both, along
with the inner solver. The vector and matrix bounds are unchanged, and custom
inner solvers can impose additional capabilities.

`BasicState` has been removed now that all of its shipped solvers use shared
progress. External solvers should choose `PointState`, `FirstOrderState`, or
`SelectedState` according to the records and selection policy they publish, and
populate them through their public coherent-update methods.

Fresh runs clear progress counters and incumbents, reevaluate the seed, and
reset the solver's evolving machinery. Reusing BFGS after a dimension change
creates an identity matrix of the new size. Stateful custom line searches must
implement `LineSearch::reset` to restore their configured starting history.
Exact solver-and-state checkpoints skip initialization and retain model,
line-search, and convergence history. Ordinary results retain progress; progress
alone cannot resume the exact trajectory.

A state-only snapshot at iteration 12 starts at iteration zero when passed to
`Executor::new`; `max_iter(8)` then runs eight more steps. Exact checkpoint
resumption retains iteration 12, so `max_iter(20)` allows eight more steps.

Shared-state `cost_evals()` now means cost calls only, and `gradient_evals()`
means gradient calls only. Residual, Jacobian, Hessian, and Hessian-product work
remains available separately in `counts()`. `best_counts()` records all
categories at the publication boundary that selected the incumbent. Use
`max_evaluations(EvaluationKind::TotalWork, limit)` to budget an explicit sum
instead of relying on a state-specific fold. Category-specific budgets and
readers count the same work. Least-squares runs typically make zero cost calls;
replace `max_cost_evals(n)` with `max_evaluations(EvaluationKind::Residual, n)`
and read `state.counts().residual_evals` and `state.counts().jacobian_evals`.
Their objective remains one-half the squared residual norm, or the configured
robust objective. Native first-order checks still use the solver's `Jᵀr`;
`PointState` does not advertise an unevaluated gradient.

Scalar solvers still expose their latest evaluated probe, which can be worse
than their retained incumbent. Cost-change convergence tests read that current
probe. Shared-state incumbents retain strict objective improvements; ties
preserve their original metadata, and NaN or positive infinity cannot establish
an incumbent. Negative infinity does not by itself establish unboundedness. Use
checked `best()` access when every published cost can be rejected.

With `serde`, the shared progress types, random search, DE, SSGA, DE injection,
global-best PSO, all CMA variants, MA-LS chains, Nelder-Mead, GBNM, NEWUOA,
BOBYQA, LINCOA, COBYLA, MADS, SLSQP,
BFGS and L-BFGS models (including bounded work buffers), gradient descent
(including momentum), projected gradient descent, SGD (including its RNG and
unpublished working iterate), nonlinear CG, Solis-Wets, simulated annealing,
basin-hopping, the barrier and augmented-Lagrangian methods, trust regions and
their built-in subproblem strategies, built-in line searches, and scalar solvers
support serialization. Least-squares solver serialization remains subject to
each solver's and backend's existing support. Their layouts replace the old
state layouts; a 1.x state payload is not a 2.0 shared-state payload, even when
both use postcard. Export parameters with the matching 1.x application and
create a fresh shared state. Exact checkpoints require matching concrete types
and Basin versions, as described under [checkpoint files](#checkpoint-files).

## Stopping conditions

Replace `target_cost` with `target_objective` and `no_improvement` with
`no_objective_improvement`. Both require `ObjectiveIncumbentState` rather
than `State` alone. `SelectedState` and `SelectedFirstOrderState` deliberately
do not implement that capability: their solvers can prefer feasibility over a
lower objective. Use an application stop that checks both the selected
constraint violation and objective when that is the intended stopping rule.
Custom states must implement the checked incumbent and objective-selection
capabilities to use these helpers.

Targets wait for an eligible incumbent. Stall checks count completed iterations,
not repeated observations; they require positive patience and wait until an
incumbent exists. Zero-delta checks retain stall age through the incumbent's
publication iteration on exact continuation. Positive-delta checks start a new
anchor when controls are reattached. Negative infinity can be an incumbent and
does not by itself establish unboundedness. Execution controls remain separate
from solver convergence settings.

Basin 2.0 removes `TerminationCriterion`, every shipped criterion type, and all
criterion re-exports. `Executor::terminate_on`, `InnerExecutor::terminate_on`,
and composed solvers' `inner_terminate_on` methods are also removed. Configure
numerical convergence on the solver and execution limits on `Executor`,
`InnerExecutor`, or `RunControl`.

  | Removed criterion                                        | Replacement                                                                             |
  | -------------------------------------------------------- | --------------------------------------------------------------------------------------- |
  | `MaxIter`, `MaxCostEvals`, `MaxGradientEvals`, `MaxTime` | `max_iter`, `max_cost_evals`, `max_gradient_evals`, `max_time` on the executor          |
  | `TargetCost`, `NoImprovement`, `NoAcceptance`            | `target_objective`, `no_objective_improvement`, `no_acceptance` on the executor          |
  | `GradientTolerance`, `RelativeGradientTolerance`         | `with_absolute_gradient_tolerance`, `with_relative_gradient_tolerance` on the solver    |
  | `ProjectedGradientTolerance`                             | `with_absolute_projected_gradient_tolerance`; bounds come from the problem              |
  | `ParamTolerance`, `RelativeParamTolerance`               | `with_absolute_step_tolerance`, `with_relative_step_tolerance`                          |
  | `CostTolerance`, `RelativeCostTolerance`                 | `with_absolute_cost_change_tolerance`, `with_relative_cost_change_tolerance`            |
  | `SimplexTolerance`                                       | `with_absolute_simplex_size_tolerance` and `with_absolute_simplex_cost_tolerance`       |
  | `CmaEsTolerance`                                         | `with_absolute_distribution_size_tolerance`                                             |
  | `RhoTolerance`                                           | `with_absolute_radius_tolerance`, or `with_absolute_step_size_tolerance` for Solis-Wets |
  | `MeshTolerance`                                          | `with_absolute_poll_size_tolerance`                                                     |
  | Custom criterion                                         | `stop_when` or `stop_when_factory`                                                      |

Setters are available only on solvers that support the corresponding test.
Settings that need additional state or vector capabilities return a
`ConfiguredSolver<...>`; prefer inferred types when composing these setters. The
compiler checks each test's required capabilities. Optional tolerances accept
`None` to disable a test, and zero requests an exact-zero threshold. Native
solver tests retain their documented norms and scaling, which can differ from
the removed generic criteria. Distinct configured tests combine with OR; simplex
size and cost spread combine with AND. Repeating a setter replaces its setting
instead of appending a criterion.

```rust
let solver = GradientDescent::new(0.1)
    .with_absolute_gradient_tolerance(1e-8);
let result = Executor::from_start(problem, solver, x0)
    .max_iter(1_000)
    .max_cost_evals(10_000)
    .stop_when(|state| {
        (state.iter() >= 500).then(|| ApplicationStop::new("user_requested"))
    })
    .run()?;
```

Execution controls run before solver convergence, including at the initialized
iteration-zero boundary. Their fixed order is iteration budget, cost budget,
gradient budget, raw evaluation budgets, time budget, target, improvement stall,
acceptance stall, and custom hooks in insertion order. This replaces criterion
registration order. Budgets are checked at boundaries; initialization and an
in-progress iteration can exceed an evaluation limit. A partial stopping step
updates evaluation counts without incrementing the completed-step count.
`TerminationReport`, `StepOutcome`, and optimization results describe why
the run stopped. A budget, target, application stop, or numerical safeguard does
not establish convergence. Native and shared convergence details are retained
in the report from ordinary `run()`.

Replace a custom criterion's `check` implementation with a closure returning
`Option<ApplicationStop>`. For reusable inner solves, move its constructor and
reset logic into a factory:

```rust
let inner = InnerExecutor::new(solver).stop_when_factory(|| {
    let mut checks = 0;
    move |_| {
        checks += 1;
        (checks == 3).then(|| ApplicationStop::new("user_requested"))
    }
});
```

`stop_when` retains captured history across borrowed runs; `stop_when_factory`
creates fresh history for each run. Composed solvers expose
`inner_stop_when_factory`. Fresh runs reset solver convergence history and
built-in clocks and stall anchors. Exact solver-and-state continuation preserves
solver convergence history; attach execution controls again. State-derived
acceptance and zero-delta improvement checks retain history through the state.

Replace `run_loop(problem, state, solver, criteria, max_iter)` with
`run_loop_with_control(problem, state, solver, &mut control)`, constructing the
controls with `RunControl::new().max_iter(max_iter)` and the appropriate
builders. The borrowed driver retains per-run evaluation counts and the same
initialization, mid-step-stop, and error-routing contracts.

Implementors of `ResumableInner` must replace `segment_criteria` with
`configure_segment(&mut self, state, control)`. Set numerical tolerances on
`self` and configure application stops on `control`. The default does nothing.
CMA-ES still derives its default distribution tolerance from each segment's
starting sigma unless the caller explicitly configured it.

Criterion objects have no serialized replacement. Inner executors still reject
serialization of closure hooks, targets, stalls, and capability controls instead
of silently discarding them; ordinary iteration, evaluation, and time budgets
remain serializable. Do not assume solver payload compatibility across major
versions. Follow the [checkpoint migration instructions](#checkpoint-files) when
moving a saved run from 1.x to 2.0.
