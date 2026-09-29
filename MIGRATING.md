# Migrating to Basin 2.0

## Checkpoint files

Basin 2.0 removes bincode and the readers for the two legacy checkpoint formats.
The `serde` feature still enables serialization and, on native targets,
checkpoint file I/O. The checkpoint writers keep their postcard formats:

  | Checkpoint                      | Accepted format                                     | Removed format                            |
  | ------------------------------- | --------------------------------------------------- | ----------------------------------------- |
  | State (`read_checkpoint`)       | `BASINST\0`, version 1, postcard payload            | Unprefixed bincode payload                |
  | Exact (`read_exact_checkpoint`) | `BASINEX\0`, version 2, postcard header and payload | Version 1 with bincode header and payload |

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

## Stopping conditions

Basin 2.0 removes `TerminationCriterion`, every shipped criterion type, and all
criterion re-exports. `Executor::terminate_on`, `InnerExecutor::terminate_on`,
and composed solvers' `inner_terminate_on` methods are also removed. Configure
numerical convergence on the solver and execution limits on `Executor`,
`InnerExecutor`, or `RunControl`.

  | Removed criterion                                        | Replacement                                                                             |
  | -------------------------------------------------------- | --------------------------------------------------------------------------------------- |
  | `MaxIter`, `MaxCostEvals`, `MaxGradientEvals`, `MaxTime` | `max_iter`, `max_cost_evals`, `max_gradient_evals`, `max_time` on the executor          |
  | `TargetCost`, `NoImprovement`, `NoAcceptance`            | `target_cost`, `no_improvement`, `no_acceptance` on the executor                        |
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
        (state.iter() >= 500).then_some(TerminationReason::UserRequested)
    })
    .run()?;
```

Execution controls run before solver convergence, including at the initialized
iteration-zero boundary. Their fixed order is iteration budget, cost budget,
gradient budget, raw evaluation budgets, time budget, target, improvement stall,
acceptance stall, and custom hooks in insertion order. This replaces criterion
registration order. Budgets are checked at boundaries; initialization and an
in-progress iteration can exceed an evaluation limit. A clean mid-step stop
updates evaluation counts without incrementing the completed-step count.
`TerminationReason`, `StepOutcome`, and optimization results still report why
the run stopped. A budget, target, application stop, or numerical safeguard does
not establish convergence. Native convergence details remain available through
`run_with_solver()` and the result's `native_convergence_tests()`.

Replace a custom criterion's `check` implementation with a closure returning
`Option<TerminationReason>`. For reusable inner solves, move its constructor and
reset logic into a factory:

```rust
let inner = InnerExecutor::new(solver).stop_when_factory(|| {
    let mut checks = 0;
    move |_| {
        checks += 1;
        (checks == 3).then_some(TerminationReason::UserRequested)
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
