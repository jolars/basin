# Migrating to Basin 4.0

## Checkpoint and snapshot names

| Use | Basin 3.x | Basin 4.0 |
| --- | --- | --- |
| Store a run for exact continuation | `ExactCheckpoint<So, S>` | `Checkpoint<So, S>` |
| Write a checkpoint with `Executor::checkpoint_with` | `ExactCheckpointWriter` | `CheckpointWriter` |
| Load a checkpoint | `read_exact_checkpoint` | `read_checkpoint` |
| Write a state snapshot with `Executor::observe_with` | `CheckpointWriter` | `StateSnapshotWriter` |
| Load a state snapshot | `read_checkpoint` | `read_state_snapshot` |

Pass a checkpoint to `Executor::resume_from_checkpoint` with the same problem.
The executor restores cumulative evaluation counts and continues without
initializing the solver again. Reapply any execution controls you need. Specify
the solver and state types that wrote the file when you load it.

```rust
type Solver = GradientDescent<Constant<f64>, Vec<f64>>;
let checkpoint: Checkpoint<Solver, FirstOrderState<Vec<f64>>> =
    read_checkpoint("run.ckpt")?;
let result = Executor::resume_from_checkpoint(problem, checkpoint)
    .max_iter(1_000)
    .run()?;
```

For a warm start, use `read_state_snapshot` and pass the concrete state to
`Executor::new`. The solver initializes a fresh run from the saved iterate
with fresh counters.

```rust
let state: FirstOrderState<Vec<f64>> = read_state_snapshot("run.snapshot")?;
let result = Executor::new(problem, solver, state).run()?;
```

## Checkpoint features

If you enabled `serde` for Basin's built-in file APIs in 3.x, switch to
`checkpoint-postcard`. This feature enables Serde and both native Postcard
file APIs, including `CheckpointStatus` and `CheckpointWriteError`.

Keep `serde` if your application serializes Basin types with its own codec.
Choose a codec that preserves the values your solver and state can contain,
including non-finite floats if applicable. Before decoding, check your codec
and settings, format revision, exact Basin version, and concrete solver and
state types, including the dependency builds that determine their layout.
Codecs that serialize the Serde struct name now see `Checkpoint`.

Use `Checkpoint`, `CheckpointSink`, and `Executor::resume_from_checkpoint`
without either feature to capture and continue a run in memory, or to supply
your own persistence code.

## Existing files

`read_checkpoint` checks the file format, exact Basin package version,
and concrete solver and state types before decoding the payload. To continue
a 3.x run exactly, open the checkpoint with the Basin version that wrote it.
To start a new run in Basin 4.0, export the saved point with that version and
pass it to `Executor::from_start` with a newly configured solver.

Open older native state snapshots with the version that wrote them. Export
the saved point and pass it to `Executor::from_start` for a fresh run in
Basin 4.0. The native state snapshot format is now version 2, and the exact
checkpoint format is version 5.

## Checked progress records

Use `current_record()` for the last published point and `selected_record()`
for the point retained by the solver's selection policy. An unevaluated seed
has no record. An available record can have a violation report and no objective
value. An evaluated NaN or infinity remains an available objective value.

For a selected record, read `iter` and `counts` for the completed iteration
and raw evaluation counts at publication. These counts include all callback
requests charged through that boundary. Use the selected record’s optional
`cost` field to check whether the objective has been evaluated.

| Basin 3.x reader | Basin 4.0 replacement |
| --- | --- |
| `cost()` | `current_record().and_then(\|record\| record.cost)` |
| `incumbent_record()` | `selected_record()`, with an optional `cost` |
| `best_param()` | `selected_record().map(\|record\| record.param)` |
| `best_cost()` | `selected_record().and_then(\|record\| record.cost)` |
| `best_iter()` | `selected_record().map(\|record\| record.iter)` |
| `best_cost_evals()` | `selected_record().map(\|record\| record.counts.cost_evals)` |

Custom states implement both record readers on `State`. Return `None` when
the corresponding record is unavailable. For `EvaluatedState`, return the
point and cost from the current record once all advertised evaluations are
available. For `ObjectiveIncumbentState`, expose the eligible incumbent and
its objective through `selected_record()`.

Implement the mapped `cost_evals()` reader on `CountsMirror`. Use
`result.counts` for authoritative run counts.

`ObserverMode::NewBest` now fires only on completed steps that publish the
first eligible objective or a strict improvement.
Switch existing `NewBest` observers and checkpoint sinks on constrained runs
to `ObserverMode::SelectionChanged` to keep receiving solver-selected
incumbents. This mode observes new selections during initialization and
partial or completed steps. Implement
`observe_selection` for a trait observer, match
`ObservationEvent::SelectionChanged` in a closure, or override
`CheckpointSink::save_with_selection` to inspect the reason in a sink.
Custom states report the reason from `State::publish_selection`.
Use `EvaluatedState::evaluated_record()` for a complete evaluated iterate.
