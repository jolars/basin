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

State snapshots can seed a fresh run through `read_state_snapshot` and
`Executor::new` when the concrete state's serialized layout matches. If the
layout differs, use the original version to load and export the point.
