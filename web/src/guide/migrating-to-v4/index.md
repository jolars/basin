# Migrating to Basin 4.0

## Checkpoint features

If you enabled `serde` in Basin 3.x to use `CheckpointWriter`,
`read_checkpoint`, `ExactCheckpointWriter`, or `read_exact_checkpoint`, switch
to `checkpoint-postcard`. This feature enables Serde and both native Postcard
file APIs, including the exact writer's status and error types.

Keep `serde` if your application serializes Basin types with its own codec.
Choose a codec that preserves the values your solver and state can contain,
including non-finite floats if applicable. For stored exact checkpoints,
record and check the format version, exact Basin version, and concrete solver
and state types before decoding. Use `ExactCheckpoint`, `CheckpointSink`,
and `Executor::resume_from_checkpoint` for in-memory continuation without
enabling either feature.

## Existing checkpoint files

`read_exact_checkpoint` checks the file format, exact Basin package version,
and concrete solver and state types before decoding the payload. To continue
a 3.x run exactly, open the checkpoint with the Basin version that wrote it.
To start a new run in Basin 4.0, export the saved point with that version and
pass it to `Executor::from_start` with a newly configured solver.

State-only files can seed a fresh run through `read_checkpoint` and
`Executor::new` when the concrete state's serialized layout matches. If the
layout differs, use the original version to load and export the point.
