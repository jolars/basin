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
