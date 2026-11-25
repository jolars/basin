# Checkpoint codec experiment

Run from this directory with Rust 1.90 or newer. Bincode-next 3.2 requires
Rust 1.90. The fixture has its own workspace so the experiment does not raise
Basin's MSRV, change the default features, or add dependencies to published
packages.

```sh
cargo test
cargo test --no-default-features
cargo bench --features bench --bench payload
```

The first command compares Postcard, MessagePack tuple and named-field modes,
and bincode-next through Serde. It saves `f64` ndarray BFGS after eight steps
with inverse-Hessian updates and seeded ndarray CMA-ES after eight generations.
CMA-ES refreshes its eigenbasis every three generations, so the split exercises
a saved eigenbasis between refreshes. A fresh executor resumes at the saved
boundary. Every subsequent step compares the serialized solver and state fields
byte for byte, plus the authoritative counts and step outcome. The tests also
check truncation and trailing bytes.

The second command uses rkyv, wincode, and native bincode-next to encode the
same explicit wire type and resume an application-owned solver. The test
reconstructs a Basin checkpoint from the solver, progress, and six raw counters.
Serde is absent from this build's dependency graph. All seven codec paths
preserve the tested `f32` and `f64` bit patterns, including both signs of zero
and infinity and NaNs with distinct payloads.

## What the fingerprint covers

In bincode-next 3.2.0, the Serde entry point produces the same bytes when
`with_fingerprint()` is set. The test runs a real BFGS checkpoint through
native `Encode` and `Decode` with a Serde field adapter. A manual
`Fingerprint` implementation checks an explicit layout revision and rejects a
different revision before decoding. The test uses the same inner payload type
for both revisions. The native derive requires `Fingerprint` on its field
types and cannot infer the Serde layout from them. Fully native fingerprinting
would require native traits throughout the stored type graph or a native
wire type with its own layout.
[Bincode-next 3.2.0 API](https://docs.rs/bincode-next/3.2.0/bincode_next/)

## Basin 4 design

- `Checkpoint` holds the solver, state, and authoritative evaluation counts.
  `resume_from_checkpoint` and `CheckpointSink` work independently of codecs.
  `StateSnapshotWriter` writes progress for warm starts. A borrowed checkpoint
  view could give multiple writers a common entry point for saving without
  cloning.
- Keep direct Serde serialization for same-build continuation. Applications
  should check codec and settings, a format revision, the Basin version, the
  solver and state types, and the dependency builds that determine their
  layout. An exact package version and matching Rust type names can miss
  changes in dependencies, features, or source revisions. Applications also
  supply the same problem and reattach execution policy. Named MessagePack
  fields can help a migration, but their presence alone gives no
  compatibility guarantee.
- Additional Serde codecs can use the existing checkpoint and sink APIs.
  If a native codec needs access to a built-in solver's saved state, try one
  private BFGS wire record first. Measure the conversion and maintenance costs
  before extending coverage. Encode only the state required for exact
  continuation, including the BFGS model and scaling status and the history
  used by stopping rules. Use explicit
  scalar widths and dimensions for numerical data. Check dimensions, counter
  layout, backend reconstruction, and solver invariants before handing the
  restored checkpoint to the executor. Keep codec-specific traits on these
  records. The fixture's native test exercises the resume boundary without
  Serde. Validate native restoration of each built-in solver on each
  supported numerical backend. Resuming a checkpoint into a different
  numerical backend would need a separate trajectory guarantee.
- If Basin supplies file I/O for several codecs, use a shared versioned
  envelope with a codec and configuration identifier, a solver schema
  identifier, and an explicit payload length. Check the envelope and use
  decoder size limits before allocating from payload lengths. Treat a
  fingerprint as a schema check. Use codec-specific byte validation and
  solver-level validation as separate steps. For rkyv, include alignment and
  archive format settings in the contract and use checked decoding when
  materializing a live solver.
  [rkyv checked decoding](https://docs.rs/rkyv/0.8.18/rkyv/fn.from_bytes.html)

## Payload benchmark

Criterion times allocation and encoding into memory and decoding into an
owned checkpoint. Solver initialization, iterations, and file I/O are outside
the timed loops. The benchmark reports the raw payload sizes, with the same
checkpoint passed to each codec. Native codecs are exercised in the tests.
There is no native payload benchmark for the built-in solvers until there is a
native wire record for them.
[Rust serialization benchmark](https://github.com/djkoloski/rust_serialization_benchmark)

The table records one run on an AMD Ryzen 9 7900, pinned to one logical CPU,
using Rust 1.90.0, Criterion 0.8.2, and ndarray 0.17.2. Times are Criterion
median estimates per operation. Postcard used 1.1.3, rmp-serde used 1.3.1,
and bincode-next used 3.2.0 with its standard configuration.

| Solver | Dimension | Codec | Bytes | Encode (µs) | Decode (µs) |
| --- | ---: | --- | ---: | ---: | ---: |
| BFGS | 8 | Postcard | 773 | 0.314 | 0.293 |
| BFGS | 8 | MessagePack tuple | 883 | 0.509 | 0.757 |
| BFGS | 8 | MessagePack named | 1,375 | 0.798 | 1.52 |
| BFGS | 8 | bincode-next Serde | 773 | 0.326 | 0.342 |
| CMA-ES | 8 | Postcard | 2,524 | 1.08 | 1.34 |
| CMA-ES | 8 | MessagePack tuple | 2,920 | 2.26 | 2.71 |
| CMA-ES | 8 | MessagePack named | 3,841 | 3.01 | 4.61 |
| CMA-ES | 8 | bincode-next Serde | 2,526 | 0.655 | 1.32 |
| BFGS | 32 | Postcard | 9,030 | 1.32 | 1.1 |
| BFGS | 32 | MessagePack tuple | 10,177 | 3.2 | 4.18 |
| BFGS | 32 | MessagePack named | 10,669 | 3.45 | 4.93 |
| BFGS | 32 | bincode-next Serde | 9,031 | 1.09 | 1.36 |
| CMA-ES | 32 | Postcard | 22,253 | 2.92 | 3.69 |
| CMA-ES | 32 | MessagePack tuple | 25,161 | 8.65 | 11.3 |
| CMA-ES | 32 | MessagePack named | 26,126 | 8.67 | 13.4 |
| CMA-ES | 32 | bincode-next Serde | 22,254 | 2.43 | 4.09 |
| BFGS | 128 | Postcard | 134,223 | 16.1 | 13.6 |
| BFGS | 128 | MessagePack tuple | 151,014 | 46.8 | 56.7 |
| BFGS | 128 | MessagePack named | 151,506 | 58.6 | 57.1 |
| BFGS | 128 | bincode-next Serde | 134,215 | 13.1 | 16.8 |
| CMA-ES | 128 | Postcard | 288,371 | 31.7 | 30.8 |
| CMA-ES | 128 | MessagePack tuple | 324,529 | 123 | 123 |
| CMA-ES | 128 | MessagePack named | 325,538 | 106 | 132 |
| CMA-ES | 128 | bincode-next Serde | 288,314 | 25.7 | 37.1 |

Postcard decoded fastest at dimensions 32 and 128 for both solvers.
Bincode-next encoded fastest in five of the six cases and produced
nearly the same payload sizes as Postcard.
