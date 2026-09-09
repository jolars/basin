# Profiling Basin

Run these commands from the Basin repository root.

## Follow the sampled cost into Basin

Use the caller chain to select the layer before opening individual hot leaves:

- Callback evaluation and derivative synthesis:
  `crates/basin/src/core/problem.rs` and `crates/basin/src/core/numdiff.rs`.
- Solver updates and trial steps: `crates/basin/src/solver/` and
  `crates/basin/src/line_search/`.
- Vector operations and backend conversions: `crates/basin/src/core/math.rs` and
  `crates/basin/src/core/math/*_backend.rs`; matrix capabilities are declared in
  `crates/basin/src/core/math/linalg.rs`.
- Driving, termination, and state bookkeeping:
  `crates/basin/src/core/executor.rs`, `crates/basin/src/core/termination.rs`,
  and `crates/basin/src/core/state/`.
- Composition and constraint adapters: `crates/basin/src/core/inner.rs`,
  `crates/basin/src/core/constraint.rs`, `crates/basin/src/core/barrier.rs`, and
  `crates/basin/src/core/augmented_lagrangian.rs`.

The universal vector tier must continue to work across supported backends. Place
richer matrix operations in the capability-based linear-algebra tier; do not
cure one measured backend hotspot by adding a mandatory BLAS dependency or an
unsupported operation to the universal tier.

## Profile a representative case

Use a profile to locate cost and ordinary unprofiled timings to establish a
speedup. Build optimized code with debug information and resolved symbols. For
Linux frame-pointer unwinding, compile with `-C force-frame-pointers=yes` and
record with `--call-graph fp`. Preserve other relevant build flags and use a the
workspace's `profiling` Cargo profile so the optimized binary retains debug
information and lives separately from ordinary release and benchmark outputs.

The development environment provides `perf`, `cargo-flamegraph`, and `samply`.
The `profiling` profile inherits release optimizations, including thin LTO and
one codegen unit, and retains full debug information without stripping. Frame
pointers still require `RUSTFLAGS`. Here is a focused Criterion flamegraph:

```sh
mkdir -p target/perf-investigation
RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C force-frame-pointers=yes" \
cargo flamegraph --profile profiling -p competitor-bench --bench gd_nm \
  --output target/perf-investigation/gd.svg \
  --cmd 'record -e cycles:u -F 997 --call-graph fp -o target/perf-investigation/gd.data' \
  -- --bench 'gd_rosenbrock_n2/basin' --exact --profile-time 10

perf report --stdio --children --no-inline \
  -i target/perf-investigation/gd.data
perf report --stdio --no-children --no-inline \
  -i target/perf-investigation/gd.data
```

The custom `-o` path is used for recording and conversion by
[cargo-flamegraph](https://github.com/flamegraph-rs/flamegraph/blob/v0.6.14/src/lib.rs#L83-L149).

Adapt the benchmark, features, filter, and duration. Verify the filter executes
the intended case. Criterion's `--profile-time` repeats work without statistical
analysis; a sampling profile still includes setup excluded by Criterion's timer.
Use a focused probe if that setup obscures the measured region. For a custom
repeated-solve executable, run `perf record` directly or use
`samply record --save-only --output <profile.json.gz> <executable> <args>`.
Obtain the executable from Cargo's JSON artifacts, not a wildcard that can pick
up stale binaries or `.d` files. Check local tool help when adapting options.

Inspect stack depth and symbol resolution before trusting percentages. Missing
callchains are a recording problem, not evidence that all cost is in leaf
functions. Try another supported unwinder or sampler if needed. If permissions
prevent sampling, report the limitation and use scoped counters or phase
timings. Do not automatically change host-wide perf permissions. Keep diagnostic
builds and allocation instrumentation out of final timing comparisons.
