# Benchmarking Basin

Run these commands from the Basin repository root.

## Reuse the existing harnesses

Inspect implementation and manifests before trusting benchmark comments; some
older comments still call `competitor-bench` by its former name, `lm-bench`.

  | Question                                      | Starting point                                                                     |
  | --------------------------------------------- | ---------------------------------------------------------------------------------- |
  | Gradient-descent step versus full run         | `crates/basin/benches/rosenbrock.rs` (custom timing loop, not Criterion)           |
  | Same solver across backends and sizes         | `crates/basin/benches/solver_backends.rs`                                          |
  | LM matrix kernels versus full solve           | `crates/basin/benches/lm_backends.rs`                                              |
  | LM versus `levenberg-marquardt`               | `crates/competitor-bench/benches/compare.rs` and `src/bin/verify.rs` in that crate |
  | GD or Nelder-Mead versus argmin               | `crates/competitor-bench/benches/gd_nm.rs` and `src/bin/verify_gd_nm.rs`           |
  | Private COBYLA driver and allocation guards   | `crates/competitor-bench/benches/cobyla.rs` and `tests/cobyla_allocations.rs`      |
  | Public COBYLA versus the `cobyla` crate       | `crates/competitor-bench/benches/gd_nm.rs` and `src/bin/verify_gd_nm.rs`           |
  | Convergence across libraries or Basin solvers | `crates/competitor-bench/src/bin/trace.rs` or `solver_compare.rs`                  |
  | Population solver thread scaling              | `crates/competitor-bench/benches/population_scaling.rs`                            |

For example, verify and then list or run a focused comparison:

```sh
cargo run --release -p competitor-bench --bin verify_gd_nm
cargo bench -p competitor-bench --bench gd_nm -- --list
cargo bench -p competitor-bench --bench gd_nm -- 'gd_rosenbrock_n2/'
cargo run --release -p competitor-bench --bin verify
cargo bench -p competitor-bench --bench compare -- 'exp_fit/'
cargo bench -p basin --features nalgebra_latest,faer_latest,problems \
  --bench lm_backends -- 'lm_gram/'
```

Run only cases relevant to the question. Inspect timed boundaries: setup
exclusion in `iter_batched` does not prove contestants charge initialization,
allocation, conversion, and teardown symmetrically. Trace observers can add
cost, and sampling by evaluation versus iteration is not interchangeable.

The LM competitor harness normally matches nalgebra 0.34 on both sides;
`basin-latest` changes Basin's side. Avoid workspace-wide or `--all-features`
timings that silently change backends or enable parallelism. Record resolved
backend versions and features; moving `*_latest` aliases are not lasting pins.

For COBYLA, read `crates/competitor-bench/investigations/cobyla.md` for the
completed investigation, comparison asymmetries, retained regression guards, and
archived experiment provenance. The public comparison is part of `gd_nm`; the
private driver retains its own benchmark. For LM, read
`crates/competitor-bench/investigations/lm/README.md` and its production
verification commands. Neither maintained comparison needs GlobalSearch.
Historical measurements are leads to remeasure, not current facts.

Extend a nearby harness when needed. Keep native competitors and instrumentation
in `competitor-bench` or an isolated investigation workspace. Profile the actual
production implementation; avoid a copied kernel that can drift from it, and do
not expose private internals as new public API solely for a probe.

## Account for Basin's execution model

- `crates/basin/src/core/problem.rs`: `Problem::counts()` holds authoritative
  per-kind `EvalCounts`. A fused `cost_and_gradient` call increments both the
  cost and gradient counters; `residual_and_jacobian` likewise increments two
  counters. Their sum is not a count of separate user callback invocations or an
  estimate of elapsed cost. Record whether the problem overrides these fused
  methods when comparing callback work.
- `crates/basin/src/core/state.rs`: `CountsMirror` maps per-run wrapper deltas
  into the concrete state's counters. Some derivative-free outer states fold all
  evaluation kinds into `cost_evals()`, including work from inner solvers.
  Inspect the state's mapping before comparing that number with a reference's
  objective-evaluation count.
- `crates/basin/src/core/numdiff.rs`: `FiniteDiff` expands a derivative request
  into multiple underlying cost or residual calls. Count those raw callbacks
  separately when comparing analytic derivatives, finite differences, or
  libraries with different evaluation accounting. Nonlinear constraint-vector
  calls also need separate instrumentation; `EvalCounts` has no constraint
  counter.
- `crates/basin/src/core/executor.rs` and `crates/basin/src/core/solver.rs`: for
  an executor-overhead comparison, a manual loop must call `Solver::init` before
  `next_iter`, honor framework and solver stops, and maintain equivalent state
  bookkeeping. The executor moves the state into `next_iter`; do not presume it
  clones the solver workspace every iteration. A mid-iteration stop can perform
  work without incrementing the completed-iteration count.
- `crates/basin/src/core/inner.rs`: distinguish a shared `Problem` wrapper from
  a fresh wrapper around an adapter. Shared-wrapper counts already include inner
  work; adapter-wrapper counts require the documented roll-up. Match restart,
  warm-start, and resume behavior when timing solver composition.
- `crates/basin/src/core/constraint.rs`: Basin's nonlinear inequalities use
  `c(x) <= 0`. Preserve feasibility when adapting a reference that uses the
  opposite sign. Charge projection, barrier, and penalty work to the layer that
  actually performs it.

Keep the exact backend release fixed for implementation comparisons. Basin's
`nalgebra_v0_34`, `ndarray_v0_17`, and `faer_v0_24` are examples of versioned
features; consult `crates/basin/Cargo.toml` for the intended release. Feature
unification selects the newest enabled release, so inspect the resolved feature
graph with the same features as the benchmark. Do not infer it solely from a
manifest alias or the parameter type's name.

## Measure a stable baseline

Build optimized binaries before measurement. Record revisions and local diffs,
lockfiles, compiler, build flags, CPU, thread counts, workload, and timing
boundaries. Keep compiler settings and dependency versions matched unless they
are the variable under investigation. Use a separate worktree or archive and
separate build outputs for historical revisions; preserve the user's checkout.

Measure uninstrumented code on an otherwise idle machine. Warm up, repeat, and
retain samples plus a median and spread or confidence interval. For tiny
operations, batch enough fresh operations to amortize clock and process costs,
consume results with `std::hint::black_box`, and reset mutable solver state
between independent solves. Measure cold initialization separately if relevant.

Criterion supports a named baseline before a change and comparison afterward:

```sh
cargo bench -p competitor-bench --bench gd_nm -- \
  'gd_rosenbrock_n2/' --save-baseline before
# After the candidate change, using the same feature and build configuration:
cargo bench -p competitor-bench --bench gd_nm -- \
  'gd_rosenbrock_n2/' --baseline before
```

Named results normally live under `target/criterion`; separate target
directories need an explicit plan to retain and compare their samples. For small
effects or machine drift, prebuild both contestants and alternate or randomize
their order across rounds. Hyperfine is useful for a repeated-solve executable,
not for timing Cargo builds or Criterion's fixed-duration statistical harness.

Use the same available CPU affinity for serial contestants when helpful. Record
Rayon and BLAS thread settings. For parallel scaling, use the existing
`population_scaling` protocol with `RAYON_NUM_THREADS=1` and the desired pool
size in separate processes; do not pin a multicore run to one CPU. Report
single-thread latency separately from parallel throughput. Native results do not
establish browser/WASM performance; measure the requested runtime.
