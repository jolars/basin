# Finite-difference benchmarks

`benches/numdiff.rs` times Basin's public finite-difference wrappers with
Criterion. `verify_numdiff` validates the same inputs and measures raw callback
work and allocation requests in separate passes. The timing executable uses the
system allocator and callbacks without counters. No solver, executor, or
`Problem` request accounting sits between the harness and the wrappers.

## Workloads

All cases use `f64` and run on Vec, nalgebra, ndarray, and faer. The default
backend versions in this crate are nalgebra 0.34, ndarray 0.17, and faer 0.24;
the workspace lockfile selects their patch versions. Keep that lockfile and
feature set fixed when comparing results. `basin-latest` selects nalgebra 0.35
instead. The unbounded finite-difference API currently requires `f64`.

| Operation | Parameter dimensions | Residual dimensions |
| --- | --- | --- |
| Gradient, fused cost/gradient, and Hessian-vector product | 4, 256 | None |
| Hessian and fused cost/gradient/Hessian | 4, 64 | None |
| Jacobian and fused residual/Jacobian | 4, 16, 64, 256 | 16, 4, 256, 64, respectively |

Each operation has forward and central stencils, cheap and expensive callbacks,
and owned and `*_into` variants. Bounded gradients, fused cost/gradient, and
Jacobians also cover active lower bounds, active upper bounds, fixed
coordinates, and interior coordinates. Those bounds repeat every four
coordinates. Fixed coordinates produce zero derivative columns.

The cheap objective is a weighted separable quadratic. The expensive objective
adds 16 cosine terms per coordinate. Residuals combine the same scalar function
with a dense linear term so both tall and wide Jacobians exercise every column.
These are deterministic synthetic workloads, with analytic gradients,
Jacobians, diagonal Hessians, and Hessian-vector products. Validation checks
every output entry, matrix dimensions, fused values, and exact Hessian symmetry
outside the measured region. Tolerances account for each stencil's truncation
and roundoff error.

Hessian-vector products use `FiniteDiff`'s fallback that differences its own
numerical gradients, with the selected stencil at both levels. A fixed step of
`1e-3` avoids amplifying roundoff through two nested differences. The other
operations use the wrappers' default adaptive steps. These HVP cases do not
measure differencing an analytic gradient.

Input construction, wrapper construction, output sizing, validation, and the
first warm-up request are outside timing and accounting. Each measured request
uses the same immutable point and direction. Owned calls include allocating
their results and dropping the previous result. `into_reused` calls repeatedly
overwrite already sized outputs, including residuals in fused calls. Both
charge the kernel's internal allocations and scratch cleanup. Fixture teardown
after the final request is excluded. These are steady-state measurements, not
cold output-resizing measurements.

## Validate and count work

Run the accounting executable separately for a sequential build, a parallel
build with one worker, and a parallel build with several workers:

```sh
mkdir -p target/numdiff
cargo run --release -p competitor-bench --bin verify_numdiff -- \
  --repeats 5 > target/numdiff/sequential.csv
RAYON_NUM_THREADS=1 cargo run --release -p competitor-bench \
  --features parallel --bin verify_numdiff -- \
  --repeats 5 > target/numdiff/parallel-t1.csv
RAYON_NUM_THREADS=4 cargo run --release -p competitor-bench \
  --features parallel --bin verify_numdiff -- \
  --repeats 5 > target/numdiff/parallel-t4.csv
```

Each CSV row reports **totals over the `repeats` column**, not per-request
averages. Divide by that column for a per-request comparison. `--filter
SUBSTRING` selects matching benchmark names, and an unmatched filter fails.
For example, `--filter numdiff/vec/hessian/cheap/central/` selects the two Vec
Hessian sizes and both output APIs. Without a filter, there are 832 rows.

The callback pass counts calls at the wrapped function boundary, including
callbacks on Rayon workers. `cost_calls` counts underlying objective calls.
`residual_calls` and `residual_into_calls` distinguish the two residual APIs;
their sum is the number of underlying residual evaluations. An owned residual
callback's internal use of its own fill method counts once. These numbers are
different from `Problem::counts()`, which counts derivative requests rather
than the raw probes. The executable checks that repeated calls preserve the
single-call callback counts and validates the final outputs.

The allocation pass uses callbacks without counters and a process-wide
counting system allocator. It includes all Rayon workers. The global pool and
each workload are warmed before measurement. `alloc_calls` includes successful
`alloc` and `alloc_zeroed` requests; `realloc_calls` counts successful reallocations.
Their sum is the allocation request count. `requested_bytes` sums the full size
of each request, including the new size of a reallocation. It measures
cumulative requested bytes, not peak or retained memory. Pool scheduling can
change parallel allocation totals between runs, so retain repeated measurements
and avoid exact parallel allocation assertions. CSV formatting, analytic
validation, fixture setup, and callback counter storage are excluded.

## Time sequential and parallel builds

Use separate processes and named baselines. A one-worker parallel build still
runs Rayon code, so measure the build without `parallel` as well. `RAYON_NUM_THREADS`
controls the global pool. The executables print the actual execution mode and
pool size to stderr before measurement. Benchmark names deliberately omit the
mode so Criterion can compare identical inputs across builds.

```sh
cargo bench -p competitor-bench --bench numdiff -- --list
cargo bench -p competitor-bench --bench numdiff -- \
  --save-baseline numdiff-sequential
RAYON_NUM_THREADS=1 cargo bench -p competitor-bench --features parallel \
  --bench numdiff -- --save-baseline numdiff-parallel-t1
RAYON_NUM_THREADS=4 cargo bench -p competitor-bench --features parallel \
  --bench numdiff -- --save-baseline numdiff-parallel-t4
```

Compare the stored samples without timing again:

```sh
RAYON_NUM_THREADS=1 cargo bench -p competitor-bench --features parallel \
  --bench numdiff -- --baseline numdiff-sequential \
  --load-baseline numdiff-parallel-t1
RAYON_NUM_THREADS=4 cargo bench -p competitor-bench --features parallel \
  --bench numdiff -- --baseline numdiff-sequential \
  --load-baseline numdiff-parallel-t4
```

Criterion retains samples and confidence intervals under `target/criterion`.
Use its regex filter to run a focused subset, with the same filter in every
configuration. For example:

```sh
cargo bench -p competitor-bench --bench numdiff -- \
  'numdiff/vec/(gradient|jacobian|hessian|hessian_product)/cheap/central/' \
  --save-baseline numdiff-sequential
```

`cargo bench ... -- --test` runs every selected case once with validation and
without collecting timings. If you invoke a prebuilt Criterion executable
directly, pass `--bench` to collect timings; without that flag, Criterion runs
in test mode. The accounting executable's instrumentation changes execution
costs and must never supply performance timings.

Record the revision and local diff, lockfile, `rustc -Vv`, `cargo -V`, CPU,
available CPU affinity, build flags, feature set, actual thread count, exact
commands, and machine load alongside saved measurements. Build before timing,
keep other CPU-heavy work out of the measurement window, and repeat or alternate
prebuilt binaries when drift matters. Compare matching names and callback work
before attributing timing changes to implementation cost. Small effects within
the confidence intervals are inconclusive. The harness does not refresh any
published website data.
