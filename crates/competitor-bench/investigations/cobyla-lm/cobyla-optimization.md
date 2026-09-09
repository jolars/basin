# COBYLA driver optimization

The [remaining-gap investigation](cobyla-variant-gap.md) compares the current
implementation directly with `cobyla` 1.0.2, separating PRIMA-related work,
public integration, and identical-input kernel costs.

The [PRIMA comparison](cobyla-prima.md) continues this work using the pinned
modern Fortran reference. The `cobyla` 1.0.2 comparisons below retain their
original NLopt-derived reference and measurement context.

The [kernel and parameter-buffer follow-up](cobyla-kernels.md) improves public
executor runtime by a further 17–23% and records the remaining gap to `cobyla`.
The measurements below describe the first optimization pass.

The optimized driver runs 1.9–2.8 times faster on the three migration cases and
makes 94–97% fewer allocation requests. The GlobalSearch adapter runs 2.2–2.5
times faster. All eight probe modes retain their baseline objective values,
returned points, objective and constraint counts, and iteration counts. The gap
to `cobyla` 1.0.2 is smaller, but remains measurable.

This follows the [initial investigation](README.md#cobyla-findings).
Measurements were completed September 8–9, 2026. The baseline binary uses Basin
`260c0ffe87bb95e1781a627f244bdf252e81242e`; its Basin sources are identical to
the pre-optimization checkout at `013bb6687258fee75dafcfdd91d93f7ecb553e73`.

## Implementation

- The two LP stages reuse their QR, multiplier, and step buffers across solves.
  Active columns are accessed by index, and absolute dot products consume slices
  directly. This removes allocations inside the active-set loop.
- Simplex updates reuse rollback storage. Inverse validation streams one product
  column at a time, preserving accumulation order, zero skipping, and the
  existing non-finite reduction behavior. Validation and inverse recovery still
  run at every original call site.
- The penalty search reuses its trial simplex and interpolation-model storage.
  Its LP step is reused only when the live objective values, constraint values,
  and inverse match the model inputs after repoling, including zero signs. NaNs
  invalidate reuse. A mismatch rebuilds the model and solves the LP again.
- Geometry scoring reuses buffers and avoids materializing distance vectors when
  an iterator suffices. Constraint-model differences are formed once per
  constraint, preserving each gradient component's summation order.
- The return filter grows geometrically up to its existing 2,000-point limit. It
  retains the same insertion, eviction, and selection rules, and no longer
  stores unused constraint vectors. Most solves retain only a few points.

The public solver API, backend requirements, feasibility tolerance, callback
order, and termination rules are unchanged. Strict objective budgets remain
enforced by the caller's callback guard. The executor may complete an iteration
and call constraints after that guard suppresses an objective evaluation, as in
the baseline sphere case.

## Timing

Environment: AMD Ryzen 9 7900, NixOS, Rust 1.89.0, release optimization, thin
LTO, and one codegen unit. All compared processes were pinned to CPU 2.
Hyperfine used three process warmups and 15 timed runs per command; every
process also performed 16 untimed solves. Solves per process were 5,000 for
camel, 400 for sphere, and 3,000 for quadratic. Allocation instrumentation was
disabled. The original investigation allowed migration between CPUs, so use the
before/after measurements here for the optimization comparison.

Mean microseconds per solve:

  | Case                  | Driver before | Driver after | Speedup | Executor before | Executor after |
  | --------------------- | ------------: | -----------: | ------: | --------------: | -------------: |
  | Camel                 |         50.76 |        19.21 |   2.64x |           55.05 |          22.40 |
  | Sphere, 10D           |       1181.11 |       624.07 |   1.89x |         1198.99 |         644.90 |
  | Constrained quadratic |         85.88 |        30.35 |   2.83x |           91.56 |          35.86 |

  | Case                  | Direct `cobyla` 1.0.2 | Old GlobalSearch adapter | Basin adapter before | Basin adapter after |
  | --------------------- | --------------------: | -----------------------: | -------------------: | ------------------: |
  | Camel                 |                  7.70 |                     6.66 |                56.89 |               23.71 |
  | Sphere, 10D           |                444.94 |                   451.86 |              1453.38 |              673.58 |
  | Constrained quadratic |                 20.50 |                    24.35 |                94.20 |               37.81 |

The optimized adapter is about 3.56, 1.49, and 1.55 times slower than the old
adapter, respectively. Shared-workstation variation affected some measurements,
especially the baseline camel and sphere adapters. The raw-driver and executor
measurements give more stable estimates of the improvement. See [timing
data](cobyla-optimization-timing.csv) for standard deviations.

## Allocations

The unchanged investigation probe records these allocation requests per solve.
Requested bytes are cumulative, not peak memory:

  | Case                  | Driver requests before / after | Driver bytes before / after | Executor requests before / after | Adapter requests before / after |
  | --------------------- | -----------------------------: | --------------------------: | -------------------------------: | ------------------------------: |
  | Camel                 |                     4759 / 263 |               253134 / 5984 |                       5060 / 484 |                      5167 / 591 |
  | Sphere, 10D           |                    27037 / 873 |            7426565 / 101584 |                     28131 / 1673 |                    28539 / 2081 |
  | Constrained quadratic |                     8521 / 377 |               357155 / 8376 |                       8887 / 645 |                      9017 / 775 |

The baseline is retained in [allocation-results.csv](allocation-results.csv).
[Before/after allocation data](cobyla-optimization-allocations.csv) includes all
eight layers and their objective and constraint counts. The new isolated
allocation test uses simpler callbacks than the investigation's raw mode, so its
counts are lower: 213, 672, and 255 requests, respectively.

## Regression coverage and reproduction

The [trace tests](../../../basin/src/solver/cobyla/regression.rs) compare every
callback's inputs and values, every iteration's incumbent and resolution, and
every driver transition against pre-optimization fixtures. They cover all three
migration cases and a non-finite objective/constraint case. Additional tests
cover filter saturation and ties, inverse repair, singular and non-finite update
rollback, immediate callback-error propagation, and projected callbacks with
strict objective guards, including exhaustion during initialization.

A separate development comparison exercised 120 cases over dimensions 1, 2, 3,
5, and 10, with unconstrained, box-constrained, and nonlinear constraints,
varied starts, and objective scales from `1e-12` to `1e12`. Complete callback
and iteration traces matched the original driver bit for bit. This was a
development check; the retained regression fixtures are the four cases described
above.

From the repository root, the new benchmark and allocation guard run without a
GlobalSearch checkout:

```sh
cargo bench -p competitor-bench --bench cobyla
cargo test -p competitor-bench --test cobyla_allocations
```

Both compile Basin's actual private driver sources. The allocation guard also
checks objective quality, evaluation counts, and iteration counts. To repeat the
historical crate and adapter comparisons, use `reproduce.py` as described in the
[investigation](README.md#reproduce), retaining the timing binary from each
checkout before comparing them with Hyperfine.

Verification passed: the full latest pure-Rust backend test suite, including
`f32` and PRIMA parity; the allocation guard; the Criterion benchmark smoke
test; workspace Clippy with all targets and features; formatting; public
rustdoc; and both default and no-default-feature WASM builds.
