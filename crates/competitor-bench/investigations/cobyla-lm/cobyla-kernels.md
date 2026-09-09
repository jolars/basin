# COBYLA kernel and parameter-buffer follow-up

The public executor runs 17–23% faster than the first optimized implementation
on the three migration cases. Results, feasibility, evaluation counts, and
termination behavior are unchanged. This gets the sphere and quadratic closer to
`cobyla` 1.0.2, but does not reach runtime parity: the public executor remains
about 3.0, 1.3, and 1.5 times slower on camel, sphere, and quadratic.

This follows the [workspace optimization](cobyla-optimization.md). Measurements
were completed September 9, 2026. The baseline is `d11d7e9` (its COBYLA sources
are those of `01e09eb`); the candidate contains the changes described below. No
public API, tolerance, backend requirement, or dependency changed.

## What changed

- Constraint-model construction reduces contiguous slices instead of indexing
  both operands inside the reduction. It preserves each component's arithmetic
  order while allowing the compiler to remove repeated bounds checks.
- Inverse residuals use fixed-size scratch for dimensions 1–4, allowing tiny
  matrix products to stay in registers. The generic kernel still handles larger
  dimensions. Zero skipping and the original NaN reduction behavior remain.
- A vertex update passes its just-computed inverse residual to repoling. It is
  reused only if repoling leaves the matrices unchanged. For dimensions above
  four, the shared update workspace also caches the most recent residual and
  both input matrices. Reuse requires exact scalar equality, including zero
  signs; NaNs in either input force a fresh product. Every recovery threshold,
  inverse-repair attempt, acceptance test, and rollback remains in place.
- Two-element signed and absolute dot products use unrolled reductions, still
  through `Scalar::sum` to retain its identity and summation order.
- The public solver fills its existing backend parameter buffer for callbacks
  and copies the selected incumbent directly from a borrowed driver slice.
  `CobylaState::update_best` uses `clone_from` to reuse the previous snapshot's
  allocation. The callback buffer and best snapshot remain independent.

## Timing and numerical comparability

AMD Ryzen 9 7900, NixOS, Rust 1.89.0, release optimization, thin LTO, one
codegen unit, and default CPU code generation. Both prebuilt probes used the
same isolated lockfile, including `cobyla = 1.0.2`, nalgebra 0.34.2, and ndarray
0.16.1. Basin used `default-features = false, features = ["nalgebra"]`; the
direct driver and executor workloads used `Vec<f64>`. Allocation accounting was
disabled. Processes were pinned to CPU 2.

Each of 15 rounds randomized the case and contestant order. Every process
performed 16 untimed warmup solves, then timed 5,000 fresh camel solves, 400
sphere solves, or 3,000 quadratic solves. The interval includes setup,
initialization, solving, result extraction, and teardown, but excludes process
startup and the independent verification after the loop. Times below are medians
in microseconds per solve; brackets show the sample interquartile range. The
[timing summary](cobyla-kernels-timing.csv) also records means and standard
deviations.

  | Case                  | Driver before          | Driver after           | Executor before        | Executor after         | Direct `cobyla` 1.0.2  |
  | --------------------- | ---------------------: | ---------------------: | ---------------------: | ---------------------: | ---------------------: |
  | Camel                 |    17.83 [17.81–17.97] |    15.83 [15.76–16.10] |    22.64 [22.50–22.80] |    18.45 [18.40–18.69] |       6.16 [6.14–6.19] |
  | Sphere, 10D           | 584.97 [583.47–587.95] | 451.31 [450.10–454.21] | 614.03 [611.54–616.06] | 470.72 [469.23–473.02] | 360.34 [358.89–361.08] |
  | Constrained quadratic |    29.85 [29.79–29.92] |    26.01 [25.93–26.14] |    35.00 [34.92–35.20] |    28.98 [28.93–29.11] |    19.90 [19.82–20.02] |

The GlobalSearch adapter using local Basin improves from 23.54 to 19.81 us on
camel, 618.28 to 475.58 us on sphere, and 36.15 to 30.44 us on quadratic.
Published website benchmarks were not regenerated.

Settings retain the migration workload: `rho_beg = 0.5`, Basin
`rho_end = sqrt(f64::EPSILON) * 0.5`, objective budgets 50/200/100, and the
original starts and bounds. Camel starts at `[0, 0]` with bounds `[-3, 3]` and
`[-2, 2]`; sphere starts at
`[2.5, -2, 1.5, -1, 0.5, 2.25, -1.75, 1.25, -0.75, 0.25]` within `[-5, 5]`;
quadratic starts at `[0.5, 0.5]` within `[0, 2]` and requires `x + y <= 1.5`.
Bound inequalities and the negated quadratic constraint preserve Basin's
`c(x) <= 0` convention.

  | Case      | Basin / reference objective calls | Basin / reference objective           | Basin violation | Basin stop       |
  | --------- | --------------------------------: | ------------------------------------: | --------------: | ---------------- |
  | Camel     |                           50 / 50 | -1.03162845272129 / -1.03162845338526 |               0 | Budget           |
  | Sphere    |                         200 / 200 |               5.72538e-9 / 9.64569e-7 |               0 | Budget           |
  | Quadratic |                          61 / 100 |             0.124999997365822 / 0.125 |      5.26836e-9 | Resolution floor |

Both budget-limited cases pass the common quality targets in the verifier;
budget exhaustion itself is not convergence. The reference uses native bounds
and zero objective/parameter tolerances and returns `MaxEvalReached` on all
three cases. It is the translated NLopt variant, while Basin follows PRIMA. The
quadratic's slightly lower Basin objective reflects its permitted feasibility
tolerance, not a better feasible optimum. Consequently the crate comparison
measures quality under the migration settings, not identical solver work.
Before/after Basin comparisons do retain identical work.

Raw driver iterations remain 41/148/50. Executor completed iterations are
41/148/49, because the converging quadratic iteration returns before the
executor increments its counter. Basin objective/constraint callback counts are
50/50, 200/201, and 61/61. The sphere still calls constraints after its
objective guard exhausts the budget. Adapter constraint counts include the
initial dimension probe. The comparison script independently recomputes the
objective and feasibility outside the timer and compares all eight probe modes'
numerical summaries between builds.

The retained driver benchmark also covers box-constrained spheres in dimensions
1, 3, 5, 20, and 40. These use `x[i] = ((7*i) % 17)/4 - 2`, bounds `[-5, 5]`,
and `20*n` objective calls with the same radii. See the [scaling
summary](cobyla-kernels-scaling.csv) for seven alternating rounds, uncertainty,
objectives, and callback/iteration counts. These are additional before/after
implementation comparisons, not measurements against `cobyla`. Returned points
and counts match exactly; all independently verified sphere objectives are below
`1e-6`.

## Profile and allocation evidence

Separate optimized profile builds used full debug information and
`-C force-frame-pointers=yes`, sampled with
`perf record -e cycles:u -F 997 --call-graph fp`. Resolved call stacks run from
the probe through `raw_solve` to the numerical kernels. An initial DWARF run had
broken call chains and was replaced by these frame-pointer recordings.

Before the changes, sphere's raw driver owned 97% of all sampled user cycles.
`inv_error` accounted for 35% self time and model construction for 24% self
time. Afterward, inverse checking including cache comparisons owns 26% inclusive
time, with the actual product at 20% self time. Model construction owns 21% self
time. These percentages have the entire sampled process as their denominator;
the inclusive and nested self shares must not be added. Fresh uninstrumented
timings, rather than the changed percentages, establish the speedup.

The remaining camel hotspot is the trust-region LP: 38% inclusive time, with 36%
in `TrstlpWork::solve` itself. Memory copying is another 14% self time. On
sphere, the LP, interpolation models, and remaining inverse products each take
about a fifth of sampled time. A focused LP benchmark with recorded identical
inputs is the next supported experiment, especially for 2D camel.

The independent allocation build reports these requests per solve:

  | Case      | Driver before / after | Executor before / after | Adapter before / after |
  | --------- | --------------------: | ----------------------: | ---------------------: |
  | Camel     |             263 / 263 |               484 / 268 |              591 / 375 |
  | Sphere    |             873 / 875 |              1673 / 880 |            2081 / 1288 |
  | Quadratic |             377 / 377 |               645 / 382 |              775 / 512 |

The inverse cache adds two allocations and 1,600 bytes on the 10D sphere. The
public executor makes 41–47% fewer requests through parameter-buffer reuse. The
[allocation data](cobyla-kernels-allocations.csv) includes cumulative requested
bytes, which are not peak memory. Its baseline is carried forward from the first
optimization's recorded after-values for the identical sources; the new values
were measured afresh. The standalone allocation guard uses simpler callbacks and
measures 213/674/255 requests on these three cases.

## Discarded experiments and verification

- Blocking inverse products into four-row accumulators helped the 2D cases but
  made sphere about 4% slower. It was discarded.
- Specializing the LP's two stages at compile time did not improve sphere beyond
  variation and slowed the 2D cases. It was discarded.
- The retained residual cache uses value comparisons rather than assuming a
  workspace always refers to the live simplex: penalty-search trial matrices,
  inverse repairs, and rollback can all change which matrices it holds.

Verification passed: the full latest pure-Rust backend suite, including PRIMA
fixtures and the four migration traces; explicit `f32` constrained solves in
dimensions 2, 3, and 5; inverse-kernel checks through dimension 17 with zero,
NaN, and infinity inputs; cache invalidation, inverse repair, and rollback
tests; public callback-error and projected-budget tests; allocation guards; all
eight Criterion cases as smoke tests; all-feature workspace Clippy; rustfmt;
public rustdoc; both WASM builds; and Python formatting/lint.

A separate development comparison checked complete callback and iteration traces
on 192 deterministic problems: dimensions 1, 2, 3, 4, 5, 10, 20, and 40,
objective scales `1e-12`, `1`, and `1e12`, varied starts, and unconstrained,
box, and nonlinear constraints. They matched the pre-optimization driver bit for
bit. This broader trace generator remains a development check; the four
migration fixtures and the added kernel/scalar tests are retained regressions.

## Reproduce

Build `reproduce.py --rounds 0` in an isolated checkout of the baseline revision
and again with the candidate, using the same generated lockfile. Preserve each
workspace's uninstrumented `cobyla-timing` binary. The script's separate
`target/release/cobyla_probe` build enables allocation accounting and must not
be substituted for it. Then run:

```sh
python3 crates/competitor-bench/investigations/cobyla-lm/compare_cobyla.py \
  --before /tmp/cobyla-before/cobyla-timing \
  --after /tmp/cobyla-after/cobyla-timing \
  --cpu 2 --rounds 15 --output /tmp/cobyla-timing-samples.csv

cargo bench -p competitor-bench --bench cobyla
cargo test -p competitor-bench --test cobyla_allocations -- --nocapture
```

The first script validates every layer before timing and prints medians and
interquartile ranges. For the added dimension cases, copy the current benchmark
and support file into the isolated baseline checkout and save a Criterion
baseline before measuring the candidate. The measurement session's binaries, raw
samples, lockfile hash, source diff, development probes, and profiles are under
the ignored `target/perf-investigation/cobyla-followup/` directory.
