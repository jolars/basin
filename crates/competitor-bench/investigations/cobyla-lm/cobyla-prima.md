# COBYLA implementation comparison with PRIMA

Under the matched settings below, Basin's public executor runs 3.7–9.6 times
faster than PRIMA v0.7.2 on the three original migration cases. The retained
1–3D LP specialization reduces kernel time by 19–27% and improves camel and
quadratic executor time by 4.7% and 5.3%, with unchanged numerical results,
callback counts, and stopping behavior.

This continues the [driver optimization](cobyla-optimization.md) and [kernel
follow-up](cobyla-kernels.md), using the modern Fortran PRIMA reference from
which Basin's COBYLA was ported. The earlier `cobyla` 1.0.2 measurements remain
historical comparisons with the translated NLopt variant.

## Reference and workload

The reference is [PRIMA v0.7.2, commit
`e1169927f10fea330c1aae60f12e1a32c45ef5f4`](https://github.com/libprima/prima/tree/e1169927f10fea330c1aae60f12e1a32c45ef5f4),
already pinned at `tools/prima` and used by Basin's numerical fixtures. The
Basin baseline is `bddb3f3a9a7c38dd589ef29c63deedf73d1eefda`, after both earlier
optimization passes. The reproducer snapshots Basin's sources and compiles the
actual private driver, with native reference linkage confined to the isolated
investigation workspace. No GlobalSearch checkout is required.

All contestants use `f64` and the same Rust objective and constraint formulas.
The three original cases retain their starts and bounds:

  | Case        | Start                                                     | Bounds                         | Objective budget |
  | ----------- | --------------------------------------------------------- | ------------------------------ | ---------------: |
  | Camel       | `[0, 0]`                                                  | `[-3, 3] × [-2, 2]`            |               50 |
  | Sphere, 10D | `[2.5, -2, 1.5, -1, 0.5, 2.25, -1.75, 1.25, -0.75, 0.25]` | `[-5, 5]`                      |              200 |
  | Quadratic   | `[0.5, 0.5]`                                              | `[0, 2]²`, with `x + y <= 1.5` |              100 |

The additional spheres have dimensions 1, 3, 5, 20, and 40, starts
`x[i] = ((7*i) % 17)/4 - 2`, bounds `[-5, 5]`, and budgets `20*n`. Both solvers
receive bounds as nonlinear inequality rows, in the same order, using
`c(x) <= 0`. The quadratic's extra row is `-(1.5 - x[0] - x[1])`. There is no
projection or separate native-bound block.

Settings are `rho_beg = 0.5`, `rho_end = sqrt(epsilon)*0.5`,
`ctol = sqrt(epsilon)`, `cweight = 1e8`, `eta1 = 0.1`, `eta2 = 0.7`,
`gamma1 = 0.5`, and `gamma2 = 2`. The investigation-only Fortran binding sets
these explicitly. PRIMA's C API derives `eta2` from `eta1`, which rounds
differently from the literal `0.7`. Objective-target stopping is disabled, and
PRIMA history and printing are disabled.

Both request a 2,000-point return filter. PRIMA's preprocessing caps its
capacity by `maxfun`, giving 20–800 points on these workloads; Basin retains its
existing lazy growth and 2,000-point limit. PRIMA's `maxfun` stops within an
iteration. Basin retains its strict objective callback guard and completes that
iteration, sometimes making one extra constraint-vector call. Neither difference
is patched away for the benchmark.

## Numerical comparison

  | Case        | Basin / PRIMA objective calls | Basin / PRIMA constraint calls | Basin / PRIMA objective                         | Violation    | Stop             |
  | ----------- | ----------------------------: | -----------------------------: | ----------------------------------------------: | -----------: | ---------------- |
  | Camel       |                       50 / 50 |                        50 / 50 |                    `-1.0316284527212884` / same |            0 | Budget           |
  | Sphere, 10D |                     200 / 200 |                      201 / 200 | `5.725381469125405e-9` / `5.725381469125366e-9` |            0 | Budget           |
  | Quadratic   |                       61 / 61 |                        61 / 61 |                    `0.12499999736582204` / same | `5.26836e-9` | Resolution floor |

The [numerical results](cobyla-prima-results.csv) include all eight workloads,
both Basin layers, returned points, native status, and wrapper counters. The
quadratic's objective is slightly below the strictly feasible optimum `0.125`
because both solvers permit its small constraint violation.

Every returned point is checked outside the timer by recomputing its objective
and all constraints. Verification calls are recorded separately. Quality
requires objective error below `1e-5` on camel and the 10D sphere, `1e-6` on
quadratic and the scaling spheres, and violation at most `sqrt(epsilon)`. All
eight workloads pass for both contestants. Budget exhaustion is recorded as a
budget stop, independently of passing these quality targets.

PRIMA and Basin reach different points on the 5D and 20D spheres. The
whole-solver table therefore measures quality under a common budget. It does not
establish identical implementation work, even though the original cases have the
same objective counts. Basin's raw iteration count includes a final converging
step; the executor counts completed iterations, giving 50 versus 49 on
quadratic. PRIMA's binding exposes no iteration count, so it is recorded as
unavailable rather than zero.

For implementation cost, a separate instrumented build records all 1,208 LP
calls from the baseline workloads as `(A, b, delta, g)`. Both implementations
replay these exact column-major inputs. Checks cover step length, linearized
violation, objective reduction, and relative step agreement. Maximum relative
step disagreement with PRIMA is `5.56e-16`; maximum scaled objective/violation
disagreement is below `2.73e-16`. These checks precede kernel timing.

## Measurements

September 9, 2026; AMD Ryzen 9 7900, NixOS, Rust 1.89.0, and GNU Fortran 15.3.0.
Basin uses `Vec<f64>`, no default features, release optimization, thin LTO, and
one codegen unit. PRIMA uses its Release build with `-O3`, 64-bit reals,
`-ffp-contract=off`, and the default heap-array/recursive-call options. Neither
build enables fast math, architecture-specific CPU tuning, or parallelism. The
resolved lockfile and compiler/build metadata are retained with each snapshot
and checked for agreement before comparison.

Each of 15 rounds randomizes contestant, case, and mode order with seed 42.
Every process is pinned to CPU 2 and performs 16 untimed warmup solves or kernel
batches. Complete-solve batches contain 5,000 camel, 400 sphere, or 3,000
quadratic solves. Scaling batches use `max(10, 40000 // (n*n))` solves. Timers
include fresh solver setup, initialization, solving, result extraction, and
teardown. Process startup, JSON output, trace loading, and independent
validation are excluded. The executor probe also extracts the current point to
distinguish it from its selected best point.

Kernel timers reuse Basin's LP workspace, as its production driver does; PRIMA
incurs its ordinary per-call scratch allocation. Basin workspace construction
and destruction are measured separately. PRIMA includes one native binding call
per LP. All timing builds exclude capture, allocation, and profiling
instrumentation.

Times are medians in microseconds per complete solve; brackets show the
interquartile range. The [timing summary](cobyla-prima-timing.csv) also retains
means and standard deviations.

  | Case        | Basin driver before    | Driver after           | Executor before        | Executor after         | PRIMA                     |
  | ----------- | ---------------------: | ---------------------: | ---------------------: | ---------------------: | ------------------------: |
  | Camel       |    15.12 [14.91–15.35] |    14.78 [14.68–15.11] |    18.55 [18.51–18.61] |    17.68 [17.54–17.84] |    158.36 [157.48–160.74] |
  | Sphere, 10D | 458.55 [457.34–464.26] | 456.42 [455.23–460.78] | 476.00 [469.67–481.99] | 471.05 [466.92–486.89] | 1725.38 [1717.46–1745.35] |
  | Quadratic   |    24.91 [24.81–25.18] |    23.43 [23.39–23.75] |    28.82 [28.70–28.97] |    27.29 [27.06–27.35] |    262.83 [262.37–266.17] |

The public executor improves by 4.7% on camel and 5.3% on quadratic. It runs
about 3.7–9.6 times faster than PRIMA on these three workloads. Basin was
already faster than this reference before the change. The 10D sphere’s small
median change is inconclusive.

  | Scaling case | Executor before              | Executor after               | Median reduction |
  | ------------ | ---------------------------: | ---------------------------: | ---------------: |
  | Sphere, 1D   |             6.09 [6.07–6.11] |             5.81 [5.79–5.89] |             4.6% |
  | Sphere, 3D   |          31.37 [31.11–31.81] |          29.11 [28.91–29.21] |             7.2% |
  | Sphere, 5D   |          86.29 [85.78–86.95] |          85.35 [84.77–85.84] |             1.1% |
  | Sphere, 20D  |    4063.43 [4039.57–4123.90] |    4037.45 [4024.89–4064.22] |             0.6% |
  | Sphere, 40D  | 50241.31 [50137.97–50923.56] | 50233.28 [49870.58–51566.47] |             0.0% |

The 1D and 3D executor gains are 4.6% and 7.2%. The larger cases show no
resolved regression. The raw 5D driver’s median increases by 2.0%, but its
paired-round ratio interval includes no change (`0.987–1.022`). The raw camel
gain is also inconclusive despite its lower median. Small general-path
differences are not treated as optimization wins.

The [paired-round intervals](cobyla-prima-ratios.csv) use the median of each
round’s after/before ratios, with 10,000 bootstrap resamples and seed 42. These
95% intervals describe timing variation; the unchanged-source rebuild control
below also illustrates build-to-build variation.

Matched LP timings are **nanoseconds per call**, using recorded baseline inputs:

  | Inputs      | Basin before        | Basin after         | PRIMA                  | Basin median reduction |
  | ----------- | ------------------: | ------------------: | ---------------------: | ---------------------: |
  | Camel       | 118.6 [117.8–119.9] |    91.1 [90.6–91.9] |    937.9 [930.8–944.8] |                  23.2% |
  | Quadratic   | 207.1 [206.5–210.6] | 168.2 [167.7–170.4] | 1551.7 [1542.9–1600.2] |                  18.8% |
  | Sphere, 1D  |    90.0 [88.7–92.1] |    65.3 [65.1–65.9] |    722.2 [718.3–737.3] |                  27.4% |
  | Sphere, 3D  | 167.0 [166.1–168.5] | 123.7 [123.5–124.9] | 1068.4 [1064.1–1078.4] |                  25.9% |
  | Sphere, 10D | 630.6 [628.3–631.6] | 623.4 [620.4–625.3] | 2437.3 [2429.3–2474.2] |                   1.1% |

The [kernel summary](cobyla-prima-kernels.csv) includes all eight input sets and
separate workspace construction timings. The small-dimension LP gains are
19–27%. These use identical mathematical inputs and preserve before/after Basin
numerical results, independently of whole-solver trajectory differences.

## Implementation and profiling

The retained change specializes the existing LP implementation for dimensions
1–3. A const dimension is carried through both LP stages, making their short
vector lengths available to the optimizer. Other dimensions use the original
dynamic dimension. Both paths share the same numerical source, workspace,
active-set rules, scaling, non-finite handling, and recovery checks.

Before the change, `TrstlpWork::solve` owns 35.6% of camel's sampled cycles
including its callees. The retained implementation reduces that inclusive share
to 31.7%. A separate three-run hardware-counter check records 6.6% fewer
instructions for 200,000 fresh camel solves. The uninstrumented timing
comparison establishes the elapsed-time effect; the changed profile share alone
does not establish a speedup. All profile percentages use the entire sampled
process as their denominator, and nested shares must not be added.

Sphere's baseline profile assigns about 21% self time to model construction, 19%
to inverse products, and 20% to the LP implementation. The PRIMA camel profile
assigns about 23% self time to allocator routines, with additional time in its
matrix products, LP, and driver. PRIMA's ordinary heap-array policy and Basin's
existing workspace reuse are material implementation differences. The
identical-input LP comparison includes those differences; it does not isolate
language or compiler cost alone.

Experiments that were not retained:

- Fusing the objective multiplier's negation and clamp reduced measured
  store-to-load conflicts by about 27%, but increased camel's cycles by about 5%
  and did not improve complete solves. It was discarded.
- Specializing the LP setup without carrying the dimension into both stages did
  not produce a useful whole-solver gain.
- Specializing only dimension 2 improved the two original 2D cases by 4–6% in a
  15-round comparison, but regressed the 1D and 3D executor cases by about 8%
  and 5%. This version was replaced by the 1–3D specialization.
- A three-round unchanged-source rebuild control varied by about 1% on the small
  executor cases. Small changes on the general path should therefore be treated
  cautiously, even with stable callback counts.

The next supported experiment is a matched-input comparison of model
construction and inverse validation at dimensions 10–40. Their measured shares
justify investigating implementation cost there; these results do not justify
weakening inverse checks or changing the algorithm.

## Reproduce and validate

From the repository root, with Python 3.11+, Git, CMake, GNU Fortran, and the
project Rust toolchain available:

```sh
git submodule update --init tools/prima
git worktree add --detach /tmp/cobyla-prima-before bddb3f3

python3 crates/competitor-bench/investigations/cobyla-lm/reproduce_cobyla_prima.py build \
  --basin /tmp/cobyla-prima-before \
  --output target/perf-investigation/cobyla-prima-repeat/before
python3 crates/competitor-bench/investigations/cobyla-lm/reproduce_cobyla_prima.py build \
  --output target/perf-investigation/cobyla-prima-repeat/after
python3 crates/competitor-bench/investigations/cobyla-lm/reproduce_cobyla_prima.py measure \
  --before target/perf-investigation/cobyla-prima-repeat/before \
  --after target/perf-investigation/cobyla-prima-repeat/after \
  --cpu 2 --rounds 15 \
  --output target/perf-investigation/cobyla-prima-repeat/samples.csv
```

Build outputs must be new directories. Historical Basin worktrees reuse the main
checkout's pinned PRIMA submodule. Each build saves `probe` (timing), `capture`
(instrumented), source snapshots, `Cargo.lock`, `metadata.json`,
`verification.json`, and LP traces. The comparison checks unchanged before/after
Basin results, counts, and stopping reasons before timing. It keeps failed
quality checks in verification output. Use
`--cases camel sphere quadratic --rounds 3` for a pilot; the default covers all
eight workloads.

For profiling, use `build --profile --output <new-directory>`, then:

```sh
perf record -e cycles:u -F 997 --call-graph fp -o /tmp/cobyla.data -- \
  taskset -c 2 <profile-directory>/probe raw camel 500000
perf report --stdio --children --no-inline -i /tmp/cobyla.data
```

The profile build adds debug information and frame pointers to both languages.
GNU Fortran's nested callback trampolines require an executable stack for this
native probe; the build sets that linker option only on the isolated executable.
It does not change Basin's build settings. Raw samples, profiles, pilot
experiments, snapshots, and verification logs from this session are retained
under ignored `target/perf-investigation/cobyla-prima/`.

Verification passed: the full latest pure-Rust backend suite (including all four
COBYLA parameter backends and `f32`), migration traces, PRIMA fixtures, public
callback/error/budget tests, the allocation guard, all eight Criterion cases as
smoke tests, all-feature workspace Clippy, public rustdoc, rustfmt, both WASM
builds, and Python lint/format checks. The isolated probe also passes Clippy
with all targets and features.

New tests check scaled dependent constraints and recovery after non-finite
models with reused LP scratch. Differential checks compare all three small
dimension specializations with the dynamic path on 1,152 deterministic models
across `f32` and `f64`, requiring bit-identical steps. The existing allocation
guard still measures 213, 674, and 255 allocation requests for camel, sphere,
and quadratic, respectively; this change adds no workspace allocation. Public
APIs, stopping policies, backend support, and published website results are
unchanged.
