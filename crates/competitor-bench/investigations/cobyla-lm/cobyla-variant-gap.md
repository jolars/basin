# Does following PRIMA explain the remaining COBYLA gap?

PRIMA-related work explains part of the remaining gap, but these measurements
**do not establish an unavoidable runtime penalty of following PRIMA**. Basin
performs more inverse validations and, on the spheres, more LP solves. It also
pays for its public solver integration and for slower execution of LPs that
produce essentially the same steps in both implementations. The balance depends
on the problem: on the 20D sphere, Basin is both faster and more accurate.

This investigation follows the [driver optimization](cobyla-optimization.md),
[kernel optimization](cobyla-kernels.md), and [Fortran PRIMA
comparison](cobyla-prima.md). It measures Basin at `525e1f4` against **`cobyla`
1.0.2**, retaining the original competitor and workloads. No production solver
code was changed.

## Comparison and controls

The crate documents its implementation as a translation of NLopt 2.7.1, which
has different bound handling, geometry perturbations, and radius adjustments
from PRIMA. NLopt itself documents modifications to Powell's original algorithm;
this is not a comparison with an untouched Powell implementation. See the
[crate's versioned API
documentation](https://docs.rs/cobyla/1.0.2/cobyla/fn.minimize.html) and
[NLopt's algorithm
description](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#cobyla-constrained-optimization-by-linear-approximations).

The matching reference remains [PRIMA v0.7.2, commit
`e1169927f10fea330c1aae60f12e1a32c45ef5f4`](https://github.com/libprima/prima/tree/e1169927f10fea330c1aae60f12e1a32c45ef5f4).
The probe reuses the existing explicit Fortran binding and the exact workloads
and settings in the [PRIMA report](cobyla-prima.md#reference-and-workload):
`f64`, `rho_beg = 0.5`, and Basin/PRIMA `rho_end = 2^-27`. Camel has two
parameters and a budget of 50 objective calls, sphere has ten and a budget of
200, and quadratic has two and a budget of 100. Starts, bounds, objective
formulas, and the quadratic constraint `x + y <= 1.5` are unchanged. Additional
spheres use the existing deterministic starts and budgets `20*n`.

Three crate configurations separate two comparison asymmetries:

- **Historical:** native bounds, with all optional stopping tolerances disabled,
  reproducing the old direct-crate comparison. Its internal `rho_end` is zero.
- **Matched floor:** native bounds and per-coordinate absolute tolerances
  `2^-27`, which set the same final radius as Basin. Function-value and relative
  tolerances remain disabled.
- **Nonlinear rows:** the matched floor, with infinite native bounds and all
  finite bounds supplied as nonlinear inequalities in Basin's row order.

Basin/PRIMA constraints use `c(x) <= 0`; the crate's callbacks use the opposite
sign. The nonlinear-row control has **identical objective evaluation traces and
returned points** to the native-bound control on all eight workloads. Its extra
scalar callback dispatch changes cost, but native bounds do not explain the
trajectory differences here. The crate exposes no iteration count.

The feasibility policies remain different. Basin and PRIMA use `ctol = 2^-26`
and a return filter with `cweight = 1e8`; the crate's constraint tolerance is
zero. Basin's objective callback guard enforces the budget but can finish an
iteration with an extra constraint-vector call. PRIMA and the crate stop inside
their drivers. Neither the shared radius floor nor the shared budget implies
identical work.

## Numerical work and solution quality

Every returned point is independently checked outside the timer. The quality
target is absolute objective error below `1e-5` for camel and the original
sphere, below `1e-6` for quadratic and the scaling spheres, and maximum
constraint violation at most `2^-26`. Budget stops are recorded separately from
passing these targets. The quadratic optimum is `0.125`; Basin's slightly lower
returned objective reflects its permitted constraint violation.

  | Case        | Basin / matched-floor crate objective calls | Basin / crate objective           | Basin / crate violation | Stop              |
  | ----------- | ------------------------------------------: | --------------------------------: | ----------------------: | ----------------- |
  | Camel       |                                     50 / 50 | `-1.0316284527` / `-1.0316284534` |                   0 / 0 | Both budget       |
  | Sphere, 10D |                                   200 / 200 |             `5.73e-9` / `9.65e-7` |                   0 / 0 | Both budget       |
  | Quadratic   |                                     61 / 81 |          `0.1249999974` / `0.125` |           `5.27e-9` / 0 | Both radius floor |

All three pass the common quality target. The historical crate configuration
instead uses all 100 calls on quadratic. Matching the radius floor therefore
makes that competitor faster, increasing the relative gap to Basin.

An instrumented, untimed trace also records the first **evaluated point** that
passes the common target. This measures evaluation efficiency; it is not a
measurement of time to termination, and does not imply the solvers would return
that point if interrupted there.

  | Case        | Basin first target evaluation | Crate first target evaluation |
  | ----------- | ----------------------------: | ----------------------------: |
  | Camel       |                            35 |                            28 |
  | Sphere, 10D |                           142 |                           160 |
  | Quadratic   |                            25 |                            31 |
  | Sphere, 3D  |                            39 |                            46 |
  | Sphere, 20D |                           308 |        Not reached within 400 |
  | Sphere, 40D |                           696 |        Not reached within 800 |

At budget exhaustion on the 20D sphere, the objectives are `3.39e-9` for Basin
and `3.42e-5` for the crate. At 40D they are `2.70e-7` and `1.43e-4`. The
crate's budget-limited results at these dimensions fail the quality target; they
must not be counted as equally successful solves.

[Complete numerical and work results](cobyla-gap-results.csv) retain all eight
workloads and six contestants, including the 1D and 5D cases. Constraint counts
are labeled by unit: Basin/PRIMA call a vector callback; the crate calls scalar
nonlinear callbacks and evaluates native bound rows internally. Diagnostic
traces recompute objective and feasibility at each evaluation without adding to
solver counters. The ordinary post-run verification adds one objective and one
constraint-vector evaluation, separately from solver work.

## Timings

September 9, 2026; AMD Ryzen 9 7900, NixOS, Rust 1.89.0, GNU Fortran 15.3.0.
Both Rust implementations are in one executable with release optimization, thin
LTO, one codegen unit, and default CPU code generation. Basin uses `Vec<f64>`
and no default features. PRIMA uses the existing Release `-O3 -ffp-contract=off`
build. The isolated lockfile, source snapshots, source hashes, compiler
information, and build flags are retained with the probe.

Each of 15 rounds randomizes case and contestant order with seed 42. All
processes are pinned to CPU 2; no compilation or profiling ran during timing.
There are 16 untimed warmup solves or LP batches per process. Solve batches use
5,000 camel, 400 sphere, 3,000 quadratic, 4,444 3D sphere, and 100 20D sphere
solves. Timing includes initialization, fresh workspaces, solving, result
extraction, and teardown. Process startup, JSON output, trace loading,
diagnostics, and independent verification are excluded. These measurements are
comparisons within this probe, not a new before/after optimization claim.

Medians in **microseconds per solve**, with interquartile ranges:

  | Case        | Basin raw driver       | Basin public solver + executor | Crate, matched floor   | Crate, nonlinear rows  |
  | ----------- | ---------------------: | -----------------------------: | ---------------------: | ---------------------: |
  | Camel       |    13.75 [13.67–14.46] |            17.32 [17.13–17.76] |       6.77 [6.73–6.82] |       7.15 [7.13–7.20] |
  | Sphere, 10D | 458.52 [456.23–460.18] |         471.30 [470.35–473.92] | 365.38 [364.62–366.66] | 374.83 [374.64–376.09] |
  | Quadratic   |    22.63 [22.30–22.74] |            26.94 [26.75–27.34] |    15.57 [15.52–15.80] |    16.07 [16.04–16.18] |

Against the matched-floor crate, paired-round public runtime ratios are **2.55
\[2.52–2.67\]**, **1.29 \[1.29–1.30\]**, and **1.71 \[1.71–1.77\]**. These
brackets are 95% bootstrap intervals, not IQRs. Against the nonlinear-row
control they are **2.42**, **1.26**, and **1.67**: the gap persists.

The historical crate medians are 6.75, 365.59, and 20.49 microseconds. Its
quadratic run continues after the matched floor is reached, which partly hides
the implementation gap in the older comparison. Fresh PRIMA medians are 156.42,
1,735.78, and 262.21 microseconds. Basin is 3.7–9.7 times faster than that
reference; Fortran PRIMA's elapsed time is not evidence of a necessary algorithm
cost floor.

The extra dimensions show why no single PRIMA overhead factor fits:

  | Case        | Basin public, microseconds   | Matched-floor crate, microseconds | Common quality target |
  | ----------- | ---------------------------: | --------------------------------: | --------------------- |
  | Sphere, 3D  |          29.15 [28.74–29.40] |               11.20 [11.17–11.29] | Both pass             |
  | Sphere, 20D | 4,065.65 [4,053.45–4,110.42] |      4,742.71 [4,734.76–4,764.18] | Only Basin passes     |

[Timing summaries](cobyla-gap-timing.csv) include means and standard deviations.
[Paired ratios](cobyla-gap-ratios.csv) use 10,000 bootstrap resamples with seed 42.
Some samples have workstation outliers; medians and IQRs are reported rather
than discarding them.

## Where the gap comes from

### Public integration is a material component

The difference between Basin's public and raw paths is **3.57, 12.77, and 4.32
microseconds**. That is approximately **34%, 12%, and 38%** of the public gap to
the matched-floor crate, using differences of timing medians.

This includes the public solver's callback parameter buffer, incumbent
selection/copying, state updates, and executor bookkeeping. It is not an
isolated measurement of the executor loop. Objective/constraint counts and
returned points agree across these two Basin layers. This component cannot be
attributed to choosing PRIMA's numerical algorithm.

### Additional validation work is real; duplicate LP/model work is not universal

Counts from separate instrumented runs of the actual private sources:

  | Case        | LP calls, Basin / crate | Model builds, Basin / crate | Inverse products, Basin / crate |
  | ----------- | ----------------------: | --------------------------: | ------------------------------: |
  | Camel       |                 41 / 45 |                     47 / 48 |                        154 / 48 |
  | Sphere, 10D |               148 / 108 |                   190 / 190 |                       295 / 190 |
  | Quadratic   |                 52 / 69 |                     63 / 89 |                        188 / 89 |

The crate column uses the matched radius floor. Basin has 3.21, 1.55, and 2.11
times as many actual inverse residual products. Counts exclude Basin's cache
hits and include initialization where applicable. Both implementations validate
the inverse: the distinction is frequency and recovery policy, not the presence
of a check in only one implementation. Each Basin run invokes full inversion
once during initialization; no subsequent inverse repair occurs in these cases.

The [PRIMA
driver](https://github.com/libprima/prima/blob/e1169927f10fea330c1aae60f12e1a32c45ef5f4/fortran/cobyla/cobylb.f90)
performs a penalty search on trial simplex data, repoles the live simplex, and
updates geometry. Its [update
routines](https://github.com/libprima/prima/blob/e1169927f10fea330c1aae60f12e1a32c45ef5f4/fortran/cobyla/update.f90)
check inverse residuals after updates and can rebuild a damaged inverse. Basin
preserves these decisions, with exact-input reuse from the earlier
optimizations.

On camel and quadratic, Basin already does **fewer** LP solves and model builds
than the matched-floor crate. PRIMA's penalty search is therefore not causing a
general doubling of those operations. It still incurs trial-buffer copying,
repoling, and checks. On the spheres, the different trajectory adds LP work: 148
versus 108 calls at 10D, and 282 versus 198 at 20D.

### Identical-input LPs leave an implementation opportunity

Both Rust LP kernels replay all **1,208** LP inputs captured from Basin's eight
workloads. The crate's private routine is compiled from its actual source in an
isolated copy. Its one-based pointer shifts receive padded scratch buffers. The
wrapper negates `A`, `b`, and the objective column to convert conventions; both
contestants reuse their normal scratch storage, with no allocation per replay.
Preparation, including this sign conversion, is inside the timer. Workspace
construction and trace loading are outside it for both.

All replays pass step-length, objective-reduction, and linearized-violation
checks. Maximum step disagreement divided by the radius is `5.56e-16`; maximum
scaled objective/violation disagreement is below `4.24e-16`. The same Basin
inputs also pass the existing comparison against Fortran PRIMA. See the [kernel
verification results](cobyla-gap-kernels.csv).

Medians in **nanoseconds per LP**, with IQRs:

  | Input set   | Basin                        | Crate                        |
  | ----------- | ---------------------------: | ---------------------------: |
  | Camel       |          91.75 [90.97–92.86] |          66.63 [66.54–66.80] |
  | Sphere, 10D |       618.80 [617.97–620.49] |       400.07 [399.08–402.24] |
  | Quadratic   |       168.87 [168.10–169.42] |       120.10 [119.80–120.44] |
  | Sphere, 3D  |       123.99 [123.70–124.56] |          88.63 [88.27–89.39] |
  | Sphere, 20D | 2,153.51 [2,148.73–2,170.41] | 1,453.28 [1,451.71–1,456.67] |

Basin's kernel takes 1.37–1.55 times as long here. Multiplying each per-call
difference by Basin's actual call count gives **1.03, 32.37, and 2.54
microseconds** on the original cases, about **15%, 35%, and 36%** of their raw
driver gaps. This is a replay-based estimate of matching the crate's per-call
cost while retaining Basin's call count, not a measured solver speedup. It does
not account for the differing distributions of LP inputs along the crate's
trajectory.

The [PRIMA LP
source](https://github.com/libprima/prima/blob/e1169927f10fea330c1aae60f12e1a32c45ef5f4/fortran/cobyla/trustregion.f90)
adds scaling for huge coefficients and expresses the active-set algorithm
through reusable QR and least-squares operations. The crate uses the older
integrated routine. These measurements do not separate safeguard arithmetic,
compiler code generation, and the organization of those operations. They show
that obtaining PRIMA's steps on these inputs does not itself require Basin's
current kernel cost. They do not establish that substituting the older routine
would preserve PRIMA behavior on degenerate, scaled, or non-finite inputs.

### Profiles locate the remaining driver costs

Separate optimized builds with debug information and frame pointers were sampled
with `perf record -e cycles:u -F 997 --call-graph fp`. Symbols and caller chains
resolve through the driver into the kernels. The profiles use 200,000 camel,
10,000 sphere, and 150,000 quadratic solves, separately for Basin's raw path and
the matched-floor crate.

  | Basin routine               | Camel | Sphere, 10D | Quadratic | Share type |
  | --------------------------- | ----: | ----------: | --------: | ---------- |
  | `TrstlpWork::solve`         | 29.1% |       20.1% |     42.8% | Inclusive  |
  | `UpdateWork::inverse_error` | 14.8% |       25.1% |     12.8% | Inclusive  |
  | `ModelWork::build`          |  7.1% |       21.9% |      5.1% | Self       |
  | `memmove`                   | 11.5% |        5.3% |      9.5% | Self       |

Every percentage uses the entire sampled process as its denominator. Inclusive
shares overlap with callees and must not be added. In particular, some copying
belongs to the LP or inverse-check paths. Profiles locate costs; they do not
measure an achievable speedup. The kernel replay supplies the independent
elapsed-time evidence for LP cost.

## Next experiments

1. **Measure public integration by component**, preserving callback counts,
   selected incumbents, and state semantics. Its measured 3.6–4.3 microseconds
   on the tiny cases warrant attention before another solver-kernel rewrite.
2. **Compare model construction and inverse products on identical captured
   matrices**, then measure trial/rollback copying and exact-input cache checks.
   Model construction has the same call count on the 10D sphere, while inverse
   products have different counts. This separates per-operation cost from
   PRIMA's additional validation work.
3. **Use the crate LP replay as a cost target for the PRIMA LP**, concentrating
   on QR updates, multiplier solves, and small-array setup. Require the existing
   scaled, dependent, and non-finite PRIMA regressions for any change. Matching
   ordinary-input steps is not grounds for removing safeguards.

The supported conclusion is a mixed explanation: PRIMA changes trajectories and
requires extra validation/bookkeeping in this port, but public integration and
per-operation execution also contribute measurably. These experiments do not
justify either assigning the entire gap to PRIMA or promising complete parity
without further implementation work.

## Reproduction and validation

From the repository root, with the project environment and initialized pinned
PRIMA submodule:

```sh
python3 -B crates/competitor-bench/investigations/cobyla-lm/reproduce_cobyla_gap.py build \
  --output target/perf-investigation/cobyla-gap-repeat/build
python3 -B crates/competitor-bench/investigations/cobyla-lm/reproduce_cobyla_gap.py measure \
  --build target/perf-investigation/cobyla-gap-repeat/build \
  --output target/perf-investigation/cobyla-gap-repeat/samples.csv \
  --cpu 2 --rounds 15
```

The build directory must be new. The script reuses the existing PRIMA builder,
then adds the pinned crate and investigation-only private-kernel access. It
builds separate timing and diagnostic binaries, checks that instrumentation does
not change any returned result/count/stop record, and emits JSON traces plus
compact result CSVs. Measurement emits raw rounds, statistical summaries, paired
ratios, and pre-timing verification records. It does not update published web
benchmarks or require GlobalSearch.

For a profiling build, add `--profile` to the build command and use its `probe`
with `perf`; the measurement command rejects profiling builds. The session's
Rust profiles were built in the baseline snapshot with:

```sh
RUSTFLAGS='-C force-frame-pointers=yes' cargo build --profile profiling \
  --manifest-path target/perf-investigation/cobyla-gap/baseline/Cargo.toml
perf record -e cycles:u -F 997 --call-graph fp -o /tmp/cobyla-gap.data -- \
  taskset -c 2 target/perf-investigation/cobyla-gap/baseline/target/profiling/cobyla-prima-probe \
  raw camel 200000
perf report --stdio --children --no-inline -i /tmp/cobyla-gap.data
```

Validation passed for 48 solver outcomes and 16 LP comparisons; a clean rebuild
reproduced all 64 numerical records exactly. All 1,208 crate LP replays pass; 48
diagnostic runs preserve their uninstrumented results and work counts. The
native-bound and nonlinear-row crate controls have identical objective traces on
all eight cases. A two-round run validates the complete measurement command.

Repository Rust formatting, standalone probe formatting, Python lint/format, and
isolated probe Clippy with all targets/features pass. Clippy uses
`-D warnings -A clippy::let_and_return` because the unmodified upstream crate
contains that lint. No production Rust or feature changes were made, so the full
backend/WASM suites were not rerun. Raw samples, profiles, exact source
snapshots, and validation outputs remain under ignored
`target/perf-investigation/cobyla-gap/`.
