# Reusing COBYLA trial buffers

This follow-up to the [variant-gap investigation](cobyla-variant-gap.md) reduces
COBYLA's allocation traffic while preserving its PRIMA numerical behavior. The
runtime target remains the NLopt-derived **`cobyla` 1.0.2** with the matched
final radius. PRIMA v0.7.2 at `e1169927f10fea330c1aae60f12e1a32c45ef5f4` remains
the numerical reference.

The retained change gives the private driver reusable storage for the trial
point, pending trust-region direction, and geometry direction. Geometry needs
its own direction because PRIMA's final short-step evaluation can still need the
pending trust-region step. The buffers are constructed after the existing model,
LP, and update workspaces. Their contents are overwritten in the same arithmetic
order as before. The public API, callback interface, initialization algorithm,
stopping rules, inverse checks, and LP kernel are unchanged.

The hypothesis was that removing frequent small allocations from `step` and
`geostep` would help the small problems identified in the previous report.
Separate allocation instrumentation confirms the removal; paired timings measure
its effect on complete solves.

## Runtime result

The public solver takes **2.7% less time on camel** and **1.9% less time on
quadratic**, using median paired ratios against the original baseline. Both
improvements persist against an unchanged-source rebuild: 2.7% and 1.4%. Public
1D, 3D, and 40D sphere timings also improve against both baselines. The 5D, 10D,
and 20D differences are unresolved against the rebuild control. No public-path
regression is resolved by these intervals.

Public solver plus executor, medians in **microseconds per solve**:

  | Case        | Before    | After     | After / before [95% interval] | After / rebuild control [95% interval] |
  | ----------- | --------: | --------: | ----------------------------: | -------------------------------------: |
  | Camel       |     20.20 |     19.55 |        0.9726 [0.9662–0.9770] |                 0.9733 [0.9674–0.9779] |
  | Sphere, 10D |    677.59 |    674.12 |        0.9934 [0.9908–0.9987] |                 0.9986 [0.9937–1.0057] |
  | Quadratic   |     31.03 |     30.40 |        0.9810 [0.9749–0.9869] |                 0.9861 [0.9806–0.9869] |
  | Sphere, 1D  |      6.36 |      6.12 |        0.9623 [0.9592–0.9694] |                 0.9661 [0.9595–0.9712] |
  | Sphere, 3D  |     33.49 |     32.77 |        0.9773 [0.9726–0.9827] |                 0.9808 [0.9747–0.9824] |
  | Sphere, 5D  |    109.18 |    108.43 |        0.9962 [0.9917–0.9989] |                 0.9999 [0.9967–1.0010] |
  | Sphere, 20D |  5,156.76 |  5,165.16 |        0.9973 [0.9932–1.0068] |                 0.9947 [0.9915–1.0027] |
  | Sphere, 40D | 58,382.55 | 57,569.65 |        0.9856 [0.9832–0.9875] |                 0.9867 [0.9826–0.9884] |

These are measurements on the new computer, not a comparison with the older
Ryzen timings. The unchanged-source rebuild is a second build of the baseline
with byte-identical generated Rust inputs. It estimates build-related variation
that a paired comparison against just one executable cannot expose. Changes in
source and workspace layout can affect generated code and heap addresses; this
experiment does not separate those effects from allocation cost.

The separate raw-driver harness has a tradeoff: 5D, 10D, and 40D sphere times
rise by **1.0%, 1.2%, and 0.9%** against the original baseline, with intervals
entirely above one; these regressions also persist against the rebuild control.
Raw camel, quadratic, and 1D sphere improve. The patch is retained for its
public-path benefit and allocation reduction, rather than as a universal runtime
improvement. The raw and public paths have different callers and optimization
contexts; subtracting their timings does not isolate executor overhead.

The remaining gap to the matched-floor crate is substantial. Paired public
runtime ratios move from **2.360 to 2.290** on camel, **1.231 to 1.223** on the
10D sphere, and **1.621 to 1.593** on quadratic. After-change ratios against the
nonlinear-row control are 2.048, 1.190, and 1.503. Trial storage closes a small
part of the gap. The existing identical-input LP replay remains the next
supported experiment for addressing execution cost in the larger cases.

The [timing summary](cobyla-trial-buffers-timing.csv) includes both Basin paths,
the rebuild control, and both crate configurations. The [paired
ratios](cobyla-trial-buffers-ratios.csv) distinguish candidate/baseline,
control/baseline, candidate/control, and Basin/crate comparisons. Ratios below
one favor the numerator. The brackets in this report are 95% bootstrap intervals
for median paired ratios; the timing CSV separately records IQRs. Median paired
ratios need not equal ratios of the displayed timing medians.

## Allocation result

Whole public solves, including setup, callbacks, result extraction, and
teardown:

  | Case        | Requests, before → after | Cumulative bytes, before → after |
  | ----------- | -----------------------: | -------------------------------: |
  | Camel       |                269 → 178 |                    7,016 → 5,560 |
  | Sphere, 10D |                881 → 504 |                 119,624 → 89,464 |
  | Quadratic   |                322 → 206 |                    9,584 → 7,728 |

Requests fall by **34%, 43%, and 36%**. Three setup allocations replace the
per-step allocations. In the raw driver, iteration-phase requests excluding
callbacks fall from **104 to 10**, **388 to 8**, and **135 to 16**. Those
remaining requests are the two row-range vectors used by `fcratio` at radius
reductions. The public path also still allocates its constraint conversion
vector, and the benchmark problem allocates each returned constraint vector.

[Allocation counts](cobyla-trial-buffers-allocations.csv) separate setup,
iteration, callback, and output phases for all eight cases and both Basin paths.
These are allocation/reallocation requests and cumulative requested bytes, not
peak live memory. The public diagnostic adds one constant allocation for its
observer list during setup. The timing binary has neither this observer nor the
counting allocator. Independent solution verification runs after allocation
counting stops.

## Numerical behavior and checks

Before and after have identical returned points, objective values, feasibility,
stop reasons, objective and constraint counts, iteration counts, evaluation
traces, and instrumented work counts. All 48 diagnostic records match, and the
eight LP trace files containing **1,208 inputs** are byte-identical. Replaying
those inputs against the pinned PRIMA and crate kernels passes the existing step
and feasibility checks. The existing PRIMA migration regression tests also pass.

  | Case        | Objective / constraint-vector calls | Public iterations | Returned objective     | Maximum violation     |
  | ----------- | ----------------------------------: | ----------------: | ---------------------: | --------------------: |
  | Camel       |                             50 / 50 |                41 |  `-1.0316284527212884` |                     0 |
  | Sphere, 10D |                           200 / 201 |               148 | `5.725381469125405e-9` |                     0 |
  | Quadratic   |                             61 / 61 |                49 |  `0.12499999736582204` | `5.26835597369768e-9` |

Camel and sphere stop at their objective budgets; quadratic reaches the final
radius. The raw quadratic harness counts its final convergence transition as an
iteration and reports 50, as it did before. Public iteration accounting is
unchanged. All eight Basin results pass the common quality target. The crate
still fails that target on the 20D and 40D spheres under the shared budget;
their timings are not equally successful solve comparisons. Full numerical
results are unchanged from [the original data](cobyla-gap-results.csv).

Regression coverage now bounds raw allocation counts more tightly, checks public
iteration allocations, and reuses one solver across changing parameter and
constraint dimensions while checking that the best-state snapshot remains
independent. The following checks passed for the retained source:

- COBYLA unit, public, allocation, and benchmark smoke tests.
- Routine Basin tests with nalgebra, ndarray, faer, problems, and parallel
  features, including the existing `f32` coverage.
- Workspace formatting and all-target, all-feature Clippy with warnings denied.
- Public rustdoc with the documented backend features.
- Default and no-default-feature `wasm32-unknown-unknown` builds.
- Standalone timing and allocation probe Clippy, plus Python formatting/lint.

## Measurement conditions

September 10, 2026; Intel Core Ultra 7 155U, NixOS, Rust 1.89.0, GNU Fortran
15.3.0. All timed processes use logical CPU 2, a performance core whose SMT
sibling is CPU 3. AC power was connected; the governor was `powersave`, with
`balance_performance` energy preference. Frequency was not fixed. The baseline
is `bdfa0541545f4a8db2209fae5553413442e2b939`.

Rust builds use release optimization, thin LTO, one codegen unit, default CPU
code generation, `Vec<f64>`, and no Basin default features. PRIMA retains
`-O3 -ffp-contract=off`. Compiler, lockfile, and reference-source hashes must
match before the comparison starts. Allocation and trace instrumentation use
separate executables.

Workloads, starts, budgets, and tolerances are those in the [previous
comparison](cobyla-variant-gap.md#comparison-and-controls): `rho_beg = 0.5`,
`rho_end = 2^-27`, `ctol = 2^-26`, and `cweight = 1e8` for Basin. The crate uses
the matched radius floor, with native bounds or the nonlinear-row control. Each
process performs 16 warmup solves. Timed batches contain 5,000 camel, 400
original 10D sphere, 3,000 quadratic, and `max(10, floor(40000 / n²))`
scaling-sphere solves. The timer includes fresh initialization, solving,
extraction, and teardown, but excludes process startup, JSON, diagnostics, and
independent verification.

The final sample contains **12 completed quiet rounds**, collected from three
attempts: nine from attempt 1, one from attempt 3, and two from attempt 7.
Unrelated Rust and document builds repeatedly interrupted the planned 15-round
comparison. A process monitor waited for five quiet five-second polls before
starting an attempt, then checked every two seconds. It terminated only the
comparison's own process group when it detected compiler or TeX work. Selection
retained completed rounds in chronological attempt order and discarded the last
completed round of each attempt as a guard around the monitor's detection
boundary. Incomplete rounds were excluded. No timing-value outlier rule was
used. Collection stopped at 12 rounds because the external work kept recurring,
rather than because of an interval or speed threshold.

Within each round, the harness randomizes cases and mode groups. Baseline,
control, and candidate run adjacently in randomized order within each Basin
mode. The two crate controls form another group. Each attempt starts the same
seed-42 schedule. Bootstrap intervals use 10,000 paired-round resamples with
seed 42; they describe this workstation sample and do not remove frequency,
thermal, or build variation.

Broader callback-buffer and initialization-buffer experiments were discarded
after exploratory measurements failed to establish a consistent added benefit.
An earlier placement of the three trial buffers also showed a small public 10D
regression and was superseded. The final comparison therefore includes the
unchanged-source rebuild control. The retained patch is limited to trial
storage; further kernel or numerical-policy changes need separate evidence.

## Reproduction and artifacts

From the repository root, with a clean checkout of the baseline available at
`/path/to/baseline` and the PRIMA submodule initialized:

```sh
artifact_root=target/perf-investigation/cobyla-buffers-repro
probe_script=crates/competitor-bench/investigations/cobyla-lm/reproduce_cobyla_gap.py
python3 -B "$probe_script" build --basin /path/to/baseline --output "$artifact_root/before"
python3 -B "$probe_script" build --output "$artifact_root/after"
python3 -B "$probe_script" build --basin "$artifact_root/before" --output "$artifact_root/control"
python3 -B "$probe_script" allocations --build "$artifact_root/before"
python3 -B "$probe_script" allocations --build "$artifact_root/after"
python3 -B "$probe_script" compare \
  --before "$artifact_root/before" --after "$artifact_root/after" \
  --stages "$artifact_root/control" --output "$artifact_root/paired.csv" \
  --cpu 2 --rounds 15
```

Run timing on an otherwise idle machine after all builds finish. `compare`
rejects changed numerical outcomes, diagnostic traces, or LP inputs before
timing and writes raw samples, summaries, ratios, verification, and metadata.
Its optional `--stages` accepts additional source snapshots or rebuild controls.

This session's raw samples, exact source snapshots, compilers, source hashes,
allocation probes, monitor logs, discarded experiments, and check logs remain
under ignored `target/perf-investigation/cobyla-buffers/`. The retained
candidate is `layout/`; `retained.*` contains the final timing artifacts.
`retained.selection.json` maps each retained round to its original attempt and
records source-file hashes. The [run manifest](cobyla-trial-buffers-run.json)
keeps compact provenance beside this report. Published web benchmark data was
not regenerated.
