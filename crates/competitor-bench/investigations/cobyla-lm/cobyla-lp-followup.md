# Sharing COBYLA's signed and absolute dot reductions

This follow-up to [trial-buffer reuse](cobyla-trial-buffers.md) retains one
private LP optimization: computing signed and absolute dot products in a shared
traversal. Public quadratic solves take **1.6% less time**, and the original 10D
sphere takes **0.9% less time**, with identical before/after numerical traces.
Both gains persist against an unchanged-source rebuild control. Camel's public
timing remains unresolved, and no public regression is resolved across the eight
workloads.

The private raw-driver harness has a different outcome: camel and quadratic take
**3.5% and 2.3% more time**, and the 3D sphere takes 1.3% more time against
baseline. These regressions also persist against the rebuild control. The patch
is retained for its public-path benefit, with these raw-driver tradeoffs
recorded. The callers have different optimization contexts; subtracting raw and
public timings does not isolate executor overhead.

## Public runtime

Public solver plus executor, medians in **microseconds per solve**:

  | Case        | Before   | After    | After / before [95% interval] | After / control [95% interval] |
  | ----------- | -------: | -------: | ----------------------------: | -----------------------------: |
  | Camel       |    15.89 |    15.96 |        0.9982 [0.9965–1.0073] |         1.0050 [0.9992–1.0111] |
  | Sphere, 10D |   472.39 |   467.96 |        0.9906 [0.9878–0.9925] |         0.9923 [0.9880–0.9926] |
  | Quadratic   |    25.25 |    24.86 |        0.9839 [0.9729–0.9876] |         0.9841 [0.9796–0.9876] |
  | Sphere, 1D  |     4.87 |     4.76 |        0.9802 [0.9770–0.9818] |         0.9770 [0.9727–0.9838] |
  | Sphere, 3D  |    27.62 |    26.74 |        0.9665 [0.9646–0.9682] |         0.9701 [0.9648–0.9727] |
  | Sphere, 5D  |    86.05 |    85.19 |        0.9894 [0.9868–0.9937] |         0.9878 [0.9789–0.9905] |
  | Sphere, 20D |  4042.29 |  3975.70 |        0.9837 [0.9830–0.9852] |         0.9835 [0.9730–0.9879] |
  | Sphere, 40D | 49797.89 | 49301.25 |        0.9893 [0.9843–0.9950] |         0.9872 [0.9725–0.9926] |

Ratios are median paired ratios, which need not equal ratios of the timing
medians. Intervals are 95% bootstrap intervals. The [timing
summary](cobyla-lp-followup-timing.csv) includes the raw path, rebuild control,
and crate contestants. The [paired ratios](cobyla-lp-followup-timing-ratios.csv)
include every before/control/candidate and Basin/crate comparison.

Against the matched-floor crate, the paired public ratio moves from **1.169 to
1.155** on the 10D sphere and **1.616 to 1.594** on quadratic. Camel remains
about **2.46**. Candidate ratios against the nonlinear-row control are 2.259,
1.126, and 1.543 for camel, sphere, and quadratic. The remaining gap is still
substantial, and these are comparisons between distinct COBYLA variants.

## Identical-input LP runtime

With the final schedule keeping baseline, control, and candidate adjacent, LP
time falls by **6.2%, 9.7%, and 15.8%** at dimensions 10, 20, and 40. The 3D and
5D sets also improve against both baselines. Quadratic and 1D replay differences
are unresolved. Camel's LP takes about **1.0% more time** against baseline and
0.9% more against the rebuild control.

Medians in **nanoseconds per LP**, with paired ratio intervals:

  | Input set   | Before  | After   | After / before [95% interval] | After / control [95% interval] |
  | ----------- | ------: | ------: | ----------------------------: | -----------------------------: |
  | Camel       |   90.94 |   91.66 |        1.0104 [1.0041–1.0163] |         1.0093 [1.0071–1.0143] |
  | Sphere, 10D |  628.72 |  589.84 |        0.9377 [0.9342–0.9444] |         0.9378 [0.9342–0.9400] |
  | Quadratic   |  168.30 |  168.10 |        0.9975 [0.9947–1.0051] |         0.9964 [0.9909–1.0077] |
  | Sphere, 1D  |   65.66 |   65.63 |        0.9985 [0.9966–1.0017] |         1.0005 [0.9851–1.0034] |
  | Sphere, 3D  |  123.94 |  121.29 |        0.9805 [0.9758–0.9855] |         0.9786 [0.9766–0.9897] |
  | Sphere, 5D  |  252.76 |  238.39 |        0.9428 [0.9383–0.9457] |         0.9448 [0.9415–0.9495] |
  | Sphere, 20D | 2151.25 | 1945.67 |        0.9033 [0.8992–0.9055] |         0.9046 [0.8997–0.9075] |
  | Sphere, 40D | 8690.93 | 7320.70 |        0.8419 [0.8347–0.8495] |         0.8463 [0.8452–0.8493] |

The [kernel summary](cobyla-lp-followup-kernels.csv) and [paired kernel
ratios](cobyla-lp-followup-kernels-ratios.csv) retain both references and the
rebuild control. These are measurements on identical mathematical inputs using
the actual private kernel source. The public solver's optimization context
differs from this replay, so its speedup is established by the separate public
timing sample.

An earlier 15-round LP sample randomized all five contestants together. It
suggested similar scaling gains, but reference runs could separate the Basin
builds. The final sample above was repeated after correcting that schedule; the
earlier data remains under `target/perf-investigation/cobyla-lp/final-lp.*`. No
solver source changed between those measurements.

## What changed

The private LP's QR addition, least-squares multipliers, and inactive-constraint
residuals need both a signed dot product and a sum of absolute products. The
candidate computes those two reductions in one traversal. Each accumulator keeps
its original initialization and arithmetic order. In particular, it obtains the
floating-point summation identity through `Scalar::sum` to preserve signed
zeros. The two-element case retains the existing unrolled reductions.

The change adds no workspace storage and preserves column scaling, rank tests,
rotations, multiplier decisions, inverse checks, callback accounting, stopping
rules, and public interfaces. The allocation probe records 178/504/206 requests
for complete public camel/sphere/quadratic solves, matching the trial-buffer
report. Their cumulative requested bytes are 5,560/89,464/7,728. These counts
include the probe's constant observer-list allocation; they are not peak memory.

An independent experiment replaced indexed QR column rotations with disjoint
slices and used slice iteration for multiplier-vector updates. It passed the
numerical checks and improved the original small-case LP medians by about 2% in
three exploratory rounds. Its public improvements were inconsistent, including
variable 1D results. That experiment was discarded. The exploratory source
snapshots, patches, and samples remain in the ignored investigation directory.

## Profile evidence and remaining work

Separate optimized builds with debug information and frame pointers were sampled
with `perf record -e cycles:u -F 997 --call-graph fp`. Resolved stacks reach the
executor, driver, and actual private kernels. Before-change public LP inclusive
shares are 28.3% on camel, 19.4% on the 10D sphere, and 38.5% on quadratic.
Sphere's QR addition alone owns 6.8% of the public sample and 32.5% of its
isolated LP replay. Each percentage uses the entire sampled process as its
denominator; inclusive shares overlap with their callees.

The paired reductions occur in QR addition, multiplier solves, and inactive-row
residuals. Source inspection identifies the repeated traversals; the replay
provides the independent elapsed-time evidence. QR addition remains about 33% of
the candidate's 10D LP profile, despite its faster replay. A hotspot's share
alone does not measure a speedup. Small-case results also reflect code
generation: the two-dimensional helper retains its existing unrolled reductions.

Model construction and inverse validation remain supported next targets for
matched-input measurements at dimensions 10–40. In the baseline public sphere
profile, model construction owns 20.4% and inverse validation owns 25.1%
inclusive. Their extra PRIMA bookkeeping and differing whole-solver trajectories
still require separation from per-operation execution cost. This LP experiment
does not support weakening numerical safeguards or promising complete parity
with the older crate.

## Numerical comparison

The practical target remains NLopt-derived `cobyla` 1.0.2 with the matched final
radius, plus its nonlinear-row control. PRIMA v0.7.2 at
`e1169927f10fea330c1aae60f12e1a32c45ef5f4` supplies the matching numerical
reference. The [variant-gap
report](cobyla-variant-gap.md#comparison-and-controls) documents their different
trajectories, feasibility policies, and bound interfaces. Both before/after
Basin implementation cost and Basin/crate quality under a common budget are
reported separately.

Before and candidate have identical results, current and selected incumbents,
evaluation traces, diagnostic work counts, and stopping reasons across all 48
diagnostic records. All eight LP traces are byte-identical and contain 1,208
inputs. Every reference replay passes the existing step, trust-region length,
objective-reduction, and linearized-violation checks. The new comparison also
requires bit-identical before/after Basin steps on those inputs before timing.

  | Case        | Objective / constraint-vector calls | Public iterations | Returned objective   | Maximum violation   |
  | ----------- | ----------------------------------: | ----------------: | -------------------: | ------------------: |
  | Camel       |                             50 / 50 |                41 |  -1.0316284527212884 |                   0 |
  | Sphere, 10D |                           200 / 201 |               148 | 5.725381469125405e-9 |                   0 |
  | Quadratic   |                             61 / 61 |                49 |  0.12499999736582204 | 5.26835597369768e-9 |

These are actual objective callbacks, distinct from wrapper counters. Camel and
sphere stop at their budgets; quadratic reaches the final radius. Independent
quality and feasibility verification runs outside timing. All eight Basin
results pass the existing targets. The crate fails the scaling-sphere target at
20D and 40D under the shared budget, so those timings do not compare equally
successful solves.

## Measurement and reproduction

Measurements use the Ryzen 9 7900 workstation, NixOS, Rust 1.89.0, and GNU
Fortran 15.3.0. Baseline and candidate were rebuilt and timed together; the
Intel trial-buffer measurements provide historical context. The baseline is
`d071c11e70c45956537b135d62c8a6769a41f8f9`. An independent rebuild of unchanged
Basin and generated probe sources supplies the control.

Rust timing builds use release optimization, thin LTO, one codegen unit, default
CPU code generation, `Vec<f64>`, and no Basin default features. PRIMA uses
`-O3 -ffp-contract=off`. Processes run on logical CPU 2, whose SMT sibling is
CPU 14. The governor is `powersave` with `balance_performance`; frequency is not
fixed. Build metadata requires matching compilers, lockfiles, reference and
probe sources, settings, and thread environments. Profiles and allocation counts
come from separate builds.

Workloads retain the original starts, bounds, and budgets, `rho_beg = 0.5`,
`rho_end = 2^-27`, `ctol = 2^-26`, and `cweight = 1e8`. Whole-solver batches use
5,000 camel, 400 original sphere, 3,000 quadratic, and
`max(10, floor(40000 / n²))` scaling-sphere solves. Each process performs 16
warmup solves. Timing includes fresh initialization, solving, result extraction,
and teardown, excluding startup, JSON, diagnostics, and independent
verification.

LP batches replay baseline traces through every contestant. Each batch repeats
the entire trace `max(10, 5 * whole_solve_batch_count)` times, following 16
warmup trace passes. Loading, workspace construction, and verification are
excluded. Input preparation and the crate's sign conversion remain timed.
Fortran retains its internal allocation behavior.

Both final comparisons completed all 15 paired rounds in a single uninterrupted
attempt. Cases and adjacent contestants are randomized with seed 42. Reported
ratios are medians of paired ratios with 95% bootstrap intervals from 10,000
resamples, also seeded with 42. Summary CSVs record timing medians and IQRs
separately. These intervals describe this sample and do not eliminate thermal,
frequency, or build-layout variation.

A process monitor requires five quiet five-second polls before an attempt, then
checks every two seconds. On compiler activity, it stops only the comparison's
process group. Monotonic process timestamps let collection retain complete
rounds that ended before the last confirmed quiet poll. No timing-value outlier
rule is used. The [run manifest](cobyla-lp-followup-run.json) records selection
and source hashes; raw data, source snapshots, logs, and profiles live under
ignored `target/perf-investigation/cobyla-lp/`.

From a clean baseline checkout and this candidate, with the pinned PRIMA
submodule initialized:

```sh
probe_script=crates/competitor-bench/investigations/cobyla-lm/reproduce_cobyla_gap.py
artifact_root=target/perf-investigation/cobyla-lp-repeat
python3 -B "$probe_script" build --basin /path/to/baseline --output "$artifact_root/before"
python3 -B "$probe_script" build --basin "$artifact_root/before" --output "$artifact_root/control"
python3 -B "$probe_script" build --output "$artifact_root/after"
python3 -B "$probe_script" compare-lp \
  --before "$artifact_root/before" --after "$artifact_root/after" \
  --stages "$artifact_root/control" --output "$artifact_root/lp.csv" \
  --cpu 2 --rounds 15
python3 -B "$probe_script" compare \
  --before "$artifact_root/before" --after "$artifact_root/after" \
  --stages "$artifact_root/control" --output "$artifact_root/public.csv" \
  --cpu 2 --rounds 15
```

Run timing after builds finish on an otherwise idle machine. `compare-lp` writes
raw samples, exact Basin/reference step records, summaries, paired ratios, and
trace hashes. Both comparisons reject incompatible builds before measurement.
Published web benchmark data was not regenerated.

Verification passed for the candidate: focused COBYLA and allocation tests; the
routine nalgebra/ndarray/faer/problems/parallel suite including `f32`; all eight
benchmark smoke cases; workspace formatting and all-target/all-feature Clippy;
public rustdoc; both WASM builds; standalone timing/allocation-probe Clippy; and
Python formatting, lint, and five harness tests. New reduction tests cover
dimensions through 40, cancellation, signed zeros, extreme magnitudes,
non-finite values, and `f32`/`f64` behavior. Harness tests check that reference
tolerance does not hide changed Basin steps, all contestants use baseline
inputs, and failed verification prevents timed batches.

## Assessment

This is a marginal public speedup: roughly 0.4 microseconds per quadratic solve
and 4.4 microseconds per original 10D sphere solve. The small production change
passes the agreed public-path acceptance criteria, but the raw-driver
regressions limit its value. The evidence supports stopping this line of LP
micro-optimization. Further work aimed at substantially closing the crate gap
would need a different cost center and a stronger expected end-to-end return.

The remaining gap is partly a comparison of PRIMA and NLopt's modified older
COBYLA, both descendants of Powell's method. It is also an implementation
comparison: identical-input LPs still cost more in Basin, and its public
integration adds work. The existing measurements do not isolate a minimum cost
for PRIMA's additional safeguards, so they establish neither that full parity is
attainable nor that the remaining gap is unavoidable. Basin already uses fewer
LP solves on camel and quadratic, while its different sphere trajectory uses
more LP solves and reaches better final accuracy.

Blanket runtime parity on these cheap objectives is therefore a weak target for
further investment. A broader comparison of time to a common accuracy and
feasibility target, including meaningful callback costs, would better establish
whether the remaining overhead matters in use. The observed first-target
evaluation counts support that investigation; they are not themselves timings or
proof of which point either solver would return if stopped early.
