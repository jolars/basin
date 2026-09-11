# Projected Nelder–Mead: lme-rs investigation

11 September 2026 · Basin `ae021d5` plus this patch · lme-rs `251e5b8`

## Result

Remove two box-bound vector clones from every projected Nelder–Mead iteration,
and the two copies during initialization. Projection now briefly borrows the
problem before the counted cost callback. No public bounds, parameters, stopping
criteria, tolerances, or simplex arithmetic change.

The initialized 20-step regression made **40 vector clones before**, and **zero
after**. Its custom vector implements only the existing required capabilities.
The bounds remain feasible throughout. Existing projected and unbounded tests
cover the numerical behavior.

Matched-work ndarray probes improve substantially for small simplexes. Complete
lme-rs fitting remains dominated by the objective: this patch does **not**
establish removal of the previously reported 27% production-fit difference.
Keep the lme-rs default unchanged pending a comparison at matched solution quality.

## The 80k-row gap is not equal work

The fixture exactly follows `benches/bench_load_production.rs` in lme-rs:
80,000 observations, 3,000 groups, seed 105, ML, `y ~ x + (x | group)`.
The response has no generated group effect, so the fitted covariance approaches
a boundary. The previously published Criterion timing did not check numerical
agreement on this case.

| Build | Objective (lower is better) | Optimizer iterations | Profiled deviance evaluations |
|:---|---:|---:|---:|
| Argmin from the full refresh | 226984.42305010345 | 44 | 89 |
| Basin 1.10.0 standard | 226983.80381439500 | 73 | 127 |
| Basin 1.10.0 adaptive experiment | 226983.80381513840 | 123 | 252 |

Each objective agrees with the final fitted model's independently evaluated
deviance. The Basin solution is better by about **0.61924**; this is not proof
of a global optimum. Fixed effects are close, but objective agreement fails the
existing fair-harness tolerance. Argmin's faster time is not a valid target at
equal quality. The adaptive experiment was discarded: it approximately doubled
evaluations without improving the objective.

About 96% of Basin's profiled prepared fit was in deviance evaluation. Tracing
its callback found 127 distinct parameter vectors and no repeated points to
cache. Trace logging and phase instrumentation were disabled for all timings.
Profile evaluation counts include documented post-fit work; they are not the
solver's iteration counter. The older profile JSON has a flattened
`fit_wall_seconds` field referring to the prepared profile; use the explicit
prepared timing and fair samples rather than interpreting it as cold-fit time.

## Matched implementation work

`probe.rs` runs projected ndarray 0.16.1 Nelder–Mead on a quadratic, with identical
initial simplexes, bounds [0, 1], standard coefficients, and 100 iterations.
It intentionally measures a fixed budget, not time to convergence. Every solve
checks its final cost and evaluation count against the reference. Initialization
and the identical result assertion are included in each timed batch. Small dimensions use batches of 300 solves; 128-D
uses batches of four. Each process has three warmup batches and eleven samples.

Prebuilt executables alternate ABBA / BAAB / ABBA, six processes per variant.
A is unchanged local Basin; B has only the bound-borrowing patch. Ratios use
the median of paired block ratios. Ranges below are **observed block ranges**,
not confidence intervals. Costs and evaluation counts are bit-identical.

| Dimension | After / before | Observed block range |
|---:|---:|:---|
| 2 | 0.498 | 0.476–0.505 |
| 3 | 0.573 | 0.542–0.580 |
| 16 | 0.850 | 0.786–0.855 |
| 128 | 0.993 | 0.920–1.016 |

The 128-D result is inconclusive. These microbenchmarks isolate an allocation
improvement; they do not imply a comparable percentage gain in mixed-model fits.
Raw results: [ndarray-comparison.json](ndarray-comparison.json).

## Complete consumer comparison

All twelve existing fair-harness workloads plus the boundary production case
were run in the same ABBA / BAAB / ABBA order. Every process ran three warmups
and eleven measurements per complete/prepared metric. **3,168 measured fit
checks** converged; objectives, fixed effects, final theta, and optimizer
iterations are bit-identical before and after. The full result also retains
each measured fit's checks and fixture hashes.

| Workload | Complete fit ratio [block range] | Prepared fit ratio [block range] |
|:---|---:|---:|
| `sleepstudy_reml` | 0.998 [0.988, 1.011] | 0.985 [0.958, 1.024] |
| `sleepstudy_weighted_reml` | 1.036 [1.018, 1.070] | 1.030 [1.009, 1.036] |
| `penicillin_crossed_reml` | 1.006 [0.925, 1.026] | 0.980 [0.977, 0.983] |
| `pastes_nested_reml` | 1.040 [0.953, 1.041] | 0.991 [0.979, 0.996] |
| `random_intercept_10k` | 1.022 [0.924, 1.134] | 1.063 [0.984, 1.166] |
| `random_intercept_50k` | 1.051 [0.818, 1.108] | 0.899 [0.895, 1.125] |
| `random_intercept_100k` | 0.919 [0.888, 0.999] | 1.009 [0.836, 1.064] |
| `large_random_slopes_100k` | 1.000 [0.931, 1.062] | 0.963 [0.944, 0.974] |
| `crossed_20k` | 0.975 [0.960, 1.031] | 0.979 [0.968, 0.983] |
| `nested_10k` | 0.965 [0.950, 1.015] | 0.992 [0.986, 0.999] |
| `cbpp_binomial_ml` | 0.976 [0.969, 0.996] | — |
| `grouseticks_poisson_ml` | 1.020 [1.013, 1.021] | — |
| `production_boundary_80k` | 0.992 [0.938, 1.033] | 0.979 [0.933, 1.062] |

Small positive and negative differences remain, including shared scalar paths
that never use the changed solver. This workstation evidence does not establish
a general end-to-end speedup. The production-case difference spans both sides
of 1.0. Raw data: [lme-comparison.json](lme-comparison.json).

## Environment and reproduction

Windows 11 build 26200, AMD Ryzen 5 8600G (12 logical CPUs), Rust/Cargo 1.98.1.
BLAS, Rayon, Polars, and Julia thread limits were one. All measured executables
were prebuilt; no compilation or other benchmark ran concurrently with timing.
The probe uses thin LTO and one codegen unit. Consumer builds retain lme-rs's
release profile and static Intel MKL. Before and after use the same local Basin
source revision and dependency graph, differing only in the solver patch.
The source checkout and lockfile in lme-rs are unchanged.

From this investigation directory, build the standalone probe using
`cargo build --release --locked`. Build another copy against unchanged Basin,
retaining both executables, then run:

```sh
python compare.py /path/to/before-probe /path/to/after-probe --output comparison.json
```

For the consumer case, copy `boundary_fixture.rs` temporarily into the lme-rs
examples directory as `bench_basin_boundary_fixture.rs`, then generate the CSV:

```sh
cargo run --release --locked --example bench_basin_boundary_fixture -- target/basin-boundary-80k.csv
```

Build lme-rs's `bench_fair_rust_julia` and `bench_perf_breakdown` examples with
`--features basin`, using a Cargo `[patch.crates-io]` override to each Basin
source. Preserve the consumer lockfile and retain separate binaries. Run the
timing binary with:

```sh
bench_fair_rust_julia time --case production_boundary_80k --data target/basin-boundary-80k.csv --formula "y ~ x + (x | group)" --model lmm --reml false --warmups 3 --repeats 11 --with-phases
```

Repeat independent alternating processes. Use the regular fair-harness case
table for the other twelve workloads. Numerical checks are extracted after
each fit timer stops. Do not compare different objectives as equivalent fits.

## Validation

- Clone regression: fails before with 40 copies; passes after with zero.
- Pure-Rust suite with `nalgebra_latest,ndarray_latest,faer_latest,problems,parallel`:
  1,355 passing test/doctest executions after excluding one confirmed baseline
  failure. This includes projected/unbounded Nelder–Mead and f32 round-trip tests.
- The full unfiltered run is **not green**: NEWUOA's
  `solver::newuoa::driver::tests::chained_rosenbrock_6d` fails its `f < 1e-6`
  assertion with `f = 0.000018666715187267427`. An isolated unchanged-source run
  produces exactly the same cost and parameters. No NEWUOA code or tolerance changed.
- Strict all-feature workspace Clippy also fails on unchanged source: five
  findings in `observer.rs`, `basin_hopping.rs`, `gbnm/geometry.rs`,
  `bobyqa/geometry.rs`, and `lincoa/getact.rs`. The four categories are
  `manual_is_multiple_of`, `needless_range_loop`, `needless_late_init`, and
  `filter_next`. All-feature/all-target workspace lint passes when only these
  known categories are allowed on the command line; no lint suppression was
  added to the repository. This limited run is not a strict lint pass.
- Default and `--no-default-features` workspace WASM builds pass.
- Required feature-rich rustdoc build and workspace formatting pass.
- The retained standalone probe builds from this directory with its lockfile.

The full-suite failure and existing strict-lint findings are separate follow-ups,
not evidence against or a reason to relax the Nelder–Mead regression. No release,
upstream push, dependency upgrade in lme-rs, or optimizer-default change is included.
