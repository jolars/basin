# Steihaug with expensive derivatives

Measured September 14, 2026. Extra derivative evaluations overturn Basin's
timing advantage on Rosenbrock at these added callback costs:

  | Problem         | Added work in Hessian only | Equal added work in each gradient and Hessian callback |
  | --------------- | -------------------------: | -----------------------------------------------------: |
  | Rosenbrock, 2D  |    About 0.45–0.50 µs/call |                                About 0.23–0.25 µs/call |
  | Rosenbrock, 20D |      About 1.4–1.6 µs/call |                                  About 0.7–0.8 µs/call |

These are approximate crossover ranges from two full sweeps and a focused sweep,
including variation between runs. They measure **added CPU work**, on top of the
inexpensive analytic derivatives, through the original GlobalSearch adapters.
They are not universal thresholds for total derivative latency. The CPU-work
proxy preserves the derivatives exactly; it does not reproduce the memory
traffic, parallelism, or shared computation of a particular autodiff or
simulation workload.

The callback counts and target outcomes reproduce the comparison behind
`TODO.md` on its original analytic workloads. The current measurements use the
local Basin checkout. Production solver code is unchanged.

[Generated crossover
plot](../../../../target/steihaug-derivatives/run-2/crossover.svg) (local
artifact).

The second full sweep gives these representative timings. Times are medians of
11 rounds, with each round averaging fresh solves over the shared successful
starts. The intervals are bootstrap 95% intervals for the median paired
argmin/Basin timing ratio across rounds. They describe within-run variation, not
uncertainty across hardware or problem populations.

  | Problem and added Hessian work | Basin, µs/solve | argmin, µs/solve | argmin/Basin ratio [95% interval] |
  | ------------------------------ | --------------: | ---------------: | --------------------------------: |
  | Rosenbrock 2D, none            |            4.52 |             9.42 |                 2.09 [2.08, 2.11] |
  | Rosenbrock 2D, 30.7 µs/call    |           703.9 |            384.3 |              0.546 [0.540, 0.547] |
  | Rosenbrock 20D, none           |           158.1 |            212.3 |              1.344 [1.335, 1.346] |
  | Rosenbrock 20D, 30.6 µs/call   |          3224.7 |           2074.8 |              0.643 [0.641, 0.643] |

At about 30 µs per Hessian call, argmin is therefore **1.83× faster in 2D and
1.55× faster in 20D**. Sphere has identical derivative counts on both sides:
adding work brings its timing ratio toward one, without the Rosenbrock reversal.

The focused sweep measured a near tie in 20D at 1.45 µs per Hessian call (ratio
1.000, interval \[0.997, 1.002\]), or 0.72 µs in each derivative callback
(1.000, \[1.000, 1.003\]). In 2D, 0.44 µs per Hessian call still slightly
favored Basin, while 0.50 µs favored argmin. Differences of a few percent should
be treated cautiously on this workstation.

The counts explain the crossover. For each library, gradient and Hessian counts
are equal on these successful runs:

  | Problem        | Shared successful starts | Basin mean G/H calls (median) | argmin mean G/H calls (median) |
  | -------------- | -----------------------: | ----------------------------: | -----------------------------: |
  | Rosenbrock 2D  |                    24/24 |                  22.75 (22.5) |                    12.21 (9.5) |
  | Rosenbrock 20D |                    15/24 |                  100.27 (102) |                     60.73 (62) |
  | Sphere 20D     |                    24/24 |                      3.83 (4) |                       3.83 (4) |

For added gradient and Hessian costs `c_g` and `c_h`, the simple cost model is
`T_B - T_A ≈ (T_B0 - T_A0) + ΔN_g c_g + ΔN_h c_h`. Use **mean** counts here:
each timed sample averages starts, whereas the original TODO quoted medians.
Each derivative count difference is 10.54 in 2D and 39.53 in 20D. The measured
cheap-callback advantage is approximately 4.7–5.3 µs and 54–63 µs per solve,
respectively. Their quotients predict the measured crossover. As callback work
dominates, the predicted Basin/argmin time ratios approach 1.86 and 1.65.

The source establishes why this comparison involves different numerical work.
[Basin's Steihaug](../../../basin/src/solver/trust_region/steihaug.rs) defaults
to `min(0.5, sqrt(||g||)) * ||g||` and a dimension-sized CG cap. [argmin
0.11.0's
implementation](https://docs.rs/crate/argmin/0.11.0/source/src/solver/trustregion/steihaug.rs)
uses `epsilon * ||g||`, with `epsilon = 10e-10 = 1e-9` in the constructor, and
no practical default CG cap. It also has an initial-gradient stop at
`||g|| < epsilon`. Boundary and negative-curvature handling differ as well.
Consequently, these whole solves do not isolate implementation overhead.

[Basin's trust-region driver](../../../basin/src/solver/trust_region.rs) already
retains the gradient and Hessian across rejected trials within an outer
iteration. In the shared successful runs, the extra Hessian calls come from
additional accepted steps, not repeated refreshes after rejection. Callback
counts, unchanged trajectory fingerprints, and the measured response to added
callback work support this diagnosis; no sampling profile or solver rewrite was
needed.

Tighter forcing is the next useful experiment for expensive exact derivatives.
The public `Steihaug::with_forcing_parameters(0.1, 1.0)` configuration already
permits it, but this investigation does not establish its speed or reliability
on these starts. GlobalSearch's current builder does not expose that setting. A
future comparison should retain all starts and count both derivative calls and
CG products. A tighter default also spends more Hessian-vector products, which
can be costly in matrix-free mode; these results do not justify a default change
for that workload.

The numerical protocol retains the original 24 deterministic starts per problem,
`f(x) - f* <= 1e-6` objective target, initial radius 1, maximum radius 100,
acceptance threshold 0.125, 1,000 outer iterations, and 20,000 objective calls.
Rosenbrock starts lie in `[-2, 2]^n`, with the first start alternating
`(-1.2, 1, ...)`. The coordinate generator and analytic formulas are retained in
[probe.rs](probe.rs). Both implementations use `f64`; Basin uses `Vec` and
`DenseMatrix` through GlobalSearch's ndarray conversions, while argmin uses
ndarray 0.16.1 directly. Native configurable convergence tests are disabled.
Basin groups rejected trials within an outer iteration, while argmin counts each
trial as an outer iteration, so equal iteration caps are not equal work.

The objective callback interrupts at the first qualifying evaluation, before the
solver can accept the point. Success means **an evaluated point reached the
target**, not that a returned solution or stationarity was certified. Outside
timing, the probe independently checks the best evaluated point's cost and
gradient, and validates the objective of any returned point. The adapters do not
expose iteration counts or detailed termination reasons on this path; the CSV
distinguishes target interruption, callback budget, returned result, and other
errors without inventing those details.

Across all original problems, target hits remained 103/120 for Basin and 100/120
for argmin: Rosenbrock 20D contributed 18/24 versus 15/24, respectively; both
hit 24/24 in Rosenbrock 2D, sphere, and ellipsoid, and 13/24 in six-hump camel.
Timing ratios use only shared successful starts. All misses remain in the
diagnostics, including three 20D starts where only Basin succeeded. The timing
reversal therefore does not establish that argmin is preferable on every start.

The full sweep uses seven work levels from zero to roughly 31 µs per callback
and two modes: Hessian only, or equal work in gradient and Hessian callbacks. A
non-inlined integer recurrence consumed through `black_box` supplies work
without sleeps or per-callback clock reads. Independent batches measure its
cost. All 2,256 diagnostic runs per full sweep check callback counts, a
fingerprint of every callback point, and the best objective against the
zero-work run. Every check passed, as did central-difference checks of the
analytic gradients and Hessians. Optional diagnostic instrumentation is disabled
for timing; the objective budget counter and target flag remain. Runner
construction is outside timing; start cloning, solver initialization, adapter
conversion, and teardown inside `solve` are included.

Measurements use an AMD Ryzen 9 7900, logical CPU 6, powersave governor with
boost enabled, Rust 1.89.0, release optimization, thin LTO, and one codegen
unit. No parallel backend feature is enabled. Each case is warmed up and
calibrated to about 35 ms per sample, with backend order alternating across 11
rounds. The first sweep's samples ranged from 22 to 96 ms, with median 35 ms.
The original TODO's baseline used Rust 1.88.0 and the published Basin 1.11.0, so
its exact speed ratios should not be substituted for the baselines measured
here.

Revisions: Basin `a58dc97`, GlobalSearch
`3de436b1f607fcc5d921ad4ab048913c3f837234`, argmin 0.11.0, and argmin-math
0.5.1. The GlobalSearch adapter sources are unchanged from the original
comparison's `70ad942` revision. [Cargo.lock](Cargo.lock) preserves the
dependency resolution. The preparation script patches the isolated benchmark to
the local Basin checkout without changing either repository's manifest.

To reproduce from the Basin root, with that GlobalSearch checkout beside it:

```sh
python3 crates/competitor-bench/investigations/steihaug-derivatives/prepare.py \
  --globalsearch ../globalsearch-rs --output target/steihaug-derivatives
cargo build --release --locked --offline \
  --manifest-path target/steihaug-derivatives/Cargo.toml \
  --target-dir target/steihaug-derivatives/build
taskset -c 6 target/steihaug-derivatives/build/release/steihaug-derivatives \
  target/steihaug-derivatives/results
taskset -c 6 target/steihaug-derivatives/build/release/steihaug-derivatives \
  target/steihaug-derivatives/refined --refine
python3 crates/competitor-bench/investigations/steihaug-derivatives/analyze.py \
  target/steihaug-derivatives/results
```

Omit `--offline` if the locked dependencies are not cached. `--verify-only` runs
the numerical and trajectory checks without timing. `analyze.py --plot` also
writes PNG and SVG crossover plots when matplotlib is installed.

Local raw artifacts are in `target/steihaug-derivatives/`: `run-1/`, `run-2/`,
and `refined/` contain starts, diagnostic records, calibration and timing
samples, and generated summaries. `environment.json` records revisions,
compiler, CPU, flags, and hashes. They are ignored and will be removed by
`cargo clean`. The probe, lockfile, analysis, and findings above are durable.

Validation passed: the focused numerical/trajectory checks, isolated probe
clippy with warnings denied, workspace all-target/all-feature clippy, Rust
formatting, and Python lint/format checks. No production Rust, public API,
backend support, or web documentation changed.
