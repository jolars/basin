# SLSQP workspace-reuse follow-up

Measured September 16, 2026, against
`a8c166ca7b731ad147cbee1cc1f6e1db6fadf564`, the allocation pass documented in
[README.md](README.md). The reported Rosenbrock solve improves from 13.802 to
10.509 microseconds (24%). The reference medians are 10.231–10.312 microseconds:
the remaining difference is approximately 2–3%, comparable to the observed run
variation. This does not establish a consistent speed advantage over either
reference. Constrained HS71 improves from 9.459 to 7.982 microseconds (16%).

## Comparable numerical work

The published case retains analytic two-dimensional Rosenbrock, `Vec<f64>`,
start `[-1.2, 1.0]`, no bounds or constraints, and the conventional objective
without half-sum scaling. Basin uses Kraft accuracy `1e-10` and a 200-iteration
cap; `slsqp` 1.0.2 and `nlopt` 0.8.1 (bundled NLopt 2.9.1) use absolute
objective-change tolerance `1e-10` and a 200-evaluation cap. None hits its cap.

Before and after, Basin takes 35 accepted steps, 49 objective evaluations, and
36 gradient evaluations, terminating with `SolverConverged`. Both references
use 59 objective and 46 gradient evaluations and return `FtolReached`. All
return `[0.9999999967339014, 0.9999999946665349]` and objective
`1.54363268165856448e-16`. Independent verification adds one objective and one
gradient calculation outside the timer; the gradient infinity norm is
`4.860250410903085e-7`. Current and best accepted points coincide for this case.
These counts cover solver callbacks. Trace setup also calculates the starting
objective once for every implementation; complete-solve timings include it.

The shared method remains Kraft's Han–Powell SLSQP, damped BFGS, original NNLS,
and inexact L1 merit line search. Basin follows
[Williams SLSQP 1.6.1](https://github.com/jacobwilliams/slsqp/tree/97884f98042624007f736dc536fa5636906ff26a).
The [NLopt-derived references](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#slsqp)
retain their different stopping rules and first-trial fused callbacks. Equal
tolerances do not equate their work. The implementation-speed claim comes from
matched before/after Basin runs, supported by identical-input kernel checks.

[workloads.rs](workloads.rs) adds analytic HS71 in four dimensions, starting
at `[1, 5, 5, 1]`, with bounds `[1, 5]`, equality `sum(x²) - 40 = 0`, and
inequality `25 - product(x) <= 0`. Its objective is
`x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]`. Accuracy and iteration cap are
`1e-10` and 200. Both versions converge after five steps, six objective and six
gradient evaluations, twelve constraint-block evaluations, and six Jacobian
evaluations. Both return objective `17.014017289134873` and
`[1.000000000000003, 4.742999642848287, 3.821149976895384, 1.379408294178592]`.
Verification independently checks the objective, bounds, equality, and
inequality outside the timer, and checks the reported stationarity and
complementarity. These verification calculations are excluded from the counts.

## Cause and retained changes

An optimized frame-pointer profile of the starting implementation collected
4,209 user-cycle samples, with resolved callchains and no lost samples.
Inclusive costs, as percentages of all sampled process cycles, were 41.85%
in `Work::prepare`, 34.04% in its `Work::qp` child, 19.54% in `lsi`, and
12.98% in `Factor::update`. `memmove` accounted for 19.44% self time. Nested
inclusive percentages must not be summed. Source inspection connected the QP
and BFGS costs to vectors and matrices allocated afresh on every step.

The changes retain arithmetic ordering, QR pivoting and rank thresholds,
non-finite checks, slack recovery, line search, convergence settings, and
evaluation accounting:

- Reuse BFGS difference, matrix-product, and rank-one-update buffers. Read
  parameters directly from the backend, update the reduced gradient in place,
  and reuse the completed search direction for the accepted displacement.
- Reuse QP input matrices, right-hand sides, QR permutation storage, and the
  Householder reflection tail. Restore variable order through permutation
  cycles instead of allocating another solution vector.
- Reuse the trial vector across backtracks and the accepted-point snapshot
  through `Clone::clone_from`.

Scratch belongs to the solver and is excluded from serialization. Every input
is rebuilt before reuse, including after failed subproblems and changes between
ordinary and slack-augmented QPs. Fresh initialization resets the workspace.
Checkpoint reconstruction starts with empty scratch; exact continuation tests
at several boundaries compare the final point, objective, gradient, counts,
iterations, and multipliers with an uninterrupted run. Public APIs and
serialized checkpoint fields are unchanged.

Separate allocator instrumentation includes initialization, callbacks, result
storage, and trace logging where present. Counts are requests and cumulative
requested bytes, not peak memory:

| Case | Requests before → after | Requested bytes before → after |
| --- | ---: | ---: |
| Rosenbrock | 658 → 130 | 15,351 → 6,207 |
| HS71 | 361 → 219 | 22,032 → 12,584 |

The Rosenbrock allocation guard failed before the changes and now passes
alongside the original result and evaluation-count checks. HS71 has its own
guard with feasibility and all four evaluation categories checked.

## Timings and limits

Linux x86-64, AMD Ryzen 9 7900, CPU affinity 4, Rust 1.89.0, release profile
with thin LTO and one codegen unit. The lockfile is unchanged. Both builds use
`competitor-bench` default features: Basin with `problems`, nalgebra 0.34.2,
and faer 0.24.4; all measured math uses serial `Vec<f64>`. No parallel feature,
profiling flags, or allocator instrumentation is linked into timing binaries.
`RUSTFLAGS`, `RAYON_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `OMP_NUM_THREADS`
were unset for ordinary timings.

The probe warms each case and records 21 batches per process. Complete solves
use 1,000 fresh solves per batch and include initialization and destruction;
Rosenbrock includes the published trace logging, while HS71 has no trace logger.
Kernels use `1000 / n` solves per batch, including input matrix clones and a
fresh workspace. Prebuilt before/after binaries alternate in five rounds
(AB, BA, AB, BA, AB), with no concurrent builds or other investigation timings.
The table pools 105 batch averages per case. IQR describes sample spread, not
a confidence interval.

| Case | Before, µs (IQR) | After, µs (IQR) | Speedup |
| --- | ---: | ---: | ---: |
| basin | 13.802 (13.742–13.870) | 10.509 (10.463–11.419) | 1.31× |
| slsqp | 10.052 (10.040–10.082) | 10.312 (10.281–10.471) | 0.97× |
| nlopt | 10.302 (10.255–10.392) | 10.231 (10.198–10.422) | 1.01× |
| basin_hs71 | 9.459 (9.432–9.501) | 7.982 (7.954–8.020) | 1.19× |
| lsei_n2_constrainedfalse | 0.122 (0.121–0.127) | 0.130 (0.130–0.131) | 0.94× |
| lsei_n2_constrainedtrue | 0.236 (0.233–0.243) | 0.242 (0.238–0.243) | 0.98× |
| lsei_n8_constrainedfalse | 2.149 (2.122–2.168) | 2.137 (2.114–2.156) | 1.01× |
| lsei_n8_constrainedtrue | 1.066 (1.052–1.087) | 1.064 (1.059–1.085) | 1.00× |
| lsei_n32_constrainedfalse | 171.037 (170.192–172.007) | 171.184 (170.589–171.936) | 1.00× |
| lsei_n32_constrainedtrue | 17.696 (17.462–18.154) | 17.496 (17.378–17.716) | 1.01× |

The cold kernels retain the [first pass's inputs](README.md#timings): upper
triangular E, `f = E * ones(n)`, and optional `sum(x) = n` and `x[0] >= 0`.
All independently check `x ≈ ones(n)`. They do not amortize workspace setup.
The smallest unconstrained cold kernel regresses by eight nanoseconds; other
kernel changes overlap variation. The benefit is repeated workspace use within
a complete solve. These measurements do not establish performance for large
or heavily active constraint systems, other backends, or WASM.

The last snapshot-reuse step reduces allocations from 178 to 130 on Rosenbrock,
but its additional timing benefit alone was inconclusive. A subsequent
experiment returned early from empty constraint assembly, after the existing
count checks. Five alternating rounds showed no speedup (10.768 versus 10.904
microseconds, overlapping IQRs), so that edit was discarded.

The final profile has 3,359 samples and no lost samples. QP work accounts for
31.47% inclusive and BFGS for 10.68%; `memmove` remains a prominent self-time
leaf (21.12%). These shares do not measure elapsed-time
improvements. A supported next experiment is an initialized manual-loop versus
`Executor` comparison with identical publication and stopping semantics, to
separate state transfers from solver math before proposing framework changes.

## Reproduction and validation

From the repository root, build both contestants before timing. The current
probe and workload source also compile against the baseline's production math:

```sh
mkdir -p target/perf-investigation/slsqp-followup
cargo build --release -p competitor-bench --bin slsqp_probe
cp target/release/slsqp_probe target/perf-investigation/slsqp-followup/after

baseline_dir=$(mktemp -d /tmp/basin-slsqp-before.XXXXXX)
git worktree add --detach "$baseline_dir" a8c166ca7b731ad147cbee1cc1f6e1db6fadf564
cp crates/competitor-bench/src/bin/slsqp_probe.rs "$baseline_dir/crates/competitor-bench/src/bin/"
cp crates/competitor-bench/investigations/slsqp/workloads.rs "$baseline_dir/crates/competitor-bench/investigations/slsqp/"
CARGO_TARGET_DIR="$baseline_dir/target" cargo build --release \
  --manifest-path "$baseline_dir/Cargo.toml" -p competitor-bench --bin slsqp_probe
cp "$baseline_dir/target/release/slsqp_probe" target/perf-investigation/slsqp-followup/before

python3 crates/competitor-bench/investigations/slsqp/compare.py \
  target/perf-investigation/slsqp-followup/before \
  target/perf-investigation/slsqp-followup/after \
  target/perf-investigation/slsqp-followup
cargo test --release -p competitor-bench --test slsqp_allocations -- --nocapture
```

`compare.py` saves raw CSV batches, separate numerical-verification output, and
`table.md`. `--summarize-only` regenerates the table from existing batches.
The run's raw profiles, staged experiments, validation logs, dependency versions,
and executable/lockfile checksums are in ignored
`target/perf-investigation/slsqp-followup/`.

Profiling remains separate from timing:

```sh
RUSTFLAGS='-C force-frame-pointers=yes' cargo build --profile profiling \
  -p competitor-bench --bin slsqp_probe
perf record -e cycles:u -F 997 --call-graph fp \
  -o target/perf-investigation/slsqp-followup/after.data \
  -- taskset -c 4 target/profiling/slsqp_probe --profile
perf report --stdio --children --no-inline --call-graph none \
  -i target/perf-investigation/slsqp-followup/after.data
```

Validation passed: focused least-squares/BFGS tests, all 28 SLSQP integration
tests with the latest pure-Rust backends and `serde`, the full routine pure-Rust
suite including `f32` round trips, allocation guards, all-feature workspace
Clippy, rustfmt, and default/no-default-feature WASM builds. Coverage includes
HS71 reference trajectories, active constraints, singular and non-finite QPs,
slack recovery, workspace reuse after failure and dimension changes, and exact
checkpoint continuation. Published website benchmark data was not regenerated.
