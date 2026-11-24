# Published benchmarks

## Competitors

How much progress does each implementation make, and how long does it take?
These comparisons follow individual solvers on shared problems. The curves show
the objective gap $f(\mathbf{x}) - f^\star$ against elapsed time. Compare the
time needed to reach a given accuracy as well as the accuracy ultimately
reached. The NIST dogleg plot uses residual evaluations on the horizontal axis.

Sizes differ by solver: a useful workload for L-BFGS can be excessive for a
simplex method. We choose larger problems instead of repeating tiny solves or
continuing iterations after reaching the target.

Configurations match where the APIs allow it. Line searches, stopping rules, and
numerical safeguards can still produce different paths. These are comparisons of
solver effectiveness, including implementation costs, rather than measurements
of identical numerical work.

Each case warms up once. Each timestamp is the median at the corresponding
sample across timing repetitions of the same deterministic run. The harness
checks that numerical traces agree across repetitions before combining
timestamps. Repetitions reduce timing noise by repeating the same starting point.

Timings include initialization and trace instrumentation. Some driving loops
construct their executor before starting the clock. LM, COBYLA, BFGS, nonlinear CG, CMA-ES, and
L-BFGS-B also include solver teardown; NLopt curves include an explicit endpoint
after optimization.
Independent solution checks and constrained-trace reconstruction happen outside
the timer. Argmin uses an observer, Basin and gomez use driving loops, and NLopt
uses objective callbacks, so sampling and instrumentation costs differ.

The objective curves retain at most about 512 samples plus the endpoint from
each full recorded trace. Feasibility plots retain all recorded trials.
Objective gaps are floored at $10^{-16}$ for display, including small negative
gaps allowed by constrained feasibility tolerances. Absolute times depend on the
machine. The [committed results](src/lib/data/competitor-benchmarks.json) record
the source revision, whether the working tree had local changes, the machine,
compiler, dependency versions, thread count, and repetition count.

From the repository root:

```sh
cargo build --locked -p competitor-bench --release --features lbfgsb-reference --bin trace
bash -c './target/release/trace > target/competitor-traces.json'
pnpm --dir web collect:competitors
```

The optional `lbfgsb-reference` feature requires a C compiler and libclang,
provided by the development environment. To verify just this case, run
`cargo run --locked --release -p competitor-bench --features lbfgsb-reference --bin verify_lbfgsb`.
Append `-- --calibrate` to remeasure untraced solves, starting at 20,000 variables
and doubling until both medians reach 20 ms. The published dimension remains
fixed between benchmark refreshes.

To verify both CMA-ES problems with the published seed and refresh period, run
`cargo run --locked --release -p competitor-bench --bin verify_cmaes`.
Append `-- --untraced` to disable intermediate trace recording. The verifier
reports timing ranges across repetitions, objective counts, and target checks.
To generate only the CMA-ES traces, pass `--cmaes` to the prebuilt `trace` binary.

Pass `--bfgs` or `--nonlinear-cg` to the prebuilt `trace` binary to generate
only that comparison.

To run the nonlinear conjugate gradient comparison with trace observers
disabled:

```sh
cargo bench --locked -p competitor-bench --bench nonlinear_cg
```

The Powell dogleg figure shows the best objective gap against residual
evaluations. Regenerate it separately from the repository root:

```sh
cargo run --release -p competitor-bench --bin verify_dogleg -- --trace > web/src/lib/data/dogleg-convergence.json
```

The [trace harness](../crates/competitor-bench/src/bin/trace.rs) and its shared
fixtures define the problems and settings. The
[collection script](scripts/collect-competitors.ts) validates the results and
adds machine and version metadata before updating the committed data.

## Backends

Each point is Criterion's mean time for a complete solve. The
[committed results](src/lib/data/backend-benchmarks.json) record the iteration
cap. The least-squares, CMA-ES, and SLSQP cases can converge before that cap. The figures use logarithmic axes for both solve time
and problem size.

The committed results also record the machine and 95% confidence intervals on
the mean. The figures show the means. Small visual differences should be read with
that measurement uncertainty in mind. Absolute times are machine-specific.

From the repository root, run the benchmark and collect the Criterion results:

```sh
cargo bench -p basin --features nalgebra,ndarray,faer,problems --bench solver_backends
pnpm --dir web collect:benchmarks
```

The [benchmark harness](../crates/basin/benches/solver_backends.rs) defines the
solver configurations and fixtures. The
[collection script](scripts/collect-benchmarks.ts) selects the published cases
from `target/criterion/` and writes the data used by the figures.
