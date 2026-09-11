# COBYLA performance investigation: closed

The GlobalSearch migration investigation located the dominant overhead in
Basin's numerical driver and delivered verified improvements with regression
protection. The original investigation is complete. Runtime parity with `cobyla`
1.0.2 was not achieved, and further LP micro-optimization had diminishing
returns. This record preserves the conclusions; superseded probes, reports, and
timing series remain in Git history.

## Findings

Workspace reuse first made the driver 1.9–2.8 times faster with 94–97% fewer
allocation requests. Kernel and parameter-buffer changes subsequently reduced
public execution time by another 17–23%. Later passes delivered small gains: the
final LP change reduced public 10D sphere time by 0.9% and quadratic time by
1.6%, while some separate raw-driver measurements regressed. Public and raw
callers have different optimization contexts, so their timing difference does
not isolate executor overhead.

The last retained comparison used Ryzen 9 7900, NixOS, Rust 1.89.0, release
optimization, thin LTO, one codegen unit, and CPU 2. Fifteen paired rounds used
randomized adjacent contestants and an unchanged-source rebuild control. Each
process performed 16 warmup solves. Timed batches contained 5,000 camel, 400
sphere, or 3,000 quadratic solves, including initialization, extraction, and
teardown. The governor was `powersave` with `balance_performance`; frequency was
not fixed. These are historical measurements, not current benchmark results.

  | Case                      | Basin public median | `cobyla` median | Paired Basin/reference ratio [95% bootstrap interval] |
  | ------------------------- | ------------------: | --------------: | ----------------------------------------------------: |
  | Six-hump camel, 2D        |            15.96 µs |         6.49 µs |                                   2.459 [2.438–2.488] |
  | Sphere, 10D               |           467.96 µs |       404.22 µs |                                   1.155 [1.152–1.159] |
  | Constrained quadratic, 2D |            24.86 µs |        15.61 µs |                                   1.594 [1.585–1.601] |

Basin follows modern PRIMA; the reference crate translates NLopt 2.7.1. Both use
`rho_beg = 0.5` and a final radius of `2^-27`. Objective budgets are 50, 200,
and 100. Optional function and relative-parameter stopping are disabled. Basin
receives bounds as nonlinear rows and uses feasibility tolerance `2^-26`; the
crate receives native bounds and uses zero constraint tolerance. Native bounds
still create internal inequality rows. The historical control supplying
nonlinear bound rows to the crate preserved its trajectories.

Actual objective calls were 50/50, 200/200, and 61/81 for Basin/reference. Both
returned points passed common objective and feasibility targets. Camel and
sphere exhausted their budgets; quadratic reached the final radius. Basin's
slightly lower quadratic objective (`0.1249999974`) reflects an allowed
constraint violation of about `5.27e-9`. The 10D sphere objectives were
`5.73e-9` and `9.65e-7`. In the scaling experiments, the reference failed the
common accuracy target at 20D and 40D, while Basin passed. Equal budgets do not
establish equal numerical work or equally successful solves.

Profiles and identical-input LP replays located remaining costs in model
construction, inverse validation, LP execution, copying, and public integration.
They establish neither an unavoidable PRIMA overhead nor attainable parity.
Removing numerical safeguards is not justified by these measurements.

## Maintained benchmarks and regression protection

The existing competitor benchmark and verifier now include the three migration
cases through [shared public adapters](../src/cobyla.rs).

The dependency requirement `cobyla = "1.0.2"` allows compatible updates;
`Cargo.lock` records the version used for a run. The table above retains the
historical 1.0.2 results.

```sh
cargo run -p competitor-bench --release --bin verify_gd_nm
cargo bench -p competitor-bench --bench gd_nm -- 'cobyla_'
cargo bench -p competitor-bench --bench cobyla
cargo test -p competitor-bench --lib cobyla::tests
cargo test -p competitor-bench --test cobyla_allocations
```

The public comparison checks returned-point accuracy, feasibility, and callback
budgets outside timing. Its verifier reports actual objective calls, constraint
callback units, iterations where available, and stopping reasons. All setup,
initialization, result extraction, and teardown are timed. Basin's callback
guard enforces the objective budget; an active iteration can still make an extra
constraint-vector call. Criterion output stays under ignored `target/`.

The separate driver benchmark covers the original cases and sphere dimensions 1,
3, 5, 20, and 40. Allocation guards, public callback/error/budget tests,
non-finite and degenerate-input tests, and PRIMA reference fixtures remain in
the normal test suites. The pinned PRIMA submodule remains the numerical
reference for those fixtures.

## Separate future opportunity

Exact models for explicitly supplied linear constraints and bounds could avoid
repeated interpolation. Basin's `FoldedConstraints` path currently interpolates
every row, while PRIMA has a separate exact-linear-model path. All three
migration workloads have only linear constraints. This is an unmeasured
numerical feature opportunity, especially at larger dimensions; it would need
validation against that PRIMA path rather than identical historical traces. Any
renewed performance study should measure complete GlobalSearch workloads across
multiple starts and realistic callback costs, with common accuracy and
feasibility targets.

## Provenance

The final optimization is commit `b742f0cba1be3b431fb99257f37d1362fb040b62`,
compared with baseline `d071c11e70c45956537b135d62c8a6769a41f8f9`. All removed
reports, summary CSVs, run manifests, source instrumentation, and historical
reproduction scripts are available at commit
`ad4ef87ee879df65b7a9ce9b9c3baf60e6470089` under
`crates/competitor-bench/investigations/cobyla-lm/`. For example:

```sh
git show ad4ef87ee879df65b7a9ce9b9c3baf60e6470089:crates/competitor-bench/investigations/cobyla-lm/cobyla-lp-followup.md
```

The original GlobalSearch comparison used migration commit
`4bf3eaa6b3677e0a2cc18f61f81603612db66b13` (Basin 1.8.0) and parent
`1f44818e396567d22cb7137c267c66da2e19f334` (`cobyla` 1.0.2). The matching
numerical reference is PRIMA v0.7.2 at
`e1169927f10fea330c1aae60f12e1a32c45ef5f4`.
