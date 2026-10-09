# Least-squares measurement pilot

The pilot connects the [frozen development NIST references](nist-reference.md)
to native measurements for normal-equation LM and pivoted-QR LM, each with
Nielsen and trust-region damping, legacy `Trf`, and full
`TrustRegionReflective`. It checks measurement integrity and returned-point
quality. It selects no policy and changes no default or numerical safeguard.

## Observation and accounting contracts

The optional public solver observations live in
[`least_squares_diagnostics.rs`](../../crates/basin/src/solver/least_squares_diagnostics.rs).
Recording is disabled by default. Fresh initialization resets its sequence;
exact checkpoint continuation retains it. Consume batches after each step and
ignore previously seen sequence numbers when execution controls stop before a
new native decision. Model and trial observations retain actual native operands,
including failing comparisons and disabled tests. Unavailable finite evidence is
absent, without translating the native comparison into convergence.

[`least_squares.rs`](../../crates/competitor-bench/src/convergence/least_squares.rs)
pairs observations with the physical ledger and authoritative executor counters.
One residual vector is one leaf call. Fused residual/Jacobian initialization is
one physical call and two logical requests. A denied request increments its
logical category but consumes no leaf call. Initialization errors expose no
public logical counters or returned point. Objectives computed from residuals
consume neither a physical cost callback nor a logical `CostFunction` request.

Native acceptance and publication are separate. Full TRF can accept a residual
trial and then fail its Jacobian callback before replacing state. Errors retain
only the last published recommendation and physical history, with no returned
point or recoverable final checkpoint. Native steps are recorded directly;
subtracting rounded callback points can erase a small step. Model solves count
attempts leading to residual trials; full TRF counts subproblem calls, not
internal secular-search iterations. Model attempts that fail before any residual
trial are outside this counter, so the complete inner-work gate stays open.

The runner freezes six routes and both native precisions. Its 80 analytic checks
use `r=x-1`, `r=1+(x-1)^2`, a non-finite trial domain outside `[0,1.1]`, exact
physical caps 0, 1, and 2, and explicit LM relative-test probes. All analytic
starts are 0.1, with unconstrained minimum 1. The explicit relative probes use
absolute gradient tolerance zero and orthogonality, model reduction, trial step,
and trust radius tolerances `1e-8` in `f64` or `1e-4` in `f32`. Radius checks
apply only to trust-region damping. These probes exercise measurement; they are
not candidate policies.

After analytic verification, the NIST phase runs 15 development datasets, both
published starts, two precisions, and six routes: 360 solves. All corpus routes
retain default solver settings. TRF receives infinite bounds, preserving the
original unconstrained NIST problem. The physical cap is `2000*(n+1)`, the
iteration cap is 10,000, and the runner's wall guard is 600 seconds per solve.
An external process guard bounds a whole phase. Timing includes instrumentation
and an unoptimized build; it cannot establish a speed advantage.

## Schema and independent verification

[`verify_least_squares.rs`](../../crates/competitor-bench/src/bin/verify_least_squares.rs)
writes five CSV files with stable predicate names and numeric fields:

- `runs.csv`: route, precision, policy, caps, outcome, stage, all passing
  criteria, physical work, denied requests, logical counts, elapsed time, last
  publication, and whether a point was returned. Debug text supplements the
  structured outcome and criteria.
- `publications.csv`: initialized and published states with iteration, physical
  work, solver cost, and native coordinates.
- `leaves.csv`: each physical call, its outcome, and its sampled point.
- `native.csv`: model and trial observations, native step, gradient, scaling,
  curvature, acceptance and publication, damping, radii, and model-solve count.
- `checks.csv`: enabled, disabled, passing, failing, or unavailable comparisons.
  Upper bounds, factored binary values, and compound model-reduction operands
  have explicit columns. Empty optional values remain empty.

Vectors use semicolon-separated native values widened exactly to `f64` and
printed with round-trip formatting. The native precision column identifies
arithmetic. Sequence numbers identify decisions within each solve. Trial leaf
ordinals identify attempted work separately from publication work.

[`check-least-squares.py`](check-least-squares.py) validates callback
accounting, acceptance, actual and predicted decrease, full TRF radius updates,
rich native criterion operands, and all passing termination predicates. Full
TRF's model prediction uses independently differentiated source formulas. The
identity checks scale 4096 native unit roundoffs by absolute operation terms,
including cancellation within dot products. They explicitly bound underflow of
squared-step products before damping rescales them. That allowance checks
integrity, not accuracy or convergence certification.

Quality verification uses independent 100-digit outward-rounded intervals and
first-order AD of the unchanged source formulas. Its AD is checked against the
reference preflight's full-Hessian AD for all development models. The verifier
checks the frozen reference source hashes and uses its fixed scales and target
eligibility. Parameter distance considers all Lanczos amplitude/rate
permutations and MGH17's pair exchange. It rejects no start or outcome because
of poor quality.

The existing native witness probe runs afterward on unique verification points,
outside solve ledgers and timers. Source-to-native data conversion, native cost
and gradient arithmetic differences, and rounding screens bound uncertainty at
each tested point. Joint objective, stationarity, and parameter tests retain the
tenfold uncertainty margins from CDP-1. Points with parameter error above 0.1
cannot pass any frozen joint target and need no additional native probe; last
published points are always verified. First published attainment and first
sampled attainment remain distinct, and errors never acquire a returned-point
score. Reference-pending targets are retained and withheld from success scores.

## Reproduction and remaining scope

Commit source and a planned manifest before retaining evidence. Run the analytic
phase and its verifier before the NIST phase. Both binaries and the checker
refuse overwriting output files. Use a new output directory for each run.

```sh
CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 \
  CARGO_TARGET_DIR=target/nist-reference-build \
  cargo build -p competitor-bench \
  --bin verify_least_squares --bin verify_nist_witness

target/nist-reference-build/debug/verify_least_squares \
  --phase analytic --output target/convergence-defaults/RUN/analytic
python dev/convergence-defaults/check-least-squares.py \
  target/convergence-defaults/RUN/analytic --output ANALYTIC-REPORT.json
python dev/convergence-defaults/test-least-squares.py \
  target/convergence-defaults/RUN/analytic

target/nist-reference-build/debug/verify_least_squares \
  --phase nist --output target/convergence-defaults/RUN/nist
python dev/convergence-defaults/check-least-squares.py \
  target/convergence-defaults/RUN/nist --output NIST-REPORT.json \
  --probe target/nist-reference-build/debug/verify_nist_witness
```

The retained run report and manifest supply exact commands and hashes. Holdout
candidate outcomes remain sealed. This run covers analytic unconstrained
quadratic loss with nalgebra 0.34; tests also cover nalgebra 0.35. Finite
bounds, fixed coordinates, robust losses, finite differences, transformed cases,
additional backends, and full inner-work measurements remain separate gates.
Local reference certificates prove neither global optimality nor start-basin
membership. Poor returned quality alone cannot establish premature termination
at the intended basin. Step 5 remains open for the other solver families and
these additional gates.

## Bounded analytic extension

The [bounded full-TRF run](runs/2026-10-09-bounded-trf-001.md) adds a `bounded`
runner phase with active-bound, mixed-fixed, all-fixed, and stationary
nonminimum fixtures in both precisions. The same five CSVs and independent
checker retain native outcomes separately from analytic returned quality. Its
manifest gives commands and source/output hashes. This extension covers full TRF
only; legacy bounded TRF and robust losses remain open.

## Robust analytic extension

The [robust-loss run](runs/2026-10-09-robust-ls-001.md) adds a `robust` phase:
Huber at two scales, soft-L1, and Cauchy, with independently known minima and
robust gradients. Seven fixtures cover outliers, the Huber transition, negative
curvature, non-finite trials, and an active bound. The six unconstrained routes
and two bounded TRF routes retain default policies separately from explicit LM
relative probes. The checker verifies robust model operands and callback
prefixes against default controls, and labels robust orthogonality separately.
Errors retain independent last-publication checks without returned quality. The
manifest records 352 solves, exact commands, and source/output hashes.
