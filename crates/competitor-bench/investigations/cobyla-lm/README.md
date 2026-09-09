# COBYLA and Levenberg–Marquardt investigation

Investigated on September 8, 2026, against Basin
`260c0ffe87bb95e1781a627f244bdf252e81242e` (1.9.0).

COBYLA's performance gap is reproducible and belongs primarily to the
numerical driver. LM's QR proposal addresses a real accuracy limitation,
but changing the factorization alone does not reproduce MINPACK's
convergence behavior. The probes below separate these questions. The
[COBYLA optimization](cobyla-optimization.md) is now implemented and verified;
the LM implementation item remains open.

## Reproduce

From the Basin repository root:

```sh
python3 crates/competitor-bench/investigations/cobyla-lm/reproduce.py \
  --globalsearch /path/to/globalsearch-rs \
  --output /tmp/basin-gaps \
  --rounds 9
```

The GlobalSearch checkout must contain the two commits recorded in TODO.md.
The script archives those commits into a new temporary workspace, pins the
historical Basin dependency to `=1.8.0`, and builds a third copy of the new
adapter against the local Basin checkout. It preserves the adapter sources,
apart from package names and dependency declarations. It also compiles
Basin's actual private COBYLA driver sources for the `raw` comparison.
Neither repository's dependency manifest or lockfile is changed. The
generated workspace retains its resolved `Cargo.lock` and metadata.

Python 3.12+, Git, Cargo, and the project development toolchain are required.
The script verifies every case, runs the LM probes, optionally measures
randomized timing rounds, and rebuilds COBYLA with allocation instrumentation
for a separate measurement. `--rounds 0` builds and verifies without the
repeated timing loop. Each COBYLA process performs 16 untimed warmup solves.

The recorded timing table below uses Hyperfine with three process warmups
and 15 runs, matching the original measurement protocol:

```sh
hyperfine --warmup 3 --runs 15 \
  '/tmp/basin-gaps/cobyla-timing old-adapter camel 5000' \
  '/tmp/basin-gaps/cobyla-timing new-adapter camel 5000'
```

Use `sphere 400` and `quadratic 3000` for the other cases. `cobyla-timing`
has no allocation instrumentation. The allocation build is
`target/release/cobyla_probe`. Its reported time is diagnostic only.

## COBYLA: findings

### Two corrections to the original setup

1. Basin uses `c(x) <= 0`. GlobalSearch uses `c(x) >= 0`; its adapter
   negates user constraints. The quadratic callback is evaluated as
   `-(1.5 - x - y)` on the Basin side, preserving the original arithmetic.
2. The historical Basin adapter imposes
   `rho_end >= sqrt(f64::EPSILON) * initial_step_size`, approximately
   `7.45e-9` here, even when parameter tolerances are zero. This accounts for
   the quadratic's 61 evaluations. A direct comparison using
   `f64::MIN_POSITIVE` instead continues substantially longer.

The old crate's native-bounds interface does not avoid the `2*n` constraint
rows: its translated NLopt implementation adds one inequality for each
finite lower and upper bound internally. See `cobyla` 1.0.2's
`nlopt_cobyla.rs`, around line 1065. Native bounds can still affect callback
handling and the numerical trajectory, but merely counting constraint rows
does not explain the gap.

### Reproduction results

Environment: AMD Ryzen 9 7900, NixOS, Rust 1.89.0, release optimization,
thin LTO, and one codegen unit. GlobalSearch default features are disabled.
The process was allowed to run on any CPU. This shared workstation differs
from the Intel machine in TODO.md. Values below are process wall time divided
by solves per process; see [timing-results.csv](timing-results.csv) for standard
deviations. The quadratic adapter run had timing outliers. Treat the ratios
as approximate, especially when comparing small differences between layers.

| Case | Direct `cobyla` 1.0.2 | Direct Basin 1.9 | Old adapter | Basin 1.8 adapter | Adapter ratio |
|---|---:|---:|---:|---:|---:|
| Camel | 6.82 us | 57.76 us | 7.12 us | 57.28 us | 8.05x |
| Sphere, 10D | 464.36 us | 1222.93 us | 451.67 us | 1199.73 us | 2.66x |
| Constrained quadratic | 20.88 us | 92.96 us | 24.06 us | 100.58 us | 4.18x |

The objective counts and final objectives reproduce the original table:
50/50 evaluations for camel, 200/200 for sphere, and 100/61 for the
quadratic. Basin's sphere objective is `5.72538e-9` versus `9.64569e-7`.
The quadratic's Basin return has constraint violation about `5.26836e-9`,
within its `sqrt(epsilon)` feasibility tolerance. Its slightly lower
objective therefore does not represent a strictly feasible improvement.
Using the historical adapter with local Basin 1.9 reproduces the same
points, objective counts, and allocation counts as Basin 1.8.

### Layer ownership

The probe modes are:

| Mode | Layer exercised |
|---|---|
| `old` | Direct `cobyla` 1.0.2 API with native bounds |
| `raw` | Private Basin driver, final incumbent extraction only |
| `manual` | Public Basin solver with a manually driven iteration loop |
| `basin` | Public Basin solver plus executor and `MaxCostEvals` |
| `projected` | The same executor with callback projection into bounds |
| `old-adapter` | GlobalSearch parent commit and `cobyla` 1.0.2 |
| `new-adapter` | GlobalSearch migration commit and Basin 1.8.0 |
| `current-adapter` | Migration adapter source and local Basin |

The raw probe still materializes a parameter vector at each callback so it
can share the public probe's callback implementation. All Basin modes retain
the objective-budget guard and bound inequalities.

Allocation requests per complete solve, measured separately from timing:

| Layer | Camel | Sphere | Quadratic |
|---|---:|---:|---:|
| Direct old crate | 16 | 16 | 19 |
| Basin private driver | 4,759 | 27,037 | 8,521 |
| Basin public solver, manual loop | 5,015 | 27,979 | 8,833 |
| Basin executor | 5,060 | 28,131 | 8,887 |
| Historical Basin adapter | 5,167 | 28,539 | 9,017 |

See [allocation-results.csv](allocation-results.csv) for requested bytes and
callback counts. Bytes are cumulative allocation requests, not peak memory.
The executor adds 45 allocations on camel; the full adapter adds 107 over
the executor. These layers cannot account for the thousands of allocations
already present in the driver. The executor moves its state into
`next_iter`; it does not clone the entire solver workspace.

The sphere makes 200 objective callbacks and 201 constraint-vector calls
through the direct Basin API. The historical adapter adds an initial
constraint-dimension probe, reaching 202 user constraint calls. Its hard
guard suppresses an objective callback after exhaustion, while the solver
finishes that iteration. An objective cap is not a cap on every callback.

User-space cycle sampling with `perf record -e cycles:u -F 997 -g
--call-graph dwarf` identifies the following hotspots. These are approximate
self-time shares, not inclusive call-tree percentages:

- Camel: allocator routines about 29%, `trstlp_sub` about 18%, and
  `inv_error` about 13%.
- Sphere: `inv_error` about 24%, `build_a` about 22%, and `trstlp_sub`
  about 16%.

The baseline code explains these measurements:

- `get_cpen` copies five simplex arrays and constructs models and a
  trust-region LP; `step` constructs models and solves another LP afterward.
- `updatepole` copies `sim` and `simi`, including when the pole stays put.
  `inv_error` forms a full matrix product, identity matrix, and difference
  vector. The product costs cubic work in the parameter dimension.
- `trstlp` and its active-set helpers allocate many short temporary vectors.
  For example, `lsqr` creates two absolute-value vectors just to take a dot
  product on each back-substitution step.
- Every solve preallocates a 2,000-point filter. Its four arrays occupy
  128,000, 512,000, and 144,000 bytes in these cases. `confilt` is written
  and compacted but is never consumed by return-point selection.

The [implemented optimization](cobyla-optimization.md) reuses scratch buffers,
removes unnecessary temporary vectors, and grows the filter lazily. It shares
the penalty search's model and LP work only when the model inputs match the
live simplex after repoling. Inverse validation and recovery still run at the
same points in the algorithm.

The follow-up records timing and allocation improvements, adds evaluation and
iteration traces, and covers filter saturation, non-finite inputs, callback
errors, and strict callback guards. It also retains a private-driver benchmark
and deterministic allocation ceilings.

## LM: accuracy and convergence are separate questions

### Fixed damped linear systems

The probe compares Cholesky of `J^T J + mu*D`, pivoted QR of the augmented
system, and an SVD reference. For

```text
J = [[1, 1], [1, 1 + 1e-8], [1, 1 - 1e-8]],
D = diag(J^T J), r = -J * [1, -1],
```

at `mu = 1e-16`, the step's absolute error against the SVD reference is
approximately `3.51e-2` for Cholesky and `7.86e-10` for QR. At `mu = 1e-20`,
Cholesky fails while QR succeeds. At `mu = 1e-3`, both routes agree closely.
This isolates the benefit of avoiding the rounded Gram matrix as damping
becomes small. Exact rank deficiency with zero damping needs a specified
solution policy; the diagnostic triangular solve is not such a policy.

### Whole solves and calibration probes

[lm-results.txt](lm-results.txt) records residual and Jacobian evaluations,
residual norms, parameter errors, and termination reasons for:

- Nearly collinear linear problems from two starts, an exactly rank-deficient
  problem, and a diagonal problem scaled across 16 orders of magnitude.
- Raw SVI with 41 log-moneyness observations over `[-0.5, 0.5]` and
  `[-0.01, 0.01]`, each from three starts.
- SSVI across four separated maturities and three closely spaced maturities,
  each from three starts.

These are deterministic synthetic observations with known parameters, not
market calibration data. Analytic Jacobians are checked by centered finite
differences before solving. Relative tolerances are `1e-12`; Basin's absolute
gradient tolerance is disabled. Each solver has at most `200*(n+1)` residual
evaluations. The prototype uses the same budget and Nielsen damping logic
as Basin, with either Cholesky or QR. It is diagnostic code, not a public
solver or a proposed rank-deficiency contract.

The narrow SVI case has Jacobian condition number about `1.48e8` at the
known solution. From start `[0.035, 0.12, -0.2, 0.01, 0.25]`:

| Solver | Residual/Jacobian calls | Residual norm | Maximum parameter error | Stop |
|---|---:|---:|---:|---|
| Basin | 1200 / 1199 | 1.974e-9 | 3.798e-2 | `MaxIter` |
| Nielsen prototype, Cholesky | 1200 / 1199 | 1.974e-9 | 3.798e-2 | `MaxIter` |
| Nielsen prototype, QR | 1200 / 1199 | 1.974e-9 | 3.798e-2 | `MaxIter` |
| `levenberg-marquardt` 0.15 | 10 / 7 | 1.388e-17 | 1.380e-11 | Relative step |

Both solvers exhaust their budgets from the other two narrow-SVI starts,
although MINPACK reaches smaller residuals. Broad SVI and both SSVI cases
reach small residuals with both solvers. Closely spaced SSVI maturities need
13–32 residual evaluations in Basin versus seven in MINPACK; replacing only
the factorization largely preserves Basin's counts.

The linear problem with column separation `1e-8` exposes an independent
stopping issue: starting from zero, both Nielsen prototypes stop on relative
cost reduction after two residual calls with parameter error about one.
MINPACK recovers the known parameters. The QR step is still heavily damped,
so more accurate factorization does not prevent the premature stop.

MINPACK's [lmder](https://netlib.org/minpack/lmder.f) factorizes the Jacobian
with column pivoting and calls a separate damping-parameter routine tied to
its trust-region radius. Its [lmpar](https://netlib.org/minpack/lmpar.f) and
[qrsolv](https://netlib.org/minpack/qrsolv.f) reuse that factorization while
applying diagonal regularization. Basin uses Nielsen's scalar damping update.
The probes therefore support investigating damping selection and stopping
tests independently of adding QR. Merely matching named tolerances does not
make the two complete algorithms equivalent.

Parameter recovery also needs an identifiability qualification. Exactly
rank-deficient systems have nonunique solutions. Unconstrained raw SVI has
the same residuals for positive and negative `sigma`; one broad-SVI start
reaches `sigma = -0.2` in both solvers. Its raw parameter error is 0.4 despite
an essentially exact fit.

### SSVI Jacobian mismatch in the downstream source

At stochastic-rs commit `1560c6cfc484327c3fe8f80092f0998f54252e06`,
[`SsviParams::phi`](https://github.com/rust-dd/stochastic-rs/blob/1560c6cfc484327c3fe8f80092f0998f54252e06/stochastic-rs-quant/src/vol_surface/ssvi/params.rs#L24)
uses `eta / (theta^gamma * (1+theta)^(1-gamma))`. The
[calibration Jacobian](https://github.com/rust-dd/stochastic-rs/blob/1560c6cfc484327c3fe8f80092f0998f54252e06/stochastic-rs-quant/src/vol_surface/ssvi/calibrate.rs#L146)
instead differentiates `eta * theta^(-gamma)`.

For the implemented residual, `dphi/dgamma` is
`phi * (ln(1+theta) - ln(theta))`. At `(rho, eta, gamma) = (-0.3, 0.5, 0.5)`,
`k = 0.5`, and `theta = 0.25`, the correct Jacobian row is approximately
`[0.113817, -0.016489, -0.013269]`; the downstream calibration code produces
`[0.126624, -0.011851, -0.008214]`. The retained SSVI probe differentiates the
actual residual consistently. Fix or account for this mismatch before using
that downstream calibration as evidence about either LM implementation.

### QR capability and API recommendation

Add QR as an explicit capability-selected route initially. These results do
not justify replacing the existing Cholesky route or adding QR requirements
to its existing `Solver` implementation.

- Define a factorization capability that retains column permutations and
  supports repeated diagonal-regularized solves. Derive column scaling from
  the Jacobian, without forming `J^T J`. Distinguish rank of the original
  Jacobian from numerical rank of the augmented system. Positive damping
  with positive scaling makes the augmented system full column rank in exact
  arithmetic; it does not remove the need for numerical safeguards.
- Specify rank thresholds and the outcome of numerical rank loss. A basic
  pivoted-QR solution is not automatically a minimum-norm solution. Require
  an explicit policy for zero damping and rank-deficient inputs, and reject
  non-finite inputs predictably. Do not silently return to normal equations.
- Preserve existing explicit type annotations and trait bounds. An additive
  wrapper or strategy-specific return type can select QR without requiring
  old Cholesky-only downstream matrix implementations to support it.
- Dense nalgebra and faer provide column-pivoted QR machinery. Nalgebra's
  high-level `ColPivQR::solve` requires a square matrix, so the stacked
  least-squares implementation must apply `Q^T`, solve the triangular system,
  and undo the permutation explicitly. The diagnostic probe does this.
  `Vec`/`DenseMatrix` and ndarray need an honest shared pure-Rust implementation.
- Basin's existing `LinearSolveLstsq` is implemented only for faer sparse and
  does not guarantee numerical rank detection or numerical column pivoting.
  Its suggestion to detect rank deficiency from residual size is insufficient:
  a rank-deficient consistent system can have zero residual. Its method-level
  promise to return `Singular` also conflicts with the trait-level caveat.
  Do not reuse this contract as a rank-revealing LM capability without resolving
  those issues. Nalgebra sparse has no current QR implementation in Basin.

The LM rustdoc claim that QR is deferred until TRF is stale, as is its claim
that regularization is unconditionally sufficient for unconstrained LM.
Replace that explanation when implementing the route with the measured
tradeoff: Cholesky retains current backend coverage; QR avoids normal-equation
roundoff near rank deficiency, while damping and termination remain separate
solver decisions.

## Verification and remaining scope

The retained reproducer was run from a new directory, and all eight COBYLA
layers passed budget, finiteness, and final-feasibility checks on all three
cases. All LM analytic-Jacobian checks passed. The generated workspace passes
Clippy with all targets and features and warnings denied. The Rust probe files
and Python script pass their formatter and lint checks.

The investigation does not establish sparse QR coverage, a production rank
policy, or real-market calibration behavior. Those remain LM implementation
follow-ups. The [COBYLA follow-up](cobyla-optimization.md) supplies the numerical
regression suite and performance guards for the implemented optimization.
