# L-BFGS-B parity fixtures

`lbfgsb_rosenbrock_5d.tsv` is the iteration-wise trajectory of Nocedal's
L-BFGS-B v3.0 Fortran source on Rosenbrock 5D, used by
`tests/lbfgsb_iter_parity.rs` to verify basin's port reproduces it
within `~1e-10`.

## Format

One line per iterate. Whitespace-separated (Fortran TSV-ish):

```
iter f x(0) x(1) x(2) x(3) x(4) g(0) g(1) g(2) g(3) g(4)
```

- `iter == 0` is the post-init state (x has been projected onto the
  feasible box; cost and gradient have been evaluated). No L-BFGS-B
  step has been taken yet.
- `iter == k > 0` is the state at the end of iteration `k`, i.e.
  after `k` accepted line searches.

Numbers are printed with `es24.16` for full f64 round-trip.

## Problem setup (locked)

- `n = 5`, `m = 5`
- Bounds `[0, 5]^5` (Fortran `nbd(i) = 2`)
- Start `(-1, 2, -1, 2, -1)` (infeasible; `active` projects it to
  `(0, 2, 0, 2, 0)`)
- `factr = 0`, `pgtol = 0`, `max_iter = 30`
- Rosenbrock in basin's standard coefficient form
  (`Σ 100 (xᵢ₊₁ − xᵢ²)² + (1 − xᵢ)²`, not the rescaled `driver1.f`
  variant).

## Regenerating

The committed `.tsv` is the artifact the test reads; you only need
to follow the steps below if you've changed the fixture parameters
(start point, bounds, `max_iter`, etc.) and need a fresh trajectory.

The L-BFGS-B v3.0 source is **not vendored** in this repo —
`references/` is gitignored, by project convention (papers and
reference implementations live there locally). Fetch the BSD-3
v3.0 tarball from Nocedal's group:

```bash
mkdir -p ../../../../references/lbfgsb-v3.0
curl -L https://users.iems.northwestern.edu/~nocedal/Software/Lbfgsb.3.0.tar.gz \
  | tar -xz --strip-components=1 -C ../../../../references/lbfgsb-v3.0
```

Then build the driver and dump a fresh fixture:

```bash
gfortran -O0 -std=legacy -o lbfgsb_driver lbfgsb_driver.f \
  ../../../../references/lbfgsb-v3.0/lbfgsb.f \
  ../../../../references/lbfgsb-v3.0/linpack.f \
  ../../../../references/lbfgsb-v3.0/blas.f \
  ../../../../references/lbfgsb-v3.0/timer.f
./lbfgsb_driver > lbfgsb_rosenbrock_5d.tsv
rm lbfgsb_driver
```

This is a manual on-demand step. CI does not rebuild the fixture.

# NEWUOA parity fixtures

`newuoa_<problem>_<n>d.tsv` are reference runs of **PRIMA's** NEWUOA (the BSD-3
C/Fortran translation of Powell's solvers, vendored at `tools/prima`, submodule
v0.7.2), used by `crates/basin/src/solver/newuoa/parity.rs` to cross-validate
basin's NEWUOA port (`docs/newuoa-roadmap.md`). The parity test is **in-crate**
(not under `tests/`) because `minimize` is `pub(crate)`; it reads these files via
`include_str!`.

Committed fixtures (`problem`, `n`, start `x0`, `rho_beg`):

- `newuoa_rosenbrock_2d.tsv` — Rosenbrock 2D (basin form), `x0=(-1.2, 1)`,
  `rho_beg=0.5`.
- `newuoa_chrosen_5d.tsv` — chained Rosenbrock 5D, `x0=(-1,…)`, `rho_beg=0.5`.
- `newuoa_arwhead_5d.tsv` — ARWHEAD 5D, `x0=(1,…)`, `rho_beg=1.0`.
- `newuoa_vardim_5d.tsv` — VARDIM 5D (the §8 Qint motivator), `x0_i=1-i/n`,
  `rho_beg=1/(2n)`.

All use `rho_end=1e-6`, `npt=2n+1`, `maxfun=500n`. The locked inputs live in the
fixture itself (the `# config` / `# x0` lines), so the test recomputes nothing.

There is intentionally **no `rosenbrock_5d`** fixture: 5D chained Rosenbrock is
already `chrosen_5d` (the only difference is the summation term order, which the
test's tier-1 objective check confirms is equivalent), and from the hard
`(-1.2, 1, …)` start basin and PRIMA converge to *different* stationary points
(basin to the global minimum `f≈1e-11`, PRIMA to a local one at `f≈3.93`) — a
property of ill-conditioned multi-basin problems, not a parity signal. The
fixture set therefore uses problems where both solvers reach the same minimizer.

## Format

One line per objective evaluation, plus three metadata lines. Whitespace-
separated, `%.17e` for full f64 round-trip:

```
# config problem=<p> n=<n> rho_beg=<..> rho_end=<..> maxfun=<..> npt=<..>
# x0 <x0_0> ... <x0_{n-1}>
<evalindex> <f> <x_0> ... <x_{n-1}>     # one per call, PRIMA's eval order
...
# final nf=<nf> rc=<rc> f=<f> x= <x_0> ... <x_{n-1}>
```

`rc` is PRIMA's return code (`0 == PRIMA_SMALL_TR_RADIUS`, the converged stop).

## What the test asserts

Three tiers (basin is paper-derived, not an FP-identical transcription of
PRIMA's Fortran, so per-eval trajectory parity is not expected past init):

1. **Objective equivalence** — the Rust objective recomputed at every traced
   point matches the fixture `f` to `1e-12` relative (catches C↔Rust drift).
2. **Initial design** — basin's first `npt` samples are the coordinate cross
   `{x0, x0 ± ρ_beg eₖ}` (§3), compared as a set (PRIMA emits all `+` then all
   `−`; basin interleaves).
3. **Final output** — basin converges (ρ reached) to the same minimizer: `f` to
   `1e-6·(1 + |f*|)` (absolute `~1e-6` when the optimum is near zero, relative
   otherwise), `x` to `1e-4` in `‖·‖∞`, `nf` within a 25% same-ballpark margin
   (not exact).

## Regenerating

The committed `.tsv` files are the artifacts the test reads; follow the steps
below only if you change the locked parameters (start, `rho_beg`, …) or the set
of problems. The generator source `newuoa_prima_driver.c` is vendored here. CI
never rebuilds the fixtures.

Inside the devenv shell (provides `cmake` and `gfortran`):

```bash
# 1. Build static libprima (Fortran + C binding) from the vendored submodule.
cmake -S tools/prima -B tools/prima/build \
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DPRIMA_ENABLE_C=ON
cmake --build tools/prima/build --target primac -j

# 2. Compile + link the generator. Link via gfortran so its runtime is found;
#    primac depends on primaf, so list primac first. -ffp-contract=off keeps the
#    objective arithmetic reproducible across compilers.
gcc -std=c99 -O2 -ffp-contract=off -DPRIMAC_STATIC \
    -I tools/prima/c/include \
    -c crates/basin/tests/fixtures/newuoa_prima_driver.c -o /tmp/newuoa_gen.o
gfortran -O2 -o /tmp/newuoa_gen /tmp/newuoa_gen.o \
    tools/prima/build/c/libprimac.a tools/prima/build/fortran/libprimaf.a -lm

# 3. Regenerate each fixture (run from crates/basin/tests/fixtures/).
/tmp/newuoa_gen rosenbrock 2 > newuoa_rosenbrock_2d.tsv
/tmp/newuoa_gen chrosen    5 > newuoa_chrosen_5d.tsv
/tmp/newuoa_gen arwhead    5 > newuoa_arwhead_5d.tsv
/tmp/newuoa_gen vardim     5 > newuoa_vardim_5d.tsv
```

`tools/prima/build/` is build output (not committed). The objective functions in
`newuoa_prima_driver.c` and in `parity.rs` must stay textually mirrored; the
tier-1 check enforces it.

# BOBYQA parity fixtures

`bobyqa_<problem>_<n>d.tsv` are reference runs of **PRIMA's** BOBYQA, the
bound-constrained sibling of NEWUOA (same vendored submodule, v0.7.2), used by
`crates/basin/src/solver/bobyqa/parity.rs` to cross-validate basin's BOBYQA
port. Unlike NEWUOA, BOBYQA has a public `Bobyqa` solver, so the parity test
drives the public `Executor` surface (recording the eval trace through a
problem wrapper) rather than a `pub(crate)` `minimize`; it still lives in-crate
to sit beside `newuoa/parity.rs` and read these files via `include_str!`.

Committed fixtures (`problem`, `n`, box, start `x0`, `rho_beg`):

- `bobyqa_rosenbrock_2d.tsv` — Rosenbrock 2D (basin form), box `[-5,5]²`,
  `x0=(-1.2, 1)`, `rho_beg=0.5`. Interior minimizer `(1,1)` with ≥ `2·rho_beg`
  slack on every coordinate, so bounds never bind and the initial design is the
  plain coordinate cross — confirms boxing doesn't perturb the unconstrained
  trajectory.
- `bobyqa_sphere_2d.tsv` — shifted sphere `Σ (xᵢ−3)²`, box `[-2,2]²`,
  `x0=(0,0)`, `rho_beg=0.5`. The unconstrained minimizer `(3,3)` lies *outside*
  the box, so the solution is the active corner `(2,2)` — exercises the TRSBOX
  active-set path and bound-aware ALTMOV.
- `bobyqa_chrosen_5d.tsv` — chained Rosenbrock 5D, wide box `[-10,10]⁵`,
  `x0=(-1,…)`, `rho_beg=0.5`. Interior minimizer, bounds never bind; a
  dimensional-scaling check.

All use `rho_end=1e-6`, `npt=2n+1`, `maxfun=500n`. Problems are chosen so PRIMA
and basin reach the *same* minimizer (see the NEWUOA chrosen_5d note above for
the multi-basin caveat that informs this choice). The locked inputs — now
including the box — live in the fixture itself (`# config` / `# x0` / `# xl` /
`# xu`), so the test recomputes nothing.

## Format

As NEWUOA's, plus two metadata lines recording the box (BOBYQA is
bound-constrained):

```
# config problem=<p> n=<n> rho_beg=<..> rho_end=<..> maxfun=<..> npt=<..>
# x0 <x0_0> ... <x0_{n-1}>
# xl <xl_0> ... <xl_{n-1}>
# xu <xu_0> ... <xu_{n-1}>
<evalindex> <f> <x_0> ... <x_{n-1}>     # one per call, PRIMA's eval order
...
# final nf=<nf> rc=<rc> f=<f> x= <x_0> ... <x_{n-1}>
```

## What the test asserts

The same three tiers as NEWUOA, with one BOBYQA adaptation in tier 2:

1. **Objective equivalence** — the Rust objective recomputed at every traced
   point matches the fixture `f` to `1e-12` relative.
2. **Initial design** — basin's first `npt` samples equal PRIMA's first `npt`
   samples *as a set* (within `1e-12`). BOBYQA's initial design is bound-aware,
   so this compares against the fixture rather than reconstructing the
   coordinate cross analytically (PRIMA emits all `+` then all `−`; basin
   interleaves).
3. **Final output** — basin converges (`SolverConverged`, ρ reached) to the
   same minimizer: `f` to `1e-6·(1 + |f*|)` (absolute `~1e-6` near a zero
   optimum, relative otherwise), `x` to `1e-4` in `‖·‖∞`, `nf` within a 25%
   same-ballpark margin.

## Regenerating

Same recipe as NEWUOA — build `libprima` once (step 1 above), then compile,
link, and run `bobyqa_prima_driver.c`:

```bash
# Compile + link the BOBYQA generator (reuses the libprima build from above).
gcc -std=c99 -O2 -ffp-contract=off -DPRIMAC_STATIC \
    -I tools/prima/c/include \
    -c crates/basin/tests/fixtures/bobyqa_prima_driver.c -o /tmp/bobyqa_gen.o
gfortran -O2 -o /tmp/bobyqa_gen /tmp/bobyqa_gen.o \
    tools/prima/build/c/libprimac.a tools/prima/build/fortran/libprimaf.a -lm

# Regenerate each fixture (run from crates/basin/tests/fixtures/).
/tmp/bobyqa_gen rosenbrock 2 > bobyqa_rosenbrock_2d.tsv
/tmp/bobyqa_gen sphere     2 > bobyqa_sphere_2d.tsv
/tmp/bobyqa_gen chrosen    5 > bobyqa_chrosen_5d.tsv
```

As with NEWUOA, CI never rebuilds these; the committed `.tsv` files are the
artifacts, and the objective fns in `bobyqa_prima_driver.c` and `parity.rs`
must stay textually mirrored (the tier-1 check enforces it).

## LINCOA fixtures

Same recipe — build `libprima` once (step 1 above), then compile, link, and run
`lincoa_prima_driver.c`. The problems are linear-inequality-constrained only
(`xl`/`xu` passed as `±INFINITY`, no equalities), so PRIMA's folded constraint
system equals the explicit `A x ≤ b` that basin's `Lincoa` folds.

```bash
# Compile + link the LINCOA generator (reuses the libprima build from above).
gcc -std=c99 -O2 -ffp-contract=off -DPRIMAC_STATIC \
    -I tools/prima/c/include \
    -c crates/basin/tests/fixtures/lincoa_prima_driver.c -o /tmp/lincoa_gen.o
gfortran -O2 -o /tmp/lincoa_gen /tmp/lincoa_gen.o \
    tools/prima/build/c/libprimac.a tools/prima/build/fortran/libprimaf.a -lm

# Regenerate each fixture (run from crates/basin/tests/fixtures/).
/tmp/lincoa_gen proj2   > lincoa_proj2_2d.tsv     # (x0-2)²+(x1-2)² s.t. x0+x1≤2
/tmp/lincoa_gen crosen2 > lincoa_crosen2_2d.tsv   # 2D Rosenbrock s.t. x0≤0.5
/tmp/lincoa_gen cquad3  > lincoa_cquad3_3d.tsv    # Σ(xᵢ-2)² s.t. x0+x1+x2≤3
```

The fixture carries the `A x ≤ b` system (`# aineq` / `# bineq`) and the final
`cstrv`; `solver/lincoa/parity.rs` rebuilds the same problem, recomputes the
objective at every traced point (tier 1), and asserts the converged `x`/`f`,
feasibility, and `nf` against PRIMA. The objective + constraint definitions in
`lincoa_prima_driver.c` and `parity.rs` must stay textually mirrored.

## COBYLA fixtures

`cobyla_<problem>_<n>d.tsv` are reference runs of **PRIMA's** COBYLA, the
nonlinearly-constrained Powell solver (same vendored submodule, v0.7.2), used by
`crates/basin/src/solver/cobyla/parity.rs` to cross-validate basin's COBYLA port.
The parity test drives the `pub(crate)` `CobylaWork` directly and reads these
files via `include_str!`.

Committed fixtures (`problem`, `n`, `m_nlcon`, start `x0`):

- `cobyla_disk_2d.tsv` — Powell (B): `min x0·x1 s.t. x0²+x1²−1 ≤ 0`, `F*=−0.5`.
  Started at the *asymmetric* feasible point `(0.7, −0.3)`: this objective has
  two sign-symmetric minima `(+,−)` and `(−,+)`, so a start off the symmetry
  axis keeps PRIMA and basin in the same basin (both reach `(√½, −√½)`).
- `cobyla_fletcher_2d.tsv` — Powell (F): `min −x0−x1 s.t. x0²−x1 ≤ 0,
  x0²+x1²−1 ≤ 0`, `F*=−√2`, `x0=(0.5, 0.5)`. Two active nonlinear constraints
  at the solution.
- `cobyla_ballsphere_3d.tsv` — convex sanity: `min Σ(xᵢ−2)² s.t. Σxᵢ²−1 ≤ 0`,
  `x0=(0.5, 0.5, 0.5)`. Unique minimizer on the unit sphere at `(1/√3)·𝟙`.

All use `rho_beg=0.5`, `rho_end=1e-6`, `maxfun=500n`, and a strictly feasible
start. Problems are chosen so PRIMA and basin reach the *same* minimizer (see the
NEWUOA chrosen_5d note above for the multi-basin caveat that motivates the disk
start). The locked inputs live in the fixture itself, so the test recomputes
nothing.

### Format

Differs from the linear-constraint fixtures: COBYLA's constraints are
*functions* (recomputed in Rust, not stored as matrix data), the simplex is
always `n+1` vertices (no `npt`), and each eval row carries the constraint
violation `cstrv = [maxᵢ cᵢ(x)]₊` so tier 1 can guard the constraint functions
too. Whitespace-separated, `%.17e`:

```
# config problem=<p> n=<n> m_nlcon=<m> rho_beg=<..> rho_end=<..> maxfun=<..>
# x0 <x0_0> ... <x0_{n-1}>
<evalindex> <f> <cstrv> <x_0> ... <x_{n-1}>   # one per calcfc call, PRIMA's order
...
# final nf=<nf> rc=<rc> f=<f> cstrv=<cstrv> x= <x_0> ... <x_{n-1}>
```

### What the test asserts

The same three tiers as the siblings, with COBYLA's constraints folded into
tier 1:

1. **Function equivalence** — the Rust objective *and* the constraint violation
   recomputed at every traced point match the fixture `f` / `cstrv` to `1e-12`
   (catches C↔Rust drift in either function).
2. **Initial design** — basin's first `n+1` samples equal PRIMA's first `n+1`
   samples *as a set* (within `1e-12`). COBYLA's simplex is built cumulatively
   with pole swaps, so this compares against the fixture rather than
   reconstructing a coordinate cross.
3. **Final output** — basin converges (ρ reached) to the same minimizer: `f` to
   `1e-6·(1+|f*|)`, `x` to `1e-4` in `‖·‖∞`, the returned point feasible
   (`cstrv ≤ 1e-6`), `nf` within a same-ballpark margin.

### Regenerating

Same recipe — build `libprima` once (step 1 above), then compile, link, and run
`cobyla_prima_driver.c`. The problems use only the nonlinear block (`m_ineq =
m_eq = 0`, `xl`/`xu = ±INFINITY`), matching basin's trait-only path.

```bash
# Compile + link the COBYLA generator (reuses the libprima build from above).
gcc -std=c99 -O2 -ffp-contract=off -DPRIMAC_STATIC \
    -I tools/prima/c/include \
    -c crates/basin/tests/fixtures/cobyla_prima_driver.c -o /tmp/cobyla_gen.o
gfortran -O2 -o /tmp/cobyla_gen /tmp/cobyla_gen.o \
    tools/prima/build/c/libprimac.a tools/prima/build/fortran/libprimaf.a -lm

# Regenerate each fixture (run from crates/basin/tests/fixtures/).
/tmp/cobyla_gen disk       > cobyla_disk_2d.tsv        # x0·x1 s.t. x0²+x1²≤1
/tmp/cobyla_gen fletcher   > cobyla_fletcher_2d.tsv    # −x0−x1 s.t. x0²≤x1, ‖x‖≤1
/tmp/cobyla_gen ballsphere > cobyla_ballsphere_3d.tsv  # Σ(xᵢ−2)² s.t. ‖x‖≤1
```

As with the others, CI never rebuilds these; the committed `.tsv` files are the
artifacts, and the objective + constraint fns in `cobyla_prima_driver.c` and
`parity.rs` must stay textually mirrored (the tier-1 check enforces it).
