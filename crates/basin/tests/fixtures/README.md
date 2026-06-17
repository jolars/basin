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
   `1e-6` relative, `x` to `1e-4` in `‖·‖∞`, `nf` within a 25% same-ballpark
   margin (not exact).

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
