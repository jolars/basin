---
description: >-
  Conventions for the optimization test-problem corpus under
  crates/basin/src/problems/ (file layout, wrapper struct, per-backend impls,
  ProblemSpec, tests, wiring).
paths:
  - "crates/basin/src/problems/**"
---

# Test-problem corpus conventions

Conventions for the optimization test-problem corpus
(`crates/basin/src/problems/`). See the root `AGENTS.md` for crate-wide tenets.

For the *workflow* of adding a brand-new problem from scratch, prefer the
`add-test-problem` project subagent (`.claude/agents/add-test-problem.md`): it
scaffolds the file, updates `ALL_SPECS`, and runs the verification gauntlet in
its own context. This file documents the *conventions* that subagent (and any
in-place edit) must follow.

## Module layout

Each problem gets one file: `src/problems/<name>.rs`. No subdirectories.

The file is structured top-to-bottom as:

1. Module-level rustdoc: formula, character, global minimum, cite the primary
   reference.
2. `use` statements (imports from `super::spec` and
   `crate::{CostFunction,    Gradient}`).
3. Raw functions on `&[f64]` slices: `pub fn <name>(x: &[f64]) -> f64` and
   `pub fn <name>_gradient(x: &[f64], out: &mut [f64])`. These are the primitive
   math; everything else routes through them.
4. The wrapper struct:
   `pub struct <Name><P = Vec<f64>>(PhantomData<fn() ->    P>)` with `new()`,
   `Default`.
5. `pub static <NAME>_SPEC: ProblemSpec` (see below).
6. `impl<P> HasSpec for <Name><P>`: blanket; pulls metadata from `<NAME>_SPEC`.
7. `CostFunction` + `Gradient` impls for `<Name><Vec<f64>>` (always-on), then
   per-backend impls each in their own
   `#[cfg(feature = "...")] mod    <backend>_impl { ... }` block. Order:
   nalgebra → ndarray → faer.
8. `#[cfg(test)] mod tests { ... }`: see Tests below.

## The wrapper struct

Always:

```rust
pub struct Foo<P = Vec<f64>>(PhantomData<fn() -> P>);
```

`PhantomData<fn() -> P>` (not `PhantomData<P>`) so the struct stays covariant
and doesn't require `P: Send + Sync` for auto-traits.

The `P = Vec<f64>` default is for downstream-with-no-backend ergonomics. **It
will not help inference inside this crate's tests** when multiple backend
features are enabled: explicit turbofish (`Foo::<Vec<f64>>::default()`) is
required there. Don't try to "fix" this with type aliases.

## Per-backend impls

- `Vec<f64>`: always present, routes through the slice-based primitives.
- `nalgebra::DVector<f64>`: gated on `feature = "nalgebra"`. Use
  `x.as_slice()`/`out.as_mut_slice()` to route through the primitives.
- `ndarray::Array1<f64>`: gated on `feature = "ndarray"`. Use
  `x.as_slice().expect("Array1 is contiguous")` and the `_mut` variant.
- `faer::Col<f64>`: gated on `feature = "faer"`. Faer's `Col` doesn't expose a
  `&[f64]` cleanly across all 0.24 APIs, so write the math elementwise inside
  the impl rather than routing through the primitives. \~10 lines of duplication
  is the right call.

Each per-backend block lives in its own `mod <backend>_impl` to keep imports
local and `#[cfg]` clutter contained.

## ProblemSpec

`pub static <NAME>_SPEC: ProblemSpec` next to the wrapper struct. Required
fields:

- `name`: canonical literature name, e.g. `"Rosenbrock"`. Title case.
- `dim`: `Dimensionality::Fixed(n)` for 2D-only problems (Beale etc.) or
  `NDimensional { min: n }` for scalable ones.
- `properties`: `Properties { ... }` literal. **Be conservative with
  `unimodal`**: for N-D problems where unimodality depends on `n` (e.g.
  Rosenbrock's spurious local min for n ≥ 4), set `false` and explain in the
  description. Same conservative rule applies to `convex` if the search domain
  isn't the whole of `R^n`.
- `references`: `&[Reference { ... }]`, **at least one entry, all real**. Not
  just URLs: citation, title, source or venue, and DOI when available. The first
  entry is the primary citation. URLs (S&B, arXiv) go in `Reference::url` as the
  publicly-accessible link, not in lieu of the citation. If no single original
  paper exists (e.g. Sphere, where De Jong popularized but didn't invent), cite
  the popularizing reference.
- `description`: 1--3 sentences for a UI tooltip. Mention the global minimum
  location and value.

Then
`impl<P> HasSpec for Foo<P> { const SPEC: &'static ProblemSpec = &FOO_SPEC; }`:
always blanket over `P`, since the spec is a property of the math, not the
backend.

## Tests

Required cases in `mod tests`:

- Value at the known global minimum equals the documented value (usually 0).
- Value at one well-known non-trivial point matches a hand-computed number.
- Gradient at the global minimum is ≈ zero.
- Gradient matches central finite-difference at a non-symmetric point
  (`tol ≈ 1e-5`). Pick a point with no zero coordinates.
- Spec-wiring sanity check: `<Foo<Vec<f64>> as HasSpec>::SPEC` resolves and has
  the expected name + at least one property + non-empty references.

Don't write integration tests for new problems unless the problem is exercising
a previously-uncovered solver path. The unit tests above cover the math; backend
integration tests for solvers already exist generically in `tests/`.

## After adding a problem

In a single pass:

1. Append `pub mod <name>;` to `src/problems.rs`.
2. Add `pub use <name>::{<name>, <name>_gradient, <Name>, <NAME>_SPEC};` to the
   re-exports.
3. Append `&<NAME>_SPEC` to `ALL_SPECS`.
4. Tick the corresponding entry in `TODO.md` under the "Test problem corpus"
   heading (`- [ ]` → `- [x]` with a `*(done)*` marker).
5. Run the verification gauntlet:
   - `cargo test --all-features`
   - `cargo clippy --all-features --all-targets -- -D warnings`
   - `cargo build --target wasm32-unknown-unknown`
   - `cargo build --target wasm32-unknown-unknown --no-default-features`

All four must pass. The wasm builds ensure the new problem doesn't pull in
anything that breaks the WASM hard constraint (see root `AGENTS.md`).

### Web visualizer (when 2D-friendly)

The `web/` solver demo (consumed via `crates/basin-wasm`) is restricted to 2D
problems on the `Vec<f64>` backend. If the new problem fits (`dim` is
`Fixed(2)`, or `NDimensional { min }` with `min <= 2`), wire it in:

- Add a variant to `ProblemKind` in `crates/basin-wasm/src/lib.rs` and extend
  `Problem2D`'s `CostFunction` + `Gradient` match arms to call the raw
  `<name>`/`<name>_gradient` functions.
- Add a `ProblemMeta` entry to the `PROBLEMS` array in `web/src/lib/problems.ts`
  with `kind`, `label`, the documented search `domain`, `minimum`, an
  `intensity` choice (`'sqrt'` for mild quadratic surfaces, `'log1p'` for
  high-dynamic-range surfaces like Rosenbrock and Beale), and a `gdAlphaDefault`
  that converges from a typical start within a few hundred iterations.
- Verify with `cd web && npm run build` (rebuilds the wasm too).

Skip the web wiring if the problem is intrinsically high-dimensional (e.g. a
function whose minimum geometry only emerges for `n >> 2`).

## Anti-patterns

- **Don't add a `Foo<P>` impl that calls `cost`/`gradient` on a generic `P` via
  traits.** Inference fails when multiple impls match. Concrete per-backend
  impls are explicit and compile-checked.
- **Don't store the math in the spec.** `ProblemSpec` is metadata only; the
  `CostFunction`/`Gradient` impls own the math.
- **Don't add a `Properties` field for a one-off tag.** The set is small on
  purpose. Add a field only when at least two problems would set it differently
  *and* a filter would care about it.
- **Don't introduce a separate `references/<name>/` directory** for these.
  Test-function references are short structured data and belong in the spec, not
  as filesystem artifacts. (Solver-paper ingestion via the `ingest-paper` skill
  is a separate concern.)
