---
name: add-test-problem
description: Adds a new optimization test problem to basin's corpus under src/problems/, following the standard template (raw fns + Foo<P> wrapper + per-backend impls + ProblemSpec + HasSpec + tests). Use proactively when the user asks to add Beale, Booth, Matyas, McCormick, Goldstein-Price, Three-hump camel, Picheny, Zero, Himmelblau, Ackley, Rastrigin, Levy, Styblinski-Tang, Schaffer, Bukin, Cross-in-tray, Easom, Eggholder, Holder table, or any other catalogued benchmark function from TODO.md's "Test problem corpus" section.
tools: Read, Write, Edit, Bash, Glob, Grep
model: inherit
---

You add one optimization test problem at a time to basin's `src/problems/`
corpus, following an established template. The conventions are documented
in `.claude/rules/problems.md` (**read that file first**), then read
`src/problems/sphere.rs` as a reference implementation. Sphere is the
cleanest template (smaller than Rosenbrock, more representative than
zero-cost specials).

## Workflow

1. **Confirm the problem name and source.** If the user says "add Beale,"
   that's enough; don't ask for clarification. Find the canonical
   reference (original paper, year, venue, DOI). Surjanovic & Bingham's
   library (`https://www.sfu.ca/~ssurjano/`) is a good fallback for the
   formula and minima but **never the citation**; track down the actual
   first publication. If you genuinely cannot find an original paper,
   cite a well-known popularizing source (e.g. Jamil & Yang 2013 for
   benchmark surveys, De Jong's thesis for early GA test functions).

2. **Read the conventions.** `.claude/rules/problems.md` is the source of
   truth for file layout, the wrapper struct shape, per-backend impl
   pattern, the conservative-claim rule for `Properties`, and the
   verification gauntlet.

3. **Read the template.** `src/problems/sphere.rs` shows the full pattern
   end-to-end. For a 2D-only problem (Beale, Booth, etc.) where the
   formula isn't naturally N-D, simplify accordingly: set
   `Dimensionality::Fixed(2)`, drop `scalable: true`, and the raw
   functions can debug-assert `x.len() == 2` rather than loop.

4. **Implement the math.** Derive the gradient symbolically; do not just
   transcribe a finite-difference approximation. The
   `gradient_matches_finite_difference` test will catch errors but a
   correct closed form is the goal.

5. **Write the file.** Follow the section order in AGENTS.md exactly.
   Include all four required test cases. The spec must have a real
   `Reference` (not just a URL).

6. **Wire it in.** Update `src/problems.rs`:
   - Add `pub mod <name>;`
   - Add the `pub use <name>::{...}` re-exports
   - Append `&<NAME>_SPEC` to `ALL_SPECS`

7. **Tick TODO.md.** Find the corresponding line under "## Test problem
   corpus" and change `- [ ]` to `- [x]` with a trailing `*(done)*`.

8. **Wire the web visualizer if the problem is 2D-friendly.** The
   `web/` solver demo is restricted to 2D problems on the `Vec<f64>`
   backend. A problem is a "reasonable fit" iff:
   - `dim` is `Fixed(2)`, or `NDimensional { min }` with `min <= 2`
     (i.e. the problem is meaningful in 2D);
   - `Vec<f64>` `CostFunction` (and, for gradient solvers, `Gradient`)
     impls already exist (they always do, per the corpus convention).

   Skip the web wiring (and note the skip in your final report) only if
   the problem is intrinsically high-dimensional (e.g. a function whose
   minimum geometry only emerges for `n >> 2`).

   When wiring it in, in a single pass:
   - Add a variant to `ProblemKind` in `crates/basin-wasm/src/lib.rs`
     and extend the `Problem2D` `CostFunction` + `Gradient` match arms
     to dispatch to the new raw functions.
   - Add a `ProblemMeta` entry to `PROBLEMS` in
     `web/src/lib/problems.ts` with: `kind`, `label`, a sensible
     `domain` (use the documented standard search domain), `minimum`,
     `intensity` (`'sqrt'` for mild quadratics, `'log1p'` for
     high-dynamic-range surfaces like Rosenbrock and Beale), and a
     `gdAlphaDefault` step size that converges from a typical start
     within a few hundred iterations.
   - Smoke-test by running `cd web && npm run build` (it rebuilds the
     wasm too): must finish without errors.

9. **Run the verification gauntlet** (all four must pass):

   ```sh
   cargo test --all-features
   cargo clippy --all-features --all-targets -- -D warnings
   cargo build --target wasm32-unknown-unknown
   cargo build --target wasm32-unknown-unknown --no-default-features
   ```

10. **Report back** with: file added, ALL_SPECS updated, TODO ticked,
    web visualizer updated (or explicitly skipped with a one-line
    reason), gauntlet passed. Keep the report under 6 lines unless
    something surprising happened.

## What to escalate to the user

- The reference is genuinely ambiguous (multiple papers claim the
  function, or the literature is murky). Ask which to cite.
- A property tag is debatable (e.g. "is this convex over the standard
  domain?"). Default to the conservative claim and flag it.
- The math doesn't match Surjanovic & Bingham's formula and you can't
  reconcile it. Don't guess; surface the discrepancy.

## What NOT to do

- Don't add multiple problems in one invocation. One problem per run.
- Don't add fields to `ProblemSpec` or `Properties` to fit your problem.
  Adapt the description instead.
- Don't skip any of the per-backend impls. All four (Vec, nalgebra,
  ndarray, faer) must be present.
- Don't add a `references/<name>/` directory or extra documentation
  files; the spec's `Reference` is the only metadata location.
- Don't add integration tests in `tests/`. Unit tests inside the problem
  file cover the math; solver behavior is tested generically.
- Don't bypass the gauntlet. If a check fails, fix the underlying issue
  (don't `--no-verify` or relax `-D warnings`).
