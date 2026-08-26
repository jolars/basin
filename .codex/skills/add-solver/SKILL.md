---
name: add-solver
description: Add one research-grounded numerical optimization solver to Basin, including its public API, state and math integration, backend tests, rustdoc references, and solver-catalog synchronization. Use for a new concrete solver, not for routine fixes to an existing solver or a line-search component alone.
---

# Add a Basin Solver

Implement one public solver per invocation and carry it through code, tests,
documentation, and catalogue integration. Preserve the solver and variant the
user requested. If no solver has been selected and the user has not delegated
the choice, identify the missing choice before implementation; recommendations
should be based on a concrete gap in Basin's current catalogue.

## Establish the Algorithm

Read `AGENTS.md`, the relevant sections of `CONTRIBUTING.md`,
`crates/basin/src/core/solver.rs`, and the closest existing solver before
designing the change. Inspect the state and math traits the algorithm appears to
need rather than assuming a new abstraction is necessary.

Research the algorithm from authoritative sources before writing production
code:

- Find at least one credible primary source: the original paper, an archival
  algorithm description, an official technical report, or an authoritative
  monograph treatment. Prefer both a paper and reference code when they exist.
- Prefer author-maintained code, an ACM TOMS implementation, or a well-maintained
  research library as the executable reference. Record its exact version or
  commit and its license.
- Lock down the named variant, equations, update ordering, defaults, convergence
  test, constraint model, and documented exceptional cases. Resolve material
  disagreements between paper, pseudocode, and code before implementation.
- Treat reference code as an oracle unless its license is compatible with
  Basin and a direct port is intentional. Do not copy or vendor code with an
  unclear or incompatible license. An independent implementation from the
  paper may still use reference outputs for parity tests.
- Put downloaded papers and external source trees under the gitignored
  `references/` directory. Commit only durable, license-compatible artifacts
  needed by tests, such as compact output fixtures and their regeneration
  driver or instructions.

If neither a credible algorithm source nor a trustworthy executable reference
can be found, explain the evidence gap before presenting the implementation as
research-based. Cite the sources that actually guided the code—do not borrow a
nearby citation merely because it is conventional.

Before coding, settle these design facts: public type and constructor, required
problem traits, constraint support, state shape, minimum math capabilities,
claimed backends, scalar genericity, solver-specific controls, and the test
oracle. Raise a breaking public-API requirement rather than quietly changing
an existing signature or trait contract.

## Design Tests First

Start with focused tests that fail for the missing solver. Exercise the public
`Executor` path, not only private kernels. Choose tests in proportion to the
algorithm, including:

- an analytic benchmark with a known solution or other paper-backed invariant;
- deterministic reference or trajectory parity when usable reference code
  exists—compare only invariants the two implementations should genuinely
  share;
- initialization and one-step invariants, including consistency among the
  current parameter, cost, derivative data, and evaluation counts;
- constraints, degeneracies, invalid configuration, and non-finite behavior
  relevant to the algorithm;
- deterministic seeds and reproducible trajectories for stochastic methods;
- every backend claimed in public rustdoc, with scale-appropriate approximate
  comparisons; and
- `f32` round-trip coverage when the new public surface stores or exposes a
  scalar-valued state.

Keep parity fixtures small and document their provenance, locked inputs,
comparison tolerances, and regeneration procedure in
`crates/basin/tests/fixtures/README.md`. Do not demand floating-point-identical
trajectories when algebraic ordering or a legitimate variant differs.

## Integrate with Basin

Put the solver in `crates/basin/src/solver/<snake_case>.rs`; use sibling files
under `solver/<snake_case>/` for a substantial implementation and never add
`mod.rs`. Follow the closest solver's public shape where the research does not
dictate a difference.

- Reuse an existing state when it honestly represents the iterate and solver
  history. Add a state only for a genuinely new shape, keep its fields
  `pub(crate)`, expose data through the appropriate traits, and carry
  `F: Scalar` with an `F = f64` default through scalar-valued public types.
- Implement the `Solver` lifecycle exactly. `init` must seed every field that
  termination criteria or `next_iter` can read at iteration zero. Each
  successful step must return mutually consistent current state. Use
  `terminate` for clean current-state convergence, return a
  `TerminationReason` for mid-step soft stops, and propagate the user's typed
  problem error for hard aborts.
- Route all objective and derivative evaluations through `Problem`; never
  maintain evaluation counters by hand. Use batch evaluation where the
  algorithm has independent points, while preserving deterministic ordering.
- Keep generic budgets and tolerances in the shared termination layer. Put only
  controls peculiar to the algorithm on the solver.
- Keep constraints on the problem side and bound the solver on the precise
  constraint traits it supports.
- Bound only the math capabilities the method needs. Keep first-order and
  derivative-free methods on the universal vector tier when possible; add a
  `linalg` capability only when every advertised implementation can provide the
  real operation in pure Rust without a fake fallback.
- Use Basin's RNG infrastructure and an explicit seed for stochastic methods.
  Gate parallel execution behind the existing `parallel` feature and preserve
  reproducibility across serial and parallel evaluation.
- Preserve the default `wasm32-unknown-unknown` build. Avoid new dependencies
  when existing math traits suffice; assess the MSRV, WASM support, feature
  semantics, and license before adding any dependency.

Add the module and re-export in `crates/basin/src/solver.rs`, then re-export the
public solver—and any state or strategy users must name—from
`crates/basin/src/lib.rs`. Keep the addition semver-compatible. Update a matching
`TODO.md` item if one exists; do not reorganize unrelated TODOs.

## Document the Contract and Evidence

The solver rustdoc should explain the implemented variant, essential update
rule, configuration, caller and state requirements, termination behavior,
numerical safeguards, and a runnable public example. Include a `# Backends`
section whose claims are verified by tests. Give complete research references,
including DOI or stable source URL when available, and identify the reference
implementation and version when parity was used.

Synchronize `web/src/routes/docs/solvers/+page.svx` in the same change:

- add the solver to the appropriate prose family with the same reference;
- use the canonical docs.rs URL containing the defining snake-case module;
- update the support matrix to match the rustdoc `# Backends` claim; and
- add a footnote when support depends on a strategy or inner solver.

The catalogue must still equal the solver re-exports in
`crates/basin/src/lib.rs`. Do not expand the visualizer or add unrelated web
pages unless the user requested that work.

## Verify

Run focused tests during development, followed by the repository checks that a
new default-path solver requires:

```text
cargo fmt --all -- --check
cargo test -p basin --features nalgebra,ndarray,faer,problems,parallel
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo doc --no-deps -p basin --all-features
cargo build --target wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown --no-default-features
```

Because the solver catalogue changes, run from `web/`:

```text
pnpm format:check
pnpm lint
pnpm check
pnpm build
```

Do not substitute a bare `cargo test --all-features`; the repository
intentionally needs an explicit BLAS/LAPACK provider to link that matrix. Report
the research basis and variant, implementation and public-surface changes,
backend coverage, parity status, and every verification result. Distinguish
failures caused by the change from pre-existing or environment failures.
