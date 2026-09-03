# AGENTS.md

This file is the operational repository guide for AI agents. Detailed design
rationale lives in `CONTRIBUTING.md`; read the relevant sections there before
changing architecture, public APIs, dependencies, or platform support.

## Project priorities

Basin is a semver-stable Rust numerical-optimization library with a generic
`Executor`/`Solver`/`State` core and support for `Vec<f64>`, nalgebra, ndarray,
and faer backends.

- Preserve public API compatibility. Treat changes to public signatures,
  required trait methods, enum variants, generic defaults, feature semantics,
  and re-exports as potentially breaking. Prefer additive changes and
  default-bodied trait methods; raise unavoidable breaking changes before
  implementing them.
- Preserve the default `wasm32-unknown-unknown` build. Gate file I/O, threads,
  rayon, and BLAS/LAPACK-linked math behind non-default features. Use `web-time`
  rather than `std::time::Instant` in default paths.
- Do not bump the package MSRV casually. Check the current CRAN toolchain before
  changing `rust-version`. The development toolchain may be newer when a
  versioned opt-in backend requires it. Every dependency selected by the
  MSRV-compatible features, including dev dependencies used during publishing
  and CI, must compile on the MSRV. Document the reason beside any MSRV-driven
  pin or feature-specific exception.

## Verification

Run focused tests while developing, then the checks matching the changed scope:

- Rust formatting: `cargo fmt --all -- --check`.
- Routine pure-Rust tests:
  `cargo test -p basin --features nalgebra_latest,ndarray_latest,faer_latest,problems,parallel`.
- Workspace lint: `cargo clippy --workspace --all-targets --all-features -- -D warnings`.
- Public documentation:
  `cargo doc --no-deps -p basin --features nalgebra_latest-lapack,ndarray_latest-blas,faer_latest,parallel,problems,serde`.
- WASM-sensitive changes:
  `cargo build --target wasm32-unknown-unknown` and
  `cargo build --target wasm32-unknown-unknown --no-default-features`.
- Web changes, from `web/`: `pnpm format:check`, `pnpm lint`, `pnpm check`, and
  `pnpm build`.

Do not assume `cargo test --all-features` links without a BLAS/LAPACK provider.
The `nalgebra-lapack` and `ndarray-blas` features intentionally select no
provider. Supply one through linker flags when those tests are required; clippy
and rustdoc only check and do not link. The package MSRV is Rust 1.87.0; CI
checks it separately. The dev environment pins Rust 1.89.0, supplies the WASM
target and project tooling, and runs all-feature clippy and rustfmt in
pre-commit.

## Architecture and repository shape

- `crates/basin/src/lib.rs` defines the documented public surface through module
  declarations and re-exports.
- `crates/basin/src/core/problem.rs` defines user-implemented cost, gradient,
  residual, Jacobian, and Hessian traits;
  `crates/basin/src/core/numdiff.rs` provides finite differences.
- `crates/basin/src/core/state.rs` and `crates/basin/src/core/state/` define
  `State`, concrete states, and the minimum-shape extension traits used by
  termination criteria.
- `crates/basin/src/core/solver.rs` defines `Solver`;
  `crates/basin/src/core/executor.rs` owns the driver loop; and
  `crates/basin/src/core/termination.rs` owns shared criteria.
- Problem-side constraints and adapters live under `crates/basin/src/core/` in
  `constraint.rs`, `barrier.rs`, and `augmented_lagrangian.rs`; composition
  contracts live in `inner.rs`.
- `crates/basin/src/core/math.rs` and `crates/basin/src/core/math/` implement the
  tiered backend layer.
- `crates/basin/src/solver.rs` and `crates/basin/src/solver/` contain concrete
  solvers; `crates/basin/src/core/rng.rs` supports stochastic algorithms.

Use `src/foo.rs` plus `src/foo/bar.rs`; never introduce `mod.rs`.

The root workspace contains `crates/basin`, `crates/basin-wasm`, and
`crates/competitor-bench`. `web/` is a separate Svelte project. Keep optional
backend, parallel, and problem integrations as features on `basin`. Add a
workspace crate only for heavy or platform-specific dependencies that do not
belong in the core crate.

## Design constraints

1. Preserve conventional optimization-framework vocabulary and the generic
   driver-loop shape unless another constraint requires divergence.
2. Each supported backend release has an exact version feature. The
   `*_latest` aliases move to the newest supported release, while the original
   unversioned features retain their Basin 1.x meanings. If dependency feature
   unification enables several releases of one backend, implement the newest
   enabled release.
3. Generic stopping criteria belong to the executor/shared termination layer;
   solver-specific controls stay on the solver. Bind each criterion to the
   minimum state shape it needs.
4. Constraints describe problems: keep them problem-side, never on state or as
   executor configuration. Solvers declare supported constraint traits;
   projection, barrier, and penalty adapters are explicit opt-ins.
5. Keep the universal vector math tier small and the richer linear-algebra tier
   capability-based. Missing operations must fail at compile time.

Public types that store or expose scalar-valued state should use `F: Scalar`
with an `F = f64` default. Carry that generic through state, solver,
termination, and math implementations when the surface requires it; do not add
a scalar parameter to a type that does not need one. Preserve the `f32`
round-trip coverage in `crates/basin/tests/f32_round_trip.rs`.

## Numerical code and backend math

- Add a regression test for behavioral fixes and analytic or reference checks
  for new numerical algorithms. Use deterministic seeds in stochastic tests,
  scale-appropriate approximate comparisons, and explicit coverage of relevant
  degenerate or non-finite inputs.
- Test every backend claimed in public rustdoc. Every solver rustdoc must include
  a `# Backends` note listing supported parameter types.
- Keep shared vector traits limited to operations every backend implements well.
  First-order and derivative-free solvers should remain generic over this tier.
- Put richer matrix operations in `crates/basin/src/core/math/linalg.rs`.
  Linear-algebra-heavy solvers must bound only the capabilities they need.
- Add backend operations only when they can be implemented honestly in pure
  Rust and without a BLAS/LAPACK link or fake stub. It is acceptable to document
  a realistic backend gap.

## Change-specific synchronization

### Test problems

For a new corpus problem, use the `add_test_problem` project subagent. Its
instructions in `.codex/agents/add-test-problem.toml` own the full file layout,
metadata, backend, testing, visualizer, and verification workflow. Add exactly
one problem per task; do not replace symbolic production gradients with finite
differences or omit a supported backend.

### Web documentation

The static SvelteKit site at `https://basin.rs/` signposts to the authoritative
docs.rs API rather than copying it. When a top-level section is added, removed,
or renamed, update the hand-maintained `web/src/routes/llms.txt/+server.ts`;
keep it a link-oriented signpost with absolute `SITE_ORIGIN` URLs and trailing
slashes. The sitemap discovers pages automatically.

### Solver catalogue

When a solver is added, removed, renamed, or changes backend support or
references, update `web/src/routes/docs/solvers/+page.svx` in the same change:

- The prose catalogue must equal the public solver re-exports in
  `crates/basin/src/lib.rs`.
- The support matrix must match each solver's rustdoc `# Backends` section. A
  check means it compiles and runs. For coverage depending on an inner solver,
  show broad coverage and add a footnote.
- Each bullet must link to docs.rs and reproduce the solver rustdoc reference.

Canonical docs.rs links include the defining snake-case submodule:
`https://docs.rs/basin/latest/basin/solver/<module>/struct.<Name>.html` or the
equivalent `line_search/<module>/...`. Root-module struct links fail. Aliases
link to the defining type—for example, both `Lbfgs` and `Lbfgsb` link to
`solver/lbfgs/struct.Lbfgs.html`. A new solver may legitimately be absent from
`/latest/` until the next release.
