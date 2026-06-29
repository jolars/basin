# Contributing

## State

Basin follows [semantic versioning]. As of 1.0.0 the public API is stable:
breaking changes ship only in a major release, while new solvers, backends, and
opt-in features arrive in minor releases.

[semantic versioning]: https://semver.org/

## What this is

Basin is a Rust library crate for numerical optimization, inspired by `argmin`.
It pairs a small generic core (problem traits you implement, a pluggable
termination layer, and an `Executor` driver loop) with a growing set of solvers
spanning first-order and quasi-Newton (gradient descent, BFGS, L-BFGS and L-BFGS-B),
derivative-free (Nelder-Mead, Brent, and Powell's model-based family
NEWUOA/BOBYQA/LINCOA/COBYLA), nonlinear least squares (Gauss-Newton,
Levenberg-Marquardt, trust-region-reflective), global and stochastic (random search,
CMA-ES, a steady-state GA, memetic combinations), and constrained methods
(projected gradient, bounded Nelder-Mead, L-BFGS-B, and CMA-ES, log-barrier,
augmented Lagrangian, and COBYLA for nonlinear inequality constraints). Solvers
are generic over the linear-algebra backend (`Vec<f64>`, nalgebra, ndarray,
faer).

## Commands

- `cargo build`: build the library.
- `cargo test`: run tests.
- `cargo test <name>`: run a single test by name.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: lint.
- `cargo doc --no-deps -p basin --all-features`: build the docs. CI runs this
  and `lib.rs` has `#![deny(rustdoc::broken_intra_doc_links)]`, so a broken or
  ambiguous intra-doc link (e.g. `[`Foo`](super::foo)` where `foo` is both a
  module and a function; link the struct `super::Foo` instead) fails the build.
  Run this before committing any rustdoc changes.
- `cargo fmt`: format (also enforced by pre-commit).

### `cargo test --all-features` needs a BLAS/LAPACK provider

`cargo clippy --all-features` and `cargo doc --all-features` work out of the box
because they only *check* and never *link*. `cargo test --all-features` is
different: it links executables, and the `nalgebra-lapack`/`ndarray-blas`
features deliberately pull in **no** BLAS/LAPACK provider crate (see the
`Cargo.toml` comments: `nalgebra-lapack` forwards `lapack-custom` precisely so
the rlib/docs build without a Fortran toolchain). So a bare
`cargo test --all-features` fails at link time with `undefined reference to
dsyev_`/`dpotrf_`—this is by design, not a missing system library. Installing
`liblapack`/`libblas` is not enough on its own; nothing tells the linker to use
them.

To actually run the LAPACK-backed tests, supply a provider at link time. With
any OpenBLAS in scope (it provides both BLAS and LAPACK symbols):

```fish
# NixOS example: point at an OpenBLAS in the store
set OB (ls -d /nix/store/*openblas-*/lib | head -1)
RUSTFLAGS="-L $OB -l openblas" cargo test -p basin --all-features
```

Outside Nix, an `-L <dir> -l openblas` (or `-l lapack -l blas` for reference
netlib) pointing at your system libraries does the same. CI does **not** run
`--all-features` *tests*; the routine local test command is the pure-Rust
feature set: `cargo test -p basin --features nalgebra,ndarray,faer,problems,parallel`.

The dev environment is provided by `devenv.nix` (loaded automatically via
`direnv` from `.envrc`). It pins Rust 1.87.0 (matches `rust-version` in
`Cargo.toml`) and adds the `wasm32-unknown-unknown` target plus tooling:
`cargo-llvm-cov`, `cargo-flamegraph`, `cargo-audit`, `cargo-deny`, `cargo-msrv`,
`samply`, `wasm-pack`, `go-task`. Pre-commit hooks run `clippy` (with
`allFeatures = true`) and `rustfmt`.

## Architecture

A generic driver loop (`Executor`) iterates a `Solver` over a `State`, calling
into user-provided `Problem` traits, until a `TerminationCriterion` fires.

- `src/lib.rs`: public re-exports only.
- `src/core.rs` + `src/core/`: the framework:
  - `problem.rs`: traits the *user* implements: `CostFunction`, `Gradient`,
    `Residual` + `Jacobian` (least squares), `Hessian` (second order).
  - `numdiff.rs`: the `FiniteDiff` wrapper: synthesizes `Gradient`/`Jacobian`/`Hessian`
    from function values via finite differences.
  - `state.rs` (+ `state/`): the `State` trait and concrete states:
    `BasicState<P>` (single iterate), `BasicSimplexState<V>` (simplex),
    `QuasiNewtonState<V, M>` (BFGS), `LbfgsState` (L-BFGS history),
    `BasicPopulationState<V>` (population). Extension traits `GradientState`/`SimplexState`/`PopulationState`
    expose the richer shape that termination
    criteria bound on. Fields are `pub(crate)`; access goes through trait
    methods.
  - `solver.rs`: the `Solver` trait: `init` (one-time setup, e.g. seeding
    cost/gradient at iter 0), `next_iter`, plus a `terminate` hook.
  - `executor.rs`: `Executor` owns problem + state + solver and drives the loop;
    `run()` returns an `OptimizationResult<S>` (final state +
    `TerminationReason`). Also `run_loop`/`Stepper`.
  - `termination.rs`: `TerminationCriterion<S>` plus shipped criteria
    (`MaxIter`, `MaxCostEvals`, `MaxGradientEvals`, the `*Tolerance`/`Relative*Tolerance`
    family, `SimplexTolerance`, `MaxTime`).
  - `constraint.rs`, `barrier.rs`, `augmented_lagrangian.rs`: constraint markers
    and the unconstrained-problem adapters (tenet 4).
  - `inner.rs`: `InnerExecutor`/`WarmStart` for solver composition.
  - `math.rs` + `math/`: the backend math layer: a shared vector tier plus the
    `linalg` tier, with per-backend impls for `Vec<f64>` (incl. `dense.rs` /
    `dense_eig.rs`), nalgebra (+sparse), faer (+sparse), and ndarray (tenet 5).
  - `rng.rs`: RNG support for stochastic solvers.
- `src/solver.rs` + `src/solver/`: concrete solvers spanning the families in
  "What this is", with pluggable line searches (`Backtracking`, `Wolfe`,
  `More-Thuente`, `Constant`) where applicable.

Module convention: **no `mod.rs`**: use `src/foo.rs` for the module file and
`src/foo/bar.rs` for submodules.

## Design tenets

These shape API decisions and are non-obvious from the code alone.

1. **Conventional vocabulary and shape.** basin uses the established
   optimization-framework vocabulary (`Executor`, `Solver`, `Problem` traits,
   `IterState`-style `State`) and a generic driver-loop architecture. Familiar
   names lower the barrier for users arriving from existing frameworks; diverge
   only when another tenet demands it.
2. **One feature per backend, one pinned version.** Each linear-algebra backend
   (`nalgebra`, `ndarray`, `faer`) is a single Cargo feature pinning one major
   version; `Vec<f64>` needs none. A backend major bump is a basin major bump.
   No per-version feature gates (`nalgebra-v0_33`/`-v0_34`): they multiply the
   test matrix and maintenance surface for little gain.
3. **Framework-level termination.** Generic stopping conditions (`max_iter`, the
   `*_tolerance` family, `max_time`, eval budgets) are configured uniformly on
   the `Executor`/shared termination layer, not per solver; solver-specific
   knobs stay on the solver. Each criterion binds on the *minimum state shape*
   it needs (e.g. `GradientTolerance` requires `S: GradientState`), so a
   derivative-free solver can't be paired with a gradient criterion by mistake.
   Because derivative-free solvers have no gradient, termination is pluggable
   and opt-in based on what the state and problem expose.
4. **First-class constraints.** Constraints describe the *problem*, so they live
   problem-side, not as executor config, never on state. Solvers declare support
   via traits; a constrained problem handed to an unconstrained solver is a
   compile error, with opt-in adapters (projection, barrier, or penalty) to wrap
   unconstrained solvers. Box bounds and linear (in)equalities ship today;
   nonlinear is future.
5. **Tiered, broadening backends.** A small universal *vector tier* (ops every
   backend implements well) keeps first-order and derivative-free solvers
   backend-generic; a richer *`linalg`tier* holds matrix ops that LA-heavy
   solvers bound on by the minimum subset they need, so a missing op is a
   compile error, not a runtime surprise. Coverage broadens over time: add an op
   to a backend the moment it can be done honestly (pure-Rust, wasm-clean, no
   BLAS/LAPACK, no stub).

## WASM as a hard constraint

Basin must build for `wasm32-unknown-unknown` out of the box: a constraint on
dependencies, not a feature. CI enforces it
(`cargo build --target wasm32-unknown-unknown`).

- Every default dep must be wasm-compatible. Anything that isn't (file I/O,
  threads, BLAS/LAPACK-linked math) sits behind a non-default feature.
- No `std::time::Instant` in default paths; use `web-time` or feature-gate the
  time-based criterion. No rayon or parallelism in default features (gate behind
  `parallel`).
- nalgebra and ndarray are wasm-fine in pure-Rust configs; pick those when both
  exist. LAPACK/BLAS acceleration of either is opt-in and off by default:
  `ndarray-blas` (forwards `ndarray/blas`) and `nalgebra-lapack` (swaps the
  nalgebra backend's Cholesky and symmetric eigendecomposition for LAPACK-backed
  ones). Both link a Fortran/BLAS toolchain, so neither builds for wasm and
  neither is in the wasm CI matrix.
- If a solver can't realistically run on wasm, document that in a per-solver
  compat note rather than weakening the guarantee.

## MSRV is externally constrained: do not bump casually

Basin's MSRV (pinned in `rust-toolchain.toml`) is set by downstream consumers,
not basin's own preferences:

- **Primary: CRAN.** A planned R-package wrapper must build under CRAN's Rust
  toolchain, which lags stable significantly. Bumping above CRAN's pin makes the
  R bindings unshippable. Don't bump `rust-version`/the `devenv.nix` pin without
  checking the current CRAN toolchain first.
- **Secondary (non-binding): Python bindings**: PyO3 and maturin track recent
  stable, so unlikely to bind tighter than CRAN.
- Every new dep (and dev-dep, which is exercised by `cargo publish --dry-run`
  and CI) must compile under the MSRV. Prefer small, stable transitive trees
  over feature-rich ones with sprawling graphs. When MSRV pain forces a pin,
  document the *reason* in `Cargo.toml` next to it so future-you doesn't lift it
  without re-checking CRAN.

## Repo structure: workspace

The workspace manifest is at the repo root (shared lockfile) with three members:

- `crates/basin`: the library.
- `crates/basin-wasm`: `wasm-bindgen` JS bindings consumed by the
  Svelte and Tailwind visualizer in `web/` (deployed to GitHub Pages). `web/` is its
  own node project, **not** a Cargo workspace member.
- `crates/competitor-bench`: benchmarks against competing libraries.

Keep optional integrations as Cargo features on `basin` itself (`nalgebra`,
`ndarray`, `faer`, `parallel`, `problems`), not new crates. Add a workspace
member only on a concrete trigger: heavy or platform-specific deps that have no
business in the core crate:

- An observer with heavy deps (TUI, slog) -> `basin-observer-foo`.
- Test problems other crates want to depend on independently →
  `basin-testfunctions`.
- Python bindings -> `basin-py`.

If the only reason is "feels tidy", keep it in `basin` behind a feature.
