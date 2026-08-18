# AGENTS.md

This file is the self-contained repository guide for AI agents.

## Project and commands

Basin is a semver-stable Rust numerical-optimization library. It has a generic
`Executor`/`Solver`/`State` core, pluggable termination, problem traits, and
first-order, quasi-Newton, derivative-free, nonlinear least-squares, global,
stochastic, and constrained solvers. Solvers are generic over `Vec<f64>`,
nalgebra, ndarray, and faer backends.

- `cargo build`: build the library.
- `cargo test` or `cargo test <name>`: run tests.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: lint.
- `cargo doc --no-deps -p basin --all-features`: verify docs, including denied
  broken intra-doc links. Link an unambiguous item when module/function names
  collide.
- `cargo fmt`: format.
- Routine pure-Rust feature tests:
  `cargo test -p basin --features nalgebra,ndarray,faer,problems,parallel`.

Do not assume `cargo test --all-features` links without a BLAS/LAPACK provider.
The `nalgebra-lapack` and `ndarray-blas` features intentionally select no
provider. Supply one through linker flags when those tests are required; the
routine CI test set is pure Rust. Clippy and rustdoc with all features only
check and do not link.

The dev environment pins Rust 1.87.0 and supplies the WASM target and project
tooling. Pre-commit runs all-feature clippy and rustfmt.

## Architecture and repository shape

- `crates/basin/src/lib.rs` contains public re-exports only.
- `core/problem.rs` defines user-implemented cost, gradient, residual, Jacobian,
  and Hessian traits; `numdiff.rs` provides finite differences.
- `core/state.rs` and `state/` define `State`, concrete state types, and the
  minimum-shape extension traits used by termination criteria.
- `core/solver.rs` defines `Solver`; `core/executor.rs` owns the driver loop and
  returns `OptimizationResult`; `core/termination.rs` owns shared criteria.
- Constraint markers and adapters live in `constraint.rs`, `barrier.rs`, and
  `augmented_lagrangian.rs`; composition contracts live in `inner.rs`.
- `core/math.rs` and `math/` implement the tiered backend layer.
- `solver.rs` and `solver/` contain concrete solvers; `rng.rs` supports
  stochastic algorithms.

Use `src/foo.rs` plus `src/foo/bar.rs`; never introduce `mod.rs`.

The root workspace contains `crates/basin`, `crates/basin-wasm`, and
`crates/competitor-bench`. `web/` is a separate Svelte project. Keep optional
backend/parallel/problem integrations as features on `basin`. Add a workspace
crate only when heavy or platform-specific dependencies have no place in the
core crate, not merely for tidiness.

## Design constraints

1. Preserve conventional optimization-framework vocabulary and the generic
   driver-loop shape unless another constraint requires divergence.
2. Each backend has one Cargo feature pinned to one major version. A backend
   major bump is a Basin major bump; do not add per-version feature gates.
3. Generic stopping criteria belong to the executor/shared termination layer,
   while solver-specific controls stay on the solver. Bind each criterion to the
   minimum state shape it needs.
4. Constraints describe problems: keep them problem-side, never on state or as
   executor configuration. Solvers declare supported constraint traits;
   projection, barrier, and penalty adapters are explicit opt-ins.
5. Keep the universal vector math tier small and the richer linear-algebra tier
   capability-based. Missing operations must fail at compile time.

WASM is a hard default-build constraint. Default dependencies and code paths
must support `wasm32-unknown-unknown`; gate file I/O, threads, rayon, and
BLAS/LAPACK-linked math behind non-default features. Use `web-time` rather than
`std::time::Instant` in default paths. Pure-Rust nalgebra/ndarray are allowed;
their LAPACK/BLAS acceleration is opt-in. Document a solver that cannot run on
WASM rather than weakening the guarantee.

Do not bump the Rust version casually. The MSRV is constrained primarily by a
planned CRAN wrapper and secondarily by Python bindings. Check the current CRAN
toolchain before changing `rust-version`, `rust-toolchain.toml`, or the dev pin.
Every dependency, including dev dependencies exercised during publishing and CI,
must compile on the MSRV. Prefer small stable dependency trees and document the
reason beside any MSRV-driven pin.

## Backend math (`crates/basin/src/core/math/**`)

The backend direction is for most solvers to support most parameter types:
`Vec<f64>`, nalgebra, ndarray, and faer. Missing capability must be a compile
error, not a reason to freeze coverage.

- Keep the shared vector tier small and universal. Traits such as `ScaledAdd`,
  `NormSquared`, `NormInfinity`, `Dot`, `ScaleInPlace`, `NegInPlace`,
  `VectorLen`, and component-wise operations belong here only when every backend
  implements them well. First-order and derivative-free solvers should stay
  generic over this tier.
- Put richer matrix operations in `core/math/linalg.rs`, including `MatVec`,
  `MatTransposeVec`, `GramMatrix`, `SymmetricEigen`, rank-one updates, SPD and
  least-squares solves, diagonal operations, matrix identity/construction, and
  dense matrix construction. LA-heavy solvers must bound only the subset they
  actually need.
- Add an implementation when it can be honest: pure Rust, WASM-clean, without a
  BLAS/LAPACK link or fake stub. The `Vec<f64>` cyclic-Jacobi symmetric
  eigensolver is the precedent. Pure-Rust Cholesky and linear solves for
  `DenseMatrix` are welcome when motivated.
- It is acceptable to omit operations that realistically require optimized
  kernels at scale. Document the gap; never add an operation no backend can
  implement honestly or stub one merely to satisfy a bound.
- Every solver rustdoc must include a `# Backends` note listing supported
  parameter types.

## Test-problem corpus (`crates/basin/src/problems/**`)

For a new problem, prefer the `add_test_problem` project subagent. Add one file
at `src/problems/<name>.rs`, with sections in this order:

1. Module rustdoc with formula, character, global minimum, and primary source.
2. Imports, then raw slice functions `name(&[f64]) -> f64` and
   `name_gradient(&[f64], &mut [f64])`.
3. `pub struct Name<P = Vec<f64>>(PhantomData<fn() -> P>);`, with `new` and
   `Default`.
4. `pub static NAME_SPEC: ProblemSpec` and blanket
   `impl<P> HasSpec for Name<P>`.
5. `CostFunction` and `Gradient` for `Vec<f64>`, then separate cfg-gated modules
   in nalgebra, ndarray, faer order.
6. Unit tests.

Use `PhantomData<fn() -> P>` to retain covariance and avoid unnecessary
`Send + Sync` requirements. The `Vec<f64>` default is downstream ergonomics;
inside feature-rich tests use an explicit `Name::<Vec<f64>>` when inference is
ambiguous.

Backend implementations must route through the slice primitives for `Vec<f64>`,
nalgebra (`as_slice`/`as_mut_slice`), and contiguous ndarray arrays. Faer's
`Col` may implement the math elementwise because its supported API does not
expose a suitable slice consistently. Keep imports and cfg clutter inside each
backend module.

`ProblemSpec` requirements:

- Use a canonical title-cased name and accurate fixed or N-dimensional shape.
- Be conservative about `unimodal` and `convex` when either depends on dimension
  or domain; explain qualifications in the description.
- Include at least one real reference with citation, title, venue/source, and
  DOI when available. A public URL supplements rather than replaces a citation.
  Prefer the original source; if none exists, cite a recognized popularizer.
- Keep the description to 1--3 sentences including the global minimizer and
  value. Metadata belongs in the spec; executable math does not.

Required unit tests cover the value at the global minimum, one hand-computed
nontrivial value, near-zero gradient at the minimum, central finite-difference
agreement at a nonsymmetric point with no zero coordinates (about `1e-5`
tolerance), and spec wiring/name/properties/references. Do not add integration
tests unless exercising a previously uncovered solver path.

After adding a problem, update `src/problems.rs` module/re-exports/`ALL_SPECS`
and mark the matching `TODO.md` corpus entry done. For a problem meaningful in
2D (`Fixed(2)` or `NDimensional { min <= 2 }`), also add its `ProblemKind` and
dispatch in `crates/basin-wasm/src/lib.rs`, and its metadata in
`web/src/lib/problems.ts`. Use the documented search domain, minimum, a useful
intensity (`sqrt` for mild quadratics, `log1p` for high dynamic range), and a
gradient-descent step that converges from a typical start. Skip web wiring only
for intrinsically high-dimensional problems.

Verify a new problem with:

```sh
cargo test --all-features
cargo clippy --all-features --all-targets -- -D warnings
cargo build --target wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown --no-default-features
cd web && npm run build # when the visualizer changed
```

Do not use generic backend-dispatch impls that create inference ambiguity, add
one-off `Properties` fields, create a `references/<name>/` directory, omit a
backend, or replace a symbolic gradient with finite differences.

## Web documentation (`web/src/routes/**`)

The static SvelteKit site is served at `https://basin.rs/` and signposts to the
authoritative docs.rs API rather than copying it. Its page routes are `/`,
`/docs`, `/docs/getting-started`, `/docs/solvers`, `/benchmarks` and its
`backends`, `competitors`, and `solvers` children, plus `/visualizer`.

The prerendered `sitemap.xml`, `robots.txt`, and `llms.txt` endpoints are
`+server.ts` routes because they need absolute URLs. They share `SITE_ORIGIN`.
The sitemap globs pages; `llms.txt` is hand-maintained. When a top-level section
is added, removed, or renamed, update `llms.txt`: core material under `Docs`,
API and external links under `Reference`, and skippable benchmarks/visualizer
links under `Optional`. Use absolute `${SITE_ORIGIN}/.../` URLs with trailing
slashes, and keep it a link-oriented signpost. Verify web changes with
`cd web && pnpm build`.

## Solver catalogue synchronization

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
equivalent `line_search/<module>/...`. Root-module struct links 404. Both
`Lbfgs` and its `Lbfgsb` alias link to `solver/lbfgs/struct.Lbfgs.html`. A newly
added solver may legitimately 404 under `/latest/` until the next release; keep
the link because it will become valid when published. Verify with the web build.

## Deliberate non-tenet choices

- **Scalar type defaults to `f64`, but the whole pipeline is `F: Scalar`.**
  Every state (`BasicState`, `BasicSimplexState`, `BasicPopulationState`,
  `QuasiNewtonState`, `LbfgsState`, `SolisWetsState`), every solver (gradient
  descent, BFGS, both L-BFGS modes, NLLS family, CMA-ES, Solis-Wets, barrier and
  AL, line searches), and every shipped termination criterion carries an
  `F = f64` default, so existing call sites resolve unchanged while `f32` works
  end-to-end (see `tests/f32_round_trip.rs`). The `F = f64` default is the
  ergonomic choice for the common case, not a constraint. When adding a scalar
  generic to new surface, commit to it properly across state, solver,
  termination, and math impls rather than adding a fake generic whose defaults
  only work in `f64`.

- **No observer KV/metadata channel.** `Observe` (`core/observer.rs`) passes
  only `&S`. There is no argmin-style stringly-typed key-value store for
  surfacing algorithm-specific scalars (step size, barrier μ, population
  diversity), because tenet 3 makes state shape the contract: observers bind on
  the minimum `State`, `GradientState`, `SimplexState`, or `PopulationState`
  shape, and a `HashMap<String, _>` side channel would erase that compile-time
  guarantee. It can be added later without a breaking change: give `Observe` a
  new default-bodied method that forwards to `observe_iter`, switch the
  executor's call site to it, and a concrete `Kv` type keeps the trait
  object-safe for `Box<dyn Observe>`. The motivation, if it arises, is that some
  solver-internal working state (LM μ, ν, diag) lives in the solver struct
  rather than the state, so exposing it on a richer state trait would not reach
  it.

- **No `Solver::name()` introspection.** The `Solver` trait (`core/solver.rs`)
  has `type Error` plus `init`, `next_iter`, and `terminate`, but no
  `name() -> &str` (argmin has one for logging and observer display). No shipped
  observer prints the solver name, so it would be unused surface frozen into
  1.0. Adding `fn name(&self) -> &str` with a default impl is additive and
  non-breaking even post-1.0, so add it if and when an observer that displays
  the name is wanted.
