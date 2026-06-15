# Powell-family DFO roadmap: NEWUOA → BOBYQA → LINCOA

Kickoff/context doc for implementing Powell's model-based derivative-free
solvers in basin. Read this first when starting that work, then the per-paper
`references/<slug>/NOTES.md` bridge docs.

## Goal and scope

Add Powell's least-Frobenius-norm, trust-region, model-based DFO family:

1. **NEWUOA** — unconstrained. Build first.
2. **BOBYQA** — bound-constrained. Reuses NEWUOA's model core, swaps the
   subproblem for a box-aware one (TRSBOX) and adds RESCUE.
3. **LINCOA** — linearly-constrained. Same core, projected-Krylov subproblem.

UOBYQA is context only (predecessor, full model), not a build target.

**Paper-anchored (tenet + memory):** implement from the papers, do not invent
variants or extensions. Cross-validate numerics against PRIMA.

## Source material (already ingested)

All under `references/<slug>/`: `source.pdf`, `source.md` (fast prose pass),
`source.marker.md` (high-fidelity LaTeX math, the one to read), and `NOTES.md`
(section→page map, licensing, roadmap framing).

- `references/newuoa/` — Powell 2006 NEWUOA software report. **Primary.**
- `references/frobenius-update/` — Powell 2004a, *Least Frobenius norm
  updating…* — **the derivation of the shared model/`H`-update core.** Read
  alongside NEWUOA §3–§4.
- `references/bobyqa/`, `references/lincoa/`, `references/uobyqa/` — for the
  later roadmap steps + context.

Start with `references/newuoa/NOTES.md` (it has the full section→page map and the
implementation pointers), then `references/newuoa/source.marker.md` §3–§7.

## Reusable design (decide once, before coding)

The three solvers share ~70% of their code. Factor that out so BOBYQA/LINCOA
drop in without a refactor.

**Shared `QuadraticModel` core** (the spine):
- interpolation set: `npt = 2n+1` points (configurable in `[n+2, ½(n+1)(n+2)]`).
- model storage: explicit `Γ` (the dense part of `∇²Q`) + coefficients `γⱼ` so
  `∇²Q = Γ + Σ γⱼ (xⱼ−x0)(xⱼ−x0)ᵀ`; gradient `∇Q(x0)` (NEWUOA §4, eq 4.27).
- factored inverse-KKT matrix `H = W⁻¹` stored as `Ξ`/`Υ` submatrices + the
  factorization `Ω = Σ sₖ zₖzₖᵀ` (NEWUOA §3 init, §4 update — the (4.11)
  Sherman–Morrison rank-2 update with α/β/τ/σ; full derivation in
  `frobenius-update`).
- the two-level **ρ/Δ radius schedule** + MOVE (point-to-drop) selection
  (NEWUOA §7) and origin shifts (§7 / frobenius-update §5).
- geometry-improving steps (BIGLAG/BIGDEN, NEWUOA §6).

**Swappable trust-region subproblem** (the only major per-solver difference):
```
trait TrustRegionSubproblem { fn solve(&self, model, delta, constraints) -> step }
```
- NEWUOA → TRSAPP (truncated CG, §5) — unconstrained.
- BOBYQA → TRSBOX (active-set/box, bobyqa §3).
- LINCOA → projected-Krylov/active-set (lincoa §3–§6).

This maps onto **tenet 4**: NEWUOA = unconstrained; BOBYQA = `BoxConstraints`
(already modeled in basin); LINCOA = `LinearInequalityConstraints` (already
modeled). The constrained solvers declare constraint support and supply their
subproblem strategy; the model core is identical.

## basin infrastructure to reuse

From the codebase survey (see also `.claude/rules/backends.md`,
`solver-composition.md`):

- **linalg tier** (`crates/basin/src/core/math/linalg.rs`): `MatVec`,
  `GramMatrix`, `LinearSolveSpd` (pure-Rust Cholesky on `Vec<f64>`, wasm-clean),
  `SymmetricEigen` (cyclic Jacobi, wasm-clean), `RankOneUpdate` /
  `GeneralRankOneUpdate` — the `H`-update algebra maps onto these. All four
  backends implement them.
- **State**: add a `NewuoaState<V, M, F>` (current iterate + interpolation set +
  costs + best) following the `CmaEsState` precedent
  (`core/state/cma_es.rs`). Keep the factored-`H` / `Γ` / `γⱼ` internals on the
  **solver struct**, not the state — consistent with LM's μ/ν living solver-side
  (see AGENTS.md provisional-choices).
- **Problem**: bind `CostFunction` only (no gradient) — NEWUOA's exact contract.
- **Termination**: `MaxIter`, `MaxCostEvals`, `CostTolerance` work as-is; add a
  small `RhoTolerance`-style criterion for the natural NEWUOA stop (ρ reaching
  `rhoend`), bound on the new state.
- Scalar stays `F: Scalar` with `F = f64` default across state+solver+
  termination+math (AGENTS.md).

## Build order

1. `QuadraticModel` + the least-Frobenius-norm **UPDATE** (NEWUOA §3 init, §4
   update). Unit-test against hand-worked small-`n` Frobenius examples and the
   `frobenius-update` derivation before wiring anything else.
2. **TRSAPP** (truncated-CG subproblem, §5) — standalone, unit-testable.
3. Driver loop: seed `2n+1` points, the ρ/Δ iteration (Figure 1 of NEWUOA §2),
   RATIO/accept-reject, MOVE selection, BIGLAG/BIGDEN geometry steps, origin
   shifts, the Qint robustness modification (§8).
4. Validate, then refactor the model core behind the `TrustRegionSubproblem`
   seam for BOBYQA/LINCOA.

## Validation

- **NLopt NEWUOA** is already linked in `crates/competitor-bench` (via
  `nlopt-sys` + cmake) — Powell's algorithm, C-translated. Use for quick
  live sanity during development (not bit-exact to PRIMA).
- **PRIMA fixtures** for authoritative parity: PRIMA is BSD-3 (safe to study and
  use as oracle); no Rust binding exists, and a native FFI must NOT be a `basin`
  dep (wasm/MSRV) — keep it out of the core. Generate reference outputs offline
  with PRIMA in the `tools/` Python env (build libprima, or its Python
  interface), dump final `x`/`f`/`#evals` (+ optional iterate trace) to fixtures
  under `tests/`, and assert parity in ordinary Rust tests. Zero Rust deps.
- **Test problems** (from NEWUOA §8): Rosenbrock, ARWHEAD, CHROSEN,
  PENALTY1–3, VARDIM (the one that motivated the Qint fix). Several already in
  `crates/basin/src/problems/`.

## Constraints to respect (basin tenets)

- wasm builds out of the box (no BLAS/LAPACK/threads in default deps).
- MSRV 1.87 (CRAN-pinned) — every new dep must compile under it; dev-deps too.
- Conventional vocabulary; framework-level termination; constraints problem-side.
- No invented variants — anchor to the papers.
