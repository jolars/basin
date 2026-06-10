---
description: When a solver is added, removed, renamed, or its backend support / references change, update the public catalogue at web/src/routes/docs/solvers/+page.svx — the prose list, the support matrix, and the inline docs.rs link + paper/port reference must all stay in sync with crates/basin/src/lib.rs re-exports and each solver's rustdoc # References / # Backends sections.
paths:
  - "crates/basin/src/solver/**/*.rs"
  - "web/src/routes/docs/solvers/+page.svx"
---

# Keep the solver catalogue web page in sync

`web/src/routes/docs/solvers/+page.svx` is the public solver catalogue
(deployed to GitHub Pages). It is hand-maintained and easily drifts from the
crate. Whenever you add, remove, rename, or change the backend support /
references of a solver, update this page in the same change.

## What must stay in sync

1. **The catalogue (prose list).** It mirrors the public solver re-exports in
   `crates/basin/src/lib.rs` (`pub use crate::solver::{…}`). The set of solvers
   named on the page must equal the set of public solvers — nothing more,
   nothing less. Cross-check with:
   `grep -nE "pub use crate::solver" crates/basin/src/lib.rs`.

2. **The support matrix.** Each solver has a row with a ✓/✗ per parameter type
   (`Vec<f64>`, nalgebra, ndarray, faer). The source of truth is the solver's
   rustdoc `# Backends` note — copy the supported backends from there. ✓ means
   it compiles and runs; ✗ means it is a compile error (the tiering rule). If a
   solver is backend-generic but its *effective* coverage depends on an inner
   solver, mark the broad coverage and add a footnote (see `DeInject`).

3. **The reference + docs.rs link.** Each bullet links the solver name to its
   docs.rs API page and cites the paper it implements (or the source it ports).
   Pull the citation verbatim from the solver's rustdoc `# References` section
   so the web page and the API docs agree.

## docs.rs link convention

- Solvers: `https://docs.rs/basin/latest/basin/solver/struct.<Name>.html`
- Line searches: `https://docs.rs/basin/latest/basin/line_search/struct.<Name>.html`
- `Lbfgsb` is a type alias over the `Lbfgs` struct → link both to
  `…/solver/struct.Lbfgs.html`.

**Caveat for newly added solvers:** `docs.rs/.../latest` resolves to the most
recent *published* release. A solver added since that release will 404 until the
next `cargo publish`. Using `/latest/` is still correct — the link self-heals on
release. Don't block the docs update on it; note it in the commit/PR.

## Verifying

Run the web build to catch malformed markdown/links:
`cd web && npm run build` (optionally `npm run dev` and eyeball the page).
