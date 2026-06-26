---
name: ingest-paper
description: Ingest a research paper PDF into references/<name>/ for use as source material when implementing a basin solver. Runs a fast pymupdf4llm pass first, then optionally a slow marker pass on pages where math and figure fidelity matters. Use when the user provides a paper to translate into a solver implementation.
---

# Paper ingestion pipeline

> **Status: untested in production.** This pipeline has been exercised exactly once
> (van-der-zander 2020, a non-optimization paper picked specifically for its mix of
> pseudocode, math, and diagrams). It needs trial-by-fire on real solver papers.
> When you use it, **note what didn't work in `references/<name>/NOTES.md`**; the
> "Optimization ideas" section at the bottom of this skill collects those for
> future iteration.

## When to use

The user provides a PDF (or URL) of a paper they want to base a solver on.
Goal: produce a stable parsed artifact that we can cross-reference while
translating the paper's algorithm into Rust.

## Two-stage pipeline

### Stage 1: fast pass (`pymupdf4llm`)

Always run this first. ~1 second per paper.

```sh
task ingest-paper PDF=<path-or-url> NAME=<slug>
```

If `PDF` is a URL, download it first into a temp path. The slug is your
choice: match the solver name you'll eventually use (`lbfgs`, `nelder-mead`,
`adam`, etc.).

Output:
- `references/<slug>/source.pdf`: the PDF
- `references/<slug>/source.md`: pymupdf4llm-parsed markdown

**What this gives you:** clean section headers, readable prose, intact
pseudocode, parseable bibliography.

**What it loses:** equations get mangled (subscripts and superscripts become bracket
noise like `aVar[ˆτyx[z][2][]]`), figures replaced with placeholders.
Hyphenation and ligatures from the source PDF mostly survive.

### Stage 2: selective marker pass (only if needed)

Read `source.md`. If the algorithm is fully captured by the pseudocode and
prose, **stop here**; most solver papers don't need stage 2.

Run stage 2 only when one of these is true on a specific page:
- A derivation matters and the equations are unreadable garbage.
- A figure (e.g. trust region geometry, line search illustration) is referenced
  in the algorithm description and needs to actually be visible.
- A table of constants or hyperparameters is critical and got dropped.

```sh
task ingest-paper-pages NAME=<slug> PAGES="<0-indexed-pages>"
```

Pages are 0-indexed (marker's convention). Examples:
- `PAGES="3"`: just page 4 of the paper (PDF page index 3)
- `PAGES="3-5"`: pages 4 through 6
- `PAGES="3,7-8"`: page 4 plus pages 8-9

Output: `references/<slug>/source.marker.md`. **Slow**, ~minutes per page on
CPU. First run also downloads ~2-3 GB of marker models.

## After ingestion: write `NOTES.md`

Create `references/<slug>/NOTES.md` with at minimum:

```md
# <Paper title>

- **Source:** <URL or citation>
- **License of any reference impl studied:** <BSD/MIT/Apache → safe to study;
  GPL/LGPL → study for understanding only, implement from paper>
- **Stage 2 pages:** <which pages got the marker pass, or "none">
- **Parser quirks:** <anything weird worth knowing for next time>
- **Algorithm sections:** <pointers like "Algorithm 2 on p. 5 is the main loop">
```

This is the bridge document between the paper artifacts and the eventual
`src/solver/<slug>.rs`. It also feeds back into the "Optimization ideas"
section below; every time the parser does something annoying, the answer
might be a tooling improvement.

## Licensing rule (load-bearing)

Reference implementations from other libraries are **read for understanding
only**. Translate from the paper, not line-by-line from the code, unless the
license is BSD/MIT/Apache-compatible with basin's MIT license.

- **GPL / LGPL** (R packages, GSL, most LAPACK/MINPACK derivatives) → no port.
  Study for understanding only.
- **BSD / MIT / Apache / MPL2** (SciPy, Ceres, Eigen) → can port with
  attribution in `NOTES.md`.

If unsure, write the algorithm from the paper's description; don't look at
the reference code while writing the Rust.

## Optimization ideas (for future-you)

The pipeline is unproven. Record concrete pain points here as you hit them
and revisit when several have accumulated:

- [ ] **Ligature normalization.** pymupdf4llm preserves `ﬁnd`, `efﬁcient`, etc.
      as Unicode ligatures. A post-processing pass (`unicodedata.normalize`)
      would clean these up. Cheap to add if it bites.
- [ ] **Auto-detect pages needing marker.** Currently a human eyeball pass.
      Possible heuristics: density of `[]` bracket sequences, count of
      `==> picture ... omitted <==` placeholders. Worth doing only if stage 2
      becomes routine.
- [ ] **GPU acceleration for marker.** Not wired into `tools/` because ROCm
      torch wheels would bloat the lockfile for every contributor. If a
      maintainer needs fast marker passes regularly, document a separate
      ROCm-enabled env in `tools/README.md`.
- [ ] **Mathpix fallback.** For papers with truly critical equation fidelity,
      paid Mathpix is faster and more accurate than marker. Worth wrapping
      behind a third recipe (`task ingest-paper-pages-mathpix`) if marker
      output is repeatedly insufficient.
- [ ] **`--use_llm` mode for marker.** Marker can call an LLM (Anthropic
      service path: `marker.services.claude.ClaudeService`) for higher-quality
      extraction. Not enabled by default; introduces nondeterminism (re-runs
      can vary), costs API credits, and creates a subtle loop where the
      artifact is shaped by an LLM before another LLM reads it. Reasonable to
      opt into per-paper if everything else fails.
- [ ] **Page-numbering offset.** Papers often have a title page or front
      matter, so "Algorithm 2 on p. 5" of the paper might be PDF page index 5
      or 6 depending on the publisher. The pymupdf4llm output doesn't make
      this offset obvious. Could paginate output with explicit PDF page
      numbers to make the mapping easier.

When adding to this list, prefer concrete observed pain over speculative
improvements. A TODO with a specific paper that hit the issue ("the L-BFGS
paper's Algorithm 7.4 came out as `Algorithm[7][.][4]`") is more actionable
than a vague "math could be better."
