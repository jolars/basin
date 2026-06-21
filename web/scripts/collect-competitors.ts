/**
 * Collect the competitor-axis convergence traces into a committed JSON file
 * the `/benchmarks/competitors` page imports.
 *
 * Reads the trace harness output at `target/competitor-traces.json` (basin
 * vs argmin suboptimality-vs-time curves), wraps it with run metadata, and
 * writes `web/src/lib/data/competitor-benchmarks.json`.
 *
 * Run with: `npm run collect:competitors` (uses tsx). Produce the input first:
 *   cargo run -p competitor-bench --release --bin trace > target/competitor-traces.json
 *
 * Like the backend collector, this is deliberately off CI — timings are
 * machine-specific and shared runners are noisy. Refresh locally and commit
 * the regenerated JSON.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { cpus } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = fileURLToPath(new URL('.', import.meta.url));
const repoRoot = resolve(scriptDir, '..', '..');
const tracesFile = join(repoRoot, 'target', 'competitor-traces.json');
const outFile = resolve(scriptDir, '..', 'src', 'lib', 'data', 'competitor-benchmarks.json');

/** Iteration budget the harness caps each solve at (`MAX_ITERS`). */
const ITERATIONS = 200;

const LIBRARY_ORDER = ['basin', 'argmin', 'gomez', 'nlopt'] as const;
type Library = (typeof LIBRARY_ORDER)[number];

/**
 * Curated (solver, problem, libraries) cases, in page order. Drives the
 * deterministic sort and the expected-count sanity check. Not every case has
 * every library — argmin has no NLLS so it's absent from those; gomez only
 * lines up with the derivative-free NM case; nlopt joins NM, L-BFGS, and the
 * NEWUOA case (where it and basin are the same algorithm) but has no GD entry
 * (its `Lbfgs` is the closest first-order analog). The NEWUOA case is the only
 * one off Rosenbrock — Styblinski–Tang at n = 5. Keep in sync with
 * `COMPETITOR_CASES` in `src/lib/data/competitors.ts` and the cases in the
 * `trace` bench (`crates/competitor-bench/src/bin/trace.rs`).
 */
const CASE_ORDER: { solver: string; problem: string; libraries: Library[] }[] = [
    { solver: 'gd', problem: 'rosenbrock', libraries: ['basin', 'argmin'] },
    { solver: 'nm', problem: 'rosenbrock', libraries: ['basin', 'argmin', 'gomez', 'nlopt'] },
    { solver: 'lbfgs', problem: 'rosenbrock', libraries: ['basin', 'argmin', 'nlopt'] },
    { solver: 'newuoa', problem: 'styblinski', libraries: ['basin', 'nlopt'] },
];

const caseIndex = (solver: string, problem: string) =>
    CASE_ORDER.findIndex((c) => c.solver === solver && c.problem === problem);

const caseAllowsLibrary = (solver: string, problem: string, lib: Library) =>
    CASE_ORDER.some(
        (c) => c.solver === solver && c.problem === problem && c.libraries.includes(lib),
    );

type TracePoint = { tNs: number; subopt: number };
type CompetitorResult = {
    solver: string;
    problem: string;
    n: number;
    library: Library;
    points: TracePoint[];
};

if (!existsSync(tracesFile)) {
    console.error(`✗ no trace output at ${tracesFile}`);
    console.error(
        '  run the harness first:\n' +
            '  cargo run -p competitor-bench --release --bin trace > target/competitor-traces.json',
    );
    process.exit(1);
}

const raw = JSON.parse(readFileSync(tracesFile, 'utf8')) as CompetitorResult[];

// Keep only curated (solver, problem, library) rows — robust to stale traces
// from an earlier case layout.
const results = raw.filter(
    (r) =>
        caseIndex(r.solver, r.problem) >= 0 &&
        LIBRARY_ORDER.includes(r.library) &&
        caseAllowsLibrary(r.solver, r.problem, r.library),
);

if (results.length === 0) {
    console.error('✗ trace file had no curated (solver, problem, library) rows — is it stale?');
    process.exit(1);
}

// Deterministic order (curated case → library) so the committed JSON has a
// stable diff.
results.sort(
    (a, b) =>
        caseIndex(a.solver, a.problem) - caseIndex(b.solver, b.problem) ||
        LIBRARY_ORDER.indexOf(a.library) - LIBRARY_ORDER.indexOf(b.library),
);

const data = {
    generatedAt: new Date().toISOString().slice(0, 10),
    env: {
        os: process.platform,
        arch: process.arch,
        cpu: cpus()[0]?.model.trim() ?? 'unknown',
    },
    iterations: ITERATIONS,
    results,
};

mkdirSync(dirname(outFile), { recursive: true });
writeFileSync(outFile, `${JSON.stringify(data, null, 2)}\n`);

const expected = CASE_ORDER.reduce((acc, c) => acc + c.libraries.length, 0);
console.log(`✓ wrote ${results.length} result(s) to ${outFile}`);
if (results.length !== expected) {
    console.warn(
        `  note: expected ${expected} rows (sum of curated libraries per case); ` +
            'is the harness run complete?',
    );
}
