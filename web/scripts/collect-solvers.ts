/**
 * Collect the solver-axis convergence traces into a committed JSON file
 * the `/benchmarks/solvers` page imports.
 *
 * Reads `target/solver-traces.json` (basin's five general solvers on
 * Rosenbrock n=2 from six curated starts, each capped on a 20 ms wall-clock
 * budget), filters and orders the rows, wraps them with run metadata, and
 * writes `web/src/lib/data/solver-benchmarks.json`.
 *
 * Run with: `npm run collect:solvers` (uses tsx). Produce the input first:
 *   cargo run -p competitor-bench --release --bin solver_compare > target/solver-traces.json
 *
 * As with the backend/competitor collectors this is deliberately off CI—
 * timings are machine-specific. Refresh locally and commit the regenerated
 * JSON.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { cpus } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = fileURLToPath(new URL(".", import.meta.url));
const repoRoot = resolve(scriptDir, "..", "..");
const tracesFile = join(repoRoot, "target", "solver-traces.json");
const outFile = resolve(
    scriptDir,
    "..",
    "src",
    "lib",
    "data",
    "solver-benchmarks.json",
);

const SOLVER_ORDER = ["gd", "nm", "bfgs", "lbfgs", "cmaes"] as const;
type Solver = (typeof SOLVER_ORDER)[number];

type TracePoint = { tNs: number; subopt: number };
type SolverResult = {
    solver: Solver;
    problem: string;
    n: number;
    start: [number, number];
    startLabel: string;
    budgetNs: number;
    points: TracePoint[];
};

if (!existsSync(tracesFile)) {
    console.error(`✗ no trace output at ${tracesFile}`);
    console.error(
        "  run the harness first:\n" +
            "  cargo run -p competitor-bench --release --bin solver_compare > target/solver-traces.json",
    );
    process.exit(1);
}

const raw = JSON.parse(readFileSync(tracesFile, "utf8")) as SolverResult[];

// Keep only known solvers (robust to stale trace files from a future lineup).
const results = raw.filter((r) => SOLVER_ORDER.includes(r.solver));

if (results.length === 0) {
    console.error(
        "✗ trace file had no rows matching the curated solver set—is it stale?",
    );
    process.exit(1);
}

// Order by (start index, solver) so the JSON diff stays stable.
const startKey = (s: [number, number]) => `${s[0]},${s[1]}`;
const startOrder: string[] = [];
for (const r of raw) {
    const k = startKey(r.start);
    if (!startOrder.includes(k)) startOrder.push(k);
}
results.sort(
    (a, b) =>
        startOrder.indexOf(startKey(a.start)) -
            startOrder.indexOf(startKey(b.start)) ||
        SOLVER_ORDER.indexOf(a.solver) - SOLVER_ORDER.indexOf(b.solver),
);

const budgetNs = results[0]?.budgetNs ?? 0;
const data = {
    generatedAt: new Date().toISOString().slice(0, 10),
    env: {
        os: process.platform,
        arch: process.arch,
        cpu: cpus()[0]?.model.trim() ?? "unknown",
    },
    budgetNs,
    results,
};

mkdirSync(dirname(outFile), { recursive: true });
writeFileSync(outFile, `${JSON.stringify(data, null, 2)}\n`);

const expectedStarts = startOrder.length;
const expected = expectedStarts * SOLVER_ORDER.length;
console.log(`✓ wrote ${results.length} result(s) to ${outFile}`);
if (results.length !== expected) {
    console.warn(
        `  note: expected ${expected} rows (${expectedStarts} starts × ${SOLVER_ORDER.length} solvers); ` +
            "is the harness run complete?",
    );
}
