/**
 * Collect the competitor-axis convergence traces into a committed JSON file
 * the `/benchmarks/competitors` page imports.
 *
 * Reads the trace harness output at `target/competitor-traces.json` (basin
 * vs competing libraries as suboptimality-vs-time curves), wraps it with run metadata, and
 * writes `web/src/lib/data/competitor-benchmarks.json`.
 *
 * Run with: `npm run collect:competitors` (uses tsx). Produce the input first:
 *   cargo run -p competitor-bench --release --features lbfgsb-reference --bin trace > target/competitor-traces.json
 *
 * Like the backend collector, this is deliberately off CI: timings are
 * machine-specific and shared runners are noisy. Refresh locally and commit
 * the regenerated JSON.
 */
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { cpus } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
    COMPETITOR_CASES as CASE_ORDER,
    LIBRARY_ORDER,
    validateCompetitorResults,
} from "../src/lib/data/competitors";

const scriptDir = fileURLToPath(new URL(".", import.meta.url));
const repoRoot = resolve(scriptDir, "..", "..");
const tracesFile = join(repoRoot, "target", "competitor-traces.json");
const outFile = resolve(
    scriptDir,
    "..",
    "src",
    "lib",
    "data",
    "competitor-benchmarks.json",
);

/** Safety cap for GD, Nelder–Mead, and L-BFGS; other caps are documented per case. */
const ITERATIONS = 10_000;

const caseIndex = (solver: string, problem: string) =>
    CASE_ORDER.findIndex((c) => c.solver === solver && c.problem === problem);

if (!existsSync(tracesFile)) {
    console.error(`✗ no trace output at ${tracesFile}`);
    console.error(
        "  run the harness first:\n" +
            "  cargo run -p competitor-bench --release --features lbfgsb-reference --bin trace > target/competitor-traces.json",
    );
    process.exit(1);
}

const results = validateCompetitorResults(
    JSON.parse(readFileSync(tracesFile, "utf8")),
);

// Deterministic order (curated case → library) so the committed JSON has a
// stable diff.
results.sort(
    (a, b) =>
        caseIndex(a.solver, a.problem) - caseIndex(b.solver, b.problem) ||
        LIBRARY_ORDER.indexOf(a.library) - LIBRARY_ORDER.indexOf(b.library) ||
        (a.seed ?? 0) - (b.seed ?? 0),
);

const packages = readFileSync(join(repoRoot, "Cargo.lock"), "utf8").split(
    "[[package]]",
);
const versions = Object.fromEntries(
    [
        "basin",
        "argmin",
        "gomez",
        "slsqp",
        "nlopt",
        "cobyla",
        "lbfgsb",
        "cmaes",
        "levenberg-marquardt",
        "nalgebra",
        "nalgebra-cmaes",
    ].map((name) => {
        const packageName = name === "nalgebra-cmaes" ? "nalgebra" : name;
        const entry = packages.find(
            (p) =>
                p.includes(`\nname = "${packageName}"\n`) &&
                (name !== "nalgebra" || /\nversion = "0\.34\./.test(p)) &&
                (name !== "nalgebra-cmaes" || /\nversion = "0\.33\./.test(p)),
        );
        const version = entry?.match(/\nversion = "([^"]+)"/)?.[1];
        if (!version) throw new Error(`Missing locked version: ${name}`);
        return [name, version];
    }),
);
const git = (...args: string[]) =>
    execFileSync("git", args, { cwd: repoRoot, encoding: "utf8" }).trim();

const data = {
    generatedAt: new Date().toISOString().slice(0, 10),
    env: {
        compiler: execFileSync("rustc", ["--version"], {
            encoding: "utf8",
        }).trim(),
        revision: git("rev-parse", "HEAD"),
        dirty: git("status", "--porcelain").length > 0,
        threads: 1,
        os: process.platform,
        arch: process.arch,
        cpu: cpus()[0]?.model.trim() ?? "unknown",
    },
    repetitions: 11,
    versions,
    iterations: ITERATIONS,
    results,
};

mkdirSync(dirname(outFile), { recursive: true });
writeFileSync(outFile, `${JSON.stringify(data, null, 2)}\n`);

const expected = CASE_ORDER.reduce(
    (acc, c) => acc + c.libraries.length * (c.seeds?.length ?? 1),
    0,
);
console.log(`✓ wrote ${results.length} result(s) to ${outFile}`);
if (results.length !== expected) throw new Error("Unexpected result count");
