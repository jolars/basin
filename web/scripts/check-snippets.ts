/**
 * Compile-check for public website snippets.
 *
 * The landing-page programs come from their code generator. Migration-guide
 * programs are extracted from specially marked Rust fences in the rendered
 * source, so CI checks the exact code readers see. Backend examples run in
 * isolated Cargo packages: feature unification must not let a newer backend
 * implementation hide a broken exact-version feature.
 *
 * Run with: `pnpm run check:snippets` (uses tsx).
 *
 * Needs a Rust toolchain on PATH. Set `KEEP_SNIPPETS=1` to leave the
 * temporary packages on disk for inspection.
 */
import { execFileSync } from "node:child_process";
import {
    mkdirSync,
    mkdtempSync,
    readFileSync,
    rmSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
    enumerateConfigs,
    generateSnippet,
} from "../src/lib/playground/codegen.ts";

type Snippet = { name: string; source: string };

type PackageCheck = {
    name: string;
    dependencies: string;
    snippets: Snippet[];
    run: boolean;
};

const scriptDir = fileURLToPath(new URL(".", import.meta.url));
const webDir = resolve(scriptDir, "..");
const basinManifestDir = resolve(webDir, "..", "crates", "basin");
const migrationPage = resolve(
    webDir,
    "src",
    "routes",
    "docs",
    "migrating-from-argmin",
    "+page.svx",
);
const keep = process.env.KEEP_SNIPPETS === "1";

function extractMigrationSnippets(): Map<string, string> {
    const page = readFileSync(migrationPage, "utf8");
    const pattern =
        /<!--\s*compile:\s*([a-z0-9-]+)\s*-->\s*```rust\s*\n([\s\S]*?)\n```/g;
    const snippets = new Map<string, string>();

    for (const match of page.matchAll(pattern)) {
        const [, name, source] = match;
        if (snippets.has(name)) {
            throw new Error(`duplicate migration snippet marker: ${name}`);
        }
        snippets.set(name, `${source}\n`);
    }

    return snippets;
}

const migrationSnippets = extractMigrationSnippets();

function migrationSnippet(name: string): Snippet {
    const source = migrationSnippets.get(name);
    if (source === undefined) {
        throw new Error(`missing migration snippet marker: ${name}`);
    }
    return { name, source };
}

function basinDependency(feature?: string): string {
    const path = JSON.stringify(basinManifestDir);
    return feature === undefined
        ? `basin = { path = ${path} }`
        : `basin = { path = ${path}, features = [${JSON.stringify(feature)}] }`;
}

const checks: PackageCheck[] = [
    {
        name: "playground",
        dependencies: basinDependency(),
        snippets: enumerateConfigs().map(({ name, config }) => ({
            name,
            source: generateSnippet(config),
        })),
        run: false,
    },
    {
        name: "migration-argmin",
        dependencies: `argmin = "0.11"\nargmin-math = { version = "0.5", features = ["vec"] }`,
        snippets: [migrationSnippet("argmin-nelder-mead")],
        run: true,
    },
    {
        name: "migration-basin",
        dependencies: basinDependency(),
        snippets: [
            migrationSnippet("basin-nelder-mead"),
            migrationSnippet("basin-lbfgs"),
            migrationSnippet("basin-gauss-newton"),
            migrationSnippet("basin-typed-error"),
            migrationSnippet("basin-finite-diff"),
            migrationSnippet("basin-box-constraints"),
            migrationSnippet("basin-observer-cancellation"),
            migrationSnippet("basin-backend-vec"),
        ],
        run: true,
    },
];

for (const version of ["0_32", "0_33", "0_34", "0_35"]) {
    const dottedVersion = version.replace("_", ".");
    checks.push({
        name: `migration-nalgebra-${version}`,
        dependencies: `${basinDependency(`nalgebra_v${version}`)}\nnalgebra = "${dottedVersion}"`,
        snippets: [migrationSnippet("basin-backend-nalgebra")],
        run: true,
    });
}

for (const version of ["0_15", "0_16", "0_17"]) {
    const dottedVersion = version.replace("_", ".");
    checks.push({
        name: `migration-ndarray-${version}`,
        dependencies: `${basinDependency(`ndarray_v${version}`)}\nndarray = "${dottedVersion}"`,
        snippets: [migrationSnippet("basin-backend-ndarray")],
        run: true,
    });
}

for (const version of ["0_22", "0_23", "0_24"]) {
    const dottedVersion = version.replace("_", ".");
    checks.push({
        name: `migration-faer-${version}`,
        dependencies: `${basinDependency(`faer_v${version}`)}\nfaer = { version = "${dottedVersion}", default-features = false, features = ["std", "linalg"] }`,
        snippets: [migrationSnippet("basin-backend-faer")],
        run: true,
    });
}

const checkedMigrationNames = new Set(
    checks
        .flatMap((check) => check.snippets)
        .map((snippet) => snippet.name)
        .filter((name) => migrationSnippets.has(name)),
);
const uncheckedMigrationNames = [...migrationSnippets.keys()].filter(
    (name) => !checkedMigrationNames.has(name),
);
if (uncheckedMigrationNames.length > 0) {
    throw new Error(
        `migration snippets lack a compile check: ${uncheckedMigrationNames.join(", ")}`,
    );
}

const checkRoot = mkdtempSync(join(tmpdir(), "basin-snippet-check-"));
const sharedTargetDir = join(checkRoot, "target");
let failed = false;

console.log(
    `Checking ${checks.reduce((total, check) => total + check.snippets.length, 0)} snippet build(s) across ${checks.length} isolated package(s)…`,
);

try {
    for (const check of checks) {
        const crateDir = join(checkRoot, check.name);
        const binDir = join(crateDir, "src", "bin");
        mkdirSync(binDir, { recursive: true });

        const cargoToml = `[package]
name = ${JSON.stringify(check.name)}
version = "0.0.0"
edition = "2021"
publish = false

[dependencies]
${check.dependencies}

[workspace]
`;
        writeFileSync(join(crateDir, "Cargo.toml"), cargoToml);

        for (const snippet of check.snippets) {
            writeFileSync(join(binDir, `${snippet.name}.rs`), snippet.source);
        }

        console.log(`  ${check.name}`);
        execFileSync("cargo", ["build", "--bins", "--quiet"], {
            cwd: crateDir,
            env: { ...process.env, CARGO_TARGET_DIR: sharedTargetDir },
            stdio: "inherit",
        });

        if (check.run) {
            for (const snippet of check.snippets) {
                execFileSync(
                    "cargo",
                    ["run", "--bin", snippet.name, "--quiet"],
                    {
                        cwd: crateDir,
                        env: {
                            ...process.env,
                            CARGO_TARGET_DIR: sharedTargetDir,
                        },
                        stdio: "inherit",
                    },
                );
            }
        }
    }

    console.log("\n✓ all public website snippets compiled and ran as required");
} catch {
    failed = true;
    console.error("\n✗ a public website snippet failed (see above)");
    if (keep)
        console.error(`  inspect the generated packages at: ${checkRoot}`);
} finally {
    if (!keep) rmSync(checkRoot, { recursive: true, force: true });
}

process.exit(failed ? 1 : 0);
