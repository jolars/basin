/**
 * Compile-check for the landing-page playground.
 *
 * Imports the *same* code generator the page uses, enumerates every
 * snippet it can produce, drops each into a throwaway Cargo package as a
 * standalone `src/bin/<name>.rs`, and runs `cargo build` against the
 * local `basin` crate. If any generated program fails to compile, this
 * exits non-zero, so the playground can never drift from the real API.
 *
 * Run with: `npm run check:snippets` (uses tsx).
 *
 * Needs a Rust toolchain on PATH. Set `KEEP_SNIPPETS=1` to leave the
 * temporary crate on disk for inspection.
 */
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
    enumerateConfigs,
    generateSnippet,
} from "../src/lib/playground/codegen.ts";

const scriptDir = fileURLToPath(new URL(".", import.meta.url));
const basinManifestDir = resolve(scriptDir, "..", "..", "crates", "basin");

const snippets = enumerateConfigs();
const keep = process.env.KEEP_SNIPPETS === "1";

const crateDir = mkdtempSync(join(tmpdir(), "basin-snippet-check-"));
const binDir = join(crateDir, "src", "bin");
mkdirSync(binDir, { recursive: true });

// A standalone package whose only purpose is to compile the snippets. The
// path dependency points at the in-repo `basin`; it is resolved within
// basin's own workspace, while this temp package is its own workspace (it
// lives outside the repo), so it is never absorbed as a member.
const cargoToml = `[package]
name = "basin-snippet-check"
version = "0.0.0"
edition = "2021"
publish = false

[dependencies]
basin = { path = ${JSON.stringify(basinManifestDir)} }

[workspace]
`;
writeFileSync(join(crateDir, "Cargo.toml"), cargoToml);

for (const { name, config } of snippets) {
    writeFileSync(join(binDir, `${name}.rs`), generateSnippet(config));
}

console.log(
    `Compiling ${snippets.length} generated snippet(s) against basin (${basinManifestDir})…`,
);

let failed = false;
try {
    execFileSync("cargo", ["build", "--bins", "--quiet"], {
        cwd: crateDir,
        stdio: "inherit",
    });
    console.log(`\n✓ all ${snippets.length} playground snippets compiled`);
} catch {
    failed = true;
    console.error(
        "\n✗ at least one playground snippet failed to compile (see above)",
    );
    if (keep) console.error(`  inspect the generated crate at: ${crateDir}`);
} finally {
    if (!keep) rmSync(crateDir, { recursive: true, force: true });
}

process.exit(failed ? 1 : 0);
