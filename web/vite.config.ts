import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { sveltekit } from "@sveltejs/kit/vite";
import tailwindcss from "@tailwindcss/vite";
import Icons from "unplugin-icons/vite";
import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";

// Read the basin crate version from Cargo.toml at config time so the
// site footer can show what version the docs/playground are pinned to.
const cargoToml = readFileSync(
    fileURLToPath(new URL("../crates/basin/Cargo.toml", import.meta.url)),
    "utf8",
);
const versionMatch = cargoToml.match(
    /\[package\][\s\S]*?\nversion\s*=\s*"([^"]+)"/,
);
if (!versionMatch) {
    throw new Error(
        "could not parse basin version from crates/basin/Cargo.toml",
    );
}
const basinVersion = versionMatch[1];

export default defineConfig({
    define: {
        __BASIN_VERSION__: JSON.stringify(basinVersion),
    },
    // `~icons/<set>/<name>` imports are resolved at build time by
    // unplugin-icons and compiled to Svelte components, so only the icons
    // actually imported are bundled (no runtime icon library). Icon data
    // comes from `@iconify-json/lucide`.
    plugins: [
        tailwindcss(),
        wasm(),
        sveltekit(),
        Icons({ compiler: "svelte" }),
    ],
    // The wasm-pack output uses ESM `import.meta.url` to find the .wasm
    // sibling. Marking the package as not-pre-bundled lets vite serve it
    // as-is in dev and preserves that resolution.
    optimizeDeps: { exclude: ["$lib/basin-wasm"] },
});
