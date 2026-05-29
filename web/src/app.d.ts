// Type declarations for `~icons/<set>/<name>` virtual modules created by
// unplugin-icons (registered in vite.config.ts). Without this reference,
// svelte-check / tsc can't resolve the imports.
/// <reference types="unplugin-icons/types/svelte" />

// See https://svelte.dev/docs/kit/types#app.d.ts
declare global {
    namespace App {
        // interface Error {}
        // interface Locals {}
        // interface PageData {}
        // interface PageState {}
        // interface Platform {}
    }

    // Injected at build time by vite.config.ts from crates/basin/Cargo.toml.
    const __BASIN_VERSION__: string;
}

export {};
