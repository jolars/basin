// `@types/prismjs` types the core module but not the per-language component
// subpaths, which are side-effect imports that register a grammar onto the
// shared Prism instance. Declare the one we use so `import` type-checks.
declare module "prismjs/components/prism-rust";
