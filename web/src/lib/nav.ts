// Shared navigation model. `href` is an app path without a leading slash.
// Calling `resolve()` prefixes the configured base, so links also work in
// deployments under a URL prefix. `section` is the first path segment, used
// for active-state matching across a whole section (e.g. any `/docs/*` page
// lights up the "Docs" link).

import type { Path } from "$app/types";

export type NavLink = {
    label: string;
    href: Path;
    /** First path segment for active-state matching. Omit for external links. */
    section?: string;
    external?: boolean;
};

// `href`s carry a trailing slash to match `trailingSlash: 'always'`
// (set in the root `+layout.ts`), so links hit the canonical URL with no
// redirect hop.

/** Top-level site navigation, shown in the header. */
export const NAV_LINKS: NavLink[] = [
    { label: "Docs", href: "docs/getting-started/", section: "docs" },
    { label: "Visualizer", href: "visualizer/", section: "visualizer" },
    { label: "Benchmarks", href: "benchmarks/", section: "benchmarks" },
];

/** Sidebar links for the docs section. */
export const DOCS_LINKS: NavLink[] = [
    { label: "Overview", href: "docs/", section: "docs" },
    {
        label: "Getting started",
        href: "docs/getting-started/",
        section: "docs",
    },
    {
        label: "Migrating from Argmin",
        href: "docs/migrating-from-argmin/",
        section: "docs",
    },
    {
        label: "Migrating convergence settings",
        href: "docs/migrating-convergence/",
        section: "docs",
    },
    { label: "Solvers", href: "docs/solvers/", section: "docs" },
];

/**
 * The active section for a route. Returns the first path segment
 * (`'docs'`, `'visualizer'`, …) or `''` for the landing page.
 *
 * Derived from `page.route.id`, which is already base-independent
 * (`/docs/solvers`, never `/basin/...`), so there's no `base` prefix to
 * strip, unlike `page.url.pathname`.
 */
export function activeSection(routeId: string | null): string {
    return (routeId ?? "").split("/")[1] ?? "";
}
