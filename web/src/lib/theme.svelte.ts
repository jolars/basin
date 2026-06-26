/**
 * Reactive theme store. Three preferences:
 *
 * - `auto`: follow the OS via `prefers-color-scheme`.
 * - `light`/`dark`: explicit override.
 *
 * The store exposes both the user's *preference* (what the toggle
 * should show as selected) and the *effective* theme actually applied
 * to the page. `effective` is derived from preference + the live media
 * query, so flipping the OS theme while in `auto` mode updates it
 * immediately.
 *
 * Persisted in `localStorage` under `basin.theme`. Defaults to `auto`
 * for first-time visitors.
 *
 * The module guards `window`/`localStorage` access for safety: the root
 * layout is server-rendered (prerendered), so this store is imported in
 * an SSR context where those globals are absent. The pre-hydration paint
 * is handled by an inline script in `app.html`; this store keeps the
 * `.dark`/`.light` class in sync after hydration.
 */

export type ThemePreference = "auto" | "light" | "dark";
export type EffectiveTheme = "light" | "dark";

const STORAGE_KEY = "basin.theme";

function readPreference(): ThemePreference {
    if (typeof localStorage === "undefined") return "auto";
    const v = localStorage.getItem(STORAGE_KEY);
    return v === "light" || v === "dark" || v === "auto" ? v : "auto";
}

function systemPrefersDark(): boolean {
    if (
        typeof window === "undefined" ||
        typeof window.matchMedia !== "function"
    )
        return false;
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

class ThemeStore {
    preference: ThemePreference = $state(readPreference());
    /** Live system preference; updated by a media-query listener. */
    private systemDark: boolean = $state(systemPrefersDark());

    effective: EffectiveTheme = $derived(
        this.preference === "auto"
            ? this.systemDark
                ? "dark"
                : "light"
            : this.preference,
    );

    constructor() {
        if (
            typeof window !== "undefined" &&
            typeof window.matchMedia === "function"
        ) {
            const mq = window.matchMedia("(prefers-color-scheme: dark)");
            const onChange = (e: MediaQueryListEvent) => {
                this.systemDark = e.matches;
            };
            // `addEventListener` is the modern API; old Safari needs
            // `addListener`. We support modern browsers only.
            mq.addEventListener("change", onChange);
        }
    }

    set(pref: ThemePreference) {
        this.preference = pref;
        if (typeof localStorage !== "undefined") {
            try {
                localStorage.setItem(STORAGE_KEY, pref);
            } catch {
                // Private mode or quota: silent fallthrough.
            }
        }
    }
}

export const theme = new ThemeStore();
