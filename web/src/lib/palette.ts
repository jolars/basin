/**
 * Per-theme colors for canvas-rendered overlays. The Tailwind classes
 * on surrounding HTML handle DOM elements; canvas drawing has to pick
 * its own RGB strings, so this module is the single source of truth.
 *
 * Keep palette keys aligned across themes so callers can index by name
 * without conditionals.
 */
export type Theme = "light" | "dark";

export type Palette = {
    /** Canvas background fill. */
    surface: string;
    /** Axis lines and ticks (cost chart). */
    axis: string;
    /** Tick labels and headline numerals. */
    text: string;
    /** Trajectory polyline + iterate dots (contour plot). */
    trajectory: string;
    /** Outline of the start-point marker. */
    startMarker: string;
    /** Cross marking the known global minimum. */
    minimum: string;
    /** Cost-vs-iter polyline. */
    cost: string;
    /** Termination-reason text in the cost chart. */
    reason: string;
    /**
     * Map a 0..1 contour rank `t` to a stroke color. `t = 0` is the
     * outermost (highest cost) contour; `t = 1` is the innermost (lowest
     * cost). The function lets the dark and light palettes pick
     * contrasting ramps without the renderer caring how.
     */
    contour: (t: number) => string;
};

const VIRIDIS_STOPS: [number, number, number][] = [
    [68, 1, 84],
    [72, 35, 116],
    [64, 67, 135],
    [52, 94, 141],
    [41, 120, 142],
    [32, 144, 140],
    [34, 167, 132],
    [68, 190, 112],
    [253, 231, 36],
];

function viridisAt(t: number): [number, number, number] {
    const x = Math.max(0, Math.min(1, t));
    const f = x * (VIRIDIS_STOPS.length - 1);
    const i = Math.floor(f);
    const u = f - i;
    const a = VIRIDIS_STOPS[i];
    const b = VIRIDIS_STOPS[Math.min(i + 1, VIRIDIS_STOPS.length - 1)];
    return [
        Math.round(a[0] + (b[0] - a[0]) * u),
        Math.round(a[1] + (b[1] - a[1]) * u),
        Math.round(a[2] + (b[2] - a[2]) * u),
    ];
}

export function paletteFor(theme: Theme): Palette {
    if (theme === "dark") {
        return {
            surface: "rgb(15, 23, 42)",
            axis: "rgba(148, 163, 184, 0.4)",
            text: "rgba(203, 213, 225, 0.85)",
            trajectory: "rgba(255, 255, 255, 0.95)",
            startMarker: "rgb(255, 255, 255)",
            minimum: "rgb(248, 113, 113)",
            cost: "rgb(56, 189, 248)",
            reason: "rgba(250, 204, 21, 0.95)",
            // On dark: brightest viridis (yellow) for inner, darkest
            // (purple) for outer.
            contour: (t) => {
                const [r, g, b] = viridisAt(t);
                return `rgba(${r}, ${g}, ${b}, 0.85)`;
            },
        };
    }
    return {
        surface: "rgb(248, 250, 252)",
        axis: "rgba(71, 85, 105, 0.5)",
        text: "rgba(30, 41, 59, 0.85)",
        trajectory: "rgba(15, 23, 42, 0.95)",
        startMarker: "rgb(15, 23, 42)",
        minimum: "rgb(220, 38, 38)",
        cost: "rgb(2, 132, 199)",
        reason: "rgba(202, 138, 4, 0.95)",
        // On light: invert the viridis ramp so the inner (low-cost)
        // contour is the *darkest* purple-blue and the outer is a
        // muted teal/yellow. Both ends still have ample contrast on a
        // near-white background.
        contour: (t) => {
            const [r, g, b] = viridisAt(1 - t);
            return `rgba(${r}, ${g}, ${b}, 0.9)`;
        },
    };
}
