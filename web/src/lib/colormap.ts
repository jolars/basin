/**
 * Tiny viridis-ish colormap. Hardcoded to avoid pulling in d3 or colormaps;
 * the bundle is small enough to care.
 *
 * Source values are 9 evenly-spaced viridis samples; we linearly interpolate
 * between them. Plenty good for a heatmap.
 */
const STOPS: [number, number, number][] = [
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

/** Map t ∈ [0, 1] to an RGBA tuple (alpha = 255). */
export function viridis(t: number): [number, number, number, number] {
    if (!Number.isFinite(t)) return [0, 0, 0, 255];
    const x = Math.max(0, Math.min(1, t));
    const f = x * (STOPS.length - 1);
    const i = Math.floor(f);
    const u = f - i;
    const a = STOPS[i];
    const b = STOPS[Math.min(i + 1, STOPS.length - 1)];
    return [
        Math.round(a[0] + (b[0] - a[0]) * u),
        Math.round(a[1] + (b[1] - a[1]) * u),
        Math.round(a[2] + (b[2] - a[2]) * u),
        255,
    ];
}

/**
 * Squash a positive cost grid into [0, 1] for colormapping. The
 * `intensity` choice keeps high-dynamic-range surfaces (Rosenbrock,
 * Beale) legible: `log1p(c) / log1p(max)` flattens the long tail and
 * keeps the basin floor distinguishable from background.
 */
export function normalizeCosts(
    costs: Float64Array,
    intensity: "linear" | "sqrt" | "log1p",
): Float64Array {
    const out = new Float64Array(costs.length);
    let max = 0;
    for (let i = 0; i < costs.length; i++) {
        const v = transform(costs[i], intensity);
        out[i] = v;
        if (v > max) max = v;
    }
    if (max <= 0) return out;
    for (let i = 0; i < out.length; i++) out[i] /= max;
    return out;
}

function transform(c: number, intensity: "linear" | "sqrt" | "log1p") {
    if (!Number.isFinite(c) || c < 0) return 0;
    switch (intensity) {
        case "sqrt":
            return Math.sqrt(c);
        case "log1p":
            return Math.log1p(c);
        default:
            return c;
    }
}
