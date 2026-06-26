/**
 * Per-problem metadata for the visualizer.
 *
 * Mirrors the corpus exposed by `basin::problems` (Sphere, Rosenbrock,
 * Beale, Booth), but only the bits the UI needs: a viewing window
 * for the contour plot, a known minimum (location + optimal value f*,
 * the latter for the suboptimality chart), and how to compress the cost
 * dynamic range when colorizing.
 *
 * Why duplicate this from Rust instead of pulling from `basin-wasm`?
 * The numbers are tiny, never change in lockstep with the algorithm,
 * and pulling them through wasm would add boilerplate for no win. If
 * the corpus grows large enough that this drifts, we can revisit.
 */
import { ProblemKind } from "./basin-wasm/basin_wasm";

/**
 * Target suboptimality `f − f*` for "converged". Doubles as the fixed
 * bottom of the cost chart's log y-axis, so the trajectory visibly stops
 * exactly when it reaches the floor. A single global value works because
 * suboptimality is already a normalized gap to the optimum; make it a
 * per-problem field on `ProblemMeta` if a problem ever needs a different
 * threshold.
 */
export const SUBOPT_TARGET = 1e-10;

export type Domain = { xmin: number; xmax: number; ymin: number; ymax: number };

export type ProblemMeta = {
    kind: ProblemKind;
    label: string;
    /** Recommended viewing window for the contour plot. */
    domain: Domain;
    /** Known global minimum location, for the marker. */
    minimum: { x: number; y: number };
    /**
     * Known global minimum value f*, subtracted from the cost to plot
     * suboptimality `f − f*` on the cost chart's (log) y-axis.
     */
    fStar: number;
    /**
     * Heatmap intensity transform. Rosenbrock and Beale have huge dynamic
     * range; a log-ish squash keeps the basin visible. Sphere and Booth
     * are mild quadratics where a square-root squash already looks fine.
     */
    intensity: "linear" | "sqrt" | "log1p";
    /** Sensible default constant step size for gradient descent. */
    gdAlphaDefault: number;
};

export const PROBLEMS: ProblemMeta[] = [
    {
        kind: ProblemKind.Sphere,
        label: "Sphere",
        domain: { xmin: -3, xmax: 3, ymin: -3, ymax: 3 },
        minimum: { x: 0, y: 0 },
        fStar: 0,
        intensity: "sqrt",
        gdAlphaDefault: 0.2,
    },
    {
        kind: ProblemKind.Rosenbrock,
        label: "Rosenbrock",
        domain: { xmin: -2, xmax: 2, ymin: -1, ymax: 3 },
        minimum: { x: 1, y: 1 },
        fStar: 0,
        intensity: "log1p",
        gdAlphaDefault: 0.001,
    },
    {
        kind: ProblemKind.Beale,
        label: "Beale",
        domain: { xmin: -4.5, xmax: 4.5, ymin: -4.5, ymax: 4.5 },
        minimum: { x: 3, y: 0.5 },
        fStar: 0,
        intensity: "log1p",
        gdAlphaDefault: 0.001,
    },
    {
        kind: ProblemKind.Booth,
        label: "Booth",
        domain: { xmin: -10, xmax: 10, ymin: -10, ymax: 10 },
        minimum: { x: 1, y: 3 },
        fStar: 0,
        intensity: "sqrt",
        gdAlphaDefault: 0.02,
    },
    {
        kind: ProblemKind.Matyas,
        label: "Matyas",
        domain: { xmin: -10, xmax: 10, ymin: -10, ymax: 10 },
        minimum: { x: 0, y: 0 },
        fStar: 0,
        intensity: "sqrt",
        gdAlphaDefault: 1.0,
    },
    {
        kind: ProblemKind.McCormick,
        label: "McCormick",
        domain: { xmin: -1.5, xmax: 4, ymin: -3, ymax: 4 },
        minimum: { x: -0.54719, y: -1.54719 },
        fStar: -1.9132229,
        intensity: "sqrt",
        gdAlphaDefault: 0.1,
    },
    {
        kind: ProblemKind.GoldsteinPrice,
        label: "Goldstein-Price",
        domain: { xmin: -2, xmax: 2, ymin: -2, ymax: 2 },
        minimum: { x: 0, y: -1 },
        fStar: 3,
        intensity: "log1p",
        gdAlphaDefault: 1e-5,
    },
    {
        kind: ProblemKind.Rastrigin,
        label: "Rastrigin",
        domain: { xmin: -5.12, xmax: 5.12, ymin: -5.12, ymax: 5.12 },
        minimum: { x: 0, y: 0 },
        fStar: 0,
        // Cosine ripple sits on top of a paraboloid; sqrt keeps the lattice
        // of local minima visible without flattening the basin floor.
        intensity: "sqrt",
        // GD on Rastrigin is "wrong tool for the job"—it stalls in the
        // nearest local well. Kept here so the slider has a sensible default
        // when users explore the (poor) local-search experience.
        gdAlphaDefault: 0.01,
    },
    {
        kind: ProblemKind.Ackley,
        label: "Ackley",
        domain: { xmin: -5, xmax: 5, ymin: -5, ymax: 5 },
        minimum: { x: 0, y: 0 },
        fStar: 0,
        // The cone+ripple already has bounded dynamic range; log/sqrt would
        // wash out the central funnel.
        intensity: "linear",
        gdAlphaDefault: 0.01,
    },
    {
        kind: ProblemKind.StyblinskiTang,
        label: "Styblinski-Tang",
        domain: { xmin: -5, xmax: 5, ymin: -5, ymax: 5 },
        minimum: { x: -2.903534, y: -2.903534 },
        // f* = -39.16599 · n at n = 2.
        fStar: -78.33233,
        intensity: "linear",
        gdAlphaDefault: 0.01,
    },
];

export function problemByKind(kind: ProblemKind): ProblemMeta {
    const m = PROBLEMS.find((p) => p.kind === kind);
    if (!m) throw new Error(`unknown problem kind: ${kind}`);
    return m;
}
