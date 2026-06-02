import { SolverKind } from './basin-wasm/basin_wasm';

/**
 * A single solver-specific control, rendered generically by `Controls`.
 *
 * The `id` is the key the value is stored under (in the visualizer's
 * `optionValues` record) and the field name passed across the wasm
 * boundary inside the `Run` options object — keep these in sync with
 * `RunOptions` in `crates/basin-wasm/src/lib.rs` (camelCase).
 */
export type SolverOption =
    | {
          id: string;
          kind: 'select';
          label: string;
          choices: { value: string; label: string }[];
          default: string;
      }
    | {
          id: string;
          kind: 'logSlider';
          label: string;
          /** Slider bounds are in log10 space (the stored value is 10^slider). */
          min: number;
          max: number;
          step: number;
          default: number;
          /** Only show this control when another option currently equals a value. */
          showIf?: { id: string; equals: string };
      }
    | {
          id: string;
          kind: 'intSlider';
          label: string;
          min: number;
          max: number;
          step: number;
          default: number;
      }
    | {
          /** Slider whose value is stored as-is (no log warp).
           *  Used for DE's F and CR — both want linear sliders.
           */
          id: string;
          kind: 'linearSlider';
          label: string;
          min: number;
          max: number;
          step: number;
          default: number;
      }
    | {
          /** A `u64` seed input with a 🎲 button to reroll it. The dice button
           *  writes a fresh `Math.floor(Math.random() * 2**31)` back via
           *  `onOptionChange`. Default `0` keeps the first-load run
           *  reproducible. */
          id: string;
          kind: 'seedField';
          label: string;
          default: number;
      };

export type SolverMeta = {
    kind: SolverKind;
    label: string;
    /** Short description shown in the UI. */
    blurb: string;
    /** Solver-specific controls, rendered in order by `Controls`. */
    options: SolverOption[];
    /**
     * Iterations advanced per animation frame. Fractional values are
     * supported: `0.25` means "one iter every 4 frames" — useful for
     * population solvers where each generation is a big visible jump and
     * you want time to read the cloud between updates. Defaults to 8 —
     * fine for single-iterate solvers (GD, NM, L-BFGS) and SSGA (whose
     * "iter" is a single offspring evaluation).
     */
    itersPerFrame?: number;
};

export const SOLVERS: SolverMeta[] = [
    {
        kind: SolverKind.GradientDescent,
        label: 'Gradient Descent',
        blurb: 'Steepest descent with a fixed step or Armijo backtracking.',
        options: [
            {
                id: 'gdLineSearch',
                kind: 'select',
                label: 'Step strategy',
                choices: [
                    { value: 'constant', label: 'Constant α' },
                    { value: 'backtracking', label: 'Backtracking' },
                ],
                default: 'constant',
            },
            {
                id: 'gdAlpha',
                kind: 'logSlider',
                label: 'Step size α',
                min: -5,
                max: 0,
                step: 0.05,
                // Overridden per-problem by the visualizer (gdAlphaDefault).
                default: 0.01,
                showIf: { id: 'gdLineSearch', equals: 'constant' },
            },
        ],
    },
    {
        kind: SolverKind.NelderMead,
        label: 'Nelder–Mead (simplex, derivative-free)',
        blurb: 'Standard reflection / expansion / contraction simplex.',
        options: [],
    },
    {
        kind: SolverKind.Lbfgs,
        label: 'L-BFGS (limited-memory quasi-Newton)',
        blurb: 'Two-loop recursion with a Moré–Thuente line search.',
        options: [
            {
                id: 'lbfgsM',
                kind: 'intSlider',
                label: 'History size m',
                min: 1,
                max: 20,
                step: 1,
                default: 10,
            },
        ],
    },
    {
        kind: SolverKind.CmaEs,
        label: 'CMA-ES (covariance matrix adaptation)',
        blurb: 'Hansen–Ostermeier evolution strategy with rank-µ + rank-1.',
        itersPerFrame: 1,
        options: [
            {
                id: 'cmaSigma',
                kind: 'logSlider',
                label: 'Initial σ',
                min: -2,
                max: 1,
                step: 0.05,
                // The wasm picks a viewport-scaled σ when this is left at NaN;
                // the slider default is a reasonable middle-of-the-road value.
                default: 0.5,
            },
            {
                id: 'cmaLambda',
                kind: 'intSlider',
                label: 'Population λ (0 = auto)',
                min: 0,
                max: 50,
                step: 1,
                default: 0,
            },
            { id: 'seed', kind: 'seedField', label: 'Seed', default: 0 },
        ],
    },
    {
        kind: SolverKind.De,
        label: 'Differential Evolution',
        blurb: 'Storn–Price DE/rand/1/bin in the viewport box.',
        // ½ gen / frame ≈ 30 gens / sec. DE on Sphere finishes in ~30 gens,
        // so the cluster contraction reads as a deliberate animation rather
        // than a flash.
        itersPerFrame: 0.5,
        options: [
            {
                id: 'dePopSize',
                kind: 'intSlider',
                label: 'Population (0 = 10n)',
                min: 0,
                max: 100,
                step: 1,
                default: 0,
            },
            {
                id: 'deF',
                kind: 'linearSlider',
                label: 'F (differential weight)',
                min: 0.1,
                max: 2,
                step: 0.05,
                default: 0.8,
            },
            {
                id: 'deCr',
                kind: 'linearSlider',
                label: 'CR (crossover probability)',
                min: 0,
                max: 1,
                step: 0.05,
                default: 0.9,
            },
            { id: 'seed', kind: 'seedField', label: 'Seed', default: 0 },
        ],
    },
    {
        kind: SolverKind.RandomSearch,
        label: 'Random Search (elitist 1+λ)',
        blurb: 'Uniform samples in the viewport box; keeps the best.',
        // RS resamples uniformly each generation, so the dots are just
        // noise that refreshes. Slow it way down (≈ 6 gens / sec) so each
        // cloud can be read before the next one replaces it.
        itersPerFrame: 0.1,
        options: [
            {
                id: 'rsLambda',
                kind: 'intSlider',
                label: 'Samples per step λ',
                min: 1,
                max: 100,
                step: 1,
                default: 16,
            },
            { id: 'seed', kind: 'seedField', label: 'Seed', default: 0 },
        ],
    },
    {
        kind: SolverKind.Ssga,
        label: 'Steady-state GA',
        blurb: 'Real-coded GA with BLX-α crossover and BGA mutation.',
        options: [
            {
                id: 'ssgaPopSize',
                kind: 'intSlider',
                label: 'Population (0 = default)',
                min: 0,
                max: 100,
                step: 1,
                default: 0,
            },
            { id: 'seed', kind: 'seedField', label: 'Seed', default: 0 },
        ],
    },
];

/** Default option values for a solver, keyed by option id. */
export function defaultOptionValues(
    meta: SolverMeta,
): Record<string, string | number> {
    const out: Record<string, string | number> = {};
    for (const opt of meta.options) out[opt.id] = opt.default;
    return out;
}
