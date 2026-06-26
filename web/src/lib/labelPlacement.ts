/**
 * Direct-labeling support: spread a set of labels along one axis so they don't
 * overlap, moving each as little as possible from its ideal position.
 *
 * This is the `last.qp` positioning method from the R package `directlabels`
 * (https://github.com/tdhock/directlabels): anchor each label at its line's
 * endpoint, then solve a small quadratic program to remove overlaps. Greedy
 * "bump the next one down" (their `last.bumpup`) is simpler but biases the whole
 * stack in one direction; the QP is symmetric and provably minimal-displacement.
 *
 * The QP is 1-D:
 *
 *     minimize  Σ (yᵢ − tᵢ)²   subject to   yᵢ₊₁ ≥ yᵢ + gap,   lo ≤ yᵢ ≤ hi
 *
 * so it has an exact O(n) solution and needs no solver. Substituting
 * zᵢ = yᵢ − i·gap turns the separation constraints into a plain monotonicity
 * constraint (zᵢ₊₁ ≥ zᵢ), which is isotonic regression: solved by
 * pool-adjacent-violators.
 */

/** Isotonic (non-decreasing) least-squares fit via pool-adjacent-violators. */
function isotonic(values: number[]): number[] {
    const val: number[] = [];
    const weight: number[] = [];
    const count: number[] = [];
    let k = -1;
    for (const v of values) {
        k++;
        val[k] = v;
        weight[k] = 1;
        count[k] = 1;
        // Merge with the previous block while it would violate monotonicity,
        // replacing both with their weighted mean.
        while (k > 0 && val[k] < val[k - 1]) {
            const w = weight[k] + weight[k - 1];
            val[k - 1] = (val[k] * weight[k] + val[k - 1] * weight[k - 1]) / w;
            weight[k - 1] = w;
            count[k - 1] += count[k];
            k--;
        }
    }
    const out: number[] = [];
    for (let b = 0; b <= k; b++) {
        for (let c = 0; c < count[b]; c++) out.push(val[b]);
    }
    return out;
}

/**
 * Place labels at `targets` (one ideal coordinate each, e.g. a line's endpoint
 * y) so neighbors are at least `gap` apart and all stay within `[lo, hi]`,
 * minimizing total squared movement. Returns new coordinates aligned with the
 * input order (so `result[i]` is where `targets[i]` should go).
 *
 * If the labels physically can't fit (`(n − 1)·gap > hi − lo`) they're spread
 * evenly across the range as a best effort.
 */
export function placeLabels(
    targets: number[],
    gap: number,
    lo: number,
    hi: number,
): number[] {
    const n = targets.length;
    if (n === 0) return [];
    if (n === 1) return [Math.min(Math.max(targets[0], lo), hi)];

    // Sort by target, remembering where each came from.
    const order = targets
        .map((_, i) => i)
        .sort((a, b) => targets[a] - targets[b]);

    // Doesn't fit: spread evenly, in target order.
    if ((n - 1) * gap > hi - lo) {
        const out = new Array<number>(n);
        order.forEach((idx, i) => {
            out[idx] = lo + ((hi - lo) * i) / (n - 1);
        });
        return out;
    }

    // Shift targets by i·gap → isotonic regression → shift back.
    const shifted = order.map((idx, i) => targets[idx] - i * gap);
    const fitted = isotonic(shifted);
    const y = fitted.map((z, i) => z + i * gap);

    // Slide the whole (now non-overlapping) stack inside the bounds. The fit
    // preserves the original span, which we've checked fits, so one nudge is
    // enough.
    if (y[0] < lo) {
        const d = lo - y[0];
        for (let i = 0; i < n; i++) y[i] += d;
    } else if (y[n - 1] > hi) {
        const d = y[n - 1] - hi;
        for (let i = 0; i < n; i++) y[i] -= d;
    }

    const out = new Array<number>(n);
    order.forEach((idx, i) => {
        out[idx] = y[i];
    });
    return out;
}
