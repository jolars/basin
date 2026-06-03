<script lang="ts">
import { formatDuration } from "./data/benchmarks";

type Point = { n: number; ns: number };
type Series = { label: string; color: string; points: Point[] };

let {
    series,
    dims,
    ariaLabel = "benchmark chart",
}: { series: Series[]; dims: number[]; ariaLabel?: string } = $props();

// Static viewBox; the SVG scales to its container via `w-full h-auto`.
const W = 380;
const H = 264;
const padL = 50;
// Right margin holds the direct end-of-line labels (one per backend),
// so we trade a little plot width for an always-visible, scroll-proof
// key that needs no separate legend.
const padR = 78;
const padT = 16;
const padB = 38;
const innerW = W - padL - padR;
const innerH = H - padT - padB;
const axisY = padT + innerH;
// Minimum vertical gap between stacked end-of-line labels.
const labelGap = 13;

// Log–log layout: x = log10(n), y = log10(time). Derived once from the
// static data (re-derives if props ever change).
const g = $derived.by(() => {
    const xs = [...dims].sort((a, b) => a - b);
    const xMin = Math.log10(xs[0]);
    const xMax = Math.log10(xs[xs.length - 1]);
    const xSpan = xMax - xMin || 1;

    const allNs = series.flatMap((s) => s.points.map((p) => p.ns)).filter((v) => v > 0);
    const yLo = Math.floor(Math.log10(Math.min(...allNs)));
    const yHi = Math.max(Math.ceil(Math.log10(Math.max(...allNs))), yLo + 1);
    const ySpan = yHi - yLo;

    const xPx = (n: number) => padL + ((Math.log10(n) - xMin) / xSpan) * innerW;
    const yPx = (ns: number) => axisY - ((Math.log10(ns) - yLo) / ySpan) * innerH;

    // x ticks at each problem size.
    const xTicks = xs.map((n) => ({ x: xPx(n), label: `${n}` }));

    // y ticks at each decade of time within range. Decades are exact, so
    // drop formatDuration's trailing zeros ("10.0 µs" → "10 µs").
    const yTicks: { y: number; label: string }[] = [];
    for (let k = Math.ceil(yLo); k <= Math.floor(yHi); k++) {
        yTicks.push({
            y: yPx(10 ** k),
            label: formatDuration(10 ** k).replace(/\.0+ /, " "),
        });
    }

    const lines = series.map((s) => {
        const pts = [...s.points]
            .sort((a, b) => a.n - b.n)
            .map((p) => ({ cx: xPx(p.n), cy: yPx(p.ns) }));
        const d = pts
            .map((p, i) => `${i ? "L" : "M"}${p.cx.toFixed(1)},${p.cy.toFixed(1)}`)
            .join(" ");
        return { label: s.label, color: s.color, d, pts };
    });

    // Direct end-of-line labels: anchor each at its line's right endpoint,
    // then nudge vertically so converging labels don't overlap. Sort by the
    // target y and push collisions downward; if the stack runs past the axis,
    // shift the whole group back up so it stays inside the plot.
    const labels = lines
        .filter((l) => l.pts.length > 0)
        .map((l) => ({
            text: l.label,
            color: l.color,
            y: l.pts[l.pts.length - 1].cy,
        }))
        .sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++) {
        if (labels[i].y - labels[i - 1].y < labelGap) {
            labels[i].y = labels[i - 1].y + labelGap;
        }
    }
    if (labels.length) {
        const overflow = labels[labels.length - 1].y - axisY;
        if (overflow > 0) for (const l of labels) l.y -= overflow;
        const underflow = padT - labels[0].y;
        if (underflow > 0) for (const l of labels) l.y += underflow;
    }
    const labelX = padL + innerW + 6;

    return { xTicks, yTicks, lines, labels, labelX };
});
</script>

<svg
    viewBox="0 0 {W} {H}"
    class="w-full h-auto"
    role="img"
    aria-label={ariaLabel}
    font-size="12"
>
    <!-- y decade gridlines + labels -->
    {#each g.yTicks as t}
        <line
            class="stroke-slate-200 dark:stroke-slate-700"
            stroke-width="1"
            x1={padL}
            x2={padL + innerW}
            y1={t.y}
            y2={t.y}
        />
        <text
            class="fill-slate-400 dark:fill-slate-500"
            x={padL - 6}
            y={t.y}
            text-anchor="end"
            dominant-baseline="middle">{t.label}</text
        >
    {/each}

    <!-- x gridlines + labels (one per problem size) -->
    {#each g.xTicks as t}
        <line
            class="stroke-slate-100 dark:stroke-slate-800"
            stroke-width="1"
            x1={t.x}
            x2={t.x}
            y1={padT}
            y2={axisY}
        />
        <text
            class="fill-slate-400 dark:fill-slate-500"
            x={t.x}
            y={axisY + 8}
            text-anchor="middle"
            dominant-baseline="hanging">{t.label}</text
        >
    {/each}

    <!-- axes -->
    <line
        class="stroke-slate-300 dark:stroke-slate-600"
        stroke-width="1"
        x1={padL}
        x2={padL}
        y1={padT}
        y2={axisY}
    />
    <line
        class="stroke-slate-300 dark:stroke-slate-600"
        stroke-width="1"
        x1={padL}
        x2={padL + innerW}
        y1={axisY}
        y2={axisY}
    />

    <!-- captions -->
    <text
        class="fill-slate-500 dark:fill-slate-400"
        x={padL}
        y={padT - 5}
        text-anchor="start">time / solve</text
    >
    <text
        class="fill-slate-500 dark:fill-slate-400"
        x={padL + innerW / 2}
        y={H - 4}
        text-anchor="middle">n (parameters)</text
    >

    <!-- series: one polyline + markers per backend -->
    {#each g.lines as line}
        <path d={line.d} fill="none" stroke-width="2" style="stroke: {line.color}" />
        {#each line.pts as p}
            <circle cx={p.cx} cy={p.cy} r="2.6" style="fill: {line.color}" />
        {/each}
    {/each}

    <!-- direct end-of-line labels: the per-chart key, in each line's color -->
    {#each g.labels as label}
        <text
            x={g.labelX}
            y={label.y}
            text-anchor="start"
            dominant-baseline="middle"
            font-weight="600"
            style="fill: {label.color}">{label.text}</text
        >
    {/each}
</svg>
