<script lang="ts">
import { formatDuration } from "./data/benchmarks";
import { placeLabels } from "./labelPlacement";

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
// Right margin gives the end-of-line labels room to ride past the last point.
const padR = 78;
const padT = 16;
const padB = 38;
const innerW = W - padL - padR;
const innerH = H - padT - padB;
const axisY = padT + innerH;

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

    // End-of-line labels, directlabels' "angled.boxes" style with graceful
    // collision handling. Each label wants to ride its own line: sit just past
    // the endpoint, tilted to the line's final slope, on a backing box. But
    // backends often tie (endpoints within a few px), so we de-overlap the
    // anchor y's with the shared minimal-displacement placer. A label the
    // placer didn't have to move stays on its line (tilted); one it pushed
    // apart drops to horizontal and grows a short leader back to its endpoint.
    const tiltCap = (22 * Math.PI) / 180;
    const off = 8;
    const ends = lines
        .filter((l) => l.pts.length > 0)
        .map((l) => {
            const end = l.pts[l.pts.length - 1];
            const prev = l.pts[l.pts.length - 2] ?? end;
            const ang = Math.max(
                -tiltCap,
                Math.min(tiltCap, Math.atan2(end.cy - prev.cy, end.cx - prev.cx)),
            );
            return { text: l.label, color: l.color, ex: end.cx, ey: end.cy, ang };
        });
    const ys = placeLabels(
        ends.map((e) => e.ey),
        15,
        padT,
        axisY,
    );
    const labels = ends.map((e, i) => {
        const w = e.text.length * 6.7 + 6;
        if (Math.abs(ys[i] - e.ey) <= 1.5) {
            // Uncrowded: ride the line, tilted to its slope.
            const x = e.ex + Math.cos(e.ang) * off;
            const y = e.ey + Math.sin(e.ang) * off;
            const deg = (e.ang * 180) / Math.PI;
            return {
                text: e.text,
                color: e.color,
                x,
                y,
                w,
                transform: `rotate(${deg.toFixed(1)} ${x.toFixed(1)} ${y.toFixed(1)})`,
                leader: null as string | null,
            };
        }
        // Crowded: fan out horizontally with a leader back to the endpoint.
        const x = e.ex + off;
        const y = ys[i];
        return {
            text: e.text,
            color: e.color,
            x,
            y,
            w,
            transform: null as string | null,
            leader: `M${e.ex.toFixed(1)},${e.ey.toFixed(1)} L${(x - 3).toFixed(1)},${y.toFixed(1)}`,
        };
    });

    return { xTicks, yTicks, lines, labels };
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

    <!-- end-of-line labels (directlabels' "angled.boxes" style): each rides its
         line's slope where there's room, else fans out with a leader -->
    {#each g.labels as label}
        {#if label.leader}
            <path
                d={label.leader}
                fill="none"
                stroke-width="1"
                stroke-opacity="0.45"
                style="stroke: {label.color}"
            />
        {/if}
        <g transform={label.transform}>
            <rect
                class="fill-white dark:fill-slate-950"
                x={label.x - 3}
                y={label.y - 8}
                width={label.w}
                height="16"
                rx="3"
                opacity="0.85"
            />
            <text
                x={label.x}
                y={label.y}
                text-anchor="start"
                dominant-baseline="middle"
                font-weight="600"
                style="fill: {label.color}">{label.text}</text
            >
        </g>
    {/each}
</svg>
