<script lang="ts">
    import { formatDuration } from "./data/benchmarks";
    import { placeLabels } from "./labelPlacement";

    type Point = { tNs: number; subopt: number };
    type Series = { label: string; color: string; points: Point[] };

    let {
        series,
        ariaLabel = "convergence chart",
        /** Drop the per-chart axis titles ("suboptimality f(x) − f*",
         * "wall-clock time"). For multi-panel grids where the caller provides
         * shared labels around the grid edges. Tightens the left + bottom
         * padding too so the plot area gets the freed-up space. */
        compact = false,
        /** Label each line directly at its end (directlabels' "angled.boxes" style)
         * instead of relying on a separate legend. Suited to roomy charts with a
         * handful of lines; leave off for dense tiny-multiples. Widens the right
         * padding to give the labels room. */
        directLabels = false,
    }: {
        series: Series[];
        ariaLabel?: string;
        compact?: boolean;
        directLabels?: boolean;
    } = $props();

    // Static viewBox; the SVG scales to its container via `w-full h-auto`.
    const W = 380;
    const H = 264;
    // Right and top need a bit of breathing room so the rightmost or topmost tick
    // labels don't bump against the viewBox edge (the last x label, e.g. "20 ms",
    // also uses an `end` text-anchor below so it tucks inside the plot edge).
    // `directLabels` widens the right margin to hold the fanned-out labels.
    const padR = $derived(directLabels ? 80 : 20);
    const padT = 16;
    // Left and bottom padding depend on `compact` so they react if the prop flips.
    // Bumped slightly to give the widest y label ("1e-10") clearance on the left.
    const padL = $derived(compact ? 42 : 60);
    const padB = $derived(compact ? 24 : 38);
    const innerW = $derived(W - padL - padR);
    const innerH = $derived(H - padT - padB);
    const axisY = $derived(padT + innerH);
    // Minimum vertical gap between fanned-out end-of-line labels.
    const labelGap = 15;

    // Log–log layout: x = log10(time), y = log10(suboptimality). The iter-0
    // sample sits at t = 0, which has no log; it's clamped to the left edge so
    // the initial objective value still shows.
    const g = $derived.by(() => {
        const allT = series
            .flatMap((s) => s.points.map((p) => p.tNs))
            .filter((v) => v > 0);
        const xLo = Math.floor(Math.log10(Math.min(...allT)));
        const xHi = Math.max(Math.ceil(Math.log10(Math.max(...allT))), xLo + 1);
        const xSpan = xHi - xLo;
        const tFloor = 10 ** xLo;

        const allY = series
            .flatMap((s) => s.points.map((p) => p.subopt))
            .filter((v) => v > 0);
        const yLo = Math.floor(Math.log10(Math.min(...allY)));
        const yHi = Math.max(Math.ceil(Math.log10(Math.max(...allY))), yLo + 1);
        const ySpan = yHi - yLo;

        const xPx = (t: number) =>
            padL + ((Math.log10(Math.max(t, tFloor)) - xLo) / xSpan) * innerW;
        const yPx = (s: number) =>
            axisY - ((Math.log10(s) - yLo) / ySpan) * innerH;

        // x ticks at each decade of time, labeled as a duration. First and last
        // tick labels use start and end anchors so they tuck inside the plot
        // edges instead of centering past them and getting clipped.
        const xTicks: {
            x: number;
            label: string;
            anchor: "start" | "middle" | "end";
        }[] = [];
        for (let k = xLo; k <= xHi; k++) {
            const anchor = k === xLo ? "start" : k === xHi ? "end" : "middle";
            xTicks.push({
                x: xPx(10 ** k),
                label: formatDuration(10 ** k).replace(/\.0+ /, " "),
                anchor,
            });
        }

        // y ticks at decades of suboptimality, thinned to ≤ ~7 labels so a wide
        // range (e.g. 1e1 → 1e-16) doesn't crowd the axis.
        const step = Math.max(1, Math.ceil(ySpan / 6));
        const yTicks: { y: number; label: string }[] = [];
        for (let k = yHi; k >= yLo; k -= step) {
            yTicks.push({ y: yPx(10 ** k), label: `1e${k}` });
        }

        const lines = series.map((s) => {
            const pts = [...s.points]
                .sort((a, b) => a.tNs - b.tNs)
                .map((p) => ({ cx: xPx(p.tNs), cy: yPx(p.subopt) }));
            const d = pts
                .map(
                    (p, i) =>
                        `${i ? "L" : "M"}${p.cx.toFixed(1)},${p.cy.toFixed(1)}`,
                )
                .join(" ");
            const end = pts[pts.length - 1];
            const prev = pts[pts.length - 2] ?? end;
            return { label: s.label, color: s.color, d, end, prev };
        });

        // End-of-line labels (opt-in), directlabels' "angled.boxes" style with
        // graceful collision handling. Each label wants to ride its line: sit just
        // past the endpoint, tilted to the final slope, on a backing box. But
        // convergence traces bottom out at the suboptimality floor at different
        // times, so endpoints cluster; we de-overlap the anchor y's with the shared
        // minimal-displacement placer. A label the placer left alone rides its line;
        // one it had to move fans into the right-margin column with a leader back to
        // its endpoint.
        type Label = {
            text: string;
            color: string;
            x: number;
            y: number;
            w: number;
            transform: string | null;
            leader: string | null;
        };
        let labels: Label[] = [];
        if (directLabels) {
            const tiltCap = (22 * Math.PI) / 180;
            const off = 8;
            const ends = lines
                .filter((l) => l.end)
                .map((l) => {
                    const ang = Math.max(
                        -tiltCap,
                        Math.min(
                            tiltCap,
                            Math.atan2(
                                l.end.cy - l.prev.cy,
                                l.end.cx - l.prev.cx,
                            ),
                        ),
                    );
                    return {
                        text: l.label,
                        color: l.color,
                        ex: l.end.cx,
                        ey: l.end.cy,
                        ang,
                    };
                });
            const ys = placeLabels(
                ends.map((e) => e.ey),
                labelGap,
                padT,
                axisY,
            );
            const colX = padL + innerW + 6;
            labels = ends.map((e, i) => {
                const w = e.text.length * 6.7 + 6;
                if (Math.abs(ys[i] - e.ey) <= 1.5) {
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
                        leader: null,
                    };
                }
                const x = colX;
                const y = ys[i];
                return {
                    text: e.text,
                    color: e.color,
                    x,
                    y,
                    w,
                    transform: null,
                    leader: `M${e.ex.toFixed(1)},${e.ey.toFixed(1)} L${(x - 3).toFixed(1)},${y.toFixed(1)}`,
                };
            });
        }

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

    <!-- x gridlines + labels (one per time decade) -->
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
            text-anchor={t.anchor}
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

    {#if !compact}
        <!-- captions: per-chart axis titles, omitted in compact mode -->
        <g transform="translate(14 {padT + innerH / 2})">
            <text
                class="fill-slate-500 dark:fill-slate-400"
                transform="rotate(-90)"
                text-anchor="middle"
                dominant-baseline="middle">suboptimality f(x) − f*</text
            >
        </g>
        <text
            class="fill-slate-500 dark:fill-slate-400"
            x={padL + innerW / 2}
            y={H - 4}
            text-anchor="middle">wall-clock time</text
        >
    {/if}

    <!-- series: one polyline per series (dense traces, no markers) -->
    {#each g.lines as line}
        <path
            d={line.d}
            fill="none"
            stroke-width="2"
            style="stroke: {line.color}"
        />
    {/each}

    <!-- end-of-line labels (opt-in, directlabels' "angled.boxes" style): each
         rides its line's slope where there's room, else fans into the right
         margin with a leader back to its endpoint -->
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
