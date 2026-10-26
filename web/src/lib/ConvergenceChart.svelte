<script lang="ts">
    import { formatDuration } from "./data/benchmarks";
    import { placeLineLabels } from "./lineLabels";

    type Point = { tNs: number; subopt: number };
    type Series = { label: string; color: string; points: Point[] };

    let {
        series,
        ariaLabel = "convergence chart",
        yLabel = "suboptimality f(x) − f*",
        xScale = "log",
        /** Drop the per-chart axis titles ("suboptimality f(x) − f*",
         * "time"). For multi-panel grids where the caller provides
         * shared labels around the grid edges. Tightens the left + bottom
         * padding too so the plot area gets the freed-up space. */
        compact = false,
        /** Place angled boxes along the curves, where other lines and labels
         * leave room. Suited to a handful of lines in a roomy chart. */
        directLabels = false,
    }: {
        series: Series[];
        ariaLabel?: string;
        yLabel?: string;
        xScale?: "log" | "linear";
        compact?: boolean;
        directLabels?: boolean;
    } = $props();

    // Static viewBox; the SVG scales to its container via `w-full h-auto`.
    const W = 380;
    const H = 264;
    // Edge anchors keep the outermost tick labels inside the viewBox.
    const padR = 20;
    const padT = 16;
    // Left and bottom padding depend on `compact` so they react if the prop flips.
    // Bumped slightly to give the widest y label ("1e-10") clearance on the left.
    const padL = $derived(compact ? 42 : 60);
    const padB = $derived(compact ? 24 : 38);
    const innerW = $derived(W - padL - padR);
    const innerH = $derived(H - padT - padB);
    const axisY = $derived(padT + innerH);

    // Logarithmic time clamps the initial sample at t = 0 to the left edge.
    const g = $derived.by(() => {
        const allT = series
            .flatMap((s) => s.points.map((p) => p.tNs))
            .filter((v) => v > 0);
        const xLo = Math.floor(Math.log10(allT.length ? Math.min(...allT) : 1));
        const xHi = Math.max(
            Math.ceil(Math.log10(allT.length ? Math.max(...allT) : 1000)),
            xLo + 1,
        );
        const xSpan = xHi - xLo;
        const tFloor = 10 ** xLo;
        const maxTime = Math.max(1, ...allT);
        const timeUnit = 10 ** Math.floor(Math.log10(maxTime));
        const timeLimit = Math.ceil(maxTime / timeUnit) * timeUnit;

        const allY = series
            .flatMap((s) => s.points.map((p) => p.subopt))
            .filter((v) => v > 0);
        const yLo = Math.floor(
            Math.log10(allY.length ? Math.min(...allY) : 1e-16),
        );
        const yHi = Math.max(
            Math.ceil(Math.log10(allY.length ? Math.max(...allY) : 1)),
            yLo + 1,
        );
        const ySpan = yHi - yLo;

        const xPx = (t: number) =>
            padL +
            (xScale === "linear"
                ? t / timeLimit
                : (Math.log10(Math.max(t, tFloor)) - xLo) / xSpan) *
                innerW;
        const yPx = (s: number) =>
            axisY - ((Math.log10(s) - yLo) / ySpan) * innerH;

        // Edge anchors keep tick labels inside the plot.
        const xTicks: {
            x: number;
            label: string;
            anchor: "start" | "middle" | "end";
        }[] = [];
        if (xScale === "linear") {
            for (let i = 0; i <= 4; i++) {
                const time = (i / 4) * timeLimit;
                xTicks.push({
                    x: xPx(time),
                    label: String(Number((time / 1e6).toPrecision(6))),
                    anchor: i === 0 ? "start" : i === 4 ? "end" : "middle",
                });
            }
        } else {
            for (let k = xLo; k <= xHi; k++) {
                const anchor =
                    k === xLo ? "start" : k === xHi ? "end" : "middle";
                xTicks.push({
                    x: xPx(10 ** k),
                    label: formatDuration(10 ** k).replace(/\.0+ /, " "),
                    anchor,
                });
            }
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
                .map((p) => ({ x: xPx(p.tNs), y: yPx(p.subopt) }));
            const d = pts
                .map(
                    (p, i) =>
                        `${i ? "L" : "M"}${p.x.toFixed(1)},${p.y.toFixed(1)}`,
                )
                .join(" ");
            return { label: s.label, color: s.color, d, points: pts };
        });

        const labels = directLabels
            ? placeLineLabels(lines, {
                  left: padL,
                  right: padL + innerW,
                  top: padT,
                  bottom: axisY,
              })
            : [];

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
            class="stroke-stone-200 dark:stroke-stone-700"
            stroke-width="1"
            x1={padL}
            x2={padL + innerW}
            y1={t.y}
            y2={t.y}
        />
        <text
            class="fill-stone-400 dark:fill-stone-500"
            x={padL - 6}
            y={t.y}
            text-anchor="end"
            dominant-baseline="middle">{t.label}</text
        >
    {/each}

    <!-- x gridlines + labels -->
    {#each g.xTicks as t}
        <line
            class="stroke-stone-100 dark:stroke-stone-800"
            stroke-width="1"
            x1={t.x}
            x2={t.x}
            y1={padT}
            y2={axisY}
        />
        <text
            class="fill-stone-400 dark:fill-stone-500"
            x={t.x}
            y={axisY + 8}
            text-anchor={t.anchor}
            dominant-baseline="hanging">{t.label}</text
        >
    {/each}

    <!-- axes -->
    <line
        class="stroke-stone-300 dark:stroke-stone-600"
        stroke-width="1"
        x1={padL}
        x2={padL}
        y1={padT}
        y2={axisY}
    />
    <line
        class="stroke-stone-300 dark:stroke-stone-600"
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
                class="fill-stone-500 dark:fill-stone-400"
                transform="rotate(-90)"
                text-anchor="middle"
                dominant-baseline="middle">{yLabel}</text
            >
        </g>
        <text
            class="fill-stone-500 dark:fill-stone-400"
            x={padL + innerW / 2}
            y={H - 4}
            text-anchor="middle">time{xScale === "linear" ? " (ms)" : ""}</text
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

    <!-- Boxes follow the local slope and use open space along each curve. -->
    {#each g.labels as label}
        {#if label.leader}
            <line
                x1={label.anchor.x}
                y1={label.anchor.y}
                x2={label.leader.x}
                y2={label.leader.y}
                stroke-width="1"
                style:stroke={label.color}
            />
        {/if}
        <g
            font-size="11"
            transform={`translate(${label.x} ${label.y}) rotate(${(label.angle * 180) / Math.PI})`}
        >
            <rect
                class="fill-white stroke-stone-400 dark:fill-stone-950 dark:stroke-stone-500"
                x={-label.width / 2}
                y={-label.height / 2}
                width={label.width}
                height={label.height}
                stroke-width="0.6"
                rx="1"
            />
            <text
                text-anchor="middle"
                dominant-baseline="central"
                style:fill={label.color}>{label.label}</text
            >
        </g>
    {/each}
</svg>
