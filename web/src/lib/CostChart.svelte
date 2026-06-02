<script lang="ts">
    import { onMount } from 'svelte';

    import { paletteFor, type Theme } from './palette';
    import { SUBOPT_TARGET } from './problems';

    type Props = {
        costs: Float64Array;
        /** Known optimal value f*; the chart plots suboptimality f − f*. */
        fStar: number;
        /** Iteration budget. Fixes the x-axis right edge so the curve
         *  descends into a fixed frame instead of the chart auto-rescaling
         *  each animation tick. */
        maxIter: number;
        /** Termination reason string when the run ends, else empty. */
        reason: string;
        theme: Theme;
    };

    let { costs, fStar, maxIter, reason, theme }: Props = $props();
    let palette = $derived(paletteFor(theme));

    let canvas: HTMLCanvasElement | undefined = $state();
    let containerWidth = $state(320);
    let containerHeight = $state(240);

    $effect(() => {
        if (!canvas) return;
        render(canvas, costs, fStar, maxIter, palette);
    });

    onMount(() => {
        const ro = new ResizeObserver((entries) => {
            for (const e of entries) {
                containerWidth = Math.max(120, Math.floor(e.contentRect.width));
                containerHeight = Math.max(
                    100,
                    Math.floor(e.contentRect.height),
                );
            }
        });
        ro.observe(canvas!.parentElement!);
        return () => ro.disconnect();
    });

    function render(
        cv: HTMLCanvasElement,
        c: Float64Array,
        fstar: number,
        xMax: number,
        pal: ReturnType<typeof paletteFor>,
    ) {
        const dpr = window.devicePixelRatio || 1;
        const w = containerWidth;
        const h = containerHeight;
        cv.width = Math.floor(w * dpr);
        cv.height = Math.floor(h * dpr);
        cv.style.width = `${w}px`;
        cv.style.height = `${h}px`;
        const ctx = cv.getContext('2d');
        if (!ctx) return;
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, w, h);

        ctx.fillStyle = pal.surface;
        ctx.fillRect(0, 0, w, h);

        if (c.length < 2) return;

        // Suboptimality s_i = f_i − f* on a log10 y-axis. The range is
        // FIXED, not data-driven: the bottom is the convergence target
        // (SUBOPT_TARGET), the top is one decade above the run's initial
        // suboptimality. Both are constant for the whole run (the initial
        // cost never changes), so nothing rescales frame-to-frame, and the
        // top is identical across solver switches from the same start.
        const yLo = Math.log10(SUBOPT_TARGET);
        const s0 = Math.max(c[0] - fstar, SUBOPT_TARGET);
        const yHi = Math.max(Math.ceil(Math.log10(s0)), yLo + 1);
        const ySpan = yHi - yLo;
        // Clamp into [yLo, yHi]: descent stays below the top, and a value
        // at/under the optimum lands on the bottom axis.
        const yOf = (v: number) =>
            Math.min(Math.log10(Math.max(v - fstar, SUBOPT_TARGET)), yHi);

        const padL = 46;
        const padR = 10;
        const padT = 18;
        const padB = 32;
        const innerW = Math.max(1, w - padL - padR);
        const innerH = Math.max(1, h - padT - padB);
        const pxOf = (yv: number) =>
            padT + innerH - ((yv - yLo) / ySpan) * innerH;

        // Axes.
        ctx.strokeStyle = pal.axis;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padL, padT);
        ctx.lineTo(padL, padT + innerH);
        ctx.lineTo(padL + innerW, padT + innerH);
        ctx.stroke();

        ctx.font = '11px sans-serif';

        // y-axis: decade gridlines + labels, thinned to ~6 across the range.
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        const kLo = Math.ceil(yLo);
        const kHi = Math.floor(yHi);
        const step = Math.max(1, Math.ceil((kHi - kLo) / 6));
        for (let k = kLo; k <= kHi; k += step) {
            const yy = pxOf(k);
            ctx.strokeStyle = pal.axis;
            ctx.globalAlpha = 0.25;
            ctx.beginPath();
            ctx.moveTo(padL, yy);
            ctx.lineTo(padL + innerW, yy);
            ctx.stroke();
            ctx.globalAlpha = 1;
            ctx.fillStyle = pal.text;
            ctx.fillText(`1e${k}`, padL - 4, yy);
        }

        // Captions: y quantity (top-left), termination reason (top-right).
        ctx.fillStyle = pal.text;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
        ctx.fillText('f − f*', padL, padT - 5);
        if (reason) {
            ctx.textAlign = 'right';
            ctx.fillStyle = pal.reason;
            ctx.fillText(reason, w - padR, padT - 5);
        }

        // x-axis: iteration index. Range is FIXED to [0, maxIter] so the
        // curve descends into a static frame as the run progresses, instead
        // of the chart auto-scaling to the most-recent iteration (which made
        // the polyline appear to compress as new points arrived).
        const xRange = Math.max(xMax, 1);
        ctx.fillStyle = pal.text;
        ctx.textBaseline = 'top';
        ctx.textAlign = 'left';
        ctx.fillText('0', padL, padT + innerH + 5);
        ctx.textAlign = 'right';
        ctx.fillText(`${xRange}`, padL + innerW, padT + innerH + 5);
        ctx.textAlign = 'center';
        ctx.fillText('iteration', padL + innerW / 2, padT + innerH + 18);

        // Polyline.
        ctx.beginPath();
        for (let i = 0; i < c.length; i++) {
            const x = padL + (i / xRange) * innerW;
            const y = pxOf(yOf(c[i]));
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = pal.cost;
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }
</script>

<canvas bind:this={canvas} class="w-full h-full block"></canvas>
