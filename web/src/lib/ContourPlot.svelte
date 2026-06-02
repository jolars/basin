<script lang="ts">
    import { onMount } from 'svelte';
    import {
        chainSegments,
        chooseLevels,
        isoContour,
        smoothChaikin,
        transform,
    } from './contours';
    import { paletteFor, type Theme } from './palette';
    import type { Domain, ProblemMeta } from './problems';

    type Props = {
        problem: ProblemMeta;
        grid: Float64Array; // length nx*ny, row-major (rows are y)
        nx: number;
        ny: number;
        trajectory: Float64Array; // flat (x, y) pairs
        /** Current generation's population as flat (x, y) pairs. Optional —
         *  empty for non-population solvers (GD, NM, L-BFGS). When non-empty,
         *  drawn as faint dots underneath the trajectory polyline. */
        population?: Float64Array;
        startPoint: { x: number; y: number };
        theme: Theme;
        onPick: (p: { x: number; y: number }) => void;
        /** Draw contour lines as a grey topographic ramp instead of the
         *  colored (viridis) palette. Trajectory/markers stay accented. */
        monochrome?: boolean;
    };

    let {
        problem,
        grid,
        nx,
        ny,
        trajectory,
        population = new Float64Array(0),
        startPoint,
        theme,
        onPick,
        monochrome = false,
    }: Props = $props();

    let palette = $derived(paletteFor(theme));

    // White contour edges in monochrome mode. Fainter on dark so the
    // (white) trajectory still reads over them; on light they sit over the
    // grey heatmap fill. `t` is unused — depth comes from the per-level
    // line width in `renderContours`.
    function whiteEdge(_t: number): string {
        return theme === 'dark'
            ? 'rgba(255, 255, 255, 0.35)'
            : 'rgba(255, 255, 255, 0.9)';
    }
    let contourStroke = $derived(monochrome ? whiteEdge : palette.contour);

    // Light-grey (dark mode: dark-grey) heatmap fill from the cost grid,
    // normalized through the same intensity transform the levels use. Drawn
    // once per contour render (not per animation frame) onto the contour
    // canvas, beneath the white edges.
    function drawHeatmap(ctx: CanvasRenderingContext2D, w: number, h: number) {
        let cmin = Infinity;
        let cmax = -Infinity;
        for (let i = 0; i < grid.length; i++) {
            const v = grid[i];
            if (!Number.isFinite(v)) continue;
            if (v < cmin) cmin = v;
            if (v > cmax) cmax = v;
        }
        const tmin = transform(Math.max(cmin, 0), problem.intensity);
        const tmax = transform(cmax, problem.intensity);
        const span = tmax - tmin || 1;
        const dark = theme === 'dark';
        const off = document.createElement('canvas');
        off.width = nx;
        off.height = ny;
        const octx = off.getContext('2d');
        if (!octx) return;
        const img = octx.createImageData(nx, ny);
        for (let j = 0; j < ny; j++) {
            for (let i = 0; i < nx; i++) {
                const v = grid[j * nx + i];
                const t = Number.isFinite(v)
                    ? Math.max(
                          0,
                          Math.min(
                              1,
                              (transform(v, problem.intensity) - tmin) / span,
                          ),
                      )
                    : 0;
                // Darker deeper in the valley: t = 0 is the lowest cost
                // (basin floor), so grey rises with t. Kept in a light band
                // overall, with the floor only moderately darker.
                const grey = dark
                    ? Math.round(45 + 45 * t)
                    : Math.round(195 + 50 * t);
                // ImageData row 0 is the top of the canvas (ymax); grid j = 0
                // is ymin, so flip vertically.
                const idx = ((ny - 1 - j) * nx + i) * 4;
                img.data[idx] = grey;
                img.data[idx + 1] = grey;
                img.data[idx + 2] = grey;
                img.data[idx + 3] = 255;
            }
        }
        octx.putImageData(img, 0, 0);
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(off, 0, 0, w, h);
    }

    let canvas: HTMLCanvasElement | undefined = $state();
    let overlay: HTMLCanvasElement | undefined = $state();
    let containerWidth = $state(640);
    let containerHeight = $state(480);

    // How many iso-contours to draw. 12 reads as "topographic map" without
    // becoming visual noise.
    const N_LEVELS = 12;

    // Re-extract iso-contours when the grid (i.e. the problem) changes.
    // This is the expensive step — O(nx * ny * n_levels) — but it only
    // runs on problem switches, not per animation frame.
    let isoLines = $derived.by(() => {
        const levels = chooseLevels(grid, problem.intensity, N_LEVELS);
        const d = problem.domain;
        return levels.map((level) => {
            const segs = isoContour(
                grid,
                nx,
                ny,
                d.xmin,
                d.xmax,
                d.ymin,
                d.ymax,
                level,
            );
            // Stitch into polylines so we can stroke each as one path,
            // then Chaikin-smooth to kill the marching-squares
            // stair-step. Two iterations is the sweet spot.
            const chains = chainSegments(segs).map((c) => smoothChaikin(c, 2));
            return { level, chains };
        });
    });

    // Render contours when contours, sizing, theme, or domain change.
    $effect(() => {
        if (!canvas) return;
        renderContours(canvas, problem.domain, isoLines, palette, contourStroke);
    });

    // Trajectory + markers go on a separate overlay so they redraw cheaply
    // every time the trajectory grows. Re-runs on theme and on population
    // updates so the search cloud refreshes alongside the best-so-far trail.
    $effect(() => {
        if (!overlay) return;
        renderOverlay(
            overlay,
            problem.domain,
            trajectory,
            population,
            startPoint,
            problem.minimum,
            palette,
        );
    });

    onMount(() => {
        const ro = new ResizeObserver((entries) => {
            for (const e of entries) {
                containerWidth = Math.max(120, Math.floor(e.contentRect.width));
                containerHeight = Math.max(
                    120,
                    Math.floor(e.contentRect.height),
                );
            }
        });
        ro.observe(canvas!.parentElement!);
        return () => ro.disconnect();
    });

    function dataToPixel(
        x: number,
        y: number,
        d: Domain,
        w: number,
        h: number,
    ) {
        const px = ((x - d.xmin) / (d.xmax - d.xmin)) * w;
        // Flip y so larger y is up.
        const py = h - ((y - d.ymin) / (d.ymax - d.ymin)) * h;
        return [px, py];
    }

    function pixelToData(
        px: number,
        py: number,
        d: Domain,
        w: number,
        h: number,
    ) {
        const x = d.xmin + (px / w) * (d.xmax - d.xmin);
        const y = d.ymin + ((h - py) / h) * (d.ymax - d.ymin);
        return { x, y };
    }

    function renderContours(
        cv: HTMLCanvasElement,
        d: Domain,
        lines: { level: number; chains: number[][] }[],
        pal: ReturnType<typeof paletteFor>,
        stroke: (t: number) => string,
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

        if (monochrome) {
            drawHeatmap(ctx, w, h);
        } else {
            ctx.fillStyle = pal.surface;
            ctx.fillRect(0, 0, w, h);
        }

        if (lines.length === 0) return;

        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';

        // Draw outermost-first so the brightest (innermost) strokes win
        // when contours crowd. `t = 0` is the outermost, `t = 1` the
        // innermost — palette decides what those map to per theme.
        for (let li = lines.length - 1; li >= 0; li--) {
            const chains = lines[li].chains;
            if (chains.length === 0) continue;
            const t = li / Math.max(lines.length - 1, 1);
            ctx.strokeStyle = stroke(1 - t);
            // Inner contours a hair thicker so the basin reads.
            ctx.lineWidth = 1 + (1 - t) * 0.6;
            ctx.beginPath();
            for (const chain of chains) {
                if (chain.length < 4) continue;
                const [px0, py0] = dataToPixel(chain[0], chain[1], d, w, h);
                ctx.moveTo(px0, py0);
                for (let k = 2; k < chain.length; k += 2) {
                    const [px, py] = dataToPixel(
                        chain[k],
                        chain[k + 1],
                        d,
                        w,
                        h,
                    );
                    ctx.lineTo(px, py);
                }
            }
            ctx.stroke();
        }
    }

    function renderOverlay(
        cv: HTMLCanvasElement,
        d: Domain,
        traj: Float64Array,
        pop: Float64Array,
        start: { x: number; y: number },
        minimum: { x: number; y: number },
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

        // Current-generation population cloud (CMA-ES, DE, RS, SSGA). Drawn
        // first so the best-so-far trail visually wins. Faint sky-blue dots
        // read as "the swarm" vs the white/accent trajectory line.
        if (pop.length >= 2) {
            ctx.fillStyle =
                theme === 'dark'
                    ? 'rgba(56, 189, 248, 0.55)' // sky-400
                    : 'rgba(2, 132, 199, 0.55)'; // sky-600
            for (let i = 0; i < pop.length; i += 2) {
                const [px, py] = dataToPixel(pop[i], pop[i + 1], d, w, h);
                ctx.beginPath();
                ctx.arc(px, py, 3, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // Trajectory line + dots.
        if (traj.length >= 4) {
            ctx.beginPath();
            for (let i = 0; i < traj.length; i += 2) {
                const [px, py] = dataToPixel(traj[i], traj[i + 1], d, w, h);
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }
            ctx.lineWidth = 2;
            ctx.strokeStyle = pal.trajectory;
            ctx.stroke();
        }
        for (let i = 2; i < traj.length; i += 2) {
            const [px, py] = dataToPixel(traj[i], traj[i + 1], d, w, h);
            ctx.beginPath();
            ctx.arc(px, py, 2, 0, Math.PI * 2);
            ctx.fillStyle = pal.trajectory;
            ctx.fill();
        }

        // Start marker (open circle).
        const [sx, sy] = dataToPixel(start.x, start.y, d, w, h);
        ctx.beginPath();
        ctx.arc(sx, sy, 6, 0, Math.PI * 2);
        ctx.lineWidth = 2;
        ctx.strokeStyle = pal.startMarker;
        ctx.stroke();

        // Known minimum: an orange disc in monochrome mode (the optimum
        // "target"); the original red cross otherwise (visualizer).
        const [mx, my] = dataToPixel(minimum.x, minimum.y, d, w, h);
        if (monochrome) {
            // Orange ring marking the optimum.
            ctx.beginPath();
            ctx.arc(mx, my, 8, 0, Math.PI * 2);
            ctx.lineWidth = 2.5;
            ctx.strokeStyle = 'rgb(249, 115, 22)'; // orange-500
            ctx.stroke();
        } else {
            ctx.lineWidth = 2;
            ctx.strokeStyle = pal.minimum;
            ctx.beginPath();
            ctx.moveTo(mx - 6, my);
            ctx.lineTo(mx + 6, my);
            ctx.moveTo(mx, my - 6);
            ctx.lineTo(mx, my + 6);
            ctx.stroke();
        }
    }

    function handlePointer(ev: PointerEvent) {
        if (!overlay) return;
        const rect = overlay.getBoundingClientRect();
        const px = ev.clientX - rect.left;
        const py = ev.clientY - rect.top;
        const p = pixelToData(
            px,
            py,
            problem.domain,
            rect.width,
            rect.height,
        );
        onPick(p);
    }
</script>

<div class="relative w-full h-full">
    <!-- Contour canvas, sized in CSS px (no longer at native grid res). -->
    <canvas bind:this={canvas} class="absolute inset-0 w-full h-full"></canvas>
    <!-- Overlay sits on top, sized to the displayed container in CSS px. -->
    <canvas
        bind:this={overlay}
        class="absolute inset-0 w-full h-full cursor-crosshair"
        onpointerdown={handlePointer}
    ></canvas>
</div>
