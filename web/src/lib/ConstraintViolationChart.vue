<script setup lang="ts">
import { computed } from "vue";
import { formatDuration } from "./data/benchmarks";

type Series = {
    label: string;
    color: string;
    points: { tNs: number; violation: number }[];
};

let {
    series,
    tolerance,
    ariaLabel = "Trial constraint violation against time",
} = defineProps<{
    series: Series[];
    tolerance: number;
    ariaLabel?: string;
}>();

const W = 380;
const H = 236;
const left = 66;
const right = 16;
const top = 14;
const bottom = 42;
const width = W - left - right;
const height = H - top - bottom;
const baseline = top + height;

const chart = computed(() => {
    const points = series.flatMap((s) => s.points);
    const maxTime = Math.max(1, ...points.map((p) => p.tNs));
    const timeUnit = 10 ** Math.floor(Math.log10(maxTime));
    const timeLimit = Math.ceil(maxTime / timeUnit) * timeUnit;
    const maxViolation = Math.max(tolerance, ...points.map((p) => p.violation));
    const upperExponent = Math.ceil(Math.log10(maxViolation));
    const upper = 10 ** upperExponent;

    // A linear region preserves zero and shows the feasibility threshold.
    // Above it, decades keep large infeasible excursions in the same view.
    const scale = (v: number) =>
        v <= tolerance ? v / tolerance : 1 + Math.log10(v / tolerance);
    const x = (time: number) => left + (time / timeLimit) * width;
    const y = (violation: number) =>
        baseline - (scale(violation) / scale(upper)) * height;
    const ticks = [0, tolerance];
    const firstExponent = Math.ceil(Math.log10(tolerance)) + 1;
    const step = Math.max(1, Math.ceil((upperExponent - firstExponent) / 3));
    for (
        let exponent = firstExponent;
        exponent <= upperExponent;
        exponent += step
    ) {
        ticks.push(10 ** exponent);
    }

    return {
        x,
        y,
        toleranceY: y(tolerance),
        xTicks: Array.from({ length: 5 }, (_, i) => ({
            x: x((i / 4) * timeLimit),
            label: String(Number((((i / 4) * timeLimit) / 1e6).toPrecision(6))),
        })),
        yTicks: ticks.map((value) => ({
            y: y(value),
            label:
                value === 0
                    ? "0"
                    : value
                          .toExponential(value === tolerance ? 2 : 0)
                          .replace("e+", "e"),
            threshold: value === tolerance,
        })),
    };
});
</script>

<template>
    <div
        class="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-stone-600 dark:text-stone-400"
    >
        <template v-for="(s, i) in series">
            <span class="inline-flex items-center gap-1.5">
                <span
                    class="h-2 w-2"
                    :class="{ 'rounded-full': i % 2 === 0 }"
                    :background-color="s.color"
                ></span>
                {{ s.label }}
            </span>
        </template>
        <span class="inline-flex items-center gap-1.5">
            <span
                class="w-4 border-t border-dashed border-stone-500 dark:border-stone-400"
            ></span>
            Feasibility tolerance
        </span>
    </div>

    <svg
        :viewBox="'0 0 ' + W + ' ' + H"
        class="h-auto w-full"
        role="img"
        :aria-label="ariaLabel"
        font-size="12"
    >
        <rect
            :x="left"
            :y="chart.toleranceY"
            :width="width"
            :height="baseline - chart.toleranceY"
            class="fill-stone-100 dark:fill-stone-800"
        ></rect>
        <template v-for="(tick, i) in chart.xTicks">
            <line
                :x1="tick.x"
                :x2="tick.x"
                :y1="top"
                :y2="baseline"
                class="stroke-stone-100 dark:stroke-stone-800"
            ></line>
            <text
                :x="tick.x"
                :y="baseline + 18"
                :text-anchor="i === 0 ? 'start' : i === 4 ? 'end' : 'middle'"
                class="fill-stone-500 dark:fill-stone-400"
            >
                {{ tick.label }}
            </text>
        </template>
        <template v-for="tick in chart.yTicks">
            <line
                :x1="left"
                :x2="W - right"
                :y1="tick.y"
                :y2="tick.y"
                :stroke-dasharray="tick.threshold ? '4 3' : undefined"
                :class="
                    tick.threshold
                        ? 'stroke-stone-500 dark:stroke-stone-400'
                        : 'stroke-stone-200 dark:stroke-stone-800'
                "
            ></line>
            <text
                :x="left - 8"
                :y="tick.y"
                text-anchor="end"
                dominant-baseline="middle"
                class="fill-stone-500 dark:fill-stone-400"
            >
                {{ tick.label }}
            </text>
        </template>
        <template v-for="(s, i) in series">
            <template v-for="point in s.points">
                <template v-if="i % 2 === 0">
                    <circle
                        :cx="chart.x(point.tNs)"
                        :cy="chart.y(point.violation)"
                        r="2.5"
                        :fill="s.color"
                        fill-opacity="0.8"
                    >
                        <title>
                            {{ s.label }}:
                            {{ point.violation.toExponential(3) }} at
                            {{ formatDuration(point.tNs) }}
                        </title>
                    </circle>
                </template>
                <template v-else>
                    <rect
                        :x="chart.x(point.tNs) - 2"
                        :y="chart.y(point.violation) - 2"
                        width="4"
                        height="4"
                        :fill="s.color"
                        fill-opacity="0.8"
                    >
                        <title>
                            {{ s.label }}:
                            {{ point.violation.toExponential(3) }} at
                            {{ formatDuration(point.tNs) }}
                        </title>
                    </rect>
                </template>
            </template>
        </template>
        <text
            x="14"
            :y="top + height / 2"
            :transform="'rotate(-90 14 ' + (top + height / 2) + ')'"
            text-anchor="middle"
            class="fill-stone-500 dark:fill-stone-400"
        >
            maximum constraint violation
        </text>
        <text
            :x="left + width / 2"
            :y="H - 4"
            text-anchor="middle"
            class="fill-stone-500 dark:fill-stone-400"
        >
            time (ms)
        </text>
    </svg>
    <p class="mt-1 text-xs text-stone-500 dark:text-stone-400">
        Each point is one trial. Zero is on the baseline; the shaded band meets
        the feasibility tolerance.
    </p>
</template>
