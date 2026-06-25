<script lang="ts">
    import { ProblemKind, SolverKind } from "./basin-wasm/basin_wasm";
    import { PROBLEMS } from "./problems";
    import { SOLVERS, type SolverOption } from "./solvers";

    type Patch = {
        problemKind?: ProblemKind;
        solverKind?: SolverKind;
        maxIter?: number;
    };

    type Props = {
        problemKind: ProblemKind;
        solverKind: SolverKind;
        /** Schema for the current solver's controls. */
        solverOptions: SolverOption[];
        /** Current option values, keyed by option id. */
        optionValues: Record<string, string | number>;
        maxIter: number;
        startPoint: { x: number; y: number };
        onChange: (patch: Patch) => void;
        onOptionChange: (id: string, value: string | number) => void;
    };

    let {
        problemKind,
        solverKind,
        solverOptions,
        optionValues,
        maxIter,
        startPoint,
        onChange,
        onOptionChange,
    }: Props = $props();

    const labelCls =
        "text-slate-700 dark:text-slate-300 uppercase text-xs tracking-wide";
    const selectCls =
        "bg-white text-slate-900 border border-slate-300 dark:bg-slate-800 dark:text-slate-100 dark:border-slate-700 rounded px-2 py-1";
    const valueCls = "font-mono text-slate-900 dark:text-slate-100";

    // A `logSlider` is hidden unless its `showIf` option currently matches.
    function visible(opt: SolverOption): boolean {
        if (opt.kind !== "logSlider" || !opt.showIf) return true;
        return optionValues[opt.showIf.id] === opt.showIf.equals;
    }
</script>

<div class="flex flex-col gap-4 text-sm">
    <label class="flex flex-col gap-1">
        <span class={labelCls}>Problem</span>
        <select
            class={selectCls}
            value={problemKind}
            onchange={(e) =>
                onChange({
                    problemKind: Number(
                        (e.currentTarget as HTMLSelectElement).value,
                    ) as ProblemKind,
                })}
        >
            {#each PROBLEMS as p}
                <option value={p.kind}>{p.label}</option>
            {/each}
        </select>
    </label>

    <label class="flex flex-col gap-1">
        <span class={labelCls}>Solver</span>
        <select
            class={selectCls}
            value={solverKind}
            onchange={(e) =>
                onChange({
                    solverKind: Number(
                        (e.currentTarget as HTMLSelectElement).value,
                    ) as SolverKind,
                })}
        >
            {#each SOLVERS as s}
                <option value={s.kind}>{s.label}</option>
            {/each}
        </select>
    </label>

    <!-- Solver-specific options, rendered from the solver's schema. -->
    {#each solverOptions as opt (opt.id)}
        {#if opt.kind === "select"}
            <label class="flex flex-col gap-1">
                <span class={labelCls}>{opt.label}</span>
                <select
                    class={selectCls}
                    value={String(optionValues[opt.id] ?? opt.default)}
                    onchange={(e) =>
                        onOptionChange(
                            opt.id,
                            (e.currentTarget as HTMLSelectElement).value,
                        )}
                >
                    {#each opt.choices as c}
                        <option value={c.value}>{c.label}</option>
                    {/each}
                </select>
            </label>
        {:else if opt.kind === "logSlider"}
            {#if visible(opt)}
                <label class="flex flex-col gap-1">
                    <span class={labelCls}
                        >{opt.label}:
                        <span class={valueCls}
                            >{Number(
                                optionValues[opt.id] ?? opt.default,
                            ).toExponential(2)}</span
                        ></span
                    >
                    <input
                        type="range"
                        min={opt.min}
                        max={opt.max}
                        step={opt.step}
                        value={Math.log10(
                            Number(optionValues[opt.id] ?? opt.default),
                        )}
                        oninput={(e) =>
                            onOptionChange(
                                opt.id,
                                Math.pow(
                                    10,
                                    Number(
                                        (e.currentTarget as HTMLInputElement)
                                            .value,
                                    ),
                                ),
                            )}
                    />
                </label>
            {/if}
        {:else if opt.kind === "intSlider"}
            <label class="flex flex-col gap-1">
                <span class={labelCls}
                    >{opt.label}:
                    <span class={valueCls}
                        >{Number(optionValues[opt.id] ?? opt.default)}</span
                    ></span
                >
                <input
                    type="range"
                    min={opt.min}
                    max={opt.max}
                    step={opt.step}
                    value={Number(optionValues[opt.id] ?? opt.default)}
                    oninput={(e) =>
                        onOptionChange(
                            opt.id,
                            Number((e.currentTarget as HTMLInputElement).value),
                        )}
                />
            </label>
        {:else if opt.kind === "linearSlider"}
            <label class="flex flex-col gap-1">
                <span class={labelCls}
                    >{opt.label}:
                    <span class={valueCls}
                        >{Number(optionValues[opt.id] ?? opt.default).toFixed(
                            2,
                        )}</span
                    ></span
                >
                <input
                    type="range"
                    min={opt.min}
                    max={opt.max}
                    step={opt.step}
                    value={Number(optionValues[opt.id] ?? opt.default)}
                    oninput={(e) =>
                        onOptionChange(
                            opt.id,
                            Number((e.currentTarget as HTMLInputElement).value),
                        )}
                />
            </label>
        {:else if opt.kind === "seedField"}
            <label class="flex flex-col gap-1">
                <span class={labelCls}>{opt.label}</span>
                <span class="flex gap-2 items-center">
                    <input
                        type="number"
                        min="0"
                        step="1"
                        class={`${selectCls} flex-1 font-mono`}
                        value={Number(optionValues[opt.id] ?? opt.default)}
                        oninput={(e) =>
                            onOptionChange(
                                opt.id,
                                Math.max(
                                    0,
                                    Math.floor(
                                        Number(
                                            (
                                                e.currentTarget as HTMLInputElement
                                            ).value,
                                        ),
                                    ),
                                ),
                            )}
                    />
                    <button
                        type="button"
                        class="px-2 py-1 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700"
                        title="Reroll seed"
                        onclick={() =>
                            onOptionChange(
                                opt.id,
                                Math.floor(Math.random() * 2147483647),
                            )}>🎲</button
                    >
                </span>
            </label>
        {/if}
    {/each}

    <label class="flex flex-col gap-1">
        <span class={labelCls}
            >Max iterations: <span class={valueCls}>{maxIter}</span></span
        >
        <input
            type="range"
            min="20"
            max="2000"
            step="20"
            value={maxIter}
            oninput={(e) =>
                onChange({
                    maxIter: Number(
                        (e.currentTarget as HTMLInputElement).value,
                    ),
                })}
        />
    </label>

    <p class="text-slate-600 dark:text-slate-400 text-xs leading-relaxed">
        Click anywhere on the contour plot to reset the starting point. The
        solver re-runs immediately. Current start: <span
            class="font-mono text-slate-800 dark:text-slate-200"
            >({startPoint.x.toFixed(2)}, {startPoint.y.toFixed(2)})</span
        >
    </p>
</div>
