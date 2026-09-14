"""Summarize paired timing rounds and the derivative-cost crossover model."""

import argparse
import csv
import json
import random
import statistics as stats
from pathlib import Path


def read(directory, filename):
    with (directory / filename).open() as stream:
        return list(csv.DictReader(stream))


def percentile(values, fraction):
    values = sorted(values)
    index = fraction * (len(values) - 1)
    low = int(index)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (
        index - low
    )


def interval(values):
    rng = random.Random(3751)
    medians = [stats.median(rng.choices(values, k=len(values))) for _ in range(10000)]
    return [percentile(medians, 0.025), percentile(medians, 0.975)]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()
diagnostics = read(args.directory, "diagnostics.csv")
timings = read(args.directory, "timings.csv")
calibration = read(args.directory, "calibration.csv")
summary = {"diagnostic_runs": len(diagnostics), "problems": {}}
for problem in dict.fromkeys(row["problem"] for row in diagnostics):
    by_backend = {
        backend: {
            row["start"]: row
            for row in diagnostics
            if row["problem"] == problem
            and row["mode"] == "baseline"
            and row["backend"] == backend
        }
        for backend in ["argmin", "basin"]
    }
    common = [
        start
        for start in by_backend["argmin"]
        if all(rows[start]["status"] == "target" for rows in by_backend.values())
    ]
    data = {
        "shared_starts": [int(start) for start in common],
        "successes": {
            backend: sum(row["status"] == "target" for row in rows.values())
            for backend, rows in by_backend.items()
        },
        "counts": {
            backend: {
                kind: {
                    "mean": stats.mean(int(rows[start][kind]) for start in common),
                    "median": stats.median(int(rows[start][kind]) for start in common),
                }
                for kind in ["cost_evals", "gradient_evals", "hessian_evals"]
            }
            for backend, rows in by_backend.items()
        },
        "sweeps": {},
    }
    summary["problems"][problem] = data
    print(f"{problem}: success {data['successes']}, shared {len(common)}")
    for mode in ["hessian", "both"]:
        subset = [
            row for row in timings if row["problem"] == problem and row["mode"] == mode
        ]
        if not subset:
            continue
        sweep = []
        for units in sorted({int(row["units"]) for row in subset}):
            samples = {
                backend: {
                    int(row["round"]): float(row["seconds_per_solve"]) * 1e6
                    for row in subset
                    if row["backend"] == backend and int(row["units"]) == units
                }
                for backend in by_backend
            }
            assert samples["argmin"].keys() == samples["basin"].keys()
            ratios = [
                samples["argmin"][r] / samples["basin"][r] for r in samples["argmin"]
            ]
            costs = [
                float(row["seconds_per_call"]) * 1e6
                for row in calibration
                if row["problem"] == problem
                and row["mode"] == mode
                and int(row["units"]) == units
            ]
            sweep.append(
                {
                    "units": units,
                    "added_us": stats.median(costs) if units else 0,
                    "added_us_iqr": [percentile(costs, 0.25), percentile(costs, 0.75)],
                    "median_us": {
                        backend: stats.median(values.values())
                        for backend, values in samples.items()
                    },
                    "iqr_us": {
                        backend: [
                            percentile(list(values.values()), q) for q in [0.25, 0.75]
                        ]
                        for backend, values in samples.items()
                    },
                    "paired_ratio": stats.median(ratios),
                    "paired_ratio_ci95": interval(ratios),
                }
            )
        counts = {
            backend: data["counts"][backend]["hessian_evals"]["mean"]
            + (
                data["counts"][backend]["gradient_evals"]["mean"]
                if mode == "both"
                else 0
            )
            for backend in by_backend
        }
        difference = counts["basin"] - counts["argmin"]
        advantage = sweep[0]["median_us"]["argmin"] - sweep[0]["median_us"]["basin"]
        crossover = advantage / difference if difference else None
        for row in sweep:
            row["predicted_us"] = {
                backend: sweep[0]["median_us"][backend]
                + counts[backend] * row["added_us"]
                for backend in by_backend
            }
            row["modeled_added_work_fraction"] = {
                backend: counts[backend] * row["added_us"] / row["median_us"][backend]
                for backend in by_backend
            }
        data["sweeps"][mode] = {"modeled_crossover_us": crossover, "samples": sweep}
        print(f"  {mode}: modeled crossover {crossover} us")
        for row in sweep:
            print(
                f"    {row['units']:5} {row['added_us']:8.3f} us/callback "
                f"A {row['median_us']['argmin']:9.3f} B {row['median_us']['basin']:9.3f} us/solve "
                f"ratio {row['paired_ratio']:.3f} CI {row['paired_ratio_ci95']}"
            )

durations = [float(row["sample_seconds"]) for row in timings]
summary["sample_seconds"] = {
    "min": min(durations),
    "median": stats.median(durations),
    "max": max(durations),
}
(args.directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
if args.plot:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, problem, title in zip(
        axes,
        ["rosenbrock2", "rosenbrock20", "sphere20"],
        ["Rosenbrock, 2D", "Rosenbrock, 20D", "Sphere, 20D"],
    ):
        for mode, label, color in [
            ("hessian", "Hessian only", "#4f46e5"),
            ("both", "Gradient and Hessian", "#d97706"),
        ]:
            rows = summary["problems"][problem]["sweeps"][mode]["samples"]
            x = [row["added_us"] for row in rows]
            y = [row["paired_ratio"] for row in rows]
            ax.plot(x, y, "o-", color=color, label=label, markersize=4)
            ax.fill_between(
                x,
                [row["paired_ratio_ci95"][0] for row in rows],
                [row["paired_ratio_ci95"][1] for row in rows],
                color=color,
                alpha=0.15,
            )
        ax.axhline(1, color="gray", linestyle="--", linewidth=1)
        ax.set_xscale("symlog", linthresh=0.03)
        ax.set_xlim(0, 40)
        ax.set_xticks([0, 0.1, 1, 10], labels=["0", "0.1", "1", "10"])
        ax.set_title(title)
        ax.set_xlabel("Added cost per callback (µs)")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("argmin time / Basin time\nAbove 1 favors Basin")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Steihaug: derivative cost reverses the Rosenbrock timing advantage")
    fig.tight_layout()
    fig.savefig(args.directory / "crossover.svg")
    fig.savefig(args.directory / "crossover.png", dpi=180)
