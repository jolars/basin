#!/usr/bin/env python3
"""Serial development probes and descriptive summaries; no default selection.

Run from the repository root. Build convergence_policies first. Each phase
creates a new directory, so repeating a measurement cannot overwrite evidence.
The trace gaps are observations, not success scores: reference uncertainties
and attainable precision floors still need an independent audit.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

POLICIES = ("default", "probe", "strict")
MODES = ("plain", "final", "trace")
KEY = ("case", "precision", "start", "solver")
MATCH = (
    "iteration",
    "termination",
    "report",
    "returned_param",
    "native_cost",
    "cost_evals",
    "gradient_evals",
    "residual_evals",
    "jacobian_evals",
)


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command_output(*args):
    return subprocess.check_output(args, text=True).strip()


def run(args):
    directory = args.directory / args.phase
    directory.mkdir(parents=True, exist_ok=False)
    binary = Path(args.binary).resolve()
    environment = os.environ.copy()
    threads = {
        name: "1"
        for name in (
            "RAYON_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
        )
    }
    environment.update(threads)
    base = [str(binary), "--warmup", "1", "--repetitions", "1"]
    if args.phase == "controls":
        configurations = [
            ["--policy", p, "--mode", m, "--max-iter", "2000"]
            for p, m in itertools.product(POLICIES, MODES)
        ]
        repetitions = args.rounds or 7
    elif args.phase == "traces":
        configurations = [
            ["--policy", p, "--mode", "trace", "--emit-trace", "--max-iter", "2000"]
            for p in POLICIES
        ]
        repetitions = args.rounds or 1
    else:
        configurations = [
            [
                "--policy",
                "strict",
                "--mode",
                "trace",
                "--emit-trace",
                "--seconds",
                str(b),
                "--max-iter",
                "100000",
            ]
            for b in args.budgets
        ]
        repetitions = args.rounds or 3
    manifest = {
        "phase": args.phase,
        "status": "running",
        "revision": command_output("git", "rev-parse", "HEAD"),
        "dirty": command_output("git", "status", "--porcelain"),
        "binary": str(binary),
        "binary_sha256": digest(binary),
        "lock_sha256": digest("Cargo.lock"),
        "compiler": command_output("rustc", "--version"),
        "system": platform.platform(),
        "cpu": command_output("lscpu"),
        "affinity": sorted(os.sched_getaffinity(0)),
        "threads": threads,
        "rounds": repetitions,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "processes": command_output("ps", "-eo", "pid,comm,pcpu", "--sort=-pcpu"),
        "runs": [],
    }
    manifest_path = directory / "manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    for block in range(repetitions):
        # Rotation and reversal distribute launch-order effects across policies.
        offset = block % len(configurations)
        order = configurations[offset:] + configurations[:offset]
        if block % 2:
            order = list(reversed(order))
        for index, configuration in enumerate(order):
            path = directory / f"block-{block + 1:02}-{index + 1:02}.csv"
            command = base + configuration
            record = {"block": block + 1, "command": command, "output": str(path)}
            manifest["runs"].append(record)
            save()
            print(f"{args.phase} {path.name}: {' '.join(configuration)}", flush=True)
            with path.open("w") as stream:
                result = subprocess.run(
                    command,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
            record.update(
                returncode=result.returncode, stderr=result.stderr, sha256=digest(path)
            )
            save()
            if result.returncode:
                raise RuntimeError(record)
    manifest.update(
        status="complete", finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    save()


def quantiles(values):
    if len(values) == 1:
        return [values[0]] * 3
    q = statistics.quantiles(values, n=4, method="inclusive")
    return [q[0], statistics.median(values), q[2]]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checked_rows(path):
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"malformed CSV: {path}")
            if row["partition"] != "development":
                raise ValueError(f"validation leakage: {path}")
            yield row


def relative_gap(row):
    scale = abs(float(row["initial_cost"]) - float(row["reference_cost"]))
    return float(row["cost_difference"]) / scale if scale > 0 else math.nan


def summarize(args):
    directory = args.directory
    controls = {}
    for path in sorted((directory / "controls").glob("*.csv")):
        for row in checked_rows(path):
            key = tuple(row[k] for k in KEY) + (row["policy"],)
            controls.setdefault(key, {}).setdefault(row["mode"], []).append(row)
    comparisons = []
    for key, modes in sorted(controls.items()):
        signatures = {
            tuple(r[k] for k in MATCH) for rows in modes.values() for r in rows
        }
        matched = len(signatures) == 1 and set(modes) == set(MODES)
        if not matched:
            raise ValueError(f"recorded and plain work differs: {key}")
        row = dict(zip(KEY + ("policy",), key))
        row["matched"] = matched
        for mode in MODES:
            rows = modes[mode]
            row[f"{mode}_n"] = len(rows)
            for field in ("solver_seconds", "harness_seconds"):
                for label, value in zip(
                    ("q1", "median", "q3"), quantiles([float(r[field]) for r in rows])
                ):
                    row[f"{mode}_{field}_{label}"] = value
        denominator = row["plain_solver_seconds_median"]
        for mode in ("trace", "final"):
            row[f"{mode}_charged_ratio"] = (
                row[f"{mode}_solver_seconds_median"] / denominator
            )
            row[f"{mode}_harness_ratio"] = (
                row[f"{mode}_harness_seconds_median"]
                / row["plain_harness_seconds_median"]
            )
        comparisons.append(row)
    write_csv(directory / "overhead.csv", comparisons)

    finals, budget_rows, crossings = [], [], []
    for phase in ("traces", "budgets"):
        for path in sorted((directory / phase).glob("*.csv")):
            group, eligible, previous_seconds, previous_passes, hit = (
                None,
                None,
                0.0,
                0,
                set(),
            )
            for row in checked_rows(path):
                key = tuple(row[k] for k in KEY) + (row["policy"],)
                if key != group:
                    group, eligible, previous_seconds, previous_passes, hit = (
                        key,
                        None,
                        0.0,
                        0,
                        set(),
                    )
                seconds = float(row["solver_seconds"])
                passes = int(row["value_passes"]) + int(row["derivative_passes"])
                if seconds < previous_seconds or passes < previous_passes:
                    raise ValueError(f"nonmonotonic trace: {path}, {key}")
                previous_seconds, previous_passes = seconds, passes
                limit = float(row["time_limit_seconds"] or "inf")
                if seconds <= limit:
                    eligible = row
                gap = relative_gap(row)
                # Keep negative discrepancies separate; these are unscored grid
                # crossings, not verified success or attainable target labels.
                grid = (
                    (1e-2, 1e-3, 1e-4)
                    if row["precision"] == "f32"
                    else (1e-2, 1e-4, 1e-6, 1e-8)
                )
                if phase == "traces":
                    for rho in grid:
                        if rho not in hit and math.isfinite(gap) and 0 <= gap <= rho:
                            hit.add(rho)
                            crossings.append(
                                dict(zip(KEY + ("policy",), key))
                                | {
                                    "rho": rho,
                                    "iteration": row["iteration"],
                                    "seconds": seconds,
                                    "value_passes": row["value_passes"],
                                    "derivative_passes": row["derivative_passes"],
                                    "relative_gap": gap,
                                    "status": "observed_crossing_unscored",
                                }
                            )
                if row["termination"]:
                    if phase == "traces":
                        finals.append(row | {"relative_gap": gap})
                    else:
                        budget_rows.append(
                            dict(zip(KEY + ("policy",), key))
                            | {
                                "source": path.name,
                                "limit": limit,
                                "actual_seconds": seconds,
                                "termination": row["termination"],
                                "report": row["report"],
                                "time_limit_exceeded": row["time_limit_exceeded"],
                                "overshoot_seconds": max(0, seconds - limit),
                                "max_segment_seconds": row["max_segment_seconds"],
                                "returned_relative_gap": gap,
                                "last_within_budget_seconds": eligible["solver_seconds"]
                                if eligible
                                else "",
                                "last_within_budget_relative_gap": relative_gap(
                                    eligible
                                )
                                if eligible
                                else "",
                                "last_within_budget_value_passes": eligible[
                                    "value_passes"
                                ]
                                if eligible
                                else "",
                                "last_within_budget_derivative_passes": eligible[
                                    "derivative_passes"
                                ]
                                if eligible
                                else "",
                            }
                        )
    write_csv(directory / "finals.csv", finals)
    write_csv(directory / "crossings.csv", crossings)
    write_csv(directory / "budget-boundaries.csv", budget_rows)
    if finals and comparisons:
        controls = {
            tuple(row[k] for k in KEY + ("policy",)): row for row in comparisons
        }
        keep = (
            "case",
            "family",
            "precision",
            "start",
            "solver",
            "policy",
            "policy_settings",
            "iteration",
            "termination",
            "report",
            "value_passes",
            "derivative_passes",
            "cost_difference",
            "relative_gap",
            "gradient_inf",
        )
        compact = []
        for row in finals:
            control = controls[tuple(row[k] for k in KEY + ("policy",))]
            compact.append(
                {k: row[k] for k in keep}
                | {
                    f"plain_seconds_{q}": control[f"plain_solver_seconds_{q}"]
                    for q in ("q1", "median", "q3")
                }
                | {
                    k: control[k]
                    for k in ("trace_charged_ratio", "trace_harness_ratio")
                }
            )
        write_csv(directory / "results.csv", compact)
    print(
        json.dumps(
            {
                "matched_comparisons": len(comparisons),
                "finals": len(finals),
                "observed_crossings": len(crossings),
                "budget_runs": len(budget_rows),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("controls", "traces", "budgets", "summarize"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--binary", default="target/release/convergence_policies")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--budgets", type=float, nargs="+", default=(1e-4, 1e-3, 1e-2))
    args = parser.parse_args()
    if args.rounds is not None and args.rounds < 1:
        parser.error("rounds must be positive")
    if any(not math.isfinite(b) or b <= 0 for b in args.budgets):
        parser.error("budgets must be finite and positive")
    if args.phase == "summarize":
        summarize(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
