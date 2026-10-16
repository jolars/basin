#!/usr/bin/env python3
"""Summarize accuracy and paired timings without discarding failed cases."""

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

BASELINE = "nielsen-cholesky"
CONFIGURATIONS = (BASELINE, "nielsen-qr", "trust-cholesky", "trust-qr")


def read_csv(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if any(None in row for row in rows):
        raise ValueError(f"{path}: malformed CSV row")
    return rows


def case_key(row):
    return row["backend"], row["scalar"], row["dataset"], row["start"]


def recovered(row, threshold):
    return row[f"recovered_{threshold}"] == "true"


def confidence(values, statistic=statistics.median):
    if not values:
        return None
    generator = random.Random(1729)
    samples = sorted(
        statistic(generator.choices(values, k=len(values)))
        for _ in range(1000)
    )
    return [samples[24], samples[974]]


def accuracy(rows):
    indexed = {}
    groups = defaultdict(list)
    for row in rows:
        key = (*case_key(row), row["configuration"])
        if key in indexed:
            raise ValueError(f"duplicate accuracy result: {key}")
        indexed[key] = row
        groups[(row["backend"], row["scalar"], row["configuration"])].append(row)
    summaries = []
    for (backend, scalar, configuration), group in sorted(groups.items()):
        summary = {
            "backend": backend,
            "scalar": scalar,
            "configuration": configuration,
            "runs": len(group),
            "terminations": dict(Counter(row["termination"] for row in group)),
            "residual_evaluations": sum(int(row["residual_evals"] or 0) for row in group),
            "jacobian_evaluations": sum(int(row["jacobian_evals"] or 0) for row in group),
        }
        if any(row["model_solves"] for row in group):
            for column in ("model_solves", "retries", "accepted", "rejected"):
                summary[column] = sum(int(row[column] or 0) for row in group)
        for threshold in ("1e3", "1e4", "1e6"):
            summary[f"recovered_{threshold}"] = sum(recovered(row, threshold) for row in group)
            tolerance = {"1e3": 1e-3, "1e4": 1e-4, "1e6": 1e-6}[threshold]
            summary[f"fit_{threshold}"] = sum(float(row["rss_relative_error"]) <= tolerance for row in group)
            gained, lost = [], []
            for row in group:
                baseline = indexed.get((*case_key(row), BASELINE))
                if baseline is None:
                    continue
                label = f"{row['dataset']}/start{row['start']}"
                if recovered(row, threshold) and not recovered(baseline, threshold):
                    gained.append(label)
                if recovered(baseline, threshold) and not recovered(row, threshold):
                    lost.append(label)
            summary[f"gained_{threshold}"] = gained
            summary[f"lost_{threshold}"] = lost
        summaries.append(summary)
    return indexed, summaries


def timings(rows, verified):
    samples = defaultdict(dict)
    for row in rows:
        key = (*case_key(row), row["configuration"])
        round_number = int(row["round"])
        if round_number in samples[key]:
            raise ValueError(f"duplicate timing round: {key}, {round_number}")
        if key not in verified:
            raise ValueError(f"timing lacks an independent accuracy result: {key}")
        value = float(row["ns_per_solve"])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"invalid timing: {key}")
        samples[key][round_number] = value
    medians = {key: statistics.median(values.values()) for key, values in samples.items()}
    case_summaries = []
    grouped = defaultdict(list)
    for key, values in sorted(samples.items()):
        backend, scalar, dataset, start, configuration = key
        baseline = samples.get((*key[:-1], BASELINE))
        if baseline is None:
            continue
        if values.keys() != baseline.keys():
            raise ValueError(f"timing rounds differ from the baseline: {key}")
        ratios = [values[round_number] / baseline[round_number] for round_number in sorted(values)]
        threshold = "1e3" if scalar == "f32" else "1e4"
        summary = {
            "backend": backend, "scalar": scalar, "dataset": dataset,
            "start": start, "configuration": configuration,
            "samples": len(values), "median_ns": medians[key],
            "median_ns_ci95": confidence(list(values.values())),
            "ratio_to_baseline": statistics.median(ratios),
            "paired_ratio_ci95": confidence(ratios),
            "accurate": recovered(verified[key], threshold),
            "baseline_accurate": recovered(verified[(*key[:-1], BASELINE)], threshold),
        }
        case_summaries.append(summary)
        grouped[(backend, scalar, configuration)].append(summary)
    comparisons = []
    for (backend, scalar, configuration), cases in sorted(grouped.items()):
        common = [case for case in cases if case["accurate"] and case["baseline_accurate"]]
        # Resample datasets together with their starts, rather than treating
        # the two starts as independent datasets in the aggregate interval.
        dataset_logs = defaultdict(list)
        for case in common:
            dataset_logs[case["dataset"]].append(math.log(case["ratio_to_baseline"]))
        values = list(dataset_logs.values())

        def geometric_ratio(datasets):
            return math.exp(statistics.mean(value for dataset in datasets for value in dataset))

        comparison = {
            "backend": backend, "scalar": scalar, "configuration": configuration,
            "timed_cases": len(cases), "jointly_accurate_cases": len(common),
            "geometric_ratio_on_joint_success": geometric_ratio(values) if values else None,
            "dataset_bootstrap_ci95": confidence(values, geometric_ratio),
            "all_case_geometric_ratio": math.exp(statistics.mean(math.log(case["ratio_to_baseline"]) for case in cases)),
        }
        profile = {}
        for factor in (1, 1.25, 1.5, 2, 5, 10):
            count = 0
            for case in cases:
                key = (backend, scalar, case["dataset"], case["start"])
                threshold = "1e3" if scalar == "f32" else "1e4"
                accurate_times = [
                    medians[(*key, candidate)] for candidate in CONFIGURATIONS
                    if (*key, candidate) in medians and recovered(verified[(*key, candidate)], threshold)
                ]
                if case["accurate"] and accurate_times and case["median_ns"] <= factor * min(accurate_times):
                    count += 1
            profile[str(factor)] = {"accurate_cases_within_factor_of_best": count, "total_cases": len(cases)}
        comparison["failure_inclusive_profile"] = profile
        comparisons.append(comparison)
    return case_summaries, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--timings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_csv(args.results)
    if not rows:
        parser.error("no accuracy rows")
    indexed, summaries = accuracy(rows)
    result = {"accuracy_rows": len(rows), "accuracy": summaries}
    if args.timings:
        cases, comparisons = timings(read_csv(args.timings), indexed)
        result.update(timing_cases=cases, timing_comparisons=comparisons)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(summaries)} configuration summaries to {args.output}")


if __name__ == "__main__":
    main()
