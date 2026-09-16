#!/usr/bin/env python3
"""Alternate prebuilt SLSQP probes and summarize their batch averages."""

import argparse
import csv
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        for round_index in range(args.rounds):
            order = ["before", "after"]
            if round_index % 2:
                order.reverse()
            for name in order:
                prefix = args.output / f"{name}-{round_index}"
                with prefix.with_suffix(".csv").open("w") as output:
                    with prefix.with_suffix(".verify").open("w") as verification:
                        subprocess.run(
                            [
                                "taskset", "-c", str(args.cpu),
                                str(getattr(args, name).resolve()),
                            ],
                            stdout=output,
                            stderr=verification,
                            check=True,
                        )
    results = {}
    for name in ["before", "after"]:
        samples = defaultdict(list)
        for round_index in range(args.rounds):
            with (args.output / f"{name}-{round_index}.csv").open() as source:
                for row in csv.DictReader(source):
                    samples[row["case"]].append(int(row["ns"]))
        results[name] = {}
        for case, values in samples.items():
            quartiles = statistics.quantiles(values, n=4)
            results[name][case] = (
                statistics.median(values), quartiles[0], quartiles[2]
            )
    assert results["before"].keys() == results["after"].keys()
    rows = [
        "| Case | Before, µs (IQR) | After, µs (IQR) | Speedup |",
        "| --- | ---: | ---: | ---: |",
    ]
    for case, before in results["before"].items():
        after = results["after"][case]
        timings = [
            f"{v[0] / 1000:.3f} ({v[1] / 1000:.3f}–{v[2] / 1000:.3f})"
            for v in [before, after]
        ]
        rows.append(
            f"| {case} | {timings[0]} | {timings[1]} | {before[0] / after[0]:.2f}× |"
        )
    table = "\n".join(rows) + "\n"
    (args.output / "table.md").write_text(table)
    print(table, end="")


if __name__ == "__main__":
    main()
