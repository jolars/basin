#!/usr/bin/env python3
"""Compare prebuilt, uninstrumented COBYLA probes from reproduce.py."""

import argparse
import csv
import json
import math
import random
import statistics
import subprocess

CASES = {"camel": 5000, "sphere": 400, "quadratic": 3000}
MODES = [
    "old",
    "raw",
    "manual",
    "basin",
    "projected",
    "old-adapter",
    "new-adapter",
    "current-adapter",
]


def run(binary, mode, case, repeats, cpu):
    command = [str(binary), mode, case, str(repeats)]
    if cpu is not None:
        command = ["taskset", "-c", str(cpu), *command]
    output = subprocess.check_output(command, text=True).strip()
    fields = output.split(",", 8)
    if len(fields) != 9 or fields[0] != mode or fields[1] != case:
        raise ValueError(f"Expected an uninstrumented probe: {output}")
    return fields


def verify(fields):
    case = fields[1]
    point = json.loads(fields[8])
    assert all(math.isfinite(x) for x in point)
    if case == "camel":
        x, y = point
        cost = (4 - 2.1 * x**2 + x**4 / 3) * x**2 + x * y + (-4 + 4 * y**2) * y**2
        violation = max(0, abs(x) - 3, abs(y) - 2)
        target, tolerance = -1.031628453489877, 1e-5
    elif case == "sphere":
        cost = sum(x * x for x in point)
        violation = max(0, max(abs(x) - 5 for x in point))
        target, tolerance = 0, 1e-5
    else:
        x, y = point
        cost = (x - 1) ** 2 + (y - 1) ** 2
        violation = max(0, -x, x - 2, -y, y - 2, x + y - 1.5)
        target, tolerance = 0.125, 1e-6
    assert math.isclose(cost, float(fields[6]), rel_tol=1e-12, abs_tol=1e-14)
    assert abs(cost - target) < tolerance
    assert violation <= math.sqrt(math.ulp(1.0))
    assert 0 < int(fields[3]) <= {"camel": 50, "sphere": 200, "quadratic": 100}[case]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error("--rounds must be at least 2")

    # Validate every layer outside timing and require unchanged numerical work.
    for case in CASES:
        for mode in MODES:
            before = run(args.before, mode, case, 1, args.cpu)
            after = run(args.after, mode, case, 1, args.cpu)
            verify(before)
            verify(after)
            assert before[3:] == after[3:], (before, after)

    rng = random.Random(42)
    rows = []
    with open(args.output, "w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(
            [
                "round",
                "revision",
                "mode",
                "case",
                "ns",
                "objective_calls",
                "constraint_calls",
                "iterations",
                "objective",
                "violation",
                "point",
            ]
        )
        for sample in range(args.rounds):
            cases = list(CASES)
            rng.shuffle(cases)
            for case in cases:
                jobs = [("reference", args.before, "old")]
                jobs += [
                    (revision, binary, mode)
                    for mode in ["raw", "basin", "current-adapter"]
                    for revision, binary in [
                        ("before", args.before),
                        ("after", args.after),
                    ]
                ]
                rng.shuffle(jobs)
                for revision, binary, mode in jobs:
                    fields = run(binary, mode, case, CASES[case], args.cpu)
                    verify(fields)
                    writer.writerow([sample, revision, *fields])
                    rows.append((case, mode, revision, float(fields[2]) / 1000))
            output.flush()

    for key in sorted({row[:3] for row in rows}):
        times = [row[3] for row in rows if row[:3] == key]
        quartiles = statistics.quantiles(times, n=4, method="inclusive")
        print(
            *key,
            f"{statistics.median(times):.2f} us; IQR {quartiles[0]:.2f}-{quartiles[2]:.2f}",
        )


if __name__ == "__main__":
    main()
