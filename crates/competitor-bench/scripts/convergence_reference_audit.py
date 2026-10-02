#!/usr/bin/env python3
"""Independently audit the three NIST models used by the policy pilot.

Use decimal observations and 60/90-digit arithmetic, with exact binary returned
parameters. Certified half-RSS rounding intervals come from the printed decimal
precision. These empirical checks are not general certified precision floors.
"""

import argparse
import ast
import csv
import struct
from decimal import Decimal, localcontext
from pathlib import Path

NAMES = ("Misra1a", "Chwirut2", "DanWood")


def dataset(name):
    text = Path(f"crates/competitor-bench/data/nist/{name}.dat").read_text()
    rss = Decimal(
        next(
            line.split(":")[1].strip()
            for line in text.splitlines()
            if line.startswith("Residual Sum of Squares:")
        )
    )
    rounding = Decimal(10) ** rss.as_tuple().exponent / 4
    data = [
        [Decimal(v) for v in line.split()]
        for line in text.rsplit("Data:", 1)[1].splitlines()[1:]
        if line.strip()
    ]
    return data, rss / 2, rounding


def initial_param(row):
    text = Path(f"crates/competitor-bench/data/nist/{row['case']}.dat").read_text()
    param = []
    for line in text.splitlines():
        words = line.split()
        if len(words) >= 6 and words[0].startswith("b") and words[1] == "=":
            value = float(words[int(row["start"]) + 1])
            if row["precision"] == "f32":
                value = struct.unpack("f", struct.pack("f", value))[0]
            param.append(value)
    return param


def cost(name, param, precision):
    with localcontext() as context:
        context.prec = precision
        data, _, _ = dataset(name)
        p = [Decimal.from_float(float(v)) for v in param]
        total = Decimal(0)
        for y, x in data:
            if name == "Misra1a":
                prediction = p[0] * (1 - (-p[1] * x).exp())
            elif name == "Chwirut2":
                prediction = (-p[0] * x).exp() / (p[1] + p[2] * x)
            else:
                prediction = p[0] * (p[1] * x.ln()).exp()
            total += (y - prediction) ** 2 / 2
        return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("finals", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.finals.open() as stream:
        rows = list(csv.DictReader(stream))
    output = []
    demonstrations = {}
    for row in rows:
        if row["case"] not in NAMES:
            continue
        param = ast.literal_eval(row["returned_param"])
        low, high = (cost(row["case"], param, precision) for precision in (60, 90))
        if abs(low - high) > max(abs(high), Decimal(1)) * Decimal("1e-50"):
            raise ValueError("decimal precision disagreement")
        _, reference, uncertainty = dataset(row["case"])
        difference = high - reference
        binary_error = abs(Decimal(row["checked_cost"]) - high)
        if row["policy"] == "strict":
            initial = cost(row["case"], initial_param(row), 90)
            scale_lower = abs(initial - reference) - uncertainty
            if scale_lower <= 0:
                raise ValueError("initial gap does not resolve reference rounding")
            upper_relative_gap = max(Decimal(0), difference + uncertainty) / scale_lower
            key = tuple(row[k] for k in ("case", "precision", "start"))
            previous = demonstrations.get(key)
            if previous is None or upper_relative_gap < previous[0]:
                demonstrations[key] = (
                    upper_relative_gap,
                    row["solver"],
                    uncertainty / scale_lower,
                )
        output.append(
            {k: row[k] for k in ("case", "precision", "start", "solver", "policy")}
            | {
                "decimal_cost": str(high),
                "reference_cost": str(reference),
                "reference_rounding": str(uncertainty),
                "decimal_difference": str(difference),
                "binary_reevaluation_error": str(binary_error),
                "negative_beyond_reference_rounding": difference < -uncertainty,
                "decimal_precision_difference": str(abs(low - high)),
            }
        )
    if not output:
        raise ValueError("no supported NIST rows")
    with args.output.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    target_rows = []
    if not demonstrations:
        raise ValueError("target audit requires strict-control rows")
    invalid_references = {
        row["case"] for row in output if row["negative_beyond_reference_rounding"]
    }
    for (name, precision, start), (upper, solver, resolution) in sorted(
        demonstrations.items()
    ):
        grid = (
            ("1e-2", "1e-3", "1e-4")
            if precision == "f32"
            else ("1e-2", "1e-4", "1e-6", "1e-8")
        )
        for rho in grid:
            target_rows.append(
                {
                    "case": name,
                    "precision": precision,
                    "start": start,
                    "rho": rho,
                    "reference_relative_rounding": str(resolution),
                    "strict_upper_relative_gap": str(upper),
                    "strict_solver": solver,
                    "demonstrated": name not in invalid_references
                    and upper <= Decimal(rho)
                    and resolution < Decimal(rho),
                }
            )
    target_path = args.output.with_name(args.output.stem + "-targets.csv")
    with target_path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(target_rows[0]))
        writer.writeheader()
        writer.writerows(target_rows)
    print(
        f"Audited {len(output)} rows; "
        f"{sum(r['negative_beyond_reference_rounding'] for r in output)} "
        "negative discrepancies beyond printed reference rounding."
    )


if __name__ == "__main__":
    main()
