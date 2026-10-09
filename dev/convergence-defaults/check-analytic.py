"""Validate schema 1 analytic traces without trusting the CLI's success exit."""

import csv
import hashlib
import json
import re
import sys
from pathlib import Path


def rows(path):
    with path.open(newline="") as stream:
        result = list(csv.DictReader(stream))
    assert all(None not in row and None not in row.values() for row in result), path
    assert all(row["schema"] == "1" for row in result), path
    return result


def main():
    directory = Path(sys.argv[1])
    summaries = rows(directory / "summary.csv")
    expected = {
        "nm-f64-paired", "nm-f32-paired", "lbfgsb-f64-default",
        "lbfgsb-f32-default", "lbfgsb-f64-central", "lbfgsb-f64-forward",
        "nm-f64-default", "nm-f64-init-cap", "nm-f64-step-cap",
    }
    assert {row["case"] for row in summaries} == expected
    assert len(summaries) == len(expected)
    retained = []
    for summary in summaries:
        name = summary["case"]
        leaves = rows(directory / f"{name}-leaves.csv")
        recommendations = rows(directory / f"{name}-recommendations.csv")
        cap, work = int(summary["cap"]), int(summary["physical_work"])
        assert work == len(leaves) <= cap, name
        assert [int(row["work"]) for row in leaves] == list(range(1, work + 1))
        assert all(row["outcome"] == "Completed" for row in leaves), name
        published_work = [int(row["work"]) for row in recommendations]
        assert published_work == sorted(published_work), name
        assert all(w <= work for w in published_work), name
        sampled = sum(row["sampled_solver_cost"] != "None" for row in leaves)
        assert int(summary["verification_calls"]) == sampled + len(recommendations)
        attained = [int(row["work"]) for row in recommendations if row["target_status"] == "Passed"]
        assert summary["first_attainment_work"] == (f"Some({min(attained)})" if attained else "None")
        if name == "nm-f64-init-cap":
            assert not recommendations and summary["logical_counts"] == "None"
            assert summary["outcome"].startswith("InitializationError(Budget")
            assert summary["returned_target_status"] == "None"
        elif name in {"nm-f64-step-cap", "nm-f64-default"}:
            assert summary["outcome"].startswith("StepError(Budget")
            assert summary["returned_target_status"] == "None"
            assert work == cap and int(summary["denied"]) == 1
        else:
            assert summary["outcome"].startswith("Stopped("), name
            assert summary["returned_target_status"] == "Some(Passed)", name
            assert recommendations[-1]["stage"].startswith("Stop("), name
        if name != "nm-f64-init-cap":
            counts = {key: int(value) for key, value in re.findall(r"(\w+_evals): (\d+)", summary["logical_counts"])}
            assert len(counts) == 6, name
            if name.endswith("central") or name.endswith("forward"):
                probes = 4 if name.endswith("central") else 3
                assert work == counts["cost_evals"] + probes * counts["gradient_evals"], name
            else:
                cost = sum(row["kind"] in {"Cost", "CostGradient"} for row in leaves)
                gradient = sum(row["kind"] in {"Gradient", "CostGradient"} for row in leaves)
                assert counts["cost_evals"] == cost + int(summary["denied"]), name
                assert counts["gradient_evals"] == gradient, name
            for key in ("residual_evals", "jacobian_evals", "hessian_evals", "hessian_product_evals"):
                assert counts[key] == 0, name
        retained.append({key: summary[key] for key in (
            "case", "precision", "physical_work", "denied", "returned_target_status",
            "last_published_target_status", "first_attainment_work", "verification_calls",
        )})
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.glob("*.csv"))}
    report = {"schema": 1, "status": "passed", "cases": retained, "sha256": hashes}
    (directory / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "cases": len(summaries), "csv_files": len(hashes)}))


if __name__ == "__main__":
    main()
