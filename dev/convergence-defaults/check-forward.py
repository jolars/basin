"""Compare strict and configured analytic runs after checking each separately."""

import csv
import hashlib
import json
import sys
from pathlib import Path


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def validated(directory):
    validation = json.loads((directory / "validation.json").read_text())
    assert validation["status"] == "passed" and len(validation["cases"]) == 9
    for name, digest in validation["sha256"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    return {row["case"]: row for row in rows(directory / "summary.csv")}


def main():
    assert len(sys.argv) == 2, "usage: check-forward.py <paired-run-directory>"
    directory = Path(sys.argv[1])
    control, configured = directory / "control", directory / "configured"
    before, after = validated(control), validated(configured)
    strict_name, configured_name = "lbfgsb-f64-forward", "lbfgsb-f64-forward-configured"
    strict, adjusted = before.pop(strict_name), after.pop(configured_name)
    assert before.keys() == after.keys()
    for name in before:
        for key in before[name]:
            if key != "instrumented_elapsed_ns":
                assert before[name][key] == after[name][key], (name, key)
        for suffix in ("leaves", "recommendations"):
            assert rows(control / f"{name}-{suffix}.csv") == rows(configured / f"{name}-{suffix}.csv"), name
    assert strict["physical_work"] == "112" and adjusted["physical_work"] == "8"
    assert strict["first_attainment_work"] == adjusted["first_attainment_work"] == "Some(8)"
    assert "termination: Failed(" in strict["outcome"]
    assert "termination: Converged(" in adjusted["outcome"]
    assert "BoundClippedGradient" in adjusted["outcome"]
    assert strict["returned_target_status"] == adjusted["returned_target_status"] == "Some(Passed)"
    assert strict["last_published_point"] == adjusted["last_published_point"] == "Some([1.0, -2.0])"
    assert rows(control / f"{strict_name}-leaves.csv")[:8] == rows(configured / f"{configured_name}-leaves.csv")
    assert rows(control / f"{strict_name}-recommendations.csv")[:-1] == rows(configured / f"{configured_name}-recommendations.csv")[:-1]
    hashes = {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(directory.glob("*/*.csv"))}
    report = {
        "status": "passed", "unchanged_controls": len(before),
        "strict_work": 112, "configured_work": 8, "first_attainment_work": 8,
        "saved_leaf_calls": 104, "identical_leaf_prefix": 8, "sha256": hashes,
    }
    (directory / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "sha256"}))


if __name__ == "__main__":
    main()
