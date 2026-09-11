from pathlib import Path
import subprocess
import json
import statistics
import argparse

parser = argparse.ArgumentParser(
    description="Compare prebuilt projected-NM probes in alternating blocks"
)
parser.add_argument("before", type=Path)
parser.add_argument("after", type=Path)
parser.add_argument("--output", type=Path, default=Path("comparison.json"))
args = parser.parse_args()
runs = []
for block, order in enumerate(["ABBA", "BAAB", "ABBA"]):
    for position, letter in enumerate(order):
        variant = {"A": "before", "B": "after"}[letter]
        result = json.loads(
            subprocess.check_output(
                [str(getattr(args, variant).resolve())], text=True
            )
        )
        runs.append(
            dict(
                block=block, position=position, variant=variant, results=result
            )
        )
for i, n in enumerate([2, 3, 16, 128]):
    assert (
        len(
            {
                (r["results"][i]["cost"], r["results"][i]["cost_evals"])
                for r in runs
            }
        )
        == 1
    )
    ratios = []
    for block in range(3):
        times = {
            variant: statistics.median(
                [
                    statistics.median(r["results"][i]["samples_seconds"])
                    for r in runs
                    if r["block"] == block and r["variant"] == variant
                ]
            )
            for variant in ["before", "after"]
        }
        ratios.append(times["after"] / times["before"])
    print(n, ratios, statistics.median(ratios))
args.output.write_text(
    json.dumps(
        dict(
            protocol=(
                "ABBA/BAAB/ABBA; 3 warmup batches and 11 measured batches "
                "per process; 100 iterations per solve"
            ),
            runs=runs,
        ),
        indent=2,
    )
)
