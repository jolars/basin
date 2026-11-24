"""Build and compare the step-overhead experiment on a pinned main revision."""

import argparse
import csv
import os
import statistics
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASELINE = "aa05436feb6595365c13b30daab629fbf774a2d9"
VARIANTS = {
    "main": (),
    "in-place": ("in-place",),
    "borrowed": ("borrowed-report",),
    "both": ("in-place", "borrowed-report"),
}
ORDERS = (
    ("main", "in-place", "both", "borrowed"),
    ("in-place", "borrowed", "main", "both"),
    ("borrowed", "both", "in-place", "main"),
    ("both", "main", "borrowed", "in-place"),
)
TESTS = (
    "owned_execution",
    "shared_progress",
    "termination_reports",
    "inner_executor",
    "levenberg_marquardt_ndarray",
    "levenberg_marquardt_stopping",
    "levenberg_marquardt_no_progress",
    "levenberg_marquardt_qr",
    "levenberg_marquardt_robust_recovery",
    "lbfgs_line_search_evaluations",
    "lbfgsb_stopping",
)


def run(*command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def build(scratch, output, variant, archive, env):
    tree = scratch / variant
    tree.mkdir()
    run("tar", "-xf", "-", cwd=tree, input=archive)
    for patch in VARIANTS[variant]:
        run("git", "apply", str(HERE / f"{patch}.patch"), cwd=tree)

    source = (HERE / "step_overhead.rs").read_text()
    if "borrowed-report" in VARIANTS[variant]:
        owned_call = "stepper.step().unwrap().report().map(read)"
        assert source.count(owned_call) == 1
        source = source.replace(
            owned_call, "stepper.step_borrowed().unwrap().map(read)"
        )
    (tree / "crates/basin/examples/step_overhead.rs").write_text(source)

    # Cargo must use independent artifacts even if the caller sets a shared
    # CARGO_TARGET_DIR in their development shell.
    target = tree / "target"
    cargo = ("cargo", "--color", "never")
    common = (
        "--locked",
        "-p",
        "basin",
        "--no-default-features",
        "--target-dir",
        str(target),
    )
    with (output / f"{variant}.build.log").open("w") as log:
        run(
            *cargo,
            "build",
            *common,
            "--release",
            "--example",
            "step_overhead",
            "--features",
            "ndarray_v0_17",
            cwd=tree,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    with (output / f"{variant}.tests.log").open("w") as log:
        test_targets = [arg for test in TESTS for arg in ("--test", test)]
        run(
            *cargo,
            "test",
            *common,
            "--features",
            "ndarray_v0_17,problems,serde",
            *test_targets,
            cwd=tree,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    executable = target / "release/examples/step_overhead"
    with (output / f"{variant}.trace").open("wb") as trace:
        run(str(executable), "--verify", stdout=trace)
    return executable


def measure(executables, output, rounds, solves, cpu):
    pin = ("taskset", "-c", str(cpu)) if cpu is not None else ()
    samples = {}
    with (output / "samples.csv").open("w", newline="") as raw:
        writer = csv.writer(raw)
        writer.writerow(("round", "order", "variant", "workload", "path", "cap", "ns"))
        for round_index in range(rounds):
            for order, variant in enumerate(ORDERS[round_index % len(ORDERS)]):
                result = run(
                    *pin,
                    str(executables[variant]),
                    str(solves),
                    capture_output=True,
                    text=True,
                )
                cases_seen = set()
                for workload, path, cap, ns in csv.reader(result.stdout.splitlines()):
                    key = (workload, path, int(cap))
                    assert key not in cases_seen, key
                    cases_seen.add(key)
                    writer.writerow((round_index + 1, order + 1, variant, *key, ns))
                    samples.setdefault(key, {}).setdefault(variant, []).append(
                        float(ns)
                    )
            raw.flush()
            print(f"Timed round {round_index + 1}/{rounds}", flush=True)

    with (output / "summary.csv").open("w", newline="") as summary:
        writer = csv.writer(summary)
        writer.writerow(
            (
                "workload",
                "path",
                "cap",
                "variant",
                "reference",
                "median_ns",
                "min_ns",
                "max_ns",
                "median_pct",
                "p25_pct",
                "p75_pct",
            )
        )
        for key, results in samples.items():
            assert all(len(results[variant]) == rounds for variant in VARIANTS), key
            for variant, reference in (
                ("main", "main"),
                ("in-place", "main"),
                ("borrowed", "main"),
                ("both", "main"),
                ("both", "in-place"),
                ("both", "borrowed"),
            ):
                values = results[variant]
                deltas = [
                    100 * (value / base - 1)
                    for value, base in zip(values, results[reference], strict=True)
                ]
                q25, _, q75 = statistics.quantiles(deltas, n=4, method="inclusive")
                metrics = (
                    statistics.median(values),
                    min(values),
                    max(values),
                    statistics.median(deltas),
                    q25,
                    q75,
                )
                writer.writerow(
                    (*key, variant, reference, *(f"{x:.3f}" for x in metrics))
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path, help="new directory for logs, traces, and timings"
    )
    parser.add_argument("--cpu", type=int, help="logical CPU for all timed batches")
    parser.add_argument(
        "--rounds", type=int, default=16, help="interleaved rounds (multiple of 4)"
    )
    parser.add_argument(
        "--solves", type=int, default=20_000, help="fresh solves per timed batch"
    )
    args = parser.parse_args()
    if args.rounds < 4 or args.rounds % 4 or args.solves < 1:
        parser.error(
            "--rounds must be a positive multiple of 4 and --solves must be positive"
        )

    output = args.output.resolve()
    output.mkdir(parents=True)
    archive = run("git", "archive", BASELINE, cwd=ROOT, capture_output=True).stdout
    env = {
        **os.environ,
        "CARGO_PROFILE_RELEASE_LTO": "fat",
        "CARGO_PROFILE_RELEASE_CODEGEN_UNITS": "1",
    }
    with (output / "environment.txt").open("w") as log:
        print(
            f"revision={BASELINE}\nrounds={args.rounds}\nsolves={args.solves}\ncpu={args.cpu}",
            file=log,
        )
        for key in (
            "CARGO_PROFILE_RELEASE_LTO",
            "CARGO_PROFILE_RELEASE_CODEGEN_UNITS",
            "RUSTFLAGS",
            "CARGO_ENCODED_RUSTFLAGS",
            "RAYON_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        ):
            print(f"{key}={env.get(key, '')}", file=log)
        log.flush()
        run("rustc", "-vV", stdout=log)
        run("uname", "-srm", stdout=log)
        run("lscpu", stdout=log)

    with tempfile.TemporaryDirectory(prefix="basin-step-overhead-") as path:
        scratch = Path(path)
        executables = {}
        for variant in VARIANTS:
            print(f"Building and verifying {variant}", flush=True)
            executables[variant] = build(scratch, output, variant, archive, env)
            assert (output / f"{variant}.trace").read_bytes() == (
                output / "main.trace"
            ).read_bytes(), variant
        measure(executables, output, args.rounds, args.solves, args.cpu)


if __name__ == "__main__":
    main()
