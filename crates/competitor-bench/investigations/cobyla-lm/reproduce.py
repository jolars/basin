#!/usr/bin/env python3
"""Build isolated probes against the two historical GlobalSearch adapters.

Requires Python 3.12+, Git, and the Basin development Rust toolchain. Historical
sources are read with git archive so the supplied checkout remains untouched.
The generated workspace keeps historical dependencies out of Basin's lockfile.
"""

import argparse
import io
import json
import random
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASIN = HERE.parents[3]
COMMITS = {
    "old": "1f44818e396567d22cb7137c267c66da2e19f334",
    "new": "4bf3eaa6b3677e0a2cc18f61f81603612db66b13",
}
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
CASES = [("camel", 5000), ("sphere", 400), ("quadratic", 3000)]


def capture(*args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True)


def prepare(output, globalsearch):
    for name, commit in COMMITS.items():
        archive = subprocess.check_output(
            ["git", "-C", str(globalsearch), "archive", commit]
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(output / name, filter="data")
    shutil.copytree(output / "new", output / "current")
    for name in ["old", "new", "current"]:
        manifest = output / name / "Cargo.toml"
        text = manifest.read_text().replace(
            'name = "globalsearch"', f'name = "globalsearch-{name}"', 1
        )
        if name == "new":
            text = text.replace(
                'version = "1.8.0", default-features',
                'version = "=1.8.0", default-features',
            )
        if name == "current":
            text = text.replace(
                'version = "1.8.0", default-features',
                f"path = {json.dumps(str(BASIN / 'crates/basin'))}, default-features",
            )
        manifest.write_text(text)

    manifest = f"""[package]
name = "basin-gap-investigation"
version = "0.0.0"
edition = "2024"
rust-version = "1.87"
[workspace]
[dependencies]
basin = {{ path = {json.dumps(str(BASIN / "crates/basin"))}, default-features = false, features = ["nalgebra"] }}
cobyla = "=1.0.2"
globalsearch-old = {{ path = "old", default-features = false }}
globalsearch-new = {{ path = "new", default-features = false }}
globalsearch-current = {{ path = "current", default-features = false }}
ndarray = "=0.16.1"
nalgebra = "=0.34.2"
levenberg-marquardt = "=0.15.0"
[features]
allocations = []
[profile.release]
lto = "thin"
codegen-units = 1
debug = 1
"""
    (output / "Cargo.toml").write_text(manifest)
    bins = output / "src/bin"
    bins.mkdir(parents=True)
    for source, name in [("cobyla.rs", "cobyla_probe.rs"), ("lm.rs", "lm_probe.rs")]:
        shutil.copyfile(HERE / source, bins / name)
    (bins / "support").mkdir()
    shutil.copyfile(HERE / "support/lm_models.rs", bins / "support/lm_models.rs")

    modules = "\n".join(
        f"#[path = {json.dumps(str(BASIN / 'crates/basin/src/solver/cobyla' / (name + '.rs')))}]\n"
        f"pub(crate) mod {name};"
        for name in [
            "driver",
            "filter",
            "geometry",
            "init",
            "linalg",
            "model",
            "trstlp",
            "update",
        ]
    )
    # Compile the actual private driver sources without exposing them in Basin.
    (output / "src/lib.rs").write_text(
        "#![allow(dead_code, clippy::needless_range_loop)]\npub use basin::core;\n"
        "pub mod raw {\n" + modules + "\n}\n" + (HERE / "raw.rs").read_text()
    )
    metadata = {
        "basin": capture("git", "rev-parse", "HEAD", cwd=BASIN).strip(),
        "basin_diff": capture("git", "diff", "--", "crates/basin", cwd=BASIN),
        "globalsearch": COMMITS,
        "rustc": capture("rustc", "-Vv"),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--globalsearch",
        type=Path,
        required=True,
        help="Local GlobalSearch Git repository containing the two commits",
    )
    parser.add_argument(
        "--output", type=Path, help="New directory for the generated workspace"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Randomized timing rounds; zero builds and verifies without timing",
    )
    args = parser.parse_args()
    if args.rounds < 0:
        parser.error("--rounds must be nonnegative")
    if args.output:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = Path(tempfile.mkdtemp(prefix="basin-gaps-"))
    prepare(output, args.globalsearch.resolve())
    print(f"Workspace and results: {output}", flush=True)
    build = [
        "cargo",
        "build",
        "--release",
        "--manifest-path",
        str(output / "Cargo.toml"),
        "--target-dir",
        str(output / "target"),
    ]
    subprocess.run(build, check=True)
    binary = output / "target/release/cobyla_probe"
    timing = output / "cobyla-timing"
    shutil.copyfile(binary, timing)
    timing.chmod(0o755)
    (output / "lm.csv").write_text(capture(str(output / "target/release/lm_probe")))
    # Validate every layer even when timing is disabled.
    (output / "verification.csv").write_text(
        "".join(
            capture(str(timing), mode, case, "1") for case, _ in CASES for mode in MODES
        )
    )
    jobs = [(case, repeats, mode) for case, repeats in CASES for mode in MODES]
    rng = random.Random(42)
    results = []
    for run in range(args.rounds):
        rng.shuffle(jobs)
        for case, repeats, mode in jobs:
            result = capture(str(timing), mode, case, str(repeats)).strip()
            results.append(
                {
                    "run": run,
                    "case": case,
                    "mode": mode,
                    "ns": float(result.split(",")[2]),
                    "output": result,
                }
            )
        (output / "timings.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"Timing round {run + 1}/{args.rounds} complete", flush=True)
    # Allocation instrumentation never runs in the timing build.
    subprocess.run(
        build + ["--features", "allocations", "--bin", "cobyla_probe"], check=True
    )
    (output / "allocations.txt").write_text(
        "".join(
            capture(str(binary), mode, case, "1") for case, _ in CASES for mode in MODES
        )
    )
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
