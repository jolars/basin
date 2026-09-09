#!/usr/bin/env python3
"""Build and compare isolated COBYLA/PRIMA probes; no GlobalSearch checkout needed."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
from pathlib import Path

import tomllib

HERE = Path(__file__).resolve().parent
BASIN = HERE.parents[3]
PRIMA_COMMIT = "e1169927f10fea330c1aae60f12e1a32c45ef5f4"
CASES = {"camel": 5000, "sphere": 400, "quadratic": 3000}
CASES.update({f"sphere_{n}": max(10, 40000 // (n * n)) for n in [1, 3, 5, 20, 40]})
MODULES = [
    "driver",
    "filter",
    "geometry",
    "init",
    "linalg",
    "model",
    "trstlp",
    "update",
]


def capture(*command, **kwargs):
    return subprocess.check_output(command, text=True, **kwargs).strip()


def run(*command, **kwargs):
    subprocess.run(command, check=True, **kwargs)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def summarize_ratios(rows, output):
    """Keep paired-round uncertainty separate from ratios of timing medians."""
    summaries = []
    keys = sorted({(r["mode"], r["case"]) for r in rows if r["revision"] == "after"})
    for mode, case in keys:
        paired = {
            revision: {
                int(r["round"]): float(r["ns"])
                for r in rows
                if (r["revision"], r["mode"], r["case"]) == (revision, mode, case)
            }
            for revision in ["before", "after"]
        }
        if paired["before"].keys() != paired["after"].keys():
            raise ValueError("Unpaired timing rounds")
        ratios = [
            paired["after"][i] / paired["before"][i] for i in sorted(paired["before"])
        ]
        rng = random.Random(42)
        boot = sorted(
            statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(10000)
        )
        summaries.append(
            {
                "mode": mode,
                "case": case,
                "rounds": len(ratios),
                "median_after_before_ratio": statistics.median(ratios),
                "bootstrap_95_lower": boot[250],
                "bootstrap_95_upper": boot[9749],
            }
        )
    if summaries:
        with output.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(summaries[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(summaries)


def probe(binary, mode, case, repeats=1, cpu=None):
    command = [str(binary), mode, str(case), str(repeats)]
    if cpu is not None:
        command = ["taskset", "-c", str(cpu), *command]
    result = json.loads(capture(*command))
    if not math.isfinite(result["ns"]) or result["ns"] <= 0:
        raise ValueError("Expected positive timing from an uninstrumented probe")
    return result


def verify(output):
    results = [
        probe(output / "probe", mode, case)
        for case in CASES
        for mode in ["prima", "raw", "executor"]
    ]
    for case in CASES:
        results.append(
            probe(output / "probe", "lp-basin", output / "traces" / f"{case}.jsonl")
        )
        results[-1]["case"] = case
    write_json(output / "verification.json", results)
    print(
        f"Verified {len(results)} solves/kernel batches; "
        f"{sum(r.get('quality_pass') is False for r in results)} quality failures; "
        f"{sum(r.get('mismatches', 0) for r in results)} LP mismatches",
        flush=True,
    )


def build(args):
    source = args.basin.resolve()
    # Historical Basin worktrees need not initialize another PRIMA checkout.
    prima = BASIN / "tools/prima"
    if capture("git", "-C", str(prima), "rev-parse", "HEAD") != PRIMA_COMMIT:
        raise ValueError(f"Expected PRIMA {PRIMA_COMMIT}")
    if capture(
        "git", "-C", str(prima), "status", "--porcelain", "--untracked-files=no"
    ):
        raise ValueError("PRIMA has tracked modifications")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / "crates/basin", output / "crates/basin")
    shutil.copyfile(source / "README.md", output / "README.md")
    shutil.copyfile(source / "Cargo.lock", output / "Cargo.lock")
    package = tomllib.loads((source / "Cargo.toml").read_text())["workspace"]["package"]
    fields = "\n".join(f"{key} = {json.dumps(value)}" for key, value in package.items())
    (output / "Cargo.toml").write_text(f"""[package]
name = "cobyla-prima-probe"
version = "0.0.0"
edition = "2024"
rust-version = "1.87"
[workspace]
members = ["crates/basin"]
resolver = "3"
[workspace.package]
{fields}
[dependencies]
basin = {{ path = "crates/basin", default-features = false }}
serde = {{ version = "1", features = ["derive"] }}
serde_json = {{ version = "1", features = ["float_roundtrip"] }}
[features]
capture = []
[profile.release]
lto = "thin"
codegen-units = 1
[profile.profiling]
inherits = "release"
debug = true
strip = "none"
""")
    src = output / "src"
    (src / "raw").mkdir(parents=True)
    shutil.copyfile(HERE / "prima_probe.rs", src / "main.rs")
    (src / "raw.rs").write_text(
        "\n".join(f"pub(crate) mod {name};" for name in MODULES) + "\n"
    )
    for name in MODULES:
        text = (source / f"crates/basin/src/solver/cobyla/{name}.rs").read_text()
        if name == "trstlp":
            marker = "        let n = self.d.len();"
            if text.count(marker) != 1:
                raise ValueError(
                    "LP capture insertion point changed; review instrumentation"
                )
            text = text.replace(
                marker,
                marker + '\n        #[cfg(feature = "capture")]\n'
                "        crate::capture_lp(n, a, b, delta, g);",
            )
        (src / "raw" / f"{name}.rs").write_text(text)

    native = output / "native"
    flags = "-O3 -ffp-contract=off"
    if args.profile:
        flags += " -g -fno-omit-frame-pointer"
    run(
        "cmake",
        "-S",
        str(prima),
        "-B",
        str(native),
        "-DBUILD_SHARED_LIBS=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DPRIMA_ENABLE_C=ON",
        "-DPRIMA_REAL_PRECISION=64",
        f"-DCMAKE_Fortran_FLAGS_RELEASE={flags}",
    )
    run("cmake", "--build", str(native), "--target", "primac", "-j", "4")
    run(
        "gfortran",
        *flags.split(),
        "-frecursive",
        "-fno-stack-arrays",
        "-c",
        "-I",
        str(native / "fortran/mod"),
        "-I",
        str(native / "c"),
        "-J",
        str(native),
        str(HERE / "prima_bridge.f90"),
        "-o",
        str(native / "bridge.o"),
    )
    run("ar", "rcs", str(native / "libbridge.a"), str(native / "bridge.o"))
    searches = [native, native / "c", native / "fortran"]
    for lib in ["libgfortran.so", "libquadmath.so"]:
        searches.append(
            Path(capture("gfortran", f"-print-file-name={lib}")).resolve().parent
        )
    # GFortran's nested callback trampolines require this for the native probe.
    directives = ["cargo:rustc-link-arg=-Wl,-z,execstack"]
    directives += [f"cargo:rustc-link-search=native={path}" for path in searches]
    directives += [
        f"cargo:rustc-link-lib={lib}"
        for lib in [
            "static=bridge",
            "static=primac",
            "static=primaf",
            "gfortran",
            "quadmath",
            "m",
        ]
    ]
    (output / "build.rs").write_text(
        "fn main() {\n"
        + "\n".join(f"    println!({json.dumps(line)});" for line in directives)
        + "\n}\n"
    )
    profile = "profiling" if args.profile else "release"
    env = os.environ.copy()
    if args.profile:
        env["RUSTFLAGS"] = (
            env.get("RUSTFLAGS", "") + " -C force-frame-pointers=yes"
        ).strip()
    command = [
        "cargo",
        "build",
        "--profile",
        profile,
        "--manifest-path",
        str(output / "Cargo.toml"),
    ]
    run(*command, env=env)
    shutil.copy2(output / f"target/{profile}/cobyla-prima-probe", output / "probe")
    run(*command, "--features", "capture", env=env)
    shutil.copy2(output / f"target/{profile}/cobyla-prima-probe", output / "capture")
    (output / "traces").mkdir()
    for case in CASES:
        trace_env = dict(env, COBYLA_LP_TRACE=str(output / "traces" / f"{case}.jsonl"))
        capture(str(output / "capture"), "raw", case, "1", env=trace_env)
    metadata = {
        "basin": capture("git", "-C", str(source), "rev-parse", "HEAD"),
        "basin_diff": capture("git", "-C", str(source), "diff", "--", "crates/basin"),
        "prima": PRIMA_COMMIT,
        "rustc": capture("rustc", "-Vv"),
        "gfortran": capture("gfortran", "--version"),
        "fortran_flags": flags,
        "rustflags": env.get("RUSTFLAGS", ""),
        "profile": profile,
        "lock_sha256": hashlib.sha256((output / "Cargo.lock").read_bytes()).hexdigest(),
        "cpu": capture("lscpu"),
        "probe_sha256": hashlib.sha256((src / "main.rs").read_bytes()).hexdigest(),
        "bridge_sha256": hashlib.sha256(
            (HERE / "prima_bridge.f90").read_bytes()
        ).hexdigest(),
        "thread_environment": {
            key: env.get(key)
            for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"]
        },
        "settings": {
            "rho_beg": 0.5,
            "rho_end": 2**-27,
            "ctol": 2**-26,
            "cweight": 1e8,
            "eta1": 0.1,
            "eta2": 0.7,
            "gamma1": 0.5,
            "gamma2": 2.0,
            "maxfilt_requested": 2000,
            "maxhist": 0,
            "bounds": "nonlinear rows",
            "scalar": "f64",
            "prima_effective_maxfilt": {
                case: (
                    50
                    if case == "camel"
                    else 100
                    if case == "quadratic"
                    else 200
                    if case == "sphere"
                    else 20 * int(case.split("_")[1])
                )
                for case in CASES
            },
        },
    }
    write_json(output / "metadata.json", metadata)
    verify(output)
    print(f"Built {output / 'probe'}", flush=True)


def measure(args):
    before = args.before.resolve()
    after = args.after.resolve() if args.after else None
    builds = [("before", before)] + ([("after", after)] if after else [])
    baseline_metadata = json.loads((before / "metadata.json").read_text())
    for _, build_dir in builds:
        metadata = json.loads((build_dir / "metadata.json").read_text())
        if metadata["profile"] != "release":
            raise ValueError("Timing requires an uninstrumented release build")
        for key in [
            "prima",
            "rustc",
            "gfortran",
            "fortran_flags",
            "rustflags",
            "lock_sha256",
        ]:
            if metadata[key] != baseline_metadata[key]:
                raise ValueError(
                    f"Mismatched {key}; rebuild with matching toolchains and lockfiles"
                )
    jobs = []
    for case, repeats in CASES.items():
        if case not in args.cases:
            continue
        jobs.append(("reference", before, "prima", case, case, repeats))
        trace = before / "traces" / f"{case}.jsonl"
        jobs.append(
            ("reference", before, "lp-prima", case, trace, max(10, repeats // 2))
        )
        for revision, directory in builds:
            for mode in ["raw", "executor"]:
                jobs.append((revision, directory, mode, case, case, repeats))
            for mode in ["lp-basin", "lp-setup"]:
                jobs.append(
                    (revision, directory, mode, case, trace, max(10, repeats // 2))
                )
    # Verify exact before/after solver work and acceptable matched LP outputs first.
    for _, directory, mode, case, value, _ in jobs:
        result = probe(directory / "probe", mode, value, cpu=args.cpu)
        if result.get("mismatches", 0):
            raise ValueError(
                f"LP mismatch on {case}; inspect verification.json before timing"
            )
        if after and directory == after and mode in ["raw", "executor"]:
            baseline = probe(before / "probe", mode, value, cpu=args.cpu)
            if result["result"] != baseline["result"]:
                raise ValueError(f"Before/after numerical work changed: {case} {mode}")
    rng = random.Random(42)
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["round", "revision", "mode", "case", "ns"],
            lineterminator="\n",
        )
        writer.writeheader()
        for sample in range(args.rounds):
            rng.shuffle(jobs)
            for revision, directory, mode, case, value, repeats in jobs:
                r = probe(directory / "probe", mode, value, repeats, args.cpu)
                row = {
                    "round": sample,
                    "revision": revision,
                    "mode": mode,
                    "case": case,
                    "ns": r["ns"],
                }
                writer.writerow(row)
                rows.append(row)
            output.flush()
            print(f"Round {sample + 1}/{args.rounds}", flush=True)
    summaries = []
    for key in sorted({(r["revision"], r["mode"], r["case"]) for r in rows}):
        values = [r["ns"] for r in rows if (r["revision"], r["mode"], r["case"]) == key]
        q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
        summaries.append(
            dict(
                zip(["revision", "mode", "case"], key),
                median_ns=statistics.median(values),
                q1_ns=q1,
                q3_ns=q3,
                mean_ns=statistics.mean(values),
                stdev_ns=statistics.stdev(values),
            )
        )
    with args.output.with_suffix(".summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(summaries[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summaries)
    summarize_ratios(rows, args.output.with_suffix(".ratios.csv"))
    write_json(
        args.output.with_suffix(".metadata.json"),
        {
            "before": str(before),
            "after": str(after) if after else None,
            "rounds": args.rounds,
            "cpu": args.cpu,
            "seed": 42,
            "solves_per_process": CASES,
            "cases": args.cases,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build")
    b.add_argument("--basin", type=Path, default=BASIN)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--profile", action="store_true")
    m = commands.add_parser("measure")
    m.add_argument("--before", type=Path, required=True)
    m.add_argument("--after", type=Path)
    m.add_argument("--output", type=Path, required=True)
    m.add_argument("--cpu", type=int, default=2)
    m.add_argument("--rounds", type=int, default=15)
    m.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    args = parser.parse_args()
    if args.command == "build":
        build(args)
    else:
        if args.rounds < 2:
            parser.error("--rounds must be at least 2")
        measure(args)


if __name__ == "__main__":
    main()
