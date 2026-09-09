#!/usr/bin/env python3
"""Compare current Basin, pinned PRIMA, and cobyla 1.0.2 in an isolated workspace."""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import statistics
from pathlib import Path

import reproduce_cobyla_prima as prima

HERE = Path(__file__).resolve().parent
MODES = [
    "raw",
    "executor",
    "prima",
    "cobyla-historical",
    "cobyla-native",
    "cobyla-rows",
]


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(directory):
    results = json.loads((directory / "diagnostics.json").read_text())
    rows = []
    for result in results:
        outcome = result["result"]
        diagnostic = result["diagnostics"]
        counts = diagnostic["basin"] or diagnostic["cobyla"]
        rows.append(
            {
                "case": result["case"],
                "mode": result["mode"],
                "objective_calls": outcome["objective_calls"],
                "constraint_calls": outcome["constraint_calls"],
                "constraint_call_unit": "scalar"
                if result["mode"].startswith("cobyla-")
                else "vector",
                "iterations": outcome["iterations"],
                "stop": outcome["stop"],
                "f": outcome["f"],
                "violation": result["violation"],
                "quality_pass": result["quality_pass"],
                "first_target_evaluation": next(
                    (p["nf"] for p in diagnostic["points"] if p["quality"]), None
                ),
                "lp_calls": counts.get("lp"),
                "model_builds": counts.get("models"),
                "inverse_products": counts.get("inverse_products"),
                "inverse_factorizations_including_init": counts.get("inverse_rebuilds"),
                "x": json.dumps(outcome["x"]),
            }
        )
    write_csv(directory / "gap-results.csv", rows)
    kernels = [
        r
        for r in json.loads((directory / "gap-verification.json").read_text())
        if r["mode"] == "lp-cobyla"
    ]
    for kernel in kernels:
        kernel.pop("ns")
    write_csv(directory / "gap-kernels.csv", kernels)


def summarize_ratios(path):
    rows = list(csv.DictReader(path.open()))
    paired = []
    for numerator, denominator in [
        ("raw", "cobyla-native"),
        ("executor", "cobyla-native"),
        ("executor", "cobyla-historical"),
        ("executor", "cobyla-rows"),
        ("lp-basin", "lp-cobyla"),
    ]:
        for row in rows:
            if row["mode"] in [numerator, denominator]:
                paired.append(
                    dict(
                        row,
                        ns=float(row["ns"]),
                        mode=f"{numerator}/{denominator}",
                        revision="after" if row["mode"] == numerator else "before",
                    )
                )
    prima.summarize_ratios(paired, path.with_suffix(".ratios.csv"))


def replace_once(text, marker, replacement):
    if text.count(marker) != 1:
        raise ValueError(f"Instrumentation point changed: {marker[:100]}")
    return text.replace(marker, replacement)


def instrument_function(text, signature, name, prefix="crate::record"):
    start = text.index(signature)
    brace = text.index("{", start)
    return (
        text[: brace + 1]
        + f'\n#[cfg(feature = "capture")] {prefix}("{name}");\n'
        + text[brace + 1 :]
    )


def build(args):
    # Reuse the pinned Fortran build, source snapshot, and verified LP capture.
    prima.build(args)
    extend_probe(args)


def extend_probe(args):
    output = args.output.resolve()
    manifest_path = output / "Cargo.toml"
    manifest_path.write_text(
        manifest_path.read_text().replace(
            "[dependencies]\n", '[dependencies]\ncobyla = "=1.0.2"\n'
        )
    )
    metadata = json.loads(
        prima.capture(
            "cargo",
            "metadata",
            "--format-version",
            "1",
            "--manifest-path",
            str(manifest_path),
        )
    )
    package = next(
        p
        for p in metadata["packages"]
        if p["name"] == "cobyla" and p["version"] == "1.0.2"
    )
    source = Path(package["manifest_path"]).parent
    old = output / "cobyla"
    shutil.copytree(source, old)
    upstream_hash = hashlib.sha256(
        (old / "src/nlopt_cobyla.rs").read_bytes()
    ).hexdigest()
    p = old / "Cargo.toml"
    p.write_text(replace_once(p.read_text(), "[features]", "[features]\ncapture = []"))
    with (old / "src/lib.rs").open("a") as stream:
        stream.write(
            '\npub use nlopt_cobyla::GapLp;\n#[cfg(feature = "capture")]\npub use nlopt_cobyla::gap_counts;\n'
        )
    text = (old / "src/nlopt_cobyla.rs").read_text()
    text = instrument_function(text, "unsafe fn trstlp(", "lp", "gap_record")
    text = replace_once(
        text,
        "                            error = 0.0f64;",
        '                            #[cfg(feature = "capture")] gap_record("inverse_products");\n                            error = 0.0f64;',
    )
    marker = "                                i__2 = mp;\n                                k = 1 as ::core::ffi::c_int;"
    text = replace_once(
        text,
        marker,
        '                                #[cfg(feature = "capture")] gap_record("models");\n'
        + marker,
    )
    (old / "src/nlopt_cobyla.rs").write_text(
        text + "\n" + (HERE / "cobyla_gap_lp.rs").read_text()
    )
    manifest = (
        (output / "Cargo.toml")
        .read_text()
        .replace('members = ["crates/basin"]', 'members = ["crates/basin", "cobyla"]')
    )
    manifest = manifest.replace('cobyla = "=1.0.2"', 'cobyla = { path = "cobyla" }')
    manifest = manifest.replace("capture = []", 'capture = ["cobyla/capture"]')
    (output / "Cargo.toml").write_text(manifest)
    main = (output / "src/main.rs").read_text()
    main += '\ninclude!("gap_extra.rs");\n'
    main = replace_once(
        main,
        '        "prima" => {',
        '        "cobyla-historical" | "cobyla-native" | "cobyla-rows" => old_solve(mode, case, &p),\n        "prima" => {',
    )
    main = replace_once(
        main,
        "        self.case.objective(x)\n",
        '        #[cfg(feature = "capture")]\n        record_point(self.case, x, self.nf.get());\n        self.case.objective(x)\n',
    )
    main = replace_once(
        main,
        '"verification_calls":{"objective":1,"constraints":1}',
        '"verification_calls":{"objective":1,"constraints":1},"diagnostics":diagnostics()',
    )
    main = replace_once(
        main,
        "    let mut work = raw::trstlp::TrstlpWork::new(n, m);",
        "    let mut old_lp = cobyla::GapLp::new(n, m);\n    let mut work = raw::trstlp::TrstlpWork::new(n, m);",
    )
    main = replace_once(
        main,
        "\n        reference_lp(lp, &mut d);",
        '\n        if mode == "lp-cobyla" {\n            d.copy_from_slice(old_lp.solve(&lp.a, &lp.b, lp.delta, &lp.g));\n        } else { reference_lp(lp, &mut d); }',
    )
    main = replace_once(
        main,
        '                "lp-prima" => {',
        '                "lp-cobyla" => {\n                    black_box(old_lp.solve(&lp.a, &lp.b, lp.delta, &lp.g));\n                }\n                "lp-prima" => {',
    )
    (output / "src/main.rs").write_text(main)
    shutil.copyfile(HERE / "cobyla_gap_extra.rs", output / "src/gap_extra.rs")
    for module, signature, name in [
        ("model", "pub(crate) fn build(", "models"),
        ("update", "fn inv_error<F:", "inverse_products"),
        ("update", "fn updatepole_with_residual<F:", "updatepole"),
        ("update", "pub(crate) fn updatexfc<F:", "updatexfc"),
        ("linalg", "pub(crate) fn inv<F:", "inverse_rebuilds"),
        ("driver", "fn get_cpen(", "get_cpen"),
        ("trstlp", "pub(crate) fn solve(", "lp"),
    ]:
        path = output / f"src/raw/{module}.rs"
        path.write_text(instrument_function(path.read_text(), signature, name))
    finish_probe(args, upstream_hash)


def finish_probe(args, upstream_hash):
    output = args.output.resolve()
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
    prima.run(*command, env=env)
    shutil.copy2(output / f"target/{profile}/cobyla-prima-probe", output / "probe")
    prima.run(*command, "--features", "capture", env=env)
    shutil.copy2(output / f"target/{profile}/cobyla-prima-probe", output / "capture")
    meta = json.loads((output / "metadata.json").read_text())
    meta.update(
        cobyla_version="1.0.2",
        cobyla_source_sha256=upstream_hash,
        lock_sha256=hashlib.sha256((output / "Cargo.lock").read_bytes()).hexdigest(),
        probe_sha256=hashlib.sha256((output / "src/main.rs").read_bytes()).hexdigest(),
        gap_extra_sha256=hashlib.sha256(
            (output / "src/gap_extra.rs").read_bytes()
        ).hexdigest(),
    )
    prima.write_json(output / "metadata.json", meta)
    results = []
    diagnostic = []
    (output / "gap-traces").mkdir()
    for case in prima.CASES:
        for mode in MODES:
            results.append(prima.probe(output / "probe", mode, case))
            trace_env = dict(
                env, COBYLA_LP_TRACE=str(output / "gap-traces" / f"{mode}-{case}.jsonl")
            )
            diagnostic.append(
                json.loads(
                    prima.capture(
                        str(output / "capture"), mode, case, "1", env=trace_env
                    )
                )
            )
            if results[-1]["result"] != diagnostic[-1]["result"]:
                raise ValueError(f"Instrumentation changed {mode} {case}")
        for mode in ["lp-basin", "lp-cobyla"]:
            result = prima.probe(
                output / "probe", mode, output / "traces" / f"{case}.jsonl"
            )
            results.append(dict(result, case=case))
    prima.write_json(output / "gap-verification.json", results)
    prima.write_json(output / "diagnostics.json", diagnostic)
    summarize_results(output)
    print(f"Gap probe ready: {output}", flush=True)


def measure(args):
    directory = args.build.resolve()
    if json.loads((directory / "metadata.json").read_text())["profile"] != "release":
        raise ValueError("Use an uninstrumented release build for timings")
    jobs = []
    for case in args.cases:
        repeats = prima.CASES[case]
        jobs.extend((mode, case, case, repeats) for mode in MODES)
        jobs.extend(
            (mode, case, directory / "traces" / f"{case}.jsonl", max(10, repeats // 2))
            for mode in ["lp-basin", "lp-cobyla"]
        )
    # Validate each contestant before timing and retain failures as outcomes.
    verified = [
        dict(prima.probe(directory / "probe", mode, value), case=case)
        for mode, case, value, _ in jobs
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prima.write_json(args.output.with_suffix(".verification.json"), verified)
    rows = []
    rng = random.Random(42)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["round", "mode", "case", "ns"], lineterminator="\n"
        )
        writer.writeheader()
        for sample in range(args.rounds):
            rng.shuffle(jobs)
            for mode, case, value, repeats in jobs:
                result = prima.probe(
                    directory / "probe", mode, value, repeats, args.cpu
                )
                row = {"round": sample, "mode": mode, "case": case, "ns": result["ns"]}
                rows.append(row)
                writer.writerow(row)
            stream.flush()
            print(f"Round {sample + 1}/{args.rounds}", flush=True)
    summaries = []
    for mode, case in sorted({(r["mode"], r["case"]) for r in rows}):
        values = [r["ns"] for r in rows if (r["mode"], r["case"]) == (mode, case)]
        q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
        summaries.append(
            {
                "mode": mode,
                "case": case,
                "median_ns": statistics.median(values),
                "q1_ns": q1,
                "q3_ns": q3,
                "mean_ns": statistics.mean(values),
                "stdev_ns": statistics.stdev(values),
            }
        )
    with args.output.with_suffix(".summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(summaries[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summaries)
    prima.write_json(
        args.output.with_suffix(".metadata.json"),
        {
            "build": str(directory),
            "cpu": args.cpu,
            "rounds": args.rounds,
            "seed": 42,
            "cases": args.cases,
            "repeats": prima.CASES,
        },
    )

    summarize_ratios(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build")
    b.add_argument("--basin", type=Path, default=prima.BASIN)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--profile", action="store_true")
    m = commands.add_parser("measure")
    m.add_argument("--build", type=Path, required=True)
    m.add_argument("--output", type=Path, required=True)
    m.add_argument("--rounds", type=int, default=15)
    m.add_argument("--cpu", type=int, default=2)
    m.add_argument(
        "--cases",
        nargs="+",
        choices=prima.CASES,
        default=["camel", "sphere", "quadratic", "sphere_3", "sphere_20"],
    )
    args = parser.parse_args()
    if args.command == "build":
        build(args)
    elif args.rounds < 2:
        parser.error("--rounds must be at least 2")
    else:
        measure(args)


if __name__ == "__main__":
    main()
