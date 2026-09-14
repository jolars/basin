"""Prepare an isolated benchmark using the real GlobalSearch adapters."""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


def output(*args):
    return subprocess.check_output(args, text=True).strip()


here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--globalsearch", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--lockfile", type=Path, default=here / "Cargo.lock")
args = parser.parse_args()
basin = here.parents[3]
destination = args.output.resolve()
destination.mkdir(parents=True, exist_ok=True)
manifest = f"""[package]
name = "steihaug-derivatives"
version = "0.0.0"
edition = "2024"
publish = false

[dependencies]
globalsearch = {{ path = {json.dumps(str(args.globalsearch.resolve()))}, default-features = false, features = ["argmin", "basin"] }}
ndarray = "=0.16.1"

[patch.crates-io]
basin = {{ path = {json.dumps(str(basin / "crates/basin"))} }}

[[bin]]
name = "steihaug-derivatives"
path = "probe.rs"

[workspace]

[profile.release]
lto = "thin"
codegen-units = 1
"""
(destination / "Cargo.toml").write_text(manifest)
shutil.copyfile(here / "probe.rs", destination / "probe.rs")
shutil.copyfile(args.lockfile, destination / "Cargo.lock")
metadata = {
    "prepared_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "rustc": output("rustc", "-Vv"),
    "cpu": output("lscpu"),
    "affinity": sorted(os.sched_getaffinity(0)),
    "RUSTFLAGS": os.environ.get("RUSTFLAGS", ""),
    "load": os.getloadavg(),
    "probe_sha256": hashlib.sha256((here / "probe.rs").read_bytes()).hexdigest(),
    "input_lock_sha256": hashlib.sha256(args.lockfile.read_bytes()).hexdigest(),
}
for name, repo in [("basin", basin), ("globalsearch", args.globalsearch)]:
    metadata[name] = {
        "path": str(repo.resolve()),
        "commit": output("git", "-C", str(repo), "rev-parse", "HEAD"),
        "status": output("git", "-C", str(repo), "status", "--short"),
    }
(destination / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(destination / "Cargo.toml")
