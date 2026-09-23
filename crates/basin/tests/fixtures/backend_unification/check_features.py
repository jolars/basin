"""Check that version-specific acceleration does not enable another adapter."""

import json
import subprocess
from pathlib import Path

manifest = Path(__file__).with_name("Cargo.toml")
for consumer, selected, other in [("older", "32", "35"), ("newer", "35", "32")]:
    metadata = json.loads(
        subprocess.check_output(
            [
                "cargo", "metadata", "--format-version", "1",
                "--manifest-path", str(manifest),
                "--features", f"basin-backend-{consumer}/lapack",
            ],
            text=True,
        )
    )
    package = next(p for p in metadata["packages"] if p["name"] == "basin")
    node = next(n for n in metadata["resolve"]["nodes"] if n["id"] == package["id"])
    features = set(node["features"])
    assert f"nalgebra_v0_{selected}-lapack" in features, features
    assert f"nalgebra_v0_{other}-lapack" not in features, features
    assert "nalgebra_v0_34" not in features, features
    assert "nalgebra_v0_34-lapack" not in features, features
    print(f"{consumer}: only the requested LAPACK adapter is enabled")
