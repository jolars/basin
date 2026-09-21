"""Regenerate solution fixtures using the authors' CG_DESCENT C 1.2."""

import hashlib
import io
from pathlib import Path
import subprocess
import tarfile
import urllib.request


FIXTURES = Path(__file__).resolve().parent
REFERENCE = FIXTURES.parents[3] / "references" / "nonlinear-cg-c-1.2"
URL = "https://www.math.lsu.edu/~hozhang/SoftArchive/CG_DESCENT-C-1.2.tar.gz"
SHA256 = "8c09fdb6f98540b214ef348060fa6834d45c32f84cdfbc200e3d97d6631d0573"


def main():
    REFERENCE.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=30) as response:
        archive_bytes = response.read()
    if hashlib.sha256(archive_bytes).hexdigest() != SHA256:
        raise RuntimeError("CG_DESCENT C 1.2 archive checksum changed")
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        for member in archive.getmembers():
            if member.isfile():
                (REFERENCE / Path(member.name).name).write_bytes(
                    archive.extractfile(member).read()
                )

    parameters = REFERENCE / "cg_descent_c.parm"
    overrides = {"restart_fac": "1000000.", "maxit_fac": "10000.", "PrintFinal": "0"}
    lines = []
    for line in parameters.read_text().splitlines():
        value, description = line.split(maxsplit=1)
        name = description.split()[0]
        lines.append(f"{overrides.get(name, value)} {description}")
    parameters.write_text("\n".join(lines) + "\n")
    executable = REFERENCE / "reference"
    subprocess.run(
        [
            "cc", "-O2", "-I", str(REFERENCE),
            str(FIXTURES / "nonlinear_cg_reference.c"),
            str(REFERENCE / "cg_descent.c"), "-lm", "-o", str(executable),
        ],
        check=True,
    )
    output = subprocess.check_output([str(executable)], cwd=REFERENCE, text=True)
    (FIXTURES / "nonlinear_cg_reference.tsv").write_text(output)


if __name__ == "__main__":
    main()
