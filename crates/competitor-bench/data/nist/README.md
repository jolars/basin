# NIST StRD nonlinear regression data

These 27 files were retrieved on October 2, 2026, from the [NIST StRD nonlinear
regression collection](https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml).
Each file retains its original observations, model, two starting points,
difficulty classification, certified parameters, residual sum of squares, and
references. The downloaded bytes, including CRLF line endings, are preserved.
[SHA256SUMS](SHA256SUMS) records their hashes.

The source URL for each file is
`https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/<name>.dat`. For
example,
[Misra1a.dat](https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Misra1a.dat)
is the source of the first focused pilot. Verify the local copies from this
directory with `sha256sum --check SHA256SUMS`. No network access is needed to
build or run the experiments.

The [fixture implementation](../../src/convergence/nist.rs) evaluates the
published models with native `f32` or `f64` arithmetic. Forward derivatives
produce the Jacobians from the same expressions. Nelson uses the logarithm of
the observed response. Basin's least-squares objective is half the published
residual sum of squares. NIST supplies no box bounds; the expanded harness does
not add finite bounds to these problems.

Certificate tests allow for rounding of the printed parameters and residual sum
of squares. In particular, the printed Lanczos1 parameters do not reproduce its
near-zero certified residual exactly. These test allowances establish
implementation consistency, not attainable optimization targets. See NIST's
[background
information](https://www.itl.nist.gov/div898/strd/nls/nls_info.shtml) and the
[experimental protocol](../../../../dev/convergence-defaults/protocol.md).
