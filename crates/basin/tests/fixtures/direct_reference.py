"""Regenerate original-DIRECT fixtures with SciPy 1.16.2 and NumPy 2.3.3.

SciPy is BSD-3-Clause; its DIRECT kernel derives from the MIT-licensed
DIRECT 2.0.4. Only independently specified objectives and numerical outputs
are committed. CI reads the TSV without Python or SciPy.
"""

from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import direct

assert scipy.__version__ == "1.16.2"
assert np.__version__ == "2.3.3"


def styblinski_tang(x):
    return 0.5 * sum(t**4 - 16 * t * t + 5 * t for t in x)


def shifted_ackley(x):
    z = x - 0.37
    return (
        -20 * np.exp(-0.2 * np.sqrt(np.mean(z * z)))
        - np.exp(np.mean(np.cos(2 * np.pi * z)))
        + 20
        + np.e
    )


# The negative stationary root gives the global minimum of the quartic.
stationary = -2.9
for _ in range(10):
    stationary -= (2 * stationary**3 - 16 * stationary + 2.5) / (
        6 * stationary * stationary - 16
    )
minimum = styblinski_tang([stationary])

lines = [
    "# SciPy 1.16.2; NumPy 2.3.3; bounds=[-5,5]^n; locally_biased=False; "
    "eps=1e-4; len_tol=0; vol_tol=0; maxiter=1000",
    "# name|dimension|budget|gap_tolerance|minimum|cost|nfev|nit|x",
]
for name, fun, n, budget, tolerance, optimum in [
    ("styblinski_tang_2d", styblinski_tang, 2, 2000, 1e-4, 2 * minimum),
    ("styblinski_tang_6d", styblinski_tang, 6, 10000, 0.5, 6 * minimum),
    ("shifted_ackley_6d", shifted_ackley, 6, 10000, 0.01, 0.0),
]:
    result = direct(
        fun,
        [(-5.0, 5.0)] * n,
        locally_biased=False,
        eps=1e-4,
        len_tol=0,
        vol_tol=0,
        maxfun=budget,
        maxiter=1000,
    )
    assert result.status == 1, (name, result.message)
    assert -1e-12 <= result.fun - optimum <= tolerance
    lines.append("|".join([
        name, str(n), str(budget), format(tolerance, ".17g"),
        format(optimum, ".17g"), format(result.fun, ".17g"),
        str(result.nfev), str(result.nit),
        ",".join(format(x, ".17g") for x in result.x),
    ]))
Path(__file__).with_suffix(".tsv").write_text("\n".join(lines) + "\n")
