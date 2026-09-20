# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.3.3", "scipy==1.16.2"]
# ///
"""Regenerate with `uv run --script robust_least_squares_reference.py`.

SciPy 1.16.2 (BSD-3-Clause) supplies reference optima and actual robust costs.
The Rust tests consume the checked-in TSV and do not require Python.
"""
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares

assert np.__version__ == "2.3.3"
assert scipy.__version__ == "1.16.2"

lines = ["# problem|loss|scale|start|solution|cost|optimality"]
for problem in ["line", "exponential"]:
    if problem == "line":
        t = np.array([-2., -1., 0., 1., 2., 3.])
        y = 1.5*t + .3 + np.array([.02, -.03, .01, -.02, 4.03, 0.])
        start = np.array([1.45, .28])

        def fun(x):
            return x[0]*t + x[1] - y

        def jac(x):
            return np.column_stack([t, np.ones_like(t)])
    else:
        t = np.array([.2, .4, .7, 1., 1.5, 2.])
        y = np.exp(-.7*t) + np.array([.01, -.02, .015, .8, -.01, .005])
        start = np.array([.65])

        def fun(x):
            return np.exp(-x[0]*t) - y

        def jac(x):
            return (-t*np.exp(-x[0]*t))[:, None]

    for loss in ["linear", "huber", "soft_l1", "cauchy", "arctan"]:
        result = least_squares(fun, start, jac=jac, method="trf",
                               loss=loss, f_scale=.1, bounds=(-10., 10.),
                               gtol=1e-12, ftol=1e-14, xtol=1e-14,
                               max_nfev=1000)
        assert result.success, (problem, loss, result.message)
        numbers = lambda x: ",".join(format(float(v), ".17g") for v in x)
        lines.append("|".join([problem, loss, ".1", numbers(start),
                               numbers(result.x), format(result.cost, ".17g"),
                               format(result.optimality, ".17g")]))

Path(__file__).with_suffix(".tsv").write_text("\n".join(lines) + "\n")
