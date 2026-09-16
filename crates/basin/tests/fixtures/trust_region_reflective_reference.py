"""Regenerate dense TRF fixtures with SciPy 1.16.2 (BSD-3-Clause).

Run with NumPy 2.3.3; the output is consumed by Rust tests without Python.
Null bounds denote the appropriate signed infinity. Fixed coordinates are
eliminated before calling SciPy, which requires strictly ordered bounds.
"""

from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares

assert scipy.__version__ == "1.16.2"
assert np.__version__ == "2.3.3"

cases = [
    dict(name="interior", a=[[1, 2], [2, -1], [1, 1]], b=[4, 3, 3],
         lower=[-10, -10], upper=[10, 10], start=[0, 0], compare_x=True),
    dict(name="active", a=[[1, 2], [2, -1], [1, 1]], b=[4, 3, 3],
         lower=[-1, -1], upper=[1, 1], start=[0, 0], compare_x=True),
    dict(name="rank_deficient", a=[[1, 1], [2, 2]], b=[1, 2],
         lower=[None, None], upper=[None, None], start=[0, 0], compare_x=False),
    dict(name="underdetermined", a=[[1, 2, 3]], b=[1],
         lower=[None]*3, upper=[None]*3, start=[0]*3, compare_x=False),
    dict(name="ill_conditioned", a=[[1, 1], [0, 1e-7]], b=[2, 1e-7],
         lower=[None]*2, upper=[None]*2, start=[0]*2, compare_x=False),
    dict(name="fixed", a=[[1, 2], [2, -1], [1, 1]], b=[4, 3, 3],
         lower=[0.5, -10], upper=[0.5, 10], start=[0.5, 0], compare_x=True),
    dict(name="rosenbrock", lower=[-2, 1.5], upper=[2, 3],
         start=[2, 2], compare_x=True),
]

for case in cases:
    lower = np.array([-np.inf if x is None else x for x in case["lower"]])
    upper = np.array([np.inf if x is None else x for x in case["upper"]])
    start = np.array(case["start"], dtype=float)
    free = lower != upper

    def expand(y):
        x = start.copy()
        x[free] = y
        return x

    if case["name"] == "rosenbrock":
        def fun(y):
            x = expand(y)
            return np.array([10*(x[1]-x[0]**2), 1-x[0]])

        def jac(y):
            x = expand(y)
            return np.array([[-20*x[0], 10], [-1, 0]])[:, free]
    else:
        a = np.array(case["a"], dtype=float)
        b = np.array(case["b"], dtype=float)

        def fun(y):
            return a @ expand(y) - b

        def jac(y):
            return a[:, free]

    result = least_squares(fun, start[free], jac=jac,
                           bounds=(lower[free], upper[free]),
                           method="trf", tr_solver="exact", loss="linear",
                           x_scale=1, gtol=1e-10, ftol=None, xtol=None,
                           max_nfev=1000)
    assert result.success, (case["name"], result.message)
    case.update(x=expand(result.x).tolist(), cost=float(result.cost),
                optimality=float(result.optimality),
                residual=result.fun.tolist(), nfev=result.nfev, njev=result.njev)

def values(xs):
    return ",".join(format(float(x), ".17g") for x in xs)


lines = ["# name|compare_x|lower|upper|start|a(row-major)|b|x|cost|optimality|nfev|njev"]
for case in cases:
    fields = [case["name"], str(int(case["compare_x"])),
              values([-np.inf if x is None else x for x in case["lower"]]),
              values([np.inf if x is None else x for x in case["upper"]]),
              values(case["start"]),
              values([x for row in case.get("a", []) for x in row]),
              values(case.get("b", [])), values(case["x"]),
              format(case["cost"], ".17g"), format(case["optimality"], ".17g"),
              str(case["nfev"]), str(case["njev"])]
    lines.append("|".join(fields))
Path(__file__).with_suffix(".tsv").write_text("\n".join(lines) + "\n")

# Lock examples in which each of SciPy's three candidates wins. These are
# quadratic-model comparisons, independent of outer stopping decisions.
from scipy.optimize._lsq.common import solve_lsq_trust_region
from scipy.optimize._lsq.trf import select_step

rng = np.random.default_rng(391)
found = set()
steps = ["# kind|j(row-major)|g|c|p|selected|predicted; d=1, x=0, bounds=+-0.25, radius=2, theta=0.995"]
for _ in range(5000):
    j = rng.normal(size=(3, 2))
    g = rng.normal(size=2)
    c = np.abs(rng.normal(size=2))
    augmented = np.vstack([j, np.diag(np.sqrt(c))])
    u, s, vt = np.linalg.svd(augmented, full_matrices=False)
    rhs = np.linalg.lstsq(augmented.T, g, rcond=None)[0]
    p, *_ = solve_lsq_trust_region(2, 5, u.T @ rhs, s, vt.T, 2.0)
    _, h, predicted = select_step(np.zeros(2), j, c, g, p.copy(), p.copy(),
                                  np.ones(2), 2.0, np.full(2, -0.25),
                                  np.full(2, 0.25), 0.995)
    if np.linalg.norm(h-p) < 1e-12:
        kind = "TrustRegion"
    elif abs(h[0]*g[1]-h[1]*g[0]) < 1e-12:
        kind = "Gradient"
    elif abs(h[0]*p[1]-h[1]*p[0]) < 1e-12:
        kind = "TrustRegion"
    else:
        kind = "Reflected"
    if kind not in found:
        found.add(kind)
        steps.append("|".join([kind, values(j.ravel()), values(g), values(c),
                               values(p), values(h), format(predicted, ".17g")]))
    if len(found) == 3:
        break
assert len(found) == 3
Path(__file__).with_name("trust_region_reflective_steps.tsv").write_text(
    "\n".join(steps) + "\n"
)
