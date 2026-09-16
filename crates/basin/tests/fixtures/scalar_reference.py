"""Generate scalar reference fixtures with SciPy 1.16.3 and NumPy 2.5.3.

Run with `roots` or `brackets` and redirect stdout to the matching TSV file.
The reference libraries remain external to the Rust test suite.
"""

import sys

import numpy as np
import scipy
from scipy import optimize
from scipy.optimize import elementwise

assert scipy.__version__ == "1.16.3"
assert np.__version__ == "2.5.3"


def emit(*values):
    print("\t".join(format(v, ".17g") if isinstance(v, float) else str(v)
                    for v in values))


if sys.argv[1] == "roots":
    print("# SciPy 1.16.3, NumPy 2.5.3, toms748 k=2, xtol=1e-12, rtol=4*eps")
    print("# name lower upper root value nfev nit first_five_evaluation_points")
    cases = [
        ("cubic", 1.0, 2.0, lambda x: x**3 - x - 2),
        ("cosine", 0.0, 1.0, lambda x: np.cos(x) - x),
        ("exponential", -2.0, 3.0, lambda x: np.exp(x) - 7),
        ("flat", 0.0, 2.0, lambda x: (x - 0.7)**3),
        ("steep", 0.0, 2.0, lambda x: x**9 - 1.1),
    ]
    for name, a, b, f in cases:
        trace = []

        def traced(x):
            trace.append(float(x))
            return f(x)

        x, result = optimize.toms748(traced, a, b, k=2, xtol=1e-12,
                                     rtol=4*np.finfo(float).eps,
                                     full_output=True)
        assert result.converged
        emit(name, a, b, float(x), float(f(x)), result.function_calls,
             result.iterations, ",".join(format(x, ".17g") for x in trace[:5]))
elif sys.argv[1] == "brackets":
    print("# SciPy 1.16.3, NumPy 2.5.3, factor=2, maxiter=2000")
    print("# kind target bound lower middle upper f_lower f_middle f_upper nfev nit")
    for target in [-100.5, 100.5, 0.35]:
        result = elementwise.bracket_root(lambda x: x - target, 0.0, 1.0,
                                          maxiter=2000)
        assert result.success
        a, b = map(float, result.bracket)
        fa, fb = map(float, result.f_bracket)
        emit("root", target, "none", a, "nan", b, fa, "nan", fb,
             result.nfev, result.nit)
    for target, bound in [(-100.5, None), (100.5, None), (0.35, None),
                          (0.999, 1.0)]:
        result = elementwise.bracket_minimum(lambda x: (x-target)**2, 0.0,
                                             xl0=-0.5, xr0=0.5, xmax=bound,
                                             maxiter=2000)
        assert result.success
        emit("minimum", target, "none" if bound is None else bound,
             *map(float, result.bracket), *map(float, result.f_bracket),
             result.nfev, result.nit)
else:
    raise SystemExit("expected roots or brackets")
