"""Locked final-output comparisons: SciPy 1.16.3 and NLopt 2.10.0."""
import numpy as np
import scipy
from scipy.optimize import minimize
import nlopt

assert scipy.__version__ == "1.16.3"
assert nlopt.__version__ == "2.10.0"


def cost(x):
    a, b, c, d = x
    return a * d * (a + b + c) + c


def grad(x):
    a, b, c, d = x
    return np.array([d * (2 * a + b + c), a * d, a * d + 1, a * (a + b + c)])


def eq(x):
    return np.dot(x, x) - 40


def iq(x):
    return 25 - np.prod(x)


def jiq(x):
    return -np.array([np.prod(np.delete(x, i)) for i in range(4)])


start = [1., 5., 5., 1.]
result = minimize(cost, start, jac=grad, method="SLSQP", bounds=[(1, 5)] * 4,
                  constraints=[{"type": "eq", "fun": eq, "jac": lambda x: 2 * x},
                               {"type": "ineq", "fun": lambda x: -iq(x), "jac": lambda x: -jiq(x)}],
                  options={"ftol": 1e-10, "maxiter": 100})
assert result.success, result.message


def callback(f, g):
    def evaluate(x, derivative):
        if derivative.size:
            derivative[:] = g(x)
        return f(x)
    return evaluate


opt = nlopt.opt(nlopt.LD_SLSQP, 4)
opt.set_lower_bounds([1.] * 4)
opt.set_upper_bounds([5.] * 4)
opt.set_min_objective(callback(cost, grad))
opt.add_equality_constraint(callback(eq, lambda x: 2 * x), 1e-10)
opt.add_inequality_constraint(callback(iq, jiq), 1e-10)
opt.set_ftol_abs(1e-10)
opt.set_xtol_abs(1e-10)
opt.set_maxeval(1000)
x_nlopt = opt.optimize(start)
assert opt.last_optimize_result() > 0
print("# library status f x0 x1 x2 x3 equality inequality projected_stationarity")
for name, status, x in [("scipy-1.16.3", result.status, result.x),
                         ("nlopt-2.10.0", opt.last_optimize_result(), x_nlopt)]:
    # Estimate multipliers from the free variables; x0 is at its lower bound.
    jac = np.stack([2*x, jiq(x)], axis=1)
    multipliers = np.linalg.lstsq(jac[1:], -grad(x)[1:], rcond=None)[0]
    lag = grad(x) + jac @ multipliers
    stationarity = np.max(np.abs(x - np.clip(x-lag, 1, 5)))
    print(name, status, *(format(v, ".17e") for v in [cost(x), *x, eq(x), iq(x), stationarity]))
