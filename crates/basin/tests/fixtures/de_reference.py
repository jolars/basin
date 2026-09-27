"""Regenerate DE references with SciPy 1.16.2 and NumPy 2.3.3.

Run: uv run --no-project --with scipy==1.16.2 --with numpy==2.3.3 python \
    crates/basin/tests/fixtures/de_reference.py

The donor cases call SciPy's mutation routines with explicit population values
and peer indices, independently of initialization, repair, and RNG differences.
The solution cases use deferred updates and no polishing. Rust tests consume
the committed TSV files without requiring Python or SciPy.
"""

from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import differential_evolution
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

assert scipy.__version__ == "1.16.2"
assert np.__version__ == "2.3.3"

mutations = ["rand1", "best1", "rand2", "best2", "randtobest1", "currenttobest1"]
population = np.array([[1, 2, 3], [2, -1, 4], [4, 5, -2],
                       [-3, 1, 2], [6, -4, 1], [0, 3, -5]], dtype=float)
peers = np.array([1, 2, 4, 5, 0])
donors = ["# SciPy 1.16.2; F=0.5; target=3; peers=1,2,4,5,0",
          "# mutation donor[0] donor[1] donor[2]"]
solutions = ["# SciPy 1.16.2; NumPy 2.3.3; deferred; polish=False; seed=42",
             "# strategy cost nfev x[0] x[1]"]


def objective(x):
    return (x[0] - 0.25)**2 + 2*(x[1] - 0.25)**2


for mutation in mutations:
    with DifferentialEvolutionSolver(
        objective, [(-2, 3), (-3, 4)], strategy=mutation + "bin",
        mutation=0.5, rng=42, polish=False,
    ) as solver:
        solver.population = population.copy()
        method = getattr(solver, "_" + mutation)
        donor = method(3, peers) if mutation == "currenttobest1" else method(peers)
        donors.append(mutation + " " + " ".join(format(x, ".17g") for x in donor))
    for crossover in ["bin", "exp"]:
        strategy = mutation + crossover
        result = differential_evolution(
            objective, [(-2, 3), (-3, 4)], strategy=strategy,
            mutation=(0.5, 1.0), recombination=0.9, popsize=12,
            maxiter=200, tol=0, atol=0, updating="deferred",
            init="random", rng=42, polish=False,
        )
        assert result.fun < 1e-12, (strategy, result.fun)
        solutions.append(f"{strategy} {result.fun:.17g} {result.nfev} "
                         + " ".join(format(x, ".17g") for x in result.x))

directory = Path(__file__).parent
(directory / "de_mutations.tsv").write_text("\n".join(donors) + "\n")
(directory / "de_solutions.tsv").write_text("\n".join(solutions) + "\n")
