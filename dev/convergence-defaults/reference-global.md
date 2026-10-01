# Global and population reference policies

Status: versioned reference comparison and draft candidates for `Direct`,
`Gbnm`, `RandomSearch`, `SimulatedAnnealing`, `CmaEs`, `BoundedCmaEs`, `De`,
`GlobalBestPso`, and `Ssga`. The [source
inventory](inventory.md#global-and-population-stopping-records) records Basin's
current behavior. This comparison selects no default. A small region,
distribution, or population spread describes search progress, not a certificate
of a global optimum.

## Geometry and restarted local search

[SciPy 1.16.2
DIRECT](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.optimize.direct.html)
defaults to `len_tol=1e-6` and `vol_tol=1e-16`, with either test able to stop
the run. Both refer to the incumbent rectangle in the unit box. The length is
half its maximum side for the default locally biased `DIRECT_L` and half its
Euclidean diagonal for original DIRECT (`locally_biased=False`). The relative
volume is the rectangle's volume divided by the original box's volume. SciPy
warns that volume decays exponentially with dimension and can stop a large
dimensional run early. It also has an optional known-minimum target, disabled by
default, and separate iteration and evaluation budgets. Its `eps=1e-4` controls
potentially optimal rectangle selection, not stopping. Basin implements original
DIRECT, enables only the normalized half-diagonal `<= 1e-6`, and uses different
cost-tie handling. An equal `len_tol` means the same geometry only for SciPy's
original variant, not its default `DIRECT_L`. All geometry checks in Basin
compose with **OR**, require a usable incumbent, and observe the partition after
a complete subdivision sweep. They measure resolution near the incumbent. A
positive-volume threshold should be compared in log space; Basin's documented
zero-threshold request currently disables both checks because its implementation
requires a positive threshold.

[Luersen and Le Riche (2004)](https://doi.org/10.1016/j.compstruc.2004.03.072)
use a bounded Nelder–Mead local search and probabilistic restarts to seek more
basins. Basin's `Gbnm` local small, flat, or degenerate simplex selects a
restart. Neither the paper's restart idea nor local simplex completion gives a
finite-run global error bound. Basin has no numerical outer default: the
executor's evaluation budget, iteration budget, time budget, or known objective
target stops the outer search. Its active local point may worsen after restart,
so quality comparisons must read the historical incumbent rather than only the
published active point.

## Distribution and population spread

The algorithm-author [pycma r4.3.0 option
table](https://github.com/CMA-ES/pycma/blob/r4.3.0/cma/options_parameters.py)
enables several **OR** stopping heuristics, among them `tolx=1e-11`,
`tolfun=1e-11`, `tolfunhist=1e-12`, and a dimension- and population-dependent
`tolstagnation`; `ftarget=-inf` disables the target by default. Its [`tolx`
implementation](https://github.com/CMA-ES/pycma/blob/r4.3.0/cma/evolution_strategy.py)
requires **both** all coordinate distribution scales
`sigma * sigma_vec.scaling * sqrt(diag(C)) < tolx` and all coordinate evolution
path values `sigma * sigma_vec.scaling * pc < tolx`. The tagged source uses the
signed `pc` values in this comparison, so it is not a norm of that path. Its
fitness-range checks use current and historical populations. Basin's optional
`CmaEs` and `BoundedCmaEs` TolX is instead the single strict test
`sigma * max_i sqrt(eigenvalue_i(C)) < tolerance`; it is disabled by default.
The maximum principal-axis standard deviation is not the maximum coordinate
standard deviation or the evolution path. Basin observes the live distribution
at an initialized iteration boundary and publishes an evaluated mean as its
representative. The bounded solver ranks penalized genotypes but publishes
clipped phenotypes and raw costs; phenotype collapse at an active bound is
therefore not genotype distribution collapse. Hansen's [CMA-ES
tutorial](https://arxiv.org/abs/1604.00772), Appendix B.3, supplies the
algorithmic TolX and failure diagnostics; its scale guidance is not a
precision-independent default for Basin's `f32` backend.

[SciPy 1.16.2 differential
evolution](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.optimize.differential_evolution.html)
defaults to `tol=0.01`, `atol=0`, and stops when
`std(population_energies) <= atol + tol * abs(mean(population_energies))`. This
is the standard deviation of objective values, not coordinate spread or
best-cost change. Its default `best1bin` mutation, immediate updates, and
optional final local polishing differ from Basin's default synchronous
`rand1bin` with no automatic polish. Basin `De` has no enabled numerical stop.
If a SciPy-style test is evaluated as a candidate, observe the full
post-generation population with its finite raw costs and keep the population
mean and variance formula explicit. A constant-cost plateau satisfies the test
even when positions remain far apart; a zero mean removes the relative term.
Such a stop is a search-exhaustion heuristic, not demonstrated solution
accuracy. The mutation, crossover, and dithering variants share the outer
stopping contract but can change how often this heuristic fires.

The [Storn and Price (1997) paper](https://doi.org/10.1023/A:1008202821328)
establishes DE's strategy family, not a universal finite-run accuracy threshold.
The original [Kennedy and Eberhart (1995)
paper](https://doi.org/10.1109/ICNN.1995.488968) and [Bratton and Kennedy
(2007)](https://doi.org/10.1109/SIS.2007.368035) similarly provide PSO
algorithmic context without an interchangeable default stop for Basin's
global-best, boundary-handling, and velocity-limit variants. `GlobalBestPso` has
no approximate default; it stops on negative-infinity global best and fails
without a usable incumbent. A generation with unchanged global best can still
move particles. `Ssga` and `RandomSearch` likewise preserve elites while
sampling new candidates, so unchanged representative cost or position is not
population convergence. They have no enabled numerical default.

## Annealing and stochastic progress

[Kirkpatrick, Gelatt, and Vecchi
(1983)](https://doi.org/10.1126/science.220.4598.671) and [Hajek
(1988)](https://doi.org/10.1287/moor.13.2.311) relate search quality to the
cooling process; they do not give a transferable finite-time tolerance for
Basin's user-supplied temperature schedules. Basin `SimulatedAnnealing` has no
approximate default. An accepted negative-infinity cost is a structural stop; a
NaN current cost fails. A rejected proposal and reannealing trigger are not
outer convergence. For `RandomSearch`, [Bergstra and Bengio
(2012)](https://jmlr.org/papers/v13/bergstra12a.html) compare fixed-budget
random search in hyperparameter optimization. Their problem domain and sampling
measure differ from Basin's general box search, but support reporting quality
against cost evaluations rather than inventing an optimality test from an
unchanged elite. For `Ssga`, Basin's replace-worst and mutation rules need an
exact implementation-level stopping comparator before any imported
population-spread threshold can be treated as more than an exploratory
candidate.

## Candidate policies to measure

Retain current behavior as the control. Treat executor budgets, known targets,
and optional stall controls as distinct outcomes from solver convergence. For
stochastic algorithms, compare quality distributions over paired seeds at equal
raw evaluation budgets; record incumbent quality even when the published
representative can worsen. Any new numerical stop must require finite inputs and
a usable incumbent, and must report structural failure or no progress
separately. Absolute coordinate scales and objective spreads need problem
normalization; calibrate `f32` and `f64` independently. No candidate below is an
accepted default.

  | Solver and variants                          | Candidate comparison                                                                                                                                                        | Observation and safeguard                                                                                                                                                                                                                                                                             |
  | -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | `Direct` original                            | Current half-diagonal `<= 1e-6` versus nearby normalized length floors; optional relative-volume grid, each alone and in **OR**.                                            | Observe the best rectangle after a complete sweep. Stratify by dimension and fixed coordinates, and report width, volume, quality, and work. Keep unrepresentable splits and all-rejected partitions out of convergence counts. Fix the zero-threshold discrepancy test-first before evaluating zero. |
  | `Gbnm`                                       | Budget-driven control versus an optional executor incumbent-stall limit, labeled as a search heuristic.                                                                     | Record each local-small/flat/degenerate restart and historical best. Compare equal budgets over seeds; never promote a local restart to outer convergence.                                                                                                                                            |
  | `RandomSearch`                               | Budget-driven control versus optional executor best-improvement stall, measured only as early search cessation.                                                             | Observe the historical best after complete sampled populations. Include flat and rare-basin objectives; unchanged elites do not establish stationarity.                                                                                                                                               |
  | `SimulatedAnnealing`, all schedules          | Budget-driven control versus optional executor best-improvement stall, stratified by cooling and reannealing choices.                                                       | Record current and best costs, accepted and rejected proposals, temperature, and restarts. A cold schedule or rejected proposal is not an optimality test; keep NaN failure distinct.                                                                                                                 |
  | `CmaEs`                                      | Disabled TolX control versus calibrated principal-axis TolX; separately test pycma-inspired coordinate-and-path and fitness-history alternatives if implementation permits. | Check the live distribution after each completed generation, before another sample. Record mean, best sample, sigma, covariance axes, and any non-finite distribution; compare thresholds relative to starting coordinate scales and precision.                                                       |
  | `BoundedCmaEs`                               | Same distribution candidates, with a separate phenotype-spread diagnostic rather than silently replacing genotype TolX.                                                     | Record clipped and unclipped samples, penalty ranking, active bounds, and raw returned quality. A clipped population can collapse while genotype variance remains large.                                                                                                                              |
  | `De`, all mutation/crossover/dither variants | Budget-driven control versus optional SciPy-style post-generation objective spread; compare an executor incumbent-stall heuristic separately.                               | Use the whole finite population and the stated `std <= atol + rtol*abs(mean)` formula. Report constant-cost false positives, zero means, population diversity, and final best cost; no automatic polish in the control.                                                                               |
  | `GlobalBestPso`, boundary/velocity variants  | Budget-driven control versus optional executor best-improvement stall; explore normalized swarm position and velocity spread only as diagnostics.                           | Read post-generation particles and historical global best. Stratify by bound contact and velocity clipping; no-incumbent failure and negative-infinity success stay separate.                                                                                                                         |
  | `Ssga`                                       | Budget-driven control versus optional executor best-improvement stall; explore finite fitness spread only as a diagnostic.                                                  | Record the whole post-generation population, diversity, and incumbent. Elitist replacement can preserve an unchanged best while new candidates explore.                                                                                                                                               |

Evidence gaps before a large sweep: a precise implementation-level stopping
comparator for Basin's GBNM, `RandomSearch`, `GlobalBestPso`, and `Ssga`
variants; whether historical fitness range or distribution path diagnostics are
implementable without altering CMA-ES state and work accounting; dimension and
coordinate-scale choices for DIRECT's length and volume grid; and an independent
quality target for every stochastic benchmark. The [protocol](protocol.md) must
settle seed pairing, evaluation accounting, and quality definitions before any
early-stop rate is interpreted.
