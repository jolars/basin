# Migrating convergence settings

Basin configures numerical convergence on the solver and execution limits on
`Executor`, `InnerExecutor`, or `RunControl`. The criterion facility and the old
setter names remain available during Basin 1.x, with deprecation notices. Their
removal is scheduled for Basin 2.0 in [TODO.md](TODO.md).

```rust
use basin::{Executor, GradientDescent};

let solver =
    GradientDescent::new(1e-3).with_absolute_gradient_tolerance(1e-6);
let result = Executor::from_start(problem, solver, x0)
    .max_iter(50_000)
    .max_cost_evals(100_000)
    .run()?;
```

## Optional checks and algorithm controls

Optional convergence setters accept either a finite nonnegative scalar or
`None`. Calling a setter again replaces that test. `None` disables it; zero
requests an exact-zero threshold. The deprecated setters retain their original
validation and zero semantics: for example, LM's `with_tol_grad(0.0)` disables
its check, so migrate that call to `with_absolute_gradient_tolerance(None)`.

Distinct enabled tests combine with OR: the first observed stop ends the run.
Nelder–Mead's enabled simplex-size and simplex-cost tests form one AND group.
Tightening a gradient threshold does not prevent a step test, budget, or
algorithm safeguard from ending the run earlier. `TerminationReason` continues
to report the stop encountered; its existing variant names remain available.

New observational checks are disabled by default. Solvers with native checks
retain their existing defaults. In particular, LM's four native defaults,
Gauss–Newton's gradient threshold, and L-BFGS-B's projected-gradient threshold
are not supplemented by second copies of the same test.

Some settings control an algorithm's progression or recovery rather than add
an optional stop. These continue to take numeric values: Powell-family initial
and final radii, MADS's minimum poll size, scalar position tolerances, GBNM's
restart/degeneracy thresholds, and line-search conditions. Setting an optional
radius check to `None` does not remove a solver's final-radius schedule.

## Replacing executor criteria

Move each numerical check from `.terminate_on(...)` to the solver. The table
lists the supported replacements; methods requiring gradients or a simplex are
available only for the corresponding solver families.

| Deprecated criterion | Replacement |
| --- | --- |
| `GradientTolerance(tol)` | `solver.with_absolute_gradient_tolerance(tol)` |
| `RelativeGradientTolerance::new(tol)` | `solver.with_relative_gradient_tolerance(tol)` |
| `ProjectedGradientTolerance::from_problem(&problem, tol)` | `solver.with_absolute_projected_gradient_tolerance(tol)` |
| `ParamTolerance::new(tol)` | `solver.with_absolute_step_tolerance(tol)` |
| `RelativeParamTolerance::new(tol)` | `solver.with_relative_step_tolerance(tol)` |
| `CostTolerance::new(tol)` | `solver.with_absolute_cost_change_tolerance(tol)` |
| `RelativeCostTolerance::new(tol)` | `solver.with_relative_cost_change_tolerance(tol)` |
| `SimplexTolerance::new(x, f)` | `solver.with_absolute_simplex_size_tolerance(x).with_absolute_simplex_cost_tolerance(f)` |
| `CmaEsTolerance::new(tol)` | `solver.with_absolute_distribution_size_tolerance(tol)` |
| `RhoTolerance::new(tol)` on Powell solvers | `solver.with_absolute_radius_tolerance(tol)` |
| `RhoTolerance::new(tol)` on Solis–Wets | `solver.with_absolute_step_size_tolerance(tol)` |
| `MeshTolerance::new(tol)` | `solver.with_absolute_poll_size_tolerance(tol)` |
| `MaxIter(n)` | `executor.max_iter(n)` |
| `MaxCostEvals(n)` | `executor.max_cost_evals(n)` |
| `MaxGradientEvals(n)` | `executor.max_gradient_evals(n)` |
| `MaxTime::new(duration)` | `executor.max_time(duration)` |
| `TargetCost(target)` | `executor.target_cost(target)` |
| `NoImprovement::new(patience, delta)` | `executor.no_improvement(patience, delta)` |
| `NoAcceptance::new(patience)` | `executor.no_acceptance(patience)` |

The common absolute gradient test uses the Euclidean norm. Its relative form
compares against the first finite gradient norm in the run. Observed step tests
use Euclidean distance; the relative step scales by the current parameter norm.
Observed relative cost change scales by the previous absolute cost. Near a zero
reference value, consider enabling an absolute check as well.

Projected-gradient checks use the infinity norm of
`x - projection(x - gradient)` with the current problem's bounds. L-BFGS-B's
canonical setter configures its native test and reports
`ProjectedGradientTolerance`; its deprecated setter retains `SolverConverged`.
Simplex checks use the maximum infinity-norm distance and absolute cost
difference from the best vertex. CMA-ES retains its strict TolX inequality,
`sigma * max_axis_std < tolerance`; zero therefore never satisfies TolX.

LM and Gauss–Newton use the infinity norm of `Jᵀr`, while TRF uses its
bound-scaled gradient. LM's relative model-reduction test includes actual
reduction, predicted reduction, and gain ratio. It differs from an observed
cost-change check. LM's relative step test uses its internal trial-step formula;
it is not a second observational iterate-change test.

## Renamed setters

| Solver or component | Deprecated name | Canonical name |
| --- | --- | --- |
| BFGS, L-BFGS, L-BFGS-B | `with_epsilon` | `with_relative_curvature_tolerance` |
| L-BFGS-B | `with_tol_pg` | `with_absolute_projected_gradient_tolerance` |
| Gauss–Newton, LM, LM QR | `with_tol_grad` | `with_absolute_gradient_tolerance` |
| TRF | `with_tol_grad` | `with_absolute_scaled_gradient_tolerance` |
| LM, LM QR | `with_tol_grad_rel` | `with_gradient_orthogonality_tolerance` |
| LM, LM QR | `with_tol_cost_rel` | `with_relative_model_reduction_tolerance` |
| LM, LM QR | `with_tol_step_rel` | `with_relative_step_tolerance` |
| LM QR | `with_rank_tolerance` | `with_relative_rank_tolerance` |
| NEWUOA, BOBYQA, LINCOA, COBYLA | `with_rho_beg`, `with_rho_end` | `with_initial_radius`, `with_final_radius` |
| Solis–Wets | `with_rho_init` | `with_initial_step_size` |
| MADS | `with_min_poll_size` | `with_minimum_poll_size` |
| GBNM | `with_small_tolerance` | `with_normalized_simplex_size_tolerance` |
| GBNM | `with_flat_tolerance` | `with_absolute_simplex_cost_tolerance` |
| GBNM | `with_degeneracy_tolerances(edge, determinant)` | `with_edge_ratio_tolerance(edge).with_normalized_determinant_tolerance(determinant)` |
| Barrier method | `with_tol` | `with_absolute_duality_gap_tolerance` |
| Barrier method | `with_phase_one_tol` | `with_absolute_phase_one_gap_tolerance` |
| Augmented Lagrangian | `with_tol` | `with_absolute_feasibility_tolerance` |
| Brent, derivative Brent, golden section | `Type::with_tol(relative, absolute)` | `Type::new().with_relative_position_tolerance(relative).with_absolute_position_tolerance(absolute)` |
| Brent root | `with_tol(relative, absolute)` | `with_relative_position_tolerance(relative).with_absolute_position_tolerance(absolute)` |
| More–Thuente | `ftol`, `gtol`, `xtol` | `with_sufficient_decrease_coefficient`, `with_curvature_coefficient`, `with_relative_bracket_tolerance` |
| Backtracking | `c` | `with_sufficient_decrease_coefficient` |
| Wolfe | `c1`, `c2` | `with_sufficient_decrease_coefficient`, `with_curvature_coefficient` |
| Hager–Zhang | `delta_sigma` | `with_wolfe_coefficients` |
| Hager–Zhang | `epsilon` | `with_relative_cost_relaxation_tolerance` |

Line-search coefficients describe sufficient decrease and curvature; they are
not convergence tolerances for the outer optimization problem. Consult each
method's rustdoc for its admissible interval and defaults.

## Application stops and composed solves

Replace custom `TerminationCriterion` implementations with closure hooks:

```rust
let result = Executor::from_start(problem, solver, x0)
    .stop_when(|state| {
        application_should_stop(state)
            .then_some(basin::TerminationReason::UserRequested)
    })
    .run()?;
```

For repeated inner solves, `InnerExecutor::stop_when_factory` and composed
`inner_stop_when_factory` methods create fresh closure history per run. A direct
`stop_when` closure keeps its captures when a `RunControl` or `InnerExecutor`
is reused. Built-in stall and clock history resets on each borrowed run.

Use `run_loop_with_control(problem, state, solver, control)` with a reusable
`RunControl` instead of `run_loop` and a criterion vector. Direct checks run in
this order: iteration, cost evaluations, gradient evaluations, time, target,
improvement stall, acceptance stall, then application hooks. Deprecated
registrations retain insertion order within the hook list. Solver convergence
follows these controls; cancellation precedes them at top-level boundaries.

The iteration default remains 1,000. Limits are observed after initialization
and between iterations. They do not cap work inside an iteration. Cost budgets
use the state's mirrored count; new gradient budgets use the wrapper's count.
Borrowed runs use per-run counts, and time starts at the first check after
initialization. A zero budget still permits initialization.

Barrier and augmented-Lagrangian methods now take a configured inner solver:

```rust
let inner = GradientDescent::with_line_search(Backtracking::new())
    .with_absolute_gradient_tolerance(1e-8);
let solver = BarrierMethod::with_inner_solver(inner)
    .with_absolute_duality_gap_tolerance(1e-8);
```

These constructors add no inner convergence threshold. The deprecated `new`
constructors retain their extra `1e-8` gradient check, including overrides from
`with_inner_grad_tol`. Migrate that value to the supplied inner solver.

`ResumableInner::configure_segment` replaces `segment_criteria`. Its default
implementation bridges existing third-party criterion implementations in 1.x.
Built-in CMA chains retain their historical per-segment TolX default unless
`with_absolute_distribution_size_tolerance` is explicitly configured, including
`None`.

## Types, history, and checkpoints

Existing third-party `Solver` implementations continue to work through the
default lifecycle hooks. Wrappers around a configured solver should forward
`reset_convergence` and `check_convergence` alongside `init` and `next_iter`.

Some setters return `ConfiguredSolver`, which carries fixed typed slots for
optional checks. Construct it through solver setters and normally let Rust infer
its type. It forwards algorithm configuration and composition traits. A solver
without a configured check retains its original backend requirements; enabling
a step or norm check adds the corresponding operations at compile time.

Fresh runs, state-only resumes, and fresh inner solves reset convergence
history. Exact solver-and-state checkpoints retain it. Rechecking the same
iteration boundary does not advance observed step or cost history. Reattach
outer execution controls, observers, and cancellation when resuming a checkpoint.

With `serde`, fixed convergence slots and inner iteration/evaluation/time
budgets serialize with their settings and solver history. An `InnerExecutor`
with application hooks, deprecated criteria, or erased target/stall checks
returns a serialization error; these cannot be silently dropped from an exact
checkpoint. Old checkpoint formats are not a stable cross-version wire format.
