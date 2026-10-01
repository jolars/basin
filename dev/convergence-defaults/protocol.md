# Experimental protocol

Status: draft requirements for step 4. No candidate grid, target values, problem
partition, budget, or selection thresholds have been frozen. This document
prepares later design work; it does not report experiments.

## Questions the experiments must answer

For each solver and applicable variant, determine whether its stopping policy
ends runs at useful accuracy, avoids substantial premature termination, and
avoids excessive work after attaining that accuracy. Evaluate the policy against
external quality measures, not its own convergence predicate.

Measure objective accuracy at several targets with known or independently
validated reference values. Specify absolute and relative scales and numerical
floors. Add stationarity and feasibility where applicable, parameter error where
identifiable, and bracket/root accuracy for root finders. Distinguish
convergence to another local minimum from premature stopping. Stochastic and
global solvers also need repeated-run success and quality within budgets.

The [TODO plan](../../TODO.md#convergence-defaults-investigation) records the
initial research leads. The reference survey must establish versioned formulas
before treating similarly named tolerances as equivalent.

## Choices to resolve before calibration

  | Choice                 | Required record                                                                   | Status  |
  | ---------------------- | --------------------------------------------------------------------------------- | ------- |
  | Cases                  | Families, dimensions, starts, constraints, derivatives, and solver applicability  | Pending |
  | Reference answers      | Provenance, verification, achievable accuracy, and treatment of uncertain optima  | Pending |
  | Quality targets        | Objective, stationarity, feasibility, parameter, and root measures with scales    | Pending |
  | Partitions             | Calibration and held-out families, with related variants kept together            | Pending |
  | Candidates             | Formulas, enabled criteria, composition, and logarithmic threshold grids          | Pending |
  | Resources              | Evaluation and time budgets, run counts, paired seeds, and retry policy           | Pending |
  | Aggregation            | Family weights, failure handling, uncertainty, and worst-case reporting           | Pending |
  | Selection              | Reliability requirements, acceptable extra work, and rules for retaining defaults | Pending |
  | Precision and backends | Separate precision calibration and finalist verification matrix                   | Pending |

Include rescaled and shifted objectives, rescaled parameters, zero and nonzero
residual minima, active bounds, fixed coordinates, rank deficiency, non-finite
inputs, and stagnation away from a solution where relevant. Cover analytic and
finite-difference derivatives and noisy evaluations where supported.

The NIST StRD harness requested in issue #109 has not been obtained. Record its
availability and include both starting points for each applicable case. If it
remains unavailable, independently assemble the cases and record provenance.
Identify corpus gaps before deciding that a solver has sufficient coverage.

## Measurement requirements

Record the returned point and independently checked quality, termination reason,
all passing criteria and measurements, observation stage, and work by evaluation
kind. Capture accepted/rejected steps and inner-solver work where needed to
explain a stop. Distinguish returned points from best sampled trials. Record
fused callback and finite-difference accounting rather than treating the sum of
evaluation categories as a count of separate calls.

Keep validation evaluations outside solve costs. Retain failures, exhausted
budgets, numerical no-progress stops, and infeasible results in the summaries.
Preserve numerical safeguards during stricter continuation. Estimate both
improvement available after an earlier stop and work beyond first attainment of
each quality target. Trace-based evaluation of alternative thresholds is valid
only when those thresholds do not change the trajectory; confirm shortlisted
policies with actual solves at the correct observation stages.

Use paired seeds and report uncertainty for stochastic methods. Timing is
supplementary to numerical quality and evaluation accounting. Document algorithm
differences before attributing a reference gap to stopping policy.

## Freezing and validation

Record the reviewed protocol revision and a decision ID before calibration. Keep
all transformed copies, dimensions, and starts from a problem family in the same
partition. Freeze the selected candidates before evaluating held-out families.
If validation results prompt retuning, record that those families have become
development data and arrange fresh validation.

Calibrate `f32` deliberately; do not derive its policy mechanically from a
double-precision default. Verify finalists on every claimed backend and relevant
variant. The [run record conventions](runs/README.md) specify what to retain for
reproduction. No experimental result can settle a default without a linked
[decision record](decisions.md).
