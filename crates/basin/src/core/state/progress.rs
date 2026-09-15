//! Shared progress storage for external and new solvers.
//!
//! [`PointState`] carries a point and its cost. [`FirstOrderState`] also
//! carries its gradient. Solvers publish complete records with `replace`;
//! the executor mirrors counts, advances completed iterations, and retains
//! strict objective improvements. Algorithm settings, models, history, RNGs,
//! and scratch buffers belong in the solver.
//!
//! These are additive state types. Existing solvers keep their existing
//! state types and constructors. Both shared states preserve all six raw
//! [`EvalCounts`] categories through their `counts` and `best_counts` readers.
//! Their [`State`] and [`GradientState`] readers use the existing Basin 1.x
//! folded accounting described by [`CountsMirror`].
//!
//! Both states implement [`EvaluatedState`], [`RawEvaluationState`],
//! [`IncumbentState`], and [`ObjectiveIncumbentState`]; [`FirstOrderState`]
//! additionally implements [`EvaluatedGradientState`]. These unsealed traits
//! also support external states. Use `require_evaluated_state` to validate
//! publication, `max_evaluations` for raw budgets, and `target_objective` or
//! `no_objective_improvement` for controls requiring objective ordering.
//!
//! # Lifecycle
//!
//! Construction supplies an unevaluated seed. In [`Solver::init`](crate::Solver::init),
//! call the state's `reset` method, reset solver-owned machinery, evaluate
//! the seed through [`Problem`](crate::Problem), and `replace` the record.
//! Each successful iteration must likewise leave a complete record. `reset`
//! retains the current parameter as a warm-start seed while clearing derived
//! values and per-run bookkeeping. [`State::reset_best`] clears only the
//! incumbent, as it does for the existing state types.
//!
//! Exact continuation uses a solver-aware [`ExactCheckpoint`](crate::ExactCheckpoint)
//! and [`Executor::resume_from_checkpoint`](crate::Executor::resume_from_checkpoint),
//! which skips initialization. Shared progress alone does not contain the
//! solver's evolution data, so these types do not implement
//! [`ExactResumeState`](super::ExactResumeState).
//!
//! # Example
//!
//! An external solver needs no custom state implementation or counters:
//!
//! ```
//! use basin::{
//!     CostFunction, Executor, FirstOrderState, Gradient, Problem, Solver,
//!     State, TerminationReason,
//! };
//! use std::convert::Infallible;
//!
//! struct Quadratic;
//! impl CostFunction for Quadratic {
//!     type Param = Vec<f64>;
//!     type Output = f64;
//!     type Error = Infallible;
//!     fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
//!         Ok(x.iter().map(|v| v * v).sum())
//!     }
//! }
//! impl Gradient for Quadratic {
//!     type Gradient = Vec<f64>;
//!     fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
//!         Ok(x.iter().map(|v| 2.0 * v).collect())
//!     }
//! }
//!
//! struct Descent;
//! impl Solver<Quadratic, FirstOrderState<Vec<f64>>> for Descent {
//!     type Error = Infallible;
//!     fn init(
//!         &mut self,
//!         problem: &mut Problem<Quadratic>,
//!         mut state: FirstOrderState<Vec<f64>>,
//!     ) -> Result<FirstOrderState<Vec<f64>>, Infallible> {
//!         state.reset();
//!         let x = state.param().clone();
//!         let (cost, gradient) = problem.cost_and_gradient(&x)?;
//!         state.replace(x, cost, gradient).unwrap();
//!         Ok(state)
//!     }
//!     fn next_iter(
//!         &mut self,
//!         problem: &mut Problem<Quadratic>,
//!         mut state: FirstOrderState<Vec<f64>>,
//!     ) -> Result<(FirstOrderState<Vec<f64>>, Option<TerminationReason>), Infallible> {
//!         let (x, _, gradient) = state.current().unwrap();
//!         let next = x.iter().zip(gradient).map(|(x, g)| x - 0.25 * g).collect();
//!         let (cost, gradient) = problem.cost_and_gradient(&next)?;
//!         state.replace(next, cost, gradient).unwrap();
//!         Ok((state, None))
//!     }
//! }
//! let result = Executor::new(
//!     Quadratic, Descent, FirstOrderState::new(vec![2.0]),
//! ).require_evaluated_state()
//!     .max_evaluations(basin::EvaluationKind::Gradient, 100)
//!     .target_objective(0.01)
//!     .max_iter(3).run().unwrap();
//! assert_eq!(result.state.cost(), 0.0625);
//! assert_eq!(result.state.counts().gradient_evals, 4);
//! ```

use super::{
    CountsMirror, EvaluatedGradientState, EvaluatedState, GradientState,
    IncumbentRef, IncumbentState, ObjectiveIncumbentState, RawEvaluationState,
    State,
};
use crate::core::math::{Scalar, VectorLen};
use crate::core::problem::EvalCounts;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
struct Incumbent<V, F: Scalar> {
    param: V,
    cost: F,
    iter: u64,
    counts: EvalCounts,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
struct Progress<V, D, F: Scalar> {
    param: V,
    // Cost and derivatives become available together, including after reset.
    evaluated: Option<(F, D)>,
    best: Option<Incumbent<V, F>>,
    iter: u64,
    counts: EvalCounts,
}

impl<V, D, F: Scalar> Progress<V, D, F> {
    fn new(param: V) -> Self {
        Self {
            param,
            evaluated: None,
            best: None,
            iter: 0,
            counts: EvalCounts::default(),
        }
    }

    fn reset(&mut self) {
        self.evaluated = None;
        self.best = None;
        self.iter = 0;
        self.counts = EvalCounts::default();
    }

    fn replace(&mut self, param: V, cost: F, derivatives: D) {
        self.param = param;
        self.evaluated = Some((cost, derivatives));
    }
}

impl<V: Clone, D, F: Scalar> Progress<V, D, F> {
    fn update_best(&mut self) {
        let Some((cost, _)) = self.evaluated.as_ref() else {
            return;
        };
        if !cost.is_nan()
            && *cost != F::infinity()
            && self.best.as_ref().is_none_or(|best| *cost < best.cost)
        {
            self.best = Some(Incumbent {
                param: self.param.clone(),
                cost: *cost,
                iter: self.iter,
                counts: self.counts,
            });
        }
    }
}

/// Shared single-point progress for derivative-free and external solvers.
///
/// [`replace`](Self::replace) publishes a matching point and cost. The
/// executor retains the lowest-cost published record separately, so a worse
/// current point cannot overwrite the historical best. Equal costs preserve
/// the incumbent and its publication metadata. NaN and positive infinity
/// never establish an incumbent. Negative infinity can be retained but does
/// not itself signal an unbounded problem.
///
/// Before evaluation, [`current`](Self::current) and [`best`](Self::best)
/// return `None`. [`State::cost`] panics without a current record, and
/// [`State::best_param`] panics without an incumbent, including when all
/// published costs are NaN or positive infinity. [`State::best_cost`] returns
/// positive infinity in that case, and best iteration/evaluation readers
/// return zero. These objective-only semantics are unsuitable for solvers
/// whose incumbent selection prioritizes constraint feasibility.
///
/// See the [module documentation](self) for initialization and continuation.
/// With `serde`, serialization is available when `V` and `F` support it.
///
/// # Backends
///
/// `Vec<F>`, `nalgebra::DVector<F>`, `ndarray::Array1<F>`, and `faer::Col<F>`,
/// with `F = f64` (default) or `f32`. No vector math capability is required:
/// scalar parameters and other `Clone` parameter types also implement [`State`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct PointState<V, F: Scalar = f64> {
    progress: Progress<V, (), F>,
}

impl<V, F: Scalar> PointState<V, F> {
    /// Read the evaluated current point and cost, or `None` for a seed.
    pub fn current(&self) -> Option<(&V, F)> {
        let (cost, ()) = self.progress.evaluated.as_ref()?;
        Some((&self.progress.param, *cost))
    }

    /// Replace the current point and cost together without changing counters
    /// or the incumbent. The executor considers the final record returned
    /// from initialization or an iteration, including a clean mid-step stop.
    /// Intermediate replacements within a step are not retained automatically.
    pub fn replace(&mut self, param: V, cost: F) {
        self.progress.replace(param, cost, ());
    }
}

/// Shared first-order progress with a matching point, cost, and gradient.
///
/// [`replace`](Self::replace) requires the point and gradient together and
/// checks their lengths before changing any field. [`GradientState::gradient`]
/// is therefore available whenever [`current`](Self::current) is available.
/// The gradient belongs to the current point; historical incumbents retain
/// only their point, cost, and publication metadata.
///
/// Incumbent selection, unavailable readers, lifecycle, and optional serde
/// support follow [`PointState`]. This state stores progress while the solver
/// owns algorithm machinery such as an inverse Hessian or L-BFGS history.
///
/// # Backends
///
/// `Vec<F>`, `nalgebra::DVector<F>`, `ndarray::Array1<F>`, and `faer::Col<F>`,
/// with `F = f64` (default) or `f32`. Replacement requires [`VectorLen`];
/// [`State`] and [`GradientState`] require only `V: Clone`.
///
/// A point-only state does not advertise a gradient capability:
///
/// ```compile_fail
/// use basin::{GradientState, PointState};
/// fn requires_gradient<S: GradientState>(_: S) {}
/// requires_gradient(PointState::<Vec<f64>>::new(vec![1.0]));
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct FirstOrderState<V, F: Scalar = f64> {
    progress: Progress<V, V, F>,
}

impl<V, F: Scalar> FirstOrderState<V, F> {
    /// Read the evaluated current point, cost, and gradient, or `None` for a seed.
    pub fn current(&self) -> Option<(&V, F, &V)> {
        let (cost, gradient) = self.progress.evaluated.as_ref()?;
        Some((&self.progress.param, *cost, gradient))
    }
}

impl<V: VectorLen, F: Scalar> FirstOrderState<V, F> {
    /// Replace the current point, cost, and gradient as one record.
    ///
    /// The point and gradient must have equal lengths; on error the entire
    /// previous state is retained. The new dimension may differ from the
    /// previous record's dimension. Numerical validity and whether a point
    /// was actually evaluated remain the solver's responsibility.
    ///
    /// Counters and the incumbent are unchanged until the executor publishes
    /// the returned state, as with [`PointState::replace`].
    pub fn replace(
        &mut self,
        param: V,
        cost: F,
        gradient: V,
    ) -> Result<(), GradientDimensionMismatch> {
        let param_len = param.vec_len();
        let gradient_len = gradient.vec_len();
        if param_len != gradient_len {
            return Err(GradientDimensionMismatch {
                param_len,
                gradient_len,
            });
        }
        self.progress.replace(param, cost, gradient);
        Ok(())
    }
}

/// A point and gradient passed to [`FirstOrderState::replace`] differ in length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GradientDimensionMismatch {
    /// Number of coordinates in the proposed point.
    pub param_len: usize,
    /// Number of coordinates in the proposed gradient.
    pub gradient_len: usize,
}

impl std::fmt::Display for GradientDimensionMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "point length {} does not match gradient length {}",
            self.param_len, self.gradient_len,
        )
    }
}

impl std::error::Error for GradientDimensionMismatch {}

fn cost_work(counts: &EvalCounts) -> u64 {
    counts.cost_evals + counts.residual_evals
}

fn gradient_work(counts: &EvalCounts) -> u64 {
    counts.gradient_evals
        + counts.jacobian_evals
        + counts.hessian_evals
        + counts.hessian_product_evals
}

macro_rules! impl_progress_state {
    ($state:ident, $cost_work:path) => {
        impl<V, F: Scalar> $state<V, F> {
            /// Construct an unevaluated seed with zero counters and no incumbent.
            pub fn new(param: V) -> Self {
                Self {
                    progress: Progress::new(param),
                }
            }

            /// Retain the current parameter as a seed and clear all evaluated
            /// data, the incumbent, iteration number, and evaluation counts.
            /// Call this during fresh solver initialization, then reevaluate
            /// the seed. Exact checkpoint resumes skip initialization.
            pub fn reset(&mut self) {
                self.progress.reset();
            }

            /// Read the retained incumbent point and cost, if one exists.
            pub fn best(&self) -> Option<(&V, F)> {
                self.progress
                    .best
                    .as_ref()
                    .map(|best| (&best.param, best.cost))
            }

            /// Raw counts mirrored from the problem at the latest publication
            /// boundary. Fresh and nested runs report per-run work; exact
            /// checkpoint resumes report cumulative work across the continuation.
            pub fn counts(&self) -> &EvalCounts {
                &self.progress.counts
            }

            /// Raw counts at the boundary that selected the incumbent, if any.
            /// These include all work charged before publication, which can
            /// exceed the work at the evaluation that found the point.
            pub fn best_counts(&self) -> Option<&EvalCounts> {
                self.progress.best.as_ref().map(|best| &best.counts)
            }
        }

        impl<V: Clone, F: Scalar> State for $state<V, F> {
            type Param = V;
            type Float = F;

            fn iter(&self) -> u64 {
                self.progress.iter
            }
            fn increment_iter(&mut self) {
                self.progress.iter += 1;
            }
            fn cost_evals(&self) -> u64 {
                $cost_work(&self.progress.counts)
            }
            fn param(&self) -> &V {
                &self.progress.param
            }
            fn cost(&self) -> F {
                self.progress
                    .evaluated
                    .as_ref()
                    .expect("current point has not been evaluated")
                    .0
            }
            fn best_param(&self) -> &V {
                &self
                    .progress
                    .best
                    .as_ref()
                    .expect("no incumbent has been selected")
                    .param
            }
            fn best_cost(&self) -> F {
                self.progress
                    .best
                    .as_ref()
                    .map_or(F::infinity(), |best| best.cost)
            }
            fn best_iter(&self) -> u64 {
                self.progress.best.as_ref().map_or(0, |best| best.iter)
            }
            fn best_cost_evals(&self) -> u64 {
                self.progress
                    .best
                    .as_ref()
                    .map_or(0, |best| $cost_work(&best.counts))
            }
            fn update_best(&mut self) {
                self.progress.update_best();
            }
            fn reset_best(&mut self) {
                self.progress.best = None;
            }
        }

        impl<V: Clone, F: Scalar> CountsMirror for $state<V, F> {
            fn mirror(&mut self, counts: &EvalCounts) {
                self.progress.counts = *counts;
            }
        }

        impl<V: Clone, F: Scalar> RawEvaluationState for $state<V, F> {
            fn raw_counts(&self) -> &EvalCounts {
                self.counts()
            }
        }

        impl<V: Clone, F: Scalar> IncumbentState for $state<V, F> {
            fn incumbent_record(&self) -> Option<IncumbentRef<'_, V, F>> {
                let best = self.progress.best.as_ref()?;
                Some(IncumbentRef {
                    param: &best.param,
                    cost: best.cost,
                    iter: best.iter,
                    counts: &best.counts,
                })
            }
        }

        impl<V: Clone, F: Scalar> ObjectiveIncumbentState for $state<V, F> {}
    };
}

impl_progress_state!(PointState, EvalCounts::total_work);
impl_progress_state!(FirstOrderState, cost_work);

impl<V: Clone, F: Scalar> EvaluatedState for PointState<V, F> {
    fn current_record(&self) -> Option<(&V, F)> {
        self.current()
    }
}

impl<V: Clone, F: Scalar> EvaluatedState for FirstOrderState<V, F> {
    fn current_record(&self) -> Option<(&V, F)> {
        self.current().map(|(param, cost, _)| (param, cost))
    }
}

impl<V: Clone, F: Scalar> EvaluatedGradientState for FirstOrderState<V, F> {
    fn current_gradient_record(&self) -> Option<(&V, F, &V)> {
        self.current()
    }
}

impl<V: Clone, F: Scalar> GradientState for FirstOrderState<V, F> {
    fn gradient(&self) -> Option<&V> {
        self.progress
            .evaluated
            .as_ref()
            .map(|(_, gradient)| gradient)
    }

    fn gradient_evals(&self) -> u64 {
        gradient_work(&self.progress.counts)
    }

    fn best_gradient_evals(&self) -> u64 {
        self.best_counts().map_or(0, gradient_work)
    }
}
