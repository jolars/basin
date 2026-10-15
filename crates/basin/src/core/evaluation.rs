//! Shared hard evaluation budgets, including work performed by adapters.
//!
//! Every executor installs its limits on the problem's context. Inner solves
//! and finite-difference probes share that ledger. Work is reserved before
//! callbacks, including complete cost batches and fused outputs.

use std::fmt;
use std::sync::{Arc, Mutex};

use crate::{EvalCounts, EvaluationKind, ExecutionLimit};

/// A callback result or a framework evaluation stop.
pub type EvaluationResult<T, E> = Result<T, EvaluationError<E>>;

mod problem;
pub use problem::ContextProblem;
pub(crate) use problem::{
    forward_constraint_jacobian_context, forward_cost_context,
    forward_equalities_context, forward_gradient_context,
    forward_hessian_context, forward_hessian_product_context,
    forward_inequalities_context, forward_jacobian_context,
    forward_mini_batch_context, forward_residual_context,
};

/// Hard limits on callback categories and their sum.
///
/// A limit is an upper bound, rather than a request to perform exactly that
/// much work. A fused call or batch that cannot fit is declined as a whole.
/// Zero permits no work in that category. Unspecified categories are unlimited.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EvaluationBudget {
    limits: [Option<u64>; 7],
    pub(crate) cost: Option<u64>,
    pub(crate) gradient: Option<u64>,
}

/// Opaque reservation passed through adapters while evaluating a cost batch.
///
/// Transparent adapters forward both [`crate::CostFunction::prepare_cost_batch`]
/// and [`crate::CostFunction::cost_in_batch`] to preserve reservations around
/// inner budgeted problems. Ordinary problems use the default implementations.
#[derive(Debug)]
pub struct CostBatchReservation {
    pub(crate) context: Option<EvaluationContext>,
}

impl EvaluationBudget {
    /// Start with every category unlimited.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set or replace a hard limit.
    pub fn with_limit(mut self, kind: EvaluationKind, limit: u64) -> Self {
        self.limits[index(kind)] = Some(limit);
        self
    }
}

fn index(kind: EvaluationKind) -> usize {
    match kind {
        EvaluationKind::Cost => 0,
        EvaluationKind::Gradient => 1,
        EvaluationKind::Residual => 2,
        EvaluationKind::Jacobian => 3,
        EvaluationKind::Hessian => 4,
        EvaluationKind::HessianProduct => 5,
        EvaluationKind::TotalWork => 6,
    }
}

/// Evidence that the next evaluation reservation could not fit.
#[derive(Clone, Debug, PartialEq)]
pub struct BudgetExhausted {
    /// Counts before the declined reservation. Declined work is not charged.
    pub counts: EvalCounts,
    /// The complete atomic request that was declined.
    pub requested: EvalCounts,
    /// Limits in the first enclosing scope that prevented the request.
    pub limits: Vec<ExecutionLimit>,
    pub(crate) scope: u64,
}

impl fmt::Display for BudgetExhausted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the next evaluation does not fit the hard budget")
    }
}
impl std::error::Error for BudgetExhausted {}

/// Evaluation control flow, separate from the original callback error.
#[derive(Debug, PartialEq)]
pub enum EvaluationError<E> {
    /// No callback was invoked for this declined request.
    BudgetExhausted(Box<BudgetExhausted>),
    /// The original callback error, preserved by ownership.
    Evaluation(E),
}

impl<E> From<Box<BudgetExhausted>> for EvaluationError<E> {
    fn from(error: Box<BudgetExhausted>) -> Self {
        Self::BudgetExhausted(error)
    }
}
impl<E> From<BudgetExhausted> for EvaluationError<E> {
    fn from(error: BudgetExhausted) -> Self {
        Self::BudgetExhausted(Box::new(error))
    }
}
impl<E> From<crate::ConstraintSolveError<E>>
    for EvaluationError<crate::ConstraintSolveError<E>>
{
    fn from(error: crate::ConstraintSolveError<E>) -> Self {
        Self::Evaluation(error)
    }
}
impl<E> From<EvaluationError<E>>
    for EvaluationError<crate::ConstraintSolveError<E>>
{
    fn from(error: EvaluationError<E>) -> Self {
        error.map_evaluation(crate::ConstraintSolveError::Evaluation)
    }
}
impl<E> EvaluationError<E> {
    /// Transform the callback error without changing budget control flow.
    pub fn map_evaluation<T>(
        self,
        map: impl FnOnce(E) -> T,
    ) -> EvaluationError<T> {
        match self {
            Self::BudgetExhausted(error) => {
                EvaluationError::BudgetExhausted(error)
            }
            Self::Evaluation(error) => EvaluationError::Evaluation(map(error)),
        }
    }
}
impl<E: fmt::Display> fmt::Display for EvaluationError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BudgetExhausted(error) => error.fmt(f),
            Self::Evaluation(error) => error.fmt(f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for EvaluationError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match self {
            Self::BudgetExhausted(error) => error.as_ref(),
            Self::Evaluation(error) => error,
        })
    }
}

#[derive(Debug)]
struct Ledger {
    budget: EvaluationBudget,
    counts: EvalCounts,
    denied: Option<BudgetExhausted>,
    scopes: Vec<BudgetScope>,
    next_scope: u64,
}

/// One synchronized evaluation ledger shared by nested and parallel work.
///
/// Problem adapters override the `*_with_context` trait methods to forward
/// this context to their probes. User callbacks need no changes. Reservations
/// charge submitted work before evaluation, so a failing callback remains
/// counted. A declined reservation changes no counters.
#[derive(Clone, Debug)]
pub struct EvaluationContext {
    ledger: Arc<Mutex<Ledger>>,
    credit: Option<Arc<Mutex<EvalCounts>>>,
}

impl EvaluationContext {
    /// Create a fresh ledger for one run.
    pub fn new(budget: EvaluationBudget) -> Self {
        Self {
            ledger: Arc::new(Mutex::new(Ledger {
                budget,
                counts: EvalCounts::default(),
                denied: None,
                scopes: Vec::new(),
                next_scope: 1,
            })),
            credit: None,
        }
    }

    /// Current submitted work, including adapter probes and inner solves.
    pub fn counts(&self) -> EvalCounts {
        self.ledger
            .lock()
            .expect("evaluation ledger poisoned")
            .counts
    }

    /// Reserve the complete request against every applicable limit atomically.
    /// Once a request is declined, later reservations remain declined.
    pub fn reserve(
        &self,
        requested: EvalCounts,
    ) -> Result<(), Box<BudgetExhausted>> {
        self.reserve_impl(requested, true)
    }

    /// Check whether a request fits without charging it.
    ///
    /// A declined request stops this scope just like a reservation. This lets
    /// fixed-work solvers stop before changing their resumable model.
    pub fn check(
        &self,
        requested: EvalCounts,
    ) -> Result<(), Box<BudgetExhausted>> {
        self.reserve_impl(requested, false)
    }

    fn reserve_impl(
        &self,
        requested: EvalCounts,
        charge: bool,
    ) -> Result<(), Box<BudgetExhausted>> {
        let mut ledger =
            self.ledger.lock().expect("evaluation ledger poisoned");
        if let Some(denied) = &ledger.denied {
            return Err(Box::new(denied.clone()));
        }
        let mut credit = self
            .credit
            .as_ref()
            .map(|credit| credit.lock().expect("evaluation credit poisoned"));
        let mut unreserved = requested;
        if let Some(credit) = credit.as_mut() {
            macro_rules! consume {
                ($($field:ident),*) => {$(
                    let prepaid = credit.$field.min(unreserved.$field);
                    unreserved.$field -= prepaid;
                    if charge { credit.$field -= prepaid; }
                )*};
            }
            consume!(
                cost_evals,
                gradient_evals,
                residual_evals,
                jacobian_evals,
                hessian_evals,
                hessian_product_evals
            );
        }
        let mut scope = 0;
        let mut limits = ledger.budget.denied(
            &ledger.counts,
            &unreserved,
            &EvalCounts::default(),
            false,
        );
        for frame in &ledger.scopes {
            if !limits.is_empty() {
                break;
            }
            limits = frame.budget.denied(
                &ledger.counts,
                &unreserved,
                &frame.baseline,
                false,
            );
            scope = frame.id;
        }
        if limits.is_empty() {
            let total = checked_total(&unreserved).and_then(|needed| {
                ledger.counts.total_work().checked_add(needed)
            });
            if total.is_none() {
                scope = 0;
                limits.push(ExecutionLimit::Evaluations {
                    kind: EvaluationKind::TotalWork,
                    observed: ledger.counts.total_work(),
                    limit: u64::MAX,
                });
            }
        }
        if !limits.is_empty() {
            let denied = BudgetExhausted {
                counts: ledger.counts,
                requested,
                limits,
                scope,
            };
            ledger.denied = Some(denied.clone());
            return Err(Box::new(denied));
        }
        if charge {
            ledger.counts.add(&unreserved);
        }
        Ok(())
    }

    /// Reserve one evaluation in each supplied category.
    ///
    /// # Panics
    /// Panics for `TotalWork`, which aggregates concrete categories.
    pub fn reserve_kinds(
        &self,
        kinds: &[EvaluationKind],
    ) -> Result<(), Box<BudgetExhausted>> {
        let mut counts = EvalCounts::default();
        for &kind in kinds {
            match kind {
                EvaluationKind::Cost => counts.cost_evals += 1,
                EvaluationKind::Gradient => counts.gradient_evals += 1,
                EvaluationKind::Residual => counts.residual_evals += 1,
                EvaluationKind::Jacobian => counts.jacobian_evals += 1,
                EvaluationKind::Hessian => counts.hessian_evals += 1,
                EvaluationKind::HessianProduct => {
                    counts.hessian_product_evals += 1
                }
                EvaluationKind::TotalWork => {
                    panic!("reserve concrete evaluation categories")
                }
            }
        }
        self.reserve(counts)
    }

    /// Wrap a borrowed problem so its ordinary callbacks use this context.
    pub fn wrap<'a, P>(&'a self, problem: &'a P) -> ContextProblem<'a, P> {
        ContextProblem {
            problem,
            context: self,
        }
    }

    pub(crate) fn prepaid(&self, credit: EvalCounts) -> Self {
        Self {
            ledger: self.ledger.clone(),
            credit: Some(Arc::new(Mutex::new(credit))),
        }
    }

    pub(crate) fn shares_ledger(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.ledger, &other.ledger)
    }

    pub(crate) fn denied(&self) -> Option<BudgetExhausted> {
        self.ledger
            .lock()
            .expect("evaluation ledger poisoned")
            .denied
            .clone()
    }

    pub(crate) fn expired(&self) -> Option<BudgetExhausted> {
        let ledger = self.ledger.lock().expect("evaluation ledger poisoned");
        std::iter::once((0, &ledger.budget, EvalCounts::default()))
            .chain(
                ledger
                    .scopes
                    .iter()
                    .map(|frame| (frame.id, &frame.budget, frame.baseline)),
            )
            .find_map(|(scope, budget, baseline)| {
                let limits = budget.denied(
                    &ledger.counts,
                    &EvalCounts::default(),
                    &baseline,
                    true,
                );
                (!limits.is_empty()).then_some(BudgetExhausted {
                    counts: ledger.counts,
                    requested: EvalCounts::default(),
                    limits,
                    scope,
                })
            })
    }

    pub(crate) fn enter(
        &self,
        budget: EvaluationBudget,
        baseline: EvalCounts,
    ) -> EvaluationScope {
        let mut ledger =
            self.ledger.lock().expect("evaluation ledger poisoned");
        let id = ledger.next_scope;
        ledger.next_scope += 1;
        ledger.scopes.push(BudgetScope {
            id,
            budget,
            baseline,
        });
        EvaluationScope {
            context: self.clone(),
            id,
        }
    }

    pub(crate) fn restore_counts(&self, counts: EvalCounts) {
        let mut ledger =
            self.ledger.lock().expect("evaluation ledger poisoned");
        assert!(
            ledger.scopes.is_empty(),
            "cannot restore counts during an active run"
        );
        assert!(
            checked_total(&counts).is_some(),
            "restored evaluation counts overflow"
        );
        ledger.counts = counts;
        ledger.denied = None;
    }
}

pub(crate) fn constraint_error<E>(
    error: crate::ConstraintSolveError<E>,
) -> crate::ConstraintSolveError<EvaluationError<E>> {
    match error {
        crate::ConstraintSolveError::InvalidConstraints(error) => {
            crate::ConstraintSolveError::InvalidConstraints(error)
        }
        crate::ConstraintSolveError::Evaluation(error) => {
            crate::ConstraintSolveError::Evaluation(
                EvaluationError::Evaluation(error),
            )
        }
    }
}

fn checked_total(counts: &EvalCounts) -> Option<u64> {
    [
        counts.cost_evals,
        counts.gradient_evals,
        counts.residual_evals,
        counts.jacobian_evals,
        counts.hessian_evals,
        counts.hessian_product_evals,
    ]
    .into_iter()
    .try_fold(0u64, u64::checked_add)
}

#[derive(Debug)]
struct BudgetScope {
    id: u64,
    budget: EvaluationBudget,
    baseline: EvalCounts,
}

pub(crate) struct EvaluationScope {
    context: EvaluationContext,
    id: u64,
}
impl EvaluationScope {
    pub(crate) fn owns(&self, error: &BudgetExhausted) -> bool {
        error.scope == self.id
    }
}
impl Drop for EvaluationScope {
    fn drop(&mut self) {
        let mut ledger = self
            .context
            .ledger
            .lock()
            .expect("evaluation ledger poisoned");
        ledger.scopes.retain(|frame| frame.id != self.id);
        if ledger
            .denied
            .as_ref()
            .is_some_and(|error| error.scope == self.id)
        {
            ledger.denied = None;
        }
    }
}
impl EvaluationBudget {
    pub(crate) fn denied(
        &self,
        counts: &EvalCounts,
        requested: &EvalCounts,
        baseline: &EvalCounts,
        reached: bool,
    ) -> Vec<ExecutionLimit> {
        let counts = counts.delta_since(baseline);
        let blocked = |kind: EvaluationKind, limit: u64| {
            let observed = kind.count(&counts);
            let needed = if kind == EvaluationKind::TotalWork {
                checked_total(requested)
            } else {
                Some(kind.count(requested))
            };
            if reached {
                observed >= limit
            } else {
                needed.is_none_or(|needed| {
                    needed > limit.saturating_sub(observed)
                })
            }
        };
        let mut limits = Vec::new();
        if let Some(limit) = self
            .cost
            .filter(|&limit| blocked(EvaluationKind::Cost, limit))
        {
            limits.push(ExecutionLimit::CostEvaluations {
                observed: counts.cost_evals,
                limit,
            });
        }
        if let Some(limit) = self
            .gradient
            .filter(|&limit| blocked(EvaluationKind::Gradient, limit))
        {
            limits.push(ExecutionLimit::GradientEvaluations {
                observed: counts.gradient_evals,
                limit,
            });
        }
        for kind in EvaluationKind::ALL {
            if let Some(limit) =
                self.limits[index(kind)].filter(|&limit| blocked(kind, limit))
            {
                limits.push(ExecutionLimit::Evaluations {
                    kind,
                    observed: kind.count(&counts),
                    limit,
                });
            }
        }
        limits
    }
}

pub(crate) fn flatten_constraint_error<E>(
    error: crate::ConstraintSolveError<EvaluationError<E>>,
) -> EvaluationError<crate::ConstraintSolveError<E>> {
    match error {
        crate::ConstraintSolveError::InvalidConstraints(error) => {
            EvaluationError::Evaluation(
                crate::ConstraintSolveError::InvalidConstraints(error),
            )
        }
        crate::ConstraintSolveError::Evaluation(error) => {
            error.map_evaluation(crate::ConstraintSolveError::Evaluation)
        }
    }
}
