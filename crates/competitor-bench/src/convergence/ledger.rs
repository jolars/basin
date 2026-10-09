//! Physical oracle calls share one cap across adapters and nested solves.

use std::sync::{Arc, Mutex};

/// Leaf operations, rather than derivative requests made to an adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OracleKind {
    Cost,
    Gradient,
    CostGradient,
    Residual,
    Jacobian,
    ResidualJacobian,
    Hessian,
    HessianProduct,
    Constraints,
    ConstraintJacobian,
}

/// A budget denial happens before evaluation and charges no physical call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OracleError {
    Budget { cap: u64 },
    Callback(&'static str),
}

impl std::fmt::Display for OracleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for OracleError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallOutcome {
    Pending,
    Completed,
    Failed(OracleError),
}

#[derive(Clone, Debug)]
pub struct LeafCall {
    pub work: u64,
    pub scope: &'static str,
    pub kind: OracleKind,
    pub point: Vec<f64>,
    /// A sampled objective is not a published recommendation.
    pub sampled_cost: Option<f64>,
    pub outcome: CallOutcome,
}

#[derive(Clone, Debug)]
pub struct LedgerSnapshot {
    pub cap: u64,
    pub calls: Vec<LeafCall>,
    pub denied: u64,
    pub cache_hits: u64,
}

impl LedgerSnapshot {
    pub fn work(&self) -> u64 {
        self.calls.len() as u64
    }

    pub fn count(&self, kind: OracleKind) -> u64 {
        self.calls.iter().filter(|c| c.kind == kind).count() as u64
    }
}

/// Clones share the cap, including across finite differences and inner solves.
#[derive(Clone, Debug)]
pub struct WorkLedger {
    shared: Arc<Mutex<LedgerSnapshot>>,
    scope: &'static str,
}

impl WorkLedger {
    pub fn new(cap: u64) -> Self {
        Self {
            shared: Arc::new(Mutex::new(LedgerSnapshot {
                cap,
                calls: Vec::new(),
                denied: 0,
                cache_hits: 0,
            })),
            scope: "outer",
        }
    }

    pub fn in_scope(&self, scope: &'static str) -> Self {
        Self {
            shared: self.shared.clone(),
            scope,
        }
    }

    pub fn snapshot(&self) -> LedgerSnapshot {
        self.shared.lock().unwrap().clone()
    }

    pub fn work(&self) -> u64 {
        self.shared.lock().unwrap().work()
    }

    pub fn cache_hit(&self) {
        self.shared.lock().unwrap().cache_hits += 1;
    }

    /// Reserve before calling user code; never hold the lock during evaluation.
    /// Failed callbacks consume their reservation, while a denied call does not.
    pub fn evaluate<T>(
        &self,
        kind: OracleKind,
        point: &[f64],
        evaluate: impl FnOnce() -> Result<(T, Option<f64>), OracleError>,
    ) -> Result<T, OracleError> {
        let index = {
            let mut ledger = self.shared.lock().unwrap();
            if ledger.work() == ledger.cap {
                ledger.denied += 1;
                return Err(OracleError::Budget { cap: ledger.cap });
            }
            let index = ledger.calls.len();
            ledger.calls.push(LeafCall {
                work: index as u64 + 1,
                scope: self.scope,
                kind,
                point: point.to_vec(),
                sampled_cost: None,
                outcome: CallOutcome::Pending,
            });
            index
        };
        let result = evaluate();
        let mut ledger = self.shared.lock().unwrap();
        let call = &mut ledger.calls[index];
        match result {
            Ok((value, cost)) => {
                call.sampled_cost = cost;
                call.outcome = CallOutcome::Completed;
                Ok(value)
            }
            Err(error) => {
                call.outcome = CallOutcome::Failed(error.clone());
                Err(error)
            }
        }
    }
}
