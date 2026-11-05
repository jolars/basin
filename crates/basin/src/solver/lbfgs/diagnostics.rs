//! Opt-in native bounded L-BFGS model construction and subproblem outcomes.

use super::{AsFloatSlice, workspace::LbfgsbWork};
use crate::{EvalCounts, Scalar};

/// Native model phase, separate from outer convergence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelPhase {
    /// Generalized Cauchy point construction.
    Cauchy,
    /// Compact free-subspace factorization (`formk`).
    FactorK,
    /// Reduced gradient construction (`cmprlb`).
    ReducedGradient,
    /// Free-subspace solve (`subsm`).
    Subspace,
    /// Compact history factorization (`formt`).
    FactorT,
    /// Usability of the selected line-search point, not Wolfe convergence.
    SelectedPoint,
}

/// Native arithmetic failure within a model phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelFailure {
    /// The compact middle-matrix triangular solve was singular.
    SingularMiddleMatrix,
    /// The first K block failed Cholesky factorization.
    NotPositiveDefiniteFirst,
    /// The second K block failed Cholesky factorization.
    NotPositiveDefiniteSecond,
    /// The T matrix failed Cholesky factorization.
    NotPositiveDefiniteT,
    /// The free-subspace K solve was singular.
    SingularK,
    /// The selected line-search point was unusable.
    UnusableSelectedPoint,
}

/// Native model snapshot and phase completion, before any restart discards it.
///
/// Arrays retain native precision and row-major capacity strides. Only leading
/// `columns` blocks are live (`2*columns` for K). A failed phase may leave partial
/// output; it does not promise a usable model or a returned recommendation.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ModelObservation<F: Scalar = f64> {
    /// Monotone sequence within one fresh initialization.
    pub sequence: u64,
    /// Phase that produced the snapshot.
    pub phase: ModelPhase,
    /// Native success or failure.
    pub succeeded: bool,
    /// Specific native arithmetic failure, absent on success.
    pub failure: Option<ModelFailure>,
    /// Retained outer point used to construct this model.
    pub base: Vec<F>,
    /// Retained native gradient at the outer point.
    pub gradient: Vec<F>,
    /// Lower coordinate bounds.
    pub lower: Vec<F>,
    /// Upper coordinate bounds.
    pub upper: Vec<F>,
    /// Number of restart attempts still available at this decision.
    pub restart_budget: u8,
    /// Authoritative callback counts at this phase.
    pub counts: EvalCounts,
    /// Capacity stride of compact history matrices.
    pub capacity: usize,
    /// Number of live curvature pairs.
    pub columns: usize,
    /// Native Hessian scaling.
    pub theta: F,
    /// Chronological displacement columns.
    pub s: Vec<Vec<F>>,
    /// Chronological gradient-difference columns.
    pub y: Vec<Vec<F>>,
    /// Native `S'Y` storage.
    pub sy: Vec<F>,
    /// Native `S'S` storage.
    pub ss: Vec<F>,
    /// Compact T factor storage.
    pub factor_t: Vec<F>,
    /// Compact K factor storage.
    pub factor_k: Vec<F>,
    /// Compact Gram cache used by `formk`.
    pub gram_k: Vec<F>,
    /// Native Cauchy or subspace point at this phase.
    pub point: Vec<F>,
    /// Reduced gradient or subspace direction storage at this phase.
    pub reduced: Vec<F>,
    /// Free-coordinate indices in native subspace order.
    pub free: Vec<usize>,
    /// Whether the successful subspace solution encountered a bound.
    pub bound_encountered: Option<bool>,
}

pub(super) struct Trace<F: Scalar> {
    pub enabled: bool,
    pub context: [Vec<F>; 4],
    pub sequence: u64,
    pub observations: Vec<ModelObservation<F>>,
}
impl<F: Scalar> Default for Trace<F> {
    fn default() -> Self {
        Self {
            enabled: false,
            context: core::array::from_fn(|_| Vec::new()),
            sequence: 0,
            observations: Vec::new(),
        }
    }
}
impl<F: Scalar> Trace<F> {
    #[allow(clippy::too_many_arguments)]
    pub fn record<V: AsFloatSlice<F>>(
        &mut self,
        phase: ModelPhase,
        succeeded: bool,
        failure: Option<ModelFailure>,
        restart_budget: u8,
        counts: EvalCounts,
        capacity: usize,
        ws: &[V],
        wy: &[V],
        sy: &[F],
        ss: &[F],
        theta: F,
        work: &LbfgsbWork<F>,
        bound_encountered: Option<bool>,
    ) {
        if !self.enabled {
            return;
        }
        self.sequence += 1;
        self.observations.push(ModelObservation {
            sequence: self.sequence,
            phase,
            succeeded,
            failure,
            base: self.context[0].clone(),
            gradient: self.context[1].clone(),
            lower: self.context[2].clone(),
            upper: self.context[3].clone(),
            restart_budget,
            counts,
            capacity,
            columns: ws.len(),
            theta,
            s: ws.iter().map(|v| v.as_float_slice().to_vec()).collect(),
            y: wy.iter().map(|v| v.as_float_slice().to_vec()).collect(),
            sy: sy.to_vec(),
            ss: ss.to_vec(),
            factor_t: work.wt.clone(),
            factor_k: work.wn.clone(),
            gram_k: work.wn1.clone(),
            point: work.z.clone(),
            reduced: work.r.clone(),
            free: work.index[..work.nfree].to_vec(),
            bound_encountered,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CostFunction, FirstOrderState, Gradient, Lbfgs, Problem, Solver,
    };
    struct Quadratic<F: Scalar> {
        lower: Vec<F>,
        upper: Vec<F>,
    }

    impl<F: Scalar> crate::Constraints for Quadratic<F> {
        type Set = crate::ConstraintSet<crate::BoxBounds<Vec<F>>>;
        fn constraints(
            &self,
        ) -> <Self::Set as crate::ConstraintDescription>::Borrowed<'_> {
            crate::ConstraintSet::new().with_bounds(crate::BoxBounds::new(
                Some(&self.lower),
                Some(&self.upper),
            ))
        }
    }
    impl<F: Scalar> CostFunction for Quadratic<F> {
        type Param = Vec<F>;
        type Output = F;
        type Error = core::convert::Infallible;
        fn cost(&self, x: &Vec<F>) -> Result<F, Self::Error> {
            Ok(x[0] * x[0] + F::from_f64(4.).unwrap() * x[1] * x[1])
        }
    }
    impl<F: Scalar> Gradient for Quadratic<F> {
        type Gradient = Vec<F>;
        fn gradient(&self, x: &Vec<F>) -> Result<Vec<F>, Self::Error> {
            Ok(vec![
                F::from_f64(2.).unwrap() * x[0],
                F::from_f64(8.).unwrap() * x[1],
            ])
        }
    }

    fn failed_subspace<F: Scalar>() {
        let f = |v| F::from_f64(v).unwrap();
        let mut p = Problem::new(Quadratic {
            lower: vec![f(-8.); 2],
            upper: vec![f(8.); 2],
        });
        let mut solver = Lbfgs::new().with_trial_diagnostics(true);
        let s = solver
            .init(&mut p, FirstOrderState::new(vec![f(4.), f(3.)]))
            .unwrap();
        let s = solver.next_iter(&mut p, s).unwrap().state;
        assert!(
            solver
                .model_observations()
                .iter()
                .any(|r| r.phase == ModelPhase::FactorT && r.succeeded)
        );
        let history = solver.history.as_mut().unwrap();
        assert!(!history.ws.is_empty());
        let work = history.work.as_mut().unwrap();
        // Preserve the cache selection while injecting an unusable factor to
        // exercise recovery evidence before the restart clears it.
        work.updatd = false;
        work.wn.fill(F::zero());
        let counts = *p.counts();
        let result = solver.next_iter(&mut p, s).unwrap();
        let failed = solver
            .model_observations()
            .iter()
            .find(|r| r.phase == ModelPhase::Subspace && !r.succeeded)
            .unwrap();
        assert_eq!(failed.counts, counts);
        assert_eq!(failed.columns, 1);
        assert_eq!(failed.restart_budget, 1);
        assert!(failed.factor_k.iter().all(|v| *v == F::zero()));
        assert!(
            solver
                .model_observations()
                .iter()
                .any(|r| r.phase == ModelPhase::Cauchy
                    && r.columns == 0
                    && r.restart_budget == 0)
        );
        assert!(result.termination.is_none());
    }
    #[test]
    fn failed_subspace_f32() {
        failed_subspace::<f32>();
    }
    #[test]
    fn failed_subspace_f64() {
        failed_subspace::<f64>();
    }
}
