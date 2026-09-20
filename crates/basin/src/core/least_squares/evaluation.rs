//! Counted evaluation and model construction shared by the NLLS drivers.

use super::{LossFunction, RobustLeastSquares};
use crate::BoxConstraints;
use crate::core::math::{Scalar, ScaleRowsInPlace, VectorIndex, VectorLen};
use crate::core::problem::{Jacobian, Problem, Residual};
use crate::core::state::NllsState;
use crate::core::termination::TerminationReason;

pub(crate) type NllsStep<V, F, E> =
    Result<(NllsState<V, F>, Option<TerminationReason>), E>;

// Separate implementations keep additional robust math capabilities off the
// existing solver bounds, including downstream minimal Jacobian types.
pub(crate) trait Evaluation<V, M, F: Scalar> {
    type Error;
    const ROBUST: bool;
    fn residual(&mut self, x: &V) -> Result<V, Self::Error>;
    fn jacobian(&mut self, x: &V) -> Result<M, Self::Error>;
    fn residual_and_jacobian(&mut self, x: &V) -> Result<(V, M), Self::Error>;
    fn cost(&self, r: &V, quadratic: impl FnOnce(&V) -> F) -> F;
    // The inner None retains raw residuals without cloning on quadratic paths.
    fn model(&self, r: &V, j: M) -> Option<(Option<V>, M)>;
    fn valid_vector(&self, vector: &V) -> bool;
}

pub(crate) trait BoundedEvaluation<V, M, F: Scalar>:
    Evaluation<V, M, F>
{
    fn lower(&self) -> &V;
    fn upper(&self) -> &V;
}

impl<P, V: Clone, M, F: Scalar> Evaluation<V, M, F> for Problem<P>
where
    P: Residual<Param = V, Output = V> + Jacobian<Jacobian = M>,
{
    type Error = P::Error;
    const ROBUST: bool = false;
    fn residual(&mut self, x: &V) -> Result<V, Self::Error> {
        Problem::residual(self, x)
    }
    fn jacobian(&mut self, x: &V) -> Result<M, Self::Error> {
        Problem::jacobian(self, x)
    }
    fn residual_and_jacobian(&mut self, x: &V) -> Result<(V, M), Self::Error> {
        Problem::residual_and_jacobian(self, x)
    }
    fn cost(&self, r: &V, quadratic: impl FnOnce(&V) -> F) -> F {
        quadratic(r)
    }
    fn model(&self, _r: &V, j: M) -> Option<(Option<V>, M)> {
        Some((None, j))
    }
    fn valid_vector(&self, _: &V) -> bool {
        true
    }
}

impl<P, L, V, M, F> Evaluation<V, M, F> for Problem<RobustLeastSquares<P, L, F>>
where
    F: Scalar,
    P: Residual<Param = V, Output = V> + Jacobian<Jacobian = M>,
    L: LossFunction<F>,
    V: Clone + VectorLen + VectorIndex<F>,
    M: ScaleRowsInPlace<F>,
{
    type Error = P::Error;
    const ROBUST: bool = true;
    fn residual(&mut self, x: &V) -> Result<V, Self::Error> {
        self.counts_mut().residual_evals += 1;
        self.inner().problem.residual(x)
    }
    fn jacobian(&mut self, x: &V) -> Result<M, Self::Error> {
        self.counts_mut().jacobian_evals += 1;
        self.inner().problem.jacobian(x)
    }
    fn residual_and_jacobian(&mut self, x: &V) -> Result<(V, M), Self::Error> {
        self.counts_mut().residual_evals += 1;
        self.counts_mut().jacobian_evals += 1;
        self.inner().problem.residual_and_jacobian(x)
    }
    fn cost(&self, r: &V, _: impl FnOnce(&V) -> F) -> F {
        self.inner().residual_cost(r)
    }
    fn model(&self, r: &V, mut j: M) -> Option<(Option<V>, M)> {
        let (residual, factors) = self.inner().model_residual(r)?;
        j.scale_rows_in_place(&factors);
        Some((Some(residual), j))
    }
    fn valid_vector(&self, v: &V) -> bool {
        (0..v.vec_len()).all(|i| v.get_scalar(i).is_finite())
    }
}

impl<P, V: Clone, M, F: Scalar> BoundedEvaluation<V, M, F> for Problem<P>
where
    P: Residual<Param = V, Output = V>
        + Jacobian<Jacobian = M>
        + BoxConstraints<Param = V>,
{
    fn lower(&self) -> &V {
        self.inner().lower()
    }
    fn upper(&self) -> &V {
        self.inner().upper()
    }
}

impl<P, L, V, M, F> BoundedEvaluation<V, M, F>
    for Problem<RobustLeastSquares<P, L, F>>
where
    F: Scalar,
    P: Residual<Param = V, Output = V>
        + Jacobian<Jacobian = M>
        + BoxConstraints<Param = V>,
    L: LossFunction<F>,
    V: Clone + VectorLen + VectorIndex<F>,
    M: ScaleRowsInPlace<F>,
{
    fn lower(&self) -> &V {
        self.inner().problem.lower()
    }
    fn upper(&self) -> &V {
        self.inner().problem.upper()
    }
}
