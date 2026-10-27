//! Counted evaluation and model construction shared by the NLLS drivers.

use super::{LossFunction, RobustLeastSquares};
use crate::core::constraint::RelaxedProblem;
use crate::core::constraint::evaluation::BoxProblem;
use crate::core::math::{Scalar, ScaleRowsInPlace, VectorIndex, VectorLen};
use crate::core::problem::{Jacobian, Problem, Residual};
use crate::core::state::PointState;

pub(crate) type NllsStep<V, F, E> =
    Result<crate::SolverStep<PointState<V, F>>, E>;

// Separate implementations keep additional robust math capabilities off the
// existing solver bounds, including downstream minimal Jacobian types.
pub(crate) trait Evaluation<V, M, F: Scalar> {
    type Error;
    const ROBUST: bool;
    fn residual(&mut self, x: &V) -> Result<V, Self::Error>;
    fn residual_into(&mut self, x: &V, out: &mut V) -> Result<(), Self::Error>;
    fn jacobian(&mut self, x: &V) -> Result<M, Self::Error>;
    fn jacobian_into(&mut self, x: &V, out: &mut M) -> Result<(), Self::Error>;
    fn jacobian_reusing(
        &mut self,
        x: &V,
        buffer: Option<M>,
    ) -> Result<M, Self::Error> {
        if let Some(mut out) = buffer {
            self.jacobian_into(x, &mut out)?;
            Ok(out)
        } else {
            self.jacobian(x)
        }
    }
    fn residual_and_jacobian(&mut self, x: &V) -> Result<(V, M), Self::Error>;
    fn residual_and_jacobian_into(
        &mut self,
        x: &V,
        r: &mut V,
        j: &mut M,
    ) -> Result<(), Self::Error>;
    fn residual_and_jacobian_reusing(
        &mut self,
        x: &V,
        mut r: V,
        j: Option<M>,
    ) -> Result<(V, M), Self::Error> {
        if let Some(mut j) = j {
            self.residual_and_jacobian_into(x, &mut r, &mut j)?;
            Ok((r, j))
        } else {
            // A custom matrix need not expose construction. One owned fused
            // evaluation seeds its storage without adding a backend bound.
            self.residual_and_jacobian(x)
        }
    }

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
        let mut out = x.clone();
        Problem::residual_into(self, x, &mut out)?;
        Ok(out)
    }
    fn residual_into(&mut self, x: &V, out: &mut V) -> Result<(), Self::Error> {
        Problem::residual_into(self, x, out)
    }
    fn jacobian(&mut self, x: &V) -> Result<M, Self::Error> {
        Problem::jacobian(self, x)
    }
    fn jacobian_into(&mut self, x: &V, out: &mut M) -> Result<(), Self::Error> {
        Problem::jacobian_into(self, x, out)
    }
    fn residual_and_jacobian(&mut self, x: &V) -> Result<(V, M), Self::Error> {
        Problem::residual_and_jacobian(self, x)
    }
    fn residual_and_jacobian_into(
        &mut self,
        x: &V,
        r: &mut V,
        j: &mut M,
    ) -> Result<(), Self::Error> {
        Problem::residual_and_jacobian_into(self, x, r, j)
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

// Relaxing a box must retain the loss, scale, and raw callback accounting.
macro_rules! robust_evaluation {
    ($problem:ty, $($field:ident).*) => {
        impl<P, L, V, M, F, W> Evaluation<V, M, F> for Problem<$problem>
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
                let mut out = x.clone();
                self.inner()$(.$field)*.problem.residual_into(x, &mut out)?;
                Ok(out)
            }
            fn residual_into(&mut self, x: &V, out: &mut V) -> Result<(), Self::Error> {
                self.counts_mut().residual_evals += 1;
                self.inner()$(.$field)*.problem.residual_into(x, out)
            }
            fn jacobian(&mut self, x: &V) -> Result<M, Self::Error> {
                self.counts_mut().jacobian_evals += 1;
                self.inner()$(.$field)*.problem.jacobian(x)
            }
            fn jacobian_into(&mut self, x: &V, out: &mut M) -> Result<(), Self::Error> {
                self.counts_mut().jacobian_evals += 1;
                self.inner()$(.$field)*.problem.jacobian_into(x, out)
            }
            fn residual_and_jacobian(
                &mut self,
                x: &V,
            ) -> Result<(V, M), Self::Error> {
                self.counts_mut().residual_evals += 1;
                self.counts_mut().jacobian_evals += 1;
                self.inner()$(.$field)*.problem.residual_and_jacobian(x)
            }
            fn residual_and_jacobian_into(&mut self, x: &V, r: &mut V, j: &mut M) -> Result<(), Self::Error> {
                self.counts_mut().residual_evals += 1;
                self.counts_mut().jacobian_evals += 1;
                self.inner()$(.$field)*.problem.residual_and_jacobian_into(x, r, j)
            }
            fn cost(&self, r: &V, _: impl FnOnce(&V) -> F) -> F {
                self.inner()$(.$field)*.residual_cost(r)
            }
            fn model(&self, r: &V, mut j: M) -> Option<(Option<V>, M)> {
                let (residual, factors) =
                    self.inner()$(.$field)*.model_residual(r)?;
                j.scale_rows_in_place(&factors);
                Some((Some(residual), j))
            }
            fn valid_vector(&self, v: &V) -> bool {
                (0..v.vec_len()).all(|i| v.get_scalar(i).is_finite())
            }
        }
    };
}
robust_evaluation!(RobustLeastSquares<P, L, F, W>,);
robust_evaluation!(RelaxedProblem<'_, RobustLeastSquares<P, L, F, W>>, problem);

impl<P, V: Clone, M, F: Scalar> BoundedEvaluation<V, M, F> for Problem<P>
where
    P: Residual<Param = V, Output = V>
        + Jacobian<Jacobian = M>
        + BoxProblem<Param = V>,
{
    fn lower(&self) -> &V {
        self.inner().lower()
    }
    fn upper(&self) -> &V {
        self.inner().upper()
    }
}

impl<P, L, V, M, F, W> BoundedEvaluation<V, M, F>
    for Problem<RobustLeastSquares<P, L, F, W>>
where
    F: Scalar,
    P: Residual<Param = V, Output = V>
        + Jacobian<Jacobian = M>
        + BoxProblem<Param = V>,
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
