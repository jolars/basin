//! Counted evaluation and model construction shared by the NLLS drivers.

use super::{LossFunction, RobustLeastSquares};
use crate::core::constraint::RelaxedProblem;
use crate::core::constraint::evaluation::BoxProblem;
use crate::core::math::{
    MatTransposeVec, Scalar, ScaleRowsInPlace, VectorIndex, VectorLen,
};
use crate::core::problem::{Jacobian, Problem, Residual};

pub(crate) type NllsStep<F, E> = Result<crate::SolverStep<F>, E>;

// Separate implementations keep additional robust math capabilities off the
// existing solver bounds, including downstream minimal Jacobian types.
pub(crate) trait Evaluation<V, M, F: Scalar, R = V> {
    type Error;
    const ROBUST: bool;
    fn residual(&mut self, x: &V) -> Result<R, Self::Error>;
    fn residual_into(&mut self, x: &V, out: &mut R) -> Result<(), Self::Error>;
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
    fn residual_and_jacobian(&mut self, x: &V) -> Result<(R, M), Self::Error>;
    fn residual_and_jacobian_into(
        &mut self,
        x: &V,
        r: &mut R,
        j: &mut M,
    ) -> Result<(), Self::Error>;
    fn residual_and_jacobian_reusing(
        &mut self,
        x: &V,
        r: Option<R>,
        j: Option<M>,
    ) -> Result<(R, M), Self::Error> {
        if let (Some(mut r), Some(mut j)) = (r, j) {
            self.residual_and_jacobian_into(x, &mut r, &mut j)?;
            Ok((r, j))
        } else {
            // A custom matrix need not expose construction. One owned fused
            // evaluation seeds its storage without adding a backend bound.
            self.residual_and_jacobian(x)
        }
    }

    fn cost(&self, r: &R, quadratic: impl FnOnce(&R) -> F) -> F;
    // The caller supplies residuals with matching lengths.
    fn cost_and_reduction(
        &self,
        current: &R,
        trial: &R,
        quadratic: impl FnOnce(&R, &R) -> (F, F),
    ) -> (F, F);
    // The inner None retains raw residuals without cloning on quadratic paths.
    fn model(&self, r: &R, j: M) -> Option<(Option<R>, M)>;
    fn gradient_model(&self, r: &R, j: M) -> Option<(V, M)>
    where
        M: MatTransposeVec<V, R>,
    {
        Some((j.mat_transpose_vec(r), j))
    }
    fn valid_vector(&self, vector: &V) -> bool;
}

pub(crate) trait BoundedEvaluation<V, M, F: Scalar, R = V>:
    Evaluation<V, M, F, R>
{
    fn lower(&self) -> &V;
    fn upper(&self) -> &V;
}

impl<P, V, M, F: Scalar, R> Evaluation<V, M, F, R> for Problem<P>
where
    P: Residual<Param = V, Output = R> + Jacobian<Jacobian = M>,
{
    type Error = P::Error;
    const ROBUST: bool = false;
    fn residual(&mut self, x: &V) -> Result<R, Self::Error> {
        Problem::residual(self, x)
    }
    fn residual_into(&mut self, x: &V, out: &mut R) -> Result<(), Self::Error> {
        Problem::residual_into(self, x, out)
    }
    fn jacobian(&mut self, x: &V) -> Result<M, Self::Error> {
        Problem::jacobian(self, x)
    }
    fn jacobian_into(&mut self, x: &V, out: &mut M) -> Result<(), Self::Error> {
        Problem::jacobian_into(self, x, out)
    }
    fn residual_and_jacobian(&mut self, x: &V) -> Result<(R, M), Self::Error> {
        Problem::residual_and_jacobian(self, x)
    }
    fn residual_and_jacobian_into(
        &mut self,
        x: &V,
        r: &mut R,
        j: &mut M,
    ) -> Result<(), Self::Error> {
        Problem::residual_and_jacobian_into(self, x, r, j)
    }
    fn cost(&self, r: &R, quadratic: impl FnOnce(&R) -> F) -> F {
        quadratic(r)
    }
    fn cost_and_reduction(
        &self,
        current: &R,
        trial: &R,
        quadratic: impl FnOnce(&R, &R) -> (F, F),
    ) -> (F, F) {
        quadratic(current, trial)
    }
    fn model(&self, _r: &R, j: M) -> Option<(Option<R>, M)> {
        Some((None, j))
    }
    fn valid_vector(&self, _: &V) -> bool {
        true
    }
}

// Relaxing a box must retain the loss, scale, and raw callback accounting.
macro_rules! robust_evaluation {
    ($problem:ty, $($field:ident).*) => {
        impl<P, L, V, M, F, W, R, const I: bool> Evaluation<V, M, F, R> for Problem<$problem>
        where
            F: Scalar,
            P: Residual<Param = V, Output = R> + Jacobian<Jacobian = M>,
            L: LossFunction<F>,
            V: VectorLen + VectorIndex<F>,
            R: Clone + VectorLen + VectorIndex<F>,
            M: ScaleRowsInPlace<F>,
        {
            type Error = P::Error;
            const ROBUST: bool = true;
            fn residual(&mut self, x: &V) -> Result<R, Self::Error> {
                self.counts_mut().residual_evals += 1;
                self.inner()$(.$field)*.problem.residual(x)
            }
            fn residual_into(&mut self, x: &V, out: &mut R) -> Result<(), Self::Error> {
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
            ) -> Result<(R, M), Self::Error> {
                self.counts_mut().residual_evals += 1;
                self.counts_mut().jacobian_evals += 1;
                self.inner()$(.$field)*.problem.residual_and_jacobian(x)
            }
            fn residual_and_jacobian_into(&mut self, x: &V, r: &mut R, j: &mut M) -> Result<(), Self::Error> {
                self.counts_mut().residual_evals += 1;
                self.counts_mut().jacobian_evals += 1;
                self.inner()$(.$field)*.problem.residual_and_jacobian_into(x, r, j)
            }
            fn cost(&self, r: &R, _: impl FnOnce(&R) -> F) -> F {
                self.inner()$(.$field)*.residual_cost(r)
            }
            fn cost_and_reduction(
                &self,
                current: &R,
                trial: &R,
                _: impl FnOnce(&R, &R) -> (F, F),
            ) -> (F, F) {
                self.inner()$(.$field)*.residual_cost_and_reduction(current, trial)
            }
            fn model(&self, r: &R, mut j: M) -> Option<(Option<R>, M)> {
                let (residual, factors) =
                    self.inner()$(.$field)*.model_residual(r)?;
                j.scale_rows_in_place(&factors);
                Some((Some(residual), j))
            }
            fn gradient_model(&self, r: &R, mut j: M) -> Option<(V, M)>
            where
                M: MatTransposeVec<V, R>,
            {
                let mut derivatives = r.clone();
                let mut factors = Vec::with_capacity(r.vec_len());
                let objective = &self.inner()$(.$field)*;
                for i in 0..r.vec_len() {
                    let term = objective.term(r.get_scalar(i))?;
                    derivatives.set_scalar(i, term.derivative);
                    factors.push(term.curvature.max(F::epsilon()).sqrt());
                }
                let gradient = j.mat_transpose_vec(&derivatives);
                j.scale_rows_in_place(&factors);
                Some((gradient, j))
            }
            fn valid_vector(&self, v: &V) -> bool {
                (0..v.vec_len()).all(|i| v.get_scalar(i).is_finite())
            }
        }
    };
}
robust_evaluation!(RobustLeastSquares<P, L, F, W, I>,);
robust_evaluation!(
    RelaxedProblem<'_, RobustLeastSquares<P, L, F, W, I>>,
    problem
);

impl<P, V, M, F: Scalar, R> BoundedEvaluation<V, M, F, R> for Problem<P>
where
    P: Residual<Param = V, Output = R>
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

impl<P, L, V, M, F, W, R, const I: bool> BoundedEvaluation<V, M, F, R>
    for Problem<RobustLeastSquares<P, L, F, W, I>>
where
    F: Scalar,
    P: Residual<Param = V, Output = R>
        + Jacobian<Jacobian = M>
        + BoxProblem<Param = V>,
    L: LossFunction<F>,
    V: VectorLen + VectorIndex<F>,
    R: Clone + VectorLen + VectorIndex<F>,
    M: ScaleRowsInPlace<F>,
{
    fn lower(&self) -> &V {
        self.inner().problem.lower()
    }
    fn upper(&self) -> &V {
        self.inner().problem.upper()
    }
}
