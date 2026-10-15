//! Problem views that route ordinary calls through an evaluation context.

use super::{EvaluationContext, EvaluationError};
use crate::core::problem::*;
use crate::{ConstraintDescription, ProblemConstraints};

/// Borrowed problem view used to propagate budgets through adapters.
pub struct ContextProblem<'a, P> {
    pub(crate) problem: &'a P,
    pub(crate) context: &'a EvaluationContext,
}

macro_rules! implement {
    ($wrapper:ty) => {
        impl<P: ProblemConstraints> ProblemConstraints for $wrapper {
            type Constraints = P::Constraints;
            fn constraint_set(
                &self,
            ) -> <P::Constraints as ConstraintDescription>::Borrowed<'_> {
                self.problem.constraint_set()
            }
        }
        impl<P: CostFunction> CostFunction for $wrapper {
            type Param = P::Param;
            type Output = P::Output;
            type Error = EvaluationError<P::Error>;
            fn cost(&self, x: &P::Param) -> Result<P::Output, Self::Error> {
                self.problem.cost_with_context(x, &self.context)
            }
            fn prepare_cost_batch(
                &self,
                points: u64,
            ) -> Result<crate::CostBatchReservation, Self::Error> {
                let requested = self.problem.cost_batch_request(points);
                self.context.reserve(requested)?;
                Ok(crate::CostBatchReservation {
                    context: Some(self.context.prepaid(requested)),
                })
            }
            fn cost_in_batch(
                &self,
                x: &P::Param,
                reservation: &crate::CostBatchReservation,
            ) -> Result<P::Output, Self::Error> {
                let context = reservation
                    .context
                    .as_ref()
                    .filter(|context| self.context.shares_ledger(context))
                    .unwrap_or(&self.context);
                self.problem.cost_with_context(x, context)
            }
        }
        impl<P: Gradient> Gradient for $wrapper {
            type Gradient = P::Gradient;
            fn gradient(
                &self,
                x: &P::Param,
            ) -> Result<P::Gradient, Self::Error> {
                self.problem.gradient_with_context(x, &self.context)
            }
            fn cost_and_gradient(
                &self,
                x: &P::Param,
            ) -> Result<(P::Output, P::Gradient), Self::Error> {
                self.problem
                    .cost_and_gradient_with_context(x, &self.context)
            }
        }
        impl<P: MiniBatchGradient> MiniBatchGradient for $wrapper {
            type Gradient = P::Gradient;
            fn n_samples(&self) -> usize {
                self.problem.n_samples()
            }
            fn batch_gradient(
                &self,
                x: &P::Param,
                batch: &[usize],
            ) -> Result<P::Gradient, Self::Error> {
                self.problem.batch_gradient_with_context(
                    x,
                    batch,
                    &self.context,
                )
            }
        }
        impl<P: Hessian> Hessian for $wrapper {
            type Hessian = P::Hessian;
            fn hessian(&self, x: &P::Param) -> Result<P::Hessian, Self::Error> {
                self.problem.hessian_with_context(x, &self.context)
            }
            fn cost_and_gradient_and_hessian(
                &self,
                x: &P::Param,
            ) -> Result<(P::Output, P::Gradient, P::Hessian), Self::Error> {
                self.problem.cost_and_gradient_and_hessian_with_context(
                    x,
                    &self.context,
                )
            }
        }
        impl<P: HessianProduct> HessianProduct for $wrapper {
            fn hessian_product(
                &self,
                x: &P::Param,
                v: &P::Param,
            ) -> Result<P::Gradient, Self::Error> {
                self.problem
                    .hessian_product_with_context(x, v, &self.context)
            }
        }
        impl<P: Residual> Residual for $wrapper {
            type Param = P::Param;
            type Output = P::Output;
            type Error = EvaluationError<P::Error>;
            fn residual(&self, x: &P::Param) -> Result<P::Output, Self::Error> {
                self.problem.residual_with_context(x, &self.context)
            }
        }
        impl<P: Jacobian> Jacobian for $wrapper {
            type Jacobian = P::Jacobian;
            fn jacobian(
                &self,
                x: &P::Param,
            ) -> Result<P::Jacobian, Self::Error> {
                self.problem.jacobian_with_context(x, &self.context)
            }
            fn residual_and_jacobian(
                &self,
                x: &P::Param,
            ) -> Result<(P::Output, P::Jacobian), Self::Error> {
                self.problem
                    .residual_and_jacobian_with_context(x, &self.context)
            }
        }
        impl<P: crate::SamplingBounds> crate::SamplingBounds for $wrapper {
            fn sampling_bounds(
                &self,
            ) -> Result<(&P::Param, &P::Param), crate::ConstraintError> {
                self.problem.sampling_bounds()
            }
        }
        impl<P: crate::NonlinearEqualityValues> crate::NonlinearEqualityValues
            for $wrapper
        {
            type Param = P::Param;
            type Values = P::Values;
            type Error = EvaluationError<P::Error>;
            fn nonlinear_equalities(
                &self,
                x: &P::Param,
            ) -> Result<P::Values, crate::ConstraintSolveError<Self::Error>>
            {
                self.problem
                    .nonlinear_equalities_with_context(x, &self.context)
            }
        }
        impl<P: crate::NonlinearInequalityValues>
            crate::NonlinearInequalityValues for $wrapper
        {
            type Param = P::Param;
            type Values = P::Values;
            type Error = EvaluationError<P::Error>;
            fn nonlinear_inequalities(
                &self,
                x: &P::Param,
            ) -> Result<P::Values, crate::ConstraintSolveError<Self::Error>>
            {
                self.problem
                    .nonlinear_inequalities_with_context(x, &self.context)
            }
        }
        impl<P: crate::ConstraintJacobian> crate::ConstraintJacobian
            for $wrapper
        {
            type Param = P::Param;
            type Jacobian = P::Jacobian;
            type Error = EvaluationError<P::Error>;
            fn constraint_jacobian(
                &self,
                x: &P::Param,
            ) -> Result<P::Jacobian, crate::ConstraintSolveError<Self::Error>>
            {
                self.problem
                    .constraint_jacobian_with_context(x, &self.context)
            }
        }
    };
}
implement!(ContextProblem<'_, P>);

// Transparent adapters preserve the innermost callback accounting.
macro_rules! forward_cost_context {
    () => {
        fn cost_with_context(
            &self,
            x: &<Self as crate::CostFunction>::Param,
            context: &crate::EvaluationContext,
        ) -> Result<
            <Self as crate::CostFunction>::Output,
            crate::EvaluationError<<Self as crate::CostFunction>::Error>,
        > {
            self.problem.cost_with_context(x, context)
        }
        fn cost_batch_request(&self, points: u64) -> crate::EvalCounts {
            self.problem.cost_batch_request(points)
        }
        fn prepare_cost_batch(
            &self,
            points: u64,
        ) -> Result<
            crate::CostBatchReservation,
            <Self as crate::CostFunction>::Error,
        > {
            self.problem.prepare_cost_batch(points)
        }
        fn cost_in_batch(
            &self,
            x: &<Self as crate::CostFunction>::Param,
            reservation: &crate::CostBatchReservation,
        ) -> Result<
            <Self as crate::CostFunction>::Output,
            <Self as crate::CostFunction>::Error,
        > {
            self.problem.cost_in_batch(x, reservation)
        }
    };
}
pub(crate) use forward_cost_context;
macro_rules! forward_gradient_context {
    () => {
        fn gradient_with_context(
            &self,
            x: &<Self as crate::CostFunction>::Param,
            context: &crate::EvaluationContext,
        ) -> Result<
            <Self as crate::Gradient>::Gradient,
            crate::EvaluationError<<Self as crate::CostFunction>::Error>,
        > {
            self.problem.gradient_with_context(x, context)
        }
        fn cost_and_gradient_with_context(
            &self,
            x: &<Self as crate::CostFunction>::Param,
            context: &crate::EvaluationContext,
        ) -> Result<
            (
                <Self as crate::CostFunction>::Output,
                <Self as crate::Gradient>::Gradient,
            ),
            crate::EvaluationError<<Self as crate::CostFunction>::Error>,
        > {
            self.problem.cost_and_gradient_with_context(x, context)
        }
    };
}
pub(crate) use forward_gradient_context;
macro_rules! forward_mini_batch_context {
    () => {
        fn batch_gradient_with_context(
            &self,
            x: &<Self as crate::CostFunction>::Param,
            batch: &[usize],
            context: &crate::EvaluationContext,
        ) -> Result<
            <Self as crate::MiniBatchGradient>::Gradient,
            crate::EvaluationError<<Self as crate::CostFunction>::Error>,
        > {
            self.problem.batch_gradient_with_context(x, batch, context)
        }
    };
}
pub(crate) use forward_mini_batch_context;
macro_rules! forward_hessian_context {
    () => {
        fn hessian_with_context(
            &self,
            x: &<Self as crate::CostFunction>::Param,
            context: &crate::EvaluationContext,
        ) -> Result<
            <Self as crate::Hessian>::Hessian,
            crate::EvaluationError<<Self as crate::CostFunction>::Error>,
        > {
            self.problem.hessian_with_context(x, context)
        }
        fn cost_and_gradient_and_hessian_with_context(
            &self,
            x: &<Self as crate::CostFunction>::Param,
            context: &crate::EvaluationContext,
        ) -> Result<
            (
                <Self as crate::CostFunction>::Output,
                <Self as crate::Gradient>::Gradient,
                <Self as crate::Hessian>::Hessian,
            ),
            crate::EvaluationError<<Self as crate::CostFunction>::Error>,
        > {
            self.problem
                .cost_and_gradient_and_hessian_with_context(x, context)
        }
    };
}
pub(crate) use forward_hessian_context;
macro_rules! forward_hessian_product_context {
    () => {
        fn hessian_product_with_context(
            &self,
            x: &<Self as crate::CostFunction>::Param,
            v: &<Self as crate::CostFunction>::Param,
            context: &crate::EvaluationContext,
        ) -> Result<
            <Self as crate::Gradient>::Gradient,
            crate::EvaluationError<<Self as crate::CostFunction>::Error>,
        > {
            self.problem.hessian_product_with_context(x, v, context)
        }
    };
}
pub(crate) use forward_hessian_product_context;
macro_rules! forward_residual_context {
    () => {
        fn residual_with_context(
            &self,
            x: &<Self as crate::Residual>::Param,
            context: &crate::EvaluationContext,
        ) -> Result<
            <Self as crate::Residual>::Output,
            crate::EvaluationError<<Self as crate::Residual>::Error>,
        > {
            self.problem.residual_with_context(x, context)
        }
    };
}
pub(crate) use forward_residual_context;
macro_rules! forward_jacobian_context {
    () => {
        fn jacobian_with_context(
            &self,
            x: &<Self as crate::Residual>::Param,
            context: &crate::EvaluationContext,
        ) -> Result<
            <Self as crate::Jacobian>::Jacobian,
            crate::EvaluationError<<Self as crate::Residual>::Error>,
        > {
            self.problem.jacobian_with_context(x, context)
        }
        fn residual_and_jacobian_with_context(
            &self,
            x: &<Self as crate::Residual>::Param,
            context: &crate::EvaluationContext,
        ) -> Result<
            (
                <Self as crate::Residual>::Output,
                <Self as crate::Jacobian>::Jacobian,
            ),
            crate::EvaluationError<<Self as crate::Residual>::Error>,
        > {
            self.problem.residual_and_jacobian_with_context(x, context)
        }
    };
}
pub(crate) use forward_jacobian_context;
macro_rules! forward_equalities_context {
    () => {
        fn nonlinear_equalities_with_context(
            &self,
            x: &<Self as crate::NonlinearEqualityValues>::Param,
            context: &crate::EvaluationContext,
        ) -> Result<
            <Self as crate::NonlinearEqualityValues>::Values,
            crate::ConstraintSolveError<
                crate::EvaluationError<
                    <Self as crate::NonlinearEqualityValues>::Error,
                >,
            >,
        > {
            self.problem.nonlinear_equalities_with_context(x, context)
        }
    };
}
pub(crate) use forward_equalities_context;
macro_rules! forward_inequalities_context {
    () => {
        fn nonlinear_inequalities_with_context(
            &self,
            x: &<Self as crate::NonlinearInequalityValues>::Param,
            context: &crate::EvaluationContext,
        ) -> Result<
            <Self as crate::NonlinearInequalityValues>::Values,
            crate::ConstraintSolveError<
                crate::EvaluationError<
                    <Self as crate::NonlinearInequalityValues>::Error,
                >,
            >,
        > {
            self.problem.nonlinear_inequalities_with_context(x, context)
        }
    };
}
pub(crate) use forward_inequalities_context;
macro_rules! forward_constraint_jacobian_context {
    () => {
        fn constraint_jacobian_with_context(
            &self,
            x: &<Self as crate::ConstraintJacobian>::Param,
            context: &crate::EvaluationContext,
        ) -> Result<
            <Self as crate::ConstraintJacobian>::Jacobian,
            crate::ConstraintSolveError<
                crate::EvaluationError<
                    <Self as crate::ConstraintJacobian>::Error,
                >,
            >,
        > {
            self.problem.constraint_jacobian_with_context(x, context)
        }
    };
}
pub(crate) use forward_constraint_jacobian_context;
