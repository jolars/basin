use super::{LossFunction, RobustLeastSquares};
use crate::core::constraint::{
    BoxConstraints, ConstraintJacobian, LinearConstraints,
    LinearEqualityConstraints, LinearInequalityConstraints,
    NonlinearConstraints, NonlinearInequalityConstraints,
};
use crate::{CostFunction, Residual, Scalar, VectorIndex, VectorLen};

impl<P, L, V, F> BoxConstraints for RobustLeastSquares<P, L, F>
where
    F: Scalar,
    P: Residual<Param = V> + BoxConstraints<Param = V>,
    <P as Residual>::Output: VectorLen + VectorIndex<F>,
    L: LossFunction<F>,
{
    fn lower(&self) -> &V {
        self.problem.lower()
    }
    fn upper(&self) -> &V {
        self.problem.upper()
    }
}

macro_rules! forward_linear {
    ($trait:ident) => {
        impl<P, L, V, F> $trait for RobustLeastSquares<P, L, F>
        where
            F: Scalar,
            P: Residual<Param = V> + $trait<Param = V>,
            <P as Residual>::Output: VectorLen + VectorIndex<F>,
            L: LossFunction<F>,
        {
            type Matrix = P::Matrix;
            fn a(&self) -> &Self::Matrix {
                self.problem.a()
            }
            fn b(&self) -> &V {
                self.problem.b()
            }
        }
    };
}
forward_linear!(LinearEqualityConstraints);
forward_linear!(LinearInequalityConstraints);

impl<P, L, V, F> LinearConstraints for RobustLeastSquares<P, L, F>
where
    F: Scalar,
    P: Residual<Param = V> + LinearConstraints<Param = V>,
    <P as Residual>::Output: VectorLen + VectorIndex<F>,
    L: LossFunction<F>,
{
    type Matrix = P::Matrix;
    fn inequalities(&self) -> Option<(&Self::Matrix, &V)> {
        self.problem.inequalities()
    }
    fn equalities(&self) -> Option<(&Self::Matrix, &V)> {
        self.problem.equalities()
    }
    fn lower(&self) -> Option<&V> {
        self.problem.lower()
    }
    fn upper(&self) -> Option<&V> {
        self.problem.upper()
    }
}

impl<P, L, V, F> NonlinearInequalityConstraints for RobustLeastSquares<P, L, F>
where
    F: Scalar,
    P: Residual<Param = V>
        + NonlinearInequalityConstraints<
            Param = V,
            Error = <P as Residual>::Error,
        >,
    <P as Residual>::Output: VectorLen + VectorIndex<F>,
    L: LossFunction<F>,
{
    fn constraints(&self, x: &V) -> Result<V, Self::Error> {
        self.problem.constraints(x)
    }
    fn num_constraints(&self) -> usize {
        self.problem.num_constraints()
    }
}

impl<P, L, V, F> NonlinearConstraints for RobustLeastSquares<P, L, F>
where
    F: Scalar,
    P: Residual<Param = V>
        + NonlinearConstraints<Param = V, Error = <P as Residual>::Error>,
    <P as Residual>::Output: VectorLen + VectorIndex<F>,
    L: LossFunction<F>,
{
    type Matrix = P::Matrix;
    fn nonlinear_constraints(&self, x: &V) -> Result<V, Self::Error> {
        self.problem.nonlinear_constraints(x)
    }
    fn num_nonlinear_constraints(&self) -> usize {
        self.problem.num_nonlinear_constraints()
    }
    fn nonlinear_equalities(&self, x: &V) -> Result<Option<V>, Self::Error> {
        self.problem.nonlinear_equalities(x)
    }
    fn num_nonlinear_equalities(&self) -> usize {
        self.problem.num_nonlinear_equalities()
    }
    fn inequalities(&self) -> Option<(&Self::Matrix, &V)> {
        self.problem.inequalities()
    }
    fn equalities(&self) -> Option<(&Self::Matrix, &V)> {
        self.problem.equalities()
    }
    fn lower(&self) -> Option<&V> {
        self.problem.lower()
    }
    fn upper(&self) -> Option<&V> {
        self.problem.upper()
    }
}

impl<P, L, V, F> ConstraintJacobian for RobustLeastSquares<P, L, F>
where
    F: Scalar,
    P: Residual<Param = V>
        + ConstraintJacobian<Param = V, Error = <P as Residual>::Error>,
    <P as Residual>::Output: VectorLen + VectorIndex<F>,
    L: LossFunction<F>,
{
    fn constraint_jacobian(
        &self,
        x: &V,
    ) -> Result<Self::Matrix, <Self as CostFunction>::Error> {
        self.problem.constraint_jacobian(x)
    }
}
