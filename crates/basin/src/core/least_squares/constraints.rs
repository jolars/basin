use super::RobustLeastSquares;

impl<P, L, F, W, const I: bool> crate::NonlinearEqualityValues
    for RobustLeastSquares<P, L, F, W, I>
where
    P: crate::Residual + crate::NonlinearEqualityValues,
    F: crate::Scalar,
{
    type Param = <P as crate::NonlinearEqualityValues>::Param;
    type Values = P::Values;
    type Error = <P as crate::NonlinearEqualityValues>::Error;
    fn nonlinear_equalities(
        &self,
        x: &Self::Param,
    ) -> Result<Self::Values, crate::ConstraintSolveError<Self::Error>> {
        self.problem.nonlinear_equalities(x)
    }
}

impl<P, L, F, W, const I: bool> crate::NonlinearInequalityValues
    for RobustLeastSquares<P, L, F, W, I>
where
    P: crate::Residual + crate::NonlinearInequalityValues,
    F: crate::Scalar,
{
    type Param = <P as crate::NonlinearInequalityValues>::Param;
    type Values = P::Values;
    type Error = <P as crate::NonlinearInequalityValues>::Error;
    fn nonlinear_inequalities(
        &self,
        x: &Self::Param,
    ) -> Result<Self::Values, crate::ConstraintSolveError<Self::Error>> {
        self.problem.nonlinear_inequalities(x)
    }
}

impl<P, L, F, W, const I: bool> crate::ConstraintJacobian
    for RobustLeastSquares<P, L, F, W, I>
where
    P: crate::Residual + crate::ConstraintJacobian,
    F: crate::Scalar,
{
    type Param = <P as crate::ConstraintJacobian>::Param;
    type Jacobian = P::Jacobian;
    type Error = <P as crate::ConstraintJacobian>::Error;
    fn constraint_jacobian(
        &self,
        x: &Self::Param,
    ) -> Result<Self::Jacobian, crate::ConstraintSolveError<Self::Error>> {
        self.problem.constraint_jacobian(x)
    }
}
