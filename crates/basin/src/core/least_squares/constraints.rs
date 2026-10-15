use super::RobustLeastSquares;

impl<P, L, F> crate::NonlinearEqualityValues for RobustLeastSquares<P, L, F>
where
    P: crate::NonlinearEqualityValues,
    F: crate::Scalar,
{
    type Param = P::Param;
    type Values = P::Values;
    type Error = P::Error;
    fn nonlinear_equalities(
        &self,
        x: &Self::Param,
    ) -> Result<Self::Values, crate::ConstraintSolveError<Self::Error>> {
        self.problem.nonlinear_equalities(x)
    }
}

impl<P, L, F> crate::NonlinearInequalityValues for RobustLeastSquares<P, L, F>
where
    P: crate::NonlinearInequalityValues,
    F: crate::Scalar,
{
    type Param = P::Param;
    type Values = P::Values;
    type Error = P::Error;
    fn nonlinear_inequalities(
        &self,
        x: &Self::Param,
    ) -> Result<Self::Values, crate::ConstraintSolveError<Self::Error>> {
        self.problem.nonlinear_inequalities(x)
    }
}

impl<P, L, F> crate::ConstraintJacobian for RobustLeastSquares<P, L, F>
where
    P: crate::ConstraintJacobian,
    F: crate::Scalar,
{
    type Param = P::Param;
    type Jacobian = P::Jacobian;
    type Error = P::Error;
    fn constraint_jacobian(
        &self,
        x: &Self::Param,
    ) -> Result<Self::Jacobian, crate::ConstraintSolveError<Self::Error>> {
        self.problem.constraint_jacobian(x)
    }
}
