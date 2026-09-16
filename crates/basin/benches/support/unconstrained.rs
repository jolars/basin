//! Empty constraint blocks let constrained solvers use the same benchmark
//! objectives and analytic derivatives as the unconstrained solvers.

use basin::{
    ConstraintJacobian, CostFunction, DenseMatrixFromFn, Gradient, MatVec,
    NonlinearConstraints, VectorLen,
};

#[derive(Default)]
pub struct Unconstrained<P>(pub P);

impl<P: CostFunction> CostFunction for Unconstrained<P> {
    type Param = P::Param;
    type Output = P::Output;
    type Error = P::Error;

    fn cost(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
        self.0.cost(x)
    }
}

impl<P: Gradient> Gradient for Unconstrained<P> {
    type Gradient = P::Gradient;

    fn gradient(&self, x: &Self::Param) -> Result<Self::Gradient, Self::Error> {
        self.0.gradient(x)
    }

    fn cost_and_gradient(
        &self,
        x: &Self::Param,
    ) -> Result<(Self::Output, Self::Gradient), Self::Error> {
        self.0.cost_and_gradient(x)
    }
}

impl<P: CostFunction<Output = f64>> NonlinearConstraints for Unconstrained<P>
where
    P::Param: DenseMatrixFromFn + VectorLen,
    <P::Param as DenseMatrixFromFn>::Matrix: MatVec<P::Param>,
{
    type Matrix = <P::Param as DenseMatrixFromFn>::Matrix;

    fn num_nonlinear_constraints(&self) -> usize {
        0
    }

    fn nonlinear_constraints(
        &self,
        x: &Self::Param,
    ) -> Result<Self::Param, Self::Error> {
        Ok(Self::Param::dense_from_fn(0, x.vec_len(), |_, _| 0.0).matvec(x))
    }
}

impl<P: CostFunction<Output = f64>> ConstraintJacobian for Unconstrained<P>
where
    P::Param: DenseMatrixFromFn + VectorLen,
    <P::Param as DenseMatrixFromFn>::Matrix: MatVec<P::Param>,
{
    fn constraint_jacobian(
        &self,
        x: &Self::Param,
    ) -> Result<Self::Matrix, Self::Error> {
        Ok(Self::Param::dense_from_fn(0, x.vec_len(), |_, _| 0.0))
    }
}
