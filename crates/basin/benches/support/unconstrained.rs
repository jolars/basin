//! Empty constraint blocks let constrained solvers use the same benchmark
//! objectives and analytic derivatives as the unconstrained solvers.

use basin::{CostFunction, Gradient};

#[derive(Default)]
pub struct Unconstrained<P>(pub P);

impl<P: CostFunction> basin::Constraints for Unconstrained<P> {
    type Set = basin::NoConstraints;
    fn constraints(
        &self,
    ) -> <Self::Set as basin::ConstraintDescription>::Borrowed<'_> {
        basin::ConstraintSet::new()
    }
}
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
