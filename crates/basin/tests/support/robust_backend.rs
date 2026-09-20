use basin::{
    BoxConstraints, CostFunction, Executor, HuberLoss, Jacobian, NllsState,
    Residual, RobustLeastSquares, Scalar, Solver, TerminationReason,
    VectorIndex, VectorLen,
};
use std::convert::Infallible;

#[derive(Clone)]
pub struct Location<V, M, F> {
    pub make: fn(&[F]) -> V,
    pub jacobian: M,
    pub lower: V,
    pub upper: V,
}

impl<V, M, F: Scalar> CostFunction for Location<V, M, F> {
    type Param = V;
    type Output = F;
    type Error = Infallible;
    fn cost(&self, _: &V) -> Result<F, Infallible> {
        panic!("the adapter must derive the objective from raw residuals")
    }
}

impl<V: VectorIndex<F>, M, F: Scalar> Residual for Location<V, M, F> {
    type Param = V;
    type Output = V;
    type Error = Infallible;
    fn residual(&self, x: &V) -> Result<V, Infallible> {
        let x = x.get_scalar(0);
        assert!(x >= self.lower.get_scalar(0) && x <= self.upper.get_scalar(0));
        Ok((self.make)(&[x, x, x, x - F::from_f64(10.0).unwrap()]))
    }
}

impl<V: VectorIndex<F>, M: Clone, F: Scalar> Jacobian for Location<V, M, F> {
    type Jacobian = M;
    fn jacobian(&self, _: &V) -> Result<M, Infallible> {
        Ok(self.jacobian.clone())
    }
}

impl<V, M, F: Scalar> BoxConstraints for Location<V, M, F> {
    fn lower(&self) -> &V {
        &self.lower
    }
    fn upper(&self) -> &V {
        &self.upper
    }
}

pub fn check<V, M, F, S>(fit: Location<V, M, F>, solver: S, tolerance: F)
where
    F: Scalar,
    V: Clone + VectorIndex<F> + VectorLen,
    M: Clone,
    S: Solver<
            RobustLeastSquares<Location<V, M, F>, HuberLoss, F>,
            NllsState<V, F>,
            Error = Infallible,
        >,
{
    let start = (fit.make)(&[F::zero()]);
    let result = Executor::new(
        RobustLeastSquares::new(fit, HuberLoss),
        solver,
        NllsState::new(start),
    )
    .max_iter(100)
    .run()
    .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        (result.param().get_scalar(0) - F::from_f64(1.0 / 3.0).unwrap()).abs()
            < tolerance
    );
    assert!(
        (result.cost() - F::from_f64(28.0 / 3.0).unwrap()).abs() < tolerance
    );
}
