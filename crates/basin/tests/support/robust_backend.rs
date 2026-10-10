use basin::{
    BoxConstraints, CauchyLoss, CostFunction, Executor, HuberLoss, Jacobian,
    PointState, Residual, RobustLeastSquares, Scalar, Solver, TerminationCode,
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

pub struct CauchyLocation<V, M, F> {
    make: fn(&[F]) -> V,
    jacobian: M,
}

impl<V: VectorIndex<F>, M, F: Scalar> Residual for CauchyLocation<V, M, F> {
    type Param = V;
    type Output = V;
    type Error = Infallible;

    fn residual(&self, x: &V) -> Result<V, Infallible> {
        Ok((self.make)(&[x.get_scalar(0) - F::one(); 4]))
    }
}

impl<V: VectorIndex<F>, M: Clone, F: Scalar> Jacobian
    for CauchyLocation<V, M, F>
{
    type Jacobian = M;

    fn jacobian(&self, _: &V) -> Result<M, Infallible> {
        Ok(self.jacobian.clone())
    }
}

pub fn check_cauchy_gradient<V, M, F, S>(fit: Location<V, M, F>, solver: S)
where
    F: Scalar,
    V: Clone + VectorIndex<F> + VectorLen,
    M: Clone,
    S: Solver<
            RobustLeastSquares<CauchyLocation<V, M, F>, CauchyLoss, F>,
            PointState<V, F>,
            Error = Infallible,
        >,
{
    // Negative initial curvature forces damping recovery before stationarity.
    let start = (fit.make)(&[-F::one()]);
    let result = Executor::new(
        RobustLeastSquares::new(
            CauchyLocation {
                make: fit.make,
                jacobian: fit.jacobian,
            },
            CauchyLoss,
        ),
        solver,
        PointState::new(start),
    )
    .max_iter(200)
    .run()
    .unwrap();
    assert_eq!(result.report.code(), TerminationCode::SolverConverged);
    let error = result.param().get_scalar(0) - F::one();
    let gradient =
        F::from_f64(4.).unwrap() * error / (F::one() + error * error);
    assert!(error.abs() < F::from_f64(1e-6).unwrap());
    assert!(gradient.abs() < F::from_f64(1e-7).unwrap());
    assert!(result.cost() < F::from_f64(1e-12).unwrap());
}

pub fn check<V, M, F, S>(fit: Location<V, M, F>, solver: S, tolerance: F)
where
    F: Scalar,
    V: Clone + VectorIndex<F> + VectorLen,
    M: Clone,
    S: Solver<
            RobustLeastSquares<Location<V, M, F>, HuberLoss, F>,
            PointState<V, F>,
            Error = Infallible,
        >,
{
    let start = (fit.make)(&[F::zero()]);
    let result = Executor::new(
        RobustLeastSquares::new(fit, HuberLoss),
        solver,
        PointState::new(start),
    )
    .max_iter(100)
    .run()
    .unwrap();
    assert_eq!(result.report.code(), TerminationCode::SolverConverged);
    assert!(
        (result.param().get_scalar(0) - F::from_f64(1.0 / 3.0).unwrap()).abs()
            < tolerance
    );
    assert!(
        (result.cost() - F::from_f64(28.0 / 3.0).unwrap()).abs() < tolerance
    );
}

pub fn check_nonfinite_damping<V, M, F, S>(fit: Location<V, M, F>, solver: S)
where
    F: Scalar,
    V: Clone + VectorIndex<F> + VectorLen,
    M: Clone,
    S: Solver<
            RobustLeastSquares<Location<V, M, F>, HuberLoss, F>,
            PointState<V, F>,
            Error = Infallible,
        >,
{
    use basin::{State, StepOutcome};
    let start = (fit.make)(&[F::zero()]);
    let mut stepper = Executor::new(
        RobustLeastSquares::new(fit, HuberLoss),
        solver,
        PointState::new(start),
    )
    .max_iter(200)
    .into_stepper()
    .unwrap();
    for _ in 0..200 {
        let point: Vec<_> = (0..stepper.state().param().vec_len())
            .map(|i| stepper.state().param().get_scalar(i))
            .collect();
        let cost = stepper.state().cost();
        let counts = *stepper.counts();
        let iteration = stepper.state().iter();
        if let StepOutcome::Stopped(report) = stepper.step().unwrap() {
            assert_eq!(report.code(), TerminationCode::SolverFailed);
            assert_eq!(stepper.state().cost(), cost);
            assert_eq!(*stepper.counts(), counts);
            assert_eq!(stepper.state().iter(), iteration);
            for (i, value) in point.iter().enumerate() {
                assert_eq!(stepper.state().param().get_scalar(i), *value);
            }
            return;
        }
    }
    panic!("non-finite damping did not stop legacy TRF");
}
