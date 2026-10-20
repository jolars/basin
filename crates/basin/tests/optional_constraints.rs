//! Constraint declarations are independent of evaluation capabilities.

use basin::{
    BoxBounds, Cobyla, Constrained, ConstraintDescription, ConstraintSet,
    ConstraintSolveError, Constraints, CostFunction, DenseMatrix, Executor,
    FiniteDiff, Gradient, GradientDescent, Jacobian, LevenbergMarquardt,
    NegInPlace, NonlinearInequalityValues, NormSquared,
    ProjectedGradientDescent, Residual, RobustLeastSquares, Scalar,
    ScaleInPlace, ScaledAdd, Slsqp, SquaredLoss, VectorLen,
};
use std::{convert::Infallible, marker::PhantomData, ops::IndexMut};

#[path = "support/backend_aliases.rs"]
mod backend_aliases;

struct Plain<V, F>(PhantomData<fn() -> (V, F)>);

impl<V: VectorLen + IndexMut<usize, Output = F>, F: Scalar> CostFunction
    for Plain<V, F>
{
    type Param = V;
    type Output = F;
    type Error = Infallible;

    fn cost(&self, x: &V) -> Result<F, Infallible> {
        let two = F::one() + F::one();
        Ok((0..x.vec_len()).map(|i| (x[i] - two).powi(2)).sum())
    }
}

impl<V: VectorLen + IndexMut<usize, Output = F> + Clone, F: Scalar> Gradient
    for Plain<V, F>
{
    type Gradient = V;

    fn gradient(&self, x: &V) -> Result<V, Infallible> {
        let two = F::one() + F::one();
        let mut gradient = x.clone();
        for i in 0..gradient.vec_len() {
            gradient[i] = two * (x[i] - two);
        }
        Ok(gradient)
    }
}

fn plain_solve<V, F>(make: impl Fn(&[F]) -> V)
where
    V: VectorLen
        + IndexMut<usize, Output = F>
        + Clone
        + ScaledAdd<F>
        + NegInPlace
        + ScaleInPlace<F>
        + NormSquared<F>,
    F: Scalar,
{
    let result = Executor::from_start(
        Plain::<V, F>(PhantomData),
        GradientDescent::new(F::from_f64(0.5).unwrap())
            .with_absolute_gradient_tolerance(F::zero()),
        make(&[F::zero(), F::one()]),
    )
    .max_iter(10)
    .run()
    .unwrap();
    assert_eq!(result.param()[0], F::one() + F::one());
    assert_eq!(result.param()[1], F::one() + F::one());
    assert_eq!(result.cost(), F::zero());
}

#[test]
fn vec_needs_no_constraint_declaration() {
    plain_solve::<Vec<f64>, f64>(|x| x.to_vec());
    plain_solve::<Vec<f32>, f32>(|x| x.to_vec());
}

#[cfg(feature = "nalgebra_all")]
#[test]
fn nalgebra_needs_no_constraint_declaration() {
    use backend_aliases::nalgebra::DVector;
    plain_solve::<DVector<f64>, f64>(DVector::from_column_slice);
    plain_solve::<DVector<f32>, f32>(DVector::from_column_slice);
}

#[cfg(feature = "ndarray_all")]
#[test]
fn ndarray_needs_no_constraint_declaration() {
    use backend_aliases::ndarray::Array1;
    plain_solve::<Array1<f64>, f64>(|x| Array1::from_vec(x.to_vec()));
    plain_solve::<Array1<f32>, f32>(|x| Array1::from_vec(x.to_vec()));
}

#[cfg(feature = "faer_all")]
#[test]
fn faer_needs_no_constraint_declaration() {
    use backend_aliases::faer::Col;
    plain_solve::<Col<f64>, f64>(|x| Col::from_fn(x.len(), |i| x[i]));
    plain_solve::<Col<f32>, f32>(|x| Col::from_fn(x.len(), |i| x[i]));
}

struct UnreadDeclaration;

impl CostFunction for UnreadDeclaration {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Plain::<Vec<f64>, f64>(PhantomData).cost(x)
    }
}

impl Gradient for UnreadDeclaration {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Plain::<Vec<f64>, f64>(PhantomData).gradient(x)
    }
}

impl Constraints for UnreadDeclaration {
    type Set = ConstraintSet<BoxBounds<Vec<f64>>>;

    fn constraints(
        &self,
    ) -> <Self::Set as ConstraintDescription>::Borrowed<'_> {
        panic!("Unconstrained solvers must not read constraint declarations.")
    }
}

#[test]
fn unconstrained_solver_does_not_read_constraint_metadata() {
    let result = Executor::from_start(
        UnreadDeclaration,
        GradientDescent::new(0.5),
        vec![0.0, 0.0],
    )
    .max_iter(2)
    .run()
    .unwrap();
    assert_eq!(result.param(), &vec![2.0, 2.0]);
}

type BoxedPlain =
    Constrained<Plain<Vec<f64>, f64>, ConstraintSet<BoxBounds<Vec<f64>>>>;

fn boxed() -> BoxedPlain {
    Constrained::new(
        Plain(PhantomData),
        ConstraintSet::new().with_bounds(BoxBounds::new(
            Some(vec![-1.0; 2]),
            Some(vec![1.0; 2]),
        )),
    )
}

#[test]
fn solver_choice_determines_whether_bounds_are_enforced() {
    let unconstrained = Executor::from_start(
        boxed(),
        GradientDescent::new(0.5),
        vec![0.0, 0.0],
    )
    .max_iter(2)
    .run()
    .unwrap();
    assert_eq!(unconstrained.param(), &vec![2.0, 2.0]);

    let constrained = Executor::from_start(
        boxed(),
        ProjectedGradientDescent::new(0.5),
        vec![0.0, 0.0],
    )
    .max_iter(2)
    .run()
    .unwrap();
    assert_eq!(constrained.param(), &vec![1.0, 1.0]);
}

#[test]
fn an_attached_set_replaces_the_inner_declaration() {
    let problem = Constrained::new(
        boxed(),
        ConstraintSet::new().with_bounds(BoxBounds::new(
            Some(vec![0.0; 2]),
            Some(vec![0.5; 2]),
        )),
    );
    let result = Executor::from_start(
        problem,
        ProjectedGradientDescent::new(0.5),
        vec![0.0, 0.0],
    )
    .max_iter(2)
    .run()
    .unwrap();
    assert_eq!(result.param(), &vec![0.5, 0.5]);
}

#[test]
fn finite_differences_need_no_constraint_declaration() {
    let result = Executor::from_start(
        FiniteDiff::new(Plain::<Vec<f64>, f64>(PhantomData)),
        GradientDescent::new(0.25),
        vec![0.0, 0.0],
    )
    .max_iter(100)
    .run()
    .unwrap();
    assert!(result.cost() < 1e-12);
}

struct PlainResidual;

impl Residual for PlainResidual {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = Infallible;

    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(x.iter().map(|x| x - 2.0).collect())
    }
}

impl Jacobian for PlainResidual {
    type Jacobian = DenseMatrix;

    fn jacobian(&self, _: &Vec<f64>) -> Result<DenseMatrix, Infallible> {
        Ok(DenseMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]))
    }
}

#[test]
fn least_squares_and_robust_loss_need_no_constraint_declaration() {
    let raw = Executor::from_start(
        PlainResidual,
        LevenbergMarquardt::new(),
        vec![0.0, 0.0],
    )
    .max_iter(100)
    .run()
    .unwrap();
    assert!(raw.cost() < 1e-12);
    let robust = Executor::from_start(
        RobustLeastSquares::new(PlainResidual, SquaredLoss),
        LevenbergMarquardt::new(),
        vec![0.0, 0.0],
    )
    .max_iter(100)
    .run()
    .unwrap();
    assert!(robust.cost() < 1e-12);
}

impl NonlinearInequalityValues for Plain<Vec<f64>, f64> {
    type Param = Vec<f64>;
    type Values = Vec<f64>;
    type Error = Infallible;

    fn nonlinear_inequalities(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, ConstraintSolveError<Infallible>> {
        Ok(vec![x[0] - 1.0])
    }
}

impl basin::ConstraintJacobian for Plain<Vec<f64>, f64> {
    type Param = Vec<f64>;
    type Jacobian = DenseMatrix;
    type Error = Infallible;

    fn constraint_jacobian(
        &self,
        _: &Vec<f64>,
    ) -> Result<DenseMatrix, ConstraintSolveError<Infallible>> {
        Ok(DenseMatrix::from_row_slice(1, 2, &[1.0, 0.0]))
    }
}

#[test]
fn constrained_solvers_consume_callbacks_without_an_inner_declaration() {
    let problem = || {
        Constrained::new(
            Plain::<Vec<f64>, f64>(PhantomData),
            ConstraintSet::new().with_nonlinear_inequalities(1),
        )
    };
    let cobyla = Executor::from_start(
        problem(),
        Cobyla::new().with_final_radius(1e-6),
        vec![0.0, 0.0],
    )
    .max_iter(1000)
    .run()
    .unwrap();
    assert!((cobyla.param()[0] - 1.0).abs() < 1e-4);
    assert!((cobyla.param()[1] - 2.0).abs() < 1e-4);
    let slsqp = Executor::from_start(problem(), Slsqp::new(), vec![0.0, 0.0])
        .max_iter(100)
        .run()
        .unwrap();
    assert!((slsqp.param()[0] - 1.0).abs() < 1e-6);
    assert!((slsqp.param()[1] - 2.0).abs() < 1e-6);
}
