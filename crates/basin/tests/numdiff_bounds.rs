use basin::{
    BoundedFiniteDiff, CostFunction, Gradient, Jacobian, Method, Residual,
};

struct Domain;

impl basin::Constraints for Domain {
    type Set = basin::NoConstraints;
    fn constraints(
        &self,
    ) -> <Self::Set as basin::ConstraintDescription>::Borrowed<'_> {
        basin::ConstraintSet::new()
    }
}
impl CostFunction for Domain {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = &'static str;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        if x[0] < 0. || x[0] > 1. || x[1] != 2. {
            return Err("outside");
        }
        Ok(x[0] * x[0] + 3. * x[0] + x[1])
    }
}
impl Residual for Domain {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = &'static str;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(vec![self.cost(x)?, x[0].powi(3)])
    }
}

#[test]
fn one_sided_second_order_and_fixed_coordinate() {
    let p = BoundedFiniteDiff::new(Domain, vec![0., 2.], vec![1., 2.])
        .jacobian_method(Method::Central);
    for x in [vec![0., 2.], vec![1., 2.]] {
        let g = p.gradient(&x).unwrap();
        assert!((g[0] - (2. * x[0] + 3.)).abs() < 1e-8, "{g:?}");
        assert_eq!(g[1], 0.);
        let j = p.jacobian(&x).unwrap();
        assert!((j.get(0, 0) - g[0]).abs() < 1e-8);
        assert_eq!(j.get(0, 1), 0.);
    }
}

#[test]
fn adjacent_float_interval_uses_the_available_probe() {
    struct Identity;

    impl basin::Constraints for Identity {
        type Set = basin::NoConstraints;
        fn constraints(
            &self,
        ) -> <Self::Set as basin::ConstraintDescription>::Borrowed<'_> {
            basin::ConstraintSet::new()
        }
    }
    impl CostFunction for Identity {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
            Ok(x[0])
        }
    }
    let lo = 1.0;
    let hi = f64::from_bits(1.0_f64.to_bits() + 1);
    let p = BoundedFiniteDiff::new(Identity, vec![lo], vec![hi]);
    assert_eq!(p.gradient(&vec![lo]).unwrap(), vec![1.]);
    assert_eq!(p.gradient(&vec![hi]).unwrap(), vec![1.]);
}

#[test]
fn sub_ulp_requested_step_still_uses_distinct_probes() {
    for step in [f64::MIN_POSITIVE, 0.2 * f64::EPSILON] {
        let p = BoundedFiniteDiff::new(Domain, vec![0., 2.], vec![1., 2.])
            .with_step(step);
        assert!((p.gradient(&vec![0.5, 2.]).unwrap()[0] - 4.).abs() < 1e-8);
    }
}
#[test]
fn empty_parameter_vector_keeps_residual_row_count() {
    struct Constant;

    impl basin::Constraints for Constant {
        type Set = basin::NoConstraints;
        fn constraints(
            &self,
        ) -> <Self::Set as basin::ConstraintDescription>::Borrowed<'_> {
            basin::ConstraintSet::new()
        }
    }
    impl Residual for Constant {
        type Param = Vec<f64>;
        type Output = Vec<f64>;
        type Error = std::convert::Infallible;
        fn residual(&self, _: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
            Ok(vec![1., 2.])
        }
    }
    let p = BoundedFiniteDiff::new(Constant, Vec::<f64>::new(), vec![]);
    let j = p.jacobian(&vec![]).unwrap();
    assert_eq!(j.nrows(), 2);
    assert_eq!(j.ncols(), 0);
}

use basin::{DenseMatrixFromFn, MatrixIndex, Scalar, VectorIndex, VectorLen};
#[path = "support/backend_aliases.rs"]
mod backend_aliases;
struct BackendDomain<V, F> {
    make: fn(&[F]) -> V,
}

impl<V: VectorIndex<F>, F: Scalar> basin::Constraints for BackendDomain<V, F> {
    type Set = basin::ConstraintSet<
        basin::NoBounds,
        basin::NoLinearConstraints,
        basin::NonlinearBlocks<basin::NonlinearBlock, basin::NonlinearBlock>,
    >;
    fn constraints(
        &self,
    ) -> <Self::Set as basin::ConstraintDescription>::Borrowed<'_> {
        basin::ConstraintSet::new()
            .with_nonlinear_equalities(1)
            .with_nonlinear_inequalities(1)
    }
}
impl<V: VectorIndex<F>, F: Scalar> CostFunction for BackendDomain<V, F> {
    type Param = V;
    type Output = F;
    type Error = &'static str;
    fn cost(&self, x: &V) -> Result<F, Self::Error> {
        let a = x.get_scalar(0);
        let b = x.get_scalar(1);
        if a < F::zero() || a > F::one() || b != F::one() {
            return Err("outside");
        }
        Ok(a * a + b)
    }
}
impl<V: VectorIndex<F>, F: Scalar> Residual for BackendDomain<V, F> {
    type Param = V;
    type Output = V;
    type Error = &'static str;
    fn residual(&self, x: &V) -> Result<V, Self::Error> {
        Ok((self.make)(&[self.cost(x)?, x.get_scalar(0)]))
    }
}
impl<V: VectorIndex<F> + DenseMatrixFromFn<F>, F: Scalar>
    basin::NonlinearInequalityValues for BackendDomain<V, F>
{
    type Param = V;
    type Values = V;
    type Error = &'static str;
    fn nonlinear_inequalities(
        &self,
        x: &Self::Param,
    ) -> Result<Self::Values, basin::ConstraintSolveError<Self::Error>> {
        (|| -> Result<V, &'static str> { Ok((self.make)(&[self.cost(x)?])) })()
            .map_err(basin::ConstraintSolveError::Evaluation)
    }
}
impl<V: VectorIndex<F> + DenseMatrixFromFn<F>, F: Scalar>
    basin::NonlinearEqualityValues for BackendDomain<V, F>
{
    type Param = V;
    type Values = V;
    type Error = &'static str;
    fn nonlinear_equalities(
        &self,
        x: &Self::Param,
    ) -> Result<Self::Values, basin::ConstraintSolveError<Self::Error>> {
        (|| -> Result<Option<V>, &'static str> {
            self.cost(x)?;
            Ok(Some((self.make)(&[x.get_scalar(0)])))
        })()
        .map_err(basin::ConstraintSolveError::Evaluation)?
        .ok_or(basin::ConstraintSolveError::InvalidConstraints(
            basin::ConstraintError::DimensionMismatch {
                block: basin::ConstraintBlock::NonlinearEqualities,
                axis: basin::ConstraintAxis::Length,
                expected: 1,
                actual: 0,
            },
        ))
    }
}

fn check_backend<V, F>(make: fn(&[F]) -> V)
where
    V: Clone
        + std::ops::Index<usize, Output = F>
        + VectorIndex<F>
        + VectorLen
        + DenseMatrixFromFn<F>
        + Sync,
    F: Scalar + Send + Sync,
    V::Matrix: MatrixIndex<F>,
{
    check_scalar_probe_results(make);
    use basin::ConstraintJacobian;
    for method in [Method::Forward, Method::Central] {
        let p = BoundedFiniteDiff::new(
            BackendDomain { make },
            make(&[F::zero(), F::one()]),
            make(&[F::one(), F::one()]),
        )
        .gradient_method(method)
        .jacobian_method(method);
        for a in [F::zero(), F::one()] {
            let x = make(&[a, F::one()]);
            let g = p.gradient(&x).unwrap();
            let j = p.jacobian(&x).unwrap();
            let c = p.constraint_jacobian(&x).unwrap();
            let tolerance =
                F::from_f64(if F::epsilon().to_f64().unwrap() > 1e-10 {
                    2e-3
                } else {
                    1e-7
                })
                .unwrap();
            assert!((g.get_scalar(0) - a - a).abs() < tolerance);
            assert_eq!(g.get_scalar(1), F::zero());
            assert!((j.matrix_entry(0, 0) - a - a).abs() < tolerance);
            assert!((j.matrix_entry(1, 0) - F::one()).abs() < tolerance);
            assert!((c.matrix_entry(0, 0) - F::one()).abs() < tolerance);
            assert!((c.matrix_entry(1, 0) - a - a).abs() < tolerance);
            assert_eq!(j.matrix_entry(0, 1), F::zero());
            assert_eq!(c.matrix_entry(1, 1), F::zero());
        }
    }
}

#[derive(Clone)]
struct ScalarObjective<V, F> {
    make: fn(&[F]) -> V,
    lower: Vec<F>,
    upper: Vec<F>,
    linear: bool,
}

impl<V: VectorLen + VectorIndex<F>, F: Scalar> CostFunction
    for ScalarObjective<V, F>
{
    type Param = V;
    type Output = F;
    type Error = &'static str;

    fn cost(&self, x: &V) -> Result<F, Self::Error> {
        let mut cost = F::zero();
        for j in 0..x.vec_len() {
            let value = x.get_scalar(j);
            if value < self.lower[j] || value > self.upper[j] {
                return Err("outside domain");
            }
            cost = cost
                + if self.linear {
                    value
                } else {
                    value * value + F::from_f64(3.0).unwrap() * value
                };
        }
        Ok(cost)
    }
}

impl<V: VectorLen + VectorIndex<F>, F: Scalar> Residual
    for ScalarObjective<V, F>
{
    type Param = V;
    type Output = V;
    type Error = &'static str;

    fn residual(&self, x: &V) -> Result<V, Self::Error> {
        Ok((self.make)(&[self.cost(x)?]))
    }
}

fn check_scalar_probe_results<V, F>(make: fn(&[F]) -> V)
where
    V: Clone + VectorLen + VectorIndex<F> + DenseMatrixFromFn<F> + Sync,
    V::Matrix: MatrixIndex<F>,
    F: Scalar + Send + Sync,
{
    let zero = F::zero();
    let one = F::one();
    let half = F::from_f64(0.5).unwrap();
    let adjacent = one + F::epsilon();
    for (values, lower, upper, linear) in [
        (vec![], vec![], vec![], false),
        (vec![one], vec![one], vec![one], false),
        (vec![one], vec![one], vec![adjacent], true),
        (vec![adjacent], vec![one], vec![adjacent], true),
        (
            vec![zero, one, half, half],
            vec![zero; 4],
            vec![one, one, one, half],
            false,
        ),
        (
            vec![half],
            vec![F::neg_infinity()],
            vec![F::infinity()],
            false,
        ),
    ] {
        let x = make(&values);
        for method in [Method::Forward, Method::Central] {
            for step in [None, Some(half), Some(F::min_positive_value())] {
                let mut p = BoundedFiniteDiff::new(
                    ScalarObjective {
                        make,
                        lower: lower.clone(),
                        upper: upper.clone(),
                        linear,
                    },
                    make(&lower),
                    make(&upper),
                )
                .gradient_method(method)
                .jacobian_method(method);
                if let Some(step) = step {
                    p = p.with_step(step);
                }
                // All free coordinates have magnitude at most one. Match the
                // gradient's step explicitly to bypass MINPACK Jacobian scaling.
                let reference_step = step.unwrap_or_else(|| match method {
                    Method::Forward => F::epsilon().sqrt(),
                    _ => F::epsilon().cbrt(),
                });
                let reference =
                    p.clone().with_step(reference_step).jacobian(&x).unwrap();
                let owned = p.gradient(&x).unwrap();
                let (cost, fused) = p.cost_and_gradient(&x).unwrap();
                assert_eq!(cost, p.cost(&x).unwrap());
                let mut into = make(&[]);
                let mut fused_into = make(&vec![one; values.len() + 1]);
                for _ in 0..2 {
                    p.gradient_into(&x, &mut into).unwrap();
                    assert_eq!(
                        p.cost_and_gradient_into(&x, &mut fused_into).unwrap(),
                        cost
                    );
                    for result in [&owned, &fused, &into, &fused_into] {
                        assert_eq!(result.vec_len(), values.len());
                        for j in 0..values.len() {
                            let actual = result.get_scalar(j);
                            assert_eq!(
                                actual.to_f64().unwrap().to_bits(),
                                reference
                                    .matrix_entry(0, j)
                                    .to_f64()
                                    .unwrap()
                                    .to_bits()
                            );
                            let expected = if lower[j] == upper[j] {
                                zero
                            } else if linear {
                                one
                            } else {
                                values[j]
                                    + values[j]
                                    + F::from_f64(3.0).unwrap()
                            };
                            let tolerance = if method == Method::Forward {
                                0.51
                            } else {
                                0.01
                            };
                            // Tiny requested steps can cancel in the objective;
                            // bit equality above still checks the original behavior.
                            if step != Some(F::min_positive_value()) || linear {
                                assert!(
                                    (actual - expected).abs().to_f64().unwrap()
                                        < tolerance,
                                    "{method:?}, step={step:?}, coordinate={j}: {actual:?} != {expected:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn vec_derivatives() {
    check_backend::<_, f64>(|v| v.to_vec());
    check_backend::<_, f32>(|v| v.to_vec());
}
#[cfg(feature = "nalgebra_all")]
#[test]
fn nalgebra_derivatives() {
    check_backend::<_, f64>(
        backend_aliases::nalgebra::DVector::from_column_slice,
    );
    check_backend::<_, f32>(
        backend_aliases::nalgebra::DVector::from_column_slice,
    );
}
#[cfg(feature = "ndarray_all")]
#[test]
fn ndarray_derivatives() {
    check_backend::<_, f64>(|v| {
        backend_aliases::ndarray::Array1::from_vec(v.to_vec())
    });
    check_backend::<_, f32>(|v| {
        backend_aliases::ndarray::Array1::from_vec(v.to_vec())
    });
}
#[cfg(feature = "faer_all")]
#[test]
fn faer_derivatives() {
    check_backend::<_, f64>(|v| {
        backend_aliases::faer::Col::from_fn(v.len(), |i| v[i])
    });
    check_backend::<_, f32>(|v| {
        backend_aliases::faer::Col::from_fn(v.len(), |i| v[i])
    });
}

fn check_unbounded_matrix_derivatives<V>(make: fn(&[f64]) -> V)
where
    V: Clone + VectorIndex + VectorLen + DenseMatrixFromFn + Sync + Send,
    V::Matrix: MatrixIndex,
{
    use basin::Hessian;
    struct Smooth<V> {
        make: fn(&[f64]) -> V,
    }

    impl<V: VectorIndex> basin::Constraints for Smooth<V> {
        type Set = basin::NoConstraints;
        fn constraints(
            &self,
        ) -> <Self::Set as basin::ConstraintDescription>::Borrowed<'_> {
            basin::ConstraintSet::new()
        }
    }
    impl<V: VectorIndex> CostFunction for Smooth<V> {
        type Param = V;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &V) -> Result<f64, Self::Error> {
            Ok(x.get_scalar(0).powi(2) + 3. * x.get_scalar(1).powi(2))
        }
    }
    impl<V: VectorIndex> Residual for Smooth<V> {
        type Param = V;
        type Output = V;
        type Error = std::convert::Infallible;
        fn residual(&self, x: &V) -> Result<V, Self::Error> {
            Ok((self.make)(&[x.get_scalar(0), 2. * x.get_scalar(1)]))
        }
    }
    let p = basin::FiniteDiff::new(Smooth { make });
    let x = make(&[0.5, 1.]);
    let h = p.hessian(&x).unwrap();
    let j = p.jacobian(&x).unwrap();
    for row in 0..2 {
        for col in 0..2 {
            let want_h = if row != col {
                0.
            } else if row == 0 {
                2.
            } else {
                6.
            };
            let want_j = if row != col {
                0.
            } else if row == 0 {
                1.
            } else {
                2.
            };
            assert!((h.matrix_entry(row, col) - want_h).abs() < 1e-6);
            assert!((j.matrix_entry(row, col) - want_j).abs() < 1e-7);
        }
    }
}
#[test]
fn vec_unbounded_matrix_derivatives() {
    check_unbounded_matrix_derivatives(|v| v.to_vec());
}
#[cfg(feature = "ndarray_all")]
#[test]
fn ndarray_unbounded_matrix_derivatives() {
    check_unbounded_matrix_derivatives(|v| {
        backend_aliases::ndarray::Array1::from_vec(v.to_vec())
    });
}
