use basin::{
    CostFunction, DenseMatrix, DerivativeCheckError, DerivativeChecker,
    Gradient, Jacobian, Residual,
};

struct Polynomial {
    wrong: bool,
}
impl CostFunction for Polynomial {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = &'static str;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        Ok(x[0].powi(2) - 0.5 * x[1].powi(3))
    }
}
impl Gradient for Polynomial {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(vec![2. * x[0], -1.5 * x[1].powi(2) + f64::from(self.wrong)])
    }
}
impl Residual for Polynomial {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = &'static str;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(vec![self.cost(x)?, x[0] * x[1], x[1].sin()])
    }
}
impl Jacobian for Polynomial {
    type Jacobian = DenseMatrix;
    fn jacobian(&self, x: &Vec<f64>) -> Result<DenseMatrix, Self::Error> {
        Ok(DenseMatrix::from_row_slice(
            3,
            2,
            &[
                2. * x[0],
                -1.5 * x[1].powi(2),
                x[1] + f64::from(self.wrong),
                x[0],
                0.,
                x[1].cos(),
            ],
        ))
    }
}

#[test]
fn checks_analytic_gradient_and_rectangular_jacobian() {
    let checker = DerivativeChecker::new();
    let x = vec![1.5, -1.5];
    let p = Polynomial { wrong: false };
    let g = checker.check_gradient(&p, &x).unwrap();
    assert!(g.passed(), "{g:?}");
    assert_eq!(g.comparisons.len(), 2);
    let j = checker.check_jacobian(&p, &x).unwrap();
    assert!(j.passed(), "{j:?}");
    assert_eq!(j.comparisons.len(), 6);
    assert!(g.direction.is_none());
    assert!(j.skipped_coordinates.is_empty());
}

#[test]
fn locates_mismatches_and_reports_error_scales() {
    let checker = DerivativeChecker::new();
    let x = vec![1.5, -1.5];
    let p = Polynomial { wrong: true };
    let g = checker.check_gradient(&p, &x).unwrap();
    assert!(!g.passed());
    let bad: Vec<_> = g.comparisons.iter().filter(|c| !c.passed()).collect();
    assert_eq!(bad.len(), 1);
    assert_eq!((bad[0].output, bad[0].coordinate), (0, Some(1)));
    assert!((bad[0].absolute_error - 1.).abs() < 1e-8);
    assert!((bad[0].relative_error - 1. / 3.375).abs() < 1e-8);
    let j = checker.check_jacobian(&p, &x).unwrap();
    let bad: Vec<_> = j.comparisons.iter().filter(|c| !c.passed()).collect();
    assert_eq!(bad.len(), 1);
    assert_eq!((bad[0].output, bad[0].coordinate), (1, Some(0)));
}

#[test]
fn directional_checks_compare_products_and_normalize_direction() {
    let checker = DerivativeChecker::new();
    let x = vec![1.5, -1.5];
    for multiplier in [1e-100, 1., 1e100] {
        let d = vec![multiplier, -2. * multiplier];
        let p = Polynomial { wrong: false };
        let g = checker.check_gradient_direction(&p, &x, &d).unwrap();
        assert!(g.passed(), "{g:?}");
        assert_eq!(g.direction, Some(vec![0.5, -1.]));
        assert_eq!(g.comparisons.len(), 1);
        assert_eq!(g.comparisons[0].coordinate, None);
        assert!((g.comparisons[0].analytic - 4.875).abs() < 1e-12);
        let j = checker.check_jacobian_direction(&p, &x, &d).unwrap();
        assert!(j.passed(), "{j:?}");
        assert_eq!(j.comparisons.len(), 3);
        let wrong = Polynomial { wrong: true };
        assert!(
            !checker
                .check_gradient_direction(&wrong, &x, &d)
                .unwrap()
                .passed()
        );
        assert!(
            !checker
                .check_jacobian_direction(&wrong, &x, &d)
                .unwrap()
                .passed()
        );
    }
}

struct Bounded;
impl CostFunction for Bounded {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = &'static str;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        if !(0.0..=1.0).contains(&x[0]) || x[1] != 2. {
            return Err("outside");
        }
        Ok(x[0].powi(2) + 3. * x[0] + x[1])
    }
}
impl Gradient for Bounded {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(vec![2. * x[0] + 3., 123.])
    }
}
impl Residual for Bounded {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = &'static str;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(vec![self.cost(x)?, x[0]])
    }
}
impl Jacobian for Bounded {
    type Jacobian = DenseMatrix;
    fn jacobian(&self, x: &Vec<f64>) -> Result<DenseMatrix, Self::Error> {
        Ok(DenseMatrix::from_row_slice(
            2,
            2,
            &[2. * x[0] + 3., 123., 1., 456.],
        ))
    }
}

#[test]
fn bounded_checks_stay_feasible_and_skip_fixed_columns() {
    let checker =
        DerivativeChecker::new().with_bounds(vec![0., 2.], vec![1., 2.]);
    for x in [vec![0., 2.], vec![1., 2.]] {
        let g = checker.check_gradient(&Bounded, &x).unwrap();
        assert!(g.passed(), "{g:?}");
        assert_eq!(g.skipped_coordinates, vec![1]);
        assert_eq!(g.comparisons.len(), 1);
        let j = checker.check_jacobian(&Bounded, &x).unwrap();
        assert!(j.passed(), "{j:?}");
        assert_eq!(j.skipped_coordinates, vec![1]);
        assert_eq!(j.comparisons.len(), 2);
        for d in [vec![2., 0.], vec![-2., 0.]] {
            assert!(
                checker
                    .check_gradient_direction(&Bounded, &x, &d)
                    .unwrap()
                    .passed()
            );
            assert!(
                checker
                    .check_jacobian_direction(&Bounded, &x, &d)
                    .unwrap()
                    .passed()
            );
        }
        assert!(matches!(
            checker.check_gradient_direction(&Bounded, &x, &vec![1., 1.]),
            Err(DerivativeCheckError::NoFeasibleDirection)
        ));
    }
}

struct NonFinite {
    at_base: bool,
    analytic: bool,
}
impl CostFunction for NonFinite {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = &'static str;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        Ok(if !self.analytic && (self.at_base || x[0] != 0.) {
            f64::INFINITY
        } else {
            x[0]
        })
    }
}
impl Gradient for NonFinite {
    type Gradient = Vec<f64>;
    fn gradient(&self, _: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(vec![if self.analytic { f64::NAN } else { 1. }])
    }
}

#[test]
fn non_finite_evaluations_are_errors_not_derivative_mismatches() {
    let checker = DerivativeChecker::new();
    for at_base in [false, true] {
        let p = NonFinite {
            at_base,
            analytic: false,
        };
        for result in [
            checker.check_gradient(&p, &vec![0.]),
            checker.check_gradient_direction(&p, &vec![0.], &vec![1.]),
        ] {
            match result {
                Err(DerivativeCheckError::NonFiniteEvaluation {
                    point,
                    output,
                    value,
                }) => {
                    assert_eq!(point[0] == 0., at_base);
                    assert_eq!(output, 0);
                    assert_eq!(value, f64::INFINITY);
                }
                other => panic!("{other:?}"),
            }
        }
    }
    assert!(matches!(
        checker.check_gradient(
            &NonFinite {
                at_base: false,
                analytic: true
            },
            &vec![0.]
        ),
        Err(DerivativeCheckError::NonFiniteDerivative { .. })
    ));
}

struct Linear {
    slope: f64,
    derivative: f64,
}
impl CostFunction for Linear {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = &'static str;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        Ok(self.slope * x[0])
    }
}
impl Gradient for Linear {
    type Gradient = Vec<f64>;
    fn gradient(&self, _: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(vec![self.derivative])
    }
}

#[test]
fn tolerances_handle_small_large_zero_and_overflowing_error_scales() {
    let checker = DerivativeChecker::new()
        .with_absolute_tolerance(1e-10)
        .with_relative_tolerance(1e-4);
    for slope in [1e-14, 1., 1e100] {
        let accepted = Linear {
            slope,
            derivative: slope * (1. + 1e-5),
        };
        assert!(
            checker
                .check_gradient(&accepted, &vec![0.])
                .unwrap()
                .passed()
        );
        let rejected = Linear {
            slope,
            derivative: slope * (1. + 1e-2) + 1e-8,
        };
        assert!(
            !checker
                .check_gradient(&rejected, &vec![0.])
                .unwrap()
                .passed()
        );
    }
    let exact = DerivativeChecker::new()
        .with_absolute_tolerance(0.)
        .with_relative_tolerance(0.);
    for slope in [0., 1.] {
        let p = Linear {
            slope,
            derivative: slope,
        };
        let report = exact.check_gradient(&p, &vec![0.]).unwrap();
        assert_eq!(report.comparisons[0].scaled_error, 0.);
        assert!(report.passed());
    }
    let p = Linear {
        slope: 0.,
        derivative: 1e-100,
    };
    assert_eq!(
        exact.check_gradient(&p, &vec![0.]).unwrap().comparisons[0]
            .scaled_error,
        f64::INFINITY
    );
    let p = Linear {
        slope: 1e308,
        derivative: -1e308,
    };
    let report = DerivativeChecker::new()
        .with_absolute_tolerance(1e308)
        .with_relative_tolerance(2.)
        .check_gradient(&p, &vec![0.])
        .unwrap();
    assert!(report.passed(), "{report:?}");
    assert_eq!(report.comparisons[0].absolute_error, f64::INFINITY);
    assert!((report.comparisons[0].relative_error - 2.).abs() < 1e-12);
    assert!((report.comparisons[0].scaled_error - 2. / 3.).abs() < 1e-12);
}

#[test]
fn adjacent_float_bounds_work_for_coordinates_and_directions() {
    let lo = 1.;
    let hi = f64::from_bits(1.0_f64.to_bits() + 1);
    let p = Linear {
        slope: 1.,
        derivative: 1.,
    };
    for method in [basin::Method::Forward, basin::Method::Central] {
        let checker = DerivativeChecker::new()
            .method(method)
            .with_bounds(vec![lo], vec![hi]);
        for x in [vec![lo], vec![hi]] {
            assert!(checker.check_gradient(&p, &x).unwrap().passed());
            for d in [vec![1.], vec![-1.]] {
                let report =
                    checker.check_gradient_direction(&p, &x, &d).unwrap();
                assert!(report.passed(), "{report:?}");
            }
        }
    }
}

#[test]
fn invalid_points_directions_and_shapes_return_errors() {
    let checker = DerivativeChecker::new();
    let p = Polynomial { wrong: false };
    for d in [
        vec![],
        vec![0., 0.],
        vec![f64::NAN, 1.],
        vec![f64::INFINITY, 0.],
    ] {
        assert!(matches!(
            checker.check_gradient_direction(&p, &vec![1., 1.], &d),
            Err(DerivativeCheckError::InvalidInput(_))
        ));
    }
    assert!(matches!(
        checker.check_jacobian(&p, &vec![f64::NAN, 1.]),
        Err(DerivativeCheckError::InvalidInput(_))
    ));
    for x in [vec![1.], vec![-1., 1.], vec![f64::INFINITY, 1.]] {
        assert!(matches!(
            checker
                .clone()
                .with_bounds(vec![0., 0.], vec![1., 1.])
                .check_gradient(&p, &x),
            Err(DerivativeCheckError::InvalidInput(_))
        ));
    }
    assert!(matches!(
        checker
            .with_bounds(vec![0., 0.], vec![1., 1.])
            .check_gradient_direction(&p, &vec![0., 0.], &vec![1., -1.]),
        Err(DerivativeCheckError::NoFeasibleDirection)
    ));
}

struct Broken {
    mode: usize,
}
impl CostFunction for Broken {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = &'static str;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        if self.mode == 0 || (self.mode == 1 && x[0] != 0.) {
            return Err("cost failed");
        }
        Ok(0.)
    }
}
impl Gradient for Broken {
    type Gradient = Vec<f64>;
    fn gradient(&self, _: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        if self.mode == 2 {
            return Err("gradient failed");
        }
        Ok(if self.mode == 3 { vec![] } else { vec![0.] })
    }
}
impl Residual for Broken {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = &'static str;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        if self.mode == 4 {
            return Err("residual failed");
        }
        if self.mode == 5 && x[0] != 0. {
            return Ok(vec![]);
        }
        Ok(vec![
            0.,
            if self.mode == 6 && x[0] != 0. {
                f64::NAN
            } else {
                0.
            },
        ])
    }
}
impl Jacobian for Broken {
    type Jacobian = DenseMatrix;
    fn jacobian(&self, _: &Vec<f64>) -> Result<DenseMatrix, Self::Error> {
        if self.mode == 7 {
            return Err("Jacobian failed");
        }
        Ok(if self.mode == 8 {
            DenseMatrix::from_row_slice(1, 2, &[0., 0.])
        } else if self.mode == 9 {
            DenseMatrix::from_row_slice(2, 1, &[0., f64::INFINITY])
        } else if self.mode == 10 {
            DenseMatrix::from_row_slice(1, 1, &[0.])
        } else {
            DenseMatrix::from_row_slice(2, 1, &[0., 0.])
        })
    }
}

#[test]
fn callback_errors_and_malformed_outputs_preserve_diagnostics() {
    let checker = DerivativeChecker::new();
    let x = vec![0.];
    for directional in [false, true] {
        let gradient = |mode| {
            if directional {
                checker.check_gradient_direction(
                    &Broken { mode },
                    &x,
                    &vec![1.],
                )
            } else {
                checker.check_gradient(&Broken { mode }, &x)
            }
        };
        let jacobian = |mode| {
            if directional {
                checker.check_jacobian_direction(
                    &Broken { mode },
                    &x,
                    &vec![1.],
                )
            } else {
                checker.check_jacobian(&Broken { mode }, &x)
            }
        };
        for mode in [0, 1] {
            assert!(matches!(
                gradient(mode),
                Err(DerivativeCheckError::Evaluation("cost failed"))
            ));
        }
        assert!(matches!(
            gradient(2),
            Err(DerivativeCheckError::Evaluation("gradient failed"))
        ));
        assert!(matches!(
            gradient(3),
            Err(DerivativeCheckError::InvalidInput(_))
        ));
        assert!(matches!(
            jacobian(4),
            Err(DerivativeCheckError::Evaluation("residual failed"))
        ));
        assert!(matches!(
            jacobian(5),
            Err(DerivativeCheckError::OutputSize {
                expected: 2,
                actual: 0
            })
        ));
        assert!(matches!(
            jacobian(6),
            Err(DerivativeCheckError::NonFiniteEvaluation { output: 1, .. })
        ));
        assert!(matches!(
            jacobian(7),
            Err(DerivativeCheckError::Evaluation("Jacobian failed"))
        ));
        assert!(matches!(
            jacobian(8),
            Err(DerivativeCheckError::InvalidInput(_))
        ));
        assert!(matches!(
            jacobian(9),
            Err(DerivativeCheckError::NonFiniteDerivative {
                source: basin::DerivativeSource::Analytic,
                output: 1,
                ..
            })
        ));
        assert!(matches!(
            jacobian(10),
            Err(DerivativeCheckError::OutputSize {
                expected: 1,
                actual: 2
            })
        ));
    }
}

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
use basin::{DenseMatrixFromFn, MatrixIndex, Scalar, VectorIndex, VectorLen};
struct Backend<V, F> {
    make: fn(&[F]) -> V,
}
impl<V: VectorIndex<F>, F: Scalar> CostFunction for Backend<V, F> {
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
impl<V: VectorIndex<F>, F: Scalar> Gradient for Backend<V, F> {
    type Gradient = V;
    fn gradient(&self, x: &V) -> Result<V, Self::Error> {
        Ok((self.make)(&[x.get_scalar(0) + x.get_scalar(0), F::one()]))
    }
}
impl<V: VectorIndex<F>, F: Scalar> Residual for Backend<V, F> {
    type Param = V;
    type Output = V;
    type Error = &'static str;
    fn residual(&self, x: &V) -> Result<V, Self::Error> {
        Ok((self.make)(&[
            self.cost(x)?,
            x.get_scalar(0),
            x.get_scalar(1),
        ]))
    }
}
impl<V: VectorIndex<F> + DenseMatrixFromFn<F>, F: Scalar> Jacobian
    for Backend<V, F>
{
    type Jacobian = V::Matrix;
    fn jacobian(&self, x: &V) -> Result<Self::Jacobian, Self::Error> {
        Ok(V::dense_from_fn(3, 2, |i, j| match (i, j) {
            (0, 0) => x.get_scalar(0) + x.get_scalar(0),
            (0, 1) | (1, 0) | (2, 1) => F::one(),
            _ => F::zero(),
        }))
    }
}
fn check_backend<V, F>(make: fn(&[F]) -> V)
where
    V: Clone + VectorLen + VectorIndex<F> + DenseMatrixFromFn<F> + Sync,
    V::Matrix: MatrixIndex<F>,
    F: Scalar + Send + Sync,
{
    for method in [basin::Method::Forward, basin::Method::Central] {
        let mut checker = DerivativeChecker::new().method(method).with_bounds(
            make(&[F::zero(), F::one()]),
            make(&[F::one(), F::one()]),
        );
        if matches!(method, basin::Method::Forward) {
            checker = checker.with_absolute_tolerance(
                F::epsilon().sqrt() * F::from_f64(10.).unwrap(),
            );
        }
        for a in [F::zero(), F::one()] {
            let x = make(&[a, F::one()]);
            let d = make(&[F::one(), F::zero()]);
            let p = Backend { make };
            for report in [
                checker.check_gradient(&p, &x).unwrap(),
                checker.check_jacobian(&p, &x).unwrap(),
                checker.check_gradient_direction(&p, &x, &d).unwrap(),
                checker.check_jacobian_direction(&p, &x, &d).unwrap(),
            ] {
                assert!(report.passed(), "{report:?}");
            }
        }
    }
}
#[test]
fn vec_f32_and_f64() {
    check_backend::<_, f64>(|x| x.to_vec());
    check_backend::<_, f32>(|x| x.to_vec());
}
#[cfg(feature = "nalgebra_all")]
#[test]
fn nalgebra_f32_and_f64() {
    check_backend::<_, f64>(
        backend_aliases::nalgebra::DVector::from_column_slice,
    );
    check_backend::<_, f32>(
        backend_aliases::nalgebra::DVector::from_column_slice,
    );
}
#[cfg(feature = "ndarray_all")]
#[test]
fn ndarray_f32_and_f64() {
    check_backend::<_, f64>(|x| {
        backend_aliases::ndarray::Array1::from_vec(x.to_vec())
    });
    check_backend::<_, f32>(|x| {
        backend_aliases::ndarray::Array1::from_vec(x.to_vec())
    });
}
#[cfg(feature = "faer_all")]
#[test]
fn faer_f32_and_f64() {
    check_backend::<_, f64>(|x| {
        backend_aliases::faer::Col::from_fn(x.len(), |i| x[i])
    });
    check_backend::<_, f32>(|x| {
        backend_aliases::faer::Col::from_fn(x.len(), |i| x[i])
    });
}

struct Counted {
    calls: std::sync::atomic::AtomicUsize,
}
impl CostFunction for Counted {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(x.iter().sum())
    }
}
impl Gradient for Counted {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(vec![1.; x.len()])
    }
}
impl Residual for Counted {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = std::convert::Infallible;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(vec![self.cost(x)?])
    }
}
impl Jacobian for Counted {
    type Jacobian = DenseMatrix;
    fn jacobian(&self, x: &Vec<f64>) -> Result<DenseMatrix, Self::Error> {
        Ok(DenseMatrix::from_row_slice(1, x.len(), &vec![1.; x.len()]))
    }
}

#[test]
fn directional_checks_use_constant_evaluations() {
    use std::sync::atomic::Ordering::Relaxed;
    for (method, probes) in
        [(basin::Method::Forward, 1), (basin::Method::Central, 2)]
    {
        let p = Counted { calls: 0.into() };
        let checker = DerivativeChecker::new().method(method);
        let x = vec![0.; 50];
        assert!(checker.check_gradient(&p, &x).unwrap().passed());
        assert_eq!(p.calls.swap(0, Relaxed), 1 + probes * 50);
        assert!(checker.check_jacobian(&p, &x).unwrap().passed());
        assert_eq!(p.calls.swap(0, Relaxed), 1 + probes * 50);
        for direction in [vec![1.; 50], vec![-1.; 50]] {
            assert!(
                checker
                    .check_gradient_direction(&p, &x, &direction)
                    .unwrap()
                    .passed()
            );
            assert_eq!(p.calls.swap(0, Relaxed), 1 + probes);
            assert!(
                checker
                    .check_jacobian_direction(&p, &x, &direction)
                    .unwrap()
                    .passed()
            );
            assert_eq!(p.calls.swap(0, Relaxed), 1 + probes);
        }
    }
}

#[test]
fn empty_and_all_fixed_checks_expose_their_lack_of_coverage() {
    let p = Counted { calls: 0.into() };
    let checker = DerivativeChecker::new();
    for report in [
        checker.check_gradient(&p, &vec![]).unwrap(),
        checker.check_jacobian(&p, &vec![]).unwrap(),
    ] {
        assert!(report.passed());
        assert!(report.comparisons.is_empty());
        assert!(report.skipped_coordinates.is_empty());
    }
    let x = vec![1., 2.];
    let checker = checker.with_bounds(x.clone(), x.clone());
    for report in [
        checker.check_gradient(&p, &x).unwrap(),
        checker.check_jacobian(&p, &x).unwrap(),
    ] {
        assert!(report.passed());
        assert!(report.comparisons.is_empty());
        assert_eq!(report.skipped_coordinates, vec![0, 1]);
    }
}

#[test]
fn finite_evaluations_with_overflowing_differences_are_identified() {
    struct Jump;
    impl CostFunction for Jump {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
            Ok(if x[0] == 0. {
                0.
            } else {
                x[0].signum() * 1e308
            })
        }
    }
    impl Gradient for Jump {
        type Gradient = Vec<f64>;
        fn gradient(&self, _: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
            Ok(vec![0.])
        }
    }
    let checker = DerivativeChecker::new();
    for result in [
        checker.check_gradient(&Jump, &vec![0.]),
        checker.check_gradient_direction(&Jump, &vec![0.], &vec![1.]),
    ] {
        assert!(matches!(
            result,
            Err(DerivativeCheckError::NonFiniteDerivative {
                source: basin::DerivativeSource::FiniteDifference,
                ..
            })
        ));
    }
}

#[test]
fn configured_stencils_agree_with_analytic_difference_formulas() {
    let p = Polynomial { wrong: false };
    let x = vec![1.5, -1.5];
    let step: f64 = 0.01;
    for (method, expected) in [
        (
            basin::Method::Forward,
            [3. + step, -3.375 + 2.25 * step - 0.5 * step.powi(2)],
        ),
        (basin::Method::Central, [3., -3.375 - 0.5 * step.powi(2)]),
    ] {
        let report = DerivativeChecker::new()
            .method(method)
            .with_step(step)
            .check_gradient(&p, &x)
            .unwrap();
        for (comparison, want) in report.comparisons.iter().zip(expected) {
            assert!((comparison.finite_difference - want).abs() < 1e-10);
        }
    }
}
