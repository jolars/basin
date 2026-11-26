use basin::{
    Bfgs, BoundedCmaEs, BoundedCmaInject, BoundedLevenbergMarquardt, BoxBounds,
    CauchyPoint, CmaEs, CmaInject, Constant, Constrained, ConstraintSet,
    CostFunction, Dogleg, Executor, FactorizePivotedQr, FiniteDiff,
    GaussNewton, Gradient, GramMatrix, Hessian, HuberLoss, Jacobian, Lbfgs,
    LevenbergMarquardt, LevenbergMarquardtQr, LinearSolveError, LinearSolveSpd,
    MaLsChCma, MatTransposeVec, MatVec, MatrixIndex, MatrixShape, Method,
    MoreSorensen, PopulationProgress, QrSolveError, RegularizedQrSolve,
    Residual, RobustLeastSquares, Scalar, SymmetricEigen, TrustRegion,
    TrustRegionReflective,
};
use rand_distr::{Distribution, StandardNormal};
use std::convert::Infallible;

fn n<F: Scalar>(x: f64) -> F {
    F::from_f64(x).unwrap()
}

struct Fit<F>(F);

impl<F: Scalar> Residual for Fit<F> {
    type Param = [F; 2];
    type Output = [F; 3];
    type Error = Infallible;

    fn residual(&self, x: &Self::Param) -> Result<Self::Output, Infallible> {
        Ok([x[0] - self.0, n::<F>(2.0) * (x[1] - n(0.5)), n(0.25)])
    }
}

impl<F: Scalar> Jacobian for Fit<F> {
    type Jacobian = [[F; 2]; 3];

    fn jacobian(&self, _: &Self::Param) -> Result<Self::Jacobian, Infallible> {
        Ok([[F::one(), F::zero()], [F::zero(), n(2.0)], [F::zero(); 2]])
    }
}

impl<F: Scalar> CostFunction for Fit<F> {
    type Param = [F; 2];
    type Output = F;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<F, Infallible> {
        Ok((x[0] - self.0).powi(2) + n::<F>(2.0) * (x[1] - n(0.5)).powi(2))
    }
}

impl<F: Scalar> Gradient for Fit<F> {
    type Gradient = [F; 2];

    fn gradient(&self, x: &Self::Param) -> Result<Self::Gradient, Infallible> {
        Ok([n::<F>(2.0) * (x[0] - self.0), n::<F>(4.0) * (x[1] - n(0.5))])
    }
}

fn check_point<F: Scalar>(actual: &[F; 2], expected: [F; 2], tolerance: F) {
    for (&actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual:?} vs {expected:?}"
        );
    }
}

type BoundedFit<F> = Constrained<Fit<F>, ConstraintSet<BoxBounds<[F; 2]>>>;

fn bounded<F: Scalar>(target: F) -> BoundedFit<F> {
    Constrained::new(
        Fit(target),
        ConstraintSet::new().with_bounds(BoxBounds::new(
            Some([n(-1.0); 2]),
            Some([F::one(); 2]),
        )),
    )
}

fn least_squares<F: Scalar + Send + Sync>() {
    let start = [F::zero(); 2];
    let target = [n(0.25), n(0.5)];
    let tolerance = n(0.002);
    macro_rules! check {
        ($problem:expr, $solver:expr, $expected:expr) => {{
            let result = Executor::from_start($problem, $solver, start)
                .max_iter(100)
                .run()
                .unwrap();
            check_point(result.param(), $expected, tolerance);
            result
        }};
    }
    check!(Fit(target[0]), GaussNewton::new_with_gram(), target);
    check!(Fit(target[0]), LevenbergMarquardt::new_with_gram(), target);
    check!(Fit(target[0]), LevenbergMarquardtQr::new(), target);
    let analytic = Fit(target[0]).jacobian(&start).unwrap();
    for method in [Method::Forward, Method::Central] {
        let finite = FiniteDiff::new_with_scalar(Fit(target[0]))
            .with_array_jacobian::<3>()
            .jacobian_method(method);
        let hessian: [[F; 2]; 2] = finite.hessian(&target).unwrap();
        check_point(&hessian[0], [n(2.0), F::zero()], tolerance);
        check_point(&hessian[1], [F::zero(), n(4.0)], tolerance);
        let mut jac = [[F::zero(); 2]; 3];
        finite.jacobian_into(&start, &mut jac).unwrap();
        for (actual, expected) in jac.iter().zip(analytic) {
            check_point(actual, expected, tolerance);
        }
        assert_eq!(finite.jacobian(&start).unwrap(), jac);
        let (residual, fused) = finite.residual_and_jacobian(&start).unwrap();
        assert_eq!(residual, Fit(target[0]).residual(&start).unwrap());
        assert_eq!(fused, jac);
        let (mut residual_out, mut jac_out) =
            ([F::zero(); 3], [[F::zero(); 2]; 3]);
        finite
            .residual_and_jacobian_into(&start, &mut residual_out, &mut jac_out)
            .unwrap();
        assert_eq!(residual_out, residual);
        assert_eq!(jac_out, jac);
        check!(finite, LevenbergMarquardtQr::new(), target);
    }
    check!(
        Fit(target[0]),
        TrustRegion::least_squares_with_gram(Dogleg),
        target
    );
    check!(
        Fit(target[0]),
        TrustRegion::least_squares_with_gram(MoreSorensen::default()),
        target
    );
    check!(
        bounded(n(2.0)),
        BoundedLevenbergMarquardt::new_with_gram(),
        [F::one(), target[1]]
    );
    check!(
        RobustLeastSquares::new(Fit(target[0]), HuberLoss).with_scale(n(0.1)),
        LevenbergMarquardtQr::new(),
        target
    );
    check!(Fit(target[0]), Bfgs::new(), target);
    let analytic_endpoint = check!(
        bounded(n(2.0)),
        TrustRegionReflective::new(),
        [F::one(), target[1]]
    );
    let finite_endpoint = check!(
        FiniteDiff::new_with_scalar(bounded(target[0]))
            .with_array_jacobian::<3>(),
        TrustRegionReflective::new(),
        target
    );
    for &value in analytic_endpoint
        .param()
        .iter()
        .chain(finite_endpoint.param())
    {
        assert!(value >= -F::one() && value <= F::one());
    }
}

fn seeded_bfgs<F: Scalar>() {
    let target = [n::<F>(0.25), n(0.5)];
    let seed = [[n(0.5), F::zero()], [F::zero(), n(0.25)]];
    let solver = || Bfgs::with_line_search(Constant(F::one()));
    macro_rules! check {
        ($solver:expr) => {{
            let result =
                Executor::from_start(Fit(target[0]), $solver, [F::zero(); 2])
                    .max_iter(1)
                    .run()
                    .unwrap();
            assert_eq!(result.param(), &target);
        }};
    }
    check!(solver().with_initial_inverse_hessian(seed));
    check!(solver().with_initial_inverse_hessian_initializer(
        |_: &[F; 2], _: &[F; 2]| seed
    ));
}

fn trust_region<F: Scalar + Send + Sync>() {
    let target = [n::<F>(0.25), n(0.5)];
    let problem = || FiniteDiff::new_with_scalar(Fit(target[0]));
    let start = [F::zero(); 2];
    macro_rules! check {
        ($solver:expr) => {{
            let result = Executor::from_start(problem(), $solver, start)
                .max_iter(100)
                .run()
                .unwrap();
            check_point(result.param(), target, n(0.002));
        }};
    }
    check!(TrustRegion::with_subproblem(basin::Steihaug::new()));
    check!(TrustRegion::with_subproblem(CauchyPoint));
    check!(TrustRegion::with_subproblem(Dogleg));
    check!(TrustRegion::with_subproblem(MoreSorensen::default()));
}

fn population<F: Scalar + Send + Sync + rand_distr::uniform::SampleUniform>()
where
    StandardNormal: Distribution<F>,
{
    let target = [n::<F>(0.25), n(0.5)];
    let cma = || CmaEs::<[F; 2], [[F; 2]; 2], F>::new(42, n(0.4));
    let bounded_cma =
        || BoundedCmaEs::<[F; 2], [[F; 2]; 2], F>::new(42, n(0.4));
    macro_rules! check {
        ($problem:expr, $solver:expr) => {{
            let result = Executor::new(
                $problem,
                $solver,
                PopulationProgress::from_point([F::zero(); 2]),
            )
            .max_iter(100)
            .run()
            .unwrap();
            check_point(result.param(), target, n(0.03));
        }};
    }
    check!(Fit(target[0]), cma());
    check!(
        bounded(target[0]),
        bounded_cma().with_eigen_update_period(3)
    );
    check!(
        Fit(target[0]),
        CmaInject::with_inner_solver(cma(), LevenbergMarquardtQr::new())
    );
    check!(
        bounded(target[0]),
        BoundedCmaInject::with_inner_solver(bounded_cma(), Lbfgs::new())
            .with_eigen_update_period(3)
    );
    check!(
        basin::SamplingDomain::new(bounded(target[0])),
        MaLsChCma::<[F; 2], [[F; 2]; 2], F>::new(42)
    );
}

#[test]
fn f32_solvers() {
    least_squares::<f32>();
    seeded_bfgs::<f32>();
    trust_region::<f32>();
    population::<f32>();
}

#[test]
fn f64_solvers() {
    least_squares::<f64>();
    seeded_bfgs::<f64>();
    trust_region::<f64>();
    population::<f64>();
    let fit = Fit(0.25);
    let point = [0.25, 0.5];
    for hessian in [
        basin::forward_difference_hessian,
        basin::central_difference_hessian,
    ] {
        let output: [[f64; 2]; 2] =
            hessian(&fit, &point, f64::EPSILON, Some(0.125)).unwrap();
        assert_eq!(output, [[2.0, 0.0], [0.0, 4.0]]);
    }
}

struct DynamicResidual<const N: usize>(usize);

impl<const N: usize> Residual for DynamicResidual<N> {
    type Param = [f64; N];
    type Output = Vec<f64>;
    type Error = Infallible;

    fn residual(&self, _: &Self::Param) -> Result<Self::Output, Self::Error> {
        Ok(vec![1.0; self.0])
    }
}

#[test]
fn array_jacobian_supports_empty_dimensions() {
    for method in [Method::Forward, Method::Central] {
        let no_params = FiniteDiff::new(DynamicResidual::<0>(2))
            .with_array_jacobian::<2>()
            .jacobian_method(method);
        assert_eq!(no_params.jacobian(&[]).unwrap(), [[], []]);

        let no_residuals = FiniteDiff::new(DynamicResidual::<2>(0))
            .with_array_jacobian::<0>()
            .jacobian_method(method);
        let expected: [[f64; 2]; 0] = [];
        assert_eq!(no_residuals.jacobian(&[1.0, 2.0]).unwrap(), expected);
    }
}

#[test]
#[should_panic(
    expected = "finite-difference residual length does not match the array Jacobian row count"
)]
fn array_jacobian_checks_the_declared_residual_length() {
    let finite =
        FiniteDiff::new(DynamicResidual::<2>(3)).with_array_jacobian::<2>();
    let _ = finite.jacobian(&[1.0, 2.0]);
}

#[test]
fn matrix_entry_access() {
    let matrix = [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
    assert_eq!(matrix.matrix_rows(), 2);
    assert_eq!(matrix.matrix_cols(), 3);
    assert_eq!(matrix.matrix_entry(0, 1), 2.0);
    assert_eq!(matrix.matrix_entry(1, 0), 4.0);
    let no_rows: [[f64; 3]; 0] = [];
    let no_columns: [[f64; 0]; 2] = [[], []];
    assert_eq!((no_rows.matrix_rows(), no_rows.matrix_cols()), (0, 3));
    assert_eq!((no_columns.matrix_rows(), no_columns.matrix_cols()), (2, 0));
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn matrix_entry_rejects_invalid_row() {
    [[1.0_f64, 2.0]].matrix_entry(1, 0);
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn matrix_entry_rejects_invalid_column() {
    [[1.0_f64, 2.0]].matrix_entry(0, 2);
}

#[test]
fn matrix_operations() {
    let a = [[1.0_f64, 2.0], [3.0, 4.0], [0.0, 1.0]];
    assert_eq!(a.matvec(&[2.0, 3.0]), [8.0, 18.0, 3.0]);
    assert_eq!(a.mat_transpose_vec(&[1.0, 2.0, 3.0]), [7.0, 13.0]);
    assert_eq!(a.gram(), [[10.0, 14.0], [14.0, 21.0]]);
    let spd = [[4.0, f64::NAN], [1.0, 3.0]];
    check_point(
        &spd.solve_spd(&[1.0, 2.0]).unwrap(),
        [1.0 / 11.0, 7.0 / 11.0],
        1e-14,
    );
    for bad in [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 2.0], [2.0, 1.0]]] {
        assert_eq!(
            bad.solve_spd(&[1.0, 2.0]),
            Err(LinearSolveError::NotPositiveDefinite)
        );
    }

    let qr = a.factorize_pivoted_qr(&[8.0, 18.0, 3.0]).unwrap();
    check_point(
        &qr.solve_regularized(0.0, &[1.0; 2], None).unwrap(),
        [2.0, 3.0],
        1e-12,
    );
    let deficient = [[1.0_f64, 2.0], [2.0, 4.0]]
        .factorize_pivoted_qr(&[1.0, 2.0])
        .unwrap();
    assert_eq!(
        deficient.solve_regularized(0.0, &[1.0; 2], None),
        Err(QrSolveError::RankDeficient)
    );
    assert!(matches!(
        a.factorize_pivoted_qr(&[0.0, f64::NAN, 0.0]),
        Err(QrSolveError::NonFinite)
    ));

    assert!([[f64::NAN]].try_eigh().is_err());
    let (vectors, values) = spd.try_eigh().unwrap();
    let symmetric = [[4.0, 1.0], [1.0, 3.0]];
    for (j, value) in values.into_iter().enumerate() {
        let column = [vectors[0][j], vectors[1][j]];
        check_point(
            &symmetric.matvec(&column),
            column.map(|v| v * value),
            1e-13,
        );
    }
    for (actual, expected) in
        vectors.gram().iter().zip([[1.0, 0.0], [0.0, 1.0]])
    {
        check_point(actual, expected, 1e-13);
    }
}
