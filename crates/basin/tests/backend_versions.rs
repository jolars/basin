//! Backend implementations must survive Cargo feature unification.

#[cfg(feature = "ndarray_v0_16")]
#[test]
fn ndarray_0_16_norm_survives_newer_features() {
    let x = ::ndarray_0_16::array![3.0_f64, 4.0];
    assert_eq!(basin::NormSquared::norm_squared(&x), 25.0);
}

#[cfg(feature = "ndarray_v0_17")]
#[test]
fn ndarray_0_17_norm() {
    let x = ndarray::array![3.0_f64, 4.0];
    assert_eq!(basin::NormSquared::norm_squared(&x), 25.0);
}

#[cfg(any(
    feature = "nalgebra_all",
    feature = "ndarray_all",
    feature = "faer_all"
))]
macro_rules! dense_checks {
    ($vector:ty, $matrix:ty, $scalar:ty, $make:expr) => {{
        use basin::{
            Bfgs, BoxConstraints, CostFunction, Executor, FactorizePivotedQr,
            Gradient, Lbfgs, LbfgsState, LinearSolveSpd, MatVec,
            MatrixIdentity, NormSquared, QuasiNewtonState, RegularizedQrSolve,
            ScaleInPlace,
        };
        use std::convert::Infallible;
        type V = $vector;
        type M = $matrix;
        type F = $scalar;
        let make = $make;
        let x = make(&[3.0, 4.0]);
        assert_eq!(NormSquared::norm_squared(&x), 25.0);
        let identity = <M as MatrixIdentity>::identity(2);
        let product = identity.matvec(&x);
        let solved = identity.solve_spd(&x).unwrap();
        let qr = identity.factorize_pivoted_qr(&x).unwrap();
        let qr_solution =
            qr.solve_regularized(0.0, &make(&[1.0, 1.0]), None).unwrap();
        for actual in [product, solved, qr_solution] {
            for i in 0..2 {
                assert!((actual[i] - x[i]).abs() < 32.0 * F::EPSILON);
            }
        }
        struct Quadratic {
            lower: V,
            upper: V,
        }
        impl CostFunction for Quadratic {
            type Param = V;
            type Output = F;
            type Error = Infallible;
            fn cost(&self, x: &V) -> Result<F, Infallible> {
                Ok(NormSquared::norm_squared(x))
            }
        }
        impl Gradient for Quadratic {
            type Gradient = V;
            fn gradient(&self, x: &V) -> Result<V, Infallible> {
                let mut gradient = x.clone();
                gradient.scale_in_place(2.0);
                Ok(gradient)
            }
        }
        impl BoxConstraints for Quadratic {
            fn lower(&self) -> &V {
                &self.lower
            }
            fn upper(&self) -> &V {
                &self.upper
            }
        }
        let problem = || Quadratic {
            lower: make(&[-5.0, -5.0]),
            upper: make(&[5.0, 5.0]),
        };
        let bfgs = Executor::new(
            problem(),
            Bfgs::<_, F>::with_line_search(basin::Wolfe::<F>::new()),
            QuasiNewtonState::<V, M, F>::new(x.clone()),
        )
        .max_iter(50)
        .run()
        .unwrap();
        assert!(bfgs.cost() < 128.0 * F::EPSILON);
        let seeded = Executor::from_start(
            problem(),
            Bfgs::<_, F>::with_line_search(basin::Wolfe::<F>::new()),
            x.clone(),
        )
        .max_iter(50)
        .run()
        .unwrap();
        assert!(seeded.cost() < 128.0 * F::EPSILON);
        let lbfgs = Executor::new(
            problem(),
            Lbfgs::<basin::solver::lbfgs::Bounded, _, F>::with_line_search(
                basin::MoreThuente::<F>::new(),
            ),
            LbfgsState::new(x, 5),
        )
        .max_iter(50)
        .run()
        .unwrap();
        assert!(lbfgs.cost() < 128.0 * F::EPSILON);
    }};
}

#[cfg(any(
    feature = "nalgebra_all",
    feature = "ndarray_all",
    feature = "faer_all"
))]
macro_rules! simplex_check {
    ($make:expr) => {{
        use basin::IntoInitialSimplex;
        let initial = ($make)(&[2.0_f64, 0.0]);
        let simplex = initial.into_initial_simplex(0.05);
        assert_eq!(simplex.len(), 3);
        assert_eq!(simplex[0][0], 2.0);
        assert_eq!(simplex[1][0], 2.1);
        assert_eq!(simplex[2][1], 0.00025);
    }};
}

#[cfg(all(
    feature = "problems",
    any(
        feature = "nalgebra_all",
        feature = "ndarray_all",
        feature = "faer_all"
    )
))]
macro_rules! corpus_checks {
    ($vector:ty, $matrix:ty, $make:expr) => {{
        use basin::problems::*;
        use basin::{
            BoxConstraints, CostFunction, Gradient, Jacobian,
            LinearEqualityConstraints, MatrixIdentity, NormSquared, Residual,
        };
        type V = $vector;
        type M = $matrix;
        let make = $make;
        let x = make(&[3.0, 4.0]);

        fn cost<P: CostFunction<Param = V, Output = f64>>() {}
        fn gradient<
            P: Gradient<Gradient = V> + CostFunction<Param = V, Output = f64>,
        >() {
        }
        fn jacobian<
            P: Jacobian<Jacobian = M> + Residual<Param = V, Output = V>,
        >() {
        }
        cost::<Ackley<V>>();
        cost::<AckleyBoxed<V>>();
        cost::<Beale<V>>();
        cost::<Booth<V>>();
        cost::<BoothBoxed<V>>();
        cost::<BoothBoxedResiduals<V>>();
        cost::<BoothResiduals<V>>();
        cost::<BukinN6<V>>();
        cost::<CrossInTray<V>>();
        cost::<Easom<V>>();
        cost::<Eggholder<V>>();
        cost::<ExponentialFit<V>>();
        cost::<GoldsteinPrice<V>>();
        cost::<Himmelblau<V>>();
        cost::<HolderTable<V>>();
        cost::<Levy<V>>();
        cost::<LevyBoxed<V>>();
        cost::<Matyas<V>>();
        cost::<McCormick<V>>();
        cost::<Picheny<V>>();
        cost::<PowellSingular<V>>();
        cost::<Rastrigin<V>>();
        cost::<RastriginBoxed<V>>();
        cost::<Rosenbrock<V>>();
        cost::<RosenbrockResiduals<V>>();
        cost::<SchafferN2<V>>();
        cost::<SchafferN4<V>>();
        cost::<Sphere<V>>();
        cost::<SphereBoxed<V>>();
        cost::<Step<V>>();
        cost::<StyblinskiTang<V>>();
        cost::<StyblinskiTangBoxed<V>>();
        cost::<ThreeHumpCamel<V>>();
        cost::<Zero<V>>();
        gradient::<Beale<V>>();
        gradient::<Booth<V>>();
        gradient::<BoothBoxed<V>>();
        gradient::<Easom<V>>();
        gradient::<GoldsteinPrice<V>>();
        gradient::<Himmelblau<V>>();
        gradient::<Levy<V>>();
        gradient::<LevyBoxed<V>>();
        gradient::<Matyas<V>>();
        gradient::<McCormick<V>>();
        gradient::<Picheny<V>>();
        gradient::<Rosenbrock<V>>();
        gradient::<SchafferN2<V>>();
        gradient::<Sphere<V>>();
        gradient::<StyblinskiTang<V>>();
        gradient::<StyblinskiTangBoxed<V>>();
        gradient::<ThreeHumpCamel<V>>();
        gradient::<Zero<V>>();
        jacobian::<BoothBoxedResiduals<V>>();
        jacobian::<BoothResiduals<V>>();
        jacobian::<PowellSingular<V>>();
        jacobian::<RosenbrockResiduals<V>>();
        let sphere = Sphere::<V>::new();
        assert_eq!(sphere.cost(&x).unwrap(), 25.0);
        assert_eq!(
            NormSquared::norm_squared(&sphere.gradient(&x).unwrap()),
            100.0
        );
        let boxed =
            SphereBoxed::<V>::new(make(&[-5.0, -5.0]), make(&[5.0, 5.0]));
        assert_eq!(boxed.lower()[0], -5.0);
        assert_eq!(boxed.upper()[1], 5.0);
        let residuals = RosenbrockResiduals::<V>::new();
        let at_minimum = make(&[1.0, 1.0]);
        assert_eq!(
            NormSquared::norm_squared(
                &residuals.residual(&at_minimum).unwrap()
            ),
            0.0
        );
        let jacobian = residuals.jacobian(&at_minimum).unwrap();
        assert_eq!(basin::MatrixIndex::matrix_entry(&jacobian, 0, 0), -20.0);
        let equality = EqualityConstrainedQuadratic::new(
            make(&[0.0, 0.0]),
            <M as MatrixIdentity>::identity(2),
            make(&[1.0, 1.0]),
        );
        assert_eq!(equality.b()[0], 1.0);
        assert_eq!(equality.cost(&x).unwrap(), 25.0);
    }};
}

#[cfg(all(
    feature = "problems",
    any(feature = "nalgebra_all", feature = "faer_all")
))]
macro_rules! sparse_problem_checks {
    (f32, $a:expr, $make:expr) => {};
    (f64, $a:expr, $make:expr) => {{
        use basin::problems::{SparseLeastSquares, SparseLeastSquaresBoxed};
        use basin::{
            BoxConstraints, Executor, GaussNewton, Jacobian, NllsState,
            Residual,
        };
        let a = $a;
        let make = $make;
        let target = make(&[1.0, 2.0]);
        let b = basin::MatVec::matvec(&a, &target);
        let bounded = SparseLeastSquaresBoxed::new(
            a.clone(),
            b.clone(),
            make(&[-5.0, -5.0]),
            make(&[5.0, 5.0]),
        );
        assert_eq!(bounded.lower()[0], -5.0);
        assert_eq!(bounded.upper()[1], 5.0);
        assert_eq!(
            basin::NormSquared::norm_squared(
                &bounded.residual(&target).unwrap()
            ),
            0.0
        );
        let jacobian = bounded.jacobian(&target).unwrap();
        let product = basin::MatVec::matvec(&jacobian, &target);
        assert_eq!(product[0], b[0]);
        let result = Executor::new(
            SparseLeastSquares::new(a, b),
            GaussNewton::new(),
            NllsState::new(make(&[0.0, 0.0])),
        )
        .max_iter(10)
        .run()
        .unwrap();
        assert!(result.cost() < 1e-10);
    }};
}
#[cfg(any(feature = "nalgebra_all", feature = "faer_all"))]
macro_rules! sparse_checks {
    ($scalar:ident, $make_vector:expr, $make_matrix:expr) => {{
        use basin::{GramMatrix, LinearSolveSpd, MatVec};
        type F = $scalar;
        let make = $make_vector;
        let a = ($make_matrix)();
        let b = make(&[1.0, 2.0]);
        let x = a.solve_spd(&b).unwrap();
        assert!((x[0] - 1.0 / 11.0).abs() < 32.0 * F::EPSILON);
        assert!((x[1] - 7.0 / 11.0).abs() < 32.0 * F::EPSILON);
        let product = a.matvec(&x);
        let gram_product = a.gram().matvec(&x);
        let twice = a.matvec(&product);
        #[cfg(feature = "problems")]
        sparse_problem_checks!($scalar, a, make);
        for i in 0..2 {
            assert!((product[i] - b[i]).abs() < 64.0 * F::EPSILON);
            assert!((gram_product[i] - twice[i]).abs() < 256.0 * F::EPSILON);
        }
    }};
}
#[cfg(feature = "nalgebra_all")]
macro_rules! nalgebra_checks {
    ($module:ident, $dependency:ident, $sparse:ident) => {
        mod $module {
            use $dependency as backend;
            #[test]
            fn f32_dense_and_solvers() {
                dense_checks!(
                    backend::DVector<f32>,
                    backend::DMatrix<f32>,
                    f32,
                    |x: &[f32]| backend::DVector::from_column_slice(x)
                );
            }
            #[test]
            fn f64_dense_and_solvers() {
                dense_checks!(
                    backend::DVector<f64>,
                    backend::DMatrix<f64>,
                    f64,
                    |x: &[f64]| backend::DVector::from_column_slice(x)
                );
            }
            #[test]
            fn simplex() {
                simplex_check!(
                    |x: &[f64]| backend::DVector::from_column_slice(x)
                );
            }
            #[cfg(feature = "problems")]
            #[test]
            fn corpus() {
                corpus_checks!(
                    backend::DVector<f64>,
                    backend::DMatrix<f64>,
                    |x: &[f64]| backend::DVector::from_column_slice(x)
                );
            }
            #[test]
            fn f32_sparse() {
                sparse_checks!(
                    f32,
                    |x: &[f32]| backend::DVector::from_column_slice(x),
                    || {
                        let mut coo = $sparse::CooMatrix::<f32>::new(2, 2);
                        coo.push(0, 0, 4.0);
                        coo.push(0, 1, 1.0);
                        coo.push(1, 0, 1.0);
                        coo.push(1, 1, 3.0);
                        $sparse::CscMatrix::from(&coo)
                    }
                );
            }

            #[test]
            fn f64_sparse() {
                sparse_checks!(
                    f64,
                    |x: &[f64]| backend::DVector::from_column_slice(x),
                    || {
                        let mut coo = $sparse::CooMatrix::<f64>::new(2, 2);
                        coo.push(0, 0, 4.0);
                        coo.push(0, 1, 1.0);
                        coo.push(1, 0, 1.0);
                        coo.push(1, 1, 3.0);
                        $sparse::CscMatrix::from(&coo)
                    }
                );
            }
        }
    };
}

#[cfg(feature = "nalgebra_v0_32")]
nalgebra_checks!(nalgebra_0_32, nalgebra_0_32, nalgebra_sparse_0_9);
#[cfg(feature = "nalgebra_v0_33")]
nalgebra_checks!(nalgebra_0_33, nalgebra_0_33, nalgebra_sparse_0_10);
#[cfg(feature = "nalgebra_v0_34")]
nalgebra_checks!(nalgebra_0_34, nalgebra_0_34, nalgebra_sparse_0_11);
#[cfg(feature = "nalgebra_v0_35")]
nalgebra_checks!(nalgebra_0_35, nalgebra, nalgebra_sparse);

#[cfg(feature = "ndarray_all")]
macro_rules! ndarray_checks {
    ($module:ident, $dependency:ident) => {
        mod $module {
            use $dependency as backend;
            #[test]
            fn f32_dense_and_solvers() {
                dense_checks!(
                    backend::Array1<f32>,
                    backend::Array2<f32>,
                    f32,
                    |x: &[f32]| backend::Array1::from_vec(x.to_vec())
                );
            }
            #[test]
            fn f64_dense_and_solvers() {
                dense_checks!(
                    backend::Array1<f64>,
                    backend::Array2<f64>,
                    f64,
                    |x: &[f64]| backend::Array1::from_vec(x.to_vec())
                );
            }
            #[test]
            fn simplex() {
                simplex_check!(|x: &[f64]| backend::Array1::from_vec(
                    x.to_vec()
                ));
            }
            #[cfg(feature = "problems")]
            #[test]
            fn corpus() {
                corpus_checks!(
                    backend::Array1<f64>,
                    backend::Array2<f64>,
                    |x: &[f64]| backend::Array1::from_vec(x.to_vec())
                );
            }
        }
    };
}

#[cfg(feature = "ndarray_v0_15")]
ndarray_checks!(ndarray_0_15, ndarray_0_15);
#[cfg(feature = "ndarray_v0_16")]
ndarray_checks!(ndarray_0_16, ndarray_0_16);
#[cfg(feature = "ndarray_v0_17")]
ndarray_checks!(ndarray_0_17, ndarray);

#[cfg(feature = "faer_all")]
macro_rules! faer_checks {
    ($module:ident, $dependency:ident) => {
        mod $module {
            use $dependency as backend;
            #[test]
            fn f32_dense_and_solvers() {
                dense_checks!(
                    backend::Col<f32>,
                    backend::Mat<f32>,
                    f32,
                    |x: &[f32]| backend::Col::from_fn(x.len(), |i| x[i])
                );
            }
            #[test]
            fn f64_dense_and_solvers() {
                dense_checks!(
                    backend::Col<f64>,
                    backend::Mat<f64>,
                    f64,
                    |x: &[f64]| backend::Col::from_fn(x.len(), |i| x[i])
                );
            }
            #[test]
            fn simplex() {
                simplex_check!(|x: &[f64]| backend::Col::from_fn(
                    x.len(),
                    |i| x[i]
                ));
            }
            #[cfg(feature = "problems")]
            #[test]
            fn corpus() {
                corpus_checks!(
                    backend::Col<f64>,
                    backend::Mat<f64>,
                    |x: &[f64]| { backend::Col::from_fn(x.len(), |i| x[i]) }
                );
            }
            #[test]
            fn f32_sparse() {
                sparse_checks!(
                    f32,
                    |x: &[f32]| backend::Col::from_fn(x.len(), |i| x[i]),
                    || {
                        use backend::sparse::{SparseColMat, Triplet};
                        SparseColMat::<usize, f32>::try_new_from_triplets(
                            2,
                            2,
                            &[
                                Triplet::new(0, 0, 4.0),
                                Triplet::new(0, 1, 1.0),
                                Triplet::new(1, 0, 1.0),
                                Triplet::new(1, 1, 3.0),
                            ],
                        )
                        .unwrap()
                    }
                );
            }

            #[test]
            fn f64_sparse() {
                sparse_checks!(
                    f64,
                    |x: &[f64]| backend::Col::from_fn(x.len(), |i| x[i]),
                    || {
                        use backend::sparse::{SparseColMat, Triplet};
                        SparseColMat::<usize, f64>::try_new_from_triplets(
                            2,
                            2,
                            &[
                                Triplet::new(0, 0, 4.0),
                                Triplet::new(0, 1, 1.0),
                                Triplet::new(1, 0, 1.0),
                                Triplet::new(1, 1, 3.0),
                            ],
                        )
                        .unwrap()
                    }
                );
            }
        }
    };
}

#[cfg(feature = "faer_v0_22")]
faer_checks!(faer_0_22, faer_0_22);
#[cfg(feature = "faer_v0_23")]
faer_checks!(faer_0_23, faer_0_23);
#[cfg(feature = "faer_v0_24")]
faer_checks!(faer_0_24, faer);
