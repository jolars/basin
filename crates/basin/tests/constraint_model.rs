//! Constraint declaration, checked evaluation, and independent storage regressions.
use basin::{
    Bobyqa, BoxBounds, Cobyla, Constrained, ConstraintAxis, ConstraintBlock,
    ConstraintDescription, ConstraintError, ConstraintSolveError, Constraints,
    CostFunction, DenseMatrix, Executor, Gradient, MatrixFinite, MatrixIndex,
    MatrixShape, NoConstraints, NoNonlinearBlock, NonlinearBlock,
    NonlinearBlocks, NonlinearInequalityValues, PointState, Problem,
    ProblemConstraints, Scalar, SelectedState, Slsqp, Solver, VectorLen,
};
use std::{
    cell::Cell,
    convert::Infallible,
    marker::PhantomData,
    ops::{Index, IndexMut},
};

#[path = "support/backend_aliases.rs"]
mod backend_aliases;

struct Disk<V, F>(PhantomData<fn() -> (V, F)>);
impl<V, F> ProblemConstraints for Disk<V, F> {
    type Constraints = Constraints<
        basin::NoBounds,
        basin::NoLinearConstraints,
        NonlinearBlocks<NoNonlinearBlock, NonlinearBlock>,
    >;
    fn constraint_set(
        &self,
    ) -> <Self::Constraints as ConstraintDescription>::Borrowed<'_> {
        Constraints::new().with_nonlinear_inequalities(1)
    }
}
impl<V: Index<usize, Output = F>, F: Scalar> CostFunction for Disk<V, F> {
    type Param = V;
    type Output = F;
    type Error = Infallible;
    fn cost(&self, x: &V) -> Result<F, Infallible> {
        Ok(x[0] * x[1])
    }
}
impl<V: Index<usize, Output = F>, F: Scalar> NonlinearInequalityValues
    for Disk<V, F>
{
    type Param = V;
    type Values = Vec<F>;
    type Error = Infallible;
    fn nonlinear_inequalities(
        &self,
        x: &V,
    ) -> Result<Vec<F>, ConstraintSolveError<Infallible>> {
        Ok(vec![x[0] * x[0] + x[1] * x[1] - F::one()])
    }
}
fn disk<V, F>(make: impl Fn(&[F]) -> V)
where
    V: Clone + VectorLen + IndexMut<usize, Output = F>,
    F: Scalar,
{
    let result = Executor::from_start(
        Disk::<V, F>(PhantomData),
        Cobyla::new().with_final_radius(F::epsilon().sqrt()),
        make(&[F::one(), F::one()]),
    )
    .max_iter(1000)
    .run()
    .unwrap();
    let tolerance = F::from_f64(128.0).unwrap() * F::epsilon().sqrt();
    assert!((result.best_cost() + F::from_f64(0.5).unwrap()).abs() < tolerance);
    assert_eq!(result.counts.cost_evals, result.counts.residual_evals);
}
#[test]
fn independent_vec_values_in_both_precisions() {
    disk::<Vec<f64>, f64>(|x| x.to_vec());
    disk::<Vec<f32>, f32>(|x| x.to_vec());
}
#[cfg(feature = "nalgebra_all")]
#[test]
fn independent_values_with_nalgebra() {
    use backend_aliases::nalgebra::DVector;
    disk::<DVector<f64>, f64>(DVector::from_column_slice);
    disk::<DVector<f32>, f32>(DVector::from_column_slice);
}
#[cfg(feature = "ndarray_all")]
#[test]
fn independent_values_with_ndarray() {
    use backend_aliases::ndarray::Array1;
    disk::<Array1<f64>, f64>(|x| Array1::from_vec(x.to_vec()));
    disk::<Array1<f32>, f32>(|x| Array1::from_vec(x.to_vec()));
}
#[cfg(feature = "faer_all")]
#[test]
fn independent_values_with_faer() {
    use backend_aliases::faer::Col;
    disk::<Col<f64>, f64>(|x| Col::from_fn(x.len(), |i| x[i]));
    disk::<Col<f32>, f32>(|x| Col::from_fn(x.len(), |i| x[i]));
}

#[derive(Clone)]
struct Fixed([f64; 2]);
impl VectorLen for Fixed {
    fn vec_len(&self) -> usize {
        2
    }
}
impl Index<usize> for Fixed {
    type Output = f64;
    fn index(&self, i: usize) -> &f64 {
        &self.0[i]
    }
}
impl IndexMut<usize> for Fixed {
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.0[i]
    }
}
#[test]
fn fixed_parameter_needs_no_matrix_or_constraint_backend() {
    disk::<Fixed, f64>(|x| Fixed([x[0], x[1]]));
}

struct BoundsProbe {
    lower: Vec<f64>,
    upper: Vec<f64>,
    calls: Cell<usize>,
}
impl ProblemConstraints for BoundsProbe {
    type Constraints = Constraints<BoxBounds<Vec<f64>>>;
    fn constraint_set(
        &self,
    ) -> <Self::Constraints as ConstraintDescription>::Borrowed<'_> {
        Constraints::new()
            .with_bounds(BoxBounds::new(Some(&self.lower), Some(&self.upper)))
    }
}
impl CostFunction for BoundsProbe {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        self.calls.set(self.calls.get() + 1);
        Ok(x.iter().map(|v| v * v).sum())
    }
}
#[test]
fn malformed_bounds_abort_before_callbacks() {
    use basin::BoundSide::{Lower, Upper};
    let cases = [
        (
            vec![f64::NAN],
            vec![1.0],
            ConstraintError::NaNBound {
                side: Lower,
                coordinate: 0,
            },
        ),
        (
            vec![0.0],
            vec![f64::NAN],
            ConstraintError::NaNBound {
                side: Upper,
                coordinate: 0,
            },
        ),
        (
            vec![f64::INFINITY],
            vec![1.0],
            ConstraintError::InvalidInfinity {
                side: Lower,
                coordinate: 0,
            },
        ),
        (
            vec![0.0],
            vec![f64::NEG_INFINITY],
            ConstraintError::InvalidInfinity {
                side: Upper,
                coordinate: 0,
            },
        ),
        (
            vec![2.0],
            vec![1.0],
            ConstraintError::InvertedBounds { coordinate: 0 },
        ),
        (
            vec![],
            vec![1.0],
            ConstraintError::DimensionMismatch {
                block: ConstraintBlock::Bounds,
                axis: ConstraintAxis::Length,
                expected: 1,
                actual: 0,
            },
        ),
    ];
    for (lower, upper, expected) in cases {
        let mut p = Problem::new(BoundsProbe {
            lower,
            upper,
            calls: Cell::new(0),
        });
        let error = Cobyla::new()
            .init(&mut p, &mut SelectedState::new(vec![0.0]))
            .unwrap_err();
        assert_eq!(
            error,
            basin::EvaluationError::Evaluation(
                ConstraintSolveError::InvalidConstraints(expected.clone())
            )
        );
        assert_eq!(p.inner().calls.get(), 0);
        assert_eq!(p.counts().total_work(), 0);
        let error = Bobyqa::new()
            .init(&mut p, &mut PointState::new(vec![0.0]))
            .unwrap_err();
        assert_eq!(
            error,
            basin::EvaluationError::Evaluation(
                ConstraintSolveError::InvalidConstraints(expected)
            )
        );
        assert_eq!(p.inner().calls.get(), 0);
        assert_eq!(p.counts().total_work(), 0);
    }
}
#[test]
fn valid_infinities_and_fixed_coordinates_are_accepted() {
    let p = BoundsProbe {
        lower: vec![1.0, f64::NEG_INFINITY],
        upper: vec![1.0, f64::INFINITY],
        calls: Cell::new(0),
    };
    let result = Executor::from_start(p, Cobyla::new(), vec![1.0, 0.0])
        .max_iter(100)
        .run()
        .unwrap();
    assert!((result.best_param()[0] - 1.0).abs() < 1e-6);
}

#[derive(Debug, PartialEq, Eq)]
struct CallbackError(usize);
struct OutputProbe {
    count: Cell<usize>,
    length: usize,
    fail: bool,
    calls: Cell<usize>,
}
impl ProblemConstraints for OutputProbe {
    type Constraints = Constraints<
        basin::NoBounds,
        basin::NoLinearConstraints,
        NonlinearBlocks<NoNonlinearBlock, NonlinearBlock>,
    >;
    fn constraint_set(
        &self,
    ) -> <Self::Constraints as ConstraintDescription>::Borrowed<'_> {
        Constraints::new().with_nonlinear_inequalities(self.count.get())
    }
}
impl CostFunction for OutputProbe {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = CallbackError;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, CallbackError> {
        Ok(x[0] * x[0])
    }
}
impl NonlinearInequalityValues for OutputProbe {
    type Param = Vec<f64>;
    type Values = Vec<f64>;
    type Error = CallbackError;
    fn nonlinear_inequalities(
        &self,
        _: &Vec<f64>,
    ) -> Result<Vec<f64>, ConstraintSolveError<CallbackError>> {
        self.calls.set(self.calls.get() + 1);
        if self.fail {
            return Err(CallbackError(37).into());
        }
        Ok(vec![-1.0; self.length])
    }
}
#[test]
fn malformed_output_and_callback_errors_preserve_attempt_counts() {
    for fail in [false, true] {
        let mut p = Problem::new(OutputProbe {
            count: Cell::new(1),
            length: 2,
            fail,
            calls: Cell::new(0),
        });
        let e = Cobyla::new()
            .init(&mut p, &mut SelectedState::new(vec![1.0]))
            .unwrap_err();
        if fail {
            assert_eq!(
                e,
                basin::EvaluationError::Evaluation(
                    ConstraintSolveError::Evaluation(CallbackError(37))
                )
            )
        } else {
            assert_eq!(
                e,
                basin::EvaluationError::Evaluation(
                    ConstraintSolveError::InvalidConstraints(
                        ConstraintError::DimensionMismatch {
                            block: ConstraintBlock::NonlinearInequalities,
                            axis: ConstraintAxis::Length,
                            expected: 1,
                            actual: 2
                        }
                    )
                )
            );
        }
        assert_eq!(p.counts().cost_evals, 1);
        assert_eq!(p.counts().residual_evals, 1);
        assert_eq!(p.inner().calls.get(), 1);
    }
}
#[test]
fn zero_count_blocks_never_invoke_callbacks() {
    let mut p = Problem::new(OutputProbe {
        count: Cell::new(0),
        length: 99,
        fail: true,
        calls: Cell::new(0),
    });
    Cobyla::new()
        .init(&mut p, &mut SelectedState::new(vec![1.0]))
        .unwrap();
    assert_eq!(p.inner().calls.get(), 0);
    assert_eq!(p.counts().residual_evals, 0);
}
#[test]
fn a_changed_declaration_is_a_typed_error() {
    let mut p = Problem::new(OutputProbe {
        count: Cell::new(1),
        length: 1,
        fail: false,
        calls: Cell::new(0),
    });
    let mut solver = Cobyla::new();
    let mut state = SelectedState::new(vec![1.0]);
    solver.init(&mut p, &mut state).unwrap();
    p.inner().count.set(2);
    assert!(matches!(
        solver.next_iter(&mut p, &mut state),
        Err(basin::EvaluationError::Evaluation(
            ConstraintSolveError::InvalidConstraints(
                ConstraintError::LayoutChanged
            )
        ))
    ));
}

struct Quadratic;
impl ProblemConstraints for Quadratic {
    type Constraints = NoConstraints;
    fn constraint_set(&self) -> NoConstraints {
        NoConstraints::EMPTY
    }
}
impl CostFunction for Quadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(x.iter().map(|v| v * v).sum())
    }
}
impl Gradient for Quadratic {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(x.iter().map(|v| 2.0 * v).collect())
    }
}
struct InequalityMatrix;
impl MatrixShape for InequalityMatrix {
    fn matrix_rows(&self) -> usize {
        1
    }
    fn matrix_cols(&self) -> usize {
        2
    }
}
impl MatrixFinite for InequalityMatrix {
    fn matrix_is_finite(&self) -> bool {
        true
    }
}
impl MatrixIndex for InequalityMatrix {
    fn matrix_entry(&self, _: usize, j: usize) -> f64 {
        if j == 0 { 1.0 } else { 0.0 }
    }
}
#[test]
fn linear_only_slsqp_uses_independent_matrices_without_a_nonlinear_jacobian() {
    let constraints = Constraints::new()
        .with_linear_equalities(basin::LinearBlock::new(
            DenseMatrix::from_row_slice(1, 2, &[1.0, 1.0]),
            vec![1.0],
        ))
        .with_linear_inequalities(basin::LinearBlock::new(
            InequalityMatrix,
            vec![0.25],
        ));
    let result = Executor::from_start(
        Constrained::new(Quadratic, constraints),
        Slsqp::new(),
        vec![0.0, 0.0],
    )
    .max_iter(100)
    .run()
    .unwrap();
    assert!((result.param()[0] - 0.25).abs() < 1e-6);
    assert!((result.param()[1] - 0.75).abs() < 1e-6);
    assert_eq!(result.counts.residual_evals, 0);
    assert_eq!(result.counts.jacobian_evals, 0);
}
#[test]
fn malformed_linear_shapes_are_errors_before_matrix_multiplication() {
    let constraints =
        Constraints::new().with_linear_inequalities(basin::LinearBlock::new(
            DenseMatrix::from_row_slice(1, 3, &[1.0; 3]),
            vec![0.0],
        ));
    let mut p = Problem::new(Constrained::new(Quadratic, constraints));
    let e = Cobyla::new()
        .init(&mut p, &mut SelectedState::new(vec![0.0; 2]))
        .unwrap_err();
    assert!(matches!(
        e,
        basin::EvaluationError::Evaluation(
            ConstraintSolveError::InvalidConstraints(
                ConstraintError::DimensionMismatch {
                    axis: ConstraintAxis::Columns,
                    ..
                }
            )
        )
    ));
    assert_eq!(p.counts().total_work(), 0);
}
#[test]
fn differentiation_bounds_use_the_same_strict_sentinel_policy() {
    let e =
        basin::BoundedFiniteDiff::try_new(Quadratic, vec![f64::NAN], vec![1.0]);
    assert!(matches!(e, Err(ConstraintError::NaNBound { .. })));
}

fn independent_products<V, M, F>(
    make: impl Fn(&[F]) -> V,
    matrix: impl Fn(&[F]) -> M,
) where
    V: VectorLen + Index<usize, Output = F>,
    M: MatrixShape
        + MatrixFinite
        + basin::MatVecInto<V, F>
        + basin::MatTransposeVecInto<V, F>,
    F: Scalar,
{
    let num = |x| F::from_f64(x).unwrap();
    let coefficients = [1.0, 2.0, 0.0, -3.0, 0.0, 4.0].map(num);
    let a = matrix(&coefficients);
    assert_eq!((a.matrix_rows(), a.matrix_cols()), (2, 3));
    assert!(a.matrix_is_finite());
    let mut values = [F::nan(); 2];
    a.matvec_into(&make(&[num(2.0), num(-1.0), num(3.0)]), &mut values);
    assert_eq!(values, [F::zero(), num(6.0)]);
    let mut param = make(&[F::nan(); 3]);
    a.mat_transpose_vec_into(&[num(2.0), num(-1.0)], &mut param);
    assert_eq!(
        [param[0], param[1], param[2]],
        [num(5.0), num(4.0), num(-4.0)]
    );
    let mut invalid = coefficients;
    invalid[1] = F::nan();
    assert!(!matrix(&invalid).matrix_is_finite());
}
macro_rules! products {
    ($module:ident,$vector:expr,$matrix:expr) => {
        mod $module {
            #[test]
            fn f64() {
                super::independent_products::<_, _, f64>($vector, $matrix);
            }
            #[test]
            fn f32() {
                super::independent_products::<_, _, f32>($vector, $matrix);
            }
        }
    };
}
products!(vec_products, |x| x.to_vec(), |a| {
    basin::DenseMatrix::from_row_slice(2, 3, a)
});
#[cfg(feature = "nalgebra_v0_32")]
products!(
    nalgebra_v0_32_products,
    ::nalgebra_0_32::DVector::from_column_slice,
    |a| ::nalgebra_0_32::DMatrix::from_row_slice(2, 3, a)
);
#[cfg(feature = "nalgebra_v0_33")]
products!(
    nalgebra_v0_33_products,
    ::nalgebra_0_33::DVector::from_column_slice,
    |a| ::nalgebra_0_33::DMatrix::from_row_slice(2, 3, a)
);
#[cfg(feature = "nalgebra_v0_34")]
products!(
    nalgebra_v0_34_products,
    ::nalgebra_0_34::DVector::from_column_slice,
    |a| ::nalgebra_0_34::DMatrix::from_row_slice(2, 3, a)
);
#[cfg(feature = "nalgebra_v0_35")]
products!(
    nalgebra_v0_35_products,
    ::nalgebra::DVector::from_column_slice,
    |a| ::nalgebra::DMatrix::from_row_slice(2, 3, a)
);
#[cfg(feature = "ndarray_v0_15")]
products!(
    ndarray_v0_15_products,
    |x| ::ndarray_0_15::Array1::from_vec(x.to_vec()),
    |a| ::ndarray_0_15::Array2::from_shape_vec((2, 3), a.to_vec()).unwrap()
);
#[cfg(feature = "ndarray_v0_16")]
products!(
    ndarray_v0_16_products,
    |x| ::ndarray_0_16::Array1::from_vec(x.to_vec()),
    |a| ::ndarray_0_16::Array2::from_shape_vec((2, 3), a.to_vec()).unwrap()
);
#[cfg(feature = "ndarray_v0_17")]
products!(
    ndarray_v0_17_products,
    |x| ::ndarray::Array1::from_vec(x.to_vec()),
    |a| ::ndarray::Array2::from_shape_vec((2, 3), a.to_vec()).unwrap()
);
#[cfg(feature = "faer_v0_22")]
products!(
    faer_v0_22_products,
    |x| ::faer_0_22::Col::from_fn(x.len(), |i| x[i]),
    |a| ::faer_0_22::Mat::from_fn(2, 3, |i, j| a[3 * i + j])
);
#[cfg(feature = "faer_v0_23")]
products!(
    faer_v0_23_products,
    |x| ::faer_0_23::Col::from_fn(x.len(), |i| x[i]),
    |a| ::faer_0_23::Mat::from_fn(2, 3, |i, j| a[3 * i + j])
);
#[cfg(feature = "faer_v0_24")]
products!(
    faer_v0_24_products,
    |x| ::faer::Col::from_fn(x.len(), |i| x[i]),
    |a| ::faer::Mat::from_fn(2, 3, |i, j| a[3 * i + j])
);

#[cfg(feature = "nalgebra_all")]
products!(nalgebra_csc_products, |x| x.to_vec(), |a| {
    use crate::backend_aliases::nalgebra_sparse::{CooMatrix, CscMatrix};
    let mut coo = CooMatrix::new(2, 3);
    for i in 0..2 {
        for j in 0..3 {
            if a[3 * i + j] != 0.0 {
                coo.push(i, j, a[3 * i + j]);
            }
        }
    }
    CscMatrix::from(&coo)
});
#[cfg(feature = "faer_all")]
products!(faer_csc_products, |x| x.to_vec(), |a| {
    use crate::backend_aliases::faer::sparse::{SparseColMat, Triplet};
    let triplets: Vec<_> = (0..2)
        .flat_map(|i| (0..3).map(move |j| Triplet::new(i, j, a[3 * i + j])))
        .collect();
    SparseColMat::<usize, _>::try_new_from_triplets(2, 3, &triplets).unwrap()
});
#[test]
fn empty_linear_transpose_overwrites_its_destination() {
    use basin::MatTransposeVecInto;
    let a = DenseMatrix::<f64>::from_row_slice(0, 3, &[]);
    let mut out = vec![f64::NAN; 3];
    a.mat_transpose_vec_into(&[], &mut out);
    assert_eq!(out, vec![0.0; 3]);
}

#[test]
fn linear_storage_does_not_require_a_parameter_backend() {
    struct Objective;
    impl ProblemConstraints for Objective {
        type Constraints = NoConstraints;
        fn constraint_set(&self) -> NoConstraints {
            NoConstraints::EMPTY
        }
    }
    impl CostFunction for Objective {
        type Param = Fixed;
        type Output = f64;
        type Error = Infallible;
        fn cost(&self, x: &Fixed) -> Result<f64, Infallible> {
            Ok(x[0] * x[0] + x[1] * x[1])
        }
    }
    let problem = || {
        Constrained::new(
            Objective,
            Constraints::new().with_linear_equalities(basin::LinearBlock::new(
                DenseMatrix::from_row_slice(1, 2, &[1.0, 1.0]),
                vec![1.0],
            )),
        )
    };
    let cobyla =
        Executor::from_start(problem(), Cobyla::new(), Fixed([1.0, 0.0]))
            .max_iter(1000)
            .run()
            .unwrap();
    let lincoa = Executor::from_start(
        problem(),
        basin::Lincoa::new(),
        Fixed([1.0, 0.0]),
    )
    .max_iter(1000)
    .run()
    .unwrap();
    for point in [cobyla.param(), lincoa.param()] {
        assert!((point[0] - 0.5).abs() < 1e-4);
        assert!((point[1] - 0.5).abs() < 1e-4);
    }
    assert_eq!(cobyla.counts.residual_evals, 0);
    assert_eq!(lincoa.counts.residual_evals, 0);
}

#[test]
fn absent_runtime_linear_blocks_have_zero_gradient_contributions() {
    let equalities =
        Constraints::new()
            .with_linear_equalities(
                basin::LinearBlock::<DenseMatrix, Vec<f64>>::optional(None),
            );
    let problem = Constrained::new(Quadratic, equalities);
    let multipliers = vec![];
    let relaxed = basin::AugmentedLagrangian::new(&problem, &multipliers, 1.0);
    let x = vec![1.0, 2.0];
    assert_eq!(relaxed.cost(&x).unwrap(), 5.0);
    assert_eq!(relaxed.gradient(&x).unwrap(), vec![2.0, 4.0]);

    let inequalities =
        Constraints::new().with_linear_inequalities(basin::LinearBlock::<
            DenseMatrix,
            Vec<f64>,
        >::optional(None));
    let problem = Constrained::new(Quadratic, inequalities);
    let relaxed = basin::LogBarrier::new(&problem, 1.0);
    assert_eq!(relaxed.cost(&x).unwrap(), 5.0);
    assert_eq!(relaxed.gradient(&x).unwrap(), vec![2.0, 4.0]);
}

#[cfg(feature = "serde")]
mod serialization {
    use basin::{DenseMatrix, LinearBlock, core::constraint::BorrowConstraint};

    type Block = LinearBlock<DenseMatrix, Vec<f64>>;

    #[test]
    fn partially_populated_linear_blocks_are_rejected() {
        #[derive(serde::Serialize)]
        struct StoredLinearBlock {
            matrix: Option<DenseMatrix>,
            rhs: Option<Vec<f64>>,
        }

        let malformed = [
            StoredLinearBlock {
                matrix: None,
                rhs: Some(vec![1.0]),
            },
            StoredLinearBlock {
                matrix: Some(DenseMatrix::from_row_slice(1, 1, &[1.0])),
                rhs: None,
            },
        ];
        for stored in malformed {
            let bytes = postcard::to_allocvec(&stored).unwrap();
            assert!(postcard::from_bytes::<Block>(&bytes).is_err());
        }
    }

    #[test]
    fn complete_and_absent_linear_blocks_round_trip() {
        let blocks = [
            Block::new(DenseMatrix::from_row_slice(1, 1, &[2.0]), vec![1.0]),
            Block::optional(None),
        ];
        for block in blocks {
            let bytes = postcard::to_allocvec(&block).unwrap();
            let restored: Block = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(restored.matrix(), block.matrix());
            assert_eq!(restored.rhs(), block.rhs());

            let borrowed = restored.borrowed();
            assert_eq!(borrowed.matrix().copied(), block.matrix());
            assert_eq!(borrowed.rhs().copied(), block.rhs());
        }
    }
}

impl Gradient for OutputProbe {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, CallbackError> {
        Ok(vec![2.0 * x[0]])
    }
}
impl basin::ConstraintJacobian for OutputProbe {
    type Param = Vec<f64>;
    type Jacobian = DenseMatrix;
    type Error = CallbackError;
    fn constraint_jacobian(
        &self,
        _: &Vec<f64>,
    ) -> Result<DenseMatrix, ConstraintSolveError<CallbackError>> {
        panic!("zero-count blocks must skip their Jacobian callback");
    }
}
#[test]
fn slsqp_skips_zero_count_values_and_jacobians() {
    let mut problem = Problem::new(OutputProbe {
        count: Cell::new(0),
        length: 99,
        fail: true,
        calls: Cell::new(0),
    });
    Slsqp::new()
        .init(
            &mut problem,
            &mut basin::SelectedFirstOrderState::new(vec![1.0]),
        )
        .unwrap();
    assert_eq!(problem.inner().calls.get(), 0);
    assert_eq!(problem.counts().residual_evals, 0);
    assert_eq!(problem.counts().jacobian_evals, 0);
}
