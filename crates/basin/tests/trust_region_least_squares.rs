use basin::solver::trust_region::GaussNewtonWorkspace;
use basin::solver::trust_region::diagnostics::Phase;
use basin::{
    ArctanLoss, CauchyPoint, DenseMatrix, DenseMatrixFromFn, Dogleg, Dot,
    Executor, GaussNewtonModel, GramMatrix, HuberLoss, Jacobian,
    LinearSolveSpd, LossFunction, MatTransposeVec, MatVec, MatrixFinite,
    NegInPlace, NormSquared, Residual, RobustLeastSquares, Scalar,
    ScaleInPlace, ScaledAdd, SquaredLoss, TerminationCode, TerminationStage,
    TrustRegion, VectorIndex, VectorLen,
};
use std::cell::Cell;
use std::convert::Infallible;

#[path = "support/backend_aliases.rs"]
mod backend_aliases;

fn num<F: Scalar>(x: f64) -> F {
    F::from_f64(x).unwrap()
}

struct Linear<V, F> {
    vector: fn(&[F]) -> V,
}

impl<V: VectorIndex<F>, F: Scalar> Residual for Linear<V, F> {
    type Param = V;
    type Output = V;
    type Error = Infallible;

    fn residual(&self, x: &V) -> Result<V, Infallible> {
        Ok((self.vector)(&[
            x.get_scalar(0) - num(2.0),
            num::<F>(2.0) * (x.get_scalar(1) - F::one()),
            num(0.5),
        ]))
    }
}

impl<V: VectorIndex<F> + DenseMatrixFromFn<F>, F: Scalar> Jacobian
    for Linear<V, F>
{
    type Jacobian = V::Matrix;

    fn jacobian(&self, _: &V) -> Result<V::Matrix, Infallible> {
        Ok(V::dense_from_fn(3, 2, |i, j| match (i, j) {
            (0, 0) => F::one(),
            (1, 1) => num(2.0),
            _ => F::zero(),
        }))
    }
}

fn check_backend<V, F>(vector: fn(&[F]) -> V)
where
    F: Scalar,
    V: Clone
        + VectorIndex<F>
        + VectorLen
        + NormSquared<F>
        + Dot<F>
        + ScaledAdd<F>
        + ScaleInPlace<F>
        + NegInPlace
        + DenseMatrixFromFn<F>,
    V::Matrix: GramMatrix
        + MatTransposeVec<V>
        + MatrixFinite
        + MatVec<V>
        + LinearSolveSpd<V>,
{
    let tolerance = num::<F>(128.0) * F::epsilon();
    let full_step = [2.0, 1.0];
    // Midpoint of the second leg, from (10/17, 20/17) to (2, 1).
    let boundary_step = [22.0 / 17.0, 37.0 / 34.0];
    let boundary_radius = 3305.0_f64.sqrt() / 34.0;

    for (radius, expected, hit_boundary) in [
        (3.0, full_step, false),
        (boundary_radius, boundary_step, true),
    ] {
        let solver: TrustRegion<Dogleg, F, GaussNewtonModel, _> =
            TrustRegion::least_squares_with(Dogleg)
                .with_trial_diagnostics(true);
        let result = Executor::from_start(
            Linear { vector },
            solver
                .with_absolute_gradient_tolerance(tolerance)
                .with_radius(num(radius)),
            vector(&[F::zero(), F::zero()]),
        )
        .require_evaluated_state()
        .max_iter(1)
        .run_with_solver()
        .unwrap();
        let (x, cost, gradient) = result.state.current().unwrap();
        for (i, expected) in expected.into_iter().enumerate() {
            assert!((x.get_scalar(i) - num(expected)).abs() <= tolerance);
        }
        let problem = Linear { vector };
        let residual = problem.residual(x).unwrap();
        assert!(
            (cost - num::<F>(0.5) * residual.norm_squared()).abs() <= tolerance
        );
        assert!(
            (gradient.get_scalar(0) - residual.get_scalar(0)).abs()
                <= tolerance
        );
        assert!(
            (gradient.get_scalar(1) - num::<F>(2.0) * residual.get_scalar(1))
                .abs()
                <= tolerance
        );
        let trial = &result.solver.solver().trial_observations()[0];
        assert_eq!(trial.phase, Phase::Accepted);
        assert_eq!(trial.hit_boundary, Some(hit_boundary));
        assert!((trial.ratio.unwrap() - F::one()).abs() <= tolerance);
        assert_eq!(result.counts.residual_evals, 2);
        assert_eq!(result.counts.jacobian_evals, 2);
        assert_eq!(result.counts.total_work(), 4);
    }
}

#[test]
fn vec_backend() {
    check_backend::<_, f64>(|x| x.to_vec());
    check_backend::<_, f32>(|x| x.to_vec());
}

#[cfg(feature = "nalgebra_all")]
#[test]
fn nalgebra_backend() {
    use backend_aliases::nalgebra::DVector;
    check_backend::<_, f64>(DVector::from_column_slice);
    check_backend::<_, f32>(DVector::from_column_slice);
}

#[cfg(feature = "ndarray_all")]
#[test]
fn ndarray_backend() {
    use backend_aliases::ndarray::Array1;
    check_backend::<_, f64>(|x| Array1::from_vec(x.to_vec()));
    check_backend::<_, f32>(|x| Array1::from_vec(x.to_vec()));
}

#[cfg(feature = "faer_all")]
#[test]
fn faer_backend() {
    use backend_aliases::faer::Col;
    check_backend::<_, f64>(|x| Col::from_fn(x.len(), |i| x[i]));
    check_backend::<_, f32>(|x| Col::from_fn(x.len(), |i| x[i]));
}

struct LinearWithConstant(f32);

impl Residual for LinearWithConstant {
    type Param = Vec<f32>;
    type Output = Vec<f32>;
    type Error = Infallible;

    fn residual(&self, x: &Vec<f32>) -> Result<Vec<f32>, Infallible> {
        Ok(vec![x[0], self.0])
    }
}

impl Jacobian for LinearWithConstant {
    type Jacobian = DenseMatrix<f32>;

    fn jacobian(&self, _: &Vec<f32>) -> Result<Self::Jacobian, Infallible> {
        Ok(DenseMatrix::from_row_slice(2, 1, &[1.0, 0.0]))
    }
}

#[test]
fn accepts_reduction_below_the_precision_of_the_cost() {
    let solver = || {
        TrustRegion::least_squares_with(Dogleg)
            .with_absolute_gradient_tolerance(0.0)
    };
    let plain =
        Executor::from_start(LinearWithConstant(1.0), solver(), vec![1e-4])
            .max_iter(4)
            .run()
            .unwrap();
    let robust = Executor::from_start(
        RobustLeastSquares::new(LinearWithConstant(1.0), SquaredLoss),
        solver(),
        vec![1e-4],
    )
    .max_iter(4)
    .run()
    .unwrap();
    for result in [plain, robust] {
        assert_eq!(result.report.code(), TerminationCode::GradientTolerance);
        assert_eq!(result.report.iteration, 1);
        let (x, cost, gradient) = result.state.current().unwrap();
        assert_eq!(x, &[0.0]);
        assert_eq!(cost, 0.5);
        assert_eq!(gradient, &[0.0]);
        assert_eq!(result.counts.residual_evals, 2);
        assert_eq!(result.counts.jacobian_evals, 2);
    }
}

struct SymmetricLinear;

impl Residual for SymmetricLinear {
    type Param = Vec<f32>;
    type Output = Vec<f32>;
    type Error = Infallible;

    fn residual(&self, x: &Vec<f32>) -> Result<Vec<f32>, Infallible> {
        Ok(vec![1.0 + x[0], x[0] - 1.0])
    }
}

impl Jacobian for SymmetricLinear {
    type Jacobian = DenseMatrix<f32>;

    fn jacobian(&self, _: &Vec<f32>) -> Result<Self::Jacobian, Infallible> {
        Ok(DenseMatrix::from_row_slice(2, 1, &[1.0, 1.0]))
    }
}

#[test]
fn squared_loss_preserves_small_reductions_in_a_linear_fit() {
    let solver = || {
        TrustRegion::least_squares_with(Dogleg)
            .with_absolute_gradient_tolerance(0.0)
    };
    let start = 1.0 / 16384.0;
    let plain = Executor::from_start(SymmetricLinear, solver(), vec![start])
        .max_iter(4)
        .run()
        .unwrap();
    let squared = Executor::from_start(
        RobustLeastSquares::new(SymmetricLinear, SquaredLoss),
        solver(),
        vec![start],
    )
    .max_iter(4)
    .run()
    .unwrap();
    let borrowed = Executor::from_start(
        RobustLeastSquares::new(
            SymmetricLinear,
            &SquaredLoss as &dyn LossFunction<f32>,
        ),
        solver(),
        vec![start],
    )
    .max_iter(4)
    .run()
    .unwrap();
    for result in [plain, squared, borrowed] {
        assert_eq!(result.report.code(), TerminationCode::GradientTolerance);
        assert_eq!(result.report.iteration, 1);
        let (x, cost, gradient) = result.state.current().unwrap();
        assert_eq!(x, &[0.0]);
        assert_eq!(cost, 1.0);
        assert_eq!(gradient, &[0.0]);
        assert_eq!(result.counts.residual_evals, 2);
        assert_eq!(result.counts.jacobian_evals, 2);
    }
}

struct CancellingReductions;

impl Residual for CancellingReductions {
    type Param = Vec<f32>;
    type Output = Vec<f32>;
    type Error = Infallible;

    fn residual(&self, x: &Vec<f32>) -> Result<Vec<f32>, Infallible> {
        let square = x[0] * x[0];
        Ok(vec![
            12288.0 * (1.0 - square),
            2.0 * square,
            2.0 * square,
            2.0 * square,
            12288.0 * square,
            x[0] - 1.0,
        ])
    }
}

impl Jacobian for CancellingReductions {
    type Jacobian = DenseMatrix<f32>;

    fn jacobian(&self, x: &Vec<f32>) -> Result<Self::Jacobian, Infallible> {
        let slope = 2.0 * x[0];
        Ok(DenseMatrix::from_row_slice(
            6,
            1,
            &[
                -12288.0 * slope,
                2.0 * slope,
                2.0 * slope,
                2.0 * slope,
                12288.0 * slope,
                1.0,
            ],
        ))
    }
}

#[test]
fn rejects_uphill_steps_when_large_residual_reductions_cancel() {
    let solver = || {
        TrustRegion::least_squares_with(Dogleg)
            .with_max_inner_attempts(1)
            .with_trial_diagnostics(true)
    };
    let plain = Executor::from_start(CancellingReductions, solver(), vec![0.0])
        .max_iter(1)
        .run_with_solver()
        .unwrap();
    // Huber is quadratic over these residuals at this scale.
    let robust = Executor::from_start(
        RobustLeastSquares::new(CancellingReductions, HuberLoss)
            .with_scale(16384.0),
        solver(),
        vec![0.0],
    )
    .max_iter(1)
    .run_with_solver()
    .unwrap();
    let reference_cost = |x| {
        CancellingReductions
            .residual(&vec![x])
            .unwrap()
            .into_iter()
            .map(|r| 0.5 * f64::from(r).powi(2))
            .sum::<f64>()
    };
    let reference_reduction = reference_cost(0.0) - reference_cost(1.0);

    for result in [plain, robust] {
        let trial = &result.solver.trial_observations()[0];
        assert_eq!(trial.predicted_reduction, Some(0.5));
        assert_eq!(f64::from(trial.ratio.unwrap() * 0.5), reference_reduction);
        assert_eq!(trial.phase, Phase::Rejected);
        let (x, cost, gradient) = result.state.current().unwrap();
        assert_eq!(x, &[0.0]);
        assert!(trial.trial_cost.unwrap() > cost);
        assert_eq!(gradient, &[-1.0]);
        assert_eq!(result.counts.residual_evals, 2);
        assert_eq!(result.counts.jacobian_evals, 1);
    }
}

#[test]
fn keeps_cost_finite_when_the_squared_norm_overflows() {
    let result = Executor::from_start(
        LinearWithConstant(2e19),
        TrustRegion::least_squares_with(Dogleg)
            .with_absolute_gradient_tolerance(0.0),
        vec![1.0],
    )
    .max_iter(4)
    .run()
    .unwrap();
    assert_eq!(result.report.code(), TerminationCode::GradientTolerance);
    assert_eq!(result.report.iteration, 1);
    assert_eq!(result.param(), &[0.0]);
    assert!((result.cost() / 2e38 - 1.0).abs() <= f32::EPSILON);
}

struct GramOverflowAtRoot;

impl Residual for GramOverflowAtRoot {
    type Param = Vec<f32>;
    type Output = Vec<f32>;
    type Error = Infallible;

    fn residual(&self, x: &Vec<f32>) -> Result<Vec<f32>, Infallible> {
        let x = x[0];
        Ok(vec![(x - 1.0) * (1.0 + 1e20 * x * x)])
    }
}

impl Jacobian for GramOverflowAtRoot {
    type Jacobian = DenseMatrix<f32>;

    fn jacobian(&self, x: &Vec<f32>) -> Result<Self::Jacobian, Infallible> {
        let x = x[0];
        let slope = 1.0 + 1e20 * x * (3.0 * x - 2.0);
        Ok(DenseMatrix::from_row_slice(1, 1, &[slope]))
    }
}

#[test]
fn publishes_the_exact_root_even_when_its_gram_matrix_overflows() {
    for (tolerance, code) in [
        (Some(0.0), TerminationCode::GradientTolerance),
        (None, TerminationCode::SolverConverged),
    ] {
        for (start, iteration) in [(1.0, 0), (0.0, 1)] {
            let result = Executor::from_start(
                GramOverflowAtRoot,
                TrustRegion::least_squares_with(Dogleg)
                    .with_absolute_gradient_tolerance(tolerance),
                vec![start],
            )
            .require_evaluated_state()
            .max_iter(4)
            .run()
            .unwrap();
            assert_eq!(result.report.code(), code);
            assert_eq!(result.report.iteration, iteration);
            let (x, cost, gradient) = result.state.current().unwrap();
            assert_eq!(x, &[1.0]);
            assert_eq!(cost, 0.0);
            assert_eq!(gradient, &[0.0]);
            assert_eq!(result.counts.residual_evals, iteration + 1);
            assert_eq!(result.counts.jacobian_evals, iteration + 1);
        }
    }
}

struct Nonlinear;

impl Residual for Nonlinear {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = Infallible;

    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(vec![x[0] * x[0] - 1.0])
    }
}

impl Jacobian for Nonlinear {
    type Jacobian = DenseMatrix;

    fn jacobian(&self, x: &Vec<f64>) -> Result<DenseMatrix, Infallible> {
        Ok(DenseMatrix::from_row_slice(1, 1, &[2.0 * x[0]]))
    }
}

type VecDogleg = TrustRegion<
    Dogleg,
    f64,
    GaussNewtonModel,
    Option<GaussNewtonWorkspace<Vec<f64>, DenseMatrix>>,
>;

fn nonlinear_solver() -> VecDogleg {
    TrustRegion::least_squares_with(Dogleg)
        .with_radius(10.0)
        .with_max_inner_attempts(1)
        .with_trial_diagnostics(true)
}

#[test]
fn rejected_iteration_reuses_the_model_and_skips_change_checks() {
    let result = Executor::from_start(
        Nonlinear,
        nonlinear_solver()
            .with_absolute_cost_change_tolerance(0.0)
            .with_absolute_step_tolerance(0.0),
        vec![0.1],
    )
    .max_iter(2)
    .run_with_solver()
    .unwrap();
    assert!((result.param()[0] - 1.3375).abs() < 1e-14);
    assert_eq!(result.counts.residual_evals, 3);
    assert_eq!(result.counts.jacobian_evals, 2);
    assert_eq!(result.counts.total_work(), 5);
    let trials = result.solver.solver().trial_observations();
    assert_eq!(trials.len(), 2);
    assert_eq!(trials[0].phase, Phase::Rejected);
    assert!(trials[0].ratio.unwrap() < 0.0);
    assert_eq!(trials[0].counts_after.jacobian_evals, 1);
    assert_eq!(trials[1].phase, Phase::Accepted);
}

struct CountedNonlinear<'a>(&'a Cell<usize>);

impl Residual for CountedNonlinear<'_> {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = Infallible;

    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Nonlinear.residual(x)
    }
}

impl<'a> Jacobian for CountedNonlinear<'a> {
    type Jacobian = CountedMatrix<'a>;

    fn jacobian(&self, x: &Vec<f64>) -> Result<Self::Jacobian, Infallible> {
        Ok(CountedMatrix {
            entry: 2.0 * x[0],
            gram_builds: self.0,
        })
    }
}

struct CountedMatrix<'a> {
    entry: f64,
    gram_builds: &'a Cell<usize>,
}

impl GramMatrix for CountedMatrix<'_> {
    fn gram(&self) -> Self {
        self.gram_builds.set(self.gram_builds.get() + 1);
        Self {
            entry: self.entry * self.entry,
            gram_builds: self.gram_builds,
        }
    }
}

impl MatrixFinite for CountedMatrix<'_> {
    fn matrix_is_finite(&self) -> bool {
        self.entry.is_finite()
    }
}

impl MatVec<Vec<f64>> for CountedMatrix<'_> {
    fn matvec(&self, v: &Vec<f64>) -> Vec<f64> {
        vec![self.entry * v[0]]
    }
}

impl MatTransposeVec<Vec<f64>> for CountedMatrix<'_> {
    fn mat_transpose_vec(&self, v: &Vec<f64>) -> Vec<f64> {
        self.matvec(v)
    }
}

#[test]
fn builds_the_gram_only_when_a_step_needs_it() {
    let gram_builds = Cell::new(0);
    let fit = || CountedNonlinear(&gram_builds);
    let start = |x| {
        Executor::from_start(
            fit(),
            TrustRegion::least_squares_with(CauchyPoint)
                .with_absolute_gradient_tolerance(0.0)
                .with_max_inner_attempts(1)
                .with_radius(10.0),
            vec![x],
        )
    };
    let stationary = start(1.0).run().unwrap();
    assert_eq!(stationary.report.code(), TerminationCode::GradientTolerance);
    assert_eq!(stationary.report.iteration, 0);
    assert_eq!(stationary.counts.residual_evals, 1);
    assert_eq!(stationary.counts.jacobian_evals, 1);
    assert_eq!(gram_builds.get(), 0);

    let seed = start(0.1)
        .require_evaluated_state()
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    assert_eq!(gram_builds.get(), 0);
    let rejected =
        Executor::resume_from_checkpoint(fit(), seed.into_checkpoint())
            .max_iter(1)
            .run_with_solver()
            .unwrap();
    assert_eq!(rejected.param(), &[0.1]);
    assert_eq!(rejected.counts.residual_evals, 2);
    assert_eq!(rejected.counts.jacobian_evals, 1);
    assert_eq!(gram_builds.get(), 1);

    let accepted =
        Executor::resume_from_checkpoint(fit(), rejected.into_checkpoint())
            .max_iter(2)
            .run()
            .unwrap();
    assert!((accepted.param()[0] - 1.3375).abs() < 1e-14);
    assert_eq!(accepted.counts.residual_evals, 3);
    assert_eq!(accepted.counts.jacobian_evals, 2);
    assert_eq!(gram_builds.get(), 1);
}

struct NoFreeParameters;

impl Residual for NoFreeParameters {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = Infallible;

    fn residual(&self, _: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(vec![3.0, -4.0])
    }
}

impl Jacobian for NoFreeParameters {
    type Jacobian = DenseMatrix;

    fn jacobian(&self, _: &Vec<f64>) -> Result<Self::Jacobian, Infallible> {
        Ok(DenseMatrix::from_row_slice(2, 0, &[]))
    }
}

#[test]
fn initializes_cost_and_gradient_without_free_parameters() {
    for (max_iter, code) in [
        (0, TerminationCode::MaxIter),
        (1, TerminationCode::SolverConverged),
    ] {
        let result = Executor::from_start(
            NoFreeParameters,
            TrustRegion::least_squares_with(Dogleg),
            vec![],
        )
        .require_evaluated_state()
        .max_iter(max_iter)
        .run()
        .unwrap();
        assert_eq!(result.report.code(), code);
        assert_eq!(result.report.iteration, 0);
        let (x, cost, gradient) = result.state.current().unwrap();
        assert!(x.is_empty());
        assert_eq!(cost, 12.5);
        assert!(gradient.is_empty());
        assert_eq!(result.counts.residual_evals, 1);
        assert_eq!(result.counts.jacobian_evals, 1);
    }
}

struct RankDeficient;

impl Residual for RankDeficient {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = Infallible;

    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(vec![x[0] - 1.0])
    }
}

impl Jacobian for RankDeficient {
    type Jacobian = DenseMatrix;

    fn jacobian(&self, _: &Vec<f64>) -> Result<DenseMatrix, Infallible> {
        Ok(DenseMatrix::from_row_slice(1, 2, &[1.0, 0.0]))
    }
}

#[test]
fn rank_deficient_dogleg_uses_the_cauchy_step() {
    let dogleg = Executor::from_start(
        RankDeficient,
        TrustRegion::least_squares_with(Dogleg),
        vec![0.0, 3.0],
    )
    .max_iter(1)
    .run()
    .unwrap();
    let cauchy = Executor::from_start(
        RankDeficient,
        TrustRegion::least_squares_with(CauchyPoint),
        vec![0.0, 3.0],
    )
    .max_iter(1)
    .run()
    .unwrap();
    assert_eq!(dogleg.state, cauchy.state);
    assert_eq!(dogleg.param(), &[1.0, 3.0]);
    assert_eq!(dogleg.cost(), 0.0);
}

enum TrialIssue {
    NonFiniteResidual,
    NonFiniteJacobian,
    GramOverflow,
    ResidualError,
    JacobianError,
}

impl Residual for TrialIssue {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = &'static str;

    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        if x[0] > 0.1 {
            match self {
                Self::NonFiniteResidual => return Ok(vec![f64::NAN]),
                Self::ResidualError => return Err("residual error"),
                _ => {}
            }
        }
        Ok(vec![x[0] - 1.0])
    }
}

impl Jacobian for TrialIssue {
    type Jacobian = DenseMatrix;

    fn jacobian(&self, x: &Vec<f64>) -> Result<DenseMatrix, Self::Error> {
        let slope = if x[0] > 0.1 {
            match self {
                Self::NonFiniteJacobian => f64::INFINITY,
                Self::GramOverflow => f64::MAX,
                Self::JacobianError => return Err("Jacobian error"),
                _ => 1.0,
            }
        } else {
            1.0
        };
        Ok(DenseMatrix::from_row_slice(1, 1, &[slope]))
    }
}

#[test]
fn non_finite_trial_cost_retries_without_a_jacobian() {
    let result = Executor::from_start(
        TrialIssue::NonFiniteResidual,
        TrustRegion::least_squares_with(Dogleg),
        vec![0.0],
    )
    .max_iter(1)
    .run()
    .unwrap();
    assert_eq!(result.param(), &[0.0625]);
    assert_eq!(result.counts.residual_evals, 4);
    assert_eq!(result.counts.jacobian_evals, 2);
}

#[test]
fn invalid_initial_model_stops_at_the_boundary() {
    for issue in [TrialIssue::NonFiniteResidual, TrialIssue::NonFiniteJacobian]
    {
        let result = Executor::from_start(
            issue,
            TrustRegion::least_squares_with(Dogleg)
                .with_absolute_gradient_tolerance(1e-8),
            vec![1.0],
        )
        .require_evaluated_state()
        .run()
        .unwrap();
        assert_eq!(result.report.code(), TerminationCode::SolverFailed);
        assert_eq!(result.report.stage, TerminationStage::Boundary);
        assert_eq!(result.report.iteration, 0);
        assert!(result.state.current().unwrap().2[0].is_nan());
        assert_eq!(result.counts.total_work(), 2);
    }
}

#[test]
fn invalid_trial_gradient_preserves_the_current_evaluation() {
    let result = Executor::from_start(
        TrialIssue::NonFiniteJacobian,
        TrustRegion::least_squares_with(Dogleg).with_trial_diagnostics(true),
        vec![0.0],
    )
    .run_with_solver()
    .unwrap();
    assert_eq!(result.report.code(), TerminationCode::SolverFailed);
    assert_eq!(
        result.report.stage,
        TerminationStage::Step { completed: false }
    );
    let (x, cost, gradient) = result.state.current().unwrap();
    assert_eq!(x, &[0.0]);
    assert_eq!(cost, 0.5);
    assert_eq!(gradient, &[-1.0]);
    assert_eq!(result.counts.residual_evals, 2);
    assert_eq!(result.counts.jacobian_evals, 2);
    assert_eq!(
        result.solver.trial_observations()[0].phase,
        Phase::InvalidModel
    );
}

#[test]
fn reports_gram_overflow_when_another_step_is_needed() {
    let accepted = Executor::from_start(
        TrialIssue::GramOverflow,
        TrustRegion::least_squares_with(Dogleg)
            .with_radius(0.5)
            .with_trial_diagnostics(true),
        vec![0.0],
    )
    .max_iter(1)
    .run_with_solver()
    .unwrap();
    assert_eq!(accepted.report.code(), TerminationCode::MaxIter);
    let (x, cost, gradient) = accepted.state.current().unwrap();
    assert_eq!(x, &[0.5]);
    assert_eq!(cost, 0.125);
    assert_eq!(gradient, &[-0.5 * f64::MAX]);
    assert_eq!(
        accepted.solver.trial_observations()[0].phase,
        Phase::Accepted
    );
    assert_eq!(accepted.counts.residual_evals, 2);
    assert_eq!(accepted.counts.jacobian_evals, 2);
    let counts = accepted.counts;

    let resumed = Executor::resume_from_checkpoint(
        TrialIssue::GramOverflow,
        accepted.into_checkpoint(),
    )
    .run()
    .unwrap();
    assert_eq!(resumed.report.code(), TerminationCode::SolverFailed);
    assert_eq!(resumed.report.iteration, 1);
    assert_eq!(
        resumed.report.stage,
        TerminationStage::Step { completed: false }
    );
    assert_eq!(resumed.param(), &[0.5]);
    assert_eq!(resumed.counts, counts);
}

#[test]
fn callback_errors_propagate() {
    for (issue, message) in [
        (TrialIssue::ResidualError, "residual error"),
        (TrialIssue::JacobianError, "Jacobian error"),
    ] {
        let result = Executor::from_start(
            issue,
            TrustRegion::least_squares_with(Dogleg),
            vec![0.0],
        )
        .run();
        assert!(matches!(result, Err(error) if error == message));
    }
}

#[cfg(feature = "serde")]
fn round_trip<T: serde::Serialize + serde::de::DeserializeOwned>(
    value: &T,
) -> T {
    postcard::from_bytes(&postcard::to_allocvec(value).unwrap()).unwrap()
}

#[test]
fn fresh_run_and_checkpoint_preserve_model_lifecycle() {
    let make =
        || Executor::from_start(Nonlinear, nonlinear_solver(), vec![0.1]);
    let expected = make().max_iter(5).run_with_solver().unwrap();
    for split in [0, 1, 2] {
        let checkpoint = make()
            .max_iter(split)
            .run_with_solver()
            .unwrap()
            .into_checkpoint();
        #[cfg(feature = "serde")]
        let checkpoint = round_trip(&checkpoint);
        let resumed = Executor::resume_from_checkpoint(Nonlinear, checkpoint)
            .max_iter(5)
            .run_with_solver()
            .unwrap();
        assert_eq!(resumed.state, expected.state);
        assert_eq!(resumed.counts, expected.counts);
        assert_eq!(resumed.solver.radius(), expected.solver.radius());
    }
    let radius = expected.solver.radius();
    let restarted = Executor::from_start(Nonlinear, expected.solver, vec![0.1])
        .max_iter(5)
        .run_with_solver()
        .unwrap();
    assert_eq!(restarted.state, expected.state);
    assert_eq!(restarted.counts, expected.counts);
    assert_eq!(restarted.solver.radius(), radius);
    assert_eq!(restarted.solver.trial_observations()[0].sequence, 1);
}

struct ScalarFit {
    residual: fn(f32) -> Vec<f32>,
    jacobian: fn(f32) -> Vec<f32>,
}

impl Residual for ScalarFit {
    type Param = Vec<f32>;
    type Output = Vec<f32>;
    type Error = Infallible;

    fn residual(&self, x: &Vec<f32>) -> Result<Vec<f32>, Infallible> {
        Ok((self.residual)(x[0]))
    }
}

impl Jacobian for ScalarFit {
    type Jacobian = DenseMatrix<f32>;

    fn jacobian(&self, x: &Vec<f32>) -> Result<Self::Jacobian, Infallible> {
        let column = (self.jacobian)(x[0]);
        Ok(DenseMatrix::from_row_slice(column.len(), 1, &column))
    }
}

#[test]
fn huber_gradient_uses_the_raw_jacobian() {
    // Adjacent f32 values whose products with the Huber row factor round equally.
    const A: f32 = f32::from_bits(0x3fb5_04f7);
    const B: f32 = f32::from_bits(0x3fb5_04f8);
    let fit = RobustLeastSquares::new(
        ScalarFit {
            residual: |x| vec![4.0 + A * x, -4.0 + B * x],
            jacobian: |_| vec![A, B],
        },
        HuberLoss,
    );
    let start = vec![0.0];
    let expected = basin::Gradient::gradient(&fit, &start).unwrap();
    assert_ne!(expected, [0.0]);
    let result = Executor::from_start(
        fit,
        TrustRegion::least_squares_with(Dogleg),
        start,
    )
    .require_evaluated_state()
    .max_iter(0)
    .run_with_solver()
    .unwrap();
    assert_eq!(result.state.current().unwrap().2, &expected);
}

#[test]
fn dogleg_accepts_small_decreases_in_the_arctan_tail() {
    let fit = RobustLeastSquares::new(
        ScalarFit {
            residual: |x| vec![1000.0 * (300.0 - x)],
            jacobian: |_| vec![-1000.0],
        },
        ArctanLoss,
    )
    .with_scale(1000.0);
    let result = Executor::from_start(
        fit,
        TrustRegion::least_squares_with(Dogleg),
        vec![0.0],
    )
    .max_iter(1000)
    .run()
    .unwrap();
    assert!(
        (result.param()[0] - 300.0).abs() < 0.001,
        "final parameter {:?}, report {:?}",
        result.param(),
        result.report
    );
}

#[test]
fn dogleg_rejects_an_uphill_step_with_nearly_canceling_reductions() {
    let c = 1.5 * 2.0_f32.powi(30);
    let d = 128.0_f32;
    let reference_cost = |x: f64| {
        let c = f64::from(c);
        let d = f64::from(d);
        0.5 * ((c + d * x * x).powi(2)
            + (c - d * x * x).powi(2)
            + (d * (x - 1.0)).powi(2))
    };
    assert_eq!(reference_cost(1.0) - reference_cost(0.0), 8192.0);
    let fit = || ScalarFit {
        residual: |x| {
            let c = 1.5 * 2.0_f32.powi(30);
            let d = 128.0;
            vec![c + d * x * x, c - d * x * x, d * (x - 1.0)]
        },
        jacobian: |x| vec![256.0 * x, -256.0 * x, 128.0],
    };
    let solver = || {
        TrustRegion::least_squares_with(Dogleg)
            .with_radius(1.0)
            .with_max_inner_attempts(1)
            .with_trial_diagnostics(true)
    };
    let plain = Executor::from_start(fit(), solver(), vec![0.0])
        .max_iter(1)
        .run_with_solver()
        .unwrap();
    let huber = Executor::from_start(
        RobustLeastSquares::new(fit(), HuberLoss).with_scale(2e9),
        solver(),
        vec![0.0],
    )
    .max_iter(1)
    .run_with_solver()
    .unwrap();
    for result in [plain, huber] {
        assert_eq!(result.param(), &[0.0]);
        let trial = &result.solver.trial_observations()[0];
        assert_eq!(trial.phase, Phase::Rejected);
        assert_eq!(trial.ratio, Some(-1.0));
        assert_eq!(result.counts.jacobian_evals, 1);
    }
}

#[test]
fn dogleg_preserves_the_cross_term_in_squared_reductions() {
    const C: f32 = 33_554_432.0;
    let fit = ScalarFit {
        residual: |x| {
            let t = x * x;
            vec![C * (1.0 - t) + t, C * t, 0.5 * (x - 1.0)]
        },
        jacobian: |x| vec![2.0 * (1.0 - C) * x, 2.0 * C * x, 0.5],
    };
    // The large residual moves to the second component and cancels.
    // At x = 1 the remaining squared cost grows from 1/8 to 1/2.
    let result = Executor::from_start(
        fit,
        TrustRegion::least_squares_with(Dogleg)
            .with_max_inner_attempts(1)
            .with_trial_diagnostics(true),
        vec![0.0],
    )
    .max_iter(1)
    .run_with_solver()
    .unwrap();
    let trial = &result.solver.trial_observations()[0];
    assert_eq!(trial.predicted_reduction, Some(0.125));
    assert_eq!(trial.ratio, Some(-3.0));
    assert_eq!(trial.phase, Phase::Rejected);
    assert_eq!(result.param(), &[0.0]);
}

#[test]
fn dogleg_rejects_an_uphill_arctan_step_with_large_finite_costs() {
    const SCALE: f32 = 2.8e19;
    const TRIAL_RESIDUAL: f32 = 3e19;
    let fit = RobustLeastSquares::new(
        ScalarFit {
            residual: |x| {
                // The blend has zero slope at both endpoints. The small
                // quadratic term gives the solver a direction at the seed.
                let blend = x * x * (3.0 - 2.0 * x);
                vec![
                    SCALE * (1.0 - blend) - x * (1.0 - x),
                    TRIAL_RESIDUAL * blend,
                ]
            },
            jacobian: |x| {
                let slope = 6.0 * x * (1.0 - x);
                vec![-SCALE * slope - (1.0 - 2.0 * x), TRIAL_RESIDUAL * slope]
            },
        },
        ArctanLoss,
    )
    .with_scale(SCALE);
    let result = Executor::from_start(
        fit,
        TrustRegion::least_squares_with(Dogleg)
            .with_max_inner_attempts(1)
            .with_trial_diagnostics(true),
        vec![0.0],
    )
    .max_iter(1)
    .run_with_solver()
    .unwrap();
    let trial = &result.solver.trial_observations()[0];
    let reference_cost = |r: f32| {
        let scale = f64::from(SCALE);
        (0.5 * scale * scale) * (f64::from(r) / scale).powi(2).atan()
    };
    let before = reference_cost(SCALE);
    let after = reference_cost(TRIAL_RESIDUAL);
    let actual_reduction = f64::from(trial.ratio.unwrap())
        * f64::from(trial.predicted_reduction.unwrap());
    let expected_reduction = before - after;
    let eps = f64::from(f32::EPSILON);
    assert!(
        (f64::from(trial.trial_cost.unwrap()) / after - 1.0).abs() < 4.0 * eps
    );
    assert!(
        (actual_reduction / expected_reduction - 1.0).abs() < 32.0 * eps,
        "reduction {actual_reduction}, expected {expected_reduction}"
    );
    assert_eq!(trial.phase, Phase::Rejected);
    assert_eq!(result.param(), &[0.0]);
    assert_eq!(result.counts.jacobian_evals, 1);
}
