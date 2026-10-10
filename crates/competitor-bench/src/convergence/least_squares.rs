//! Native least-squares observations with a physical leaf-call ledger.

use std::time::{Duration, Instant};

use basin::solver::least_squares_diagnostics::{
    LeastSquaresDiagnostics, LeastSquaresObservation,
};
use basin::{
    BoxConstraints, CostFunction, EvalCounts, Executor, Jacobian, PointState,
    Residual, Scalar, Solver, State, StepOutcome,
};
#[cfg(not(feature = "basin-latest"))]
use nalgebra::{DMatrix, DVector};
#[cfg(feature = "basin-latest")]
use nalgebra_latest::{DMatrix, DVector};

use super::ledger::{OracleError, OracleKind, WorkLedger};
use super::nist::{Dataset, Nist};
use super::runner::{
    Measurement, PublicationStage, Recommendation, RunOutcome,
};

/// Analytic fixtures have a known minimum and independently supplied derivatives.
#[derive(Clone, Copy, Debug)]
pub enum AnalyticModel {
    Linear,
    /// Identity residuals with target (1, -1), for box KKT checks.
    BoxLinear,
    /// Two inliers and one outlier distinguish robust from squared-loss minima.
    RobustOutlier,
    /// Symmetric residuals give a known smooth robust minimum.
    RobustSymmetric,
    /// Four identifiable coordinates, each with three linear residuals.
    RobustLinear4,
    /// Two identifiable sums and two flat directions, with constant residuals.
    RobustRankDeficient4,
    /// A stationary maximum in the first coordinate, to separate stops from quality.
    BoxStationary,
    Nonzero,
    NonFiniteTrial,
}

/// Loss dispatch retains the production adapter and its native safeguards.
#[derive(Clone, Copy, Debug)]
pub enum PilotLoss {
    Huber,
    SoftL1,
    Cauchy,
    Arctan,
}
impl<F: Scalar> basin::LossFunction<F> for PilotLoss {
    fn evaluate(&self, z: F) -> basin::LossEvaluation<F> {
        match self {
            Self::Huber => basin::HuberLoss.evaluate(z),
            Self::SoftL1 => basin::SoftL1Loss.evaluate(z),
            Self::Cauchy => basin::CauchyLoss.evaluate(z),
            Self::Arctan => basin::ArctanLoss.evaluate(z),
        }
    }
    fn evaluate_scaled(&self, r: F, scale: F) -> basin::LossEvaluation<F> {
        match self {
            Self::Huber => basin::HuberLoss.evaluate_scaled(r, scale),
            Self::SoftL1 => basin::SoftL1Loss.evaluate_scaled(r, scale),
            Self::Cauchy => basin::CauchyLoss.evaluate_scaled(r, scale),
            Self::Arctan => basin::ArctanLoss.evaluate_scaled(r, scale),
        }
    }
}

/// Unit-scale analytic cases are measurement controls, not calibration targets.
#[derive(Clone, Copy, Debug)]
pub struct RobustFixture {
    pub name: &'static str,
    pub model: AnalyticModel,
    pub loss: PilotLoss,
    pub scale: f64,
    pub start: &'static [f64],
    /// A representative minimum; rank-deficient models have a minimizer set.
    pub reference: &'static [f64],
    pub bounds: Option<(f64, f64)>,
}

pub const ROBUST_FIXTURES: [RobustFixture; 7] = [
    RobustFixture {
        name: "robust_huber_outlier",
        model: AnalyticModel::RobustOutlier,
        loss: PilotLoss::Huber,
        scale: 1.,
        start: &[0.1],
        reference: &[0.5],
        bounds: None,
    },
    RobustFixture {
        name: "robust_huber_scaled",
        model: AnalyticModel::RobustOutlier,
        loss: PilotLoss::Huber,
        scale: 0.5,
        start: &[0.1],
        reference: &[0.25],
        bounds: None,
    },
    RobustFixture {
        name: "robust_soft_l1",
        model: AnalyticModel::RobustSymmetric,
        loss: PilotLoss::SoftL1,
        scale: 1.,
        start: &[0.1],
        reference: &[0.],
        bounds: None,
    },
    RobustFixture {
        name: "robust_cauchy",
        model: AnalyticModel::Linear,
        loss: PilotLoss::Cauchy,
        scale: 1.,
        start: &[-1.],
        reference: &[1.],
        bounds: None,
    },
    RobustFixture {
        name: "robust_huber_kink",
        model: AnalyticModel::Linear,
        loss: PilotLoss::Huber,
        scale: 0.5,
        start: &[0.5],
        reference: &[1.],
        bounds: None,
    },
    RobustFixture {
        name: "robust_nonfinite",
        model: AnalyticModel::NonFiniteTrial,
        loss: PilotLoss::SoftL1,
        scale: 1.,
        start: &[0.1],
        reference: &[1.],
        bounds: None,
    },
    RobustFixture {
        name: "robust_huber_bound",
        model: AnalyticModel::RobustOutlier,
        loss: PilotLoss::Huber,
        scale: 0.5,
        start: &[0.1],
        reference: &[0.125],
        bounds: Some((0., 0.125)),
    },
];

/// Larger controls keep all losses on identical full-rank and rank-deficient models.
pub const ROBUST_EXTENDED_FIXTURES: [RobustFixture; 9] = [
    RobustFixture {
        name: "robust_arctan",
        model: AnalyticModel::Linear,
        loss: PilotLoss::Arctan,
        scale: 1.,
        start: &[-1.],
        reference: &[1.],
        bounds: None,
    },
    extended_fixture(
        "robust_huber_linear4",
        AnalyticModel::RobustLinear4,
        PilotLoss::Huber,
    ),
    extended_fixture(
        "robust_soft_l1_linear4",
        AnalyticModel::RobustLinear4,
        PilotLoss::SoftL1,
    ),
    extended_fixture(
        "robust_cauchy_linear4",
        AnalyticModel::RobustLinear4,
        PilotLoss::Cauchy,
    ),
    extended_fixture(
        "robust_arctan_linear4",
        AnalyticModel::RobustLinear4,
        PilotLoss::Arctan,
    ),
    extended_fixture(
        "robust_huber_rank4",
        AnalyticModel::RobustRankDeficient4,
        PilotLoss::Huber,
    ),
    extended_fixture(
        "robust_soft_l1_rank4",
        AnalyticModel::RobustRankDeficient4,
        PilotLoss::SoftL1,
    ),
    extended_fixture(
        "robust_cauchy_rank4",
        AnalyticModel::RobustRankDeficient4,
        PilotLoss::Cauchy,
    ),
    extended_fixture(
        "robust_arctan_rank4",
        AnalyticModel::RobustRankDeficient4,
        PilotLoss::Arctan,
    ),
];

const fn extended_fixture(
    name: &'static str,
    model: AnalyticModel,
    loss: PilotLoss,
) -> RobustFixture {
    let (start, reference): (&[f64], &[f64]) = match model {
        AnalyticModel::RobustLinear4 => {
            (&[-1., 1., -0.5, 0.5], &[1., -1., 0.5, -0.5])
        }
        AnalyticModel::RobustRankDeficient4 => {
            (&[-0.5, -0.5, 0.5, 0.5], &[0.5, 0.5, -0.5, -0.5])
        }
        _ => panic!("expected a four-parameter robust model"),
    };
    RobustFixture {
        name,
        model,
        loss,
        scale: 1.,
        start,
        reference,
        bounds: None,
    }
}

pub fn robust_fixture(name: &str) -> Option<&'static RobustFixture> {
    ROBUST_FIXTURES
        .iter()
        .chain(ROBUST_EXTENDED_FIXTURES.iter())
        .find(|fixture| fixture.name == name)
}

/// Fixed stopping ablations retain the algorithm and its numerical safeguards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RobustStoppingPolicy {
    DefaultGradient,
    AllRelative,
    ModelReduction,
    TrialStep,
    ModelOrStep,
    NormalizedGradient,
    TrustRadius,
}

impl RobustStoppingPolicy {
    pub const ALL: [Self; 7] = [
        Self::DefaultGradient,
        Self::AllRelative,
        Self::ModelReduction,
        Self::TrialStep,
        Self::ModelOrStep,
        Self::NormalizedGradient,
        Self::TrustRadius,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::DefaultGradient => "robust_gradient_default",
            Self::AllRelative => "robust_relative_probe",
            Self::ModelReduction => "robust_model_probe",
            Self::TrialStep => "robust_step_probe",
            Self::ModelOrStep => "robust_model_step_probe",
            Self::NormalizedGradient => "robust_gradient_probe",
            Self::TrustRadius => "robust_radius_probe",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|policy| policy.name() == name)
    }

    pub fn configure<V, M, F: Scalar>(
        self,
        solver: basin::LevenbergMarquardt<V, M, F>,
        tolerance: F,
    ) -> basin::LevenbergMarquardt<V, M, F> {
        if self == Self::DefaultGradient {
            return solver;
        }
        // Keep exact stationarity enabled when isolating a relative predicate.
        let mut solver = solver.with_absolute_gradient_tolerance(F::zero());
        if matches!(self, Self::AllRelative | Self::NormalizedGradient) {
            solver = solver.with_gradient_orthogonality_tolerance(tolerance);
        }
        if matches!(
            self,
            Self::AllRelative | Self::ModelReduction | Self::ModelOrStep
        ) {
            solver = solver.with_relative_model_reduction_tolerance(tolerance);
        }
        if matches!(
            self,
            Self::AllRelative | Self::TrialStep | Self::ModelOrStep
        ) {
            solver = solver.with_relative_step_tolerance(tolerance);
        }
        if matches!(self, Self::AllRelative | Self::TrustRadius) {
            solver = solver.with_relative_trust_radius_tolerance(tolerance);
        }
        solver
    }
}

#[derive(Clone, Debug)]
pub enum Model<F: Scalar> {
    Nist(Box<Nist<F>>),
    Analytic(AnalyticModel),
}

#[derive(Clone, Debug)]
pub struct Instrumented<F: Scalar> {
    pub ledger: WorkLedger,
    pub model: Model<F>,
    pub fused: bool,
    lower: DVector<F>,
    upper: DVector<F>,
}

impl<F: Scalar> Instrumented<F> {
    pub fn nist(dataset: Dataset, ledger: WorkLedger) -> Self {
        let n = dataset.reference.len();
        Self {
            model: Model::Nist(Box::new(Nist::new(dataset))),
            ledger,
            fused: true,
            lower: DVector::from_element(n, F::neg_infinity()),
            upper: DVector::from_element(n, F::infinity()),
        }
    }
    pub fn analytic(model: AnalyticModel, ledger: WorkLedger) -> Self {
        let n = match model {
            AnalyticModel::RobustLinear4
            | AnalyticModel::RobustRankDeficient4 => 4,
            AnalyticModel::BoxLinear | AnalyticModel::BoxStationary => 2,
            _ => 1,
        };
        Self {
            model: Model::Analytic(model),
            ledger,
            fused: true,
            lower: DVector::from_element(n, F::neg_infinity()),
            upper: DVector::from_element(n, F::infinity()),
        }
    }
    /// Set fixture bounds without changing its residual model or ledger.
    pub fn with_bounds(mut self, lower: Vec<F>, upper: Vec<F>) -> Self {
        assert_eq!(lower.len(), upper.len());
        self.lower = DVector::from_vec(lower);
        self.upper = DVector::from_vec(upper);
        self
    }
    fn values(
        &self,
        x: &DVector<F>,
    ) -> Result<(DVector<F>, DMatrix<F>), OracleError> {
        let (r, j) = match &self.model {
            Model::Nist(model) => {
                let r = model
                    .residuals(x.as_slice())
                    .map_err(|_| OracleError::Callback("NIST residual"))?;
                let j = model
                    .jacobian_rows(x.as_slice())
                    .map_err(|_| OracleError::Callback("NIST Jacobian"))?;
                let rows = r.len();
                (
                    DVector::from_vec(r),
                    DMatrix::from_fn(rows, x.len(), |i, k| j[i][k]),
                )
            }
            Model::Analytic(AnalyticModel::RobustOutlier) => (
                DVector::from_vec(vec![
                    x[0],
                    x[0],
                    x[0] - F::from_f64(8.).unwrap(),
                ]),
                DMatrix::from_element(3, 1, F::one()),
            ),
            Model::Analytic(AnalyticModel::RobustSymmetric) => (
                DVector::from_vec(vec![
                    x[0] - F::from_f64(2.).unwrap(),
                    x[0] + F::from_f64(2.).unwrap(),
                ]),
                DMatrix::from_element(2, 1, F::one()),
            ),
            Model::Analytic(
                model @ (AnalyticModel::RobustLinear4
                | AnalyticModel::RobustRankDeficient4),
            ) => {
                let rank_deficient =
                    matches!(model, AnalyticModel::RobustRankDeficient4);
                let differences = if rank_deficient {
                    vec![x[0] + x[1] - F::one(), x[2] + x[3] + F::one()]
                } else {
                    [1., -1., 0.5, -0.5]
                        .iter()
                        .enumerate()
                        .map(|(k, target)| x[k] - F::from_f64(*target).unwrap())
                        .collect()
                };
                let rows =
                    differences.len() * 3 + if rank_deficient { 2 } else { 0 };
                let mut residual = DVector::zeros(rows);
                let mut jacobian = DMatrix::zeros(rows, 4);
                for (k, difference) in differences.into_iter().enumerate() {
                    for (i, weight) in [1., 2., -1.].iter().enumerate() {
                        let weight = F::from_f64(*weight).unwrap();
                        residual[3 * k + i] = weight * difference;
                        if rank_deficient {
                            jacobian[(3 * k + i, 2 * k)] = weight;
                            jacobian[(3 * k + i, 2 * k + 1)] = weight;
                        } else {
                            jacobian[(3 * k + i, k)] = weight;
                        }
                    }
                }
                if rank_deficient {
                    residual[rows - 2] = F::from_f64(2.).unwrap();
                    residual[rows - 1] = -F::from_f64(2.).unwrap();
                }
                (residual, jacobian)
            }
            Model::Analytic(AnalyticModel::BoxLinear) => (
                DVector::from_vec(vec![x[0] - F::one(), x[1] + F::one()]),
                DMatrix::identity(2, 2),
            ),
            Model::Analytic(AnalyticModel::BoxStationary) => (
                DVector::from_vec(vec![
                    F::from_f64(2.).unwrap() - x[0] * x[0],
                    x[1] + F::one(),
                ]),
                DMatrix::from_diagonal(&DVector::from_vec(vec![
                    -F::from_f64(2.).unwrap() * x[0],
                    F::one(),
                ])),
            ),
            Model::Analytic(model) => {
                let d = x[0] - F::one();
                let (r, j) = match model {
                    AnalyticModel::RobustOutlier
                    | AnalyticModel::RobustSymmetric
                    | AnalyticModel::RobustLinear4
                    | AnalyticModel::RobustRankDeficient4
                    | AnalyticModel::BoxLinear
                    | AnalyticModel::BoxStationary => {
                        unreachable!()
                    }
                    AnalyticModel::Linear => (d, F::one()),
                    AnalyticModel::Nonzero => {
                        (F::one() + d * d, F::from_f64(2.).unwrap() * d)
                    }
                    AnalyticModel::NonFiniteTrial
                        if x[0] < F::zero()
                            || x[0] > F::from_f64(1.1).unwrap() =>
                    {
                        (F::nan(), F::nan())
                    }
                    AnalyticModel::NonFiniteTrial => {
                        (F::one() + d * d, F::from_f64(2.).unwrap() * d)
                    }
                };
                (DVector::from_element(1, r), DMatrix::from_element(1, 1, j))
            }
        };
        Ok((r, j))
    }
    fn point(x: &DVector<F>) -> Vec<f64> {
        x.iter().map(|v| v.to_f64().unwrap()).collect()
    }
}

impl<F: Scalar> Residual for Instrumented<F> {
    type Param = DVector<F>;
    type Output = DVector<F>;
    type Error = OracleError;
    fn residual(&self, x: &DVector<F>) -> Result<DVector<F>, OracleError> {
        self.ledger
            .evaluate(OracleKind::Residual, &Self::point(x), || {
                Ok((self.values(x)?.0, None))
            })
    }
}

impl<F: Scalar> Jacobian for Instrumented<F> {
    type Jacobian = DMatrix<F>;
    fn jacobian(&self, x: &DVector<F>) -> Result<DMatrix<F>, OracleError> {
        self.ledger
            .evaluate(OracleKind::Jacobian, &Self::point(x), || {
                Ok((self.values(x)?.1, None))
            })
    }
    fn residual_and_jacobian(
        &self,
        x: &DVector<F>,
    ) -> Result<(DVector<F>, DMatrix<F>), OracleError> {
        if !self.fused {
            return Ok((self.residual(x)?, self.jacobian(x)?));
        }
        self.ledger.evaluate(
            OracleKind::ResidualJacobian,
            &Self::point(x),
            || Ok((self.values(x)?, None)),
        )
    }
}

impl<F: Scalar> CostFunction for Instrumented<F> {
    type Param = DVector<F>;
    type Output = F;
    type Error = OracleError;
    fn cost(&self, x: &DVector<F>) -> Result<F, OracleError> {
        self.ledger.evaluate(OracleKind::Cost, &Self::point(x), || {
            let r = self.values(x)?.0;
            let cost = r.iter().fold(F::zero(), |sum, v| sum + *v * *v)
                / F::from_f64(2.).unwrap();
            Ok((cost, Some(cost.to_f64().unwrap())))
        })
    }
}

impl<F: Scalar> BoxConstraints for Instrumented<F> {
    fn lower(&self) -> &DVector<F> {
        &self.lower
    }
    fn upper(&self) -> &DVector<F> {
        &self.upper
    }
}

#[derive(Clone, Debug)]
pub struct NativeRecord<F: Scalar> {
    pub observation: LeastSquaresObservation<F>,
    pub starting_iteration: u64,
    /// Work immediately before this native decision or attempted trial callback.
    pub work_before: u64,
    /// Completed trial's physical call ordinal, absent after a budget denial.
    pub residual_work: Option<u64>,
    /// Publication is separate from the numerical acceptance decision.
    pub published: bool,
}

#[derive(Clone, Debug)]
pub struct Measured<F: Scalar> {
    pub run: Measurement<F>,
    pub native: Vec<NativeRecord<F>>,
}

fn recommendation<F: Scalar>(
    state: &PointState<DVector<F>, F>,
    counts: &EvalCounts,
    ledger: &WorkLedger,
    stage: PublicationStage,
) -> Recommendation {
    Recommendation {
        point: state.param().iter().map(|x| x.to_f64().unwrap()).collect(),
        solver_cost: state.cost().to_f64().unwrap(),
        iteration: state.iter(),
        work: ledger.work(),
        counts: *counts,
        stage,
    }
}

/// Controls and errors preserve the same ownership rules as the original runner.
pub fn measure<F, P, So>(
    executor: Executor<P, PointState<DVector<F>, F>, So>,
    ledger: &WorkLedger,
    wall_cap: Duration,
) -> Measured<F>
where
    F: Scalar,
    So: Solver<P, PointState<DVector<F>, F>, Error = OracleError>
        + LeastSquaresDiagnostics<F>,
{
    let start = Instant::now();
    let mut stepper = match executor.into_stepper() {
        Ok(stepper) => stepper,
        Err(error) => {
            return Measured {
                run: Measurement {
                    recommendations: Vec::new(),
                    counts: None,
                    outcome: RunOutcome::InitializationError(error),
                    ledger: ledger.snapshot(),
                    elapsed: start.elapsed(),
                },
                native: Vec::new(),
            };
        }
    };
    let mut recommendations = vec![recommendation(
        stepper.state(),
        stepper.counts(),
        ledger,
        PublicationStage::Initialization,
    )];
    let mut native = Vec::new();
    let mut sequence = 0;
    let outcome = loop {
        if start.elapsed() >= wall_cap {
            break RunOutcome::WallLimit;
        }
        let iteration = stepper.state().iter();
        let before = ledger.work();
        let result = stepper.step();
        let leaves = ledger.snapshot();
        let mut trials = leaves
            .calls
            .iter()
            .filter(|c| c.work > before && c.kind == OracleKind::Residual);
        let observations: Vec<_> = stepper
            .solver()
            .least_squares_observations()
            .iter()
            .filter(|o| o.sequence > sequence)
            .cloned()
            .collect();
        let model_work =
            trials.clone().next().map_or(leaves.work(), |c| c.work - 1);
        for observation in observations {
            sequence = observation.sequence;
            let leaf = observation.trial.then(|| trials.next()).flatten();
            let published = result.is_ok()
                && observation.accepted == Some(true)
                && leaf.is_some_and(|c| {
                    recommendation(
                        stepper.state(),
                        stepper.counts(),
                        ledger,
                        PublicationStage::Boundary,
                    )
                    .point
                        == c.point
                });
            let work_before = leaf.map_or(
                if observation.trial {
                    leaves.work()
                } else {
                    model_work
                },
                |c| c.work - 1,
            );
            native.push(NativeRecord {
                observation,
                starting_iteration: iteration,
                work_before,
                residual_work: leaf.map(|c| c.work),
                published,
            });
        }
        match result {
            Ok(StepOutcome::Continue) => recommendations.push(recommendation(
                stepper.state(),
                stepper.counts(),
                ledger,
                PublicationStage::Boundary,
            )),
            Ok(StepOutcome::Stopped(report)) => {
                recommendations.push(recommendation(
                    stepper.state(),
                    stepper.counts(),
                    ledger,
                    PublicationStage::Stop(report.stage),
                ));
                break RunOutcome::Stopped(report);
            }
            Err(error) => break RunOutcome::StepError(error),
        }
    };
    Measured {
        run: Measurement {
            recommendations,
            outcome,
            counts: Some(*stepper.counts()),
            ledger: ledger.snapshot(),
            elapsed: start.elapsed(),
        },
        native,
    }
}
