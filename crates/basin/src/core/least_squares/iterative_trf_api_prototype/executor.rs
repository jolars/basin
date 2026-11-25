use super::*;
use crate::State;

type Fit = Fixture<Vec<f64>, Vec<f64>, f64>;
type Objective = RobustLeastSquares<Fit, SoftL1Loss>;
type CountedObjective = RawCallbacks<Objective, f64>;

fn objective(fail: Option<u8>) -> CountedObjective {
    let mut fit = Fixture::new(<[f64]>::to_vec, <[f64]>::to_vec);
    fit.fail = fail;
    RawCallbacks(
        RobustLeastSquares::new(fit, SoftL1Loss),
        std::marker::PhantomData,
    )
}

#[derive(Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct ModelDriver {
    residual: Vec<f64>,
    factors: Vec<f64>,
    #[cfg_attr(feature = "serde", serde(skip))]
    scratch: OperatorScratch,
}

#[derive(Default)]
struct OperatorScratch {
    trial: Option<Vec<f64>>,
    product: Option<Vec<f64>>,
    transpose: Option<Vec<f64>>,
}

impl<P> crate::Solver<RawCallbacks<P, f64>, crate::PointState<Vec<f64>>>
    for ModelDriver
where
    P: OperatorObjective<f64>,
    P::Raw: BoxProblem<Param = Vec<f64>>
        + Residual<Param = Vec<f64>, Output = Vec<f64>, Error = &'static str>,
{
    type Error = &'static str;

    fn init(
        &mut self,
        problem: &mut Problem<RawCallbacks<P, f64>>,
        mut state: crate::PointState<Vec<f64>>,
    ) -> Result<crate::PointState<Vec<f64>>, Self::Error> {
        let x = state.param().clone();
        let model = linearize(problem, x.clone())?.unwrap();
        state.replace(x, model.cost);
        self.residual = model.residual;
        self.factors = model.factors;
        self.scratch = OperatorScratch::default();
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<RawCallbacks<P, f64>>,
        state: crate::PointState<Vec<f64>>,
    ) -> Result<crate::SolverStep<crate::PointState<Vec<f64>>>, Self::Error>
    {
        let x = state.param();
        let mut trial = x.clone();
        trial[0] = 5.0;
        let r = residual_reusing(problem, &trial, &mut self.scratch.trial)?;
        assert!(problem.inner().0.cost(r).is_infinite());
        for direction in [vec![1.0, 0.0], vec![0.0, 1.0]] {
            if let Some(out) = &mut self.scratch.product {
                problem.jacobian_product_into(x, &direction, out)?;
            } else {
                self.scratch.product =
                    Some(problem.jacobian_product(x, &direction)?);
            }
            multiply(self.scratch.product.as_mut().unwrap(), &self.factors);
        }
        let mut weighted = self.residual.clone();
        multiply(&mut weighted, &self.factors);
        if let Some(out) = &mut self.scratch.transpose {
            problem.transpose_jacobian_product_into(x, &weighted, out)?;
        } else {
            self.scratch.transpose =
                Some(problem.transpose_jacobian_product(x, &weighted)?);
        }
        assert!(
            self.scratch
                .product
                .as_ref()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(
            self.scratch
                .transpose
                .as_ref()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
        Ok(crate::SolverStep::completed(state))
    }
}

fn residual_reusing<'a, P: Residual>(
    problem: &mut Problem<P>,
    x: &P::Param,
    scratch: &'a mut Option<P::Output>,
) -> Result<&'a P::Output, P::Error> {
    if let Some(out) = scratch {
        problem.residual_into(x, out)?;
    } else {
        *scratch = Some(problem.residual(x)?);
    }
    Ok(scratch.as_ref().unwrap())
}

fn controlled<S: crate::RawEvaluationState>(
    control: crate::RunControl<S>,
) -> crate::RunControl<S> {
    use crate::EvaluationKind::*;
    control
        .max_evaluations(JacobianProduct, 4)
        .max_evaluations(TransposeJacobianProduct, 2)
        .max_evaluations(TotalWork, 9)
}

#[test]
fn executor_budgets_and_exact_resume() {
    use crate::{Executor, ObserverMode, TerminationCode};
    let seen = Rc::new(RefCell::new(Vec::new()));
    let trace = seen.clone();
    let objective_full = objective(None);
    let in_place = objective_full.0.inner().in_place.clone();
    let full = Executor::new(
        objective_full,
        ModelDriver::default(),
        crate::PointState::new(vec![1.0, -0.5]),
    )
    .max_evaluations(crate::EvaluationKind::JacobianProduct, 4)
    .max_evaluations(crate::EvaluationKind::TransposeJacobianProduct, 2)
    .max_evaluations(crate::EvaluationKind::TotalWork, 9)
    .observe_solver(
        move |state, _, _| trace.borrow_mut().push(*state.counts()),
        ObserverMode::Always,
    )
    .run_with_solver()
    .unwrap();
    assert_eq!(full.iter(), 2);
    assert_eq!(in_place.each_ref().map(std::cell::Cell::get), [1, 3, 1]);
    use crate::{EvaluationKind, ExecutionLimit, Termination};
    assert_eq!(
        full.report.termination,
        Termination::Limit(vec![
            ExecutionLimit::Evaluations {
                kind: EvaluationKind::JacobianProduct,
                observed: 4,
                limit: 4
            },
            ExecutionLimit::Evaluations {
                kind: EvaluationKind::TransposeJacobianProduct,
                observed: 2,
                limit: 2
            },
            ExecutionLimit::Evaluations {
                kind: EvaluationKind::TotalWork,
                observed: 9,
                limit: 9
            },
        ])
    );
    assert_eq!(full.report.code(), TerminationCode::MaxEvaluations);
    assert_eq!(full.counts.jacobian_product_evals, 4);
    assert_eq!(full.counts.transpose_jacobian_product_evals, 2);
    assert_eq!(full.counts.total_work(), 9);
    assert_eq!(full.state.counts(), &full.counts);
    assert_eq!(seen.borrow().last(), Some(&full.counts));

    let split_run = Executor::new(
        objective(None),
        ModelDriver::default(),
        crate::PointState::new(vec![1.0, -0.5]),
    );
    #[cfg(all(feature = "checkpoint-postcard", not(target_arch = "wasm32")))]
    let path = std::env::temp_dir().join(format!(
        "basin-operator-checkpoint-{}.ckpt",
        std::process::id()
    ));
    #[cfg(all(feature = "checkpoint-postcard", not(target_arch = "wasm32")))]
    let split_run = split_run.checkpoint_with(
        crate::ExactCheckpointWriter::new(&path),
        ObserverMode::Always,
    );
    let mut stepper = split_run.into_stepper().unwrap();
    stepper.step().unwrap();
    let checkpoint = stepper.into_checkpoint().unwrap();
    assert_eq!(checkpoint.state().iter(), 1);
    assert_eq!(checkpoint.state().counts(), checkpoint.counts());
    #[cfg(feature = "serde")]
    let restored =
        postcard::from_bytes(&postcard::to_allocvec(&checkpoint).unwrap())
            .unwrap();
    let resume = |checkpoint: crate::ExactCheckpoint<
        ModelDriver,
        crate::PointState<Vec<f64>>,
    >| {
        let result =
            Executor::resume_from_checkpoint(objective(None), checkpoint)
                .max_evaluations(EvaluationKind::JacobianProduct, 4)
                .max_evaluations(EvaluationKind::TransposeJacobianProduct, 2)
                .max_evaluations(EvaluationKind::TotalWork, 9)
                .run_with_solver()
                .unwrap();
        assert_eq!(result.iter(), full.iter());
        assert_eq!(result.cost(), full.cost());
        assert_eq!(result.param(), full.param());
        assert_eq!(result.solver.residual, full.solver.residual);
        assert_eq!(result.solver.factors, full.solver.factors);
        assert_eq!(result.report, full.report);
        assert_eq!(result.counts, full.counts);
        assert_eq!(result.state.counts(), &full.counts);
    };
    resume(checkpoint);
    #[cfg(feature = "serde")]
    resume(restored);
    #[cfg(all(feature = "checkpoint-postcard", not(target_arch = "wasm32")))]
    {
        let restored = crate::read_exact_checkpoint(&path);
        std::fs::remove_file(&path).unwrap();
        resume(restored.unwrap());
    }
}

#[test]
fn nested_product_counts_are_relative_and_mergeable() {
    use crate::core::executor::run_loop_with_control;
    use crate::{PointState, RunControl};
    let mut outer = Problem::new(objective(None));
    let x = vec![1.0, -0.5];
    outer.jacobian_product(&x, &x).unwrap();
    outer.transpose_jacobian_product(&x, &vec![1.0; 3]).unwrap();
    let baseline = *outer.counts();
    let mut driver = ModelDriver::default();
    let mut control = controlled(RunControl::new());
    let inner = run_loop_with_control(
        &mut outer,
        PointState::new(x.clone()),
        &mut driver,
        &mut control,
    )
    .unwrap();
    assert_eq!(inner.counts, outer.counts().delta_since(&baseline));
    assert_eq!(inner.state.counts(), &inner.counts);
    assert_eq!(inner.counts.jacobian_product_evals, 4);
    assert_eq!(inner.counts.transpose_jacobian_product_evals, 2);

    let mut adapter = counted(objective(None).0.into_inner());
    let adapted = run_loop_with_control(
        &mut adapter,
        PointState::new(x),
        &mut driver,
        &mut control,
    )
    .unwrap();
    assert_eq!(adapted.counts, inner.counts);
    let baseline = *outer.counts();
    outer.counts_mut().add(adapter.counts());
    assert_eq!(outer.counts().delta_since(&baseline), inner.counts);
}

#[test]
fn internal_probes_stay_out_of_executor_budgets() {
    let fit = Fixture::new(<[f64]>::to_vec, <[f64]>::to_vec);
    let trace = fit.trace.clone();
    let problem = RawCallbacks(
        RobustLeastSquares::new(ProbingProducts(fit), SoftL1Loss),
        std::marker::PhantomData,
    );
    let run = crate::Executor::new(
        problem,
        ModelDriver::default(),
        crate::PointState::new(vec![1.0, -0.5]),
    )
    .max_evaluations(crate::EvaluationKind::TotalWork, 5)
    .run()
    .unwrap();
    assert_eq!(run.iter(), 1);
    assert_eq!(run.counts.residual_evals, 2);
    assert_eq!(run.counts.jacobian_evals, 0);
    assert_eq!(run.counts.jacobian_product_evals, 2);
    assert_eq!(run.counts.transpose_jacobian_product_evals, 1);
    assert_eq!(run.counts.total_work(), 5);
    assert_eq!(trace.borrow().len(), 9);
}

#[derive(Clone, Default)]
struct CheckpointCapture(Rc<RefCell<Vec<crate::EvalCounts>>>);

impl crate::CheckpointSink<ModelDriver, crate::PointState<Vec<f64>>>
    for CheckpointCapture
{
    fn save(
        &mut self,
        _: &ModelDriver,
        state: &crate::PointState<Vec<f64>>,
        counts: &crate::EvalCounts,
    ) {
        assert_eq!(state.counts(), counts);
        self.0.borrow_mut().push(*counts);
    }
}

#[test]
fn executor_product_failure_preserves_charge_and_discards_state() {
    for kind in [1, 2] {
        let capture = CheckpointCapture::default();
        let observations = Rc::new(std::cell::Cell::new(0));
        let seen = observations.clone();
        let mut run = crate::Executor::new(
            objective(Some(kind)),
            ModelDriver::default(),
            crate::PointState::new(vec![1.0, -0.5]),
        )
        .observe_solver(
            move |_, _, _| seen.set(seen.get() + 1),
            crate::ObserverMode::Always,
        )
        .checkpoint_with(capture.clone(), crate::ObserverMode::Always)
        .into_stepper()
        .unwrap();
        let before = observations.get();
        assert_eq!(run.step().unwrap_err(), "operator callback failed");
        assert_eq!(
            run.counts().jacobian_product_evals,
            if kind == 1 { 1 } else { 2 }
        );
        assert_eq!(
            run.counts().transpose_jacobian_product_evals,
            u64::from(kind == 2)
        );
        assert_eq!(observations.get(), before);
        assert!(capture.0.borrow().is_empty());
        assert!(run.into_checkpoint().is_none());
    }
}

struct BorrowedResidual<'a, P>(&'a P);

impl<P: Residual> Residual for BorrowedResidual<'_, P> {
    type Param = P::Param;
    type Output = P::Output;
    type Error = P::Error;

    fn residual(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
        self.0.residual(x)
    }

    fn residual_into(
        &self,
        x: &Self::Param,
        out: &mut Self::Output,
    ) -> Result<(), Self::Error> {
        self.0.residual_into(x, out)
    }
}

struct ResidualOnly(Fit);

impl Residual for ResidualOnly {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = &'static str;

    fn residual(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
        self.0.residual(x)
    }

    fn residual_into(
        &self,
        x: &Self::Param,
        out: &mut Self::Output,
    ) -> Result<(), Self::Error> {
        self.0.residual_into(x, out)
    }
}

#[test]
fn residual_only_robust_scratch_is_seeded_and_overwritten() {
    let raw = ResidualOnly(Fixture::new(<[f64]>::to_vec, <[f64]>::to_vec));
    let loss = RobustLeastSquares::<_, _, f64>::new(raw, SoftL1Loss);
    let mut problem = Problem::new(BorrowedResidual(loss.inner()));
    let mut scratch = None;
    let first =
        residual_reusing(&mut problem, &vec![1.0, -0.5], &mut scratch).unwrap();
    let storage = first.as_ptr();
    let cost = loss.residual_cost(first);
    let second =
        residual_reusing(&mut problem, &vec![2.0, 0.0], &mut scratch).unwrap();
    assert_eq!(second.as_ptr(), storage);
    assert_ne!(loss.residual_cost(second), cost);
    assert_eq!(loss.inner().0.in_place[0].get(), 1);
    assert_eq!(problem.counts().residual_evals, 2);
}

#[test]
fn dense_robust_scratch_preserves_fused_dispatch_with_custom_matrix() {
    use super::evaluation::Evaluation;
    struct Matrix([f64; 2]);
    impl crate::ScaleRowsInPlace for Matrix {
        fn scale_rows_in_place(&mut self, factors: &[f64]) {
            for (row, factor) in self.0.iter_mut().zip(factors) {
                *row *= factor;
            }
        }
    }
    struct Dense {
        fused_into: std::cell::Cell<u64>,
    }
    impl Residual for Dense {
        type Param = Vec<f64>;
        type Output = Vec<f64>;
        type Error = std::convert::Infallible;
        fn residual(&self, _: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
            panic!("expected fused callback")
        }
    }
    impl Jacobian for Dense {
        type Jacobian = Matrix;
        fn jacobian(&self, _: &Vec<f64>) -> Result<Matrix, Self::Error> {
            panic!("expected fused callback")
        }
        fn residual_and_jacobian(
            &self,
            x: &Vec<f64>,
        ) -> Result<(Vec<f64>, Matrix), Self::Error> {
            Ok((vec![x[0], x[0] * x[0]], Matrix([1.0, 2.0 * x[0]])))
        }
        fn residual_and_jacobian_into(
            &self,
            x: &Vec<f64>,
            r: &mut Vec<f64>,
            j: &mut Matrix,
        ) -> Result<(), Self::Error> {
            self.fused_into.set(self.fused_into.get() + 1);
            r.copy_from_slice(&[x[0], x[0] * x[0]]);
            j.0 = [1.0, 2.0 * x[0]];
            Ok(())
        }
    }
    let mut problem = Problem::new(RobustLeastSquares::<_, _, f64>::new(
        Dense {
            fused_into: 0.into(),
        },
        SoftL1Loss,
    ));
    let mut scratch = crate::EvaluationBuffers::default();
    for x in [1.0, 2.0] {
        let (residual, matrix) = problem
            .residual_and_jacobian_reusing(
                &vec![x],
                scratch.vector.take(),
                scratch.matrix.take(),
            )
            .unwrap();
        assert_eq!(residual, vec![x, x * x]);
        assert_eq!(matrix.0, [1.0, 2.0 * x]);
        assert!(problem.model(&residual, Matrix(matrix.0)).is_some());
        scratch.vector = Some(residual);
        scratch.matrix = Some(matrix);
    }
    assert_eq!(problem.inner().inner().fused_into.get(), 1);
    assert_eq!(problem.counts().residual_evals, 2);
    assert_eq!(problem.counts().jacobian_evals, 2);
}

#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
#[test]
fn independent_scratch_allows_shared_robust_callbacks_to_overlap() {
    use std::sync::{Condvar, Mutex};
    use std::time::Duration;

    #[derive(Default)]
    struct SharedResidual {
        arrivals: Mutex<u32>,
        ready: Condvar,
    }
    impl Residual for SharedResidual {
        type Param = f64;
        type Output = Vec<f64>;
        type Error = std::convert::Infallible;
        fn residual(&self, x: &f64) -> Result<Vec<f64>, Self::Error> {
            Ok(vec![*x])
        }
        fn residual_into(
            &self,
            x: &f64,
            out: &mut Vec<f64>,
        ) -> Result<(), Self::Error> {
            let mut arrivals = self.arrivals.lock().unwrap();
            *arrivals += 1;
            self.ready.notify_all();
            let (arrivals, _) = self
                .ready
                .wait_timeout_while(
                    arrivals,
                    Duration::from_secs(5),
                    |arrivals| *arrivals < 2,
                )
                .unwrap();
            assert_eq!(*arrivals, 2);
            out[0] = *x;
            Ok(())
        }
    }

    let objective = RobustLeastSquares::<_, _, f64>::new(
        SharedResidual::default(),
        SoftL1Loss,
    );
    std::thread::scope(|scope| {
        for x in [1.0, 2.0] {
            let objective = &objective;
            scope.spawn(move || {
                let mut problem =
                    Problem::new(BorrowedResidual(objective.inner()));
                let mut scratch = None;
                residual_reusing(&mut problem, &0.0, &mut scratch).unwrap();
                let residual =
                    residual_reusing(&mut problem, &x, &mut scratch).unwrap();
                assert_eq!(residual, &[x]);
                assert!(objective.residual_cost(residual) > 0.0);
                assert_eq!(problem.counts().residual_evals, 2);
            });
        }
    });
}
