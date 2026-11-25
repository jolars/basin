use std::convert::Infallible;

use basin::{
    Checkpoint, CostFunction, CountsMirror, EvalCounts, Executor, Problem,
    Solver, SolverStep, State, StepOutcome,
};
use bincode_next::{Decode, Encode, Fingerprint};
use rkyv::{Archive, Deserialize, Serialize, rancor::Error};
use wincode::{SchemaRead, SchemaWrite};

#[derive(Debug, PartialEq)]
struct Halving {
    scale: f64,
}

#[derive(Debug, PartialEq)]
struct Progress {
    point: f64,
    cost: f64,
    iter: u64,
    cost_evals: u64,
}

impl State for Progress {
    type Param = f64;
    type Float = f64;

    fn iter(&self) -> u64 {
        self.iter
    }

    fn increment_iter(&mut self) {
        self.iter += 1;
    }

    fn param(&self) -> &f64 {
        &self.point
    }

    fn update_best(&mut self) {}
    fn reset_best(&mut self) {}
}

impl CountsMirror for Progress {
    fn cost_evals(&self) -> u64 {
        self.cost_evals
    }
    fn mirror(&mut self, counts: &EvalCounts) {
        self.cost_evals = counts.cost_evals;
    }
}

struct Square;

impl CostFunction for Square {
    type Param = f64;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &f64) -> Result<f64, Infallible> {
        Ok(x * x)
    }
}

impl Solver<Square, Progress> for Halving {
    type Error = Infallible;

    fn init(
        &mut self,
        problem: &mut Problem<Square>,
        mut state: Progress,
    ) -> Result<Progress, Infallible> {
        self.scale = 0.5;
        state.cost = problem.cost(&state.point)?;
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<Square>,
        mut state: Progress,
    ) -> Result<SolverStep<Progress>, Infallible> {
        state.point *= self.scale;
        self.scale *= 0.75;
        state.cost = problem.cost(&state.point)?;
        Ok(SolverStep::completed(state))
    }
}

#[derive(
    Archive,
    Deserialize,
    Serialize,
    Decode,
    Encode,
    Fingerprint,
    SchemaRead,
    SchemaWrite,
)]
struct WireCheckpointV1 {
    scale: f64,
    point: f64,
    cost: f64,
    iter: u64,
    eval_counts: [u64; 6],
}

impl From<Checkpoint<Halving, Progress>> for WireCheckpointV1 {
    fn from(checkpoint: Checkpoint<Halving, Progress>) -> Self {
        let (solver, state, counts) = checkpoint.into_parts();
        Self {
            scale: solver.scale,
            point: state.point,
            cost: state.cost,
            iter: state.iter,
            eval_counts: [
                counts.cost_evals,
                counts.gradient_evals,
                counts.residual_evals,
                counts.jacobian_evals,
                counts.hessian_evals,
                counts.hessian_product_evals,
            ],
        }
    }
}

impl From<WireCheckpointV1> for Checkpoint<Halving, Progress> {
    fn from(wire: WireCheckpointV1) -> Self {
        let mut counts = EvalCounts::default();
        [
            counts.cost_evals,
            counts.gradient_evals,
            counts.residual_evals,
            counts.jacobian_evals,
            counts.hessian_evals,
            counts.hessian_product_evals,
        ] = wire.eval_counts;
        Self::from_parts(
            Halving { scale: wire.scale },
            Progress {
                point: wire.point,
                cost: wire.cost,
                iter: wire.iter,
                cost_evals: counts.cost_evals,
            },
            counts,
        )
    }
}

#[test]
fn native_codecs_resume_an_application_solver() {
    let make = || {
        Executor::new(
            Square,
            Halving { scale: 0.0 },
            Progress {
                point: 2.0,
                cost: f64::INFINITY,
                iter: 0,
                cost_evals: 0,
            },
        )
        .max_iter(7)
        .into_stepper()
        .unwrap()
    };
    let mut reference = make();
    let mut split = make();
    for _ in 0..3 {
        assert_eq!(reference.step().unwrap(), StepOutcome::Continue);
        assert_eq!(split.step().unwrap(), StepOutcome::Continue);
    }
    let wire = WireCheckpointV1::from(split.into_checkpoint().unwrap());
    assert_eq!(wire.eval_counts, [4, 0, 0, 0, 0, 0]);

    let rkyv_bytes = rkyv::to_bytes::<Error>(&wire).unwrap();
    let archived =
        rkyv::from_bytes::<WireCheckpointV1, Error>(&rkyv_bytes).unwrap();
    let wincode_bytes = wincode::serialize(&wire).unwrap();
    let wincode =
        wincode::deserialize_exact::<WireCheckpointV1>(&wincode_bytes).unwrap();
    let config = bincode_next::config::standard().with_fingerprint();
    let bincode_bytes = bincode_next::encode_to_vec(&wire, config).unwrap();
    let (bincode, consumed) = bincode_next::decode_from_slice::<
        WireCheckpointV1,
        _,
    >(&bincode_bytes, config)
    .unwrap();
    assert_eq!(consumed, bincode_bytes.len());

    let mut resumed = [archived, wincode, bincode].map(|wire| {
        Executor::resume_from_checkpoint(Square, wire.into())
            .max_iter(7)
            .into_stepper()
            .unwrap()
    });
    loop {
        for stepper in &resumed {
            assert_eq!(stepper.solver(), reference.solver());
            assert_eq!(stepper.state(), reference.state());
            assert_eq!(stepper.counts(), reference.counts());
        }
        let expected = reference.step().unwrap();
        for stepper in &mut resumed {
            assert_eq!(stepper.step().unwrap(), expected);
        }
        if expected.report().is_some() {
            assert_eq!(reference.iter(), 7);
            assert_eq!(reference.counts().cost_evals, 8);
            break;
        }
    }
}
