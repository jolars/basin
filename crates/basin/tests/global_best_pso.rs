use std::convert::Infallible;

use basin::{
    BoxConstraints, CostFunction, Executor, GlobalBestPso, GlobalBestPsoState,
    PopulationState, PsoBoundaryHandling, PsoVelocityLimit, State,
    TerminationReason,
};
use rand::TryRng;

#[derive(Clone)]
struct BoundedSphere<F> {
    lower: Vec<F>,
    upper: Vec<F>,
}

impl<F> CostFunction for BoundedSphere<F>
where
    F: basin::Scalar,
{
    type Param = Vec<F>;
    type Output = F;
    type Error = Infallible;

    fn cost(&self, x: &Vec<F>) -> Result<F, Self::Error> {
        Ok(x.iter().map(|xi| *xi * *xi).sum())
    }
}

impl<F> BoxConstraints for BoundedSphere<F>
where
    F: basin::Scalar,
{
    fn lower(&self) -> &Vec<F> {
        &self.lower
    }

    fn upper(&self) -> &Vec<F> {
        &self.upper
    }
}

#[derive(Clone, Debug)]
struct ConstantRng(u64);

impl TryRng for ConstantRng {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(self.0 as u32)
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        Ok(self.0)
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        for chunk in dst.chunks_mut(8) {
            let bytes = self.0.to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct SequenceRng {
    values: [u64; 8],
    next: usize,
}

impl SequenceRng {
    fn new() -> Self {
        Self {
            values: [
                0x2000_0000_0000_0000,
                0x4000_0000_0000_0000,
                0x6000_0000_0000_0000,
                0x8000_0000_0000_0000,
                0xa000_0000_0000_0000,
                0xc000_0000_0000_0000,
                0xe000_0000_0000_0000,
                0x4000_0000_0000_0000,
            ],
            next: 0,
        }
    }
}

impl TryRng for SequenceRng {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(self.try_next_u64()? as u32)
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        let value = self.values[self.next];
        self.next += 1;
        Ok(value)
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        for chunk in dst.chunks_mut(8) {
            let bytes = self.try_next_u64()?.to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
        Ok(())
    }
}

fn one_dimensional_problem() -> BoundedSphere<f64> {
    BoundedSphere {
        lower: vec![-1.0],
        upper: vec![1.0],
    }
}

#[test]
fn defaults_follow_the_standard_pso_coefficient_profile() {
    let solver = GlobalBestPso::<f64>::new(42);

    assert_eq!(GlobalBestPso::<f64>::default_swarm_size(0), 10);
    assert_eq!(GlobalBestPso::<f64>::default_swarm_size(9), 16);
    assert!((solver.inertia() - 1.0 / (2.0 * 2.0_f64.ln())).abs() < 1e-15);
    assert!((solver.cognitive() - (0.5 + 2.0_f64.ln())).abs() < 1e-15);
    assert!((solver.social() - (0.5 + 2.0_f64.ln())).abs() < 1e-15);
}

#[test]
fn warm_start_updates_velocity_and_position_synchronously() {
    let solver = GlobalBestPso::new_with_rng(ConstantRng(0))
        .with_swarm_size(2)
        .with_inertia(0.5)
        .with_cognitive(0.0)
        .with_social(0.0)
        .with_boundary_handling(PsoBoundaryHandling::Preserve);
    let state = GlobalBestPsoState::from_positions_and_velocities(
        vec![vec![-0.5], vec![0.5]],
        vec![vec![0.2], vec![-0.4]],
    );

    let result = Executor::new(one_dimensional_problem(), solver, state)
        .max_iter(1)
        .run()
        .unwrap();

    assert_eq!(result.state.candidates(), &[vec![0.3], vec![-0.4]]);
    assert_eq!(result.state.velocities(), &[vec![-0.2], vec![0.1]]);
    assert!((result.state.personal_best_costs()[0] - 0.09).abs() < 1e-15);
    assert!((result.state.personal_best_costs()[1] - 0.16).abs() < 1e-15);
    assert_eq!(result.state.global_best_position(), &vec![0.3]);
    assert_eq!(result.cost_evals(), 4);
}

#[test]
fn positions_only_warm_start_uses_half_displacement_velocity() {
    let solver = GlobalBestPso::new_with_rng(ConstantRng(0));
    let state = GlobalBestPsoState::from_positions(vec![vec![0.5]]);
    let result = Executor::new(
        BoundedSphere {
            lower: vec![0.0],
            upper: vec![1.0],
        },
        solver,
        state,
    )
    .max_iter(0)
    .run()
    .unwrap();

    assert_eq!(result.state.candidates(), &[vec![0.5]]);
    assert_eq!(result.state.velocities(), &[vec![-0.25]]);
}

#[test]
fn one_generation_matches_argmin_0_11_global_best_reference() {
    let solver = GlobalBestPso::new_with_rng(SequenceRng::new())
        .with_swarm_size(2)
        .with_inertia(0.7)
        .with_cognitive(1.2)
        .with_social(1.4)
        .with_boundary_handling(PsoBoundaryHandling::Preserve);
    let state = GlobalBestPsoState::from_positions_and_velocities(
        vec![vec![-0.5, 0.25], vec![0.75, -0.5]],
        vec![vec![0.2, -0.1], vec![-0.3, 0.4]],
    );
    let result = Executor::new(
        BoundedSphere {
            lower: vec![-1.0; 2],
            upper: vec![1.0; 2],
        },
        solver,
        state,
    )
    .max_iter(1)
    .run()
    .unwrap();

    let fixture = include_str!("fixtures/global_best_pso_argmin_0_11.csv");
    let expected: Vec<Vec<f64>> = fixture
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            line.split(',')
                .map(|field| field.parse::<f64>().unwrap())
                .collect()
        })
        .collect();
    for ((position, &cost), row) in result
        .state
        .candidates()
        .iter()
        .zip(result.state.costs())
        .zip(&expected)
    {
        assert!(
            (cost - row[0]).abs() < 2e-15,
            "cost {cost:?}, reference {:?}, position {position:?}",
            row[0]
        );
        assert!((position[0] - row[1]).abs() < 2e-15);
        assert!((position[1] - row[2]).abs() < 2e-15);
    }
}

#[test]
fn boundary_and_velocity_policies_are_orthogonal() {
    let run = |boundary| {
        let solver = GlobalBestPso::new_with_rng(ConstantRng(0))
            .with_swarm_size(1)
            .with_inertia(1.0)
            .with_cognitive(0.0)
            .with_social(0.0)
            .with_boundary_handling(boundary);
        let state = GlobalBestPsoState::from_positions_and_velocities(
            vec![vec![0.9]],
            vec![vec![0.4]],
        );
        Executor::new(one_dimensional_problem(), solver, state)
            .max_iter(1)
            .run()
            .unwrap()
    };

    assert_eq!(
        run(PsoBoundaryHandling::Absorb).state.velocities(),
        &[vec![0.0]]
    );
    assert_eq!(
        run(PsoBoundaryHandling::Preserve).state.velocities(),
        &[vec![0.4]]
    );
    assert_eq!(
        run(PsoBoundaryHandling::Reflect { damping: 0.5 })
            .state
            .velocities(),
        &[vec![-0.2]]
    );

    let solver = GlobalBestPso::new_with_rng(ConstantRng(0))
        .with_swarm_size(1)
        .with_inertia(1.0)
        .with_cognitive(0.0)
        .with_social(0.0)
        .with_velocity_limit(PsoVelocityLimit::SpanFraction(0.1));
    let state = GlobalBestPsoState::from_positions_and_velocities(
        vec![vec![0.0]],
        vec![vec![0.9]],
    );
    let result = Executor::new(one_dimensional_problem(), solver, state)
        .max_iter(1)
        .run()
        .unwrap();

    assert_eq!(result.state.candidates(), &[vec![0.2]]);
    assert_eq!(result.state.velocities(), &[vec![0.2]]);
}

#[test]
fn span_fraction_limits_velocity_when_box_width_overflows() {
    let lower = -3e38_f32;
    let upper = 3e38_f32;
    assert!((upper - lower).is_infinite());

    let run = |fraction| {
        let solver = GlobalBestPso::new_with_rng(ConstantRng(0))
            .with_swarm_size(1)
            .with_inertia(1.0_f32)
            .with_cognitive(0.0)
            .with_social(0.0)
            .with_velocity_limit(PsoVelocityLimit::SpanFraction(fraction));
        let state = GlobalBestPsoState::from_positions_and_velocities(
            vec![vec![0.0_f32]],
            vec![vec![1e38_f32]],
        );
        Executor::new(
            BoundedSphere {
                lower: vec![lower],
                upper: vec![upper],
            },
            solver,
            state,
        )
        .max_iter(1)
        .run()
        .unwrap()
    };

    let stopped = run(0.0);
    assert_eq!(stopped.state.candidates(), &[vec![0.0]]);
    assert_eq!(stopped.state.velocities(), &[vec![0.0]]);

    let limited = run(0.01);
    let expected = 6e36_f32;
    assert!(
        (limited.state.candidates()[0][0] - expected).abs() < 1e-6 * expected
    );
    assert!(
        (limited.state.velocities()[0][0] - expected).abs() < 1e-6 * expected
    );
}

#[test]
fn non_finite_motion_is_isolated_without_poisoning_the_swarm() {
    let solver = GlobalBestPso::new_with_rng(ConstantRng(0))
        .with_swarm_size(1)
        .with_inertia(f64::MAX)
        .with_cognitive(0.0)
        .with_social(0.0);
    let state = GlobalBestPsoState::from_positions_and_velocities(
        vec![vec![0.0]],
        vec![vec![f64::MAX]],
    );
    let result = Executor::new(one_dimensional_problem(), solver, state)
        .max_iter(1)
        .run()
        .unwrap();

    assert_eq!(result.state.candidates(), &[vec![0.0]]);
    assert_eq!(result.state.velocities(), &[vec![0.0]]);
}

#[test]
fn same_seed_is_reproducible_and_sphere_converges() {
    let run = || {
        Executor::new(
            BoundedSphere {
                lower: vec![-5.0; 3],
                upper: vec![5.0; 3],
            },
            GlobalBestPso::new(1729),
            GlobalBestPsoState::<Vec<f64>>::new(),
        )
        .max_iter(300)
        .run()
        .unwrap()
    };
    let (a, b) = (run(), run());

    assert_eq!(a.param(), b.param());
    assert_eq!(a.cost(), b.cost());
    assert!(a.cost() < 1e-8, "final cost was {}", a.cost());
    assert_eq!(
        a.cost_evals(),
        GlobalBestPso::<f64>::default_swarm_size(3) as u64 * 301
    );
}

#[test]
fn f32_state_solver_and_objective_round_trip() {
    let result = Executor::new(
        BoundedSphere {
            lower: vec![-3.0_f32; 2],
            upper: vec![3.0_f32; 2],
        },
        GlobalBestPso::<f32>::new(91).with_swarm_size(12),
        GlobalBestPsoState::<Vec<f32>, f32>::new(),
    )
    .max_iter(150)
    .run()
    .unwrap();

    assert!(result.cost() < 1e-5);
}

struct Rastrigin {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl CostFunction for Rastrigin {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        Ok(10.0 * x.len() as f64
            + x.iter()
                .map(|xi| xi * xi - 10.0 * (std::f64::consts::TAU * xi).cos())
                .sum::<f64>())
    }
}

impl BoxConstraints for Rastrigin {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }

    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

#[test]
fn converges_on_a_multimodal_rastrigin_problem() {
    let result = Executor::new(
        Rastrigin {
            lower: vec![-5.12; 2],
            upper: vec![5.12; 2],
        },
        GlobalBestPso::new(2025).with_swarm_size(30),
        GlobalBestPsoState::<Vec<f64>>::new(),
    )
    .max_iter(500)
    .run()
    .unwrap();

    assert!(result.cost() < 1e-8, "final cost was {}", result.cost());
}

struct PartialNan {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl CostFunction for PartialNan {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        if x[0] == 0.5 {
            Ok(f64::NAN)
        } else {
            Ok(x[0] * x[0])
        }
    }
}

impl BoxConstraints for PartialNan {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }

    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

#[test]
fn a_finite_evaluation_replaces_a_nan_personal_best() {
    let solver = GlobalBestPso::new_with_rng(ConstantRng(0))
        .with_swarm_size(2)
        .with_inertia(1.0)
        .with_cognitive(0.0)
        .with_social(0.0);
    let state = GlobalBestPsoState::from_positions_and_velocities(
        vec![vec![0.5], vec![0.25]],
        vec![vec![-0.5], vec![0.0]],
    );
    let result = Executor::new(
        PartialNan {
            lower: vec![-1.0],
            upper: vec![1.0],
        },
        solver,
        state,
    )
    .max_iter(1)
    .run()
    .unwrap();

    assert_eq!(result.state.personal_best_costs(), &[0.0, 0.0625]);
    assert_eq!(result.cost(), 0.0);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ApplicationError {
    Abort,
}

struct FailingProblem {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl CostFunction for FailingProblem {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = ApplicationError;

    fn cost(&self, _: &Vec<f64>) -> Result<f64, Self::Error> {
        Err(ApplicationError::Abort)
    }
}

impl BoxConstraints for FailingProblem {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }

    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

#[test]
fn typed_problem_errors_propagate_unchanged() {
    let result = Executor::new(
        FailingProblem {
            lower: vec![-1.0],
            upper: vec![1.0],
        },
        GlobalBestPso::new(3).with_swarm_size(2),
        GlobalBestPsoState::<Vec<f64>>::new(),
    )
    .max_iter(1)
    .run();

    assert!(matches!(result, Err(ApplicationError::Abort)));
}

struct AlwaysNan {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl CostFunction for AlwaysNan {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, _: &Vec<f64>) -> Result<f64, Self::Error> {
        Ok(f64::NAN)
    }
}

impl BoxConstraints for AlwaysNan {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }

    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

#[test]
fn all_non_comparable_initial_costs_stop_cleanly() {
    let result = Executor::new(
        AlwaysNan {
            lower: vec![-1.0],
            upper: vec![1.0],
        },
        GlobalBestPso::new(1).with_swarm_size(3),
        GlobalBestPsoState::<Vec<f64>>::new(),
    )
    .max_iter(2)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverFailed);
    assert_eq!(result.state.iter(), 0);
    assert_eq!(result.cost_evals(), 3);
}

struct AlwaysNegativeInfinity {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl CostFunction for AlwaysNegativeInfinity {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, _: &Vec<f64>) -> Result<f64, Self::Error> {
        Ok(f64::NEG_INFINITY)
    }
}

impl BoxConstraints for AlwaysNegativeInfinity {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }

    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

#[test]
fn negative_infinity_is_a_clean_global_optimum_stop() {
    let result = Executor::new(
        AlwaysNegativeInfinity {
            lower: vec![-1.0],
            upper: vec![1.0],
        },
        GlobalBestPso::new(1).with_swarm_size(2),
        GlobalBestPsoState::<Vec<f64>>::new(),
    )
    .max_iter(2)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert_eq!(result.state.iter(), 0);
}
