use argmin::core::{CostFunction, PopulationState, Problem, Solver, State};
use argmin::solver::particleswarm::{Particle, ParticleSwarm};
use rand::RngCore;

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
                0x1000_0000_0000_0000,
            ],
            next: 0,
        }
    }
}

impl RngCore for SequenceRng {
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    fn next_u64(&mut self) -> u64 {
        let value = self.values[self.next];
        self.next += 1;
        value
    }

    fn fill_bytes(&mut self, dst: &mut [u8]) {
        for chunk in dst.chunks_mut(8) {
            let bytes = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
    }
}

struct Sphere;

impl CostFunction for Sphere {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, x: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        Ok(x.iter().map(|xi| xi * xi).sum())
    }
}

fn main() -> Result<(), argmin::core::Error> {
    let particles = vec![
        Particle::new(vec![-0.5, 0.25], 0.3125, vec![0.2, -0.1]),
        Particle::new(vec![0.75, -0.5], 0.8125, vec![-0.3, 0.4]),
    ];
    let state = PopulationState::new().population(particles);
    let mut solver = ParticleSwarm::new((vec![-1.0; 2], vec![1.0; 2]), 2)
        .with_rng_generator(SequenceRng::new())
        .with_inertia_factor(0.7)?
        .with_cognitive_factor(1.2)?
        .with_social_factor(1.4)?;
    let mut problem = Problem::new(Sphere);
    let (state, _) = solver.init(&mut problem, state)?;
    let (state, _) = solver.next_iter(&mut problem, state)?;

    let mut rows: Vec<_> = state
        .get_population()
        .unwrap()
        .iter()
        .map(|particle| {
            let cost = particle.position.iter().map(|xi| xi * xi).sum::<f64>();
            (cost, particle.position.clone())
        })
        .collect();
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (cost, position) in rows {
        println!(
            "{cost:.17e},{:.17e},{:.17e}",
            position[0], position[1]
        );
    }
    Ok(())
}
