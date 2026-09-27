use std::{
    convert::Infallible,
    ops::{Index, IndexMut},
};

use basin::{
    BasicPopulationState, BoxConstraints, CostFunction, De, Executor,
    PopulationState, SampleUniformBox, Scalar, ScaleInPlace, ScaledAdd, State,
    VectorLen,
};
use rand_distr::uniform::SampleUniform;

struct Quadratic<V> {
    lower: V,
    upper: V,
}

impl<V, F> CostFunction for Quadratic<V>
where
    V: VectorLen + Index<usize, Output = F>,
    F: Scalar,
{
    type Param = V;
    type Output = F;
    type Error = Infallible;

    fn cost(&self, x: &V) -> Result<Self::Output, Infallible> {
        let quarter = F::from_f64(0.25).unwrap();
        Ok((0..x.vec_len())
            .map(|i| F::from_usize(i + 1).unwrap() * (x[i] - quarter).powi(2))
            .sum())
    }
}

impl<V, F> BoxConstraints for Quadratic<V>
where
    V: VectorLen + Index<usize, Output = F>,
    F: Scalar,
{
    fn lower(&self) -> &V {
        &self.lower
    }
    fn upper(&self) -> &V {
        &self.upper
    }
}

pub fn check<V, F>(make: fn(&[F]) -> V)
where
    F: Scalar + SampleUniform + Send + Sync,
    V: Clone
        + VectorLen
        + SampleUniformBox
        + ScaledAdd<F>
        + ScaleInPlace<F>
        + Index<usize, Output = F>
        + IndexMut<usize, Output = F>
        + Sync,
{
    let num = |x| F::from_f64(x).unwrap();
    for mutation in super::MUTATIONS {
        for crossover in super::CROSSOVERS {
            let problem = Quadratic {
                lower: make(&[num(-2.0), num(-3.0)]),
                upper: make(&[num(3.0), num(4.0)]),
            };
            let solver = De::new(42)
                .with_pop_size(24)
                .with_mutation(mutation)
                .with_crossover(crossover)
                .with_dither(num(0.5), num(1.0));
            let mut stepper = Executor::new(
                problem,
                solver,
                BasicPopulationState::with_size(1),
            )
            .max_iter(200)
            .into_stepper()
            .unwrap();
            let mut previous = stepper.state().cost();
            for _ in 0..200 {
                stepper.step().unwrap();
                let state = stepper.state();
                assert!(state.cost() <= previous);
                previous = state.cost();
                assert!(
                    state.costs().windows(2).all(|pair| pair[0] <= pair[1])
                );
                for (x, &cost) in state.candidates().iter().zip(state.costs()) {
                    assert!(x[0] >= num(-2.0) && x[0] <= num(3.0));
                    assert!(x[1] >= num(-3.0) && x[1] <= num(4.0));
                    let expected = (x[0] - num(0.25)).powi(2)
                        + num(2.0) * (x[1] - num(0.25)).powi(2);
                    assert_eq!(cost, expected);
                }
            }
            let tolerance = if F::epsilon() > num(1e-10) {
                num(1e-5)
            } else {
                num(1e-12)
            };
            assert!(
                stepper.state().cost() < tolerance,
                "{mutation:?} {crossover:?}: {:?}",
                stepper.state().cost()
            );
            assert_eq!(stepper.state().cost_evals(), 24 * 201);
        }
    }
}
