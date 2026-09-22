use basin::core::parallel::{MaybeSend, MaybeSync};
use basin::{
    BoxConstraints, CostFunction, Direct, Executor, PointState, Scalar, State,
    TerminationReason, VectorIndex, VectorLen,
};
use std::convert::Infallible;

struct Quadratic<V, F> {
    lower: V,
    upper: V,
    target: Vec<F>,
}

impl<V: VectorIndex<F>, F: Scalar> CostFunction for Quadratic<V, F> {
    type Param = V;
    type Output = F;
    type Error = Infallible;

    fn cost(&self, x: &V) -> Result<F, Infallible> {
        Ok(self
            .target
            .iter()
            .enumerate()
            .map(|(i, &target)| {
                let value = x.get_scalar(i);
                assert!(value >= self.lower.get_scalar(i));
                assert!(value <= self.upper.get_scalar(i));
                (value - target).powi(2)
            })
            .sum())
    }
}

impl<V: VectorIndex<F>, F: Scalar> BoxConstraints for Quadratic<V, F> {
    fn lower(&self) -> &V {
        &self.lower
    }
    fn upper(&self) -> &V {
        &self.upper
    }
}

pub fn check<V, F>(make: fn(&[F]) -> V)
where
    F: Scalar + MaybeSend + MaybeSync,
    V: Clone + VectorIndex<F> + VectorLen + MaybeSync,
{
    let num = |v| F::from_f64(v).unwrap();
    for (lower, upper, target) in [
        (vec![num(-2.0)], vec![num(5.0)], vec![num(0.37)]),
        (
            vec![num(-1.0), num(2.0), num(-4.0)],
            vec![num(3.0), num(2.0), num(1.0)],
            vec![num(0.37), num(2.0), num(-1.23)],
        ),
        (vec![num(-1.0)], vec![num(3.0)], vec![num(3.0)]),
    ] {
        let n = target.len();
        let problem = Quadratic {
            lower: make(&lower),
            upper: make(&upper),
            target,
        };
        let result = Executor::new(
            problem,
            Direct::new(),
            PointState::new(make(&vec![num(99.0); n])),
        )
        .require_evaluated_state()
        .target_objective(num(1e-6))
        .max_cost_evals(5000)
        .max_iter(200)
        .run()
        .unwrap();
        assert_eq!(
            result.reason,
            TerminationReason::TargetCost,
            "cost={:?}",
            result.cost()
        );
        assert!(result.cost() <= num(1e-6));
        assert_eq!(result.state.current().unwrap().1, result.state.best_cost());
        assert_eq!(result.state.counts().gradient_evals, 0);
        assert_eq!(result.state.counts().cost_evals, result.state.cost_evals());
    }
}
