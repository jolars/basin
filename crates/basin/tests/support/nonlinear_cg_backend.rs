use basin::{
    CgUpdate, CostFunction, Dot, Executor, FirstOrderState, Gradient,
    NegInPlace, NonlinearCg, NormInfinity, NormSquared, Scalar, ScaledAdd,
    TerminationReason, VectorIndex, VectorLen,
};
use std::convert::Infallible;

#[derive(Clone, Copy)]
pub enum Kind {
    Quadratic,
    Rosenbrock,
    IllConditioned,
}

pub struct Objective<V, F> {
    pub make: fn(&[F]) -> V,
    pub kind: Kind,
}

impl<V: VectorIndex<F> + VectorLen, F: Scalar> Objective<V, F> {
    fn evaluate(&self, x: &V) -> (F, V) {
        let num = |v| F::from_f64(v).unwrap();
        let (cost, gradient) = match self.kind {
            Kind::Quadratic => {
                let a = x.get_scalar(0);
                let b = x.get_scalar(1);
                let g = vec![num(3.0) * a + b, a + num(2.0) * b];
                ((a * g[0] + b * g[1]) / num(2.0), g)
            }
            Kind::Rosenbrock => {
                let a = x.get_scalar(0);
                let b = x.get_scalar(1);
                let r = b - a * a;
                (
                    num(100.0) * r * r + (F::one() - a).powi(2),
                    vec![
                        num(-400.0) * a * r + num(2.0) * (a - F::one()),
                        num(200.0) * r,
                    ],
                )
            }
            Kind::IllConditioned => {
                // A Householder reflection rotates a diagonal quadratic without
                // changing its known eigenvalues or minimizer.
                let n = x.vec_len();
                let mean = (0..n).map(|i| x.get_scalar(i)).sum::<F>()
                    / F::from_usize(n).unwrap();
                let z: Vec<_> =
                    (0..n).map(|i| x.get_scalar(i) - num(2.0) * mean).collect();
                let weighted: Vec<_> = z
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| num(10.0).powi(i as i32) * v)
                    .collect();
                let cost =
                    z.iter().zip(&weighted).map(|(&a, &b)| a * b).sum::<F>()
                        / num(2.0);
                let mean = weighted.iter().copied().sum::<F>()
                    / F::from_usize(n).unwrap();
                (
                    cost,
                    weighted.into_iter().map(|v| v - num(2.0) * mean).collect(),
                )
            }
        };
        (cost, (self.make)(&gradient))
    }
}

impl<V: VectorIndex<F> + VectorLen, F: Scalar> CostFunction
    for Objective<V, F>
{
    type Param = V;
    type Output = F;
    type Error = Infallible;
    fn cost(&self, x: &V) -> Result<F, Infallible> {
        Ok(self.evaluate(x).0)
    }
}

impl<V: VectorIndex<F> + VectorLen, F: Scalar> Gradient for Objective<V, F> {
    type Gradient = V;
    fn gradient(&self, x: &V) -> Result<V, Infallible> {
        Ok(self.evaluate(x).1)
    }
    fn cost_and_gradient(&self, x: &V) -> Result<(F, V), Infallible> {
        Ok(self.evaluate(x))
    }
}

pub fn check<V, F>(make: fn(&[F]) -> V)
where
    F: Scalar,
    V: Clone
        + VectorIndex<F>
        + VectorLen
        + Dot<F>
        + NegInPlace
        + NormInfinity<F>
        + NormSquared<F>
        + ScaledAdd<F>,
{
    let num = |v| F::from_f64(v).unwrap();
    let single = F::epsilon() > num(1e-10);
    let tolerance = if single { num(1e-3) } else { num(1e-7) };
    for update in [CgUpdate::HagerZhang, CgUpdate::PolakRibierePlus] {
        for (kind, start) in [
            (Kind::Quadratic, vec![num(2.0), num(-1.0)]),
            (Kind::Rosenbrock, vec![num(-1.2), F::one()]),
            (
                Kind::IllConditioned,
                vec![F::one(); if single { 3 } else { 7 }],
            ),
        ] {
            let result = Executor::new(
                Objective { make, kind },
                NonlinearCg::new()
                    .with_update(update)
                    .with_absolute_gradient_tolerance(tolerance),
                FirstOrderState::new(make(&start)),
            )
            .require_evaluated_state()
            .max_iter(10_000)
            .run()
            .unwrap();
            assert_eq!(
                result.reason,
                TerminationReason::GradientTolerance,
                "update={update:?}, cost={:?}, iter={}",
                result.cost(),
                result.iter()
            );
            let (x, cost, gradient) = result.state.current().unwrap();
            assert!(gradient.norm_squared().sqrt() <= tolerance);
            let (expected_cost, expected_gradient) =
                Objective { make, kind }.evaluate(x);
            assert_eq!(cost, expected_cost);
            let mut error = gradient.clone();
            error.scaled_add(-F::one(), &expected_gradient);
            assert_eq!(error.norm_infinity(), F::zero());
            let target = if matches!(kind, Kind::Rosenbrock) {
                F::one()
            } else {
                F::zero()
            };
            for i in 0..x.vec_len() {
                assert!(
                    (x.get_scalar(i) - target).abs() < num(5.0) * tolerance
                );
            }
        }
    }
}
