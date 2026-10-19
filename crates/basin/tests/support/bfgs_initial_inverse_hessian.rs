use basin::{
    Bfgs, Constant, CostFunction, DenseBackend, DenseMatrixFromFn, Dot,
    Executor, GeneralRankOneUpdate, Gradient, MatVec, MatrixIdentity,
    MatrixIndex, NegInPlace, NormSquared, Scalar, ScaleInPlace, ScaledAdd,
    State, VectorIndex, VectorLen,
};
use std::convert::Infallible;

pub struct Quadratic<V, F> {
    pub make: fn(&[F]) -> V,
}

impl<V, F> basin::ProblemConstraints for Quadratic<V, F> {
    type Constraints = basin::NoConstraints;
    fn constraint_set(&self) -> basin::NoConstraints {
        basin::NoConstraints::EMPTY
    }
}

impl<V: VectorIndex<F>, F: Scalar> CostFunction for Quadratic<V, F> {
    type Param = V;
    type Output = F;
    type Error = Infallible;
    fn cost(&self, x: &V) -> Result<F, Infallible> {
        let a = x.get_scalar(0);
        let b = x.get_scalar(1);
        Ok(F::from_f64(2.0).unwrap() * a * a + a * b + b * b)
    }
}

impl<V: VectorIndex<F>, F: Scalar> Gradient for Quadratic<V, F> {
    type Gradient = V;
    fn gradient(&self, x: &V) -> Result<V, Infallible> {
        let a = x.get_scalar(0);
        let b = x.get_scalar(1);
        Ok((self.make)(&[
            F::from_f64(4.0).unwrap() * a + b,
            a + F::from_f64(2.0).unwrap() * b,
        ]))
    }
}

pub fn check<V, F>(make: fn(&[F]) -> V)
where
    F: Scalar,
    V: DenseBackend<F>
        + DenseMatrixFromFn<F, Matrix = <V as DenseBackend<F>>::Matrix>
        + Clone
        + VectorLen
        + VectorIndex<F>
        + Dot<F>
        + NormSquared<F>
        + ScaledAdd<F>
        + ScaleInPlace<F>
        + NegInPlace,
    <V as DenseBackend<F>>::Matrix: Clone
        + MatrixIdentity
        + MatrixIndex<F>
        + MatVec<V>
        + ScaleInPlace<F>
        + GeneralRankOneUpdate<V, F>,
{
    let num = |v| F::from_f64(v).unwrap();
    let tolerance = num(64.0) * F::epsilon();
    let close = |a: F, b: F| {
        assert!((a - b).abs() <= tolerance, "{a:?} != {b:?}");
    };
    // A = [[4, 1], [1, 2]], so A^-1 = [[2, -1], [-1, 4]] / 7.
    // Exact inverse seeds remain unchanged by BFGS on this quadratic.
    let inverse = || {
        V::dense_from_fn(2, 2, |i, j| {
            num([[2.0, -1.0], [-1.0, 4.0]][i][j] / 7.0)
        })
    };
    let make_run = || {
        Executor::from_start(
            Quadratic { make },
            Bfgs::<V, F, _, Constant<F>>::with_line_search(Constant(num(0.25)))
                .with_initial_inverse_hessian(inverse()),
            make(&[num(2.0), num(-1.0)]),
        )
    };
    let first = make_run().max_iter(1).run_with_solver().unwrap();
    close(first.state.param().get_scalar(0), num(1.5));
    close(first.state.param().get_scalar(1), num(-0.75));
    for i in 0..2 {
        for j in 0..2 {
            close(
                first.solver.inverse_hessian().unwrap().matrix_entry(i, j),
                inverse().matrix_entry(i, j),
            );
        }
    }
    let expected = make_run().max_iter(4).run_with_solver().unwrap();
    let resumed = Executor::resume_from_checkpoint(
        Quadratic { make },
        first.into_checkpoint(),
    )
    .max_iter(4)
    .run_with_solver()
    .unwrap();
    assert_eq!(resumed.counts, expected.counts);
    assert_eq!(resumed.state.iter(), 4);
    for i in 0..2 {
        assert_eq!(
            resumed.state.param().get_scalar(i),
            expected.state.param().get_scalar(i),
        );
        for j in 0..2 {
            assert_eq!(
                resumed.solver.inverse_hessian().unwrap().matrix_entry(i, j),
                expected
                    .solver
                    .inverse_hessian()
                    .unwrap()
                    .matrix_entry(i, j),
            );
        }
    }

    let seeded = Executor::from_start(
        Quadratic { make },
        Bfgs::<V, F, _, Constant<F>>::with_line_search(Constant(F::one()))
            .with_initial_inverse_hessian_initializer(|x: &V, g: &V| {
                close(x.get_scalar(0), num(2.0));
                close(g.get_scalar(0), num(7.0));
                close(g.get_scalar(1), F::zero());
                V::dense_from_fn(x.vec_len(), x.vec_len(), |i, j| {
                    if i == j {
                        num(0.1) / g.get_scalar(i).abs().max(num(1e-3))
                    } else {
                        F::zero()
                    }
                })
            }),
        make(&[num(2.0), num(-1.0)]),
    )
    .max_iter(1)
    .run()
    .unwrap();
    close(seeded.param().get_scalar(0), num(1.9));
    close(seeded.param().get_scalar(1), num(-1.0));
    assert_eq!(seeded.counts.cost_evals, 2);
    assert_eq!(seeded.counts.gradient_evals, 2);
}
