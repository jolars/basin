use basin::{
    BoxConstraints, CostFunction, DenseMatrixFromFn, Executor, Jacobian,
    MatrixIndex, Residual, Scalar, TerminationReason, TrustRegionReflective,
    VectorIndex, VectorLen,
};
use std::convert::Infallible;

fn num<F: Scalar>(x: f64) -> F {
    F::from_f64(x).unwrap()
}

struct Fit<V, F> {
    make: fn(&[F]) -> V,
    lower: V,
    upper: V,
    deficient: bool,
}

impl<V, F: Scalar> CostFunction for Fit<V, F> {
    type Param = V;
    type Output = F;
    type Error = Infallible;
    fn cost(&self, _: &V) -> Result<F, Infallible> {
        panic!("residual-only solver")
    }
}
impl<V: VectorIndex<F>, F: Scalar> Residual for Fit<V, F> {
    type Param = V;
    type Output = V;
    type Error = Infallible;
    fn residual(&self, x: &V) -> Result<V, Infallible> {
        for i in 0..3 {
            assert!(
                x.get_scalar(i) >= self.lower.get_scalar(i)
                    && x.get_scalar(i) <= self.upper.get_scalar(i)
            );
        }
        let r = if self.deficient {
            let sum = x.get_scalar(1) + x.get_scalar(2) - num(3.0);
            vec![sum, num::<F>(2.0) * sum]
        } else {
            vec![
                x.get_scalar(0) - F::one(),
                x.get_scalar(1) - num(3.0),
                x.get_scalar(2) + num(2.0),
            ]
        };
        Ok((self.make)(&r))
    }
}
impl<V: VectorIndex<F> + DenseMatrixFromFn<F>, F: Scalar> Jacobian
    for Fit<V, F>
{
    type Jacobian = V::Matrix;
    fn jacobian(&self, _: &V) -> Result<V::Matrix, Infallible> {
        Ok(V::dense_from_fn(
            if self.deficient { 2 } else { 3 },
            3,
            |i, j| {
                if self.deficient {
                    if j == 0 {
                        F::zero()
                    } else {
                        F::from_usize(i + 1).unwrap()
                    }
                } else if i == j {
                    F::one()
                } else {
                    F::zero()
                }
            },
        ))
    }
}
impl<V, F: Scalar> BoxConstraints for Fit<V, F> {
    fn lower(&self) -> &V {
        &self.lower
    }
    fn upper(&self) -> &V {
        &self.upper
    }
}

pub fn check<V, F>(make: fn(&[F]) -> V)
where
    F: Scalar,
    V: Clone + VectorIndex<F> + VectorLen + DenseMatrixFromFn<F>,
    V::Matrix: MatrixIndex<F>,
{
    for deficient in [false, true] {
        let problem = Fit {
            make,
            lower: make(&[num(0.5), num(-10.0), F::neg_infinity()]),
            upper: make(&[num(0.5), F::one(), F::infinity()]),
            deficient,
        };
        let tol: F = if F::epsilon() > num(1e-10) {
            num(1e-4)
        } else {
            num(1e-9)
        };
        let result = Executor::from_start(
            problem,
            TrustRegionReflective::<F>::new()
                .with_absolute_scaled_gradient_tolerance(tol),
            make(&[F::zero(); 3]),
        )
        .max_iter(200)
        .run()
        .unwrap();
        assert_eq!(result.reason, TerminationReason::SolverConverged);
        assert_eq!(result.param().get_scalar(0), num(0.5));
        if deficient {
            assert!(result.cost() < num(1e-6));
        } else {
            assert!(
                (result.param().get_scalar(1) - F::one()).abs() < num(1e-3)
            );
            assert!(
                (result.param().get_scalar(2) + num(2.0)).abs() < num(1e-3)
            );
            assert!((result.cost() - num(2.125)).abs() < num(1e-3));
        }
    }
    check_small_radius(make);
    check_small_secular_root(make);
}

struct Diagonal<V, F> {
    make: fn(&[F]) -> V,
    diagonal: Vec<F>,
    target: Vec<F>,
    lower: V,
    upper: V,
}

impl<V, F: Scalar> CostFunction for Diagonal<V, F> {
    type Param = V;
    type Output = F;
    type Error = Infallible;
    fn cost(&self, _: &V) -> Result<F, Infallible> {
        panic!("residual-only solver")
    }
}

impl<V: VectorIndex<F>, F: Scalar> Residual for Diagonal<V, F> {
    type Param = V;
    type Output = V;
    type Error = Infallible;
    fn residual(&self, x: &V) -> Result<V, Infallible> {
        let r: Vec<F> = self
            .diagonal
            .iter()
            .enumerate()
            .map(|(i, &d)| d * (x.get_scalar(i) - self.target[i]))
            .collect();
        Ok((self.make)(&r))
    }
}

impl<V: VectorIndex<F> + DenseMatrixFromFn<F>, F: Scalar> Jacobian
    for Diagonal<V, F>
{
    type Jacobian = V::Matrix;
    fn jacobian(&self, _: &V) -> Result<V::Matrix, Infallible> {
        let n = self.diagonal.len();
        Ok(V::dense_from_fn(n, n, |i, j| {
            if i == j { self.diagonal[i] } else { F::zero() }
        }))
    }
}

impl<V, F: Scalar> BoxConstraints for Diagonal<V, F> {
    fn lower(&self) -> &V {
        &self.lower
    }
    fn upper(&self) -> &V {
        &self.upper
    }
}

fn check_small_radius<V, F>(make: fn(&[F]) -> V)
where
    F: Scalar,
    V: Clone + VectorIndex<F> + VectorLen + DenseMatrixFromFn<F>,
    V::Matrix: MatrixIndex<F>,
{
    for start in [F::one(), -F::one(), num::<F>(1e6)] {
        for relative_radius in [num::<F>(1e-5), F::epsilon().sqrt()] {
            let radius = relative_radius * start.abs();
            for max_iter in [1, 100] {
                let problem = Diagonal {
                    make,
                    diagonal: vec![F::one()],
                    target: vec![num::<F>(2.0) * start],
                    lower: make(&[F::neg_infinity()]),
                    upper: make(&[F::infinity()]),
                };
                let result = Executor::from_start(
                    problem,
                    TrustRegionReflective::new().with_initial_radius(radius),
                    make(&[start]),
                )
                .max_iter(max_iter)
                .run()
                .unwrap();
                assert_ne!(result.reason, TerminationReason::SolverFailed);
                let x = result.param().get_scalar(0);
                if max_iter == 1 {
                    assert_eq!(result.iter(), 1);
                    assert!(
                        ((x - start).abs() - radius).abs()
                            <= num::<F>(4.0) * F::epsilon() * start.abs()
                    );
                } else {
                    assert_eq!(
                        result.reason,
                        TerminationReason::SolverConverged
                    );
                    assert!((x / start - num(2.0)).abs() < num(1e-6));
                }
            }
        }
    }
}

fn check_small_secular_root<V, F>(make: fn(&[F]) -> V)
where
    F: Scalar,
    V: Clone + VectorIndex<F> + VectorLen + DenseMatrixFromFn<F>,
    V::Matrix: MatrixIndex<F>,
{
    let small = if F::epsilon() > num(1e-10) {
        num(1e-4)
    } else {
        num(1e-10)
    };
    let problem = Diagonal {
        make,
        diagonal: vec![F::one(), small],
        target: vec![num(3.0), num(10.0)],
        lower: make(&[F::neg_infinity(); 2]),
        upper: make(&[F::infinity(); 2]),
    };
    let result = Executor::from_start(
        problem,
        TrustRegionReflective::new(),
        make(&[num(2.0), F::zero()]),
    )
    .max_iter(1)
    .run()
    .unwrap();
    assert_ne!(result.reason, TerminationReason::SolverFailed);
    let h0 = result.param().get_scalar(0) - num(2.0);
    let h1 = result.param().get_scalar(1);
    assert!((h0.hypot(h1) - num(2.0)).abs() < num(1e-5));
    assert!((h0 - F::one()).abs() < num(0.02));
    assert!((h1 - num::<F>(3.0).sqrt()).abs() < num(0.02));
    assert!(result.cost() < num(1e-4));
}
