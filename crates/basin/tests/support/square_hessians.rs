use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use basin::{
    CostFunction, DenseBackend, DenseMatrixFromFn, FiniteDiff, Hessian,
    MatrixIndex, MatrixShape, Method, Scalar, SquareMatrixFromFn, VectorIndex,
    VectorLen,
};

struct Quadratic<V, F> {
    types: PhantomData<(V, F)>,
    calls: AtomicUsize,
    fail_mixed: bool,
}

impl<V: VectorIndex<F> + VectorLen, F: Scalar> CostFunction
    for Quadratic<V, F>
{
    type Param = V;
    type Output = F;
    type Error = &'static str;

    fn cost(&self, x: &V) -> Result<F, Self::Error> {
        self.calls.fetch_add(1, Relaxed);
        if x.vec_len() == 0 {
            return Ok(F::zero());
        }
        let a = x.get_scalar(0);
        let b = x.get_scalar(1);
        if self.fail_mixed && a > F::zero() && b > F::zero() {
            return Err("mixed probe");
        }
        Ok(a * a + a * b + (b + b) * b)
    }
}

pub(super) fn check<V, F>(make: impl Fn(&[F]) -> V)
where
    F: Scalar + Send + Sync,
    V: Clone + VectorLen + VectorIndex<F> + DenseBackend<F> + Send + Sync,
    V: DenseMatrixFromFn<F, Matrix = <V as DenseBackend<F>>::Matrix>,
    <V as DenseBackend<F>>::Matrix: MatrixIndex<F> + SquareMatrixFromFn<F>,
{
    let number = |value| F::from_f64(value).unwrap();
    let x = make(&[number(1.0), number(-2.0)]);
    let problem = |fail_mixed| Quadratic::<V, F> {
        types: PhantomData,
        calls: AtomicUsize::new(0),
        fail_mixed,
    };
    let check_entries = |out: &<V as DenseBackend<F>>::Matrix| {
        assert_eq!((out.matrix_rows(), out.matrix_cols()), (2, 2));
        for (i, row) in [[2.0, 1.0], [1.0, 4.0]].into_iter().enumerate() {
            for (j, value) in row.into_iter().enumerate() {
                assert_eq!(out.matrix_entry(i, j), number(value));
            }
        }
    };

    for (method, probes) in [(Method::Forward, 6), (Method::Central, 9)] {
        let adapter = |fail_mixed| {
            FiniteDiff::new_with_scalar(problem(fail_mixed))
                .hessian_method(method)
                .with_step(number(0.125))
        };
        let fd = adapter(false);
        let owned = fd.hessian(&x).unwrap();
        check_entries(&owned);
        assert_eq!(fd.get_ref().calls.swap(0, Relaxed), probes);

        let mut out = V::dense_from_fn(2, 3, |_, _| F::nan());
        fd.hessian_into(&x, &mut out).unwrap();
        check_entries(&out);
        assert_eq!(fd.get_ref().calls.swap(0, Relaxed), probes);

        let (expected_cost, mut gradient, hessian) =
            fd.cost_and_gradient_and_hessian(&x).unwrap();
        check_entries(&hessian);
        assert_eq!(fd.get_ref().calls.swap(0, Relaxed), probes + 4);

        let mut out = V::dense_from_fn(2, 2, |_, _| F::nan());
        let cost = fd
            .cost_and_gradient_and_hessian_into(&x, &mut gradient, &mut out)
            .unwrap();
        assert_eq!(cost, expected_cost);
        check_entries(&out);
        assert_eq!(fd.get_ref().calls.swap(0, Relaxed), probes + 4);

        let zero = make(&[F::zero(); 2]);
        let failing = adapter(true);
        assert_eq!(failing.hessian_into(&zero, &mut out), Err("mixed probe"));
        check_entries(&out);
        assert_eq!(
            failing.cost_and_gradient_and_hessian_into(
                &zero,
                &mut gradient,
                &mut out,
            ),
            Err("mixed probe")
        );
        check_entries(&out);

        fd.hessian_into(&make(&[]), &mut out).unwrap();
        assert_eq!((out.matrix_rows(), out.matrix_cols()), (0, 0));
        assert_eq!(fd.get_ref().calls.load(Relaxed), 1);
    }
}
