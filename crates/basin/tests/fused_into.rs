use std::cell::RefCell;

use basin::{
    CostFunction, Gradient, Hessian, Jacobian, MatrixIndex, Residual, Scalar,
    VectorIndex, VectorLen,
};

#[path = "support/backend_aliases.rs"]
mod backend_aliases;

type Pending<T> = RefCell<Option<Result<T, &'static str>>>;

// No bounds here: existing fused implementations must retain their dispatch
// without requiring math, cloning, or construction capabilities.
struct FusedOnly<'a, P, C, V, M> {
    param: &'a P,
    pair: Pending<(C, V)>,
    residual_pair: Pending<(V, M)>,
    triple: Pending<(C, V, M)>,
}

fn take<T>(output: &Pending<T>) -> Result<T, &'static str> {
    output
        .borrow_mut()
        .take()
        .expect("fused callback called twice")
}

impl<P, C, V, M> CostFunction for FusedOnly<'_, P, C, V, M> {
    type Param = P;
    type Output = C;
    type Error = &'static str;

    fn cost(&self, _: &P) -> Result<C, Self::Error> {
        panic!("unexpected separate cost");
    }
}

impl<P, C, V, M> Gradient for FusedOnly<'_, P, C, V, M> {
    type Gradient = V;

    fn gradient(&self, _: &P) -> Result<V, Self::Error> {
        panic!("unexpected separate gradient");
    }

    fn gradient_into(&self, _: &P, _: &mut V) -> Result<(), Self::Error> {
        panic!("a separate in-place override must not bypass fused dispatch");
    }

    fn cost_and_gradient(&self, param: &P) -> Result<(C, V), Self::Error> {
        assert!(std::ptr::eq(param, self.param));
        take(&self.pair)
    }
}

impl<P, C, V, M> Residual for FusedOnly<'_, P, C, V, M> {
    type Param = P;
    type Output = V;
    type Error = &'static str;

    fn residual(&self, _: &P) -> Result<V, Self::Error> {
        panic!("unexpected separate residual");
    }

    fn residual_into(&self, _: &P, _: &mut V) -> Result<(), Self::Error> {
        panic!("a separate in-place override must not bypass fused dispatch");
    }
}

impl<P, C, V, M> Jacobian for FusedOnly<'_, P, C, V, M> {
    type Jacobian = M;

    fn jacobian(&self, _: &P) -> Result<M, Self::Error> {
        panic!("unexpected separate Jacobian");
    }

    fn jacobian_into(&self, _: &P, _: &mut M) -> Result<(), Self::Error> {
        panic!("a separate in-place override must not bypass fused dispatch");
    }

    fn residual_and_jacobian(&self, param: &P) -> Result<(V, M), Self::Error> {
        assert!(std::ptr::eq(param, self.param));
        take(&self.residual_pair)
    }
}

impl<P, C, V, M> Hessian for FusedOnly<'_, P, C, V, M> {
    type Hessian = M;

    fn hessian(&self, _: &P) -> Result<M, Self::Error> {
        panic!("unexpected separate Hessian");
    }

    fn hessian_into(&self, _: &P, _: &mut M) -> Result<(), Self::Error> {
        panic!("a separate in-place override must not bypass fused dispatch");
    }

    fn cost_and_gradient_and_hessian(
        &self,
        param: &P,
    ) -> Result<(C, V, M), Self::Error> {
        assert!(std::ptr::eq(param, self.param));
        take(&self.triple)
    }
}

fn check_defaults<V, M, F>(
    vector: fn(&[F]) -> V,
    matrix: fn(usize, usize, &[F]) -> M,
) where
    V: VectorLen + VectorIndex<F>,
    M: MatrixIndex<F>,
    F: Scalar,
{
    let f = |x| F::from_f64(x).unwrap();
    let x = vector(&[f(2.0), f(-1.0)]);
    let gradient = [f(4.0), f(-2.0)];
    let residual = [f(2.0), f(-1.0), f(1.0)];
    let jacobian = [f(1.0), f(0.0), f(0.0), f(1.0), f(1.0), f(1.0)];
    let hessian = [f(2.0), f(0.0), f(0.0), f(2.0)];

    for (len, rows, cols) in [(0, 0, 0), (1, 1, 1), (2, 3, 2), (4, 2, 3)] {
        let p = FusedOnly {
            param: &x,
            pair: RefCell::new(Some(Ok((f(5.0), vector(&gradient))))),
            residual_pair: RefCell::new(Some(Ok((
                vector(&residual),
                matrix(3, 2, &jacobian),
            )))),
            triple: RefCell::new(Some(Ok((
                f(5.0),
                vector(&gradient),
                matrix(2, 2, &hessian),
            )))),
        };
        let mut v = vector(&vec![F::nan(); len]);
        let mut m = matrix(rows, cols, &vec![F::nan(); rows * cols]);
        assert_eq!(p.cost_and_gradient_into(&x, &mut v).unwrap(), f(5.0));
        check_vector(&v, &gradient);
        p.residual_and_jacobian_into(&x, &mut v, &mut m).unwrap();
        check_vector(&v, &residual);
        check_matrix(&m, 3, 2, &jacobian);
        assert_eq!(
            p.cost_and_gradient_and_hessian_into(&x, &mut v, &mut m)
                .unwrap(),
            f(5.0)
        );
        check_vector(&v, &gradient);
        check_matrix(&m, 2, 2, &hessian);
        assert!(p.pair.borrow().is_none());
        assert!(p.residual_pair.borrow().is_none());
        assert!(p.triple.borrow().is_none());
    }
}

fn check_vector<F: Scalar>(v: &(impl VectorLen + VectorIndex<F>), x: &[F]) {
    assert_eq!(v.vec_len(), x.len());
    for (i, &value) in x.iter().enumerate() {
        assert_eq!(v.get_scalar(i), value);
    }
}

fn check_matrix<F: Scalar>(
    m: &impl MatrixIndex<F>,
    rows: usize,
    cols: usize,
    x: &[F],
) {
    assert_eq!((m.matrix_rows(), m.matrix_cols()), (rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            assert_eq!(m.matrix_entry(i, j), x[i * cols + j]);
        }
    }
}

#[test]
fn defaults_vec_f32_and_f64() {
    check_defaults::<_, _, f32>(
        |x| x.to_vec(),
        basin::DenseMatrix::from_row_slice,
    );
    check_defaults::<_, _, f64>(
        |x| x.to_vec(),
        basin::DenseMatrix::from_row_slice,
    );
}

#[cfg(feature = "nalgebra_all")]
#[test]
fn defaults_nalgebra_f32_and_f64() {
    use backend_aliases::nalgebra::{DMatrix, DVector};
    check_defaults::<_, _, f32>(
        DVector::from_column_slice,
        DMatrix::from_row_slice,
    );
    check_defaults::<_, _, f64>(
        DVector::from_column_slice,
        DMatrix::from_row_slice,
    );
}

#[cfg(feature = "ndarray_all")]
#[test]
fn defaults_ndarray_f32_and_f64() {
    use backend_aliases::ndarray::{Array1, Array2};
    check_defaults::<_, _, f32>(
        |x| Array1::from_vec(x.to_vec()),
        |r, c, x| Array2::from_shape_vec((r, c), x.to_vec()).unwrap(),
    );
    check_defaults::<_, _, f64>(
        |x| Array1::from_vec(x.to_vec()),
        |r, c, x| Array2::from_shape_vec((r, c), x.to_vec()).unwrap(),
    );
}

#[cfg(feature = "faer_all")]
#[test]
fn defaults_faer_f32_and_f64() {
    use backend_aliases::faer::{Col, Mat};
    check_defaults::<_, _, f32>(
        |x| Col::from_fn(x.len(), |i| x[i]),
        |r, c, x| Mat::from_fn(r, c, |i, j| x[i * c + j]),
    );
    check_defaults::<_, _, f64>(
        |x| Col::from_fn(x.len(), |i| x[i]),
        |r, c, x| Mat::from_fn(r, c, |i, j| x[i * c + j]),
    );
}

#[test]
fn defaults_support_unconstrained_types_and_trait_objects_and_preserve_errors()
{
    #[derive(Debug, PartialEq)]
    struct Cost;
    struct Param;
    struct Vector;
    struct Matrix;
    let x = Param;
    for fail in [false, true] {
        let p = FusedOnly {
            param: &x,
            pair: RefCell::new(Some(if fail {
                Err("pair")
            } else {
                Ok((Cost, Vector))
            })),
            residual_pair: RefCell::new(Some(if fail {
                Err("residual pair")
            } else {
                Ok((Vector, Matrix))
            })),
            triple: RefCell::new(Some(if fail {
                Err("triple")
            } else {
                Ok((Cost, Vector, Matrix))
            })),
        };
        let gradient: &dyn Gradient<
            Param = Param,
            Output = Cost,
            Error = &'static str,
            Gradient = Vector,
        > = &p;
        assert_eq!(
            gradient.cost_and_gradient_into(&x, &mut Vector),
            if fail { Err("pair") } else { Ok(Cost) }
        );
        let jacobian: &dyn Jacobian<
            Param = Param,
            Output = Vector,
            Error = &'static str,
            Jacobian = Matrix,
        > = &p;
        assert_eq!(
            jacobian.residual_and_jacobian_into(&x, &mut Vector, &mut Matrix),
            if fail { Err("residual pair") } else { Ok(()) }
        );
        let hessian: &dyn Hessian<
            Param = Param,
            Output = Cost,
            Error = &'static str,
            Gradient = Vector,
            Hessian = Matrix,
        > = &p;
        assert_eq!(
            hessian.cost_and_gradient_and_hessian_into(
                &x,
                &mut Vector,
                &mut Matrix
            ),
            if fail { Err("triple") } else { Ok(Cost) }
        );
    }
}
