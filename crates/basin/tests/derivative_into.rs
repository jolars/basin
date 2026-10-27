use std::cell::RefCell;

use basin::{
    CostFunction, Gradient, Hessian, HessianProduct, Jacobian, MatrixIndex,
    MiniBatchGradient, Residual, Scalar, VectorIndex, VectorLen,
};

#[path = "support/backend_aliases.rs"]
mod backend_aliases;

type Pending<T> = RefCell<Option<Result<T, &'static str>>>;

// Keeping this fixture unconstrained checks that the defaults need no math,
// cloning, or construction capabilities on the associated types.
struct OwnedOnly<'a, P, V, M> {
    param: &'a P,
    direction: &'a P,
    batch: &'a [usize],
    vectors: [Pending<V>; 3],
    matrices: [Pending<M>; 2],
}

fn take<T>(output: &Pending<T>) -> Result<T, &'static str> {
    output
        .borrow_mut()
        .take()
        .expect("owned callback called twice")
}

impl<P, V, M> CostFunction for OwnedOnly<'_, P, V, M> {
    type Param = P;
    type Output = ();
    type Error = &'static str;

    fn cost(&self, _: &P) -> Result<(), Self::Error> {
        panic!("a separate derivative must not evaluate the cost");
    }
}

impl<P, V, M> Gradient for OwnedOnly<'_, P, V, M> {
    type Gradient = V;

    fn gradient(&self, param: &P) -> Result<V, Self::Error> {
        assert!(std::ptr::eq(param, self.param));
        take(&self.vectors[0])
    }
}

impl<P, V, M> MiniBatchGradient for OwnedOnly<'_, P, V, M> {
    type Gradient = V;

    fn n_samples(&self) -> usize {
        4
    }

    fn batch_gradient(
        &self,
        param: &P,
        batch: &[usize],
    ) -> Result<V, Self::Error> {
        assert!(std::ptr::eq(param, self.param));
        assert_eq!(batch, self.batch);
        take(&self.vectors[1])
    }
}

impl<P, V, M> Residual for OwnedOnly<'_, P, V, M> {
    type Param = P;
    type Output = V;
    type Error = &'static str;

    fn residual(&self, _: &P) -> Result<V, Self::Error> {
        panic!("a separate Jacobian must not evaluate the residual");
    }
}

impl<P, V, M> Jacobian for OwnedOnly<'_, P, V, M> {
    type Jacobian = M;

    fn jacobian(&self, param: &P) -> Result<M, Self::Error> {
        assert!(std::ptr::eq(param, self.param));
        take(&self.matrices[0])
    }
}

impl<P, V, M> Hessian for OwnedOnly<'_, P, V, M> {
    type Hessian = M;

    fn hessian(&self, param: &P) -> Result<M, Self::Error> {
        assert!(std::ptr::eq(param, self.param));
        take(&self.matrices[1])
    }
}

impl<P, V, M> HessianProduct for OwnedOnly<'_, P, V, M> {
    fn hessian_product(&self, param: &P, v: &P) -> Result<V, Self::Error> {
        assert!(std::ptr::eq(param, self.param));
        assert!(std::ptr::eq(v, self.direction));
        take(&self.vectors[2])
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
    let direction = vector(&[f(-2.0), f(3.0)]);
    let batch = [3, 1];
    let expected_vectors =
        [[f(1.0), f(2.0)], [f(3.0), f(4.0)], [f(5.0), f(6.0)]];
    let jacobian = [f(1.0), f(2.0), f(3.0), f(4.0), f(5.0), f(6.0)];
    let hessian = [f(2.0), f(3.0), f(3.0), f(4.0)];

    for (len, rows, cols) in [(0, 0, 0), (1, 1, 1), (2, 3, 2), (4, 2, 3)] {
        let mut p = basin::Problem::new(OwnedOnly {
            param: &x,
            direction: &direction,
            batch: &batch,
            vectors: expected_vectors
                .map(|v| RefCell::new(Some(Ok(vector(&v))))),
            matrices: [
                RefCell::new(Some(Ok(matrix(3, 2, &jacobian)))),
                RefCell::new(Some(Ok(matrix(2, 2, &hessian)))),
            ],
        });
        for (kind, expected) in expected_vectors.iter().enumerate() {
            let mut out = vector(&vec![F::nan(); len]);
            match kind {
                0 => p.gradient_into(&x, &mut out).unwrap(),
                1 => p.batch_gradient_into(&x, &batch, &mut out).unwrap(),
                _ => p.hessian_product_into(&x, &direction, &mut out).unwrap(),
            }
            assert_eq!(out.vec_len(), 2);
            for (i, &value) in expected.iter().enumerate() {
                assert_eq!(out.get_scalar(i), value);
            }
        }

        for (kind, expected) in [&jacobian[..], &hessian[..]].iter().enumerate()
        {
            let mut out = matrix(rows, cols, &vec![F::nan(); rows * cols]);
            let expected_rows = if kind == 0 {
                p.jacobian_into(&x, &mut out).unwrap();
                3
            } else {
                p.hessian_into(&x, &mut out).unwrap();
                2
            };
            assert_eq!(
                (out.matrix_rows(), out.matrix_cols()),
                (expected_rows, 2)
            );
            for row in 0..expected_rows {
                for col in 0..2 {
                    assert_eq!(
                        out.matrix_entry(row, col),
                        expected[row * 2 + col]
                    );
                }
            }
        }
        assert!(p.inner().vectors.iter().all(|v| v.borrow().is_none()));
        assert!(p.inner().matrices.iter().all(|m| m.borrow().is_none()));
        let mut expected_counts = basin::EvalCounts::default();
        expected_counts.gradient_evals = 2;
        expected_counts.jacobian_evals = 1;
        expected_counts.hessian_evals = 1;
        expected_counts.hessian_product_evals = 1;
        assert_eq!(*p.counts(), expected_counts);
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
fn defaults_propagate_errors_with_unconstrained_types_and_trait_objects() {
    struct Param;
    struct Vector;
    struct Matrix;
    let x = Param;
    let direction = Param;
    let batch = [3, 1];
    let p = OwnedOnly {
        param: &x,
        direction: &direction,
        batch: &batch,
        vectors: std::array::from_fn(|_| {
            RefCell::new(Some(Err("vector failed")))
        }),
        matrices: std::array::from_fn(|_| {
            RefCell::new(Some(Err("matrix failed")))
        }),
    };
    let gradient: &dyn Gradient<
        Param = Param,
        Output = (),
        Error = &'static str,
        Gradient = Vector,
    > = &p;
    assert_eq!(
        gradient.gradient_into(&x, &mut Vector),
        Err("vector failed")
    );
    let batch_gradient: &dyn MiniBatchGradient<
        Param = Param,
        Output = (),
        Error = &'static str,
        Gradient = Vector,
    > = &p;
    assert_eq!(
        batch_gradient.batch_gradient_into(&x, &batch, &mut Vector),
        Err("vector failed")
    );
    let product: &dyn HessianProduct<
        Param = Param,
        Output = (),
        Error = &'static str,
        Gradient = Vector,
    > = &p;
    assert_eq!(
        product.hessian_product_into(&x, &direction, &mut Vector),
        Err("vector failed")
    );
    let jacobian: &dyn Jacobian<
        Param = Param,
        Output = Vector,
        Error = &'static str,
        Jacobian = Matrix,
    > = &p;
    assert_eq!(
        jacobian.jacobian_into(&x, &mut Matrix),
        Err("matrix failed")
    );
    let hessian: &dyn Hessian<
        Param = Param,
        Output = (),
        Error = &'static str,
        Gradient = Vector,
        Hessian = Matrix,
    > = &p;
    assert_eq!(hessian.hessian_into(&x, &mut Matrix), Err("matrix failed"));
    assert!(p.vectors.iter().all(|v| v.borrow().is_none()));
    assert!(p.matrices.iter().all(|m| m.borrow().is_none()));
}

struct Reuse;

fn fill_vector(x: &[f64], out: &mut Vec<f64>) {
    out.resize(x.len(), f64::NAN);
    for (value, &xi) in out.iter_mut().zip(x) {
        *value = 2.0 * xi;
    }
}

fn fill_matrix(rows: usize, cols: usize, out: &mut Vec<Vec<f64>>) {
    out.resize_with(rows, Vec::new);
    for (i, row) in out.iter_mut().enumerate() {
        row.resize(cols, f64::NAN);
        for (j, value) in row.iter_mut().enumerate() {
            *value = if i == j { 2.0 } else { 0.0 };
        }
    }
}

impl CostFunction for Reuse {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = &'static str;

    fn cost(&self, _: &Vec<f64>) -> Result<f64, Self::Error> {
        panic!("unexpected cost callback");
    }
}

impl Gradient for Reuse {
    type Gradient = Vec<f64>;

    fn gradient(&self, _: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        panic!("unexpected owned gradient");
    }

    fn gradient_into(
        &self,
        x: &Vec<f64>,
        out: &mut Vec<f64>,
    ) -> Result<(), Self::Error> {
        fill_vector(x, out);
        Ok(())
    }
}

impl MiniBatchGradient for Reuse {
    type Gradient = Vec<f64>;

    fn n_samples(&self) -> usize {
        4
    }

    fn batch_gradient(
        &self,
        _: &Vec<f64>,
        _: &[usize],
    ) -> Result<Vec<f64>, Self::Error> {
        panic!("unexpected owned batch gradient");
    }

    fn batch_gradient_into(
        &self,
        x: &Vec<f64>,
        batch: &[usize],
        out: &mut Vec<f64>,
    ) -> Result<(), Self::Error> {
        assert_eq!(batch, &[3, 1]);
        fill_vector(x, out);
        Ok(())
    }
}

impl Residual for Reuse {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = &'static str;

    fn residual(&self, _: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        panic!("unexpected residual callback");
    }
}

impl Jacobian for Reuse {
    type Jacobian = Vec<Vec<f64>>;

    fn jacobian(&self, _: &Vec<f64>) -> Result<Self::Jacobian, Self::Error> {
        panic!("unexpected owned Jacobian");
    }

    fn jacobian_into(
        &self,
        x: &Vec<f64>,
        out: &mut Self::Jacobian,
    ) -> Result<(), Self::Error> {
        fill_matrix(x.len() + 1, x.len(), out);
        Ok(())
    }
}

impl Hessian for Reuse {
    type Hessian = Vec<Vec<f64>>;

    fn hessian(&self, _: &Vec<f64>) -> Result<Self::Hessian, Self::Error> {
        panic!("unexpected owned Hessian");
    }

    fn hessian_into(
        &self,
        x: &Vec<f64>,
        out: &mut Self::Hessian,
    ) -> Result<(), Self::Error> {
        fill_matrix(x.len(), x.len(), out);
        Ok(())
    }
}

impl HessianProduct for Reuse {
    fn hessian_product(
        &self,
        _: &Vec<f64>,
        _: &Vec<f64>,
    ) -> Result<Vec<f64>, Self::Error> {
        panic!("unexpected owned Hessian product");
    }

    fn hessian_product_into(
        &self,
        _: &Vec<f64>,
        v: &Vec<f64>,
        out: &mut Vec<f64>,
    ) -> Result<(), Self::Error> {
        fill_vector(v, out);
        Ok(())
    }
}

#[test]
fn overrides_resize_and_reuse_storage_without_calling_owned_methods() {
    let x = vec![2.0, -1.0];
    let v = vec![-2.0, 3.0];
    for len in [0, 1, 2, 4] {
        let mut out = vec![f64::NAN; len];
        Reuse.gradient_into(&x, &mut out).unwrap();
        assert_eq!(out, [4.0, -2.0]);
        let allocation = out.as_ptr();
        out.fill(f64::NAN);
        Reuse.batch_gradient_into(&x, &[3, 1], &mut out).unwrap();
        assert_eq!(out, [4.0, -2.0]);
        assert_eq!(out.as_ptr(), allocation);
        out.fill(f64::NAN);
        Reuse.hessian_product_into(&x, &v, &mut out).unwrap();
        assert_eq!(out, [-4.0, 6.0]);
        assert_eq!(out.as_ptr(), allocation);

        let mut matrix = vec![vec![f64::NAN; len]; len];
        Reuse.jacobian_into(&x, &mut matrix).unwrap();
        assert_eq!(matrix, [vec![2.0, 0.0], vec![0.0, 2.0], vec![0.0, 0.0]]);
        let allocation = matrix.as_ptr();
        let rows = [matrix[0].as_ptr(), matrix[1].as_ptr()];
        Reuse.hessian_into(&x, &mut matrix).unwrap();
        assert_eq!(matrix, [vec![2.0, 0.0], vec![0.0, 2.0]]);
        assert_eq!(matrix.as_ptr(), allocation);
        assert_eq!([matrix[0].as_ptr(), matrix[1].as_ptr()], rows);
    }
}
