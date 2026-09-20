#[path = "support/backend_aliases.rs"]
mod backend_aliases;
#[path = "support/robust_backend.rs"]
mod robust_backend;

use basin::{
    GaussNewton, LevenbergMarquardt, LevenbergMarquardtQr, LmDamping, Trf,
    TrustRegionReflective,
};
use robust_backend::{Location, check};

macro_rules! check_solvers {
    ($v:ty, $m:ty, $f:ty, $make:expr, $matrix:expr, $dense:ident) => {{
        type V = $v;
        type M = $m;
        type F = $f;
        let make: fn(&[F]) -> V = $make;
        let fit = Location {
            make,
            jacobian: $matrix,
            lower: make(&[-100.0]),
            upper: make(&[100.0]),
        };
        let tolerance: F = if F::EPSILON > 1e-10 { 1e-4 } else { 1e-7 };
        let gradient: F = if F::EPSILON > 1e-10 { 1e-4 } else { 1e-9 };
        check(
            fit.clone(),
            GaussNewton::<V, M, F>::default()
                .with_absolute_gradient_tolerance(gradient),
            tolerance,
        );
        for damping in [LmDamping::Nielsen, LmDamping::TrustRegion] {
            check(
                fit.clone(),
                LevenbergMarquardt::<V, M, F>::default()
                    .with_damping(damping)
                    .with_absolute_gradient_tolerance(gradient),
                tolerance,
            );
        }
        check(
            fit.clone(),
            Trf::<V, M, F>::default()
                .with_absolute_scaled_gradient_tolerance(gradient),
            tolerance,
        );
        $dense!(fit, tolerance, gradient);
    }};
}

macro_rules! dense_solvers {
    ($fit:ident, $tol:ident, $grad:ident) => {{
        for damping in [LmDamping::Nielsen, LmDamping::TrustRegion] {
            check(
                $fit.clone(),
                LevenbergMarquardtQr::new()
                    .with_damping(damping)
                    .with_absolute_gradient_tolerance($grad),
                $tol,
            );
        }
        check(
            $fit,
            TrustRegionReflective::new()
                .with_absolute_scaled_gradient_tolerance($grad),
            $tol,
        );
    }};
}
#[allow(unused_macros)]
macro_rules! sparse_solvers {
    ($fit:ident, $tol:ident, $grad:ident) => {};
}

#[test]
fn vec_f64() {
    check_solvers!(
        Vec<f64>,
        basin::DenseMatrix<f64>,
        f64,
        |x| x.to_vec(),
        basin::DenseMatrix::from_row_slice(4, 1, &[1.0; 4]),
        dense_solvers
    );
}
#[test]
fn vec_f32() {
    check_solvers!(
        Vec<f32>,
        basin::DenseMatrix<f32>,
        f32,
        |x| x.to_vec(),
        basin::DenseMatrix::from_row_slice(4, 1, &[1.0; 4]),
        dense_solvers
    );
}

#[cfg(feature = "nalgebra_all")]
mod nalgebra {
    use super::*;
    use backend_aliases::nalgebra::{DMatrix, DVector};
    use backend_aliases::nalgebra_sparse::{CooMatrix, CscMatrix};
    macro_rules! tests {
        ($dense:ident, $sparse:ident, $f:ty) => {
            #[test]
            fn $dense() {
                check_solvers!(
                    DVector<$f>,
                    DMatrix<$f>,
                    $f,
                    DVector::from_column_slice,
                    DMatrix::from_element(4, 1, 1.0),
                    dense_solvers
                );
            }
            #[test]
            fn $sparse() {
                let mut coo = CooMatrix::new(4, 1);
                for i in 0..4 {
                    coo.push(i, 0, 1.0);
                }
                check_solvers!(
                    DVector<$f>,
                    CscMatrix<$f>,
                    $f,
                    DVector::from_column_slice,
                    CscMatrix::from(&coo),
                    sparse_solvers
                );
            }
        };
    }
    tests!(dense_f64, sparse_f64, f64);
    tests!(dense_f32, sparse_f32, f32);
}

#[cfg(feature = "ndarray_all")]
mod ndarray {
    use super::*;
    use backend_aliases::ndarray::{Array1, Array2};
    #[test]
    fn f64() {
        check_solvers!(
            Array1<f64>,
            Array2<f64>,
            f64,
            |x| Array1::from_vec(x.to_vec()),
            Array2::ones((4, 1)),
            dense_solvers
        );
    }
    #[test]
    fn f32() {
        check_solvers!(
            Array1<f32>,
            Array2<f32>,
            f32,
            |x| Array1::from_vec(x.to_vec()),
            Array2::ones((4, 1)),
            dense_solvers
        );
    }
}

#[cfg(feature = "faer_all")]
mod faer {
    use super::*;
    use backend_aliases::faer::sparse::{SparseColMat, Triplet};
    use backend_aliases::faer::{Col, Mat};
    macro_rules! tests {
        ($dense:ident, $sparse:ident, $f:ty) => {
            #[test]
            fn $dense() { check_solvers!(Col<$f>, Mat<$f>, $f, |x| Col::from_fn(x.len(), |i| x[i]), Mat::from_fn(4,1,|_,_|1.0), dense_solvers); }
            #[test]
            fn $sparse() {
                let entries: Vec<_> = (0..4).map(|i| Triplet::new(i,0,1.0)).collect();
                let matrix = SparseColMat::try_new_from_triplets(4,1,&entries).unwrap();
                check_solvers!(Col<$f>, SparseColMat<usize,$f>, $f, |x| Col::from_fn(x.len(), |i| x[i]), matrix, sparse_solvers);
            }
        };
    }
    tests!(dense_f64, sparse_f64, f64);
    tests!(dense_f32, sparse_f32, f32);
}
