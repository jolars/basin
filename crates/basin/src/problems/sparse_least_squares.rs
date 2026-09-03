//! Sparse linear least-squares fixture.
//!
//! `r(x) = A · x − b` with a sparse design matrix `A`. The Jacobian
//! `J(x) = A` is constant in `x`; the residual is exactly linear, so
//! Gauss-Newton converges in a single iteration. This is the load-
//! bearing sparse fixture for S2b: the simplest possible problem
//! that exercises sparse `MatVec`, sparse `Jacobian::Output`, and
//! sparse `LinearSolveSpd`/`LinearSolveLstsq` end-to-end.
//!
//! Cost (LM convention): `f(x) = ½ ‖A · x − b‖²`. The closed-form
//! minimizer is `x* = (AᵀA)⁻¹Aᵀb` when `A` has full column rank.
//!
//! Unlike the analytic problems in this corpus, `SparseLeastSquares`
//! carries data on the struct (the design matrix and target). The
//! generic parameters `M` and `V` pin the matrix and vector backend.
//! Per-backend impls live in feature-gated submodules below.

use super::spec::{
    Dimensionality, HasSpec, ProblemSpec, Properties, Reference,
};

/// Sparse linear least-squares problem `min_x ½ ‖A · x − b‖²`. Holds
/// the design matrix and target on the struct; `M` is the sparse
/// matrix backend and `V` is the matching vector backend.
pub struct SparseLeastSquares<M, V> {
    /// Design matrix `A` (typically tall and sparse).
    pub a: M,
    /// Target vector `b`.
    pub b: V,
}

impl<M, V> SparseLeastSquares<M, V> {
    /// Build a problem from a design matrix and target. Both backends
    /// need to be the matching `(M, V)` pair for a per-backend
    /// `Residual`/`Jacobian` impl below to apply.
    pub fn new(a: M, b: V) -> Self {
        Self { a, b }
    }
}

/// Catalog entry for the sparse linear least-squares fixture.
pub static SPARSE_LEAST_SQUARES_SPEC: ProblemSpec = ProblemSpec {
    name: "Sparse least squares",
    // Dimension comes from the runtime design matrix; the catalog
    // entry advertises the family rather than a fixed n.
    dim: Dimensionality::NDimensional { min: 1 },
    properties: Properties {
        smooth: true,
        differentiable: true,
        convex: true,
        unimodal: true,
        separable: false,
        scalable: true,
    },
    references: &[Reference {
        citation: "Björck (1996)",
        title: "Numerical Methods for Least Squares Problems",
        source: "SIAM",
        doi: Some("10.1137/1.9781611971484"),
        url: None,
    }],
    description: "Linear least-squares problem r(x) = A·x − b with a \
                  sparse design matrix A. Residual is exactly linear, \
                  so Gauss-Newton converges in one iteration. The \
                  S2b sparse fixture.",
};

impl<M, V> HasSpec for SparseLeastSquares<M, V> {
    const SPEC: &'static ProblemSpec = &SPARSE_LEAST_SQUARES_SPEC;
}

/// Sparse linear least-squares problem with explicit element-wise box
/// bounds: `min_x ½ ‖A · x − b‖²` subject to `lower ≤ x ≤ upper`.
/// Suitable for [`Trf`](crate::solver::Trf) when bounds make the
/// closed-form least-squares minimum infeasible.
pub struct SparseLeastSquaresBoxed<M, V> {
    /// Design matrix `A` (typically tall and sparse).
    pub a: M,
    /// Target vector `b`.
    pub b: V,
    /// Element-wise lower bound on `x`.
    pub lower: V,
    /// Element-wise upper bound on `x`.
    pub upper: V,
}

impl<M, V> SparseLeastSquaresBoxed<M, V> {
    /// Build a bounded sparse linear-least-squares problem. Caller must
    /// ensure `lower[i] ≤ upper[i]` per component.
    pub fn new(a: M, b: V, lower: V, upper: V) -> Self {
        Self { a, b, lower, upper }
    }
}

impl<M, V> HasSpec for SparseLeastSquaresBoxed<M, V> {
    const SPEC: &'static ProblemSpec = &SPARSE_LEAST_SQUARES_SPEC;
}

#[cfg(feature = "nalgebra_all")]
mod nalgebra_impl {
    use super::{SparseLeastSquares, SparseLeastSquaresBoxed};
    use crate::core::math::{MatVec, ScaledAdd};
    use crate::{BoxConstraints, CostFunction, Jacobian, Residual};
    use nalgebra::DVector;
    use nalgebra_sparse::CscMatrix;

    impl CostFunction for SparseLeastSquares<CscMatrix<f64>, DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &DVector<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            // ½ ‖A·x − b‖² (LM convention).
            let mut r = self.a.matvec(x);
            r.scaled_add(-1.0, &self.b);
            Ok(0.5 * r.iter().map(|v| v * v).sum::<f64>())
        }
    }

    impl Residual for SparseLeastSquares<CscMatrix<f64>, DVector<f64>> {
        type Param = DVector<f64>;
        type Output = DVector<f64>;
        type Error = std::convert::Infallible;
        fn residual(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, std::convert::Infallible> {
            let mut r = self.a.matvec(x);
            r.scaled_add(-1.0, &self.b);
            Ok(r)
        }
    }

    impl Jacobian for SparseLeastSquares<CscMatrix<f64>, DVector<f64>> {
        type Jacobian = CscMatrix<f64>;
        fn jacobian(
            &self,
            _x: &DVector<f64>,
        ) -> Result<CscMatrix<f64>, std::convert::Infallible> {
            // J(x) = A, constant in x for linear residuals.
            Ok(self.a.clone())
        }
    }

    impl CostFunction for SparseLeastSquaresBoxed<CscMatrix<f64>, DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &DVector<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            let mut r = self.a.matvec(x);
            r.scaled_add(-1.0, &self.b);
            Ok(0.5 * r.iter().map(|v| v * v).sum::<f64>())
        }
    }

    impl Residual for SparseLeastSquaresBoxed<CscMatrix<f64>, DVector<f64>> {
        type Param = DVector<f64>;
        type Output = DVector<f64>;
        type Error = std::convert::Infallible;
        fn residual(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, std::convert::Infallible> {
            let mut r = self.a.matvec(x);
            r.scaled_add(-1.0, &self.b);
            Ok(r)
        }
    }

    impl Jacobian for SparseLeastSquaresBoxed<CscMatrix<f64>, DVector<f64>> {
        type Jacobian = CscMatrix<f64>;
        fn jacobian(
            &self,
            _x: &DVector<f64>,
        ) -> Result<CscMatrix<f64>, std::convert::Infallible> {
            Ok(self.a.clone())
        }
    }

    impl BoxConstraints for SparseLeastSquaresBoxed<CscMatrix<f64>, DVector<f64>> {
        fn lower(&self) -> &DVector<f64> {
            &self.lower
        }
        fn upper(&self) -> &DVector<f64> {
            &self.upper
        }
    }
}

#[cfg(feature = "faer_all")]
mod faer_impl {
    use super::{SparseLeastSquares, SparseLeastSquaresBoxed};
    use crate::core::math::{MatVec, ScaledAdd};
    use crate::{BoxConstraints, CostFunction, Jacobian, Residual};
    use faer::Col;
    use faer::sparse::SparseColMat;

    impl CostFunction for SparseLeastSquares<SparseColMat<usize, f64>, Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            // ½ ‖A·x − b‖² (LM convention).
            let mut r = self.a.matvec(x);
            r.scaled_add(-1.0, &self.b);
            let mut s = 0.0;
            for i in 0..r.nrows() {
                s += r[i] * r[i];
            }
            Ok(0.5 * s)
        }
    }

    impl Residual for SparseLeastSquares<SparseColMat<usize, f64>, Col<f64>> {
        type Param = Col<f64>;
        type Output = Col<f64>;
        type Error = std::convert::Infallible;
        fn residual(
            &self,
            x: &Col<f64>,
        ) -> Result<Col<f64>, std::convert::Infallible> {
            let mut r = self.a.matvec(x);
            r.scaled_add(-1.0, &self.b);
            Ok(r)
        }
    }

    impl Jacobian for SparseLeastSquares<SparseColMat<usize, f64>, Col<f64>> {
        type Jacobian = SparseColMat<usize, f64>;
        fn jacobian(
            &self,
            _x: &Col<f64>,
        ) -> Result<SparseColMat<usize, f64>, std::convert::Infallible>
        {
            // J(x) = A, constant in x for linear residuals.
            Ok(self.a.clone())
        }
    }

    impl CostFunction
        for SparseLeastSquaresBoxed<SparseColMat<usize, f64>, Col<f64>>
    {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            let mut r = self.a.matvec(x);
            r.scaled_add(-1.0, &self.b);
            let mut s = 0.0;
            for i in 0..r.nrows() {
                s += r[i] * r[i];
            }
            Ok(0.5 * s)
        }
    }

    impl Residual for SparseLeastSquaresBoxed<SparseColMat<usize, f64>, Col<f64>> {
        type Param = Col<f64>;
        type Output = Col<f64>;
        type Error = std::convert::Infallible;
        fn residual(
            &self,
            x: &Col<f64>,
        ) -> Result<Col<f64>, std::convert::Infallible> {
            let mut r = self.a.matvec(x);
            r.scaled_add(-1.0, &self.b);
            Ok(r)
        }
    }

    impl Jacobian for SparseLeastSquaresBoxed<SparseColMat<usize, f64>, Col<f64>> {
        type Jacobian = SparseColMat<usize, f64>;
        fn jacobian(
            &self,
            _x: &Col<f64>,
        ) -> Result<SparseColMat<usize, f64>, std::convert::Infallible>
        {
            Ok(self.a.clone())
        }
    }

    impl BoxConstraints
        for SparseLeastSquaresBoxed<SparseColMat<usize, f64>, Col<f64>>
    {
        fn lower(&self) -> &Col<f64> {
            &self.lower
        }
        fn upper(&self) -> &Col<f64> {
            &self.upper
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        // Only need a type: concrete generics don't matter for the SPEC constant.
        type Probe = SparseLeastSquares<(), ()>;
        let spec = <Probe as HasSpec>::SPEC;
        assert_eq!(spec.name, "Sparse least squares");
        assert!(spec.properties.convex);
        assert!(spec.properties.scalable);
        assert!(matches!(spec.dim, Dimensionality::NDimensional { min: 1 }));
        assert!(!spec.references.is_empty());
    }

    #[cfg(feature = "faer_all")]
    #[test]
    fn faer_residual_at_zero_returns_minus_b() {
        use crate::Residual;
        use faer::Col;
        use faer::sparse::{SparseColMat, Triplet};

        // 3×2 design with two nonzeros per row.
        let triplets = [
            Triplet::new(0_usize, 0_usize, 1.0),
            Triplet::new(1, 1, 1.0),
            Triplet::new(2, 0, 1.0),
            Triplet::new(2, 1, 1.0),
        ];
        let a =
            SparseColMat::<usize, f64>::try_new_from_triplets(3, 2, &triplets)
                .unwrap();
        let b = Col::<f64>::from_fn(3, |i| [1.0, 2.0, 4.0][i]);
        let prob = SparseLeastSquares::new(a, b);
        let r = prob.residual(&Col::<f64>::zeros(2)).unwrap();
        // r(0) = A·0 − b = −b.
        assert_eq!(r.nrows(), 3);
        assert_eq!(r[0], -1.0);
        assert_eq!(r[1], -2.0);
        assert_eq!(r[2], -4.0);
    }
}
