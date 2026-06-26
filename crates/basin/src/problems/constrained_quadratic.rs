//! Linearly-constrained quadratic fixture.
//!
//! `min f(x) = Σᵢ (xᵢ − cᵢ)²` subject to `A x ≤ b`. The objective is a
//! convex isotropic quadratic centered at `c`, so the *unconstrained*
//! minimizer is `c`; when `c` violates a constraint the constrained
//! minimizer is the Euclidean projection of `c` onto the feasible polytope.
//! That closed form makes this the load-bearing fixture for the log-barrier
//! [`BarrierMethod`](crate::solver::BarrierMethod): a single active row
//! `x₀ + x₁ ≤ 2` with `c = (2, 2)` has the analytic optimum `(1, 1)`.
//!
//! Like [`SparseLeastSquares`](crate::problems::SparseLeastSquares), the
//! constraint and objective data live on the struct; the generic parameters
//! `M` and `V` pin the matrix and vector backend, and per-backend impls live
//! in feature-gated submodules below.

use super::spec::{Dimensionality, HasSpec, ProblemSpec, Properties, Reference};

/// Linearly-constrained quadratic `min Σ (xᵢ − cᵢ)² s.t. A x ≤ b`. Holds the
/// objective center `c`, constraint matrix `A`, and right-hand side `b`; `M`
/// is the (dense) matrix backend and `V` the matching vector backend.
pub struct ConstrainedQuadratic<M, V> {
    /// Center `c` of the quadratic; the unconstrained minimizer.
    pub c: V,
    /// Constraint matrix `A` (`m × n`, dense).
    pub a: M,
    /// Right-hand side `b ∈ ℝᵐ`.
    pub b: V,
}

impl<M, V> ConstrainedQuadratic<M, V> {
    /// Build the fixture from an objective center and the constraints
    /// `A x ≤ b`.
    pub fn new(c: V, a: M, b: V) -> Self {
        Self { c, a, b }
    }
}

/// Catalog entry for the linearly-constrained quadratic fixture.
pub static CONSTRAINED_QUADRATIC_SPEC: ProblemSpec = ProblemSpec {
    name: "Constrained quadratic",
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
        citation: "Boyd & Vandenberghe (2004)",
        title: "Convex Optimization",
        source: "Cambridge University Press",
        doi: None,
        url: Some("https://web.stanford.edu/~boyd/cvxbook/"),
    }],
    description: "Isotropic quadratic Σ(xᵢ − cᵢ)² minimized subject to linear \
                  inequalities A·x ≤ b. The constrained optimum is the \
                  projection of c onto the feasible polytope—the fixture \
                  for the log-barrier BarrierMethod.",
};

impl<M, V> HasSpec for ConstrainedQuadratic<M, V> {
    const SPEC: &'static ProblemSpec = &CONSTRAINED_QUADRATIC_SPEC;
}

#[cfg(feature = "nalgebra")]
mod nalgebra_impl {
    use super::ConstrainedQuadratic;
    use crate::{CostFunction, Gradient, LinearInequalityConstraints};
    use nalgebra::{DMatrix, DVector};

    impl CostFunction for ConstrainedQuadratic<DMatrix<f64>, DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
            Ok((x - &self.c).norm_squared())
        }
    }

    impl Gradient for ConstrainedQuadratic<DMatrix<f64>, DVector<f64>> {
        type Gradient = DVector<f64>;
        fn gradient(&self, x: &DVector<f64>) -> Result<DVector<f64>, std::convert::Infallible> {
            Ok(2.0 * (x - &self.c))
        }
    }

    impl LinearInequalityConstraints for ConstrainedQuadratic<DMatrix<f64>, DVector<f64>> {
        type Matrix = DMatrix<f64>;
        fn a(&self) -> &DMatrix<f64> {
            &self.a
        }
        fn b(&self) -> &DVector<f64> {
            &self.b
        }
    }
}

#[cfg(feature = "faer")]
mod faer_impl {
    use super::ConstrainedQuadratic;
    use crate::{CostFunction, Gradient, LinearInequalityConstraints};
    use faer::{Col, Mat};

    impl CostFunction for ConstrainedQuadratic<Mat<f64>, Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            let mut s = 0.0;
            for i in 0..x.nrows() {
                let d = x[i] - self.c[i];
                s += d * d;
            }
            Ok(s)
        }
    }

    impl Gradient for ConstrainedQuadratic<Mat<f64>, Col<f64>> {
        type Gradient = Col<f64>;
        fn gradient(&self, x: &Col<f64>) -> Result<Col<f64>, std::convert::Infallible> {
            Ok(Col::from_fn(x.nrows(), |i| 2.0 * (x[i] - self.c[i])))
        }
    }

    impl LinearInequalityConstraints for ConstrainedQuadratic<Mat<f64>, Col<f64>> {
        type Matrix = Mat<f64>;
        fn a(&self) -> &Mat<f64> {
            &self.a
        }
        fn b(&self) -> &Col<f64> {
            &self.b
        }
    }
}

mod vec_impl {
    use super::ConstrainedQuadratic;
    use crate::core::math::DenseMatrix;
    use crate::{CostFunction, Gradient, LinearInequalityConstraints};

    impl CostFunction for ConstrainedQuadratic<DenseMatrix, Vec<f64>> {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(x.iter()
                .zip(&self.c)
                .map(|(xi, ci)| (xi - ci).powi(2))
                .sum())
        }
    }

    impl Gradient for ConstrainedQuadratic<DenseMatrix, Vec<f64>> {
        type Gradient = Vec<f64>;
        fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
            Ok(x.iter()
                .zip(&self.c)
                .map(|(xi, ci)| 2.0 * (xi - ci))
                .collect())
        }
    }

    impl LinearInequalityConstraints for ConstrainedQuadratic<DenseMatrix, Vec<f64>> {
        type Matrix = DenseMatrix;
        fn a(&self) -> &DenseMatrix {
            &self.a
        }
        fn b(&self) -> &Vec<f64> {
            &self.b
        }
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::ConstrainedQuadratic;
    use crate::{CostFunction, Gradient, LinearInequalityConstraints};
    use ndarray::{Array1, Array2};

    impl CostFunction for ConstrainedQuadratic<Array2<f64>, Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Array1<f64>) -> Result<f64, std::convert::Infallible> {
            Ok((x - &self.c).mapv(|v| v * v).sum())
        }
    }

    impl Gradient for ConstrainedQuadratic<Array2<f64>, Array1<f64>> {
        type Gradient = Array1<f64>;
        fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, std::convert::Infallible> {
            Ok((x - &self.c).mapv(|v| 2.0 * v))
        }
    }

    impl LinearInequalityConstraints for ConstrainedQuadratic<Array2<f64>, Array1<f64>> {
        type Matrix = Array2<f64>;
        fn a(&self) -> &Array2<f64> {
            &self.a
        }
        fn b(&self) -> &Array1<f64> {
            &self.b
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        type Probe = ConstrainedQuadratic<(), ()>;
        let spec = <Probe as HasSpec>::SPEC;
        assert_eq!(spec.name, "Constrained quadratic");
        assert!(spec.properties.convex);
        assert!(matches!(spec.dim, Dimensionality::NDimensional { min: 1 }));
        assert!(!spec.references.is_empty());
    }
}
