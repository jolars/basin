//! N-dimensional Sphere function.
//!
//! `f(x) = Σᵢ xᵢ²`
//!
//! Smooth, convex, separable, unimodal. Global minimum at `x = (0, …, 0)`
//! with `f = 0`. The trivial canary problem—every solver should solve it
//! cleanly; failure indicates the implementation is broken.

use core::marker::PhantomData;

use super::spec::{Dimensionality, HasSpec, ProblemSpec, Properties, Reference};
use crate::{CostFunction, Gradient};

/// Evaluates the Sphere function at `x`.
pub fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}

/// Writes the Sphere gradient at `x` into `out`. Lengths must match.
pub fn sphere_gradient(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), out.len());
    for (g, &v) in out.iter_mut().zip(x.iter()) {
        *g = 2.0 * v;
    }
}

/// Pre-wrapped Sphere problem. Generic over the parameter backend `P`;
/// the default `P = Vec<f64>` lets you write `Sphere::default()` for the
/// common case. Backend impls (`nalgebra::DVector<f64>`, `ndarray::Array1<f64>`,
/// `faer::Col<f64>`) are gated behind their respective features.
pub struct Sphere<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> Sphere<P> {
    /// Build a freshly typed problem instance. Pair with one of the
    /// backend-specific impl blocks (Vec, nalgebra, ndarray, faer).
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for Sphere<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Catalog entry for this problem.
pub static SPHERE_SPEC: ProblemSpec = ProblemSpec {
    name: "Sphere",
    dim: Dimensionality::NDimensional { min: 1 },
    properties: Properties {
        smooth: true,
        differentiable: true,
        convex: true,
        unimodal: true,
        separable: true,
        scalable: true,
    },
    references: &[Reference {
        citation: "De Jong (1975)",
        title: "An Analysis of the Behavior of a Class of Genetic Adaptive Systems",
        source: "PhD thesis, University of Michigan",
        doi: None,
        url: Some("https://hdl.handle.net/2027.42/4507"),
    }],
    description: "Sum of squares: f(x) = Σ xᵢ². Convex, separable, unimodal. \
                  Global minimum at x = (0, …, 0), value 0. The canonical \
                  trivial canary—every solver should solve it cleanly.",
};

impl<P> HasSpec for Sphere<P> {
    const SPEC: &'static ProblemSpec = &SPHERE_SPEC;
}

impl CostFunction for Sphere<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(sphere(x))
    }
}

impl Gradient for Sphere<Vec<f64>> {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; x.len()];
        sphere_gradient(x, &mut out);
        Ok(out)
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_impl {
    use super::{Sphere, sphere, sphere_gradient};
    use crate::{CostFunction, Gradient};
    use nalgebra::DVector;

    impl CostFunction for Sphere<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(sphere(x.as_slice()))
        }
    }

    impl Gradient for Sphere<DVector<f64>> {
        type Gradient = DVector<f64>;
        fn gradient(&self, x: &DVector<f64>) -> Result<DVector<f64>, std::convert::Infallible> {
            let mut out = DVector::zeros(x.len());
            sphere_gradient(x.as_slice(), out.as_mut_slice());
            Ok(out)
        }
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::{Sphere, sphere, sphere_gradient};
    use crate::{CostFunction, Gradient};
    use ndarray::Array1;

    impl CostFunction for Sphere<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Array1<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(sphere(x.as_slice().expect("Array1 is contiguous")))
        }
    }

    impl Gradient for Sphere<Array1<f64>> {
        type Gradient = Array1<f64>;
        fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, std::convert::Infallible> {
            let mut out = Array1::zeros(x.len());
            sphere_gradient(
                x.as_slice().expect("Array1 is contiguous"),
                out.as_slice_mut().expect("Array1 is contiguous"),
            );
            Ok(out)
        }
    }
}

#[cfg(feature = "faer")]
mod faer_impl {
    use super::Sphere;
    use crate::{CostFunction, Gradient};
    use faer::Col;

    impl CostFunction for Sphere<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            let n = x.nrows();
            let mut s = 0.0;
            for i in 0..n {
                s += x[i] * x[i];
            }
            Ok(s)
        }
    }

    impl Gradient for Sphere<Col<f64>> {
        type Gradient = Col<f64>;
        fn gradient(&self, x: &Col<f64>) -> Result<Col<f64>, std::convert::Infallible> {
            let n = x.nrows();
            Ok(Col::<f64>::from_fn(n, |i| 2.0 * x[i]))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_minimum_is_zero_at_origin() {
        assert_eq!(sphere(&[0.0]), 0.0);
        assert_eq!(sphere(&[0.0, 0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn sphere_known_value() {
        assert_eq!(sphere(&[1.0, 2.0, 3.0]), 14.0);
    }

    #[test]
    fn sphere_gradient_zero_at_origin() {
        let mut g = vec![0.0; 5];
        sphere_gradient(&[0.0; 5], &mut g);
        for v in g {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn sphere_gradient_matches_finite_difference() {
        let x = [-1.2, 1.0, 0.7, 0.4];
        let mut g = vec![0.0; x.len()];
        sphere_gradient(&x, &mut g);
        let h = 1e-6;
        for i in 0..x.len() {
            let mut xp = x;
            let mut xm = x;
            xp[i] += h;
            xm[i] -= h;
            let fd = (sphere(&xp) - sphere(&xm)) / (2.0 * h);
            assert!((g[i] - fd).abs() < 1e-6, "i={i}, g={}, fd={fd}", g[i]);
        }
    }

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        let spec = <Sphere<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(spec.name, "Sphere");
        assert!(spec.properties.convex);
        assert!(spec.properties.separable);
        assert!(spec.properties.unimodal);
        assert!(matches!(spec.dim, Dimensionality::NDimensional { min: 1 }));
        assert!(!spec.references.is_empty());
    }
}
