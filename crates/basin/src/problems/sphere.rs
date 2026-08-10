//! N-dimensional Sphere function.
//!
//! `f(x) = Σᵢ xᵢ²`
//!
//! Smooth, convex, separable, unimodal. Global minimum at `x = (0, …, 0)`
//! with `f = 0`. The trivial canary problem: every solver should solve it
//! cleanly; failure indicates the implementation is broken.

use core::marker::PhantomData;

use super::spec::{
    Dimensionality, HasSpec, ProblemSpec, Properties, Reference,
};
use crate::{BoxConstraints, CostFunction, Gradient};

/// Standard lower bound on each coordinate (De Jong 1975).
pub const STANDARD_LOWER: f64 = -5.12;
/// Standard upper bound on each coordinate (De Jong 1975).
pub const STANDARD_UPPER: f64 = 5.12;

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
                  trivial canary: every solver should solve it cleanly.",
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
    fn gradient(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, std::convert::Infallible> {
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
        fn cost(
            &self,
            x: &DVector<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(sphere(x.as_slice()))
        }
    }

    impl Gradient for Sphere<DVector<f64>> {
        type Gradient = DVector<f64>;
        fn gradient(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, std::convert::Infallible> {
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
        fn cost(
            &self,
            x: &Array1<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(sphere(x.as_slice().expect("Array1 is contiguous")))
        }
    }

    impl Gradient for Sphere<Array1<f64>> {
        type Gradient = Array1<f64>;
        fn gradient(
            &self,
            x: &Array1<f64>,
        ) -> Result<Array1<f64>, std::convert::Infallible> {
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
        fn gradient(
            &self,
            x: &Col<f64>,
        ) -> Result<Col<f64>, std::convert::Infallible> {
            let n = x.nrows();
            Ok(Col::<f64>::from_fn(n, |i| 2.0 * x[i]))
        }
    }
}

// ----------------------------------------------------------------------
// Boxed (constrained) form
// ----------------------------------------------------------------------
// Carries element-wise bounds on the struct so it can implement
// `BoxConstraints` for solvers that require explicit box constraints
// (CMA-ES variants, projected methods). The standard `[−5.12, 5.12]ⁿ`
// search domain is the most common choice; `with_standard_bounds(n)`
// is a shortcut for that case.

/// Sphere function with explicit element-wise box bounds, suitable for
/// solvers that require [`BoxConstraints`] (e.g. CMA-ES variants like
/// MA-LSCh-CMA). Carries the bounds as data on the problem (tenet 4 in
/// `crate::core` and `CONTRIBUTING.md`) and routes the cost through the
/// same raw [`sphere`] free function as the unconstrained [`Sphere`].
///
/// The standard search domain `[−5.12, 5.12]ⁿ` from De Jong (1975) is
/// the common case; build it with [`SphereBoxed::with_standard_bounds`].
pub struct SphereBoxed<P> {
    lower: P,
    upper: P,
}

impl<P> SphereBoxed<P> {
    /// Build a Sphere problem with arbitrary element-wise bounds.
    /// Caller must ensure `lower[i] ≤ upper[i]` per component.
    pub fn new(lower: P, upper: P) -> Self {
        Self { lower, upper }
    }
}

impl<P> HasSpec for SphereBoxed<P> {
    const SPEC: &'static ProblemSpec = &SPHERE_SPEC;
}

impl SphereBoxed<Vec<f64>> {
    /// Build the canonical Sphere instance on `[−5.12, 5.12]ⁿ` for the
    /// requested dimension `n`.
    pub fn with_standard_bounds(n: usize) -> Self {
        Self {
            lower: vec![STANDARD_LOWER; n],
            upper: vec![STANDARD_UPPER; n],
        }
    }
}

impl CostFunction for SphereBoxed<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(sphere(x))
    }
}

impl BoxConstraints for SphereBoxed<Vec<f64>> {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_boxed_impl {
    use super::{STANDARD_LOWER, STANDARD_UPPER, SphereBoxed, sphere};
    use crate::{BoxConstraints, CostFunction};
    use nalgebra::DVector;

    impl SphereBoxed<DVector<f64>> {
        /// Build the canonical Sphere instance on `[−5.12, 5.12]ⁿ` for
        /// the requested dimension `n`.
        pub fn with_standard_bounds(n: usize) -> Self {
            Self {
                lower: DVector::from_element(n, STANDARD_LOWER),
                upper: DVector::from_element(n, STANDARD_UPPER),
            }
        }
    }

    impl CostFunction for SphereBoxed<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &DVector<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(sphere(x.as_slice()))
        }
    }

    impl BoxConstraints for SphereBoxed<DVector<f64>> {
        fn lower(&self) -> &DVector<f64> {
            &self.lower
        }
        fn upper(&self) -> &DVector<f64> {
            &self.upper
        }
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_boxed_impl {
    use super::{STANDARD_LOWER, STANDARD_UPPER, SphereBoxed, sphere};
    use crate::{BoxConstraints, CostFunction};
    use ndarray::Array1;

    impl SphereBoxed<Array1<f64>> {
        /// Build the canonical Sphere instance on `[−5.12, 5.12]ⁿ` for
        /// the requested dimension `n`.
        pub fn with_standard_bounds(n: usize) -> Self {
            Self {
                lower: Array1::from_elem(n, STANDARD_LOWER),
                upper: Array1::from_elem(n, STANDARD_UPPER),
            }
        }
    }

    impl CostFunction for SphereBoxed<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &Array1<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(sphere(x.as_slice().expect("Array1 is contiguous")))
        }
    }

    impl BoxConstraints for SphereBoxed<Array1<f64>> {
        fn lower(&self) -> &Array1<f64> {
            &self.lower
        }
        fn upper(&self) -> &Array1<f64> {
            &self.upper
        }
    }
}

#[cfg(feature = "faer")]
mod faer_boxed_impl {
    use super::{STANDARD_LOWER, STANDARD_UPPER, SphereBoxed};
    use crate::{BoxConstraints, CostFunction};
    use faer::Col;

    impl SphereBoxed<Col<f64>> {
        /// Build the canonical Sphere instance on `[−5.12, 5.12]ⁿ` for
        /// the requested dimension `n`.
        pub fn with_standard_bounds(n: usize) -> Self {
            Self {
                lower: Col::<f64>::from_fn(n, |_| STANDARD_LOWER),
                upper: Col::<f64>::from_fn(n, |_| STANDARD_UPPER),
            }
        }
    }

    // faer's `Col` doesn't expose a `&[f64]` directly across all 0.24
    // APIs we care about, so we evaluate elementwise here rather than
    // routing through the slice-based primitive.
    impl CostFunction for SphereBoxed<Col<f64>> {
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

    impl BoxConstraints for SphereBoxed<Col<f64>> {
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

    #[test]
    fn boxed_form_exposes_standard_bounds() {
        let p = SphereBoxed::<Vec<f64>>::with_standard_bounds(10);
        let lo = <SphereBoxed<Vec<f64>> as BoxConstraints>::lower(&p);
        let hi = <SphereBoxed<Vec<f64>> as BoxConstraints>::upper(&p);
        assert_eq!(lo.len(), 10);
        assert_eq!(hi.len(), 10);
        for &v in lo {
            assert_eq!(v, STANDARD_LOWER);
        }
        for &v in hi {
            assert_eq!(v, STANDARD_UPPER);
        }
    }

    #[test]
    fn boxed_form_shares_cost_with_unboxed() {
        let unboxed: Sphere<Vec<f64>> = Sphere::default();
        let boxed = SphereBoxed::<Vec<f64>>::with_standard_bounds(3);
        let x = vec![0.3, -0.7, 1.2];
        assert!(
            (unboxed.cost(&x).unwrap() - boxed.cost(&x).unwrap()).abs() < 1e-12
        );
    }

    #[test]
    fn boxed_form_reuses_sphere_spec() {
        let spec = <SphereBoxed<Vec<f64>> as HasSpec>::SPEC;
        // Same static: both wrappers point at the one Sphere entry.
        assert!(core::ptr::eq(spec, &SPHERE_SPEC));
    }
}
