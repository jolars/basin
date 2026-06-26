//! N-dimensional Ackley function.
//!
//! ```text
//! f(x) = −a·exp(−b·√((1/n)·Σ xᵢ²)) − exp((1/n)·Σ cos(c·xᵢ)) + a + e
//! ```
//!
//! with the conventional constants `a = 20`, `b = 0.2`, `c = 2π`. A nearly-flat
//! outer region (dominated by the first exponential) surrounds a deep central
//! funnel toward the global minimum at `x = (0, …, 0)` with `f = 0`, while the
//! cosine term studs the whole surface with a regular lattice of shallow local
//! minima. The `√(Σ xᵢ²)` term has a cusp at the origin, so the function is
//! **not differentiable at its own minimum**: basin treats it as cost-only,
//! for global and derivative-free solvers (cf. [`Rastrigin`](super::Rastrigin)).
//! Standard search domain is `[−32.768, 32.768]^n` (Bäck 1996).

use core::marker::PhantomData;

use super::spec::{Dimensionality, HasSpec, ProblemSpec, Properties, Reference};
use crate::{BoxConstraints, CostFunction};

/// First Ackley constant (overall amplitude of the funnel).
const A: f64 = 20.0;
/// Second Ackley constant (funnel decay rate).
const B: f64 = 0.2;

/// Standard lower bound on each coordinate (Bäck 1996).
pub const STANDARD_LOWER: f64 = -32.768;
/// Standard upper bound on each coordinate (Bäck 1996).
pub const STANDARD_UPPER: f64 = 32.768;

/// Evaluates the Ackley function at `x`. Works for any `n >= 1`.
pub fn ackley(x: &[f64]) -> f64 {
    debug_assert!(!x.is_empty());
    let n = x.len() as f64;
    let c = 2.0 * core::f64::consts::PI;
    let mut sum_sq = 0.0;
    let mut sum_cos = 0.0;
    for &v in x.iter() {
        sum_sq += v * v;
        sum_cos += (c * v).cos();
    }
    -A * (-B * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp() + A + core::f64::consts::E
}

/// Pre-wrapped Ackley problem. Cost-only (non-differentiable at the origin), so
/// no `Gradient` impl is provided. Generic over the parameter backend `P`; the
/// default `P = Vec<f64>` lets you write `Ackley::default()` for the common
/// case. Backend impls (`nalgebra::DVector<f64>`, `ndarray::Array1<f64>`,
/// `faer::Col<f64>`) are gated behind their respective features.
///
/// Carries no constraint metadata. For solvers that need explicit box bounds
/// (e.g. CMA-ES variants), use [`AckleyBoxed`].
pub struct Ackley<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> Ackley<P> {
    /// Build a freshly typed problem instance. Pair with one of the
    /// backend-specific impl blocks (Vec, nalgebra, ndarray, faer).
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for Ackley<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Catalog entry for this problem.
pub static ACKLEY_SPEC: ProblemSpec = ProblemSpec {
    name: "Ackley",
    dim: Dimensionality::NDimensional { min: 1 },
    properties: Properties {
        // √(Σ xᵢ²) has a cusp at the origin (the global minimum).
        smooth: false,
        differentiable: false,
        convex: false,
        unimodal: false,
        // The exp of a mean couples all coordinates, not separable.
        separable: false,
        scalable: true,
    },
    references: &[
        Reference {
            citation: "Ackley (1987)",
            title: "A Connectionist Machine for Genetic Hillclimbing",
            source: "Kluwer Academic Publishers, Boston, MA",
            doi: Some("10.1007/978-1-4613-1997-9"),
            url: None,
        },
        Reference {
            citation: "Bäck (1996)",
            title: "Evolutionary Algorithms in Theory and Practice",
            source: "Oxford University Press (a = 20, b = 0.2, c = 2π, domain [−32.768, 32.768]ⁿ)",
            doi: None,
            url: None,
        },
    ],
    description: "Nearly-flat outer region around a deep central funnel to the \
                  global minimum at x = (0, …, 0), value 0, with a cosine \
                  lattice of shallow local minima. Constants a = 20, b = 0.2, \
                  c = 2π; standard domain [−32.768, 32.768]ⁿ. Non-differentiable \
                  at the origin (cusp), so cost-only for global solvers.",
};

impl<P> HasSpec for Ackley<P> {
    const SPEC: &'static ProblemSpec = &ACKLEY_SPEC;
}

impl CostFunction for Ackley<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(ackley(x))
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_impl {
    use super::{Ackley, ackley};
    use crate::CostFunction;
    use nalgebra::DVector;

    impl CostFunction for Ackley<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(ackley(x.as_slice()))
        }
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::{Ackley, ackley};
    use crate::CostFunction;
    use ndarray::Array1;

    impl CostFunction for Ackley<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Array1<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(ackley(x.as_slice().expect("Array1 is contiguous")))
        }
    }
}

#[cfg(feature = "faer")]
mod faer_impl {
    use super::{A, Ackley, B};
    use crate::CostFunction;
    use faer::Col;

    // faer's `Col` doesn't expose a `&[f64]` directly across all 0.24 APIs we
    // care about, so we evaluate elementwise here rather than routing through
    // the slice-based primitive.
    impl CostFunction for Ackley<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            let n = x.nrows();
            let c = 2.0 * core::f64::consts::PI;
            let mut sum_sq = 0.0;
            let mut sum_cos = 0.0;
            for i in 0..n {
                let v = x[i];
                sum_sq += v * v;
                sum_cos += (c * v).cos();
            }
            let nf = n as f64;
            Ok(
                -A * (-B * (sum_sq / nf).sqrt()).exp() - (sum_cos / nf).exp()
                    + A
                    + core::f64::consts::E,
            )
        }
    }
}

// ----------------------------------------------------------------------
// Boxed (constrained) form
// ----------------------------------------------------------------------

/// Ackley function with explicit element-wise box bounds, suitable for solvers
/// that require [`BoxConstraints`] (e.g. CMA-ES variants). Carries the bounds
/// as data on the problem and routes the cost through the same raw [`ackley`]
/// free function as the unconstrained [`Ackley`]. The standard search domain
/// `[−32.768, 32.768]ⁿ` is the common case; build it with
/// [`AckleyBoxed::with_standard_bounds`].
pub struct AckleyBoxed<P> {
    lower: P,
    upper: P,
}

impl<P> AckleyBoxed<P> {
    /// Build an Ackley problem with arbitrary element-wise bounds. Caller must
    /// ensure `lower[i] ≤ upper[i]` per component.
    pub fn new(lower: P, upper: P) -> Self {
        Self { lower, upper }
    }
}

impl<P> HasSpec for AckleyBoxed<P> {
    const SPEC: &'static ProblemSpec = &ACKLEY_SPEC;
}

impl AckleyBoxed<Vec<f64>> {
    /// Build the canonical Ackley instance on `[−32.768, 32.768]ⁿ` for the
    /// requested dimension `n`.
    pub fn with_standard_bounds(n: usize) -> Self {
        Self {
            lower: vec![STANDARD_LOWER; n],
            upper: vec![STANDARD_UPPER; n],
        }
    }
}

impl CostFunction for AckleyBoxed<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(ackley(x))
    }
}

impl BoxConstraints for AckleyBoxed<Vec<f64>> {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_boxed_impl {
    use super::{AckleyBoxed, STANDARD_LOWER, STANDARD_UPPER, ackley};
    use crate::{BoxConstraints, CostFunction};
    use nalgebra::DVector;

    impl AckleyBoxed<DVector<f64>> {
        /// Build the canonical Ackley instance on `[−32.768, 32.768]ⁿ` for the
        /// requested dimension `n`.
        pub fn with_standard_bounds(n: usize) -> Self {
            Self {
                lower: DVector::from_element(n, STANDARD_LOWER),
                upper: DVector::from_element(n, STANDARD_UPPER),
            }
        }
    }

    impl CostFunction for AckleyBoxed<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(ackley(x.as_slice()))
        }
    }

    impl BoxConstraints for AckleyBoxed<DVector<f64>> {
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
    use super::{AckleyBoxed, STANDARD_LOWER, STANDARD_UPPER, ackley};
    use crate::{BoxConstraints, CostFunction};
    use ndarray::Array1;

    impl AckleyBoxed<Array1<f64>> {
        /// Build the canonical Ackley instance on `[−32.768, 32.768]ⁿ` for the
        /// requested dimension `n`.
        pub fn with_standard_bounds(n: usize) -> Self {
            Self {
                lower: Array1::from_elem(n, STANDARD_LOWER),
                upper: Array1::from_elem(n, STANDARD_UPPER),
            }
        }
    }

    impl CostFunction for AckleyBoxed<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Array1<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(ackley(x.as_slice().expect("Array1 is contiguous")))
        }
    }

    impl BoxConstraints for AckleyBoxed<Array1<f64>> {
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
    use super::{A, AckleyBoxed, B, STANDARD_LOWER, STANDARD_UPPER};
    use crate::{BoxConstraints, CostFunction};
    use faer::Col;

    impl AckleyBoxed<Col<f64>> {
        /// Build the canonical Ackley instance on `[−32.768, 32.768]ⁿ` for the
        /// requested dimension `n`.
        pub fn with_standard_bounds(n: usize) -> Self {
            Self {
                lower: Col::<f64>::from_fn(n, |_| STANDARD_LOWER),
                upper: Col::<f64>::from_fn(n, |_| STANDARD_UPPER),
            }
        }
    }

    impl CostFunction for AckleyBoxed<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            let n = x.nrows();
            let c = 2.0 * core::f64::consts::PI;
            let mut sum_sq = 0.0;
            let mut sum_cos = 0.0;
            for i in 0..n {
                let v = x[i];
                sum_sq += v * v;
                sum_cos += (c * v).cos();
            }
            let nf = n as f64;
            Ok(
                -A * (-B * (sum_sq / nf).sqrt()).exp() - (sum_cos / nf).exp()
                    + A
                    + core::f64::consts::E,
            )
        }
    }

    impl BoxConstraints for AckleyBoxed<Col<f64>> {
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
    fn minimum_is_zero_at_origin() {
        assert!(ackley(&[0.0]).abs() < 1e-12);
        assert!(ackley(&[0.0, 0.0]).abs() < 1e-12);
        assert!(ackley(&[0.0; 10]).abs() < 1e-12);
    }

    #[test]
    fn known_value_at_unit_point() {
        // f(1, 1): mean(x²) = 1, mean(cos 2π) = 1, so
        //   f = −20·exp(−0.2) − exp(1) + 20 + e = −20·exp(−0.2) + 20 ≈ 3.62538…
        let f = ackley(&[1.0, 1.0]);
        assert!((f - 3.6253849384).abs() < 1e-9, "got {f}");
    }

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        let spec = <Ackley<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(spec.name, "Ackley");
        assert!(!spec.properties.smooth);
        assert!(!spec.properties.differentiable);
        assert!(spec.properties.scalable);
        assert!(matches!(spec.dim, Dimensionality::NDimensional { min: 1 }));
        assert!(!spec.references.is_empty());
    }

    #[test]
    fn boxed_form_exposes_standard_bounds() {
        let p = AckleyBoxed::<Vec<f64>>::with_standard_bounds(5);
        let lo = <AckleyBoxed<Vec<f64>> as BoxConstraints>::lower(&p);
        let hi = <AckleyBoxed<Vec<f64>> as BoxConstraints>::upper(&p);
        assert_eq!(lo.len(), 5);
        assert_eq!(hi.len(), 5);
        for &v in lo {
            assert_eq!(v, STANDARD_LOWER);
        }
        for &v in hi {
            assert_eq!(v, STANDARD_UPPER);
        }
    }

    #[test]
    fn boxed_form_shares_cost_with_unboxed() {
        let unboxed: Ackley<Vec<f64>> = Ackley::default();
        let boxed = AckleyBoxed::<Vec<f64>>::with_standard_bounds(3);
        let x = vec![0.3, -0.7, 1.2];
        assert!((unboxed.cost(&x).unwrap() - boxed.cost(&x).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn boxed_form_reuses_ackley_spec() {
        let spec = <AckleyBoxed<Vec<f64>> as HasSpec>::SPEC;
        assert!(core::ptr::eq(spec, &ACKLEY_SPEC));
    }
}
