//! N-dimensional Rastrigin function.
//!
//! `f(x) = A·n + Σᵢ [xᵢ² − A·cos(2π·xᵢ)]`  with `A = 10`.
//!
//! Highly multimodal: a parabolic bowl `Σ xᵢ²` modulated by a cosine
//! ripple of amplitude `A` creates a dense lattice of local minima on
//! integer offsets. The global minimum sits at `x = (0, …, 0)` with
//! `f = 0`. Separable (the sum decomposes per coordinate), which is a
//! useful diagnostic — solvers that exploit separability handle it far
//! better than non-separable multimodal functions like Schwefel or
//! Ackley.
//!
//! The canonical search domain is `[−5.12, 5.12]^n`, set by Mühlenbein
//! et al. (1991) when they generalized Rastrigin's original 2D
//! formulation to arbitrary `n`. This is what the GA / evolutionary
//! optimization literature has used ever since (CEC competitions,
//! Bergmeir's MA-LSCh-CMA paper, etc.).
//!
//! Gradient is intentionally omitted: the function exists in basin's
//! corpus to exercise *global* solvers (CMA-ES variants, memetic
//! algorithms) which use cost evaluations only. Local gradient methods
//! stall in the nearest cosine pit and learn nothing about basin
//! geometry.

use core::marker::PhantomData;

use super::spec::{Dimensionality, HasSpec, ProblemSpec, Properties, Reference};
use crate::{BoxConstraints, CostFunction};

/// Rastrigin amplitude constant. Fixed at 10 by Mühlenbein et al.
/// (1991); essentially every paper that benchmarks on Rastrigin uses
/// this value.
const A: f64 = 10.0;

/// Standard lower bound on each coordinate (Mühlenbein et al. 1991).
pub const STANDARD_LOWER: f64 = -5.12;
/// Standard upper bound on each coordinate (Mühlenbein et al. 1991).
pub const STANDARD_UPPER: f64 = 5.12;

/// Evaluates the Rastrigin function at `x`. Works for any `n >= 1`.
pub fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let two_pi = 2.0 * core::f64::consts::PI;
    let mut s = A * n;
    for &v in x.iter() {
        s += v * v - A * (two_pi * v).cos();
    }
    s
}

/// Pre-wrapped Rastrigin problem. Generic over the parameter backend
/// `P`; the default `P = Vec<f64>` lets you write `Rastrigin::default()`
/// for the common case. Backend impls (`nalgebra::DVector<f64>`,
/// `ndarray::Array1<f64>`, `faer::Col<f64>`) are gated behind their
/// respective features.
///
/// Carries no constraint metadata. For solvers that need explicit box
/// bounds (e.g. CMA-ES variants), use [`RastriginBoxed`].
pub struct Rastrigin<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> Rastrigin<P> {
    /// Build a freshly typed problem instance. Pair with one of the
    /// backend-specific impl blocks (Vec, nalgebra, ndarray, faer).
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for Rastrigin<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Catalogue entry for this problem.
pub static RASTRIGIN_SPEC: ProblemSpec = ProblemSpec {
    name: "Rastrigin",
    dim: Dimensionality::NDimensional { min: 1 },
    properties: Properties {
        smooth: true,
        differentiable: true,
        // Non-convex: the cosine term creates many bumps.
        convex: false,
        // Highly multimodal — many local minima on an integer lattice.
        unimodal: false,
        // f(x) = Σᵢ gᵢ(xᵢ) with gᵢ(t) = A + t² − A·cos(2π·t); the
        // additive constant A·n is shared across coordinates but the
        // sum still decomposes per coordinate.
        separable: true,
        scalable: true,
    },
    references: &[
        Reference {
            citation: "Rastrigin (1974)",
            title: "Systems of extremal control",
            source: "Nauka, Moscow (in Russian)",
            doi: None,
            url: None,
        },
        Reference {
            citation: "Mühlenbein, Schomisch & Born (1991)",
            title: "The parallel genetic algorithm as function optimizer",
            source: "Parallel Computing, 17(6–7), 619–632",
            doi: Some("10.1016/S0167-8191(05)80052-3"),
            url: None,
        },
    ],
    description: "Highly multimodal: parabolic bowl Σ xᵢ² modulated by a \
                  cosine ripple of amplitude 10, giving a dense lattice of \
                  local minima. Global minimum at x = (0, …, 0), value 0. \
                  Standard search domain is [−5.12, 5.12]ⁿ (Mühlenbein \
                  et al. 1991). Separable, so coordinate-wise solvers \
                  handle it far better than non-separable multimodal \
                  functions like Schwefel.",
};

impl<P> HasSpec for Rastrigin<P> {
    const SPEC: &'static ProblemSpec = &RASTRIGIN_SPEC;
}

impl CostFunction for Rastrigin<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(rastrigin(x))
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_impl {
    use super::{Rastrigin, rastrigin};
    use crate::CostFunction;
    use nalgebra::DVector;

    impl CostFunction for Rastrigin<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(rastrigin(x.as_slice()))
        }
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::{Rastrigin, rastrigin};
    use crate::CostFunction;
    use ndarray::Array1;

    impl CostFunction for Rastrigin<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Array1<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(rastrigin(x.as_slice().expect("Array1 is contiguous")))
        }
    }
}

#[cfg(feature = "faer")]
mod faer_impl {
    use super::{A, Rastrigin};
    use crate::CostFunction;
    use faer::Col;

    // faer's `Col` doesn't expose a `&[f64]` directly across all 0.24 APIs
    // we care about, so we evaluate elementwise here rather than routing
    // through the slice-based primitive.
    impl CostFunction for Rastrigin<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            let n = x.nrows();
            let two_pi = 2.0 * core::f64::consts::PI;
            let mut s = A * n as f64;
            for i in 0..n {
                let v = x[i];
                s += v * v - A * (two_pi * v).cos();
            }
            Ok(s)
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

/// Rastrigin function with explicit element-wise box bounds, suitable
/// for solvers that require [`BoxConstraints`] (e.g. CMA-ES variants
/// like MA-LSCh-CMA). Carries the bounds as data on the problem (tenet
/// 4 in `crate::core` / `AGENTS.md`) and routes the cost through the
/// same raw [`rastrigin`] free function as the unconstrained
/// [`Rastrigin`].
///
/// The standard search domain `[−5.12, 5.12]ⁿ` from Mühlenbein et al.
/// (1991) is the common case; build it with
/// [`RastriginBoxed::with_standard_bounds`].
pub struct RastriginBoxed<P> {
    lower: P,
    upper: P,
}

impl<P> RastriginBoxed<P> {
    /// Build a Rastrigin problem with arbitrary element-wise bounds.
    /// Caller must ensure `lower[i] ≤ upper[i]` per component.
    pub fn new(lower: P, upper: P) -> Self {
        Self { lower, upper }
    }
}

impl<P> HasSpec for RastriginBoxed<P> {
    const SPEC: &'static ProblemSpec = &RASTRIGIN_SPEC;
}

impl RastriginBoxed<Vec<f64>> {
    /// Build the canonical Rastrigin instance on `[−5.12, 5.12]ⁿ` for
    /// the requested dimension `n`.
    pub fn with_standard_bounds(n: usize) -> Self {
        Self {
            lower: vec![STANDARD_LOWER; n],
            upper: vec![STANDARD_UPPER; n],
        }
    }
}

impl CostFunction for RastriginBoxed<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(rastrigin(x))
    }
}

impl BoxConstraints for RastriginBoxed<Vec<f64>> {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_boxed_impl {
    use super::{RastriginBoxed, STANDARD_LOWER, STANDARD_UPPER, rastrigin};
    use crate::{BoxConstraints, CostFunction};
    use nalgebra::DVector;

    impl RastriginBoxed<DVector<f64>> {
        /// Build the canonical Rastrigin instance on `[−5.12, 5.12]ⁿ`
        /// for the requested dimension `n`.
        pub fn with_standard_bounds(n: usize) -> Self {
            Self {
                lower: DVector::from_element(n, STANDARD_LOWER),
                upper: DVector::from_element(n, STANDARD_UPPER),
            }
        }
    }

    impl CostFunction for RastriginBoxed<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(rastrigin(x.as_slice()))
        }
    }

    impl BoxConstraints for RastriginBoxed<DVector<f64>> {
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
    use super::{RastriginBoxed, STANDARD_LOWER, STANDARD_UPPER, rastrigin};
    use crate::{BoxConstraints, CostFunction};
    use ndarray::Array1;

    impl RastriginBoxed<Array1<f64>> {
        /// Build the canonical Rastrigin instance on `[−5.12, 5.12]ⁿ`
        /// for the requested dimension `n`.
        pub fn with_standard_bounds(n: usize) -> Self {
            Self {
                lower: Array1::from_elem(n, STANDARD_LOWER),
                upper: Array1::from_elem(n, STANDARD_UPPER),
            }
        }
    }

    impl CostFunction for RastriginBoxed<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Array1<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(rastrigin(x.as_slice().expect("Array1 is contiguous")))
        }
    }

    impl BoxConstraints for RastriginBoxed<Array1<f64>> {
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
    use super::{A, RastriginBoxed, STANDARD_LOWER, STANDARD_UPPER};
    use crate::{BoxConstraints, CostFunction};
    use faer::Col;

    impl RastriginBoxed<Col<f64>> {
        /// Build the canonical Rastrigin instance on `[−5.12, 5.12]ⁿ`
        /// for the requested dimension `n`.
        pub fn with_standard_bounds(n: usize) -> Self {
            Self {
                lower: Col::<f64>::from_fn(n, |_| STANDARD_LOWER),
                upper: Col::<f64>::from_fn(n, |_| STANDARD_UPPER),
            }
        }
    }

    impl CostFunction for RastriginBoxed<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            let n = x.nrows();
            let two_pi = 2.0 * core::f64::consts::PI;
            let mut s = A * n as f64;
            for i in 0..n {
                let v = x[i];
                s += v * v - A * (two_pi * v).cos();
            }
            Ok(s)
        }
    }

    impl BoxConstraints for RastriginBoxed<Col<f64>> {
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
    fn rastrigin_minimum_is_zero_at_origin() {
        assert!(rastrigin(&[0.0]).abs() < 1e-12);
        assert!(rastrigin(&[0.0, 0.0]).abs() < 1e-12);
        assert!(rastrigin(&[0.0; 10]).abs() < 1e-12);
        assert!(rastrigin(&[0.0; 30]).abs() < 1e-12);
    }

    #[test]
    fn rastrigin_known_value_at_unit_offsets() {
        // At integer offsets cos(2π·k) = 1, so each coordinate
        // contributes A + k² − A = k². The constant A·n cancels with
        // the per-coordinate −A·cos = −A. Total:
        //   f(x) = A·n + Σᵢ (xᵢ² − A) = Σᵢ xᵢ²
        // For x = (1, 1, ..., 1), f = n. For n = 3 that's 3.
        assert!((rastrigin(&[1.0, 1.0, 1.0]) - 3.0).abs() < 1e-9);
        assert!((rastrigin(&[2.0, 2.0]) - 8.0).abs() < 1e-9);
    }

    #[test]
    fn rastrigin_local_minimum_at_half_integer_offset() {
        // The nearest local minima of the 1D component
        // g(t) = A + t² − A·cos(2π·t) lie near t ≈ ±1 (not exactly,
        // because the parabola tilts the cosine pits). Just verify
        // the value at t = 1: g(1) = A + 1 − A·1 = 1, so the local
        // pit value is exactly 1 there.
        assert!((rastrigin(&[1.0]) - 1.0).abs() < 1e-12);
        assert!((rastrigin(&[-1.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rastrigin_matches_definition_at_irregular_point() {
        // Hand-compute f(0.3, -0.7):
        //   A·n = 20
        //   0.3² + (-0.7)² = 0.09 + 0.49 = 0.58
        //   cos(2π·0.3) + cos(2π·(-0.7)) = cos(0.6π) + cos(-1.4π)
        //     = cos(0.6π) + cos(1.4π)         (cos is even)
        //     ≈ -0.30901699 + -0.30901699
        //     ≈ -0.61803398
        //   f = 20 + 0.58 − 10·(−0.61803398) = 20.58 + 6.1803398
        //     ≈ 26.7603398
        let f = rastrigin(&[0.3, -0.7]);
        assert!((f - 26.7603398874989).abs() < 1e-9, "got {f}");
    }

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        let spec = <Rastrigin<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(spec.name, "Rastrigin");
        assert!(spec.properties.smooth);
        assert!(spec.properties.differentiable);
        assert!(spec.properties.separable);
        assert!(spec.properties.scalable);
        assert!(!spec.properties.convex);
        assert!(!spec.properties.unimodal);
        assert!(matches!(spec.dim, Dimensionality::NDimensional { min: 1 }));
        assert!(!spec.references.is_empty());
    }

    #[test]
    fn boxed_form_exposes_standard_bounds() {
        let p = RastriginBoxed::<Vec<f64>>::with_standard_bounds(10);
        let lo = <RastriginBoxed<Vec<f64>> as BoxConstraints>::lower(&p);
        let hi = <RastriginBoxed<Vec<f64>> as BoxConstraints>::upper(&p);
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
        let unboxed: Rastrigin<Vec<f64>> = Rastrigin::default();
        let boxed = RastriginBoxed::<Vec<f64>>::with_standard_bounds(3);
        let x = vec![0.3, -0.7, 1.2];
        assert!((unboxed.cost(&x).unwrap() - boxed.cost(&x).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn boxed_form_reuses_rastrigin_spec() {
        let spec = <RastriginBoxed<Vec<f64>> as HasSpec>::SPEC;
        // Same static — both wrappers point at the one Rastrigin entry.
        assert!(core::ptr::eq(spec, &RASTRIGIN_SPEC));
    }
}
