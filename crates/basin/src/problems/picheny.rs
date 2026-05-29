//! 2D Picheny log-rescaled Goldstein-Price function.
//!
//! ```text
//! f̄(x) = (1 / 2.427) · (ln(GP(x̄)) − 8.693),   x̄ᵢ = 4·xᵢ − 2
//! ```
//!
//! where `GP` is the standard [Goldstein-Price](super::GoldsteinPrice)
//! function. Introduced by Picheny, Wagner & Ginsbourger (2013) to put a
//! standard benchmark on the unit square `[0, 1]²` and tame Goldstein-Price's
//! enormous dynamic range: the natural-log transform compresses the ~10⁶ spread
//! into an `O(1)` surface with mean ≈ 0 and unit-ish variance, so it has the
//! same landscape shape but very different conditioning — handy for probing
//! line-search and step-control behavior. The inputs are mapped back onto
//! Goldstein-Price's `[-2, 2]²` domain by `x̄ = 4x − 2`. Global minimum at
//! `x = (0.5, 0.25)` (the image of GP's minimizer `(0, −1)`), value
//! `(ln 3 − 8.693) / 2.427 ≈ −3.1291`.

use core::marker::PhantomData;

use super::goldstein_price::{goldstein_price, goldstein_price_gradient};
use super::spec::{Dimensionality, HasSpec, ProblemSpec, Properties, Reference};
use crate::{CostFunction, Gradient};

/// Scale divisor in the log-rescaling (Picheny et al. 2013).
const SCALE: f64 = 2.427;
/// Additive shift in the log-rescaling (Picheny et al. 2013).
const SHIFT: f64 = 8.693;

/// Standard lower bound on each coordinate (the unit square).
pub const STANDARD_LOWER: f64 = 0.0;
/// Standard upper bound on each coordinate (the unit square).
pub const STANDARD_UPPER: f64 = 1.0;

/// Maps a `[0, 1]²` point onto Goldstein-Price's `[-2, 2]²` domain.
#[inline]
fn rescale(x: &[f64]) -> [f64; 2] {
    [4.0 * x[0] - 2.0, 4.0 * x[1] - 2.0]
}

/// Evaluates the Picheny log-Goldstein-Price function at `x`. Requires
/// `x.len() == 2`.
pub fn picheny(x: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), 2);
    let xbar = rescale(x);
    (goldstein_price(&xbar).ln() - SHIFT) / SCALE
}

/// Writes the Picheny gradient at `x` into `out`. Both slices must have
/// length 2.
pub fn picheny_gradient(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), 2);
    debug_assert_eq!(out.len(), 2);
    let xbar = rescale(x);
    let gp = goldstein_price(&xbar);
    let mut gpg = [0.0; 2];
    goldstein_price_gradient(&xbar, &mut gpg);
    // Chain rule: ∂f̄/∂xᵢ = (1/SCALE)·(1/GP)·(∂GP/∂x̄ᵢ)·(dx̄ᵢ/dxᵢ), dx̄ᵢ/dxᵢ = 4.
    let s = 4.0 / (SCALE * gp);
    out[0] = s * gpg[0];
    out[1] = s * gpg[1];
}

/// Pre-wrapped Picheny problem (fixed 2D). Generic over the parameter backend
/// `P`; the default `P = Vec<f64>` lets you write `Picheny::default()` for the
/// common case. Backend impls (`nalgebra::DVector<f64>`, `ndarray::Array1<f64>`,
/// `faer::Col<f64>`) are gated behind their respective features.
pub struct Picheny<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> Picheny<P> {
    /// Build a freshly typed problem instance. Pair with one of the
    /// backend-specific impl blocks (Vec, nalgebra, ndarray, faer).
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for Picheny<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Catalogue entry for this problem.
pub static PICHENY_SPEC: ProblemSpec = ProblemSpec {
    name: "Picheny",
    dim: Dimensionality::Fixed(2),
    properties: Properties {
        smooth: true,
        differentiable: true,
        convex: false,
        // Inherits Goldstein-Price's saddles/plateaus; conservative call.
        unimodal: false,
        separable: false,
        scalable: false,
    },
    references: &[
        Reference {
            citation: "Picheny, Wagner & Ginsbourger (2013)",
            title: "A benchmark of kriging-based infill criteria for noisy optimization",
            source: "Structural and Multidisciplinary Optimization, 48(3), 607–626",
            doi: Some("10.1007/s00158-013-0919-4"),
            url: None,
        },
        Reference {
            citation: "Goldstein & Price (1971)",
            title: "On Descent from Local Minima",
            source: "Mathematics of Computation, 25(115), 569–574",
            doi: Some("10.2307/2005219"),
            url: None,
        },
    ],
    description: "Log-rescaled Goldstein-Price on the unit square [0, 1]²: the \
                  natural-log transform compresses GP's ~10⁶ dynamic range into \
                  an O(1) surface, same shape but different conditioning. Global \
                  minimum at x = (0.5, 0.25), value ≈ −3.1291; useful for \
                  probing line-search and step-control behavior.",
};

impl<P> HasSpec for Picheny<P> {
    const SPEC: &'static ProblemSpec = &PICHENY_SPEC;
}

impl CostFunction for Picheny<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(picheny(x))
    }
}

impl Gradient for Picheny<Vec<f64>> {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; x.len()];
        picheny_gradient(x, &mut out);
        Ok(out)
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_impl {
    use super::{Picheny, picheny, picheny_gradient};
    use crate::{CostFunction, Gradient};
    use nalgebra::DVector;

    impl CostFunction for Picheny<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(picheny(x.as_slice()))
        }
    }

    impl Gradient for Picheny<DVector<f64>> {
        type Gradient = DVector<f64>;
        fn gradient(&self, x: &DVector<f64>) -> Result<DVector<f64>, std::convert::Infallible> {
            let mut out = DVector::zeros(x.len());
            picheny_gradient(x.as_slice(), out.as_mut_slice());
            Ok(out)
        }
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::{Picheny, picheny, picheny_gradient};
    use crate::{CostFunction, Gradient};
    use ndarray::Array1;

    impl CostFunction for Picheny<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Array1<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(picheny(x.as_slice().expect("Array1 is contiguous")))
        }
    }

    impl Gradient for Picheny<Array1<f64>> {
        type Gradient = Array1<f64>;
        fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, std::convert::Infallible> {
            let mut out = Array1::zeros(x.len());
            picheny_gradient(
                x.as_slice().expect("Array1 is contiguous"),
                out.as_slice_mut().expect("Array1 is contiguous"),
            );
            Ok(out)
        }
    }
}

#[cfg(feature = "faer")]
mod faer_impl {
    // Routes through the slice-based primitives via a small stack array, since
    // the math is fixed-2D and reuses the Goldstein-Price free functions.
    use super::{Picheny, picheny, picheny_gradient};
    use crate::{CostFunction, Gradient};
    use faer::Col;

    impl CostFunction for Picheny<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            debug_assert_eq!(x.nrows(), 2);
            Ok(picheny(&[x[0], x[1]]))
        }
    }

    impl Gradient for Picheny<Col<f64>> {
        type Gradient = Col<f64>;
        fn gradient(&self, x: &Col<f64>) -> Result<Col<f64>, std::convert::Infallible> {
            debug_assert_eq!(x.nrows(), 2);
            let mut g = [0.0; 2];
            picheny_gradient(&[x[0], x[1]], &mut g);
            Ok(Col::<f64>::from_fn(2, |i| g[i]))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimum_at_rescaled_optimum() {
        // GP minimizer (0, −1) maps to x = (0.5, 0.25); GP value there is 3.
        let expected = (3.0_f64.ln() - SHIFT) / SCALE;
        let f = picheny(&[0.5, 0.25]);
        assert!((f - expected).abs() < 1e-12, "got {f}, want {expected}");
        // Sanity on the documented numeric value.
        assert!((f - (-3.1291)).abs() < 1e-3, "got {f}");
    }

    #[test]
    fn known_value_at_domain_center() {
        // x = (0.5, 0.5) maps to x̄ = (0, 0); GP(0, 0) = 600.
        let expected = (600.0_f64.ln() - SHIFT) / SCALE;
        assert!((picheny(&[0.5, 0.5]) - expected).abs() < 1e-12);
    }

    #[test]
    fn gradient_zero_at_minimum() {
        let mut g = vec![0.0; 2];
        picheny_gradient(&[0.5, 0.25], &mut g);
        for v in g {
            assert!(v.abs() < 1e-9);
        }
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let x = [0.3, 0.7];
        let mut g = vec![0.0; x.len()];
        picheny_gradient(&x, &mut g);
        let h = 1e-6;
        for i in 0..x.len() {
            let mut xp = x;
            let mut xm = x;
            xp[i] += h;
            xm[i] -= h;
            let fd = (picheny(&xp) - picheny(&xm)) / (2.0 * h);
            let rel = (g[i] - fd).abs() / (1.0 + fd.abs());
            assert!(rel < 1e-5, "i={i}, g={}, fd={fd}", g[i]);
        }
    }

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        let spec = <Picheny<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(spec.name, "Picheny");
        assert!(spec.properties.smooth);
        assert!(spec.properties.differentiable);
        assert!(matches!(spec.dim, Dimensionality::Fixed(2)));
        assert!(!spec.references.is_empty());
    }
}
