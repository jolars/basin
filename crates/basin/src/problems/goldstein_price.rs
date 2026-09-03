//! 2D Goldstein-Price function.
//!
//! ```text
//! f(x, y) = [1 + (x + y + 1)² · (19 − 14x + 3x² − 14y + 6xy + 3y²)]
//!         · [30 + (2x − 3y)² · (18 − 32x + 12x² + 48y − 36xy + 27y²)]
//! ```
//!
//! Smooth 2D polynomial test function with a single global minimum at
//! `(x, y) = (0, −1)` with `f = 3`. Usual search domain is
//! `x, y ∈ [-2, 2]`. The function has a very wide dynamic range, values
//! reach ~10⁶ near the corners of the standard domain, which makes it a
//! useful stress test for step-size control on first-order methods.
//! Quartic polynomial overall (degree 8 once the two factors multiply
//! out), with several saddle points and gentle plateaus around the
//! minimum.

use core::marker::PhantomData;

use super::spec::{
    Dimensionality, HasSpec, ProblemSpec, Properties, Reference,
};
use crate::{CostFunction, Gradient};

/// Evaluates the Goldstein-Price function at `x`. Requires `x.len() == 2`.
pub fn goldstein_price(x: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), 2);
    let (a, b) = (x[0], x[1]);
    let u = a + b + 1.0;
    let p =
        19.0 - 14.0 * a + 3.0 * a * a - 14.0 * b + 6.0 * a * b + 3.0 * b * b;
    let v = 2.0 * a - 3.0 * b;
    let q =
        18.0 - 32.0 * a + 12.0 * a * a + 48.0 * b - 36.0 * a * b + 27.0 * b * b;
    let big_a = 1.0 + u * u * p;
    let big_b = 30.0 + v * v * q;
    big_a * big_b
}

/// Writes the Goldstein-Price gradient at `x` into `out`. Both slices must have
/// length 2.
pub fn goldstein_price_gradient(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), 2);
    debug_assert_eq!(out.len(), 2);
    let (a, b) = (x[0], x[1]);

    // First factor A = 1 + u² · P, with u = x + y + 1.
    let u = a + b + 1.0;
    let p =
        19.0 - 14.0 * a + 3.0 * a * a - 14.0 * b + 6.0 * a * b + 3.0 * b * b;
    // ∂P/∂x = ∂P/∂y = -14 + 6x + 6y; ∂u/∂x = ∂u/∂y = 1, hence ∂A/∂x = ∂A/∂y.
    let dp = -14.0 + 6.0 * a + 6.0 * b;
    let big_a = 1.0 + u * u * p;
    let dadx = 2.0 * u * p + u * u * dp;
    let dady = dadx;

    // Second factor B = 30 + v² · Q, with v = 2x - 3y.
    let v = 2.0 * a - 3.0 * b;
    let q =
        18.0 - 32.0 * a + 12.0 * a * a + 48.0 * b - 36.0 * a * b + 27.0 * b * b;
    // ∂Q/∂x = -32 + 24x - 36y; ∂Q/∂y = 48 - 36x + 54y.
    // ∂(v²)/∂x = 4v; ∂(v²)/∂y = -6v.
    let dqdx = -32.0 + 24.0 * a - 36.0 * b;
    let dqdy = 48.0 - 36.0 * a + 54.0 * b;
    let big_b = 30.0 + v * v * q;
    let dbdx = 4.0 * v * q + v * v * dqdx;
    let dbdy = -6.0 * v * q + v * v * dqdy;

    // Product rule: f = A · B.
    out[0] = dadx * big_b + big_a * dbdx;
    out[1] = dady * big_b + big_a * dbdy;
}

/// Pre-wrapped Goldstein-Price problem (fixed 2D). Generic over the parameter
/// backend `P`; the default `P = Vec<f64>` lets you write
/// `GoldsteinPrice::default()` for the common case. Backend impls
/// (`nalgebra::DVector<f64>`, `ndarray::Array1<f64>`, `faer::Col<f64>`) are
/// gated behind their respective features.
pub struct GoldsteinPrice<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> GoldsteinPrice<P> {
    /// Build a freshly typed problem instance. Pair with one of the
    /// backend-specific impl blocks (Vec, nalgebra, ndarray, faer).
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for GoldsteinPrice<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Catalog entry for this problem.
pub static GOLDSTEIN_PRICE_SPEC: ProblemSpec = ProblemSpec {
    name: "Goldstein-Price",
    dim: Dimensionality::Fixed(2),
    properties: Properties {
        smooth: true,
        differentiable: true,
        convex: false,
        // Single strict local minimum on the usual domain x, y ∈ [-2, 2],
        // but the function has saddle points and the literature uses
        // "unimodal" inconsistently: keep the conservative call.
        unimodal: false,
        separable: false,
        scalable: false,
    },
    references: &[Reference {
        citation: "Goldstein & Price (1971)",
        title: "On Descent from Local Minima",
        source: "Mathematics of Computation, 25(115), 569–574",
        doi: Some("10.2307/2005219"),
        url: None,
    }],
    description: "Smooth 2D polynomial benchmark with a single global \
                  minimum at (x, y) = (0, −1), value 3. Usual search domain \
                  is x, y ∈ [-2, 2]; values reach ~10⁶ near the corners, so \
                  the wide dynamic range stresses step-size control on \
                  first-order methods.",
};

impl<P> HasSpec for GoldsteinPrice<P> {
    const SPEC: &'static ProblemSpec = &GOLDSTEIN_PRICE_SPEC;
}

impl CostFunction for GoldsteinPrice<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(goldstein_price(x))
    }
}

impl Gradient for GoldsteinPrice<Vec<f64>> {
    type Gradient = Vec<f64>;
    fn gradient(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, std::convert::Infallible> {
        let mut out = vec![0.0; x.len()];
        goldstein_price_gradient(x, &mut out);
        Ok(out)
    }
}

#[cfg(feature = "nalgebra_all")]
mod nalgebra_impl {
    use super::{GoldsteinPrice, goldstein_price, goldstein_price_gradient};
    use crate::{CostFunction, Gradient};
    use nalgebra::DVector;

    impl CostFunction for GoldsteinPrice<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &DVector<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(goldstein_price(x.as_slice()))
        }
    }

    impl Gradient for GoldsteinPrice<DVector<f64>> {
        type Gradient = DVector<f64>;
        fn gradient(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, std::convert::Infallible> {
            let mut out = DVector::zeros(x.len());
            goldstein_price_gradient(x.as_slice(), out.as_mut_slice());
            Ok(out)
        }
    }
}

#[cfg(feature = "ndarray_all")]
mod ndarray_impl {
    use super::{GoldsteinPrice, goldstein_price, goldstein_price_gradient};
    use crate::{CostFunction, Gradient};
    use ndarray::Array1;

    impl CostFunction for GoldsteinPrice<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &Array1<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(goldstein_price(x.as_slice().expect("Array1 is contiguous")))
        }
    }

    impl Gradient for GoldsteinPrice<Array1<f64>> {
        type Gradient = Array1<f64>;
        fn gradient(
            &self,
            x: &Array1<f64>,
        ) -> Result<Array1<f64>, std::convert::Infallible> {
            let mut out = Array1::zeros(x.len());
            goldstein_price_gradient(
                x.as_slice().expect("Array1 is contiguous"),
                out.as_slice_mut().expect("Array1 is contiguous"),
            );
            Ok(out)
        }
    }
}

#[cfg(feature = "faer_all")]
mod faer_impl {
    use super::GoldsteinPrice;
    use crate::{CostFunction, Gradient};
    use faer::Col;

    // faer's `Col` doesn't expose a `&[f64]` directly across all 0.24 APIs we
    // care about, so we evaluate elementwise here rather than routing through
    // the slice-based primitives.
    impl CostFunction for GoldsteinPrice<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            debug_assert_eq!(x.nrows(), 2);
            let (a, b) = (x[0], x[1]);
            let u = a + b + 1.0;
            let p = 19.0 - 14.0 * a + 3.0 * a * a - 14.0 * b
                + 6.0 * a * b
                + 3.0 * b * b;
            let v = 2.0 * a - 3.0 * b;
            let q = 18.0 - 32.0 * a + 12.0 * a * a + 48.0 * b - 36.0 * a * b
                + 27.0 * b * b;
            Ok((1.0 + u * u * p) * (30.0 + v * v * q))
        }
    }

    impl Gradient for GoldsteinPrice<Col<f64>> {
        type Gradient = Col<f64>;
        fn gradient(
            &self,
            x: &Col<f64>,
        ) -> Result<Col<f64>, std::convert::Infallible> {
            debug_assert_eq!(x.nrows(), 2);
            let (a, b) = (x[0], x[1]);
            let u = a + b + 1.0;
            let p = 19.0 - 14.0 * a + 3.0 * a * a - 14.0 * b
                + 6.0 * a * b
                + 3.0 * b * b;
            let dp = -14.0 + 6.0 * a + 6.0 * b;
            let big_a = 1.0 + u * u * p;
            let dadx = 2.0 * u * p + u * u * dp;
            let dady = dadx;

            let v = 2.0 * a - 3.0 * b;
            let q = 18.0 - 32.0 * a + 12.0 * a * a + 48.0 * b - 36.0 * a * b
                + 27.0 * b * b;
            let dqdx = -32.0 + 24.0 * a - 36.0 * b;
            let dqdy = 48.0 - 36.0 * a + 54.0 * b;
            let big_b = 30.0 + v * v * q;
            let dbdx = 4.0 * v * q + v * v * dqdx;
            let dbdy = -6.0 * v * q + v * v * dqdy;

            let g0 = dadx * big_b + big_a * dbdx;
            let g1 = dady * big_b + big_a * dbdy;
            Ok(Col::<f64>::from_fn(2, |i| if i == 0 { g0 } else { g1 }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goldstein_price_minimum_is_three_at_known_optimum() {
        assert!((goldstein_price(&[0.0, -1.0]) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn goldstein_price_known_value_at_origin() {
        // At (0, 0): u = 1, P = 19, A = 1 + 1·19 = 20.
        // v = 0, B = 30 + 0 = 30. f = 20 · 30 = 600.
        assert!((goldstein_price(&[0.0, 0.0]) - 600.0).abs() < 1e-9);
    }

    #[test]
    fn gradient_zero_at_minimum() {
        let mut g = vec![0.0; 2];
        goldstein_price_gradient(&[0.0, -1.0], &mut g);
        for v in g {
            assert!(v.abs() < 1e-9);
        }
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let x = [-1.2, 0.7];
        let mut g = vec![0.0; x.len()];
        goldstein_price_gradient(&x, &mut g);

        let h = 1e-6;
        for i in 0..x.len() {
            let mut xp = x;
            let mut xm = x;
            xp[i] += h;
            xm[i] -= h;
            let fd = (goldstein_price(&xp) - goldstein_price(&xm)) / (2.0 * h);
            // Function magnitude here is ~10³–10⁴, so absolute tolerance is
            // loose but relative error is still small.
            let rel = (g[i] - fd).abs() / (1.0 + fd.abs());
            assert!(rel < 1e-5, "i={i}, g={}, fd={fd}", g[i]);
        }
    }

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        let spec = <GoldsteinPrice<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(spec.name, "Goldstein-Price");
        assert!(spec.properties.smooth);
        assert!(spec.properties.differentiable);
        assert!(matches!(spec.dim, Dimensionality::Fixed(2)));
        assert!(!spec.references.is_empty());
    }
}
