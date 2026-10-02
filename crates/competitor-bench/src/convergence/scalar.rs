//! Scalar controls distinguish function accuracy from position accuracy.

use super::{Work, dual::Real};
use basin::{BoxConstraints, CostFunction, Gradient, Scalar};
use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
};

#[derive(Clone, Copy, Debug)]
pub enum Kind {
    RootQuadratic,
    RootScaled,
    RootFlat,
    RootSine,
    MinimumQuadratic,
    MinimumFlat,
    MinimumExponential,
}

#[derive(Clone, Debug)]
pub struct Case {
    pub name: &'static str,
    pub family: &'static str,
    pub validation: bool,
    pub kind: Kind,
    pub reference: f64,
    pub intervals: [[f64; 2]; 2],
}
impl Case {
    pub fn is_root(&self) -> bool {
        matches!(
            self.kind,
            Kind::RootQuadratic
                | Kind::RootScaled
                | Kind::RootFlat
                | Kind::RootSine
        )
    }
    pub fn value<T: Real>(&self, x: T) -> T {
        match self.kind {
            Kind::RootQuadratic => x.square() - T::number(2.0),
            Kind::RootScaled => {
                T::number(1e-12) * (x.square() - T::number(2.0))
            }
            Kind::RootFlat => {
                (x - T::number(1.0)).square() * (x - T::number(1.0))
            }
            Kind::RootSine => x.sin(),
            Kind::MinimumQuadratic => (x - T::number(0.25)).square(),
            Kind::MinimumFlat => (x - T::number(0.25)).square().square(),
            Kind::MinimumExponential => x.exp() - x - T::number(1.0),
        }
    }
    pub fn derivative<T: Real>(&self, x: T) -> T {
        match self.kind {
            Kind::RootQuadratic => T::number(2.0) * x,
            Kind::RootScaled => T::number(2e-12) * x,
            Kind::RootFlat => T::number(3.0) * (x - T::number(1.0)).square(),
            Kind::RootSine => x.cos(),
            Kind::MinimumQuadratic => T::number(2.0) * (x - T::number(0.25)),
            Kind::MinimumFlat => {
                T::number(4.0)
                    * (x - T::number(0.25)).square()
                    * (x - T::number(0.25))
            }
            Kind::MinimumExponential => x.exp() - T::number(1.0),
        }
    }
    pub fn second<T: Real>(&self, x: T) -> T {
        match self.kind {
            Kind::RootQuadratic => T::number(2.0),
            Kind::RootScaled => T::number(2e-12),
            Kind::RootFlat => T::number(6.0) * (x - T::number(1.0)),
            Kind::RootSine => -x.sin(),
            Kind::MinimumQuadratic => T::number(2.0),
            Kind::MinimumFlat => {
                T::number(12.0) * (x - T::number(0.25)).square()
            }
            Kind::MinimumExponential => x.exp(),
        }
    }
}
pub fn cases() -> Vec<Case> {
    use Kind::*;
    [
        RootQuadratic,
        RootScaled,
        RootFlat,
        RootSine,
        MinimumQuadratic,
        MinimumFlat,
        MinimumExponential,
    ]
    .into_iter()
    .map(|kind| {
        let (name, family, validation, reference, intervals) = match kind {
            RootQuadratic => (
                "root_simple",
                "root_square",
                false,
                2_f64.sqrt(),
                [[0.0, 2.0], [1.0, 3.0]],
            ),
            RootScaled => (
                "root_scaled",
                "root_square",
                false,
                2_f64.sqrt(),
                [[0.0, 2.0], [1.0, 3.0]],
            ),
            RootFlat => (
                "root_multiple",
                "flat_polynomial",
                false,
                1.0,
                [[-0.2, 2.0], [0.8, 3.0]],
            ),
            RootSine => (
                "root_sine",
                "trigonometric",
                true,
                std::f64::consts::PI,
                [[2.5, 4.0], [2.0, 3.5]],
            ),
            MinimumQuadratic => (
                "minimum_quadratic",
                "quadratic",
                false,
                0.25,
                [[-1.0, 2.0], [0.0, 3.0]],
            ),
            MinimumFlat => (
                "minimum_flat",
                "flat_polynomial",
                false,
                0.25,
                [[-1.0, 2.0], [0.0, 3.0]],
            ),
            MinimumExponential => (
                "minimum_exponential",
                "scalar_exponential",
                true,
                0.0,
                [[-2.0, 1.0], [-1.0, 3.0]],
            ),
        };
        Case {
            name,
            family,
            validation,
            reference,
            intervals,
            kind,
        }
    })
    .collect()
}

#[derive(Clone)]
pub struct Fixture<F> {
    pub case: Case,
    pub work: Arc<Mutex<Work>>,
    lower: F,
    upper: F,
}
impl<F: Scalar + Real> Fixture<F> {
    pub fn new(case: Case, interval: usize) -> Self {
        let [lo, hi] = case.intervals[interval];
        Self {
            case,
            work: Arc::default(),
            lower: F::from_f64(lo).unwrap(),
            upper: F::from_f64(hi).unwrap(),
        }
    }
}
impl<F: Scalar + Real> CostFunction for Fixture<F> {
    type Param = F;
    type Output = F;
    type Error = Infallible;
    fn cost(&self, x: &F) -> Result<F, Infallible> {
        let cost = self.case.value(*x);
        let mut work = self.work.lock().unwrap();
        work.value_passes += 1;
        work.observe(&[*x], cost);
        Ok(cost)
    }
}
impl<F: Scalar + Real> Gradient for Fixture<F> {
    type Gradient = F;
    fn gradient(&self, x: &F) -> Result<F, Infallible> {
        self.work.lock().unwrap().derivative_passes += 1;
        Ok(self.case.derivative(*x))
    }
}
impl<F: Scalar + Real> BoxConstraints for Fixture<F> {
    fn lower(&self) -> &F {
        &self.lower
    }
    fn upper(&self) -> &F {
        &self.upper
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scalar_references_and_derivatives() {
        for case in cases() {
            assert!(case.value(case.reference).abs() < 1e-14);
            for [a, b] in case.intervals {
                if case.is_root() {
                    assert!(case.value(a) * case.value(b) < 0.0);
                }
                for x in [a, (a + b) / 2.0, b] {
                    let h = 1e-5;
                    let d = (case.value(x + h) - case.value(x - h)) / (2.0 * h);
                    let dd = (case.derivative(x + h) - case.derivative(x - h))
                        / (2.0 * h);
                    assert!((case.derivative(x) - d).abs() < 1e-7);
                    assert!((case.second(x) - dd).abs() < 1e-7);
                    assert!(
                        (case.value(x as f32) as f64 - case.value(x)).abs()
                            < 1e-5
                    );
                }
            }
        }
    }
}
