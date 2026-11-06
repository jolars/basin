//! Model and projected-search cases with independently known box minima.
#![allow(dead_code)]

use basin::{
    BoxBounds, ConstraintSet, Constraints, CostFunction, Dot, FirstOrderState,
    Gradient, Lbfgsb, OptimizationResult, Scalar, ScaledAdd, VectorLen,
    solver::lbfgs::{AsFloatSlice, AsFloatSliceMut},
};
use std::convert::Infallible;

#[derive(Clone, Copy, Debug)]
pub enum Case {
    Rosenbrock2,
    Interior50,
    Active8One,
    Active8NonlinearOne,
    Active50Two,
}

impl Case {
    pub const ALL: [Self; 5] = [
        Self::Rosenbrock2,
        Self::Interior50,
        Self::Active8One,
        Self::Active8NonlinearOne,
        Self::Active50Two,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::Rosenbrock2 => "rosenbrock2",
            Self::Interior50 => "interior50",
            Self::Active8One => "active8_one",
            Self::Active8NonlinearOne => "active8_nonlinear_one",
            Self::Active50Two => "active50_two",
        }
    }
    pub fn history(self) -> usize {
        if matches!(self, Self::Active8One | Self::Active8NonlinearOne) {
            4
        } else {
            10
        }
    }
}

pub struct Quadratic<V, F: Scalar = f64> {
    pub case: Case,
    pub lower: V,
    pub upper: V,
    start: V,
    target: Vec<F>,
    center: Vec<F>,
    pub optimum: Vec<F>,
}

impl<V, F: Scalar> Quadratic<V, F> {
    pub fn new(case: Case, make: fn(&[F]) -> V) -> Self {
        let value = |x| F::from_f64(x).unwrap();
        let (n, free) = match case {
            Case::Rosenbrock2 => (2, 2),
            Case::Interior50 => (50, 50),
            Case::Active8One | Case::Active8NonlinearOne => (8, 1),
            Case::Active50Two => (50, 2),
        };
        let active = free < n;
        let optimum = vec![
            value(if matches!(case, Case::Rosenbrock2) {
                1.0
            } else {
                0.3
            });
            n
        ];
        let mut result = Self {
            case,
            lower: make(&vec![value(if active { 0.0 } else { -10.0 }); n]),
            upper: make(&vec![value(if active { 1.0 } else { 10.0 }); n]),
            start: make(
                &(0..n)
                    .map(|j| {
                        value(
                            if j == 0
                                && matches!(case, Case::Active8NonlinearOne)
                            {
                                0.9
                            } else if active {
                                0.6
                            } else {
                                1.5
                            },
                        )
                    })
                    .collect::<Vec<_>>(),
            ),
            target: vec![value(0.3); n],
            center: vec![value(0.3); n],
            optimum,
        };
        if active {
            for j in free..n {
                result.target[j] = value(2.0);
                result.center[j] = F::one();
                result.optimum[j] = F::one();
            }
        }
        if matches!(case, Case::Rosenbrock2) {
            result.start = make(&[value(-1.2), F::one()]);
        }
        result
    }
    pub fn tolerance(&self) -> F {
        F::from_f64(if F::epsilon() > F::from_f64(1e-10).unwrap() {
            0.01
        } else if matches!(self.case, Case::Rosenbrock2) {
            1e-8
        } else {
            1e-5
        })
        .unwrap()
    }
    pub fn solver(&self) -> Lbfgsb<V, F> {
        Lbfgsb::new()
            .with_m_capacity(self.case.history())
            .with_relative_cost_reduction_tolerance(None)
            .with_absolute_projected_gradient_tolerance(self.tolerance())
    }
    pub fn start(&self) -> V
    where
        V: Clone,
    {
        self.start.clone()
    }
}

impl<V, F: Scalar> Constraints for Quadratic<V, F> {
    type Set = ConstraintSet<BoxBounds<V>>;
    fn constraints(
        &self,
    ) -> <Self::Set as basin::ConstraintDescription>::Borrowed<'_> {
        ConstraintSet::new()
            .with_bounds(BoxBounds::new(Some(&self.lower), Some(&self.upper)))
    }
}

impl<V: AsFloatSlice<F>, F: Scalar> CostFunction for Quadratic<V, F> {
    type Param = V;
    type Output = F;
    type Error = Infallible;
    fn cost(&self, x: &V) -> Result<F, Infallible> {
        let x = x.as_float_slice();
        if matches!(self.case, Case::Rosenbrock2) {
            let a = x[1] - x[0] * x[0];
            let b = F::one() - x[0];
            return Ok(F::from_f64(100.0).unwrap() * a * a + b * b);
        }
        let sum: F = x.iter().zip(&self.center).map(|(x, c)| *x - *c).sum();
        let coupling =
            F::from_f64(0.25).unwrap() / F::from_usize(x.len()).unwrap();
        let squares: F = x
            .iter()
            .zip(&self.target)
            .enumerate()
            .map(|(j, (x, t))| {
                let d = *x - *t;
                if j == 0 && matches!(self.case, Case::Active8NonlinearOne) {
                    let a = F::from_f64(4.0).unwrap() * d;
                    (a.exp() - F::one() - a) / F::from_f64(8.0).unwrap()
                } else {
                    F::from_usize(j + 1).unwrap() * d * d
                }
            })
            .sum();
        Ok((squares + coupling * sum * sum) / F::from_f64(2.0).unwrap())
    }
}

impl<V: AsFloatSliceMut<F> + Clone, F: Scalar> Gradient for Quadratic<V, F> {
    type Gradient = V;
    fn gradient(&self, x: &V) -> Result<V, Infallible> {
        let mut out = x.clone();
        self.gradient_into(x, &mut out)?;
        Ok(out)
    }
    fn gradient_into(&self, x: &V, out: &mut V) -> Result<(), Infallible> {
        let x = x.as_float_slice();
        let out = out.as_float_slice_mut();
        if matches!(self.case, Case::Rosenbrock2) {
            let a = x[1] - x[0] * x[0];
            out[0] = -F::from_f64(400.0).unwrap() * x[0] * a
                - F::from_f64(2.0).unwrap() * (F::one() - x[0]);
            out[1] = F::from_f64(200.0).unwrap() * a;
        } else {
            let sum: F = x.iter().zip(&self.center).map(|(x, c)| *x - *c).sum();
            let correction = F::from_f64(0.25).unwrap() * sum
                / F::from_usize(x.len()).unwrap();
            for j in 0..x.len() {
                out[j] = if j == 0
                    && matches!(self.case, Case::Active8NonlinearOne)
                {
                    ((F::from_f64(4.0).unwrap() * (x[j] - self.target[j]))
                        .exp()
                        - F::one())
                        / F::from_f64(4.0).unwrap()
                        + correction
                } else {
                    F::from_usize(j + 1).unwrap() * (x[j] - self.target[j])
                        + correction
                };
            }
        }
        Ok(())
    }
}

impl<V, F> Quadratic<V, F>
where
    F: Scalar,
    V: AsFloatSliceMut<F> + Clone + Dot<F> + ScaledAdd<F> + VectorLen,
{
    pub fn verify(&self, result: &OptimizationResult<FirstOrderState<V, F>>) {
        let x = result.param().as_float_slice();
        for (j, &value) in x.iter().enumerate() {
            assert!(
                value.is_finite()
                    && value >= self.lower.as_float_slice()[j]
                    && value <= self.upper.as_float_slice()[j]
            );
            let tol =
                F::from_f64(if F::epsilon() > F::from_f64(1e-10).unwrap() {
                    0.05
                } else {
                    1e-4
                })
                .unwrap();
            assert!(
                (value - self.optimum[j]).abs() < tol,
                "{} coordinate {j}: {:?} vs {:?}",
                self.case.name(),
                value,
                self.optimum[j]
            );
        }
        let gradient = self.gradient(result.param()).unwrap();
        for (j, &value) in x.iter().enumerate() {
            let clipped = value
                - (value - gradient.as_float_slice()[j])
                    .max(self.lower.as_float_slice()[j])
                    .min(self.upper.as_float_slice()[j]);
            assert!(
                clipped.abs() <= F::from_f64(10.0).unwrap() * self.tolerance()
            );
        }
    }
}

impl<V, F: Scalar> Constraints for &Quadratic<V, F> {
    type Set = ConstraintSet<BoxBounds<V>>;
    fn constraints(
        &self,
    ) -> <Self::Set as basin::ConstraintDescription>::Borrowed<'_> {
        (**self).constraints()
    }
}
impl<V: AsFloatSlice<F>, F: Scalar> CostFunction for &Quadratic<V, F> {
    type Param = V;
    type Output = F;
    type Error = Infallible;
    fn cost(&self, x: &V) -> Result<F, Infallible> {
        (**self).cost(x)
    }
}
impl<V: AsFloatSliceMut<F> + Clone, F: Scalar> Gradient for &Quadratic<V, F> {
    type Gradient = V;
    fn gradient(&self, x: &V) -> Result<V, Infallible> {
        (**self).gradient(x)
    }
    fn gradient_into(&self, x: &V, out: &mut V) -> Result<(), Infallible> {
        (**self).gradient_into(x, out)
    }
}
