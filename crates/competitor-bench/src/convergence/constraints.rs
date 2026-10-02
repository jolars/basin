//! Constrained controls with explicit feasibility and reference metadata.

use super::{
    Work,
    dual::{Dual, Real},
};
use basin::{
    ConstraintJacobian, CostFunction, DenseMatrix, Gradient,
    NonlinearConstraints, Scalar,
};
use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
};

#[derive(Clone, Copy, Debug)]
pub enum Kind {
    DiskActive,
    DiskInactive,
    Ellipse,
    Parabola,
    Equality,
    Bounds,
    Fixed,
    Hs71,
}

#[derive(Clone, Debug)]
pub struct Case {
    pub name: &'static str,
    pub family: &'static str,
    pub validation: bool,
    pub kind: Kind,
    pub starts: [Vec<f64>; 2],
    pub reference: Vec<f64>,
    pub reference_cost: f64,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

impl Case {
    pub fn cost<T: Real>(&self, p: &[T]) -> T {
        if matches!(self.kind, Kind::Hs71) {
            return p[0] * p[3] * (p[0] + p[1] + p[2]) + p[2];
        }
        let target = match self.kind {
            Kind::DiskActive => [2.0, 0.0],
            Kind::DiskInactive => [0.25, -0.25],
            Kind::Ellipse => [3.0, 0.0],
            Kind::Parabola => [1.0, 0.0],
            Kind::Equality => [1.0, 2.0],
            _ => [2.0, -1.0],
        };
        ((p[0] - T::number(target[0])).square()
            + (p[1] - T::number(target[1])).square())
            / T::number(2.0)
    }
    /// Equality rows precede inequality rows, following Basin's contract.
    pub fn constraints<T: Real>(&self, p: &[T]) -> Vec<T> {
        match self.kind {
            Kind::DiskActive | Kind::DiskInactive => {
                vec![p[0].square() + p[1].square() - T::number(1.0)]
            }
            Kind::Ellipse => vec![
                p[0].square() / T::number(4.0) + p[1].square() - T::number(1.0),
            ],
            Kind::Parabola => vec![p[0].square() - p[1]],
            Kind::Equality => vec![p[0] + p[1] - T::number(1.0)],
            Kind::Hs71 => vec![
                p.iter().fold(T::number(-40.0), |a, &v| a + v.square()),
                T::number(25.0) - p[0] * p[1] * p[2] * p[3],
            ],
            Kind::Bounds | Kind::Fixed => vec![],
        }
    }
    pub fn n_equalities(&self) -> usize {
        usize::from(matches!(self.kind, Kind::Hs71 | Kind::Equality))
    }
    pub fn n_inequalities(&self) -> usize {
        usize::from(matches!(
            self.kind,
            Kind::Hs71
                | Kind::DiskActive
                | Kind::DiskInactive
                | Kind::Ellipse
                | Kind::Parabola
        ))
    }
    pub fn violation(&self, p: &[f64]) -> f64 {
        if p.iter().any(|x| !x.is_finite()) {
            return f64::NAN;
        }
        let mut v: f64 = 0.0;
        for ((&x, &lo), &hi) in p.iter().zip(&self.lower).zip(&self.upper) {
            v = v.max(lo - x).max(x - hi);
        }
        for (i, x) in self.constraints(p).into_iter().enumerate() {
            if !x.is_finite() {
                return f64::NAN;
            }
            v = v.max(if i < self.n_equalities() { x.abs() } else { x });
        }
        v
    }
}

pub fn cases() -> Vec<Case> {
    use Kind::*;
    [
        DiskActive,
        DiskInactive,
        Ellipse,
        Parabola,
        Equality,
        Bounds,
        Fixed,
        Hs71,
    ]
    .into_iter()
    .map(|kind| {
        let (name, family, validation, reference, lower, upper) = match kind {
            DiskActive => (
                "disk_active",
                "disk",
                false,
                vec![1.0, 0.0],
                vec![-f64::INFINITY; 2],
                vec![f64::INFINITY; 2],
            ),
            DiskInactive => (
                "disk_inactive",
                "disk",
                false,
                vec![0.25, -0.25],
                vec![-f64::INFINITY; 2],
                vec![f64::INFINITY; 2],
            ),
            // Coordinate changes of a disk stay in its development family.
            Ellipse => (
                "ellipse_active",
                "disk",
                false,
                vec![2.0, 0.0],
                vec![-f64::INFINITY; 2],
                vec![f64::INFINITY; 2],
            ),
            Parabola => {
                let discriminant = (1.0_f64 / 16.0 + 1.0 / 216.0).sqrt();
                let x =
                    (0.25 + discriminant).cbrt() + (0.25 - discriminant).cbrt();
                (
                    "parabolic_inequality",
                    "parabola",
                    true,
                    vec![x, x * x],
                    vec![-f64::INFINITY; 2],
                    vec![f64::INFINITY; 2],
                )
            }
            Equality => (
                "linear_equality",
                "linear_equality",
                false,
                vec![0.0, 1.0],
                vec![-f64::INFINITY; 2],
                vec![f64::INFINITY; 2],
            ),
            Bounds => (
                "active_bounds",
                "box_quadratic",
                false,
                vec![1.0, 0.0],
                vec![0.0; 2],
                vec![1.0; 2],
            ),
            Fixed => (
                "fixed_coordinate",
                "box_quadratic",
                false,
                vec![1.0, 0.0],
                vec![0.0; 2],
                vec![1.0, 0.0],
            ),
            Hs71 => (
                "hs71",
                "hs71",
                true,
                vec![
                    1.0,
                    4.742999642848332,
                    3.8211499768953514,
                    1.3794082941785437,
                ],
                vec![1.0; 4],
                vec![5.0; 4],
            ),
        };
        let starts = if matches!(kind, Hs71) {
            [vec![1.0, 5.0, 5.0, 1.0], vec![2.0, 4.0, 4.0, 2.0]]
        } else if matches!(kind, Fixed) {
            [vec![0.25, 0.0], vec![0.75, 0.0]]
        } else {
            [vec![0.25, 0.25], vec![0.75, 0.5]]
        };
        let mut case = Case {
            name,
            family,
            validation,
            kind,
            starts,
            reference,
            reference_cost: 0.0,
            lower,
            upper,
        };
        case.reference_cost = case.cost(&case.reference);
        case
    })
    .collect()
}

#[derive(Clone)]
pub struct Fixture<F> {
    pub case: Case,
    pub work: Arc<Mutex<Work>>,
    lower: Vec<F>,
    upper: Vec<F>,
}
impl<F: Scalar + Real> Fixture<F> {
    pub fn new(case: Case) -> Self {
        let lower = case
            .lower
            .iter()
            .map(|&x| F::from_f64(x).unwrap())
            .collect();
        let upper = case
            .upper
            .iter()
            .map(|&x| F::from_f64(x).unwrap())
            .collect();
        Self {
            case,
            work: Arc::default(),
            lower,
            upper,
        }
    }
    fn variables(&self, p: &[F]) -> Vec<Dual<F>> {
        p.iter()
            .enumerate()
            .map(|(i, &x)| Dual::variable(x, i))
            .collect()
    }
}
impl<F: Scalar + Real> CostFunction for Fixture<F> {
    type Param = Vec<F>;
    type Output = F;
    type Error = Infallible;
    fn cost(&self, p: &Vec<F>) -> Result<F, Infallible> {
        let cost = self.case.cost(p);
        let mut work = self.work.lock().unwrap();
        work.value_passes += 1;
        work.observe(p, cost);
        Ok(cost)
    }
}
impl<F: Scalar + Real> Gradient for Fixture<F> {
    type Gradient = Vec<F>;
    fn gradient(&self, p: &Vec<F>) -> Result<Vec<F>, Infallible> {
        let cost = self.case.cost(&self.variables(p));
        let mut work = self.work.lock().unwrap();
        work.derivative_passes += 1;
        work.observe(p, cost.value);
        Ok(cost.derivative[..p.len()].to_vec())
    }
}
impl<F: Scalar + Real> NonlinearConstraints for Fixture<F> {
    type Matrix = DenseMatrix<F>;
    fn lower(&self) -> Option<&Vec<F>> {
        Some(&self.lower)
    }
    fn upper(&self) -> Option<&Vec<F>> {
        Some(&self.upper)
    }
    fn num_nonlinear_equalities(&self) -> usize {
        self.case.n_equalities()
    }
    fn num_nonlinear_constraints(&self) -> usize {
        self.case.n_inequalities()
    }
    fn nonlinear_constraints(&self, p: &Vec<F>) -> Result<Vec<F>, Infallible> {
        self.work.lock().unwrap().value_passes += 1;
        Ok(self.case.constraints(p)[self.case.n_equalities()..].to_vec())
    }
    fn nonlinear_equalities(
        &self,
        p: &Vec<F>,
    ) -> Result<Option<Vec<F>>, Infallible> {
        if self.case.n_equalities() == 0 {
            return Ok(None);
        }
        self.work.lock().unwrap().value_passes += 1;
        Ok(Some(
            self.case.constraints(p)[..self.case.n_equalities()].to_vec(),
        ))
    }
}
impl<F: Scalar + Real> ConstraintJacobian for Fixture<F> {
    fn constraint_jacobian(
        &self,
        p: &Vec<F>,
    ) -> Result<DenseMatrix<F>, Infallible> {
        let rows = self.case.constraints(&self.variables(p));
        self.work.lock().unwrap().derivative_passes += 1;
        let values: Vec<_> = rows
            .iter()
            .flat_map(|r| r.derivative[..p.len()].iter().copied())
            .collect();
        Ok(DenseMatrix::from_row_slice(rows.len(), p.len(), &values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn references_and_constraint_derivatives() {
        for case in cases() {
            assert!(case.violation(&case.reference) < 1e-8, "{}", case.name);
            let fixture = Fixture::<f64>::new(case.clone());
            for p in case.starts.iter().chain(std::iter::once(&case.reference))
            {
                let grad = fixture.gradient(p).unwrap();
                let rows = case.constraints(&fixture.variables(p));
                for col in 0..p.len() {
                    let mut a = p.clone();
                    let mut b = p.clone();
                    a[col] += 1e-5;
                    b[col] -= 1e-5;
                    assert!(
                        (grad[col] - (case.cost(&a) - case.cost(&b)) / 2e-5)
                            .abs()
                            < 1e-7
                    );
                    for ((row, a), b) in rows
                        .iter()
                        .zip(case.constraints(&a))
                        .zip(case.constraints(&b))
                    {
                        assert!(
                            (row.derivative[col] - (a - b) / 2e-5).abs() < 1e-7
                        );
                    }
                }
                let p32: Vec<_> = p.iter().map(|&x| x as f32).collect();
                let f32_fixture = Fixture::<f32>::new(case.clone());
                assert!(
                    (f32_fixture.cost(&p32).unwrap() as f64 - case.cost(p))
                        .abs()
                        < 1e-4
                );
                for (a, b) in
                    f32_fixture.gradient(&p32).unwrap().iter().zip(&grad)
                {
                    assert!((*a as f64 - b).abs() < 1e-4);
                }
            }
        }
        let hs71 = cases()
            .into_iter()
            .find(|c| matches!(c.kind, Kind::Hs71))
            .unwrap();
        assert!((hs71.reference_cost - 17.014017289134).abs() < 1e-10);
    }
}
