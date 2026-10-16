//! Four fixed variants isolate units and an irreducible residual floor.

use super::{DMatrix, DVector, Fixture, compare};
use basin::{Jacobian, Residual};

#[derive(Clone)]
struct Affine {
    name: &'static str,
    residual_scale: f64,
    coordinate_units: [f64; 2],
    floor: f64,
}

impl basin::ProblemConstraints for Affine {
    type Constraints = basin::NoConstraints;

    fn constraint_set(
        &self,
    ) -> <Self::Constraints as basin::ConstraintDescription>::Borrowed<'_> {
        basin::Constraints::new()
    }
}

impl Residual for Affine {
    type Param = DVector<f64>;
    type Output = DVector<f64>;
    type Error = String;

    fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, String> {
        Ok(DVector::from_vec(vec![
            self.residual_scale * (x[0] / self.coordinate_units[0] - 0.3),
            self.residual_scale * (x[1] / self.coordinate_units[1] - 0.7),
            self.residual_scale * self.floor,
        ]))
    }
}

impl Jacobian for Affine {
    type Jacobian = DMatrix<f64>;

    fn jacobian(&self, _: &DVector<f64>) -> Result<DMatrix<f64>, String> {
        Ok(DMatrix::from_row_slice(
            3,
            2,
            &[
                self.residual_scale / self.coordinate_units[0],
                0.,
                0.,
                self.residual_scale / self.coordinate_units[1],
                0.,
                0.,
            ],
        ))
    }
}

impl Fixture for Affine {
    fn id(&self) -> &str {
        self.name
    }

    fn starts(&self) -> Vec<Vec<f64>> {
        vec![vec![0., self.coordinate_units[1]]]
    }

    fn reference_point(&self) -> Vec<f64> {
        vec![
            0.3 * self.coordinate_units[0],
            0.7 * self.coordinate_units[1],
        ]
    }

    fn reference_rss(&self) -> f64 {
        (self.residual_scale * self.floor).powi(2)
    }
}

pub fn run() {
    for (name, residual_scale, coordinate_units, floor) in [
        ("affine-unit", 1., [1., 1.], 0.),
        ("affine-small-residual", 1e-6, [1., 1.], 0.),
        ("affine-coordinate-units", 1., [1., 1e6], 0.),
        ("affine-residual-floor", 1., [1., 1.], 2.),
    ] {
        compare(Affine {
            name,
            residual_scale,
            coordinate_units,
            floor,
        });
    }
}
