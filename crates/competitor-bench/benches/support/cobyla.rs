//! Compile Basin's actual private driver to guard the layer owning the gap.

#![allow(dead_code, clippy::needless_range_loop)]

#[path = "../../../basin/src/solver/cobyla/driver.rs"]
mod driver;
#[path = "../../../basin/src/solver/cobyla/filter.rs"]
mod filter;
#[path = "../../../basin/src/solver/cobyla/geometry.rs"]
mod geometry;
#[path = "../../../basin/src/solver/cobyla/init.rs"]
mod init;
#[path = "../../../basin/src/solver/cobyla/linalg.rs"]
mod linalg;
#[path = "../../../basin/src/solver/cobyla/model.rs"]
mod model;
#[path = "../../../basin/src/solver/cobyla/trstlp.rs"]
mod trstlp;
#[path = "../../../basin/src/solver/cobyla/update.rs"]
mod update;

use std::{cell::Cell, convert::Infallible};

use driver::{CobylaWork, Transition};

#[derive(Clone, Copy)]
pub enum Case {
    Camel,
    Sphere,
    SphereN(usize),
    Quadratic,
}

pub const CASES: [(Case, &str); 3] = [
    (Case::Camel, "camel"),
    (Case::Sphere, "sphere_10d"),
    (Case::Quadratic, "quadratic"),
];

pub const SCALING_CASES: [(Case, &str); 5] = [
    (Case::SphereN(1), "sphere_1d"),
    (Case::SphereN(3), "sphere_3d"),
    (Case::SphereN(5), "sphere_5d"),
    (Case::SphereN(20), "sphere_20d"),
    (Case::SphereN(40), "sphere_40d"),
];

pub struct Outcome {
    pub cost: f64,
    pub point: Vec<f64>,
    pub evaluations: usize,
    pub constraint_calls: usize,
    pub iterations: usize,
}

impl Case {
    pub fn solve(self) -> Outcome {
        let (start, bounds, budget) = match self {
            Self::Camel => (vec![0.0, 0.0], vec![(-3.0, 3.0), (-2.0, 2.0)], 50),
            Self::Sphere => (
                vec![2.5, -2.0, 1.5, -1.0, 0.5, 2.25, -1.75, 1.25, -0.75, 0.25],
                vec![(-5.0, 5.0); 10],
                200,
            ),
            Self::SphereN(n) => (
                (0..n).map(|i| ((i * 7) % 17) as f64 / 4.0 - 2.0).collect(),
                vec![(-5.0, 5.0); n],
                20 * n,
            ),
            Self::Quadratic => (vec![0.5, 0.5], vec![(0.0, 2.0); 2], 100),
        };
        let m = 2 * start.len() + usize::from(matches!(self, Self::Quadratic));
        let nf = Cell::new(0);
        let nc = Cell::new(0);
        let mut eval = |x: &[f64]| -> Result<_, Infallible> {
            let f = if nf.get() == budget {
                f64::INFINITY
            } else {
                nf.set(nf.get() + 1);
                match self {
                    Self::Camel => {
                        (4.0 - 2.1 * x[0].powi(2) + x[0].powi(4) / 3.0)
                            * x[0].powi(2)
                            + x[0] * x[1]
                            + (-4.0 + 4.0 * x[1].powi(2)) * x[1].powi(2)
                    }
                    Self::Sphere | Self::SphereN(_) => {
                        x.iter().map(|v| v * v).sum()
                    }
                    Self::Quadratic => {
                        (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2)
                    }
                }
            };
            nc.set(nc.get() + 1);
            let mut c = Vec::with_capacity(m);
            if matches!(self, Self::Quadratic) {
                c.push(-(1.5 - x[0] - x[1]));
            }
            for (&v, &(lower, upper)) in x.iter().zip(&bounds) {
                c.extend([lower - v, v - upper]);
            }
            Ok((f, c))
        };
        let (mut work, _, _) = CobylaWork::try_init(
            start,
            m,
            0.5,
            f64::EPSILON.sqrt() * 0.5,
            &mut eval,
        )
        .unwrap();
        let mut iterations = 0;
        while nc.get() < budget && iterations < 10000 {
            let transition = work.step(&mut eval).unwrap();
            iterations += 1;
            if matches!(transition, Transition::Converged | Transition::Failed)
            {
                break;
            }
        }
        let (point, cost) = work.best();
        Outcome {
            point,
            cost,
            evaluations: nf.get(),
            constraint_calls: nc.get(),
            iterations,
        }
    }
}
