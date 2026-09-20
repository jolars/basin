use basin::{
    ArctanLoss, BoxConstraints, CauchyLoss, CostFunction, DenseMatrix,
    Executor, GaussNewton, HuberLoss, Jacobian, LevenbergMarquardt,
    LevenbergMarquardtQr, LmDamping, LossFunction, NllsState, Residual,
    RobustLeastSquares, SoftL1Loss, Solver, SquaredLoss, Trf,
    TrustRegionReflective,
};

#[derive(Clone)]
struct Fit {
    exponential: bool,
    lower: Vec<f64>,
    upper: Vec<f64>,
}
impl Fit {
    fn observations(&self) -> (Vec<f64>, Vec<f64>) {
        if self.exponential {
            let t: Vec<f64> = vec![0.2, 0.4, 0.7, 1., 1.5, 2.];
            let y = t
                .iter()
                .zip([0.01, -0.02, 0.015, 0.8, -0.01, 0.005])
                .map(|(&t, e)| (-0.7 * t).exp() + e)
                .collect();
            (t, y)
        } else {
            let t = vec![-2., -1., 0., 1., 2., 3.];
            let y = t
                .iter()
                .zip([0.02, -0.03, 0.01, -0.02, 4.03, 0.])
                .map(|(&t, e)| 1.5 * t + 0.3 + e)
                .collect();
            (t, y)
        }
    }
}
impl CostFunction for Fit {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, _: &Vec<f64>) -> Result<f64, Self::Error> {
        panic!("raw cost must not be called")
    }
}
impl Residual for Fit {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = std::convert::Infallible;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        let (t, y) = self.observations();
        Ok(t.iter()
            .zip(y)
            .map(|(&t, y)| {
                if self.exponential {
                    (-x[0] * t).exp() - y
                } else {
                    x[0] * t + x[1] - y
                }
            })
            .collect())
    }
}
impl Jacobian for Fit {
    type Jacobian = DenseMatrix;
    fn jacobian(&self, x: &Vec<f64>) -> Result<DenseMatrix, Self::Error> {
        let (t, _) = self.observations();
        Ok(DenseMatrix::from_fn(t.len(), x.len(), |i, j| {
            if self.exponential {
                -t[i] * (-x[0] * t[i]).exp()
            } else if j == 0 {
                t[i]
            } else {
                1.0
            }
        }))
    }
}
impl BoxConstraints for Fit {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

fn check<'a, S>(
    fit: Fit,
    loss: &'a dyn LossFunction,
    scale: f64,
    start: Vec<f64>,
    expected: &[f64],
    cost: f64,
    solver: S,
) where
    S: Solver<
            RobustLeastSquares<Fit, &'a dyn LossFunction>,
            NllsState<Vec<f64>>,
            Error = std::convert::Infallible,
        >,
{
    let objective = RobustLeastSquares::new(fit, loss).with_scale(scale);
    let result =
        Executor::new(objective.clone(), solver, NllsState::new(start))
            .max_iter(1000)
            .run()
            .unwrap();
    for (a, b) in result.param().iter().zip(expected) {
        assert!(
            (a - b).abs() < 2e-5,
            "{}: {:?} != {:?}",
            std::any::type_name::<S>(),
            result.param(),
            expected
        );
    }
    assert!(
        (result.cost() - cost).abs() < 1e-9,
        "{}: {} != {}",
        std::any::type_name::<S>(),
        result.cost(),
        cost
    );
    assert!(
        (result.cost() - objective.cost(result.param()).unwrap()).abs() < 1e-12
    );
}

#[test]
fn solvers_match_scipy_on_outlier_contaminated_fits() {
    for line in include_str!("fixtures/robust_least_squares_reference.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let fields: Vec<_> = line.split('|').collect();
        let loss: &dyn LossFunction = match fields[1] {
            "linear" => &SquaredLoss,
            "huber" => &HuberLoss,
            "soft_l1" => &SoftL1Loss,
            "cauchy" => &CauchyLoss,
            "arctan" => &ArctanLoss,
            _ => unreachable!(),
        };
        let numbers = |s: &str| {
            s.split(',')
                .map(|v| v.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        };
        let start = numbers(fields[3]);
        let expected = numbers(fields[4]);
        let fit = Fit {
            exponential: fields[0] == "exponential",
            lower: vec![-10.; start.len()],
            upper: vec![10.; start.len()],
        };
        let scale = fields[2].parse().unwrap();
        let cost = fields[5].parse().unwrap();
        macro_rules! run {
            ($solver:expr) => {
                check(
                    fit.clone(),
                    loss,
                    scale,
                    start.clone(),
                    &expected,
                    cost,
                    $solver,
                )
            };
        }
        // Full Gauss–Newton has no globalization; test its local convergence.
        check(
            fit.clone(),
            loss,
            scale,
            expected.iter().map(|x| x + 1e-4).collect(),
            &expected,
            cost,
            GaussNewton::new(),
        );
        for damping in [LmDamping::Nielsen, LmDamping::TrustRegion] {
            run!(LevenbergMarquardt::new().with_damping(damping));
            run!(LevenbergMarquardtQr::new().with_damping(damping));
        }
        run!(Trf::new());
        run!(TrustRegionReflective::new());
    }
}
