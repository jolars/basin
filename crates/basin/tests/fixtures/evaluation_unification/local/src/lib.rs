use std::{cell::Cell, rc::Rc};

use basin::{
    BoundedFiniteDiff, CmaEs, CostFunction, DenseMatrix, DerivativeCheckError,
    DerivativeChecker, Executor, FiniteDiff, Gradient, GradientDescent,
    Hessian, Jacobian, PopulationProgress, Residual,
    central_difference_gradient, forward_difference_hessian,
    forward_difference_jacobian,
};

pub trait CostBatch<P: CostFunction> {
    fn costs(
        &self,
        problem: &P,
        points: &[P::Param],
    ) -> Result<Vec<P::Output>, P::Error>;
}

pub trait GradientPair<P: Gradient<Param = Vec<f64>, Gradient = Vec<f64>>> {
    fn gradients(
        &self,
        problem: &P,
        points: [&Vec<f64>; 2],
    ) -> Result<[Vec<f64>; 2], P::Error>;
}

#[derive(Clone, Copy)]
pub struct Serial;

impl<P: CostFunction> CostBatch<P> for Serial {
    fn costs(
        &self,
        problem: &P,
        points: &[P::Param],
    ) -> Result<Vec<P::Output>, P::Error> {
        points.iter().map(|point| problem.cost(point)).collect()
    }
}

pub fn gradient<P>(problem: &P, point: &Vec<f64>) -> Result<Vec<f64>, P::Error>
where
    P: Gradient<Param = Vec<f64>, Gradient = Vec<f64>>,
{
    let mut output = point.clone();
    problem.gradient_into(point, &mut output)?;
    Ok(output)
}

impl<P> GradientPair<P> for Serial
where
    P: Gradient<Param = Vec<f64>, Gradient = Vec<f64>>,
{
    fn gradients(
        &self,
        problem: &P,
        [first, second]: [&Vec<f64>; 2],
    ) -> Result<[Vec<f64>; 2], P::Error> {
        Ok([gradient(problem, first)?, gradient(problem, second)?])
    }
}

pub struct Difference<P, Mode> {
    pub problem: P,
    pub mode: Mode,
}

pub const STEP: f64 = 1.0 / 1024.0;

impl<P: CostFunction<Param = Vec<f64>, Output = f64>, Mode> CostFunction
    for Difference<P, Mode>
{
    type Param = Vec<f64>;
    type Output = f64;
    type Error = P::Error;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        self.problem.cost(x)
    }
}

impl<P, Mode> Gradient for Difference<P, Mode>
where
    P: CostFunction<Param = Vec<f64>, Output = f64>,
    Mode: CostBatch<P>,
{
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        let mut points = Vec::with_capacity(2 * x.len());
        for j in 0..x.len() {
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus[j] += STEP;
            minus[j] -= STEP;
            points.extend([plus, minus]);
        }
        Ok(self
            .mode
            .costs(&self.problem, &points)?
            .chunks_exact(2)
            .map(|pair| (pair[0] - pair[1]) / (2.0 * STEP))
            .collect())
    }
}

#[derive(Clone, Copy)]
pub enum Stencil {
    Forward,
    Central,
}

pub fn product<P, Mode>(
    mode: &Mode,
    problem: &P,
    x: &Vec<f64>,
    v: &[f64],
    stencil: Stencil,
) -> Result<Vec<f64>, P::Error>
where
    P: Gradient<Param = Vec<f64>, Gradient = Vec<f64>>,
    Mode: GradientPair<P>,
{
    assert_eq!(x.len(), v.len());
    if v.iter().all(|&coordinate| coordinate == 0.0) {
        return Ok(vec![0.0; x.len()]);
    }
    let probe = |step| x.iter().zip(v).map(|(&x, &v)| x + step * v).collect();
    let plus = probe(STEP);
    let (positive, negative, width) = match stencil {
        Stencil::Forward => {
            let [base, plus] = mode.gradients(problem, [x, &plus])?;
            (plus, base, STEP)
        }
        Stencil::Central => {
            let minus = probe(-STEP);
            let [plus, minus] = mode.gradients(problem, [&plus, &minus])?;
            (plus, minus, 2.0 * STEP)
        }
    };
    Ok(positive
        .into_iter()
        .zip(negative)
        .map(|(a, b)| (a - b) / width)
        .collect())
}

pub fn quadratic(x: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(j, &x)| (j + 1) as f64 * x * x / 2.0)
        .sum()
}

#[derive(Default)]
pub struct Local {
    pub calls: Rc<Cell<usize>>,
    pub error: Option<Rc<str>>,
}

impl CostFunction for Local {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Rc<str>;

    fn cost(&self, point: &Vec<f64>) -> Result<f64, Self::Error> {
        self.calls.set(self.calls.get() + 1);
        match &self.error {
            Some(error) => Err(Rc::clone(error)),
            None => Ok(quadratic(point)),
        }
    }
}

impl Gradient for Local {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(x.iter()
            .enumerate()
            .map(|(j, x)| (j + 1) as f64 * x)
            .collect())
    }
}

impl Residual for Local {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = Rc<str>;

    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        self.cost(x).map(|value| vec![value])
    }
}

impl Jacobian for Local {
    type Jacobian = DenseMatrix;

    fn jacobian(&self, x: &Vec<f64>) -> Result<DenseMatrix, Self::Error> {
        let gradient = self.gradient(x)?;
        Ok(DenseMatrix::from_row_slice(1, gradient.len(), &gradient))
    }
}

pub fn check_local_derivatives() {
    let x = vec![2.0, 1.0];
    let fd = FiniteDiff::new(Local::default()).with_step(STEP);
    assert_eq!(fd.gradient(&x).unwrap(), [2.0, 2.0]);
    assert_eq!(
        fd.hessian(&x).unwrap(),
        DenseMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0])
    );
    let bounded = fd.with_bounds(vec![0.0; 2], vec![3.0; 2]);
    assert_eq!(bounded.gradient(&x).unwrap(), [2.0, 2.0]);
    assert!(bounded.jacobian(&x).is_ok());

    let checker = DerivativeChecker::new().with_step(STEP);
    assert!(
        checker
            .check_gradient(bounded.get_ref(), &x)
            .unwrap()
            .passed()
    );
    assert!(
        checker
            .check_jacobian(bounded.get_ref(), &x)
            .unwrap()
            .passed()
    );

    let error: Rc<str> = "local failure".into();
    let problem = Local {
        error: Some(Rc::clone(&error)),
        ..Local::default()
    };
    let actual = central_difference_gradient(&problem, &x, f64::EPSILON, None)
        .unwrap_err();
    assert!(Rc::ptr_eq(&error, &actual));
    let actual = forward_difference_jacobian(&problem, &x, f64::EPSILON, None)
        .unwrap_err();
    assert!(Rc::ptr_eq(&error, &actual));
    let actual = forward_difference_hessian(&problem, &x, f64::EPSILON, None)
        .unwrap_err();
    assert!(Rc::ptr_eq(&error, &actual));
    match checker.check_gradient(&problem, &x).unwrap_err() {
        DerivativeCheckError::Evaluation(actual) => {
            assert!(Rc::ptr_eq(&error, &actual))
        }
        other => panic!("expected the callback error, got {other:?}"),
    }
    let actual = BoundedFiniteDiff::new(problem, vec![0.0; 2], vec![3.0; 2])
        .gradient(&x)
        .unwrap_err();
    assert!(Rc::ptr_eq(&error, &actual));
}

pub fn check_local_runs() {
    let problem = Local::default();
    let calls = Rc::clone(&problem.calls);
    let hook_calls = Rc::new(Cell::new(0));
    let observed = Rc::clone(&hook_calls);
    let result = Executor::from_start(
        FiniteDiff::new(problem).with_step(STEP),
        GradientDescent::new(0.25),
        vec![2.0, 1.0],
    )
    .max_iter(2)
    .into_local()
    .stop_when(move |_| {
        observed.set(observed.get() + 1);
        None
    })
    .run()
    .unwrap();
    assert_eq!(result.param(), &[1.125, 0.25]);
    assert_eq!(result.counts.gradient_evals, 3);
    assert_eq!(result.cost_evals(), 3);
    assert_eq!(calls.get(), 15);
    assert_eq!(hook_calls.get(), 2);

    let problem = Local::default();
    let calls = Rc::clone(&problem.calls);
    let result = Executor::new(
        problem,
        CmaEs::<Vec<f64>, DenseMatrix>::new(7, 0.25).with_lambda(4),
        PopulationProgress::from_point(vec![2.0, 1.0]),
    )
    .max_iter(2)
    .run()
    .unwrap();
    assert_eq!(result.cost_evals(), 15);
    assert_eq!(calls.get(), 15);
}

#[test]
fn local_runs() {
    check_local_runs();
    check_local_derivatives();
}
