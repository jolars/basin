//! Check the actual initial objective coordinates across all NM contestants.

use std::{cell::RefCell, convert::Infallible, rc::Rc};

use argmin::{
    core::Executor as ArgminExecutor,
    solver::neldermead::NelderMead as ArgminNelderMead,
};
use basin::problems::rosenbrock;
use basin::{CostFunction, NelderMead, Problem, SimplexProgress, Solver};
use competitor_bench::nm;
use gomez::nalgebra::{Dyn, IsContiguous, Storage, Vector};
use gomez::{Domain, OptimizerDriver};

#[derive(Clone)]
struct Recorded {
    n: usize,
    points: Rc<RefCell<Vec<Vec<f64>>>>,
}

impl Recorded {
    fn cost_at(&self, point: &[f64]) -> f64 {
        self.points.borrow_mut().push(point.to_vec());
        rosenbrock(point)
    }
}

impl CostFunction for Recorded {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, point: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(self.cost_at(point))
    }
}

impl argmin::core::CostFunction for Recorded {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, point: &Vec<f64>) -> Result<f64, argmin::core::Error> {
        Ok(self.cost_at(point))
    }
}

impl gomez::Problem for Recorded {
    type Field = f64;
    fn domain(&self) -> Domain<f64> {
        Domain::unconstrained(self.n)
    }
}

impl gomez::Function for Recorded {
    fn apply<S>(&self, point: &Vector<f64, Dyn, S>) -> f64
    where
        S: Storage<f64, Dyn> + IsContiguous,
    {
        self.cost_at(point.as_slice())
    }
}

#[test]
fn initial_objective_callbacks_receive_identical_vertices() {
    for n in [2, 12, 24] {
        let points = Rc::new(RefCell::new(Vec::new()));
        let problem = Recorded {
            n,
            points: points.clone(),
        };
        let expected = nm::initial_simplex(n);
        let check = |library| {
            let actual = points.borrow();
            assert!(
                actual.len() >= expected.len(),
                "{library}: missing initialization calls"
            );
            let bits = |vertices: &[Vec<f64>]| {
                vertices
                    .iter()
                    .flatten()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                bits(&actual[..expected.len()]),
                bits(&expected),
                "{library}, n={n}"
            );
        };

        let mut wrapped = Problem::new(problem.clone());
        let _ = NelderMead::new()
            .init(
                &mut wrapped,
                SimplexProgress::from_simplex(nm::initial_simplex(n)),
            )
            .unwrap();
        check("basin");
        points.borrow_mut().clear();

        ArgminExecutor::new(
            problem.clone(),
            ArgminNelderMead::new(nm::initial_simplex(n)),
        )
        .configure(|state| state.max_iters(0))
        .run()
        .unwrap();
        check("argmin");
        points.borrow_mut().clear();

        OptimizerDriver::builder(&problem)
            .with_initial(nm::start(n))
            .with_algo(nm::gomez_solver)
            .build()
            .next()
            .unwrap();
        check("gomez");
        points.borrow_mut().clear();

        let mut optimizer = nlopt::Nlopt::new(
            nlopt::Algorithm::Neldermead,
            n,
            |point, _, data: &mut Recorded| data.cost_at(point),
            nlopt::Target::Minimize,
            problem.clone(),
        );
        optimizer.set_initial_step(&nm::initial_steps(n)).unwrap();
        optimizer.set_maxeval((n + 1) as u32).unwrap();
        let _ = optimizer.optimize(&mut nm::start(n));
        check("nlopt");
    }
}
