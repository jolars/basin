use basin::{
    BrentDerivative, CostFunction, Executor, FirstOrderState, Gbnm, Gradient,
    PointState, Problem, Scalar, Solver, State, TerminationCode, VectorLen,
};
use std::convert::Infallible;

struct Linear<F> {
    lower: F,
    upper: F,
}

impl<F: Scalar> basin::ProblemConstraints for Linear<F> {
    type Constraints = basin::Constraints<basin::BoxBounds<F>>;
    fn constraint_set(
        &self,
    ) -> <Self::Constraints as basin::ConstraintDescription>::Borrowed<'_> {
        basin::Constraints::new().with_bounds(basin::BoxBounds::new(
            Some(&self.lower),
            Some(&self.upper),
        ))
    }
}
impl<F: Scalar> CostFunction for Linear<F> {
    type Param = F;
    type Output = F;
    type Error = Infallible;

    fn cost(&self, x: &F) -> Result<F, Infallible> {
        assert!(*x >= self.lower && *x <= self.upper);
        Ok(*x)
    }
}

impl<F: Scalar> Gradient for Linear<F> {
    type Gradient = F;

    fn gradient(&self, x: &F) -> Result<F, Infallible> {
        assert!(*x >= self.lower && *x <= self.upper);
        Ok(F::one())
    }
}

fn brent_domain<F: Scalar + VectorLen>() {
    let mut problem = Problem::new(Linear {
        lower: F::zero(),
        upper: F::one(),
    });
    let mut solver =
        BrentDerivative::new().with_absolute_position_tolerance(F::one());
    let mut state = FirstOrderState::new(F::from_f64(0.5).unwrap());
    solver.init(&mut problem, &mut state).unwrap();
    // A composed policy can suppress the native width stop and request more steps.
    for _ in 0..5 {
        let step = solver.next_iter(&mut problem, &mut state).unwrap();
        assert!(*state.param() >= F::zero() && *state.param() <= F::one());
        if step.termination.is_some() {
            break;
        }
    }
}

struct FixedBox<F> {
    point: Vec<F>,
    cost: F,
}

impl<F: Scalar> basin::ProblemConstraints for FixedBox<F> {
    type Constraints = basin::Constraints<basin::BoxBounds<Vec<F>>>;
    fn constraint_set(
        &self,
    ) -> <Self::Constraints as basin::ConstraintDescription>::Borrowed<'_> {
        basin::Constraints::new().with_bounds(basin::BoxBounds::new(
            Some(&self.point),
            Some(&self.point),
        ))
    }
}
impl<F: Scalar> CostFunction for FixedBox<F> {
    type Param = Vec<F>;
    type Output = F;
    type Error = Infallible;

    fn cost(&self, x: &Vec<F>) -> Result<F, Infallible> {
        assert_eq!(x, &self.point);
        Ok(self.cost)
    }
}

#[test]
fn brent_trials_remain_inside_the_domain_when_width_stopping_is_suppressed() {
    brent_domain::<f64>();
    brent_domain::<f32>();
}

#[test]
fn gbnm_evaluates_an_all_fixed_box_once() {
    macro_rules! check {
        ($f:ty) => {{
            for (cost, expected) in [
                (5.0 as $f, TerminationCode::SolverConverged),
                (<$f>::NAN, TerminationCode::SolverFailed),
                (<$f>::INFINITY, TerminationCode::SolverFailed),
            ] {
                let result = Executor::new(
                    FixedBox {
                        point: vec![1.0 as $f, 2.0 as $f],
                        cost,
                    },
                    Gbnm::new(42),
                    PointState::new(vec![99.0 as $f, -99.0 as $f]),
                )
                .max_iter(10)
                .run()
                .unwrap();
                assert_eq!(result.report.code(), expected);
                assert_eq!(result.param(), &vec![1.0 as $f, 2.0 as $f]);
                assert_eq!(result.state.cost_evals(), 1);
                assert_eq!(result.state.iter(), 0);
            }
        }};
    }
    check!(f64);
    check!(f32);
}
