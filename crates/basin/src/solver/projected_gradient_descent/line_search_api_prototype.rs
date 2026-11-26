use super::*;
use crate::line_search::projected_armijo::ProjectedArmijo;
use crate::solver::lbfgs::AsFloatSliceMut;
use crate::{
    Backtracking, BoxBounds, ConstraintDescription, ConstraintSet, Constraints,
    Executor, GradientState, TerminationCode,
};
use std::{cell::RefCell, rc::Rc};

struct ProjectedDescent<F: Scalar>(ProjectedArmijo<F>);

impl<P, V, F> Solver<P, FirstOrderState<V, F>> for ProjectedDescent<F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>
        + Gradient<Gradient = V>
        + BoxProblem<Param = V>,
    V: ScaledAdd<F>
        + NegInPlace
        + ClampInPlace
        + Clone
        + VectorLen
        + std::ops::Index<usize, Output = F>
        + AsFloatSliceMut<F>,
{
    type Error = crate::ConstraintSolveError<P::Error>;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        state: FirstOrderState<V, F>,
    ) -> Result<FirstOrderState<V, F>, Self::Error> {
        ProjectedGradientDescent::new(F::one()).init(problem, state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        state: &mut FirstOrderState<V, F>,
    ) -> Result<crate::SolverStep<F>, Self::Error> {
        let (x, cost, gradient) =
            state.current().expect("initialized progress");
        let mut direction = gradient.clone();
        direction.neg_in_place();
        // Own the bound values before passing the wrapper mutably to the search.
        let lower = problem.inner().lower().as_float_slice().to_vec();
        let upper = problem.inner().upper().as_float_slice().to_vec();
        let Some((_, accepted)) = self.0.search(
            problem, x, cost, gradient, &direction, &lower, &upper, &mut None,
        )?
        else {
            return Ok(crate::SolverStep::stopped(
                Termination::numerical_failure(
                    "Projected Armijo search exhausted.",
                ),
            ));
        };
        state
            .replace(accepted.param, accepted.cost, accepted.gradient)
            .expect("matching gradient dimension");
        Ok(crate::SolverStep::completed())
    }
}

#[derive(Debug, PartialEq)]
enum TrialError {
    OutsideBox,
    Gradient,
}

struct Probes<F> {
    costs: RefCell<Vec<[F; 2]>>,
    gradients: RefCell<Vec<[F; 2]>>,
}

struct StrictBox<V, F> {
    lower: V,
    upper: V,
    make: fn(Vec<F>) -> V,
    probes: Rc<Probes<F>>,
    fail_gradient: bool,
}

impl<V, F> Constraints for StrictBox<V, F> {
    type Set = ConstraintSet<BoxBounds<V>>;
    fn constraints(
        &self,
    ) -> <Self::Set as ConstraintDescription>::Borrowed<'_> {
        ConstraintSet::new()
            .with_bounds(BoxBounds::new(Some(&self.lower), Some(&self.upper)))
    }
}

impl<V: AsFloatSliceMut<F>, F: Scalar> StrictBox<V, F> {
    fn check(&self, param: &V) -> Result<[F; 2], TrialError> {
        let &[x, y] = param.as_float_slice() else {
            panic!("expected two coordinates")
        };
        if [x, y].into_iter().all(|v| v >= F::zero() && v <= F::one()) {
            Ok([x, y])
        } else {
            Err(TrialError::OutsideBox)
        }
    }
}

fn scalar<F: Scalar>(value: f64) -> F {
    F::from_f64(value).unwrap()
}

impl<V: AsFloatSliceMut<F>, F: Scalar> CostFunction for StrictBox<V, F> {
    type Param = V;
    type Output = F;
    type Error = TrialError;

    fn cost(&self, param: &V) -> Result<F, TrialError> {
        let [x, y] = self.check(param)?;
        self.probes.costs.borrow_mut().push([x, y]);
        let dy = y - scalar(0.5);
        Ok(scalar::<F>(-8.0) * x + scalar::<F>(32.0) * dy * dy)
    }
}

impl<V: AsFloatSliceMut<F>, F: Scalar> Gradient for StrictBox<V, F> {
    type Gradient = V;

    fn gradient(&self, param: &V) -> Result<V, TrialError> {
        let [x, y] = self.check(param)?;
        self.probes.gradients.borrow_mut().push([x, y]);
        if self.fail_gradient && [x, y] == point([1.0, 0.6875]) {
            return Err(TrialError::Gradient);
        }
        Ok((self.make)(vec![
            scalar(-8.0),
            scalar::<F>(64.0) * (y - scalar(0.5)),
        ]))
    }
}

fn point<F: Scalar>(values: [f64; 2]) -> [F; 2] {
    values.map(scalar)
}

fn fixture<V: Clone, F: Scalar>(
    make: fn(Vec<F>) -> V,
) -> (StrictBox<V, F>, FirstOrderState<V, F>) {
    let problem = StrictBox {
        lower: make(vec![F::zero(); 2]),
        upper: make(vec![F::one(); 2]),
        make,
        probes: Rc::new(Probes {
            costs: RefCell::default(),
            gradients: RefCell::default(),
        }),
        fail_gradient: false,
    };
    (
        problem,
        FirstOrderState::new(make(point([0.75, 0.4375]).to_vec())),
    )
}

fn search<F: Scalar>() -> ProjectedArmijo<F> {
    ProjectedArmijo::new().with_sufficient_decrease_coefficient(scalar(0.25))
}

fn check_projected<V, F>(make: fn(Vec<F>) -> V)
where
    F: Scalar,
    V: ScaledAdd<F>
        + NegInPlace
        + ClampInPlace
        + Clone
        + VectorLen
        + std::ops::Index<usize, Output = F>
        + AsFloatSliceMut<F>,
{
    for fused in [false, true] {
        let (problem, state) = fixture(make);
        let probes = problem.probes.clone();
        let search = if fused {
            search().with_fused_evaluations()
        } else {
            search()
        };
        let result = Executor::new(problem, ProjectedDescent(search), state)
            .max_iter(1)
            .run()
            .unwrap();
        let seed = point([0.75, 0.4375]);
        let accepted = point([1.0, 0.6875]);
        let trials = [seed, point([1.0, 1.0]), point([1.0, 0.9375]), accepted];
        assert_eq!(*probes.costs.borrow(), trials);
        if fused {
            assert_eq!(*probes.gradients.borrow(), trials);
        } else {
            assert_eq!(*probes.gradients.borrow(), [seed, accepted]);
        }
        assert_eq!(result.iter(), 1);
        assert_eq!(result.cost_evals(), 4);
        assert_eq!(result.state.gradient_evals(), if fused { 4 } else { 2 });
        assert_eq!(result.param().as_float_slice(), &accepted);
        assert_eq!(
            result.selected_record().unwrap().param.as_float_slice(),
            &accepted
        );
        assert_eq!(
            result.state.gradient().unwrap().as_float_slice(),
            &point([-8.0, 12.0])
        );
        assert_eq!(
            result.current_record().unwrap().cost.unwrap(),
            scalar(-6.875)
        );
        let initial_cost = scalar::<F>(-5.875);
        let coefficient = scalar::<F>(0.25);
        // The projection clips the first displacement from 0.5 to 0.25.
        assert!(
            result.current_record().unwrap().cost.unwrap()
                <= initial_cost + coefficient * scalar(-3.0)
        );
        assert!(
            result.current_record().unwrap().cost.unwrap()
                > initial_cost + coefficient * scalar(-5.0)
        );
    }
}

macro_rules! backend {
    ($name:ident, $float:ty, $make:expr) => {
        #[test]
        fn $name() {
            check_projected::<_, $float>($make);
        }
    };
}

backend!(vec_f32, f32, |v| v);
backend!(vec_f64, f64, |v| v);

#[cfg(feature = "nalgebra")]
backend!(nalgebra_f32, f32, nalgebra::DVector::from_vec);
#[cfg(feature = "nalgebra")]
backend!(nalgebra_f64, f64, nalgebra::DVector::from_vec);
#[cfg(feature = "ndarray")]
backend!(ndarray_f32, f32, ndarray::Array1::from_vec);
#[cfg(feature = "ndarray")]
backend!(ndarray_f64, f64, ndarray::Array1::from_vec);
#[cfg(feature = "faer")]
backend!(faer_f32, f32, |v| faer::Col::from_fn(v.len(), |i| v[i]));
#[cfg(feature = "faer")]
backend!(faer_f64, f64, |v| faer::Col::from_fn(v.len(), |i| v[i]));

#[test]
fn exhausted_search_publishes_attempts_without_completing_a_step() {
    let (problem, state) = fixture::<_, f64>(|v| v);
    let probes = problem.probes.clone();
    let result =
        Executor::new(problem, ProjectedDescent(search().max_iter(1)), state)
            .max_iter(1)
            .run()
            .unwrap();
    assert_eq!(result.report.code(), TerminationCode::SolverFailed);
    assert_eq!(result.iter(), 0);
    assert_eq!(
        result.state.current(),
        Some((&vec![0.75, 0.4375], -5.875, &vec![-8.0, -4.0]))
    );
    assert_eq!(result.selected_record().unwrap().param, &[0.75, 0.4375]);
    assert_eq!(
        result.state.selected_record().unwrap().cost.unwrap(),
        -5.875
    );
    assert_eq!(*probes.costs.borrow(), [[0.75, 0.4375], [1.0, 1.0]]);
    assert_eq!(result.cost_evals(), 2);
    assert_eq!(result.state.gradient_evals(), 1);
}

#[test]
fn feasible_trial_propagates_typed_gradient_error() {
    let (mut problem, state) = fixture::<_, f64>(|v| v);
    problem.fail_gradient = true;
    let probes = problem.probes.clone();
    let result = Executor::new(problem, ProjectedDescent(search()), state)
        .max_iter(1)
        .run();
    assert!(matches!(
        result,
        Err(crate::ConstraintSolveError::Evaluation(
            TrialError::Gradient
        ))
    ));
    assert_eq!(*probes.gradients.borrow(), [[0.75, 0.4375], [1.0, 0.6875]]);
}

#[test]
fn straight_ray_backtracking_probes_outside_box_before_projection() {
    let (problem, state) = fixture::<_, f64>(|v| v);
    let result = Executor::new(
        problem,
        ProjectedGradientDescent::with_line_search(Backtracking::new()),
        state,
    )
    .max_iter(1)
    .run();
    assert!(matches!(
        result,
        Err(crate::ConstraintSolveError::Evaluation(
            TrialError::OutsideBox
        ))
    ));
}
