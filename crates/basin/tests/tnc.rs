use basin::{
    BoxBounds, ConstraintDescription, ConstraintSet, Constraints, CostFunction,
    Dot, Executor, FirstOrderState, Gradient, HessianProduct, Scalar,
    ScaledAdd, Tnc, VectorIndex, VectorLen,
};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

#[path = "support/backend_aliases.rs"]
mod backends;

struct Quadratic<V, F> {
    lo: V,
    hi: V,
    fail_hvp: bool,
    calls: AtomicUsize,
    scalar: PhantomData<F>,
}
impl<V, F: Scalar> Constraints for Quadratic<V, F> {
    type Set = ConstraintSet<BoxBounds<V>>;
    fn constraints(
        &self,
    ) -> <Self::Set as ConstraintDescription>::Borrowed<'_> {
        ConstraintSet::new()
            .with_bounds(BoxBounds::new(Some(&self.lo), Some(&self.hi)))
    }
}
impl<V: VectorIndex<F> + VectorLen, F: Scalar> CostFunction
    for Quadratic<V, F>
{
    type Param = V;
    type Output = F;
    type Error = &'static str;
    fn cost(&self, x: &V) -> Result<F, Self::Error> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        for i in 0..x.vec_len() {
            if x.get_scalar(i) < self.lo.get_scalar(i)
                || x.get_scalar(i) > self.hi.get_scalar(i)
            {
                return Err("outside bounds");
            }
        }
        Ok((0..x.vec_len())
            .map(|i| {
                let target =
                    F::from_f64(if i == 0 { 1.2 } else { 0.2 }).unwrap();
                let d = x.get_scalar(i) - target;
                F::from_usize(i + 1).unwrap() * d * d
            })
            .sum())
    }
}
impl<V: Clone + VectorIndex<F> + VectorLen, F: Scalar> Gradient
    for Quadratic<V, F>
{
    type Gradient = V;
    fn gradient(&self, x: &V) -> Result<V, Self::Error> {
        let mut g = x.clone();
        for i in 0..x.vec_len() {
            let t = F::from_f64(if i == 0 { 1.2 } else { 0.2 }).unwrap();
            g.set_scalar(
                i,
                F::from_usize(2 * (i + 1)).unwrap() * (x.get_scalar(i) - t),
            );
        }
        Ok(g)
    }
}
impl<V: Clone + VectorIndex<F> + VectorLen, F: Scalar> HessianProduct
    for Quadratic<V, F>
{
    fn hessian_product(&self, _: &V, v: &V) -> Result<V, Self::Error> {
        if self.fail_hvp {
            return Err("exact product failed");
        }
        let mut out = v.clone();
        for i in 0..v.vec_len() {
            out.set_scalar(
                i,
                F::from_usize(2 * (i + 1)).unwrap() * v.get_scalar(i),
            );
        }
        Ok(out)
    }
}
fn problem<V, F: Scalar>(make: fn(&[F]) -> V) -> Quadratic<V, F> {
    Quadratic {
        lo: make(&[F::zero(); 3]),
        hi: make(&[F::one(); 3]),
        fail_hvp: false,
        calls: AtomicUsize::new(0),
        scalar: PhantomData,
    }
}
fn exercise<V, F>(make: fn(&[F]) -> V)
where
    F: Scalar,
    V: Clone + VectorLen + VectorIndex<F> + Dot<F> + ScaledAdd<F>,
{
    let tolerance = F::epsilon().sqrt() * F::from_f64(10.0).unwrap();
    for exact in [false, true] {
        for memory in [0, 10] {
            let p = problem(make);
            let x = make(&[F::zero(), F::one(), F::one()]);
            let solver = Tnc::new()
                .with_memory(memory)
                .with_relative_cg_residual_tolerance(F::from_f64(0.5).unwrap())
                .with_absolute_projected_gradient_tolerance(tolerance);
            let result = if exact {
                Executor::new(
                    p,
                    solver.exact_products(),
                    FirstOrderState::new(x),
                )
                .max_iter(500)
                .run()
                .unwrap()
            } else {
                Executor::new(p, solver, FirstOrderState::new(x))
                    .max_iter(500)
                    .run()
                    .unwrap()
            };
            assert!(
                (result.param().get_scalar(0) - F::one()).abs() <= tolerance
            );
            for i in 1..3 {
                assert!(
                    (result.param().get_scalar(i) - F::from_f64(0.2).unwrap())
                        .abs()
                        <= tolerance
                );
            }
            assert!(
                matches!(
                    result.report.termination,
                    basin::Termination::Converged(_)
                ),
                "{:?}",
                result.report
            );
        }
    }
}
#[test]
fn tnc_vec_f64() {
    exercise::<Vec<f64>, f64>(|s| s.to_vec());
}
#[test]
fn tnc_vec_f32() {
    exercise::<Vec<f32>, f32>(|s| s.to_vec());
}
#[cfg(feature = "nalgebra_all")]
#[test]
fn tnc_nalgebra() {
    exercise::<backends::nalgebra::DVector<f64>, f64>(|s| {
        backends::nalgebra::DVector::from_column_slice(s)
    });
    exercise::<backends::nalgebra::DVector<f32>, f32>(|s| {
        backends::nalgebra::DVector::from_column_slice(s)
    });
}
#[cfg(feature = "ndarray_all")]
#[test]
fn tnc_ndarray() {
    exercise::<backends::ndarray::Array1<f64>, f64>(|s| {
        backends::ndarray::Array1::from_vec(s.to_vec())
    });
    exercise::<backends::ndarray::Array1<f32>, f32>(|s| {
        backends::ndarray::Array1::from_vec(s.to_vec())
    });
}
#[cfg(feature = "faer_all")]
#[test]
fn tnc_faer() {
    exercise::<backends::faer::Col<f64>, f64>(|s| {
        backends::faer::Col::from_fn(s.len(), |i| s[i])
    });
    exercise::<backends::faer::Col<f32>, f32>(|s| {
        backends::faer::Col::from_fn(s.len(), |i| s[i])
    });
}
#[test]
fn exact_product_error_remains_typed() {
    let mut p = problem::<Vec<f64>, f64>(|s| s.to_vec());
    p.fail_hvp = true;
    let error =
        Executor::from_start(p, Tnc::new().exact_products(), vec![0.5; 3])
            .run()
            .err()
            .unwrap();
    assert_eq!(
        error,
        basin::ConstraintSolveError::Evaluation("exact product failed")
    );
}
#[test]
fn malformed_bounds_are_rejected_without_callbacks() {
    let mut p = problem::<Vec<f64>, f64>(|s| s.to_vec());
    p.hi[0] = -1.0;
    let mut wrapped = basin::Problem::new(p);
    let mut solver = Tnc::new();
    assert!(
        basin::Solver::init(
            &mut solver,
            &mut wrapped,
            FirstOrderState::new(vec![0.5; 3])
        )
        .is_err()
    );
    assert_eq!(wrapped.inner().calls.load(Ordering::Relaxed), 0);
}
#[test]
fn stationary_fixed_coordinates_need_no_curvature() {
    let mut p = problem::<Vec<f64>, f64>(|s| s.to_vec());
    p.lo = vec![1.0, 0.2, 0.2];
    p.hi = p.lo.clone();
    p.fail_hvp = true;
    let r = Executor::from_start(
        p,
        Tnc::new().exact_products(),
        vec![1.0, 0.2, 0.2],
    )
    .run()
    .unwrap();
    assert!(matches!(
        r.report.termination,
        basin::Termination::Converged(_)
    ));
}
#[test]
fn failed_search_keeps_current_progress() {
    let p = problem::<Vec<f64>, f64>(|s| s.to_vec());
    let r = Executor::from_start(
        p,
        Tnc::with_line_search(basin::Constant(-1.0)),
        vec![0.5; 3],
    )
    .run()
    .unwrap();
    assert_eq!(r.param(), &vec![0.5; 3]);
    assert!(matches!(
        r.report.termination,
        basin::Termination::Failed(_)
    ));
}

#[test]
fn exact_continuation_preserves_curvature_and_accounting() {
    let p = || problem::<Vec<f64>, f64>(|s| s.to_vec());
    let first = Executor::from_start(p(), Tnc::new(), vec![0.5; 3])
        .max_iter(1)
        .run_with_solver()
        .unwrap();
    let checkpoint = first.into_checkpoint();
    #[cfg(feature = "serde")]
    let checkpoint: basin::ExactCheckpoint<
        Tnc<basin::Wolfe, Vec<f64>>,
        FirstOrderState<Vec<f64>>,
    > = serde_json::from_str(&serde_json::to_string(&checkpoint).unwrap())
        .unwrap();
    let resumed = Executor::resume_from_checkpoint(p(), checkpoint)
        .max_iter(500)
        .run()
        .unwrap();
    let whole = Executor::from_start(p(), Tnc::new(), vec![0.5; 3])
        .max_iter(500)
        .run()
        .unwrap();
    assert_eq!(resumed.param(), whole.param());
    assert_eq!(resumed.cost(), whole.cost());
    assert_eq!(resumed.counts, whole.counts);
}

#[test]
fn cancellation_and_budgets_keep_basins_boundary_semantics() {
    let token = basin::CancellationToken::new();
    token.cancel();
    let cancelled = Executor::from_start(
        problem::<Vec<f64>, f64>(|s| s.to_vec()),
        Tnc::new(),
        vec![0.5; 3],
    )
    .with_cancellation_token(token)
    .run()
    .unwrap();
    assert!(matches!(
        cancelled.report.termination,
        basin::Termination::Cancelled
    ));
    assert_eq!(cancelled.counts.cost_evals, 1);
    let limited = Executor::from_start(
        problem::<Vec<f64>, f64>(|s| s.to_vec()),
        Tnc::new(),
        vec![0.5; 3],
    )
    .max_cost_evals(1)
    .run()
    .unwrap();
    assert!(matches!(
        limited.report.termination,
        basin::Termination::Limit(_)
    ));
    assert_eq!(limited.counts.cost_evals, 1);
}

#[test]
fn nonpositive_curvature_falls_back_to_feasible_descent() {
    struct Concave {
        lo: Vec<f64>,
        hi: Vec<f64>,
    }
    impl Constraints for Concave {
        type Set = ConstraintSet<BoxBounds<Vec<f64>>>;
        fn constraints(
            &self,
        ) -> <Self::Set as ConstraintDescription>::Borrowed<'_> {
            ConstraintSet::new()
                .with_bounds(BoxBounds::new(Some(&self.lo), Some(&self.hi)))
        }
    }
    impl CostFunction for Concave {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = &'static str;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
            if !(0.0..=1.0).contains(&x[0]) {
                return Err("outside bounds");
            }
            Ok(-x[0] * x[0])
        }
    }
    impl Gradient for Concave {
        type Gradient = Vec<f64>;
        fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
            Ok(vec![-2.0 * x[0]])
        }
    }
    impl HessianProduct for Concave {
        fn hessian_product(
            &self,
            _: &Vec<f64>,
            v: &Vec<f64>,
        ) -> Result<Vec<f64>, Self::Error> {
            Ok(vec![-2.0 * v[0]])
        }
    }
    let r = Executor::from_start(
        Concave {
            lo: vec![0.0],
            hi: vec![1.0],
        },
        Tnc::new().exact_products(),
        vec![0.2],
    )
    .run()
    .unwrap();
    assert_eq!(r.param(), &vec![1.0]);
    assert!(matches!(
        r.report.termination,
        basin::Termination::Converged(_)
    ));
}

#[cfg(feature = "problems")]
#[test]
fn nonlinear_rosenbrock_converges_in_physical_coordinates() {
    let p = basin::Constrained::new(
        basin::problems::Rosenbrock::<Vec<f64>>::new(),
        ConstraintSet::new().with_bounds(BoxBounds::new(
            Some(vec![-2.0; 2]),
            Some(vec![2.0; 2]),
        )),
    );
    let r = Executor::from_start(p, Tnc::new(), vec![-1.2, 1.0])
        .max_iter(1000)
        .run()
        .unwrap();
    assert!(r.cost() < 1e-10, "{:?}", r.report);
}

#[test]
fn nonfinite_initial_data_never_reports_convergence() {
    struct Data {
        cost: f64,
        gradient: f64,
    }
    impl CostFunction for Data {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, _: &Vec<f64>) -> Result<f64, Self::Error> {
            Ok(self.cost)
        }
    }
    impl Gradient for Data {
        type Gradient = Vec<f64>;
        fn gradient(&self, _: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
            Ok(vec![self.gradient])
        }
    }
    for (start, cost, gradient) in [
        (0.5, f64::NAN, 0.0),
        (0.5, f64::INFINITY, 0.0),
        (0.5, 0.0, f64::NAN),
        (0.5, 0.0, f64::INFINITY),
        (f64::NAN, 0.0, 0.0),
        (f64::INFINITY, 0.0, 0.0),
    ] {
        let problem = basin::Constrained::new(
            Data { cost, gradient },
            ConstraintSet::new()
                .with_bounds(BoxBounds::new(Some(vec![0.0]), Some(vec![1.0]))),
        );
        let result = Executor::from_start(problem, Tnc::new(), vec![start])
            .run()
            .unwrap();
        assert!(matches!(
            result.report.termination,
            basin::Termination::Failed(_)
        ));
        assert_eq!(result.iter(), 0);
        assert_eq!(result.counts.cost_evals, 1);
        assert_eq!(result.counts.gradient_evals, 1);
    }
}

#[test]
fn invalid_configuration_panics_before_a_solve() {
    for tolerance in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(
            std::panic::catch_unwind(|| {
                Tnc::<_, Vec<f64>>::new()
                    .with_absolute_projected_gradient_tolerance(tolerance)
            })
            .is_err()
        );
    }
    for tolerance in [0.0, 1.0, -0.5, f64::NAN, f64::INFINITY] {
        assert!(
            std::panic::catch_unwind(|| {
                Tnc::<_, Vec<f64>>::new()
                    .with_relative_cg_residual_tolerance(tolerance)
            })
            .is_err()
        );
    }
    for step in [0.0, f64::EPSILON, 0.2, f64::NAN, f64::INFINITY] {
        assert!(
            std::panic::catch_unwind(|| {
                Tnc::<_, Vec<f64>>::new().with_difference_step(step)
            })
            .is_err()
        );
    }
    assert!(
        std::panic::catch_unwind(|| {
            Tnc::<_, Vec<f64>>::new().with_max_cg_iterations(0)
        })
        .is_err()
    );
}

#[test]
fn disabled_stationarity_does_not_stop_at_iteration_zero() {
    let result = Executor::from_start(
        problem::<Vec<f64>, f64>(|s| s.to_vec()),
        Tnc::new().with_absolute_projected_gradient_tolerance(None),
        vec![1.0, 0.2, 0.2],
    )
    .max_iter(0)
    .run()
    .unwrap();
    assert!(matches!(
        result.report.termination,
        basin::Termination::Limit(_)
    ));
}
