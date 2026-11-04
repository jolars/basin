//! Retained evaluations must belong to the returned step, including fallback paths.
use std::cell::RefCell;
use std::marker::PhantomData;

use basin::{
    CostFunction, Dot, Gradient, LineSearch, LineSearchBounds,
    LineSearchOutcome, Problem, Scalar, ScaledAdd, VectorIndex, VectorLen,
    Wolfe,
};

#[path = "support/backend_aliases.rs"]
mod backends;

struct Sphere<V, F = f64> {
    costs: RefCell<Vec<V>>,
    gradients: RefCell<Vec<V>>,
    fused: RefCell<Vec<V>>,
    fail_cost: bool,
    fail_gradient: bool,
    scalar: PhantomData<F>,
}
impl<V, F> Default for Sphere<V, F> {
    fn default() -> Self {
        Self {
            costs: RefCell::new(Vec::new()),
            gradients: RefCell::new(Vec::new()),
            fused: RefCell::new(Vec::new()),
            fail_cost: false,
            fail_gradient: false,
            scalar: PhantomData,
        }
    }
}
#[derive(Debug, PartialEq)]
enum Error {
    Cost,
    Gradient,
}
impl<V: Clone + Dot<F>, F: Scalar> CostFunction for Sphere<V, F> {
    type Param = V;
    type Output = F;
    type Error = Error;
    fn cost(&self, x: &V) -> Result<F, Error> {
        self.costs.borrow_mut().push(x.clone());
        if self.fail_cost {
            return Err(Error::Cost);
        }
        Ok(x.dot(x))
    }
}
impl<V: Clone + Dot<F> + ScaledAdd<F>, F: Scalar> Gradient for Sphere<V, F> {
    type Gradient = V;
    fn gradient(&self, x: &V) -> Result<V, Error> {
        self.gradients.borrow_mut().push(x.clone());
        if self.fail_gradient {
            return Err(Error::Gradient);
        }
        let mut g = x.clone();
        g.scaled_add(F::one(), x);
        Ok(g)
    }

    fn cost_and_gradient(&self, x: &V) -> Result<(F, V), Error> {
        self.fused.borrow_mut().push(x.clone());
        if self.fail_cost {
            return Err(Error::Cost);
        }
        if self.fail_gradient {
            return Err(Error::Gradient);
        }
        let mut g = x.clone();
        g.scaled_add(F::one(), x);
        Ok((x.dot(x), g))
    }
}

fn exercise<V, F>(make: fn(&[F]) -> V)
where
    V: Clone + Dot<F> + ScaledAdd<F> + VectorIndex<F> + VectorLen,
    F: Scalar,
{
    let one = F::one();
    let two = one + one;
    let half = one / two;
    let x = make(&[one, one]);
    let g = make(&[two, two]);
    let d = make(&[-two, -two]);
    // Direct acceptance, Armijo-triggered zoom, and slope-flip-triggered zoom.
    for (initial, fused) in [half, one, F::from_f64(0.75).unwrap()]
        .into_iter()
        .flat_map(|initial| [false, true].map(|fused| (initial, fused)))
    {
        let mut p = Problem::new(Sphere::<V, F>::default());
        let mut search = Wolfe::new().alpha_init(initial);
        if initial < one && initial > half {
            search.c2 = F::from_f64(0.1).unwrap();
        }
        let (c1, c2) = (search.c1, search.c2);
        let result = if fused {
            search
                .with_fused_evaluations()
                .next_with_evaluation(&mut p, &x, two, &g, &d)
        } else {
            search.next_with_evaluation(&mut p, &x, two, &g, &d)
        }
        .unwrap();
        let LineSearchOutcome::Step(alpha) = result.outcome else {
            panic!("expected a step")
        };
        let evaluation = result.evaluation.expect("accepted evaluation");
        let mut expected = x.clone();
        expected.scaled_add(alpha, &d);
        for i in 0..expected.vec_len() {
            assert_eq!(evaluation.param.get_scalar(i), expected.get_scalar(i));
            assert_eq!(
                evaluation.gradient.get_scalar(i),
                two * expected.get_scalar(i)
            );
        }
        assert_eq!(evaluation.cost, expected.dot(&expected));
        assert_eq!(
            p.counts().gradient_evals,
            (p.inner().gradients.borrow().len()
                + p.inner().fused.borrow().len()) as u64
        );
        assert_eq!(
            p.counts().cost_evals,
            (p.inner().costs.borrow().len() + p.inner().fused.borrow().len())
                as u64
        );
        assert!(evaluation.cost <= two + c1 * alpha * g.dot(&d));
        assert!(evaluation.gradient.dot(&d).abs() <= -c2 * g.dot(&d));
        // Direct accept and Armijo zoom each need just one gradient; the
        // rejected x + 1*d point must remain a cost-only probe.
        if initial == half || initial == one {
            assert_eq!(
                p.counts().gradient_evals,
                if fused && initial == one { 2 } else { 1 }
            );
            assert_eq!(
                p.counts().cost_evals,
                if initial == one { 2 } else { 1 }
            );
        }
        if fused {
            assert!(p.inner().costs.borrow().is_empty());
            assert!(p.inner().gradients.borrow().is_empty());
            assert_eq!(p.counts().cost_evals, p.counts().gradient_evals);
        }
    }
    // A cap can satisfy Armijo without strong curvature. Its retained values
    // must still describe the capped step, and every callback stays on the ray.
    let max = F::from_f64(0.01).unwrap();
    for fused in [false, true] {
        let mut p = Problem::new(Sphere::<V, F>::default());
        let mut search = Wolfe::new();
        let bounds = LineSearchBounds::new(max, max);
        let result = if fused {
            search
                .with_fused_evaluations()
                .next_with_bounds(&mut p, &x, two, &g, &d, bounds)
        } else {
            search.next_with_bounds(&mut p, &x, two, &g, &d, bounds)
        }
        .unwrap();
        assert!(
            matches!(result.outcome, LineSearchOutcome::Step(a) if a == max)
        );
        let e = result.evaluation.unwrap();
        assert_eq!(e.cost, e.param.dot(&e.param));
        for i in 0..x.vec_len() {
            assert_eq!(e.param.get_scalar(i), one + max * -two);
            assert_eq!(e.gradient.get_scalar(i), two * e.param.get_scalar(i));
        }
        assert_eq!(p.counts().cost_evals, 1);
        assert_eq!(p.counts().gradient_evals, 1);
    }
}

#[test]
fn vec_scalars() {
    exercise::<Vec<f64>, f64>(|s| s.to_vec());
    exercise::<Vec<f32>, f32>(|s| s.to_vec());
}
#[cfg(feature = "nalgebra_all")]
#[test]
fn nalgebra_scalars() {
    exercise::<backends::nalgebra::DVector<f64>, f64>(
        backends::nalgebra::DVector::from_column_slice,
    );
    exercise::<backends::nalgebra::DVector<f32>, f32>(
        backends::nalgebra::DVector::from_column_slice,
    );
}
#[cfg(feature = "ndarray_all")]
#[test]
fn ndarray_scalars() {
    exercise::<backends::ndarray::Array1<f64>, f64>(|s| {
        backends::ndarray::Array1::from_vec(s.to_vec())
    });
    exercise::<backends::ndarray::Array1<f32>, f32>(|s| {
        backends::ndarray::Array1::from_vec(s.to_vec())
    });
}
#[cfg(feature = "faer_all")]
#[test]
fn faer_scalars() {
    exercise::<backends::faer::Col<f64>, f64>(|s| {
        backends::faer::Col::from_fn(s.len(), |i| s[i])
    });
    exercise::<backends::faer::Col<f32>, f32>(|s| {
        backends::faer::Col::from_fn(s.len(), |i| s[i])
    });
}

#[test]
fn exhaustion_does_not_attach_another_trials_evaluation() {
    // Expansion exhausts after evaluating alpha=.01, but returns alpha=.02.
    // Zoom exhausts after rejecting alpha=2 and returns the starting point.
    // A tight curvature tolerance can also exhaust zoom at a positive low
    // endpoint, after reusing the earlier trial's gradient buffer.
    for (initial, expected) in [(0.01, 0.02), (4.0, 0.0), (0.75, 0.375)] {
        let mut p = Problem::new(Sphere::<Vec<f64>>::default());
        let result = Wolfe::new()
            .with_curvature_coefficient(0.01)
            .alpha_init(initial)
            .max_iter(1)
            .next_with_evaluation(
                &mut p,
                &vec![1.0],
                1.0,
                &vec![2.0],
                &vec![-2.0],
            )
            .unwrap();
        assert!(
            matches!(result.outcome, LineSearchOutcome::Step(a) if a == expected)
        );
        assert!(result.evaluation.is_none());
    }
}

#[test]
fn hard_errors_remain_typed_and_counted() {
    for (fail_cost, fused) in [false, true]
        .into_iter()
        .flat_map(|fail_cost| [false, true].map(|fused| (fail_cost, fused)))
    {
        let mut p = Problem::new(Sphere::<Vec<f64>> {
            fail_cost,
            fail_gradient: !fail_cost,
            ..Default::default()
        });
        let mut search = Wolfe::new().alpha_init(0.5);
        let (x, g, d) = (vec![1.0], vec![2.0], vec![-2.0]);
        let error = if fused {
            search
                .with_fused_evaluations()
                .next_with_evaluation(&mut p, &x, 1.0, &g, &d)
        } else {
            search.next_with_evaluation(&mut p, &x, 1.0, &g, &d)
        }
        .unwrap_err();
        assert_eq!(
            error,
            if fail_cost {
                Error::Cost
            } else {
                Error::Gradient
            }
        );
        assert_eq!(p.counts().cost_evals, 1);
        assert_eq!(p.counts().gradient_evals, u64::from(fused || !fail_cost));
    }
}

#[test]
fn non_descent_and_nan_directions_do_not_evaluate() {
    for direction in [0.0, 1.0, f64::NAN] {
        let mut p = Problem::new(Sphere::<Vec<f64>>::default());
        let result = Wolfe::new()
            .next_with_evaluation(
                &mut p,
                &vec![1.0],
                1.0,
                &vec![2.0],
                &vec![direction],
            )
            .unwrap();
        assert!(matches!(result.outcome, LineSearchOutcome::Step(0.0)));
        assert!(result.evaluation.is_none());
        assert_eq!(*p.counts(), Default::default());
    }
}

#[cfg(feature = "serde")]
#[test]
fn fused_wolfe_round_trips_without_changing_existing_wolfe_binary_layout() {
    let old_fields = (1e-4_f64, 0.9_f64, 1.0_f64, 10.0_f64, 25_u32);
    assert_eq!(
        postcard::to_allocvec(&Wolfe::<f64>::new()).unwrap(),
        postcard::to_allocvec(&old_fields).unwrap()
    );
    let search = Wolfe::<f64>::new().with_fused_evaluations();
    let bytes = postcard::to_allocvec(&search).unwrap();
    let restored: basin::FusedWolfe = postcard::from_bytes(&bytes).unwrap();
    let mut restored: basin::FusedWolfe =
        serde_json::from_str(&serde_json::to_string(&restored).unwrap())
            .unwrap();
    let mut p = Problem::new(Sphere::<Vec<f64>>::default());
    let result = restored
        .next_with_evaluation(&mut p, &vec![1.0], 1.0, &vec![2.0], &vec![-2.0])
        .unwrap();
    assert!(result.evaluation.is_some());
    assert_eq!(p.inner().fused.borrow().len(), 2);
    assert!(p.inner().costs.borrow().is_empty());
    assert!(p.inner().gradients.borrow().is_empty());
}
