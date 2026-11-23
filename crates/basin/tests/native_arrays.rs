use basin::{
    BoxBounds, Constrained, ConstraintSet, CostFunction, Executor, FiniteDiff,
    Gradient, GradientDescent, Lbfgs, NelderMead, PopulationProgress,
    ProjectedGradientDescent, RandomSearch, Scalar,
};
use std::convert::Infallible;

struct Quadratic<F>([F; 2]);

impl<F: Scalar> CostFunction for Quadratic<F> {
    type Param = [F; 2];
    type Output = F;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<F, Infallible> {
        Ok((x[0] - self.0[0]).powi(2) + (x[1] - self.0[1]).powi(2))
    }
}

impl<F: Scalar> Gradient for Quadratic<F> {
    type Gradient = [F; 2];

    fn gradient(&self, x: &Self::Param) -> Result<Self::Gradient, Infallible> {
        let two = F::one() + F::one();
        Ok(std::array::from_fn(|i| two * (x[i] - self.0[i])))
    }
}

fn check<F: Scalar + Send + Sync + rand_distr::uniform::SampleUniform>() {
    let n = |v| F::from_f64(v).unwrap();
    let start = [n(-0.5), n(-0.5)];
    let target = [n(0.25), n(0.5)];
    let bounded = || {
        Constrained::new(
            Quadratic([n(2.0), target[1]]),
            ConstraintSet::new().with_bounds(BoxBounds::new(
                Some([n(-1.0); 2]),
                Some([n(1.0); 2]),
            )),
        )
    };
    let check_point = |point: &[F; 2], expected: [F; 2], tolerance: F| {
        for i in 0..2 {
            assert!((point[i] - expected[i]).abs() < tolerance, "{point:?}");
        }
    };
    let check_box_point = |point: &[F; 2], expected, tolerance| {
        check_point(point, expected, tolerance);
        assert!(point.iter().all(|&x| x >= n(-1.0) && x <= n(1.0)));
    };
    let tolerance = n(0.002);
    let result = Executor::from_start(
        FiniteDiff::new_with_scalar(Quadratic(target)),
        GradientDescent::new(n(0.25)),
        start,
    )
    .max_iter(25)
    .run()
    .unwrap();
    check_point(result.param(), target, tolerance);

    let result =
        Executor::from_start(Quadratic(target), NelderMead::new(), start)
            .max_iter(200)
            .run()
            .unwrap();
    check_point(result.param(), target, tolerance);

    let result =
        Executor::from_start(bounded(), NelderMead::new().projected(), start)
            .max_iter(200)
            .run()
            .unwrap();
    let box_target = [n(1.0), target[1]];
    check_box_point(result.param(), box_target, tolerance);

    let result = Executor::from_start(
        bounded(),
        ProjectedGradientDescent::new(n(0.25)),
        start,
    )
    .max_iter(25)
    .run()
    .unwrap();
    check_box_point(result.param(), box_target, tolerance);
    assert_eq!(result.param()[0], box_target[0]);

    let result = Executor::from_start(
        Quadratic(target),
        Lbfgs::new().unbounded(),
        start,
    )
    .max_iter(40)
    .run()
    .unwrap();
    check_point(result.param(), target, tolerance);

    let result = Executor::from_start(bounded(), Lbfgs::new(), start)
        .max_iter(40)
        .run()
        .unwrap();
    check_box_point(result.param(), box_target, tolerance);
    assert_eq!(result.param()[0], box_target[0]);

    let result = Executor::new(
        bounded(),
        RandomSearch::new(64, 42),
        PopulationProgress::empty(),
    )
    .max_iter(150)
    .run()
    .unwrap();
    check_box_point(result.param(), box_target, n(0.1));
}

#[test]
fn f32_solvers() {
    check::<f32>();
}

#[test]
fn f64_solvers() {
    check::<f64>();
}
