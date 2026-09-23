use basin::{Bfgs, Executor, IntoInitialSimplex, NormSquared};

pub fn check() {
    let nalgebra = nalgebra::DVector::from_vec(vec![3.0_f64, 4.0]);
    let ndarray = ndarray::array![3.0_f64, 4.0];
    let faer = faer::Col::from_fn(2, |i| [3.0_f64, 4.0][i]);
    assert_eq!(NormSquared::norm_squared(&nalgebra), 25.0);
    assert_eq!(NormSquared::norm_squared(&ndarray), 25.0);
    assert_eq!(NormSquared::norm_squared(&faer), 25.0);
    assert_eq!(ndarray.into_initial_simplex(0.05).len(), 3);
    let problem = basin::problems::Sphere::<nalgebra::DVector<f64>>::new();
    let result = Executor::from_start(problem, Bfgs::new(), nalgebra)
        .max_iter(20)
        .run()
        .unwrap();
    assert!(result.cost() < 1e-10);
}

#[test]
fn selected_versions_work() {
    check();
}
