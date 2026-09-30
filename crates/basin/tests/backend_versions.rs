//! Backend implementations must survive Cargo feature unification.

#[cfg(feature = "ndarray_v0_16")]
#[test]
fn ndarray_0_16_norm_survives_newer_features() {
    let x = ::ndarray_0_16::array![3.0_f64, 4.0];
    assert_eq!(basin::NormSquared::norm_squared(&x), 25.0);
}

#[cfg(feature = "ndarray_v0_17")]
#[test]
fn ndarray_0_17_norm() {
    let x = ndarray::array![3.0_f64, 4.0];
    assert_eq!(basin::NormSquared::norm_squared(&x), 25.0);
}

macro_rules! dense_checks {
    ($vector:ty, $matrix:ty, $scalar:ty, $make:expr) => {{
        use basin::{
            Bfgs, BoxConstraints, CostFunction, Executor, FactorizePivotedQr,
            FirstOrderState, Gradient, Lbfgs, LinearSolveSpd, MatVec,
            MatrixIdentity, NormSquared, RegularizedQrSolve, ScaleInPlace,
            State,
        };
        use std::convert::Infallible;
        type V = $vector;
        type M = $matrix;
        type F = $scalar;
        let make = $make;
        let x = make(&[3.0, 4.0]);
        assert_eq!(NormSquared::norm_squared(&x), 25.0);
        let identity = <M as MatrixIdentity>::identity(2);
        let product = identity.matvec(&x);
        let solved = identity.solve_spd(&x).unwrap();
        let qr = identity.factorize_pivoted_qr(&x).unwrap();
        let qr_solution =
            qr.solve_regularized(0.0, &make(&[1.0, 1.0]), None).unwrap();
        for actual in [product, solved, qr_solution] {
            for i in 0..2 {
                assert!((actual[i] - x[i]).abs() < 32.0 * F::EPSILON);
            }
        }
        struct Quadratic {
            lower: V,
            upper: V,
            a: M,
            b: V,
        }
        impl CostFunction for Quadratic {
            type Param = V;
            type Output = F;
            type Error = Infallible;
            fn cost(&self, x: &V) -> Result<F, Infallible> {
                Ok(NormSquared::norm_squared(x))
            }
        }
        impl Gradient for Quadratic {
            type Gradient = V;
            fn gradient(&self, x: &V) -> Result<V, Infallible> {
                let mut gradient = x.clone();
                gradient.scale_in_place(2.0);
                Ok(gradient)
            }
        }
        impl basin::MiniBatchGradient for Quadratic {
            type Gradient = V;
            fn n_samples(&self) -> usize {
                7
            }
            fn batch_gradient(
                &self,
                x: &V,
                _: &[usize],
            ) -> Result<V, Infallible> {
                self.gradient(x)
            }
        }
        impl basin::Hessian for Quadratic {
            type Hessian = M;
            fn hessian(&self, x: &V) -> Result<M, Infallible> {
                let mut h = <M as MatrixIdentity>::identity(
                    basin::VectorLen::vec_len(x),
                );
                h.scale_in_place(2.0);
                Ok(h)
            }
        }
        impl basin::HessianProduct for Quadratic {
            fn hessian_product(&self, _: &V, v: &V) -> Result<V, Infallible> {
                self.gradient(v)
            }
        }
        impl BoxConstraints for Quadratic {
            fn lower(&self) -> &V {
                &self.lower
            }
            fn upper(&self) -> &V {
                &self.upper
            }
        }
        impl basin::LinearEqualityConstraints for Quadratic {
            type Matrix = M;
            fn a(&self) -> &M {
                &self.a
            }
            fn b(&self) -> &V {
                &self.b
            }
        }
        impl basin::LinearInequalityConstraints for Quadratic {
            type Matrix = M;
            fn a(&self) -> &M {
                &self.a
            }
            fn b(&self) -> &V {
                &self.b
            }
        }
        impl basin::LinearConstraints for Quadratic {
            type Matrix = M;
            fn inequalities(&self) -> Option<(&M, &V)> {
                Some((&self.a, &self.b))
            }
        }
        impl basin::NonlinearInequalityConstraints for Quadratic {
            fn num_constraints(&self) -> usize {
                2
            }
            fn constraints(&self, x: &V) -> Result<V, Infallible> {
                let mut constraints = x.clone();
                constraints[0] = 1.0 - x[0];
                constraints[1] = -1.0;
                Ok(constraints)
            }
        }
        impl basin::NonlinearConstraints for Quadratic {
            type Matrix = M;
            fn num_nonlinear_constraints(&self) -> usize {
                2
            }
            fn nonlinear_constraints(&self, x: &V) -> Result<V, Infallible> {
                basin::NonlinearInequalityConstraints::constraints(self, x)
            }
            fn lower(&self) -> Option<&V> {
                Some(&self.lower)
            }
            fn upper(&self) -> Option<&V> {
                Some(&self.upper)
            }
        }
        impl basin::ConstraintJacobian for Quadratic {
            fn constraint_jacobian(&self, x: &V) -> Result<M, Infallible> {
                let mut diagonal = x.clone();
                diagonal[0] = -1.0;
                diagonal[1] = 0.0;
                Ok(<M as basin::MatrixFromDiagonal<V>>::from_diagonal(
                    &diagonal,
                ))
            }
        }
        let problem = || Quadratic {
            lower: make(&[-5.0, -5.0]),
            upper: make(&[5.0, 5.0]),
            a: <M as MatrixIdentity>::identity(2),
            b: make(&[1.0, 1.0]),
        };
        macro_rules! powell_run {
            ($solver:expr, $seed:expr) => {{
                let result = Executor::from_start(problem(), $solver, $seed)
                    .require_evaluated_state()
                    .max_iter(200)
                    .run_with_solver()
                    .unwrap();
                let (point, cost) = result.state.current().unwrap();
                assert_eq!(cost, NormSquared::norm_squared(point));
                assert!(cost < 128.0 * F::EPSILON);
                assert_eq!(result.state.counts(), &result.counts);
                assert_eq!(result.state.cost_evals(), result.counts.cost_evals);
                assert_eq!(
                    result.counts.total_work(),
                    result.counts.cost_evals
                );
                assert!(result.solver.rho().unwrap() <= F::EPSILON.sqrt());
            }};
        }
        powell_run!(
            basin::Newuoa::new().with_final_radius(F::EPSILON.sqrt()),
            x.clone()
        );
        powell_run!(
            basin::Bobyqa::new().with_final_radius(F::EPSILON.sqrt()),
            x.clone()
        );
        powell_run!(
            basin::Lincoa::new().with_final_radius(F::EPSILON.sqrt()),
            make(&[0.5, 0.5])
        );
        macro_rules! cobyla_run {
            ($problem:expr, $blocks:expr) => {{
                let result = Executor::from_start(
                    $problem,
                    basin::Cobyla::new().with_final_radius(F::EPSILON.sqrt()),
                    make(&[0.0, 0.0]),
                )
                .require_evaluated_state()
                .max_iter(200)
                .run_with_solver()
                .unwrap();
                let (point, cost, violation) = result.state.current().unwrap();
                assert_eq!(cost, NormSquared::norm_squared(point));
                assert!((cost - 1.0).abs() < 8.0 * F::EPSILON.sqrt());
                assert!(violation < 8.0 * F::EPSILON.sqrt());
                assert_eq!(result.state.current(), result.state.best());
                assert_eq!(result.state.counts(), &result.counts);
                assert_eq!(
                    result.counts.residual_evals,
                    $blocks * result.counts.cost_evals
                );
                assert_eq!(result.state.cost_evals(), result.counts.cost_evals);
                assert!(result.solver.rho().unwrap() <= F::EPSILON.sqrt());
            }};
        }
        cobyla_run!(problem(), 1);
        cobyla_run!(basin::FoldedConstraints::new(problem()), 2);
        macro_rules! mads_run {
            ($solver:expr, $target:expr, $blocks:expr) => {{
                let result = Executor::from_start(
                    problem(),
                    $solver.with_minimum_poll_size(F::EPSILON.sqrt()),
                    make(&[0.5, 0.5]),
                )
                .require_evaluated_state()
                .max_iter(200)
                .run_with_solver()
                .unwrap();
                assert_eq!(
                    result.state.cost(),
                    NormSquared::norm_squared(result.state.param())
                );
                assert!(
                    (result.state.cost() - $target).abs()
                        < 8.0 * F::EPSILON.sqrt()
                );
                assert_eq!(result.state.counts(), &result.counts);
                assert_eq!(
                    result.counts.residual_evals,
                    $blocks * result.counts.cost_evals
                );
                assert_eq!(result.state.cost_evals(), result.counts.cost_evals);
                assert!(
                    result.solver.poll_size().unwrap() <= F::EPSILON.sqrt()
                );
                assert!(result.solver.mesh_index().unwrap() > 0);
            }};
        }
        mads_run!(basin::Mads::new(), 0.0, 0);
        mads_run!(basin::Mads::new().bounded(), 0.0, 0);
        mads_run!(basin::Mads::new().constrained(), 1.0, 1);
        let slsqp = Executor::from_start(
            problem(),
            basin::Slsqp::new()
                .with_absolute_accuracy_tolerance(F::EPSILON.sqrt()),
            make(&[0.0, 0.5]),
        )
        .require_evaluated_state()
        .max_iter(50)
        .run_with_solver()
        .unwrap();
        let (point, cost, gradient, violation) = slsqp.state.current().unwrap();
        assert_eq!(cost, NormSquared::norm_squared(point));
        assert_eq!(gradient, &problem().gradient(point).unwrap());
        assert!((cost - 1.0).abs() < 8.0 * F::EPSILON.sqrt());
        assert!(violation < 8.0 * F::EPSILON.sqrt());
        assert_eq!(slsqp.solver.failure(), None);
        assert_eq!(slsqp.state.counts(), &slsqp.counts);
        assert_eq!(slsqp.state.cost_evals(), slsqp.counts.cost_evals);
        assert_eq!(
            basin::GradientState::gradient_evals(&slsqp.state),
            slsqp.counts.gradient_evals
        );
        assert!(slsqp.counts.residual_evals > 0);
        assert!(slsqp.counts.jacobian_evals > 0);
        macro_rules! cma_run {
            ($solver:expr, $bounded:expr) => {{
                let result = Executor::new(
                    problem(),
                    $solver,
                    basin::PopulationProgress::from_point(make(&[0.4, -0.4])),
                )
                .require_evaluated_state()
                .max_iter(35)
                .run_with_solver()
                .unwrap();
                for (point, cost) in result
                    .state
                    .candidates()
                    .iter()
                    .zip(result.state.costs().iter().copied())
                    .chain(result.state.current())
                    .chain(result.state.best())
                {
                    assert_eq!(cost, NormSquared::norm_squared(point));
                    if $bounded {
                        for i in 0..2 {
                            assert!((-5.0..=5.0).contains(&point[i]));
                        }
                    }
                }
                assert!(result.state.best_cost() < 0.01);
                assert_eq!(result.state.counts(), &result.counts);
                assert_eq!(result.state.cost_evals(), result.counts.cost_evals);
                result
            }};
        }
        let cma = cma_run!(
            basin::CmaEs::<V, M, F>::new(91, 0.7)
                .with_lambda(6)
                .with_stds(make(&[1.0, 0.3])),
            false
        );
        assert_eq!(cma.counts.cost_evals, 36 * 7);
        let bounded_cma = cma_run!(
            basin::BoundedCmaEs::<V, M, F>::new(91, 0.7).with_lambda(6),
            true
        );
        assert_eq!(bounded_cma.counts.cost_evals, 36 * 7);
        cma_run!(
            basin::CmaInject::with_inner_solver(
                basin::CmaEs::<V, M, F>::new(91, 0.7).with_lambda(6),
                basin::NelderMead::adaptive()
            )
            .with_inner_max_iter(3),
            false
        );
        let injected_cma = cma_run!(
            basin::BoundedCmaInject::with_inner_solver(
                basin::BoundedCmaEs::<V, M, F>::new(91, 0.7).with_lambda(6),
                basin::Lbfgsb::new()
            )
            .with_inner_max_iter(3),
            true
        );
        assert!(injected_cma.counts.gradient_evals > 0);
        macro_rules! chain_run {
            ($solver:expr) => {{
                let result = Executor::new(
                    problem(),
                    $solver
                        .with_pop_size(4)
                        .with_nfrec(2)
                        .with_ls_intensity(18),
                    basin::PopulationProgress::empty(),
                )
                .require_evaluated_state()
                .max_iter(15)
                .run_with_solver()
                .unwrap();
                for (point, &cost) in
                    result.state.candidates().iter().zip(result.state.costs())
                {
                    assert_eq!(cost, NormSquared::norm_squared(point));
                }
                assert!(result.state.best_cost() < 0.2);
                assert_eq!(result.state.counts(), &result.counts);
                assert_eq!(result.state.cost_evals(), result.counts.cost_evals);
            }};
        }
        chain_run!(basin::MaLsChCma::<V, M, F>::new(71));
        chain_run!(basin::MaLsChSw::<V, F>::new(71));
        macro_rules! population_run {
            ($solver:expr) => {{
                let result = Executor::new(
                    problem(),
                    $solver,
                    basin::PopulationProgress::from_population(vec![
                        make(&[3.0, 4.0]),
                        make(&[-3.0, -4.0]),
                        make(&[8.0, -8.0]),
                        make(&[-2.0, 1.0]),
                    ]),
                )
                .require_evaluated_state()
                .max_iter(40)
                .run_with_solver()
                .unwrap();
                for (point, &cost) in
                    result.state.candidates().iter().zip(result.state.costs())
                {
                    assert_eq!(cost, NormSquared::norm_squared(point));
                    for i in 0..2 {
                        assert!((-5.0..=5.0).contains(&point[i]));
                    }
                }
                let (best, cost) = result.state.best().unwrap();
                assert_eq!(cost, NormSquared::norm_squared(best));
                assert!(cost <= 5.0);
                assert_eq!(result.state.counts(), &result.counts);
                result
            }};
        }
        let random = population_run!(basin::RandomSearch::new(4, 73));
        assert_eq!(random.state.cost_evals(), 4 * 41);
        let de = population_run!(basin::De::new(73).with_pop_size(4));
        assert_eq!(de.state.cost_evals(), 4 * 41);
        let ssga = population_run!(basin::Ssga::new(73).with_pop_size(4));
        assert_eq!(ssga.state.cost_evals(), 4 + 2 * 40);
        let injected = population_run!(
            basin::DeInject::with_inner_solver(
                basin::De::new(73).with_pop_size(4),
                basin::Lbfgsb::new()
            )
            .with_inner_max_iter(10)
            .with_refine_every(3)
        );
        assert!(injected.state.best_cost() < 128.0 * F::EPSILON);
        assert!(injected.counts.gradient_evals > 0);
        assert_eq!(injected.state.cost_evals(), injected.counts.cost_evals);
        let swarm = Executor::new(
            problem(),
            basin::GlobalBestPso::new(73).with_swarm_size(16),
            basin::PopulationProgress::empty(),
        )
        .require_evaluated_state()
        .max_iter(150)
        .run_with_solver()
        .unwrap();
        assert!(swarm.state.best_cost() < 128.0 * F::EPSILON);
        assert_eq!(swarm.state.cost_evals(), 16 * 151);
        assert_eq!(swarm.state.counts(), &swarm.counts);
        assert_eq!(swarm.solver.velocities().len(), 16);
        for (i, (point, &cost)) in swarm
            .state
            .candidates()
            .iter()
            .zip(swarm.state.costs())
            .enumerate()
        {
            assert_eq!(cost, NormSquared::norm_squared(point));
            assert_eq!(
                swarm.solver.personal_best_costs()[i],
                NormSquared::norm_squared(
                    &swarm.solver.personal_best_positions()[i]
                )
            );
        }
        let gbnm =
            Executor::from_start(problem(), basin::Gbnm::new(73), x.clone())
                .require_evaluated_state()
                .max_iter(200)
                .run_with_solver()
                .unwrap();
        let (best, cost) = gbnm.state.best().unwrap();
        assert_eq!(cost, NormSquared::norm_squared(best));
        assert!(cost < 1e-3);
        assert_eq!(
            gbnm.state.current().unwrap().0[0],
            gbnm.solver.vertices()[0][0]
        );
        assert_eq!(gbnm.state.counts(), &gbnm.counts);
        assert_eq!(gbnm.state.cost_evals(), gbnm.counts.total_work());
        let simplex = Executor::from_start(
            problem(),
            basin::NelderMead::adaptive(),
            x.clone(),
        )
        .require_evaluated_state()
        .max_iter(200)
        .run()
        .unwrap();
        assert!(simplex.best_cost() < 128.0 * F::EPSILON);
        assert_eq!(
            simplex.state.counts().total_work(),
            simplex.state.counts().cost_evals
        );
        assert_eq!(
            simplex.state.current().unwrap().1,
            NormSquared::norm_squared(simplex.state.current().unwrap().0)
        );
        let projected_simplex = Executor::from_start(
            problem(),
            basin::NelderMead::new().projected(),
            x.clone(),
        )
        .require_evaluated_state()
        .max_iter(200)
        .run()
        .unwrap();
        assert!(projected_simplex.best_cost() < 128.0 * F::EPSILON);
        assert_eq!(
            projected_simplex.state.counts().total_work(),
            projected_simplex.state.counts().cost_evals
        );
        let bfgs = Executor::new(
            problem(),
            Bfgs::<_, F>::with_line_search(basin::Wolfe::<F>::new()),
            FirstOrderState::<V, F>::new(x.clone()),
        )
        .max_iter(50)
        .run()
        .unwrap();
        assert!(bfgs.cost() < 128.0 * F::EPSILON);
        let seeded = Executor::from_start(
            problem(),
            Bfgs::<_, F>::with_line_search(basin::Wolfe::<F>::new()),
            x.clone(),
        )
        .max_iter(50)
        .run()
        .unwrap();
        assert!(seeded.cost() < 128.0 * F::EPSILON);
        let lbfgs = Executor::new(
            problem(),
            Lbfgs::<_, F>::with_line_search(basin::MoreThuente::<F>::new())
                .with_m_capacity(5),
            FirstOrderState::new(x.clone()),
        )
        .max_iter(50)
        .run()
        .unwrap();
        assert!(lbfgs.cost() < 128.0 * F::EPSILON);
        let unbounded = Executor::from_start(
            problem(),
            Lbfgs::<_, F>::new().unbounded().with_m_capacity(5),
            x.clone(),
        )
        .require_evaluated_state()
        .max_iter(50)
        .run()
        .unwrap();
        assert!(unbounded.cost() < 128.0 * F::EPSILON);
        let descent = Executor::from_start(
            problem(),
            basin::GradientDescent::new(0.25).with_momentum(0.2),
            x.clone(),
        )
        .require_evaluated_state()
        .max_iter(50)
        .run()
        .unwrap();
        assert!(descent.cost() < 128.0 * F::EPSILON);
        let hopping = Executor::from_start(
            problem(),
            basin::BasinHopping::new(basin::GradientDescent::new(0.25), 73)
                .with_inner_max_iter(25)
                .with_adaptive_interval(2),
            x.clone(),
        )
        .require_evaluated_state()
        .max_iter(3)
        .run()
        .unwrap();
        assert!(hopping.cost() < 128.0 * F::EPSILON);
        assert_eq!(hopping.state.counts().cost_evals, 104);
        assert_eq!(hopping.state.counts().gradient_evals, 104);
        let constrained = Executor::from_start(
            problem(),
            basin::AugmentedLagrangianMethod::with_inner_solver(
                basin::GradientDescent::new(0.05),
            )
            .with_inner_max_iter(100),
            make(&[0.0, 0.0]),
        )
        .require_evaluated_state()
        .max_iter(6)
        .run()
        .unwrap();
        let (_, best_cost, violation) = constrained.state.best().unwrap();
        assert!(violation < 1e-4);
        assert!((best_cost - 2.0).abs() < 1e-3);
        for start in [make(&[0.0, 0.0]), x.clone()] {
            let barrier = Executor::from_start(
                problem(),
                basin::BarrierMethod::with_inner_solver(
                    basin::GradientDescent::with_line_search(
                        basin::Backtracking::<F>::new(),
                    )
                    .with_absolute_gradient_tolerance(1e-5 as F),
                ),
                start,
            )
            .require_evaluated_state()
            .max_iter(20)
            .run()
            .unwrap();
            let (best, cost) = barrier.state.best().unwrap();
            assert!(best[0] < 1.0 && best[1] < 1.0);
            assert!(cost < 1e-4);
            assert_eq!(cost, NormSquared::norm_squared(best));
            assert!(barrier.state.counts().gradient_evals > 0);
        }
        let annealing = Executor::from_start(
            problem(),
            basin::SimulatedAnnealing::new(
                |x: &V, _: F, _: &mut basin::core::rng::ChaCha8Rng| {
                    let mut candidate = x.clone();
                    candidate.scale_in_place(0.5);
                    candidate
                },
                1.0,
                basin::TemperatureSchedule::<F>::geometric(0.9),
                73,
            )
            .with_reannealing_fixed(7),
            x.clone(),
        )
        .require_evaluated_state()
        .max_iter(30)
        .run_with_solver()
        .unwrap();
        assert!(annealing.cost() < 128.0 * F::EPSILON);
        assert_eq!(annealing.state.accepted_moves(), 30);
        assert_eq!(annealing.state.rejected_moves(), 0);
        assert_eq!(annealing.state.last_accepted_iter(), 30);
        assert_eq!(annealing.state.counts().cost_evals, 31);
        assert_eq!(annealing.state.counts().total_work(), 31);
        assert_eq!(annealing.solver.reannealings(), 4);
        let stochastic = Executor::from_start(
            problem(),
            basin::Sgd::new(0.25, 2, 73).with_momentum(0.2),
            x.clone(),
        )
        .require_evaluated_state()
        .max_iter(50)
        .run()
        .unwrap();
        assert!(stochastic.cost() < 128.0 * F::EPSILON);
        assert_eq!(stochastic.state.counts().gradient_evals, 50);
        assert_eq!(stochastic.state.counts().cost_evals, 17);
        let random_walk = Executor::from_start(
            problem(),
            basin::SolisWets::new(73).with_initial_step_size(0.4),
            x.clone(),
        )
        .require_evaluated_state()
        .max_iter(500)
        .run()
        .unwrap();
        assert!(random_walk.cost() < 1e-4);
        let projected = Executor::from_start(
            problem(),
            basin::ProjectedGradientDescent::new(0.25),
            x.clone(),
        )
        .require_evaluated_state()
        .max_iter(50)
        .run()
        .unwrap();
        assert!(projected.cost() < 128.0 * F::EPSILON);
        let exact = Executor::from_start(
            problem(),
            basin::TrustRegion::with_subproblem(basin::Steihaug::new()),
            x.clone(),
        )
        .require_evaluated_state()
        .max_iter(50)
        .run()
        .unwrap();
        assert!(exact.cost() < 128.0 * F::EPSILON);
        let matrix_free = Executor::from_start(
            problem(),
            basin::TrustRegion::matrix_free_with(basin::Steihaug::new()),
            x,
        )
        .require_evaluated_state()
        .max_iter(50)
        .run()
        .unwrap();
        assert!(matrix_free.cost() < 128.0 * F::EPSILON);
    }};
}

#[test]
fn vec_f32_dense_and_solvers() {
    dense_checks!(Vec<f32>, basin::DenseMatrix<f32>, f32, |x: &[f32]| x
        .to_vec());
}

#[test]
fn vec_f64_dense_and_solvers() {
    dense_checks!(Vec<f64>, basin::DenseMatrix, f64, |x: &[f64]| x.to_vec());
}

#[cfg(any(
    feature = "nalgebra_all",
    feature = "ndarray_all",
    feature = "faer_all"
))]
macro_rules! simplex_check {
    ($make:expr) => {{
        use basin::IntoInitialSimplex;
        let initial = ($make)(&[2.0_f64, 0.0]);
        let simplex = initial.into_initial_simplex(0.05);
        assert_eq!(simplex.len(), 3);
        assert_eq!(simplex[0][0], 2.0);
        assert_eq!(simplex[1][0], 2.1);
        assert_eq!(simplex[2][1], 0.00025);
    }};
}

#[cfg(all(
    feature = "problems",
    any(
        feature = "nalgebra_all",
        feature = "ndarray_all",
        feature = "faer_all"
    )
))]
macro_rules! corpus_checks {
    ($vector:ty, $matrix:ty, $make:expr) => {{
        use basin::problems::*;
        use basin::{
            BoxConstraints, CostFunction, Gradient, Jacobian,
            LinearEqualityConstraints, MatrixIdentity, NormSquared, Residual,
        };
        type V = $vector;
        type M = $matrix;
        let make = $make;
        let x = make(&[3.0, 4.0]);

        fn cost<P: CostFunction<Param = V, Output = f64>>() {}
        fn gradient<
            P: Gradient<Gradient = V> + CostFunction<Param = V, Output = f64>,
        >() {
        }
        fn jacobian<
            P: Jacobian<Jacobian = M> + Residual<Param = V, Output = V>,
        >() {
        }
        cost::<Ackley<V>>();
        cost::<AckleyBoxed<V>>();
        cost::<Beale<V>>();
        cost::<Booth<V>>();
        cost::<BoothBoxed<V>>();
        cost::<BoothBoxedResiduals<V>>();
        cost::<BoothResiduals<V>>();
        cost::<BukinN6<V>>();
        cost::<CrossInTray<V>>();
        cost::<Easom<V>>();
        cost::<Eggholder<V>>();
        cost::<ExponentialFit<V>>();
        cost::<GoldsteinPrice<V>>();
        cost::<Himmelblau<V>>();
        cost::<HolderTable<V>>();
        cost::<Levy<V>>();
        cost::<LevyBoxed<V>>();
        cost::<Matyas<V>>();
        cost::<McCormick<V>>();
        cost::<Picheny<V>>();
        cost::<PowellSingular<V>>();
        cost::<Rastrigin<V>>();
        cost::<RastriginBoxed<V>>();
        cost::<Rosenbrock<V>>();
        cost::<RosenbrockResiduals<V>>();
        cost::<SchafferN2<V>>();
        cost::<SchafferN4<V>>();
        cost::<Sphere<V>>();
        cost::<SphereBoxed<V>>();
        cost::<Step<V>>();
        cost::<StyblinskiTang<V>>();
        cost::<StyblinskiTangBoxed<V>>();
        cost::<ThreeHumpCamel<V>>();
        cost::<Zero<V>>();
        gradient::<Beale<V>>();
        gradient::<Booth<V>>();
        gradient::<BoothBoxed<V>>();
        gradient::<Easom<V>>();
        gradient::<GoldsteinPrice<V>>();
        gradient::<Himmelblau<V>>();
        gradient::<Levy<V>>();
        gradient::<LevyBoxed<V>>();
        gradient::<Matyas<V>>();
        gradient::<McCormick<V>>();
        gradient::<Picheny<V>>();
        gradient::<Rosenbrock<V>>();
        gradient::<SchafferN2<V>>();
        gradient::<Sphere<V>>();
        gradient::<StyblinskiTang<V>>();
        gradient::<StyblinskiTangBoxed<V>>();
        gradient::<ThreeHumpCamel<V>>();
        gradient::<Zero<V>>();
        jacobian::<BoothBoxedResiduals<V>>();
        jacobian::<BoothResiduals<V>>();
        jacobian::<PowellSingular<V>>();
        jacobian::<RosenbrockResiduals<V>>();
        let sphere = Sphere::<V>::new();
        assert_eq!(sphere.cost(&x).unwrap(), 25.0);
        assert_eq!(
            NormSquared::norm_squared(&sphere.gradient(&x).unwrap()),
            100.0
        );
        let boxed =
            SphereBoxed::<V>::new(make(&[-5.0, -5.0]), make(&[5.0, 5.0]));
        assert_eq!(boxed.lower()[0], -5.0);
        assert_eq!(boxed.upper()[1], 5.0);
        let residuals = RosenbrockResiduals::<V>::new();
        let at_minimum = make(&[1.0, 1.0]);
        assert_eq!(
            NormSquared::norm_squared(
                &residuals.residual(&at_minimum).unwrap()
            ),
            0.0
        );
        let jacobian = residuals.jacobian(&at_minimum).unwrap();
        assert_eq!(basin::MatrixIndex::matrix_entry(&jacobian, 0, 0), -20.0);
        let equality = EqualityConstrainedQuadratic::new(
            make(&[0.0, 0.0]),
            <M as MatrixIdentity>::identity(2),
            make(&[1.0, 1.0]),
        );
        assert_eq!(equality.b()[0], 1.0);
        assert_eq!(equality.cost(&x).unwrap(), 25.0);
    }};
}

#[cfg(all(
    feature = "problems",
    any(feature = "nalgebra_all", feature = "faer_all")
))]
macro_rules! sparse_problem_checks {
    (f32, $a:expr, $make:expr) => {};
    (f64, $a:expr, $make:expr) => {{
        use basin::problems::{SparseLeastSquares, SparseLeastSquaresBoxed};
        use basin::{
            BoxConstraints, Executor, GaussNewton, Jacobian, PointState,
            Residual,
        };
        let a = $a;
        let make = $make;
        let target = make(&[1.0, 2.0]);
        let b = basin::MatVec::matvec(&a, &target);
        let bounded = SparseLeastSquaresBoxed::new(
            a.clone(),
            b.clone(),
            make(&[-5.0, -5.0]),
            make(&[5.0, 5.0]),
        );
        assert_eq!(bounded.lower()[0], -5.0);
        assert_eq!(bounded.upper()[1], 5.0);
        assert_eq!(
            basin::NormSquared::norm_squared(
                &bounded.residual(&target).unwrap()
            ),
            0.0
        );
        let jacobian = bounded.jacobian(&target).unwrap();
        let product = basin::MatVec::matvec(&jacobian, &target);
        assert_eq!(product[0], b[0]);
        let result = Executor::new(
            SparseLeastSquares::new(a, b),
            GaussNewton::new(),
            PointState::new(make(&[0.0, 0.0])),
        )
        .max_iter(10)
        .run()
        .unwrap();
        assert!(result.cost() < 1e-10);
    }};
}
#[cfg(any(feature = "nalgebra_all", feature = "faer_all"))]
macro_rules! sparse_checks {
    ($scalar:ident, $make_vector:expr, $make_matrix:expr) => {{
        use basin::{GramMatrix, LinearSolveSpd, MatVec};
        type F = $scalar;
        let make = $make_vector;
        let a = ($make_matrix)();
        let b = make(&[1.0, 2.0]);
        let x = a.solve_spd(&b).unwrap();
        assert!((x[0] - 1.0 / 11.0).abs() < 32.0 * F::EPSILON);
        assert!((x[1] - 7.0 / 11.0).abs() < 32.0 * F::EPSILON);
        let product = a.matvec(&x);
        let gram_product = a.gram().matvec(&x);
        let twice = a.matvec(&product);
        #[cfg(feature = "problems")]
        sparse_problem_checks!($scalar, a, make);
        for i in 0..2 {
            assert!((product[i] - b[i]).abs() < 64.0 * F::EPSILON);
            assert!((gram_product[i] - twice[i]).abs() < 256.0 * F::EPSILON);
        }
    }};
}
#[cfg(feature = "nalgebra_all")]
macro_rules! nalgebra_checks {
    ($module:ident, $dependency:ident, $sparse:ident) => {
        mod $module {
            use $dependency as backend;
            #[test]
            fn f32_dense_and_solvers() {
                dense_checks!(
                    backend::DVector<f32>,
                    backend::DMatrix<f32>,
                    f32,
                    |x: &[f32]| backend::DVector::from_column_slice(x)
                );
            }
            #[test]
            fn f64_dense_and_solvers() {
                dense_checks!(
                    backend::DVector<f64>,
                    backend::DMatrix<f64>,
                    f64,
                    |x: &[f64]| backend::DVector::from_column_slice(x)
                );
            }
            #[test]
            fn simplex() {
                simplex_check!(
                    |x: &[f64]| backend::DVector::from_column_slice(x)
                );
            }
            #[cfg(feature = "problems")]
            #[test]
            fn corpus() {
                corpus_checks!(
                    backend::DVector<f64>,
                    backend::DMatrix<f64>,
                    |x: &[f64]| backend::DVector::from_column_slice(x)
                );
            }
            #[test]
            fn f32_sparse() {
                sparse_checks!(
                    f32,
                    |x: &[f32]| backend::DVector::from_column_slice(x),
                    || {
                        let mut coo = $sparse::CooMatrix::<f32>::new(2, 2);
                        coo.push(0, 0, 4.0);
                        coo.push(0, 1, 1.0);
                        coo.push(1, 0, 1.0);
                        coo.push(1, 1, 3.0);
                        $sparse::CscMatrix::from(&coo)
                    }
                );
            }

            #[test]
            fn f64_sparse() {
                sparse_checks!(
                    f64,
                    |x: &[f64]| backend::DVector::from_column_slice(x),
                    || {
                        let mut coo = $sparse::CooMatrix::<f64>::new(2, 2);
                        coo.push(0, 0, 4.0);
                        coo.push(0, 1, 1.0);
                        coo.push(1, 0, 1.0);
                        coo.push(1, 1, 3.0);
                        $sparse::CscMatrix::from(&coo)
                    }
                );
            }
        }
    };
}

#[cfg(feature = "nalgebra_v0_32")]
nalgebra_checks!(nalgebra_0_32, nalgebra_0_32, nalgebra_sparse_0_9);
#[cfg(feature = "nalgebra_v0_33")]
nalgebra_checks!(nalgebra_0_33, nalgebra_0_33, nalgebra_sparse_0_10);
#[cfg(feature = "nalgebra_v0_34")]
nalgebra_checks!(nalgebra_0_34, nalgebra_0_34, nalgebra_sparse_0_11);
#[cfg(feature = "nalgebra_v0_35")]
nalgebra_checks!(nalgebra_0_35, nalgebra, nalgebra_sparse);

#[cfg(feature = "ndarray_all")]
macro_rules! ndarray_checks {
    ($module:ident, $dependency:ident) => {
        mod $module {
            use $dependency as backend;
            #[test]
            fn f32_dense_and_solvers() {
                dense_checks!(
                    backend::Array1<f32>,
                    backend::Array2<f32>,
                    f32,
                    |x: &[f32]| backend::Array1::from_vec(x.to_vec())
                );
            }
            #[test]
            fn f64_dense_and_solvers() {
                dense_checks!(
                    backend::Array1<f64>,
                    backend::Array2<f64>,
                    f64,
                    |x: &[f64]| backend::Array1::from_vec(x.to_vec())
                );
            }
            #[test]
            fn simplex() {
                simplex_check!(|x: &[f64]| backend::Array1::from_vec(
                    x.to_vec()
                ));
            }
            #[cfg(feature = "problems")]
            #[test]
            fn corpus() {
                corpus_checks!(
                    backend::Array1<f64>,
                    backend::Array2<f64>,
                    |x: &[f64]| backend::Array1::from_vec(x.to_vec())
                );
            }
        }
    };
}

#[cfg(feature = "ndarray_v0_15")]
ndarray_checks!(ndarray_0_15, ndarray_0_15);
#[cfg(feature = "ndarray_v0_16")]
ndarray_checks!(ndarray_0_16, ndarray_0_16);
#[cfg(feature = "ndarray_v0_17")]
ndarray_checks!(ndarray_0_17, ndarray);

#[cfg(feature = "faer_all")]
macro_rules! faer_checks {
    ($module:ident, $dependency:ident) => {
        mod $module {
            use $dependency as backend;
            #[test]
            fn f32_dense_and_solvers() {
                dense_checks!(
                    backend::Col<f32>,
                    backend::Mat<f32>,
                    f32,
                    |x: &[f32]| backend::Col::from_fn(x.len(), |i| x[i])
                );
            }
            #[test]
            fn f64_dense_and_solvers() {
                dense_checks!(
                    backend::Col<f64>,
                    backend::Mat<f64>,
                    f64,
                    |x: &[f64]| backend::Col::from_fn(x.len(), |i| x[i])
                );
            }
            #[test]
            fn simplex() {
                simplex_check!(|x: &[f64]| backend::Col::from_fn(
                    x.len(),
                    |i| x[i]
                ));
            }
            #[cfg(feature = "problems")]
            #[test]
            fn corpus() {
                corpus_checks!(
                    backend::Col<f64>,
                    backend::Mat<f64>,
                    |x: &[f64]| { backend::Col::from_fn(x.len(), |i| x[i]) }
                );
            }
            #[test]
            fn f32_sparse() {
                sparse_checks!(
                    f32,
                    |x: &[f32]| backend::Col::from_fn(x.len(), |i| x[i]),
                    || {
                        use backend::sparse::{SparseColMat, Triplet};
                        SparseColMat::<usize, f32>::try_new_from_triplets(
                            2,
                            2,
                            &[
                                Triplet::new(0, 0, 4.0),
                                Triplet::new(0, 1, 1.0),
                                Triplet::new(1, 0, 1.0),
                                Triplet::new(1, 1, 3.0),
                            ],
                        )
                        .unwrap()
                    }
                );
            }

            #[test]
            fn f64_sparse() {
                sparse_checks!(
                    f64,
                    |x: &[f64]| backend::Col::from_fn(x.len(), |i| x[i]),
                    || {
                        use backend::sparse::{SparseColMat, Triplet};
                        SparseColMat::<usize, f64>::try_new_from_triplets(
                            2,
                            2,
                            &[
                                Triplet::new(0, 0, 4.0),
                                Triplet::new(0, 1, 1.0),
                                Triplet::new(1, 0, 1.0),
                                Triplet::new(1, 1, 3.0),
                            ],
                        )
                        .unwrap()
                    }
                );
            }
        }
    };
}

#[cfg(feature = "faer_v0_22")]
faer_checks!(faer_0_22, faer_0_22);
#[cfg(feature = "faer_v0_23")]
faer_checks!(faer_0_23, faer_0_23);
#[cfg(feature = "faer_v0_24")]
faer_checks!(faer_0_24, faer);
