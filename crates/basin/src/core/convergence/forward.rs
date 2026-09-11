//! Forward algorithm builders without creating another convergence layer.
use super::*;
use crate::solver::lbfgs::Lbfgs;
use crate::solver::*;
use crate::{CountsMirror, FactorizePivotedQr};

// Keep algorithm configuration available after setting optional convergence.
macro_rules! forward_setting {
    ($solver:ident, $name:ident, $arg:ident: $value:ty) => {
        #[doc = concat!("Configure [`", stringify!($name), "`](", stringify!($solver), "::", stringify!($name), ") while retaining convergence settings.")]
        pub fn $name(self, $arg: $value) -> Self {
            self.map_solver(|solver| solver.$name($arg))
        }
    };
}

impl<CG, CX, CC, CT, So, F: Scalar>
    ConfiguredSolver<BarrierMethod<So, F>, CG, CX, CC, CT>
{
    forward_setting!(BarrierMethod, mu0, mu0: F);
    forward_setting!(BarrierMethod, with_reduction, reduction: F);
    forward_setting!(BarrierMethod, with_absolute_duality_gap_tolerance, tol: impl Into<Option<F>>);
    forward_setting!(BarrierMethod, with_absolute_phase_one_gap_tolerance, phase_one_tol: F);
    forward_setting!(BarrierMethod, with_inner_max_iter, inner_max_iter: u64);
}

impl<CG, CX, CC, CT, S, F: Scalar>
    ConfiguredSolver<Bfgs<S, F>, CG, CX, CC, CT>
{
    forward_setting!(Bfgs, with_relative_curvature_tolerance, epsilon: F);
}

impl<CG, CX, CC, CT, F: Scalar>
    ConfiguredSolver<NelderMead<Unbounded, F>, CG, CX, CC, CT>
{
    /// Configure [`projected`](NelderMead::projected) while retaining convergence settings.
    pub fn projected(
        self,
    ) -> ConfiguredSolver<NelderMead<Projected, F>, CG, CX, CC, CT> {
        self.map_solver(|solver| solver.projected())
    }
}

impl<CG, CX, CC, CT, S, F: Scalar>
    ConfiguredSolver<Lbfgs<crate::solver::lbfgs::Bounded, S, F>, CG, CX, CC, CT>
{
    forward_setting!(Lbfgs, with_absolute_projected_gradient_tolerance, value: impl Into<Option<F>>);
    /// Configure [`unbounded`](Lbfgs::unbounded) while retaining convergence settings.
    pub fn unbounded(
        self,
    ) -> ConfiguredSolver<
        Lbfgs<crate::solver::lbfgs::Unbounded, S, F>,
        CG,
        CX,
        CC,
        CT,
    > {
        self.map_solver(|solver| solver.unbounded())
    }
}

impl<CG, CX, CC, CT, S, F: Scalar>
    ConfiguredSolver<
        Lbfgs<crate::solver::lbfgs::Unbounded, S, F>,
        CG,
        CX,
        CC,
        CT,
    >
{
    /// Configure [`bounded`](Lbfgs::bounded) while retaining convergence settings.
    pub fn bounded(
        self,
    ) -> ConfiguredSolver<
        Lbfgs<crate::solver::lbfgs::Bounded, S, F>,
        CG,
        CX,
        CC,
        CT,
    > {
        self.map_solver(|solver| solver.bounded())
    }
}

impl<CG, CX, CC, CT, Mode, S, F: Scalar>
    ConfiguredSolver<Lbfgs<Mode, S, F>, CG, CX, CC, CT>
{
    forward_setting!(Lbfgs, with_relative_curvature_tolerance, epsilon: F);
    forward_setting!(Lbfgs, with_m_capacity, m_capacity: usize);
}

impl<CG, CX, CC, CT, F: Scalar> ConfiguredSolver<Newuoa<F>, CG, CX, CC, CT> {
    forward_setting!(Newuoa, with_absolute_radius_tolerance, value: impl Into<Option<F>>);
    forward_setting!(Newuoa, with_initial_radius, rho_beg: F);
    forward_setting!(Newuoa, with_final_radius, rho_end: F);
    forward_setting!(Newuoa, with_npt, npt: usize);
}

impl<CG, CX, CC, CT, F: Scalar>
    ConfiguredSolver<Bobyqa<crate::solver::bobyqa::Bounded, F>, CG, CX, CC, CT>
{
    forward_setting!(Bobyqa, with_absolute_radius_tolerance, value: impl Into<Option<F>>);
    forward_setting!(Bobyqa, with_initial_radius, rho_beg: F);
    forward_setting!(Bobyqa, with_final_radius, rho_end: F);
    forward_setting!(Bobyqa, with_npt, npt: usize);
}

impl<CG, CX, CC, CT, I, V, F>
    ConfiguredSolver<DeInject<I, V, F>, CG, CX, CC, CT>
where
    F: Scalar,
    I: MemeticInner<V, F>,
    I::State: CountsMirror,
{
    forward_setting!(DeInject, with_k, k: usize);
    forward_setting!(DeInject, with_refine_every, n: u64);
    forward_setting!(DeInject, with_inner_max_iter, n: u64);
    /// Configure [`inner_stop_when_factory`](DeInject::inner_stop_when_factory) while retaining convergence settings.
    pub fn inner_stop_when_factory<Mk, CheckFn>(self, make: Mk) -> Self
    where
        Mk: FnMut() -> CheckFn + 'static,
        CheckFn: FnMut(&I::State) -> Option<TerminationReason> + 'static,
    {
        self.map_solver(|solver| solver.inner_stop_when_factory(make))
    }
}

impl<CG, CX, CC, CT, I, V, F>
    ConfiguredSolver<DeInject<I, V, F>, CG, CX, CC, CT>
where
    F: Scalar,
    I: MemeticInner<V, F>,
    I::State: CountsMirror,
{
    forward_setting!(DeInject, with_ls_intensity, evals: u64);
}

impl<CG, CX, CC, CT, V, LS> ConfiguredSolver<MaLsCh<V, LS>, CG, CX, CC, CT> {
    forward_setting!(MaLsCh, with_pop_size, pop_size: usize);
    forward_setting!(MaLsCh, with_blx_alpha, alpha: f64);
    forward_setting!(MaLsCh, with_nam_pool, pool: usize);
    forward_setting!(MaLsCh, with_mutation_prob, p: f64);
    forward_setting!(MaLsCh, with_bga_range_fraction, f: f64);
    forward_setting!(MaLsCh, with_ls_intensity, istr: u64);
    forward_setting!(MaLsCh, with_ls_improvement_threshold, delta: f64);
    forward_setting!(MaLsCh, with_nfrec, n: u64);
    forward_setting!(MaLsCh, with_initial_scale_fallback, scale: f64);
}

impl<CG, CX, CC, CT, F: Scalar> ConfiguredSolver<De<F>, CG, CX, CC, CT> {
    forward_setting!(De, with_pop_size, pop_size: usize);
    forward_setting!(De, with_f, f: F);
    forward_setting!(De, with_cr, cr: f64);
}

impl<CG, CX, CC, CT, Mode, F: Scalar>
    ConfiguredSolver<Mads<Mode, F>, CG, CX, CC, CT>
{
    forward_setting!(Mads, with_absolute_poll_size_tolerance, value: impl Into<Option<F>>);
    forward_setting!(Mads, with_initial_poll_size, poll_size_init: F);
    forward_setting!(Mads, with_minimum_poll_size, poll_size_min: F);
}

impl<CG, CX, CC, CT, F: Scalar>
    ConfiguredSolver<
        Mads<crate::solver::nelder_mead::Unbounded, F>,
        CG,
        CX,
        CC,
        CT,
    >
{
    /// Configure [`bounded`](Mads::bounded) while retaining convergence settings.
    pub fn bounded(self) -> ConfiguredSolver<Mads<Bounded, F>, CG, CX, CC, CT> {
        self.map_solver(|solver| solver.bounded())
    }
    /// Configure [`constrained`](Mads::constrained) while retaining convergence settings.
    pub fn constrained(
        self,
    ) -> ConfiguredSolver<Mads<Constrained, F>, CG, CX, CC, CT> {
        self.map_solver(|solver| solver.constrained())
    }
}

impl<CG, CX, CC, CT, F: Scalar> ConfiguredSolver<Brent<F>, CG, CX, CC, CT> {
    forward_setting!(Brent, with_absolute_position_tolerance, value: F);
    forward_setting!(Brent, with_relative_position_tolerance, value: F);
}

impl<CG, CX, CC, CT, F: Scalar> ConfiguredSolver<Gbnm<F>, CG, CX, CC, CT> {
    /// Configure [`with_nm_params`](Gbnm::with_nm_params) while retaining convergence settings.
    pub fn with_nm_params(self, alpha: F, beta: F, gamma: F, delta: F) -> Self {
        self.map_solver(|solver| {
            solver.with_nm_params(alpha, beta, gamma, delta)
        })
    }
    forward_setting!(Gbnm, with_restart_candidates, count: usize);
    forward_setting!(Gbnm, with_gaussian_variance_factor, factor: F);
    forward_setting!(Gbnm, with_initial_simplex_fraction, fraction: F);
    /// Configure [`with_probabilistic_simplex_fractions`](Gbnm::with_probabilistic_simplex_fractions) while retaining convergence settings.
    pub fn with_probabilistic_simplex_fractions(self, min: F, max: F) -> Self {
        self.map_solver(|solver| {
            solver.with_probabilistic_simplex_fractions(min, max)
        })
    }
    /// Configure [`with_test_simplex_fractions`](Gbnm::with_test_simplex_fractions) while retaining convergence settings.
    pub fn with_test_simplex_fractions(self, small: F, large: F) -> Self {
        self.map_solver(|solver| {
            solver.with_test_simplex_fractions(small, large)
        })
    }
    forward_setting!(Gbnm, with_normalized_simplex_size_tolerance, tolerance: F);
    forward_setting!(Gbnm, with_absolute_simplex_cost_tolerance, tolerance: F);
    forward_setting!(Gbnm, with_edge_ratio_tolerance, value: F);
    forward_setting!(Gbnm, with_normalized_determinant_tolerance, value: F);
}

impl<CG, CX, CC, CT, N, F, R>
    ConfiguredSolver<SimulatedAnnealing<N, F, R>, CG, CX, CC, CT>
where
    F: Scalar,
{
    forward_setting!(SimulatedAnnealing, with_reannealing, reannealing: Reannealing);
    forward_setting!(SimulatedAnnealing, with_reannealing_fixed, iterations: u64);
    forward_setting!(SimulatedAnnealing, with_reannealing_accepted, iterations: u64);
    forward_setting!(SimulatedAnnealing, with_reannealing_best, iterations: u64);
}

impl<CG, CX, CC, CT, F: Scalar> ConfiguredSolver<SolisWets<F>, CG, CX, CC, CT> {
    forward_setting!(SolisWets, with_absolute_step_size_tolerance, value: impl Into<Option<F>>);
    forward_setting!(SolisWets, with_initial_step_size, rho_init: F);
    forward_setting!(SolisWets, with_bias_gain, bias_gain: F);
    forward_setting!(SolisWets, with_bias_memory, bias_memory: F);
    forward_setting!(SolisWets, with_bias_decay, bias_decay: F);
    forward_setting!(SolisWets, with_expand_threshold, expand_threshold: u32);
    forward_setting!(SolisWets, with_contract_threshold, contract_threshold: u32);
    forward_setting!(SolisWets, with_expand_factor, expand_factor: F);
    forward_setting!(SolisWets, with_contract_factor, contract_factor: F);
}

impl<CG, CX, CC, CT, V, M, F: Scalar>
    ConfiguredSolver<Trf<V, M, F>, CG, CX, CC, CT>
{
    forward_setting!(Trf, with_absolute_scaled_gradient_tolerance, value: impl Into<Option<F>>);
    forward_setting!(Trf, with_tau, tau: F);
    forward_setting!(Trf, with_rstep, rstep: F);
    forward_setting!(Trf, with_theta, theta: F);
    forward_setting!(Trf, with_max_inner_attempts, n: u32);
}

impl<CG, CX, CC, CT, F: Scalar>
    ConfiguredSolver<BrentDerivative<F>, CG, CX, CC, CT>
{
    forward_setting!(BrentDerivative, with_absolute_position_tolerance, value: F);
    forward_setting!(BrentDerivative, with_relative_position_tolerance, value: F);
}

impl<CG, CX, CC, CT, V, M, F: Scalar>
    ConfiguredSolver<CmaEs<V, M, F>, CG, CX, CC, CT>
{
    forward_setting!(CmaEs, with_absolute_distribution_size_tolerance, value: impl Into<Option<F>>);
    forward_setting!(CmaEs, with_lambda, lambda: usize);
}

impl<CG, CX, CC, CT, V, M, F: Scalar>
    ConfiguredSolver<BoundedCmaEs<V, M, F>, CG, CX, CC, CT>
{
    forward_setting!(BoundedCmaEs, with_absolute_distribution_size_tolerance, value: impl Into<Option<F>>);
    forward_setting!(BoundedCmaEs, with_lambda, lambda: usize);
}

impl<CG, CX, CC, CT, L, V, F: Scalar>
    ConfiguredSolver<GradientDescent<L, V, F>, CG, CX, CC, CT>
{
    forward_setting!(GradientDescent, with_momentum, beta: F);
}

impl<CG, CX, CC, CT, F: Scalar> ConfiguredSolver<Lincoa<F>, CG, CX, CC, CT> {
    forward_setting!(Lincoa, with_absolute_radius_tolerance, value: impl Into<Option<F>>);
    forward_setting!(Lincoa, with_initial_radius, rho_beg: F);
    forward_setting!(Lincoa, with_final_radius, rho_end: F);
    forward_setting!(Lincoa, with_npt, npt: usize);
}

impl<CG, CX, CC, CT, I, V, F>
    ConfiguredSolver<
        BasinHopping<I, V, F, RandomDisplacement<F>, Metropolis<F>>,
        CG,
        CX,
        CC,
        CT,
    >
where
    F: Scalar,
    I: WarmStart<V>,
    <I as InitialState<V>>::State: State + CountsMirror,
{
    forward_setting!(BasinHopping, with_stepsize, stepsize: F);
    forward_setting!(BasinHopping, with_temperature, temperature: F);
}

impl<CG, CX, CC, CT, I, V, F, S, A>
    ConfiguredSolver<BasinHopping<I, V, F, S, A>, CG, CX, CC, CT>
where
    F: Scalar,
    I: WarmStart<V>,
    <I as InitialState<V>>::State: State + CountsMirror,
{
    /// Configure [`with_step_taker`](BasinHopping::with_step_taker) while retaining convergence settings.
    #[allow(clippy::type_complexity)] // Preserve the typed algorithm conversion.
    pub fn with_step_taker<S2>(
        self,
        step: S2,
    ) -> ConfiguredSolver<BasinHopping<I, V, F, S2, A>, CG, CX, CC, CT> {
        self.map_solver(|solver| solver.with_step_taker(step))
    }
    /// Configure [`with_acceptance_test`](BasinHopping::with_acceptance_test) while retaining convergence settings.
    #[allow(clippy::type_complexity)] // Preserve the typed algorithm conversion.
    pub fn with_acceptance_test<A2>(
        self,
        accept: A2,
    ) -> ConfiguredSolver<BasinHopping<I, V, F, S, A2>, CG, CX, CC, CT> {
        self.map_solver(|solver| solver.with_acceptance_test(accept))
    }
    forward_setting!(BasinHopping, with_adaptive, adaptive: bool);
    forward_setting!(BasinHopping, with_adaptive_interval, interval: u64);
    forward_setting!(BasinHopping, with_target_accept_rate, rate: F);
    forward_setting!(BasinHopping, with_stepwise_factor, factor: F);
    forward_setting!(BasinHopping, with_inner_max_iter, n: u64);
    /// Configure [`inner_stop_when_factory`](BasinHopping::inner_stop_when_factory) while retaining convergence settings.
    pub fn inner_stop_when_factory<Mk, CheckFn>(self, make: Mk) -> Self
    where
        Mk: FnMut() -> CheckFn + 'static,
        CheckFn: FnMut(&<I as InitialState<V>>::State) -> Option<TerminationReason>
            + 'static,
    {
        self.map_solver(|solver| solver.inner_stop_when_factory(make))
    }
}

impl<CG, CX, CC, CT, F: Scalar>
    ConfiguredSolver<GoldenSection<F>, CG, CX, CC, CT>
{
    forward_setting!(GoldenSection, with_absolute_position_tolerance, value: F);
    forward_setting!(GoldenSection, with_relative_position_tolerance, value: F);
}

impl<CG, CX, CC, CT, F: Scalar, R>
    ConfiguredSolver<GlobalBestPso<F, R>, CG, CX, CC, CT>
{
    forward_setting!(GlobalBestPso, with_swarm_size, swarm_size: usize);
    forward_setting!(GlobalBestPso, with_inertia, inertia: F);
    forward_setting!(GlobalBestPso, with_cognitive, cognitive: F);
    forward_setting!(GlobalBestPso, with_social, social: F);
    forward_setting!(GlobalBestPso, with_boundary_handling, boundary_handling: PsoBoundaryHandling<F>);
    forward_setting!(GlobalBestPso, with_velocity_limit, velocity_limit: PsoVelocityLimit<F>);
}

impl<CG, CX, CC, CT, I, V, M, F>
    ConfiguredSolver<BoundedCmaInject<I, V, M, F>, CG, CX, CC, CT>
where
    F: Scalar,
    I: MemeticInner<V, F>,
    I::State: CountsMirror,
{
    forward_setting!(BoundedCmaInject, with_k, k: usize);
    forward_setting!(BoundedCmaInject, with_c_y, c_y: F);
    forward_setting!(BoundedCmaInject, with_inner_max_iter, n: u64);
    forward_setting!(BoundedCmaInject, with_absolute_distribution_size_tolerance, value: impl Into<Option<F>>);
    /// Configure [`inner_stop_when_factory`](BoundedCmaInject::inner_stop_when_factory) while retaining convergence settings.
    pub fn inner_stop_when_factory<Mk, CheckFn>(self, make: Mk) -> Self
    where
        Mk: FnMut() -> CheckFn + 'static,
        CheckFn: FnMut(&I::State) -> Option<TerminationReason> + 'static,
    {
        self.map_solver(|solver| solver.inner_stop_when_factory(make))
    }
}

impl<CG, CX, CC, CT, F: Scalar> ConfiguredSolver<Cobyla<F>, CG, CX, CC, CT> {
    forward_setting!(Cobyla, with_absolute_radius_tolerance, value: impl Into<Option<F>>);
    forward_setting!(Cobyla, with_initial_radius, rho_beg: F);
    forward_setting!(Cobyla, with_final_radius, rho_end: F);
}

impl<CG, CX, CC, CT, So, V, F: Scalar>
    ConfiguredSolver<AugmentedLagrangianMethod<So, V, F>, CG, CX, CC, CT>
{
    forward_setting!(AugmentedLagrangianMethod, rho0, rho0: F);
    forward_setting!(AugmentedLagrangianMethod, with_rho_increase, rho_increase: F);
    forward_setting!(AugmentedLagrangianMethod, with_feasibility_decrease, feasibility_decrease: F);
    forward_setting!(AugmentedLagrangianMethod, with_absolute_feasibility_tolerance, tol: impl Into<Option<F>>);
    forward_setting!(AugmentedLagrangianMethod, with_inner_max_iter, inner_max_iter: u64);
}

impl<CG, CX, CC, CT, V, M, F: Scalar>
    ConfiguredSolver<LevenbergMarquardt<V, M, F>, CG, CX, CC, CT>
{
    forward_setting!(LevenbergMarquardt, with_absolute_gradient_tolerance, value: impl Into<Option<F>>);
    forward_setting!(LevenbergMarquardt, with_gradient_orthogonality_tolerance, value: impl Into<Option<F>>);
    forward_setting!(LevenbergMarquardt, with_relative_model_reduction_tolerance, value: impl Into<Option<F>>);
    forward_setting!(LevenbergMarquardt, with_relative_step_tolerance, value: impl Into<Option<F>>);
    forward_setting!(LevenbergMarquardt, with_tau, tau: F);
    forward_setting!(LevenbergMarquardt, with_max_inner_attempts, n: u32);
    forward_setting!(LevenbergMarquardt, with_no_progress_check, enabled: bool);
}

impl<CG, CX, CC, CT, V, M, F: Scalar>
    ConfiguredSolver<LevenbergMarquardt<V, M, F>, CG, CX, CC, CT>
{
    /// Configure [`with_pivoted_qr`](LevenbergMarquardt::with_pivoted_qr) while retaining convergence settings.
    pub fn with_pivoted_qr(
        self,
    ) -> ConfiguredSolver<LevenbergMarquardtQr<V, M, F>, CG, CX, CC, CT>
    where
        M: FactorizePivotedQr<V, F>,
    {
        self.map_solver(|solver| solver.with_pivoted_qr())
    }
}

impl<CG, CX, CC, CT, V, M, F: Scalar>
    ConfiguredSolver<LevenbergMarquardtQr<V, M, F>, CG, CX, CC, CT>
where
    M: FactorizePivotedQr<V, F>,
{
    forward_setting!(LevenbergMarquardtQr, with_relative_rank_tolerance, tol: F);
    forward_setting!(LevenbergMarquardtQr, with_absolute_gradient_tolerance, value: impl Into<Option<F>>);
    forward_setting!(LevenbergMarquardtQr, with_gradient_orthogonality_tolerance, value: impl Into<Option<F>>);
    forward_setting!(LevenbergMarquardtQr, with_relative_model_reduction_tolerance, value: impl Into<Option<F>>);
    forward_setting!(LevenbergMarquardtQr, with_relative_step_tolerance, value: impl Into<Option<F>>);
    forward_setting!(LevenbergMarquardtQr, with_tau, value: F);
    forward_setting!(LevenbergMarquardtQr, with_max_inner_attempts, value: u32);
    forward_setting!(LevenbergMarquardtQr, with_no_progress_check, enabled: bool);
}

impl<CG, CX, CC, CT, V, M, F: Scalar>
    ConfiguredSolver<GaussNewton<V, M, F>, CG, CX, CC, CT>
{
    forward_setting!(GaussNewton, with_absolute_gradient_tolerance, value: impl Into<Option<F>>);
}

impl<CG, CX, CC, CT, F: Scalar> ConfiguredSolver<Ssga<F>, CG, CX, CC, CT> {
    forward_setting!(Ssga, with_pop_size, pop_size: usize);
    forward_setting!(Ssga, with_blx_alpha, alpha: F);
    forward_setting!(Ssga, with_nam_pool, pool: usize);
    forward_setting!(Ssga, with_mutation_prob, p: f64);
    forward_setting!(Ssga, with_bga_range_fraction, f: F);
    forward_setting!(Ssga, with_offspring_per_step, n: usize);
}

impl<CG, CX, CC, CT, Sub, F: Scalar, Mode>
    ConfiguredSolver<TrustRegion<Sub, F, Mode>, CG, CX, CC, CT>
{
    forward_setting!(TrustRegion, with_radius, radius: F);
    forward_setting!(TrustRegion, with_max_radius, max_radius: F);
    forward_setting!(TrustRegion, with_eta, eta: F);
    forward_setting!(TrustRegion, with_max_inner_attempts, n: u32);
}

impl<CG, CX, CC, CT, I, V, M, F>
    ConfiguredSolver<CmaInject<I, V, M, F>, CG, CX, CC, CT>
where
    F: Scalar,
    I: MemeticInner<V, F>,
    I::State: CountsMirror,
{
    forward_setting!(CmaInject, with_k, k: usize);
    forward_setting!(CmaInject, with_c_y, c_y: F);
    forward_setting!(CmaInject, with_inner_max_iter, n: u64);
    forward_setting!(CmaInject, with_absolute_distribution_size_tolerance, value: impl Into<Option<F>>);
    /// Configure [`inner_stop_when_factory`](CmaInject::inner_stop_when_factory) while retaining convergence settings.
    pub fn inner_stop_when_factory<Mk, CheckFn>(self, make: Mk) -> Self
    where
        Mk: FnMut() -> CheckFn + 'static,
        CheckFn: FnMut(&I::State) -> Option<TerminationReason> + 'static,
    {
        self.map_solver(|solver| solver.inner_stop_when_factory(make))
    }
}

impl<CG, CX, CC, CT, V, F: Scalar> ConfiguredSolver<Sgd<V, F>, CG, CX, CC, CT> {
    forward_setting!(Sgd, with_momentum, beta: F);
    forward_setting!(Sgd, with_cost_eval_every, period: usize);
}
