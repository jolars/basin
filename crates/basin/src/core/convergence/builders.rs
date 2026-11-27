use super::*;
use crate::Lbfgs;
use crate::core::run_control::HookMode;
use crate::solver::*;

macro_rules! cost_setters {
    ([$($generic:tt)*] $solver:ty, [$($bound:tt)*]) => {
        impl<$($generic)*> $solver where $($bound)* {
            /// Stop when the absolute change between consecutive observed costs is at most the tolerance.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Distinct enabled tests combine with OR.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting. Change tests need two observed iterates. Trust-region
            /// solvers skip rejected iterations and compare accepted iterates.
            pub fn with_absolute_cost_change_tolerance<Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, (), (), CostChecks<Q>, ()>
             { ConfiguredSolver::new(self).set_absolute_cost_change_tolerance(value) }
            /// Stop when the observed cost change is at most the tolerance times the previous absolute cost.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Distinct enabled tests combine with OR.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting. Change tests need two observed iterates. Trust-region
            /// solvers skip rejected iterations and compare accepted iterates.
            pub fn with_relative_cost_change_tolerance<Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, (), (), CostChecks<Q>, ()>
             { ConfiguredSolver::new(self).set_relative_cost_change_tolerance(value) }
        }
        impl<CG, CX, CC, CT, $($generic)*> ConfiguredSolver<$solver, CG, CX, CC, CT> where $($bound)* {
            /// Stop when the absolute change between consecutive observed costs is at most the tolerance.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Distinct enabled tests combine with OR.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting. Change tests need two observed iterates. Trust-region
            /// solvers skip rejected iterations and compare accepted iterates.
            pub fn with_absolute_cost_change_tolerance<Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, CG, CX, CostChecks<Q>, CT>
            where CC: Into<CostChecks<Q>> { self.set_absolute_cost_change_tolerance(value) }
            /// Stop when the observed cost change is at most the tolerance times the previous absolute cost.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Distinct enabled tests combine with OR.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting. Change tests need two observed iterates. Trust-region
            /// solvers skip rejected iterations and compare accepted iterates.
            pub fn with_relative_cost_change_tolerance<Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, CG, CX, CostChecks<Q>, CT>
            where CC: Into<CostChecks<Q>> { self.set_relative_cost_change_tolerance(value) }
        }
    };
}
macro_rules! step_setters {
    ([$($generic:tt)*] $solver:ty, [$($bound:tt)*]) => {
        impl<$($generic)*> $solver where $($bound)* {
            /// Stop when the Euclidean distance between consecutive observed iterates is at most the tolerance.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Distinct enabled tests combine with OR.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting. Change tests need two observed iterates. Trust-region
            /// solvers skip rejected iterations and compare accepted iterates.
            pub fn with_absolute_step_tolerance<Param, Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, (), StepChecks<Param,Q>, (), ()>
             { ConfiguredSolver::new(self).set_absolute_step_tolerance(value) }
            /// Stop when the Euclidean iterate change is at most the tolerance times the current iterate norm.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Distinct enabled tests combine with OR.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting. Change tests need two observed iterates. Trust-region
            /// solvers skip rejected iterations and compare accepted iterates.
            pub fn with_relative_step_tolerance<Param, Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, (), StepChecks<Param,Q>, (), ()>
             { ConfiguredSolver::new(self).set_relative_step_tolerance(value) }
        }
        impl<CG, CX, CC, CT, $($generic)*> ConfiguredSolver<$solver, CG, CX, CC, CT> where $($bound)* {
            /// Stop when the Euclidean distance between consecutive observed iterates is at most the tolerance.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Distinct enabled tests combine with OR.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting. Change tests need two observed iterates. Trust-region
            /// solvers skip rejected iterations and compare accepted iterates.
            pub fn with_absolute_step_tolerance<Param, Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, CG, StepChecks<Param,Q>, CC, CT>
            where CX: Into<StepChecks<Param,Q>> { self.set_absolute_step_tolerance(value) }
            /// Stop when the Euclidean iterate change is at most the tolerance times the current iterate norm.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Distinct enabled tests combine with OR.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting. Change tests need two observed iterates. Trust-region
            /// solvers skip rejected iterations and compare accepted iterates.
            pub fn with_relative_step_tolerance<Param, Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, CG, StepChecks<Param,Q>, CC, CT>
            where CX: Into<StepChecks<Param,Q>> { self.set_relative_step_tolerance(value) }
        }
    };
}
macro_rules! gradient_setters {
    ([$($generic:tt)*] $solver:ty, [$($bound:tt)*]) => {
        impl<$($generic)*> $solver where $($bound)* {
            /// Stop when the Euclidean gradient norm is at most the tolerance.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Distinct enabled tests combine with OR.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting. Change tests need two observed iterates.
            pub fn with_absolute_gradient_tolerance<Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, GradientChecks<Q>, (), (), ()>
             { ConfiguredSolver::new(self).set_absolute_gradient_tolerance(value) }
            /// Stop when the Euclidean gradient norm is at most the tolerance times its initial norm.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Distinct enabled tests combine with OR.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting. Change tests need two observed iterates.
            pub fn with_relative_gradient_tolerance<Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, GradientChecks<Q>, (), (), ()>
             { ConfiguredSolver::new(self).set_relative_gradient_tolerance(value) }
        }
        impl<CG, CX, CC, CT, $($generic)*> ConfiguredSolver<$solver, CG, CX, CC, CT> where $($bound)* {
            /// Stop when the Euclidean gradient norm is at most the tolerance.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Distinct enabled tests combine with OR.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting. Change tests need two observed iterates.
            pub fn with_absolute_gradient_tolerance<Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, GradientChecks<Q>, CX, CC, CT>
            where CG: Into<GradientChecks<Q>> { self.set_absolute_gradient_tolerance(value) }
            /// Stop when the Euclidean gradient norm is at most the tolerance times its initial norm.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Distinct enabled tests combine with OR.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting. Change tests need two observed iterates.
            pub fn with_relative_gradient_tolerance<Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, GradientChecks<Q>, CX, CC, CT>
            where CG: Into<GradientChecks<Q>> { self.set_relative_gradient_tolerance(value) }
        }
    };
}
macro_rules! projected_setters {
    ([$($generic:tt)*] $solver:ty, [$($bound:tt)*]) => {
        impl<$($generic)*> $solver where $($bound)* {
            /// Stop when the infinity norm of `x - project(x - gradient)` is at most the tolerance; use the current problem bounds.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Distinct enabled tests combine with OR.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting. Change tests need two observed iterates.
            pub fn with_absolute_projected_gradient_tolerance<Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, ProjectedGradientCheck<Q>, (), (), ()>
             { ConfiguredSolver::new(self).set_absolute_projected_gradient_tolerance(value) }
        }
        impl<CG, CX, CC, CT, $($generic)*> ConfiguredSolver<$solver, CG, CX, CC, CT> where $($bound)* {
            /// Stop when the infinity norm of `x - project(x - gradient)` is at most the tolerance; use the current problem bounds.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Distinct enabled tests combine with OR.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting. Change tests need two observed iterates.
            pub fn with_absolute_projected_gradient_tolerance<Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, ProjectedGradientCheck<Q>, CX, CC, CT>
            where CG: Into<ProjectedGradientCheck<Q>> { self.set_absolute_projected_gradient_tolerance(value) }
        }
    };
}
cost_setters!([L, V, F: Scalar] GradientDescent<L, V, F>, []);
step_setters!([L, V, F: Scalar] GradientDescent<L, V, F>, []);
cost_setters!([L, V, F: Scalar] NonlinearCg<L, V, F>, []);
step_setters!([L, V, F: Scalar] NonlinearCg<L, V, F>, []);
gradient_setters!([L, V, F: Scalar] NonlinearCg<L, V, F>, []);
cost_setters!([V, F: Scalar] Sgd<V,F>, []);
step_setters!([V, F: Scalar] Sgd<V,F>, []);
cost_setters!([V, L, M, I, F: Scalar] Bfgs<V,F,M,L,I>, [V: crate::DenseBackend<F>]);
step_setters!([V, L, M, I, F: Scalar] Bfgs<V,F,M,L,I>, [V: crate::DenseBackend<F>]);
cost_setters!([V,Mode,L,F:Scalar] Lbfgs<V,F,Mode,L>, []);
step_setters!([V,Mode,L,F:Scalar] Lbfgs<V,F,Mode,L>, []);
cost_setters!([L,F:Scalar] ProjectedGradientDescent<L,F>, []);
step_setters!([L,F:Scalar] ProjectedGradientDescent<L,F>, []);
cost_setters!([Sub,F:Scalar,Mode,W] TrustRegion<Sub,F,Mode,W>, []);
step_setters!([Sub,F:Scalar,Mode,W] TrustRegion<Sub,F,Mode,W>, []);
step_setters!([V,Mode,F:Scalar] NelderMead<V,F,Mode>, []);
cost_setters!([V,M,F:Scalar] GaussNewton<V,M,F>, []);
step_setters!([V,M,F:Scalar] GaussNewton<V,M,F>, []);
cost_setters!([V,M,F:Scalar] LevenbergMarquardt<V,M,F>, []);
cost_setters!([V,M,F:Scalar,R] LevenbergMarquardtQr<V,M,F,R>, [M: crate::FactorizePivotedQr<V,F,R>]);
cost_setters!([V,M,F:Scalar] BoundedLevenbergMarquardt<V,M,F>, []);
step_setters!([V,M,F:Scalar] BoundedLevenbergMarquardt<V,M,F>, []);
cost_setters!([F:Scalar,R,M] TrustRegionReflective<F,R,M>, []);
step_setters!([F:Scalar,R,M] TrustRegionReflective<F,R,M>, []);
cost_setters!([F:Scalar] Brent<F>, []);
step_setters!([F:Scalar] Brent<F>, []);
cost_setters!([F:Scalar] BrentDerivative<F>, []);
step_setters!([F:Scalar] BrentDerivative<F>, []);
cost_setters!([F:Scalar] GoldenSection<F>, []);
step_setters!([F:Scalar] GoldenSection<F>, []);
cost_setters!([V, F:Scalar] SolisWets<V,F>, []);
step_setters!([V, F:Scalar] SolisWets<V,F>, []);
cost_setters!([F:Scalar] Newuoa<F>, []);
step_setters!([F:Scalar] Newuoa<F>, []);
cost_setters!([Mode,F:Scalar] Bobyqa<Mode,F>, []);
step_setters!([Mode,F:Scalar] Bobyqa<Mode,F>, []);
cost_setters!([F:Scalar] Lincoa<F>, []);
step_setters!([F:Scalar] Lincoa<F>, []);
cost_setters!([V,F:Scalar] Cobyla<V,F>, []);
step_setters!([V,F:Scalar] Cobyla<V,F>, []);
cost_setters!([Mode,F:Scalar] Mads<Mode,F>, []);
step_setters!([Mode,F:Scalar] Mads<Mode,F>, []);
cost_setters!([V,F:Scalar] Gbnm<V,F>, []);
step_setters!([V,F:Scalar] Gbnm<V,F>, []);
cost_setters!([V,M,F:Scalar,Evaluation] CmaEs<V,M,F,Evaluation>, []);
step_setters!([V,M,F:Scalar,Evaluation] CmaEs<V,M,F,Evaluation>, []);
cost_setters!([V,M,F:Scalar] BoundedCmaEs<V,M,F>, []);
step_setters!([V,M,F:Scalar] BoundedCmaEs<V,M,F>, []);
cost_setters!([F:Scalar,Evaluation] De<F,Evaluation>, []);
step_setters!([F:Scalar,Evaluation] De<F,Evaluation>, []);
cost_setters!([F:Scalar] Ssga<F>, []);
step_setters!([F:Scalar] Ssga<F>, []);
cost_setters!([Evaluation] RandomSearch<Evaluation>, []);
step_setters!([Evaluation] RandomSearch<Evaluation>, []);
cost_setters!([N,F:Scalar,R] SimulatedAnnealing<N,F,R>, []);
step_setters!([N,F:Scalar,R] SimulatedAnnealing<N,F,R>, []);
cost_setters!([V,F:Scalar,R,Evaluation] GlobalBestPso<V,F,R,Evaluation>, []);
step_setters!([V,F:Scalar,R,Evaluation] GlobalBestPso<V,F,R,Evaluation>, []);
cost_setters!([So,F:Scalar] BarrierMethod<So,F>, []);
step_setters!([So,F:Scalar] BarrierMethod<So,F>, []);
cost_setters!([So,V,F:Scalar] AugmentedLagrangianMethod<So,V,F>, []);
step_setters!([So,V,F:Scalar] AugmentedLagrangianMethod<So,V,F>, []);
cost_setters!([V,LS,F:Scalar] MaLsCh<V,LS,F>, [LS: crate::ResumableInner<V,F>]);
step_setters!([V,LS,F:Scalar] MaLsCh<V,LS,F>, [LS: crate::ResumableInner<V,F>]);
cost_setters!([I,V,M,F:Scalar,Evaluation,H:HookMode] CmaInject<I,V,M,F,Evaluation,H>, [I: MemeticInner<V,F>]);
step_setters!([I,V,M,F:Scalar,Evaluation,H:HookMode] CmaInject<I,V,M,F,Evaluation,H>, [I: MemeticInner<V,F>]);
cost_setters!([I,V,M,F:Scalar,H:HookMode] BoundedCmaInject<I,V,M,F,H>, [I: MemeticInner<V,F>]);
step_setters!([I,V,M,F:Scalar,H:HookMode] BoundedCmaInject<I,V,M,F,H>, [I: MemeticInner<V,F>]);
cost_setters!([I,V,F:Scalar,Evaluation,H:HookMode] DeInject<I,V,F,Evaluation,H>, [I: MemeticInner<V,F>]);
step_setters!([I,V,F:Scalar,Evaluation,H:HookMode] DeInject<I,V,F,Evaluation,H>, [I: MemeticInner<V,F>]);
cost_setters!([V,I,T,A,F:Scalar,H:HookMode] BasinHopping<I,V,F,T,A,H>, [I: crate::WarmStart<V>]);
step_setters!([V,I,T,A,F:Scalar,H:HookMode] BasinHopping<I,V,F,T,A,H>, [I: crate::WarmStart<V>]);
gradient_setters!([L, V, F: Scalar] GradientDescent<L, V, F>, []);
gradient_setters!([V, L, M, I, F: Scalar] Bfgs<V,F,M,L,I>, [V: crate::DenseBackend<F>]);
gradient_setters!([V,L,F:Scalar] Lbfgs<V,F,crate::solver::lbfgs::Unbounded,L>, []);
gradient_setters!([Sub,F:Scalar,Mode,W] TrustRegion<Sub,F,Mode,W>, []);
gradient_setters!([F:Scalar] BrentDerivative<F>, []);
projected_setters!([L,F:Scalar] ProjectedGradientDescent<L,F>, []);

macro_rules! absolute_step_setters {
    ([$($generic:tt)*] $solver:ty, [$($bound:tt)*]) => {
        impl<$($generic)*> $solver where $($bound)* {
            /// Stop when the Euclidean distance between consecutive observed iterates is at most the tolerance.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Distinct enabled tests combine with OR.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting. Change tests need two observed iterates. Trust-region
            /// solvers skip rejected iterations and compare accepted iterates.
            pub fn with_absolute_step_tolerance<Param, Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, (), StepChecks<Param,Q>, (), ()>
             { ConfiguredSolver::new(self).set_absolute_step_tolerance(value) }
        }
        impl<CG, CX, CC, CT, $($generic)*> ConfiguredSolver<$solver, CG, CX, CC, CT> where $($bound)* {
            /// Stop when the Euclidean distance between consecutive observed iterates is at most the tolerance.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Distinct enabled tests combine with OR.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting. Change tests need two observed iterates. Trust-region
            /// solvers skip rejected iterations and compare accepted iterates.
            pub fn with_absolute_step_tolerance<Param, Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, CG, StepChecks<Param,Q>, CC, CT>
            where CX: Into<StepChecks<Param,Q>> { self.set_absolute_step_tolerance(value) }
        }
    };
}

absolute_step_setters!([V,M,F:Scalar] LevenbergMarquardt<V,M,F>, []);
absolute_step_setters!([V,M,F:Scalar,R] LevenbergMarquardtQr<V,M,F,R>, [M: crate::FactorizePivotedQr<V,F,R>]);
