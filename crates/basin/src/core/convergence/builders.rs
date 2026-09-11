use super::*;
use crate::Lbfgs;
use crate::solver::*;

macro_rules! cost_setters {
    ([$($generic:tt)*] $solver:ty, [$($bound:tt)*]) => {
        impl<$($generic)*> $solver where $($bound)* {
            /// Stop when the absolute change between consecutive observed costs is at most the tolerance.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Distinct enabled tests combine with OR.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting. Change tests need two observed iterates.
            pub fn with_absolute_cost_change_tolerance<Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, (), (), CostChecks<Q>, ()>
             { ConfiguredSolver::new(self).set_absolute_cost_change_tolerance(value) }
            /// Stop when the observed cost change is at most the tolerance times the previous absolute cost.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Distinct enabled tests combine with OR.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting. Change tests need two observed iterates.
            pub fn with_relative_cost_change_tolerance<Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, (), (), CostChecks<Q>, ()>
             { ConfiguredSolver::new(self).set_relative_cost_change_tolerance(value) }
        }
        impl<CG, CX, CC, CT, $($generic)*> ConfiguredSolver<$solver, CG, CX, CC, CT> where $($bound)* {
            /// Stop when the absolute change between consecutive observed costs is at most the tolerance.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Distinct enabled tests combine with OR.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting. Change tests need two observed iterates.
            pub fn with_absolute_cost_change_tolerance<Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, CG, CX, CostChecks<Q>, CT>
            where CC: Into<CostChecks<Q>> { self.set_absolute_cost_change_tolerance(value) }
            /// Stop when the observed cost change is at most the tolerance times the previous absolute cost.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Distinct enabled tests combine with OR.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting. Change tests need two observed iterates.
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
            /// this setting. Change tests need two observed iterates.
            pub fn with_absolute_step_tolerance<Param, Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, (), StepChecks<Param,Q>, (), ()>
             { ConfiguredSolver::new(self).set_absolute_step_tolerance(value) }
            /// Stop when the Euclidean iterate change is at most the tolerance times the current iterate norm.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Distinct enabled tests combine with OR.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting. Change tests need two observed iterates.
            pub fn with_relative_step_tolerance<Param, Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, (), StepChecks<Param,Q>, (), ()>
             { ConfiguredSolver::new(self).set_relative_step_tolerance(value) }
        }
        impl<CG, CX, CC, CT, $($generic)*> ConfiguredSolver<$solver, CG, CX, CC, CT> where $($bound)* {
            /// Stop when the Euclidean distance between consecutive observed iterates is at most the tolerance.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Distinct enabled tests combine with OR.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting. Change tests need two observed iterates.
            pub fn with_absolute_step_tolerance<Param, Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, CG, StepChecks<Param,Q>, CC, CT>
            where CX: Into<StepChecks<Param,Q>> { self.set_absolute_step_tolerance(value) }
            /// Stop when the Euclidean iterate change is at most the tolerance times the current iterate norm.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Distinct enabled tests combine with OR.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting. Change tests need two observed iterates.
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
macro_rules! simplex_setters {
    ([$($generic:tt)*] $solver:ty, [$($bound:tt)*]) => {
        impl<$($generic)*> $solver where $($bound)* {
            /// Require the maximum infinity-norm distance from the best simplex vertex to be at most the tolerance.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Combines with the other enabled simplex condition using AND.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting.
            pub fn with_absolute_simplex_size_tolerance<Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, (), (), (), SimplexChecks<Q>>
             { ConfiguredSolver::new(self).set_absolute_simplex_size_tolerance(value) }
            /// Require the maximum absolute cost difference from the best simplex vertex to be at most the tolerance.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Combines with the other enabled simplex condition using AND.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting.
            pub fn with_absolute_simplex_cost_tolerance<Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, (), (), (), SimplexChecks<Q, ()>>
             { ConfiguredSolver::new(self).set_absolute_simplex_cost_tolerance(value) }
        }
        impl<CG, CX, CC, CT, $($generic)*> ConfiguredSolver<$solver, CG, CX, CC, CT> where $($bound)* {
            /// Require the maximum infinity-norm distance from the best simplex vertex to be at most the tolerance.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Combines with the other enabled simplex condition using AND.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting.
            pub fn with_absolute_simplex_size_tolerance<Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, CG, CX, CC, SimplexChecks<Q>>
            where CT: Into<SimplexChecks<Q>> { self.set_absolute_simplex_size_tolerance(value) }
        }
        impl<CG, CX, CC, $($generic)*> ConfiguredSolver<$solver, CG, CX, CC> where $($bound)* {
            /// Require the maximum absolute cost difference from the best simplex vertex to be at most the tolerance.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Combines with the other enabled simplex condition using AND.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting.
            pub fn with_absolute_simplex_cost_tolerance<Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, CG, CX, CC, SimplexChecks<Q, ()>>
             { self.set_absolute_simplex_cost_tolerance(value) }
        }
        impl<CG, CX, CC, Q: Scalar, Size, $($generic)*> ConfiguredSolver<$solver, CG, CX, CC, SimplexChecks<Q, Size>> where $($bound)* {
            /// Require the maximum absolute cost difference from the best simplex vertex to be at most the tolerance.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Combines with the other enabled simplex condition using AND.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting.
            pub fn with_absolute_simplex_cost_tolerance(self, value: impl Into<Option<Q>>) -> Self
             { self.set_absolute_simplex_cost_tolerance(value) }
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
cost_setters!([V, F: Scalar] Sgd<V,F>, []);
step_setters!([V, F: Scalar] Sgd<V,F>, []);
cost_setters!([L, F: Scalar] Bfgs<L,F>, []);
step_setters!([L, F: Scalar] Bfgs<L,F>, []);
cost_setters!([Mode,L,F:Scalar] Lbfgs<Mode,L,F>, []);
step_setters!([Mode,L,F:Scalar] Lbfgs<Mode,L,F>, []);
cost_setters!([L] ProjectedGradientDescent<L>, []);
step_setters!([L] ProjectedGradientDescent<L>, []);
cost_setters!([Sub,F:Scalar,Mode] TrustRegion<Sub,F,Mode>, []);
step_setters!([Sub,F:Scalar,Mode] TrustRegion<Sub,F,Mode>, []);
cost_setters!([Mode,F:Scalar] NelderMead<Mode,F>, []);
step_setters!([Mode,F:Scalar] NelderMead<Mode,F>, []);
cost_setters!([V,M,F:Scalar] GaussNewton<V,M,F>, []);
step_setters!([V,M,F:Scalar] GaussNewton<V,M,F>, []);
cost_setters!([V,M,F:Scalar] LevenbergMarquardt<V,M,F>, []);
cost_setters!([V,M,F:Scalar] LevenbergMarquardtQr<V,M,F>, [M: crate::FactorizePivotedQr<V,F>]);
cost_setters!([V,M,F:Scalar] Trf<V,M,F>, []);
step_setters!([V,M,F:Scalar] Trf<V,M,F>, []);
cost_setters!([F:Scalar] Brent<F>, []);
step_setters!([F:Scalar] Brent<F>, []);
cost_setters!([F:Scalar] BrentDerivative<F>, []);
step_setters!([F:Scalar] BrentDerivative<F>, []);
cost_setters!([F:Scalar] GoldenSection<F>, []);
step_setters!([F:Scalar] GoldenSection<F>, []);
cost_setters!([F:Scalar] SolisWets<F>, []);
step_setters!([F:Scalar] SolisWets<F>, []);
cost_setters!([F:Scalar] Newuoa<F>, []);
step_setters!([F:Scalar] Newuoa<F>, []);
cost_setters!([Mode,F:Scalar] Bobyqa<Mode,F>, []);
step_setters!([Mode,F:Scalar] Bobyqa<Mode,F>, []);
cost_setters!([F:Scalar] Lincoa<F>, []);
step_setters!([F:Scalar] Lincoa<F>, []);
cost_setters!([F:Scalar] Cobyla<F>, []);
step_setters!([F:Scalar] Cobyla<F>, []);
cost_setters!([Mode,F:Scalar] Mads<Mode,F>, []);
step_setters!([Mode,F:Scalar] Mads<Mode,F>, []);
cost_setters!([F:Scalar] Gbnm<F>, []);
step_setters!([F:Scalar] Gbnm<F>, []);
cost_setters!([V,M,F:Scalar] CmaEs<V,M,F>, []);
step_setters!([V,M,F:Scalar] CmaEs<V,M,F>, []);
cost_setters!([V,M,F:Scalar] BoundedCmaEs<V,M,F>, []);
step_setters!([V,M,F:Scalar] BoundedCmaEs<V,M,F>, []);
cost_setters!([F:Scalar] De<F>, []);
step_setters!([F:Scalar] De<F>, []);
cost_setters!([F:Scalar] Ssga<F>, []);
step_setters!([F:Scalar] Ssga<F>, []);
cost_setters!([] RandomSearch, []);
step_setters!([] RandomSearch, []);
cost_setters!([N,F:Scalar,R] SimulatedAnnealing<N,F,R>, []);
step_setters!([N,F:Scalar,R] SimulatedAnnealing<N,F,R>, []);
cost_setters!([F:Scalar,R] GlobalBestPso<F,R>, []);
step_setters!([F:Scalar,R] GlobalBestPso<F,R>, []);
cost_setters!([So,F:Scalar] BarrierMethod<So,F>, []);
step_setters!([So,F:Scalar] BarrierMethod<So,F>, []);
cost_setters!([So,V,F:Scalar] AugmentedLagrangianMethod<So,V,F>, []);
step_setters!([So,V,F:Scalar] AugmentedLagrangianMethod<So,V,F>, []);
cost_setters!([V,LS] MaLsCh<V,LS>, []);
step_setters!([V,LS] MaLsCh<V,LS>, []);
cost_setters!([I,V,M,F:Scalar] CmaInject<I,V,M,F>, [I: MemeticInner<V,F>]);
step_setters!([I,V,M,F:Scalar] CmaInject<I,V,M,F>, [I: MemeticInner<V,F>]);
cost_setters!([I,V,M,F:Scalar] BoundedCmaInject<I,V,M,F>, [I: MemeticInner<V,F>]);
step_setters!([I,V,M,F:Scalar] BoundedCmaInject<I,V,M,F>, [I: MemeticInner<V,F>]);
cost_setters!([I,V,F:Scalar] DeInject<I,V,F>, [I: MemeticInner<V,F>]);
step_setters!([I,V,F:Scalar] DeInject<I,V,F>, [I: MemeticInner<V,F>]);
cost_setters!([V,I,T,A,F:Scalar] BasinHopping<I,V,F,T,A>, [I: crate::WarmStart<V>]);
step_setters!([V,I,T,A,F:Scalar] BasinHopping<I,V,F,T,A>, [I: crate::WarmStart<V>]);
gradient_setters!([L, V, F: Scalar] GradientDescent<L, V, F>, []);
gradient_setters!([L, F: Scalar] Bfgs<L,F>, []);
gradient_setters!([L,F:Scalar] Lbfgs<crate::solver::lbfgs::Unbounded,L,F>, []);
gradient_setters!([Sub,F:Scalar,Mode] TrustRegion<Sub,F,Mode>, []);
gradient_setters!([F:Scalar] BrentDerivative<F>, []);
simplex_setters!([Mode,F:Scalar] NelderMead<Mode,F>, []);
projected_setters!([L] ProjectedGradientDescent<L>, []);

macro_rules! absolute_step_setters {
    ([$($generic:tt)*] $solver:ty, [$($bound:tt)*]) => {
        impl<$($generic)*> $solver where $($bound)* {
            /// Stop when the Euclidean distance between consecutive observed iterates is at most the tolerance.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Distinct enabled tests combine with OR.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting. Change tests need two observed iterates.
            pub fn with_absolute_step_tolerance<Param, Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, (), StepChecks<Param,Q>, (), ()>
             { ConfiguredSolver::new(self).set_absolute_step_tolerance(value) }
        }
        impl<CG, CX, CC, CT, $($generic)*> ConfiguredSolver<$solver, CG, CX, CC, CT> where $($bound)* {
            /// Stop when the Euclidean distance between consecutive observed iterates is at most the tolerance.
            ///
            /// Disabled by default. `None` disables; zero requests an exact-zero
            /// threshold. Values must be finite and nonnegative. Distinct enabled tests combine with OR.
            /// Evaluated at initialized iteration boundaries; repeated calls replace
            /// this setting. Change tests need two observed iterates.
            pub fn with_absolute_step_tolerance<Param, Q: Scalar>(self, value: impl Into<Option<Q>>) -> ConfiguredSolver<$solver, CG, StepChecks<Param,Q>, CC, CT>
            where CX: Into<StepChecks<Param,Q>> { self.set_absolute_step_tolerance(value) }
        }
    };
}

absolute_step_setters!([V,M,F:Scalar] LevenbergMarquardt<V,M,F>, []);
absolute_step_setters!([V,M,F:Scalar] LevenbergMarquardtQr<V,M,F>, [M: crate::FactorizePivotedQr<V,F>]);
