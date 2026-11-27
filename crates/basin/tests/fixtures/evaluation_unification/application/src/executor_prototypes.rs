//! Compare two ways to store hooks in a configured executor. Both install the
//! saved hooks into Basin's existing executor on the thread that calls `run`.

use basin::{
    ApplicationStop, CheckpointSink, CountsMirror, Executor, InitialState,
    ObservationEvent, Observe, ObserverMode, OptimizationResult, Solver, State,
};

type LocalExecutor<P, S, So> = Executor<P, S, So, basin::LocalHooks>;

/// Erase hook types while keeping their `Send` guarantee. Use `into_local`
/// before registering a hook that captures local state.
///
/// ```compile_fail,E0277
/// use basin::{CostFunction, Gradient, GradientDescent, ObserverMode};
/// use basin_evaluation_application::executor_prototypes::ExplicitExecutor;
/// use std::rc::Rc;
///
/// fn configure<P>(problem: P)
/// where
///     P: Gradient<Gradient = Vec<f64>> + CostFunction<Param = Vec<f64>, Output = f64> + Send,
/// {
///     let label = Rc::new(String::from("progress"));
///     ExplicitExecutor::from_start(problem, GradientDescent::new(0.25), vec![2.0])
///         .observe_solver(move |_, _, _| println!("{label}"), ObserverMode::Always);
/// }
/// ```
pub type ExplicitExecutor<P, S, So> = Configured<P, S, So, SendHooks<P, S, So>>;

/// Keep each registered hook's concrete type so Rust can infer `Send` from the
/// captures. Each registration returns a different type.
///
/// ```compile_fail,E0277
/// use basin::{CostFunction, Gradient, GradientDescent, ObserverMode};
/// use basin_evaluation_application::executor_prototypes::InferredExecutor;
/// use std::rc::Rc;
///
/// fn assert_send<T: Send>(_: T) {}
/// fn configure<P>(problem: P)
/// where
///     P: Gradient<Gradient = Vec<f64>> + CostFunction<Param = Vec<f64>, Output = f64> + Send,
/// {
///     let label = Rc::new(String::from("progress"));
///     let run = InferredExecutor::from_start(problem, GradientDescent::new(0.25), vec![2.0])
///         .observe_solver(move |_, _, _| println!("{label}"), ObserverMode::Always);
///     assert_send(run);
/// }
/// ```
///
/// A runtime flag cannot change the hook types of a single binding.
///
/// ```compile_fail,E0308
/// use basin::{CostFunction, Gradient, GradientDescent, ObserverMode};
/// use basin_evaluation_application::executor_prototypes::InferredExecutor;
///
/// fn configure<P>(problem: P, log_progress: bool)
/// where
///     P: Gradient<Gradient = Vec<f64>> + CostFunction<Param = Vec<f64>, Output = f64>,
/// {
///     let mut run = InferredExecutor::from_start(problem, GradientDescent::new(0.25), vec![2.0]);
///     if log_progress {
///         run = run.observe_solver(|_, _, event| println!("{event:?}"), ObserverMode::Always);
///     }
///     let _ = run;
/// }
/// ```
pub type InferredExecutor<P, S, So> = Configured<P, S, So, InferredHooks<()>>;

pub struct Configured<P, S: State, So, H> {
    problem: P,
    state: S,
    solver: So,
    max_iter: Option<u64>,
    hooks: H,
}

impl<P, S: State, So, H: Default> Configured<P, S, So, H> {
    pub fn new(problem: P, solver: So, state: S) -> Self {
        Self {
            problem,
            state,
            solver,
            max_iter: None,
            hooks: H::default(),
        }
    }

    pub fn from_start<V>(problem: P, solver: So, initial: V) -> Self
    where
        So: InitialState<V, State = S>,
    {
        let state = solver.seed(&initial);
        Self::new(problem, solver, state)
    }
}

impl<P, S, So, H> Configured<P, S, So, H>
where
    S: State + CountsMirror,
    So: Solver<P, S>,
{
    pub fn max_iter(mut self, limit: u64) -> Self {
        self.max_iter = Some(limit);
        self
    }

    pub fn observe_with<O>(
        self,
        observer: O,
        mode: ObserverMode,
    ) -> Configured<P, S, So, H::Output>
    where
        O: Observe<S> + 'static,
        H: AddHook<StateObserver<O>>,
    {
        self.map_hooks(|hooks| hooks.add(StateObserver(observer, mode)))
    }

    pub fn observe_solver<C>(
        self,
        callback: C,
        mode: ObserverMode,
    ) -> Configured<P, S, So, H::Output>
    where
        C: FnMut(&S, &So, ObservationEvent<'_, S::Float>) + 'static,
        H: AddHook<SolverObserver<C>>,
    {
        self.map_hooks(|hooks| hooks.add(SolverObserver(callback, mode)))
    }

    pub fn stop_when<C>(self, check: C) -> Configured<P, S, So, H::Output>
    where
        C: FnMut(&S) -> Option<ApplicationStop> + 'static,
        H: AddHook<StopHook<C>>,
    {
        self.map_hooks(|hooks| hooks.add(StopHook(check)))
    }

    pub fn checkpoint_with<C>(
        self,
        sink: C,
        mode: ObserverMode,
    ) -> Configured<P, S, So, H::Output>
    where
        C: CheckpointSink<So, S> + 'static,
        H: AddHook<CheckpointHook<C>>,
    {
        self.map_hooks(|hooks| hooks.add(CheckpointHook(sink, mode)))
    }

    pub fn run(self) -> Result<OptimizationResult<S>, So::Error>
    where
        H: Install<P, S, So>,
    {
        let mut executor =
            Executor::new(self.problem, self.solver, self.state).into_local();
        if let Some(limit) = self.max_iter {
            executor = executor.max_iter(limit);
        }
        self.hooks.install(executor).run()
    }
}

impl<P, S: State, So, H> Configured<P, S, So, H> {
    fn map_hooks<R>(self, f: impl FnOnce(H) -> R) -> Configured<P, S, So, R> {
        Configured {
            problem: self.problem,
            state: self.state,
            solver: self.solver,
            max_iter: self.max_iter,
            hooks: f(self.hooks),
        }
    }
}

impl<P, S: State, So> ExplicitExecutor<P, S, So> {
    /// Allow local hooks while preserving any hooks already registered.
    ///
    /// ```compile_fail,E0277
    /// use basin::FirstOrderState;
    /// use basin_evaluation_application::executor_prototypes::ExplicitExecutor;
    ///
    /// fn assert_send<T: Send>(_: T) {}
    /// let run = ExplicitExecutor::new((), (), FirstOrderState::new(vec![2.0_f64]));
    /// assert_send(run.into_local());
    /// ```
    pub fn into_local(self) -> Configured<P, S, So, LocalHooks<P, S, So>> {
        self.map_hooks(|hooks| {
            LocalHooks(
                hooks
                    .0
                    .into_iter()
                    .map(|hook| -> LocalInstall<P, S, So> { hook })
                    .collect(),
            )
        })
    }
}

pub trait Install<P, S: State, So> {
    fn install(
        self,
        executor: LocalExecutor<P, S, So>,
    ) -> LocalExecutor<P, S, So>;
}

pub trait AddHook<T> {
    type Output;

    fn add(self, hook: T) -> Self::Output;
}

type SendInstall<P, S, So> =
    Box<dyn FnOnce(LocalExecutor<P, S, So>) -> LocalExecutor<P, S, So> + Send>;
type LocalInstall<P, S, So> =
    Box<dyn FnOnce(LocalExecutor<P, S, So>) -> LocalExecutor<P, S, So>>;

pub struct SendHooks<P, S: State, So>(Vec<SendInstall<P, S, So>>);
pub struct LocalHooks<P, S: State, So>(Vec<LocalInstall<P, S, So>>);

impl<P, S: State, So> Default for SendHooks<P, S, So> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<P, S: State, So, T: Install<P, S, So> + Send + 'static> AddHook<T>
    for SendHooks<P, S, So>
{
    type Output = Self;

    fn add(mut self, hook: T) -> Self {
        self.0
            .push(Box::new(move |executor| hook.install(executor)));
        self
    }
}

impl<P, S: State, So, T: Install<P, S, So> + 'static> AddHook<T>
    for LocalHooks<P, S, So>
{
    type Output = Self;

    fn add(mut self, hook: T) -> Self {
        self.0
            .push(Box::new(move |executor| hook.install(executor)));
        self
    }
}

impl<P, S: State, So> Install<P, S, So> for SendHooks<P, S, So> {
    fn install(
        self,
        executor: LocalExecutor<P, S, So>,
    ) -> LocalExecutor<P, S, So> {
        self.0
            .into_iter()
            .fold(executor, |executor, hook| hook(executor))
    }
}

impl<P, S: State, So> Install<P, S, So> for LocalHooks<P, S, So> {
    fn install(
        self,
        executor: LocalExecutor<P, S, So>,
    ) -> LocalExecutor<P, S, So> {
        self.0
            .into_iter()
            .fold(executor, |executor, hook| hook(executor))
    }
}

#[derive(Default)]
pub struct InferredHooks<H>(H);

impl<H, T> AddHook<T> for InferredHooks<H> {
    type Output = InferredHooks<(H, T)>;

    fn add(self, hook: T) -> Self::Output {
        InferredHooks((self.0, hook))
    }
}

impl<P, S: State, So, H: Install<P, S, So>> Install<P, S, So>
    for InferredHooks<H>
{
    fn install(
        self,
        executor: LocalExecutor<P, S, So>,
    ) -> LocalExecutor<P, S, So> {
        self.0.install(executor)
    }
}

impl<P, S: State, So> Install<P, S, So> for () {
    fn install(
        self,
        executor: LocalExecutor<P, S, So>,
    ) -> LocalExecutor<P, S, So> {
        executor
    }
}

impl<P, S: State, So, H: Install<P, S, So>, T: Install<P, S, So>>
    Install<P, S, So> for (H, T)
{
    fn install(
        self,
        executor: LocalExecutor<P, S, So>,
    ) -> LocalExecutor<P, S, So> {
        let (previous, hook) = self;
        hook.install(previous.install(executor))
    }
}

pub struct StateObserver<O>(O, ObserverMode);
pub struct SolverObserver<C>(C, ObserverMode);
pub struct StopHook<C>(C);
pub struct CheckpointHook<C>(C, ObserverMode);

impl<P, S, So, O> Install<P, S, So> for StateObserver<O>
where
    S: State + CountsMirror,
    So: Solver<P, S>,
    O: Observe<S> + 'static,
{
    fn install(
        self,
        executor: LocalExecutor<P, S, So>,
    ) -> LocalExecutor<P, S, So> {
        executor.observe_with(self.0, self.1)
    }
}

impl<P, S, So, C> Install<P, S, So> for SolverObserver<C>
where
    S: State + CountsMirror,
    So: Solver<P, S>,
    C: FnMut(&S, &So, ObservationEvent<'_, S::Float>) + 'static,
{
    fn install(
        self,
        executor: LocalExecutor<P, S, So>,
    ) -> LocalExecutor<P, S, So> {
        executor.observe_solver(self.0, self.1)
    }
}

impl<P, S, So, C> Install<P, S, So> for StopHook<C>
where
    S: State + CountsMirror,
    So: Solver<P, S>,
    C: FnMut(&S) -> Option<ApplicationStop> + 'static,
{
    fn install(
        self,
        executor: LocalExecutor<P, S, So>,
    ) -> LocalExecutor<P, S, So> {
        executor.stop_when(self.0)
    }
}

impl<P, S, So, C> Install<P, S, So> for CheckpointHook<C>
where
    S: State + CountsMirror,
    So: Solver<P, S>,
    C: CheckpointSink<So, S> + 'static,
{
    fn install(
        self,
        executor: LocalExecutor<P, S, So>,
    ) -> LocalExecutor<P, S, So> {
        executor.checkpoint_with(self.0, self.1)
    }
}
