use std::sync::Arc;

use basin::{CostFunction, Gradient};
use basin_evaluation_local::{CostBatch, GradientPair, gradient};
use rayon::{ThreadPool, prelude::*};

/// An explicit threaded policy for shareable callbacks.
///
/// ```compile_fail
/// use basin_evaluation_local::{CostBatch, Local};
/// use basin_evaluation_threaded::Threaded;
/// fn requires_threaded_local(mode: &Threaded, problem: &Local) {
///     mode.costs(problem, &[vec![1.0]]).unwrap();
/// }
/// ```
#[derive(Clone)]
pub struct Threaded(Arc<ThreadPool>);

impl Threaded {
    pub fn new(pool: ThreadPool) -> Self {
        Self(Arc::new(pool))
    }
}

impl<P> CostBatch<P> for Threaded
where
    P: CostFunction + Sync,
    P::Param: Sync,
    P::Output: Send,
    P::Error: Send,
{
    fn costs(
        &self,
        problem: &P,
        points: &[P::Param],
    ) -> Result<Vec<P::Output>, P::Error> {
        self.0.install(|| {
            points.par_iter().map(|point| problem.cost(point)).collect()
        })
    }
}

impl<P> GradientPair<P> for Threaded
where
    P: Gradient<Param = Vec<f64>, Gradient = Vec<f64>> + Sync,
    P::Error: Send,
{
    fn gradients(
        &self,
        problem: &P,
        [first, second]: [&Vec<f64>; 2],
    ) -> Result<[Vec<f64>; 2], P::Error> {
        let (first, second) = self.0.install(|| {
            rayon::join(
                || gradient(problem, first),
                || gradient(problem, second),
            )
        });
        Ok([first?, second?])
    }
}

pub fn two_workers() -> Threaded {
    Threaded::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap(),
    )
}

/// Run a test on a two-worker pool.
///
/// ```compile_fail
/// use basin::{CmaEs, DenseMatrix, Executor, PopulationProgress};
/// use basin_evaluation_local::Local;
/// Executor::new(
///     Local::default(),
///     CmaEs::<Vec<f64>, DenseMatrix>::new(7, 0.25).with_parallel_evaluation(),
///     PopulationProgress::from_point(vec![2.0, 1.0]),
/// ).run().unwrap();
/// ```
///
/// Threaded finite differences require a shareable problem, even when the
/// callback error can be sent between threads.
///
/// ```compile_fail
/// use basin::{CostFunction, FiniteDiff, Gradient};
/// use std::{cell::Cell, convert::Infallible};
/// struct Local(Cell<f64>);
/// impl CostFunction for Local {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
///         Ok(self.0.get() * x[0].powi(2))
///     }
/// }
/// let fd = FiniteDiff::new(Local(Cell::new(1.0))).with_parallel_evaluation();
/// fd.gradient(&vec![1.0]).unwrap();
/// ```
///
/// Callback errors must be sendable, even when the problem is shareable.
///
/// ```compile_fail
/// use basin::{CostFunction, FiniteDiff, Hessian};
/// use std::rc::Rc;
/// struct LocalError;
/// impl CostFunction for LocalError {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = Rc<str>;
///     fn cost(&self, _: &Vec<f64>) -> Result<f64, Self::Error> {
///         Err(Rc::from("evaluation failed"))
///     }
/// }
/// FiniteDiff::new(LocalError).with_parallel_evaluation().hessian(&vec![1.0]);
/// ```
pub fn on_two_workers<T: Send>(run: impl FnOnce() -> T + Send) -> T {
    two_workers().0.install(run)
}
