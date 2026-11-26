//! Evaluation modes and indexed maps for independent callbacks.
//!
//! The Rayon maps collect successful results in input order. Each job reuses
//! its own scratch value. A range map with one worker uses an ordinary loop
//! and a single scratch value.

/// Map `f` over `0..n`, threading a per-task scratch value from `init`, and
/// collect successful values in index order on the active Rayon pool.
/// Returns a callback error if any evaluation fails.
#[cfg(feature = "parallel")]
pub(crate) fn try_map_range_with<S, T, E, FInit, F>(
    n: usize,
    init: FInit,
    f: F,
) -> Result<Vec<T>, E>
where
    T: Send,
    E: Send,
    FInit: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize) -> Result<T, E> + Sync + Send,
{
    use rayon::prelude::*;
    if rayon::current_num_threads() == 1 {
        let mut scratch = init();
        return (0..n).map(|j| f(&mut scratch, j)).collect();
    }
    (0..n).into_par_iter().map_init(init, f).collect()
}

/// Map `f` over `items`, threading a per-task scratch value from `init`, and
/// collect successful values in slice order on the active Rayon pool.
/// Returns a callback error if any evaluation fails.
#[cfg(feature = "parallel")]
pub(crate) fn try_map_slice_with<I, S, T, E, FInit, F>(
    items: &[I],
    init: FInit,
    f: F,
) -> Result<Vec<T>, E>
where
    I: Sync,
    T: Send,
    E: Send,
    FInit: Fn() -> S + Sync + Send,
    F: Fn(&mut S, &I) -> Result<T, E> + Sync + Send,
{
    use rayon::prelude::*;
    items.par_iter().map_init(init, f).collect()
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use std::{
        cell::RefCell,
        rc::Rc,
        sync::atomic::{AtomicUsize, Ordering::Relaxed},
    };

    use super::{try_map_range_with, try_map_slice_with};

    #[test]
    fn local_scratch_preserves_range_and_slice_order() {
        for workers in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            pool.install(|| {
                for n in [0, 1, 63, 1025] {
                    let initialized = AtomicUsize::new(0);
                    // Job-local scratch need not be Send. A scheduling change
                    // must not accidentally require that extra capability.
                    let init = || {
                        initialized.fetch_add(1, Relaxed);
                        Rc::new(RefCell::new(vec![0; 2]))
                    };
                    let evaluate = |scratch: &mut Rc<RefCell<Vec<usize>>>,
                                    j| {
                        let mut scratch = scratch.borrow_mut();
                        scratch[0] = j;
                        scratch[1] = 3 * j;
                        Ok::<_, ()>(scratch[1])
                    };
                    let result = try_map_range_with(n, init, evaluate).unwrap();
                    let expected: Vec<_> = (0..n).map(|j| 3 * j).collect();
                    assert_eq!(result, expected);
                    if workers == 1 {
                        assert_eq!(initialized.swap(0, Relaxed), 1);
                    }
                    let items: Vec<_> = (0..n).rev().collect();
                    let result =
                        try_map_slice_with(&items, init, |scratch, &j| {
                            evaluate(scratch, j)
                        })
                        .unwrap();
                    assert_eq!(
                        result,
                        expected.into_iter().rev().collect::<Vec<_>>()
                    );
                }
            });
        }
    }

    #[test]
    fn failed_jobs_propagate_the_original_error() {
        for workers in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            pool.install(|| {
                let calls = AtomicUsize::new(0);
                let init = || Rc::new(RefCell::new(0));
                let evaluate = |scratch: &mut Rc<RefCell<usize>>, j| {
                    calls.fetch_add(1, Relaxed);
                    *scratch.borrow_mut() = j;
                    if j == 17 { Err("probe failed") } else { Ok(j) }
                };
                assert_eq!(
                    try_map_range_with(1025, init, evaluate),
                    Err("probe failed")
                );
                if workers == 1 {
                    assert_eq!(calls.swap(0, Relaxed), 18);
                }
                let items: Vec<_> = (0..1025).collect();
                assert_eq!(
                    try_map_slice_with(&items, init, |scratch, &j| evaluate(
                        scratch, j
                    )),
                    Err("probe failed")
                );
            });
        }
    }
}

/// Evaluate callbacks on the calling thread.
#[derive(Clone, Copy, Debug, Default)]
pub struct SerialEvaluation;

/// Evaluate independent callbacks on the active Rayon pool.
#[cfg(feature = "parallel")]
#[derive(Clone, Copy, Debug, Default)]
pub struct ParallelEvaluation;

#[doc(hidden)]
pub trait CostBatch<P: crate::CostFunction> {
    fn evaluate(
        problem: &mut crate::Problem<P>,
        params: &[P::Param],
    ) -> Result<Vec<P::Output>, P::Error>;
}

impl<P: crate::CostFunction> CostBatch<P> for SerialEvaluation {
    fn evaluate(
        problem: &mut crate::Problem<P>,
        params: &[P::Param],
    ) -> Result<Vec<P::Output>, P::Error> {
        problem.cost_batch(params)
    }
}

#[cfg(feature = "parallel")]
impl<P> CostBatch<P> for ParallelEvaluation
where
    P: crate::CostFunction + Sync,
    P::Param: Sync,
    P::Output: Send,
    P::Error: Send,
{
    fn evaluate(
        problem: &mut crate::Problem<P>,
        params: &[P::Param],
    ) -> Result<Vec<P::Output>, P::Error> {
        problem.cost_batch_parallel(params)
    }
}
