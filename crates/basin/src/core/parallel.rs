//! Internal parallelism plumbing for the opt-in `parallel` feature.
//!
//! Doc-hidden: this is crate-internal infrastructure, not public API. It lets
//! a loop over *independent* evaluations (finite-difference probes and
//! per-generation population fitness) fan across a `rayon` thread pool when the
//! `parallel` feature is on, preserving numerical results and their order.
//!
//! # `MaybeSync`/`MaybeSend`
//!
//! Auto-trait *shims*. With `parallel` on they alias [`Sync`]/[`Send`]; with
//! it off they are blanket-implemented for every type, so a bound like
//! `P: MaybeSync` imposes nothing. This lets one set of function signatures
//! carry the thread-safety bounds the `rayon` path needs without forcing them
//! (and breaking the wasm and no-threads build) on the sequential path.
//!
//! # `try_map_range_with`/`try_map_slice_with`
//!
//! Fallible indexed maps that collect `Result`s in input order. Each task gets
//! its own reusable scratch value via an `init` closure (a per-job probe
//! buffer in the `rayon` case, a single buffer in the sequential case), so a
//! body that mutates-then-resets one coordinate of a cloned vector reuses that
//! buffer instead of cloning per index. Because the output is collected in
//! order and every index is independent, the result is bit-identical whether or
//! not `parallel` is enabled.
//!
//! Finite-difference range maps use an ordinary loop with one scratch value
//! when the current pool has only one worker. Multi-worker maps retain Rayon's
//! work stealing; `map_init` may initialize scratch more than once per worker.

/// `Sync` when `parallel` is on, an unconstrained marker otherwise. See the
/// [module docs](self).
#[cfg(feature = "parallel")]
pub trait MaybeSync: Sync {}
#[cfg(feature = "parallel")]
impl<T: Sync + ?Sized> MaybeSync for T {}

/// `Send` when `parallel` is on, an unconstrained marker otherwise. See the
/// [module docs](self).
#[cfg(feature = "parallel")]
pub trait MaybeSend: Send {}
#[cfg(feature = "parallel")]
impl<T: Send + ?Sized> MaybeSend for T {}

/// Unconstrained marker (the `parallel`-off counterpart). See the
/// [module docs](self).
#[cfg(not(feature = "parallel"))]
pub trait MaybeSync {}
#[cfg(not(feature = "parallel"))]
impl<T: ?Sized> MaybeSync for T {}

/// Unconstrained marker (the `parallel`-off counterpart). See the
/// [module docs](self).
#[cfg(not(feature = "parallel"))]
pub trait MaybeSend {}
#[cfg(not(feature = "parallel"))]
impl<T: ?Sized> MaybeSend for T {}

/// Map `f` over `0..n`, threading a per-task scratch value from `init`, and
/// collect the `Result`s in index order. `rayon`-parallel under `parallel`,
/// sequential otherwise; identical output either way. Short-circuits on the
/// first `Err`.
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

#[cfg(not(feature = "parallel"))]
pub(crate) fn try_map_range_with<S, T, E, FInit, F>(
    n: usize,
    init: FInit,
    f: F,
) -> Result<Vec<T>, E>
where
    FInit: FnOnce() -> S,
    F: Fn(&mut S, usize) -> Result<T, E>,
{
    let mut scratch = init();
    (0..n).map(|j| f(&mut scratch, j)).collect()
}

/// Map `f` over `items`, threading a per-task scratch value from `init`, and
/// collect the `Result`s in slice order. `rayon`-parallel under `parallel`,
/// sequential otherwise; identical output either way. Short-circuits on the
/// first `Err`.
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

#[cfg(not(feature = "parallel"))]
pub(crate) fn try_map_slice_with<I, S, T, E, FInit, F>(
    items: &[I],
    init: FInit,
    f: F,
) -> Result<Vec<T>, E>
where
    FInit: FnOnce() -> S,
    F: Fn(&mut S, &I) -> Result<T, E>,
{
    let mut scratch = init();
    items.iter().map(|it| f(&mut scratch, it)).collect()
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
