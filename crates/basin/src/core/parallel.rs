//! Internal parallelism plumbing for the opt-in `parallel` feature.
//!
//! Doc-hidden: this is crate-internal infrastructure, not public API. It lets
//! a loop over *independent* evaluations (finite-difference probes and
//! per-generation population fitness) fan across a `rayon` thread pool when the
//! `parallel` feature is on, while staying allocation- and result-identical to
//! the sequential path when it is off.
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
//! its own reusable scratch value via an `init` closure (a per-thread probe
//! buffer in the `rayon` case, a single buffer in the sequential case), so a
//! body that mutates-then-resets one coordinate of a cloned vector reuses that
//! buffer instead of cloning per index. Because the output is collected in
//! order and every index is independent, the result is bit-identical whether or
//! not `parallel` is enabled.

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
