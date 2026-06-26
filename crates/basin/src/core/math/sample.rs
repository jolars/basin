//! Component-wise uniform sample in a box, plus standard-normal sampling.
//!
//! Construction primitives for stochastic solvers that draw fresh
//! candidates. [`SampleUniformBox`] is the box-sampling op used by
//! [`RandomSearch`](crate::solver::RandomSearch); [`SampleStandardNormal`]
//! is the per-component `N(0, 1)` op used by CMA-ES (`y_k = B D z_k` with
//! `z_k ~ N(0, I)`, Hansen 2016 eq. 38).

use core::ops::Index;

use rand::Rng;

use super::{Scalar, VectorLen};

/// Per-component uniform sample in `[lower[i], upper[i]]`. Returns a
/// fresh value of the same shape as `lower`/`upper`.
///
/// # Contract
///
/// - **Caller must:** pass `lower` and `upper` of equal length, with
///   `lower[i] ≤ upper[i]` and both **finite** for every component.
///   Equal bounds are allowed (the corresponding component is pinned to
///   that value). Uniform sampling over an unbounded interval is
///   undefined, so a non-finite (`±∞`/`NaN`) bound is a contract
///   violation: implementors panic with a message naming the offending
///   coordinate (via the shared `assert_finite_box` helper), rather than
///   letting `rand`'s internal `NonFinite` rejection surface as an
///   opaque dependency unwrap.
/// - **Implementor must:** call
///   `Rng::random_range` once per component
///   on the inclusive range `lower[i]..=upper[i]`, and return a freshly
///   allocated value of length `lower.len()`. Sampling is component-
///   independent: the same `(rng, lower, upper)` produces the same
///   draws across backends only modulo iteration order.
/// - **Implementor must:** advance `rng` by exactly `n` `f64` draws so
///   that reproducibility is well-defined for fixed-seed RNGs.
///
/// Bound semantics match `rand::distr::Uniform::new_inclusive`:
/// `lower[i] == upper[i]` deterministically returns that value.
pub trait SampleUniformBox: Sized {
    /// Sample a fresh value with each component drawn uniformly from
    /// `[lower[i], upper[i]]`.
    fn sample_uniform_box<R: Rng + ?Sized>(lower: &Self, upper: &Self, rng: &mut R) -> Self;
}

/// Assert every component of `lower`/`upper` is finite: the
/// precondition every [`SampleUniformBox`] impl shares.
///
/// Panics naming the first offending coordinate, converting `rand`'s
/// opaque internal `NonFinite` unwrap (`random_range` on a `±∞` range)
/// into an actionable basin-level message. Every backend impl calls this
/// after its length-mismatch check, so the clear failure mode is uniform
/// across `De`/`DeInject`/`Ssga`/`RandomSearch`/`MaLsChCma` and
/// any future consumer that seeds uniformly in the box.
///
/// Assumes `lower.vec_len() == upper.vec_len()` (callers check the length
/// mismatch first).
pub(crate) fn assert_finite_box<V, F>(lower: &V, upper: &V)
where
    F: Scalar,
    V: VectorLen + Index<usize, Output = F>,
{
    for i in 0..lower.vec_len() {
        assert!(
            lower[i].is_finite() && upper[i].is_finite(),
            "sample_uniform_box requires finite bounds, got lower[{i}] = {:?}, upper[{i}] = {:?}",
            lower[i],
            upper[i]
        );
    }
}

/// Per-component standard-normal sample. Returns a fresh value of the
/// same shape as `template` with each component drawn independently from
/// `N(0, 1)`.
///
/// The vehicle for CMA-ES sampling: `z_k ~ N(0, I)` (Hansen 2016 eq. 38)
/// composed with `y_k = B D z_k` (eq. 39) gives `y_k ~ N(0, C)`.
///
/// # Contract
///
/// - **Caller must:** pass a `template` whose length equals the desired
///   sample length. The contents are not read; only the shape matters.
/// - **Implementor must:** sample each component independently from
///   `rand_distr::StandardNormal` and return a freshly allocated value
///   of the same length as `template`.
/// - **Implementor must:** advance `rng` by exactly the number of `f64`
///   draws that `rand_distr::StandardNormal` performs per component, so
///   that reproducibility is well-defined for fixed-seed RNGs across
///   backends modulo iteration order.
pub trait SampleStandardNormal: Sized {
    /// Sample a fresh value with each component drawn from `N(0, 1)`.
    /// `template` provides the shape (`len`); its values are unused.
    fn sample_standard_normal<R: Rng + ?Sized>(template: &Self, rng: &mut R) -> Self;
}
