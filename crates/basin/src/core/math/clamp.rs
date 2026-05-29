//! Component-wise clamp into a box `[lower, upper]`.
//!
//! First-class projection primitive for box-constrained solvers
//! ([`ProjectedGradientDescent`](crate::solver::ProjectedGradientDescent);
//! shared with the projected-gradient termination metric
//! [`ProjectedGradientTolerance`](crate::core::termination::ProjectedGradientTolerance)).

/// In-place component-wise `self[i] ← clamp(self[i], lower[i], upper[i])`.
///
/// # Contract
///
/// - **Caller must:** ensure `lower[i] ≤ upper[i]` for every component
///   and that neither bound is `NaN`. Backend impls use
///   `x.max(lower).min(upper)` (the `Float` API lacks `clamp`), which
///   matches `f64::clamp`'s output on well-ordered, finite bounds but
///   silently mishandles `NaN` rather than panicking.
/// - **Caller must:** pass values of the same shape as `self`. Backend
///   impls assert this and panic on mismatch.
/// - **Implementor must:** mutate `self` in place with no allocation.
pub trait ClampInPlace {
    /// Clamp every component of `self` into `[lower, upper]` element-wise.
    fn clamp_in_place(&mut self, lower: &Self, upper: &Self);
}
