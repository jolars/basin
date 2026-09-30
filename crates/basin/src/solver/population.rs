//! Initialization shared by solvers that sample a finite box.

use crate::core::math::{SampleUniformBox, Scalar, VectorLen};
use crate::core::rng::Rng;

pub(crate) fn prepare_population<V, F, R>(
    members: &mut Vec<V>,
    lower: &V,
    upper: &V,
    size: usize,
    rng: &mut R,
) where
    F: Scalar,
    V: SampleUniformBox
        + VectorLen
        + std::ops::Index<usize, Output = F>
        + std::ops::IndexMut<usize, Output = F>,
    R: Rng + ?Sized,
{
    let n = lower.vec_len();
    assert!(n > 0, "population search requires non-empty bounds");
    assert_eq!(
        upper.vec_len(),
        n,
        "population bounds must have equal lengths"
    );
    for j in 0..n {
        assert!(
            lower[j].is_finite()
                && upper[j].is_finite()
                && lower[j] <= upper[j],
            "population search requires finite ordered bounds"
        );
    }
    if members.is_empty() {
        members.extend(
            (0..size).map(|_| V::sample_uniform_box(lower, upper, rng)),
        );
    } else {
        assert_eq!(
            members.len(),
            size,
            "initial population must match the solver's configured size"
        );
        for member in members {
            assert_eq!(
                member.vec_len(),
                n,
                "initial population must match the bounds' dimension"
            );
            for j in 0..n {
                assert!(
                    member[j].is_finite(),
                    "initial population coordinates must be finite"
                );
                // Project seeds before invoking a possibly domain-limited objective.
                member[j] = member[j].max(lower[j]).min(upper[j]);
            }
        }
    }
}
