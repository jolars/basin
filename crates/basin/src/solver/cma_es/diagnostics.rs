//! Borrowed views of the live CMA distribution at solver boundaries.

use crate::core::math::Scalar;

/// Distribution diagnostics borrow solver-owned models without evaluating the problem.
///
/// Available after initialization. A completed generation updates these records;
/// an error can leave partially updated machinery, so it does not promise a
/// coherent checkpoint. Fresh initialization resets the distribution, while exact
/// solver-aware continuation retains it. The axes are square roots of covariance
/// eigenvalues, before multiplication by the overall step size. After a
/// covariance update, the solver floors each eigenvalue at `1e-30` before
/// taking its square root. Initialization retains the configured axes.
/// With [`CmaEs::with_eigen_update_period`](super::CmaEs::with_eigen_update_period),
/// the axes describe the last refreshed sampling distribution and can lag the
/// covariance, which still updates every generation.
/// The backend's symmetric eigensolver reads the covariance's lower triangle;
/// rank-one update rounding can make its upper triangle differ. Coordinate
/// standard deviations instead use the covariance diagonal. The two evolution
/// paths are algorithm state and need not equal either of those scales.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct DistributionRef<'a, V, M, F: Scalar = f64> {
    /// Live distribution mean, before bounded phenotype repair.
    pub mean: &'a V,
    /// Overall distribution step size.
    pub sigma: F,
    /// Configured principal-axis size threshold; `None` disables the TolX test.
    pub distribution_tolerance: Option<F>,
    /// Live covariance matrix in genotype coordinates.
    pub covariance: &'a M,
    /// Principal-axis standard deviations, including the post-update eigenvalue floor.
    pub principal_axis_stds: &'a V,
    /// Covariance adaptation path, in genotype coordinates.
    pub covariance_path: &'a V,
    /// Step-size adaptation path, in whitened coordinates.
    pub step_size_path: &'a V,
    /// Completed distribution generations since fresh initialization.
    pub generation: u64,
}
