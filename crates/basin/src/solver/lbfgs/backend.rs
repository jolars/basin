//! Backend-agnostic slice views over the parameter vector.
//!
//! L-BFGS-B's inner numerics (cauchy, subsm, formk, compact-form
//! helpers) all operate on `&[f64]` / `&mut [f64]`. To stay generic
//! over the user-chosen parameter backend (`Vec<f64>`, nalgebra
//! `DVector<f64>`, faer `Col<f64>`, ndarray `Array1<f64>`), the
//! top-level solver views each vector as a contiguous f64 slice at
//! iteration boundaries. The two
//! tiny traits below capture that view; impls are per-backend and
//! `pub(crate)` to keep them off the public API surface.
//!
//! The trait carries an `F = f64` default so the unbounded L-BFGS
//! solver path can run with `F = f32` while the L-BFGS-B bounded path
//! (cauchy / subsm / formk and friends — still hard-coded to `&[f64]`)
//! resolves through the default unchanged.

/// Read-only view of a parameter vector as a contiguous `&[F]`.
pub(crate) trait AsFloatSlice<F = f64> {
    /// Borrow the underlying storage as a contiguous slice.
    fn as_float_slice(&self) -> &[F];
}

/// Mutable companion to [`AsFloatSlice`].
pub(crate) trait AsFloatSliceMut<F = f64>: AsFloatSlice<F> {
    /// Borrow the underlying storage as a contiguous mutable slice.
    fn as_float_slice_mut(&mut self) -> &mut [F];
}

impl<F> AsFloatSlice<F> for Vec<F> {
    fn as_float_slice(&self) -> &[F] {
        self.as_slice()
    }
}
impl<F> AsFloatSliceMut<F> for Vec<F> {
    fn as_float_slice_mut(&mut self) -> &mut [F] {
        self.as_mut_slice()
    }
}

#[cfg(feature = "nalgebra")]
impl<F: nalgebra::Scalar> AsFloatSlice<F> for nalgebra::DVector<F> {
    fn as_float_slice(&self) -> &[F] {
        self.as_slice()
    }
}
#[cfg(feature = "nalgebra")]
impl<F: nalgebra::Scalar> AsFloatSliceMut<F> for nalgebra::DVector<F> {
    fn as_float_slice_mut(&mut self) -> &mut [F] {
        self.as_mut_slice()
    }
}

#[cfg(feature = "faer")]
impl<F: faer_traits::ComplexField> AsFloatSlice<F> for faer::Col<F> {
    fn as_float_slice(&self) -> &[F] {
        self.try_as_col_major()
            .expect("faer::Col<F> backing storage must be col-major contiguous")
            .as_slice()
    }
}
#[cfg(feature = "faer")]
impl<F: faer_traits::ComplexField> AsFloatSliceMut<F> for faer::Col<F> {
    fn as_float_slice_mut(&mut self) -> &mut [F] {
        self.try_as_col_major_mut()
            .expect("faer::Col<F> backing storage must be col-major contiguous")
            .as_slice_mut()
    }
}

#[cfg(feature = "ndarray")]
impl<F> AsFloatSlice<F> for ndarray::Array1<F> {
    fn as_float_slice(&self) -> &[F] {
        self.as_slice().expect("Array1 is contiguous")
    }
}
#[cfg(feature = "ndarray")]
impl<F> AsFloatSliceMut<F> for ndarray::Array1<F> {
    fn as_float_slice_mut(&mut self) -> &mut [F] {
        self.as_slice_mut().expect("Array1 is contiguous")
    }
}
