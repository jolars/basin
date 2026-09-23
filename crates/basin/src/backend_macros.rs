//! Instantiate adapters independently for each enabled backend version.
//!
//! Bodies live in a child module, so the caller imports its parent scope to
//! preserve the shared implementations' `super::` paths.

#[cfg(feature = "nalgebra_v0_32")]
#[allow(unused_imports)]
pub(crate) mod nalgebra_0_32 {
    pub(crate) use ::nalgebra_0_32::*;
    pub(crate) use ::nalgebra_0_32::{
        ClosedAdd as ClosedAddAssign, ClosedDiv as ClosedDivAssign,
        ClosedMul as ClosedMulAssign, ClosedSub as ClosedSubAssign,
    };
}

#[cfg(feature = "nalgebra_all")]
macro_rules! nalgebra_version {
    ($version:ident, $backend:path, $sparse:path, $lapack:path, $feature:literal,
     $nalgebra:ident, $nalgebra_sparse:ident, $nalgebra_lapack:ident, $body:tt) => {
        mod $version {
            use $backend as $nalgebra;
            #[cfg(feature = $feature)]
            #[allow(unused_imports)]
            use $lapack as $nalgebra_lapack;
            #[allow(unused_imports)]
            use $sparse as $nalgebra_sparse;

            // These gates belong to this version, even when other versions
            // independently enable or disable acceleration.
            #[allow(unused_macros)]
            macro_rules! if_nalgebra_lapack {
                ($item:item) => {
                    #[cfg(feature = $feature)]
                    $item
                };
            }
            #[allow(unused_macros)]
            macro_rules! if_not_nalgebra_lapack {
                ($item:item) => {
                    #[cfg(not(feature = $feature))]
                    $item
                };
            }
            $crate::backend_macros::items!($body);
        }
    };
}

#[cfg(any(
    feature = "nalgebra_all",
    feature = "ndarray_all",
    feature = "faer_all"
))]
macro_rules! items {
    ({ $($items:item)* }) => { $($items)* };
}
#[cfg(any(
    feature = "nalgebra_all",
    feature = "ndarray_all",
    feature = "faer_all"
))]
pub(crate) use items;
#[cfg(feature = "nalgebra_all")]
pub(crate) use nalgebra_version;

#[cfg(feature = "nalgebra_all")]
macro_rules! nalgebra_versions {
    ($module:ident($nalgebra:ident, $nalgebra_sparse:ident, $nalgebra_lapack:ident); $($body:item)*) => {
        mod $module {
            use super::*;
            $crate::backend_macros::nalgebra_versions!($nalgebra, $nalgebra_sparse, $nalgebra_lapack, { $($body)* });
        }
    };
    ($nalgebra:ident, $nalgebra_sparse:ident, $nalgebra_lapack:ident, $body:tt) => {
        #[cfg(feature = "nalgebra_v0_32")]
        $crate::backend_macros::nalgebra_version!(
            v0_32,
            $crate::backend_macros::nalgebra_0_32,
            ::nalgebra_sparse_0_9,
            ::nalgebra_lapack_0_24,
            "nalgebra_v0_32-lapack",
            $nalgebra,
            $nalgebra_sparse,
            $nalgebra_lapack,
            $body
        );
        #[cfg(feature = "nalgebra_v0_33")]
        $crate::backend_macros::nalgebra_version!(
            v0_33,
            ::nalgebra_0_33,
            ::nalgebra_sparse_0_10,
            ::nalgebra_lapack_0_25,
            "nalgebra_v0_33-lapack",
            $nalgebra,
            $nalgebra_sparse,
            $nalgebra_lapack,
            $body
        );
        #[cfg(feature = "nalgebra_v0_34")]
        $crate::backend_macros::nalgebra_version!(
            v0_34,
            ::nalgebra_0_34,
            ::nalgebra_sparse_0_11,
            ::nalgebra_lapack_0_27,
            "nalgebra_v0_34-lapack",
            $nalgebra,
            $nalgebra_sparse,
            $nalgebra_lapack,
            $body
        );
        #[cfg(feature = "nalgebra_v0_35")]
        $crate::backend_macros::nalgebra_version!(
            v0_35,
            ::nalgebra,
            ::nalgebra_sparse,
            ::nalgebra_lapack,
            "nalgebra_v0_35-lapack",
            $nalgebra,
            $nalgebra_sparse,
            $nalgebra_lapack,
            $body
        );
    };
}
#[cfg(feature = "nalgebra_all")]
pub(crate) use nalgebra_versions;

#[cfg(feature = "ndarray_all")]
macro_rules! ndarray_versions {
    ($module:ident($ndarray:ident); $($body:item)*) => {
        mod $module {
            use super::*;
            $crate::backend_macros::ndarray_versions!($ndarray, { $($body)* });
        }
    };
    ($ndarray:ident, $body:tt) => {
        #[cfg(feature = "ndarray_v0_15")]
        mod v0_15 {
            use ::ndarray_0_15 as $ndarray;
            $crate::backend_macros::items!($body);
        }
        #[cfg(feature = "ndarray_v0_16")]
        mod v0_16 {
            use ::ndarray_0_16 as $ndarray;
            $crate::backend_macros::items!($body);
        }
        #[cfg(feature = "ndarray_v0_17")]
        mod v0_17 {
            use ::ndarray as $ndarray;
            $crate::backend_macros::items!($body);
        }
    };
}
#[cfg(feature = "ndarray_all")]
pub(crate) use ndarray_versions;

#[cfg(feature = "faer_all")]
macro_rules! faer_versions {
    ($module:ident($faer:ident, $faer_traits:ident); $($body:item)*) => {
        mod $module {
            use super::*;
            $crate::backend_macros::faer_versions!($faer, $faer_traits, { $($body)* });
        }
    };
    ($faer:ident, $faer_traits:ident, $body:tt) => {
        #[cfg(feature = "faer_v0_22")]
        mod v0_22 {
            use ::faer_0_22 as $faer;
            #[allow(unused_imports)]
            use ::faer_traits_0_22 as $faer_traits;
            $crate::backend_macros::items!($body);
        }
        #[cfg(feature = "faer_v0_23")]
        mod v0_23 {
            use ::faer_0_23 as $faer;
            #[allow(unused_imports)]
            use ::faer_traits_0_23 as $faer_traits;
            $crate::backend_macros::items!($body);
        }
        #[cfg(feature = "faer_v0_24")]
        mod v0_24 {
            use ::faer as $faer;
            #[allow(unused_imports)]
            use ::faer_traits as $faer_traits;
            $crate::backend_macros::items!($body);
        }
    };
}
#[cfg(feature = "faer_all")]
pub(crate) use faer_versions;
