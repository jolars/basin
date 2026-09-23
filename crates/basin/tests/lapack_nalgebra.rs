//! LAPACK-backed nalgebra factorizations (the versioned `*-lapack` features).
//!
//! These exercise the LAPACK-selected impls in `core::math::nalgebra_backend`
//! (LAPACK Cholesky for `LinearSolveSpd` and `dsyev` for `SymmetricEigen`),
//! mirroring the pure-Rust unit tests in that module so both code paths are
//! checked against the same expectations.
//!
//! Running this test *links* LAPACK, so it needs a provider supplied at link
//! time. CI installs a system OpenBLAS and links it via `RUSTFLAGS`; see
//! `.github/workflows/ci.yml`.
//! Locally (devenv exposes an LP64 OpenBLAS as `$OPENBLAS_LP64_LIB`):
//!
//! ```sh
//! RUSTFLAGS="-L $OPENBLAS_LP64_LIB -l openblas" \
//!   cargo test -p basin --features nalgebra_latest-lapack --test lapack_nalgebra
//! ```
#![cfg(all(
    feature = "nalgebra_all",
    any(
        feature = "nalgebra_v0_32-lapack",
        feature = "nalgebra_v0_33-lapack",
        feature = "nalgebra_v0_34-lapack",
        feature = "nalgebra_v0_35-lapack"
    )
))]

#[cfg(feature = "nalgebra_v0_32")]
mod nalgebra_0_32 {
    use ::nalgebra_0_32 as backend;
    include!("support/nalgebra_factorizations.rs");
}

#[cfg(feature = "nalgebra_v0_33")]
mod nalgebra_0_33 {
    use ::nalgebra_0_33 as backend;
    include!("support/nalgebra_factorizations.rs");
}

#[cfg(feature = "nalgebra_v0_34")]
mod nalgebra_0_34 {
    use ::nalgebra_0_34 as backend;
    include!("support/nalgebra_factorizations.rs");
}

#[cfg(feature = "nalgebra_v0_35")]
mod nalgebra_0_35 {
    use ::nalgebra as backend;
    include!("support/nalgebra_factorizations.rs");
}
