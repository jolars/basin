//! Standard optimization test problems.
//!
//! Each problem is exposed three ways:
//! - Raw functions on `&[f64]` slices (e.g.
//!   [`rosenbrock()`](crate::problems::rosenbrock::rosenbrock)) for callers that
//!   want to plug the math into their own wrappers, benchmarks, or visualizations.
//! - A pre-wrapped struct (e.g. [`Rosenbrock`](crate::problems::Rosenbrock))
//!   implementing
//!   [`CostFunction`](crate::CostFunction) and [`Gradient`](crate::Gradient)
//!   for each enabled backend, so it can be handed straight to an
//!   [`Executor`](crate::Executor).
//! - A static [`ProblemSpec`](crate::problems::ProblemSpec) with metadata
//!   (dimensionality, mathematical
//!   properties, references) for use in catalog UIs and filtering. Iterate
//!   [`ALL_SPECS`](crate::problems::ALL_SPECS) to enumerate the corpus.
//!
//! Most functions in this corpus are cataloged in Jamil & Yang (2013),
//! *A Literature Survey of Benchmark Functions For Global Optimisation
//! Problems*, arXiv:1308.4008. Per-problem references on each
//! [`ProblemSpec`](crate::problems::ProblemSpec) cite the original source where
//! applicable.
//!
//! Gated behind the `problems` feature (default-on). Disable with
//! `default-features = false` to drop the corpus from the build.

pub mod ackley;
pub mod beale;
pub mod booth;
pub mod bukin;
pub mod constrained_quadratic;
pub mod cross_in_tray;
pub mod easom;
pub mod eggholder;
pub mod equality_constrained_quadratic;
pub mod exponential_fit;
pub mod goldstein_price;
pub mod himmelblau;
pub mod holder_table;
pub mod levy;
pub mod matyas;
pub mod mccormick;
pub mod picheny;
pub mod powell_singular;
pub mod rastrigin;
pub mod rosenbrock;
pub mod schaffer;
pub mod sparse_least_squares;
pub mod spec;
pub mod sphere;
pub mod step;
pub mod styblinski_tang;
pub mod three_hump_camel;
pub mod zero;

pub use ackley::{ACKLEY_SPEC, Ackley, AckleyBoxed, ackley};
pub use beale::{BEALE_SPEC, Beale, beale, beale_gradient};
pub use booth::{
    BOOTH_SPEC, Booth, BoothBoxed, BoothBoxedResiduals, BoothResiduals, booth, booth_gradient,
    booth_residuals, booth_residuals_jacobian,
};
pub use bukin::{BUKIN_N6_SPEC, BukinN6, bukin_n6};
pub use constrained_quadratic::{CONSTRAINED_QUADRATIC_SPEC, ConstrainedQuadratic};
pub use cross_in_tray::{CROSS_IN_TRAY_SPEC, CrossInTray, cross_in_tray};
pub use easom::{EASOM_SPEC, Easom, easom, easom_gradient};
pub use eggholder::{EGGHOLDER_SPEC, Eggholder, eggholder};
pub use equality_constrained_quadratic::{
    EQUALITY_CONSTRAINED_QUADRATIC_SPEC, EqualityConstrainedQuadratic,
};
pub use exponential_fit::{
    EXPONENTIAL_FIT_SPEC, ExponentialFit, exponential_fit, exponential_fit_jacobian,
    exponential_fit_residuals,
};
pub use goldstein_price::{
    GOLDSTEIN_PRICE_SPEC, GoldsteinPrice, goldstein_price, goldstein_price_gradient,
};
pub use himmelblau::{HIMMELBLAU_SPEC, Himmelblau, himmelblau, himmelblau_gradient};
pub use holder_table::{HOLDER_TABLE_SPEC, HolderTable, holder_table};
pub use levy::{LEVY_SPEC, Levy, LevyBoxed, levy, levy_gradient};
pub use matyas::{MATYAS_SPEC, Matyas, matyas, matyas_gradient};
pub use mccormick::{MCCORMICK_SPEC, McCormick, mccormick, mccormick_gradient};
pub use picheny::{PICHENY_SPEC, Picheny, picheny, picheny_gradient};
pub use powell_singular::{
    POWELL_SINGULAR_SPEC, PowellSingular, powell_singular, powell_singular_jacobian,
    powell_singular_residuals,
};
pub use rastrigin::{RASTRIGIN_SPEC, Rastrigin, RastriginBoxed, rastrigin};
pub use rosenbrock::{
    ROSENBROCK_SPEC, Rosenbrock, RosenbrockResiduals, rosenbrock, rosenbrock_gradient,
    rosenbrock_residuals, rosenbrock_residuals_jacobian,
};
pub use schaffer::{
    SCHAFFER_N2_SPEC, SCHAFFER_N4_SPEC, SchafferN2, SchafferN4, schaffer_n2, schaffer_n2_gradient,
    schaffer_n4,
};
pub use sparse_least_squares::{
    SPARSE_LEAST_SQUARES_SPEC, SparseLeastSquares, SparseLeastSquaresBoxed,
};
pub use spec::{Dimensionality, HasSpec, ProblemSpec, Properties, Reference};
pub use sphere::{SPHERE_SPEC, Sphere, SphereBoxed, sphere, sphere_gradient};
pub use step::{STEP_SPEC, Step, step};
pub use styblinski_tang::{
    STYBLINSKI_TANG_SPEC, StyblinskiTang, StyblinskiTangBoxed, styblinski_tang,
    styblinski_tang_gradient,
};
pub use three_hump_camel::{
    THREE_HUMP_CAMEL_SPEC, ThreeHumpCamel, three_hump_camel, three_hump_camel_gradient,
};
pub use zero::{ZERO_SPEC, Zero, zero, zero_gradient};

/// All cataloged problem specs, for browsing and filtering. Append new
/// problems here as they're added.
pub static ALL_SPECS: &[&ProblemSpec] = &[
    &ROSENBROCK_SPEC,
    &SPHERE_SPEC,
    &BEALE_SPEC,
    &BOOTH_SPEC,
    &CONSTRAINED_QUADRATIC_SPEC,
    &EQUALITY_CONSTRAINED_QUADRATIC_SPEC,
    &MATYAS_SPEC,
    &MCCORMICK_SPEC,
    &STEP_SPEC,
    &GOLDSTEIN_PRICE_SPEC,
    &POWELL_SINGULAR_SPEC,
    &RASTRIGIN_SPEC,
    &SPARSE_LEAST_SQUARES_SPEC,
    &EXPONENTIAL_FIT_SPEC,
    &THREE_HUMP_CAMEL_SPEC,
    &PICHENY_SPEC,
    &ZERO_SPEC,
    &HIMMELBLAU_SPEC,
    &ACKLEY_SPEC,
    &LEVY_SPEC,
    &STYBLINSKI_TANG_SPEC,
    &SCHAFFER_N2_SPEC,
    &SCHAFFER_N4_SPEC,
    &BUKIN_N6_SPEC,
    &CROSS_IN_TRAY_SPEC,
    &EASOM_SPEC,
    &EGGHOLDER_SPEC,
    &HOLDER_TABLE_SPEC,
];
