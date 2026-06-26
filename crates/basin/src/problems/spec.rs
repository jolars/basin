//! Metadata describing each problem in the corpus.
//!
//! Two layers:
//! - A flat [`ProblemSpec`] static per problem, aggregated into a slice
//!   ([`crate::problems::ALL_SPECS`]) so callers (e.g. a web UI) can iterate
//!   and filter the catalog without touching the math.
//! - A [`HasSpec`] trait, blanket-implemented for the wrapped problem types,
//!   so generic code can recover a spec from a problem instance.

/// Bibliographic reference to the source defining a problem.
///
/// Strings are free-form to keep this lightweight; the web app just renders
/// them. `doi` and `url` are optional: pick whichever the source actually has.
pub struct Reference {
    /// Short human-readable citation, e.g. `"Rosenbrock (1960)"`.
    pub citation: &'static str,
    /// Full title of the work.
    pub title: &'static str,
    /// Venue, journal, or book, plus pages, e.g. `"The Computer Journal, 3(3), 175–184"`.
    pub source: &'static str,
    /// DOI without scheme, e.g. `"10.1093/comjnl/3.3.175"`.
    pub doi: Option<&'static str>,
    /// Stable URL (e.g. publisher page, arXiv, or Surjanovic & Bingham link).
    pub url: Option<&'static str>,
}

/// Whether the problem has a fixed dimensionality or scales over `n`.
pub enum Dimensionality {
    /// Fixed at `n` dimensions (e.g. 2 for Beale).
    Fixed(usize),
    /// Defined for any `n >= min`.
    NDimensional {
        /// Minimum admissible problem dimension.
        min: usize,
    },
}

/// Boolean tags describing mathematical character. Each problem sets only the
/// fields that hold; defaults to all `false`. Used by the web UI for filtering.
///
/// Add fields here only when a new problem actually needs the distinction;
/// the field set is small on purpose.
#[derive(Clone, Copy, Debug, Default)]
pub struct Properties {
    /// `C^∞` (or sufficiently smooth that gradient methods see no kinks).
    pub smooth: bool,
    /// Gradient exists everywhere on the domain.
    pub differentiable: bool,
    /// Convex over the standard search domain.
    pub convex: bool,
    /// Single global minimum, no spurious local minima. Be conservative:
    /// for N-D problems where unimodality depends on `n`, prefer `false` and
    /// note the caveat in [`ProblemSpec::description`].
    pub unimodal: bool,
    /// `f(x) = Σ_i g_i(x_i)`: decomposes into per-coordinate functions.
    pub separable: bool,
    /// Defined for any `n >= some min` (matches `Dimensionality::NDimensional`).
    pub scalable: bool,
}

/// Static description of a cataloged problem. The `Cost`/`Gradient` impls
/// live on the corresponding wrapper struct; this is just the metadata.
pub struct ProblemSpec {
    /// Canonical name as used in the literature, e.g. `"Rosenbrock"`.
    pub name: &'static str,
    /// Whether the problem dimension is fixed or scales over `n`.
    pub dim: Dimensionality,
    /// Mathematical character tags used by the catalog UI for filtering.
    pub properties: Properties,
    /// One or more sources. The first is the primary citation; later entries
    /// are useful surveys, popularizations, or variants.
    pub references: &'static [Reference],
    /// 1–3 sentence description suitable for a UI tooltip.
    pub description: &'static str,
}

/// Recovers the [`ProblemSpec`] for a wrapped problem type at compile time.
/// Implemented blanket-style for the corpus types, e.g.
/// `impl<P> HasSpec for Rosenbrock<P>`.
pub trait HasSpec {
    /// The catalog entry for this problem type.
    const SPEC: &'static ProblemSpec;
}
