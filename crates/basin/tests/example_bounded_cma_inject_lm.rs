//! Worked example: `BoundedCmaInject` with Levenberg-Marquardt inner.
//!
//! Booth-as-residuals on a tight `[-1, 1]²` box: the unconstrained
//! minimum `(1, 3)` is outside the box, so the bound-active constrained
//! optimum sits on the box corner `(1, 1)`. Bounded CMA-ES does global
//! exploration under the adaptive BoundPenalty (Hansen/pycma); each
//! generation's best `k` candidates are polished by an unconstrained LM
//! inner working off the same residual + Jacobian; the refined points
//! are Mahalanobis-clipped (Hansen 2011 eq. 4) and injected back into
//! the population.
//!
//! Run: `cargo test --test example_bounded_cma_inject_lm --features
//! nalgebra -- --nocapture`. The `--nocapture` flag is what lets the
//! progress prints land in your terminal.

#![cfg(feature = "nalgebra")]

use basin::problems::BoothBoxedResiduals;
use basin::{BoundedCmaEs, BoundedCmaInject, CmaEsState, Executor, LevenbergMarquardt};
use nalgebra::{DMatrix, DVector};

#[test]
fn example_bounded_cma_inject_lm_on_booth_corner() {
    // -----------------------------------------------------------------
    // 1. Problem: Booth as residuals (so LM has a Jacobian to chew on)
    //    with bounds `[-1, 1]²` that exclude the unconstrained min
    //    `(1, 3)`. The corner `(1, 1)` is the constrained optimum.
    // -----------------------------------------------------------------
    let lower = DVector::from_vec(vec![-1.0, -1.0]);
    let upper = DVector::from_vec(vec![1.0, 1.0]);
    let problem = BoothBoxedResiduals::<DVector<f64>>::new(lower, upper);

    // -----------------------------------------------------------------
    // 2. Outer: bounded CMA-ES with default population size.
    //    Start at the opposite corner so CMA has to drift across the
    //    box; σ = 0.3 keeps the initial distribution inside.
    // -----------------------------------------------------------------
    let m0 = DVector::from_vec(vec![-0.5, -0.5]);
    let cma = BoundedCmaEs::<DVector<f64>, DMatrix<f64>>::new(42);

    // -----------------------------------------------------------------
    // 3. Memetic wrapper: top `k = 1` candidate per generation gets
    //    polished by LM. LM's tol_grad = 1e-8 default terminates the
    //    inner cleanly once ‖Jᵀr‖_∞ ≤ 1e-8 (2–3 iters on this
    //    quadratic), so we don't need to cap inner iters explicitly.
    // -----------------------------------------------------------------
    let solver = BoundedCmaInject::with_inner_solver(cma, LevenbergMarquardt::new()).with_k(1);

    // -----------------------------------------------------------------
    // 4. Drive.
    // -----------------------------------------------------------------
    let state = CmaEsState::<DVector<f64>, DMatrix<f64>>::new(m0, 0.3);
    let result = Executor::new(problem, solver, state)
        .max_iter(100)
        .run()
        .unwrap();

    let p = result.param();
    eprintln!();
    eprintln!("=== BoundedCmaInject + Levenberg-Marquardt on BoothBoxedResiduals ===");
    eprintln!("bounds:           [-1, 1]²");
    eprintln!("unconstrained min: (1, 3)  (outside the box)");
    eprintln!("constrained min:   (1, 1)  (corner of the box)");
    eprintln!();
    eprintln!("final iterate:    ({:.10}, {:.10})", p[0], p[1]);
    eprintln!("final cost:       {:e}", result.cost());
    eprintln!("outer iters:      {}", result.iter());
    eprintln!(
        "cost evals:       {}  (rolls up LM residual + Jacobian calls)",
        result.cost_evals()
    );
    eprintln!("termination:      {:?}", result.reason);
    eprintln!();

    // -----------------------------------------------------------------
    // 5. Sanity check: we should be at the corner (1, 1) within the
    //    BoundPenalty's tolerance. Inner LM doesn't see the bounds (it's
    //    unconstrained), so the polished iterates may temporarily step
    //    toward the unconstrained min (1, 3); BoundPenalty repairs them
    //    back into feasibility on the next generation.
    //
    //    Tolerance is deliberately loose (5e-2): CMA-ES diagonalizes its
    //    covariance each generation, so the walk's exact trajectory depends
    //    on which symmetric eigendecomposition is linked. The default
    //    pure-Rust nalgebra path lands within ~5e-3, but `--all-features`
    //    enables `nalgebra-lapack`, swapping in LAPACK's `dsyev`, whose
    //    different (but equally valid) eigenvectors nudge the sampling path
    //    and leave the final iterate ~2.4e-2 off the corner. This is a
    //    "did we reach the corner" sanity check, not a precision test, so
    //    the bound holds across both backends.
    // -----------------------------------------------------------------
    let err = (p[0] - 1.0).abs().max((p[1] - 1.0).abs());
    assert!(
        err <= 5e-2,
        "expected ≈ (1, 1) within 5e-2, got ({}, {})—err = {}",
        p[0],
        p[1],
        err
    );
}
