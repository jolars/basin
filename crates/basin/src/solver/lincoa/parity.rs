//! Cross-validation of the LINCOA driver against PRIMA v0.7.2.
//!
//! PRIMA is the BSD-3 C/Fortran reference translation of Powell's solvers
//! (vendored at `tools/prima`). For each linearly-constrained test problem we
//! generate a fixture with PRIMA's LINCOA (`tests/fixtures/lincoa_prima_driver.c`
//! → `tests/fixtures/lincoa_<problem>_<n>d.tsv`; see that file and the fixtures
//! `README.md`) recording the locked inputs (including the `A x ≤ b` system), the
//! full evaluation trace, and the final `x`/`f`/`cstrv`/`nf`. This module replays
//! the same problem through basin's [`LincoaWork`] driver and asserts a tiered
//! parity.
//!
//! The fixtures are inequality-only (no box, no equalities), so PRIMA's folded
//! constraint system equals the explicit `A x ≤ b`, which is exactly what basin's
//! [`fold_constraints`] produces — the two solvers see the same feasible region.
//!
//! # What is and isn't asserted
//!
//! basin's LINCOA is ported from Powell 2015 / PRIMA but is not an FP-identical
//! transcription, so the per-evaluation trajectory diverges once trust-region
//! iterations begin. The assertions mirror the NEWUOA parity harness:
//!
//! 1. **Objective equivalence (tight, `1e-12`).** The Rust objective recomputed
//!    at every point PRIMA evaluated must match the fixture value.
//! 2. **Initial design (tight, order-independent).** basin's first `npt` samples
//!    are the coordinate cross `{x0, x0 ± ρ_beg eₖ}`.
//! 3. **Final output (loose).** basin reaches the same constrained minimizer:
//!    `f` to `1e-6` relative, `x` to `1e-4` in `‖·‖∞`, the returned point feasible
//!    (`cstrv ≈ 0`), and `nf` within a margin.

use super::driver::{LincoaWork, Transition};
use super::init::fold_constraints;

/// One PRIMA reference run, parsed from a `lincoa_*.tsv` fixture.
struct Fixture {
    problem: String,
    n: usize,
    m_ineq: usize,
    rho_beg: f64,
    rho_end: f64,
    max_fun: usize,
    npt: usize,
    x0: Vec<f64>,
    /// `A` in row-major, `m_ineq × n`.
    aineq: Vec<f64>,
    bineq: Vec<f64>,
    /// Every objective evaluation PRIMA made, in order: `(f, x)`.
    evals: Vec<(f64, Vec<f64>)>,
    final_f: f64,
    final_x: Vec<f64>,
}

/// `key=value` token → the value as `&str` (panics on a mismatched key).
fn kv<'a>(tok: &'a str, key: &str) -> &'a str {
    let (k, v) = tok
        .split_once('=')
        .unwrap_or_else(|| panic!("expected key=value, got {tok:?}"));
    assert_eq!(k, key, "expected key {key:?}, got {k:?}");
    v
}

fn parse_fixture(text: &str) -> Fixture {
    let mut problem = None;
    let mut n = None;
    let mut m_ineq = None;
    let mut rho_beg = None;
    let mut rho_end = None;
    let mut max_fun = None;
    let mut npt = None;
    let mut x0 = None;
    let mut aineq = None;
    let mut bineq = None;
    let mut evals = Vec::new();
    let mut final_line = None;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("# config") {
            let t: Vec<&str> = rest.split_whitespace().collect();
            problem = Some(kv(t[0], "problem").to_string());
            n = Some(kv(t[1], "n").parse().unwrap());
            m_ineq = Some(kv(t[2], "m_ineq").parse().unwrap());
            rho_beg = Some(kv(t[3], "rho_beg").parse().unwrap());
            rho_end = Some(kv(t[4], "rho_end").parse().unwrap());
            max_fun = Some(kv(t[5], "maxfun").parse().unwrap());
            npt = Some(kv(t[6], "npt").parse().unwrap());
        } else if let Some(rest) = line.strip_prefix("# x0") {
            x0 = Some(
                rest.split_whitespace()
                    .map(|s| s.parse().unwrap())
                    .collect::<Vec<f64>>(),
            );
        } else if let Some(rest) = line.strip_prefix("# aineq") {
            aineq = Some(
                rest.split_whitespace()
                    .map(|s| s.parse().unwrap())
                    .collect::<Vec<f64>>(),
            );
        } else if let Some(rest) = line.strip_prefix("# bineq") {
            bineq = Some(
                rest.split_whitespace()
                    .map(|s| s.parse().unwrap())
                    .collect::<Vec<f64>>(),
            );
        } else if let Some(rest) = line.strip_prefix("# final") {
            final_line = Some(rest.to_string());
        } else if line.starts_with('#') {
            continue;
        } else {
            let mut t = line.split_whitespace();
            let _idx: usize = t.next().unwrap().parse().unwrap();
            let f: f64 = t.next().unwrap().parse().unwrap();
            let x: Vec<f64> = t.map(|s| s.parse().unwrap()).collect();
            evals.push((f, x));
        }
    }

    // `# final nf=<nf> rc=<rc> f=<f> cstrv=<cstrv> x= <x0> .. <x(n-1)>`
    let fl = final_line.expect("fixture missing `# final` line");
    let mut t = fl.split_whitespace();
    let _final_nf: usize = kv(t.next().unwrap(), "nf").parse().unwrap();
    let _rc: i32 = kv(t.next().unwrap(), "rc").parse().unwrap();
    let final_f: f64 = kv(t.next().unwrap(), "f").parse().unwrap();
    let _cstrv: f64 = kv(t.next().unwrap(), "cstrv").parse().unwrap();
    assert_eq!(t.next(), Some("x="), "expected `x=` before final x");
    let final_x: Vec<f64> = t.map(|s| s.parse().unwrap()).collect();

    Fixture {
        problem: problem.expect("problem"),
        n: n.expect("n"),
        m_ineq: m_ineq.expect("m_ineq"),
        rho_beg: rho_beg.expect("rho_beg"),
        rho_end: rho_end.expect("rho_end"),
        max_fun: max_fun.expect("maxfun"),
        npt: npt.expect("npt"),
        x0: x0.expect("x0"),
        aineq: aineq.expect("aineq"),
        bineq: bineq.expect("bineq"),
        evals,
        final_f,
        final_x,
    }
}

/// The test objectives, mirrored textually with the C harness
/// (`tests/fixtures/lincoa_prima_driver.c`).
fn objective(problem: &str, x: &[f64]) -> f64 {
    match problem {
        // (x0-2)^2 + (x1-2)^2
        "proj2" => (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2),
        // (1-x0)^2 + 100 (x1 - x0^2)^2
        "crosen2" => {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        }
        // sum (x_i - 2)^2
        "cquad3" => x.iter().map(|&xi| (xi - 2.0).powi(2)).sum(),
        other => panic!("unknown problem {other:?}"),
    }
}

/// Run the full tiered parity check for one fixture.
fn check_parity(text: &str) {
    let fx = parse_fixture(text);
    let n = fx.n;
    assert_eq!(fx.x0.len(), n);
    assert_eq!(fx.npt, 2 * n + 1, "fixtures use the recommended npt = 2n+1");
    assert_eq!(fx.aineq.len(), fx.m_ineq * n);

    // --- Tier 1: objective equivalence over PRIMA's whole explored trajectory.
    for (k, (f_prima, x)) in fx.evals.iter().enumerate() {
        let f_rust = objective(&fx.problem, x);
        let tol = 1e-12 * f_prima.abs().max(1.0);
        assert!(
            (f_rust - f_prima).abs() <= tol,
            "{} eval {}: objective mismatch C vs Rust: prima={:.17e} rust={:.17e} diff={:.3e}",
            fx.problem,
            k,
            f_prima,
            f_rust,
            (f_rust - f_prima).abs(),
        );
    }

    // Build the inequality rows (row-major A) and fold them as basin's public
    // solver would (no box, no equalities).
    let ineq: Vec<(Vec<f64>, f64)> = (0..fx.m_ineq)
        .map(|j| (fx.aineq[j * n..j * n + n].to_vec(), fx.bineq[j]))
        .collect();
    let (amat, bvec) = fold_constraints::<f64>(n, &fx.x0, None, None, &[], &ineq);

    // Drive basin once, recording every evaluation it makes.
    let trace = std::cell::RefCell::new(Vec::<Vec<f64>>::new());
    let mut eval = |x: &[f64]| -> Result<f64, std::convert::Infallible> {
        trace.borrow_mut().push(x.to_vec());
        Ok(objective(&fx.problem, x))
    };
    let (mut work, _bx, _bf) = LincoaWork::try_init(
        fx.x0.clone(),
        amat,
        bvec,
        fx.rho_beg,
        fx.rho_end,
        fx.npt,
        &mut eval,
    )
    .unwrap();

    let mut converged = false;
    for _ in 0..fx.max_fun {
        let out = work.step(&mut eval).unwrap();
        if matches!(out.transition, Transition::Converged) {
            converged = true;
            break;
        }
        if trace.borrow().len() >= fx.max_fun {
            break;
        }
    }
    let (basin_x, basin_f) = work.best();
    let nf = trace.borrow().len();

    // --- Tier 2: basin's first npt samples are the coordinate cross (§3).
    let trace_ref = trace.borrow();
    assert!(
        trace_ref.len() >= fx.npt,
        "{}: basin made only {} evals, fewer than npt={}",
        fx.problem,
        trace_ref.len(),
        fx.npt,
    );
    let initial = &trace_ref[..fx.npt];
    let mut expected: Vec<Vec<f64>> = vec![fx.x0.clone()];
    for k in 0..n {
        let mut xp = fx.x0.clone();
        xp[k] += fx.rho_beg;
        let mut xm = fx.x0.clone();
        xm[k] -= fx.rho_beg;
        expected.push(xp);
        expected.push(xm);
    }
    for want in &expected {
        let found = initial.iter().any(|got| {
            got.len() == want.len() && got.iter().zip(want).all(|(a, b)| (a - b).abs() <= 1e-12)
        });
        assert!(
            found,
            "{}: initial design missing point {:?}",
            fx.problem, want
        );
    }
    drop(trace_ref);

    // --- Tier 3: converged final output matches PRIMA's.
    assert!(
        converged,
        "{}: basin stopped on the eval budget, not convergence",
        fx.problem
    );

    let f_tol = 1e-6 * (1.0 + fx.final_f.abs());
    assert!(
        (basin_f - fx.final_f).abs() <= f_tol,
        "{}: final f mismatch: prima={:.17e} basin={:.17e} diff={:.3e} tol={:.3e}",
        fx.problem,
        fx.final_f,
        basin_f,
        (basin_f - fx.final_f).abs(),
        f_tol,
    );

    let x_inf = basin_x
        .iter()
        .zip(&fx.final_x)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        x_inf <= 1e-4,
        "{}: final x mismatch (||.||_inf = {:.3e} > 1e-4)\n  prima={:?}\n  basin={:?}",
        fx.problem,
        x_inf,
        fx.final_x,
        basin_x,
    );

    // The returned point must be feasible: max_j (a_jᵀ x − b_j) ≈ 0.
    let cstrv = (0..fx.m_ineq).fold(0.0_f64, |acc, j| {
        let ax: f64 = (0..n).map(|i| fx.aineq[j * n + i] * basin_x[i]).sum();
        acc.max(ax - fx.bineq[j])
    });
    assert!(
        cstrv <= 1e-6,
        "{}: returned point infeasible, cstrv = {:.3e}",
        fx.problem,
        cstrv
    );

    // nf is the weakest signal (paper-derived port → different but valid
    // trajectory); a same-ballpark sanity bound, not a parity claim.
    let nf_margin = (0.5 * fx.evals.len() as f64).max(15.0);
    let nf_diff = (nf as f64 - fx.evals.len() as f64).abs();
    assert!(
        nf_diff <= nf_margin,
        "{}: nf out of margin: prima={} basin={} diff={} margin={:.1}",
        fx.problem,
        fx.evals.len(),
        nf,
        nf_diff,
        nf_margin,
    );
}

macro_rules! parity_test {
    ($name:ident, $file:literal) => {
        #[test]
        fn $name() {
            check_parity(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/",
                $file
            )));
        }
    };
}

parity_test!(proj2_matches_prima, "lincoa_proj2_2d.tsv");
parity_test!(crosen2_matches_prima, "lincoa_crosen2_2d.tsv");
parity_test!(cquad3_matches_prima, "lincoa_cquad3_3d.tsv");
