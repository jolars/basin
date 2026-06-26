//! Cross-validation of the COBYLA driver against PRIMA v0.7.2.
//!
//! PRIMA is the BSD-3 C/Fortran reference translation of Powell's solvers
//! (vendored at `tools/prima`). For each nonlinearly-constrained test problem we
//! generate a fixture with PRIMA's COBYLA (`tests/fixtures/cobyla_prima_driver.c`
//! → `tests/fixtures/cobyla_<problem>_<n>d.tsv`; see that file and the fixtures
//! `README.md`) recording the locked inputs, the full evaluation trace (each
//! point's `f` and constraint violation `cstrv`), and the final
//! `x`/`f`/`cstrv`/`nf`. This module replays the same problem through basin's
//! [`CobylaWork`] driver and asserts a tiered parity.
//!
//! The fixtures use only the nonlinear constraint block (`m_ineq = m_eq = 0`,
//! `xl`/`xu = ±∞`), which is exactly the trait-only path basin's [`Cobyla`]
//! drives (its `m_lcon = 0`)—so the two solvers see the same feasible region.
//!
//! # What is and isn't asserted
//!
//! basin's COBYLA is ported from Powell 1994 / PRIMA but is not an FP-identical
//! transcription, so the per-evaluation trajectory diverges once trust-region
//! iterations begin. The assertions mirror the NEWUOA/LINCOA parity harnesses,
//! with COBYLA's nonlinear constraints added to tier 1:
//!
//! 1. **Function equivalence (tight, `1e-12`).** The Rust objective *and*
//!    constraint violation recomputed at every point PRIMA evaluated must match
//!    the fixture values—guards both functions against C↔Rust drift.
//! 2. **Initial design (tight, order-independent).** basin's first `n+1` samples
//!    equal PRIMA's first `n+1` samples as a set (COBYLA's `n+1`-vertex simplex
//!    is built cumulatively with pole swaps, so this compares against the
//!    fixture rather than reconstructing it analytically).
//! 3. **Final output (loose).** basin reaches the same constrained minimizer:
//!    `f` to `1e-6` relative, `x` to `1e-4` in `‖·‖∞`, the returned point
//!    feasible (`cstrv ≈ 0`), and `nf` within a margin.

use super::driver::{CobylaWork, Transition};

/// One PRIMA reference run, parsed from a `cobyla_*.tsv` fixture.
struct Fixture {
    problem: String,
    n: usize,
    m_nlcon: usize,
    rho_beg: f64,
    rho_end: f64,
    max_fun: usize,
    x0: Vec<f64>,
    /// Every objective/constraint evaluation PRIMA made, in order: `(f, cstrv, x)`.
    evals: Vec<(f64, f64, Vec<f64>)>,
    final_f: f64,
    final_cstrv: f64,
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
    let mut m_nlcon = None;
    let mut rho_beg = None;
    let mut rho_end = None;
    let mut max_fun = None;
    let mut x0 = None;
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
            m_nlcon = Some(kv(t[2], "m_nlcon").parse().unwrap());
            rho_beg = Some(kv(t[3], "rho_beg").parse().unwrap());
            rho_end = Some(kv(t[4], "rho_end").parse().unwrap());
            max_fun = Some(kv(t[5], "maxfun").parse().unwrap());
        } else if let Some(rest) = line.strip_prefix("# x0") {
            x0 = Some(
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
            let cstrv: f64 = t.next().unwrap().parse().unwrap();
            let x: Vec<f64> = t.map(|s| s.parse().unwrap()).collect();
            evals.push((f, cstrv, x));
        }
    }

    // `# final nf=<nf> rc=<rc> f=<f> cstrv=<cstrv> x= <x0> .. <x(n-1)>`
    let fl = final_line.expect("fixture missing `# final` line");
    let mut t = fl.split_whitespace();
    let _final_nf: usize = kv(t.next().unwrap(), "nf").parse().unwrap();
    let _rc: i32 = kv(t.next().unwrap(), "rc").parse().unwrap();
    let final_f: f64 = kv(t.next().unwrap(), "f").parse().unwrap();
    let final_cstrv: f64 = kv(t.next().unwrap(), "cstrv").parse().unwrap();
    assert_eq!(t.next(), Some("x="), "expected `x=` before final x");
    let final_x: Vec<f64> = t.map(|s| s.parse().unwrap()).collect();

    Fixture {
        problem: problem.expect("problem"),
        n: n.expect("n"),
        m_nlcon: m_nlcon.expect("m_nlcon"),
        rho_beg: rho_beg.expect("rho_beg"),
        rho_end: rho_end.expect("rho_end"),
        max_fun: max_fun.expect("maxfun"),
        x0: x0.expect("x0"),
        evals,
        final_f,
        final_cstrv,
        final_x,
    }
}

/// The test objectives, mirrored textually with the C harness
/// (`tests/fixtures/cobyla_prima_driver.c`).
fn objective(problem: &str, x: &[f64]) -> f64 {
    match problem {
        // x0 * x1
        "disk" => x[0] * x[1],
        // -x0 - x1
        "fletcher" => -x[0] - x[1],
        // sum (x_i - 2)^2
        "ballsphere" => x.iter().map(|&xi| (xi - 2.0).powi(2)).sum(),
        other => panic!("unknown problem {other:?}"),
    }
}

/// The test constraints `c(x) ≤ 0`, mirrored textually with the C harness.
fn constraints(problem: &str, x: &[f64]) -> Vec<f64> {
    match problem {
        // x0^2 + x1^2 - 1 ≤ 0
        "disk" => vec![x[0] * x[0] + x[1] * x[1] - 1.0],
        // x0^2 - x1 ≤ 0 ; x0^2 + x1^2 - 1 ≤ 0
        "fletcher" => vec![x[0] * x[0] - x[1], x[0] * x[0] + x[1] * x[1] - 1.0],
        // sum x_i^2 - 1 ≤ 0
        "ballsphere" => vec![x.iter().map(|&xi| xi * xi).sum::<f64>() - 1.0],
        other => panic!("unknown problem {other:?}"),
    }
}

/// Constraint violation `[ maxᵢ cᵢ ]₊`, matching the C driver's `cstrv` column
/// and basin's `eval_moderated`.
fn violation(constr: &[f64]) -> f64 {
    constr.iter().fold(0.0_f64, |a, &c| a.max(c)).max(0.0)
}

/// Run the full tiered parity check for one fixture.
fn check_parity(text: &str) {
    let fx = parse_fixture(text);
    let n = fx.n;
    let m = fx.m_nlcon;
    assert_eq!(fx.x0.len(), n);

    // --- Tier 1: objective + constraint equivalence over PRIMA's trajectory.
    for (k, (f_prima, cstrv_prima, x)) in fx.evals.iter().enumerate() {
        let f_rust = objective(&fx.problem, x);
        let c_rust = constraints(&fx.problem, x);
        assert_eq!(c_rust.len(), m, "{}: constraint count mismatch", fx.problem);
        let cstrv_rust = violation(&c_rust);
        let f_tol = 1e-12 * f_prima.abs().max(1.0);
        assert!(
            (f_rust - f_prima).abs() <= f_tol,
            "{} eval {}: objective mismatch C vs Rust: prima={:.17e} rust={:.17e} diff={:.3e}",
            fx.problem,
            k,
            f_prima,
            f_rust,
            (f_rust - f_prima).abs(),
        );
        let c_tol = 1e-12 * cstrv_prima.abs().max(1.0);
        assert!(
            (cstrv_rust - cstrv_prima).abs() <= c_tol,
            "{} eval {}: cstrv mismatch C vs Rust: prima={:.17e} rust={:.17e} diff={:.3e}",
            fx.problem,
            k,
            cstrv_prima,
            cstrv_rust,
            (cstrv_rust - cstrv_prima).abs(),
        );
    }

    // Drive basin once, recording every evaluation it makes. The eval closure
    // returns the (raw) objective and constraint vector, as `CobylaWork` expects.
    let trace = std::cell::RefCell::new(Vec::<Vec<f64>>::new());
    let mut eval = |x: &[f64]| -> Result<(f64, Vec<f64>), std::convert::Infallible> {
        trace.borrow_mut().push(x.to_vec());
        Ok((objective(&fx.problem, x), constraints(&fx.problem, x)))
    };
    let (mut work, _bx, _bf) =
        CobylaWork::try_init(fx.x0.clone(), m, fx.rho_beg, fx.rho_end, &mut eval).unwrap();

    let mut converged = false;
    for _ in 0..fx.max_fun {
        let transition = work.step(&mut eval).unwrap();
        if matches!(transition, Transition::Converged) {
            converged = true;
            break;
        }
        if matches!(transition, Transition::Failed) {
            break;
        }
        if trace.borrow().len() >= fx.max_fun {
            break;
        }
    }
    let (basin_x, basin_f) = work.best();
    let nf = trace.borrow().len();

    // --- Tier 2: basin's first n+1 samples equal PRIMA's as a set.
    let npt = n + 1;
    let trace_ref = trace.borrow();
    assert!(
        trace_ref.len() >= npt,
        "{}: basin made only {} evals, fewer than n+1={}",
        fx.problem,
        trace_ref.len(),
        npt,
    );
    let initial = &trace_ref[..npt];
    for (f_prima, _cstrv, want) in &fx.evals[..npt] {
        let found = initial.iter().any(|got| {
            got.len() == want.len() && got.iter().zip(want).all(|(a, b)| (a - b).abs() <= 1e-12)
        });
        assert!(
            found,
            "{}: initial design missing PRIMA point {:?} (f={:.6e})",
            fx.problem, want, f_prima
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

    // The returned point must be feasible: [ maxᵢ cᵢ(x) ]₊ ≈ 0. PRIMA's own
    // final cstrv is also ≈ 0 (sanity-check the fixture is a feasible solution).
    assert!(
        fx.final_cstrv <= 1e-6,
        "{}: fixture final point infeasible, cstrv = {:.3e}",
        fx.problem,
        fx.final_cstrv
    );
    let cstrv = violation(&constraints(&fx.problem, &basin_x));
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

parity_test!(disk_matches_prima, "cobyla_disk_2d.tsv");
parity_test!(fletcher_matches_prima, "cobyla_fletcher_2d.tsv");
parity_test!(ballsphere_matches_prima, "cobyla_ballsphere_3d.tsv");
