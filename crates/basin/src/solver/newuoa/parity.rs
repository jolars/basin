//! Cross-validation of the NEWUOA [`minimize`] driver against PRIMA v0.7.2.
//!
//! PRIMA is the BSD-3 C/Fortran reference translation of Powell's solvers
//! (vendored at `tools/prima`). For each test problem we generate a fixture
//! with PRIMA's NEWUOA (`tests/fixtures/newuoa_prima_driver.c` →
//! `tests/fixtures/newuoa_<problem>_<n>d.tsv`; see that file and the fixtures
//! `README.md`) recording the locked inputs, the full evaluation trace, and the
//! final `x`/`f`/`nf`. This module replays the same problem through basin and
//! asserts a tiered parity.
//!
//! Because `minimize` is `pub(crate)` (NEWUOA has no public `Solver` surface
//! yet—that is the next roadmap step), this lives in-crate rather than under
//! `tests/`, and reads the fixtures via [`include_str!`] so it is independent of
//! the test's working directory.
//!
//! # What is and isn't asserted
//!
//! basin's NEWUOA is derived from Powell's 2006 paper (aligned to PRIMA's tuning
//! in the load-bearing spots), **not** a line-by-line transcription of PRIMA's
//! Fortran, so the per-evaluation arithmetic order differs and the eval
//! sequences diverge once trust-region iterations begin. The assertions are
//! therefore structured to be meaningful without demanding FP-identical internal
//! arithmetic:
//!
//! 1. **Objective equivalence (tight).** The C objective in the harness and the
//!    Rust objective here must be the same function; we recompute the Rust
//!    objective at *every* point PRIMA evaluated and require agreement to
//!    `1e-12` relative. This catches any coefficient / index / sign drift
//!    between the two definitions across the whole region PRIMA explored.
//! 2. **Initial design (tight, order-independent).** basin's first `npt` samples
//!    must be exactly the coordinate cross `{x0, x0 ± ρ_beg eₖ}` (§3). PRIMA
//!    evaluates all `+` then all `−`; basin interleaves `+,−` per coordinate, so
//!    we compare as a *set*, not positionally.
//! 3. **Final output (loose).** basin must converge (ρ reached, not the eval
//!    budget) to the same minimizer PRIMA found: `f` to `1e-6` relative, `x` to
//!    `1e-4` in `‖·‖∞`, and the evaluation count `nf` within a margin (not
//!    exact—a non-transcription port will not match `nf` to the unit).

use super::driver::{NewuoaConfig, NewuoaStop, minimize};

/// One PRIMA reference run, parsed from a `newuoa_*.tsv` fixture.
struct Fixture {
    problem: String,
    n: usize,
    rho_beg: f64,
    rho_end: f64,
    max_fun: usize,
    npt: usize,
    x0: Vec<f64>,
    /// Every objective evaluation PRIMA made, in PRIMA's order: `(f, x)`.
    evals: Vec<(f64, Vec<f64>)>,
    final_nf: usize,
    final_f: f64,
    final_x: Vec<f64>,
}

/// `key=value` token → the value parsed as `T` (panics on a mismatched key).
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
    let mut rho_beg = None;
    let mut rho_end = None;
    let mut max_fun = None;
    let mut npt = None;
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
            rho_beg = Some(kv(t[2], "rho_beg").parse().unwrap());
            rho_end = Some(kv(t[3], "rho_end").parse().unwrap());
            max_fun = Some(kv(t[4], "maxfun").parse().unwrap());
            npt = Some(kv(t[5], "npt").parse().unwrap());
        } else if let Some(rest) = line.strip_prefix("# x0") {
            x0 = Some(
                rest.split_whitespace()
                    .map(|s| s.parse().unwrap())
                    .collect::<Vec<f64>>(),
            );
        } else if let Some(rest) = line.strip_prefix("# final") {
            final_line = Some(rest.to_string());
        } else if line.starts_with('#') {
            continue; // any other comment line
        } else {
            // eval row: idx f x0 .. x(n-1)
            let mut t = line.split_whitespace();
            let _idx: usize = t.next().unwrap().parse().unwrap();
            let f: f64 = t.next().unwrap().parse().unwrap();
            let x: Vec<f64> = t.map(|s| s.parse().unwrap()).collect();
            evals.push((f, x));
        }
    }

    // `# final nf=<nf> rc=<rc> f=<f> x= <x0> .. <x(n-1)>`
    let fl = final_line.expect("fixture missing `# final` line");
    let mut t = fl.split_whitespace();
    let final_nf: usize = kv(t.next().unwrap(), "nf").parse().unwrap();
    let _rc: i32 = kv(t.next().unwrap(), "rc").parse().unwrap();
    let final_f: f64 = kv(t.next().unwrap(), "f").parse().unwrap();
    assert_eq!(t.next(), Some("x="), "expected `x=` before final x");
    let final_x: Vec<f64> = t.map(|s| s.parse().unwrap()).collect();

    Fixture {
        problem: problem.expect("problem"),
        n: n.expect("n"),
        rho_beg: rho_beg.expect("rho_beg"),
        rho_end: rho_end.expect("rho_end"),
        max_fun: max_fun.expect("maxfun"),
        npt: npt.expect("npt"),
        x0: x0.expect("x0"),
        evals,
        final_nf,
        final_f,
        final_x,
    }
}

/// The four test objectives, mirrored textually with the C harness
/// (`tests/fixtures/newuoa_prima_driver.c`). Keep the summation order, the
/// coefficients, and the index base identical to that file.
fn objective(problem: &str, x: &[f64]) -> f64 {
    match problem {
        // basin coefficient form: sum 100 (x_{i+1}-x_i^2)^2 + (1-x_i)^2
        "rosenbrock" => {
            let mut s = 0.0;
            for i in 0..x.len() - 1 {
                let t = x[i + 1] - x[i] * x[i];
                s += 100.0 * t * t + (1.0 - x[i]) * (1.0 - x[i]);
            }
            s
        }
        // chrosen.f90 form: sum (x_i-1)^2 + 100 (x_{i+1}-x_i^2)^2
        "chrosen" => {
            let mut s = 0.0;
            for i in 0..x.len() - 1 {
                let a = x[i] - 1.0;
                let b = x[i + 1] - x[i] * x[i];
                s += a * a + 100.0 * b * b;
            }
            s
        }
        // sum_{i<n-1} (x_i^2 + x_{n-1}^2)^2 - 4 x_i + 3
        "arwhead" => {
            let n = x.len();
            let last = n - 1;
            let mut s = 0.0;
            for i in 0..n - 1 {
                let t = x[i] * x[i] + x[last] * x[last];
                s += t * t - 4.0 * x[i] + 3.0;
            }
            s
        }
        // sum (x_i-1)^2 + (sum i (x_i-1))^2 + (sum i (x_i-1))^4
        "vardim" => {
            let (mut sq, mut lin) = (0.0, 0.0);
            for (i, &xi) in x.iter().enumerate() {
                let d = xi - 1.0;
                sq += d * d;
                lin += (i as f64 + 1.0) * d;
            }
            sq + lin * lin + lin * lin * lin * lin
        }
        other => panic!("unknown problem {other:?}"),
    }
}

/// Run the full tiered parity check for one fixture.
fn check_parity(text: &str) {
    let fx = parse_fixture(text);
    let n = fx.n;
    assert_eq!(fx.x0.len(), n);
    assert_eq!(fx.npt, 2 * n + 1, "fixtures use the recommended npt = 2n+1");

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

    // Drive basin once, recording every evaluation it makes.
    let cfg = NewuoaConfig {
        rho_beg: fx.rho_beg,
        rho_end: fx.rho_end,
        max_fun: fx.max_fun,
        npt: fx.npt,
    };
    let mut trace: Vec<Vec<f64>> = Vec::new();
    let outcome = {
        let trace = std::cell::RefCell::new(&mut trace);
        minimize(fx.x0.clone(), &cfg, |x| {
            trace.borrow_mut().push(x.to_vec());
            objective(&fx.problem, x)
        })
    };

    // --- Tier 2: basin's first npt samples are the coordinate cross (§3),
    // compared as an (order-independent) set since PRIMA and basin emit the
    // initial design in different orders.
    assert!(
        trace.len() >= fx.npt,
        "{}: basin made only {} evals, fewer than npt={}",
        fx.problem,
        trace.len(),
        fx.npt,
    );
    let initial = &trace[..fx.npt];
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

    // --- Tier 3: converged final output matches PRIMA's.
    assert_eq!(
        outcome.stop,
        NewuoaStop::RhoReached,
        "{}: basin stopped on the eval budget, not convergence",
        fx.problem,
    );

    let f_tol = 1e-6 * (1.0 + fx.final_f.abs());
    assert!(
        (outcome.f - fx.final_f).abs() <= f_tol,
        "{}: final f mismatch: prima={:.17e} basin={:.17e} diff={:.3e} tol={:.3e}",
        fx.problem,
        fx.final_f,
        outcome.f,
        (outcome.f - fx.final_f).abs(),
        f_tol,
    );

    let x_inf = outcome
        .x
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
        outcome.x,
    );

    // nf is the weakest signal: a paper-derived port takes a different (but
    // valid) trust-region trajectory, so the eval count differs by a chunk
    // (~15-20% observed) while still reaching the same minimizer. The margin is
    // a same-ballpark sanity bound, not a parity claim—see the module docs.
    let nf_margin = (0.25 * fx.final_nf as f64).max(10.0);
    let nf_diff = (outcome.nf as f64 - fx.final_nf as f64).abs();
    assert!(
        nf_diff <= nf_margin,
        "{}: nf out of margin: prima={} basin={} diff={} margin={:.1}",
        fx.problem,
        fx.final_nf,
        outcome.nf,
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

parity_test!(rosenbrock_2d_matches_prima, "newuoa_rosenbrock_2d.tsv");
parity_test!(chrosen_5d_matches_prima, "newuoa_chrosen_5d.tsv");
parity_test!(arwhead_5d_matches_prima, "newuoa_arwhead_5d.tsv");
parity_test!(vardim_5d_matches_prima, "newuoa_vardim_5d.tsv");
