//! Evaluation and iteration traces recorded before optimizing the driver.
//!
//! Recorded from Basin commit 013bb6687258fee75dafcfdd91d93f7ecb553e73.
//! These retain the raw callback inputs and values, per-iteration incumbents,
//! resolution reductions, and stopping transitions, including guarded costs
//! after budget exhaustion. The non-finite case covers extreme-barrier inputs.

use std::{cell::Cell, convert::Infallible};

use super::driver::{CobylaWork, Transition};

fn trace(case: &str) -> String {
    let (start, bounds, budget) = match case {
        "camel" => (vec![0.0, 0.0], vec![(-3.0, 3.0), (-2.0, 2.0)], 50),
        "sphere" => (
            vec![2.5, -2.0, 1.5, -1.0, 0.5, 2.25, -1.75, 1.25, -0.75, 0.25],
            vec![(-5.0, 5.0); 10],
            200,
        ),
        "quadratic" => (vec![0.5, 0.5], vec![(0.0, 2.0); 2], 100),
        "nonfinite" => (vec![0.0, 0.0], vec![(-1.0, 1.0); 2], 100),
        _ => unreachable!(),
    };
    let n = start.len();
    let m = 2 * n + usize::from(case == "quadratic");
    let nf = Cell::new(0);
    let evaluations = std::cell::RefCell::new(String::new());
    let mut eval = |x: &[f64]| -> Result<_, Infallible> {
        let f = if nf.get() >= budget {
            f64::INFINITY
        } else {
            nf.set(nf.get() + 1);
            match case {
                "camel" => {
                    (4.0 - 2.1 * x[0].powi(2) + x[0].powi(4) / 3.0)
                        * x[0].powi(2)
                        + x[0] * x[1]
                        + (-4.0 + 4.0 * x[1].powi(2)) * x[1].powi(2)
                }
                "sphere" => x.iter().map(|v| v * v).sum(),
                "quadratic" => (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2),
                "nonfinite" if x[0] > 0.25 => f64::NAN,
                "nonfinite" if x[1] > 0.25 => f64::INFINITY,
                "nonfinite" => (x[0] - 0.1).powi(2) + (x[1] - 0.1).powi(2),
                _ => unreachable!(),
            }
        };
        let mut c = Vec::with_capacity(m);
        if case == "quadratic" {
            c.push(-(1.5 - x[0] - x[1]));
        }
        for (&v, &(lower, upper)) in x.iter().zip(&bounds) {
            c.extend([lower - v, v - upper]);
        }
        if case == "nonfinite" && x[0] < -0.25 {
            c[0] = f64::NAN;
            c[1] = f64::NEG_INFINITY;
        }
        let mut log = evaluations.borrow_mut();
        log.push_str(&format!("eval {} {f:.17e}", nf.get()));
        for v in x.iter().chain(&c) {
            log.push_str(&format!(" {v:.17e}"));
        }
        log.push('\n');
        Ok((f, c))
    };
    let (mut work, _, _) = CobylaWork::try_init(
        start,
        m,
        0.5,
        f64::EPSILON.sqrt() * 0.5,
        &mut eval,
    )
    .unwrap();
    let mut out = evaluations.take();
    for _ in 0..10000 {
        if nf.get() >= budget {
            break;
        }
        let transition = work.step(&mut eval).unwrap();
        out.push_str(&evaluations.take());
        let (x, f) = work.best();
        out.push_str(&format!("{transition:?} {:.17e} {f:.17e}", work.rho()));
        for v in x {
            out.push_str(&format!(" {v:.17e}"));
        }
        out.push('\n');
        if matches!(transition, Transition::Converged | Transition::Failed) {
            break;
        }
    }
    out
}

#[test]
fn migration_traces() {
    for (case, expected) in [
        (
            "camel",
            include_str!("../../../tests/fixtures/cobyla_migration_camel.txt"),
        ),
        (
            "sphere",
            include_str!("../../../tests/fixtures/cobyla_migration_sphere.txt"),
        ),
        (
            "quadratic",
            include_str!(
                "../../../tests/fixtures/cobyla_migration_quadratic.txt"
            ),
        ),
        (
            "nonfinite",
            include_str!(
                "../../../tests/fixtures/cobyla_migration_nonfinite.txt"
            ),
        ),
    ] {
        let actual = trace(case);
        let actual: Vec<_> = actual.split_whitespace().collect();
        let expected: Vec<_> = expected.split_whitespace().collect();
        assert_eq!(actual.len(), expected.len(), "{case}: trace length");
        for (i, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            if let (Ok(a), Ok(b)) =
                (actual.parse::<f64>(), expected.parse::<f64>())
            {
                assert!(
                    a == b
                        || (a.is_nan() && b.is_nan())
                        || (a.is_finite()
                            && b.is_finite()
                            && (a - b).abs() <= 8e-14 * b.abs().max(1.0)),
                    "{case}, token {i}: {a} != {b}",
                );
            } else {
                assert_eq!(*actual, expected, "{case}, token {i}");
            }
        }
    }
}
