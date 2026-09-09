// Included only by the isolated gap probe.
fn old_solve(mode: &str, case: Case, p: &Problem) -> Outcome {
    let rows = mode == "cobyla-rows";
    let historical = mode == "cobyla-historical";
    let bounds: Vec<_> = (0..case.n())
        .map(|i| {
            if rows {
                (f64::NEG_INFINITY, f64::INFINITY)
            } else {
                case.bounds(i)
            }
        })
        .collect();
    let row_count = if rows {
        case.m()
    } else {
        usize::from(matches!(case, Case::Quadratic))
    };
    let constraints: Vec<_> = (0..row_count)
        .map(|row| {
            move |x: &[f64], _: &mut ()| {
                // The crate accepts positive inequalities, despite one contrary API sentence.
                p.nc.set(p.nc.get() + 1);
                let offset = usize::from(matches!(case, Case::Quadratic));
                if offset == 1 && row == 0 {
                    1.5 - x[0] - x[1]
                } else {
                    let i = (row - offset) / 2;
                    let (lo, hi) = case.bounds(i);
                    if (row - offset) % 2 == 0 {
                        x[i] - lo
                    } else {
                        hi - x[i]
                    }
                }
            }
        })
        .collect();
    let result = cobyla::minimize(
        |x: &[f64], _: &mut ()| p.objective(x, false),
        &case.start(),
        &bounds,
        &constraints,
        (),
        case.budget(),
        cobyla::RhoBeg::All(0.5),
        Some(cobyla::StopTols {
            xtol_abs: if historical {
                vec![]
            } else {
                vec![f64::EPSILON.sqrt() * 0.5; case.n()]
            },
            ..Default::default()
        }),
    );
    let (x, f, stop, status) = match result {
        Ok((s, x, f)) => (
            x,
            f,
            match s {
                cobyla::SuccessStatus::MaxEvalReached => "budget",
                cobyla::SuccessStatus::XtolReached => "rho_end",
                _ => "other",
            },
            format!("{s:?}"),
        ),
        Err((s, x, f)) => (x, f, "failed", format!("{s:?}")),
    };
    #[cfg(feature = "capture")]
    eprintln!("cobyla status: {status}");
    let _ = status;
    Outcome {
        x,
        f,
        objective_calls: 0,
        constraint_calls: 0,
        wrapper_calls: None,
        iterations: None,
        stop,
        native_status: None,
        current: None,
    }
}

#[cfg(feature = "capture")]
thread_local! {
    static WORK: std::cell::RefCell<std::collections::BTreeMap<&'static str, usize>> = const { std::cell::RefCell::new(std::collections::BTreeMap::new()) };
    static POINTS: std::cell::RefCell<Vec<serde_json::Value>> = const { std::cell::RefCell::new(Vec::new()) };
}
#[cfg(feature = "capture")]
fn record(name: &'static str) {
    WORK.with(|w| *w.borrow_mut().entry(name).or_default() += 1);
}
#[cfg(feature = "capture")]
fn record_point(case: Case, x: &[f64], nf: usize) {
    let f = case.objective(x);
    let mut c = vec![0.0; case.m()];
    case.constraints(x, &mut c);
    let cv = c.into_iter().fold(0.0, f64::max);
    POINTS.with(|p| {
        p.borrow_mut().push(serde_json::json!({
            "nf": nf, "x": x, "f": f, "cv": cv, "quality": case.quality(f, cv)
        }))
    });
}
fn diagnostics() -> serde_json::Value {
    #[cfg(feature = "capture")]
    {
        serde_json::json!({"basin": WORK.with(|w| w.borrow().clone()),
        "cobyla": cobyla::gap_counts(), "points": POINTS.with(|p| p.borrow().clone())})
    }
    #[cfg(not(feature = "capture"))]
    {
        serde_json::Value::Null
    }
}
