//! Scalar convergence controls. Root and minimum records use separate schemas.

use basin::{
    Brent, BrentDerivative, BrentRoot, FirstOrderState, GoldenSection,
    HalleyRoot, NewtonRoot, PointState, SecantRoot, Toms748Root,
};
use competitor_bench::convergence::{scalar, trace};
use std::{cell::Cell, convert::Infallible, time::Instant};

macro_rules! run_precision {
    ($f:ty, $roots:expr, $validation:expr, $list:expr) => {
        for case in scalar::cases().into_iter().filter(|c| {
            c.is_root() == $roots && ($list || $validation || !c.validation)
        }) {
            let partition = if case.validation {
                "validation"
            } else {
                "development"
            };
            if $list {
                println!(
                    "{}",
                    trace::csv([
                        case.name.into(),
                        case.family.into(),
                        partition.into(),
                        stringify!($f).into(),
                        case.reference.to_string(),
                        format!("{:?}", case.intervals)
                    ])
                );
                continue;
            }
            for (start, &[lo, hi]) in case.intervals.iter().enumerate() {
                let (lo, hi) = (lo as $f, hi as $f);
                if $roots {
                    for solver in [
                        "BrentRoot",
                        "SecantRoot",
                        "NewtonRoot",
                        "HalleyRoot",
                        "Toms748Root",
                    ] {
                        let calls = Cell::new([0_u64; 3]);
                        let count = |i: usize| {
                            let mut c = calls.get();
                            c[i] += 1;
                            calls.set(c);
                        };
                        let f = |x: $f| {
                            count(0);
                            Ok::<_, Infallible>(case.value(x))
                        };
                        let df = |x: $f| {
                            count(1);
                            Ok::<_, Infallible>(case.derivative(x))
                        };
                        let ddf = |x: $f| {
                            count(2);
                            Ok::<_, Infallible>(case.second(x))
                        };
                        let began = Instant::now();
                        let result = match solver {
                            "BrentRoot" => BrentRoot::new(lo, hi)
                                .solve(f)
                                .map_err(|e| format!("{e:?}")),
                            "SecantRoot" => SecantRoot::new(lo, hi)
                                .solve(f)
                                .map_err(|e| format!("{e:?}")),
                            "NewtonRoot" => NewtonRoot::new(lo, hi)
                                .solve(f, df)
                                .map_err(|e| format!("{e:?}")),
                            "HalleyRoot" => HalleyRoot::new(lo, hi)
                                .solve(f, df, ddf)
                                .map_err(|e| format!("{e:?}")),
                            _ => Toms748Root::new(lo, hi)
                                .solve(f)
                                .map_err(|e| format!("{e:?}")),
                        };
                        let seconds = began.elapsed().as_secs_f64();
                        let mut fields = vec![
                            case.name.into(),
                            case.family.into(),
                            partition.into(),
                            stringify!($f).into(),
                            (start + 1).to_string(),
                            solver.into(),
                        ];
                        match result {
                            Ok(result) => {
                                let root = result.root() as f64;
                                let (a, b) = result.bracket();
                                assert_eq!(
                                    calls.get(),
                                    [
                                        result.function_evals(),
                                        result.derivative_evals(),
                                        result.second_derivative_evals()
                                    ]
                                );
                                fields.extend([
                                    format!("{:?}", result.reason()),
                                    result.iterations().to_string(),
                                    root.to_string(),
                                    case.value(root).abs().to_string(),
                                    (root - case.reference).abs().to_string(),
                                    (b as f64 - a as f64).to_string(),
                                    result.value().to_string(),
                                ]);
                            }
                            Err(error) => {
                                fields.extend([
                                    error,
                                    String::new(),
                                    "NaN".into(),
                                    "NaN".into(),
                                    "NaN".into(),
                                    "NaN".into(),
                                    "NaN".into(),
                                ]);
                            }
                        }
                        fields.extend(calls.get().map(|n| n.to_string()));
                        fields.extend([
                            calls.get().iter().sum::<u64>().to_string(),
                            seconds.to_string(),
                        ]);
                        println!("{}", trace::csv(fields));
                    }
                } else {
                    macro_rules! solve {
                        ($name:expr, $solver:expr, $state:expr) => {{
                            let fixture =
                                scalar::Fixture::<$f>::new(case.clone(), start);
                            let work = fixture.work.clone();
                            let rows = trace::run(
                                fixture,
                                $solver,
                                $state,
                                work,
                                200,
                                2000,
                                |p| {
                                    let x = *p as f64;
                                    trace::Quality {
                                        param: vec![x],
                                        cost: case.value(x),
                                        difference: case.value(x)
                                            - case.value(case.reference),
                                        gradient_inf: case.derivative(x).abs(),
                                        violation: (lo as f64 - x)
                                            .max(x - hi as f64)
                                            .max(0.0),
                                        parameter_error: (x - case.reference)
                                            .abs(),
                                    }
                                },
                            );
                            trace::print_rows(
                                &[
                                    "scalar_minimum".into(),
                                    case.name.into(),
                                    case.family.into(),
                                    partition.into(),
                                    stringify!($f).into(),
                                    "scalar".into(),
                                    (start + 1).to_string(),
                                    $name.into(),
                                    String::new(),
                                ],
                                &rows,
                                false,
                            );
                        }};
                    }
                    let middle = (lo + hi) / 2.0;
                    solve!(
                        "Brent",
                        Brent::new(),
                        PointState::<$f, $f>::new(middle)
                    );
                    solve!(
                        "GoldenSection",
                        GoldenSection::new(),
                        PointState::<$f, $f>::new(middle)
                    );
                    solve!(
                        "BrentDerivative",
                        BrentDerivative::new(),
                        FirstOrderState::<$f, $f>::new(middle)
                    );
                }
            }
        }
    };
}

fn main() {
    let mut validation = false;
    let mut roots = true;
    let mut list = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--include-validation" => validation = true,
            "--minima" => roots = false,
            "--list" => list = true,
            "--help" => {
                println!(
                    "convergence_scalar [--minima] [--list] [--include-validation]\nBoth precisions and two intervals per case are always included."
                );
                return;
            }
            _ => {
                eprintln!("unknown option {arg}");
                std::process::exit(2);
            }
        }
    }
    if list {
        println!("case,family,partition,precision,reference,intervals");
    } else if roots {
        println!(
            "case,family,partition,precision,start,solver,termination,iterations,root,checked_residual,position_error,bracket_width,native_value,function_evals,derivative_evals,second_derivative_evals,callback_evals,solver_seconds"
        );
    } else {
        println!("{}", trace::HEADER);
    }
    run_precision!(f32, roots, validation, list);
    run_precision!(f64, roots, validation, list);
}
