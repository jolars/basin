// `!(a > b)` comparisons below are deliberate NaN-preserving ports of PRIMA's
// `~(a > b)` masks (`a <= b` would exclude NaN, changing tie-breaking); keep them.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

//! Return-point filter for COBYLA (PRIMA `selectx`/`savefilt`/`isbetter`).
//!
//! COBYLA is a trust-region method on an L-infinity merit function, but the
//! point it *returns* is chosen from a filter of mutually non-dominated
//! `(f, cstrv)` pairs, so a low-`F` infeasible iterate never shadows a feasible
//! one. `isbetter` is the domination test; `savefilt` inserts a new point
//! (evicting dominated ones, or the worst by merit when full); `selectx` picks
//! the final index by the merit `φ = f + cweight · max(cstrv − ctol, 0)`.

use crate::core::math::Scalar;

fn funcmax<F: Scalar>() -> F {
    // PRIMA FUNCMAX = 2^100 (a large sub-overflow sentinel); CONSTRMAX = FUNCMAX.
    F::from_f64(2f64.powi(100)).unwrap()
}

/// Is `(f1, c1)` strictly better than `(f2, c2)`? `c*` are constraint
/// violations (`≥ 0`); `ctol` is the feasibility tolerance. PRIMA `isbetter`.
pub(crate) fn isbetter<F: Scalar>(f1: F, c1: F, f2: F, c2: F, ctol: F) -> bool {
    let realmax = F::max_value();
    let eps = F::epsilon();
    let ten = F::from_f64(10.0).unwrap();
    let constrmax = funcmax::<F>();
    let mut better = false;
    better = better || ((f1.is_nan() || c1.is_nan()) && !(f2.is_nan() || c2.is_nan()));
    better = better || (f1 < f2 && c1 <= c2);
    better = better || (f1 <= f2 && c1 < c2);
    let cref = ten * eps.max(ctol.min(F::from_f64(1.0e-2).unwrap() * constrmax));
    better = better || (f1 < realmax && c1 <= ctol && (c2 > ctol.max(cref) || c2.is_nan()));
    better
}

/// Insert `(x, f, cstrv, constr)` into the filter unless dominated; evict points
/// it dominates, or the merit-worst point if the filter is full. PRIMA
/// `savefilt`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn savefilt<F: Scalar>(
    x: &[F],
    f: F,
    cstrv: F,
    constr: &[F],
    n: usize,
    m: usize,
    ctol: F,
    cweight: F,
    maxfilt: usize,
    nfilt: &mut usize,
    xfilt: &mut [F],
    ffilt: &mut [F],
    cfilt: &mut [F],
    confilt: &mut [F],
) {
    let zero = F::zero();
    let realmax = F::max_value();
    // Return immediately if any filter column is better than X.
    for k in 0..*nfilt {
        if isbetter(ffilt[k], cfilt[k], f, cstrv, ctol) {
            return;
        }
    }
    // Decide which columns to keep (those X does not dominate).
    let mut keep: Vec<bool> = (0..*nfilt)
        .map(|k| !isbetter(f, cstrv, ffilt[k], cfilt[k], ctol))
        .collect();

    let count_keep = keep.iter().filter(|&&b| b).count();
    if count_keep == maxfilt {
        // Evict the merit-worst column.
        let phi: Vec<F> = (0..*nfilt)
            .map(|k| {
                let cs = (cfilt[k] - ctol).max(zero);
                if cweight <= zero {
                    ffilt[k]
                } else if cweight.is_infinite() {
                    cs
                } else {
                    ffilt[k].max(-realmax) + cweight * cs
                }
            })
            .collect();
        let phimax = phi.iter().cloned().fold(F::neg_infinity(), F::max);
        let cref = (0..*nfilt)
            .filter(|&k| phi[k] >= phimax)
            .map(|k| (cfilt[k] - ctol).max(zero))
            .fold(F::neg_infinity(), F::max);
        let fref = (0..*nfilt)
            .filter(|&k| (cfilt[k] - ctol).max(zero) >= cref)
            .map(|k| ffilt[k])
            .fold(F::neg_infinity(), F::max);
        // kworst = first index maximizing cfilt among ffilt >= fref.
        let cmax = (0..*nfilt)
            .filter(|&k| ffilt[k] >= fref)
            .map(|k| cfilt[k])
            .fold(F::neg_infinity(), F::max);
        let kworst = (0..*nfilt)
            .find(|&k| ffilt[k] >= fref && !(cfilt[k] < cmax))
            .unwrap_or(0);
        keep[kworst] = false;
    }

    // Compact kept columns, then append the new point.
    let mut new_n = 0usize;
    for k in 0..*nfilt {
        if keep[k] {
            if new_n != k {
                for r in 0..n {
                    xfilt[r + new_n * n] = xfilt[r + k * n];
                }
                ffilt[new_n] = ffilt[k];
                cfilt[new_n] = cfilt[k];
                for r in 0..m {
                    confilt[r + new_n * m] = confilt[r + k * m];
                }
            }
            new_n += 1;
        }
    }
    for r in 0..n {
        xfilt[r + new_n * n] = x[r];
    }
    ffilt[new_n] = f;
    cfilt[new_n] = cstrv;
    for r in 0..m {
        confilt[r + new_n * m] = constr[r];
    }
    *nfilt = new_n + 1;
}

/// Select the index of the point to return from the filter, by the merit
/// function `φ = f + cweight · max(cstrv − ctol, 0)`. PRIMA `selectx`.
pub(crate) fn selectx<F: Scalar>(ffilt: &[F], cfilt: &[F], cweight: F, ctol: F) -> usize {
    let nhist = ffilt.len();
    let zero = F::zero();
    let eps = F::epsilon();
    let two = F::from_f64(2.0).unwrap();
    let realmax = F::max_value();
    let funcmax = funcmax::<F>();
    let constrmax = funcmax;

    // Reference bounds (tiering against the large sentinels).
    let (mut fref, mut cref) = if (0..nhist).any(|k| ffilt[k] < funcmax && cfilt[k] < constrmax) {
        (funcmax, constrmax)
    } else if (0..nhist).any(|k| ffilt[k] < realmax && cfilt[k] < constrmax) {
        (realmax, constrmax)
    } else if (0..nhist).any(|k| ffilt[k] < funcmax && cfilt[k] < realmax) {
        (funcmax, realmax)
    } else {
        (realmax, realmax)
    };

    if !(0..nhist).any(|k| ffilt[k] < fref && cfilt[k] < cref) {
        return nhist - 1;
    }

    let cshift: Vec<F> = (0..nhist).map(|k| (cfilt[k] - ctol).max(zero)).collect();
    let cmin = (0..nhist)
        .filter(|&k| ffilt[k] < fref)
        .map(|k| cshift[k])
        .fold(F::infinity(), F::min);
    let cref2 = eps.max(two * cmin);
    let phi: Vec<F> = (0..nhist)
        .map(|k| {
            if cweight <= zero {
                ffilt[k]
            } else if cweight.is_infinite() {
                cshift[k]
            } else {
                ffilt[k].max(-realmax) + cweight * cshift[k]
            }
        })
        .collect();
    let phimin = (0..nhist)
        .filter(|&k| ffilt[k] < fref && cshift[k] <= cref2)
        .map(|k| phi[k])
        .fold(F::infinity(), F::min);
    cref = (0..nhist)
        .filter(|&k| ffilt[k] < fref && phi[k] <= phimin)
        .map(|k| cshift[k])
        .fold(F::infinity(), F::min);
    fref = (0..nhist)
        .filter(|&k| cshift[k] <= cref)
        .map(|k| ffilt[k])
        .fold(F::infinity(), F::min);
    // kopt = first index minimizing cfilt among ffilt <= fref.
    let cmin2 = (0..nhist)
        .filter(|&k| ffilt[k] <= fref)
        .map(|k| cfilt[k])
        .fold(F::infinity(), F::min);
    (0..nhist)
        .find(|&k| ffilt[k] <= fref && !(cfilt[k] > cmin2))
        .unwrap_or(nhist - 1)
}

/// Moderate a value into `[−FUNCMAX, FUNCMAX]`, mapping `NaN` to `FUNCMAX`
/// (PRIMA's `moderatef`/`moderatec` extreme barrier). Keeps the filter free of
/// `NaN`/`+Inf` and lets soft-rejected points (`+Inf` cost) participate sanely.
pub(crate) fn moderatef<F: Scalar>(f: F) -> F {
    let fmax = funcmax::<F>();
    if f.is_nan() {
        fmax
    } else {
        f.min(fmax).max(-fmax)
    }
}
