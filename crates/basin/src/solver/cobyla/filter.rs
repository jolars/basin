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
    better = better
        || ((f1.is_nan() || c1.is_nan()) && !(f2.is_nan() || c2.is_nan()));
    better = better || (f1 < f2 && c1 <= c2);
    better = better || (f1 <= f2 && c1 < c2);
    let cref =
        ten * eps.max(ctol.min(F::from_f64(1.0e-2).unwrap() * constrmax));
    better = better
        || (f1 < realmax && c1 <= ctol && (c2 > ctol.max(cref) || c2.is_nan()));
    better
}

/// Insert `(x, f, cstrv)` into the filter unless dominated; evict points
/// it dominates, or the merit-worst point if the filter is full. PRIMA
/// `savefilt`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn savefilt<F: Scalar>(
    x: &[F],
    f: F,
    cstrv: F,
    n: usize,
    ctol: F,
    cweight: F,
    maxfilt: usize,
    nfilt: &mut usize,
    xfilt: &mut Vec<F>,
    ffilt: &mut Vec<F>,
    cfilt: &mut Vec<F>,
) {
    let zero = F::zero();
    let realmax = F::max_value();
    // Return immediately if any filter column is better than X.
    for k in 0..*nfilt {
        if isbetter(ffilt[k], cfilt[k], f, cstrv, ctol) {
            return;
        }
    }
    let count_keep = (0..*nfilt)
        .filter(|&k| !isbetter(f, cstrv, ffilt[k], cfilt[k], ctol))
        .count();
    let kworst = if count_keep == maxfilt {
        // Evict the merit-worst column.
        let phi = |k: usize| {
            let cs = (cfilt[k] - ctol).max(zero);
            if cweight <= zero {
                ffilt[k]
            } else if cweight.is_infinite() {
                cs
            } else {
                ffilt[k].max(-realmax) + cweight * cs
            }
        };
        let phimax = (0..*nfilt).map(phi).fold(F::neg_infinity(), F::max);
        let cref = (0..*nfilt)
            .filter(|&k| phi(k) >= phimax)
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
        Some(kworst)
    } else {
        None
    };

    // Compact kept columns, then append the new point.
    let mut new_n = 0usize;
    for k in 0..*nfilt {
        if Some(k) != kworst && !isbetter(f, cstrv, ffilt[k], cfilt[k], ctol) {
            if new_n != k {
                for r in 0..n {
                    xfilt[r + new_n * n] = xfilt[r + k * n];
                }
                ffilt[new_n] = ffilt[k];
                cfilt[new_n] = cfilt[k];
            }
            new_n += 1;
        }
    }
    xfilt.truncate(new_n * n);
    ffilt.truncate(new_n);
    cfilt.truncate(new_n);
    // Most solves retain only a few nondominated points. Grow geometrically,
    // bounded by MAXFILT, instead of allocating the full history at startup.
    let capacity = (new_n + 1).max(4).next_power_of_two().min(maxfilt);
    for (values, width) in
        [(&mut *xfilt, n), (&mut *ffilt, 1), (&mut *cfilt, 1)]
    {
        if values.capacity() < (new_n + 1) * width {
            values.reserve_exact(capacity * width - values.len());
        }
    }
    xfilt.extend_from_slice(x);
    ffilt.push(f);
    cfilt.push(cstrv);
    *nfilt = new_n + 1;
}

/// Select the index of the point to return from the filter, by the merit
/// function `φ = f + cweight · max(cstrv − ctol, 0)`. PRIMA `selectx`.
pub(crate) fn selectx<F: Scalar>(
    ffilt: &[F],
    cfilt: &[F],
    cweight: F,
    ctol: F,
) -> usize {
    let nhist = ffilt.len();
    let zero = F::zero();
    let eps = F::epsilon();
    let two = F::from_f64(2.0).unwrap();
    let realmax = F::max_value();
    let funcmax = funcmax::<F>();
    let constrmax = funcmax;

    // Reference bounds (tiering against the large sentinels).
    let (mut fref, mut cref) = if (0..nhist)
        .any(|k| ffilt[k] < funcmax && cfilt[k] < constrmax)
    {
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

    let cshift = |k: usize| (cfilt[k] - ctol).max(zero);
    let cmin = (0..nhist)
        .filter(|&k| ffilt[k] < fref)
        .map(cshift)
        .fold(F::infinity(), F::min);
    let cref2 = eps.max(two * cmin);
    let phi = |k: usize| {
        if cweight <= zero {
            ffilt[k]
        } else if cweight.is_infinite() {
            cshift(k)
        } else {
            ffilt[k].max(-realmax) + cweight * cshift(k)
        }
    };
    let phimin = (0..nhist)
        .filter(|&k| ffilt[k] < fref && cshift(k) <= cref2)
        .map(phi)
        .fold(F::infinity(), F::min);
    cref = (0..nhist)
        .filter(|&k| ffilt[k] < fref && phi(k) <= phimin)
        .map(cshift)
        .fold(F::infinity(), F::min);
    fref = (0..nhist)
        .filter(|&k| cshift(k) <= cref)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Filter {
        count: usize,
        x: Vec<f64>,
        f: Vec<f64>,
        c: Vec<f64>,
    }

    impl Filter {
        fn save(&mut self, id: f64, f: f64, c: f64, limit: usize, weight: f64) {
            savefilt(
                &[id, -id],
                f,
                c,
                2,
                0.0,
                weight,
                limit,
                &mut self.count,
                &mut self.x,
                &mut self.f,
                &mut self.c,
            );
            assert_eq!(self.x.len(), 2 * self.count);
            assert_eq!(self.f.len(), self.count);
            assert_eq!(self.c.len(), self.count);
            assert!(self.count <= limit);
            assert!(self.x.capacity() <= 2 * limit);
            assert!(self.f.capacity() <= limit);
            assert!(self.c.capacity() <= limit);
        }
    }

    #[test]
    fn lazy_filter_saturates_evicts_and_compacts() {
        let mut filter = Filter::default();
        filter.save(0.0, 3000.0, 1.0, 2000, 2.0);
        assert!(filter.x.capacity() < 4000);
        for k in 1..2002 {
            filter.save(k as f64, 3000.0 - k as f64, k as f64 + 1.0, 2000, 2.0);
        }
        assert_eq!(filter.count, 2000);
        for k in 0..1999 {
            assert_eq!(&filter.x[2 * k..2 * k + 2], &[k as f64, -(k as f64)]);
        }
        assert_eq!(&filter.x[3998..], &[2001.0, -2001.0]);
        assert_eq!(selectx(&filter.f, &filter.c, 2.0, 0.0), 0);

        filter.save(-1.0, 0.0, 0.0, 2000, 2.0);
        assert_eq!(filter.count, 1);
        assert_eq!(filter.x, [-1.0, 1.0]);
        filter.save(-2.0, 1.0, 0.0, 2000, 2.0);
        assert_eq!(filter.count, 1);
    }

    #[test]
    fn filter_ties_evict_and_select_the_first_matching_point() {
        for weight in [0.0, 1.0, f64::INFINITY] {
            let mut filter = Filter::default();
            for k in 0..4 {
                filter.save(k as f64, 1.0, 1.0, 3, weight);
            }
            assert_eq!(filter.x, [1.0, -1.0, 2.0, -2.0, 3.0, -3.0]);
            assert_eq!(selectx(&filter.f, &filter.c, weight, 0.0), 0);
        }
    }

    #[test]
    fn moderated_nonfinite_values_do_not_hide_a_feasible_point() {
        let sentinel = funcmax::<f64>();
        assert_eq!(moderatef(f64::NAN), sentinel);
        assert_eq!(moderatef(f64::INFINITY), sentinel);
        assert_eq!(moderatef(f64::NEG_INFINITY), -sentinel);
        let mut filter = Filter::default();
        filter.save(0.0, sentinel, sentinel, 3, 1e8);
        filter.save(1.0, -sentinel, 1.0, 3, 1e8);
        filter.save(2.0, 1.0, 0.0, 3, 1e8);
        let best = selectx(&filter.f, &filter.c, 1e8, 0.0);
        assert_eq!(filter.x[2 * best], 2.0);
        assert_eq!(
            selectx(&[sentinel, sentinel], &[sentinel, sentinel], 1e8, 0.0),
            0
        );
        assert_eq!(selectx(&[f64::MAX; 2], &[f64::MAX; 2], 1e8, 0.0), 1);
    }
}
