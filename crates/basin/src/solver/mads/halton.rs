//! Deterministic Halton low-discrepancy sequence for OrthoMADS poll directions.
//!
//! OrthoMADS (Abramson, Audet, Dennis & Le Digabel 2009, §3.1) replaces LTMADS's
//! random direction matrix with a deterministic construction seeded by the Halton
//! sequence. This module provides the radical-inverse generator and the
//! per-iteration Halton direction `u_t ∈ [0,1]^n`, plus the prime helper for the
//! Halton seed `t₀ = p_n`. Every value is a pure function of `(t, n)`—no RNG,
//! fully reproducible (see `references/orthomads-2009/NOTES.md`).

/// The radical-inverse function `u_{t,p} = Σ_r a_{t,r,p} · p^{-(1+r)}`, where the
/// `a_{t,r,p}` are the base-`p` digits of `t` (OrthoMADS §3.1).
pub(crate) fn radical_inverse(t: usize, base: usize) -> f64 {
    debug_assert!(base >= 2, "Halton base must be a prime ≥ 2");
    let mut result = 0.0;
    let mut f = 1.0 / base as f64;
    let mut i = t;
    while i > 0 {
        result += f * (i % base) as f64;
        i /= base;
        f /= base as f64;
    }
    result
}

/// The `t`th Halton direction `u_t = (u_{t,p_1}, …, u_{t,p_n}) ∈ [0,1]^n`, using
/// the first `n` primes as bases (OrthoMADS §3.1).
pub(crate) fn halton_direction(t: usize, n: usize) -> Vec<f64> {
    first_n_primes(n)
        .into_iter()
        .map(|p| radical_inverse(t, p))
        .collect()
}

/// The first `n` prime numbers `p_1 = 2, p_2 = 3, p_3 = 5, …`.
pub(crate) fn first_n_primes(n: usize) -> Vec<usize> {
    let mut primes: Vec<usize> = Vec::with_capacity(n);
    let mut candidate = 2;
    while primes.len() < n {
        if primes.iter().all(|&p| candidate % p != 0) {
            primes.push(candidate);
        }
        candidate += 1;
    }
    primes
}

/// The Halton seed `t₀ = p_n`, the `n`th prime. OrthoMADS starts the sequence at
/// `p_n` to remove the low-index correlation between dimensions (§3.1).
pub(crate) fn halton_seed(n: usize) -> usize {
    *first_n_primes(n)
        .last()
        .expect("n ≥ 1 for a non-empty problem")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    #[test]
    fn first_primes() {
        assert_eq!(first_n_primes(4), vec![2, 3, 5, 7]);
        assert_eq!(halton_seed(2), 3); // p_2 = 3 (used by Figure 1, n=2)
        assert_eq!(halton_seed(4), 7); // p_4 = 7 (used by Table 2, n=4)
    }

    /// Table 1 of OrthoMADS (n = 4, t = 0..7): exact Halton directions.
    #[test]
    fn halton_table_1() {
        // (t, [u_{t,2}, u_{t,3}, u_{t,5}, u_{t,7}]) read off Table 1.
        let rows: [(usize, [f64; 4]); 8] = [
            (0, [0.0, 0.0, 0.0, 0.0]),
            (1, [1.0 / 2.0, 1.0 / 3.0, 1.0 / 5.0, 1.0 / 7.0]),
            (2, [1.0 / 4.0, 2.0 / 3.0, 2.0 / 5.0, 2.0 / 7.0]),
            (3, [3.0 / 4.0, 1.0 / 9.0, 3.0 / 5.0, 3.0 / 7.0]),
            (4, [1.0 / 8.0, 4.0 / 9.0, 4.0 / 5.0, 4.0 / 7.0]),
            (5, [5.0 / 8.0, 7.0 / 9.0, 1.0 / 25.0, 5.0 / 7.0]),
            (6, [3.0 / 8.0, 2.0 / 9.0, 6.0 / 25.0, 6.0 / 7.0]),
            (7, [7.0 / 8.0, 5.0 / 9.0, 11.0 / 25.0, 1.0 / 49.0]),
        ];
        for (t, expected) in rows {
            let u = halton_direction(t, 4);
            for (got, want) in u.iter().zip(expected.iter()) {
                approx(*got, *want);
            }
        }
    }

    /// The footnote example: u_{5,3} = 1·3^-2 + 2·3^-1 = 7/9.
    #[test]
    fn radical_inverse_footnote() {
        approx(radical_inverse(5, 3), 7.0 / 9.0);
    }
}
