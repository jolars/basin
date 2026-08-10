//! OrthoMADS poll-direction geometry (Abramson, Audet, Dennis & Le Digabel
//! 2009, §3.2–§3.4): the adjusted Halton direction `q_{t,ℓ}`, the
//! scaled-Householder integer basis `H`, the maximal positive basis
//! `D_k = [H −H]`, and the ℓ↔(Δᵐ, Δᵖ) mesh and poll sizes.
//!
//! All arithmetic is flat `f64`/`i64` and depends only on `(t, ℓ, n)`: it is
//! backend- and scalar-`F`-independent. The direction outputs are exact integer
//! vectors, so poll points `x_k + Δᵐ·d` land exactly on the mesh. Verified
//! against the paper's Tables 1 and 2 and Figure 1 (see the golden tests below and
//! `references/orthomads-2009/NOTES.md`).

use super::halton::halton_direction;

/// Mesh size `Δᵐ = min{1, 4^{-ℓ}}` and poll size `Δᵖ = 2^{-ℓ}` for mesh index
/// `ℓ` (OrthoMADS eq. (1), **signed ℓ**). Init `ℓ = 0 ⇒ Δᵐ = Δᵖ = 1`; an
/// unsuccessful iteration does `ℓ ← ℓ+1` (refine), a success `ℓ ← ℓ−1`
/// (coarsen). Powers of two, so exact in `f64`/`f32`.
pub(crate) fn mesh_and_poll_size(ell: i32) -> (f64, f64) {
    let poll = 2f64.powi(-ell); // Δᵖ = 2^{-ℓ}
    let mesh = 4f64.powi(-ell).min(1.0); // Δᵐ = min{1, 4^{-ℓ}}
    (mesh, poll)
}

/// The adjusted Halton direction `q_{t,ℓ} ∈ ℤⁿ` (OrthoMADS §3.2): scale and
/// round the centered Halton direction `2u_t − e` to an integer vector whose
/// norm is as close as possible to `2^{|ℓ|/2}` without exceeding it.
///
/// `q_t(α) = round(α (2u_t−e)/‖2u_t−e‖)` (round half away from zero). `‖q_t(α)‖`
/// is a nondecreasing step function whose steps occur at the breakpoints
/// `(2j+1)/(2|c_i|)`; `q_{t,ℓ}` is its value on the highest plateau with
/// `‖q_t(α)‖ ≤ 2^{|ℓ|/2}` (problem (5), solved exactly via those breakpoints).
pub(crate) fn adjusted_halton_direction(
    t: usize,
    ell: i32,
    n: usize,
) -> Vec<i64> {
    let u = halton_direction(t, n);
    // Centered direction w = 2u − e and its unit vector c = w/‖w‖.
    let w: Vec<f64> = u.iter().map(|&ui| 2.0 * ui - 1.0).collect();
    let w_norm = w.iter().map(|wi| wi * wi).sum::<f64>().sqrt();
    // 2u_t − e = 0 only for n = t = 1, which the algorithm never uses (it seeds
    // the Halton index at t₀ = p_n ≥ 2).
    debug_assert!(w_norm > 0.0, "2u_t − e = 0 only for n = t = 1");
    let c: Vec<f64> = w.iter().map(|wi| wi / w_norm).collect();

    let target = 2f64.powf(f64::from(ell.unsigned_abs()) / 2.0); // 2^{|ℓ|/2}
    let target_sq = target * target; // 2^{|ℓ|}, exact for the ℓ range we hit

    let q_at = |alpha: f64| -> Vec<i64> {
        c.iter().map(|ci| (alpha * ci).round() as i64).collect()
    };
    let norm_sq = |q: &[i64]| -> i64 { q.iter().map(|qi| qi * qi).sum() };

    // Beyond α where the largest single component alone exceeds the target,
    // ‖q‖ > target for certain: bound the breakpoint enumeration there.
    let c_max = c.iter().fold(0.0_f64, |m, ci| m.max(ci.abs()));
    let alpha_ub = (target + 0.5) / c_max;

    let mut breakpoints: Vec<f64> = Vec::new();
    for ci in &c {
        let aci = ci.abs();
        if aci == 0.0 {
            continue;
        }
        let mut j = 0u64;
        loop {
            let bp = (2 * j + 1) as f64 / (2.0 * aci);
            if bp > alpha_ub {
                break;
            }
            breakpoints.push(bp);
            j += 1;
        }
    }
    breakpoints.sort_by(|a, b| a.partial_cmp(b).expect("finite breakpoints"));

    // Walk plateaus low → high. `q` is constant on `[b_k, b_{k+1})`; evaluate
    // just past each breakpoint (a tiny nudge avoids fp round-at-the-step
    // ambiguity). Keep the highest-norm `q` that is still feasible; stop at the
    // first infeasible one (the norm is monotone).
    let mut best = vec![0i64; n]; // plateau [0, b_1): q = 0
    for &bp in &breakpoints {
        let q = q_at(bp + bp * 1e-12);
        if (norm_sq(&q) as f64) <= target_sq {
            best = q;
        } else {
            break;
        }
    }
    best
}

/// The scaled-Householder integer basis `H = ‖q‖²·Iₙ − 2·q·qᵀ` (OrthoMADS eq.
/// (6)). Returns the `n` columns; each is an integer vector of norm `‖q‖²`, and
/// the columns are mutually orthogonal.
pub(crate) fn householder_basis(q: &[i64]) -> Vec<Vec<i64>> {
    let n = q.len();
    let q_norm_sq: i64 = q.iter().map(|qi| qi * qi).sum();
    // Column j = H e_j = ‖q‖²·e_j − 2·q_j·q.
    (0..n)
        .map(|j| {
            (0..n)
                .map(|i| {
                    let diag = if i == j { q_norm_sq } else { 0 };
                    diag - 2 * q[j] * q[i]
                })
                .collect()
        })
        .collect()
}

/// The OrthoMADS poll set `D_k = [H −H]` (2n directions, the maximal positive
/// basis) for Halton index `t`, mesh index `ℓ`, dimension `n` (OrthoMADS §3.4).
pub(crate) fn poll_directions(t: usize, ell: i32, n: usize) -> Vec<Vec<i64>> {
    let q = adjusted_halton_direction(t, ell, n);
    let h = householder_basis(&q);
    let mut dirs: Vec<Vec<i64>> = Vec::with_capacity(2 * n);
    dirs.extend(h.iter().cloned());
    dirs.extend(h.iter().map(|col| col.iter().map(|&x| -x).collect()));
    dirs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm_sq(q: &[i64]) -> i64 {
        q.iter().map(|qi| qi * qi).sum()
    }

    fn dot(a: &[i64], b: &[i64]) -> i64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn mesh_sizes_eq_1() {
        assert_eq!(mesh_and_poll_size(0), (1.0, 1.0));
        assert_eq!(mesh_and_poll_size(3), (1.0 / 64.0, 1.0 / 8.0)); // Figure 1
        // After successes ℓ goes negative: mesh capped at 1, poll grows.
        assert_eq!(mesh_and_poll_size(-2), (1.0, 4.0));
    }

    /// Table 2 of OrthoMADS (n = 4, eight pairs with t = ℓ + 7, t₀ = p_4 = 7):
    /// exact adjusted Halton directions and their squared norms.
    #[test]
    fn adjusted_direction_table_2() {
        let rows: [(usize, i32, [i64; 4], i64); 8] = [
            (7, 0, [0, 0, 0, -1], 1),
            (8, 1, [-1, 1, 0, 0], 2),
            (9, 2, [0, -1, 1, -1], 3),
            (10, 3, [-1, -1, -2, 0], 6),
            (11, 4, [2, 2, -2, 1], 13),
            (12, 5, [-3, -4, 0, 2], 29),
            (13, 6, [3, 0, 3, 6], 54),
            (14, 7, [-1, 5, 6, -8], 126),
        ];
        for (t, ell, q_expected, nsq_expected) in rows {
            let q = adjusted_halton_direction(t, ell, 4);
            assert_eq!(q, q_expected.to_vec(), "q_{{{t},{ell}}}");
            assert_eq!(norm_sq(&q), nsq_expected, "‖q_{{{t},{ell}}}‖²");
        }
    }

    /// Figure 1 of OrthoMADS, end to end: n = 2, (t, ℓ) = (6, 3).
    #[test]
    fn figure_1_full_chain() {
        let q = adjusted_halton_direction(6, 3, 2);
        assert_eq!(q, vec![-1, -2]);

        let h = householder_basis(&q);
        // H e_1 = (3, −4), H e_2 = (−4, −3).
        assert_eq!(h[0], vec![3, -4]);
        assert_eq!(h[1], vec![-4, -3]);

        // Columns orthogonal, each of norm ‖q‖² = 5 (so squared norm 25 = ‖q‖⁴).
        assert_eq!(dot(&h[0], &h[1]), 0);
        assert_eq!(norm_sq(&h[0]), 25);
        assert_eq!(norm_sq(&h[1]), 25);
        assert_eq!((norm_sq(&h[0]) as f64).sqrt(), 5.0); // ‖H e_j‖ = ‖q‖²

        // Poll set D_k = [H −H] has 2n = 4 directions.
        let dirs = poll_directions(6, 3, 2);
        assert_eq!(dirs.len(), 4);
        assert_eq!(dirs[2], vec![-3, 4]);
        assert_eq!(dirs[3], vec![4, 3]);

        // Every d ∈ D_k satisfies Δᵐ‖d‖ = 5/64 < Δᵖ = 1/8 (caption).
        let (mesh, poll) = mesh_and_poll_size(3);
        for d in &dirs {
            let dist = mesh * (norm_sq(d) as f64).sqrt();
            assert!((dist - 5.0 / 64.0).abs() < 1e-12, "Δᵐ‖d‖ = {dist}");
            assert!(dist < poll);
        }
    }

    /// The columns of `[H −H]` are mutually orthogonal (any non-collinear pair),
    /// so the cosine measure is the optimal `1/√n` (OrthoMADS eq. (2)).
    #[test]
    fn orthogonal_basis_in_higher_dim() {
        let dirs = poll_directions(11, 4, 4);
        assert_eq!(dirs.len(), 8);
        let nsq = norm_sq(&dirs[0]);
        for i in 0..4 {
            assert_eq!(norm_sq(&dirs[i]), nsq, "equal column norms");
            for j in 0..4 {
                if i != j {
                    assert_eq!(
                        dot(&dirs[i], &dirs[j]),
                        0,
                        "columns {i},{j} orthogonal"
                    );
                }
            }
        }
    }
}
