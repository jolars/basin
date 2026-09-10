//! Small dense column-major linear-algebra helpers for the COBYLA port.
//!
//! COBYLA's simplex bookkeeping works on tiny `n × n`/`n × (n+1)` matrices.
//! Rather than route these through the backend `linalg` tier (which targets
//! large LA-heavy solvers), the port keeps its own pure-`Vec<F>` column-major
//! scratch (exactly as the LINCOA active-set QR does), so it is backend-generic
//! and wasm-clean. Matrices are stored column-major: column `j` of an
//! `r × c` matrix occupies `a[j*r .. (j+1)*r]`.
//!
//! Faithful to PRIMA's `linalg_mod` helpers used by COBYLA (`planerot`,
//! `isminor`, `inv`, the matrix products).

use crate::core::math::Scalar;

/// `Σᵢ a[i]·b[i]`.
pub(crate) fn dot<F: Scalar>(a: &[F], b: &[F]) -> F {
    // Two-dimensional LPs benefit from unrolled reductions. Keep Scalar::sum
    // so its identity and accumulation order also apply to signed zeros.
    if let ([a0, a1], [b0, b1]) = (a, b) {
        return [*a0 * *b0, *a1 * *b1].into_iter().sum();
    }
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// `Σᵢ |a[i]|·|b[i]|`, preserving the signed dot product's summation order.
pub(crate) fn dot_abs<F: Scalar>(a: &[F], b: &[F]) -> F {
    if let ([a0, a1], [b0, b1]) = (a, b) {
        return [a0.abs() * b0.abs(), a1.abs() * b1.abs()].into_iter().sum();
    }
    a.iter().zip(b).map(|(&x, &y)| x.abs() * y.abs()).sum()
}

/// Signed and absolute dot products with the same reduction order.
pub(crate) fn dot_pair<F: Scalar>(a: &[F], b: &[F]) -> (F, F) {
    if a.len() == 2 && b.len() == 2 {
        return (dot(a, b), dot_abs(a, b));
    }
    // Scalar::sum starts floating-point reductions at negative zero. Reusing
    // its identity preserves the signs of empty and all-negative-zero sums.
    let identity: F = std::iter::empty().sum();
    a.iter().zip(b).fold(
        (identity, identity),
        |(signed, absolute), (&x, &y)| {
            (signed + x * y, absolute + x.abs() * y.abs())
        },
    )
}

/// Column `j` of an `r × c` column-major matrix.
pub(crate) fn col<F>(a: &[F], r: usize, j: usize) -> &[F] {
    &a[j * r..(j + 1) * r]
}

/// Column-major `n × n` identity.
pub(crate) fn eye<F: Scalar>(n: usize) -> Vec<F> {
    let mut q = vec![F::zero(); n * n];
    for i in 0..n {
        q[i + i * n] = F::one();
    }
    q
}

/// Fortran `SIGN(a, b)`: `|a|` with the sign of `b` (and `+|a|` when `b == 0`).
pub(crate) fn fsign<F: Scalar>(a: F, b: F) -> F {
    if b >= F::zero() { a.abs() } else { -a.abs() }
}

/// `√(x₀² + x₁²)`, overflow-aware (PRIMA `hypotenuse`).
pub(crate) fn hypotenuse<F: Scalar>(x0: F, x1: F) -> F {
    let (a, b) = (x0.abs(), x1.abs());
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    if hi <= F::zero() {
        F::zero()
    } else {
        let t = lo / hi;
        hi * (t * t + F::one()).sqrt()
    }
}

/// Tests whether `x` is negligible compared to `refv` (PRIMA `isminor`, used by
/// Powell throughout COBYLA). In exact arithmetic this is `x == 0`; in floating
/// point it is true when a nonzero `x` is attributable to rounding given `refv`.
pub(crate) fn isminor<F: Scalar>(x: F, refv: F) -> bool {
    let sensitivity = F::from_f64(0.1).unwrap();
    let two = F::from_f64(2.0).unwrap();
    let refa = refv.abs() + sensitivity * x.abs();
    let refb = refv.abs() + two * sensitivity * x.abs();
    refv.abs() >= refa || refa >= refb
}

/// 2×2 Givens rotation: returns `(c, s)` so that, with `G = [[c, s], [-s, c]]`,
/// `G·[x₀, x₁]ᵀ = [r, 0]ᵀ`. Faithful to PRIMA `planerot`, including the
/// sign-preserving degenerate cases that keep `G` continuous and orthogonal.
pub(crate) fn planerot<F: Scalar>(x0: F, x1: F) -> (F, F) {
    let zero = F::zero();
    let one = F::one();
    let eps = F::epsilon();
    if x0.is_nan() || x1.is_nan() {
        return (one, zero);
    }
    if x0.is_infinite() && x1.is_infinite() {
        let r = one / (one + one).sqrt();
        return (fsign(r, x0), fsign(r, x1));
    }
    if x0.abs() <= zero && x1.abs() <= zero {
        return (one, zero);
    }
    if x1.abs() <= eps * x0.abs() {
        return (fsign(one, x0), zero);
    }
    if x0.abs() <= eps * x1.abs() {
        return (zero, fsign(one, x1));
    }
    if x0.abs() > x1.abs() {
        let t = x1 / x0;
        let u = fsign((one + t * t).sqrt(), x0);
        (one / u, t / u)
    } else {
        let t = x0 / x1;
        let u = fsign((one + t * t).sqrt(), x1);
        (t / u, one / u)
    }
}

/// Inverse of an `n × n` column-major matrix by Gauss-Jordan elimination with
/// partial pivoting. Returns `None` if `A` is numerically singular. Mirrors the
/// role of PRIMA's `inv` (used to refresh `SIMI` from scratch when the
/// rank-one-updated copy drifts).
pub(crate) fn inv<F: Scalar>(a: &[F], n: usize) -> Option<Vec<F>> {
    // Work on an augmented [A | I]; reduce A to I, leaving the inverse in place.
    let mut m = a.to_vec(); // column-major working copy of A
    let mut out = eye::<F>(n);
    for c in 0..n {
        // Pivot: largest |A[r, c]| over r >= c.
        let mut piv = c;
        let mut best = m[c + c * n].abs();
        for r in (c + 1)..n {
            let v = m[r + c * n].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best <= F::zero() || !best.is_finite() {
            return None;
        }
        if piv != c {
            for j in 0..n {
                m.swap(c + j * n, piv + j * n);
                out.swap(c + j * n, piv + j * n);
            }
        }
        let d = m[c + c * n];
        for j in 0..n {
            m[c + j * n] = m[c + j * n] / d;
            out[c + j * n] = out[c + j * n] / d;
        }
        for r in 0..n {
            if r == c {
                continue;
            }
            let f = m[r + c * n];
            if f == F::zero() {
                continue;
            }
            for j in 0..n {
                m[r + j * n] = m[r + j * n] - f * m[c + j * n];
                out[r + j * n] = out[r + j * n] - f * out[c + j * n];
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_dots_preserve_separate_reductions() {
        fn check<F: Scalar>() {
            let f = |x| F::from_f64(x).unwrap();
            let values = [
                F::zero(),
                -F::zero(),
                f(1.0),
                f(-1.0),
                F::epsilon(),
                -F::epsilon(),
                F::min_positive_value(),
                F::max_value(),
                -F::max_value(),
                F::infinity(),
                F::neg_infinity(),
                F::nan(),
            ];
            for n in [0, 1, 2, 3, 5, 10, 20, 40] {
                for sample in 0..96 {
                    let a: Vec<_> = (0..n)
                        .map(|i| values[(i * 7 + sample) % values.len()])
                        .collect();
                    let b: Vec<_> = (0..n)
                        .map(|i| values[(i * 5 + sample / 8) % values.len()])
                        .collect();
                    for (a, b) in [(&a[..], &b[..]), (&a[..n / 2], &b[..])] {
                        let (signed, absolute) = dot_pair(a, b);
                        for (actual, expected) in
                            [(signed, dot(a, b)), (absolute, dot_abs(a, b))]
                        {
                            if expected.is_nan() {
                                assert!(actual.is_nan());
                            } else {
                                assert_eq!(
                                    actual.to_f64().unwrap().to_bits(),
                                    expected.to_f64().unwrap().to_bits(),
                                    "dimension {n}, sample {sample}"
                                );
                            }
                        }
                    }
                }
                for value in [F::zero(), -F::zero(), f(1e-30), f(1e30)] {
                    let a = vec![value; n];
                    let b = vec![f(-1.0); n];
                    let (signed, absolute) = dot_pair(&a, &b);
                    assert_eq!(
                        signed.to_f64().unwrap().to_bits(),
                        dot(&a, &b).to_f64().unwrap().to_bits()
                    );
                    assert_eq!(
                        absolute.to_f64().unwrap().to_bits(),
                        dot_abs(&a, &b).to_f64().unwrap().to_bits()
                    );
                }
                // Keep long-vector cancellation checks finite: the exceptional
                // palette above necessarily includes NaNs at these lengths.
                let large = F::one() / (F::epsilon() * F::epsilon());
                let finite = [large, F::one(), -large, -F::one()];
                for sample in 0..32 {
                    let a: Vec<_> = (0..n)
                        .map(|i| finite[(i + sample) % finite.len()])
                        .collect();
                    let b: Vec<_> = (0..n)
                        .map(|i| f(((i * 13 + sample) % 11) as f64 - 5.0))
                        .collect();
                    let (signed, absolute) = dot_pair(&a, &b);
                    for (actual, expected) in
                        [(signed, dot(&a, &b)), (absolute, dot_abs(&a, &b))]
                    {
                        assert!(actual.is_finite() && expected.is_finite());
                        assert_eq!(
                            actual.to_f64().unwrap().to_bits(),
                            expected.to_f64().unwrap().to_bits(),
                            "finite dimension {n}, sample {sample}"
                        );
                    }
                }
            }
        }
        check::<f32>();
        check::<f64>();
    }
}
