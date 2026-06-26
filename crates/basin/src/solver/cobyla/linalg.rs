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
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Column `j` of an `r × c` column-major matrix.
pub(crate) fn col<F>(a: &[F], r: usize, j: usize) -> &[F] {
    &a[j * r..(j + 1) * r]
}

/// `xᵀ A = Aᵀ x` for an `r × c` column-major `A` and length-`r` `x`; result
/// length `c` (PRIMA's `matprod(x, A)` with `x` a row vector).
pub(crate) fn row_times_mat<F: Scalar>(x: &[F], a: &[F], r: usize, c: usize) -> Vec<F> {
    (0..c).map(|j| dot(x, col(a, r, j))).collect()
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

/// Max absolute entry of a slice; `NaN` if any entry is `NaN`.
pub(crate) fn maxabs<F: Scalar>(a: &[F]) -> F {
    a.iter().fold(F::zero(), |acc, &x| {
        if x.is_nan() {
            F::nan()
        } else {
            acc.max(x.abs())
        }
    })
}
