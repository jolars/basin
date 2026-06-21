//! COBYLA initial simplex (PRIMA `initxfc`).
//!
//! Builds the regular initial simplex `sim(:, j) = ρ_beg eⱼ` around the pole
//! `sim(:, n) = x0`, evaluating the objective and constraints at each vertex.
//! As each off-pole vertex is found better (by raw `F`) than the pole, it is
//! swapped to the pole and the displacement bookkeeping is kept lower-triangular
//! (Powell's construction). `simi = inv(sim[:, 0..n])` is formed at the end.

use crate::core::math::Scalar;

use super::driver::EvalFn;
use super::filter::moderatef;
use super::linalg::inv;

/// The initialized simplex and its bookkeeping.
pub(crate) struct InitOut<F> {
    pub(crate) sim: Vec<F>,    // n × (n+1)
    pub(crate) simi: Vec<F>,   // n × n
    pub(crate) fval: Vec<F>,   // n+1
    pub(crate) conmat: Vec<F>, // m × (n+1)
    pub(crate) cval: Vec<F>,   // n+1
}

/// Evaluate the moderated `(f, constr)` at the `n+1` simplex vertices and form
/// `sim`/`simi`. `f0`/`constr0` are the (already moderated) values at `x0`.
/// `eval(x)` returns the moderated objective and constraint vector at `x`.
pub(crate) fn initxfc<F, E>(
    n: usize,
    m: usize,
    x0: &[F],
    f0: F,
    constr0: &[F],
    rhobeg: F,
    eval: &mut EvalFn<F, E>,
) -> Result<InitOut<F>, E>
where
    F: Scalar,
{
    let zero = F::zero();
    let np = n + 1;
    let mut sim = vec![zero; n * np];
    for j in 0..n {
        sim[j + j * n] = rhobeg;
    }
    for r in 0..n {
        sim[r + n * n] = x0[r];
    }
    let mut simi = vec![zero; n * n];
    for i in 0..n {
        simi[i + i * n] = F::one() / rhobeg;
    }
    let mut fval = vec![zero; np];
    let mut conmat = vec![zero; m * np];
    let mut cval = vec![zero; np];

    for k in 0..np {
        // x = current pole coordinates.
        let mut x: Vec<F> = (0..n).map(|r| sim[r + n * n]).collect();
        let (jstore, f, constr) = if k == 0 {
            (n, f0, constr0.to_vec())
        } else {
            let jd = k - 1;
            x[jd] = x[jd] + rhobeg;
            let (f, c) = eval(&x)?;
            (jd, moderatef(f), c.iter().map(|&v| moderatef(v)).collect())
        };
        let cstrv = constr.iter().cloned().fold(zero, F::max).max(zero);
        fval[jstore] = f;
        for r in 0..m {
            conmat[r + jstore * m] = constr[r];
        }
        cval[jstore] = cstrv;

        if jstore < n && fval[jstore] < fval[n] {
            fval.swap(jstore, n);
            cval.swap(jstore, n);
            for r in 0..m {
                conmat.swap(r + jstore * m, r + n * m);
            }
            for r in 0..n {
                sim[r + n * n] = x[r];
            }
            for l in 0..=jstore {
                sim[jstore + l * n] = -rhobeg;
            }
        }
    }

    if let Some(fresh) = inv(&sim[..n * n], n) {
        simi.copy_from_slice(&fresh);
    }

    Ok(InitOut {
        sim,
        simi,
        fval,
        conmat,
        cval,
    })
}
