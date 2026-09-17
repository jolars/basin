//! Ordered products for the newest row of the compact history matrices.

use super::backend::AsFloatSlice;
use crate::core::math::Scalar;

/// Compute the newest S column's products with the preceding Y and S columns.
pub(crate) fn newest_products<F: Scalar, V: AsFloatSlice<F>>(
    s: &[F],
    ws: &[V],
    wy: &[V],
    sy: &mut [F],
    ss: &mut [F],
) {
    let col = ws.len();
    let n = s.len();
    assert_eq!(wy.len(), col);
    assert_eq!(sy.len(), col);
    assert_eq!(ss.len(), col);

    // Interleave independent reductions to hide addition latency without
    // reassociating a sum. Two history columns share each load of the new S
    // column, and their own storage remains contiguous across coordinates.
    let paired = col / 2 * 2;
    for j in (0..paired).step_by(2) {
        let s0 = ws[j].as_float_slice();
        let s1 = ws[j + 1].as_float_slice();
        let y0 = wy[j].as_float_slice();
        let y1 = wy[j + 1].as_float_slice();
        assert_eq!(s0.len(), n);
        assert_eq!(s1.len(), n);
        assert_eq!(y0.len(), n);
        assert_eq!(y1.len(), n);
        // Match the signed-zero identity used by the backends' float sums.
        let mut sy0 = -F::zero();
        let mut sy1 = -F::zero();
        let mut ss0 = -F::zero();
        let mut ss1 = -F::zero();
        for i in 0..n {
            let si = s[i];
            sy0 = sy0 + si * y0[i];
            ss0 = ss0 + s0[i] * si;
            sy1 = sy1 + si * y1[i];
            ss1 = ss1 + s1[i] * si;
        }
        sy[j] = sy0;
        sy[j + 1] = sy1;
        ss[j] = ss0;
        ss[j + 1] = ss1;
    }

    if paired < col {
        let sj = ws[paired].as_float_slice();
        let yj = wy[paired].as_float_slice();
        assert_eq!(sj.len(), n);
        assert_eq!(yj.len(), n);
        let mut syj = -F::zero();
        let mut ssj = -F::zero();
        for i in 0..n {
            syj = syj + s[i] * yj[i];
            ssj = ssj + sj[i] * s[i];
        }
        sy[paired] = syj;
        ss[paired] = ssj;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_independent_products<F: Scalar>() {
        for n in 0..=9 {
            for col in 0..=7 {
                let s = vec![F::one(); n];
                let ws: Vec<Vec<_>> = (0..col)
                    .map(|j| {
                        (0..n)
                            .map(|i| {
                                F::from_f64(
                                    [-0.0, 1e16, 1.0, -1e16][(i + j) % 4],
                                )
                                .unwrap()
                            })
                            .collect()
                    })
                    .collect();
                let mut wy = ws.clone();
                if n > 0 && col > 1 {
                    wy[1][0] = F::nan();
                }
                if n > 0 && col > 4 {
                    wy[4][0] = F::infinity();
                }
                let mut sy = vec![F::zero(); col];
                let mut ss = vec![F::zero(); col];
                newest_products(&s, &ws, &wy, &mut sy, &mut ss);
                for j in 0..col {
                    for (actual, column) in [(sy[j], &wy[j]), (ss[j], &ws[j])] {
                        let expected: F = column.iter().copied().sum();
                        if expected.is_nan() {
                            assert!(actual.is_nan());
                        } else {
                            assert_eq!(actual, expected);
                            assert_eq!(
                                actual.is_sign_negative(),
                                expected.is_sign_negative()
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn products_preserve_cancellation_and_nonfinite_columns_f64() {
        check_independent_products::<f64>();
    }

    #[test]
    fn products_preserve_cancellation_and_nonfinite_columns_f32() {
        check_independent_products::<f32>();
    }
}
