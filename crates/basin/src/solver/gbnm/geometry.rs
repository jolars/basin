use crate::core::math::{ClampInPlace, Scalar, VectorIndex, VectorLen};

pub(super) fn free_coordinates<V, F>(lower: &V, upper: &V) -> Vec<usize>
where
    V: VectorLen + VectorIndex<F>,
    F: Scalar,
{
    assert_eq!(
        lower.vec_len(),
        upper.vec_len(),
        "GBNM bounds length mismatch"
    );
    assert!(lower.vec_len() >= 1, "GBNM requires a non-empty box");

    let mut free = Vec::new();
    for i in 0..lower.vec_len() {
        let lo = lower.get_scalar(i);
        let hi = upper.get_scalar(i);
        assert!(
            lo.is_finite() && hi.is_finite(),
            "GBNM requires finite bounds, got lower[{i}] = {lo:?}, upper[{i}] = {hi:?}"
        );
        assert!(
            lo <= hi,
            "GBNM requires lower <= upper at coordinate {i}, got {lo:?} > {hi:?}"
        );
        if lo < hi {
            free.push(i);
        }
    }
    assert!(
        !free.is_empty(),
        "GBNM requires at least one free coordinate"
    );
    free
}

pub(super) fn scaled_minimum_width<V, F>(
    lower: &V,
    upper: &V,
    free: &[usize],
    fraction: F,
) -> F
where
    V: VectorIndex<F>,
    F: Scalar,
{
    free.iter().fold(F::infinity(), |width, &i| {
        let lower = lower.get_scalar(i);
        let upper = upper.get_scalar(i);
        let span = upper - lower;
        let scaled = if span.is_finite() {
            span * fraction
        } else {
            // Scaling first keeps a fractional width representable when the
            // finite endpoints are too far apart to subtract directly.
            let scaled = upper * fraction - lower * fraction;
            if scaled.is_finite() {
                scaled
            } else {
                F::max_value()
            }
        };
        width.min(scaled)
    })
}

pub(super) fn regular_simplex<V, F>(
    center: &V,
    edge: F,
    lower: &V,
    upper: &V,
    free: &[usize],
) -> Vec<V>
where
    V: Clone + ClampInPlace + VectorIndex<F>,
    F: Scalar,
{
    let n = F::from_usize(free.len()).unwrap();
    let root = F::from_usize(free.len() + 1).unwrap().sqrt();
    let denominator = n * F::from_f64(2.0).unwrap().sqrt();
    let p = edge * ((root + n - F::one()) / denominator);
    let q = edge * ((root - F::one()) / denominator);

    let directions: Vec<F> = free
        .iter()
        .map(|&i| {
            let center = center.get_scalar(i);
            let room_below = center - lower.get_scalar(i);
            let room_above = upper.get_scalar(i) - center;
            if room_above >= room_below {
                F::one()
            } else {
                -F::one()
            }
        })
        .collect();

    let mut vertices = Vec::with_capacity(free.len() + 1);
    vertices.push(center.clone());
    for &axis in free {
        let mut vertex = center.clone();
        for (&i, &direction) in free.iter().zip(&directions) {
            let offset = if i == axis { p } else { q };
            vertex.set_scalar(i, center.get_scalar(i) + direction * offset);
        }
        vertex.clamp_in_place(lower, upper);
        vertices.push(vertex);
    }
    vertices
}

pub(super) fn normalized_l1_distance<V, F>(
    left: &V,
    right: &V,
    lower: &V,
    upper: &V,
    free: &[usize],
) -> F
where
    V: VectorIndex<F>,
    F: Scalar,
{
    free.iter()
        .map(|&i| {
            normalized_coordinate_distance(
                left.get_scalar(i),
                right.get_scalar(i),
                lower.get_scalar(i),
                upper.get_scalar(i),
            )
        })
        .sum()
}

fn normalized_coordinate_distance<F: Scalar>(
    left: F,
    right: F,
    lower: F,
    upper: F,
) -> F {
    let difference = (left - right).abs();
    let width = upper - lower;
    if difference.is_finite() && width.is_finite() {
        return difference / width;
    }

    let scale = lower.abs().max(upper.abs());
    ((left / scale) - (right / scale)).abs()
        / ((upper / scale) - (lower / scale))
}

pub(super) fn simplex_is_small<V, F>(
    vertices: &[V],
    lower: &V,
    upper: &V,
    free: &[usize],
    tolerance: F,
) -> bool
where
    V: VectorIndex<F>,
    F: Scalar,
{
    vertices[1..].iter().all(|vertex| {
        normalized_l1_distance(vertex, &vertices[0], lower, upper, free)
            < tolerance
    })
}

pub(super) fn point_on_bounds<V, F>(
    point: &V,
    lower: &V,
    upper: &V,
    free: &[usize],
) -> bool
where
    V: VectorIndex<F>,
    F: Scalar,
{
    free.iter().copied().any(|i| {
        point.get_scalar(i) <= lower.get_scalar(i)
            || point.get_scalar(i) >= upper.get_scalar(i)
    })
}

fn simplex_touches_bounds<V, F>(
    vertices: &[V],
    lower: &V,
    upper: &V,
    free: &[usize],
) -> bool
where
    V: VectorIndex<F>,
    F: Scalar,
{
    vertices
        .iter()
        .any(|vertex| point_on_bounds(vertex, lower, upper, free))
}

fn determinant<F: Scalar>(mut matrix: Vec<Vec<F>>) -> F {
    let n = matrix.len();
    let mut det = F::one();
    for column in 0..n {
        let mut pivot = column;
        for row in column + 1..n {
            if matrix[row][column].abs() > matrix[pivot][column].abs() {
                pivot = row;
            }
        }
        if matrix[pivot][column] == F::zero() {
            return F::zero();
        }
        if pivot != column {
            matrix.swap(pivot, column);
            det = -det;
        }
        let diagonal = matrix[column][column];
        det = det * diagonal;
        for row in column + 1..n {
            let factor = matrix[row][column] / diagonal;
            for j in column + 1..n {
                matrix[row][j] = matrix[row][j] - factor * matrix[column][j];
            }
        }
    }
    det
}

fn normalize_edge<F: Scalar>(edge: &mut [F]) -> Option<F> {
    let scale = edge
        .iter()
        .fold(F::zero(), |scale, value| scale.max(value.abs()));
    if scale == F::zero() {
        return None;
    }

    let scaled_norm = edge
        .iter()
        .map(|value| {
            let scaled = *value / scale;
            scaled * scaled
        })
        .sum::<F>()
        .sqrt();
    for component in edge {
        *component = (*component / scale) / scaled_norm;
    }
    Some(scale.ln() + scaled_norm.ln())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn simplex_is_degenerate<V, F>(
    vertices: &[V],
    lower: &V,
    upper: &V,
    free: &[usize],
    small: bool,
    edge_ratio_tolerance: F,
    shape_tolerance: F,
) -> bool
where
    V: VectorIndex<F>,
    F: Scalar,
{
    if small || simplex_touches_bounds(vertices, lower, upper, free) {
        return false;
    }

    let mut edge_matrix = Vec::with_capacity(free.len());
    // Log norms preserve edge ratios even when a rescaled Euclidean norm
    // would lie outside the scalar's representable range.
    let mut min_log_norm = F::infinity();
    let mut max_log_norm = F::neg_infinity();
    for vertex in &vertices[1..] {
        let mut edge: Vec<F> = free
            .iter()
            .map(|&i| vertex.get_scalar(i) - vertices[0].get_scalar(i))
            .collect();
        let log_scale = if edge.iter().all(|component| component.is_finite()) {
            F::zero()
        } else {
            let scale = free.iter().fold(F::zero(), |scale, &i| {
                scale
                    .max(vertex.get_scalar(i).abs())
                    .max(vertices[0].get_scalar(i).abs())
            });
            edge = free
                .iter()
                .map(|&i| {
                    vertex.get_scalar(i) / scale
                        - vertices[0].get_scalar(i) / scale
                })
                .collect();
            scale.ln()
        };
        let Some(log_norm) = normalize_edge(&mut edge) else {
            return true;
        };
        let log_norm = log_scale + log_norm;
        min_log_norm = min_log_norm.min(log_norm);
        max_log_norm = max_log_norm.max(log_norm);
        edge_matrix.push(edge);
    }

    let edge_ratio = (min_log_norm - max_log_norm).exp();
    // Normalizing first is equivalent to dividing `det(E)` by the product of
    // edge norms, but avoids overflow and underflow in that product.
    let normalized_shape = determinant(edge_matrix).abs();
    edge_ratio < edge_ratio_tolerance || normalized_shape < shape_tolerance
}

pub(super) fn log_parzen_density<V, F>(
    point: &V,
    starts: &[V],
    local_optima: &[(V, F)],
    lower: &V,
    upper: &V,
    free: &[usize],
    variance_factor: F,
) -> F
where
    V: VectorIndex<F>,
    F: Scalar,
{
    let half = F::from_f64(0.5).unwrap();
    starts
        .iter()
        .chain(local_optima.iter().map(|(point, _)| point))
        .fold(F::neg_infinity(), |log_sum, start| {
            let scaled_distance: F = free
                .iter()
                .map(|&i| {
                    let z = normalized_coordinate_distance(
                        point.get_scalar(i),
                        start.get_scalar(i),
                        lower.get_scalar(i),
                        upper.get_scalar(i),
                    );
                    z * z
                })
                .sum();
            let exponent = -half * scaled_distance / variance_factor;
            if log_sum == F::neg_infinity() {
                exponent
            } else {
                let max_exponent = log_sum.max(exponent);
                max_exponent
                    + ((log_sum - max_exponent).exp()
                        + (exponent - max_exponent).exp())
                    .ln()
            }
        })
}

#[cfg(test)]
mod tests {
    use super::{
        determinant, log_parzen_density, normalized_l1_distance,
        point_on_bounds, regular_simplex, simplex_is_degenerate,
    };

    fn distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn regular_simplex_has_the_requested_edge_length() {
        let center = vec![0.0, 0.0, 9.0];
        let lower = vec![-10.0, -10.0, 9.0];
        let upper = vec![10.0, 10.0, 9.0];
        let simplex = regular_simplex(&center, 2.5, &lower, &upper, &[0, 1]);

        assert_eq!(simplex.len(), 3);
        for i in 0..simplex.len() {
            for j in i + 1..simplex.len() {
                assert!(
                    (distance(&simplex[i], &simplex[j]) - 2.5).abs() < 1e-12
                );
            }
        }
        assert!(simplex.iter().all(|vertex| vertex[2] == 9.0));
    }

    #[test]
    fn regular_simplex_points_inward_from_upper_bounds() {
        let center = vec![1.0, 0.0];
        let lower = vec![-1.0, -1.0];
        let upper = vec![1.0, 1.0];
        let simplex = regular_simplex(&center, 0.1, &lower, &upper, &[0, 1]);

        assert!(simplex.iter().all(|vertex| vertex[0] <= 1.0));
        assert!(simplex[1..].iter().any(|vertex| vertex[0] < 1.0));
        assert!(simplex[1..].iter().all(|vertex| vertex != &center));
    }

    #[test]
    fn pivoted_determinant_handles_a_row_swap() {
        let value: f64 = determinant(vec![vec![0.0, 2.0], vec![3.0, 4.0]]);
        assert!((value + 6.0).abs() < 1e-12);
    }

    #[test]
    fn parzen_density_is_lower_away_from_previous_starts() {
        let lower = vec![0.0, 0.0];
        let upper = vec![10.0, 10.0];
        let starts = vec![vec![1.0, 1.0], vec![2.0, 2.0]];
        let near = log_parzen_density(
            &vec![1.5, 1.5],
            &starts,
            &[],
            &lower,
            &upper,
            &[0, 1],
            0.01,
        );
        let far = log_parzen_density(
            &vec![9.0, 9.0],
            &starts,
            &[],
            &lower,
            &upper,
            &[0, 1],
            0.01,
        );
        assert!(far < near);

        let with_far_optimum = log_parzen_density(
            &vec![9.0, 9.0],
            &starts,
            &[(vec![9.0, 9.0], 0.0)],
            &lower,
            &upper,
            &[0, 1],
            0.01,
        );
        assert!(with_far_optimum > far);
    }

    #[test]
    fn normalized_geometry_handles_overflowing_f32_box_widths() {
        let lower = vec![-3.0e38_f32, -3.0e38];
        let upper = vec![3.0e38_f32, 3.0e38];
        let center = vec![0.0_f32, 0.0];
        let corner = vec![3.0e38_f32, 3.0e38];

        let distance =
            normalized_l1_distance(&center, &corner, &lower, &upper, &[0, 1]);
        assert!(distance.is_finite());
        assert!((distance - 1.0).abs() < 1.0e-6);

        let near = log_parzen_density(
            &vec![1.0e37_f32, 1.0e37],
            std::slice::from_ref(&center),
            &[],
            &lower,
            &upper,
            &[0, 1],
            0.01,
        );
        let far = log_parzen_density(
            &corner,
            std::slice::from_ref(&center),
            &[],
            &lower,
            &upper,
            &[0, 1],
            0.01,
        );
        assert!(near.is_finite());
        assert!(far.is_finite());
        assert!(far < near);
    }

    #[test]
    fn normalized_determinant_detects_collinearity() {
        let lower = vec![-10.0, -10.0];
        let upper = vec![10.0, 10.0];
        let simplex = vec![vec![0.0, 0.0], vec![2.0, 0.0], vec![1.0, 0.0]];
        assert!(simplex_is_degenerate(
            &simplex,
            &lower,
            &upper,
            &[0, 1],
            false,
            1e-7,
            1e-7,
        ));
    }

    #[test]
    fn scaled_edge_norm_handles_extreme_f32_magnitudes() {
        for edge in [1.0e20_f32, 1.0e-30_f32] {
            let scale = edge * 10.0;
            let lower = vec![-scale, -scale];
            let upper = vec![scale, scale];
            let simplex = regular_simplex(
                &vec![0.0_f32, 0.0],
                edge,
                &lower,
                &upper,
                &[0, 1],
            );
            assert!(!simplex_is_degenerate(
                &simplex,
                &lower,
                &upper,
                &[0, 1],
                false,
                1.0e-7,
                1.0e-7,
            ));
        }
    }

    #[test]
    fn scaled_edge_norm_handles_overflowing_f32_differences() {
        let lower = vec![-3.0e38_f32, -3.0e38];
        let upper = vec![3.0e38_f32, 3.0e38];
        let simplex = vec![
            vec![-2.0e38_f32, -2.0e38],
            vec![2.0e38_f32, -2.0e38],
            vec![-2.0e38_f32, 2.0e38],
        ];

        assert!(!simplex_is_degenerate(
            &simplex,
            &lower,
            &upper,
            &[0, 1],
            false,
            1.0e-7,
            1.0e-7,
        ));

        let collinear = vec![
            vec![-2.0e38_f32, -2.0e38],
            vec![2.0e38_f32, 2.0e38],
            vec![1.0e38_f32, 1.0e38],
        ];
        assert!(simplex_is_degenerate(
            &collinear,
            &lower,
            &upper,
            &[0, 1],
            false,
            1.0e-7,
            1.0e-7,
        ));
    }

    #[test]
    fn pinned_coordinates_do_not_count_as_active_bounds() {
        let lower = vec![-5.0, 2.0];
        let upper = vec![5.0, 2.0];
        assert!(!point_on_bounds(&vec![0.0, 2.0], &lower, &upper, &[0],));
    }
}
