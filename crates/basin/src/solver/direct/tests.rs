use super::{Rectangle, Work, interpolate, potentially_optimal};

fn work(n: usize) -> Work<f64> {
    let mut work = Work::new(
        vec![(0.0, 1.0); n],
        (0..n).collect(),
        Rectangle::new(vec![0.5; n], 0.0),
    );
    work.consider(0);
    work
}

#[test]
fn lower_hull_matches_exhaustive_slope_intervals() {
    // Integer costs and binary radii make the independent interval calculation
    // exact, including collinearity and equality at the selection threshold.
    for encoded in 0..625 {
        let mut code = encoded;
        let points: Vec<_> = [0.125, 0.25, 0.5, 1.0]
            .into_iter()
            .map(|d| {
                let f = f64::from(code % 5 - 2);
                code /= 5;
                (d, f)
            })
            .collect();
        for epsilon in [0.0, 0.125, 0.5] {
            let best =
                points.iter().map(|&(_, f)| f).fold(f64::INFINITY, f64::min);
            let expected: Vec<_> = points
                .iter()
                .enumerate()
                .filter_map(|(j, &(dj, fj))| {
                    let mut lower = 0.0_f64;
                    let mut upper = f64::INFINITY;
                    for &(di, fi) in &points {
                        if di < dj {
                            lower = lower.max((fj - fi) / (dj - di));
                        }
                        if di > dj {
                            upper = upper.min((fi - fj) / (di - dj));
                        }
                    }
                    lower = lower.max((fj - best + epsilon * best.abs()) / dj);
                    (upper > 0.0 && lower <= upper).then_some(j)
                })
                .collect();
            assert_eq!(
                potentially_optimal(&points, epsilon),
                expected,
                "points={points:?}, epsilon={epsilon}"
            );
        }
    }
}

#[test]
fn collinear_and_flat_diagrams_and_extreme_costs() {
    assert_eq!(
        potentially_optimal(&[(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)], 1e-4),
        [0, 1, 2]
    );
    assert_eq!(
        potentially_optimal(&[(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)], 0.0),
        [2]
    );
    assert_eq!(
        potentially_optimal(&[(1.0, -f64::MAX), (2.0, f64::MAX)], 1e-4),
        [0, 1]
    );
    assert_eq!(
        potentially_optimal(&[(1.0, 1.0), (2.0, 2.0)], f64::MAX),
        [1]
    );
}

#[test]
fn cost_scaling_preserves_collinearity_even_for_subnormals() {
    for scale in [1.0, f64::from_bits(1), 2.0_f64.powi(1000)] {
        let points = [
            (0.125, -3.0 * scale),
            (0.25, -3.0 * scale),
            (0.5, -scale),
            (1.0, 3.0 * scale),
        ];
        assert_eq!(potentially_optimal(&points, 0.5), [1, 2, 3]);
    }
    for scale in [1.0_f32, f32::from_bits(1), 2.0_f32.powi(100)] {
        let points = [
            (0.125, -3.0 * scale),
            (0.25, -3.0 * scale),
            (0.5, -scale),
            (1.0, 3.0 * scale),
        ];
        assert_eq!(potentially_optimal(&points, 0.5), [1, 2, 3]);
    }
}

#[test]
fn mixed_magnitude_costs_preserve_small_improvements() {
    let points = [
        (1.0_f32 / 54.0, -1e-9),
        (1.0 / 18.0, 0.0),
        (1.0 / 6.0, 1e38),
    ];
    assert_eq!(potentially_optimal(&points, 1e-4), [0, 1, 2]);
    let points = [
        (1.0_f64 / 54.0, -1e-100),
        (1.0 / 18.0, 0.0),
        (1.0 / 6.0, 1e300),
    ];
    assert_eq!(potentially_optimal(&points, 1e-4), [0, 1, 2]);
}

#[test]
fn mixed_magnitude_costs_preserve_improvement_threshold() {
    for small in [1e-9_f32, f32::from_bits(1)] {
        let points = [
            (0.125, -3.0 * small),
            (0.25, -2.0 * small),
            (0.5, -small),
            (1.0, 1e38),
        ];
        assert_eq!(potentially_optimal(&points, 0.0), [0, 2, 3]);
        assert_eq!(potentially_optimal(&points, 0.5), [2, 3]);
    }
}

#[test]
fn trisection_orders_axes_conserves_volume_and_preserves_centers() {
    let mut work = work(3);
    let probes = work.probes(&vec![0.0; 3], 0).unwrap();
    assert_eq!(probes.axes, [0, 1, 2]);
    let centers = probes.centers.clone();
    // Axis 1 is most promising, then axis 2, then axis 0.
    work.divide(
        0,
        probes.axes,
        probes.centers,
        vec![5.0, 6.0, -4.0, 2.0, -1.0, 1.0],
    );
    assert_eq!(work.rectangles[0].depths, [1, 1, 1]);
    assert_eq!(work.rectangles[1].depths, [1, 1, 1]);
    assert_eq!(work.rectangles[3].depths, [0, 1, 0]);
    assert_eq!(work.rectangles[5].depths, [0, 1, 1]);
    assert_eq!(work.best, Some(3));
    for (r, center) in work.rectangles[1..].iter().zip(centers) {
        assert_eq!(r.center, center);
        for (&c, &depth) in r.center.iter().zip(&r.depths) {
            let half = 0.5 / 3.0_f64.powi(depth as i32);
            assert!(c >= half - 1e-15 && c <= 1.0 - half + 1e-15);
        }
    }
    let volume: f64 = work
        .rectangles
        .iter()
        .map(|r| 3.0_f64.powi(-(r.level() as i32)))
        .sum();
    assert!((volume - 1.0).abs() < 1e-15);
    // Only the two remaining longest axes may be divided in this child.
    assert_eq!(work.probes(&vec![0.0; 3], 3).unwrap().axes, [0, 2]);
}

#[test]
fn exact_cost_ties_and_oldest_largest_rejection() {
    let mut work = work(1);
    let probes = work.probes(&vec![0.0], 0).unwrap();
    work.divide(0, probes.axes, probes.centers, vec![0.0, 1e-14]);
    assert_eq!(work.select(1e-4), [0, 1]);
    assert_eq!(work.best, Some(0));
    work.rectangles[0].cost = f64::NAN;
    work.rectangles[1].cost = f64::INFINITY;
    work.best = Some(2);
    assert_eq!(work.select(1e-4), [0, 2]);
    let probes = work.probes(&vec![0.0], 0).unwrap();
    work.divide(0, probes.axes, probes.centers, vec![f64::INFINITY; 2]);
    assert_eq!(work.select(1e-4), [1, 2]);
}

#[test]
fn interpolation_remains_finite_and_ordered_at_extreme_bounds() {
    for (lo, hi) in [
        (-f64::MAX, f64::MAX),
        (f64::MAX / 2.0, f64::MAX),
        (-f64::MAX, -f64::MAX / 2.0),
        (0.0, f64::MIN_POSITIVE),
    ] {
        let mut previous = lo;
        for i in 0..=100 {
            let value = interpolate(lo, hi, f64::from(i) / 100.0);
            assert!(value.is_finite() && value >= previous && value <= hi);
            previous = value;
        }
    }
}
