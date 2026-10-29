//! CMA-ES eigensolver cost on identical covariances, separate from trajectories.
//!
//! The reference uses the same nalgebra release and eigensolver settings as
//! `cmaes` 0.2.2. The matched variant also uses Basin's tolerance and iteration
//! cap to separate precision choices from backend cost. All kernels include
//! input cloning and output destruction.

use std::{hint::black_box, time::Duration};

use basin::SymmetricEigen;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

#[cfg(not(feature = "basin-latest"))]
use nalgebra::DMatrix;
#[cfg(feature = "basin-latest")]
use nalgebra_latest::DMatrix;

fn covariance(n: usize) -> DMatrix<f64> {
    // A dense rotation with a known positive spectrum exercises learned,
    // ill-conditioned covariances without depending on either solver's RNG.
    let q = DMatrix::from_fn(n, n, |i, j| {
        let scale = if i == 0 { 1.0 } else { 2.0 };
        (scale / n as f64).sqrt()
            * (std::f64::consts::PI * i as f64 * (j as f64 + 0.5) / n as f64)
                .cos()
    });
    let d =
        DMatrix::from_diagonal(&q.column(0).map_with_location(|i, _, _| {
            10_f64.powf(-6.0 * i as f64 / (n - 1) as f64)
        }));
    q.transpose() * d * q
}

fn eigen_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmaes_eigen");
    for n in [8, 40, 80] {
        let matrix = covariance(n);
        let reference = nalgebra_reference::DMatrix::from_column_slice(
            n,
            n,
            matrix.as_slice(),
        );
        let (basis, values) = matrix.try_eigh().unwrap();
        let reconstructed =
            &basis * DMatrix::from_diagonal(&values) * basis.transpose();
        assert!((&reconstructed - &matrix).norm() / matrix.norm() < 1e-8);
        let eig = reference.clone().try_symmetric_eigen(1e-20, 0).unwrap();
        let reconstructed = &eig.eigenvectors
            * nalgebra_reference::DMatrix::from_diagonal(&eig.eigenvalues)
            * eig.eigenvectors.transpose();
        assert!((&reconstructed - &reference).norm() / reference.norm() < 1e-8);

        group.bench_with_input(
            BenchmarkId::new("basin", n),
            &matrix,
            |b, a| {
                b.iter(|| black_box(black_box(a).try_eigh().unwrap()));
            },
        );
        for (name, eps, limit) in [
            ("cmaes_nalgebra", 1e-20, 0),
            ("cmaes_nalgebra_matched", 1e-10, (n * 30).max(64)),
        ] {
            group.bench_with_input(
                BenchmarkId::new(name, n),
                &reference,
                |b, a| {
                    b.iter(|| {
                        black_box(
                            black_box(a)
                                .clone()
                                .try_symmetric_eigen(eps, limit)
                                .unwrap(),
                        )
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(40)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(2));
    targets = eigen_kernels
}
criterion_main!(benches);
