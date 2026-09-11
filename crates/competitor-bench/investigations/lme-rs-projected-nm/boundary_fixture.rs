//! Export the exact boundary-heavy production random-slope workload for profiling.
//! Run in release mode with the output CSV path as the first argument.
use polars::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("output CSV path required"))?;
    // Same recipe, seed, and dimensions as bench_load_production's random slopes.
    let mut rng = StdRng::seed_from_u64(105);
    let normal = Normal::new(0.0, 1.0)?;
    let mut y = Vec::with_capacity(80_000);
    let mut x = Vec::with_capacity(80_000);
    let mut group = Vec::with_capacity(80_000);
    for _ in 0..80_000 {
        let g = rng.random_range(0..3_000);
        let xi = normal.sample(&mut rng);
        y.push(1.0 + 1.25 * xi + normal.sample(&mut rng));
        x.push(xi);
        group.push(format!("G{}", g));
    }
    let mut data = df!("y" => y, "x" => x, "group" => group)?;
    CsvWriter::new(std::fs::File::create(path)?).finish(&mut data)?;
    Ok(())
}
