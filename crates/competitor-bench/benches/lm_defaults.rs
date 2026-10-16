//! Criterion entry point using exactly the verification runner's settings.

use std::{hint::black_box, time::Duration};

use competitor_bench::{
    lm_defaults::*,
    nist::{Nist, datasets},
};
use criterion::{Criterion, criterion_group, criterion_main};

fn benchmark<B: Backend>(criterion: &mut Criterion) {
    for dataset in datasets().unwrap() {
        let model = Nist::<B::Float>::new(dataset);
        for start in 0..2 {
            let x = model.dataset.start::<B::Float>(start);
            let mut group = criterion.benchmark_group(format!(
                "lm_defaults/{}/{}/{}/start{}",
                B::NAME,
                B::PRECISION,
                model.dataset.id,
                start + 1
            ));
            for configuration in Configuration::ALL {
                group.bench_function(configuration.name(), |b| {
                    let verification =
                        B::solve(&model, x.clone(), configuration, false)
                            .expect("the selected fixture must evaluate");
                    let result = &verification.result;
                    eprintln!(
                        "verify {}/{}/{}/start{}/{}: iterations={}, residuals={}, jacobians={}, termination={:?}, parameters={:?}",
                        B::NAME,
                        B::PRECISION,
                        model.dataset.id,
                        start + 1,
                        configuration.name(),
                        result.iter(),
                        result.counts.residual_evals,
                        result.counts.jacobian_evals,
                        result.report.code(),
                        B::values(result.param()),
                    );
                    drop(verification);
                    b.iter(|| {
                        drop(black_box(B::solve(
                            &model,
                            x.clone(),
                            configuration,
                            false,
                        )));
                    });
                });
            }
            group.finish();
        }
    }
}

fn comparisons(criterion: &mut Criterion) {
    benchmark::<VecF64>(criterion);
    benchmark::<NalgebraF64>(criterion);
    benchmark::<NdarrayF64>(criterion);
    benchmark::<FaerF64>(criterion);
    benchmark::<VecF32>(criterion);
    benchmark::<NalgebraF32>(criterion);
    benchmark::<NdarrayF32>(criterion);
    benchmark::<FaerF32>(criterion);
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(1));
    targets = comparisons
}
criterion_main!(benches);
