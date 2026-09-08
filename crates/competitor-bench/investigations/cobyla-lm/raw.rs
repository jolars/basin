/// Run the private driver with the historical adapter's radius floor.
pub fn raw_solve(
    mut eval: impl FnMut(&[f64]) -> (f64, Vec<f64>),
    x: Vec<f64>,
    m: usize,
    budget: usize,
) -> (Vec<f64>, f64, usize) {
    let nf = std::cell::Cell::new(0);
    let mut eval = |x: &[f64]| -> Result<_, std::convert::Infallible> {
        nf.set(nf.get() + 1);
        Ok(eval(x))
    };
    let (mut work, _, _) = raw::driver::CobylaWork::try_init(
        x,
        m,
        0.5,
        f64::EPSILON.sqrt() * 0.5,
        &mut eval,
    )
    .unwrap();
    let mut iterations = 0;
    while nf.get() < budget && iterations < 10000 {
        let reason = work.step(&mut eval).unwrap();
        iterations += 1;
        if matches!(
            reason,
            raw::driver::Transition::Converged
                | raw::driver::Transition::Failed
        ) {
            break;
        }
    }
    let (x, f) = work.best();
    (x, f, iterations)
}
