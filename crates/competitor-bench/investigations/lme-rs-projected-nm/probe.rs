use basin::{
    BasicSimplexState, BoxConstraints, CostFunction, NelderMead, Problem,
    Solver, State,
};
use ndarray::Array1;
use std::{convert::Infallible, hint::black_box, time::Instant};
struct Quadratic {
    lower: Array1<f64>,
    upper: Array1<f64>,
}
impl CostFunction for Quadratic {
    type Param = Array1<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Self::Param) -> Result<f64, Infallible> {
        Ok(x.iter().map(|x| (x - 0.25).powi(2)).sum())
    }
}
impl BoxConstraints for Quadratic {
    fn lower(&self) -> &Array1<f64> {
        &self.lower
    }
    fn upper(&self) -> &Array1<f64> {
        &self.upper
    }
}
fn solve(n: usize) -> (f64, u64) {
    let mut problem = Problem::new(Quadratic {
        lower: Array1::zeros(n),
        upper: Array1::ones(n),
    });
    let start = Array1::from_elem(n, 0.8);
    let mut vertices = vec![start.clone()];
    for i in 0..n {
        let mut vertex = start.clone();
        vertex[i] += 0.2;
        vertices.push(vertex);
    }
    let mut solver = NelderMead::new().projected();
    let mut state = solver
        .init(&mut problem, BasicSimplexState::from_simplex(vertices))
        .unwrap();
    for _ in 0..100 {
        state = solver.next_iter(&mut problem, state).unwrap().0;
    }
    (state.cost(), problem.counts().cost_evals)
}
fn main() {
    let mut reports = Vec::new();
    for n in [2, 3, 16, 128] {
        let batch = if n < 20 { 300 } else { 4 };
        let reference = solve(n);
        let mut samples = Vec::new();
        for sample in 0..14 {
            let timer = Instant::now();
            for _ in 0..batch {
                assert_eq!(black_box(solve(n)), reference);
            }
            let elapsed = timer.elapsed().as_secs_f64() / batch as f64;
            if sample >= 3 {
                samples.push(elapsed);
            }
        }
        reports.push(serde_json::json!({"dimension":n,"iterations":100,"cost":reference.0,"cost_evals":reference.1,"batch":batch,"samples_seconds":samples}));
    }
    println!("{}", serde_json::to_string_pretty(&reports).unwrap());
}
