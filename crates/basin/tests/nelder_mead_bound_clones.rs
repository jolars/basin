//! Projected steps must borrow box bounds instead of allocating copies.
use basin::{
    BasicSimplexState, BoxConstraints, CostFunction, NelderMead, Problem,
    Solver,
};
use basin::{ClampInPlace, ScaleInPlace, ScaledAdd};
use std::{cell::Cell, convert::Infallible, rc::Rc};

struct Vector(Vec<f64>, Rc<Cell<usize>>);
impl Clone for Vector {
    fn clone(&self) -> Self {
        self.1.set(self.1.get() + 1);
        Self(self.0.clone(), self.1.clone())
    }
}
impl ScaleInPlace for Vector {
    fn scale_in_place(&mut self, scalar: f64) {
        for x in &mut self.0 {
            *x *= scalar;
        }
    }
}
impl ScaledAdd for Vector {
    fn scaled_add(&mut self, scalar: f64, other: &Self) {
        for (x, y) in self.0.iter_mut().zip(&other.0) {
            *x += scalar * y;
        }
    }
}
impl ClampInPlace for Vector {
    fn clamp_in_place(&mut self, lower: &Self, upper: &Self) {
        for ((x, lo), hi) in self.0.iter_mut().zip(&lower.0).zip(&upper.0) {
            *x = x.clamp(*lo, *hi);
        }
    }
}
struct Quadratic {
    lower: Vector,
    upper: Vector,
}
impl CostFunction for Quadratic {
    type Param = Vector;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vector) -> Result<f64, Infallible> {
        for ((value, lo), hi) in
            x.0.iter().zip(&self.lower.0).zip(&self.upper.0)
        {
            assert!(*value >= *lo && *value <= *hi);
        }
        Ok(x.0.iter().map(|value| (value - 0.25).powi(2)).sum())
    }
}
impl BoxConstraints for Quadratic {
    fn lower(&self) -> &Vector {
        &self.lower
    }
    fn upper(&self) -> &Vector {
        &self.upper
    }
}
#[test]
fn projected_iterations_do_not_clone_bounds() {
    let clones = Rc::new(Cell::new(0));
    let vector = |values| Vector(values, clones.clone());
    let mut problem = Problem::new(Quadratic {
        lower: vector(vec![0.0; 2]),
        upper: vector(vec![1.0; 2]),
    });
    let state = BasicSimplexState::from_simplex(vec![
        vector(vec![0.5, 0.5]),
        vector(vec![0.7, 0.5]),
        vector(vec![0.5, 0.7]),
    ]);
    let mut solver = NelderMead::new().projected();
    let mut state = solver.init(&mut problem, state).unwrap();
    clones.set(0);
    for _ in 0..20 {
        state = solver.next_iter(&mut problem, state).unwrap().0;
    }
    assert_eq!(
        clones.get(),
        0,
        "initialized projected steps must not clone vectors"
    );
    assert!(problem.counts().cost_evals > 20);
}
