# basin <img src='https://raw.githubusercontent.com/jolars/basin/main/images/logo.png' align="right" width="139" />

[![CI](https://github.com/jolars/basin/actions/workflows/ci.yml/badge.svg)](https://github.com/jolars/basin/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/basin.svg)](https://crates.io/crates/basin)
[![docs.rs](https://img.shields.io/docsrs/basin)](https://docs.rs/basin)

A numerical optimization library for Rust, inspired by [argmin]. It pairs a
small generic core --- problem traits you implement, a pluggable termination
layer, and a driver loop (`Executor`) --- with a growing set of solvers spanning
first-order, derivative-free, nonlinear least-squares, and evolutionary methods.
Solvers are generic over the linear-algebra backend, constraints are
first-class, and the default build compiles to `wasm32-unknown-unknown` with no
BLAS/LAPACK or threads.

Narrative documentation lives at [basin.bz/docs]; the rustdoc reference
is at [docs.rs/basin]. There is also an in-browser [solver visualizer].

## Install

```sh
cargo add basin
```

basin works on plain `Vec<f64>` out of the box. Linear-algebra backends are
opt-in, one feature each:

```sh
cargo add basin --features nalgebra   # or: ndarray, faer
```

## Example

Implement `CostFunction` (and `Gradient`, when the solver needs derivatives),
then hand the problem, a solver, and an initial state to the `Executor`:

```rust
use basin::{BasicState, CostFunction, Executor, Gradient, GradientDescent, GradientTolerance};

struct Rosenbrock;

impl CostFunction for Rosenbrock {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, x: &Vec<f64>) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
    }
}

impl Gradient for Rosenbrock {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Vec<f64> {
        vec![
            -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2)),
            200.0 * (x[1] - x[0].powi(2)),
        ]
    }
}

let result = Executor::new(
    Rosenbrock,
    GradientDescent::new(1e-3),
    BasicState::new(vec![-1.2, 1.0]),
)
    .max_iter(50_000)
    .terminate_on(GradientTolerance(1e-6))
    .run();

println!("x = {:?}, f = {}, stopped: {:?}", result.param(), result.cost(), result.reason);
```

Termination criteria are framework-level: the same ones compose across solvers,
and they are bound to the state a solver actually exposes --- so asking for a
gradient tolerance on a derivative-free solver is a compile error, not a runtime
surprise.

## Solvers

- **First-order / quasi-Newton:** gradient descent (with momentum and pluggable
  line searches), BFGS, L-BFGS, L-BFGS-B.
- **Derivative-free:** Nelder--Mead, Brent (1D).
- **Nonlinear least squares:** Gauss--Newton, Levenberg--Marquardt, trust-region
  reflective.
- **Global / stochastic:** random search, CMA-ES, a steady-state genetic
  algorithm, and memetic combinations.
- **Constrained:** box bounds via projected gradient descent, bounded
  Nelder--Mead, L-BFGS-B, and bounded CMA-ES.

See [Solvers] for which backends each one supports.

## Backends

Parameters and linear algebra are generic over the backend. `Vec<f64>` needs no
features; [nalgebra], [ndarray], and [faer] are enabled one feature each, each
pinning a single major version. First-order and derivative-free solvers run on
any backend; linear-algebra-heavy solvers may require a specific one and say so
in their docs.

The default build is wasm-friendly --- no BLAS/LAPACK and no threads.
Parallelism and BLAS-backed paths are behind opt-in features (`parallel`,
`ndarray-blas`).

## Status

Early-stage alpha: the public API is still iterating and breaking changes are
expected.

## License

Licensed under either of

- Apache License, Version 2.0
  ([LICENSE-APACHE](https://github.com/jolars/basin/blob/main/LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license
  ([LICENSE-MIT](https://github.com/jolars/basin/blob/main/LICENSE-MIT) or
  <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

[argmin]: https://github.com/argmin-rs/argmin
[nalgebra]: https://nalgebra.org
[ndarray]: https://github.com/rust-ndarray/ndarray
[faer]: https://github.com/sarah-quinones/faer-rs
[basin.bz/docs]: https://basin.bz/docs/
[docs.rs/basin]: https://docs.rs/basin
[solver visualizer]: https://basin.bz/visualizer/
[Solvers]: https://basin.bz/docs/solvers/
