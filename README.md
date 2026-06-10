# Basin <picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jolars/basin/main/images/logo-dark.png" /><img src="https://raw.githubusercontent.com/jolars/basin/main/images/logo.png" align="right" width="189" alt="basin logo" /></picture>

[![CI](https://github.com/jolars/basin/actions/workflows/ci.yml/badge.svg)](https://github.com/jolars/basin/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/basin.svg)](https://crates.io/crates/basin)
[![docs.rs](https://img.shields.io/docsrs/basin)](https://docs.rs/basin)

A numerical optimization library for Rust, inspired by [argmin]. It pairs a
small generic core, problem traits you implement, a pluggable termination layer,
and a driver loop (`Executor`), with a growing set of solvers spanning
first-order, derivative-free, nonlinear least-squares, and evolutionary methods.
Solvers are generic over the linear-algebra backend, constraints are
first-class, and the default build compiles to `wasm32-unknown-unknown` with no
BLAS/LAPACK or threads.

Narrative documentation lives at [basin.rs/docs]; the rustdoc reference is at
[docs.rs/basin]. There is also an in-browser [solver visualizer] and a
[benchmarks site] comparing Basin against competing crates and across backends
and solvers.

## Install

```sh
cargo add basin
```

Basin works on plain `Vec<f64>` out of the box. Linear-algebra backends are
opt-in, one feature each:

```sh
cargo add basin --features nalgebra  # or: ndarray, faer
```

## Example

Implement `CostFunction` (and `Gradient`, when the solver needs derivatives),
then hand the problem, a solver, and an initial state to the `Executor`:

```rust
use basin::{BasicState, CostFunction, Executor, Gradient, GradientDescent, GradientTolerance};
use std::convert::Infallible;

struct Rosenbrock;

impl CostFunction for Rosenbrock {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        Ok((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2))
    }
}

impl Gradient for Rosenbrock {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(vec![
            -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2)),
            200.0 * (x[1] - x[0].powi(2)),
        ])
    }
}

let result = Executor::new(Rosenbrock, GradientDescent::new(1e-3), BasicState::new(vec![-1.2, 1.0]))
    .max_iter(50_000).terminate_on(GradientTolerance(1e-6))
    .run()
    .unwrap();

println!("x = {:?}, f = {}, stopped: {:?}", result.param(), result.cost(), result.reason);
```

Termination criteria are framework-level: the same ones compose across solvers,
and they are bound to the state a solver actually exposes --- so asking for a
gradient tolerance on a derivative-free solver is a compile error, not a runtime
surprise.

## Solvers

- **First-order/quasi-Newton:** gradient descent (with momentum and pluggable
  line searches), BFGS, L-BFGS, L-BFGS-B.
- **Derivative-free:** Nelder--Mead, Brent (1D).
- **Nonlinear least squares:** Gauss--Newton, Levenberg--Marquardt, trust-region
  reflective.
- **Global/stochastic:** random search, CMA-ES, a steady-state genetic
  algorithm, and memetic combinations.
- **Constrained:** box bounds via projected gradient descent, bounded
  Nelder--Mead, L-BFGS-B, and bounded CMA-ES; log-barrier and augmented
  Lagrangian wrappers for more general constraints.

See [Solvers] for which backends each one supports.

## Backends

Parameters and linear algebra are generic over the backend. `Vec<f64>` needs no
features; [nalgebra], [ndarray], and [faer] are enabled one feature each, each
pinning a single major version. First-order and derivative-free solvers run on
any backend; linear-algebra-heavy solvers may require a specific one and say so
in their docs.

basin pins one major version per backend. Each basin 1.x release supports
exactly these versions:

  | Backend    | Feature    | Version                            |
  | ---------- | ---------- | ---------------------------------- |
  | [nalgebra] | `nalgebra` | 0.34 (with `nalgebra-sparse` 0.11) |
  | [ndarray]  | `ndarray`  | 0.17                               |
  | [faer]     | `faer`     | 0.24                               |

`Vec<f64>` is built in and needs no features. A backend major-version bump is a
breaking change and ships only in a basin major release; within the 1.x series
these pins are fixed.

The default build is wasm-friendly: no BLAS/LAPACK and no threads. Parallelism
and BLAS-backed paths are behind opt-in features (`parallel`).

## Status

The public API is still iterating and breaking changes are expected. WebAssembly
bindings (`basin-wasm`) power the visualizer but are not published to a package
registry yet.

## Acknowledgements

Basin owes a substantial intellectual debt to [argmin]: the overall shape of the
crate: the `Executor` driver loop, the `Solver` / `Problem` trait split,
per-solver `State`, and the pluggable termination layer is borrowed from it, and
several solver implementations and test-problem conventions were modeled on
argmin's. Thanks to the argmin authors and contributors for a library that is a
pleasure to learn from.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

[argmin]: https://github.com/argmin-rs/argmin
[nalgebra]: https://nalgebra.rs
[ndarray]: https://github.com/rust-ndarray/ndarray
[faer]: https://faer.veganb.tw
[basin.rs/docs]: https://basin.rs/docs/
[docs.rs/basin]: https://docs.rs/basin
[solver visualizer]: https://basin.rs/visualizer/
[benchmarks site]: https://basin.rs/benchmarks/
[Solvers]: https://basin.rs/docs/solvers/
