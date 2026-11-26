# Evaluation feature unification fixture

The application depends on a local library and a threaded library. Only the
threaded library requests Basin's `parallel` feature. The local library runs
finite differences and population evaluation with an `Rc`-backed callback and
error, both in isolation and with unified features.

The population tests use Basin's CMA-ES. They verify that a threaded run
overlaps objective calls and matches the serial population and counts.
The finite-difference prototype uses fixture-defined policies through Basin's
public executor. The tests compare forward and central Hessian-vector products
with analytic results and Basin's serial helpers. A rendezvous verifies overlap,
and a synthesized gradient exercises nested dispatch on the same pool.

A run factory can move its configuration to a worker and create the executor
and local hooks there. An already-built executor contains non-`Send` hook
storage. Hooks that capture existing `Rc` state run on their owning thread.

Run from this directory:

```sh
cargo test -p basin-evaluation-local
cargo test --workspace
cargo tree -p basin-evaluation-local -e features -i basin
cargo tree -p basin-evaluation-application -e features -i basin
```
