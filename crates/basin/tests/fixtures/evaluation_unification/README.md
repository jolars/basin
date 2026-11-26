# Evaluation feature unification fixture

The application depends on a local library and a threaded library. Only the
threaded library requests Basin's `parallel` feature. The local library uses
Basin's finite-difference adapters, derivative checker, standalone helpers,
and CMA-ES with an `Rc`-backed callback and error. These tests run both in
isolation and with unified features.

The application tests verify that serial finite-difference probes stay on the
calling thread and that the threaded modes of `FiniteDiff` and `CmaEs` overlap
objective calls and match serial solver results and counts.

The Hessian-vector product prototype uses fixture-defined policies. Tests
compare forward and central products with analytic results and Basin's serial
helpers. A rendezvous verifies overlap, and a synthesized gradient exercises
nested dispatch on the same pool.

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
