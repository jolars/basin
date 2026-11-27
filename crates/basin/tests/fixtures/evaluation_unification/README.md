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

A run factory can send its configuration to a worker and create local hooks
there. The application tests also send a configured Basin executor directly,
with observers, application stops, and checkpoint sinks already attached.
The tests cover a memetic solver with a transferable inner stop factory and
an inner factory that captures local state.

## Configured executor prototypes

The [prototypes](application/src/executor_prototypes.rs) compare boxed hooks
with an explicit local switch against inferring transferability from each
hook's concrete type. Basin uses the explicit switch. That keeps the executor
type stable across registrations within each mode, including registrations in
a loop. Inferred hooks can accept local captures through the ordinary
registration methods, but each registration changes the executor's type.

Run from this directory:

```sh
cargo test -p basin-evaluation-local
cargo test --workspace
cargo test -p basin-evaluation-application --test executor_transfer
cargo tree -p basin-evaluation-local -e features -i basin
cargo tree -p basin-evaluation-application -e features -i basin
```
