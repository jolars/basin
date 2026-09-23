# Backend feature unification fixture

The two libraries independently request different versions of all three external
backends. The application combines them without configuring Basin directly.
Run from the repository root:

```sh
cargo test --manifest-path crates/basin/tests/fixtures/backend_unification/Cargo.toml -p basin-backend-older
cargo test --manifest-path crates/basin/tests/fixtures/backend_unification/Cargo.toml -p basin-backend-newer
cargo run --manifest-path crates/basin/tests/fixtures/backend_unification/Cargo.toml -p basin-backend-application
python3 crates/basin/tests/fixtures/backend_unification/check_features.py
```

Keep this fixture in its own workspace so the main workspace cannot supply a
missing feature accidentally.

The metadata check verifies that each LAPACK feature enables only its matching
adapter and does not activate the frozen legacy feature alias. It does not link
BLAS or LAPACK.
