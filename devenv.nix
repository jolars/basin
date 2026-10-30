{
  config,
  pkgs,
  ...
}:
{
  # LP64 OpenBLAS for running the nalgebra LAPACK feature tests locally.
  # The `nalgebra-lapack` feature forwards `lapack-custom`, so LAPACK symbols are supplied
  # at link time—point RUSTFLAGS at this path:
  #   RUSTFLAGS="-L $OPENBLAS_LP64_LIB -l openblas" \
  #     cargo test -p basin --features nalgebra-lapack --test lapack_nalgebra
  # `openblasCompat` is the LP64 (32-bit int) build that matches `lapack-sys`;
  # the default `openblas` is ILP64 on 64-bit and segfaults at runtime.
  env.OPENBLAS_LP64_LIB = "${pkgs.openblasCompat}/lib";

  # Nix manages pnpm; downloaded standalone binaries cannot run on NixOS.
  env.pnpm_config_pm_on_fail = "ignore";

  # bindgen loads libclang for the optional C L-BFGS-B benchmark reference.
  env.LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

  packages = with pkgs; [
    go-task
    llvmPackages.bintools
    liteparse
    cargo-llvm-cov
    cargo-flamegraph
    cargo-audit
    cargo-deny
    cargo-msrv
    hyperfine
    util-linux # taskset keeps benchmark CPU affinity reproducible.
    valgrind
    gnuplot
    samply
    pprof
    wasm-pack
    lychee
    perf
    quartoMinimal
    shfmt
    resvg # SVG → PNG rasteriser for the `task logo` asset pipeline
    cmake # for nlopt-sys (competitor-bench only; builds bundled NLopt 2.9.1)
    llvmPackages.clang # for the optional lbfgsb reference and bindgen headers.
  ];

  languages = {
    rust = {
      enable = true;

      toolchainFile = ./rust-toolchain.toml;
    };

    fortran = {
      enable = true;
    };

    r = {
      enable = true;
    };

    python = {
      enable = true;

      directory = "./tools";

      venv.enable = true;
      uv = {
        enable = true;
        sync = {
          enable = true;
          allGroups = true;
        };
      };
    };

    javascript = {
      enable = true;

      pnpm.enable = true;
    };

    typescript = {
      enable = true;
    };
  };

  git-hooks = {
    hooks = {
      clippy = {
        enable = true;

        settings = {
          allFeatures = true;
        };
      };

      rustfmt = {
        enable = true;
      };

      # Cargo fmt does not follow include! into the shared backend sources.
      rustfmt-shared = {
        enable = true;
        name = "rustfmt (shared sources)";
        entry = "${config.languages.rust.toolchainPackage}/bin/rustfmt --edition 2024";
        files = "^crates/basin/(src/core/math/(nalgebra|nalgebra_sparse|ndarray|faer|faer_sparse)_backend|tests/support/nalgebra_factorizations|tests/fixtures/backend_unification/check)\\.rs$";
        language = "system";
      };

      # The downstream fixture has its own workspace outside the root workspace.
      rustfmt-fixtures = {
        enable = true;
        name = "rustfmt (backend fixtures)";
        entry = "${config.git-hooks.hooks.rustfmt.package}/bin/cargo-fmt fmt --manifest-path crates/basin/tests/fixtures/backend_unification/Cargo.toml --all";
        files = "^crates/basin/tests/fixtures/backend_unification/.*\\.rs$";
        language = "system";
        pass_filenames = false;
      };
    };
  };
}
