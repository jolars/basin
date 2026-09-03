{
  pkgs,
  ...
}:
{
  # LP64 OpenBLAS for running the nalgebra LAPACK feature tests locally.
  # The latest feature forwards `lapack-custom`, so LAPACK symbols are supplied
  # at link time—point RUSTFLAGS at this path:
  #   RUSTFLAGS="-L $OPENBLAS_LP64_LIB -l openblas" \
  #     cargo test -p basin --features nalgebra_latest-lapack --test lapack_nalgebra
  # `openblasCompat` is the LP64 (32-bit int) build that matches `lapack-sys`;
  # the default `openblas` is ILP64 on 64-bit and segfaults at runtime.
  env.OPENBLAS_LP64_LIB = "${pkgs.openblasCompat}/lib";

  packages = with pkgs; [
    go-task
    llvmPackages.bintools
    liteparse
    cargo-llvm-cov
    cargo-flamegraph
    cargo-audit
    cargo-deny
    cargo-msrv
    gnuplot
    samply
    pprof
    wasm-pack
    perf
    go-task
    quartoMinimal
    shfmt
    resvg # SVG → PNG rasteriser for the `task logo` asset pipeline
    cmake # for nlopt-sys (competitor-bench only; builds bundled NLopt 2.9.1)
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
    };
  };
}
