{
  pkgs,
  lib,
  ...
}:
{
  # The ROCm torch wheel (used by marker's GPU second pass, see tools/) is a
  # manylinux binary that dynaically links a handful of base system libs absent
  # from NixOS's default loader path. Exposed as a scoped variable (not a global
  # LD_LIBRARY_PATH, which would shadow libs for the Rust/R/gcc toolchains) and
  # applied only to the marker command in Taskfile.yml's ingest-paper-pages.
  env.TORCH_ROCM_LIB_PATH = lib.makeLibraryPath [
    pkgs.zstd
    pkgs.bzip2
    pkgs.xz
    pkgs.zlib
    pkgs.stdenv.cc.cc.lib
  ];

  # LP64 OpenBLAS for running the `nalgebra-lapack` feature's tests locally.
  # That feature forwards `lapack-custom`, so LAPACK symbols are supplied at
  # link time — point RUSTFLAGS at this path:
  #   RUSTFLAGS="-L $OPENBLAS_LP64_LIB -l openblas" \
  #     cargo test -p basin --features nalgebra-lapack --test lapack_nalgebra
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
