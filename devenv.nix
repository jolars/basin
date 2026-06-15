{
  pkgs,
  ...
}:
{
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
