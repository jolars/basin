# Changelog

## [0.5.0](https://github.com/jolars/basin/compare/v0.4.0...v0.5.0) (2026-05-26)

### Breaking changes
- make barrier/augmented-Lagrangian methods inner-solver-agnostic ([`5a4f369`](https://github.com/jolars/basin/commit/5a4f369773ff3828fe14853deaf2982b9630ad2f))
- rename BoxConstrained → BoxConstraints ([`7d60b0f`](https://github.com/jolars/basin/commit/7d60b0ffdf42cc655a2df8b60ddd1f08467cdc2e))

### Features
- add `MatVec` + `MatTransposeVec` to vec, ndarray backends ([`320173f`](https://github.com/jolars/basin/commit/320173f5bf0ae0389163a3586dcdfde01f6e0509))
- make barrier/augmented-Lagrangian methods inner-solver-agnostic ([`5a4f369`](https://github.com/jolars/basin/commit/5a4f369773ff3828fe14853deaf2982b9630ad2f))
- add linear equality constraints (Ax = b) and augmented Lagrangian method ([`aa038bc`](https://github.com/jolars/basin/commit/aa038bcff613d173dae584e437fa38151ee2cbf9))
- add linear inequality constraints (Ax ≤ b) and log-barrier method ([`c7f2e24`](https://github.com/jolars/basin/commit/c7f2e24f493a7d8ae3c92016e747761ce8c67f67))
- **web:** live contour beside the playground code (phase 2) ([`8a21eda`](https://github.com/jolars/basin/commit/8a21edabadc86ed89c114e79f54cbe855dbd263c))
- **web:** add interactive code-gen playground to landing page ([`29f10b2`](https://github.com/jolars/basin/commit/29f10b2e65bae65e3ce60983fb05c5206d99419b))
## [0.4.0](https://github.com/jolars/basin/compare/v0.3.0...v0.4.0) (2026-05-21)

### Features
- add per-coordinate initial step-size to CMA-ES (with_stds) ([`fb97d0a`](https://github.com/jolars/basin/commit/fb97d0afd8d6c389cd747ab7ea0942cc4838dc77))
- add numerical gradients/hessians/jacobians ([`bef3fdc`](https://github.com/jolars/basin/commit/bef3fdcd2dd5b5350faafbe510fa75de8a895993))
- add Polyak momentum to GD solver ([`9b9ab93`](https://github.com/jolars/basin/commit/9b9ab9351e6814e9310cd4970918855d7ce5a8f9))
- **web:** restructure into landing + docs + visualizer site ([`727de34`](https://github.com/jolars/basin/commit/727de34e4afc51431003b171b850ae74dfc0efcb))
## [0.3.0](https://github.com/jolars/basin/compare/v0.2.0...v0.3.0) (2026-05-20)

### Breaking changes
- **nlls:** damp Levenberg-Marquardt with Marquardt diagonal scaling ([`fdde644`](https://github.com/jolars/basin/commit/fdde64488cc788052f2016959657a86e5e142ba4)), closes [#6](https://github.com/jolars/basin/issues/6)

### Features
- add competitor-bench crate for comparisons ([`bbacbf6`](https://github.com/jolars/basin/commit/bbacbf6d12aa486a3eac521b34a6f47f5cd5a66b))
- **nlls:** add MINPACK ftol/xtol convergence tests to Levenberg-Marquardt ([`2bad473`](https://github.com/jolars/basin/commit/2bad473685f87d0a6b472e546689a1ea567819ca)), closes [#8](https://github.com/jolars/basin/issues/8)
- **termination:** add RelativeGradientTolerance framework criterion ([`70b5cb6`](https://github.com/jolars/basin/commit/70b5cb6f6c9e12f6c07ace6c256e583167cbd7ab))
- **nlls:** add MINPACK gtol (relative gradient) test to Levenberg-Marquardt ([`5a60145`](https://github.com/jolars/basin/commit/5a601451ea4c41b4a1eee2424cb5d58766962fbc))

### Bug Fixes
- **nlls:** damp Levenberg-Marquardt with Marquardt diagonal scaling ([`fdde644`](https://github.com/jolars/basin/commit/fdde64488cc788052f2016959657a86e5e142ba4)), closes [#6](https://github.com/jolars/basin/issues/6)

### Performance Improvements
- **nlls:** reuse JᵀJ and Jᵀr across rejected Levenberg-Marquardt steps ([`deb9bae`](https://github.com/jolars/basin/commit/deb9bae565397b928062db836e517f852446639c))
## [0.2.0](https://github.com/jolars/basin/compare/v0.1.0...v0.2.0) (2026-05-19)

### Breaking changes
- bump MSRV to 1.91.1 and unwind CRAN-related dep pins ([`2a64a51`](https://github.com/jolars/basin/commit/2a64a518d55db9b106fa2b2d08729462725da175))

### Features
- add `ndarray-blas` feature ([`b424d9c`](https://github.com/jolars/basin/commit/b424d9c9cba0dd11b900c252464a20bba7bde395))
- add `parallel` feature ([`7919632`](https://github.com/jolars/basin/commit/79196320d669d80a16581a03fdbea1917e6821a5))
- implement L-BFGS (unbounded) ([`4df3dbb`](https://github.com/jolars/basin/commit/4df3dbb1296ffdc55d786addc08842226d3cd905))

### Performance Improvements
- **nlls:** cache r and J across iterations in GN, LM, TRF ([`aef1197`](https://github.com/jolars/basin/commit/aef11970d2fe24beabd5b63f93fd3f4633829049))
## [0.1.0](https://github.com/jolars/basin/compare/v0.0.1...v0.1.0) (2026-05-19)

### Breaking changes
- rename step_size to line_search ([`15b0d42`](https://github.com/jolars/basin/commit/15b0d4240e00c2c53920ad3fafdac3142bb9f3e3))
- simpplify API ([`b19d6b7`](https://github.com/jolars/basin/commit/b19d6b7688e41fd79fe61e59c6f14d2793f592b1))
- update return value to `OptimizationResult` ([`7943351`](https://github.com/jolars/basin/commit/794335183ceb812b09014cbd3f0adf4c0582001a))

### Features
- add `<Mode>` trait for NelderMead solver ([`2f4421a`](https://github.com/jolars/basin/commit/2f4421a361d429e56df0fcc7f7d8c97c16f8c0d4))
- add bounded cma-es injected solver ([`38b3a9b`](https://github.com/jolars/basin/commit/38b3a9b58d6ccd315cde791ea2695cce57627fb8))
- add L-BFGS-B solver ([`570db88`](https://github.com/jolars/basin/commit/570db8893a7bc5c175e2f8fe16efe0a07121112e))
- add Moré-Thuente line search ([`991a2e9`](https://github.com/jolars/basin/commit/991a2e9fa5e80eae6225b91cdc2bc25f4af83d66))
- add MA-LSCh-CMA solver ([`7c37901`](https://github.com/jolars/basin/commit/7c379010ea821f65230ab969734de56c0e537f9e))
- add SSGA solver ([`1c4fc46`](https://github.com/jolars/basin/commit/1c4fc4675a48b24c9c2b01aa71195b0ac3bc8472))
- add Rastrigin test problem ([`6cbc73e`](https://github.com/jolars/basin/commit/6cbc73e93f4f0f18f25d6d9a83192c77f93a7e0e))
- add CmaInject solver ([`f46b685`](https://github.com/jolars/basin/commit/f46b6853b337cbadc847640923fdf961492ccc2f))
- add inner executor ([`3638d02`](https://github.com/jolars/basin/commit/3638d02c546dda32c322786c8636af2c89e94af5))
- add bounded CMA-ES solver ([`2508b6d`](https://github.com/jolars/basin/commit/2508b6d8fba04df305272bde69009255c32d2e4a))
- add CMA-ES solver ([`d5161c2`](https://github.com/jolars/basin/commit/d5161c215185ba218e1526f0478de9b56080df47))
- add random search solver ([`a84b3d0`](https://github.com/jolars/basin/commit/a84b3d0178906f4bab40038b5473e085f2986baa))
- add a trf solver ([`ba2a5b7`](https://github.com/jolars/basin/commit/ba2a5b74ce223f4bfb44d29b055037f83f517dfe))
- implement projected gradient ([`6137503`](https://github.com/jolars/basin/commit/6137503675eced844b71275391f5a7e7577bf30d))
- implement Levenberg-Marquardt solver ([`edac7a2`](https://github.com/jolars/basin/commit/edac7a2ae48601de71c939a90718d400140b45e0))
- add sparse linear algebra backends ([`4544ebd`](https://github.com/jolars/basin/commit/4544ebd2882227aa3ead4be2fce59db614298a88))
- add Gauss Newton solver ([`7da924b`](https://github.com/jolars/basin/commit/7da924b8cf5651c3645e9a332b5aeeb1c351500f))
- design linalg trait math backend ([`51a2130`](https://github.com/jolars/basin/commit/51a21303f76227fac0c061c09a5e6ea02b3176b1))
- add Residual and Jacobian traits ([`04e4d79`](https://github.com/jolars/basin/commit/04e4d79a707b45cea23d544120b004d79136b75a))
- **web:** make curves smoother ([`ccd6bb8`](https://github.com/jolars/basin/commit/ccd6bb864944a9174f5b76f0c7ab7b969baa0695))
- add Goldstein-Price problem ([`b64fd80`](https://github.com/jolars/basin/commit/b64fd800a74e8ce26c744cdac0a01d051a622676))
- add Matyas and McCormick problems ([`8ee942b`](https://github.com/jolars/basin/commit/8ee942b895620220584b8202cdca4478223369c9))
- **web:** add theme toggle ([`ab4f0d7`](https://github.com/jolars/basin/commit/ab4f0d75c766306c7ff7f870af31b4b0cf980e7b))
- add wasm crate and web app ([`9f3b2bb`](https://github.com/jolars/basin/commit/9f3b2bb49710936cbc082dbee7622b7d2b69f663))
- add Booth and Beale problems ([`f395e61`](https://github.com/jolars/basin/commit/f395e61ec62c79cb90ce7112e810e83166fa4e80))
- add Sphere test problem ([`8a41cc0`](https://github.com/jolars/basin/commit/8a41cc0cff7094c41d86fe5bef421d527a8b04ea))
- add scaffolding and metadata for example problems ([`790807d`](https://github.com/jolars/basin/commit/790807d64c9948a5c018191a513acdce0e9b0669))
- consolidate problems into a separate module ([`639567b`](https://github.com/jolars/basin/commit/639567b402a7ba0e3bac1afebaf08f2d9089090e))
- allow `next_iter` to report mid-termination ([`b1b2b36`](https://github.com/jolars/basin/commit/b1b2b3664e7acdeddd12063dad536e6abc71c9b7))
- implement bfgs solver ([`6c546f3`](https://github.com/jolars/basin/commit/6c546f3b78296dcd29337b8926d2d4a62888f5fb))
- rename step_size to line_search ([`15b0d42`](https://github.com/jolars/basin/commit/15b0d4240e00c2c53920ad3fafdac3142bb9f3e3))
- add Brent method ([`e1c9605`](https://github.com/jolars/basin/commit/e1c9605d79ee7a48b064fff54585dd1e678ebd1f))
- add `run_loop` to compose inner solvers ([`67438a0`](https://github.com/jolars/basin/commit/67438a0aee40fa4ccdf9569146b97959df1dac12))
- add faer backend ([`c5ca8dd`](https://github.com/jolars/basin/commit/c5ca8dd10e0da2a9719d6dbb96d1fa14728f8973))
- add ndarray backend ([`a3932ed`](https://github.com/jolars/basin/commit/a3932edbd90eb10c949f59f154218e5598ef2438))
- add gradient evals counts ([`845c986`](https://github.com/jolars/basin/commit/845c98657fa543831e55c14cd9375552bed43fff))
- implement cost eval tracking ([`bc9de31`](https://github.com/jolars/basin/commit/bc9de31099362555954a85d4dfe28a3ff68e0de1))
- add BasicSimplexState ([`5864c1c`](https://github.com/jolars/basin/commit/5864c1c48114635b82c5ad40a924a879b01018d9))
- simpplify API ([`b19d6b7`](https://github.com/jolars/basin/commit/b19d6b7688e41fd79fe61e59c6f14d2793f592b1))
- update return value to `OptimizationResult` ([`7943351`](https://github.com/jolars/basin/commit/794335183ceb812b09014cbd3f0adf4c0582001a))
- **solvers:** implement Nelder-Mead ([`6ddd8d6`](https://github.com/jolars/basin/commit/6ddd8d601f39932281eabf3cc5167195cbef5730))
- add termination criteria ([`ecafbdd`](https://github.com/jolars/basin/commit/ecafbdde0da260e982173241f5d0d0e1edbaafc7))
- add nalgebra backend ([`4b97e17`](https://github.com/jolars/basin/commit/4b97e1753d6043ced20443e01b795d8d8a4df91f))
- add `StepSize` trait and add to GD solver ([`d26ce45`](https://github.com/jolars/basin/commit/d26ce4531cdcabd28d52f0b0a3c6012133e06f0c))
- add gradient descent ([`d7b2fdc`](https://github.com/jolars/basin/commit/d7b2fdc8c4a1c037575e60d35c59f4d04c97de9b))

### Bug Fixes
- **web:** decrease height of plot window ([`e964039`](https://github.com/jolars/basin/commit/e964039cd8477d1b96dad988bfb357a4cc170a26))
