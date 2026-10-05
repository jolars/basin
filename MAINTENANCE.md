# Maintenance and releases

## Supported release lines

Basin develops 2.0 on `main` and maintains the latest 1.x release on `v1`, which
starts from v1.15.1. The 1.x line receives correctness, regression, and security
fixes throughout 2.0 development and for at least six months after the first
stable 2.0.0 release. We will record the earliest end-of-support date here when
2.0.0 ships and announce the end of support in the release notes. No end date is
set while 2.0 is in development.

Only the latest 1.x patch release is maintained; earlier minor releases do not
have separate maintenance branches. Applications can use `basin = "1"` and
update their lockfiles to receive fixes. The `v1` branch preserves its public
API, convergence defaults, backend feature meanings, Rust 1.87-compatible
feature set, and default WASM build. New features and breaking changes target
2.x.

The published [1.x API reference](https://docs.rs/basin/1/basin/) remains
available when the main documentation moves to 2.x. Report the Basin version,
backend features, and a reproducer when opening an issue.

## Backporting fixes

Fixes normally land on `main` first. For a bug affecting 1.x, cherry-pick the
fix and its regression test with `git cherry-pick -x <commit>` onto a branch
based on `origin/v1`. Adapt the implementation if the APIs have diverged, run
the relevant checks from [CONTRIBUTING.md](CONTRIBUTING.md#commands), and open a
PR targeting `v1`. A fix developed on `v1` should also be ported to `main` when
applicable.

Keep the conventional `fix:` commit or PR title so Versionary proposes a patch
release. Review the planned version before merging a release PR: Versionary does
not enforce a version range from the branch name.

Backport source changes and tests. Each branch owns its versions, changelog, and
`.versionary-manifest.json`; let Versionary generate those independently. Do not
merge `main` into `v1` or cherry-pick release bookkeeping between them.

## Release automation

CI checks pushes and PRs targeting both base branches. Versionary 1.5.0 or newer
runs after the CI checks pass, with one release job at a time per branch.
`VERSIONARY_BASE_BRANCH` selects the base branch, while `release-branch` in that
branch's `versionary.jsonc` names the generated release PR branch:

  | Base branch | Generated release PR branch | Purpose                           |
  | ----------- | --------------------------- | --------------------------------- |
  | `main`      | `versionary/release`        | Prepare the next 2.x release.     |
  | `v1`        | `versionary/release-v1`     | Release compatible fixes for 1.x. |

The commit starting 2.0 development carries `Release-As: 2.0.0`, so Versionary
targets 2.0.0 on `main` from the start. Keep its release PR open while
implementing the planned breaking changes. Merging that PR is the decision to
publish 2.0.0. Publish further 1.x releases only from `v1`.

Merging a release PR creates its tag and GitHub Release. The existing Publish
Crates workflow publishes the tagged crate to crates.io. The `RELEASE_TOKEN`
must allow the tag and release events to trigger these downstream workflows.

While 1.x is the stable line, its `release-latest` setting remains `true`. The
setting controls GitHub's Latest designation; it does not turn a release into a
prerelease or prevent crates.io publication. Publishing beta or RC versions
needs a separate prerelease policy.

## Publishing 2.0.0

Before merging the first stable 2.0 release PR:

1. Set `release-latest` to `false` in `versionary.jsonc` on the `v1` branch, and
   finish any in-flight 1.x release jobs. Keep it `true` on `main`. Later 1.x
   patches will then leave 2.x marked as Latest.
2. Record the earliest end-of-support date for 1.x here, at least six months
   after the 2.0.0 release date, and update both branches' README policy.
3. Publish migration guidance and retain links to the 1.x API reference.
4. Merge the reviewed 2.0.0 release PR after all required checks pass, then
   verify crate publication and website deployment.

## Website deployment

The Website workflow builds and checks both branches. Deployment starts from a
published release, after its GitHub metadata exists. Only a stable
`vMAJOR.MINOR.PATCH` tag that is currently marked Latest may deploy to
<https://basin.rs/> on Cloudflare Workers. A 2.0 prerelease or a later 1.x
maintenance release cannot replace the current stable documentation. Deployments
are serialized and recheck Latest immediately before deploying.

The build job uploads the checked static site as an artifact. The deployment
job publishes that artifact and the Worker together, using the
`cloudflare-workers` environment. It needs the `CLOUDFLARE_API_TOKEN` repository
secret and `CLOUDFLARE_ACCOUNT_ID` repository variable. See the
[hosting guide](web/worker/README.md) for local preview, credentials, and domain
migration. Keep the deployment workflow and Worker configuration synchronized
between `main` and `v1`.

To redeploy, manually run the Website workflow on `main` or `v1` and set
`release_tag` to the current stable tag. The workflow builds the tagged site's
sources with the selected branch's hosting tools, then rechecks Latest before
publishing. Leave the input empty to build and check the selected branch
without deploying. Tags that already contain the Workers workflow can also be
dispatched directly. Do not rerun older tags' GitHub Pages deployment workflows.
