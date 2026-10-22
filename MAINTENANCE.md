# Maintenance and releases

## Supported release lines

Basin prepares 3.0 on `main` and maintains the latest 1.x release on `v1`, which
starts from v1.15.1. The optional constraint API on `main` is a breaking change
and must ship as 3.0.0. Keep published 2.x versions available; do not yank or
replace them for this redesign. The 1.x line receives correctness, regression,
and security fixes through at least April 19, 2027, and for at least six months
after the first stable 2.0.0 release, whichever is later. We will announce the
end of support in the release notes.

Only the latest 1.x patch release is maintained; earlier minor releases do not
have separate maintenance branches. Applications can use `basin = "1"` and
update their lockfiles to receive fixes. The `v1` branch preserves its public
API, convergence defaults, backend feature meanings, Rust 1.87-compatible
feature set, and default WASM build. New features and breaking changes land on
`main`. Do not backport the 3.0 constraint API to 2.x or 1.x.

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
  | `main`      | `versionary/release`        | Prepare the next 3.x release.     |
  | `v1`        | `versionary/release-v1`     | Release compatible fixes for 1.x. |

The commit starting 2.0 development carries `Release-As: 2.0.0`, so Versionary
targets 2.0.0 on `main` from the start. Keep its release PR open while
implementing the planned breaking changes. Merging that PR is the decision to
publish 2.0.0. Publish further 1.x releases only from `v1`.

Merging a release PR creates its tag and GitHub Release. The existing Publish
Crates workflow publishes the tagged crate to crates.io. The `RELEASE_TOKEN`
must allow the tag and release events to trigger these downstream workflows.

The `main` branch sets `release-latest` to `true`; the `v1` branch sets it to
`false` for the 2.0 handoff. Maintenance releases therefore leave GitHub's
Latest designation unchanged. The setting does not turn a release into a
prerelease or prevent crates.io publication. Publishing beta or RC versions
needs a separate prerelease policy.

## Website deployment

The Website workflow builds and checks both branches. Deployment starts from a
published release, after its GitHub metadata exists. Only a stable
`vMAJOR.MINOR.PATCH` tag that is currently marked Latest may deploy to
<https://basin.rs/> on Cloudflare Workers. A 2.0 prerelease or a later 1.x
maintenance release cannot replace the current stable documentation. Deployments
are serialized and recheck Latest immediately before deploying.

The build job uploads the checked static site as an artifact. The deployment job
publishes that artifact and the Worker together, using the `cloudflare-workers`
environment. It needs the `CLOUDFLARE_API_TOKEN` repository secret and
`CLOUDFLARE_ACCOUNT_ID` repository variable. See the [hosting
guide](web/worker/README.md) for local preview, credentials, and domain
migration. Keep the deployment workflow and Worker configuration synchronized
between `main` and `v1`.

To redeploy, manually run the Website workflow on `main` or `v1` and set
`release_tag` to the current stable tag. The workflow builds the tagged site's
sources with the selected branch's hosting tools, then rechecks Latest before
publishing. Leave the input empty to build and check the selected branch without
deploying. Tags that already contain the Workers workflow can also be dispatched
directly. Do not rerun older tags' GitHub Pages deployment workflows.
