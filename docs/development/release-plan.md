<a name="velocicyber-beta-release-plan"></a>

# ![Velocicyber beta release plan](../../assets/docs/headings/h1-velocicyber-beta-release-plan.svg)

<a name="scope-and-boundaries"></a>

## ![Scope and boundaries](../../assets/docs/headings/h2-scope-and-boundaries.svg)

- Work in this release repository only. The working prototype is read-only.
- Keep the GitHub repository private. Never copy credentials, SSH keys, runtime state, conversations, training data, or model weights into Git.
- Never launch paid GPUs or upload datasets as part of release validation.
- Update this checklist and the checkpoint after each milestone.

<a name="milestones"></a>

## ![Milestones](../../assets/docs/headings/h2-milestones.svg)

- [x] Create an isolated source-only copy and record prototype file hashes privately.
- [x] Verify GitHub authentication and check that the requested repository is absent.
- [x] Replace owner-specific configuration and remove the prototype-only GPU model.
- [x] Add first-run credential setup with hidden input, skip choices, and private file permissions.
- [x] Prompt for separate user-owned main LLM and sentinel dataset IDs; optionally create private datasets only after confirmation.
- [x] Provide a portable installation path and optional pinned Pi model downloads.
- [x] Preserve engine/runtime compatibility and third-party notices.
- [x] Audit source/configuration, staged Git content, and the final published commit for credentials and machine-specific identifiers.
- [x] Run offline regression, first-run setup, and clean-install checks.
- [x] Create the private Velocicyber GitHub repository, push the audited source, and verify its visibility and commit.
- [x] Verify prototype source/configuration hashes are unchanged (106 files, including permissions).

<a name="public-release-gates"></a>

## ![Public-release gates](../../assets/docs/headings/h2-public-release-gates.svg)

- [ ] Choose a license for original harness code before making the repository public.
- [ ] Run a clean Pi install and opt-in model installation on a separate test installation.
- [ ] Conduct a wider security review; automated scanning cannot establish absence of all vulnerabilities.
- [x] Replace the former repository with independent fresh history and verify
  that the old email-bearing commit is unavailable in the replacement.

<a name="checkpoint"></a>

## ![Checkpoint](../../assets/docs/headings/h2-checkpoint.svg)

The isolated release installs in a fresh Python environment. First-run prompts were exercised in a terminal with dummy dataset IDs and skipped keys; no provider calls or downloads were made by setup validation. Automated setup tests cover hidden credentials, private files, skipped inputs, separate datasets, explicit creation consent and optional model download dispatch.

Source and Git-index audits found no prototype credentials or owner-specific deployment identifiers. Loopback/wildcard bindings are intentional; the private IP in the sanitizer test is synthetic. Recorded prototype hashes and modes remain unchanged. Git history is fresh; no prototype history is included.

Validation completed: **472 tests passed** in 186.44 seconds, including the official engine against local fake model servers. Ruff, dependency consistency and Git whitespace checks passed. Test fixtures now explicitly opt into collection and register their fake model without adding it to the shipped menu. API alias checks use canonical provider IDs. The release additionally guards a menu-mount timer race. No GPU launches, actual inference, dataset uploads, or model-weight downloads are part of this validation.

Published to the private **Velocicyber** repository. GitHub confirmed private visibility, a matching commit, and exactly the 115 audited files. The published commit audit found no credential/private-identifier matches. Pi downloads are optional and show sizes first. Source, configuration and test changes remain entirely inside this release copy.

<a name="next-session"></a>

## ![Next session](../../assets/docs/headings/h2-next-session.svg)

The initial release preparation is complete. Work only in this release copy; do not change the prototype. Before public release, resolve the public-release gates above. Live provider tests or dataset uploads require explicit approval. The optional Pi model installers have not downloaded additional weights during initial release preparation.

<a name="curl-installation-follow-up"></a>

## ![Curl installation follow-up](../../assets/docs/headings/h2-curl-installation-follow-up.svg)

- [x] Add a standalone curl-based downloader with commit-pinned source archives, private GitHub authentication and a source-only option.
- [x] Document authenticated private installation and the future public command.
- [x] Verify archive safety, credential handling and preservation of existing directories with offline tests (18 passed).
- [x] Publish and validate the real private GitHub download into a temporary directory.

No changes are made to the working prototype or repository visibility.

Curl validation: **18 focused tests passed**, plus a real authenticated GitHub entrypoint/source download, fresh Python dependency installation and offline `doctor` check. All **117 tracked source files** matched the published commit. The test used a destination containing spaces, did not save its GitHub credential, and removed its temporary installation afterward. No model weights, paid inference or GPU resources were started.

<a name="structure-refactor"></a>

## ![Structure refactor](../../assets/docs/headings/h2-structure-refactor.svg)

- [x] Save the original release structure as Git tag `pre-structure-refactor-20261021`.
- [x] Group application modules into engine, providers, models, datasets, sentinel and UI packages.
- [x] Extract reusable UI widgets, command definitions, styles and pure dashboard rendering.
- [x] Group runtime pins under `config/locks/` and separate install/runtime/maintenance/validation scripts.
- [x] Shorten the root README and move detailed guides into `docs/`.
- [x] Validate all offline/native-engine regressions and a clean source installation.
- [x] Audit the staged refactor and verify the prototype is unchanged.
- [x] Publish and verify the private refactor.

User-facing launch commands, configuration locations, runtime data and model
selection behavior stay the same. Internal Python import paths and direct
script paths have changed; see [the source map](architecture.md).

Refactor validation: the full suite passed **488 tests** and exposed two test
imports still using the former module layout. After correcting those imports,
all **17 upload tests passed**, including both failures. This validates all
**490 distinct regressions**, with the native engine using local fake providers.
No application behavior changes were needed to resolve the test failures.

A fresh source archive installed into a new directory containing spaces. The
CLI, offline doctor, maintenance wrapper, Pi installer help and offline UI all
worked. Dependency checks passed, and a built wheel contains every new package
and the extracted UI stylesheet. Ruff, Git whitespace checks and documentation
link checks passed. The 30 moved non-UI modules have identical syntax trees
after accounting for import and path changes. All five runtime/model pins are
unchanged. The source/index security audit found no prototype credentials or
deployment identifiers; all 106 recorded prototype file hashes and modes match.

Internal Python and direct script paths are reorganized. The supported root
launch/install commands and user configuration locations remain intact. The
current ARM64-only runtime installers are documented; x86-64 Linux and WSL2
portability is a separate future change, not a claimed result of this refactor.

GitHub confirmed that the refactored repository remains private and contains
exactly the 131 audited files. The pre-refactor checkpoint tag is also published
as `pre-structure-refactor-20261021`. The working prototype was not modified.

<a name="upstream-sources-index"></a>

## ![Upstream sources index](../../assets/docs/headings/h2-upstream-sources-index.svg)

- [x] Add [sources and upstream references](../sources.md) for the engine,
  provider integrations, serving runtimes, UI library and all registered model cards.
- [x] Document configured versions and authoritative pins alongside integration
  modules; distinguish hosted services and resolved GPU weight revisions from
  fixed local-model artifacts.
- [x] Link the index from the README and third-party notices.

Validation: all 51 local links in the new index resolve. Documented engine,
SDK, Pi runtime, sentinel runtime and Python dependency pins match the checked-in
configuration. Official documentation/model-card pages were inspected; the
OrcaCyber model-card URL returned HTTP 200 in a public, unauthenticated check.
This change is documentation-only; the application regression suite was not
rerun. No providers were invoked for inference, and no model weights or training
records were transferred.

<a name="sage-model-integration--2026-10-23"></a>

## ![SAGE model integration — 2026-10-23](../../assets/docs/headings/h2-sage-model-integration--2026-10-23.svg)

The owner explicitly requested this integration in the working prototype first,
then in the private release. This is an authorized exception to the earlier
prototype-preservation instruction for this model change only.

- [x] Inspect public model configuration, full weight index and pinned vLLM support.
- [x] Add the prototype GPU profile, menu entry, runtime pin and projected timings.
- [x] Port the focused integration into the release and update model/source docs.
- [x] Complete offline planning, deployment, readiness and menu regression checks.
- [x] Audit and publish to the private GitHub repository.
- [ ] Validate inference/tool use on Lambda after explicit GPU-test authorization.

Checkpoint: recommend one B200 (180 GB), with a conservative 96 GiB profile floor
and 32K initial context. The complete indexed checkpoint is 71.9 GB, including
MTP weights; speculative decoding is disabled. Fresh/cached estimates are
projected. No GPU launches, model-weight downloads or dataset uploads were made.
Published SAGE integration commit `ee2cffb` to the private repository. GitHub
confirmed the matching commit and all 133 audited files. Authentication was
restored using the owner-provided replacement credential; no credentials were committed.

SAGE validation: **72 prototype tests and 88 release tests passed**, covering the
new model plus existing GPU profiles and first-run configuration. Ruff and Git
whitespace checks passed. Source scanning found no credentials or deployment
identifiers; the only flagged IP remains the pre-existing synthetic sanitizer
fixture. The preservation audit confirms exactly four explicitly authorized
prototype source/configuration edits, plus the new offline SAGE test. All other
recorded prototype hashes and permissions remain unchanged, apart from the
owner's subsequent credential replacement. A private backup of the four original
files and a before/after hash record are retained.

<a name="responsible-use-disclaimer--2026-10-30"></a>

## ![Responsible-use disclaimer — 2026-10-30](../../assets/docs/headings/h2-responsible-use-disclaimer--2026-10-30.svg)

- [x] Add a prominent native GitHub CAUTION alert beneath the README branding.
- [x] State the authorized red-team/blue-team research purpose, restrictions on
  illegal or unethical use, user authorization/scope responsibilities, possible
  civil/criminal consequences, and an as-is/liability disclaimer limited by
  applicable law.
- [x] Verify the alert using GitHub's Markdown renderer and check the patch for
  whitespace errors. Existing README edits and branding are preserved.

Documentation-only change; application tests were not rerun. No inference,
GPU operations, uploads of training data, or prototype changes were performed.

<a name="readme-accuracy-review--2026-10-30"></a>

## ![README accuracy review — 2026-10-30](../../assets/docs/headings/h2-readme-accuracy-review--2026-10-30.svg)

- [x] Review every section of the root README against the current release code,
  model registries, slash-menu definitions, runtime pins and setup workflow.
- [x] Update all four README files: root, SCOPE, SSH keys and chat templates.
- [x] Correct claims about arbitrary model support, guaranteed scope enforcement,
  built-in training, GPU memory implying faster inference, and automatic engine
  syncing. Document current models, opt-in data handling, manual updates,
  supported platform, and live-validation limitations.
- [x] Correct the linked installation note: engine platform guards exist; automatic
  package selection for additional platforms is not implemented. Describe GPU
  profiles as registered rather than implying every profile is live-validated.
- [x] Preserve the branding and responsible-use disclaimer verbatim.

Validation: all 26 local documentation links/anchors resolve; all 16
main-menu commands are represented and match the release registry. Counts match
2 API models, 2 Pi coding models, 5 GPU profiles and 2 sentinel choices. Startup
permissions/defaults, the BugTrace template hash/source pin and engine platform
guards match the documentation. GitHub rendered the caution alert and all three
tables. Git whitespace checks passed. This is a documentation-only update;
application tests and live provider calls were not repeated.

<a name="beta-release-wording--2026-10-30"></a>

## ![Beta release wording — 2026-10-30](../../assets/docs/headings/h2-beta-release-wording--2026-10-30.svg)

- [x] Label the release **Beta — under active development** and remove the former
  release-stage wording from the README, security notes and contributor guidance.
- [x] Rename this checklist to `release-plan.md` and update documentation links.
- [x] Preserve the responsible-use disclaimer, release gates and repository
  visibility requirements. Private-download instructions remain conditional.

Validation: local documentation links resolve, obsolete release-stage references
are absent, and Git whitespace checks pass. This changes release wording only;
application tests and live provider calls were not repeated. The working
prototype and repository visibility are unchanged.

<a name="complete-model-guide--2026-10-30"></a>

## ![Complete model guide — 2026-10-30](../../assets/docs/headings/h2-complete-model-guide--2026-10-30.svg)

- [x] Expand `docs/models.md` beyond the SAGE-specific addition to cover all five
  Lambda profiles, both API models, both Pi coding models and both sentinels.
- [x] Document per-profile GPU/host/disk requirements, fresh/cached estimates,
  serving backends, context limits and current live-validation status.
- [x] Explain configured menu overrides, shared Qwen weights and sentinel
  installation behavior without changing application code or defaults.

Validation: model entries, resource floors, contexts and startup ranges checked
against the release registries and lock files; local documentation links and Git
whitespace checked. Documentation-only changes; no provider calls, inference,
model downloads or application tests were needed. Prototype files are preserved.

<a name="provider-privacy-links--2026-10-30"></a>

## ![Provider privacy links — 2026-10-30](../../assets/docs/headings/h2-provider-privacy-links--2026-10-30.svg)

- [x] Add official Hugging Face, Lambda Cloud and Adverserial AI privacy-policy
  links to `SECURITY.md` after opening and verifying each provider's page.
- [x] Preserve existing security guidance and check the documentation diff for
  whitespace errors. No application code, configuration or prototype changes.

<a name="documentation-styling-and-model-risk-warning--2026-10-30"></a>

## ![Documentation styling and model-risk warning — 2026-10-30](../../assets/docs/headings/h2-documentation-styling-and-model-risk-warning--2026-10-30.svg)

- [x] Add block-letter neon-green SVG headings to reader-facing Markdown while
  retaining heading labels as alt text and existing section links as anchors.
- [x] Keep body text selectable and the existing README branding intact; provide
  a dependency-free generator for consistent future heading updates.
- [x] Expand the README caution and SECURITY privacy section with a model-backdoor
  warning citing the user-specified paper, arXiv:2406.02619. Distinguish
  artifact integrity checks from assurance about model behavior.

Validation: 68 SVG headings across 12 documents previewed on light and dark
backgrounds; all 163 local links resolve. GitHub rendered the headings, custom
anchors, both paper links and the existing red caution alert. Original text is
preserved, generation is repeatable, and the generator passes Ruff. Documentation
and artwork only; application tests, provider inference and GPU operations were
not needed.

<a name="commit-email-privacy--2026-10-30"></a>

## ![Commit email privacy — 2026-10-30](../../assets/docs/headings/h2-commit-email-privacy--2026-10-30.svg)

- [x] Replace the owner's personal commit email with their verified GitHub
  noreply address in the published history; the existing checkpoint tag is clean.
- [x] Set this release checkout's future commit identity to the verified noreply
  address and set the account's primary email visibility to private.
- [x] Verify all 29 pre-checkpoint commits: file trees, messages, timestamps and
  parent relationships are preserved. The rewrite changes 23 commit IDs; prior
  commit signatures cannot be retained after their signed metadata changes.
- [x] Publish using an exact force-with-lease and sync this clean release checkout.
- [x] Superseded by the fresh repository replacement on 2026-11-04 below.
  The former repository was deleted; its retained history was not imported
  into the replacement. Private recovery records remain outside tracked files.

Other existing clones retain their old history. Re-clone before further work,
or carefully reapply unpublished changes onto the cleaned history; merging or
pushing an old branch can reintroduce the personal address. The working prototype
and separate validation checkout were not modified. This is a metadata cleanup,
not an application change; no provider inference or GPU operations were needed.

<a name="fresh-repository-replacement--2026-11-04"></a>

## ![Fresh repository replacement — 2026-11-04](../../assets/docs/headings/h2-fresh-repository-replacement--2026-11-04.svg)

- [x] Back up the original repository and copy its tracked source into an
  independent repository with a fresh root commit and a noreply identity.
- [x] Preserve both README branding images as local assets, so they do not
  depend on attachments belonging to the original repository.
- [x] Verify all 205 source/assets files, credential scanning, and the absence
  of the old email-bearing commit in the replacement repository.
- [x] Confirm the owner deleted the original private repository after GitHub
  rejected API deletion with HTTP 403. The original repository ID returns 404.
- [x] Rename the verified replacement to `beau-warren/Velocicyber` and switch
  the release checkout to a fresh clone of its independent history.

The replacement is private at the original project URL. Its initial import
contains the existing application code unchanged, with the two README images
stored locally and their links updated. The migration checkpoint adds one SVG,
bringing the tracked total to 206 files. The old email-bearing commit returns
404 through the replacement repository's API, and new commits use the verified
GitHub noreply identity. The historical checkpoints above describe work done
before this replacement; their original commits and tags are not included in
the new repository.

The working prototype is unchanged. A private recovery bundle, previous local
Git metadata and migration records are retained outside tracked files. Existing
clones must be re-cloned, or have unpublished changes carefully reapplied, before
pushing; do not merge or push previous history into this replacement. Application
tests were not repeated because executable code did not change.

<a name="optional-desktop-launcher--2026-11-04"></a>

## ![Optional desktop launcher — 2026-11-04](../../assets/docs/headings/h2-optional-desktop-launcher--2026-11-04.svg)

- [x] Offer a desktop launcher and bundled neon V icon during first-run setup,
  with explicit opt-in, Enter to skip, and a later retry through setup.
- [x] Respect configured Linux desktop folders, skip unavailable/headless
  desktops, preserve existing shortcuts and reuse matching installations.
- [x] Generate installation-specific paths using desktop-entry quoting and
  launch in the user's default terminal without starting anything at setup time.
- [x] Keep shortcut failures optional so the remaining setup can complete.

Validation: 43 focused onboarding, desktop and CLI tests passed. Native
`desktop-file-validate` and a Gio launch of a harmless temporary fixture verified
paths containing spaces, quotes, dollar signs, backticks, percent signs, equals
signs and backslashes. Ruff passed. All shortcut tests used temporary directories;
no real desktop shortcuts, model inference or GPU resources were created. The
working prototype is preserved.


<a name="completed-response-display--2026-11-04"></a>

## ![Completed response display — 2026-11-04](../../assets/docs/headings/h2-completed-response-display--2026-11-04.svg)

- [x] Prevent delayed provider previews from repeating an already displayed
  answer under “Partial response”. Native completion and proxy progress arrive
  independently; ignore preview text already covered by a completed response.
- [x] Preserve unfinished text after interruption, including after an earlier
  completed message, and reset preview tracking for each follow-up prompt.
- [x] Keep the change in the interface layer; inference requests, stored
  conversations, dataset records and the engine remain unchanged.

Validation: reproduced the duplicate with an offline delayed-callback fixture
before the fix. All 47 interface tests passed, including seven new cases for
late full/prefix previews, interruptions and follow-up turns. Ruff passed.
No live model calls, GPU launches or dataset uploads were needed.


<a name="unified-harness-and-engine-updates--2026-11-04"></a>

## ![Unified harness and engine updates — 2026-11-04](../../assets/docs/headings/h2-unified-harness-and-engine-updates--2026-11-04.svg)

- [x] Check official harness `main` and stable engine releases once per launch;
  one indicator reports an available update for either component or both.
- [x] Add `/update` with explicit confirmation of exact target revisions and
  automatic component selection; keep engine-only maintenance available.
- [x] Stage and test source before activation; preserve credentials, local
  configuration, SSH keys, models, workspaces, conversations and datasets.
- [x] Support clean official Git checkouts and receipt-backed curl installs,
  refusing local edits/conflicts and retaining source/environment rollback.
- [x] Keep private GitHub credentials out of Git remotes, model environments,
  cached status and redirected download authorization.

Validation: all 566 offline tests passed with the installed ARM64 engine,
including Git/archive update and rollback, dependency rollback, dirty trees,
active-session locking, failed candidate tests, private access failures,
redirect credential isolation, combined status and explicit menu approval.
Ruff passed for changed Python files. No paid inference, GPU launches, dataset
uploads or real engine update were performed. Existing installations need a
one-time source refresh to gain the updater before discovering subsequent changes.


<a name="lambda-ssh-setup-prompt--2026-11-04"></a>

## ![Lambda SSH setup prompt — 2026-11-04](../../assets/docs/headings/h2-lambda-ssh-setup-prompt--2026-11-04.svg)

- [x] Prompt for the registered Lambda SSH key name and private key file during
  setup; support numbered files already in ssh-keys and explicit external import.
- [x] Keep key contents out of configuration and terminal output, save name/path
  options in owner-only Modules.txt, and protect selected/imported keys with 0600.
- [x] Offer existing installations just this new section once after updating;
  skipping is remembered, and running setup explicitly reopens it.
- [x] Point GPU preflight messages to setup instead of requiring manual file edits.

Validation: 54 focused onboarding, SSH, desktop and CLI tests passed, including
fresh setup, one-time migration, external key import, private-file permissions,
non-key rejection and preservation of existing credentials/dataset settings.
Ruff passed. No Lambda requests or instance launches were made. This change is
published after updater support so a test installation can detect it normally.

<a name="automatic-repo-updates-on-launch--2026-11-05"></a>

## ![Automatic repo updates on launch — 2026-11-05](../../assets/docs/headings/h2-automatic-repo-updates-on-launch--2026-11-05.svg)

- [x] Add an opt-in `/update` preference for automatic source updates before the
  UI opens, defaulting to OFF. Engine upgrades retain explicit manual approval.
- [x] Reuse source staging, offline validation, private-file preservation,
  active-session exclusion and rollback. Restart with the validated source and
  reuse the completed source check instead of querying GitHub again.
- [x] Defer safely on failed checks/downloads, failed tests, local edits or active
  sessions. Block UI startup if interrupted activation needs rollback. Disable
  automatic updates after deliberate source rollback.
- [x] Keep demo, setup, doctor and headless run free of automatic installation.
  Preserve the repo-first workflow: publish fixes, then update consumer installs
  through the harness updater.

Validation: 601 full offline tests passed against the installed ARM64 0.0.59
engine. After adding the download-timeout regression, all 23 startup-update tests
passed; the earlier focused updater/interface run passed 83 tests. Ruff passed.
Checks cover opt-in/cancel/disable, restart flags, one-use revision-bound handoff,
stale update caches, interrupted recovery, active sessions and rollback behavior.
Existing Pi model weights were verified against their pinned SHA-256 hashes for
reuse by a separate consumer install, with independent runtimes and installation
receipts. No live inference, GPU launches, engine upgrades or dataset uploads
were performed; the prototype's source and configuration were preserved.

<a name="secure-setup-from-the-command-menu--2026-11-05"></a>

## ![Secure setup from the command menu — 2026-11-05](../../assets/docs/headings/h2-secure-setup-from-the-command-menu--2026-11-05.svg)

- [x] Add filterable `/setup` to reopen skipped credential, dataset, SSH,
  desktop-launcher and optional-install steps using the existing setup wizard.
- [x] Require confirmation before ending the session and terminating its GPU;
  finish UI/model teardown before hidden prompts. Retain saved conversations.
- [x] Exclude concurrent harness sessions during setup, reject command-line
  credential arguments, and retain hidden-input refusal and owner-only writes.
- [x] Reload configuration and return to the main menu, including after an
  interrupted wizard. Reuse completed version checks when the engine is unchanged.
- [x] Review all four READMEs and update only the main and SSH setup sections;
  scope and chat-template documentation remains accurate without changes.

Validation: 142 focused CLI, interface, setup, SSH, desktop and startup-update
checks passed. A further 81 CLI/maintenance checks passed, including actual
headless UI teardown and reopening, exclusive setup locking, interruptions and
update/rollback regressions. Ruff passed. All credentials, desktop shortcuts and
configuration writes in tests used temporary fixtures. No live inference, GPU
launches, provider credential probes or dataset uploads were performed.

<a name="gpu-upload-progress-refresh--2026-11-05"></a>

## ![GPU upload progress refresh — 2026-11-05](../../assets/docs/headings/h2-gpu-upload-progress-refresh--2026-11-05.svg)

- [x] Reduce the streamed transfer-progress display interval from 300 to 60
  seconds, preserving immediate initial/completion events and loading-stage changes.
- [x] Review all READMEs; document the interval in the main README only.

Validation: 23 targeted GPU, serving and interface tests passed, including the
59/60-second boundary, immediate completion and rejection of invalid progress.
Ruff passed. No GPU operations or live provider calls were needed.
