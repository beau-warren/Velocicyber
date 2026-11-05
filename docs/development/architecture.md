<a name="architecture-and-development"></a>

# ![Architecture and development](../../assets/docs/headings/h1-architecture-and-development.svg)

Velocicyber orchestrates a separately installed Open Interpreter engine. The
Python application owns configuration, model selection, provider lifecycles,
the terminal UI and data pipelines. Engine binaries remain upstream artifacts;
this repository does not fork or rebuild their implementation.

<a name="source-map"></a>

## ![Source map](../../assets/docs/headings/h2-source-map.svg)

| Location | Responsibility |
| --- | --- |
| `velocicyber/cli.py` | Supported CLI entrypoint and headless commands |
| `velocicyber/config.py`, `onboarding.py`, `state.py` | Installation configuration, first-run setup and runtime state |
| `velocicyber/engine/` | Engine subprocess, capture proxy, context/compaction, sessions and staged updates |
| `velocicyber/providers/` | API gateway, Lambda lifecycle/tunnels, Pi serving and hardware readings |
| `velocicyber/models/` | Registered GPU/Pi profiles, engine model metadata and startup estimates |
| `velocicyber/datasets/` | Raw records, curation, upload receipts and local retention |
| `velocicyber/sentinel/` | Asynchronous scope observation, sentinel runtime and separate training records |
| `velocicyber/ui/` | Application orchestration, reusable menus, command definitions and rendering |
| `config/locks/` | Pinned engine, model and serving runtime metadata |
| `scripts/install/` | Explicit installation/build tools |
| `scripts/runtime/` | Local serving helpers and code transferred to Lambda instances |
| `scripts/maintenance/` | Interactive engine update entrypoints |
| `scripts/validation/` | Explicit operator checks, including live provider tests |
| `tests/` | Offline regressions and native-engine tests against fake providers |

Paths in this document are relative to the repository root. Keep `install.sh`,
`velocicyber.sh` and `bootstrap.py` at the root: they are the user-facing install
and launch interfaces. Personal `.env`, `Modules.txt`, `SCOPE/` content and
`ssh-keys/` content keep their existing locations. Runtime state and model
weights remain under the Git-ignored `.runtime/` directory.

<a name="runtime-flow"></a>

## ![Runtime flow](../../assets/docs/headings/h2-runtime-flow.svg)

1. The CLI loads settings and completes optional first-run setup.
2. The UI selects a registered model profile and asks its provider adapter to
   prepare the runtime. Headless commands use the same engine and data modules.
3. The engine runner manages the upstream subprocess and capture proxy. Session
   history, compaction and completion handling remain outside UI rendering.
4. The optional sentinel observes executable actions asynchronously and can stop
   the task. It is not a pre-execution approval barrier.
5. With consent, the dataset pipelines curate main-agent and sentinel records
   separately and upload using the shared permission controls.

`ui/app.py` owns lifecycle and event handling. `widgets.py` owns filterable
menus and input behavior; `commands.py` defines menu entries. `dashboard.py`
renders supplied state without polling or mutation. Provider polling stays in
the application timer. `styles.tcss` is included as Python package data.

Runtime scripts must remain usable in their deployment context. In particular,
`serve_model.py` and `tensorfold_runtime.py` are transferred to a GPU host and
cannot depend on an installed copy of the whole harness.

The Python subpackages are internal implementation details, not a stable plugin
API. Existing shell/CLI entrypoints and configuration formats are preserved.
This pass separates responsibilities; model registration is still explicit
Python configuration rather than a third-party plugin system.

<a name="extending-models"></a>

## ![Extending models](../../assets/docs/headings/h2-extending-models.svg)

- Configuration: `Modules.txt` overrides the tracked examples for this installation.
- GPU profiles: register hardware requirements, pinned serving backend, tool parser and tests in `velocicyber/models/gpu_profiles.py` (with engine metadata in `velocicyber/models/catalog.py`); add the repository to `LOCAL_MODEL_OPTIONS`.
- Pi profiles: `velocicyber/models/pi_profiles.py` plus a pinned model/runtime lock file.
- API profiles: `velocicyber/config.py`; current API routing uses the Adverserial confidential gateway. Adding a different API provider requires a routing adapter and tests, not just replacing its URL.
- Sentinel profiles: `velocicyber/sentinel/profiles.py`, independently of the main model selection.

Arbitrary repository names are not automatically supported: validate memory, tokenizer, tool calls and serving dependencies for each new profile.

<a name="engine-updates-and-development"></a>

## ![Engine updates and development](../../assets/docs/headings/h2-engine-updates-and-development.svg)

The execution engine is installed separately from an official pinned package. It is not rebuilt or patched in this repository. `./velocicyber.sh update-engine` asks for approval, stages the candidate, runs compatibility checks and activates it only after those checks pass. Rollback is available with `--rollback`.

`velocicyber/updates/` combines the engine checker with source revision checks and
a source updater. `/update` chooses the available components and offers a persistent,
opt-in source update preference. With `AUTO_UPDATE_HARNESS=ON`, normal UI startup
runs the same staging, validation, exclusive maintenance lock and backup path
before opening the UI; it restarts with the validated source and transfers the
completed check to avoid a second network check. Engine upgrades stay manual.
Failed or blocked updates preserve the working version; interrupted activation
requires rollback. Source rollback disables automatic updates until re-enabled. See [update behavior](../installation.md#updates-and-rollback).

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check bootstrap.py velocicyber scripts tests --no-cache
```

Tests use local fixtures. Native engine integration tests skip if the pinned engine is absent. Scripts named `live_*`, `validate_*`, `smoke_api.py` and `watch_gpu_test.py` are explicit operator tools, not part of installation or the offline test suite; inspect their behavior before running them against paid providers or datasets.

The validation directory contains tools that can call providers, execute live
inference, upload data or update local configuration. They are never invoked by
the offline test suite or the curl bootstrap. `preflight.py` may save a uniquely
matched Lambda SSH key name into `Modules.txt`; it is not a read-only command.

See [the release checklist](release-plan.md), [security boundaries](../../SECURITY.md)
and [third-party notices](../../THIRD_PARTY_NOTICES.md) before release.

<a name="documentation-styling"></a>

## ![Documentation styling](../../assets/docs/headings/h2-documentation-styling.svg)

GitHub strips custom font/color CSS from Markdown. Reader-facing headings use
small, locally stored SVGs with neon-green block lettering on a dark background;
body text uses the viewer's normal font. Heading alt text preserves the labels,
and explicit anchors preserve section links. Agent instructions and the scope
example remain plain text. The existing README branding and red caution alert
are separate from the heading artwork.

After adding or renaming a heading, run:

```bash
python3 scripts/maintenance/style_docs.py
```

The generator draws its own simple lettering, needs no downloaded font or extra
Python dependency, and can be run repeatedly. Review the resulting Markdown and
SVG changes together. Remove unreferenced heading assets when retiring sections.
