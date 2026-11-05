import argparse
import asyncio
import json
import sys
from contextlib import ExitStack
from pathlib import Path

from velocicyber import __version__
from velocicyber.config import Settings
from velocicyber.datasets.pipeline import Curator, Filters, Scrubber
from velocicyber.datasets.upload import Uploader
from velocicyber.engine.installation import engine_lease
from velocicyber.engine.runner import Engine
from velocicyber.sentinel.data import SentinelStore
from velocicyber.state import State


def run_menu_setup(settings):
    """Run hidden terminal prompts after all model clients and UI input have closed."""
    from velocicyber.onboarding import run_setup

    try:
        with engine_lease(settings.runtime, maintenance=True):
            run_setup(settings.root, force=True)
    except (EOFError, KeyboardInterrupt):
        print("\nSetup interrupted. Fields already saved were kept; use /setup to continue.", flush=True)
    except (OSError, ValueError, RuntimeError) as exc:
        print("Setup could not finish: " + str(exc), file=sys.stderr, flush=True)
    try:
        input("Press Enter to return to Velocicyber…")
    except (EOFError, KeyboardInterrupt):
        pass


def main():
    parser = argparse.ArgumentParser(description="Velocicyber — modular coding harness")
    parser.add_argument("--version", action="version", version="Velocicyber " + __version__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workspace", type=Path)
    parser.add_argument(
        "--sandbox",
        choices=("workspace", "open"),
        default="workspace",
        help="Filesystem access: workspace (default), or open using your Linux account permissions",
    )
    parser.add_argument("--demo", action="store_true", help="Offline UI demo; no external requests")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("doctor", help="Check local setup without contacting providers")
    sub.add_parser("setup", help="Configure credentials, datasets and optional local downloads")
    maintenance = sub.add_parser("update", help="Check and manually update the harness, engine, or both")
    maintenance.add_argument("--harness-only", action="store_true", help="Update only Velocicyber source")
    modes = maintenance.add_mutually_exclusive_group()
    modes.add_argument("--stage-only", action="store_true", help="Test available updates without activating")
    modes.add_argument("--rollback-harness", action="store_true", help="Restore the previous harness update")
    update = sub.add_parser(
        "update-engine", help="Manually approve, test and install an engine update"
    )
    mode = update.add_mutually_exclusive_group()
    mode.add_argument(
        "--stage-only", action="store_true", help="Download and test without activation"
    )
    mode.add_argument(
        "--rollback", action="store_true", help="Restore the previous engine and history"
    )
    update.add_argument("--target", help="Optional stable Rust release tag")
    curate = sub.add_parser("curate", help="Rebuild sanitized derived datasets locally")
    curate.add_argument("--allow-compaction", action=argparse.BooleanOptionalAction, default=True)
    run = sub.add_parser("run", help="Run one prompt with Velocicyber and the configured API model")
    run.add_argument("prompt")
    run.add_argument("--collect", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument("--upload", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument("--allow-compaction", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--resume", metavar="SESSION_ID")
    run.add_argument("--task-id")
    run.add_argument("--verifier", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.demo and args.command == "setup":
        parser.error("Setup changes configuration; run it without --demo.")
    settings = Settings.load(args.root)
    from velocicyber.updates.updater import PENDING
    if (settings.runtime / PENDING).exists() and not (
        args.command == "update" and args.rollback_harness
    ):
        parser.exit(1, "Interrupted harness update. Run ./velocicyber.sh update --rollback-harness first.\n")
    startup_checker = None
    if not args.demo and args.command is None:
        from velocicyber.updates.startup import startup_updates

        try:
            startup_checker = startup_updates(settings, lambda message: print(message, flush=True))
        except (OSError, ValueError, RuntimeError) as exc:
            parser.exit(1, str(exc) + "\n")
    if not args.demo and args.command in (None, "setup"):
        from velocicyber.onboarding import run_setup

        try:
            run_setup(args.root, force=args.command == "setup")
        except (EOFError, KeyboardInterrupt):
            parser.exit(1, "Setup interrupted. Run ./velocicyber.sh setup to continue.\n")
        except (ValueError, RuntimeError, OSError) as exc:
            parser.exit(1, "Setup incomplete: " + str(exc) + "\n")
        if args.command == "setup":
            return
    settings = Settings.load(args.root)
    lease = ExitStack()
    try:
        if args.command in (None, "run"):
            lease.enter_context(engine_lease(settings.runtime))
        if args.command == "doctor":
            print("Selected GPU model:", settings.local_model or "none — choose in /local")
            print("GPU model choices:", ", ".join(settings.gpu_model_choices) or "MISSING")
            print("Raspberry Pi CPU model:", settings.pi_model or "not configured")
            print("Main LLM dataset:", settings.dataset_repo or "MISSING")
            print("Sentinel dataset:", settings.options.get("SENTINEL_FINETUNING_REPO") or "MISSING")
            for key in ("ADVERSARIAL_API", "LAMBDA_API", "HUGGINGFACE_API"):
                print(key + ":", "configured" if settings.secrets.get(key) else "MISSING")
            try:
                engine_path = settings.engine()
            except RuntimeError:
                engine_path = "not installed; run ./velocicyber.sh setup"
            print("Velocicyber execution engine:", engine_path)
            for key in ("LAMBDA_REGION", "LAMBDA_SSH_KEY_NAME", "SSH_PRIVATE_KEY"):
                print(
                    key + ":",
                    "configured" if settings.options.get(key) else "needed before GPU launch",
                )
            print("No provider requests made.")
        elif args.command == "update":
            from velocicyber.updates.manager import interactive_updates

            if args.demo:
                raise ValueError("Updates are disabled in demo mode.")
            interactive_updates(settings, harness_only=args.harness_only,
                                stage_only=args.stage_only, rollback_harness=args.rollback_harness)
        elif args.command == "update-engine":
            from velocicyber.engine.updater import interactive_update

            if args.demo:
                raise ValueError("Engine maintenance is disabled in demo mode.")
            interactive_update(
                settings,
                action="rollback" if args.rollback else "stage" if args.stage_only else "update",
                tag=args.target,
            )
        elif args.command == "curate":
            print(
                Curator(
                    settings.data,
                    Scrubber(
                        settings.secrets.values(),
                        settings.options.get("PRIVATE_HOSTS", "").split(","),
                    ),
                    Filters(allow_compaction=args.allow_compaction),
                ).run()
            )
        elif args.command == "run":
            state = State(
                collection=args.collect,
                upload_approved=args.upload,
                allow_compaction=args.allow_compaction,
                sandbox_open=args.sandbox == "open",
            )
            engine = Engine(settings, state, args.workspace)
            try:
                if args.resume:
                    engine.resume_session(args.resume)
                if args.task_id:
                    engine.task_id = args.task_id
                if args.verifier is not None:
                    engine.verifier = args.verifier
                asyncio.run(
                    engine.run(
                        args.prompt,
                        lambda e: print(engine.scrubber.text(json.dumps(e)), flush=True),
                    )
                )
                if state.collection and state.upload_approved:
                    artifact = Curator(
                        settings.data,
                        engine.scrubber,
                        Filters(allow_compaction=state.allow_compaction),
                    ).run()
                    receipt = Uploader(settings).upload(artifact, approved=True)
                    print(
                        json.dumps({"type": "dataset.uploaded", "receipt": receipt})
                        if receipt
                        else "No eligible data to upload."
                    )
                    sentinel = SentinelStore(settings).upload(
                        approved=True, allow_compaction=state.allow_compaction
                    )
                    if sentinel:
                        print(json.dumps({"type": "sentinel.uploaded", "receipt": sentinel}))
            finally:
                engine.close()
        else:
            from velocicyber.ui.app import VelocicyberApp

            previous_engine_check = None
            while True:
                app = VelocicyberApp(
                    settings,
                    demo=args.demo,
                    workspace=args.workspace,
                    sandbox_open=args.sandbox == "open",
                )
                if startup_checker is not None:
                    startup_checker.settings = settings
                    app.updates.source = startup_checker
                    app.startup_update_message = startup_checker.startup_notice
                if previous_engine_check is not None and app.updates.engine.identity == previous_engine_check.identity:
                    previous_engine_check.settings = settings
                    app.updates.engine = previous_engine_check
                app.run()
                lease.close()  # The UI and its model sessions have now shut down.
                if app.maintenance_action == "setup":
                    run_menu_setup(settings)
                    settings = Settings.load(args.root)
                    startup_checker = app.updates.source
                    startup_checker.startup_notice = ""
                    previous_engine_check = app.updates.engine
                    lease.enter_context(engine_lease(settings.runtime))
                    continue
                if app.maintenance_action:
                    from velocicyber.engine.updater import interactive_update

                    try:
                        if app.maintenance_action == "components":
                            from velocicyber.updates.manager import interactive_updates
                            interactive_updates(settings, plan=app.maintenance_plan, approved=True)
                        elif app.maintenance_action == "harness-rollback":
                            from velocicyber.updates.manager import interactive_updates
                            interactive_updates(settings, rollback_harness=True, approved=True)
                        else:
                            interactive_update(settings, action=app.maintenance_action, approved=True)
                    except RuntimeError as exc:
                        print(str(exc), file=sys.stderr, flush=True)
                        raise
                    finally:
                        try:
                            input("Press Enter to close this window…")
                        except EOFError:
                            pass
                break
    except (ValueError, RuntimeError, PermissionError) as exc:
        parser.exit(1, str(exc) + "\n")
    finally:
        lease.close()


if __name__ == "__main__":
    main()
