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

            app = VelocicyberApp(
                settings,
                demo=args.demo,
                workspace=args.workspace,
                sandbox_open=args.sandbox == "open",
            )
            app.run()
            lease.close()  # The UI and its model sessions have now shut down.
            if app.maintenance_action:
                from velocicyber.engine.updater import interactive_update

                try:
                    interactive_update(settings, action=app.maintenance_action, approved=True)
                except RuntimeError as exc:
                    print(str(exc), file=sys.stderr, flush=True)
                    raise
                finally:
                    try:
                        input("Press Enter to close this window…")
                    except EOFError:
                        pass
    except (ValueError, RuntimeError, PermissionError) as exc:
        parser.exit(1, str(exc) + "\n")
    finally:
        lease.close()


if __name__ == "__main__":
    main()
