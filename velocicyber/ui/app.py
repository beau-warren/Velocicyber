"""Terminal controls, with one status refresh when restoring a saved GPU."""

import asyncio
import shlex
import time
from collections import deque
from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Footer, Input, Label, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from velocicyber.config import API_MODELS, PI_MODEL_NOTICE
from velocicyber.datasets.pipeline import dataset_summary, delete_local_finetuning_data
from velocicyber.datasets.upload import Uploader
from velocicyber.engine.installation import take_update_notice
from velocicyber.engine.runner import Engine
from velocicyber.models.gpu_profiles import MOG_CONTEXT_PROFILES, MOG_SEC_REPO, gpu_profile
from velocicyber.models.startup import model_details, model_option_label
from velocicyber.providers import temperature
from velocicyber.providers.lambda_cloud import MEMORY_CHOICES, SINGLE_MEMORY_CHOICES, GPUManager
from velocicyber.providers.pi import PiModelManager
from velocicyber.providers.pricing import memory_price_options
from velocicyber.sentinel.data import SentinelStore
from velocicyber.sentinel.observer import load_scope
from velocicyber.sentinel.profiles import SENTINEL_MODELS, local_sentinel_active, sentinel_profile
from velocicyber.sentinel.runtime import SentinelRuntime
from velocicyber.sentinel.scope import read_scope
from velocicyber.state import GPUStatus, State
from velocicyber.ui.commands import COMMANDS, CONTROLS
from velocicyber.ui.dashboard import dashboard_text
from velocicyber.ui.presentation import (
    CODE_GRAY,
    NEON_BLUE,
    NEON_GREEN,
    NEON_PINK,
    compact_title_art,
    status_guide,
    title_art,
    transcript_text,
)
from velocicyber.ui.scope_editor import ScopeEditor
from velocicyber.ui.transcript import TranscriptLog
from velocicyber.ui.widgets import Menu, Prompt
from velocicyber.updates import UpdateChecker
from velocicyber.updates.manager import describe


class VelocicyberApp(App):
    TITLE = "Velocicyber"
    CSS_PATH = "styles.tcss"
    BINDINGS = [
        ("f2", "menu", "Menu"),
        ("ctrl+n", "new_task", "New task"),
        Binding("ctrl+x", "cancel_task", "Stop task", priority=True),
        ("f3", "conversation", "Conversation"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, settings, *, demo=False, workspace=None, sandbox_open=False):
        super().__init__()
        self.settings = settings
        self.state = State.restore(settings.runtime / "gpu.json")
        self.state.sandbox_open = sandbox_open
        self.engine = Engine(settings, self.state, workspace)
        self.gpu = GPUManager(settings, self.state)
        self.pi = PiModelManager(settings, self.state)
        self.sentinel = SentinelRuntime(settings, self.state)
        self.sentinel_store = SentinelStore(settings)
        self.sentinel_busy = False
        self.uploader = Uploader(settings)
        self.updates = UpdateChecker(settings)
        self.update_check_running = False
        self.startup_update_message = ""
        self.maintenance_action = ""
        self.maintenance_plan = {}
        self.compacting = False
        self.demo, self.frame, self.busy = demo, 0, False
        self.gpu_busy = False
        self.pi_busy = False
        self.data_busy = False
        self.exit_after_termination = False
        self.launch_task = None
        self.termination_poll_seconds = 30
        self.restoring_gpu = False
        self.conversation = False
        self.activity = "Ready for a task"
        self.task_started = 0.0
        self.task_failed = False
        self.completion_notice = ""
        self.stop_requested = False
        self.queued_prompts = deque()
        self.queue_paused = False
        self.session_ending = False
        self.command_items = {}
        self.live_text = ""
        self.shown_live_text = ""
        self.completed_responses = set()
        self.reply_model = ""
        self.reply_provider = ""
        self.pi_install_notified = False
        self.pi_status_checked = 0.0
        self.pi_temperature_c = None
        self.pi_temperature_checked = None
        self.header_width = None

    def compose(self) -> ComposeResult:
        with Vertical(id="masthead"):
            yield Static(title_art(), id="title", markup=False)
            with Vertical(id="status-panel"):
                yield Static(id="update-status", markup=False)
                yield Static(id="dashboard", markup=False)
        with Vertical(id="content"):
            yield TranscriptLog(id="log", min_width=1, wrap=True, markup=False, highlight=False)
            with VerticalScroll(id="main-menu"):
                with Vertical(id="home-copy"):
                    yield Static(status_guide(), id="status-guide", markup=False)
                    yield Static(
                        Text.assemble(
                            ("/new", f"bold {NEON_PINK}"),
                            " to start a new session\n",
                            ("/resume", f"bold {NEON_GREEN}"),
                            " to pick up where you left off.",
                        ),
                        id="home-actions",
                        markup=False,
                    )
        yield TranscriptLog(
            id="live-response", min_width=1, wrap=True, markup=False, highlight=False
        )
        yield Static(id="activity", markup=False)
        yield Static(id="prompt-queue", markup=False)
        yield Static(id="home-notice", markup=False)
        yield OptionList(id="commands", markup=False, compact=True)
        yield Prompt(placeholder="Type /new or /resume, or / to browse commands", id="prompt")
        yield Footer()
        yield Static(id="home-update-status", markup=False)

    def on_mount(self):
        self.animate()
        self.set_interval(0.2, self.animate)
        self.query_one(
            "#commands", OptionList
        ).border_title = "Commands · ↑↓ choose · Enter open · Esc close"
        self.query_one("#prompt", Input).focus()
        self.query_one("#live-response", RichLog).border_title = "Velocicyber · live response"
        if self.demo:
            self.query_one("#prompt", Input).placeholder = "Type /new or /resume · offline demo"
        elif self.state.instance_id or self.state.launch_uncertain:
            self.restoring_gpu = True
            self.say("Checking the saved GPU's current status once.")
            self.gpu_operation("refresh")
        if not self.demo:
            if self.startup_update_message:
                self.say(self.startup_update_message)
            if tag := take_update_notice(self.settings.runtime):
                self.notify(f"Velocicyber engine updated to {tag}.", title="Update complete")
            self.refresh_updates()

    @work(group="updates", exit_on_error=False)
    async def refresh_updates(self):
        if self.update_check_running:
            return
        self.update_check_running = True
        try:
            await self.updates.check()
            self.animate()
        finally:
            self.update_check_running = False

    def on_resize(self, event):
        self.update_header_layout(event.size.width - 4)
        self.call_after_refresh(self.update_header_layout)

    def update_header_layout(self, width=None):
        if not self.query("#status-panel"):
            return
        if width is None:
            width = self.size.width - 4  # Match the masthead's two-column padding.
        layout = (width, self.size.height, self.conversation)
        if not width or layout == self.header_width:
            return
        self.header_width = layout
        art = title_art()
        title_width = max(map(len, art.plain.splitlines()))
        # Home only needs room for the title plus the existing outer padding.
        # Sessions also reserve a gap and enough width for their status column.
        compact = width < title_width + (2 + 52 if self.conversation else 0)
        panel_width = min(76, max(1, width - (18 if compact else title_width + 2)))
        self.query_one("#status-panel").styles.width = panel_width
        title = self.query_one("#title", Static)
        banner = compact_title_art(padded=self.conversation) if compact else art
        masthead = self.query_one("#masthead")
        if self.conversation:
            title.styles.width = "1fr"
            masthead.styles.padding = (0, 2, 0, 2)
        else:
            banner_width = max(line.cell_len for line in banner.split("\n"))
            if compact:
                # Center the wordmark over the icon without disturbing its pixel geometry.
                lines = banner.split("\n")
                lines[0].pad_left((banner_width - lines[0].cell_len) // 2)
                banner = Text("\n").join(lines)
            title.styles.width = min(width, banner_width)
            guide = status_guide()
            copy_width = max(1, min(width, max(line.cell_len for line in guide.split("\n"))))
            self.query_one("#home-copy").styles.width = copy_width
            actions = Text(self.query_one("#home-actions", Static).render().plain)
            group_height = (
                len(banner.plain.splitlines())
                + len(guide.wrap(self.console, copy_width))
                + len(actions.wrap(self.console, copy_width))
                + 2  # Space above the guide and the command instructions.
            )
            # Center in the area above the command input and footer; short windows scroll.
            top = max(1, (self.size.height - 5 - group_height) // 2)
            masthead.styles.padding = (top, 2, 0, 2)
        title.styles.height = len(banner.plain.splitlines())
        title.update(banner)

    def update_release_indicator(self):
        label, color = ("UPDATES: DEMO", CODE_GRAY) if self.demo else self.updates.indicator()
        home_badge = self.query_one("#home-update-status", Static)
        for badge in (self.query_one("#update-status", Static), home_badge):
            badge.update(Text(label, style=f"bold {color}", justify="right", no_wrap=True))
            badge.tooltip = "Open /updates-info for harness and engine versions; /update to install."
        home_badge.styles.width = min(len(label), max(1, self.size.width - 4))
        home_badge.styles.offset = (max(2, self.size.width - len(label) - 2), 0)

    def setup_menu(self):
        if self.demo:
            self.say("DEMO: setup is disabled. Open the normal harness to configure it.")
            return
        if self.busy or self.gpu_busy or self.pi_busy or self.data_busy or self.sentinel_busy or self.session_ending:
            self.say("Finish the current task or resource operation before opening setup.")
            return
        self.push_screen(Menu(
            "Secure setup — credentials, datasets, SSH, desktop launcher and optional installs.\n"
            "This ends the session, stops local models and terminates its GPU.\n"
            "Saved conversations remain available through /resume.\n"
            "Credentials use hidden terminal prompts outside the chat. Open setup?",
            [(False, "Cancel — keep coding"), (True, "Open secure setup")], id="setup-confirm"),
            lambda approved: self.begin_setup() if approved else None)

    def begin_setup(self):
        if self.demo or self.busy or self.gpu_busy or self.pi_busy or self.data_busy or self.sentinel_busy or self.session_ending:
            self.say("Finish the current task or resource operation before opening setup.")
            return
        self.maintenance_action = "setup"
        self.action_quit()

    def update_menu(self):
        if self.demo:
            self.say("DEMO: updates are disabled.")
            return
        if self.busy or self.gpu_busy or self.pi_busy or self.data_busy or self.sentinel_busy:
            self.say("Finish the current task or resource operation before updating.")
            return
        if self.update_check_running:
            self.say("The launch-time update check is still running. Open /update when it finishes.")
            return
        plan = self.updates.plan()
        choices = [(None, "Cancel")]
        if plan:
            choices.append(("components", "Update " + describe(plan) + " — test, back up, then activate"))
        choices.extend([("engine-tools", "Engine test / rollback options"),
                        ("harness-rollback", "Roll back the last Velocicyber update")])
        from velocicyber.updates.startup import automatic_updates
        choices.append(("automatic-source", "Automatic repo updates on launch: "
                        + ("ON" if automatic_updates(self.settings) else "OFF")))
        summary = "\n".join(component + ": " + item["target"][:18] for component, item in plan.items())
        self.push_screen(Menu("Updates" + ("\n" + summary if plan else " — no verified updates available; see /updates-info"), choices),
                         lambda action: self.confirm_updates(action, plan))

    def confirm_updates(self, action, plan):
        if action == "automatic-source":
            from velocicyber.updates.startup import automatic_updates
            if automatic_updates(self.settings):
                self.set_automatic_source_updates(False)
            else:
                self.push_screen(Menu(
                    "Automatically test and install official repo updates on future launches?\n"
                    "Active sessions are left alone. Engine upgrades still require approval.",
                    [(False, "Cancel"), (True, "Enable automatic repo updates on launch")]),
                    lambda approved: self.set_automatic_source_updates(True) if approved else None)
            return
        if action == "engine-tools":
            self.engine_update_menu()
            return
        if action not in {"components", "harness-rollback"}:
            return
        description = ("Update " + describe(plan) + ". Test before activation; retain rollback backups."
                       if action == "components" else "Restore the previous Velocicyber source and Python environment.")
        self.push_screen(Menu(description + "\nThis closes the harness, stops the Pi model and terminates its GPU.\nProceed?",
                              [(False, "Cancel"), (True, "Yes — proceed")]),
                         lambda approved: self.begin_updates(action, plan) if approved else None)

    def set_automatic_source_updates(self, enabled):
        from velocicyber.updates.startup import set_automatic_updates
        try:
            set_automatic_updates(self.settings, enabled)
        except (OSError, ValueError) as exc:
            self.say("Could not save the automatic update preference: " + str(exc))
            return
        self.say("Automatic repo updates on launch: " + ("ON" if enabled else "OFF")
                 + ". Engine upgrades remain manual.")

    def begin_updates(self, action, plan):
        if self.demo or self.busy or self.gpu_busy or self.pi_busy or self.data_busy or self.sentinel_busy:
            self.say("Finish the current operation before updating.")
            return
        self.maintenance_plan = plan
        self.maintenance_action = action
        self.action_quit()

    def engine_update_menu(self):
        if self.demo:
            self.say("DEMO: engine maintenance is disabled.")
            return
        if self.busy or self.gpu_busy or self.pi_busy or self.data_busy or self.sentinel_busy:
            self.say("Finish the current task or resource operation before updating the engine.")
            return
        self.push_screen(
            Menu(
                "Engine maintenance — choose an action",
                [
                    (None, "Cancel"),
                    ("update", "Update engine — verify, test, back up, then activate"),
                    ("stage", "Download and test only — keep the current engine"),
                    ("rollback", "Roll back engine and restore pre-update conversations"),
                ],
            ),
            self.confirm_engine_maintenance,
        )

    def confirm_engine_maintenance(self, action):
        if action not in {"update", "stage", "rollback"}:
            return
        descriptions = {
            "update": "Download and test the new engine. Activate only if tests pass; back up conversations first.",
            "stage": "Download and test the new engine without activating it.",
            "rollback": "Restore the previous engine and pre-update history. Preserve newer history in a separate archive.",
        }
        self.push_screen(
            Menu(
                descriptions[action]
                + "\nThis closes the harness, stops the Pi model and terminates its GPU.\nProceed?",
                [(False, "Cancel — keep coding"), (True, "Yes — proceed")],
            ),
            lambda approved: self.begin_engine_maintenance(action) if approved else None,
        )

    def begin_engine_maintenance(self, action):
        if self.demo or self.busy or self.gpu_busy or self.pi_busy or self.data_busy:
            self.say("Engine maintenance is unavailable while another operation is running.")
            return
        self.maintenance_action = action
        self.action_quit()

    def say(self, message, *, kind="text"):
        message = self.engine.scrubber.text(str(message))
        self.query_one("#log", RichLog).write(transcript_text(message, kind))
        if not self.conversation:
            notice = self.query_one("#home-notice", Static)
            notice.update(message)
            notice.add_class("has-message")

    def animate(self):
        if not self.query("#dashboard") or not self.query("#home-update-status"):
            return  # A timer can already be queued while the screen is unmounting.
        self.frame += 1
        self.update_header_layout()
        self.update_release_indicator()
        if self.settings.pi_model and time.monotonic() - self.pi_status_checked >= 5:
            self.pi_status_checked = time.monotonic()
            status = self.pi.install_status()
            if status.startswith("Downloaded") and not self.pi_install_notified:
                self.pi_install_notified = True
                self.say(
                    "Raspberry Pi model installation complete. Select it in Local Model Selection to load it."
                )
            if not self.pi_busy and not self.state.pi_ready:
                self.state.pi_stage = status
        if self.state.private_target == "pi":
            now = time.monotonic()
            if (
                self.pi_temperature_checked is None
                or now - self.pi_temperature_checked >= temperature.REFRESH_SECONDS
            ):
                self.pi_temperature_c = temperature.cpu_temperature_c()
                self.pi_temperature_checked = now
        display = dashboard_text(
            self.settings, self.state, restoring_gpu=self.restoring_gpu,
            sentinel_busy=self.sentinel_busy, pi_temperature_c=self.pi_temperature_c,
        )
        dashboard = self.query_one("#dashboard", Static)
        dashboard.update(display)
        dashboard.tooltip = display.plain  # Full names remain accessible when a row is shortened.
        if self.state.private_target == "pi":
            dashboard.tooltip += (
                "\nCPU temperature: red at 167°F; thermal throttling starts around 176°F."
            )
        self.update_chat_model()
        self.update_activity()
        if isinstance(self.screen, Menu) and self.screen.id == "local-models":
            self.update_local_menu()

    def action_conversation(self):
        self.show_conversation()

    def update_chat_model(self):
        log = self.query_one("#log", RichLog)
        if not self.conversation:
            log.border_title = ""
            return
        provider = (
            self.reply_provider if self.busy and self.reply_provider else self.state.active_provider
        )
        model = (
            self.reply_model if self.busy and self.reply_model else self.chat_model_name(provider)
        )
        title = Text("Chat model: " + model, style=f"bold {NEON_GREEN}")
        title.append(
            " · Pi CPU" if provider == "pi" else " · GPU" if provider == "local" else " · API"
        )
        if provider == "local" and not self.state.model_ready:
            title.append(" (not ready)", style="yellow")
        if provider == "pi" and not self.state.pi_ready:
            title.append(" (not ready)", style="yellow")
        log.border_title = title

    def chat_model_name(self, provider=None):
        provider = provider or self.state.active_provider
        return (
            self.settings.pi_model
            if provider == "pi"
            else self.settings.local_model
            if provider == "local"
            else self.settings.api_model_name
        )

    def show_conversation(self):
        self.conversation = True
        if isinstance(self.screen, Menu):
            self.screen.dismiss(None)
        self.screen_stack[0].set_class(True, "conversation")
        self.query_one("#commands", OptionList).display = False
        prompt = self.query_one("#prompt", Input)
        prompt.placeholder = "Give the agent a task, or type \\ to find a command"
        prompt.focus()
        self.animate()

    def show_home(self):
        self.conversation = False
        self.screen_stack[0].set_class(False, "conversation")
        self.query_one("#main-menu", VerticalScroll).scroll_home(animate=False)
        self.query_one("#commands", OptionList).display = False
        prompt = self.query_one("#prompt", Input)
        prompt.placeholder = "Type /new or /resume, or / to browse commands"
        prompt.focus()
        self.animate()

    def update_activity(self):
        self.update_prompt_queue()
        label = self.activity
        timing = ""
        if self.busy:
            elapsed = int(time.monotonic() - self.task_started)
            spinner = "|/-\\"[self.frame % 4]
            label = f"{spinner} {label}"
            timing = f" · {elapsed // 60}:{elapsed % 60:02d}"
        stage = self.state.sentinel_stage
        sentinel = "Sentinel " + (
            "off" if not self.state.sentinel_enabled and not self.sentinel_busy else
            {"REVIEWING": "working", "OBSERVING": "waiting", "READY": "idle",
             "STOPPED TASK": "stopped", "ERROR": "error"}.get(stage, stage.lower())
        )
        activity = self.query_one("#activity", Static)
        width = activity.content_size.width or max(1, self.size.width - 4)
        if self.busy and width >= 72:
            timing += " · Ctrl+X stops"
        text = Text(self.engine.scrubber.text(label), style=NEON_BLUE)
        text.truncate(max(1, width - len(sentinel) - len(timing) - 3), overflow="ellipsis")
        text.append(" · " + sentinel, style=f"bold {NEON_PINK}")
        text.append(timing, style=NEON_BLUE)
        activity.update(text)
        if self.live_text != self.shown_live_text:
            preview = self.query_one("#live-response", RichLog)
            if self.live_text and preview.text_selection is not None:
                return  # Keep a streaming selection stable until the user clears it.
            visibility_changed = preview.display != bool(self.live_text)
            preview.clear()
            preview.display = bool(self.live_text)
            if self.live_text:
                preview.write(
                    transcript_text(self.engine.scrubber.text(self.live_text)), scroll_end=True
                )
            self.shown_live_text = self.live_text
            if visibility_changed:
                self.call_after_refresh(self.query_one("#log", RichLog).scroll_end, animate=False)

    def update_prompt_queue(self):
        widget = self.query_one("#prompt-queue", Static)
        widget.display = bool(self.queued_prompts) and self.conversation
        if not self.queued_prompts:
            return
        status = "paused" if self.queue_paused else "after current task"
        text = Text(
            f"Queued: {len(self.queued_prompts)} · {status} · /queue",
            style=f"bold {NEON_PINK}",
        )
        preview = Text(
            "Next: " + " ".join(self.engine.scrubber.text(self.queued_prompts[0]).split()),
            style=CODE_GRAY,
        )
        preview.truncate(max(1, self.size.width - 4), overflow="ellipsis")
        text.append("\n")
        text.append(preview)
        widget.update(text)

    def queue_prompt(self, prompt):
        self.queued_prompts.append(prompt)
        if self.stop_requested:
            self.queue_paused = True
        self.update_prompt_queue()
        self.say(f"Prompt queued · {len(self.queued_prompts)} waiting. Use /queue to manage.")

    def clear_prompt_queue(self, reason=""):
        count = len(self.queued_prompts)
        self.queued_prompts.clear()
        self.queue_paused = False
        self.update_prompt_queue()
        if count:
            self.say(f"Cleared {count} queued prompt(s)" + (f" — {reason}." if reason else "."))

    def open_prompt_queue(self):
        details = "\n\n".join(
            f"{i}. {self.engine.scrubber.text(prompt)}"
            for i, prompt in enumerate(self.queued_prompts, 1)
        )
        self.push_screen(
            Menu(
                f"Queued prompts: {len(self.queued_prompts)}"
                + (" · paused" if self.queue_paused else "")
                + "\nRun in order in this conversation.",
                [
                    ("resume", "Resume queued prompts"),
                    ("pause", "Pause queue after the current task"),
                    ("clear", "Clear all queued prompts"),
                ],
                id="prompt-queue-menu",
                details=details or "No prompts queued.",
            ),
            self.queue_selected,
        )

    def queue_selected(self, action):
        if not action or self.session_ending:
            return
        if action == "clear":
            self.clear_prompt_queue()
        elif action == "pause":
            self.queue_paused = True
            self.update_prompt_queue()
        elif action == "resume":
            if not self.queued_prompts:
                self.say("No prompts queued.")
                return
            if self.busy and (self.stop_requested or self.task_failed):
                self.say("Wait for the current task to stop before resuming the queue.")
                return
            if not self.busy and (blocker := self.task_start_blocker()):
                self.say(blocker)
                return
            self.queue_paused = False
            if not self.busy:
                self.run_prompt(self.queued_prompts.popleft())
            self.update_prompt_queue()

    def task_start_blocker(self):
        if self.session_ending:
            return "The session is ending; no new tasks will be started."
        if self.sentinel_busy:
            return "Wait for Sentinel startup or shutdown before starting a task."
        if self.data_busy:
            return "Wait for local data deletion to finish before starting a task."
        if self.gpu_busy or (self.pi_busy and self.state.active_provider == "pi"):
            return "Wait for model startup to finish before starting a task."
        return ""

    def action_menu(self):
        if self.busy:
            self.say("Wait for the current task or stop it before changing session settings.")
            return
        if isinstance(self.screen, Menu):
            self.screen.dismiss(None)
        prompt = self.query_one("#prompt", Input)
        prompt.value = "\\"
        prompt.cursor_position = 1
        prompt.focus()
        self.filter_commands(Input.Changed(prompt, prompt.value))

    def menu_selected(self, value):
        if value is None:
            return
        if self.busy:
            self.say("Wait for the current task or stop it before changing session settings.")
            return
        if value == "gpu":
            self.push_screen(
                Menu(
                    "GPU / GPU Cluster",
                    [
                        ("on", "ON — choose total GPU memory (VRAM)"),
                        ("terminate", "OFF — terminate instance"),
                        ("refresh", "Refresh instance status"),
                        ("connect", "Reconnect inference tunnel"),
                    ],
                ),
                self.gpu_selected,
            )
        elif value == "local":
            self.push_screen(
                Menu(
                    self.local_menu_heading(),
                    [(repo, repo) for repo in self.settings.model_choices]
                    + [("gpu-controls", "Open GPU / GPU Cluster controls")]
                    + (
                        [("pi-stop", "Stop Raspberry Pi CPU model")]
                        if self.settings.pi_model
                        else []
                    ),
                    id="local-models",
                    groups=(
                        {
                            i: "Raspberry Pi CPU — no GPUs"
                            if repo in self.settings.pi_model_choices
                            else "Lambda GPU"
                            for i, repo in enumerate(self.settings.model_choices)
                        }
                        if self.settings.pi_model
                        else None
                    ),
                ),
                self.review_local_model,
            )
            self.call_after_refresh(self.update_local_menu)
        elif value == "api":
            self.push_screen(
                Menu(
                    "API Model",
                    [
                        (key, profile[0] + " — Adverserial.ai")
                        for key, profile in API_MODELS.items()
                    ],
                ),
                self.select_api_model,
            )
        elif value == "sentinel":
            self.push_screen(
                Menu(
                    "Sentinel — " + sentinel_profile(self.state).label + "\n"
                    "Scope: SCOPE/SCOPE.MD · Reviews tool actions.\n"
                    "ON observes asynchronously and stops possible scope violations.\n"
                    "Actions may finish before review; stopping cannot undo them.\n"
                    "Sentinel uses Pi RAM; a running Pi coding model can be swapped after confirmation.\n"
                    "Collection and upload permissions also apply to sentinel records.",
                    [(True, "/ON"), (False, "/OFF"), ("models", "Select model"),
                     ("scope", "/SCOPE — Edit scope restrictions")],
                    id="sentinel-menu",
                ),
                self.set_sentinel,
            )
        elif value == "data-collection":
            self.push_screen(
                Menu(
                    "Data Collection",
                    [
                        (True, "ON — collect fine-tuning data"),
                        (False, "OFF — stop collecting"),
                        ("delete", "Delete all fine-tuning data"),
                        ("upload-manual", "Manual Upload"),
                        ("allow-compaction", "Allow compaction?"),
                        ("advanced", "Advanced"),
                    ],
                ),
                self.data_selected,
            )
        elif value == "upload-perms":
            self.push_screen(
                Menu(
                    "Automatic uploads of eligible, sanitized datasets to\n"
                    + self.settings.dataset_repo
                    + "\nCurrently "
                    + ("ON" if self.state.upload_approved else "OFF")
                    + ". Raw logs stay local.",
                    [(True, "ON — automatic uploads"), (False, "OFF — keep data local")],
                ),
                self.set_upload,
            )
        elif value == "terminate":
            self.confirm_termination()
        elif value == "resume":
            self.open_resume_menu()
        elif value == "sandbox":
            self.push_screen(
                Menu(
                    "Filesystem access · currently "
                    + ("OPEN" if self.state.sandbox_open else "WORKSPACE")
                    + "\nWorkspace isolates tools from personal files and credentials; copy needed "
                    "files into the workspace. Open allows file changes and commands across this machine with your "
                    "Linux account's permissions, without per-command approval. No added root access. "
                    "Collection/upload settings still apply. Resets on restart.",
                    [(False, "Workspace — isolated by default"), (True, "Open — access this machine")],
                ),
                self.set_sandbox,
            )

    def set_sandbox(self, enabled):
        if enabled is None:
            return
        if self.busy:
            self.say("Wait for the current task or stop it before changing filesystem access.")
            return
        try:
            reset = self.engine.set_sandbox_open(enabled)
        except (RuntimeError, ValueError, OSError) as exc:
            self.say(str(exc))
            return
        if reset:
            self.query_one("#log", RichLog).clear()
            self.say(
                "Started a new conversation in "
                + str(self.engine.workspace)
                + ". The external conversation remains saved under /resume. Verifier cleared."
            )
        self.say(
            "Filesystem access: OPEN — the next task can work outside the workspace."
            if enabled
            else "Filesystem access: WORKSPACE — the next task uses an isolated workspace."
        )
        self.animate()

    @work(group="resume", exclusive=True)
    async def open_resume_menu(self):
        try:
            await asyncio.to_thread(self.engine.sessions.import_legacy)
            sessions = await asyncio.to_thread(self.engine.sessions.list)
            if not sessions:
                self.say(
                    "No saved conversations yet. Conversations are saved as you use the agent."
                )
                return
            self.push_screen(
                Menu(
                    "Resume conversation · type to filter by title, date or model",
                    [
                        (
                            row["id"],
                            f"{row['updated_at'][:16].replace('T', ' ')} UTC · {row['title']} · {row['model']}",
                        )
                        for row in sessions
                    ],
                    id="saved-sessions",
                ),
                self.resume_selected,
            )
        except Exception as exc:
            self.say("Could not list saved conversations: " + str(exc))

    def resume_selected(self, session_id):
        if not session_id:
            return
        if self.busy or self.gpu_busy:
            self.say("Wait for the current task or GPU operation before resuming a conversation.")
            return
        try:
            saved = self.engine.resume_session(session_id)
            self.query_one("#log", RichLog).clear()
            self.clear_prompt_queue("resuming a conversation")
            self.command_items = {}
            self.live_text = ""
            # Restored output belongs in the conversation, not main-menu notifications.
            self.show_conversation()
            for event in saved.events():
                if event.get("type") == "user.prompt":
                    self.say("\nYou: " + event["text"])
                elif event.get("type") != "dataset.prepared":
                    self.engine_event(event)
            self.task_failed = self.stop_requested = False
            self.activity = "Conversation resumed — enter a follow-up"
            self.animate()
            model = self.chat_model_name()
            self.say("Resumed: " + saved.metadata["title"] + ". Next reply uses " + model + ".")
        except Exception as exc:
            self.say("Could not resume conversation: " + str(exc))

    def local_menu_heading(self):
        if self.restoring_gpu:
            status = "Checking saved GPU status…"
        elif self.gpu_busy:
            status = "Wait for the current GPU operation to finish."
        elif self.state.gpu_status == GPUStatus.RUNNING:
            status = "GPU is running. Select a model to use."
        elif self.can_select_local_model():
            status = "Select a model for the next GPU launch. Turn GPUs ON to use it."
        else:
            status = (
                "Wait for GPU startup/shutdown, or refresh its status before selecting a model."
            )
        return "Local Model Selection\n" + status

    def can_select_local_model(self):
        return not (self.gpu_busy or self.restoring_gpu) and (
            self.state.gpu_status == GPUStatus.RUNNING
            or (
                self.state.gpu_status == GPUStatus.STOPPED
                and not self.state.instance_id
                and not self.state.launch_uncertain
            )
        )

    def update_local_menu(self):
        if not isinstance(self.screen, Menu) or self.screen.id != "local-models":
            return
        # A timer can run between pushing the menu and mounting its children.
        if not self.screen.is_mounted or not self.screen.query(Label):
            return
        self.screen.query_one(Label).update(self.local_menu_heading())
        disabled = not self.can_select_local_model()
        self.screen.set_disabled(
            {
                i
                for i, repo in enumerate(self.settings.model_choices)
                if self.pi_busy or (disabled and repo not in self.settings.pi_model_choices)
            }
        )

    def gpu_model_ready(self, repo, mog_context=None):
        if not (self.state.model_ready and repo == self.settings.local_model):
            return False
        if repo == MOG_SEC_REPO:
            profile = gpu_profile(repo, mog_context) if mog_context else self.settings.gpu_profile(repo)
            return (self.state.model_context_model == repo
                    and self.state.model_context_window == profile.context_window)
        return True

    def review_local_model(self, repo, *, mog_context=None):
        if repo in {"gpu-controls", "pi-stop"}:
            self.select_local_model(repo)
        elif repo == MOG_SEC_REPO and repo in self.settings.model_choices and mog_context is None:
            self.push_screen(
                Menu(
                    repo + "\nChoose the context window",
                    [("64k", "64K context — lighter · 80 GB H100"),
                     ("1m", "1M context (1 million) — experimental · 180 GB B200")],
                    id="mog-context",
                ),
                lambda variant: self.review_local_model(repo, mog_context=variant)
                if variant in MOG_CONTEXT_PROFILES else None,
            )
        elif repo in self.settings.model_choices:
            self.push_screen(
                Menu(
                    repo
                    + (f"\nContext: {mog_context.upper()}" if mog_context else "")
                    + ("\n" + PI_MODEL_NOTICE if repo in self.settings.pi_model_choices else ""),
                    [
                        ("confirm", "Confirm selected model choice"),
                        ("details", "More Details"),
                    ],
                    id="local-model-choice",
                ),
                lambda action: self.local_model_choice(repo, action, mog_context=mog_context),
            )

    def local_model_choice(self, repo, action, *, mog_context=None):
        if action == "confirm":
            self.select_local_model(repo, mog_context=mog_context)
        elif action == "back":
            self.review_local_model(repo, mog_context=mog_context)
        elif action == "details":
            ready = (
                self.state.pi_ready and repo == self.settings.pi_model
                if repo in self.settings.pi_model_choices
                else self.gpu_model_ready(repo, mog_context)
            )
            self.push_screen(
                Menu(
                    "Model Details\n" + repo + "\nPgUp / PgDn or mouse wheel to scroll",
                    [("back", "Back"), ("confirm", "Confirm selected model choice")],
                    id="local-model-details",
                    details=model_details(repo, ready=ready, variant=mog_context or ""),
                ),
                lambda choice: self.local_model_choice(repo, choice, mog_context=mog_context),
            )

    def select_local_model(self, repo, *, download_approved=False, mog_context=None):
        if self.busy or self.pi_busy or self.sentinel_busy:
            self.say("Wait for the current task or model operation to finish.")
            return
        if repo == "pi-stop":
            self.pi_operation("stop")
        elif repo and repo in self.settings.pi_model_choices:
            if local_sentinel_active(self.state) or self.sentinel_busy:
                self.say("Turn /sentinel OFF before loading a Pi coding model.")
                return
            if not self.demo and not download_approved:
                try:
                    installed = self.pi.installed(repo)
                except (OSError, ValueError, KeyError) as exc:
                    self.say("Could not validate the Pi model installation: " + str(exc))
                    return
                if installed is None:
                    self.push_screen(
                        Menu(
                            repo + "\nThis model is not downloaded yet onto this local device. "
                            "Would you like to download it now?",
                            [(False, "No — Cancel"), (True, "Yes — Download now")],
                            id="pi-download-confirm",
                        ),
                        lambda approved: self.select_local_model(repo, download_approved=True)
                        if approved is True else None,
                    )
                    return
            if repo != self.settings.pi_model:
                self.state.pi_ready = False
                self.settings.options["PI_LOCAL_MODEL"] = repo
                self.pi_status_checked = 0
                self.pi_install_notified = False
            self.state.private_target = "pi"
            self.state.active_provider = "pi"
            if not self.conversation:
                self.command("new")
            self.say(PI_MODEL_NOTICE)
            self.pi_operation("start", allow_download=download_approved)
        elif repo == "gpu-controls":
            self.menu_selected("gpu")
        elif repo:
            if repo not in self.settings.gpu_model_choices:
                self.say("Unknown GPU model selection.")
                return
            if not self.can_select_local_model():
                self.say("Wait for the GPU operation to finish, or refresh its status first.")
            elif self.state.gpu_status == GPUStatus.STOPPED:
                if repo == MOG_SEC_REPO:
                    profile = gpu_profile(repo, mog_context) if mog_context else self.settings.gpu_profile(repo)
                    self.settings.options["MOG_CONTEXT_PROFILE"] = profile.variant
                self.gpu.spec = None
                self.settings.local_model = repo
                self.state.private_target = "gpu"
                self.state.model_ready = False
                self.state.active_provider = "api"
                if not self.conversation:
                    self.command("new")
                context = f" ({self.settings.local_context_window():,} tokens)" if repo == MOG_SEC_REPO else ""
                self.say("Selected for the next GPU launch: " + repo + context + ". Turn GPUs ON to use it.")
                self.animate()
            else:
                self.state.private_target = "gpu"
                if not self.conversation:
                    self.command("new")
                if repo == MOG_SEC_REPO:
                    self.gpu_operation("load", repo, mog_context=mog_context)
                else:
                    self.gpu_operation("load", repo)

    @work(group="pi", exclusive=True)
    async def pi_operation(self, operation, *, allow_download=False):
        self.pi_busy = True
        try:
            if operation == "stop":
                await self.pi.stop()
                self.say("Raspberry Pi CPU model stopped; memory released.")
            else:
                if self.demo:
                    from velocicyber.models.pi_profiles import PI_PROFILES

                    if self.pi.repo != self.settings.pi_model:
                        await self.pi.stop()
                        self.pi.repo = self.settings.pi_model
                    self.state.pi_ready = True
                    self.state.pi_context_window = PI_PROFILES[
                        self.settings.pi_model
                    ].context_window
                    self.state.pi_stage = "Ready"
                    self.state.active_provider = self.state.private_target = "pi"
                else:
                    await self.pi.start(self.say, allow_download=allow_download)
                if self.state.active_provider == "pi" and not self.engine.running:
                    self.engine.new_task()
        except Exception as exc:
            self.say(str(exc))
        finally:
            self.pi_busy = False
            self.animate()

    def select_api_model(self, model):
        if self.busy or self.sentinel_busy:
            self.say("Wait for the current task or Sentinel operation to finish.")
            return
        if model in API_MODELS:
            selected = API_MODELS[model][1]
            if selected != self.settings.api_model:
                self.state.api_health = "unverified"
            self.settings.api_model = selected
            self.state.active_provider = "api"
            if not self.conversation:
                self.command("new")
            else:
                self.engine.new_task()
            self.say(
                f"Selected {self.settings.api_model_name}. A new task will use the API; GPUs are optional."
            )
            self.animate()

    @on(Input.Changed, "#prompt")
    def filter_commands(self, event):
        menu = self.query_one("#commands", OptionList)
        value = event.value
        if not value.startswith(("/", "\\")):
            menu.display = False
            return
        query = value[1:].lower()
        # Preserve commands with arguments, including spaces in verifier commands.
        if any(query.startswith(name + " ") for name in ("task", "verifier", "new")):
            menu.display = False
            return
        words = query.split()
        matches = [
            Option(f"{value[0]}{name:<16} {label}", id=name)
            for name, label in COMMANDS
            if all(word in f"{name} {label}".lower() for word in words)
        ]
        menu.clear_options()
        menu.add_options(matches or [Option("No matching commands", disabled=True)])
        menu.highlighted = 0 if matches else None
        height = min(7, len(matches) or 1) + 2
        menu.styles.height = height
        menu.styles.offset = (0, -height)
        menu.display = True

    @on(OptionList.OptionSelected, "#commands")
    def command_chosen(self, event):
        event.stop()
        if event.option_id:
            self.activate_command(event.option_id)

    def activate_command(self, name):
        prompt = self.query_one("#prompt", Input)
        prefix = prompt.value[:1] or "\\"
        self.query_one("#commands", OptionList).display = False
        prompt.value = ""
        prompt.focus()
        if name in {"task", "verifier"}:
            prompt.value = prefix + name + " "
            prompt.cursor_position = len(prompt.value)
        elif name in dict(CONTROLS):
            self.menu_selected(name)
        else:
            self.command(name)

    def sentinel_change_idle(self):
        return not (self.busy or self.pi_busy or self.sentinel_busy or self.gpu_busy
                    or self.restoring_gpu or self.data_busy or self.session_ending)

    def sentinel_change_blocked(self):
        if not self.sentinel_change_idle():
            self.say("Wait for the current task or model operation before changing Sentinel.")
            return True
        return False

    def set_sentinel(self, enabled):
        if enabled is None:
            return
        if enabled == "models":
            self.push_screen(Menu("Sentinel model", [
                (key, profile.label + " — Raspberry Pi")
                for key, profile in SENTINEL_MODELS.items()
            ], id="sentinel-models"), self.select_sentinel_model)
            return
        if enabled == "scope":
            if self.sentinel_change_blocked():
                return
            try:
                editor = ScopeEditor(self.settings.root, read_scope(self.settings.root),
                                     can_save=self.sentinel_change_idle)
            except (OSError, ValueError) as error:
                self.say(f"Cannot open scope editor: {error}")
                return
            self.push_screen(editor, self.scope_edited)
            return
        self.request_sentinel(enabled, self.state.sentinel_model)

    def scope_edited(self, saved):
        if saved:
            self.say("SCOPE.MD saved. Sentinel will use it for the next task.")

    def select_sentinel_model(self, model):
        if model is not None:
            self.request_sentinel(True, model)

    def sentinel_swap_plan(self):
        provider = self.state.active_provider
        if provider == "pi":
            provider = ("local" if self.state.model_ready and self.settings.local_model
                        and self.state.gpu_status == GPUStatus.RUNNING else "api")
        return (self.pi.repo, self.state.active_provider, provider, self.chat_model_name(provider))

    def request_sentinel(self, enabled, model):
        if model not in SENTINEL_MODELS:
            self.say("Unknown Sentinel model.")
            return
        if self.sentinel_change_blocked():
            return
        pi_running = (self.state.active_provider == "pi" or self.state.pi_ready
                      or (self.pi.process is not None and self.pi.process.returncode is None))
        if enabled and SENTINEL_MODELS[model].local and pi_running:
            plan = self.sentinel_swap_plan()
            provider = "Lambda GPU" if plan[2] == "local" else "API"
            details = ("Stop Pi coding model: " + plan[0]
                       + "\nStart Sentinel: " + SENTINEL_MODELS[model].label
                       + "\nFuture prompts and conversation context will use "
                       + plan[3] + " (" + provider + ") for coding."
                       + "\nSaved conversations are kept. Queued prompts will be paused.")
            self.push_screen(Menu(
                "Swap the Pi coding model for Sentinel?",
                [(False, "Cancel — keep the Pi coding model"),
                 (True, "Yes — stop Pi coding and start Sentinel")],
                id="sentinel-swap", details=details),
                lambda approved: self.confirm_sentinel_swap(model, plan) if approved else None)
            return
        self.begin_sentinel_operation(enabled, model)

    def confirm_sentinel_swap(self, model, plan):
        if self.sentinel_change_blocked():
            return
        if plan != self.sentinel_swap_plan():
            self.say("The selected coding model changed. Review the updated swap before continuing.")
            self.call_after_refresh(self.request_sentinel, True, model)
            return
        self.begin_sentinel_operation(True, model, swap_plan=plan)

    def begin_sentinel_operation(self, enabled, model, *, swap_plan=None):
        # Reserve the operation synchronously, before the worker can yield to a
        # queued prompt or another model-selection callback.
        self.sentinel_busy = True
        if swap_plan is not None:
            self.pi_busy = True
        self.sentinel_operation(enabled, model, swap_plan=swap_plan)

    @work(group="sentinel", exclusive=True)
    async def sentinel_operation(self, enabled, model=None, *, swap_plan=None):
        self.sentinel_busy = True
        swapped = False
        previous_stage = self.state.sentinel_stage
        try:
            if self.demo:
                if model is not None:
                    self.state.sentinel_model = model
                self.say("DEMO: Sentinel settings preview only; no model is downloaded or loaded.")
                return
            if enabled:
                # Invalid/missing scope must not unload a working coding model.
                load_scope(self.settings.root)
            if swap_plan is not None:
                if swap_plan != self.sentinel_swap_plan():
                    raise RuntimeError("The coding model changed; select Sentinel again to review the swap.")
                if self.queued_prompts:
                    self.queue_paused = True
                    self.update_prompt_queue()
                    self.say("Queue paused for the model swap; use /queue when ready.")
                self.state.sentinel_stage = "UNLOADING PI MODEL"
                self.animate()
                await self.pi.stop()  # Wait for exit and release the shared Pi lock/RAM.
                swapped = True
                if swap_plan != self.sentinel_swap_plan():
                    raise RuntimeError("The coding model changed during shutdown; select a coding model and retry Sentinel.")
                self.state.active_provider = swap_plan[2]
                if swap_plan[2] == "local":
                    self.state.private_target = "gpu"
                self.say("Pi coding model stopped. Next coding model: " + swap_plan[3] + ".")
            if model is not None and model != self.state.sentinel_model:
                await self.sentinel.stop()
                self.state.sentinel_model = model
            if enabled:
                await self.sentinel.start(self.say)
                self.say("Sentinel ON — " + sentinel_profile(self.state).label
                         + " observing tool actions against SCOPE/SCOPE.MD.")
            else:
                await self.sentinel.stop()
                self.say("Sentinel OFF — observer model unloaded.")
        except Exception as exc:
            if swapped:
                self.state.sentinel_enabled = True
                self.state.sentinel_ready = False
                self.state.sentinel_stage = "ERROR"
                self.say("Sentinel is unavailable after the swap. Retry /sentinel or explicitly turn it OFF before coding.")
            elif swap_plan is not None:
                self.state.sentinel_stage = previous_stage
            self.say("Sentinel: " + str(exc))
        finally:
            if swap_plan is not None:
                self.pi_busy = False
            self.sentinel_busy = False
            self.animate()

    def set_collection(self, value):
        if self.data_busy:
            self.say("Wait for local data deletion to finish before changing collection settings.")
            return
        if value is not None:
            self.state.collection = value
            self.say("Fine-tuning Data Collection: " + ("ON" if value else "OFF"))

    def data_in_use(self):
        return (
            self.busy
            or self.data_busy
            or any(worker.group in {"agent", "upload"} for worker in self.workers)
        )

    def data_selected(self, value):
        if value == "advanced":
            self.push_screen(
                Menu(
                    "Advanced data collection settings\n"
                    + "Task ID: "
                    + (self.engine.task_id or "automatic")
                    + "\nVerification command: "
                    + self.engine.scrubber.text(shlex.join(self.engine.verifier) or "OFF"),
                    [
                        ("task", "Task ID — group repeated attempts"),
                        ("verifier", "Verification command — check completed turns"),
                    ],
                    id="data-advanced",
                ),
                self.data_advanced_selected,
            )
        elif value == "allow-compaction":
            self.push_screen(
                Menu(
                    "Allow compaction? Currently "
                    + ("YES" if self.state.allow_compaction else "NO")
                    + "\nInclude compacted sessions in fine-tuning collection and uploads.\n"
                    "Collection and upload permissions still apply.\n"
                    "Automatic context compaction and /resume remain available.",
                    [(True, "YES"), (False, "NO")],
                ),
                self.set_allow_compaction,
            )
        elif value == "upload-manual":
            if self.data_in_use():
                self.say("Wait for the current task, upload, or deletion before uploading data.")
                return
            self.upload_manual()
        elif value == "delete":
            if self.data_in_use():
                self.say(
                    "Wait for the current task, upload, or deletion to finish before deleting data."
                )
                return
            self.push_screen(
                Menu(
                    "Delete ALL locally stored fine-tuning data?\n"
                    "Folders: velocicyber_finetuning-data/ and velocicyber_sentinel-data/\n"
                    "Removes raw episodes, derived datasets, and upload receipts.\n"
                    "Collection and uploads will be switched OFF.\n"
                    "Hugging Face data stays intact. Local deletion cannot be undone.\n"
                    "Saved conversations are kept.",
                    [("cancel", "Cancel"), ("delete-all", "Delete all local data permanently")],
                ),
                self.confirm_data_deletion,
            )
        elif isinstance(value, bool):
            self.set_collection(value)

    def data_advanced_selected(self, value):
        if value not in {"task", "verifier"}:
            return
        if self.busy:
            self.say("Wait for the current task or press Ctrl+X.")
            return
        self.say(
            "Enter a task ID after /task, then press Enter to start a new conversation. "
            "Reuse the same ID for repeated attempts at the same task."
            if value == "task"
            else "Enter a command after /verifier, such as python3 -m pytest -q. "
            "It runs after each completed turn. Press Enter to save; enter off to disable it."
        )
        self.activate_command(value)

    def set_allow_compaction(self, value):
        if value is None:
            return
        if self.data_in_use():
            self.say("Wait for the current task or data operation before changing this setting.")
            return
        self.state.allow_compaction = value
        self.say(
            "Allow compaction? "
            + ("YES" if value else "NO")
            + (
                " — compacted sessions are eligible for collection and upload."
                if value
                else " — only non-compacted sessions are eligible for collection and upload."
            )
        )

    def confirm_data_deletion(self, answer):
        if answer != "delete-all":
            return
        if self.data_in_use():
            self.say("Data is in use. Wait for the current task or upload, then try again.")
            return
        self.data_busy = True
        self.state.collection = False
        self.state.upload_approved = False
        self.delete_finetuning_data()

    @work(group="data_delete")
    async def delete_finetuning_data(self):
        try:
            if self.demo:
                self.say("DEMO: local fine-tuning data would be deleted; no files changed.")
                return
            self.say("Deleting local fine-tuning data…")
            removed = await asyncio.to_thread(
                delete_local_finetuning_data, self.settings.root, approved=True
            )
            sentinel_removed = await asyncio.to_thread(self.sentinel_store.delete, approved=True)
            removed = removed or sentinel_removed
            self.say(
                "All local fine-tuning data deleted. Hugging Face data is unchanged."
                if removed
                else "No local fine-tuning data found."
            )
        except (OSError, ValueError) as exc:
            self.say("Local data deletion failed: " + str(exc))
        finally:
            self.data_busy = False
            self.animate()

    def set_upload(self, value):
        if self.data_busy:
            self.say("Wait for local data deletion to finish before changing upload settings.")
            return
        if value is not None:
            self.state.upload_approved = value
            self.say("Dataset uploads: " + ("ON" if value else "OFF"))

    def gpu_selected(self, value):
        if value == "terminate":
            self.confirm_termination()
            return
        if self.gpu_busy or self.sentinel_busy:
            self.say("A GPU or Sentinel operation is in progress.")
            return
        if value in ("on", "load", "connect") and not self.settings.local_model:
            self.say("Choose and confirm a GPU model in /local first.")
            self.menu_selected("local")
            return
        if value == "load" and self.gpu_model_ready(self.settings.local_model):
            self.state.active_provider = "local"
            self.state.private_target = "gpu"
            self.engine.new_task()
            self.say("Selected the loaded private model.")
            return
        if value == "on":
            if self.state.instance_id or self.state.launch_uncertain:
                self.say("An instance already exists or launch is unresolved. Refresh it first.")
                return
            self.push_screen(
                Menu(
                    "Total GPU memory (VRAM across the instance)\n"
                    "Estimated USD / hr for the whole instance, before tax.\n"
                    + "Selected model: "
                    + self.settings.local_model
                    + (
                        "\n" + self.settings.gpu_profile().notice
                        if self.settings.gpu_profile()
                        else ""
                    ),
                    memory_price_options(self.settings.runtime),
                    groups={
                        i: "Single" if size in SINGLE_MEMORY_CHOICES else "Cluster"
                        for i, size in enumerate(MEMORY_CHOICES)
                    },
                ),
                self.prepare_launch,
            )
        elif value in ("refresh", "load", "connect"):
            self.gpu_operation(value)

    def confirm_termination(self):
        if self.demo:
            self.state.gpu_status = GPUStatus.STOPPED
            self.state.model_ready = False
            if self.state.active_provider == "local":
                self.state.active_provider = "api"
            self.say("DEMO: GPU stopped.")
        elif self.state.instance_id:
            self.push_screen(
                Menu(
                    "Terminate Lambda instance " + self.state.instance_id + "?",
                    [(True, "Terminate Instance")],
                ),
                lambda answer: self.terminate_during_startup() if answer else None,
            )
        else:
            self.say("No tracked instance. Refresh if a previous launch was interrupted.")

    @work(group="termination", exclusive=True)
    async def terminate_during_startup(self):
        # A user must be able to stop billing while model downloads are still running.
        workers = self.workers.cancel_group(self, "plan") + self.workers.cancel_group(self, "gpu")
        await asyncio.gather(*(worker.wait() for worker in workers), return_exceptions=True)
        # Cancelling a worker cannot cancel an in-flight cloud launch request.
        # Wait for its tracked result before selecting the instance to terminate.
        if self.launch_task:
            await asyncio.gather(self.launch_task, return_exceptions=True)
        self.gpu_busy = False
        if self.state.launch_uncertain and not self.state.instance_id:
            try:
                await asyncio.to_thread(self.gpu.refresh)
            except Exception as exc:
                self.say("Session remains open: could not reconcile the launch. " + str(exc))
                self.exit_after_termination = False
                return
        if not self.state.instance_id:
            if self.exit_after_termination:
                self.exit()
            return
        self.gpu_operation("terminate")

    @work(exclusive=True, group="plan")
    async def prepare_launch(self, memory):
        if memory is None:
            return
        if not self.settings.local_model:
            self.say("Choose and confirm a GPU model in /local first.")
            return
        if self.demo:
            self.state.gpu_status = GPUStatus.INITIALIZED
            self.say(f"DEMO: {memory} GB GPU initialized.")
            await asyncio.sleep(0.5)
            self.state.gpu_status = GPUStatus.RUNNING
            self.state.model_ready = True
            self.state.active_provider = "local"
            self.state.private_target = "gpu"
            return
        self.gpu_busy = True
        self.say("Checking model metadata and available Lambda capacity once.")
        try:
            plan = await asyncio.to_thread(self.gpu.plan, memory)
            approved = await self.push_screen_wait(
                Menu(
                    f"Launch {plan.instance_type} in {plan.region}?\n"
                    f"{plan.gpu_count} GPU(s) · {plan.total_vram_gib} GB total VRAM · ${plan.hourly_cents / 100:.2f}/hour\n"
                    f"Model: {plan.model.repo}\n"
                    f"Context: {self.settings.local_context_window():,} tokens\n"
                    "Billing continues until termination.",
                    [(True, "Approve launch and load model")],
                )
            )
            if approved:
                self.launch_task = asyncio.create_task(
                    asyncio.to_thread(self.gpu.launch, plan, approved=True)
                )
                await asyncio.shield(self.launch_task)
                self.say("Instance initialized. Waiting for Lambda to report active.")
                for _ in range(60):
                    await asyncio.sleep(30)
                    status = await asyncio.to_thread(self.gpu.refresh)
                    if status == "active":
                        await self.load_model()
                        break
                    if status in ("terminated", "unhealthy"):
                        raise RuntimeError("Instance became " + status)
                else:
                    self.say("Startup wait expired; refresh or terminate the tracked instance.")
        except Exception as exc:
            self.gpu.record_error("launch/load", exc, memory=memory)
            self.say(str(exc))
        finally:
            self.gpu_busy = False

    async def load_model(self):
        profile = self.settings.gpu_profile()
        self.say("Loading the selected model. " + model_option_label(
            self.settings.local_model, variant=profile.variant if profile else ""
        ))
        try:
            await self.gpu.deploy()
            for _ in range(60):
                if await self.gpu.check_ready():
                    if self.state.active_provider == "local":
                        if not self.engine.running:
                            self.engine.new_task()
                        self.say("Private model is ready. Inference now uses the GPU.")
                    else:
                        self.say("GPU model is ready; Raspberry Pi inference remains selected.")
                    return
                await self.gpu.check_model_process()
                await asyncio.sleep(30)
            self.say("Model readiness timed out. GPU remains billable; inspect it or terminate it.")
        finally:
            if self.state.model_stage != "failed":
                self.state.model_stage = ""

    @work(group="gpu")
    async def gpu_operation(self, operation, repo=None, *, mog_context=None):
        if self.gpu_busy:
            return
        self.gpu_busy = True
        try:
            if self.demo:
                if operation == "load" and repo:
                    if repo == MOG_SEC_REPO and mog_context:
                        self.settings.options["MOG_CONTEXT_PROFILE"] = gpu_profile(repo, mog_context).variant
                    self.settings.local_model = repo
                    self.state.model_context_window = self.settings.local_context_window()
                    self.state.model_context_model = repo
                    self.state.model_ready = True
                    self.state.active_provider = "local"
                self.say("DEMO: " + operation)
            elif operation == "load":
                repo = repo or self.settings.local_model
                if self.gpu_model_ready(repo, mog_context):
                    self.state.active_provider = "local"
                    self.engine.new_task()
                    self.say("Selected the loaded private model.")
                    return
                if repo == MOG_SEC_REPO and mog_context:
                    spec = await asyncio.to_thread(self.gpu.inspect_model, repo, profile_variant=mog_context)
                else:
                    spec = await asyncio.to_thread(self.gpu.inspect_model, repo)
                if (
                    await asyncio.to_thread(self.gpu.refresh) != "active"
                    or self.state.gpu_status != GPUStatus.RUNNING
                ):
                    raise RuntimeError("The tracked GPU is not active.")
                if spec.required_gib > self.state.total_vram_gib:
                    raise ValueError(
                        f"This model needs at least ~{spec.required_gib} GiB; "
                        "choose a larger instance before switching."
                    )
                self.gpu.validate_loaded_hardware(spec)
                if repo == MOG_SEC_REPO:
                    self.settings.options["MOG_CONTEXT_PROFILE"] = spec.profile_variant
                self.settings.local_model = repo
                self.state.model_ready = False
                self.state.active_provider = "api"
                await self.load_model()
            elif operation == "connect":
                status = await asyncio.to_thread(self.gpu.refresh)
                if status != "active" or self.state.gpu_status != GPUStatus.RUNNING:
                    raise RuntimeError("The tracked GPU is not active.")
                await self.gpu.connect()
                self.say(
                    "Model ready." if await self.gpu.check_ready() else "Model is not ready yet."
                )
            elif operation == "terminate":
                await asyncio.to_thread(self.gpu.terminate, approved=True)
                await self.gpu.close_tunnel()
                self.say("Termination requested; waiting for confirmation.")
                for _ in range(20):
                    await asyncio.sleep(self.termination_poll_seconds)
                    if await asyncio.to_thread(self.gpu.refresh) == "terminated":
                        self.say("GPUs: OFF — termination confirmed.")
                        if self.exit_after_termination:
                            self.exit()
                        break
                else:
                    self.say("Termination is not yet confirmed. Refresh or check Lambda console.")
            else:
                status = await asyncio.to_thread(self.gpu.refresh)
                self.say("Lambda status: " + status)
                if status == "active" and not self.state.model_ready:
                    self.say("GPU is running. Use GPU / GPU Cluster → Reconnect inference tunnel.")
                if status == "terminated" and self.exit_after_termination:
                    self.exit()
        except Exception as exc:
            self.gpu.record_error(operation, exc, model=repo)
            self.say(str(exc))
        finally:
            self.gpu_busy = False
            if operation == "refresh":
                self.restoring_gpu = False
            self.animate()

    @on(Input.Submitted, "#prompt")
    def submitted(self, event):
        menu = self.query_one("#commands", OptionList)
        if menu.display and menu.highlighted_option and menu.highlighted_option.id:
            self.activate_command(menu.highlighted_option.id)
            return
        value = event.value.strip()
        if not value:
            return
        if value in ("/", "\\"):
            event.input.value = ""
            self.action_menu()
        elif value.startswith(("/", "\\")):
            event.input.value = ""
            self.command(value[1:])
        elif self.session_ending:
            self.say(self.task_start_blocker())
        elif self.busy:
            event.input.value = ""
            self.queue_prompt(value)
        elif blocker := self.task_start_blocker():
            self.say(blocker)
        else:
            event.input.value = ""
            self.run_prompt(value)

    def command(self, value):
        if value.strip() == "quit":
            self.action_quit()
            return
        if value.strip() == "queue":
            self.open_prompt_queue()
            return
        if self.busy:
            self.say("Wait for the current task or press Ctrl+X.")
            return
        name, _, rest = value.partition(" ")
        if name in dict(CONTROLS):
            self.menu_selected(name)
        elif name == "chat":
            self.show_conversation()
        elif name == "home":
            self.show_home()
        elif name == "help":
            self.say(
                "Type \\ or / to filter commands · /new new conversation · "
                "/resume saved conversations · /chat conversation · /home main menu · "
                "/sandbox filesystem access · "
                "/sentinel scope observer ON/OFF · "
                "/data-collection collection, manual upload and Advanced task/verification settings · "
                "/queue view, pause, resume or clear queued prompts · "
                "/updates-info harness/engine versions · "
                "/update manual update/rollback · /setup secure configuration · /quit\n"
                "Drag session text to select · Ctrl+C copies · Ctrl+A selects the focused log.\n"
                "If clipboard access is blocked, Shift-drag and use your terminal's Copy command."
            )
        elif name == "updates-info":
            self.say(
                "DEMO: update checks are disabled; no GitHub requests."
                if self.demo
                else self.updates.details()
            )
            if not self.demo:
                self.say("Open /update to install the available harness/engine updates or roll back.")
        elif name == "setup":
            if rest.strip():
                self.say("Use /setup without arguments. Enter credentials only in its hidden prompts.")
            else:
                self.setup_menu()
        elif name == "update":
            self.update_menu()
        elif name == "update-engine":
            self.engine_update_menu()
        elif name in ("new", "task"):
            self.engine.new_task(rest.strip() or None)
            self.query_one("#log", RichLog).clear()
            self.clear_prompt_queue("starting a new conversation")
            self.command_items = {}
            self.live_text = ""
            self.completion_notice = ""
            self.task_failed = self.stop_requested = self.compacting = False
            self.activity = "Ready for a task"
            self.show_conversation()
            self.say("Started a new task." + (" Task ID: " + rest if rest else ""))
        elif name == "verifier":
            try:
                verifier = [] if rest.strip() == "off" else shlex.split(rest)
            except ValueError as exc:
                self.say("Invalid verification command: " + str(exc))
                return
            self.engine.verifier = verifier
            self.say(
                "Verifier: "
                + (shlex.join(self.engine.verifier) or "none; episodes remain unverified")
            )
        elif name == "upload-manual":
            self.data_selected("upload-manual")
        else:
            self.say("Unknown command. Type /help.")

    def pending_response_text(self, text):
        # Provider previews arrive independently of native engine completion events.
        # A late preview may be the whole answer or an earlier prefix of it.
        preview = text.strip()
        if preview and any(reply.startswith(preview) for reply in self.completed_responses):
            return ""
        return text

    def engine_event(self, event):
        kind = event.get("type")
        if kind == "sentinel.stopped":
            if not self.task_failed or event.get("active_process_stopped"):
                self.stop_requested = True
            self.say("SENTINEL: " + event["message"] + (
                " Task processes were stopped."
                if event.get("active_process_stopped") else
                " No active process remained; completed actions cannot be undone."
            ), kind="sentinel_stop")
            return
        if kind == "sentinel.review":
            return
        if kind == "context.compaction.started":
            self.compacting = True
            self.live_text = ""
            self.activity = "Compacting conversation…"
            self.say(
                "Compacting conversation… summarizing progress to make room for the next steps."
            )
            return
        if kind in ("context.compaction.completed", "context.compaction.failed"):
            self.compacting = False
            completed = kind.endswith("completed")
            self.activity = "Continuing task" if completed else "Compaction failed — see output"
            self.say(
                "Compaction complete — continuing the task."
                if completed
                else "Compaction failed — see the model error below."
            )
            return
        if self.compacting and kind in ("provider.progress", "item.started", "item.updated"):
            return
        if kind == "provider.progress":
            if not self.busy:
                return
            self.activity = event["stage"]
            self.live_text = self.pending_response_text(event.get("text", ""))
        elif kind == "tool.error":
            self.say(
                f"Tool error ({event.get('tool', 'tool')}): {event['message']}\n"
                "The model has received the error and may retry with corrected arguments."
            )
        elif kind == "engine.notice":
            if event.get("code") == "missing_final_summary":
                self.live_text = ""
                self.completion_notice = (
                    "Verified complete — no model summary"
                    if event.get("completion_status") == "verified_no_summary"
                    else "Task paused — no model summary; enter a follow-up"
                    if event.get("completion_status") == "response_incomplete"
                    else "Actions completed — no model summary"
                )
            self.say("Harness: " + event["message"])
        elif kind in ("thread.started", "turn.started"):
            self.activity = "Waiting for model"
        elif kind in ("item.started", "item.updated", "item.completed"):
            item = event["item"]
            if item.get("type") == "agent_message":
                if kind == "item.completed":
                    if item.get("text", "").strip():
                        self.completed_responses.add(item["text"].strip())
                        self.live_text = ""
                        self.say("\nVelocicyber:\n" + item["text"])
                else:
                    self.live_text = self.pending_response_text(item.get("text", ""))
                self.activity = "Writing response"
            elif item.get("type") == "command_execution":
                command = item["command"]
                if isinstance(command, list):
                    command = shlex.join(command)
                item_id = item.get("id") or command
                if item_id not in self.command_items:
                    self.say("\n$ " + command, kind="command")
                    self.command_items[item_id] = ""
                output = item.get("aggregated_output", "")
                previous = self.command_items[item_id]
                extra = output[len(previous) :] if output.startswith(previous) else output
                if extra:
                    self.say(extra, kind="output")
                self.command_items[item_id] = output
                if kind == "item.completed":
                    self.say(
                        f"Command {item.get('status', 'finished')} · exit {item.get('exit_code')}"
                    )
                    self.activity = "Waiting for model"
                else:
                    self.activity = "Running: " + command
            elif item.get("type") == "reasoning":
                self.activity = "Model working"
            elif item.get("type") == "error":
                self.say("Engine notice: " + item.get("message", "Unspecified engine notice."))
            elif kind == "item.completed":
                self.say(
                    item.get("text")
                    or item.get("message")
                    or (item.get("type", "tool").replace("_", " ") + " completed.")
                )
            else:
                self.activity = "Working: " + item.get("type", "tool").replace("_", " ")
        elif kind in ("engine.error", "error", "turn.failed"):
            self.task_failed = True
            self.activity = "Task failed — see output"
            error = event.get("error") or event.get("message") or "The model task failed."
            if isinstance(error, dict):
                error = error.get("message", "The model task failed.")
            self.say(str(error))
        elif kind == "turn.completed":
            self.activity = "Finishing task"
        elif kind == "dataset.prepared":
            self.say(
                "This interaction was saved locally; dataset prepared: "
                + Path(event["path"]).name[:12]
            )
            self.show_dataset_summary(event)
            if event.get("eligible", 0) > 0 and not self.state.upload_approved:
                self.say("Hugging Face upload is OFF. Use /upload-perms to enable it.")
        elif kind == "dataset.skipped" and event.get("reason") == "compaction_disabled":
            self.say(
                "Fine-tuning collection skipped: this conversation was compacted and "
                "Allow compaction? is NO. Your conversation remains available in /resume."
            )

    def show_dataset_summary(self, summary):
        if "eligible" not in summary:
            return
        reasons = "; ".join(
            f"{name.replace('_', ' ')}: {count}"
            for name, count in sorted(summary.get("exclusion_reasons", {}).items())
        )
        self.say(
            "Local collection totals (including earlier interactions):\n"
            f"{summary['eligible']} eligible rows; "
            f"{summary['excluded']} excluded" + (f" ({reasons})" if reasons else "") + "."
        )
        if summary.get("unverified"):
            self.say(
                f"Includes {summary['unverified']} completed conversations labeled unverified "
                "(no passing test result)."
            )
        if not summary["eligible"]:
            self.say("Hugging Face: nothing eligible to upload.")

    @work(group="agent")
    async def run_prompt(self, prompt):
        if self.session_ending:
            return
        # Workers start asynchronously: two quick submissions must never run
        # two engine processes against the same conversation.
        if self.busy:
            self.queue_prompt(prompt)
            return
        if blocker := self.task_start_blocker():
            self.say(blocker)
            return
        if not self.queued_prompts:
            self.queue_paused = False
        try:
            while True:
                await self.run_prompt_turn(prompt)
                if self.stop_requested or self.task_failed:
                    self.queue_paused = True
                    if self.queued_prompts and not self.session_ending:
                        self.say("Queue paused. Use /queue to resume or clear pending prompts.")
                self.update_prompt_queue()
                if self.session_ending or self.queue_paused or not self.queued_prompts:
                    break
                if blocker := self.task_start_blocker():
                    self.queue_paused = True
                    self.say(blocker + " Queue paused; use /queue to resume.")
                    self.update_prompt_queue()
                    break
                prompt = self.queued_prompts.popleft()
        except BaseException:
            self.queue_paused = True
            self.update_prompt_queue()
            raise

    async def run_prompt_turn(self, prompt):
        self.busy = True
        self.reply_provider = self.state.active_provider
        self.reply_model = self.chat_model_name(self.reply_provider)
        self.task_started = time.monotonic()
        self.task_failed = self.stop_requested = False
        self.completion_notice = ""
        self.compacting = False
        self.command_items = {}
        self.live_text = ""
        self.completed_responses.clear()
        self.activity = "Starting agent"
        self.show_conversation()
        self.say("\nYou: " + prompt)
        uploads_attempted = False
        try:
            if self.demo:
                await asyncio.sleep(0.3)
                self.say(
                    "DEMO: Terminal controls are working. Start without --demo to use Velocicyber."
                )
            else:
                if (self.state.active_provider == "local" and self.state.model_ready
                        and self.state.instance_ip
                        and (not self.gpu.tunnel or self.gpu.tunnel.returncode is not None)):
                    self.say("Reconnecting the inference tunnel before sending this task…")
                    try:
                        await self.gpu.connect()
                    except Exception:
                        self.state.model_ready = False
                        raise
                await self.engine.run(prompt, self.engine_event)
                if self.state.collection and self.state.upload_approved:
                    try:
                        uploads_attempted = True
                        await self.flush_uploads()
                    except Exception as exc:
                        self.say(
                            "Hugging Face upload failed: "
                            + str(exc)
                            + ". Use Manual Upload to retry."
                        )
        except Exception as exc:
            if not self.task_failed:
                self.say(str(exc))
            self.task_failed = True
        finally:
            # A sentinel stop is a failed/interrupted main turn but still yields
            # useful observer records. Use the same collection/upload permission.
            if (not self.demo and not uploads_attempted and self.state.sentinel_enabled
                    and self.state.collection
                    and self.state.upload_approved):
                try:
                    await self.flush_uploads()
                except Exception as exc:
                    self.say("Upload pending: " + str(exc))
            if partial := self.pending_response_text(self.live_text):
                self.say("Partial response:\n" + partial)
            self.live_text = ""
            self.busy = False
            self.compacting = False
            self.reply_provider = self.reply_model = ""
            self.activity = (
                "Task stopped — enter another task"
                if self.stop_requested
                else "Task failed — see output above"
                if self.task_failed
                else self.completion_notice or "Task complete — enter a follow-up"
            )
            self.update_activity()
            self.update_chat_model()

    async def flush_uploads(self):
        if self.data_busy:
            raise RuntimeError("Wait for local data deletion to finish before uploading.")
        if self.demo:
            self.say("DEMO: no datasets uploaded.")
            return
        if not self.state.upload_approved:
            raise PermissionError("Use the menu to approve dataset uploads for this session first.")
        errors = []
        try:
            await self.flush_main_uploads()
        except Exception as exc:
            errors.append("Main dataset: " + str(exc))
        try:
            receipt = await asyncio.to_thread(
                self.sentinel_store.upload, approved=True,
                allow_compaction=self.state.allow_compaction,
            )
            if receipt:
                self.say(
                    f"Sentinel: {receipt['records']} observations uploaded; local copies cleared. "
                    "Model predictions require review before training."
                )
        except Exception as exc:
            errors.append("Sentinel dataset: " + str(exc))
        if errors:
            raise RuntimeError("; ".join(errors))

    async def flush_main_uploads(self):
        # Re-curate once instead of replaying every cumulative historical snapshot.
        path = await asyncio.to_thread(self.engine.curator().run)
        receipt = await asyncio.to_thread(self.uploader.upload, path, approved=True)
        if receipt is None:
            self.show_dataset_summary(dataset_summary(path))
        else:
            self.say(
                "Dataset already up to date; no duplicate upload."
                if receipt.get("unchanged")
                else "Unique dataset published: " + receipt["training_hash"][:12]
            )
            cleanup = receipt.get("local_cleanup", {})
            if cleanup.get("complete"):
                self.say(
                    f"Local training copies cleared for {cleanup['episodes']} uploaded episodes. "
                    "Saved conversations are available in /resume."
                )
            elif cleanup:
                self.say("Upload succeeded; local cleanup pending: " + cleanup["error"])

    @work(group="upload", exclusive=True)
    async def upload_manual(self):
        try:
            await self.flush_uploads()
        except Exception as exc:
            self.say("Upload pending: " + str(exc))

    def action_new_task(self):
        self.command("new")

    async def action_cancel_task(self):
        self.queue_paused = True
        self.update_prompt_queue()
        if self.busy:
            self.stop_requested = True
            self.activity = "Stopping task"
            self.update_activity()
        await self.engine.cancel()

    @work(group="session_end", exclusive=True)
    async def finish_task_and_quit(self):
        self.say("Ending session: stopping the current task.")
        await self.engine.cancel()
        # Let the engine record the interrupted task before closing the session.
        workers = [worker for worker in self.workers if worker.group == "agent"]
        await asyncio.gather(*(worker.wait() for worker in workers), return_exceptions=True)
        self.busy = False
        self.action_quit()

    def action_quit(self):
        self.session_ending = True
        self.clear_prompt_queue("ending the session")
        if self.exit_after_termination and self.gpu_busy:
            self.say("Ending session: waiting for GPU termination confirmation.")
            return
        if self.busy:
            self.finish_task_and_quit()
            return
        if not self.demo and (
            self.state.instance_id or self.state.launch_uncertain or self.gpu_busy
        ):
            self.exit_after_termination = True
            self.say("Ending session: terminating its GPU before closing.")
            self.terminate_during_startup()
        else:
            self.exit()

    async def on_unmount(self):
        self.session_ending = True
        self.queued_prompts.clear()
        await self.engine.cancel()
        await self.sentinel.stop()
        await self.pi.stop()
        await self.gpu.close_tunnel()
        self.engine.close()
        if self.gpu._client:
            self.gpu._client.close()
