"""Filterable menus and command-aware prompt input."""

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from velocicyber.ui.commands import command_name


class Prompt(Input):
    BINDINGS = [
        Binding("up", "previous_command", show=False),
        Binding("down", "next_command", show=False),
        Binding("tab", "complete_command", show=False),
        Binding("escape", "close_commands", show=False),
    ]

    def check_action(self, action, parameters):
        if action in {"previous_command", "next_command", "complete_command", "close_commands"}:
            return self.app.query_one("#commands", OptionList).display
        return True

    def action_previous_command(self):
        self.app.query_one("#commands", OptionList).action_cursor_up()

    def action_next_command(self):
        self.app.query_one("#commands", OptionList).action_cursor_down()

    def action_complete_command(self):
        option = self.app.query_one("#commands", OptionList).highlighted_option
        if option and option.id:
            self.value = (
                self.value[0] + command_name(option.id)
                + (" " if option.id in {"task", "verifier"} else "")
            )
            self.cursor_position = len(self.value)

    def action_close_commands(self):
        self.app.query_one("#commands", OptionList).display = False

class Menu(ModalScreen):
    DEFAULT_CSS = """
    Menu { align: center bottom; background: transparent; }
    Menu > Vertical { width: 100%; height: auto; max-height: 90%; margin: 0 2 2 2;
        background: #000000; }
    Menu Label { height: auto; max-height: 7; padding: 0 1;
        color: #39ff14; background: #000000; }
    Menu OptionList { height: auto; max-height: 10; border: round #ff39c8;
        background: #000000; color: #39ff14; }
    Menu OptionList > .option-list--option-highlighted { background: #241020; color: #ff39c8; }
    Menu OptionList > .option-list--option-hover { background: #102418; }
    Menu Input { height: 3; margin: 0; background: #000000; border: tall #ff39c8; }
    Menu Input:focus { background-tint: transparent; }
    Menu.with-details > Vertical { height: 90%; }
    Menu #model-details-body { height: 1fr; min-height: 1; padding: 0 1; }
    """
    BINDINGS = [
        Binding("escape", "dismiss(None)", "Close"),
        Binding("up", "previous_option", show=False),
        Binding("down", "next_option", show=False),
        Binding("tab", "complete_option", show=False, priority=True),
        Binding("pageup", "page_details(-1)", show=False, priority=True),
        Binding("pagedown", "page_details(1)", show=False, priority=True),
    ]

    def __init__(self, title, options, *, id=None, groups=None, details=None):
        super().__init__(id=id)
        self.heading, self.options = title, options
        self.groups = groups or {}
        self.disabled_choices = set()
        self.details = details
        self.set_class(details is not None, "with-details")

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self.heading, markup=False)
            if self.details is not None:
                with VerticalScroll(id="model-details-body"):
                    yield Static(self.details, markup=False)
            yield OptionList(id="menu-options", markup=False, compact=True)
            yield Input(
                placeholder="Type to filter · ↑↓ choose · Enter select · Esc close",
                id="menu-filter",
            )

    def on_mount(self):
        self.filter_options()
        self.query_one(Input).focus()

    def check_action(self, action, parameters):
        if action == "page_details":
            return self.details is not None
        return super().check_action(action, parameters)

    def action_page_details(self, direction):
        body = self.query_one("#model-details-body", VerticalScroll)
        if direction < 0:
            body.scroll_page_up(animate=False)
        else:
            body.scroll_page_down(animate=False)

    @on(Input.Changed, "#menu-filter")
    def filter_options(self):
        words = self.query_one(Input).value.lower().lstrip("\\/").split()
        choices = self.query_one(OptionList)
        previous = choices.highlighted_option.id if choices.highlighted_option else None
        matches, shown_groups = [], set()
        for index, (_, label) in enumerate(self.options):
            group = self.groups.get(index)
            if not all(word in f"{group or ''} {label}".lower() for word in words):
                continue
            if group and group not in shown_groups:
                matches.append(Option(f"── {group} ──", disabled=True))
                shown_groups.add(group)
            matches.append(
                Option(label, id=f"choice-{index}", disabled=index in self.disabled_choices)
            )
        choices.clear_options()
        choices.add_options(matches or [Option("No matching options", disabled=True)])
        enabled = [i for i, option in enumerate(matches) if not option.disabled]
        choices.highlighted = next(
            (i for i in enabled if matches[i].id == previous), enabled[0] if enabled else None
        )

    def set_disabled(self, indexes):
        if self.disabled_choices != indexes:
            self.disabled_choices = indexes
            self.filter_options()

    def action_previous_option(self):
        self.query_one(OptionList).action_cursor_up()

    def action_next_option(self):
        self.query_one(OptionList).action_cursor_down()

    def action_complete_option(self):
        option = self.query_one(OptionList).highlighted_option
        if option and option.id:
            index = int(option.id.split("-")[1])
            field = self.query_one(Input)
            field.value = self.options[index][1].splitlines()[0]
            field.cursor_position = len(field.value)

    @on(Input.Submitted, "#menu-filter")
    def submit_option(self, event):
        event.stop()
        self.query_one(OptionList).action_select()

    @on(OptionList.OptionSelected, "#menu-options")
    def choose(self, event):
        event.stop()
        if event.option_id:
            self.dismiss(self.options[int(event.option_id.split("-")[1])][0])
