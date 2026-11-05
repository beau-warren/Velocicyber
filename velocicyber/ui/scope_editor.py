"""A personal scope editor with explicit save/cancel and no transcript logging."""

from textual import on
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static, TextArea

from velocicyber.sentinel.scope import MAX_SCOPE_BYTES, save_scope


class ScopeEditor(ModalScreen):
    DEFAULT_CSS = """
    ScopeEditor { align: center middle; background: #000000 75%; }
    ScopeEditor > Vertical { width: 95%; max-width: 110; height: 90%;
        background: #000000; border: round #ff39c8; padding: 0 1; }
    ScopeEditor Label { height: auto; color: #39ff14; }
    ScopeEditor TextArea { height: 1fr; min-height: 3; margin: 1 0;
        background: #000000; color: #e3ffe7; border: tall #39ff14; }
    ScopeEditor TextArea:focus { border: tall #ff39c8; }
    ScopeEditor #scope-error { height: auto; max-height: 3; color: #ff5555; }
    ScopeEditor #scope-size { height: 1; color: #00cfff; }
    ScopeEditor Horizontal { height: 3; }
    ScopeEditor Button { min-width: 12; margin-right: 2; background: #000000;
        color: #39ff14; border: tall #ff39c8; }
    ScopeEditor Button:focus { background: #241020; }
    """
    BINDINGS = [
        Binding("ctrl+s", "save", "Save", priority=True),
        Binding("escape", "dismiss(False)", "Cancel", priority=True),
    ]

    def __init__(self, root, body, *, can_save):
        super().__init__(id="scope-editor")
        self.root, self.original, self.can_save = root, body, can_save
        self.text = (body or b"").decode("utf-8")

    def compose(self):
        with Vertical():
            yield Label("SCOPE/SCOPE.MD — Edit scope restrictions", markup=False)
            yield Label("Applies to the next task. Ctrl+S saves · Esc cancels · Tab selects buttons.")
            yield TextArea(self.text, id="scope-text", soft_wrap=True, tab_behavior="focus")
            yield Static(id="scope-size", markup=False)
            yield Static("", id="scope-error", markup=False)
            with Horizontal():
                yield Button("Save", id="scope-save")
                yield Button("Cancel", id="scope-cancel")

    def on_mount(self):
        self.query_one(TextArea).focus()
        self.update_size()

    @on(TextArea.Changed, "#scope-text")
    def update_size(self):
        size = len(self.query_one(TextArea).text.encode("utf-8"))
        self.query_one("#scope-size", Static).update(f"{size:,} / {MAX_SCOPE_BYTES:,} UTF-8 bytes")

    def action_save(self):
        if not self.can_save():
            self.query_one("#scope-error", Static).update("Wait for the current task or model operation before saving.")
            return
        try:
            save_scope(self.root, self.query_one(TextArea).text, expected=self.original)
        except (OSError, ValueError) as error:
            self.query_one("#scope-error", Static).update(str(error))
            return
        self.dismiss(True)

    @on(Button.Pressed)
    def button_pressed(self, event):
        event.stop()
        if event.button.id == "scope-save":
            self.action_save()
        elif event.button.id == "scope-cancel":
            self.dismiss(False)
