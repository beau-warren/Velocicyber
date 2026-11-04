"""Selectable styled logs with clipboard text independent of visual wrapping."""

from rich.cells import cell_len
from rich.text import Text
from textual.binding import Binding
from textual.strip import Strip
from textual.widgets import RichLog


class TranscriptLog(RichLog):
    BINDINGS = [
        Binding("ctrl+c,ctrl+shift+c", "copy_selection", "Copy", show=True),
        Binding("ctrl+a", "select_all", "Select all", show=False),
        Binding("escape", "clear_selection", "Clear selection", show=False),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._copy_text = ""
        self._copy_offsets = []

    def write(self, content, width=None, expand=False, shrink=True, scroll_end=None, animate=False):
        # RichLog defers writes until mounted; record source text only when rendered.
        if not self._size_known:
            return super().write(content, width, expand, shrink, scroll_end, animate)
        text = content.copy() if isinstance(content, Text) else Text(str(content))
        text.expand_tabs()
        start_line = len(self.lines)
        if self.text_selection is not None:
            scroll_end = False
        elif scroll_end is None:
            scroll_end = self.auto_scroll and self.is_vertical_scroll_end
        super().write(text, width, expand, shrink, scroll_end, animate)
        new_lines = self.lines[start_line:]
        source = text.plain
        offsets, cursor = [], 0
        for line in new_lines:
            visible = line.text.rstrip(" ")
            start = source.find(visible, cursor)
            if start < 0 or source[cursor:start].strip():
                # A nonstandard Rich transform: copying displayed text is safer
                # than associating the selection with unrelated source characters.
                source = "\n".join(line.text for line in new_lines)
                offsets, cursor = [], 0
                for rendered in new_lines:
                    offsets.append((cursor, cursor + len(rendered.text)))
                    cursor += len(rendered.text) + 1
                break
            end = start + len(visible)
            offsets.append((start, end))
            cursor = end
            if not visible:
                while cursor < len(source) and source[cursor] == " ":
                    cursor += 1
            if source[cursor : cursor + 1] == "\n":
                cursor += 1
        base = len(self._copy_text)
        self._copy_offsets.extend((base + start, base + end) for start, end in offsets)
        self._copy_text += source + ("" if source.endswith("\n") else "\n")
        return self

    def clear(self):
        if self.is_attached and self.text_selection is not None:
            self.action_clear_selection()
        self._copy_text = ""
        self._copy_offsets.clear()
        return super().clear()

    def get_selection(self, selection):
        if not self._copy_offsets:
            return "", "\n"

        def source_offset(offset, default):
            if offset is None:
                return default
            if offset.y >= len(self._copy_offsets):
                return len(self._copy_text)
            start, end = self._copy_offsets[max(0, offset.y)]
            return min(end, start + max(0, offset.x))

        start = source_offset(selection.start, 0)
        end = source_offset(selection.end, len(self._copy_text))
        return self._copy_text[start:end], "\n"

    def render_line(self, y):
        scroll_x, scroll_y = self.scroll_offset
        row = y + scroll_y
        width = self.scrollable_content_region.width
        if row >= len(self.lines):
            return Strip.blank(width, self.rich_style)
        line = self.lines[row]
        selection = self.text_selection
        if selection and (span := selection.get_span(row)) is not None:
            start, end = span
            plain = line.text
            start_cell = cell_len(plain[:start])
            end_cell = line.cell_length if end < 0 else cell_len(plain[:end])
            style = self.screen.get_component_rich_style("screen--selection")
            line = Strip.join(
                [
                    line.crop(0, start_cell),
                    line.crop(start_cell, end_cell).apply_style(style),
                    line.crop(end_cell),
                ]
            )
        # Crop uses terminal cells; selection uses Unicode character indices.
        character_start = len(line.text) - len(line.crop(scroll_x).text)
        return (
            line.crop_extend(scroll_x, scroll_x + width, self.rich_style)
            .apply_offsets(character_start, row)
            .apply_style(self.rich_style)
        )

    def action_copy_selection(self):
        text = self.screen.get_selected_text()
        if text:
            self.app.copy_to_clipboard(text)

    def action_select_all(self):
        self.text_select_all()

    def action_clear_selection(self):
        self.screen.selections = {
            widget: selection
            for widget, selection in self.screen.selections.items()
            if widget is not self
        }
