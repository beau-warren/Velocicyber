import pytest
from rich.text import Text

from velocicyber.ui.app import VelocicyberApp
from velocicyber.ui.transcript import TranscriptLog


async def drag(pilot, start, end):
    await pilot.mouse_down(offset=start)
    await pilot.hover(offset=end)
    await pilot.mouse_up(offset=end)
    await pilot.pause()


async def test_mouse_selection_and_copy_preserve_styling_and_unicode(settings, monkeypatch):
    app = VelocicyberApp(settings, demo=True)
    monkeypatch.setattr(app, "notify", lambda *a, **kw: pytest.fail("Unexpected copy popup"))
    async with app.run_test(size=(80, 24)) as pilot:
        app.show_conversation()
        log = app.query_one("#log", TranscriptLog)
        log.clear()
        log.write(Text("α 界 beta\n    print('hello')", style="#ff39c8"))
        await pilot.pause()
        x, y = log.content_region.offset
        await drag(pilot, (x + 2, y), (x + 4, y))
        assert app.screen.get_selected_text() == "界"
        await pilot.press("ctrl+c")
        assert app.clipboard == "界"
        assert app.is_running
        assert any(
            segment.style.color.get_truecolor().hex == "#ff39c8"
            for segment in log.lines[0]
            if segment.style and segment.style.color
        )
        await drag(pilot, (x, y), (x + 18, y + 1))
        await pilot.press("ctrl+shift+c")
        assert app.clipboard == "α 界 beta\n    print('hello')"
        await pilot.press("escape")
        assert log.text_selection is None
        log.action_copy_selection()  # Copying without a selection is quiet too.


async def test_copy_wrapped_command_preserves_original_line_breaks_and_select_all(settings):
    app = VelocicyberApp(settings, demo=True)
    async with app.run_test(size=(80, 24)) as pilot:
        app.show_conversation()
        log = app.query_one("#log", TranscriptLog)
        log.clear()
        source = "$ printf '%s' " + "word " * 28 + "\n    next command\n\nlast line"
        log.write(Text(source, style="#39ff14"))
        await pilot.pause()
        assert len(log.lines) > len(source.splitlines())
        x, y = log.content_region.offset
        # Select the first logical line through its last wrapped display row.
        command_rows = next(i for i, line in enumerate(log.lines) if "next command" in line.text)
        end = log.lines[command_rows - 1].text.rstrip()
        await drag(pilot, (x, y), (x + len(end), y + command_rows - 1))
        await pilot.press("ctrl+c")
        assert app.clipboard == source.splitlines()[0].rstrip()
        await pilot.press("ctrl+a", "ctrl+c")
        assert app.clipboard == source
        log.clear()
        log.write("A different session")
        await pilot.pause()
        assert log.text_selection is None
        await pilot.press("ctrl+a", "ctrl+c")
        assert app.clipboard == "A different session"


async def test_selection_survives_new_output_and_scroll_position(settings):
    app = VelocicyberApp(settings, demo=True)
    async with app.run_test(size=(80, 24)) as pilot:
        app.show_conversation()
        log = app.query_one("#log", TranscriptLog)
        log.clear()
        for i in range(30):
            log.write(f"line {i}")
        await pilot.pause()
        log.scroll_to(y=10, animate=False, immediate=True)
        await pilot.pause()
        x, y = log.content_region.offset
        await drag(pilot, (x, y), (x + 7, y + 1))
        selected = app.screen.get_selected_text()
        position = log.scroll_y
        log.write("New streaming output")
        await pilot.pause()
        assert log.scroll_y == position
        await pilot.press("ctrl+c")
        assert app.clipboard == selected == "line 10\nline 11"


async def test_selection_after_horizontal_scroll_uses_character_offsets(settings):
    app = VelocicyberApp(settings, demo=True)
    async with app.run_test(size=(80, 24)) as pilot:
        app.show_conversation()
        log = app.query_one("#log", TranscriptLog)
        log.clear()
        log.write("界 α " + "abcdefghij" * 14, width=200)
        await pilot.pause()
        log.scroll_to(x=5, y=0, animate=False, immediate=True)
        await pilot.pause()
        x, y = log.content_region.offset
        await drag(pilot, (x, y), (x + 5, y))
        await pilot.press("ctrl+c")
        assert app.clipboard == "abcde"
