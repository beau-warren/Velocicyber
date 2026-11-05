import os
import stat

import pytest
from textual.widgets import Static, TextArea

from velocicyber.sentinel.scope import load_scope, read_scope, save_scope
from velocicyber.ui.app import VelocicyberApp


@pytest.fixture
def scope_file(settings):
    folder = settings.root / 'SCOPE'
    folder.mkdir()
    path = folder / 'SCOPE.MD'
    path.write_text('Only operate inside the workspace.\n')
    return path


def test_save_creates_private_scope_and_loader_sees_new_rules(settings):
    assert read_scope(settings.root) is None
    text = 'Allowed: workspace/docs only.\nDo not delete files.\n'
    save_scope(settings.root, text, expected=None)
    assert load_scope(settings.root)[0] == text
    assert stat.S_IMODE((settings.root / 'SCOPE/SCOPE.MD').stat().st_mode) == 0o600
    assert not list((settings.root / 'SCOPE').glob('.scope-*.tmp'))


@pytest.mark.parametrize('text', ['', ' \n ', 'EXAMPLE ONLY: replace', 'é' * 3001])
def test_invalid_edits_preserve_original(scope_file, settings, text):
    original = scope_file.read_bytes()
    with pytest.raises(ValueError):
        save_scope(settings.root, text, expected=original)
    assert scope_file.read_bytes() == original


def test_concurrent_edit_is_not_overwritten(scope_file, settings):
    original = read_scope(settings.root)
    scope_file.write_text('Changed in another editor.')
    with pytest.raises(ValueError, match='changed outside'):
        save_scope(settings.root, 'Stale replacement.', expected=original)
    assert scope_file.read_text() == 'Changed in another editor.'


@pytest.mark.parametrize('target', ['file', 'folder', 'fifo', 'directory'])
def test_rejects_symlinks_and_nonregular_files(settings, target):
    root = settings.root
    folder = root / 'SCOPE'
    outside = root / 'other'
    outside.mkdir()
    marker = outside / 'SCOPE.MD'
    marker.write_text('Unchanged.')
    if target == 'folder':
        folder.symlink_to(outside, target_is_directory=True)
    else:
        folder.mkdir()
        path = folder / 'SCOPE.MD'
        if target == 'file':
            path.symlink_to(marker)
        elif target == 'fifo':
            os.mkfifo(path)
        else:
            path.mkdir()
    with pytest.raises(ValueError, match='regular file'):
        read_scope(root)
    with pytest.raises(ValueError, match='regular file'):
        save_scope(root, 'Replacement.', expected=None)
    assert marker.read_text() == 'Unchanged.'


def test_failed_replace_retains_scope_and_cleans_tempfile(scope_file, settings, monkeypatch):
    original = scope_file.read_bytes()
    def fail(*args, **kwargs):
        raise OSError('Fixture disk failure')
    monkeypatch.setattr(os, 'replace', fail)
    with pytest.raises(OSError):
        save_scope(settings.root, 'New rules.', expected=original)
    assert scope_file.read_bytes() == original
    assert not list(scope_file.parent.glob('.scope-*.tmp'))


async def open_editor(app, pilot):
    app.command('sentinel')
    await pilot.pause()
    assert ('scope', '/SCOPE — Edit scope restrictions') in app.screen.options
    await pilot.press(*'scope', 'enter')
    await pilot.pause()
    assert app.screen.id == 'scope-editor'


@pytest.mark.parametrize('size', [(80, 24), (55, 20)])
async def test_menu_editor_saves_without_logging_contents(scope_file, settings, monkeypatch, size):
    app = VelocicyberApp(settings, demo=True)
    notices = []
    async with app.run_test(size=size) as pilot:
        monkeypatch.setattr(app, 'say', notices.append)
        await open_editor(app, pilot)
        field = app.screen.query_one(TextArea)
        assert field.text == scope_file.read_text()
        field.load_text('Confidential authorization: workspace only.\n')
        await pilot.press('ctrl+s')
        await pilot.pause()
        assert app.screen.id != 'scope-editor'
        assert load_scope(settings.root)[0] == 'Confidential authorization: workspace only.\n'
        assert notices == ['SCOPE.MD saved. Sentinel will use it for the next task.']
        assert not app.state.sentinel_enabled
        assert not (settings.root / 'velocicyber_sentinel-data').exists()
        assert not (settings.root / 'velocicyber_finetuning-data').exists()


@pytest.mark.parametrize('cancel', ['escape', 'button'])
async def test_cancel_does_not_create_or_enable_scope(settings, cancel):
    app = VelocicyberApp(settings, demo=True)
    async with app.run_test() as pilot:
        await open_editor(app, pilot)
        app.screen.query_one(TextArea).load_text('Not saved.')
        if cancel == 'button':
            await pilot.click('#scope-cancel')
        else:
            await pilot.press('escape')
        await pilot.pause()
        assert read_scope(settings.root) is None
        assert not app.state.sentinel_enabled


async def test_invalid_save_keeps_editor_and_input(scope_file, settings):
    app = VelocicyberApp(settings, demo=True)
    original = scope_file.read_bytes()
    async with app.run_test() as pilot:
        await open_editor(app, pilot)
        app.screen.query_one(TextArea).load_text('')
        await pilot.click('#scope-save')
        assert app.screen.id == 'scope-editor'
        assert '6,000' in str(app.screen.query_one('#scope-error', Static).render())
        assert scope_file.read_bytes() == original
        app.screen.query_one(TextArea).load_text('Corrected scope.')
        await pilot.press('ctrl+s')
        await pilot.pause()
        assert scope_file.read_text() == 'Corrected scope.'


@pytest.mark.parametrize('flag', ['busy', 'pi_busy', 'sentinel_busy', 'gpu_busy',
                                  'restoring_gpu', 'data_busy', 'session_ending'])
async def test_busy_guard_on_open_and_save(scope_file, settings, flag):
    app = VelocicyberApp(settings, demo=True)
    original = scope_file.read_bytes()
    async with app.run_test() as pilot:
        setattr(app, flag, True)
        app.set_sentinel('scope')
        await pilot.pause()
        assert app.screen.id != 'scope-editor'
        setattr(app, flag, False)
        app.set_sentinel('scope')
        await pilot.pause()
        app.screen.query_one(TextArea).load_text('New scope.')
        setattr(app, flag, True)
        await pilot.press('ctrl+s')
        assert app.screen.id == 'scope-editor'
        assert 'Wait' in str(app.screen.query_one('#scope-error', Static).render())
        assert scope_file.read_bytes() == original
        setattr(app, flag, False)
        await pilot.press('escape')
