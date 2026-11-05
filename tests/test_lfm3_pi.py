"""Offline LFM3 installation/routing checks; no real model weights or inference."""

import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.widgets import Static

from scripts.install import pi_conversion as conversion
from velocicyber.config import Settings
from velocicyber.models.pi_profiles import LFM3_PI_REPO, PI_PROFILES
from velocicyber.models.startup import model_details
from velocicyber.providers.lambda_cloud import GPUManager
from velocicyber.providers.pi import PiModelManager
from velocicyber.state import GPUStatus, State
from velocicyber.ui.app import VelocicyberApp

ROOT = Path(__file__).resolve().parents[1]


def gguf(architecture='lfm3', kind=7):
    def string(value):
        data = value.encode()
        return struct.pack('<Q', len(data)) + data
    return (b'GGUF' + struct.pack('<IQQ', 3, 1, 2)
            + string('general.architecture') + struct.pack('<I', 8) + string(architecture)
            + string('general.file_type') + struct.pack('<II', 4, kind))


def test_profile_is_cpu_only_and_storage_details_are_model_specific(settings):
    configured = Settings.load(ROOT)
    assert LFM3_PI_REPO in configured.pi_model_choices
    assert LFM3_PI_REPO not in configured.gpu_model_choices
    details = model_details(LFM3_PI_REPO)
    for text in ('3.46 GB', '1.8–2.0 GB', '12 GiB', '4 GiB', '16K', 'validation pending'):
        assert text in details
    assert '11.4 GiB weights' not in details
    manager = GPUManager(settings, State())
    with pytest.raises(ValueError, match='local CPU'):
        manager.inspect_model(LFM3_PI_REPO)
    assert manager._client is None


def test_pinned_source_has_only_verified_data_and_compatible_runtime():
    cfg = json.loads((ROOT / PI_PROFILES[LFM3_PI_REPO].lock_file).read_text())
    assert cfg['repo'] == LFM3_PI_REPO and cfg['runtime']['tag'] == 'b12124'
    assert cfg['conversion']['outtype'] == 'q8_0'
    assert cfg['context_window'] == 16384 and cfg['enable_thinking']
    files = cfg['source_files']
    assert all(not f['filename'].endswith('.py') and len(f['sha256']) == 64 for f in files)
    assert sum(f['weight_bytes'] for f in files) < 3_480_000_000
    assert next(f for f in files if f['filename'] == 'model.safetensors')['weight_bytes'] == 3457635448
    requirements = ROOT / 'config/locks/pi-conversion.requirements.txt'
    assert hashlib.sha256(requirements.read_bytes()).hexdigest() == cfg['conversion']['requirements_sha256']


async def test_menu_uses_pi_without_changing_gpu_status(settings):
    settings.options.update(PI_LOCAL_MODEL='', PI_LOCAL_MODEL_OPTIONS=LFM3_PI_REPO)
    app = VelocicyberApp(settings, demo=True)
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.press('backslash', *'local', 'enter', *'lfm3', 'enter')
        assert app.screen.id == 'local-model-choice'
        assert app.screen.heading.count('No GPUs Needed!') == 1
        await pilot.press(*'more', 'enter')
        assert 'Q8_0' in app.screen.details and '12 GiB' in app.screen.details
        await pilot.press(*'confirm', 'enter')
        await app.workers.wait_for_complete()
        assert app.state.active_provider == 'pi' and app.state.pi_ready
        assert app.state.pi_context_window == 16384
        assert app.state.gpu_status == GPUStatus.STOPPED and not app.state.model_ready
        assert app.gpu._client is None
        assert 'GPUs: OFF' in str(app.query_one('#dashboard', Static).render())


@pytest.fixture
def converted(settings, monkeypatch):
    cfg = json.loads((ROOT / PI_PROFILES[LFM3_PI_REPO].lock_file).read_text())
    settings.options['PI_LOCAL_MODEL'] = LFM3_PI_REPO
    manager = PiModelManager(settings, State())
    (settings.root / PI_PROFILES[LFM3_PI_REPO].lock_file).write_text(json.dumps(cfg))
    events = []
    calls = []
    monkeypatch.setattr(conversion.shutil, 'disk_usage', lambda p: SimpleNamespace(free=20 * 1024**3))
    monkeypatch.setattr('velocicyber.providers.pi.available_memory', lambda: 8 * 1024**3)

    def runtime(pin, destination):
        binary = destination / 'runtime/llama-server'
        binary.parent.mkdir(exist_ok=True)
        binary.write_bytes(b'fixture binary')
        binary.chmod(0o700)
        return binary, {'runtime/llama-server': conversion.file_hash(binary)}

    def download(pin, stage, report):
        calls.append(pin['filename'])
        cache = stage / 'downloads'
        cache.mkdir(exist_ok=True)
        path = cache / pin['filename']
        path.write_text(json.dumps({'fixture':True}) if path.suffix == '.json' else 'fixture model data')
        return path

    def prepare(cfg, stage, run):
        source = stage / 'source'
        source.mkdir(exist_ok=True)
        (source / 'LICENSE').write_text('Fixture runtime license')
        return Path('/fixture/python'), source

    def convert(cfg, model_dir, output, python, checkout, run):
        assert (model_dir / 'model.safetensors').is_file()
        output.write_bytes(gguf())
        return conversion.file_hash(output)

    monkeypatch.setattr(conversion, 'install_runtime', runtime)
    monkeypatch.setattr(conversion, 'download', download)
    monkeypatch.setattr(conversion, 'prepare_converter', prepare)
    monkeypatch.setattr(conversion, 'convert', convert)
    return manager, cfg, calls, lambda *args, **kw: events.append((args, kw))


def test_success_keeps_receipt_runtime_license_and_cleans_temporary_source(converted):
    manager, cfg, calls, notify = converted
    result = conversion.install_converted(cfg, manager.folder, notify)
    assert calls == [f['filename'] for f in cfg['source_files']]
    assert result['converted_weight_bytes'] == len(gguf())
    assert Path(result['model_path']).read_bytes() == gguf()
    assert (manager.folder / 'LICENSE.model').exists()
    assert (manager.folder / 'LICENSE.llama.cpp').exists()
    assert not (manager.folder / 'conversion-work').exists()
    assert conversion.valid_receipt(cfg, manager.folder, result)
    (manager.folder / 'installed.json').write_text(json.dumps(result))
    assert manager.installed()['model_sha256'] == result['model_sha256']
    assert conversion.install_converted(cfg, manager.folder, notify) == result
    assert len(calls) == len(cfg['source_files'])  # No second download or conversion.


def test_failed_conversion_retains_verified_inputs_for_retry(converted, monkeypatch):
    manager, cfg, _, notify = converted
    def fail(*a):
        raise RuntimeError('Fixture conversion failed')
    monkeypatch.setattr(conversion, 'convert', fail)
    with pytest.raises(RuntimeError, match='conversion failed'):
        conversion.install_converted(cfg, manager.folder, notify)
    assert (manager.folder / 'conversion-work/model/model.safetensors').is_file()
    assert not (manager.folder / 'conversion.json').exists()
    assert not (manager.folder / cfg['filename']).exists()


@pytest.mark.parametrize('target', ['model_path', 'binary_path'])
def test_changed_artifacts_are_not_reused(converted, target):
    manager, cfg, calls, notify = converted
    result = conversion.install_converted(cfg, manager.folder, notify)
    Path(result[target]).write_bytes(b'changed')
    assert not conversion.valid_receipt(cfg, manager.folder, result)
    with pytest.raises(ValueError, match='installation changed'):
        conversion.install_converted(cfg, manager.folder, notify)
    assert len(calls) == len(cfg['source_files'])


@pytest.mark.parametrize('resource', ['disk', 'memory'])
def test_insufficient_resources_stop_before_any_download(converted, monkeypatch, resource):
    manager, cfg, calls, notify = converted
    if resource == 'disk':
        monkeypatch.setattr(conversion.shutil, 'disk_usage', lambda p: SimpleNamespace(free=1))
    else:
        monkeypatch.setattr('velocicyber.providers.pi.available_memory', lambda: 1)
    with pytest.raises(RuntimeError, match='GiB'):
        conversion.install_converted(cfg, manager.folder, notify)
    assert not calls


def test_symlinked_work_folder_cannot_be_deleted_or_used(converted, tmp_path):
    manager, cfg, calls, notify = converted
    outside = tmp_path / 'outside'
    outside.mkdir()
    marker = outside / 'keep'
    marker.write_text('preserve')
    (manager.folder / 'conversion-work').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        conversion.install_converted(cfg, manager.folder, notify)
    assert marker.read_text() == 'preserve' and not calls


@pytest.mark.parametrize('architecture,kind,accepted', [('lfm3', 7, True), ('qwen2', 7, False), ('lfm3', 1, False)])
def test_converter_uses_offline_cpu_configuration_and_checks_output(settings, architecture, kind, accepted):
    cfg = json.loads((ROOT / PI_PROFILES[LFM3_PI_REPO].lock_file).read_text())
    model_dir = settings.root / 'model'
    model_dir.mkdir()
    output = settings.root / 'generated.gguf'
    def run(args, **kwargs):
        assert args[args.index('--outtype') + 1] == 'q8_0'
        assert '--use-temp-file' in args
        assert kwargs['env']['HF_HUB_OFFLINE'] == '1'
        assert kwargs['env']['TRANSFORMERS_OFFLINE'] == '1'
        assert not any('API' in k or 'TOKEN' in k for k in kwargs['env'] if k != 'TOKENIZERS_PARALLELISM')
        output.write_bytes(gguf(architecture, kind))
    if accepted:
        assert conversion.convert(cfg, model_dir, output, Path('/python'), settings.root, run) == conversion.file_hash(output)
    else:
        with pytest.raises(ValueError):
            conversion.convert(cfg, model_dir, output, Path('/python'), settings.root, run)


@pytest.mark.parametrize("stage,label", [("converting", "Converting"),
                                          ("preparing_converter", "Preparing converter"),
                                          ("failed", "Setup stopped")])
def test_background_conversion_stage_is_visible(settings, stage, label):
    settings.options["PI_LOCAL_MODEL"] = LFM3_PI_REPO
    manager = PiModelManager(settings, State())
    (manager.folder / "build.json").write_text(json.dumps({"stage": stage}))
    assert label in manager.install_status()


def test_internal_runtime_library_links_are_allowed_but_cannot_escape(converted, monkeypatch, tmp_path):
    manager, cfg, _, notify = converted
    original = conversion.install_runtime
    def runtime(pin, folder):
        binary, files = original(pin, folder)
        library = binary.parent / "library.so.1"
        library.write_bytes(b"fixture library")
        link = binary.parent / "library.so"
        link.symlink_to(library.name)
        for path in (library, link):
            files[str(path.relative_to(folder))] = conversion.file_hash(path)
        return binary, files
    monkeypatch.setattr(conversion, "install_runtime", runtime)
    result = conversion.install_converted(cfg, manager.folder, notify)
    assert conversion.valid_receipt(cfg, manager.folder, result)
    outside = tmp_path / "outside-library"
    outside.write_bytes(b"fixture library")
    link = manager.folder / "runtime/library.so"
    link.unlink()
    link.symlink_to(outside)
    assert not conversion.valid_receipt(cfg, manager.folder, result)
