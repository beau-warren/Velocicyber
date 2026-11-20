import json
import shutil
import tomllib
from pathlib import Path

import httpx
import pytest
from packaging.requirements import Requirement

from velocicyber.engine.installation import engine_lease
from velocicyber.updates import UpdateChecker
from velocicyber.updates.manager import apply_updates, interactive_updates
from velocicyber.updates.source import REMOTE, REPOSITORY, SourceChecker, git
from velocicyber.updates.updater import BACKUP, PENDING, SourceUpdater, manifest, source_path

OLD, NEW = "a" * 40, "b" * 40


def receipt(settings, commit=OLD):
    (settings.runtime / "source-install.json").write_text(json.dumps({"repository": REPOSITORY, "commit": commit}))


@pytest.mark.parametrize("remote_status,expected", [
    ("ahead", "UPDATE AVAILABLE"), ("behind", "LOCAL SOURCE CHANGES"),
    ("diverged", "LOCAL SOURCE CHANGES"), ("identical", "UP TO DATE"),
])
async def test_source_checks_once_and_private_token_is_not_cached(settings, remote_status, expected):
    receipt(settings)
    settings.secrets["GITHUB_API"] = "fixture_github_key"
    requests = []

    def respond(request):
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer fixture_github_key"
        assert request.url.host == "api.github.com"
        return httpx.Response(200, json={"sha": NEW} if request.url.path.endswith("/main") else {"status": remote_status})

    checker = SourceChecker(settings)
    await checker.check(transport=httpx.MockTransport(respond))
    await checker.check(transport=httpx.MockTransport(respond))
    assert len(requests) == 2
    assert checker.indicator()[0] == expected
    assert "fixture_github_key" not in checker.path.read_text()
    assert checker.path.stat().st_mode & 0o777 == 0o600
    assert SourceChecker(settings).indicator()[0] == expected


async def test_identical_source_needs_only_one_request_and_new_install_invalidates_cache(settings):
    receipt(settings)
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={"sha": OLD})

    checker = SourceChecker(settings)
    await checker.check(transport=httpx.MockTransport(respond))
    assert checker.indicator()[0] == "UP TO DATE" and len(calls) == 1
    receipt(settings, NEW)
    assert SourceChecker(settings).result == {}


async def test_failed_private_access_is_explicit_and_preserves_known_update(settings):
    receipt(settings)
    checker = SourceChecker(settings)
    await checker.check(transport=httpx.MockTransport(lambda _: httpx.Response(404)))
    assert checker.indicator()[0] == "UPDATE CHECK FAILED"
    assert "GITHUB_API" in checker.details()
    checker = SourceChecker(settings)
    await checker.check(transport=httpx.MockTransport(lambda r: httpx.Response(
        200, json={"sha": NEW} if r.url.path.endswith("/main") else {"status": "ahead"})))
    checker = SourceChecker(settings)
    await checker.check(transport=httpx.MockTransport(lambda _: httpx.Response(503)))
    assert checker.indicator()[0] == "UPDATE AVAILABLE" and "failed" in checker.details()


async def test_unknown_install_never_contacts_github(settings):
    checker = SourceChecker(settings)
    await checker.check(transport=httpx.MockTransport(lambda _: pytest.fail("Unexpected network")))
    assert checker.indicator()[0] == "VERSION UNKNOWN"


@pytest.mark.parametrize("source,engine,expected,components", [
    ("UP TO DATE", "UP TO DATE", "UP TO DATE", set()),
    ("UPDATE AVAILABLE", "UP TO DATE", "UPDATE AVAILABLE", {"harness"}),
    ("UP TO DATE", "UPDATE AVAILABLE", "UPDATE AVAILABLE", {"engine"}),
    ("UPDATE AVAILABLE", "UPDATE AVAILABLE", "UPDATE AVAILABLE", {"harness", "engine"}),
    ("UPDATE CHECK FAILED", "UPDATE AVAILABLE", "UPDATE AVAILABLE", {"engine"}),
    ("UP TO DATE", "UPDATE CHECK FAILED", "UPDATE CHECK FAILED", set()),
    ("CHECKING UPDATES", "UP TO DATE", "CHECKING UPDATES", set()),
])
def test_combined_indicator_and_component_selection(settings, monkeypatch, source, engine, expected, components):
    checker = UpdateChecker(settings)
    monkeypatch.setattr(checker.source, "indicator", lambda: (source, "yellow"))
    monkeypatch.setattr(checker.engine, "indicator", lambda: (engine, "yellow"))
    checker.source.installed = OLD
    checker.source.result = {"latest": NEW}
    checker.engine.result = {"latest": "rust-v0.0.59"}
    assert checker.indicator()[0] == expected
    assert set(checker.plan()) == components


@pytest.fixture
def source_update(settings, tmp_path, monkeypatch):
    baseline, candidate = tmp_path / "baseline-fixture", tmp_path / "candidate-fixture"
    for folder, content in ((baseline, "old"), (candidate, "new")):
        folder.mkdir()
        (folder / "app.py").write_text(content)
        (folder / "requirements.lock").write_text("")
        (folder / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (baseline / "removed.py").write_text("old removed file")
    (candidate / "added.py").write_text("new added file")
    for file in baseline.iterdir():
        shutil.copy2(file, settings.root / file.name)
    receipt(settings)
    (settings.root / ".env").write_text("LAMBDA_API=private-fixture")
    (settings.root / "Modules.txt").write_text("LAMBDA_SSH_KEY_NAME=fixture")
    (settings.root / "ssh-keys").mkdir()
    (settings.root / "ssh-keys/key.pem").write_text("private-key-fixture")
    (settings.root / "workspace").mkdir()
    (settings.root / "workspace/keep.txt").write_text("user work")
    (settings.runtime / "conversation.json").write_text("private conversation")
    updater = SourceUpdater(settings, lambda _: None)

    def download(commit, destination):
        shutil.copytree(baseline if commit == OLD else candidate, destination)
        return manifest(destination)

    monkeypatch.setattr(updater, "download", download)
    monkeypatch.setattr(updater, "validate", lambda *_: False)
    return updater, {"installed": OLD, "target": NEW}, candidate


def assert_private_preserved(root):
    assert (root / ".env").read_text() == "LAMBDA_API=private-fixture"
    assert (root / "Modules.txt").read_text() == "LAMBDA_SSH_KEY_NAME=fixture"
    assert (root / "ssh-keys/key.pem").read_text() == "private-key-fixture"
    assert (root / "workspace/keep.txt").read_text() == "user work"
    assert (root / ".runtime/conversation.json").read_text() == "private conversation"


def test_archive_update_and_rollback_preserve_local_data(source_update):
    updater, plan, _ = source_update
    updater.run(plan, approved=True)
    assert (updater.root / "app.py").read_text() == "new"
    assert not (updater.root / "removed.py").exists()
    assert (updater.root / "added.py").is_file()
    assert not (updater.runtime / PENDING).exists()
    assert (updater.runtime / BACKUP).is_file()
    assert_private_preserved(updater.root)
    updater.run(approved=True, rollback=True)
    assert (updater.root / "app.py").read_text() == "old"
    assert (updater.root / "removed.py").is_file()
    assert not (updater.root / "added.py").exists()
    assert_private_preserved(updater.root)


@pytest.mark.parametrize("failure", ["approval", "changed", "collision", "tests", "active"])
def test_source_refuses_unsafe_activation(source_update, monkeypatch, failure):
    updater, plan, _ = source_update
    if failure == "changed":
        (updater.root / "app.py").write_text("local edits")
    elif failure == "collision":
        (updater.root / "added.py").write_text("user file")
    elif failure == "tests":
        def fail(*args):
            raise RuntimeError("Candidate test failed")
        monkeypatch.setattr(updater, "validate", fail)
    before = (updater.root / "app.py").read_text()
    if failure == "active":
        with engine_lease(updater.runtime), pytest.raises(RuntimeError):
            updater.run(plan, approved=True)
    else:
        with pytest.raises((RuntimeError, PermissionError)):
            updater.run(plan, approved=failure != "approval")
    assert (updater.root / "app.py").read_text() == before
    assert not (updater.runtime / BACKUP).exists()
    assert_private_preserved(updater.root)


@pytest.mark.parametrize("name", [".env", "Modules.txt", "../escape", ".runtime/model", "workspace/task.py", "ssh-keys/key.pem", "/tmp/file"])
def test_source_rejects_private_paths(name):
    with pytest.raises(ValueError):
        source_path(name)


@pytest.mark.parametrize("install_failure", [False, True])
def test_dependency_environment_has_rollback(source_update, monkeypatch, install_failure):
    updater, plan, _ = source_update
    (updater.root / ".venv").mkdir()
    (updater.root / ".venv/original").write_text("old environment")

    def validate(source, run):
        (run / "venv").mkdir()
        return True

    def command(*args, **kwargs):
        if install_failure:
            raise RuntimeError("Dependency activation failed")

    monkeypatch.setattr(updater, "validate", validate)
    monkeypatch.setattr(updater, "command", command)
    if install_failure:
        with pytest.raises(RuntimeError, match="activation failed"):
            updater.run(plan, approved=True)
    else:
        updater.run(plan, approved=True)
        assert (updater.root / ".venv").is_symlink()
        updater.run(approved=True, rollback=True)
    assert (updater.root / ".venv/original").read_text() == "old environment"
    assert (updater.root / "app.py").read_text() == "old"
    assert not (updater.runtime / PENDING).exists()


def test_stage_only_leaves_installation_untouched(source_update):
    updater, plan, _ = source_update
    updater.run(plan, approved=True, stage_only=True)
    assert (updater.root / "app.py").read_text() == "old"
    assert not (updater.runtime / BACKUP).exists()
    assert_private_preserved(updater.root)


@pytest.mark.parametrize("components", [{"engine"}, {"harness"}, {"engine", "harness"}])
def test_manual_update_dispatch_is_exact(settings, monkeypatch, components):
    from velocicyber.updates import manager
    calls = []
    monkeypatch.setattr(manager.EngineUpdater, "run", lambda self, **kw: calls.append(("engine", kw)))
    monkeypatch.setattr(manager.SourceUpdater, "run", lambda self, plan, **kw: calls.append(("harness", kw)))
    plan = {key: {"installed": OLD, "target": NEW if key == "harness" else "rust-v0.0.59"} for key in components}
    with pytest.raises(PermissionError):
        apply_updates(settings, plan)
    interactive_updates(settings, plan=plan, input_fn=lambda _: "")
    assert calls == []
    interactive_updates(settings, plan=plan, input_fn=lambda _: "UPDATE")
    assert [key for key, _ in calls] == [key for key in ("engine", "harness") if key in components]
    assert all(kwargs["approved"] for _, kwargs in calls)


def test_git_update_and_rollback(source_update, tmp_path, monkeypatch):
    updater, _, candidate = source_update
    root = updater.root
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    (root / ".gitignore").write_text(".env\nModules.txt\n.runtime/\nssh-keys/\nworkspace/\nconfig/\nbaseline-fixture/\ncandidate-fixture/\n")
    git(root, "add", "app.py", "removed.py", "requirements.lock", "pyproject.toml", ".gitignore")
    git(root, "commit", "-m", "old")
    old = git(root, "rev-parse", "HEAD")
    remote = tmp_path / "remote"
    git(root, "clone", str(root), str(remote))
    git(remote, "config", "user.name", "Fixture")
    git(remote, "config", "user.email", "fixture@example.invalid")
    for file in candidate.iterdir():
        shutil.copy2(file, remote / file.name)
    (remote / "removed.py").unlink()
    git(remote, "add", "-A")
    git(remote, "commit", "-m", "new")
    target = git(remote, "rev-parse", "HEAD")
    # The fake remote lives beside the install so it never counts as local edits.
    moved = tmp_path.parent / (tmp_path.name + "-remote")
    remote.rename(moved)
    git(root, "remote", "add", "origin", REMOTE)
    shutil.copy2(root / ".gitignore", candidate / ".gitignore")

    def fetch(commit, run):
        git(root, "fetch", str(moved), commit)

    monkeypatch.setattr(updater, "fetch_git", fetch)
    updater.run({"installed": old, "target": target}, approved=True)
    assert git(root, "rev-parse", "HEAD") == target
    assert_private_preserved(root)
    updater.run(approved=True, rollback=True)
    assert git(root, "rev-parse", "HEAD") == old
    assert (root / "app.py").read_text() == "old"
    assert_private_preserved(root)


@pytest.mark.parametrize("report", ["<testsuite/>", "<testsuite><testcase><skipped/></testcase></testsuite>",
                                  "<testsuite><testcase><failure/></testcase></testsuite>"])
def test_candidate_validation_rejects_empty_skipped_or_failed_suite(source_update, monkeypatch, report):
    updater, _, source = source_update
    monkeypatch.undo()
    run = updater.runtime / "validation-fixture"
    run.mkdir()

    def command(*args, **kwargs):
        (run / "tests.xml").write_text(report)

    monkeypatch.setattr(updater, "command", command)
    with pytest.raises(RuntimeError, match="test suite"):
        updater.validate(source, run)


def test_source_download_strips_auth_on_redirect_and_unpacks_bounded_archive(settings, tmp_path):
    import io
    import tarfile
    from pathlib import Path
    shutil.copy2(Path(__file__).resolve().parents[1] / "bootstrap.py", settings.root / "bootstrap.py")
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as tar:
        for name in ("install.sh", "pyproject.toml", "velocicyber/cli.py"):
            info = tarfile.TarInfo("source/" + name)
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
    calls = []
    settings.secrets["GITHUB_API"] = "fixture_private_token"

    def respond(request):
        calls.append(request)
        if request.url.host == "api.github.com":
            assert request.headers["Authorization"] == "Bearer fixture_private_token"
            return httpx.Response(302, headers={"Location": "https://codeload.github.com/fixture"})
        assert "Authorization" not in request.headers
        return httpx.Response(200, content=payload.getvalue())

    files = SourceUpdater(settings).download(NEW, tmp_path / "download", transport=httpx.MockTransport(respond))
    assert set(files) == {"install.sh", "pyproject.toml", "velocicyber/cli.py"}
    assert len(calls) == 2
    with pytest.raises(ValueError, match="redirect"):
        SourceUpdater(settings).download(NEW, tmp_path / "bad-download", transport=httpx.MockTransport(
            lambda _: httpx.Response(302, headers={"Location": "https://untrusted.example/source"})))


def test_explicit_rollback_disables_automatic_reapplication(source_update):
    updater, plan, _ = source_update
    updater.run(plan, approved=True)
    updater.settings.options["AUTO_UPDATE_HARNESS"] = "ON"
    updater.run(approved=True, rollback=True)
    assert updater.settings.options["AUTO_UPDATE_HARNESS"] == "OFF"
    assert "AUTO_UPDATE_HARNESS=OFF" in (updater.root / "Modules.txt").read_text()
    assert (updater.root / "app.py").read_text() == "old"


def test_legacy_updater_receives_build_requirements_in_dependency_lock():
    # Older updaters install this lock, then activate with --no-build-isolation.
    # PEP 517's temporary build environment does not supply that backend later.
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    locked = {req.name: req for line in (root / "requirements.lock").read_text().splitlines()
              if line.strip() and not line.startswith("#") for req in [Requirement(line)]}
    for value in project["build-system"]["requires"]:
        requirement = Requirement(value)
        pin = locked[requirement.name]
        specs = list(pin.specifier)
        assert len(specs) == 1 and specs[0].operator == "=="
        assert specs[0].version in requirement.specifier


@pytest.mark.parametrize("build_failure", [False, True])
def test_dependency_build_is_checked_before_tests_or_activation(source_update, monkeypatch, build_failure):
    updater, _, source = source_update
    (source / "requirements.lock").write_text("build-fixture==1\n")
    run = updater.runtime / "build-validation"
    run.mkdir()
    calls = []

    def command(arguments, folder, log, **kwargs):
        calls.append(arguments)
        if "pip" in arguments and "-e" in arguments and build_failure:
            raise RuntimeError("Build backend unavailable")
        if "pytest" in arguments:
            (run / "tests.xml").write_text("<testsuite><testcase/></testsuite>")

    monkeypatch.setattr(updater, "command", command)
    if build_failure:
        with pytest.raises(RuntimeError, match="Build backend unavailable"):
            SourceUpdater.validate(updater, source, run)
        assert not any("pytest" in args for args in calls)
    else:
        assert SourceUpdater.validate(updater, source, run) is True
        assert "pytest" in calls[-1]
    editable = next(args for args in calls if "-e" in args)
    assert "--no-deps" in editable and "--no-build-isolation" in editable
    assert editable[-2:] == ["-e", "."]
    assert (updater.root / "app.py").read_text() == "old"
    assert not (updater.runtime / PENDING).exists()
    assert_private_preserved(updater.root)


def test_managed_environment_symlink_does_not_count_as_local_source_changes(settings):
    root = settings.root
    git(root, "init", "-b", "main")
    shutil.copy2(Path(__file__).resolve().parents[1] / ".gitignore", root / ".gitignore")
    environment = settings.runtime / "validated-venv"
    environment.mkdir()
    (root / ".venv").symlink_to(environment, target_is_directory=True)
    assert git(root, "status", "--porcelain", "--untracked-files=all", "--", ".venv") == ""
