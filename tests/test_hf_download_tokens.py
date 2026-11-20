"""Offline checks that download credentials cannot inherit dataset write access."""

import json
import stat
import time
from types import SimpleNamespace

import httpx
import pytest

from velocicyber import onboarding
from velocicyber.config import Settings, read_assignments
from velocicyber.datasets.pipeline import Scrubber
from velocicyber.datasets.upload import Uploader
from velocicyber.providers import huggingface as auth
from velocicyber.providers.lambda_cloud import GPUManager, LaunchPlan, ModelSpec
from velocicyber.sentinel.data import SentinelStore
from velocicyber.state import GPUStatus, State


@pytest.fixture(autouse=True)
def isolated_credentials(monkeypatch):
    for key, _ in onboarding.CREDENTIALS:
        monkeypatch.delenv(key, raising=False)


def identity(role="read", **access):
    return {"auth": {"accessToken": {"role": role, **access}}}


def fine_grained(permissions, **extra):
    return identity("fineGrained", fineGrained={
        "scoped": [{"entity": {"type": "model"}, "permissions": permissions}], **extra,
    })


@pytest.mark.parametrize("value,accepted", [
    (identity(), True), (fine_grained(["repo.read", "repo.content.read"]), True),
    (fine_grained([], canReadGatedRepos=True), True),
    (identity("write"), False), (identity("god"), False), (identity("unknown"), False),
    (fine_grained(["repo.write"]), False), (fine_grained(["repo.read", "repo.content.write"]), False),
    (fine_grained(["repo.future-permission"]), False),
    (fine_grained(["repo.read"], global_scope=True), False),
    (fine_grained(["repo.read"], **{"global": ["discussion.write"]}), False),
    (fine_grained(["repo.read"], **{"global": ["post.write"]}), False),
    (fine_grained(["repo.read"], **{"global": None}), False),
    (fine_grained("repo.read"), False), (fine_grained([{}]), False),
    (fine_grained([], canReadGatedRepos="true"), False),
    (identity("fineGrained"), False), (identity("fineGrained", fineGrained={}), False),
    ({}, False), (None, False), ([], False),
])
def test_only_explicit_read_permissions_are_accepted(value, accepted):
    assert auth.read_only_identity(value) is accepted


def test_verification_uses_only_the_download_token_over_https():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=identity())

    auth.validate_download_token("read-fixture", upload_token="upload-fixture",
                                 transport=httpx.MockTransport(handler))
    assert len(seen) == 1
    assert str(seen[0].url) == auth.WHOAMI_URL
    assert seen[0].headers["Authorization"] == "Bearer read-fixture"
    assert "upload-fixture" not in str(seen[0].headers)


@pytest.mark.parametrize("status", [301, 302, 401, 403, 429, 500])
def test_failed_or_redirected_verification_never_discloses_the_token(status):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(status, text="read-private-fixture rejected",
                              headers={"location": "http://untrusted.example/verify"})

    with pytest.raises(auth.DownloadTokenError, match="Could not verify") as caught:
        auth.validate_download_token("read-private-fixture", transport=httpx.MockTransport(handler))
    assert len(seen) == 1 and "read-private-fixture" not in str(caught.value)
    assert caught.value.__suppress_context__


def test_network_failure_does_not_leak_credentials():
    def fail(_):
        raise httpx.ReadTimeout("read-private-fixture")

    with pytest.raises(auth.DownloadTokenError) as caught:
        auth.validate_download_token("read-private-fixture", transport=httpx.MockTransport(fail))
    assert "read-private-fixture" not in str(caught.value)


def test_upload_token_reuse_is_blocked_before_network(monkeypatch):
    monkeypatch.setattr("httpx.Client", lambda **kw: pytest.fail("Upload token must not be sent"))
    with pytest.raises(auth.DownloadTokenError, match="separate"):
        auth.validate_download_token("upload-fixture", upload_token="upload-fixture")


def test_public_downloads_never_fall_back_to_upload_or_environment_tokens(settings, monkeypatch):
    settings.secrets["HUGGINGFACE_API"] = "upload-private-fixture"
    monkeypatch.setenv("HF_TOKEN", "ambient-private-fixture")
    monkeypatch.setattr("httpx.Client", lambda **kw: pytest.fail("Anonymous download needs no validation"))
    assert auth.model_download_token(settings) is False


def test_config_accepts_read_token_environment_override_without_env_entry(tmp_path, monkeypatch):
    monkeypatch.setenv(auth.READ_TOKEN_KEY, "read-environment-fixture")
    settings = Settings.load(tmp_path)
    assert settings.secrets[auth.READ_TOKEN_KEY] == "read-environment-fixture"
    assert "read-environment-fixture" not in repr(settings)
    assert "read-environment-fixture" not in Scrubber(settings.secrets.values()).text("read-environment-fixture")


@pytest.mark.parametrize("download_token", ["", "read-fixture"])
def test_model_metadata_uses_only_download_credentials(settings, monkeypatch, download_token):
    settings.secrets.update(HUGGINGFACE_API="upload-private-fixture", HUGGINGFACE_READ_API=download_token)
    validations, tokens = [], []
    monkeypatch.setattr(auth, "validate_download_token", lambda token, **kw: validations.append((token, kw)))

    def api(*, token):
        tokens.append(token)
        return SimpleNamespace(model_info=lambda *a, **kw: SimpleNamespace(sha="revision", siblings=[
            SimpleNamespace(rfilename="model.gguf", size=1024, lfs=SimpleNamespace(sha256="a" * 64)),
        ]))

    monkeypatch.setattr("huggingface_hub.HfApi", api)

    def header(repo, revision, filename, token):
        tokens.append(token)
        return {"general.architecture": "qwen3", "qwen3.context_length": 8192}

    monkeypatch.setattr("velocicyber.providers.gguf.inspect_header", header)
    manager = GPUManager(settings, State())
    assert manager.inspect_model().backend == "llama.cpp"
    assert tokens == [download_token or False] * 2
    assert bool(validations) is bool(download_token)


async def test_changed_download_token_blocks_cached_model_and_gpu_actions(settings, monkeypatch):
    settings.secrets.update(HUGGINGFACE_API="upload-private-fixture", HUGGINGFACE_READ_API="unsafe-fixture")
    verified = []

    def reject(token, **kwargs):
        verified.append(token)
        raise auth.DownloadTokenError("The model-download token must be read-only.")

    monkeypatch.setattr(auth, "validate_download_token", reject)
    monkeypatch.setattr("asyncio.create_subprocess_exec", lambda *a, **kw: pytest.fail("SSH must not start"))
    manager = GPUManager(settings, State())
    manager.spec = ModelSpec(settings.local_model, "revision", 1024, "", "", 24, backend="llama.cpp")
    with pytest.raises(auth.DownloadTokenError):
        manager.inspect_model()
    plan = LaunchPlan("gpu-fixture", "region-fixture", 40, 1, 100, manager.spec, time.time())
    with pytest.raises(auth.DownloadTokenError):
        manager.launch(plan, approved=True)
    assert manager._client is None and not manager.state.launch_uncertain
    manager.state.gpu_status = GPUStatus.RUNNING
    manager.state.total_vram_gib = 40
    with pytest.raises(auth.DownloadTokenError):
        await manager.deploy()
    assert verified == ["unsafe-fixture"] * 3


def test_both_dataset_pipelines_keep_using_the_upload_token(settings, monkeypatch):
    settings.secrets.update(HUGGINGFACE_API="upload-private-fixture", HUGGINGFACE_READ_API="read-private-fixture")
    tokens = []

    def api(*, token):
        tokens.append(token)
        return SimpleNamespace(
            dataset_info=lambda *a, **kw: SimpleNamespace(sha="revision", siblings=[]),
            create_commit=lambda **kw: SimpleNamespace(oid="b" * 40),
        )

    monkeypatch.setattr("huggingface_hub.HfApi", api)
    Uploader(settings).client()
    store = SentinelStore(settings)
    store.append({"scope": "fixture", "verdict": "continue"})
    store.upload(approved=True)  # Fully mocked; no real Hub requests or uploads.
    assert tokens == ["upload-private-fixture"] * 2


def existing_setup(tmp_path):
    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    marker = runtime / onboarding.MARKER
    marker.write_text('{"schema":1,"lambda_ssh_setup":1,"completed_at":"original"}')
    (tmp_path / ".env").write_text("HUGGINGFACE_API=upload-private-fixture\nCUSTOM=keep\n")
    return marker


@pytest.mark.parametrize("answer", ["", "skip for now", "read-private-fixture"])
def test_existing_setup_prompts_only_for_read_token_once(tmp_path, monkeypatch, capsys, answer):
    marker = existing_setup(tmp_path)
    prompts, validations = [], []
    monkeypatch.setattr(onboarding, "validate_download_token", lambda token, **kw: validations.append((token, kw)))

    def secret(prompt):
        prompts.append(prompt)
        return answer

    def no_prompt(_):
        pytest.fail("Existing settings must not be prompted again")

    assert onboarding.run_setup(tmp_path, interactive=True, input_fn=no_prompt, secret_fn=secret)
    assert len(prompts) == 1 and "read-only" in prompts[0]
    values = read_assignments(tmp_path / ".env")
    assert values["HUGGINGFACE_API"] == "upload-private-fixture" and values["CUSTOM"] == "keep"
    assert values.get(auth.READ_TOKEN_KEY, "") == (answer if answer.startswith("read-") else "")
    assert bool(validations) is answer.startswith("read-")
    assert stat.S_IMODE((tmp_path / ".env").stat().st_mode) == 0o600
    assert json.loads(marker.read_text())["hf_download_token_setup"] == 1
    assert json.loads(marker.read_text())["completed_at"] == "original"
    assert not onboarding.run_setup(tmp_path, interactive=True, input_fn=no_prompt, secret_fn=no_prompt)
    output = capsys.readouterr().out + marker.read_text()
    assert "read-private-fixture" not in output and "upload-private-fixture" not in output


def test_headless_launch_defers_download_token_prompt(tmp_path):
    marker = existing_setup(tmp_path)
    assert not onboarding.run_setup(tmp_path, interactive=False)
    assert "hf_download_token_setup" not in json.loads(marker.read_text())


def test_setup_rejects_write_token_and_can_retry_without_displaying_it(tmp_path, monkeypatch, capsys):
    existing_setup(tmp_path)
    seen = []

    def handler(request):
        seen.append(request.headers["Authorization"])
        return httpx.Response(200, json=identity("write" if len(seen) == 1 else "read"))

    monkeypatch.setattr(onboarding, "validate_download_token", lambda token, **kw: auth.validate_download_token(
        token, transport=httpx.MockTransport(handler), **kw,
    ))
    answers = iter(["write-private-fixture", "read-private-fixture"])
    assert onboarding.run_setup(tmp_path, interactive=True, secret_fn=lambda _: next(answers))
    assert read_assignments(tmp_path / ".env")[auth.READ_TOKEN_KEY] == "read-private-fixture"
    assert seen == ["Bearer write-private-fixture", "Bearer read-private-fixture"]
    output = capsys.readouterr().out
    assert "must be read-only" in output
    assert all(value not in output for value in ("write-private-fixture", "read-private-fixture"))


def test_first_run_accepts_distinct_tokens_without_enabling_uploads(tmp_path, monkeypatch, capsys):
    for key, _ in onboarding.CREDENTIALS:
        monkeypatch.delenv(key, raising=False)
    checks = []
    monkeypatch.setattr(onboarding, "validate_download_token", lambda token, **kw: checks.append((token, kw)))
    monkeypatch.setattr(onboarding, "desktop_directory", lambda: None)
    answers = iter(["", "", "n", "", ""])
    secrets = iter(["", "", "upload-private-fixture", "read-private-fixture"])
    assert onboarding.run_setup(tmp_path, interactive=True, input_fn=lambda _: next(answers),
                                secret_fn=lambda _: next(secrets))
    assert checks == [("read-private-fixture", {"upload_token": "upload-private-fixture"})]
    assert json.loads((tmp_path / ".runtime" / onboarding.MARKER).read_text())["hf_download_token_setup"] == 1
    assert not State().collection and not State().upload_approved
    output = capsys.readouterr().out
    assert all(value not in output for value in ("upload-private-fixture", "read-private-fixture"))


def test_setup_can_add_the_download_token_after_skipping_it(tmp_path, monkeypatch):
    marker = existing_setup(tmp_path)
    marker.write_text('{"schema":1,"lambda_ssh_setup":1,"hf_download_token_setup":1}')
    for key, _ in onboarding.CREDENTIALS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(onboarding, "desktop_directory", lambda: None)
    checks = []
    monkeypatch.setattr(onboarding, "validate_download_token", lambda token, **kw: checks.append(token))
    secrets = iter(["", "", "", "read-private-fixture"])
    answers = iter(["", "", "n", "", ""])
    assert onboarding.run_setup(tmp_path, force=True, interactive=True,
                                secret_fn=lambda _: next(secrets), input_fn=lambda _: next(answers))
    assert checks == ["read-private-fixture"]
    values = read_assignments(tmp_path / ".env")
    assert values[auth.READ_TOKEN_KEY] == "read-private-fixture"
    assert values["HUGGINGFACE_API"] == "upload-private-fixture"


def test_failed_validation_can_be_skipped_without_saving_the_token(tmp_path, monkeypatch, capsys):
    existing_setup(tmp_path)
    original = (tmp_path / ".env").read_bytes()

    def fail(token, **kwargs):
        raise auth.DownloadTokenError("Could not verify the Hugging Face model-download token.")

    monkeypatch.setattr(onboarding, "validate_download_token", fail)
    secrets = iter(["unverified-private-fixture", ""])
    assert onboarding.run_setup(tmp_path, interactive=True, secret_fn=lambda _: next(secrets))
    assert (tmp_path / ".env").read_bytes() == original
    assert "unverified-private-fixture" not in capsys.readouterr().out
