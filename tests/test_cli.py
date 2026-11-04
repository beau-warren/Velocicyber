import sys

import pytest

from velocicyber import cli
from velocicyber.datasets.pipeline import RawStore, Scrubber


@pytest.mark.parametrize("sandbox_open", [False, True])
@pytest.mark.parametrize(
    "flags,collect,upload,compaction",
    [
        ([], False, False, True),
        (["--collect", "--upload"], True, True, True),
        (["--collect"], True, False, True),
        (["--upload"], False, False, True),
        (["--no-collect"], False, False, True),
        (["--no-upload"], False, False, True),
        (["--no-allow-compaction"], False, False, False),
    ],
)
def test_run_collection_upload_opt_in(
    settings, episode, monkeypatch, flags, collect, upload, compaction, sandbox_open
):
    calls = []

    class Engine:
        def __init__(self, settings, state, workspace=None):
            self.state = state
            self.scrubber = Scrubber()

        async def run(self, prompt, callback):
            assert self.state.sandbox_open == sandbox_open
            assert self.state.collection == collect
            assert self.state.allow_compaction == compaction
            if collect:
                RawStore(settings.data).append(episode)

        def close(self):
            calls.append("closed")

    class Uploader:
        def __init__(self, settings):
            pass

        def upload(self, artifact, *, approved):
            assert approved and (artifact / "manifest.json").exists()
            calls.append("uploaded")
            return {"training_hash": "fixture"}

    monkeypatch.setattr(cli.Settings, "load", lambda root: settings)
    monkeypatch.setattr(cli, "Engine", Engine)
    monkeypatch.setattr(cli, "Uploader", Uploader)
    access_flags = ["--sandbox", "open"] if sandbox_open else []
    monkeypatch.setattr(sys, "argv", ["velocicyber", *access_flags, "run", "Fixture task", *flags])
    cli.main()
    assert ("uploaded" in calls) == upload
    assert calls[-1] == "closed"
