"""Private, durable conversation history, independent of fine-tuning collection."""

import fcntl
import json
import os
import re
import uuid
from pathlib import Path

from velocicyber.engine.capture import utcnow


def private_write(path, text):
    temporary = path.with_suffix(".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class Conversation:
    def __init__(self, folder, lock, metadata):
        self.folder, self.lock, self.metadata = folder, lock, metadata

    @property
    def home(self):
        return self.folder / "engine"

    def save(self, **changes):
        changes.setdefault("updated_at", utcnow())
        self.metadata.update(changes)
        private_write(self.folder / "session.json", json.dumps(self.metadata) + "\n")

    def append(self, event):
        fd = os.open(self.folder / "transcript.jsonl", os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a+b") as stream:
            if stream.seek(0, os.SEEK_END):
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) != b"\n":
                    stream.write(b"\n")
            stream.write((json.dumps(event, ensure_ascii=False) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())

    def events(self):
        path = self.folder / "transcript.jsonl"
        if path.exists():
            with path.open() as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue  # A power loss may leave a partial final line.
                    if isinstance(event, dict):
                        yield event

    def close(self):
        if self.lock:
            if self.metadata.get("status") == "running":
                self.save(status="interrupted")
            self.lock.close()
            self.lock = None


class SessionStore:
    def __init__(self, settings, scrubber):
        self.settings, self.scrubber = settings, scrubber
        self.root = settings.runtime / "conversations"
        if self.root.is_symlink():
            raise ValueError("Conversation storage must not be a symlink.")
        self.root.mkdir(mode=0o700, exist_ok=True)

    def folder(self, session_id):
        if not re.fullmatch(r"[a-f0-9]{32}", session_id):
            raise ValueError("Invalid saved conversation ID.")
        folder = self.root / session_id
        if folder.is_symlink() or folder.resolve().parent != self.root.resolve():
            raise ValueError("Saved conversation path escapes storage.")
        return folder

    def acquire(self, folder):
        fd = os.open(folder / "session.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        lock = os.fdopen(fd, "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise RuntimeError(
                "This conversation is open in another harness. Close it there first."
            ) from None
        return lock

    def create(self, workspace, system_prompt, *, session_id=None):
        session_id = session_id or uuid.uuid4().hex
        folder = self.folder(session_id)
        folder.mkdir(mode=0o700)
        (folder / "engine").mkdir(mode=0o700)
        private_write(folder / "engine" / "system-prompt.md", system_prompt)
        conversation = Conversation(
            folder,
            self.acquire(folder),
            {
                "schema_version": 1,
                "id": session_id,
                "title": "",
                "thread_id": None,
                "workspace": str(workspace),
                "task_id": None,
                "verifier": [],
                "created_at": utcnow(),
                "status": "new",
                "provider": "",
                "model": "",
            },
        )
        conversation.save()
        return conversation

    def open(self, session_id, *, allow_external_workspace=False):
        folder = self.folder(session_id)
        lock = self.acquire(folder)
        try:
            if (folder / "session.json").is_symlink():
                raise ValueError("Saved metadata must not be a symlink.")
            metadata = json.loads((folder / "session.json").read_text())
            if metadata.get("schema_version") != 1 or metadata.get("id") != session_id:
                raise ValueError("Unsupported saved conversation metadata.")
            workspace = Path(metadata["workspace"]).resolve()
            if not workspace.is_dir():
                raise ValueError("The saved workspace is missing.")
            if not allow_external_workspace and not workspace.is_relative_to(
                self.settings.root.resolve()
            ):
                raise ValueError(
                    "The saved workspace is outside this project. "
                    "Select /sandbox > Open before resuming it."
                )
            home = (folder / "engine").resolve()
            if (
                not home.is_relative_to(folder)
                or not (home / "system-prompt.md").is_file()
                or not (home / "system-prompt.md").resolve().is_relative_to(home)
            ):
                raise ValueError("Saved engine history is missing or outside this conversation.")
            uuid.UUID(metadata["thread_id"])
            if not any(home.glob(f"sessions/**/rollout-*-{metadata['thread_id']}.jsonl")):
                raise ValueError("The saved model conversation history is missing.")
            conversation = Conversation(folder, lock, metadata)
            if metadata["status"] == "running":
                conversation.save(status="interrupted")
            return conversation
        except BaseException:
            lock.close()
            raise

    def list(self):
        rows = []
        for path in self.root.glob("*/session.json"):
            try:
                if self.folder(path.parent.name) != path.parent or path.is_symlink():
                    continue
                row = json.loads(path.read_text())
                if row.get("thread_id") and row.get("schema_version") == 1:
                    rows.append(row)
            except (ValueError, OSError):
                continue
        return sorted(rows, key=lambda row: row.get("updated_at", ""), reverse=True)

    def import_legacy(self):
        """Copy surviving temporary rollouts; never change or delete their source."""
        imported = 0
        for path in self.settings.runtime.glob("session-*/sessions/**/*.jsonl"):
            if path.is_symlink() or not path.resolve().is_relative_to(self.settings.runtime):
                continue
            try:
                # Ignore a partially written trailing line from a still-open old app.
                records = []
                for line in path.read_text().splitlines():
                    try:
                        records.append(json.loads(line))
                    except ValueError:
                        continue
                meta = records[0]["payload"]
                native_id = str(uuid.UUID(meta["id"]))
                session_id = uuid.UUID(native_id).hex
                if self.folder(session_id).exists():
                    continue
                workspace = Path(meta["cwd"]).resolve()
                if not workspace.is_relative_to(self.settings.root) or not workspace.is_dir():
                    continue
                legacy_home = next(p for p in path.parents if p.parent == self.settings.runtime)
                prompt_file = legacy_home / "system-prompt.md"
                if not prompt_file.is_file():
                    continue
                conversation = self.create(
                    workspace, prompt_file.read_text(), session_id=session_id
                )
                try:
                    target = conversation.home / path.relative_to(legacy_home)
                    target.parent.mkdir(parents=True, mode=0o700)
                    private_write(target, "".join(json.dumps(row) + "\n" for row in records))
                    title, model = "Imported conversation", ""
                    for record in records:
                        payload = record.get("payload", {})
                        if record.get("type") == "turn_context":
                            model = payload.get("model", model)
                        if (
                            record.get("type") != "event_msg"
                            or payload.get("type") != "item_completed"
                        ):
                            continue
                        item = payload.get("item", {})
                        text = "".join(
                            part.get("text", "")
                            for part in item.get("content", [])
                            if isinstance(part, dict) and part.get("type") in ("text", "Text")
                        )
                        if item.get("type") == "UserMessage":
                            if title == "Imported conversation":
                                title = " ".join(text.split())[:100]
                            event = {"type": "user.prompt", "text": text}
                        elif item.get("type") == "AgentMessage":
                            event = {
                                "type": "item.completed",
                                "item": {"type": "agent_message", "text": text},
                            }
                        elif item.get("type") == "CommandExecution":
                            event = {
                                "type": "item.completed",
                                "item": {**item, "type": "command_execution"},
                            }
                        else:
                            continue
                        conversation.append(self.scrubber.clean(event))
                    conversation.save(
                        thread_id=native_id,
                        title=self.scrubber.text(title),
                        model=model,
                        provider="api" if self.settings.is_api_model(model) else "local",
                        status="imported",
                        created_at=meta.get("timestamp", utcnow()),
                        updated_at=records[-1].get("timestamp", meta.get("timestamp", utcnow())),
                        legacy_source=str(path.relative_to(self.settings.root)),
                    )
                    imported += 1
                finally:
                    conversation.close()
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                continue
        return imported
