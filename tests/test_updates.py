import asyncio
import json
import time

import httpx
import pytest
from textual.widgets import Input, OptionList, RichLog, Static

from velocicyber.engine.releases import REPOSITORY, UpdateChecker
from velocicyber.ui.app import VelocicyberApp


@pytest.fixture
def update_settings(settings):
    (settings.root / "config/locks/upstream.lock.json").write_text(json.dumps({
        "repository": REPOSITORY, "tag": "rust-v0.0.44",
    }))
    engine = settings.root / "test-engine"
    engine.write_text("#!/bin/sh\nprintf 'interpreter 0.0.45\\n'\n")
    engine.chmod(0o755)
    settings.options["INTERPRETER_BIN"] = str(engine)
    return settings


def release(tag, **kwargs):
    return {"tag_name": tag, "prerelease": False, "draft": False, **kwargs}


def github(releases, advisories=None, calls=None):
    def request(request):
        assert request.url.host == "api.github.com"
        assert "authorization" not in request.headers
        if calls is not None:
            calls.append(request.url.path)
        if request.url.path.endswith("/releases/latest"):
            return httpx.Response(200, json=releases[0] if releases else {})
        return httpx.Response(
            200, json=releases if request.url.path.endswith("/releases") else advisories or [],
        )

    return httpx.MockTransport(request)


@pytest.mark.parametrize("latest,expected", [
    ("rust-v0.0.45", "UP TO DATE"),
    ("rust-v0.0.46", "UPDATE AVAILABLE"),
    ("rust-v0.0.100", "UPDATE AVAILABLE"),
    ("rust-v0.0.9", "UP TO DATE"),
])
async def test_uses_running_binary_and_stable_rust_versions(update_settings, latest, expected):
    checker = UpdateChecker(update_settings)
    await checker.check(transport=github([
        release("v99.0.0"),  # The separate Python distribution must not trigger the badge.
        release("rust-v2.0.0", prerelease=True),
        release("rust-v3.0.0", draft=True),
        release("rust-v4.0.0-beta.1"),
        release(latest),
    ]))
    assert checker.result["installed"] == "rust-v0.0.45"  # Not the stale lock's 0.0.44.
    assert checker.result["latest"] == latest
    assert checker.indicator()[0] == expected
    assert checker.indicator()[1] == ("red" if expected == "UPDATE AVAILABLE" else "#39ff14")


async def test_checks_once_per_launch_despite_recent_cache_or_elapsed_time(
    update_settings, monkeypatch,
):
    now = time.time()
    monkeypatch.setattr("velocicyber.engine.releases.time.time", lambda: now)
    calls = []
    transport = github([release("rust-v0.0.45")], calls=calls)
    checker = UpdateChecker(update_settings)
    await checker.check(transport=transport)
    now += 24 * 60 * 60
    await checker.check(transport=transport)
    assert len(calls) == 2  # No periodic refresh, even after a day.
    again = UpdateChecker(update_settings)
    assert again.indicator()[0] == "UP TO DATE"
    await again.check(transport=transport)
    # Even an immediate relaunch checks afresh and can discover a newer release.
    reopened = UpdateChecker(update_settings)
    await reopened.check(transport=github([release("rust-v0.0.46")], calls=calls))
    assert len(calls) == 6
    assert reopened.indicator()[0] == "UPDATE AVAILABLE"


@pytest.mark.parametrize("failure", ["offline", "rate-limit", "malformed", "empty"])
async def test_failed_check_is_amber_and_cached(update_settings, failure):
    calls = []

    def request(request):
        calls.append(request.url)
        if failure == "offline":
            raise httpx.ConnectError("offline", request=request)
        if failure == "rate-limit":
            return httpx.Response(403, json={"message": "rate limited"})
        return httpx.Response(200, json={} if failure == "malformed" else [])

    transport = httpx.MockTransport(request)
    checker = UpdateChecker(update_settings)
    await checker.check(transport=transport)
    assert checker.indicator() == ("UPDATE CHECK FAILED", "yellow")
    await checker.check(transport=transport)
    assert len(calls) == 2  # No automatic retry during this launch.
    restored = UpdateChecker(update_settings)
    await restored.check(transport=transport)
    assert restored.indicator() == ("UPDATE CHECK FAILED", "yellow")
    assert len(calls) == 4  # Reopening retries even a recent cached failure.


async def test_advisory_failure_cannot_look_fully_current(update_settings):
    def request(request):
        if request.url.path.endswith("/releases/latest"):
            return httpx.Response(200, json=release("rust-v0.0.45"))
        return httpx.Response(429)

    checker = UpdateChecker(update_settings)
    await checker.check(transport=httpx.MockTransport(request))
    assert checker.indicator() == ("UPDATE CHECK FAILED", "yellow")
    assert "advisory check failed (HTTP 429)" in checker.details()


async def test_known_update_retained_when_next_launch_check_fails(update_settings):
    checker = UpdateChecker(update_settings)
    await checker.check(transport=github([release("rust-v0.0.46")]))
    checker = UpdateChecker(update_settings)
    await checker.check(transport=httpx.MockTransport(lambda _: httpx.Response(503)))
    assert checker.indicator() == ("UPDATE AVAILABLE", "red")
    assert "check failed" in checker.details()


async def test_advisories_visible_without_claiming_vulnerability(update_settings):
    checker = UpdateChecker(update_settings)
    await checker.check(transport=github([release("rust-v0.0.46")], [
        {"ghsa_id": "GHSA-test", "state": "published", "severity": "high", "summary": "Fix"},
        {"ghsa_id": "GHSA-withdrawn", "state": "published", "withdrawn_at": "2026-01-01"},
    ]))
    assert "Published repository security advisories: 1" in checker.details()
    assert "GHSA-test (high): Fix" in checker.details()
    assert "affected packages and versions" in checker.details()
    assert REPOSITORY + "/security/advisories" in checker.details()


async def test_unidentified_binary_does_not_fall_back_to_pin(update_settings):
    engine = update_settings.engine()
    engine.write_text("#!/bin/sh\necho unexpected\n")
    checker = UpdateChecker(update_settings)
    await checker.check(transport=github([release("rust-v0.0.44")]))
    assert checker.indicator()[1] == "yellow"
    assert "Could not read the installed engine version" in checker.details()


async def test_changed_engine_invalidates_cache(update_settings):
    checker = UpdateChecker(update_settings)
    await checker.check(transport=github([release("rust-v0.0.45")]))
    update_settings.engine().write_text("#!/bin/sh\nprintf 'interpreter 0.0.46\\n'\n# changed\n")
    restored = UpdateChecker(update_settings)
    assert not restored.checked_this_launch and restored.indicator()[0] == "CHECKING UPDATES"


async def test_missing_configuration_never_uses_network(settings):
    checker = UpdateChecker(settings)
    await checker.check(transport=httpx.MockTransport(lambda _: pytest.fail("Unexpected request")))
    assert checker.indicator() == ("VERSION UNKNOWN", "yellow")


@pytest.mark.parametrize("content", ["{broken", "[]", '{"schema": 1, "checked_at": "bad"}'])
def test_corrupt_cache_is_ignored(update_settings, content):
    (update_settings.runtime / "upstream-updates.json").write_text(content)
    checker = UpdateChecker(update_settings)
    assert not checker.checked_this_launch and not checker.result


async def test_dashboard_check_is_nonblocking_and_does_not_consume_log_space(
    update_settings, monkeypatch,
):
    started, finish = asyncio.Event(), asyncio.Event()
    checks = []

    async def check(self):
        checks.append(self)
        started.set()
        await finish.wait()
        self.result = {
            "installed": "rust-v0.0.45", "latest": "rust-v0.0.46", "checked_at": time.time(),
        }

    monkeypatch.setattr(UpdateChecker, "check", check)
    app = VelocicyberApp(update_settings)
    async with app.run_test(size=(80, 24)) as pilot:
        await asyncio.wait_for(started.wait(), 2)
        await pilot.pause()
        home_badge = app.query_one("#home-update-status", Static)
        assert "CHECKING UPDATES" in str(home_badge.render())
        assert home_badge.display and home_badge.region.y == 0
        assert home_badge.region.right == app.size.width - 2
        assert home_badge.region.bottom <= app.query_one("#title").region.y
        app.show_conversation()
        app.refresh_updates()  # Overlapping calls cannot interrupt the launch check.
        await pilot.pause()
        assert len(checks) == 1
        assert "CHECKING UPDATES" in str(app.query_one("#update-status", Static).render())
        app.query_one("#prompt", Input).value = "/updates-info"
        await pilot.pause()
        assert app.query_one("#commands", OptionList).option_count == 1
        finish.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        badge = app.query_one("#update-status", Static)
        assert "UPDATE AVAILABLE" in str(badge.render())
        assert "UPDATE AVAILABLE" in str(home_badge.render()) and not home_badge.display
        assert badge.region.right == app.query_one("#masthead").content_region.right
        assert badge.region.y == app.query_one("#title").region.y
        assert app.query_one("#masthead").region.height == 9
        dashboard = app.query_one("#dashboard")
        assert dashboard.region.height == 7
        assert dashboard.region.y == badge.region.bottom
        assert dashboard.region.x == badge.region.x
        assert dashboard.region.x >= app.query_one("#title").region.right + 2
        await pilot.press("enter")
        text = "\n".join(line.text for line in app.query_one("#log", RichLog).lines)
        assert "rust-v0.0.46" in text
        app.updates.result["latest"] = "rust-v0.0.45"
        app.animate()
        assert "UP TO DATE" in str(badge.render())
        app.command("home")
        await pilot.pause()
        assert home_badge.display and "UP TO DATE" in str(home_badge.render())
        assert home_badge.region.y == 0 and home_badge.region.right == app.size.width - 2
        app.updates.result["error"] = "Fixture: release check failed"
        app.animate()
        await pilot.pause()
        assert "UPDATE CHECK FAILED" in str(home_badge.render())
        assert home_badge.region.y == 0 and home_badge.region.right == app.size.width - 2
        assert len(checks) == 1


async def test_demo_never_checks_updates(update_settings, monkeypatch):
    monkeypatch.setattr(UpdateChecker, "check", lambda _: pytest.fail("Demo contacted GitHub"))
    app = VelocicyberApp(update_settings, demo=True)
    async with app.run_test() as pilot:
        assert "UPDATES: DEMO" in str(app.query_one("#update-status", Static).render())
        app.command("updates-info")
        await pilot.pause()
        assert not app.workers
