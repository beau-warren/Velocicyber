import json
from types import SimpleNamespace

import pytest

from velocicyber.providers.lambda_cloud import GPUManager, ModelSpec
from velocicyber.providers.pricing import (
    PUBLISHED_RANGES,
    cache_catalog_prices,
    catalog_price_ranges,
    memory_price_options,
)
from velocicyber.state import State


def offer(vram, count, cents, *, architecture="x86_64", available=False):
    return {
        "instance_type": {
            "name": f"gpu_{count}x_fixture",
            "gpu_description": f"Fixture {vram} GB",
            "architecture": architecture,
            "specs": {"gpus": count, "memory_gib": 1800},
            "price_cents_per_hour": cents,
        },
        "regions_with_capacity_available": [{"name": "fixture"}] if available else [],
    }


def test_ranges_use_exact_vram_category_and_total_instance_api_price():
    catalog = {
        "pcie": offer(80, 1, 329),
        "sxm": offer(80, 1, 429),
        "two40": offer(40, 2, 398),  # Not a single 80 GB option.
        "cluster": offer(80, 8, 3192),
        "cheaper-cluster": offer(80, 8, 2232),
        "arm": offer(80, 1, 100, architecture="arm64"),
        "unknown-price": offer(48, 1, None),
        "negative-price": offer(48, 1, -1),
        "malformed": {},
    }
    ranges = catalog_price_ranges(catalog)
    assert ranges == {80: (329, 429), 640: (2232, 3192)}
    # Include listed rates even with no current capacity. Host RAM is not GPU VRAM.
    assert 1800 not in ranges and 48 not in ranges


def test_menu_uses_cached_rates_and_published_fallback_without_network(settings):
    catalog = {"a": offer(80, 1, 350), "b": offer(80, 1, 450), "c": offer(180, 4, 2800)}
    cache_catalog_prices(settings.runtime, catalog)
    first = dict(memory_price_options(settings.runtime))
    assert first[80] == "80 GB — ($3.50 – $4.50) / hr"
    assert first[720] == "720 GB total — $28.00 / hr"  # Already the total for four GPUs.
    assert first[48] == "48 GB — $1.09 / hr"
    assert first[1800] == "1800 GB total — price unavailable"
    assert dict(memory_price_options(settings.runtime)) == first
    snapshot = json.loads((settings.runtime / "gpu-pricing.json").read_text())
    assert snapshot["ranges"]["720"] == [2800, 2800]
    assert "regions_with_capacity_available" not in snapshot


@pytest.mark.parametrize("body", ["broken", "[]", '{"schema":1}', '{"schema":2}'])
def test_bad_or_unsupported_cache_keeps_offline_menu_usable(settings, body):
    (settings.runtime / "gpu-pricing.json").write_text(body)
    options = dict(memory_price_options(settings.runtime))
    assert options[80] == "80 GB — ($3.29 – $4.29) / hr"
    assert len(options) == 10
    assert PUBLISHED_RANGES[640] == (2232, 3192)


def test_existing_capacity_check_refreshes_prices_without_an_extra_provider_call(settings):
    (settings.root / "fake-key").write_text("fixture")
    settings.options.update(LAMBDA_SSH_KEY_NAME="fixture", SSH_PRIVATE_KEY="fake-key")
    calls = []

    def catalog():
        calls.append("catalog")
        return {"fixture": offer(80, 1, 375)}

    manager = GPUManager(settings, State(), SimpleNamespace(catalog=catalog))
    manager.spec = ModelSpec(settings.local_model, "fixture", 1024, "jinja", "auto", 0)
    with pytest.raises(ValueError, match="No available"):
        manager.plan(80)
    assert dict(memory_price_options(settings.runtime))[80] == "80 GB — $3.75 / hr"
    assert calls == ["catalog"]
    assert not manager.state.instance_id and not manager.state.launch_uncertain


def test_price_cache_write_failure_does_not_block_a_valid_launch_plan(settings, monkeypatch):
    (settings.root / "fake-key").write_text("fixture")
    settings.options.update(LAMBDA_SSH_KEY_NAME="fixture", SSH_PRIVATE_KEY="fake-key")
    manager = GPUManager(settings, State(), SimpleNamespace(
        catalog=lambda: {"fixture": offer(80, 1, 375, available=True)},
    ))
    manager.spec = ModelSpec(settings.local_model, "fixture", 1024, "jinja", "auto", 0)

    def fail(*args, **kwargs):
        raise OSError("fixture disk failure")

    monkeypatch.setattr(type(settings.root), "write_text", fail)
    assert manager.plan(80).hourly_cents == 375
    assert not manager.state.instance_id
