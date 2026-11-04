"""Offline GPU-menu estimates, refreshed by the existing Lambda capacity check."""

import json
from datetime import datetime, timezone

from velocicyber.providers.lambda_cloud import MEMORY_CHOICES, SINGLE_MEMORY_CHOICES, total_vram

# Whole-instance USD cents/hour, before tax. Public prices checked 2026-10-19:
# https://lambda.ai/instances (website cluster rates are per GPU; multiplied here).
# These ranges include hardware variants, independently of current capacity or model fit.
PUBLISHED_RANGES = {
    24: (69, 129),
    40: (199, 199),
    48: (109, 109),
    80: (329, 429),
    180: (699, 699),
    360: (1378, 1378),
    640: (2232, 3192),
    720: (2716, 2716),
    1440: (5352, 5352),
}


def catalog_price_ranges(catalog):
    """Group total-instance API quotes by exact VRAM and single/cluster category."""
    prices = {}
    for row in catalog.values():
        try:
            kind = row["instance_type"]
            count = kind["specs"]["gpus"]
            cents = kind["price_cents_per_hour"]
            if (
                kind.get("architecture", "x86_64") != "x86_64"
                or type(count) is not int or count < 1
                or type(cents) is not int or cents < 0
            ):
                continue
            memory = total_vram(kind)
            if memory not in MEMORY_CHOICES:
                continue
            if (memory in SINGLE_MEMORY_CHOICES) != (count == 1):
                continue
            # The API price is already for the entire instance; do not multiply again.
            prices.setdefault(memory, []).append(cents)
        except (KeyError, TypeError, AttributeError, ValueError):
            continue
    return {memory: (min(values), max(values)) for memory, values in prices.items()}


def cache_catalog_prices(runtime, catalog):
    ranges = catalog_price_ranges(catalog)
    if not ranges:
        return
    snapshot = {
        "schema": 1,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "currency": "USD",
        "unit": "cents_per_instance_hour",
        "ranges": ranges,
    }
    try:
        path = runtime / "gpu-pricing.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(snapshot, indent=2) + "\n")
        temporary.chmod(0o600)
        temporary.replace(path)
    except OSError:
        pass  # A display-cache failure must not prevent the current launch plan.


def memory_price_options(runtime):
    """Build labels without contacting a provider or implying current availability."""
    ranges = dict(PUBLISHED_RANGES)
    try:
        snapshot = json.loads((runtime / "gpu-pricing.json").read_text())
        if (
            snapshot.get("schema") == 1 and snapshot.get("currency") == "USD"
            and snapshot.get("unit") == "cents_per_instance_hour"
            and isinstance(snapshot.get("ranges"), dict)
        ):
            for key, values in snapshot["ranges"].items():
                memory = int(key)
                if (
                    memory in MEMORY_CHOICES and isinstance(values, list) and len(values) == 2
                    and all(type(value) is int and value >= 0 for value in values)
                    and values[0] <= values[1]
                ):
                    ranges[memory] = tuple(values)
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    options = []
    for memory in MEMORY_CHOICES:
        label = f"{memory} GB" + (" total" if memory not in SINGLE_MEMORY_CHOICES else "")
        if memory in ranges:
            low, high = ranges[memory]
            price = f"${low / 100:.2f}" if low == high else f"(${low / 100:.2f} – ${high / 100:.2f})"
            label += f" — {price} / hr"
        else:
            label += " — price unavailable"
        options.append((memory, label))
    return options
