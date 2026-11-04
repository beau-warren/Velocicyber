"""Read the Pi CPU's local thermal sensor without spawning a subprocess."""

from pathlib import Path

# Warn 5°C before the Pi's usual 80°C thermal throttling threshold.
# https://www.raspberrypi.com/news/heating-and-cooling-raspberry-pi-5/
WARNING_C = 75.0
REFRESH_SECONDS = 5


def cpu_temperature_c():
    zone = Path("/sys/class/thermal/thermal_zone0")
    try:
        if (zone / "type").read_text().strip() != "cpu-thermal":
            return None
        value = int((zone / "temp").read_text().strip()) / 1000
    except (OSError, ValueError):
        return None
    return value if -40 <= value <= 150 else None
