"""One launch-time indicator for the harness and its separate execution engine."""

import asyncio

from velocicyber.engine.releases import UpdateChecker as EngineChecker
from velocicyber.updates.source import SourceChecker


class UpdateChecker:
    def __init__(self, settings):
        self.source = SourceChecker(settings)
        self.engine = EngineChecker(settings)

    async def check(self, *, transport=None):
        await asyncio.gather(self.source.check(transport=transport), self.engine.check(transport=transport))

    def indicator(self):
        states = [self.source.indicator(), self.engine.indicator()]
        for name in ("UPDATE AVAILABLE", "UPDATE CHECK FAILED", "CHECKING UPDATES",
                     "VERSION UNKNOWN", "LOCAL SOURCE CHANGES"):
            if any(label == name for label, _ in states):
                return name, "red" if name == "UPDATE AVAILABLE" else "yellow"
        return "UP TO DATE", "#39ff14"

    def plan(self):
        plan = {}
        if self.source.indicator()[0] == "UPDATE AVAILABLE":
            plan["harness"] = {"installed": self.source.installed,
                               "target": self.source.result["latest"]}
        if self.engine.indicator()[0] == "UPDATE AVAILABLE":
            plan["engine"] = {"target": self.engine.result["latest"]}
        return plan

    def details(self):
        return self.source.details() + "\n\n" + self.engine.details() + (
            "\n\nUse /update to approve available components or change automatic repo updates on launch. Engine upgrades remain manual."
        )
