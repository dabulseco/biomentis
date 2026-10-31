"""The completion gate: checks a research answer must pass before it is final.

Each checker returns a list of ``Finding``. Only blocking findings fail the
gate, and a user's "accept as is" resolves one for the run by its ``key``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class Finding:
    check: str          # "arithmetic" | "quantity" | "figure" | "claim" | "evidence" | "rubric" | "review" | "regression"
    severity: str       # "blocking" | "warning"
    quote: str          # exact text from the answer
    issue: str
    fix: str = ""
    key: str = ""       # stable id for "accept as is"

    def __post_init__(self) -> None:
        if not self.key:
            self.key = hashlib.sha1((self.check + self.quote).encode()).hexdigest()[:10]


@dataclass
class GateResult:
    round: int
    findings: list[Finding]
    reviewer_state: str  # "ok" | "off" | "failed"
    accepted: set[str] = field(default_factory=set)

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "blocking" and f.key not in self.accepted]

    @property
    def passed(self) -> bool:
        return not self.blocking
