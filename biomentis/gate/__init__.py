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
    key: str = ""       # stable id for "accept as is" (default: hash of check + quote + issue)

    def __post_init__(self) -> None:
        if not self.key:
            # The issue is part of it: two findings on one quote (say a blocking
            # derivation and a code-output warning) must not share an "accept as is".
            self.key = hashlib.sha1((self.check + self.quote + self.issue).encode()).hexdigest()[:10]


@dataclass
class GateResult:
    round: int
    findings: list[Finding]
    reviewer_state: str  # "ok" | "off" | "failed"
    accepted: set[str] = field(default_factory=set)
    # Checks that failed or did not run this round (a checker crashed, the
    # quantity extraction failed or skipped a section, no rubric to check
    # with the reviewer on). A result that passed with any is not "Passed".
    skipped: list[str] = field(default_factory=list)

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "blocking" and f.key not in self.accepted]

    @property
    def passed(self) -> bool:
        return not self.blocking

    @property
    def accepted_blocking(self) -> list[Finding]:
        """The blocking findings that pass only because the user accepted them as is (one per key)."""
        seen: dict[str, Finding] = {}
        for f in self.findings:
            if f.severity == "blocking" and f.key in self.accepted:
                seen.setdefault(f.key, f)
        return list(seen.values())
