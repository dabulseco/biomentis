"""Pause control: lets an agent thread block until the user picks Continue or Stop.

There is deliberately no timeout. A paused run waits for as long as the user
takes; `wait`'s poll interval only sets how often the thread wakes up.
"""

from __future__ import annotations

import threading

from . import Finding

_CHOICES = ("continue", "stop")


class RoundControl:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._pending: dict | None = None
        self._decision: str | None = None
        self.accepted: set[str] = set()

    def request(self, round_number: int, findings: list[Finding], reason: str) -> None:
        """Record that the run is about to pause. `reason` is "round" or "context"."""
        with self._cond:
            if self._decision != "stop":
                self._decision = None  # a stale Continue must not skip this pause
            self._pending = {"round": round_number, "findings": list(findings), "reason": reason}
            self._cond.notify_all()

    @property
    def pending(self) -> dict | None:
        """The pause awaiting the user, or None. A decided pause is no longer
        pending, even before the run thread wakes up and takes the decision,
        so a UI does not show it again on the rerun after the click."""
        with self._cond:
            if self._pending is None or self._decision is not None:
                return None
            return {**self._pending, "findings": list(self._pending["findings"])}

    def wait(self, poll_seconds: float = 0.5) -> str:
        """Block until a decision is set; return "continue" or "stop"."""
        with self._cond:
            while self._decision is None:
                self._cond.wait(poll_seconds)
            choice = self._decision
            self._decision = None
            self._pending = None
            return choice

    def decide(self, choice: str, accepted_keys: set[str] = frozenset()) -> None:
        if choice not in _CHOICES:
            raise ValueError(f"choice must be one of {_CHOICES}, got {choice!r}")
        with self._cond:
            if self._decision == "stop":
                return  # stop is sticky: nothing replaces it
            if choice == "continue":
                if self._pending is None:
                    return  # stale or duplicate submit, nothing to continue
                self.accepted |= set(accepted_keys)
            self._decision = choice
            self._cond.notify_all()

    def stop(self) -> None:
        self.decide("stop")

    def reset(self) -> None:
        """Clear everything, including a sticky stop. Called at the start of a run."""
        with self._cond:
            self._decision = None
            self._pending = None
            self.accepted = set()
            self._cond.notify_all()
