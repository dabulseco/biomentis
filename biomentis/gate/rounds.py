"""Round policy for the completion gate: pure logic, no I/O."""

from __future__ import annotations

from biomentis.gate import GateResult

AUTO_ROUND_LIMIT = 3


def _effective_passed(result: GateResult) -> bool:
    """Passed, ignoring reviewer-dependent findings when the reviewer did not run."""
    if result.reviewer_state in ("off", "failed"):
        return not [f for f in result.blocking if f.check not in ("rubric", "review")]
    return result.passed


def next_action(round_done: int, planned: int, result: GateResult) -> str:
    """Return "critique", "await_decision" or "end"."""
    if round_done < planned:
        return "critique"
    if _effective_passed(result):
        return "end"
    if round_done < AUTO_ROUND_LIMIT:
        return "critique"
    return "await_decision"


def final_label(result: GateResult, stopped: bool, rounds: int) -> str:
    n = f"{rounds} round" if rounds == 1 else f"{rounds} rounds"
    if stopped:
        return "⛔ Stopped by user — did not pass"
    if result.reviewer_state == "off":
        return "⚠️ Not fully checked (reviewer off)"
    if result.reviewer_state == "failed":
        return "⚠️ Not fully checked (review failed)"
    if result.passed:
        return f"✅ Passed all checks after {n}"
    return f"❌ Did not pass after {n}"  # defensive: not passed, not stopped, reviewer ok
