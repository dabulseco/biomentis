"""Tests for the completion gate's round policy (pure logic).

Run with:
    python -m biomentis.agent.tests.test_gate_rounds
"""

from __future__ import annotations

import sys

from biomentis.gate import Finding, GateResult
from biomentis.gate.rounds import AUTO_ROUND_LIMIT, final_label, next_action

_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


def _res(check: str | None = "arithmetic", state: str = "ok", rnd: int = 0) -> GateResult:
    fs = [Finding(check, "blocking", "q", "bad")] if check else []
    return GateResult(rnd, fs, state)


def main() -> int:
    _check("limit is 3", AUTO_ROUND_LIMIT == 3)
    fail = _res()
    ok = _res(None)
    table = [
        ("planned=1 round 0 fails", 0, 1, fail, "critique"),
        ("planned=1 round 1 fails", 1, 1, fail, "critique"),
        ("round 2 fails", 2, 1, fail, "critique"),
        ("round 3 fails", 3, 1, fail, "await_decision"),
        ("round 4 fails", 4, 1, fail, "await_decision"),
        ("planned=2 round 1 passes", 1, 2, ok, "critique"),
        ("planned=2 round 2 passes", 2, 2, ok, "end"),
        ("planned=0 round 0 passes", 0, 0, ok, "end"),
        ("reviewer off, only rubric blocking", 1, 1, _res("rubric", "off"), "end"),
        ("reviewer failed, only review blocking", 1, 1, _res("review", "failed"), "end"),
        ("reviewer off, arithmetic still blocks", 1, 1, _res("arithmetic", "off"), "critique"),
        ("reviewer off, arithmetic blocks at limit", 3, 1, _res("arithmetic", "off"), "await_decision"),
        ("reviewer ok, rubric blocks", 1, 1, _res("rubric", "ok"), "critique"),
    ]
    for name, done, planned, res, want in table:
        got = next_action(done, planned, res)
        _check(name, got == want, f"{got!r} != {want!r}")

    _check("label stopped", final_label(fail, True, 2) == "⛔ Stopped by user — did not pass")
    _check("label stopped beats off", final_label(_res(None, "off"), True, 2) == "⛔ Stopped by user — did not pass")
    _check("label off", final_label(_res(None, "off"), False, 2) == "⚠️ Not fully checked (reviewer off)")
    _check("label failed", final_label(_res(None, "failed"), False, 2) == "⚠️ Not fully checked (review failed)")
    _check("label passed plural", final_label(ok, False, 2) == "✅ Passed all checks after 2 rounds")
    _check("label passed singular", final_label(ok, False, 1) == "✅ Passed all checks after 1 round")
    _check("label passed zero", final_label(ok, False, 0) == "✅ Passed all checks after 0 rounds")
    _check("label did not pass", final_label(fail, False, 3) == "❌ Did not pass after 3 rounds")

    print(f"\n{len(_FAILED)} failed")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
