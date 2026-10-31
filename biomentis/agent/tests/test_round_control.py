"""Tests for the pause control: a run blocks from round 4 until Continue or Stop.

Run with:
    python -m biomentis.agent.tests.test_round_control
"""

from __future__ import annotations

import inspect
import sys
import threading
import time

from biomentis.gate import Finding

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


def _wait_in_thread(control):
    out: dict = {}

    def work():
        t0 = time.monotonic()
        out["choice"] = control.wait()
        out["took"] = time.monotonic() - t0

    th = threading.Thread(target=work, daemon=True)
    th.start()
    return th, out


def _wait_for_pending(control, limit: float = 2.0) -> bool:
    end = time.monotonic() + limit
    while time.monotonic() < end:
        if control.pending is not None:
            return True
        time.sleep(0.01)
    return False


def main() -> int:
    from biomentis.gate.control import RoundControl
    from biomentis.run_worker import BackgroundRun

    finding = Finding("arithmetic", "blocking", "2+2=5", "wrong")

    print("continue after a long wait")
    c = RoundControl()
    _check("pending is None before request", c.pending is None)
    c.request(4, [finding], "round")
    p = c.pending
    _check("pending describes the request",
           p is not None and p["round"] == 4 and p["reason"] == "round" and p["findings"] == [finding], str(p))
    th, out = _wait_in_thread(c)
    time.sleep(1.5)
    _check("still blocked after 1.5 s", th.is_alive())
    c.decide("continue", {"abc"})
    th.join(3)
    _check("wait returns continue", out.get("choice") == "continue", str(out))
    _check("accepted keys recorded", c.accepted == {"abc"}, str(c.accepted))
    _check("pending cleared after wait", c.pending is None)

    print("accepted keys accumulate; cycle restarts fresh")
    c.request(5, [], "context")
    th, out = _wait_in_thread(c)
    _wait_for_pending(c)
    c.decide("continue", {"def"})
    th.join(3)
    _check("second pause continues", out.get("choice") == "continue")
    _check("accepted accumulates", c.accepted == {"abc", "def"}, str(c.accepted))

    print("pending is a copy")
    c.request(6, [finding], "round")
    snap = c.pending
    snap["round"] = 99
    snap["findings"].clear()
    _check("mutating pending does not change state",
           c.pending["round"] == 6 and c.pending["findings"] == [finding])
    th, out = _wait_in_thread(c)
    c.decide("stop")
    th.join(3)
    _check("stop decision returned", out.get("choice") == "stop")

    print("invalid choice")
    try:
        c.decide("maybe")
        _check("invalid choice raises ValueError", False, "no exception")
    except ValueError:
        _check("invalid choice raises ValueError", True)

    print("stop while not waiting")
    c2 = RoundControl()
    c2.stop()
    t0 = time.monotonic()
    _check("next wait returns stop at once", c2.wait() == "stop" and time.monotonic() - t0 < 0.3)

    print("BackgroundRun.cancel stops a waiting control")
    c3 = RoundControl()
    run = BackgroundRun("p")
    run.round_control = c3
    c3.request(4, [finding], "round")
    th, out = _wait_in_thread(c3)
    _wait_for_pending(c3)
    t0 = time.monotonic()
    run.cancel()
    th.join(3)
    _check("cancel returns stop within 1 s",
           out.get("choice") == "stop" and time.monotonic() - t0 < 1.0, str(out))
    _check("run is marked cancelled", run.cancelled)
    _check("round_control defaults to None", BackgroundRun("p").round_control is None)
    BackgroundRun("p").cancel()
    _check("cancel without a control still works", True)

    print("no timeouts")
    _check("wait has no timeout parameter", "timeout" not in inspect.signature(RoundControl.wait).parameters)
    import biomentis.gate.control as mod
    src = inspect.getsource(mod)
    _check("module never reads timeout_seconds", "timeout_seconds" not in src)

    print(f"\n{len(_PASSED)} passed, {len(_FAILED)} failed")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
