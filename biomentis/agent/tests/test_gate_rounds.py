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

    # Checks that did not run: never "Passed".
    skipped = GateResult(2, [], "ok", skipped=["arithmetic check", "rubric coverage"])
    _check("label: passed but checks skipped",
           final_label(skipped, False, 2) == "⚠️ Not fully checked (arithmetic check, rubric coverage did not run)",
           final_label(skipped, False, 2))
    _check("label: reviewer off keeps priority over skipped",
           final_label(GateResult(2, [], "off", skipped=["figure check"]), False, 2)
           == "⚠️ Not fully checked (reviewer off)")
    _check("label: review failed keeps priority over skipped",
           final_label(GateResult(2, [], "failed", skipped=["figure check"]), False, 2)
           == "⚠️ Not fully checked (review failed)")
    _check("label: stopped keeps priority over skipped",
           final_label(GateResult(2, [], "ok", skipped=["figure check"]), True, 2) == "⛔ Stopped by user — did not pass")
    _check("label: not passed and skipped is still not passed",
           final_label(GateResult(3, list(fail.findings), "ok", skipped=["figure check"]), False, 3)
           == "❌ Did not pass after 3 rounds")

    # Findings accepted as is: the pass depended on them, so the label says so.
    a, b = Finding("arithmetic", "blocking", "q1", "bad 1"), Finding("figure", "blocking", "q2", "bad 2")
    accepted = GateResult(4, [a, b, Finding("figure", "blocking", "q2", "bad 2")], "ok", accepted={a.key, b.key})
    _check("accepted_blocking: one per key", len(accepted.accepted_blocking) == 2, repr(accepted.accepted_blocking))
    _check("label: passed with accepted findings",
           final_label(accepted, False, 4) == "✅ Passed all checks after 4 rounds (2 accepted as is)",
           final_label(accepted, False, 4))
    stale = GateResult(4, [Finding("figure", "warning", "q", "w")], "ok", accepted={"some-old-key"})
    _check("label: accepted keys that no longer block do not count",
           final_label(stale, False, 4) == "✅ Passed all checks after 4 rounds", final_label(stale, False, 4))

    test_run_gate_skipped()

    print(f"\n{len(_FAILED)} failed")
    return 1 if _FAILED else 0


def test_run_gate_skipped() -> None:
    print("\n[run_gate] checks that did not run are recorded")
    import json
    from types import SimpleNamespace

    from langchain_core.messages import AIMessage, HumanMessage

    import biomentis.gate.runner as runner

    msgs = [HumanMessage(content="What is 24+3?"),
            AIMessage(content="<execute>print(query_pubmed('sums'))</execute>"),
            AIMessage(content="<observation>Title: a paper</observation>"),
            AIMessage(content="<execute>print(24 + 3)</execute>"),
            AIMessage(content="<observation>27</observation>"),
            AIMessage(content="<solution>The total is 24+3 = 27 items.</solution>")]

    class _LLM:
        def __init__(self, replies):
            self.replies = list(replies)

        def invoke(self, messages):
            reply = self.replies.pop(0) if self.replies else "not json"
            if isinstance(reply, Exception):
                raise reply
            return AIMessage(content=reply)

    def agent(gate_replies, review_llm=None):
        return SimpleNamespace(llm=None, gate_llm=_LLM(gate_replies), round_control=None, review_llm=review_llm,
                               run_gate_results=[], _gate_prev_claims=None)

    empty = json.dumps({"quantities": []})
    result = runner.run_gate(agent([empty]), msgs, 0)
    _check("all checks ran: nothing skipped", result.skipped == [], str(result.skipped))

    result = runner.run_gate(agent(["not json", "still not json"]), msgs, 0)
    _check("a section with no usable extraction is skipped",
           result.skipped == ["quantity extraction for 1 section"], str(result.skipped))

    original = runner.check_arithmetic
    runner.check_arithmetic = lambda answer: 1 / 0
    try:
        result = runner.run_gate(agent([empty]), msgs, 0)
    finally:
        runner.check_arithmetic = original
    _check("a crashed checker is skipped", result.skipped == ["arithmetic check"], str(result.skipped))

    original = runner.extract_quantities
    runner.extract_quantities = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))
    try:
        result = runner.run_gate(agent([]), msgs, 0)
    finally:
        runner.extract_quantities = original
    _check("extraction raised: quantity checks skipped", result.skipped == ["quantity checks"], str(result.skipped))

    class _Review:
        error = ""
    rev_agent = agent([empty], review_llm=object())
    rev_agent.run_review = lambda messages, rubric=None, focus_numbers=None: _Review()
    original = runner.review_findings
    runner.review_findings = lambda review: []
    try:
        result = runner.run_gate(rev_agent, msgs, 0, rubric=None)
    finally:
        runner.review_findings = original
    _check("reviewer on, no rubric: rubric coverage skipped",
           result.reviewer_state == "ok" and result.skipped == ["rubric coverage"], f"{result.reviewer_state} {result.skipped}")
    _check("... label: not fully checked",
           final_label(result, False, 0) == "⚠️ Not fully checked (rubric coverage did not run)",
           final_label(result, False, 0))


if __name__ == "__main__":
    sys.exit(main())
