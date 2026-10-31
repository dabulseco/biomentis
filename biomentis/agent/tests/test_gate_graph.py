"""Tests for the completion gate wired into the agent graph.

The graph is driven end to end with a scripted model; only the code steps
really run. Each scripted run consults a (stubbed) literature tool, makes a
calculation and "draws" a figure, so the evidence check is satisfied and the
arithmetic check alone decides the gate.

What these tests pin down:

  (a) A wrong sum fails round 0, the critique goes back to the model, the
      corrected answer passes in round 1; reviewer off → "Not fully checked".
  (b) An answer that keeps the error pauses before round 4; Stop ends the run
      as "Stopped by user".
  (c) With completion_gate=False the old graph runs: no gate, no status.
  (d) The step trace holds one gate record per gate, and the report counts
      gate failures by check.
  (e) The context-limit pause: Stop ends the run without calling the model;
      Continue proceeds, and the pause is asked once per run.
  (f) Finding keys differ when only the issue differs.

Run with:
    python -m biomentis.agent.tests.test_gate_graph
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

os.environ["BIOMENTIS_ARTIFACT_DIR"] = tempfile.mkdtemp(prefix="biomentis_gate_graph_")

from langchain_core.messages import AIMessage, HumanMessage

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


class _Scripted:
    def __init__(self, replies: list):
        self.replies = list(replies)
        self.requests: list = []

    def invoke(self, messages):
        self.requests.append(list(messages))
        if not self.replies:
            raise RuntimeError("called more times than scripted")
        return AIMessage(content=self.replies.pop(0))


class _NoQuantities:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=json.dumps({"quantities": []}))


_EVIDENCE_STEPS = [
    "<execute>def query_pubmed(q):\n    return 'Title: a paper'\nprint(query_pubmed('sums'))</execute>",
    "<execute>total = 24 + 3\nprint('Plot captured.', total)</execute>",
]
_WRONG = "<solution>The total is 24+3 = 28 items.</solution>"
_RIGHT = "<solution>The total is 24+3 = 27 items.</solution>"


def _agent(llm, critic=None, planned=1, gate=True, trace_dir=None):
    from biomentis.agent.a1 import A1

    agent = A1(path="./data", use_tool_retriever=False, expected_data_lake_files=[],
               trace=bool(trace_dir), trace_dir=trace_dir)
    agent.llm = llm
    agent._creative_llm = critic or _Scripted([])
    agent.gate_llm = _NoQuantities()
    agent.configure(self_critic=planned > 0, test_time_scale_round=planned, completion_gate=gate)
    agent.review_llm = None
    return agent


def test_fix_after_one_round(trace_dir: str) -> None:
    print("\n(a) a wrong sum fails, the critique fixes it, round 1 passes")
    from biomentis.run_quality import GATE_FEEDBACK_PREFIX

    llm = _Scripted(_EVIDENCE_STEPS + [_WRONG, _RIGHT])
    critic = _Scripted(["Check the sum."])
    agent = _agent(llm, critic, planned=1, trace_dir=trace_dir)
    _, final = agent.go("What is 24+3?")
    results = agent.run_gate_results
    _check("two gates ran", len(results) == 2, str(len(results)))
    _check("round 0 failed on arithmetic",
           results and not results[0].passed and [f.check for f in results[0].blocking] == ["arithmetic"],
           repr(results[0].findings) if results else "")
    _check("round 1 passed", len(results) == 2 and results[1].passed and results[1].round == 1,
           repr(results[1].findings) if len(results) == 2 else "")
    feedback = [m for m in llm.requests[3] if isinstance(m, HumanMessage)
                and m.content.startswith(GATE_FEEDBACK_PREFIX)] if len(llm.requests) > 3 else []
    _check("the model got the gate feedback", len(feedback) == 1, str(len(feedback)))
    if feedback:
        text = feedback[0].content
        _check("the feedback quotes the wrong sum", "24+3 = 28" in text, text)
        _check("the critic's text follows the feedback",
               "Check the sum." in text and text.index("24+3 = 28") < text.index("Check the sum."), text)
    _check("the critic ran once", len(critic.requests) == 1, str(len(critic.requests)))
    _check("the corrected answer is final", "24+3 = 27" in final, final)
    _check("status: not fully checked (reviewer off)",
           agent.final_status == "⚠️ Not fully checked (reviewer off)", agent.final_status)
    _check("reviewer state off", all(r.reviewer_state == "off" for r in results), "")
    _check("quantities extracted with the gate model", agent.gate_llm.calls == 2, str(agent.gate_llm.calls))


def test_pause_then_stop() -> None:
    print("\n(b) the error stays: pause before round 4, Stop ends the run")
    llm = _Scripted(_EVIDENCE_STEPS + [_WRONG] * 4)
    critic = _Scripted(["Check the sum."])
    agent = _agent(llm, critic, planned=1)
    done = threading.Event()
    box: dict = {}

    def run():
        try:
            box["final"] = agent.go("What is 24+3?")[1]
        except Exception as exc:  # reported below
            box["error"] = exc
        finally:
            done.set()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    deadline = time.time() + 120
    while agent.round_control.pending is None and not done.is_set() and time.time() < deadline:
        time.sleep(0.05)
    pending = agent.round_control.pending
    _check("the run paused", pending is not None, f"done={done.is_set()} error={box.get('error')!r}")
    _check("before round 4", pending is not None and pending["round"] == 4, str(pending))
    _check("the pause lists the blocking finding",
           pending is not None and [f.check for f in pending["findings"]] == ["arithmetic"], str(pending))
    _check("four gates ran before it", len(agent.run_gate_results) == 4, str(len(agent.run_gate_results)))
    agent.round_control.decide("stop")
    worker.join(timeout=60)
    _check("the run ended after Stop", not worker.is_alive() and "error" not in box, repr(box.get("error")))
    _check("status: stopped by user", agent.final_status == "⛔ Stopped by user — did not pass", agent.final_status)
    _check("no model call after the pause", len(llm.requests) == 6, str(len(llm.requests)))
    _check("the critic ran only in the planned round", len(critic.requests) == 1, str(len(critic.requests)))
    _check("no further pause requested", agent.round_control.pending is None, str(agent.round_control.pending))


def test_gate_off() -> None:
    print("\n(c) completion_gate=False keeps the old graph")
    llm = _Scripted(["<solution>The total is 24+3 = 28 items.</solution>"])
    agent = _agent(llm, planned=0, gate=False)
    _, final = agent.go("What is 24+3?")
    _check("one call, answer kept", len(llm.requests) == 1 and "24+3 = 28" in final, str(len(llm.requests)))
    _check("no gate ran", agent.run_gate_results == [] and agent.final_status == "", agent.final_status)
    _check("no gate nodes in the graph", "gate" not in agent.app.get_graph().nodes, str(agent.app.get_graph().nodes))
    _check("the old review node is there", "independent_review" in agent.app.get_graph().nodes, "")

    gated = _agent(_Scripted([]), planned=0, gate=True)
    nodes = set(gated.app.get_graph().nodes)
    _check("gate graph nodes", {"draft_rubric", "generate", "execute", "gate", "critique", "await_decision",
                                "finalize"} <= nodes and "independent_review" not in nodes, str(nodes))


def test_trace(trace_dir: str) -> None:
    print("\n(d) gate records in the step trace")
    from biomentis.eval.step_trace import build_report, format_report, load_traces

    records = load_traces(trace_dir)
    gates = [r for r in records if r["type"] == "gate"]
    _check("two gate records", len(gates) == 2, str(gates))
    if len(gates) == 2:
        _check("the first failed on arithmetic",
               gates[0]["passed"] is False and gates[0]["blocking"] == ["arithmetic"] and gates[0]["round"] == 0,
               str(gates[0]))
        _check("the second passed", gates[1]["passed"] is True and gates[1]["round"] == 1, str(gates[1]))
        _check("record fields", set(gates[0]) >= {"round", "passed", "reviewer_state", "blocking", "warnings"},
               str(gates[0]))
    report = build_report(records)
    _check("report counts gate failures by check", report.get("gate_failures") == {"arithmetic": 1},
           str(report.get("gate_failures")))
    text = format_report(report)
    _check("the text report shows them", "arithmetic" in text and "gate" in text.lower(), text[-600:])


def _wrap_up_fit(agent):
    from biomentis.context_budget import WRAP_UP_INSTRUCTION

    calls = {"n": 0}

    def fit(messages):
        calls["n"] += 1
        agent._context_wrap_ups = getattr(agent, "_context_wrap_ups", 0) + 1
        return [*messages, HumanMessage(content=WRAP_UP_INSTRUCTION)], SimpleNamespace(
            must_wrap_up=True, must_hard_stop_after=lambda n: False, changed=False)

    agent._fit_request = fit
    return calls


def _run_in_thread(agent, prompt):
    box: dict = {}

    def run():
        try:
            box["final"] = agent.go(prompt)[1]
        except Exception as exc:
            box["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    deadline = time.time() + 60
    while agent.round_control.pending is None and worker.is_alive() and time.time() < deadline:
        time.sleep(0.05)
    return worker, box


def test_context_pause() -> None:
    print("\n(e) the context-limit pause")
    llm = _Scripted([])
    agent = _agent(llm, planned=0)
    _wrap_up_fit(agent)
    worker, box = _run_in_thread(agent, "What is 24+3?")
    pending = agent.round_control.pending
    _check("paused for the context", pending is not None and pending["reason"] == "context"
           and pending["round"] == 0 and pending["findings"] == [], str(pending))
    agent.round_control.decide("stop")
    worker.join(timeout=30)
    _check("Stop ends the run", not worker.is_alive() and "error" not in box, repr(box.get("error")))
    _check("without calling the model", llm.requests == [], str(len(llm.requests)))
    _check("status: stopped by user", agent.final_status == "⛔ Stopped by user — did not pass", agent.final_status)

    llm = _Scripted(_EVIDENCE_STEPS + [_RIGHT])
    agent = _agent(llm, planned=0)
    _wrap_up_fit(agent)
    worker, box = _run_in_thread(agent, "What is 24+3?")
    _check("paused again in a new run", agent.round_control.pending is not None, str(box))
    agent.round_control.decide("continue")
    worker.join(timeout=60)
    _check("Continue proceeds to the answer", not worker.is_alive() and "24+3 = 27" in box.get("final", ""),
           repr(box))
    _check("asked once per run", agent.round_control.pending is None and len(llm.requests) == 3, str(len(llm.requests)))
    from biomentis.context_budget import WRAP_UP_INSTRUCTION
    _check("the wrap-up instruction was sent",
           llm.requests and llm.requests[0][-1].content == WRAP_UP_INSTRUCTION, "")


def test_keys() -> None:
    print("\n(f) finding keys")
    from biomentis.gate import Finding

    a = Finding("quantity", "blocking", "total 27", "derivation is wrong")
    b = Finding("quantity", "warning", "total 27", "not found in any code output")
    _check("same quote, different issue: different keys", a.key != b.key, a.key)
    _check("same finding: same key", a.key == Finding("quantity", "blocking", "total 27", "derivation is wrong").key)


def test_feedback_message() -> None:
    print("\n(g) the gate feedback message")
    from biomentis.gate import Finding, GateResult
    from biomentis.gate.runner import gate_feedback
    from biomentis.run_quality import GATE_FEEDBACK_PREFIX, current_run, run_evidence

    result = GateResult(1, [Finding("figure", "warning", "Fig 2", "no title"),
                            Finding("arithmetic", "blocking", "24+3 = 28", "24+3 is 27, not 28", fix="24+3 = 27")], "off")
    text = gate_feedback(result)
    _check("starts with the prefix", text.startswith(GATE_FEEDBACK_PREFIX), text)
    _check("blocking before warnings", text.index("24+3 = 28") < text.index("Fig 2"), text)
    result.accepted.add(result.findings[1].key)
    _check("an accepted finding is not sent back", "24+3 = 28" not in gate_feedback(result), gate_feedback(result))
    msgs = [HumanMessage(content="Q"), AIMessage(content="<solution>A</solution>"), HumanMessage(content=text)]
    _check("it does not start a new run", current_run(msgs) == msgs, str(len(current_run(msgs))))
    _check("it counts as a critique round", run_evidence(msgs).critique_rounds == 1, "")


def main() -> int:
    trace_dir = tempfile.mkdtemp(prefix="biomentis_gate_trace_")
    test_fix_after_one_round(trace_dir)
    test_pause_then_stop()
    test_gate_off()
    test_trace(trace_dir)
    test_context_pause()
    test_keys()
    test_feedback_message()
    print(f"\n{len(_PASSED)} passed, {len(_FAILED)} failed")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
