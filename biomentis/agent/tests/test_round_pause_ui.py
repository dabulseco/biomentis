"""Tests for the completion gate in the UI: pause panel, gate events, final status.

What these tests pin down:

  (a) A paused run, driven through the REAL `A1.launch_streamlit_demo` render
      loop, shows the pause panel: "Round 4 needs your decision", both
      blocking findings with an "Accept as is" box each, Continue and Stop.
      The script does not block on the empty event queue while paused.
  (b) A rerun with no click keeps the panel, and the run stays paused.
  (c) Ticking one "Accept as is" and clicking Continue calls
      `decide("continue", {key})`, and the run proceeds to its end.
  (d) The final status is rendered above the final answer, and the exported
      main page begins with it.
  (e) The existing "Stop run" button, pressed during the pause, stops the
      control through `BackgroundRun.round_control` and the run ends.
  (f) Real graph, no UI: `start_run` gets the agent's control, the events
      carry one "🧪 Checks after round N" per gate and a round-pause event,
      and `cancel()` during the round-4 pause ends the run within a second
      or two as "⛔ Stopped by user — did not pass", with the run-quality
      summary's new lines.
  (g) `render_round_pause` on its own: the context reason, key prefixes
      (one panel per paused run in a batch), and Stop.
  (h) The sidebar: "Completion checks" replaces the evidence toggle, and
      `configure` gets `completion_gate=`.

Run with:
    python -m biomentis.agent.tests.test_round_pause_ui
"""

from __future__ import annotations

import glob
import os
import sys
import tempfile
import time

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
STOPPED = "⛔ Stopped by user — did not pass"

# The real render loop over a stub agent whose stream pauses like the gate
# graph does: it yields its events, requests a round-4 decision and blocks in
# `wait()` with no events until the user answers.
_HARNESS = '''
import sys, os
sys.path.insert(0, {repo!r})
os.environ["BIOMENTIS_RUN_DIR"] = {tmp!r}
os.environ["BIOMNI_DOWNLOADS_DIR"] = {tmp!r}
if {inline!r}:
    os.environ["BIOMENTIS_BACKGROUND_RUNS"] = "0"

import streamlit as st
from types import SimpleNamespace

from biomentis.agent.a1 import A1
from biomentis.gate import Finding
from biomentis.gate.control import RoundControl
from biomentis.run_worker import get_run
from biomentis.ui_core import UIEvent

SESSION = {session!r}
FINDINGS = [
    Finding("arithmetic", "blocking", "24+3 = 28", "24+3 is 27, not 28", "Recompute the sum in code.", key="k1"),
    Finding("figure", "blocking", "Figure 2", "Figure 2 is cited but was never drawn", key="k2"),
]


class SpyControl(RoundControl):
    def __init__(self):
        super().__init__()
        self.calls = []

    def decide(self, choice, accepted_keys=frozenset()):
        self.calls.append((choice, sorted(accepted_keys)))
        super().decide(choice, accepted_keys)


st.session_state.setdefault("biomni_session_key", SESSION)
st.session_state.setdefault("biomni_agent_key", "test")
if "_agent" not in st.session_state:
    st.session_state["_agent"] = SimpleNamespace(
        main_history_copy=[],
        path=os.path.join({tmp!r}, "agent"),
        llm=SimpleNamespace(model_name="test-model"),
        round_control=SpyControl(),
    )
agent = st.session_state["_agent"]


def fake_stream(agent, prompt, files, history, thread_id):
    yield UIEvent(type="solution", channel="main", content="the answer", title="Answer")
    yield UIEvent(type="round_pause", channel="main", title="⏸ Paused for your decision",
                  content="Round 4 needs your decision")
    # Like A1._await_user: inline, a run that needs a decision stops.
    if getattr(agent, "pause_allowed", True):
        agent.round_control.request(4, FINDINGS, "round")
        agent.round_control.wait()
    yield UIEvent(type="final_status", channel="main", content={stopped!r})
    yield UIEvent(type="complete", content="finished", title="Complete")


A1.launch_streamlit_demo(agent, stream_fn=fake_stream)

run = get_run(SESSION)
st.markdown(
    "RESULT alive=" + str(run is not None and run.is_alive())
    + " runner_control=" + str(run is not None and run.round_control is agent.round_control)
    + " active=" + str(st.session_state.get("biomni_run_active"))
)
st.markdown("CALLS=" + repr(agent.round_control.calls))
st.markdown("CONTENTS=" + repr([e.get("content") for e in st.session_state.biomni_transcript]))
st.markdown("OUTDIR=" + str(st.session_state.get("biomni_output_dir")))
'''


def _line(at, prefix: str) -> str:
    for m in at.markdown:
        if m.value.startswith(prefix):
            return m.value
    return "(missing)"


def _texts(node) -> list[str]:
    """Every element's text in render order, walking the whole tree."""
    out: list[str] = []
    children = getattr(node, "children", None)
    if children:
        for child in children.values():
            out.extend(_texts(child))
        return out
    icon = getattr(node, "icon", None)
    prefix = icon + " " if isinstance(icon, str) and icon and not icon.startswith(":") else ""
    for attr in ("value", "label", "body"):
        try:
            value = getattr(node, attr)
        except Exception:
            continue
        if isinstance(value, str) and value:
            out.append(prefix + value)
            break
    return out


def _paused_app(session: str, inline: bool = False):
    from streamlit.testing.v1 import AppTest

    tmp = tempfile.mkdtemp(prefix="biomni_round_pause_")
    path = os.path.join(tmp, "app.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_HARNESS.format(repo=_REPO, tmp=tmp, session=session, stopped=STOPPED, inline=inline))
    at = AppTest.from_file(path, default_timeout=30).run()
    at.text_area[0].set_value("the task")
    send = [b for b in at.button if b.label == "Send"]
    assert send, f"no Send button among {[b.label for b in at.button]}"
    t0 = time.monotonic()
    at = send[0].click().run()
    return at, tmp, time.monotonic() - t0


def _panel_shown(at) -> tuple[bool, str]:
    warnings = [w.value for w in at.warning]
    return any("Round 4 needs your decision" in w for w in warnings), str(warnings)


def _wait_until(cond, limit: float = 5.0) -> bool:
    end = time.monotonic() + limit
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


def test_panel_continue_and_export() -> None:
    print("\n(a) a paused run shows the pause panel")
    at, tmp, took = _paused_app("round-pause-continue")
    _check("no exception", not at.exception, str(at.exception))
    _check("the script did not block while paused", took < 20, f"{took:.1f}s")
    shown, detail = _panel_shown(at)
    _check("reason: Round 4 needs your decision", shown, detail)
    boxes = [c for c in at.checkbox if c.label == "Accept as is"]
    _check("two Accept as is boxes", len(boxes) == 2, str([c.label for c in at.checkbox]))
    _check("keyed accept_<finding.key>", sorted(c.key for c in boxes) == ["accept_k1", "accept_k2"],
           str([c.key for c in boxes]))
    page = " ".join(_texts(at.main))
    _check("both findings listed", "24+3 is 27, not 28" in page and "Figure 2 is cited" in page, page[:800])
    labels = [b.label for b in at.button]
    _check("Continue and Stop buttons", "Continue" in labels and "Stop" in labels, str(labels))
    _check("the runner holds the agent's control", "runner_control=True" in _line(at, "RESULT"),
           _line(at, "RESULT"))
    _check("still running", "alive=True" in _line(at, "RESULT"), _line(at, "RESULT"))

    print("\n(b) a rerun with no click keeps the panel")
    at.run()
    _check("no exception after rerun", not at.exception, str(at.exception))
    shown, detail = _panel_shown(at)
    _check("panel survives the rerun", shown, detail)
    _check("no decision was made", _line(at, "CALLS=") == "CALLS=[]", _line(at, "CALLS="))
    _check("still running after rerun", "alive=True" in _line(at, "RESULT"), _line(at, "RESULT"))

    print("\n(c) accept one, Continue")
    at.checkbox(key="accept_k1").check()
    at.button(key="round_continue").click()
    at.run()
    _check("no exception after Continue", not at.exception, str(at.exception))
    _check("decide('continue', {k1})", "('continue', ['k1'])" in _line(at, "CALLS="), _line(at, "CALLS="))
    # The worker resumes on its own thread; a rerun picks up its events.
    for _ in range(20):
        if "alive=False" in _line(at, "RESULT"):
            break
        time.sleep(0.1)
        at.run()
    _check("the run finished", "alive=False" in _line(at, "RESULT"), _line(at, "RESULT"))
    shown, _ = _panel_shown(at)
    _check("the panel is gone", not shown, str([w.value for w in at.warning]))

    print("\n(d) the final status sits above the answer, and the export begins with it")
    at.run()  # a settled page, with nothing left over from the redraws
    contents = _line(at, "CONTENTS=")
    _check("status before answer in the transcript",
           STOPPED in contents and "the answer" in contents
           and contents.index(STOPPED) < contents.index("'the answer'"), contents)
    page = _texts(at.main)
    joined = "\n".join(page)
    _check("status rendered before the answer",
           STOPPED in joined and "the answer" in joined and joined.index(STOPPED) < joined.index("the answer"),
           joined[:1500])
    export = [b for b in at.button if b.label.startswith("📥 Export run")]
    _check("export button present", bool(export), str([b.label for b in at.button]))
    if export:
        at = export[0].click().run()
        _check("no exception on export", not at.exception, str(at.exception))
        outdir = _line(at, "OUTDIR=").removeprefix("OUTDIR=")
        mains = glob.glob(os.path.join(outdir, "biomentis_main_*.html"))
        _check("main page written", len(mains) == 1, str(mains))
        if mains:
            with open(mains[0], encoding="utf-8") as f:
                html_text = f.read()
            body = html_text[html_text.index('<div class="d-flex flex-column">'):]
            _check("main export begins with the status",
                   STOPPED in body and body.index(STOPPED) < body.index("the task") < body.index("the answer"),
                   body[:1200])


def test_stop_run_button_during_pause() -> None:
    print("\n(e) the Stop run button during the pause")
    at, _, _ = _paused_app("round-pause-stop-run")
    shown, detail = _panel_shown(at)
    _check("paused", shown, detail)
    stop = [b for b in at.button if b.key == "biomni_stop_run"]
    _check("Stop run button shown while paused", bool(stop), str([b.key for b in at.button]))
    if not stop:
        return
    t0 = time.monotonic()
    at = stop[0].click().run()
    _check("no exception after Stop run", not at.exception, str(at.exception))
    _check("the control was stopped", "('stop', [])" in _line(at, "CALLS="), _line(at, "CALLS="))
    for _ in range(20):
        if "alive=False" in _line(at, "RESULT"):
            break
        time.sleep(0.1)
        at.run()
    _check("the run ended promptly", "alive=False" in _line(at, "RESULT") and time.monotonic() - t0 < 10,
           _line(at, "RESULT"))
    _check("the stopped status reached the transcript", STOPPED in _line(at, "CONTENTS="), _line(at, "CONTENTS="))


def test_real_graph_cancel_during_pause() -> None:
    print("\n(f) real graph: gate events, pause event, cancel during the pause")
    from biomentis.agent.tests.test_gate_graph import _EVIDENCE_STEPS, _WRONG, _agent, _Scripted
    from biomentis.run_worker import clear_run, start_run
    from biomentis.ui_core import stream_agent_events

    llm = _Scripted(_EVIDENCE_STEPS + [_WRONG] * 4)
    agent = _agent(llm, _Scripted(["Check the sum."] * 3), planned=1)
    runner = start_run(
        "real-graph-pause",
        "What is 24+3?",
        lambda: stream_agent_events(agent, "What is 24+3?", [], [], 91),
        round_control=agent.round_control,
    )
    _check("start_run sets the runner's control", runner.round_control is agent.round_control)
    paused = _wait_until(lambda: agent.round_control.pending is not None, 120)
    _check("the run paused before round 4", paused and agent.round_control.pending["round"] == 4,
           str(agent.round_control.pending))
    _wait_until(lambda: any(e.type == "round_pause" for e in runner.events), 5)
    titles = [e.title for e in runner.events if e.type == "gate_result"]
    _check("one gate_result per gate", titles == [f"🧪 Checks after round {n}" for n in range(4)], str(titles))
    gate_events = [e for e in runner.events if e.type == "gate_result"]
    _check("gate results go to the inner panel", all(e.channel == "inner" for e in gate_events))
    _check("gate result lists the finding",
           bool(gate_events) and "arithmetic" in gate_events[0].content and "24+3 = 28" in gate_events[0].content,
           gate_events[0].content if gate_events else "")
    pauses = [e for e in runner.events if e.type == "round_pause"]
    _check("a round_pause event on the main channel",
           len(pauses) == 1 and pauses[0].channel == "main" and "Round 4 needs your decision" in pauses[0].content,
           repr(pauses))
    reasoning = [e.content for e in runner.events if e.type == "reasoning"]
    _check("no reasoning repeated by the gate nodes", len(reasoning) == len(set(reasoning)), str(reasoning))

    t0 = time.monotonic()
    runner.cancel()
    ended = _wait_until(lambda: runner.done, 5)
    took = time.monotonic() - t0
    _check("cancel ends the paused run promptly", ended and took < 2.0, f"{took:.2f}s")
    _check("no model call after the pause", len(llm.requests) == 6, str(len(llm.requests)))
    statuses = [e for e in runner.events if e.type == "final_status"]
    _check("final_status event: stopped",
           len(statuses) == 1 and statuses[0].content == STOPPED and statuses[0].channel == "main", repr(statuses))
    quality = [e.content for e in runner.events if e.type == "status" and e.title == "📊 Run quality"]
    q = quality[0] if quality else ""
    _check("summary: status", f"**Status:** {STOPPED}" in q, q)
    _check("summary: rounds", "**Rounds:** used 3 of planned 1" in q, q)
    _check("summary: accepted", "**Accepted as is:** 0 findings" in q, q)
    _check("summary: rubric (the helper presets an uploaded one)", "**Rubric:** uploaded" in q, q)
    clear_run("real-graph-pause")


_PANEL_HARNESS = '''
import sys
sys.path.insert(0, {repo!r})
import streamlit as st
from biomentis.gate import Finding
from biomentis.gate.control import RoundControl
from biomentis.ui_gate import render_round_pause

if "_c" not in st.session_state:
    c = RoundControl()
    c.request(2, [Finding("quantity", "blocking", "x = 5", "x is 4 elsewhere", key="q1")], "context")
    other = RoundControl()
    other.request(5, [Finding("figure", "blocking", "Fig 1", "no figure", key="f1")], "round")
    idle = RoundControl()
    st.session_state["_c"] = (c, other, idle)
c, other, idle = st.session_state["_c"]
render_round_pause(c, "run1_")
render_round_pause(other, "run2_")
render_round_pause(idle, "run3_")
st.markdown("DECIDED=" + repr((c._decision, other._decision)))
'''


def test_render_round_pause_alone() -> None:
    print("\n(g) render_round_pause: context reason, prefixes, Stop")
    from streamlit.testing.v1 import AppTest

    path = os.path.join(tempfile.mkdtemp(prefix="biomni_panel_"), "app.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_PANEL_HARNESS.format(repo=_REPO))
    at = AppTest.from_file(path, default_timeout=30).run()
    _check("no exception", not at.exception, str(at.exception))
    warnings = [w.value for w in at.warning]
    _check("context reason text", any("The context window is nearly full" in w for w in warnings), str(warnings))
    _check("round reason text", any("Round 5 needs your decision" in w for w in warnings), str(warnings))
    _check("only two panels (idle control renders nothing)", len(warnings) == 2, str(warnings))
    keys = sorted(c.key for c in at.checkbox)
    _check("checkbox keys carry the prefix", keys == ["run1_accept_q1", "run2_accept_f1"], str(keys))
    bkeys = sorted(b.key for b in at.button)
    _check("button keys carry the prefix",
           bkeys == ["run1_round_continue", "run1_round_stop", "run2_round_continue", "run2_round_stop"], str(bkeys))
    at.button(key="run2_round_stop").click().run()
    _check("Stop stops only that run", _line(at, "DECIDED=") == "DECIDED=(None, 'stop')", _line(at, "DECIDED="))
    warnings = [w.value for w in at.warning]
    _check("a decided panel is no longer shown", len(warnings) == 1, str(warnings))


def test_sidebar_toggle() -> None:
    print("\n(h) sidebar: Completion checks")
    with open(os.path.join(_REPO, "streamlit_app.py"), encoding="utf-8") as f:
        src = f.read()
    _check("the old evidence toggle is gone", "Require evidence before answering" not in src)
    _check("Completion checks toggle", '"Completion checks"' in src and 'key="biomentis_completion_gate"' in src)
    _check("default from config", "value=default_config.completion_gate" in src)
    help_text = (
        "Checks every answer (arithmetic, figures, sources, labels, consistency and, if a reviewer is set, "
        "rubric coverage). Up to 3 revision rounds run automatically; from round 4 the run pauses for you "
        "to continue or stop."
    )
    flat = " ".join(src.split())
    joined = flat.replace('" "', "")
    _check("help text verbatim", help_text in joined, "")
    _check("configure gets completion_gate", "completion_gate=st.session_state.get(\"biomentis_completion_gate\"" in flat)
    _check("warning when background runs are off",
           "if not background_runs_enabled():" in flat
           and "Background runs are off (BIOMENTIS_BACKGROUND_RUNS=0): a run that needs your decision at "
           "round 4 will stop instead of pausing." in joined)
    _check("evidence_gate keeps the same value",
           "evidence_gate=st.session_state.get(\"biomentis_completion_gate\"" in flat)


def test_status_placement_and_restore() -> None:
    print("\n(i) status placement: live helper and a restored journal")
    from biomentis.run_journal import entries_for_run
    from biomentis.ui_core import UIEvent, add_transcript_entry, transcript_entry_for_event

    status = transcript_entry_for_event(UIEvent(type="final_status", channel="main", content=STOPPED))
    t = [{"panel": "main", "role": "user", "content": "q"}]
    _check("no answer yet: appended", add_transcript_entry(t, dict(status)) is False and t[-1]["content"] == STOPPED)
    t = [{"panel": "main", "role": "user", "content": "q"},
         transcript_entry_for_event(UIEvent(type="solution", channel="main", content="a0", title="A")),
         {"panel": "user", "role": "user", "content": "q2"},
         transcript_entry_for_event(UIEvent(type="solution", channel="main", content="a1", title="A")),
         transcript_entry_for_event(UIEvent(type="solution", channel="main", content="a2", title="A")),
         transcript_entry_for_event(UIEvent(type="status", channel="main", content="quality"))]
    _check("inserted above this turn's last answer", add_transcript_entry(t, dict(status)) is True
           and [e["content"] for e in t] == ["q", "a0", "q2", "a1", STOPPED, "a2", "quality"],
           str([e["content"] for e in t]))
    run = {"prompt": "q", "events": [
        {"type": "solution", "channel": "main", "content": "ans", "title": "A"},
        {"type": "final_status", "channel": "main", "content": STOPPED},
    ]}
    restored = [e["content"] for e in entries_for_run(run)]
    _check("a restored run puts the status above the answer", restored == ["q", STOPPED, "ans"], str(restored))


def test_inline_run_stops_instead_of_pausing() -> None:
    print("\n(j) background runs off: a run that needs a decision stops, it does not hang")
    import threading

    from biomentis.agent.tests.test_gate_graph import _EVIDENCE_STEPS, _WRONG, _agent, _Scripted
    from biomentis.ui_core import stream_agent_events

    llm = _Scripted(_EVIDENCE_STEPS + [_WRONG] * 4)
    agent = _agent(llm, _Scripted(["Check the sum."] * 3), planned=1)
    agent.pause_allowed = False
    box: dict = {}

    def run():
        try:
            box["events"] = list(stream_agent_events(agent, "What is 24+3?", [], [], 92))
        except Exception as exc:  # reported below
            box["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=120)
    _check("the run ended (bounded join)", not worker.is_alive() and "error" not in box, repr(box.get("error")))
    _check("it never paused", agent.round_control.pending is None)
    events = box.get("events") or []
    statuses = [e.content for e in events if e.type == "final_status"]
    _check("final status: stopped", statuses == [STOPPED], str(statuses))
    _check("no round_pause event", not any(e.type == "round_pause" for e in events))
    _check("four gates ran, no model call after round 3",
           len(agent.run_gate_results) == 4 and len(llm.requests) == 6,
           f"{len(agent.run_gate_results)} gates, {len(llm.requests)} calls")

    print("\n(j2) the UI's inline path sets pause_allowed False")
    t0 = time.monotonic()
    at, _, _ = _paused_app("round-pause-inline", inline=True)
    _check("no exception", not at.exception, str(at.exception))
    _check("the inline script finished promptly", time.monotonic() - t0 < 25, f"{time.monotonic() - t0:.1f}s")
    _check("no pause panel", not _panel_shown(at)[0], str([w.value for w in at.warning]))
    _check("the stopped status reached the transcript", STOPPED in _line(at, "CONTENTS="), _line(at, "CONTENTS="))


_DUP_HARNESS = '''
import sys
sys.path.insert(0, {repo!r})
import streamlit as st
from biomentis.gate import Finding
from biomentis.gate.control import RoundControl
from biomentis.ui_gate import render_round_pause

if "_c" not in st.session_state:
    c = RoundControl()
    same = Finding("arithmetic", "blocking", "2+2 = 5", "2+2 is 4")
    c.request(4, [same, Finding("arithmetic", "blocking", "2+2 = 5", "2+2 is 4")], "round")
    st.session_state["_c"] = c
render_round_pause(st.session_state["_c"])
st.markdown("ACCEPTED=" + repr(sorted(st.session_state["_c"].accepted)))
'''


def test_duplicate_findings_share_one_box() -> None:
    print("\n(k) two identical findings: one box, no duplicate widget id")
    from streamlit.testing.v1 import AppTest

    from biomentis.gate import Finding

    key = Finding("arithmetic", "blocking", "2+2 = 5", "2+2 is 4").key
    path = os.path.join(tempfile.mkdtemp(prefix="biomni_dup_"), "app.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_DUP_HARNESS.format(repo=_REPO))
    at = AppTest.from_file(path, default_timeout=30).run()
    _check("no exception", not at.exception, str(at.exception))
    boxes = [c.key for c in at.checkbox]
    _check("one Accept as is box", boxes == [f"accept_{key}"], str(boxes))
    if boxes:
        at.checkbox(key=f"accept_{key}").check()
        at.button(key="round_continue").click().run()
        _check("accepting it accepts the key", _line(at, "ACCEPTED=") == f"ACCEPTED={[key]!r}", _line(at, "ACCEPTED="))


def run_all() -> int:
    test_status_placement_and_restore()
    test_render_round_pause_alone()
    test_panel_continue_and_export()
    test_stop_run_button_during_pause()
    test_real_graph_cancel_during_pause()
    test_sidebar_toggle()
    test_inline_run_stops_instead_of_pausing()
    test_duplicate_findings_share_one_box()
    print(f"\n{len(_PASSED)} passed, {len(_FAILED)} failed")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(run_all())
