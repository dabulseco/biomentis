"""Tests for the repeat-run tool in the sidebar ("R&D: repeat this prompt").

What these tests pin down:

  (a) The real app with `default_config.repeat_runs` False (BIOMNI_REPEAT_RUNS
      =false): no "Show repeat-run tool" toggle, no "Run this prompt K times"
      button, read at render time.
  (b) With it true: the toggle is shown (on by default) and so is the button;
      toggle off hides the button.
  (c) A stub batch with one paused run, registered for the app's session: a
      pause panel keyed by run index (`repeat2_`), and Continue calls that
      run's control. While the batch is active the button is disabled, and
      the interactive Send button is disabled too.
  (d) An interactive run in flight disables the button.
  (e) A real `RepeatBatch` over stub agents, started from the button: the
      rubric (an uploaded canonical one, so no model) is shown, the paused
      run gets its panel, Continue lets it finish, the progress table has
      run #, status and rounds, and the report can be downloaded.

Run with:
    python -m biomentis.agent.tests.test_repeat_runs_ui
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


_REPO = Path(__file__).resolve().parents[3]
TOGGLE = "Show repeat-run tool"
BUTTON = "Run this prompt K times"
SESSION = "repeat-ui-test-session"


def _buttons(at, label: str) -> list:
    return [b for b in at.button if b.label == label]


def _toggles(at, label: str) -> list:
    return [t for t in at.toggle if t.label == label]


def _wait_until(cond, limit: float = 10.0) -> bool:
    end = time.monotonic() + limit
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


class _StubBatch:
    """Just the surface the UI reads: run 1 finished, run 2 paused before round 4."""

    def __init__(self):
        from biomentis.agent.tests.test_repeat_runs import RUBRIC
        from biomentis.eval.repeat_report import RunOutcome
        from biomentis.gate import Finding
        from biomentis.gate.control import RoundControl

        self.k = 2
        self.rubric = RUBRIC
        self.report = None
        self.report_error = ""
        self.calls: list = []
        batch = self

        class Spy(RoundControl):
            def decide(self, choice, accepted_keys=frozenset()):
                batch.calls.append((choice, sorted(accepted_keys)))
                super().decide(choice, accepted_keys)

        self.control = Spy()
        self.control.request(4, [Finding("arithmetic", "blocking", "3 × 12 = 38", "3 × 12 is 36", key="a1")], "round")
        self._outcomes = [RunOutcome(index=1, run_id="run-01", final_status="✅ Passed all checks after 0 rounds",
                                     rounds_used=0, gate_results=[], final_answer="A", telemetry="")]

    def paused(self):
        return [(2, self.control)] if self.control.pending is not None else []

    def waiting(self):
        return ["Run 2: paused at round 4, waiting for Continue or Stop"] if self.control.pending else \
            ["Run 2: continued, waiting its turn"]

    def outcomes(self):
        return list(self._outcomes)

    def done(self):
        return False


def test_real_app() -> None:
    from streamlit.testing.v1 import AppTest

    from biomentis import ui_repeat
    from biomentis.config import default_config

    saved = default_config.repeat_runs
    at = AppTest.from_file(str(_REPO / "streamlit_app.py"), default_timeout=300)
    at.session_state["biomni_session_key"] = SESSION
    try:
        print("\n(a) BIOMNI_REPEAT_RUNS=false: no toggle, no button")
        os.environ["BIOMNI_REPEAT_RUNS"] = "false"
        default_config.repeat_runs = False
        at.run()
        _check("no exception", not at.exception, str(at.exception))
        _check("no toggle", not _toggles(at, TOGGLE), str([t.label for t in at.toggle]))
        _check("no button", not _buttons(at, BUTTON), str([b.label for b in at.button]))
        _check("no prompt box", not [t for t in at.text_area if t.key == "biomentis_repeat_prompt"])

        print("\n(b) true: toggle on by default shows the button; off hides it")
        os.environ["BIOMNI_REPEAT_RUNS"] = "true"
        default_config.repeat_runs = True
        at.run()
        _check("no exception", not at.exception, str(at.exception))
        toggles = _toggles(at, TOGGLE)
        _check("the toggle is shown", len(toggles) == 1 and toggles[0].key == "biomentis_show_repeat_runs",
               str([(t.label, t.key) for t in at.toggle]))
        _check("on by default", bool(toggles) and toggles[0].value is True)
        buttons = _buttons(at, BUTTON)
        _check("the button is shown", len(buttons) == 1, str([b.label for b in at.button]))
        k = [n for n in at.number_input if n.key == "biomentis_repeat_k"]
        _check("K defaults to 5, 1-10", len(k) == 1 and k[0].value == 5 and k[0].min == 1 and k[0].max == 10,
               str([(n.key, n.value) for n in at.number_input]))
        _check("empty prompt: the button is disabled", bool(buttons) and buttons[0].disabled)
        at.text_area(key="biomentis_repeat_prompt").set_value("What is 24+3?")
        at.run()
        _check("a prompt enables the button", _buttons(at, BUTTON) and not _buttons(at, BUTTON)[0].disabled,
               str([(b.label, b.disabled) for b in at.button]))
        if toggles:
            at.toggle(key="biomentis_show_repeat_runs").set_value(False)
            at.run()
            _check("toggle off: no button", not _buttons(at, BUTTON), str([b.label for b in at.button]))
            _check("toggle off: the toggle stays", len(_toggles(at, TOGGLE)) == 1)
            _toggles(at, TOGGLE)[0].set_value(True)
            at.run()

        print("\n(c) a stub batch with a paused run: panel keyed by run index, Continue calls its control")
        stub = _StubBatch()
        ui_repeat.set_entry(SESSION, ui_repeat.RepeatEntry("What is 24+3?", 2, batch=stub))
        at.run()
        _check("no exception", not at.exception, str(at.exception))
        warnings = [w.value for w in at.warning]
        _check("the pause panel is shown", any("Round 4 needs your decision" in w for w in warnings), str(warnings))
        keys = sorted(b.key for b in at.button if b.key and b.key.startswith("repeat"))
        _check("its buttons keyed by run index", keys == ["repeat2_round_continue", "repeat2_round_stop"], str(keys))
        boxes = [c.key for c in at.checkbox if c.key and c.key.startswith("repeat")]
        _check("its accept box keyed by run index", boxes == ["repeat2_accept_a1"], str(boxes))
        _check("the button is disabled while the batch runs",
               _buttons(at, BUTTON) and _buttons(at, BUTTON)[0].disabled, "")
        send = [b for b in at.button if b.label == "Send"]
        _check("Send is disabled while the batch runs", bool(send) and send[0].disabled,
               str([(b.label, b.disabled) for b in send]))
        frames = [d.value for d in at.dataframe]
        _check("a progress table: run #, status, rounds",
               bool(frames) and list(frames[0].columns) == ["Run #", "Status", "Rounds"]
               and list(frames[0]["Status"]) == ["✅ Passed all checks after 0 rounds", "paused at round 4"],
               str(frames[0] if frames else None))
        if keys:
            at.checkbox(key="repeat2_accept_a1").check()
            at.button(key="repeat2_round_continue").click().run()
            _check("Continue calls that run's control", stub.calls == [("continue", ["a1"])], str(stub.calls))
            frames = [d.value for d in at.dataframe]
            _check("the run is then continued, waiting its turn",
                   bool(frames) and list(frames[0]["Status"])[1] == "continued, waiting its turn",
                   str(frames[0] if frames else None))

        print("\n(d) an interactive run in flight disables the button")
        ui_repeat.clear_entry(SESSION)
        at.session_state["biomni_run_active"] = True
        at.run()
        buttons = _buttons(at, BUTTON)
        _check("disabled during an interactive run", bool(buttons) and buttons[0].disabled,
               str([(b.label, b.disabled) for b in at.button]))
        captions = " ".join(c.value for c in at.caption)
        _check("with a caption saying why", "interactive run" in captions, captions[:600])
        at.session_state["biomni_run_active"] = False
        at.run()
    finally:
        default_config.repeat_runs = saved
        ui_repeat.clear_entry(SESSION)
        try:
            tutor = at.session_state["biomni_tutor"]
            shutil.rmtree(tutor.kb.path, ignore_errors=True)
            shutil.rmtree(tutor.logger.dir, ignore_errors=True)
        except Exception:
            pass


_HARNESS = '''
import sys
sys.path.insert(0, {repo!r})
import streamlit as st
from pathlib import Path
from biomentis.agent.tests.test_repeat_runs import SCRIPTS, StubAgent
from biomentis.gate.rubric import format_rubric_markdown
from biomentis.agent.tests.test_repeat_runs import RUBRIC
from biomentis.ui_repeat import render_repeat_runs

st.session_state.setdefault("biomni_session_key", {session!r})
scripts = iter([SCRIPTS[0], SCRIPTS[1]])

def make_agent():
    agent = StubAgent(next(scripts))
    agent.llm = None
    return agent

with st.sidebar:
    render_repeat_runs(make_agent, format_rubric_markdown(RUBRIC), "rounds=1, completion gate on",
                       out_root=Path({out!r}))
'''


def test_real_batch() -> None:
    print("\n(e) a real batch over stub agents, started from the button")
    from streamlit.testing.v1 import AppTest

    from biomentis import ui_repeat

    tmp = tempfile.mkdtemp(prefix="biomni_repeat_ui_")
    session = "repeat-ui-real-batch"
    path = os.path.join(tmp, "app.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_HARNESS.format(repo=str(_REPO), session=session, out=os.path.join(tmp, "runs")))
    try:
        at = AppTest.from_file(path, default_timeout=30).run()
        _check("no exception", not at.exception, str(at.exception))
        at.text_area(key="biomentis_repeat_prompt").set_value("Design a 16S study.")
        at.number_input(key="biomentis_repeat_k").set_value(2)
        at.run()
        _buttons(at, BUTTON)[0].click().run()
        _check("no exception after start", not at.exception, str(at.exception))
        entry = ui_repeat.get_entry(session)
        _check("an entry in the registry", entry is not None)
        if entry is None:
            return
        started = _wait_until(lambda: entry.batch is not None and bool(entry.batch.paused()))
        _check("run 2 paused (bounded wait)", started, entry.error)
        if not started:
            return
        at.run()
        _check("the button is disabled while it runs", _buttons(at, BUTTON)[0].disabled)
        page = " ".join(m.value for m in at.markdown)
        _check("the rubric is shown", "[D1] A sample plan per soil" in page, page[:600])
        _check("run 2's panel", any(b.key == "repeat2_round_continue" for b in at.button),
               str([b.key for b in at.button]))
        at.button(key="repeat2_round_continue").click().run()
        finished = entry.batch.wait(10)
        _check("the batch finishes after Continue (bounded wait)", finished, str(entry.batch.waiting()))
        at.run()
        _check("no exception when done", not at.exception, str(at.exception))
        frames = [d.value for d in at.dataframe]
        rows = frames[0].to_dict("records") if frames else []
        _check("two rows with their final status and rounds",
               [r["Run #"] for r in rows] == [1, 2]
               and rows[0]["Status"].startswith("✅") and rows[1]["Rounds"] == 3,
               str(rows))
        downloads = at.get("download_button")
        _check("a report download", len(downloads) == 1, str(downloads))
        report = entry.batch.report
        captions = " ".join(c.value for c in at.caption)
        _check("the report path shown", report is not None and str(report) in captions, captions[:600])
        _check("the button is enabled again", not _buttons(at, BUTTON)[0].disabled)
    finally:
        ui_repeat.clear_entry(session)
        shutil.rmtree(tmp, ignore_errors=True)


def run_all() -> int:
    test_real_batch()
    test_real_app()
    print(f"\n{len(_PASSED)} passed, {len(_FAILED)} failed")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(run_all())
