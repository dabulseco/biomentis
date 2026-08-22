"""Tests for surviving a stray click: the background worker and the journal.

A research run used to be a `for event in stream(...)` loop on the Streamlit
script thread. Streamlit stops a running script by raising `RerunException`
from the next `st.*` call, so any click anywhere on the page — the sidebar
model picker, an export button, a tab — tore the loop down, dropped the only
reference to the generator, and garbage-collected twenty minutes of work.

`biomentis/run_worker.py` moves the stream onto a thread the registry owns, and
`biomentis/run_journal.py` writes every event to disk as it is produced. What
these tests pin down:

  1. The worker keeps producing while nothing is consuming — the whole claim
  2. A consumer that dies mid-run and re-attaches gets the remainder exactly
     once: no gaps, no repeats
  3. Cancel stops at the next event boundary and keeps what was produced
  4. A worker that raises records the error instead of vanishing
  5. The journal is complete and readable while the run is still going
  6. A journal with no `run_end` reads back as "interrupted", which is the
     case worth finding after a crash
  7. `code_script` recovers every generated code block
  8. `transcript_entry_for_event` is the single mapping used by both the live
     UI and a restored journal, so a restored run renders identically
  9. `launch_streamlit_demo` re-attaches to an in-flight run on a rerun that
     submitted nothing, renders only the events past the cursor, and does not
     re-render the ones already in the transcript

Run with:
    python -m biomentis.agent.tests.test_background_run
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time

from biomentis.run_journal import RunJournal, code_script, entries_for_run, list_runs, load_run
from biomentis.run_worker import BackgroundRun, background_runs_enabled, clear_run, get_run, start_run
from biomentis.ui_core import UIEvent, transcript_entry_for_event

# --- Tiny test harness (matches test_smoke_e2e.py) -----------------------

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


def _wait_for(predicate, timeout: float = 5.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _events(n: int) -> list[UIEvent]:
    return [UIEvent(type="status", content=f"step {i}") for i in range(n)]


# --- 1-4: the worker ------------------------------------------------------


def test_worker_runs_without_a_consumer() -> None:
    """The claim the whole design rests on: nobody is reading, and the agent
    keeps working anyway."""
    print("\n[1] the worker produces while nothing consumes")
    gate = threading.Event()

    def stream():
        yield UIEvent(type="status", content="a")
        gate.wait(5)
        yield UIEvent(type="status", content="b")

    run = BackgroundRun("task").start(stream)
    _check("first event lands with no consumer attached", _wait_for(lambda: len(run.events) == 1))
    _check("run is still alive while blocked", run.is_alive())

    # This is the stray click: whatever was consuming is gone. The worker
    # neither knows nor cares.
    gate.set()
    _check("second event still arrives", _wait_for(lambda: len(run.events) == 2))
    _check("run finishes on its own", _wait_for(lambda: run.done))
    _check("no error recorded", run.error is None, repr(run.error))


def test_reattach_after_teardown_is_exact() -> None:
    print("\n[2] a torn-down consumer re-attaches without gaps or repeats")
    run = BackgroundRun("task").start(lambda: iter(_events(5)))
    _wait_for(lambda: run.done)

    # First consumer reads two events, then "dies" (breaks out) the way a
    # RerunException would take it out.
    seen_first = []
    cursor = 0
    for event in run.events_from(cursor):
        seen_first.append(event.content)
        cursor += 1
        if cursor == 2:
            break

    seen_second = [e.content for e in run.events_from(cursor)]
    _check("first consumer saw the head", seen_first == ["step 0", "step 1"], str(seen_first))
    _check(
        "second consumer saw exactly the tail",
        seen_second == ["step 2", "step 3", "step 4"],
        str(seen_second),
    )
    _check("nothing was delivered twice", len(seen_first) + len(seen_second) == 5)
    _check("pending_from reports what a consumer still owes", run.pending_from(2) == 3)


def test_cancel_stops_at_the_next_boundary() -> None:
    print("\n[3] cancel stops the run and keeps what it produced")
    started = threading.Event()

    def stream():
        for i in range(50):
            started.set()
            yield UIEvent(type="status", content=f"step {i}")
            time.sleep(0.01)

    run = BackgroundRun("task").start(stream)
    started.wait(5)
    run.cancel()
    _check("run ends", _wait_for(lambda: run.done))
    _check("marked cancelled", run.cancelled)
    _check("kept the work it had already done", len(run.events) >= 1, f"{len(run.events)} events")
    _check("did not run to completion", len(run.events) < 50, f"{len(run.events)} events")


def test_worker_error_is_captured() -> None:
    print("\n[4] a worker that raises records the error rather than vanishing")

    def stream():
        yield UIEvent(type="status", content="a")
        raise RuntimeError("agent exploded")

    with tempfile.TemporaryDirectory() as tmp:
        journal = RunJournal("task", tmp)
        run = BackgroundRun("task", journal=journal).start(stream)
        _wait_for(lambda: run.done)
        _check("error is on the run object", isinstance(run.error, RuntimeError), repr(run.error))
        _check("the event before the failure is kept", [e.content for e in run.events] == ["a"])

        loaded = load_run(journal.path)
        _check("journal records the failure", loaded["status"] == "error", loaded["status"])
        _check(
            "journal detail names the exception",
            "agent exploded" in (loaded.get("detail") or ""),
            repr(loaded.get("detail")),
        )


# --- 5-7: the journal -----------------------------------------------------


def test_journal_is_readable_mid_run() -> None:
    print("\n[5] the journal is complete and readable while the run is going")
    gate = threading.Event()

    def stream():
        yield UIEvent(type="code", content="print('hi')", language="python", title="🛠️ Code")
        gate.wait(5)
        yield UIEvent(type="solution", channel="main", content="the answer", title="✅ Answer")

    with tempfile.TemporaryDirectory() as tmp:
        journal = RunJournal("find the answer", tmp, mode="research")
        run = BackgroundRun("find the answer", journal=journal).start(stream)
        _wait_for(lambda: len(run.events) == 1)

        # Read it from disk with the run still in flight — this is what makes
        # a crash survivable rather than merely logged.
        mid = load_run(journal.path)
        _check("prompt is on disk", mid["prompt"] == "find the answer", mid["prompt"])
        _check("the first event is on disk", len(mid["events"]) == 1, str(mid["events"]))
        _check("an unfinished run reads as interrupted", mid["status"] == "interrupted", mid["status"])
        _check("run metadata is kept", mid["meta"].get("mode") == "research", str(mid["meta"]))

        gate.set()
        _wait_for(lambda: run.done)
        final = load_run(journal.path)
        _check("both events are on disk", len(final["events"]) == 2, str(len(final["events"])))
        _check("a finished run reads as complete", final["status"] == "complete", final["status"])
        _check(
            "content survives the round trip",
            final["events"][1]["content"] == "the answer",
            str(final["events"][1]),
        )

        summaries = list_runs(tmp)
        _check("list_runs finds it", len(summaries) == 1, str(summaries))
        _check("summary counts the code blocks", summaries[0]["code_blocks"] == 1, str(summaries[0]))
        _check("summary knows there is an answer", summaries[0]["has_answer"] is True, str(summaries[0]))


def test_code_script_recovers_every_block() -> None:
    print("\n[6] every generated code block comes back as one file")
    with tempfile.TemporaryDirectory() as tmp:
        journal = RunJournal("build something", tmp)
        journal.append(UIEvent(type="code", content="import pandas as pd", language="python"))
        journal.append(UIEvent(type="observation", content="ok"))
        journal.append(UIEvent(type="code", content="echo hello", language="bash"))
        journal.finish("complete")

        script = code_script(load_run(journal.path))
        _check("python block is verbatim", "import pandas as pd" in script, script)
        _check("non-python block is preserved", "echo hello" in script, script)
        _check("the task is recorded in the header", "build something" in script, script[:300])
        _check("block count is stated", "2 code block(s)" in script, script[:300])
        _check(
            "the file is valid python",
            _compiles(script),
            script,
        )


def _compiles(source: str) -> bool:
    try:
        compile(source, "<code_script>", "exec")
    except SyntaxError:
        return False
    return True


def test_restored_entries_match_live_entries() -> None:
    print("\n[7] a restored run renders identically to the live one")
    live_events = [
        UIEvent(type="status", content="starting"),
        UIEvent(type="reasoning", content="thinking", title="🤔 Reasoning"),
        UIEvent(type="code", content="print(1)", language="python", title="🛠️ Code"),
        UIEvent(type="observation", content="1", duration=1.5, collapsible=True),
        UIEvent(type="solution", channel="main", content="answer", title="✅ Answer"),
        UIEvent(type="complete", content="done", title="🔄 Complete"),
    ]

    live_code_entries: list[dict] = []
    live = [transcript_entry_for_event(e, live_code_entries) for e in live_events]
    live = [e for e in live if e is not None]

    with tempfile.TemporaryDirectory() as tmp:
        journal = RunJournal("a task", tmp)
        for event in live_events:
            journal.append(event)
        journal.finish("complete")
        restored = entries_for_run(load_run(journal.path))

    # The restore prepends the user's prompt, which the live path renders
    # separately at submit time.
    _check("restore leads with the prompt", restored[0].get("role") == "user", str(restored[0]))
    _check(
        "every event round-trips to the same entry",
        restored[1:] == live,
        f"\n  restored={restored[1:]}\n  live={live}",
    )
    _check(
        "the observation retro-titled its code block in both",
        live[2]["title"] == "🛠️ Code (done in 1.50s)" == restored[3]["title"],
        f"{live[2]['title']!r} vs {restored[3]['title']!r}",
    )


def test_background_runs_can_be_disabled() -> None:
    print("\n[8] the opt-out env var is honored")
    previous = os.environ.get("BIOMENTIS_BACKGROUND_RUNS")
    try:
        os.environ.pop("BIOMENTIS_BACKGROUND_RUNS", None)
        _check("on by default", background_runs_enabled() is True)
        os.environ["BIOMENTIS_BACKGROUND_RUNS"] = "0"
        _check("BIOMENTIS_BACKGROUND_RUNS=0 turns it off", background_runs_enabled() is False)
    finally:
        if previous is None:
            os.environ.pop("BIOMENTIS_BACKGROUND_RUNS", None)
        else:
            os.environ["BIOMENTIS_BACKGROUND_RUNS"] = previous


def test_registry_round_trip() -> None:
    print("\n[9] the registry keeps a run across script executions")
    key = "session-under-test"
    run = start_run(key, "task", lambda: iter(_events(2)))
    _check("get_run finds it by session key", get_run(key) is run)
    _wait_for(lambda: run.done)
    clear_run(key)
    _check("clear_run forgets it", get_run(key) is None)


# --- 10: the real render loop --------------------------------------------

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Drives the REAL `A1.launch_streamlit_demo` render loop over a
# pre-seeded background run, which is what a rerun lands in after a stray
# click: no prompt submitted, a worker already holding events.
_HARNESS = '''
import sys, os
sys.path.insert(0, {repo!r})
os.environ["BIOMENTIS_RUN_DIR"] = {run_dir!r}

import streamlit as st
from types import SimpleNamespace

from biomentis.agent.a1 import A1
from biomentis.run_worker import get_run, start_run
from biomentis.ui_core import UIEvent

SESSION = "apptest-session"
EVENTS = [
    UIEvent(type="status", content="starting"),
    UIEvent(type="code", content="print(1)", language="python", title="Code"),
    UIEvent(type="observation", content="1", duration=0.5),
    UIEvent(type="solution", channel="main", content="the answer", title="Answer"),
    UIEvent(type="complete", content="done", title="Complete"),
]

st.session_state.setdefault("biomni_session_key", SESSION)
# Skip the in-method model picker, which would need a real llm object.
st.session_state.setdefault("biomni_agent_key", "test")

if not st.session_state.get("_seeded"):
    st.session_state["_seeded"] = True
    run = start_run(SESSION, "the task", lambda: iter(EVENTS))
    while run.is_alive():
        pass
    # A consumer that got through the first two events before a stray click
    # took the script down.
    st.session_state.biomni_run_cursor = {cursor}
    st.session_state.biomni_transcript = [
        {{"panel": "main", "role": "user", "content": "the task"}},
    ]
    st.session_state.biomni_run_active = True
    st.session_state.biomni_run_started_at = 0.0

agent = SimpleNamespace(
    main_history_copy=[],
    path=os.path.join({run_dir!r}, "agent"),
    llm=SimpleNamespace(model_name="test-model"),
)
A1.launch_streamlit_demo(agent)

st.markdown(
    "RESULT cursor=" + str(st.session_state.get("biomni_run_cursor"))
    + " entries=" + str(len(st.session_state.biomni_transcript))
    + " active=" + str(st.session_state.get("biomni_run_active"))
    + " attached=" + str(get_run(SESSION) is not None)
)
st.markdown("CONTENTS=" + repr([e.get("content") for e in st.session_state.biomni_transcript]))
'''


def _result_line(at, prefix: str) -> str:
    for m in at.markdown:
        if m.value.startswith(prefix):
            return m.value
    return "(missing)"


def _app_test(cursor: int):
    from streamlit.testing.v1 import AppTest

    tmp = tempfile.mkdtemp(prefix="biomni_bgrun_")
    path = os.path.join(tmp, "app.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_HARNESS.format(repo=_REPO, run_dir=tmp, cursor=cursor))
    return AppTest.from_file(path, default_timeout=90).run()


def test_render_loop_reattaches_to_a_live_run() -> None:
    print("\n[10] the render loop re-attaches to an in-flight run")
    at = _app_test(cursor=2)
    _check("no exception", not at.exception, str(at.exception))

    result = _result_line(at, "RESULT")
    contents = _result_line(at, "CONTENTS=")
    _check("cursor advanced to the end of the run", "cursor=5" in result, result)
    _check(
        "only the un-consumed events were appended (1 user + 3)",
        "entries=4" in result,
        f"{result}\n  {contents}",
    )
    _check(
        "the events already in the transcript were not re-rendered",
        "'starting'" not in contents and "print(1)" not in contents,
        contents,
    )
    _check("the answer landed", "'the answer'" in contents, contents)
    _check("the run is no longer marked active", "active=False" in result, result)
    _check("a fully consumed run is dropped from the registry", "attached=False" in result, result)


def test_render_loop_consumes_a_whole_run_from_scratch() -> None:
    print("\n[11] attaching at cursor 0 renders the whole run")
    at = _app_test(cursor=0)
    _check("no exception", not at.exception, str(at.exception))
    result = _result_line(at, "RESULT")
    contents = _result_line(at, "CONTENTS=")
    _check("every event was consumed", "cursor=5" in result, result)
    _check("1 user entry + 5 events", "entries=6" in result, f"{result}\n  {contents}")
    _check("the code block is there", "print(1)" in contents, contents)



# Submits a prompt through the real form and lets the run start on the
# worker, which is the path tests 10-11 skip. `stream_fn` stands in for the
# tutor wrapper: same signature, and like a real tutor step it exhausts at a
# pause instead of running to `complete`.
_SUBMIT_HARNESS = '''
import sys, os
sys.path.insert(0, {repo!r})
os.environ["BIOMENTIS_RUN_DIR"] = {run_dir!r}

import streamlit as st
from types import SimpleNamespace

from biomentis.agent.a1 import A1
from biomentis.run_worker import get_run
from biomentis.ui_core import UIEvent

st.session_state.setdefault("biomni_session_key", "apptest-submit")
st.session_state.setdefault("biomni_agent_key", "test")

TUTOR = {tutor}


def fake_stream(agent, prompt, files, history, thread_id):
    yield UIEvent(type="status", content="working on " + prompt)
    yield UIEvent(type="code", content="print('x')", language="python", title="Code")
    if TUTOR:
        # A tutor step ends at the gate, not at completion. `step_id`/`run_id`
        # are attached after construction, as ui_tutor's `_make_paused_event`
        # does — they are not UIEvent dataclass fields.
        paused = UIEvent(type="paused", content="paused", title="Paused")
        paused.step_id = 1
        paused.run_id = "r1"
        yield paused
    else:
        yield UIEvent(type="solution", channel="main", content="done", title="Answer")
        yield UIEvent(type="complete", content="finished", title="Complete")


agent = SimpleNamespace(
    main_history_copy=[],
    path=os.path.join({run_dir!r}, "agent"),
    llm=SimpleNamespace(model_name="test-model"),
)
A1.launch_streamlit_demo(agent, stream_fn=fake_stream)

st.markdown(
    "RESULT cursor=" + str(st.session_state.get("biomni_run_cursor"))
    + " entries=" + str(len(st.session_state.biomni_transcript))
    + " active=" + str(st.session_state.get("biomni_run_active"))
    + " attached=" + str(get_run("apptest-submit") is not None)
    + " journal=" + str(bool(st.session_state.get("biomni_run_journal")))
)
st.markdown("CONTENTS=" + repr([e.get("content") for e in st.session_state.biomni_transcript]))
'''


def _submit_app(tutor: bool):
    from streamlit.testing.v1 import AppTest

    tmp = tempfile.mkdtemp(prefix="biomni_submit_")
    path = os.path.join(tmp, "app.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_SUBMIT_HARNESS.format(repo=_REPO, run_dir=tmp, tutor=tutor))
    at = AppTest.from_file(path, default_timeout=90).run()
    at.text_area[0].set_value("the task")
    # Not `at.button[0]` — the two "Export to HTML" buttons render first.
    send = [b for b in at.button if b.label == "Send"]
    assert send, f"no Send button among {[b.label for b in at.button]}"
    return send[0].click().run(), tmp


def test_submitting_a_prompt_starts_a_background_run() -> None:
    print("\n[12] submitting a prompt runs the turn on the worker")
    at, run_dir = _submit_app(tutor=False)
    _check("no exception", not at.exception, str(at.exception))

    result = _result_line(at, "RESULT")
    contents = _result_line(at, "CONTENTS=")
    _check("the whole turn was consumed", "cursor=4" in result, result)
    _check("1 user entry + 4 events", "entries=5" in result, f"{result}\n  {contents}")
    _check("the answer reached the transcript", "'done'" in contents, contents)
    _check("the run ended", "active=False" in result, result)
    _check("the finished run left the registry", "attached=False" in result, result)
    _check("a journal path was recorded", "journal=True" in result, result)

    runs = list_runs(run_dir)
    _check("the run was journaled to disk", len(runs) == 1, str(runs))
    if runs:
        _check("the journal holds the prompt", runs[0]["prompt"] == "the task", str(runs[0]))
        _check("the journal is marked complete", runs[0]["status"] == "complete", str(runs[0]))
        _check("the generated code was captured", runs[0]["code_blocks"] == 1, str(runs[0]))


def test_a_tutor_step_ends_at_the_gate_and_frees_the_registry() -> None:
    print("\n[13] a tutor step runs on the worker and clears for the next one")
    at, _ = _submit_app(tutor=True)
    _check("no exception", not at.exception, str(at.exception))

    result = _result_line(at, "RESULT")
    contents = _result_line(at, "CONTENTS=")
    _check("the step's three events were consumed", "cursor=3" in result, result)
    _check("the pause reached the transcript", "'paused'" in contents, contents)
    _check(
        "the registry is free for the next Continue click",
        "attached=False" in result,
        result,
    )
    # A walkthrough is not over at a pause: the run banner must keep saying so,
    # exactly as it did before runs moved onto a thread.
    _check("the walkthrough is still marked active", "active=True" in result, result)

def run_all() -> int:
    print("=" * 60)
    print("Background run + journal tests")
    print("=" * 60)
    test_worker_runs_without_a_consumer()
    test_reattach_after_teardown_is_exact()
    test_cancel_stops_at_the_next_boundary()
    test_worker_error_is_captured()
    test_journal_is_readable_mid_run()
    test_code_script_recovers_every_block()
    test_restored_entries_match_live_entries()
    test_background_runs_can_be_disabled()
    test_registry_round_trip()
    test_render_loop_reattaches_to_a_live_run()
    test_render_loop_consumes_a_whole_run_from_scratch()
    test_submitting_a_prompt_starts_a_background_run()
    test_a_tutor_step_ends_at_the_gate_and_frees_the_registry()

    print("\n" + "=" * 60)
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    print("=" * 60)
    for name, detail in _FAILED:
        print(f"  ✗ {name}: {detail}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(run_all())
