"""Tests for how a run's completion and its timing are reported.

Regression origin (run of 2026-08-19): the executor log ended with two
completion banners back to back —

    🔄 Complete  · total time: 2m 15.5s
    ✅ Done      · agent compute time: 15m 25.3s · session time: 54m 20.3s
                 · total time: 0.0s

Two `complete` events are produced for one tutor run: the inner stream's
"returning the result to the main interface" (`ui_core.stream_agent_events`)
and the tutor's own "✅ Done" (`ui_tutor.tutor_wrapped_stream`). The first one
stopped the run clock, so the second was stamped `total time: 0.0s` — and it
also overwrote the duration the run banner reads.

The fix has two halves:

  1. `_advance_run_live` drops the inner stream's completion while the tutor
     is gating the run; the walkthrough ends when the student finishes it,
     not when the agent stops, and the "✅ Done" event describes that.
  2. A `complete` event may carry its own `duration`, which tells
     `launch_streamlit_demo` the producer already accounted for the timing
     and to render the content as-is rather than appending a second total.

Run with:
    python -m biomentis.agent.tests.test_run_completion
"""

from __future__ import annotations

import os
import re
import sys
import tempfile


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


_REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Drives a real TutorEngine + the real `tutor_wrapped_stream` over a scripted
# agent, pumping it the way repeated Continue clicks would.
_HARNESS = '''
import sys, tempfile
sys.path.insert(0, {repo!r})
from types import SimpleNamespace

import streamlit as st
from langchain_core.messages import AIMessage

from biomentis.agent.tutor.engine import TutorEngine
from biomentis.ui_tutor import tutor_wrapped_stream

MESSAGES = [
    "Thinking about it.\\n<execute>print('work')</execute>",
    "Done.\\n<solution>The answer.</solution>",
]


def _stream(inputs, stream_mode=None, config=None):
    history = list(inputs["messages"])
    for content in MESSAGES:
        history.append(AIMessage(content=content))
        yield {{"messages": list(history)}}


agent = SimpleNamespace(
    app=SimpleNamespace(stream=_stream),
    use_tool_retriever=False,
    path=tempfile.mkdtemp(),
    user_task="",
)

if "biomni_tutor" not in st.session_state:
    eng = TutorEngine("completion-test", llm=None, path=tempfile.mkdtemp())
    eng.enabled = True
    st.session_state["biomni_tutor"] = eng

# Pump the walkthrough to exhaustion: each call advances one step and
# pauses, exactly as a Continue click does.
seen = []
for _ in range(40):
    before = len(seen)
    for ev in tutor_wrapped_stream(agent, "task", [], [], 1):
        seen.append(ev)
    run = st.session_state.get("biomni_tutor_run")
    if run is not None and run.get("phase") == "done":
        break
    if len(seen) == before:
        break

completes = [e for e in seen if e.type == "complete"]
st.write("COMPLETES=" + repr(len(completes)))
st.write("DURATIONS=" + repr([e.duration for e in completes]))
st.write("TITLES=" + repr([e.title for e in completes]))
'''


def _app_test():
    from streamlit.testing.v1 import AppTest

    path = os.path.join(tempfile.mkdtemp(prefix="biomni_complete_"), "app.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_HARNESS.format(repo=_REPO))
    return AppTest.from_file(path, default_timeout=120).run()


def _line(at, prefix: str) -> str:
    for m in at.markdown:
        if m.value.startswith(prefix):
            return m.value
    return "(missing)"


# --- Tests ----------------------------------------------------------------


def test_one_completion_per_tutor_run() -> None:
    print("\n[1] A tutor run reports exactly one completion")
    at = _app_test()
    _check("harness ran without error", not at.exception, str(at.exception))
    _check(
        "exactly one complete event survives the walkthrough",
        _line(at, "COMPLETES=") == "COMPLETES=1",
        _line(at, "COMPLETES="),
    )
    _check(
        "and it is the tutor's own ✅ Done, not the inner narration",
        "✅ Done" in _line(at, "TITLES="),
        _line(at, "TITLES="),
    )
    _check(
        "it carries its own duration, so no second total is appended",
        _line(at, "DURATIONS=") != "DURATIONS=[None]"
        and _line(at, "DURATIONS=") != "(missing)",
        _line(at, "DURATIONS="),
    )


def test_research_mode_completion_still_self_times() -> None:
    """The unwrapped stream must keep its own completion, untimed by itself."""
    print("\n[2] Research mode still emits its own untimed completion")
    from types import SimpleNamespace

    from langchain_core.messages import AIMessage

    from biomentis.ui_core import stream_agent_events

    def _stream(inputs, stream_mode=None, config=None):
        history = list(inputs["messages"])
        history.append(AIMessage(content="Done.\n<solution>The answer.</solution>"))
        yield {"messages": history}

    agent = SimpleNamespace(
        app=SimpleNamespace(stream=_stream),
        use_tool_retriever=False,
        path="/tmp",
        user_task="",
    )
    events = list(stream_agent_events(agent, "task", [], [], thread_id=1))
    completes = [e for e in events if e.type == "complete"]
    _check("exactly one completion", len(completes) == 1, f"got {len(completes)}")
    _check(
        "it carries no duration, so the UI stamps the run clock onto it",
        completes and completes[0].duration is None,
        f"duration={completes[0].duration if completes else '(none)'}",
    )


def test_complete_handler_is_timing_aware() -> None:
    """The dispatch lives inside `launch_streamlit_demo`; assert its shape."""
    print("\n[3] The complete handler distinguishes the three timing cases")
    src_path = os.path.join(_REPO, "biomentis", "agent", "a1.py")
    with open(src_path, encoding="utf-8") as f:
        src = f.read()

    # `a1.py` has two `complete` handlers — the Gradio one and the Streamlit
    # one. Anchor on the run clock, which only the Streamlit dispatch owns.
    handler = src[src.rindex('elif event.type == "complete":') :][:4000]
    _check(
        "anchored on the Streamlit dispatch, not the Gradio one",
        "biomni_run_started_at" in handler,
        "the anchored handler does not touch the run clock",
    )
    _check(
        "a self-timed event is rendered as-is",
        re.search(r"if event\.duration is not None:", handler) is not None,
        "no `event.duration is not None` branch",
    )
    _check(
        "a second completion is not stamped 'total time: 0.0s'",
        "_stamp = None" in handler,
        "no branch that skips stamping when no run is in flight",
    )
    _check(
        "the banner duration is only overwritten when there is one to record",
        "if _stamp is not None:" in handler,
        "the stashed duration can still be clobbered",
    )


def test_inner_completion_is_dropped_while_gating() -> None:
    print("\n[4] The inner stream's completion is dropped during a walkthrough")
    src_path = os.path.join(_REPO, "biomentis", "ui_tutor.py")
    with open(src_path, encoding="utf-8") as f:
        src = f.read()

    advance = src[src.index("def _advance_run_live(") :][:3000]
    _check(
        "_advance_run_live skips complete events",
        re.search(r'if event\.type == "complete":\s*\n(\s*#.*\n)*\s*continue', advance)
        is not None,
        "no `complete` -> continue guard in the advance loop",
    )
    _check(
        "the tutor's own Done carries a duration",
        "duration=_wall," in src,
        "the ✅ Done event does not carry its own timing",
    )


def run_all() -> int:
    print("=" * 60)
    print("Run-completion tests")
    print("=" * 60)
    test_one_completion_per_tutor_run()
    test_research_mode_completion_still_self_times()
    test_complete_handler_is_timing_aware()
    test_inner_completion_is_dropped_while_gating()

    print("\n" + "=" * 60)
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    print("=" * 60)
    for name, detail in _FAILED:
        print(f"  ✗ {name}: {detail}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(run_all())
