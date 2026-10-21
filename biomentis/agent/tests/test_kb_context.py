"""Tests for the knowledge base reaching the agent, and for not losing it.

Until now the KB fed only the tutor chat and the instruction cards. A user
who uploaded course material and then asked the *agent* a question got an
answer that never saw it, with nothing on screen to say so. And the KB could
be emptied with a single unconfirmed click on "Clear KB", which sits right
next to "Add to KB" — which is what happened to an uploaded PDF on
2026-10-21.

What these tests pin down:

  1.  An empty KB yields no context, without loading the embedding model
  2.  A populated KB yields the passages, their sources, and how to cite them
  3.  The agent's message carries the context; `user_task` stays the prompt
  4.  The run announces which KB sources it is using
  5.  The tutor wrapper hands the session's KB to the agent (tutor off)
  6.  ...and with instructional mode on
  7.  "Clear KB" asks for confirmation before deleting anything

Run with:
    python -m biomentis.agent.tests.test_kb_context
"""

from __future__ import annotations

import os
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


_RUMEN = (
    "Rumen fluid must be collected under anaerobic conditions and frozen at -80C "
    "within thirty minutes, or the Bacteroidetes to Firmicutes ratio shifts. "
)


def _kb_with(text: str, name: str = "rumen_notes.txt"):
    from biomentis.agent.tutor.kb import KnowledgeBase

    root = tempfile.mkdtemp()
    kb = KnowledgeBase("kbctx", path=root)
    path = os.path.join(root, name)
    with open(path, "w") as f:
        f.write(text)
    kb.add_files([path])
    return kb


class _RecordingLLM:
    """Answers at once and remembers what it was sent."""

    def __init__(self):
        self.calls: list[list] = []

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.calls.append(messages)
        return AIMessage(content="<solution>done</solution>")


def _agent():
    from biomentis.agent.a1 import A1

    agent = A1(path="./data", use_tool_retriever=False, expected_data_lake_files=[], trace=False)
    agent.llm = _RecordingLLM()
    agent.configure()
    return agent


def _sent_text(llm) -> str:
    return "\n".join(str(m.content) for m in llm.calls[0])


# --- 1-2: building the context --------------------------------------------


def test_empty_kb_gives_no_context() -> None:
    print("\n[1] an empty KB adds nothing and loads nothing")
    from biomentis.agent.tutor.kb import KnowledgeBase

    kb = KnowledgeBase("empty", path=tempfile.mkdtemp())
    ctx = kb.context_for_task("how do I sample the rumen?")
    _check("no context text", ctx.text == "", repr(ctx.text))
    _check("no sources", ctx.sources == [], repr(ctx.sources))
    _check("the embedding model was not loaded", kb._embeddings is None)


def test_populated_kb_gives_passages_and_sources() -> None:
    print("\n[2] a populated KB yields passages, sources and citation rules")
    kb = _kb_with(_RUMEN * 40)
    ctx = kb.context_for_task("how should rumen fluid be stored?")
    _check("the passage text is included", "anaerobic conditions" in ctx.text, ctx.text[:300])
    _check("the source is named", "rumen_notes.txt" in ctx.text, ctx.text[:300])
    _check("the agent is told to cite it", "cite" in ctx.text.lower(), ctx.text[:300])
    _check("sources are reported", ctx.sources == ["rumen_notes.txt"], repr(ctx.sources))


# --- 3-4: the agent receives it -------------------------------------------


def test_agent_message_carries_the_context() -> None:
    print("\n[3] the agent's message carries the KB context")
    from biomentis.ui_core import stream_agent_events

    agent = _agent()
    list(
        stream_agent_events(
            agent, "plan a rumen study", [], [], 11, context="KB_PASSAGE_MARKER", context_note="note"
        )
    )
    sent = _sent_text(agent.llm)
    _check("the context reached the model", "KB_PASSAGE_MARKER" in sent, sent[-400:])
    _check("so did the prompt", "plan a rumen study" in sent, sent[-400:])
    _check("user_task is the prompt alone", agent.user_task == "plan a rumen study", repr(agent.user_task))


def test_run_announces_its_sources() -> None:
    print("\n[4] the run says which KB sources it is using")
    from biomentis.ui_core import stream_agent_events

    agent = _agent()
    events = list(
        stream_agent_events(
            agent, "plan a rumen study", [], [], 12, context="ctx", context_note="📚 Using rumen_notes.txt"
        )
    )
    first = events[0] if events else None
    _check(
        "the first event is the note",
        first is not None and first.type == "status" and "rumen_notes.txt" in first.content,
        repr(first),
    )

    quiet = list(stream_agent_events(_agent(), "plan a rumen study", [], [], 13))
    _check("no note without context", all("📚" not in e.content for e in quiet), repr(quiet[:2]))


# --- 5-6: the tutor wrapper supplies the session's KB ---------------------


def _session_with_kb(kb, enabled: bool):
    import streamlit as st

    from biomentis.agent.tutor import TutorEngine

    tutor = TutorEngine(session_id="kbctx-wrap", llm=_RecordingLLM(), path=tempfile.mkdtemp())
    tutor.kb = kb
    tutor.enabled = enabled
    st.session_state["biomni_tutor"] = tutor
    st.session_state["biomni_tutor_run"] = None
    return tutor


def test_wrapper_passes_kb_with_tutor_off() -> None:
    print("\n[5] tutor off: the wrapper gives the agent the session's KB")
    from biomentis.ui_tutor import tutor_wrapped_stream

    _session_with_kb(_kb_with(_RUMEN * 40), enabled=False)
    agent = _agent()
    events = list(tutor_wrapped_stream(agent, "how should rumen fluid be stored?", [], [], 21))
    sent = _sent_text(agent.llm)
    _check("the KB passage reached the model", "anaerobic conditions" in sent, sent[-400:])
    _check("the run named its source", any("rumen_notes.txt" in e.content for e in events if e.type == "status"))


def test_wrapper_passes_kb_with_tutor_on() -> None:
    print("\n[6] tutor on: the wrapper gives the agent the session's KB")
    from biomentis.ui_tutor import tutor_wrapped_stream

    tutor = _session_with_kb(_kb_with(_RUMEN * 40), enabled=True)
    tutor.modes = ()  # no teaching cards: nothing else calls an LLM
    agent = _agent()
    list(tutor_wrapped_stream(agent, "how should rumen fluid be stored?", [], [], 22))
    _check("the model was called", bool(agent.llm.calls), "no calls")
    sent = _sent_text(agent.llm) if agent.llm.calls else ""
    _check("the KB passage reached the model", "anaerobic conditions" in sent, sent[-400:])


# --- 7: clearing needs a second click -------------------------------------


def test_clear_needs_confirmation() -> None:
    print("\n[7] Clear KB asks before deleting")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("streamlit_app.py", default_timeout=300)
    at.run()
    up = next(u for u in at.get("file_uploader") if u.key == "biomni_tutor_kb_uploader")
    up.upload("rumen_notes.txt", (_RUMEN * 40).encode(), "text/plain")
    at.run()
    next(b for b in at.button if b.key == "biomni_tutor_kb_add").click()
    at.run()
    kb = at.session_state["biomni_tutor"].kb
    _check("the upload was indexed", kb.stats().sources == 1, repr(kb.stats()))

    next(b for b in at.button if b.key == "biomni_tutor_kb_clear").click()
    at.run()
    _check("one click deletes nothing", kb.stats().sources == 1, repr(kb.stats()))
    confirm = [b for b in at.button if b.key == "biomni_tutor_kb_clear_confirm"]
    _check("a confirm button appears", bool(confirm), str([b.key for b in at.button]))
    if not confirm:
        return

    confirm[0].click()
    at.run()
    _check("confirming clears it", kb.stats().sources == 0, repr(kb.stats()))
    # Asserted on the warning, not the buttons: after `st.rerun()` AppTest
    # keeps the previous run's buttons in its tree until the next run.
    _check(
        "and the confirmation prompt goes away",
        not any("Delete every document" in w.value for w in at.warning),
        str([w.value for w in at.warning]),
    )

    # The app writes its session folders under the real data dir; don't
    # leave this test's behind.
    import shutil

    shutil.rmtree(kb.path, ignore_errors=True)
    shutil.rmtree(at.session_state["biomni_tutor"].logger.dir, ignore_errors=True)


def run_all() -> int:
    print("=" * 60)
    print("Knowledge base → agent tests")
    print("=" * 60)
    for test in (
        test_empty_kb_gives_no_context,
        test_populated_kb_gives_passages_and_sources,
        test_agent_message_carries_the_context,
        test_run_announces_its_sources,
        test_wrapper_passes_kb_with_tutor_off,
        test_wrapper_passes_kb_with_tutor_on,
        test_clear_needs_confirmation,
    ):
        try:
            test()
        except Exception as e:
            _check(f"{test.__name__} ran", False, repr(e))

    print("\n" + "=" * 60)
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    print("=" * 60)
    for name, detail in _FAILED:
        print(f"  ✗ {name}: {detail}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(run_all())
