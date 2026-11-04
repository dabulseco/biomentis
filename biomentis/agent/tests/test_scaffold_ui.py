"""Tests for question support in the Streamlit UI.

Spec: docs/superpowers/specs/2026-11-04-scaffolded-questions-design.md

What these tests pin down:

  1.  Sidebar control: Students / Guide me by default, every level
      selectable (Answer directly too), Teacher locks at the first question
      and the radios disappear, an edit to a vanished widget's key changes
      nothing, an env preset is locked from the first render
  2.  Page chat: a direct answer is today's flow; a guided question opens a
      thread (caption, Hint, Show me the answer, new placeholder); Show me
      the answer releases it into history; a correct reply earns it; a new
      question mid-thread releases the old one and is answered; a rerun
      re-renders an open thread unchanged; level "off" is today's flow
  3.  Per-step box: a thread inside the step's expander; released into
      today's Q/A shape plus its rungs; two steps keep separate threads;
      level "off" is today's flow

Run with:
    python -m biomentis.agent.tests.test_scaffold_ui
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

from langchain_core.messages import AIMessage

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


_CLASSIFY = json.dumps({"bloom_level": "Apply", "dok_level": 2, "rubric_hit": [], "confidence": 0.7})
_ANSWER = "An E-value counts the hits you'd expect by chance in a database this size."
_FOLLOW_UP = "What would happen at a cutoff of 0.05?"
_RUNG_1 = "What would you expect to see by chance in a very large database?"
_RUNG_2 = "Think about how the database size changes that count."


def _profile(**kw) -> dict:
    p = {"type": "conceptual", "bloom": "Apply", "dok": 2, "specific": True,
         "shows_reasoning": False, "wants_direct": False}
    p.update(kw)
    return p


def _answer(profile: dict) -> str:
    return json.dumps({"answer": _ANSWER, "follow_up": _FOLLOW_UP, "citations": [], "also_consider": [],
                       "plot_code": "", "structure": "", "key_points": ["expected number of chance hits"],
                       "question_profile": profile})


class _RouterLLM:
    """Answers the answer call, the classifier and the rung calls by prompt.

    `rungs` is a queue of rung replies; `answers`, when given, a queue of
    answer-call replies (the last one repeats).
    """

    def __init__(self, answer: str, rungs=(), answers=()):
        self.model = "fake"
        self.answers = list(answers) or [answer]
        self.rungs = list(rungs)
        self.rung_calls = 0

    def invoke(self, messages):
        system = messages[0].content
        if system.startswith("You are a patient tutor in a biomedical research workflow"):
            self.rung_calls += 1
            reply = self.rungs.pop(0)
            return AIMessage(content=reply if isinstance(reply, str) else json.dumps(reply))
        if system.startswith("You are a careful educational assessor"):
            return AIMessage(content=_CLASSIFY)
        reply = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return AIMessage(content=reply)


def _rung(message=_RUNG_1, verdict="none") -> dict:
    return {"verdict": verdict, "message": message}


def _engine(llm, level="standard", env: dict | None = None):
    from biomentis.agent.tutor import TutorEngine

    keys = ("BIOMENTIS_SCAFFOLD_LEVEL", "BIOMENTIS_SCAFFOLD_CONTROL")
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        os.environ.update(env or {})
        engine = TutorEngine(session_id="scaffold-ui", llm=llm, path=tempfile.mkdtemp())
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    if not engine.scaffold_locked:
        engine.set_scaffold(level=level)
    return engine


def _records(engine, kind) -> list[dict]:
    if not os.path.exists(engine.logger.path):
        return []
    with open(engine.logger.path) as f:
        return [r for r in (json.loads(line) for line in f if line.strip()) if r.get("kind") == kind]


# --- harness apps (run in this process; the engine rides in session state) ---


def _sidebar_app():
    import streamlit as st

    from biomentis.ui_tutor import render_tutor_sidebar

    render_tutor_sidebar(st.session_state["tutor"])


def _chat_app():
    import streamlit as st

    from biomentis.ui_tutor import render_tutor_chat_panel

    render_tutor_chat_panel(st.session_state["tutor"])


def _step_app():
    import streamlit as st

    from biomentis.ui_tutor import _render_step_qa

    run = st.session_state.setdefault("run", {"prompt": "Find homologs of the query protein", "step_qa": {}})
    for step_id in (1, 2):
        _render_step_qa(st.container(), step_id, run, st.session_state["tutor"], "r1")


def _app(fn, engine):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(fn, default_timeout=30)
    at.session_state["tutor"] = engine
    return at.run()


_CONTROL = "biomni_tutor_scaffold_control"
_LEVEL = "biomni_tutor_scaffold_level"
_THREAD = "biomni_tutor_scaffold_thread"
_HISTORY = "biomni_tutor_chat_history"


def _radio(at, key):
    found = [r for r in at.radio if r.key == key]
    return found[0] if found else None


def _captions(at) -> list[str]:
    return [c.value for c in at.caption]


def _ask(at, text: str) -> None:
    at.text_area(key="biomni_tutor_chat_textarea").input(text)
    # After an `st.rerun()` AppTest can also list the previous run's
    # submit button; the live one is the last.
    [b for b in at.button if b.label == "Ask"][-1].click()
    at.run()


def _ask_step(at, step_id: int, text: str) -> None:
    at.text_input(key=f"biomni_tutor_step_ask_input_r1_{step_id}").input(text)
    [b for b in at.button if b.label == "💬 Ask about this step"][step_id - 1].click()
    at.run()


def _history(at) -> list[dict]:
    return list(at.session_state[_HISTORY])


def _exc(at) -> str:
    return "; ".join(e.value for e in at.exception)


# --- 1: sidebar control --------------------------------------------------------


def test_sidebar_defaults() -> None:
    print("\n[1a] sidebar defaults and choices")
    engine = _engine(_RouterLLM(_answer(_profile())), level="standard")
    at = _app(_sidebar_app, engine)
    _check("no exception", not at.exception, _exc(at))
    control, level = _radio(at, _CONTROL), _radio(at, _LEVEL)
    _check("control radio shown", control is not None)
    _check("level radio shown", level is not None)
    if control is None or level is None:
        return
    _check("students choose by default", control.value == "students", repr(control.value))
    _check("Guide me by default", level.value == "standard", repr(level.value))
    _check("level labels", list(level.options) == ["Answer directly", "A nudge first", "Guide me", "Challenge me"],
           repr(level.options))
    _check("control labels", list(control.options) == ["Students", "Teacher"], repr(control.options))
    for value in ("off", "light", "challenge", "standard"):
        _radio(at, _LEVEL).set_value(value).run()
        _check(f"{value} selectable", engine.scaffold_level == value, engine.scaffold_level)
    _check("no lock caption for students",
           not any("Locks for this session" in c for c in _captions(at)), repr(_captions(at)))


def test_sidebar_teacher_lock() -> None:
    print("\n[1b] Teacher locks at the first question")
    engine = _engine(_RouterLLM(_answer(_profile(type="definition"))), level="standard")
    at = _app(_sidebar_app, engine)
    _radio(at, _CONTROL).set_value("teacher").run()
    _check("control is teacher", engine.scaffold_control == "teacher")
    _check("lock warning caption",
           any("Locks for this session when the first question is asked." in c for c in _captions(at)),
           repr(_captions(at)))
    _radio(at, _LEVEL).set_value("challenge").run()
    _check("teacher sets the level before the lock", engine.scaffold_level == "challenge")
    engine.start_question("What is an E-value?", surface="chat", ask_kwargs={"task": "t", "mode": "chat"})
    at.run()
    _check("locked", engine.scaffold_locked)
    _check("radios gone", _radio(at, _CONTROL) is None and _radio(at, _LEVEL) is None,
           repr([r.key for r in at.radio]))
    _check("locked caption",
           any(c == "Question support: **Challenge me**, set by your teacher" for c in _captions(at)),
           repr(_captions(at)))
    at.session_state[_LEVEL] = "off"
    at.session_state[_CONTROL] = "students"
    at.run()
    _check("session-state edit leaves the level", engine.scaffold_level == "challenge", engine.scaffold_level)
    _check("session-state edit leaves the control", engine.scaffold_control == "teacher")
    _check("still no radios", _radio(at, _LEVEL) is None)


def test_sidebar_env_preset() -> None:
    print("\n[1c] env preset is locked from the first render")
    engine = _engine(_RouterLLM(_answer(_profile())),
                     env={"BIOMENTIS_SCAFFOLD_CONTROL": "teacher", "BIOMENTIS_SCAFFOLD_LEVEL": "challenge"})
    at = _app(_sidebar_app, engine)
    _check("no exception", not at.exception, _exc(at))
    _check("no radios", _radio(at, _CONTROL) is None and _radio(at, _LEVEL) is None)
    _check("locked caption",
           any(c == "Question support: **Challenge me**, set by your teacher" for c in _captions(at)),
           repr(_captions(at)))


# --- 2: page chat --------------------------------------------------------------


def test_chat_direct() -> None:
    print("\n[2a] a definition at Guide me is answered directly")
    llm = _RouterLLM(_answer(_profile(type="definition")))
    at = _app(_chat_app, _engine(llm))
    _ask(at, "What is an E-value?")
    _check("no exception", not at.exception, _exc(at))
    h = _history(at)
    _check("question + answer", [t["role"] for t in h] == ["user", "assistant"], repr(h))
    _check("answer text", h[-1]["content"].startswith(f"{_ANSWER}\n\n**Follow-up:** {_FOLLOW_UP}"), h[-1]["content"])
    _check("took suffix", "(took " in h[-1]["content"])
    _check("no thread", not at.session_state[_THREAD] if _THREAD in at.session_state else True)
    _check("no rung call", llm.rung_calls == 0)
    _check("no guided caption", not any("Guided" in c for c in _captions(at)))


def _open_thread(rungs, level="standard"):
    llm = _RouterLLM(_answer(_profile()), rungs=rungs)
    engine = _engine(llm, level=level)
    at = _app(_chat_app, engine)
    _ask(at, "Why filter BLAST hits at E < 1e-5?")
    return at, llm, engine


def test_chat_thread_opens() -> None:
    print("\n[2b] a conceptual question opens a thread")
    at, llm, _ = _open_thread([_rung()])
    _check("no exception", not at.exception, _exc(at))
    thread = at.session_state[_THREAD] if _THREAD in at.session_state else None
    _check("thread stored as a dict", isinstance(thread, dict), repr(thread))
    _check("only the question in history", [t["role"] for t in _history(at)] == ["user"], repr(_history(at)))
    _check("step caption", "🧭 Guided · step 1 of up to 2" in _captions(at), repr(_captions(at)))
    md = [m.value for m in at.markdown]
    _check("rung shown", _RUNG_1 in md, repr(md))
    _check("answer not shown", not any(_ANSWER in m for m in md))
    _check("hint button", any(b.key == "biomni_tutor_scaffold_hint" and b.label == "💡 Hint" for b in at.button))
    _check("reveal button",
           any(b.key == "biomni_tutor_scaffold_reveal" and b.label == "Show me the answer" for b in at.button))
    _check("placeholder", at.text_area(key="biomni_tutor_chat_textarea").placeholder
           == "Your thinking… (or ask something new)")

    # Review Focus 5: a rerun with no input changes nothing.
    before = json.dumps(thread, sort_keys=True, default=str)
    at.run()
    after = json.dumps(at.session_state[_THREAD], sort_keys=True, default=str)
    _check("rerun leaves the thread", before == after)
    _check("rerun makes no call", llm.rung_calls == 1, str(llm.rung_calls))
    _check("rerun re-renders the rung", "🧭 Guided · step 1 of up to 2" in _captions(at))


def test_chat_reveal() -> None:
    print("\n[2c] Show me the answer releases it")
    at, _, engine = _open_thread([_rung()])
    at.button(key="biomni_tutor_scaffold_reveal").click().run()
    _check("no exception", not at.exception, _exc(at))
    _check("thread cleared", not at.session_state[_THREAD] if _THREAD in at.session_state else True)
    h = _history(at)
    _check("question, rung, answer", [t["role"] for t in h] == ["user", "assistant", "assistant"], repr(h))
    _check("rung flagged", h[1].get("scaffold_rung") is True and h[1]["content"] == _RUNG_1, repr(h[1]))
    _check("answer has today's keys", {"citations", "bloom_level", "dok_level", "also_consider"} <= set(h[2]))
    _check("answer text", h[2]["content"].startswith(_ANSWER), h[2]["content"])
    _check("no-sources caption on the answer only",
           sum("No KB sources cited" in c for c in _captions(at)) == 1, repr(_captions(at)))
    _check("scaffold record", [r["released"] for r in _records(engine, "scaffold")] == ["button"])
    _check("no buttons", not any(b.key in ("biomni_tutor_scaffold_hint", "biomni_tutor_scaffold_reveal")
                                 for b in at.button))


def test_chat_hint() -> None:
    print("\n[2d] Hint adds the next step")
    at, llm, _ = _open_thread([_rung(), _rung(_RUNG_2)])
    at.button(key="biomni_tutor_scaffold_hint").click().run()
    _check("no exception", not at.exception, _exc(at))
    _check("second step caption", "🧭 Guided · step 2 of up to 2" in _captions(at), repr(_captions(at)))
    _check("two rungs on the thread", len(at.session_state[_THREAD]["rungs"]) == 2)
    _check("hint counted", at.session_state[_THREAD]["hints_requested"] == 1)


def test_chat_earned() -> None:
    print("\n[2e] a correct reply earns the answer")
    at, _, engine = _open_thread([_rung(), _rung("Exactly.", verdict="correct")])
    _ask(at, "Because chance hits grow with database size")
    _check("no exception", not at.exception, _exc(at))
    h = _history(at)
    _check("question, rung, reply, answer", [t["role"] for t in h] == ["user", "assistant", "user", "assistant"],
           repr(h))
    _check("reply flagged", h[2].get("scaffold_rung") is True)
    _check("✓ prefix", h[3]["content"].startswith("✓ That's the core of it."), h[3]["content"])
    _check("released earned", [r["released"] for r in _records(engine, "scaffold")] == ["earned"])


def test_chat_new_question() -> None:
    print("\n[2f] a new question mid-thread")
    rungs = [_rung(), _rung("", verdict="new_question"), _rung("What is a BLOSUM matrix built from?")]
    at, _, engine = _open_thread(rungs)
    _ask(at, "Which scoring matrix should I use?")
    _check("no exception", not at.exception, _exc(at))
    h = _history(at)
    _check("old thread then new question",
           [t["role"] for t in h] == ["user", "assistant", "assistant", "user"], repr(h))
    _check("old answer released", h[2]["content"].startswith(_ANSWER))
    _check("new question last", h[3]["content"] == "Which scoring matrix should I use?")
    _check("released abandoned", [r["released"] for r in _records(engine, "scaffold")] == ["abandoned"])
    thread = at.session_state[_THREAD] if _THREAD in at.session_state else None
    _check("new thread open", bool(thread) and thread["question"] == "Which scoring matrix should I use?",
           repr(thread))


def test_chat_highlight_reveal() -> None:
    print("\n[2h] wanting the answer straight under Challenge me highlights the way out")
    llm = _RouterLLM(_answer(_profile(wants_direct=True)), rungs=[_rung()])
    at = _app(_chat_app, _engine(llm, level="challenge"))
    _ask(at, "Just tell me why we filter at E < 1e-5")
    _check("no exception", not at.exception, _exc(at))
    _check("one step", "🧭 Guided · step 1 of up to 1" in _captions(at), repr(_captions(at)))
    _check("reveal is primary", at.button(key="biomni_tutor_scaffold_reveal").proto.type == "primary")

    at, _, _ = _open_thread([_rung()])
    _check("otherwise secondary", at.button(key="biomni_tutor_scaffold_reveal").proto.type == "secondary")


def test_chat_off() -> None:
    print("\n[2g] level off is today's flow")
    llm = _RouterLLM(_answer(_profile()))
    engine = _engine(llm, level="off")
    at = _app(_chat_app, engine)
    _ask(at, "Why filter BLAST hits at E < 1e-5?")
    _check("no exception", not at.exception, _exc(at))
    h = _history(at)
    _check("one answer", [t["role"] for t in h] == ["user", "assistant"], repr(h))
    _check("no caption", not any("Guided" in c for c in _captions(at)))
    _check("no rung call", llm.rung_calls == 0)
    _check("no scaffold record", _records(engine, "scaffold") == [])
    _check("qa logged", len(_records(engine, "qa")) == 1)


# --- 3: per-step box -----------------------------------------------------------


def _step_thread(at, step_id):
    run = at.session_state["run"]
    return run.get("step_qa_thread", {}).get(step_id)


def test_step_thread() -> None:
    print("\n[3a] a step question opens a thread inside its expander")
    llm = _RouterLLM(_answer(_profile()), rungs=[_rung(), _rung(_RUNG_2)])
    engine = _engine(llm)
    at = _app(_step_app, engine)
    _ask_step(at, 1, "Why filter at E < 1e-5 here?")
    _check("no exception", not at.exception, _exc(at))
    thread = _step_thread(at, 1)
    _check("thread on step 1", isinstance(thread, dict), repr(thread))
    _check("no thread on step 2", _step_thread(at, 2) is None)
    _check("no Q/A yet", not at.session_state["run"]["step_qa"].get(1))
    _check("caption", "🧭 Guided · step 1 of up to 2" in _captions(at), repr(_captions(at)))
    _check("rung inside the step expander",
           any(_RUNG_1 in m.value for e in at.expander for m in e.markdown), repr([e.label for e in at.expander]))
    _check("buttons keyed per step",
           {"biomni_tutor_scaffold_hint_r1_1", "biomni_tutor_scaffold_reveal_r1_1"} <= {b.key for b in at.button})
    _check("placeholder", at.text_input(key="biomni_tutor_step_ask_input_r1_1").placeholder
           == "Your thinking… (or ask something new)")

    # Step 2 gets its own thread.
    llm.rungs.append(_rung("What does the database size do here?"))
    _ask_step(at, 2, "Why this cutoff for step 2?")
    _check("step 2 thread", isinstance(_step_thread(at, 2), dict))
    _check("step 1 thread untouched", _step_thread(at, 1) == thread)

    at.button(key="biomni_tutor_scaffold_reveal_r1_1").click().run()
    _check("no exception after reveal", not at.exception, _exc(at))
    qa = at.session_state["run"]["step_qa"].get(1) or []
    _check("one Q/A on step 1", len(qa) == 1, repr(qa))
    if qa:
        entry = qa[0]
        _check("today's shape", {"question", "answer", "citations", "also_consider", "bloom_level",
                                 "dok_level", "rubric_hit", "confidence", "failed"} <= set(entry), repr(entry))
        _check("no role key", "role" not in entry and "content" not in entry)
        _check("answer text", entry["answer"].startswith(_ANSWER))
        _check("rungs kept", [r["message"] for r in entry.get("rungs", [])] == [_RUNG_1])
    _check("step 1 thread cleared", _step_thread(at, 1) is None)
    _check("step 2 thread still open", isinstance(_step_thread(at, 2), dict))
    _check("guided steps expander", any(e.label == "🧭 Guided steps (1)" for e in at.expander),
           repr([e.label for e in at.expander]))
    recs = _records(engine, "scaffold")
    _check("scaffold record for the step", [(r["surface"], r["step_id"]) for r in recs] == [("step", 1)], repr(recs))


def test_step_off() -> None:
    print("\n[3b] level off in the step box is today's flow")
    llm = _RouterLLM(_answer(_profile()))
    engine = _engine(llm, level="off")
    at = _app(_step_app, engine)
    _ask_step(at, 1, "Why filter at E < 1e-5 here?")
    _check("no exception", not at.exception, _exc(at))
    qa = at.session_state["run"]["step_qa"].get(1) or []
    _check("answered at once", len(qa) == 1 and qa[0]["answer"].startswith(_ANSWER), repr(qa))
    _check("no rungs key", qa and "rungs" not in qa[0])
    _check("no thread", _step_thread(at, 1) is None)
    _check("no rung call", llm.rung_calls == 0)


def run_all() -> int:
    tests = [
        test_sidebar_defaults,
        test_sidebar_teacher_lock,
        test_sidebar_env_preset,
        test_chat_direct,
        test_chat_thread_opens,
        test_chat_reveal,
        test_chat_hint,
        test_chat_earned,
        test_chat_new_question,
        test_chat_highlight_reveal,
        test_chat_off,
        test_step_thread,
        test_step_off,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback

            traceback.print_exc()
            _FAILED.append((t.__name__, repr(e)))
            print(f"  ✗ {t.__name__} raised {e!r}")
    print(f"\nResults: {len(_PASSED)} passed, {len(_FAILED)} failed")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(run_all())
