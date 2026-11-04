"""Tests for question support run end to end through `TutorEngine`.

Spec: docs/superpowers/specs/2026-11-04-scaffolded-questions-design.md

What these tests pin down:

  1.  A question that gets a direct answer is released at once and logged
  2.  A question worth guiding opens a thread with its first step, and
      nothing is logged until the answer is released
  3.  Each way an answer is released: earned, button, exhausted (by reply
      and by hint), frustration, abandoned (with the new question handed
      back), error (fail open)
  4.  A stretch question under "challenge" for an expert question only
  5.  A thread survives a round trip through session state mid-exchange
  6.  Settings: changes are logged; teacher control locks at the first
      question; an env preset is locked from the start
  7.  At level "off" the log is exactly what it was before (no scaffold
      records)

Run with:
    python -m biomentis.agent.tests.test_scaffold_engine
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


_CLASSIFY = json.dumps({"bloom_level": "Analyze", "dok_level": 3, "rubric_hit": [], "confidence": 0.7})
_ANSWER = "An E-value counts the hits you'd expect by chance in a database this size."
_FOLLOW_UP = "What would happen at a cutoff of 0.05?"


def _profile(**kw) -> dict:
    p = {"type": "conceptual", "bloom": "Apply", "dok": 2, "specific": True,
         "shows_reasoning": False, "wants_direct": False}
    p.update(kw)
    return p


def _answer(profile: dict | None, stretch: str = "") -> str:
    reply = {"answer": _ANSWER, "follow_up": _FOLLOW_UP, "citations": [], "also_consider": [],
             "plot_code": "", "structure": "", "key_points": ["expected number of chance hits"],
             "stretch_question": stretch}
    if profile is not None:
        reply["question_profile"] = profile
    return json.dumps(reply)


class _RouterLLM:
    """Answers the answer call, the classifier and the rung calls by prompt.

    `rungs` is a queue of rung replies (dicts, strings, or exceptions).
    """

    def __init__(self, answer: str, rungs=()):
        self.model = "fake"
        self.answer = answer
        self.rungs = list(rungs)
        self.rung_calls = 0

    def invoke(self, messages):
        system = messages[0].content
        if system.startswith("You are a patient tutor in a biomedical research workflow"):
            self.rung_calls += 1
            reply = self.rungs.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return AIMessage(content=reply if isinstance(reply, str) else json.dumps(reply))
        if system.startswith("You are a careful educational assessor"):
            return AIMessage(content=_CLASSIFY)
        return AIMessage(content=self.answer)


def _rung(message="What would you expect with 250 million sequences?", verdict="none") -> dict:
    return {"verdict": verdict, "message": message}


def _engine(llm, level="standard", control=None):
    from biomentis.agent.tutor import TutorEngine

    saved = {k: os.environ.pop(k, None) for k in ("BIOMENTIS_SCAFFOLD_LEVEL", "BIOMENTIS_SCAFFOLD_CONTROL")}
    try:
        if control:
            os.environ["BIOMENTIS_SCAFFOLD_CONTROL"] = control
        engine = TutorEngine(session_id="scaffold-engine", llm=llm, path=tempfile.mkdtemp())
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    if not engine.scaffold_locked:
        engine.set_scaffold(level=level)
    return engine


def _records(engine, kind=None) -> list[dict]:
    path = engine.logger.path
    if not os.path.exists(path):
        return []
    with open(path) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    return [r for r in recs if kind is None or r.get("kind") == kind]


_ASK = {"task": "Find homologs of the query protein", "mode": "chat", "plot_mode": "student"}
_Q = "Why filter BLAST hits at E < 1e-5?"


def _start(engine, question=_Q, step_id=None):
    return engine.start_question(question, surface="chat", step_id=step_id, ask_kwargs=dict(_ASK))


# --- 1-2: direct vs guided -------------------------------------------------


def test_direct() -> None:
    print("\n[1] a direct answer is released at once")
    engine = _engine(_RouterLLM(_answer(_profile(type="logistics"))))
    step = _start(engine, "Where is the export button?")
    _check("released at once", step.released_turn is not None)
    _check("the turn has today's content",
           step.released_turn["content"] == f"{_ANSWER}\n\n**Follow-up:** {_FOLLOW_UP}", repr(step.released_turn))
    _check("the turn dict has today's keys",
           {"role", "content", "citations", "bloom_level", "dok_level", "rubric_hit", "confidence", "failed",
            "also_consider", "plot_code", "plot_png", "plot_error", "structure_path", "plot_mode"}
           <= set(step.released_turn), sorted(step.released_turn))
    _check("one qa record", len(_records(engine, "qa")) == 1)
    sc = _records(engine, "scaffold")
    _check("one scaffold record, released direct",
           len(sc) == 1 and sc[0]["released"] == "direct" and sc[0]["plan_reason"] == "type_logistics", repr(sc))
    _check("no rung calls", engine.llm.rung_calls == 0)


def test_guided_opens_thread() -> None:
    print("\n[2] a question worth guiding opens a thread")
    engine = _engine(_RouterLLM(_answer(_profile()), rungs=[_rung()]))
    step = _start(engine, step_id=4)
    _check("the thread is open", step.released_turn is None and step.thread is not None)
    t = step.thread
    _check("its first step is the plan's first move",
           t.rungs and t.rungs[0]["move"] == "predict" and t.rungs[0]["message"].startswith("What would you"),
           repr(t.rungs))
    _check("standard/developing plans two steps", t.plan.moves == ("predict", "hint"), repr(t.plan))
    _check("the held-back answer is on the thread", t.answer.get("answer_body") == _ANSWER)
    _check("the task is kept for later steps", t.task == _ASK["task"])
    _check("nothing logged yet", _records(engine) == [] or all(r["kind"] == "scaffold_setting" for r in _records(engine)),
           repr(_records(engine)))
    _check("the chat history is untouched", engine.chat.history == [])


# --- 3: releases -----------------------------------------------------------


def _open(rungs, profile=None, level="standard"):
    engine = _engine(_RouterLLM(_answer(profile or _profile()), rungs=rungs), level=level)
    return engine, _start(engine).thread


def test_releases() -> None:
    print("\n[3] each way the answer is released")
    engine, t = _open([_rung(), _rung("Exactly right.", "correct")])
    step = engine.reply(t, "it's how many hits you'd get by chance")
    _check("correct → earned", step.thread.released == "earned", repr(step.thread.released))
    _check("earned opens with the acknowledgment", step.released_turn["content"].startswith("✓ That's the core of it."))
    _check("the reply is kept on the thread", step.thread.rungs[0]["reply"].startswith("it's how many"))
    qa = _records(engine, "qa")
    _check("the qa record carries the thread", len(qa) == 1 and qa[0]["scaffold"]["released"] == "earned", repr(qa))
    _check("history has the student's reply", "by chance" in engine.chat.history[-2].content)

    engine, t = _open([_rung()])
    step = engine.reveal(t)
    _check("button → released", step.thread.released == "button" and step.released_turn is not None)

    engine, t = _open([_rung(), _rung("Closer. Now think about database size.", "partial"),
                       _rung("Okay.", "partial")])
    step = engine.reply(t, "it filters weak matches")
    _check("a partial reply opens the next step",
           step.released_turn is None and len(step.thread.rungs) == 2 and step.thread.rungs[1]["move"] == "hint",
           repr(step.thread.rungs))
    step = engine.reply(step.thread, "bigger databases give more chance hits")
    _check("a reply after the last step → exhausted", step.thread.released == "exhausted", repr(step.thread.released))

    engine, t = _open([_rung(), _rung("Think about database size.")])
    step = engine.hint(t)
    _check("hint opens the next step", step.released_turn is None and len(step.thread.rungs) == 2)
    _check("the hint is counted", step.thread.hints_requested == 1)
    step = engine.hint(step.thread)
    _check("hint with no steps left → exhausted", step.thread.released == "exhausted")
    _check("no extra rung call for it", engine.llm.rung_calls == 2, engine.llm.rung_calls)

    engine, t = _open([_rung(), _rung("Fair enough.", "gave_up")])
    _check("just tell me → frustration", engine.reply(t, "just tell me").thread.released == "frustration")

    engine, t = _open([_rung(), _rung("Sure.", "new_question")])
    step = engine.reply(t, "what does blastp stand for?")
    _check("a new question → abandoned", step.thread.released == "abandoned")
    _check("the new question is handed back", step.reask == "what does blastp stand for?", repr(step.reask))
    _check("it isn't recorded as a reply", step.thread.rungs[0]["reply"] == "")

    engine = _engine(_RouterLLM(_answer(_profile()), rungs=["junk", RuntimeError("down")]))
    step = _start(engine)
    _check("a failing first step releases the answer (error)",
           step.released_turn is not None and step.thread.released == "error", repr(step.thread))
    engine, t = _open([_rung(), "junk", "junk"])
    step = engine.reply(t, "some attempt here")
    _check("a failing later step releases the answer (error)", step.thread.released == "error")

    engine = _engine(_RouterLLM(_answer({"type": "nonsense"})))
    step = _start(engine)
    _check("a malformed profile → direct", step.released_turn is not None and step.thread.plan.reason == "profile_failed")


def test_stretch() -> None:
    print("\n[4] the stretch question")
    expert = _profile(bloom="Evaluate", shows_reasoning=True)
    stretch = "How would a 10x larger database change your cutoff?"
    engine = _engine(_RouterLLM(_answer(expert, stretch), rungs=[_rung()]), level="challenge")
    step = engine.reveal(_start(engine).thread)
    _check("challenge + expert gets the stretch question",
           step.released_turn["content"].endswith(f"**Stretch question:** {stretch}"), step.released_turn["content"])
    adv = _profile(bloom="Analyze")
    engine = _engine(_RouterLLM(_answer(adv, stretch), rungs=[_rung()]), level="challenge")
    step = engine.reveal(_start(engine).thread)
    _check("challenge + advanced keeps the follow-up", step.released_turn["content"].endswith(_FOLLOW_UP))


def test_round_trip_mid_thread() -> None:
    print("\n[5] a thread continues after a round trip through session state")
    from biomentis.agent.tutor.scaffold import ScaffoldThread

    engine, t = _open([_rung(), _rung("Exactly.", "correct")])
    stored = json.loads(json.dumps(t.to_dict()))
    step = engine.reply(stored, "chance hits in a big database")
    _check("a dict thread is accepted", isinstance(step.thread, ScaffoldThread))
    _check("and released as earned", step.thread.released == "earned")


# --- 6-7: settings and off ------------------------------------------------


def test_settings_and_lock() -> None:
    print("\n[6] settings, logging, and the teacher lock")
    engine = _engine(_RouterLLM(_answer(_profile(type="logistics"))), level="standard")
    _check("defaults: students, unlocked", engine.scaffold_control == "students" and not engine.scaffold_locked)
    _check("a level change is accepted", engine.set_scaffold(level="challenge") and engine.scaffold_level == "challenge")
    _check("a junk level is ignored", engine.set_scaffold(level="max") and engine.scaffold_level == "challenge")
    settings = _records(engine, "scaffold_setting")
    _check("each change is logged once", [r.get("level") for r in settings] == ["challenge"], repr(settings))

    engine.set_scaffold(control="teacher")
    _check("teacher before any question isn't locked yet", not engine.scaffold_locked)
    _check("it can still be changed", engine.set_scaffold(level="light") and engine.scaffold_level == "light")
    _start(engine, "Where is the export button?")
    _check("the first question locks it", engine.scaffold_locked)
    _check("a locked level can't change", engine.set_scaffold(level="off") is False and engine.scaffold_level == "light")
    _check("a locked control can't change",
           engine.set_scaffold(control="students") is False and engine.scaffold_control == "teacher")

    students = _engine(_RouterLLM(_answer(_profile(type="logistics"))))
    _start(students, "Where is the export button?")
    _check("students control never locks", not students.scaffold_locked)
    _check("students can choose off", students.set_scaffold(level="off") and students.scaffold_level == "off")

    preset = _engine(_RouterLLM(_answer(_profile())), control="teacher")
    _check("an env preset is teacher and locked from the start",
           preset.scaffold_control == "teacher" and preset.scaffold_locked)


def test_off_logs_like_before() -> None:
    print("\n[7] at level off nothing new is logged")
    engine = _engine(_RouterLLM(_answer(None)), level="off")
    before = len(_records(engine))
    step = _start(engine)
    recs = _records(engine)[before:]
    _check("released directly", step.released_turn is not None)
    _check("only a qa record", [r["kind"] for r in recs] == ["qa"], repr([r["kind"] for r in recs]))
    _check("with no scaffold key", "scaffold" not in recs[0])
    _check("the system prompt had no question-support keys", engine.llm.rung_calls == 0)


def run_all() -> int:
    print("=" * 60)
    print("Question support: TutorEngine threads")
    print("=" * 60)
    for test in (
        test_direct,
        test_guided_opens_thread,
        test_releases,
        test_stretch,
        test_round_trip_mid_thread,
        test_settings_and_lock,
        test_off_logs_like_before,
    ):
        try:
            test()
        except Exception as e:
            import traceback

            traceback.print_exc()
            _check(f"{test.__name__} ran", False, repr(e))

    print("\n" + "=" * 60)
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    print("=" * 60)
    for name, detail in _FAILED:
        print(f"  ✗ {name}: {detail}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(run_all())
