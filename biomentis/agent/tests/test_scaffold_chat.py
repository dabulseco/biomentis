"""Tests for `TutorChat.ask` with question support (scaffold levels).

Spec: docs/superpowers/specs/2026-11-04-scaffolded-questions-design.md

What these tests pin down:

  1.  At level "off", ask() is exactly what it was before question support:
      the same messages to the model, the same turn, the same log record.
      The baseline was captured from the unmodified chat.py with
      `--capture` and is kept in fixtures/scaffold_off_baseline.json.
  2.  Above "off", the same call also asks for the question profile and the
      key points, and they land on the turn
  3.  A stretch question is asked for only under "challenge"
  4.  A malformed profile leaves an empty dict (the engine then answers
      directly)
  5.  The answer body and follow-up are kept apart on the turn at every level
  6.  record=False defers the log record and the history; record() writes
      them, with the guided exchange when there was one
  7.  A soft failure is recorded at once and says so

Run with:
    python -m biomentis.agent.tests.test_scaffold_chat
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


BASELINE = os.path.join(os.path.dirname(__file__), "fixtures", "scaffold_off_baseline.json")
_CLASSIFY = json.dumps({"bloom_level": "Analyze", "dok_level": 3, "rubric_hit": [], "confidence": 0.7})
_REPLY = {
    "answer": "An E-value counts the hits you'd expect by chance in a database this size.",
    "follow_up": "What would happen at a cutoff of 0.05?",
    "citations": [],
    "also_consider": [{"point": "Database size", "why": "E-values scale with it."}],
    "plot_code": "",
    "structure": "",
}
_PROFILE = {"type": "conceptual", "bloom": "Analyze", "dok": 3, "specific": True,
            "shows_reasoning": False, "wants_direct": False}


class _LLM:
    """Replies with `reply` to the answer call and a classification after."""

    def __init__(self, reply: dict | str):
        self.model = "fake"
        self.reply = reply if isinstance(reply, str) else json.dumps(reply)
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.reply if len(self.calls) == 1 else _CLASSIFY)


def _chat(reply, root: str | None = None):
    from biomentis.agent.tutor.chat import ChatTurn, TutorChat
    from biomentis.agent.tutor.log import SessionLogger

    logger = SessionLogger("scaffold-chat", path=root or tempfile.mkdtemp())
    chat = TutorChat(_LLM(reply), logger=logger)
    chat.push_recent_card({"event_type": "code", "what": "Ran BLAST", "why": "Find homologs"})
    chat.push_recent_card({"event_type": "observation", "what": "412 hits", "why": "Filter next"})
    chat.history = [
        ChatTurn(role="user", content="What is BLAST?"),
        ChatTurn(role="assistant", content="A sequence similarity search."),
    ]
    return chat


def _ask(chat, **kw):
    return chat.ask(
        "Why filter BLAST hits at E < 1e-5?",
        context="step_id=4",
        task="Find homologs of the query protein",
        transcript="[code] Ran BLAST\nblastp -evalue 1e-5",
        last_answer="Kept 37 hits.",
        mode="chat",
        step_id=4,
        plot_mode="student",
        **kw,
    )


def _log_records(chat) -> list[dict]:
    if not os.path.exists(chat.logger.path):
        return []
    with open(chat.logger.path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _snapshot(chat, turn) -> dict:
    """What the baseline compares: the messages, the turn, the log, the history."""
    from dataclasses import asdict

    msgs = [[type(m).__name__, m.content] for m in chat.llm.calls[0]]
    records = _log_records(chat)
    for r in records:
        r.pop("ts", None)
    return {
        "messages": msgs,
        "turn": asdict(turn),
        "log": records,
        "history": [[t.role, t.content] for t in chat.history],
    }


def capture() -> None:
    """Write the baseline from the current chat.py. Run once, before changing it."""
    chat = _chat(_REPLY)
    snap = _snapshot(chat, _ask(chat))
    with open(BASELINE, "w") as f:
        json.dump(snap, f, indent=1, ensure_ascii=False)
    print(f"wrote {BASELINE}")


# --- 1: off is unchanged --------------------------------------------------


def test_off_matches_baseline() -> None:
    print("\n[1] level off is exactly the old behaviour")
    with open(BASELINE) as f:
        base = json.load(f)
    chat = _chat(_REPLY)
    snap = _snapshot(chat, _ask(chat, scaffold_level="off"))
    _check("same messages to the model", snap["messages"] == base["messages"])
    _check("same log record", snap["log"] == base["log"], f"{snap['log']} != {base['log']}")
    _check("same history", snap["history"] == base["history"])
    old_keys = {k: v for k, v in snap["turn"].items() if k in base["turn"]}
    _check("same turn fields", old_keys == base["turn"], f"{old_keys} != {base['turn']}")
    default = _chat(_REPLY)
    _check("scaffold_level defaults to off",
           _snapshot(default, _ask(default))["messages"] == base["messages"])


# --- 2-5: profile, key points, stretch, body ------------------------------


def test_profile_and_key_points() -> None:
    print("\n[2] above off, the profile and key points come back on the turn")
    reply = dict(_REPLY, question_profile=_PROFILE,
                 key_points=[" expected chance hits ", "database size", "", 7, "multiple testing", "extra"])
    chat = _chat(reply)
    turn = _ask(chat, scaffold_level="standard")
    system = chat.llm.calls[0][0].content
    _check("the system prompt asks for the profile", "question_profile" in system)
    _check("the system prompt asks for key points", "key_points" in system)
    _check("one call, no extra", len(chat.llm.calls) == 2, f"{len(chat.llm.calls)} calls")
    _check("the profile is on the turn", turn.question_profile == _PROFILE, repr(turn.question_profile))
    _check("key points are cleaned and capped at 3",
           turn.key_points == ["expected chance hits", "database size", "multiple testing"],
           repr(turn.key_points))


def test_stretch_only_under_challenge() -> None:
    print("\n[3] the stretch question is requested only under challenge")
    reply = dict(_REPLY, question_profile=_PROFILE, key_points=[],
                 stretch_question="How would a 10x larger database change your cutoff?")
    for level in ("light", "standard"):
        chat = _chat(reply)
        turn = _ask(chat, scaffold_level=level)
        _check(f"{level}: not requested", "stretch_question" not in chat.llm.calls[0][0].content)
        _check(f"{level}: ignored if sent anyway", turn.stretch_question == "", repr(turn.stretch_question))
    chat = _chat(reply)
    turn = _ask(chat, scaffold_level="challenge")
    _check("challenge: requested", "stretch_question" in chat.llm.calls[0][0].content)
    _check("challenge: on the turn", turn.stretch_question.startswith("How would a 10x"), repr(turn.stretch_question))


def test_malformed_profile() -> None:
    print("\n[4] a malformed profile leaves an empty dict")
    for bad in ("conceptual", {"type": "rhetorical"}, None, ["x"]):
        chat = _chat(dict(_REPLY, question_profile=bad))
        turn = _ask(chat, scaffold_level="standard")
        _check(f"profile {bad!r} → {{}}", turn.question_profile == {}, repr(turn.question_profile))
    chat = _chat(_REPLY)
    _check("a missing profile → {}", _ask(chat, scaffold_level="standard").question_profile == {})


def test_body_and_follow_up() -> None:
    print("\n[5] the answer body and follow-up are kept apart")
    for level in ("off", "standard"):
        chat = _chat(dict(_REPLY, question_profile=_PROFILE))
        turn = _ask(chat, scaffold_level=level)
        _check(f"{level}: answer_body", turn.answer_body == _REPLY["answer"], repr(turn.answer_body))
        _check(f"{level}: follow_up", turn.follow_up == _REPLY["follow_up"], repr(turn.follow_up))
        _check(f"{level}: content still has both", turn.content.endswith("**Follow-up:** " + _REPLY["follow_up"]))


# --- 6-7: deferred recording ----------------------------------------------


def test_deferred_record() -> None:
    print("\n[6] record=False defers the log and the history")
    from biomentis.agent.tutor.scaffold import ScaffoldPlan, ScaffoldThread

    chat = _chat(dict(_REPLY, question_profile=_PROFILE))
    history_before = list(chat.history)
    turn = _ask(chat, scaffold_level="standard", record=False)
    _check("no log record yet", _log_records(chat) == [], repr(_log_records(chat)))
    _check("history unchanged", chat.history == history_before)
    _check("the turn says it isn't recorded", turn.recorded is False)

    thread = ScaffoldThread(
        question="Why filter BLAST hits at E < 1e-5?", surface="chat", step_id=4, level="standard",
        control="students", profile=_PROFILE, tier="advanced",
        plan=ScaffoldPlan(direct=False, moves=("probe", "contrast"), reason="table"), answer={},
        rungs=[{"move": "probe", "message": "What are you assuming?", "reply": "that hits are real",
                "verdict": "partial", "leak_retry": False}],
        released="button",
    )
    chat.record("Why filter BLAST hits at E < 1e-5?", turn, step_id=4, thread=thread)
    recs = _log_records(chat)
    _check("one qa record", len(recs) == 1 and recs[0]["kind"] == "qa", repr(recs))
    _check("it carries the thread", recs and recs[0].get("scaffold", {}).get("released") == "button")
    _check("it has today's qa fields",
           recs and all(k in recs[0] for k in ("question", "answer", "bloom_level", "dok_level", "citations",
                                                 "also_consider", "step_id", "plot_mode")))
    _check("history gained two turns", len(chat.history) == len(history_before) + 2)
    _check("the user turn carries the student's replies",
           "that hits are real" in chat.history[-2].content and "1e-5" in chat.history[-2].content,
           repr(chat.history[-2].content))
    _check("the assistant turn is the released answer", chat.history[-1] is turn)
    _check("the turn is now recorded", turn.recorded is True)

    plain = _chat(_REPLY)
    t2 = _ask(plain, scaffold_level="standard", record=False)
    plain.record("Why filter BLAST hits at E < 1e-5?", t2, step_id=4)
    recs2 = _log_records(plain)
    _check("without a thread there's no scaffold key", recs2 and "scaffold" not in recs2[0], repr(recs2))
    _check("without a thread the user turn is the question",
           plain.history[-2].content == "Why filter BLAST hits at E < 1e-5?")


def test_soft_failure_recorded() -> None:
    print("\n[7] a soft failure is recorded at once")
    class _Junk:
        model = "fake"

        def invoke(self, messages):
            return AIMessage(content="not json at all")

    chat = _chat(_REPLY)
    chat.llm = _Junk()
    turn = _ask(chat, scaffold_level="standard", record=False)
    _check("it failed", turn.failed)
    _check("it is already recorded", turn.recorded is True)
    _check("one qa record", len(_log_records(chat)) == 1)


def run_all() -> int:
    print("=" * 60)
    print("Question support: TutorChat.ask")
    print("=" * 60)
    for test in (
        test_off_matches_baseline,
        test_profile_and_key_points,
        test_stretch_only_under_challenge,
        test_malformed_profile,
        test_body_and_follow_up,
        test_deferred_record,
        test_soft_failure_recorded,
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
    if "--capture" in sys.argv:
        capture()
        sys.exit(0)
    sys.exit(run_all())
