"""Tests for question support: guiding a student to an answer in steps.

Spec: docs/superpowers/specs/2026-11-04-scaffolded-questions-design.md

What these tests pin down:

  1.  The question profile parses defensively
  2.  The sophistication tier, including "expert" (two of three signals)
  3.  The policy: which questions are answered directly, and the guided
      steps for each level and tier
  4.  The env defaults for level and control
  5.  The give-up patterns, which must not fire on a real attempt
  6.  When a guided exchange releases the answer, and why
  7.  A thread survives a round trip through a plain dict
  8.  The released answer text (today's text at "off")
  9.  The leak check, which skips key points too short to check
 10.  A fallback text for every move
 11.  The rung call: retries, leak regeneration, fallback, fail open

Run with:
    python -m biomentis.agent.tests.test_scaffold
"""

from __future__ import annotations

import os
import sys

# --- Tiny test harness (matches test_chat_visuals.py) ---------------------

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


def _profile(**kw):
    from biomentis.agent.tutor.scaffold import QuestionProfile

    base = dict(
        type="conceptual", bloom="Understand", dok=2, specific=True, shows_reasoning=False, wants_direct=False
    )
    base.update(kw)
    return QuestionProfile(**base)


# --- 1-4: profile, tier, policy, defaults ---------------------------------


def test_profile_parse() -> None:
    print("\n[1] the question profile parses defensively")
    from biomentis.agent.tutor.scaffold import QuestionProfile

    p = QuestionProfile.parse(
        {"type": " Conceptual ", "bloom": "analyze", "dok": "3", "specific": "true",
         "shows_reasoning": True, "wants_direct": "no"}
    )
    _check("type is normalized", p is not None and p.type == "conceptual", repr(p))
    _check("bloom is matched case-insensitively", p is not None and p.bloom == "Analyze", repr(p))
    _check("dok is coerced to int", p is not None and p.dok == 3, repr(p))
    _check("string booleans are read", p is not None and p.specific is True and p.wants_direct is False, repr(p))

    _check("None is rejected", QuestionProfile.parse(None) is None)
    _check("a list is rejected", QuestionProfile.parse(["conceptual"]) is None)
    _check("an unknown type is rejected", QuestionProfile.parse({"type": "rhetorical"}) is None)
    high = QuestionProfile.parse({"type": "evaluative", "dok": 9})
    _check("dok is clamped to 4", high is not None and high.dok == 4, repr(high))
    low = QuestionProfile.parse({"type": "evaluative", "dok": "x"})
    _check("a junk dok falls back to 2", low is not None and low.dok == 2, repr(low))
    odd = QuestionProfile.parse({"type": "evaluative", "bloom": "Memorize"})
    _check("an unknown bloom falls back to Understand", odd is not None and odd.bloom == "Understand", repr(odd))


def test_tier() -> None:
    print("\n[2] the sophistication tier")
    from biomentis.agent.tutor.scaffold import tier

    _check("vague recall question is novice",
           tier(_profile(bloom="Remember", dok=1, specific=False)) == "novice")
    _check("specific understand question is developing",
           tier(_profile(bloom="Understand", dok=2, specific=True)) == "developing")
    _check("apply question is developing", tier(_profile(bloom="Apply", dok=2)) == "developing")
    _check("analyze alone is advanced", tier(_profile(bloom="Analyze", dok=2)) == "advanced")
    _check("evaluate alone is advanced (one expert signal)", tier(_profile(bloom="Evaluate", dok=2)) == "advanced")
    _check("dok 3 alone is advanced", tier(_profile(bloom="Apply", dok=3)) == "advanced")
    _check("reasoning alone is advanced", tier(_profile(shows_reasoning=True)) == "advanced")
    _check("reasoning + evaluate is expert",
           tier(_profile(shows_reasoning=True, bloom="Evaluate", dok=2)) == "expert")
    _check("reasoning + dok 3 is expert", tier(_profile(shows_reasoning=True, bloom="Apply", dok=3)) == "expert")
    _check("create + dok 4 is expert", tier(_profile(bloom="Create", dok=4)) == "expert")
    _check("analyze + dok 3 is only advanced (analyze is not an expert signal)",
           tier(_profile(bloom="Analyze", dok=3)) == "advanced")


def test_policy_table() -> None:
    print("\n[3a] the guided steps for each level and tier")
    from biomentis.agent.tutor.scaffold import decide

    expected = {
        ("light", "novice"): ("orient",),
        ("light", "developing"): ("predict",),
        ("light", "advanced"): ("probe",),
        ("standard", "novice"): ("orient",),
        ("standard", "developing"): ("predict", "hint"),
        ("standard", "advanced"): ("probe", "contrast"),
        ("challenge", "novice"): ("orient", "hint"),
        ("challenge", "developing"): ("predict", "partial"),
        ("challenge", "advanced"): ("probe", "find_flaw", "extend"),
    }
    for (level, t), moves in expected.items():
        plan = decide(level, _profile(), t)
        _check(f"{level}/{t} → {' → '.join(moves)}",
               not plan.direct and plan.moves == moves, repr(plan))

    expert = decide("challenge", _profile(), "expert")
    _check("expert uses the advanced column",
           expert.moves == ("probe", "find_flaw", "extend"), repr(expert))


def test_policy_direct_and_stretch() -> None:
    print("\n[3b] direct answers, procedural questions, stretch questions")
    from biomentis.agent.tutor.scaffold import LEVELS, decide

    failed = decide("standard", None, "developing")
    _check("no profile → direct (profile_failed)", failed.direct and failed.reason == "profile_failed", repr(failed))
    _check("off → direct", decide("off", _profile(), "expert").direct)
    for level in LEVELS:
        _check(f"logistics is direct at {level}", decide(level, _profile(type="logistics"), "expert").direct)
        _check(f"agent_action is direct at {level}", decide(level, _profile(type="agent_action"), "expert").direct)
    for level in ("off", "light", "standard"):
        _check(f"definition is direct at {level}", decide(level, _profile(type="definition"), "advanced").direct)
    d = decide("challenge", _profile(type="definition"), "novice")
    _check("definition under challenge gets one predict step",
           not d.direct and d.moves == ("predict",), repr(d))

    for level in ("light", "standard"):
        _check(f"wants_direct at {level} is direct",
               decide(level, _profile(wants_direct=True), "advanced").direct)
    wd = decide("challenge", _profile(wants_direct=True), "advanced")
    _check("wants_direct under challenge gets one step, with the reveal highlighted",
           not wd.direct and len(wd.moves) == 1 and wd.highlight_reveal, repr(wd))
    _check("the reveal is not highlighted otherwise",
           not decide("challenge", _profile(), "advanced").highlight_reveal)

    proc = decide("challenge", _profile(type="procedural"), "advanced")
    _check("procedural loses one step", proc.moves == ("probe", "find_flaw"), repr(proc))
    proc1 = decide("light", _profile(type="procedural"), "novice")
    _check("procedural keeps at least one step", proc1.moves == ("orient",), repr(proc1))

    for level in LEVELS:
        for t in ("novice", "developing", "advanced", "expert"):
            plan = decide(level, _profile(), t)
            if len(plan.moves) > 3:
                _check(f"hard cap at {level}/{t}", False, repr(plan))

    _check("stretch for challenge + expert", decide("challenge", _profile(), "expert").stretch)
    _check("no stretch for challenge + advanced", not decide("challenge", _profile(), "advanced").stretch)
    _check("no stretch for standard + expert", not decide("standard", _profile(), "expert").stretch)
    _check("no stretch when answered directly",
           not decide("challenge", _profile(type="logistics"), "expert").stretch)


def test_env_defaults() -> None:
    print("\n[4] level and control defaults from the environment")
    from biomentis.agent.tutor.scaffold import scaffold_control_default, scaffold_level_default

    saved = {k: os.environ.pop(k, None) for k in ("BIOMENTIS_SCAFFOLD_LEVEL", "BIOMENTIS_SCAFFOLD_CONTROL")}
    try:
        _check("level defaults to standard", scaffold_level_default() == "standard")
        _check("control defaults to students, unlocked", scaffold_control_default() == ("students", False))
        os.environ["BIOMENTIS_SCAFFOLD_LEVEL"] = " Challenge "
        _check("a valid level is read", scaffold_level_default() == "challenge")
        os.environ["BIOMENTIS_SCAFFOLD_LEVEL"] = "maximum"
        _check("a junk level is standard", scaffold_level_default() == "standard")
        os.environ["BIOMENTIS_SCAFFOLD_CONTROL"] = "TEACHER"
        _check("teacher control is preset and locked", scaffold_control_default() == ("teacher", True))
        os.environ["BIOMENTIS_SCAFFOLD_CONTROL"] = "admin"
        _check("a junk control is students", scaffold_control_default() == ("students", False))
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


# --- 5-8: thread state and release ----------------------------------------


def _thread(moves=("predict", "hint"), stretch=False, **answer):
    from biomentis.agent.tutor.scaffold import ScaffoldPlan, ScaffoldThread

    ans = {"answer_body": "E-values count expected chance hits.", "follow_up": "What would 1e-10 change?",
           "stretch_question": "", "key_points": []}
    ans.update(answer)
    return ScaffoldThread(
        question="Why filter at E < 1e-5?",
        surface="chat",
        step_id=None,
        level="standard",
        control="students",
        profile={"type": "conceptual"},
        tier="developing",
        plan=ScaffoldPlan(direct=False, moves=tuple(moves), stretch=stretch, reason="table"),
        answer=ans,
    )


def _add_rung(thread, move, reply="", verdict="none"):
    thread.rungs.append({"move": move, "message": "m", "reply": reply, "verdict": verdict, "leak_retry": False})


def test_give_up_patterns() -> None:
    print("\n[5] give-up patterns")
    from biomentis.agent.tutor.scaffold import is_give_up

    for text in ("just tell me", "Just tell me the answer!", "idk", "I don't know", "i dont know.",
                 "no idea", "I give up", "?", "??", "can you just tell me please", "idk, just tell me"):
        _check(f"give-up: {text!r}", is_give_up(text))
    for text in ("I don't know if the threshold matters, but maybe it filters noise",
                 "false positives", "tell me if I'm right: it's about chance hits", "no", "",
                 "maybe 0.05 of them?"):
        _check(f"not a give-up: {text!r}", not is_give_up(text))


def test_release_rules() -> None:
    print("\n[6] when the answer is released")
    t = _thread()
    _add_rung(t, "predict")
    _check("new_question → abandoned", t.release_reason_for("new_question", "what is BLAST?") == "abandoned")
    _check("correct → earned", t.release_reason_for("correct", "chance hits") == "earned")
    _check("short correct reply is earned, not frustration",
           t.release_reason_for("correct", "false positives") == "earned")
    _check("gave_up → frustration", t.release_reason_for("gave_up", "hmm") == "frustration")
    _check("give-up text → frustration even if judged partial",
           t.release_reason_for("partial", "just tell me") == "frustration")
    _check("a real attempt keeps going", t.release_reason_for("partial", "it removes weak matches") is None)

    t2 = _thread()
    _add_rung(t2, "predict", reply="um ok", verdict="off_track")
    _add_rung(t2, "hint")
    t2.plan = t2.plan.__class__(direct=False, moves=("predict", "hint", "partial"), stretch=False, reason="table")
    _check("two short off-track replies → frustration",
           t2.release_reason_for("off_track", "dunno") == "frustration")
    _check("short reply after a short one, but partial, keeps going",
           t2.release_reason_for("partial", "fewer hits") is None)

    t3 = _thread()
    _add_rung(t3, "predict", reply="false positives", verdict="partial")
    _add_rung(t3, "hint")
    t3.plan = t3.plan.__class__(direct=False, moves=("predict", "hint", "partial"), stretch=False, reason="table")
    _check("a short partial reply earlier doesn't count toward frustration",
           t3.release_reason_for("off_track", "dunno") is None)

    t4 = _thread(moves=("predict", "hint"))
    _add_rung(t4, "predict", reply="it filters stuff out", verdict="partial")
    _add_rung(t4, "hint")
    _check("all steps used → exhausted", t4.release_reason_for("partial", "it removes weak matches") == "exhausted")
    _check("correct on the last step is still earned", t4.release_reason_for("correct", "chance hits") == "earned")

    t5 = _thread(moves=("probe", "find_flaw", "extend"))
    _check("next_move walks the plan", t5.next_move() == "probe")
    for m in ("probe", "find_flaw", "extend"):
        _add_rung(t5, m)
    _check("next_move is None when the plan is spent", t5.next_move() is None)


def test_thread_round_trip() -> None:
    print("\n[7] a thread survives a round trip through a dict")
    from biomentis.agent.tutor.scaffold import ScaffoldThread

    t = _thread(stretch=True, key_points=["expected chance hits"])
    _add_rung(t, "predict", reply="chance", verdict="partial")
    t.hints_requested = 1
    t.released = "earned"
    d = t.to_dict()
    import json

    json.dumps(d)  # must be plain data
    back = ScaffoldThread.from_dict(d)
    _check("every field survives", back.to_dict() == d, f"{back.to_dict()} != {d}")
    _check("the plan comes back as a ScaffoldPlan", back.plan.moves == ("predict", "hint") and back.plan.stretch)


def test_compose_released() -> None:
    print("\n[8] the released answer text")
    from biomentis.agent.tutor.scaffold import compose_released

    t = _thread()
    t.released = "button"
    _check("today's composition when nothing special happened",
           compose_released(t) == "E-values count expected chance hits.\n\n**Follow-up:** What would 1e-10 change?",
           repr(compose_released(t)))
    t.released = "earned"
    _check("earned opens with the acknowledgment",
           compose_released(t).startswith("✓ That's the core of it.\n\nE-values"), repr(compose_released(t)))
    s = _thread(stretch=True, stretch_question="How would a smaller database change this?")
    s.released = "earned"
    out = compose_released(s)
    _check("a stretch question replaces the follow-up",
           out.endswith("**Stretch question:** How would a smaller database change this?")
           and "Follow-up" not in out, repr(out))
    s2 = _thread(stretch=True, stretch_question="")
    s2.released = "button"
    _check("no stretch question → the follow-up stays", compose_released(s2).endswith("1e-10 change?"))
    n = _thread(follow_up="")
    n.released = "exhausted"
    _check("no follow-up → the body alone", compose_released(n) == "E-values count expected chance hits.")


# --- 9-10: leak check and fallbacks ---------------------------------------


def test_leak_check() -> None:
    print("\n[9] the leak check")
    from biomentis.agent.tutor.scaffold import leaked_point

    points = ["expected number of chance hits", "noise", "E-value"]
    _check("a verbatim leak is caught",
           leaked_point("It's the expected number of chance hits.", points) == "expected number of chance hits")
    _check("case, punctuation and bold are ignored",
           leaked_point("It is the **Expected number** of chance-hits!", points) == "expected number of chance hits")
    _check("short key points are never checked",
           leaked_point("Think about noise and the E-value here.", points) is None)
    _check("no leak → None", leaked_point("What does the E-value count?", points) is None)
    _check("one 14-character word is below the bar",
           leaked_point("Consider overdispersion here.", ["overdispersion!!"]) is None)
    _check("15+ characters is checkable",
           leaked_point("Think about multiple-testing correction.", ["multiple testing correction"])
           == "multiple testing correction")
    _check("part of a word doesn't count",
           leaked_point("unexpected number of chance hitsx", ["expected number of chance hits"]) is None)


def test_fallbacks() -> None:
    print("\n[10] a fallback for every move")
    from biomentis.agent.tutor.scaffold import MOVES, fallback_rung

    for move in MOVES:
        with_step = fallback_rung(move, step_id=3)
        without = fallback_rung(move, step_id=None)
        _check(f"{move} has a fallback", bool(with_step.strip()) and bool(without.strip()))
        _check(f"{move} fallback has no unfilled placeholder", "{" not in with_step + without)
    _check("orient names the step", "step 3" in fallback_rung("orient", step_id=3))
    _check("orient without a step names the last step",
           "the agent's last step" in fallback_rung("orient", step_id=None))


# --- 11: the rung call ----------------------------------------------------


class _SeqLLM:
    """Returns the queued replies in order and records each call's messages."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[list] = []

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.calls.append(list(messages))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return AIMessage(content=reply if isinstance(reply, str) else __import__("json").dumps(reply))


def _rung(llm, thread=None, move="predict", reply=""):
    from biomentis.agent.tutor.scaffold import next_rung

    thread = thread or _thread(key_points=["expected number of chance hits"])
    return next_rung(llm, thread, move, task="Find homologs", cards=[{"what": "Ran BLAST"}], latest_reply=reply)


def test_rung_call() -> None:
    print("\n[11] the rung call")
    from biomentis.agent.tutor.scaffold import FALLBACK_RUNGS, RungFailed, fallback_rung

    good = {"verdict": "none", "message": "What would you expect with 250 million sequences?"}
    llm = _SeqLLM(good)
    verdict, message, leak_retry = _rung(llm)
    _check("a good reply comes back", (verdict, message, leak_retry) == ("none", good["message"], False),
           repr((verdict, message, leak_retry)))
    system = llm.calls[0][0].content
    user = llm.calls[0][-1].content
    _check("the prompt says never to state the answer", "never state" in system.lower())
    _check("the prompt defines the move", "MOVE: predict" in user and "predict an outcome" in user.lower(), user[:400])
    _check("the prompt carries the held-back answer", "E-values count expected chance hits." in user)
    _check("the prompt says which step this is", "1 of 2" in user, user[:400])

    llm = _SeqLLM("not json", good)
    _check("an unparseable reply is retried", _rung(llm)[1] == good["message"] and len(llm.calls) == 2)
    llm = _SeqLLM({"verdict": "none", "message": "  "}, good)
    _check("an empty message is retried", _rung(llm)[1] == good["message"])
    # A verdict that releases the answer needs no next step: an empty
    # message is fine there (else a new question mid-thread was lost).
    for verdict in ("new_question", "correct", "gave_up"):
        llm = _SeqLLM({"verdict": verdict, "message": ""})
        _check(f"{verdict} needs no message", _rung(llm, reply="something")[:2] == (verdict, "")
               and len(llm.calls) == 1, repr(llm.calls[1:]))

    try:
        _rung(_SeqLLM("nope", RuntimeError("down")))
        _check("two failures raise RungFailed", False, "no exception")
    except RungFailed:
        _check("two failures raise RungFailed", True)

    leak = {"verdict": "none", "message": "It's the expected number of chance hits, see?"}
    llm = _SeqLLM(leak, good)
    verdict, message, leak_retry = _rung(llm)
    _check("a leak is regenerated once", message == good["message"] and leak_retry, repr((message, leak_retry)))
    _check("the regeneration names what leaked",
           "expected number of chance hits" in llm.calls[1][-1].content, llm.calls[1][-1].content)

    llm = _SeqLLM(leak, leak)
    _, message, leak_retry = _rung(llm)
    _check("a second leak falls back to the fixed text",
           message == FALLBACK_RUNGS["predict"] and leak_retry, repr(message))
    llm = _SeqLLM(leak, "garbage")
    _check("a failed regeneration also falls back", _rung(llm)[1] == FALLBACK_RUNGS["predict"])

    t = _thread(key_points=[])
    _add_rung(t, "predict")
    llm = _SeqLLM({"verdict": "partial", "message": "Right, and now why that many?"})
    verdict, _, _ = _rung(llm, thread=t, move="hint", reply="lots of false hits")
    _check("a verdict on a reply comes back", verdict == "partial", verdict)
    _check("the reply is in the prompt", "lots of false hits" in llm.calls[0][-1].content)
    llm = _SeqLLM({"verdict": "brilliant", "message": "Go on."})
    _check("an unknown verdict is none", _rung(llm, thread=t, move="hint", reply="x")[0] == "none")
    llm = _SeqLLM({"verdict": "correct", "message": "Hm?"})
    _check("rung 1 has no reply to judge, so its verdict is none", _rung(llm)[0] == "none")

    # A hint (no reply after an earlier step): seen live, a model echoed the
    # previous step word for word instead of making the new move.
    earlier = "Could you clarify what happens to the number of hits at 1e-10?"
    t = _thread(moves=("probe", "find_flaw", "extend"), key_points=[])
    t.rungs.append({"move": "probe", "message": earlier, "reply": "", "verdict": "none", "leak_retry": False})
    fresh = {"verdict": "none", "message": "Here is a flawed version: a stricter cutoff keeps more hits. What's wrong?"}
    llm = _SeqLLM(fresh)
    _rung(llm, thread=t, move="find_flaw")
    user = llm.calls[0][-1].content
    _check("a hint is named in the prompt", "asked for a hint" in user, user[-400:])
    _check("the prompt says not to repeat a step", "repeat" in llm.calls[0][0].content.lower())

    llm = _SeqLLM({"verdict": "none", "message": earlier}, fresh)
    verdict, message, retried = _rung(llm, thread=t, move="find_flaw")
    _check("a repeated step is regenerated once", message == fresh["message"] and retried, repr(message))
    _check("the regeneration says it repeated", "repeat" in llm.calls[1][-1].content.lower(),
           llm.calls[1][-1].content)
    llm = _SeqLLM({"verdict": "none", "message": f"  {earlier.upper()} "}, {"verdict": "none", "message": earlier})
    _check("a repeat twice falls back to the move's fixed text",
           _rung(llm, thread=t, move="find_flaw")[1] == fallback_rung("find_flaw", None))


def run_all() -> int:
    print("=" * 60)
    print("Question support (scaffold) tests")
    print("=" * 60)
    for test in (
        test_profile_parse,
        test_tier,
        test_policy_table,
        test_policy_direct_and_stretch,
        test_env_defaults,
        test_give_up_patterns,
        test_release_rules,
        test_thread_round_trip,
        test_compose_released,
        test_leak_check,
        test_fallbacks,
        test_rung_call,
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
