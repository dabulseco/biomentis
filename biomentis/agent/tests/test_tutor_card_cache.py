"""Tests for teaching cards that fail, and for what the tutor teaches about.

Regression origin (session of 2026-10-25): every tutor-mode run showed
"🎓 Teaching card unavailable for this step (LLM call failed)" on step 1,
whatever the model. Three things compounded:

  1.  A failed card was cached like a good one, in both card caches, and
      neither cache key named the model. `ui_tutor._CARD_CACHE` lives for
      the whole Streamlit process, so one failure on `gemma4:26b-mlx` was
      served — instantly, without a call — to every later run whose step
      had the same text, including after switching models.
  2.  Step 1 always had the same text, because it was not agent work: it
      was the parse-error correction (`PARSE_ERROR_NUDGE`), a HumanMessage
      that happens to contain "<execute>", so `stream_agent_events` cut it
      at the tag and showed it as "🤔 Reasoning". The tutor then paused on
      it and tried to teach it.
  3.  The agent's first reply had no tags at all, which is what triggered
      the correction in the first place (see test_runaway_generation).

What these tests pin down:

  1.  The correction is shown as a status line, never as a reasoning step
  2.  Self-critic feedback that mentions a tag is not a reasoning step either
  3.  A failed card is not cached by the generator — the next try calls again
  4.  A good card is still cached
  5.  A different model gets its own card
  6.  The UI-level card cache follows the same two rules
  7.  A reply with unescaped LaTeX (`$\\Delta\\lambda$`) still parses
  8.  One bad card reply is retried once, and only once
  9.  A tutor chat reply with no answer text is recovered or retried once
      (session of 2026-10-26: "(tutor returned an empty answer)")

Run with:
    python -m biomentis.agent.tests.test_tutor_card_cache
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

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


# --- Fakes ----------------------------------------------------------------

_CARD_JSON = json.dumps(
    {
        "technical": {"what": "Samples diameters", "why": "Inputs for the Brus equation"},
        "scientific": {"what": "Size sets the band gap", "why": "Quantum confinement"},
        "bloom_target": "Apply",
        "dok_target": 2,
    }
)


class _ScriptedLLM:
    """Plays back a list of outcomes: a string reply, or an exception to raise."""

    def __init__(self, outcomes: list, model: str = "fake-model"):
        self.model = model
        self.outcomes = list(outcomes)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else _CARD_JSON
        if isinstance(outcome, Exception):
            raise outcome
        return AIMessage(content=outcome)


def _event(content: str = "d = np.linspace(2, 8, 50)"):
    return SimpleNamespace(type="code", title="🛠️ Executing code...", content=content, file_path=None)


def _scripted_agent(messages: list):
    """An agent whose graph appends `messages` one state at a time."""
    history: list = []

    def _stream(inputs, stream_mode=None, config=None):
        history.extend(inputs["messages"])
        for m in messages:
            history.append(m)
            yield {"messages": list(history)}

    return SimpleNamespace(app=SimpleNamespace(stream=_stream), use_tool_retriever=False, path="/tmp", user_task="")


def _stream(messages: list) -> list:
    from biomentis.ui_core import stream_agent_events

    return list(stream_agent_events(_scripted_agent(messages), "Design a green quantum dot.", [], [], thread_id=1))


# --- Tests ----------------------------------------------------------------


def test_correction_is_a_status_not_reasoning() -> None:
    print("\n[1] The parse-error correction is a status line, not a reasoning step")
    from biomentis.utils import PARSE_ERROR_NUDGE

    events = _stream(
        [
            AIMessage(content="Let me think about quantum dots at length..."),
            HumanMessage(content=PARSE_ERROR_NUDGE),
            AIMessage(content="Compare the three materials.\n<execute>print(1)</execute>"),
        ]
    )
    reasoning = [e.content for e in events if e.type == "reasoning"]
    _check(
        "the correction text never becomes a reasoning step",
        not any("Each response must include" in r for r in reasoning),
        f"reasoning steps: {reasoning}",
    )
    _check(
        "the real reasoning after it is still shown",
        reasoning == ["Compare the three materials."],
        f"reasoning steps: {reasoning}",
    )
    statuses = [e.content for e in events if e.type == "status"]
    _check(
        "a status line says the model is being asked to retry",
        any("no <execute> or <solution> tag" in s for s in statuses),
        f"statuses: {statuses}",
    )


def test_critic_feedback_is_not_reasoning() -> None:
    print("\n[2] Self-critic feedback that mentions a tag is not a reasoning step")
    events = _stream(
        [
            AIMessage(content="<solution>InP</solution>"),
            HumanMessage(
                content="Wait... this is not enough to solve the task. Here are some feedbacks for "
                "improvement:\nRun the Brus model in an <execute> block before answering."
            ),
        ]
    )
    reasoning = [e.content for e in events if e.type == "reasoning"]
    _check("no reasoning step is made from the feedback", reasoning == [], f"reasoning steps: {reasoning}")


def test_failed_card_is_not_cached() -> None:
    print("\n[3] A failed card is not cached by the generator")
    from biomentis.agent.tutor.instruction import InstructionGenerator

    # Two failures: one card's worth of attempts (see test [8] for the retry).
    llm = _ScriptedLLM([ConnectionError("ollama busy"), ConnectionError("ollama busy"), _CARD_JSON])
    gen = InstructionGenerator(llm)
    first = gen.generate(_event())
    second = gen.generate(_event())
    _check("the first card is the failure placeholder", first._generation_failed, repr(first))
    _check("the next step calls the model again", llm.calls == 3, f"calls={llm.calls}")
    _check("and gets a real card", not second._generation_failed and second.what == "Samples diameters", repr(second))

    bad = "Sure! Here is a teaching note without any JSON."
    llm = _ScriptedLLM([bad, bad, _CARD_JSON])
    gen = InstructionGenerator(llm)
    gen.generate(_event())
    second = gen.generate(_event())
    _check("an unparseable reply is not cached either", not second._generation_failed, f"calls={llm.calls}")


def test_good_card_is_cached() -> None:
    print("\n[4] A good card is still cached")
    from biomentis.agent.tutor.instruction import InstructionGenerator

    llm = _ScriptedLLM([_CARD_JSON])
    gen = InstructionGenerator(llm)
    a = gen.generate(_event())
    b = gen.generate(_event())
    _check("the same step is not re-billed", llm.calls == 1, f"calls={llm.calls}")
    _check("and the cached card is the same card", a is b)


def test_model_switch_gets_its_own_card() -> None:
    print("\n[5] A different model gets its own card")
    from biomentis.agent.tutor.instruction import InstructionGenerator

    mlx = _ScriptedLLM([_CARD_JSON], model="gemma4:26b-mlx")
    cloud = _ScriptedLLM([_CARD_JSON], model="gemma4:31b-cloud")
    gen = InstructionGenerator(mlx)
    gen.generate(_event())
    gen.llm = cloud
    gen.generate(_event())
    _check("the new model is called for the same step", cloud.calls == 1, f"cloud calls={cloud.calls}")


def test_ui_card_cache() -> None:
    print("\n[6] The UI-level card cache follows the same rules")
    from biomentis import ui_tutor
    from biomentis.agent.tutor.instruction import InstructionGenerator

    def _tutor(llm):
        return SimpleNamespace(
            modes=("technical", "scientific"),
            kb=None,
            llm=llm,
            instruction_gen=InstructionGenerator(llm),
            chat=SimpleNamespace(push_recent_card=lambda card: None),
            logger=SimpleNamespace(log=lambda *a, **k: None),
        )

    run = {"prompt": "Design a green quantum dot.", "step_id": 1}
    ui_tutor._CARD_CACHE.clear()
    llm = _ScriptedLLM([ConnectionError("ollama busy"), ConnectionError("ollama busy"), _CARD_JSON])
    tutor = _tutor(llm)
    first = ui_tutor._generate_or_get_card(_event(), run, tutor)
    second = ui_tutor._generate_or_get_card(_event(), run, tutor)
    _check("a failure is not served again from the UI cache", first._generation_failed and not second._generation_failed,
           f"first failed={first._generation_failed}, second failed={second._generation_failed}")

    ui_tutor._CARD_CACHE.clear()
    mlx = _ScriptedLLM([_CARD_JSON], model="gemma4:26b-mlx")
    cloud = _ScriptedLLM([_CARD_JSON], model="gemma4:31b-cloud")
    ui_tutor._generate_or_get_card(_event(), run, _tutor(mlx))
    ui_tutor._generate_or_get_card(_event(), run, _tutor(cloud))
    _check("the UI cache key names the model", cloud.calls == 1, f"cloud calls={cloud.calls}")
    ui_tutor._CARD_CACHE.clear()


def test_latex_in_a_reply_still_parses() -> None:
    print("\n[7] A reply with unescaped LaTeX still parses")
    from biomentis.agent.tutor import critic
    from biomentis.agent.tutor.instruction import InstructionGenerator, _extract_json

    # What gemma4:26b-mlx actually wrote, 4/4 times, for a step about
    # "color shift per nanometer": `\D` and `\l` are not JSON escapes, so a
    # strict parse rejected the whole otherwise-complete card.
    latex = (
        '{"technical": {"what": "Defines the metric ($\\Delta\\lambda / \\Delta d$)", "why": "Stability"},'
        ' "scientific": {"what": "Quantum confinement", "why": "Smaller dots, wider gap"}}'
    )
    data = _extract_json(latex)
    _check("the card JSON parses", data is not None)
    _check(
        "the LaTeX survives as written",
        bool(data) and data["technical"]["what"] == "Defines the metric ($\\Delta\\lambda / \\Delta d$)",
        repr(data and data["technical"]["what"]),
    )

    escaped = '{"what": "line one\\nline two, a \\"quote\\", and \\\\Delta"}'
    data = _extract_json(escaped)
    _check(
        "properly escaped text is untouched",
        bool(data) and data["what"] == 'line one\nline two, a "quote", and \\Delta',
        repr(data and data["what"]),
    )

    card = InstructionGenerator(_ScriptedLLM([latex])).generate(_event())
    _check("the generator makes a real card from it", not card._generation_failed, repr(card))
    _check("the critic parses it too", critic._extract_json(latex) is not None)


def test_one_bad_reply_is_retried() -> None:
    print("\n[8] One bad reply is retried; two give up")
    from biomentis.agent.tutor.instruction import InstructionGenerator

    for label, first in (("an exception", ConnectionError("ollama busy")),
                         ("a non-JSON reply", "Here is a teaching note."),
                         ("an all-empty card", '{"technical": {}, "scientific": {}}')):
        llm = _ScriptedLLM([first, _CARD_JSON])
        card = InstructionGenerator(llm).generate(_event())
        _check(f"after {label}, the retry gives a real card", not card._generation_failed and llm.calls == 2,
               f"failed={card._generation_failed} calls={llm.calls}")

    llm = _ScriptedLLM(["nope", "still nope", _CARD_JSON])
    card = InstructionGenerator(llm).generate(_event())
    _check("two bad replies give the placeholder, without a third call",
           card._generation_failed and llm.calls == 2, f"failed={card._generation_failed} calls={llm.calls}")


def test_empty_chat_answer() -> None:
    print("\n[9] A chat reply with no answer text is recovered or retried")
    from biomentis.agent.tutor.chat import TutorChat

    good = json.dumps({"answer": "Yes: ZnTe shifts least per nm.", "follow_up": "", "citations": []})
    for label, reply in (("a nested object", json.dumps({"response": {"answer": "Yes: nested."}})),
                         ("a renamed key", json.dumps({"Response": "Yes: renamed."})),
                         ("only a follow_up", json.dumps({"answer": "", "follow_up": "Yes: in follow_up."}))):
        llm = _ScriptedLLM([reply])
        turn = TutorChat(llm).ask("Is drift related to size?", mode="chat")
        _check(f"{label} is recovered without a retry", turn.content.startswith("Yes:") and llm.calls == 2,
               f"content={turn.content!r} calls={llm.calls}")

    for label, first in (("an empty answer", json.dumps({"answer": "", "follow_up": ""})),
                         ("a non-JSON reply", "Sure, here you go."),
                         ("an exception", ConnectionError("ollama busy"))):
        llm = _ScriptedLLM([first, good])
        turn = TutorChat(llm).ask("Is drift related to size?", mode="chat")
        _check(f"after {label}, the retry gives the answer", turn.content.startswith("Yes: ZnTe"),
               f"content={turn.content!r}")

    llm = _ScriptedLLM([json.dumps({"answer": ""}), json.dumps({"answer": ""})])
    turn = TutorChat(llm).ask("Is drift related to size?", mode="chat")
    _check("two empty replies still say so, without a third answer call",
           "empty answer" in turn.content and llm.calls == 3, f"content={turn.content!r} calls={llm.calls}")


def run_all() -> int:
    print("=" * 60)
    print("Tutor card cache tests")
    print("=" * 60)
    test_correction_is_a_status_not_reasoning()
    test_critic_feedback_is_not_reasoning()
    test_failed_card_is_not_cached()
    test_good_card_is_cached()
    test_model_switch_gets_its_own_card()
    test_ui_card_cache()
    test_latex_in_a_reply_still_parses()
    test_one_bad_reply_is_retried()
    test_empty_chat_answer()

    print("\n" + "=" * 60)
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    print("=" * 60)
    for name, detail in _FAILED:
        print(f"  ✗ {name}: {detail}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(run_all())
