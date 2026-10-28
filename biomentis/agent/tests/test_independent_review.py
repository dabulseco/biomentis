"""Tests for the independent review of an answer by a second model, and temperatures.

The self-critique runs on the agent's own model and kept a 50x scale-up error
through two drafts of a CHO workflow (deepseek-v4.1-flash:cloud, 2026-10-28).
A different model (the Critic picked in the sidebar) now reviews each answer:
the changes the self-critique made, the final answer's numbers, units and
claims against the run's own evidence, and the rationale of each decision.

What these tests pin down:

  1.  The reviewer gets the task, the first draft, the self-critique, the final
      answer and each step's code and output, cut to fit
  2.  Its JSON reply is parsed, fenced or not; a reply without a valid verdict
      is a failed review, not a verdict
  3.  The correction request quotes each problem and asks for numbers to be
      recomputed in code; the review is formatted for the answer panel
  4.  "needs rework" sends the problems back for exactly one correction round,
      then the reviewer checks again and both reviews are kept
  5.  "accept" ends the run; no reviewer means no review; a failed review call
      keeps the answer; a second "needs rework" is reported, not acted on
  6.  With self-critique on, the review comes after the critique rounds; an
      aborted run is not reviewed
  7.  The UI shows the reviews under the answer, before the run-quality line,
      which names the model, temperatures and reviewer
  8.  set_temperatures rebuilds the clients only when a value changes

Run with:
    python -m biomentis.agent.tests.test_independent_review
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


def _review_json(verdict: str, problems: list | None = None) -> str:
    return json.dumps({
        "verdict": verdict,
        "summary": f"Overall: {verdict}.",
        "changes": [{"change": "agitation 32 rpm -> 33 rpm", "status": "invalid", "reason": "P/V uses rev/s"}],
        "problems": problems if problems is not None else [
            {"kind": "unit", "quote": "33 rpm", "issue": "P/V computed with rpm, not rev/s", "fix": "about 124 rpm"}
        ],
        "rationale": [{"decision": "AEX before CEX", "sound": "yes", "note": "standard order"}],
    })


def test_review_inputs() -> None:
    print("\n[1] what the reviewer is given")
    from biomentis.independent_review import build_review_prompt, review_inputs

    msgs = [
        HumanMessage(content="Old question"),
        AIMessage(content="<solution>old</solution>"),
        HumanMessage(content="Design a 1000 L process."),
        AIMessage(content="Compute P/V.\n<execute>print(5 * 1000 * 0.53**3 * 0.258**5)</execute>"),
        AIMessage(content="<observation>0.87</observation>"),
        AIMessage(content="<solution>Draft: 32 rpm</solution>"),
        HumanMessage(content="Wait... this is not enough to solve the task. Here are some feedbacks for improvement:\nCheck rpm."),
        AIMessage(content="<solution>Final: 33 rpm</solution>"),
    ]
    inputs = review_inputs(msgs)
    _check("the task is this run's prompt", inputs["task"] == "Design a 1000 L process.", inputs["task"])
    _check("first draft", inputs["first_draft"] == "Draft: 32 rpm", inputs["first_draft"])
    _check("final answer", inputs["final_answer"] == "Final: 33 rpm", inputs["final_answer"])
    _check("the critique", inputs["critiques"] and "Check rpm." in inputs["critiques"][0], str(inputs["critiques"]))
    _check("the evidence has code and output", "0.258**5" in inputs["evidence"] and "0.87" in inputs["evidence"],
           inputs["evidence"])
    prompt = build_review_prompt(inputs)
    for section in ("TASK:", "EVIDENCE", "FIRST_DRAFT:", "SELF_CRITIQUE", "FINAL_ANSWER:"):
        _check(f"prompt has {section}", section in prompt)

    long = [HumanMessage(content="Q")]
    for i in range(80):
        long += [AIMessage(content=f"<execute>print({i})</execute>"), AIMessage(content="<observation>" + "x" * 10_000 + "</observation>")]
    long.append(AIMessage(content="<solution>A</solution>"))
    _check("a long run's evidence is cut to fit", len(review_inputs(long)["evidence"]) < 140_000,
           str(len(review_inputs(long)["evidence"])))
    same = build_review_prompt(review_inputs([HumanMessage(content="Q"), AIMessage(content="<solution>A</solution>")]))
    _check("an unrevised answer says so", "did not change it" in same, same[-300:])


def test_parse_review() -> None:
    print("\n[2] reading the reviewer's reply")
    from biomentis.independent_review import parse_review

    r = parse_review(_review_json("needs_rework"))
    _check("verdict", r.verdict == "needs_rework" and r.needs_rework, str(r))
    _check("problems", r.problems and r.problems[0]["quote"] == "33 rpm", str(r.problems))
    fenced = parse_review("Here you go:\n```json\n" + _review_json("accept", problems=[]) + "\n```")
    _check("a fenced reply parses", fenced.verdict == "accept" and not fenced.needs_rework, str(fenced))
    _check("'Accept with corrections' is normalized",
           parse_review(json.dumps({"verdict": "Accept with corrections"})).verdict == "accept_with_corrections")
    bad = parse_review(json.dumps({"verdict": "great"}))
    _check("an unknown verdict is an error", bad.error and not bad.needs_rework, str(bad))
    junk = parse_review("I think it is fine.")
    _check("prose is an error", junk.error and not junk.needs_rework, str(junk))


def test_feedback_and_format() -> None:
    print("\n[3] the correction request, and the review as shown")
    from biomentis.independent_review import feedback_for_agent, format_review, parse_review
    from biomentis.run_quality import REVIEW_FEEDBACK_PREFIX

    review = parse_review(_review_json("needs_rework"))
    fb = feedback_for_agent(review, "Ollama: gpt-oss:120b")
    _check("recognizable", fb.startswith(REVIEW_FEEDBACK_PREFIX))
    _check("quotes the problem and the fix", '"33 rpm"' in fb and "about 124 rpm" in fb, fb)
    _check("lists invalid changes", "agitation 32 rpm -> 33 rpm" in fb, fb)
    _check("asks to recompute in code and answer again", "recompute it in code" in fb and "<solution></solution>" in fb, fb)
    title, body = format_review(review, "Ollama: gpt-oss:120b", after_correction=False)
    _check("titled", title == "🔎 Independent review", title)
    _check("verdict and reviewer shown", "Needs rework" in body and "gpt-oss:120b" in body, body)
    _check("changes and rationale shown", "✗ agitation" in body and "✓ AEX before CEX" in body, body)
    title, _ = format_review(review, "x", after_correction=True)
    _check("the second is the re-review", title == "🔎 Re-review after correction", title)
    _, body = format_review(parse_review("nope"), "x", after_correction=False)
    _check("a failed review says why", "No review" in body, body)


class _Scripted:
    def __init__(self, replies: list):
        self.replies = list(replies)
        self.requests: list = []

    def invoke(self, messages):
        self.requests.append(list(messages))
        if not self.replies:
            raise RuntimeError("called more times than scripted")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return AIMessage(content=reply)


def _agent(llm, reviewer=None, self_critic=False, rounds=0, critic=None):
    from biomentis.agent.a1 import A1

    agent = A1(path="./data", use_tool_retriever=False, expected_data_lake_files=[], trace=False)
    agent.llm = llm
    if critic is not None:
        agent._creative_llm = critic
    agent.configure(self_critic=self_critic, test_time_scale_round=rounds)
    agent.review_llm = reviewer
    agent.review_model_name = "gpt-oss:120b" if reviewer else ""
    return agent


def test_correction_round() -> None:
    print("\n[4] needs rework: one correction round, then a re-review")
    from biomentis.run_quality import REVIEW_FEEDBACK_PREFIX

    llm = _Scripted(["<solution>Draft: 33 rpm</solution>", "<execute>print(124)</execute>", "<solution>Fixed: 124 rpm</solution>"])
    reviewer = _Scripted([_review_json("needs_rework"), _review_json("accept", problems=[])])
    agent = _agent(llm, reviewer)
    log, final = agent.go("Design a 1000 L process.")
    _check("reviewed twice", len(reviewer.requests) == 2, str(len(reviewer.requests)))
    _check("the agent got the problems", any(
        isinstance(m, HumanMessage) and m.content.startswith(REVIEW_FEEDBACK_PREFIX) for m in llm.requests[1]),
        "no review feedback in the second request" if len(llm.requests) > 1 else "")
    _check("the corrected answer is final", "<solution>Fixed: 124 rpm</solution>" in final, final)
    _check("both reviews kept", [r["after_correction"] for r in agent.run_reviews] == [False, True],
           str(agent.run_reviews))
    _check("the re-review saw the correction", "Fixed: 124 rpm" in reviewer.requests[1][-1].content)
    _check("its first draft was the original", "Draft: 33 rpm" in reviewer.requests[1][-1].content)


def test_other_outcomes() -> None:
    print("\n[5] accept, no reviewer, a failed call, and a second needs-rework")
    llm = _Scripted(["<solution>Good</solution>"])
    reviewer = _Scripted([_review_json("accept", problems=[])])
    agent = _agent(llm, reviewer)
    _, final = agent.go("Q")
    _check("accept: one review, no correction", len(reviewer.requests) == 1 and len(llm.requests) == 1, "")
    _check("accept: answer kept", "<solution>Good</solution>" in final, final)

    llm = _Scripted(["<solution>Good</solution>"])
    agent = _agent(llm, reviewer=None)
    _, final = agent.go("Q")
    _check("no reviewer: no review", agent.run_reviews == [] and len(llm.requests) == 1, str(agent.run_reviews))

    llm = _Scripted(["<solution>Good</solution>"])
    reviewer = _Scripted([RuntimeError("model not found")])
    agent = _agent(llm, reviewer)
    _, final = agent.go("Q")
    _check("failed call: answer kept", "<solution>Good</solution>" in final, final)
    _check("failed call: reported", agent.run_reviews and "failed" in agent.run_reviews[0]["review"].error,
           str(agent.run_reviews))

    llm = _Scripted(["<solution>A</solution>", "<solution>B</solution>"])
    reviewer = _Scripted([_review_json("needs_rework"), _review_json("needs_rework")])
    agent = _agent(llm, reviewer)
    _, final = agent.go("Q")
    _check("second needs-rework: no third round", len(llm.requests) == 2 and len(reviewer.requests) == 2,
           f"{len(llm.requests)} agent, {len(reviewer.requests)} reviewer calls")
    _check("second needs-rework: reported", agent.run_reviews[-1]["review"].needs_rework, str(agent.run_reviews))
    _check("the corrected answer is still the result", "<solution>B</solution>" in final, final)


def test_with_self_critique() -> None:
    print("\n[6] after the self-critique; not after an aborted run")
    from biomentis.run_quality import CRITIC_FEEDBACK_PREFIX

    llm = _Scripted(["<solution>Draft</solution>", "<solution>Revised</solution>"])
    critic = _Scripted(["Check the units."])
    reviewer = _Scripted([_review_json("accept", problems=[])])
    agent = _agent(llm, reviewer, self_critic=True, rounds=1, critic=critic)
    _, final = agent.go("Q")
    _check("critique, then review", len(critic.requests) == 1 and len(reviewer.requests) == 1,
           f"{len(critic.requests)} critic, {len(reviewer.requests)} reviewer")
    prompt = reviewer.requests[0][-1].content if reviewer.requests else ""
    _check("the reviewer saw draft, critique and revision",
           "Draft" in prompt and "Check the units." in prompt and "Revised" in prompt, prompt[-400:])
    _check("the critique text isn't mistaken for the task", CRITIC_FEEDBACK_PREFIX not in prompt.split("EVIDENCE")[0], "")

    llm = _Scripted(["no tags"] * 3)
    reviewer = _Scripted([])
    agent = _agent(llm, reviewer)
    _, final = agent.go("Q")
    _check("an aborted run is not reviewed", reviewer.requests == [] and agent.run_reviews == [], final)


def _scripted_agent(messages: list, reviews: list):
    history: list = []

    def _stream(inputs, stream_mode=None, config=None):
        history.extend(inputs["messages"])
        for m in messages:
            history.append(m)
            yield {"messages": list(history)}

    agent = SimpleNamespace(
        app=SimpleNamespace(stream=_stream), use_tool_retriever=False, path="/tmp", user_task="",
        llm=SimpleNamespace(model="deepseek-v4.1-flash:cloud"), temperature=0.0, creative_temperature=0.7,
        self_critic=True, review_llm=SimpleNamespace(model="gpt-oss:120b"), review_model_name="gpt-oss:120b",
    )
    # stream_agent_events resets reviews at the start of a run, like A1.reset_review.
    agent.reset_review = lambda: setattr(agent, "run_reviews", list(reviews))
    return agent


def test_ui() -> None:
    print("\n[7] reviews and settings in the UI")
    from biomentis.independent_review import feedback_for_agent, parse_review
    from biomentis.ui_core import stream_agent_events

    first = parse_review(_review_json("needs_rework"))
    second = parse_review(_review_json("accept", problems=[]))
    reviews = [
        {"review": first, "reviewer": "gpt-oss:120b", "after_correction": False},
        {"review": second, "reviewer": "gpt-oss:120b", "after_correction": True},
    ]
    events = list(stream_agent_events(_scripted_agent([
        AIMessage(content="<solution>Draft</solution>"),
        HumanMessage(content=feedback_for_agent(first, "gpt-oss:120b")),
        AIMessage(content="<solution>Fixed</solution>"),
    ], reviews), "Q", [], [], thread_id=1))
    statuses = [e.content for e in events if e.type == "status" and e.channel != "main"]
    _check("the correction round is a status line", any("one correction round" in s for s in statuses), str(statuses))
    main_titles = [e.title for e in events if e.channel == "main"]
    _check("answer, review, re-review, then run quality",
           main_titles[-3:] == ["🔎 Independent review", "🔎 Re-review after correction", "📊 Run quality"], str(main_titles))
    quality = [e.content for e in events if e.title == "📊 Run quality"]
    _check("settings name model, temperatures and reviewer",
           quality and "temperature 0" in quality[0] and "self-critique temperature 0.7" in quality[0]
           and "independent review by gpt-oss:120b" in quality[0], quality[0] if quality else "")
    _check("the correction round is counted", quality and "1 correction round from the independent review" in quality[0],
           quality[0] if quality else "")


def test_set_temperatures() -> None:
    print("\n[8] temperatures")
    from biomentis.agent.a1 import A1

    agent = A1(path="./data", use_tool_retriever=False, expected_data_lake_files=[], trace=False)
    agent.set_model("llama3", source="Ollama")
    before = agent.llm
    _check("unchanged values don't rebuild", agent.set_temperatures(agent.temperature, agent.creative_temperature) is False
           and agent.llm is before)
    _check("a change rebuilds", agent.set_temperatures(0.0, 0.5) is True and agent.llm is not before)
    _check("the agent client is at 0", agent.llm.temperature == 0.0, str(agent.llm.temperature))
    _check("the critique client follows", agent.creative_llm.temperature == 0.5, str(agent.creative_llm.temperature))
    _check("the model is kept", agent.llm.model == "llama3", agent.llm.model)


def run_all() -> int:
    print("=" * 60)
    print("Independent review tests")
    print("=" * 60)
    for test in (
        test_review_inputs,
        test_parse_review,
        test_feedback_and_format,
        test_correction_round,
        test_other_outcomes,
        test_with_self_critique,
        test_ui,
        test_set_temperatures,
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
    return 0 if not _FAILED else 1


if __name__ == "__main__":
    sys.exit(run_all())
