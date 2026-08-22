"""Tests for how `stream_agent_events` surfaces the agent's answer.

Regression origin (run of 2026-08-19): the main panel showed nothing but a
plan checklist labelled "✅ Answer", while 27 steps of real research sat in
the executor log and the actual final answer was never displayed.

Two things combined:

  1. `streamlit_app.py` runs the agent with `self_critic=True`, so the first
     `<solution>` does NOT end the run — `routing_function` routes it to the
     `self_critic` node, which injects "this is not enough to solve the task"
     and sends control back to `generate`. An agent turn can therefore
     produce several `<solution>` blocks, the last being the real answer.
  2. `stream_agent_events` latched on the FIRST `<solution>` (`... and not
     solution_found`) and ignored every later one, and gated its end-of-run
     fallback on the same flag. Once the first solution was accepted,
     nothing else could ever reach the main panel.

The trigger was the model wrapping its opening plan in `<solution>` — a
compliant reading of "every response must include either <execute> or
<solution>", since a plan is neither code nor an answer.

Run with:
    python -m biomentis.agent.tests.test_solution_streaming
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from biomentis.ui_core import stream_agent_events


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


# --- Fake agent -----------------------------------------------------------


def _fake_agent(message_contents: list[str]):
    """An agent whose graph emits one state per scripted message.

    Mirrors what `agent.app.stream(..., stream_mode="values")` yields: each
    item is the whole state, and `stream_agent_events` reads
    `state["messages"][-1]`.
    """
    history: list = []

    def _stream(inputs, stream_mode=None, config=None):
        history.extend(inputs["messages"])
        for content in message_contents:
            history.append(AIMessage(content=content))
            yield {"messages": list(history)}

    return SimpleNamespace(
        app=SimpleNamespace(stream=_stream),
        use_tool_retriever=False,
        path="/tmp",
        user_task="",
    )


def _run(message_contents: list[str]) -> list:
    agent = _fake_agent(message_contents)
    return list(
        stream_agent_events(
            agent,
            "Explain personalized mRNA cancer vaccines.",
            [],
            [HumanMessage(content="hi")],
            thread_id=1,
        )
    )


def _main_texts(events) -> list[str]:
    """Content of everything that reaches the main (answer) panel."""
    return [e.content for e in events if getattr(e, "channel", None) == "main"]


# --- Tests ----------------------------------------------------------------

_PLAN = "Plan:\n1. [ ] Search literature\n2. [ ] Synthesize the table"
_ANSWER = "Personalized mRNA vaccines are made by sequencing the tumor... <full table>"


def test_revised_solution_reaches_the_main_panel() -> None:
    """The exact shape of the 2026-08-19 run."""
    print("\n[1] A solution revised after self-critique reaches the main panel")
    events = _run(
        [
            f"I'll research this systematically.\n<solution>{_PLAN}</solution>",
            "<execute>query_pubmed('mRNA cancer vaccine')</execute>",
            "Now I can synthesize.\n<solution>" + _ANSWER + "</solution>",
        ]
    )
    main = _main_texts(events)

    _check(
        "the final answer reaches the main panel",
        any(_ANSWER in m for m in main),
        f"main panel got: {[m[:60] for m in main]}",
    )
    _check(
        "the final answer is the LAST thing shown",
        bool(main) and _ANSWER in main[-1],
        f"last main entry: {main[-1][:80] if main else '(none)'}",
    )
    _check(
        "the plan is still shown, not silently swallowed",
        any(_PLAN in m for m in main),
        f"main panel got: {[m[:60] for m in main]}",
    )
    titles = [e.title for e in events if getattr(e, "channel", None) == "main"]
    _check(
        "the revision is labelled as one, so two answers aren't confusing",
        any("revised after self-critique" in (t or "") for t in titles),
        f"titles: {titles}",
    )


def test_single_solution_is_unchanged() -> None:
    """The common (non-self-critic) path must behave exactly as before."""
    print("\n[2] A single-solution run is unchanged")
    events = _run(
        [
            "<execute>print('work')</execute>",
            f"Done.\n<solution>{_ANSWER}</solution>",
        ]
    )
    main = _main_texts(events)
    _check("exactly one main-panel entry", len(main) == 1, f"got {len(main)}")
    _check("it is the answer", main and _ANSWER in main[0], f"got {main[:1]}")
    titles = [e.title for e in events if getattr(e, "channel", None) == "main"]
    _check("titled '✅ Answer', not a revision", titles == ["✅ Answer"], f"{titles}")


def test_identical_solution_is_not_repeated() -> None:
    print("\n[3] The same solution repeated across states is shown once")
    events = _run(
        [
            f"<solution>{_ANSWER}</solution>",
            f"<solution>{_ANSWER}</solution>",
        ]
    )
    main = _main_texts(events)
    _check("shown once, not twice", len(main) == 1, f"got {len(main)}: {[m[:40] for m in main]}")


def test_run_with_no_solution_still_summarizes() -> None:
    print("\n[4] A run that never answers still surfaces something")
    events = _run(
        [
            "<execute>print('work')</execute>",
            "I could not finish this task.",
        ]
    )
    main = _main_texts(events)
    _check("the main panel is not left empty", len(main) == 1, f"got {len(main)}")
    _check(
        "the trailing text is salvaged",
        main and "could not finish" in main[0],
        f"got {main[:1]}",
    )


def test_solution_only_in_final_state_is_shown() -> None:
    """A closing solution that the loop never got to emit must still land."""
    print("\n[5] A solution seen only in the terminal state is still shown")
    # `stream_agent_events` skips states whose message content is not a str;
    # this drives the end-of-run gate rather than the in-loop branch.
    agent = _fake_agent([])
    final = [HumanMessage(content="q"), AIMessage(content=f"<solution>{_ANSWER}</solution>")]

    def _stream(inputs, stream_mode=None, config=None):
        yield {"messages": final}

    agent.app.stream = _stream
    events = list(stream_agent_events(agent, "task", [], [], thread_id=1))
    main = _main_texts(events)
    _check("the answer is shown exactly once", len(main) == 1, f"got {len(main)}")
    _check("and it is the answer", main and _ANSWER in main[0], f"got {main[:1]}")


def test_prompt_forbids_plans_in_solution_tags() -> None:
    """The upstream half of the fix: don't let a plan BE a solution."""
    print("\n[6] The system prompt rules a plan out of <solution>")
    import os

    src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "a1.py"
    )
    with open(src, encoding="utf-8") as f:
        prompt_src = f.read()
    _check(
        "the prompt states that a plan is not a solution",
        "A PLAN IS NOT A SOLUTION" in prompt_src,
        "the rule is missing from the base prompt",
    )


def run_all() -> int:
    print("=" * 60)
    print("Answer-streaming tests")
    print("=" * 60)
    test_revised_solution_reaches_the_main_panel()
    test_single_solution_is_unchanged()
    test_identical_solution_is_not_repeated()
    test_run_with_no_solution_still_summarizes()
    test_solution_only_in_final_state_is_shown()
    test_prompt_forbids_plans_in_solution_tags()

    print("\n" + "=" * 60)
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    print("=" * 60)
    for name, detail in _FAILED:
        print(f"  ✗ {name}: {detail}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(run_all())
