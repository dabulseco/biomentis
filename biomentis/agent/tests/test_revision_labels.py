"""Each revised answer is titled by what sent it back.

The metagenomics Run 3 bug: a revision caused by the independent reviewer was
titled "revised after self-critique #2".

Run with:
    python -m biomentis.agent.tests.test_revision_labels
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from biomentis.run_quality import CRITIC_FEEDBACK_PREFIX, GATE_FEEDBACK_PREFIX, REVIEW_FEEDBACK_PREFIX
from biomentis.ui_core import revision_source, revision_title, stream_agent_events

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


def _sol(text: str) -> AIMessage:
    return AIMessage(content=f"<solution>{text}</solution>")


def _scripted_agent(script: list):
    history: list = []

    def _stream(inputs, stream_mode=None, config=None):
        history.extend(inputs["messages"])
        for msg in script:
            history.append(msg)
            yield {"messages": list(history)}

    return SimpleNamespace(app=SimpleNamespace(stream=_stream), use_tool_retriever=False, path="/tmp", user_task="")


def _titles(script: list) -> list[str]:
    agent = _scripted_agent(script)
    events = stream_agent_events(agent, "Explain it.", [], [HumanMessage(content="hi")], thread_id=1)
    return [e.title for e in events if e.type == "solution"]


def test_titles_follow_the_trigger() -> None:
    script = [
        _sol("draft one"),
        HumanMessage(content=CRITIC_FEEDBACK_PREFIX + " Check the numbers."),
        _sol("draft two"),
        HumanMessage(content=REVIEW_FEEDBACK_PREFIX + " and found problems."),
        _sol("draft three"),
        HumanMessage(content=GATE_FEEDBACK_PREFIX + ": arithmetic."),
        _sol("draft four"),
    ]
    titles = _titles(script)
    _check("first title is plain", titles[:1] == ["✅ Answer"], f"{titles}")
    _check(
        "critic revision",
        titles[1:2] == ["✅ Answer (round 1: revised after self-critique)"],
        f"{titles}",
    )
    _check(
        "reviewer revision",
        titles[2:3] == ["✅ Answer (round 2: revised after independent review)"],
        f"{titles}",
    )
    _check(
        "gate revision",
        titles[3:4] == ["✅ Answer (round 3: revised after failed checks)"],
        f"{titles}",
    )


def test_helpers() -> None:
    _check("first", revision_title(0, "first") == "✅ Answer")
    _check("source reviewer", revision_source([HumanMessage(content=REVIEW_FEEDBACK_PREFIX + "x")]) == "reviewer")
    _check("source none", revision_source([HumanMessage(content="hi")]) == "first")


def run_all() -> int:
    print("Revision label tests")
    test_titles_follow_the_trigger()
    test_helpers()
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(run_all())
