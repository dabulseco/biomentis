"""Tests for ending a model turn at its first closing tag.

The agent's protocol is "one `<execute>` or one `<solution>` per turn", and a
provider stop sequence is supposed to enforce it. On the default local path it
never did: `ChatOllama` has a field named `stop` and no `stop_sequences`
alias, so the name every other branch uses is accepted with no warning, no
`model_kwargs`, and the value simply disappears. The branch looked identical
to its working neighbours and did nothing.

Two things went wrong as a result, and both are about what the model wrote
*after* `</execute>`:

  * a fabricated `<observation>` entered the message history, and the model
    reasoned over its own fiction as if it were a real result
  * a trailing `<solution>` won the branch test in `generate` — `answer_match`
    is checked before `execute_match` — so the run ended on an answer built
    from output that was never computed

`utils.truncate_after_first_tag` enforces the rule locally, for every
provider, and `llm._check_stop_sequences` makes the next silent drop audible.

What these tests pin down:

  1-7.  Truncation: what it cuts, what it leaves, and that it never removes
        anything the extractors would have used
  8.    The branch flip — the same message ends the run before truncation and
        executes after it
  9.    End to end through a real A1 and a scripted LLM: the code runs, and
        the fabricated answer never becomes the run's answer
  10.   Every provider branch either applies stop sequences or is a declared
        exemption with a reason
  11.   The guard fires on an undeclared drop
  12.   The upstream behavior that caused all this, pinned so an SDK fix or
        regression is noticed

Run with:
    python -m biomentis.agent.tests.test_stop_sequences
"""

from __future__ import annotations

import re
import sys

from biomentis.llm import STOP_SEQUENCE_EXEMPT_CLIENTS, get_llm, stop_sequences_applied
from biomentis.utils import truncate_after_first_tag

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


# The response that motivated all of this: real code, invented output, and an
# answer drawn from the invention.
_HAZARD = (
    "Let me check.\n"
    "<execute>print(compute())</execute>\n"
    "<observation>42</observation>\n"
    "Great, the answer is <solution>42</solution>"
)


# --- 1-7: truncation ------------------------------------------------------


def test_truncation_cuts_the_hazard() -> None:
    print("\n[1] the fabricated tail is cut")
    kept, dropped = truncate_after_first_tag(_HAZARD)
    _check("the execute block survives", kept.endswith("</execute>"), repr(kept))
    _check("the invented observation is gone", "<observation>" not in kept, repr(kept))
    _check("the answer built on it is gone", "<solution>" not in kept, repr(kept))
    _check("the reasoning before the tag survives", kept.startswith("Let me check."), repr(kept))
    _check("what was cut is reported back", "<observation>42</observation>" in dropped, repr(dropped))


def test_a_clean_turn_is_untouched() -> None:
    print("\n[2] a well-formed turn is left alone")
    for message in (
        "Thinking.\n<execute>print(1)</execute>",
        "<solution>done</solution>",
        "<think>a plan</think>\n<execute>print(1)</execute>",
    ):
        kept, dropped = truncate_after_first_tag(message)
        _check(f"unchanged: {message[:28]!r}", kept == message and dropped == "", f"{kept!r} / {dropped!r}")


def test_think_is_not_a_turn_boundary() -> None:
    print("\n[3] </think> does not end a turn")
    kept, _ = truncate_after_first_tag("<think>plan</think>\n<execute>print(1)</execute>\ntrailing")
    _check("the execute block after the thinking survives", "<execute>print(1)</execute>" in kept, repr(kept))
    _check("only the trailing prose is cut", "trailing" not in kept, repr(kept))


def test_no_tags_is_a_passthrough() -> None:
    print("\n[4] a tagless message is a passthrough")
    kept, dropped = truncate_after_first_tag("I am not going to use tags today.")
    _check("returned unchanged", kept == "I am not going to use tags today.", repr(kept))
    _check("nothing dropped", dropped == "", repr(dropped))
    # This matters: `generate` must still reach its parse-error branch, which
    # is what re-prompts the model. Truncation must not disguise the failure.


def test_trailing_whitespace_is_not_a_drop() -> None:
    print("\n[5] trailing whitespace does not count as a drop")
    kept, dropped = truncate_after_first_tag("<execute>print(1)</execute>\n\n  \n")
    _check("nothing reported as dropped", dropped == "", repr(dropped))
    _check("message returned as-is", kept.endswith("\n"), repr(kept))


def test_tags_are_case_insensitive() -> None:
    print("\n[6] closing tags match case-insensitively")
    kept, dropped = truncate_after_first_tag("<EXECUTE>x</EXECUTE> junk")
    _check("uppercase tag still ends the turn", kept == "<EXECUTE>x</EXECUTE>", repr(kept))
    _check("the junk is cut", dropped == "junk", repr(dropped))
    # `generate`'s own matchers use re.IGNORECASE, so this has to agree.


def test_truncation_never_removes_what_the_extractor_would_use() -> None:
    print("\n[7] nothing the extractors read is ever cut")
    # The extractors are non-greedy and only ever read the FIRST block, so
    # truncating at the first closing tag cannot cost anything that was used.
    for message in (
        _HAZARD,
        "<execute>a</execute> then <execute>b</execute>",
        "<execute>code with </execute> inside a string</execute>",
    ):
        kept, _ = truncate_after_first_tag(message)
        before = re.search(r"<execute>(.*?)</execute>", message, re.DOTALL | re.IGNORECASE)
        after = re.search(r"<execute>(.*?)</execute>", kept, re.DOTALL | re.IGNORECASE)
        _check(
            f"same code extracted: {message[:30]!r}",
            (before is None and after is None) or (before and after and before.group(1) == after.group(1)),
            f"{before and before.group(1)!r} vs {after and after.group(1)!r}",
        )


# --- 8: the branch flip ---------------------------------------------------


def test_truncation_flips_the_branch() -> None:
    print("\n[8] the same message ends the run before, and executes after")

    def branch(message: str) -> str:
        """`generate`'s decision: answer_match is tested before execute_match."""
        if re.search(r"<solution>(.*?)</solution>", message, re.DOTALL | re.IGNORECASE):
            return "solution"
        if re.search(r"<execute>(.*?)</execute>", message, re.DOTALL | re.IGNORECASE):
            return "execute"
        return "none"

    _check("without truncation the run ends on the invented answer", branch(_HAZARD) == "solution")
    kept, _ = truncate_after_first_tag(_HAZARD)
    _check("with truncation the code runs instead", branch(kept) == "execute")

    # And the ordering this depends on is still what a1.py does.
    src = open(_A1_PATH, encoding="utf-8").read()
    answer_at = src.index("if answer_match:")
    execute_at = src.index("elif execute_match:")
    truncate_at = src.index("msg, dropped_tail = truncate_after_first_tag(msg)")
    _check("answer_match is still tested first", answer_at < execute_at)
    _check("truncation runs before the branch test", truncate_at < answer_at, f"{truncate_at} vs {answer_at}")


import os  # noqa: E402

_A1_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "a1.py"
)


# --- 9: end to end --------------------------------------------------------


class _ScriptedLLM:
    """Returns canned responses in order. Enough for the generate node, which
    only needs `.invoke(messages) -> AIMessage`."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[list] = []

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.calls.append(messages)
        if not self._responses:
            return AIMessage(content="<solution>ran out of script</solution>")
        return AIMessage(content=self._responses.pop(0))


def test_end_to_end_the_code_runs_and_the_fiction_does_not_win() -> None:
    print("\n[9] end to end: the code runs, the invented answer does not")
    from biomentis.agent.a1 import A1

    agent = A1(path="./data", use_tool_retriever=False, expected_data_lake_files=[], trace=False)
    agent.llm = _ScriptedLLM(
        [
            # An unstopped model's turn: real code, invented result, answer
            # drawn from the invention.
            "Let me compute it.\n"
            "<execute>print('REAL_OUTPUT')</execute>\n"
            "<observation>INVENTED_OUTPUT</observation>\n"
            "So the answer is <solution>INVENTED_ANSWER</solution>",
            # What it says once it has actually seen the real output.
            "<solution>REAL_ANSWER</solution>",
        ]
    )
    agent.configure()

    log, final = agent.go("compute it")
    history = "\n".join(log)

    _check("the code actually executed", "REAL_OUTPUT" in history, history[-600:])
    _check("the run's answer is the real one", "REAL_ANSWER" in final, repr(final))
    _check("the invented answer is not the run's answer", "INVENTED_ANSWER" not in final, repr(final))
    _check(
        "the invented observation never entered the history",
        "INVENTED_OUTPUT" not in history,
        history[-600:],
    )
    _check("the model was asked again after the real output", len(agent.llm.calls) == 2, str(len(agent.llm.calls)))


# --- 10-12: the guard -----------------------------------------------------

# Every source `get_llm` supports, with the credentials each needs to
# construct (not to call) a client.
_PROVIDERS = [
    ("Ollama", "llama3", {}),
    ("OpenAI", "gpt-4o", {"OPENAI_API_KEY": "test"}),
    ("OpenAI", "gpt-5", {"OPENAI_API_KEY": "test"}),
    ("Anthropic", "claude-sonnet-4-5", {"ANTHROPIC_API_KEY": "test"}),
    ("Gemini", "gemini-2.5-pro", {"GEMINI_API_KEY": "test"}),
    ("Groq", "llama-3.3-70b", {"GROQ_API_KEY": "test"}),
    ("Bedrock", "anthropic.claude-v2", {"AWS_REGION": "us-east-1"}),
    ("Custom", "my-model", {}),
]


def test_every_provider_stops_or_declares_why_not() -> None:
    print("\n[10] every provider applies stop sequences or declares why not")
    stops = ["</execute>", "</solution>"]
    skipped: list[str] = []

    for source, model, env in _PROVIDERS:
        previous = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            kwargs = {"base_url": "http://localhost:8000/v1"} if source == "Custom" else {}
            llm = get_llm(model, source=source, stop_sequences=stops, api_key="test", **kwargs)
        except ImportError as exc:
            # Report rather than pass quietly: an uninstalled provider is an
            # untested provider, and saying nothing would read as coverage.
            skipped.append(f"{source}/{model} ({str(exc).split(':')[0]})")
            continue
        except Exception as exc:
            skipped.append(f"{source}/{model} ({type(exc).__name__}: {str(exc)[:60]})")
            continue
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        name = type(llm).__name__
        applied = stop_sequences_applied(llm)
        # Exemption first: the gpt-5 subclass sets `stop` on the client and
        # then strips it from the payload, so "carries them" would overstate
        # what actually reaches the API.
        exempt = name in STOP_SEQUENCE_EXEMPT_CLIENTS
        _check(
            f"{source}/{model} -> {name}: {'declared exempt' if exempt else 'applies' if applied else 'DROPS'}",
            applied or exempt,
            "stop sequences vanished and the class is not in STOP_SEQUENCE_EXEMPT_CLIENTS",
        )

    if skipped:
        print(f"    (not constructible here, so untested: {', '.join(skipped)})")
    _check("every exemption carries a reason", all(STOP_SEQUENCE_EXEMPT_CLIENTS.values()))


def test_the_guard_fires_on_an_undeclared_drop() -> None:
    print("\n[11] an undeclared drop is reported")
    import io
    from contextlib import redirect_stdout

    from biomentis.llm import _check_stop_sequences

    class _DropsThem:
        stop = None

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        _check_stop_sequences(_DropsThem(), ["</execute>"])
    printed = buffer.getvalue()
    _check("it warns", "WARNING" in printed, repr(printed))
    _check("it names the class", "_DropsThem" in printed, repr(printed))
    _check("it points at the registry", "STOP_SEQUENCE_EXEMPT_CLIENTS" in printed, repr(printed))

    # Quiet in the two cases that are not bugs.
    for label, llm, stops in (
        ("nothing was asked for", _DropsThem(), None),
        ("the client carries them", type("_Keeps", (), {"stop": ["</execute>"]})(), ["</execute>"]),
    ):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            _check_stop_sequences(llm, stops)
        _check(f"silent when {label}", buffer.getvalue() == "", repr(buffer.getvalue()))


def test_chatollama_still_swallows_the_wrong_kwarg() -> None:
    print("\n[12] the upstream behavior that caused this is pinned")
    import warnings

    from langchain_ollama import ChatOllama

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        llm = ChatOllama(model="x", temperature=0.2, stop_sequences=["</execute>"])
        warned = [str(w.message) for w in caught]

    options = llm._chat_params([]).get("options", {})
    _check("`stop_sequences=` is accepted without error", llm is not None)
    _check("...and without a warning", not warned, str(warned))
    _check("...and never reaches the request", "stop" not in options, str(options))
    _check("the field it actually reads is `stop`", "stop" in ChatOllama.model_fields)
    _check("there is no `stop_sequences` alias", "stop_sequences" not in ChatOllama.model_fields)

    # If any of these start failing, langchain-ollama changed and the
    # exemption in STOP_SEQUENCE_EXEMPT_CLIENTS is worth revisiting.
    correct = ChatOllama(model="x", temperature=0.2, stop=["</execute>"])
    _check(
        "`stop=` does reach the request",
        correct._chat_params([]).get("options", {}).get("stop") == ["</execute>"],
        str(correct._chat_params([]).get("options")),
    )


def run_all() -> int:
    print("=" * 60)
    print("Turn-boundary (stop sequence) tests")
    print("=" * 60)
    test_truncation_cuts_the_hazard()
    test_a_clean_turn_is_untouched()
    test_think_is_not_a_turn_boundary()
    test_no_tags_is_a_passthrough()
    test_trailing_whitespace_is_not_a_drop()
    test_tags_are_case_insensitive()
    test_truncation_never_removes_what_the_extractor_would_use()
    test_truncation_flips_the_branch()
    test_end_to_end_the_code_runs_and_the_fiction_does_not_win()
    test_every_provider_stops_or_declares_why_not()
    test_the_guard_fires_on_an_undeclared_drop()
    test_chatollama_still_swallows_the_wrong_kwarg()

    print("\n" + "=" * 60)
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    print("=" * 60)
    for name, detail in _FAILED:
        print(f"  ✗ {name}: {detail}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(run_all())
