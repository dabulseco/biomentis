"""Tests for a run that never ends because the model never ends.

Two failures stacked up into a UI that "just spins" with an empty Executor tab
(seen with gemma4:26b-mlx on 2026-10-21):

  * The parse-error guard in `generate` was meant to give up after two
    corrections. It counted the corrections by looking for "There are no tags"
    in `AIMessage`s — but the correction is a `HumanMessage`, so the count was
    always 0 and a model that kept answering without tags was re-asked until
    the recursion limit (500).
  * `ChatOllama` carries no stop sequences (see STOP_SEQUENCE_EXEMPT_CLIENTS)
    and had no output cap, and the generate node waited for the whole reply.
    One call ran 8m14s and produced ~41k tokens, all of which
    `truncate_after_first_tag` would have thrown away, while the UI showed
    nothing.

What these tests pin down:

  1.  Repeated parse errors end the run after two corrections
  2.  ...counting only the current streak, not earlier turns in the history
  3.  On Ollama, generate stops reading the reply at its first closing tag
      and closes the stream, so the daemon stops generating
  4.  What it keeps is exactly what truncation would have kept
  5.  A reply with no closing tag is read to the end
  6.  The Ollama client carries an output cap, overridable by env var
  7.  turn_from_reasoning finds the block a model wrote in its thinking
  8.  A reply with empty content and the turn in its thinking runs that turn,
      with no "no tag" retry (deepseek-v4.1-flash:cloud, 2026-10-26)
  9.  coerce_invoke_syntax reads a turn written as a tool call
 10.  A tool-call-style turn runs as code, and the results the model made up
      after it are dropped (claude-sonnet-5-5, 2026-10-26)
 11.  On Claude, observations go as user turns, so no request ends on an
      assistant turn (prefill is a 400 on current Claude models); other
      models' requests are unchanged
 12.  The correction doesn't ask for the model's "thinking process" (Claude
      refused that as reasoning extraction), and a refusal ends the run with
      a message instead of being asked again

Run with:
    python -m biomentis.agent.tests.test_runaway_generation
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator

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


class _NoTagsLLM:
    """Never uses tags. Raises past `limit` calls so an unbounded retry loop
    fails the test instead of running to the recursion limit."""

    def __init__(self, limit: int = 12, then: list[str] | None = None):
        self.calls = 0
        self.limit = limit
        self._then = list(then or [])

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.calls += 1
        if self.calls > self.limit:
            raise RuntimeError(f"still retrying after {self.limit} calls")
        if self._then and self.calls > 1:
            return AIMessage(content=self._then.pop(0))
        return AIMessage(content="Here is my plan, with no tags at all.")


def _agent(llm):
    from biomentis.agent.a1 import A1

    agent = A1(path="./data", use_tool_retriever=False, expected_data_lake_files=[], trace=False)
    agent.llm = llm
    agent.configure()
    return agent


# --- 1-2: the parse-error guard -------------------------------------------


def test_repeated_parse_errors_end_the_run() -> None:
    print("\n[1] two corrections, then the run ends")
    llm = _NoTagsLLM()
    agent = _agent(llm)
    try:
        _, final = agent.go("do something")
    except Exception as e:  # the runaway shows up as the limit tripping
        final = f"raised {e!r}"

    _check("the model is asked three times, not until the limit", llm.calls == 3, str(llm.calls))
    _check("the run says why it stopped", "repeated parsing errors" in final, repr(final[-300:]))


def test_earlier_turns_do_not_count_toward_the_streak() -> None:
    print("\n[2] parse errors from earlier turns don't cut this one short")
    from langchain_core.messages import AIMessage, HumanMessage

    from biomentis.agent.a1 import PARSE_ERROR_NUDGE

    # One fresh parse error, then a valid answer.
    llm = _NoTagsLLM(then=["<solution>RECOVERED</solution>"])
    agent = _agent(llm)

    # A previous turn that hit two parse errors before it recovered.
    history = [
        HumanMessage(content="earlier task"),
        AIMessage(content="no tags"),
        HumanMessage(content=PARSE_ERROR_NUDGE),
        AIMessage(content="still no tags"),
        HumanMessage(content=PARSE_ERROR_NUDGE),
        AIMessage(content="<solution>earlier answer</solution>"),
        HumanMessage(content="this task"),
    ]
    result = agent.app.invoke(
        {"messages": history, "next_step": None},
        {"recursion_limit": 50, "configurable": {"thread_id": 7}},
    )
    final = result["messages"][-1].content

    _check("this turn's error got a retry", llm.calls == 2, str(llm.calls))
    _check("and the retry's answer is the result", "RECOVERED" in final, repr(final))


# --- 3-6: Ollama replies end at the first closing tag ---------------------


def _streaming_ollama(pieces: list[str], endless_tail: str | None = None):
    """A real `ChatOllama` whose transport is replaced by canned chunks.

    Overriding `_stream` keeps LangChain's own `.stream()` plumbing in the
    path. With `endless_tail`, the reply never ends on its own — the shape of
    the 41k-token runaway.
    """
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk
    from langchain_ollama import ChatOllama

    class _Canned(ChatOllama):
        yielded: int = 0
        closed: bool = False

        def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]:
            try:
                for piece in pieces:
                    self.yielded += 1
                    yield ChatGenerationChunk(message=AIMessageChunk(content=piece))
                while endless_tail is not None:
                    self.yielded += 1
                    if self.yielded > 5000:
                        raise RuntimeError("read the endless reply to the end")
                    yield ChatGenerationChunk(message=AIMessageChunk(content=endless_tail))
            finally:
                self.closed = True

        def invoke(self, *args, **kwargs):  # the bug being fixed: waiting for the end
            from langchain_core.messages import AIMessage

            text = "".join(c.message.content for c in self._stream([]))
            return AIMessage(content=text)

    return _Canned(model="canned", temperature=0.0)


def test_ollama_reply_stops_at_the_first_closing_tag() -> None:
    print("\n[3] an endless Ollama reply is cut at its first closing tag")
    from biomentis.agent.a1 import _invoke_llm_with_retry

    llm = _streaming_ollama(
        ["Plan.\n<exe", "cute>print(1)</exe", "cute>\n"],
        endless_tail="<observation>made up</observation> ",
    )
    try:
        response = _invoke_llm_with_retry(llm, [], stop_at_first_tag=True)
        text = response.content
    except Exception as e:
        text = f"raised {e!r}"

    _check("it returns instead of reading forever", not text.startswith("raised"), text[:200])
    _check("the stream is closed, so the daemon stops", llm.closed, f"yielded={llm.yielded}")
    _check("it read only a few chunks past the tag", llm.yielded <= 5, f"yielded={llm.yielded}")
    _check("the tag split across chunks was still found", "</execute>" in text, repr(text[:200]))


def test_what_is_kept_matches_truncation() -> None:
    print("\n[4] the early stop keeps what truncation would have kept")
    from biomentis.agent.a1 import _invoke_llm_with_retry
    from biomentis.utils import truncate_after_first_tag

    pieces = ["<think>maybe</think>", "\n<solution>42", "</solution>", " and then more", " and more"]
    llm = _streaming_ollama(pieces)
    response = _invoke_llm_with_retry(llm, [], stop_at_first_tag=True)
    expected, _ = truncate_after_first_tag("".join(pieces))
    kept, _ = truncate_after_first_tag(response.content)
    _check("same turn as before the change", kept == expected, f"{kept!r} vs {expected!r}")


def test_reply_without_a_tag_is_read_to_the_end() -> None:
    print("\n[5] a reply with no closing tag is read in full")
    from biomentis.agent.a1 import _invoke_llm_with_retry

    pieces = ["Just ", "prose, ", "no tags."]
    llm = _streaming_ollama(pieces)
    response = _invoke_llm_with_retry(llm, [], stop_at_first_tag=True)
    _check("whole reply returned", response.content == "Just prose, no tags.", repr(response.content))


def test_ollama_client_has_an_output_cap() -> None:
    print("\n[6] the Ollama client carries an output cap")
    from biomentis.llm import OLLAMA_NUM_PREDICT_DEFAULT, get_llm

    llm = get_llm("llama3", source="Ollama")
    _check("a default cap is set", llm.num_predict == OLLAMA_NUM_PREDICT_DEFAULT, str(llm.num_predict))
    _check(
        "...and it reaches the request",
        llm._chat_params([]).get("options", {}).get("num_predict") == OLLAMA_NUM_PREDICT_DEFAULT,
        str(llm._chat_params([]).get("options")),
    )

    os.environ["BIOMENTIS_OLLAMA_NUM_PREDICT"] = "1234"
    try:
        overridden = get_llm("llama3", source="Ollama")
    finally:
        del os.environ["BIOMENTIS_OLLAMA_NUM_PREDICT"]
    _check("the env var overrides it", overridden.num_predict == 1234, str(overridden.num_predict))


def test_turn_from_reasoning() -> None:
    print("\n[7] the turn is found in the thinking")
    from biomentis.utils import turn_from_reasoning

    code = "<execute>print(1)</execute>"
    _check("block after thinking", turn_from_reasoning("Let me compute.\n" + code) == code)
    _check("a plan that mentions <execute> is skipped",
           turn_from_reasoning("I'll write an <execute> block.\n" + code) == code)
    _check("solution", turn_from_reasoning("Done.\n<solution>ZnTe</solution>") == "<solution>ZnTe</solution>")
    _check("only up to the first closing tag",
           turn_from_reasoning(code + " more <solution>x</solution>") == code)
    _check("no closing tag, nothing", turn_from_reasoning("thinking <execute>print(1)") == "")
    _check("mismatched tags, nothing", turn_from_reasoning("<execute>x</solution>") == "")
    _check("empty", turn_from_reasoning(None) == "" and turn_from_reasoning("") == "")


class _ThinkingLLM:
    """Content empty, the turn in reasoning_content; then a normal answer."""

    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="", additional_kwargs={
                "reasoning_content": "Plan: compute it.\n<execute>print(6 * 7)</execute>"})
        return AIMessage(content="<solution>42</solution>")


def test_turn_in_thinking_runs() -> None:
    print("\n[8] a turn written in the thinking is run, not retried")
    from biomentis.agent.a1 import PARSE_ERROR_NUDGE

    llm = _ThinkingLLM()
    agent = _agent(llm)
    log, final = agent.go("multiply")
    _check("the code ran", any("<observation>42" in m for m in log), str(log[-3:])[:300])
    _check("no retry nudge", not any(PARSE_ERROR_NUDGE[:40] in m for m in log), "")
    _check("two calls, not three", llm.calls == 2, str(llm.calls))
    _check("answered", "<solution>42</solution>" in final, final)


_INVOKE_TURN = (
    "I'll compute it.\n\n<invoke name=\"execute\">\n<parameter name=\"code\">\nprint(6 * 7)\n</parameter>\n</invoke>\n"
    ":<observe>\n41\n</parameter>\n</invoke>\n:So the answer is 41."
)


def test_coerce_invoke_syntax() -> None:
    print("\n[9] a turn written as a tool call")
    from biomentis.utils import AGENT_STOP_SEQUENCES, coerce_invoke_syntax

    got = coerce_invoke_syntax(_INVOKE_TURN)
    _check("becomes an execute block", got == "I'll compute it.\n<execute>\nprint(6 * 7)\n</execute>", repr(got))
    _check("made-up results dropped", "41" not in got and "observe" not in got, repr(got))
    cut = '<invoke name="execute">\n<parameter name="code">\nx = 1\n</parameter>\n'  # stopped at </invoke>
    _check("works when stopped at </invoke>", coerce_invoke_syntax(cut) == "<execute>\nx = 1\n</execute>",
           repr(coerce_invoke_syntax(cut)))
    answer = '<invoke name="solution"><parameter name="text">ZnTe</parameter></invoke>'
    _check("answer-like name becomes a solution", coerce_invoke_syntax(answer) == "<solution>\nZnTe\n</solution>")
    tagged = "<execute>print(1)</execute> <invoke name=\"execute\"><parameter name=\"code\">2</parameter>"
    _check("a turn with tags is left alone", coerce_invoke_syntax(tagged) == tagged)
    _check("plain prose is left alone", coerce_invoke_syntax("No tags here.") == "No tags here.")
    _check("agent stops at </invoke>", "</invoke>" in AGENT_STOP_SEQUENCES)


class _InvokeLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.calls += 1
        return AIMessage(content=_INVOKE_TURN if self.calls == 1 else "<solution>42</solution>")


def test_invoke_turn_runs() -> None:
    print("\n[10] a tool-call-style turn runs as code")
    llm = _InvokeLLM()
    log, final = _agent(llm).go("multiply")
    _check("the code ran", any("<observation>42" in m for m in log), str(log[-3:])[:300])
    _check("its made-up result is not in the history", not any(":So the answer is 41" in m for m in log), "")
    _check("two calls", llm.calls == 2, str(llm.calls))
    _check("answered", "<solution>42</solution>" in final, final)


class _RecordingLLM:
    """Runs one step, then answers; records every request."""

    def __init__(self, model: str):
        self.model = model
        self.requests: list = []

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.requests.append(list(messages))
        if len(self.requests) == 1:
            return AIMessage(content="<execute>print(42)</execute>")
        return AIMessage(content="<solution>42</solution>")


def test_claude_requests_end_on_a_user_turn() -> None:
    print("\n[11] Claude requests never end on an assistant turn")
    from langchain_core.messages import AIMessage, HumanMessage

    llm = _RecordingLLM("claude-sonnet-5-5")
    _, final = _agent(llm).go("print 42")
    second = llm.requests[1] if len(llm.requests) > 1 else []
    _check("ran to its answer", "<solution>42</solution>" in final, final)
    _check("the request after a step ends on a user turn", bool(second) and isinstance(second[-1], HumanMessage),
           type(second[-1]).__name__ if second else "no second request")
    _check("that turn is the observation", bool(second) and second[-1].content.startswith("<observation>42"),
           second[-1].content[:60] if second else "")

    other = _RecordingLLM("gemma4:31b-cloud")
    _agent(other).go("print 42")
    last = other.requests[1][-1] if len(other.requests) > 1 else None
    _check("other models unchanged", isinstance(last, AIMessage), type(last).__name__)


class _RefusingLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="<execute>print(1)</execute>")
        return AIMessage(content="", response_metadata={"stop_reason": "refusal"})


def test_refusal_ends_the_run() -> None:
    print("\n[12] the correction wording, and refusals")
    from biomentis.agent.a1 import MODEL_REFUSED_MESSAGE, PARSE_ERROR_NUDGE

    _check("the correction doesn't ask for reasoning", "thinking" not in PARSE_ERROR_NUDGE.lower(), PARSE_ERROR_NUDGE)
    _check("it names both tags", "<execute>" in PARSE_ERROR_NUDGE and "<solution>" in PARSE_ERROR_NUDGE)
    llm = _RefusingLLM()
    log, final = _agent(llm).go("anything")
    _check("a refusal is not asked again", llm.calls == 2, str(llm.calls))
    _check("the run says why it stopped", final == MODEL_REFUSED_MESSAGE, final)
    _check("no correction was sent", not any(PARSE_ERROR_NUDGE in m for m in log), "")


def run_all() -> int:
    print("=" * 60)
    print("Runaway generation tests")
    print("=" * 60)
    for test in (
        test_repeated_parse_errors_end_the_run,
        test_earlier_turns_do_not_count_toward_the_streak,
        test_ollama_reply_stops_at_the_first_closing_tag,
        test_what_is_kept_matches_truncation,
        test_reply_without_a_tag_is_read_to_the_end,
        test_ollama_client_has_an_output_cap,
        test_turn_from_reasoning,
        test_turn_in_thinking_runs,
        test_coerce_invoke_syntax,
        test_invoke_turn_runs,
        test_claude_requests_end_on_a_user_turn,
        test_refusal_ends_the_run,
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
