"""Tests for keeping a run inside the model's context window.

The failure this prevents, verbatim from a real run:

    Run failed: ResponseError: The prompt is too long: 262960,
    model maximum context length: 262144 (status code: 400)

Twenty minutes of work lost, and the prompt that "caused" it was 1,212 tokens
— 0.46% of the window. The window was filled by the run itself: a ~42k-token
system prompt before a word is exchanged (32k of it the descriptions of all
224 tools), plus an observation per execute step, capped at 10,000 characters
and never released. Around 3-4k tokens a step against 262k, with
`recursion_limit` at 500: nothing stopped the loop before the context did.

What these tests pin down:

  1-2.  Estimation, and that a request under budget is left completely alone
  3-5.  Trimming: observations collapse oldest-first, the system prompt, the
        task and the recent exchange are never touched, and the caller's own
        message list is never mutated
  6.    A request is never knowingly sent over the limit, even when a single
        message is larger than the whole budget
  7.    The agent is told to finish while there is still room to answer
  8.    An agent that ignores that is eventually stopped, rather than looping
  9-10. Limit resolution and the refusal to invent one
  11.   The provider's own limit is learned from its error and persisted
  12.   A context-length 400 is not retried
  13.   Both LLM call sites in the graph are guarded, not just `generate`

Run with:
    python -m biomentis.agent.tests.test_context_budget
"""

from __future__ import annotations

import os
import sys
import tempfile

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from biomentis.context_budget import (
    HARD_STOP_AFTER_WRAP_UPS,
    KNOWN_CONTEXT_LIMITS,
    estimate_messages_tokens,
    estimate_tokens,
    fit_messages,
    is_context_length_error,
    learn_limit_from_error,
    resolve_context_limit,
)

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


class _FakeLLM:
    def __init__(self, model: str = "deepseek-v4-pro:cloud"):
        self.model = model


def _run_history(steps: int = 30, observation_chars: int = 10_000) -> list:
    """A run shaped like a real one: system prompt, task, then step/observation
    pairs with observations at the 10,000-character cap the execute node
    applies."""
    messages = [
        SystemMessage(content="SYSTEM " * 6_000),
        HumanMessage(content="design species-specific assays for Hawaiian Gracilaria"),
    ]
    for i in range(steps):
        messages.append(AIMessage(content=f"Reasoning for step {i}. " * 60 + f"<execute>step_{i}()</execute>"))
        messages.append(AIMessage(content=f"<observation>{'x' * observation_chars}</observation>"))
    return messages


# --- 1-2 ------------------------------------------------------------------


def test_estimation_is_sane() -> None:
    print("\n[1] token estimation")
    _check("empty is zero", estimate_tokens("") == 0)
    _check("a word is a token or two", 1 <= estimate_tokens("Gracilaria") <= 6, str(estimate_tokens("Gracilaria")))
    _check(
        "message framing is counted",
        estimate_messages_tokens([HumanMessage(content="hi")]) > estimate_tokens("hi"),
    )
    big = estimate_messages_tokens(_run_history(30))
    _check("a 30-step run is large enough to matter", big > 50_000, f"{big:,}")


def test_a_request_under_budget_is_untouched() -> None:
    print("\n[2] a request that fits is left alone")
    messages = _run_history(2)
    fit = fit_messages(messages, limit=1_000_000)
    _check("nothing changed", not fit.changed, fit.summary())
    _check("no wrap-up demanded", not fit.must_wrap_up)
    _check("same messages returned", [m.content for m in fit.messages] == [m.content for m in messages])


# --- 3-6: trimming --------------------------------------------------------


def test_observations_collapse_oldest_first() -> None:
    print("\n[3] observations collapse, oldest first")
    messages = _run_history(30)
    fit = fit_messages(messages, limit=60_000, reserve=4_000)
    _check("it fits afterwards", fit.tokens_after <= fit.budget, f"{fit.tokens_after:,} vs {fit.budget:,}")
    _check("observations were collapsed", fit.trimmed_observations > 0, fit.summary())

    kept = [m.content for m in fit.messages if m.content.startswith("<observation>")]
    collapsed = [c for c in kept if "dropped to stay within" in c]
    intact = [c for c in kept if "dropped to stay within" not in c]
    _check("some survive", intact, "every observation was collapsed")
    _check("the collapsed ones say why", collapsed and "context window" in collapsed[0], str(collapsed[:1]))
    _check(
        "the survivors are the most recent",
        fit.messages[-1].content == messages[-1].content,
        "the newest observation was collapsed",
    )


def test_the_system_prompt_and_task_are_never_trimmed() -> None:
    print("\n[4] the system prompt and the task survive")
    messages = _run_history(40)
    fit = fit_messages(messages, limit=30_000, reserve=2_000)
    _check("system prompt intact", fit.messages[0].content == messages[0].content)
    _check(
        "the task is still there",
        any(m.content == messages[1].content for m in fit.messages),
        "the user's task was trimmed away",
    )
    # Without the task the agent has nothing to answer, and without the system
    # prompt it stops emitting the tags the loop parses.


def test_the_callers_history_is_not_mutated() -> None:
    print("\n[5] the run's own history is untouched")
    messages = _run_history(30)
    before = [m.content for m in messages]
    fit = fit_messages(messages, limit=40_000, reserve=4_000)
    _check("caller's list unchanged", [m.content for m in messages] == before)
    _check("a new list was returned", fit.messages is not messages)
    # This is what keeps the transcript and the journal complete: only the
    # outgoing request is trimmed, and the trim is recomputed next call.


def test_an_oversized_single_message_is_still_made_to_fit() -> None:
    print("\n[6] a request is never knowingly sent over the limit")
    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="the task"),
        AIMessage(content="<observation>" + "y" * 400_000 + "</observation>"),
        AIMessage(content="one more"),
    ]
    fit = fit_messages(messages, limit=20_000, reserve=2_000)
    _check("it fits", fit.tokens_after <= fit.budget, f"{fit.tokens_after:,} vs {fit.budget:,}")
    _check("something was cut", fit.changed, fit.summary())


# --- 7-8: wrapping up -----------------------------------------------------


def test_the_agent_is_asked_to_finish_before_the_ceiling() -> None:
    print("\n[7] the agent is told to finish while it can still answer")
    # Nearly full, but nothing left to collapse: every observation is already
    # small, so the bulk is reasoning the trimmer will not touch.
    messages = [SystemMessage(content="SYS " * 25_000), HumanMessage(content="task")]
    fit = fit_messages(messages, limit=30_000, reserve=2_000)
    _check("wrap-up requested", fit.must_wrap_up, fit.summary())
    _check("and there is still room to answer in", fit.tokens_after < fit.limit, f"{fit.tokens_after:,}")

    roomy = fit_messages(_run_history(2), limit=1_000_000)
    _check("not requested when there is plenty of room", not roomy.must_wrap_up)


def test_an_agent_that_will_not_stop_is_stopped() -> None:
    print("\n[8] an agent that ignores the nudge is eventually stopped")
    messages = [SystemMessage(content="SYS " * 25_000), HumanMessage(content="task")]
    fit = fit_messages(messages, limit=30_000, reserve=2_000)
    _check("no hard stop on the first nudge", not fit.must_hard_stop_after(1))
    _check("no hard stop on the second", not fit.must_hard_stop_after(2))
    _check(
        f"hard stop once nudged {HARD_STOP_AFTER_WRAP_UPS} times",
        fit.must_hard_stop_after(HARD_STOP_AFTER_WRAP_UPS),
    )

    roomy = fit_messages(_run_history(2), limit=1_000_000)
    _check("never a hard stop when there is room", not roomy.must_hard_stop_after(99))


# --- 9-11: limits ---------------------------------------------------------


def test_limit_resolution() -> None:
    print("\n[9] the limit comes from the right place")
    _check(
        "the model that failed is known",
        resolve_context_limit(_FakeLLM("deepseek-v4-pro:cloud")) == 262_144,
        str(resolve_context_limit(_FakeLLM("deepseek-v4-pro:cloud"))),
    )
    previous = os.environ.get("BIOMNI_MAX_CONTEXT_TOKENS")
    try:
        os.environ["BIOMNI_MAX_CONTEXT_TOKENS"] = "12345"
        _check("an explicit override wins", resolve_context_limit(_FakeLLM("deepseek-v4-pro:cloud")) == 12345)
        os.environ["BIOMNI_MAX_CONTEXT_TOKENS"] = "not-a-number"
        _check("a junk override is ignored, not fatal", resolve_context_limit(_FakeLLM("gpt-4o")) == 128_000)
    finally:
        if previous is None:
            os.environ.pop("BIOMNI_MAX_CONTEXT_TOKENS", None)
        else:
            os.environ["BIOMNI_MAX_CONTEXT_TOKENS"] = previous
    _check("every known limit is a real number", all(v > 0 for v in KNOWN_CONTEXT_LIMITS.values()))


def test_an_unknown_model_gets_no_guard() -> None:
    print("\n[10] an unknown model gets no guard rather than a guess")
    _check("no limit invented", resolve_context_limit(_FakeLLM("some-model-nobody-has-heard-of")) is None)
    fit = fit_messages(_run_history(30), limit=None)
    _check("nothing is trimmed without a limit", not fit.changed, fit.summary())
    _check("and no wrap-up is demanded", not fit.must_wrap_up)
    # A guessed ceiling that is too low cuts a run short for no reason; one
    # that is too high protects nothing. Silence is the honest default.


def test_the_limit_is_learned_from_the_error() -> None:
    print("\n[11] the provider's own limit is learned and persisted")
    import biomentis.context_budget as cb

    previous_dir = os.environ.get("BIOMENTIS_RUN_DIR")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BIOMENTIS_RUN_DIR"] = tmp
        cb._learned_limits.clear()
        try:
            llm = _FakeLLM("mystery-model-v1")
            _check("unknown to begin with", resolve_context_limit(llm) is None)

            error = RuntimeError(
                "The prompt is too long: 262960, model maximum context length: 262144 (status code: 400)"
            )
            learned = learn_limit_from_error(llm, error)
            _check("the number in the error is extracted", learned == 262_144, str(learned))
            _check("and used from then on", resolve_context_limit(llm) == 262_144)

            # A fresh process must not have to relearn it.
            cb._learned_limits.clear()
            _check("it survives a restart", resolve_context_limit(_FakeLLM("mystery-model-v1")) == 262_144)

            _check("an unrelated error teaches nothing", learn_limit_from_error(llm, RuntimeError("boom")) is None)
        finally:
            cb._learned_limits.clear()
            if previous_dir is None:
                os.environ.pop("BIOMENTIS_RUN_DIR", None)
            else:
                os.environ["BIOMENTIS_RUN_DIR"] = previous_dir


# --- 12-13: wiring --------------------------------------------------------


def test_a_context_error_is_not_retried() -> None:
    print("\n[12] a context-length 400 is not retried")
    from biomentis.agent.a1 import _invoke_llm_with_retry

    _check("recognised", is_context_length_error(RuntimeError("The prompt is too long: 262960")))
    _check("and the generic form too", is_context_length_error(RuntimeError("context_length_exceeded")))
    _check("a real transient error is not", not is_context_length_error(RuntimeError("TLS handshake timeout")))

    class _Boom:
        model = "deepseek-v4-pro:cloud"

        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            raise RuntimeError("The prompt is too long: 262960, model maximum context length: 262144")

    llm = _Boom()
    try:
        _invoke_llm_with_retry(llm, [HumanMessage(content="hi")], max_retries=3, base_delay=0.01)
        _check("it raises", False, "no exception")
    except RuntimeError:
        _check("it raises", True)
    _check("exactly one attempt, not three", llm.calls == 1, f"{llm.calls} attempts")

    # For contrast: a transient error still gets its retries.
    class _Flaky(_Boom):
        def invoke(self, messages):
            self.calls += 1
            raise RuntimeError("TLS handshake timeout")

    flaky = _Flaky()
    try:
        _invoke_llm_with_retry(flaky, [HumanMessage(content="hi")], max_retries=3, base_delay=0.01)
    except RuntimeError:
        pass
    _check("transient errors are still retried", flaky.calls == 3, f"{flaky.calls} attempts")


def test_both_llm_call_sites_are_guarded() -> None:
    print("\n[13] every LLM call in the graph goes through the guard")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "a1.py")
    src = open(path, encoding="utf-8").read()

    invocations = src.count("_invoke_llm_with_retry(")
    # One definition, one call in `generate`, one in `execute_self_critic`.
    _check("still two call sites", invocations == 3, f"{invocations} occurrences")
    _check("generate fits its request", "messages, context_fit = self._fit_request(messages)" in src)
    _check("the self-critic fits its request too", "critic_messages, _ = self._fit_request(" in src)
    # The critic is the largest request the graph makes — it re-sends the whole
    # history — so leaving it unguarded would move the crash rather than fix it.
    _check(
        "the hard stop skips the critic round",
        "self.critic_count = 10**9" in src,
        "a forced end would still fire a full-history critic call",
    )



def test_end_to_end_a_run_that_would_have_died() -> None:
    print("\n[14] end to end: the run that would have returned a 400")
    from langchain_core.messages import AIMessage

    from biomentis.agent.a1 import A1
    from biomentis.context_budget import WRAP_UP_INSTRUCTION

    class _Stubborn:
        """Keeps calling for more code and ignores every request to finish —
        the worst case for a guard that only asks nicely."""

        model = "deepseek-v4-pro:cloud"

        def __init__(self):
            self.calls = 0
            self.nudges = 0

        def invoke(self, messages):
            self.calls += 1
            if any(WRAP_UP_INSTRUCTION in getattr(m, "content", "") for m in messages):
                self.nudges += 1
            return AIMessage(content=f"Step {self.calls}.\n<execute>print('D' * 30000)</execute>")

    previous = os.environ.get("BIOMNI_MAX_CONTEXT_TOKENS")
    os.environ["BIOMNI_MAX_CONTEXT_TOKENS"] = "60000"
    try:
        agent = A1(path="./data", use_tool_retriever=False, expected_data_lake_files=[], trace=False)
        agent.llm = _Stubborn()
        agent.configure()
        log, final = agent.go("a very long research task")
    finally:
        if previous is None:
            os.environ.pop("BIOMNI_MAX_CONTEXT_TOKENS", None)
        else:
            os.environ["BIOMNI_MAX_CONTEXT_TOKENS"] = previous

    history = "\n".join(log)
    _check("the agent was asked to finish", agent.llm.nudges > 0, f"{agent.llm.nudges} nudges")
    _check(
        "the run ended instead of crashing",
        "Context exhausted" in history or "Context exhausted" in final,
        history[-400:],
    )
    _check(
        "and ended quickly, not at the recursion limit",
        agent.llm.calls < 20,
        f"{agent.llm.calls} LLM calls (recursion_limit is 500)",
    )
    # Before the guard this run made one doomed 262k-token request, retried it
    # three times, and took the whole run down with it.



def run_all() -> int:
    print("=" * 60)
    print("Context budget tests")
    print("=" * 60)
    test_estimation_is_sane()
    test_a_request_under_budget_is_untouched()
    test_observations_collapse_oldest_first()
    test_the_system_prompt_and_task_are_never_trimmed()
    test_the_callers_history_is_not_mutated()
    test_an_oversized_single_message_is_still_made_to_fit()
    test_the_agent_is_asked_to_finish_before_the_ceiling()
    test_an_agent_that_will_not_stop_is_stopped()
    test_limit_resolution()
    test_an_unknown_model_gets_no_guard()
    test_the_limit_is_learned_from_the_error()
    test_a_context_error_is_not_retried()
    test_both_llm_call_sites_are_guarded()
    test_end_to_end_a_run_that_would_have_died()

    print("\n" + "=" * 60)
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    print("=" * 60)
    for name, detail in _FAILED:
        print(f"  ✗ {name}: {detail}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(run_all())
