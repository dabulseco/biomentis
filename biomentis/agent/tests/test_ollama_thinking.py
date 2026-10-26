"""Tests for how the Ollama client sets thinking (`think`) per model.

Regression origin (session of 2026-10-25): every tutor-mode run failed its
first agent turn, on both models the user had, in opposite ways. The client
sent no `think` setting, so Ollama applied its own default (thinking on for
gemma4), and:

  * `gemma4:31b-cloud` returned an EMPTY reply — 1.9-3.6k tokens generated,
    `done_reason: stop`, zero characters of content. 3/3 empty with `think`
    unset and 3/3 empty with `think=false`; 3/3 tagged turns with
    `think=true`, where the thinking arrives separately and the content is
    the turn itself.
  * `gemma4:26b-mlx` degenerated on the agent's 46k-token prompt into
    "volume-volume-volume-..." until the 8192-token cap, with `think` unset
    or on. 3/3 clean, tagged turns with `think=false`.

Either way the reply had no `<execute>`/`<solution>`, the parse-error
correction fired, and the run stalled on step 1. Step 1 had worked on the
cloud model from August through 2026-10-21.

`think=true` cannot simply be sent everywhere: Ollama refuses it with a 400
for a model without the `thinking` capability (`qwen2.5-coder:1.5b`).
`think=false` is accepted by every model.

What these tests pin down:

  1.  A local model gets `think=false`, with no capability lookup needed
  2.  A cloud model that can think gets `think=true`
  3.  A cloud model that cannot think, or whose capabilities can't be read,
      gets no `think` setting at all — never a request Ollama would refuse
  4.  BIOMENTIS_OLLAMA_THINK overrides the rule in both directions
  5.  The setting reaches the request Ollama is sent

Run with:
    python -m biomentis.agent.tests.test_ollama_thinking
"""

from __future__ import annotations

import os
import sys

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


# --- Fake capability lookup ------------------------------------------------


class _Capabilities:
    """Stands in for `llm._ollama_can_think` and records what it was asked."""

    def __init__(self, thinking: dict[str, bool | Exception]):
        self.thinking = thinking
        self.asked: list[str] = []

    def __call__(self, model: str, base_url: str | None) -> bool:
        self.asked.append(model)
        answer = self.thinking.get(model, False)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _build(model: str, caps: _Capabilities):
    from biomentis import llm as llm_mod

    original = llm_mod._ollama_can_think
    llm_mod._ollama_can_think = caps
    try:
        return llm_mod.get_llm(model, source="Ollama")
    finally:
        llm_mod._ollama_can_think = original


# --- Tests ----------------------------------------------------------------


def test_local_model_thinks_off() -> None:
    print("\n[1] A local model gets think=false")
    caps = _Capabilities({"gemma4:26b-mlx": True})
    llm = _build("gemma4:26b-mlx", caps)
    _check("reasoning is False", llm.reasoning is False, repr(llm.reasoning))
    _check("no capability lookup was needed", caps.asked == [], f"asked: {caps.asked}")


def test_cloud_model_that_can_think_thinks_on() -> None:
    print("\n[2] A cloud model that can think gets think=true")
    for model in ("gemma4:31b-cloud", "deepseek-v4.1-flash:cloud"):
        llm = _build(model, _Capabilities({model: True}))
        _check(f"{model}: reasoning is True", llm.reasoning is True, repr(llm.reasoning))


def test_cloud_model_that_cannot_think_is_left_alone() -> None:
    print("\n[3] A cloud model without thinking, or unreadable, gets no setting")
    llm = _build("tiny:cloud", _Capabilities({"tiny:cloud": False}))
    _check("no thinking capability -> reasoning unset", llm.reasoning is None, repr(llm.reasoning))

    llm = _build("gemma4:31b-cloud", _Capabilities({"gemma4:31b-cloud": ConnectionError("daemon down")}))
    _check("lookup failed -> reasoning unset", llm.reasoning is None, repr(llm.reasoning))


def test_env_override() -> None:
    print("\n[4] BIOMENTIS_OLLAMA_THINK overrides the rule")
    caps = _Capabilities({"gemma4:31b-cloud": True, "gemma4:26b-mlx": True})
    try:
        os.environ["BIOMENTIS_OLLAMA_THINK"] = "true"
        local = _build("gemma4:26b-mlx", caps)
        os.environ["BIOMENTIS_OLLAMA_THINK"] = "false"
        cloud = _build("gemma4:31b-cloud", caps)
        os.environ["BIOMENTIS_OLLAMA_THINK"] = "auto"
        auto = _build("gemma4:31b-cloud", caps)
    finally:
        os.environ.pop("BIOMENTIS_OLLAMA_THINK", None)
    _check("true turns thinking on for a local model", local.reasoning is True, repr(local.reasoning))
    _check("false turns thinking off for a cloud model", cloud.reasoning is False, repr(cloud.reasoning))
    _check("auto uses the rule", auto.reasoning is True, repr(auto.reasoning))


def test_setting_reaches_the_request() -> None:
    print("\n[5] The setting reaches the request")
    local = _build("gemma4:26b-mlx", _Capabilities({}))
    cloud = _build("gemma4:31b-cloud", _Capabilities({"gemma4:31b-cloud": True}))
    _check("local request carries think=false", local._chat_params([]).get("think") is False,
           repr(local._chat_params([]).get("think")))
    _check("cloud request carries think=true", cloud._chat_params([]).get("think") is True,
           repr(cloud._chat_params([]).get("think")))


def run_all() -> int:
    print("=" * 60)
    print("Ollama thinking tests")
    print("=" * 60)
    test_local_model_thinks_off()
    test_cloud_model_that_can_think_thinks_on()
    test_cloud_model_that_cannot_think_is_left_alone()
    test_env_override()
    test_setting_reaches_the_request()

    print("\n" + "=" * 60)
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    print("=" * 60)
    for name, detail in _FAILED:
        print(f"  ✗ {name}: {detail}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(run_all())
