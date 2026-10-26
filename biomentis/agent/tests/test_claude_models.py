"""Tests for running the agent on current Claude models.

Before these changes the model picker offered retired Claude ids, every
current Claude model rejected the request (the agent always sends
`temperature`, a 400 on these models), and a thinking model's reply arrived
as a list of blocks that the agent and tutor read as a string.

What these tests pin down:

  1.  claude_rejects_sampling: which Claude ids reject `temperature`
  2.  The request sent to a current model carries no temperature; an older
      model keeps it
  3.  A reply with a thinking block reaches callers as plain text; a reply
      that calls a tool keeps its blocks
  4.  The picker lists current ids with a real key, and no Claude models
      when the key is the .env.example placeholder
  5.  The pinned defaults (SQL helper, web search) are current models

Run with:
    python -m biomentis.agent.tests.test_claude_models
"""

from __future__ import annotations

import os
import sys
import tempfile

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


_FAKE_KEY = "sk-ant-api03-" + "T" * 40


def _with_key(key: str | None):
    """Set the only key tiers that apply outside the repo's own .env."""
    from biomentis.config import default_config

    default_config.anthropic_key = None
    if key is None:
        os.environ.pop("ANTHROPIC_API_KEY", None)
    else:
        os.environ["ANTHROPIC_API_KEY"] = key


def test_rejects_sampling() -> None:
    print("\n[1] which models reject temperature")
    from biomentis.llm import claude_rejects_sampling

    for model in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-1",
                  "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "anthropic.claude-opus-5-5",
                  "us.anthropic.claude-sonnet-5-5"):
        _check(f"{model} rejects", claude_rejects_sampling(model))
    for model in ("claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5", "claude-sonnet-4-5",
                  "gemma4:31b-cloud", "gpt-5", "", None):
        _check(f"{model!r} keeps it", not claude_rejects_sampling(model))


def test_request_payload() -> None:
    print("\n[2] no temperature in requests to current models")
    from langchain_core.messages import HumanMessage

    from biomentis.llm import get_llm

    _with_key(_FAKE_KEY)
    new = get_llm("claude-opus-5-5", temperature=0.2, source="Anthropic", stop_sequences=["</execute>"])
    payload = new._get_request_payload([HumanMessage(content="hi")])
    _check("current model: no temperature", "temperature" not in payload, str(sorted(payload)))
    _check("stop sequences still sent", payload.get("stop_sequences") == ["</execute>"], str(payload.get("stop_sequences")))
    old = get_llm("claude-sonnet-4-6", temperature=0.2, source="Anthropic")
    _check("older model keeps it", old._get_request_payload([HumanMessage(content="hi")]).get("temperature") == 0.2)
    _with_key(None)


def test_reply_is_text() -> None:
    print("\n[3] replies reach callers as text")
    from anthropic.types import Message, TextBlock, ThinkingBlock, ToolUseBlock, Usage

    from biomentis.llm import get_llm

    _with_key(_FAKE_KEY)
    llm = get_llm("claude-opus-5-5", source="Anthropic")

    def reply(content):
        return Message(id="m", type="message", role="assistant", model="claude-opus-5-5", content=content,
                       stop_reason="end_turn", stop_sequence=None, usage=Usage(input_tokens=1, output_tokens=1))

    thought = ThinkingBlock(type="thinking", thinking="", signature="sig")
    text = TextBlock(type="text", text="Plan.\n<execute>print(1)</execute>")
    msg = llm._format_output(reply([thought, text])).generations[0].message
    _check("thinking dropped, text kept", msg.content == "Plan.\n<execute>print(1)</execute>", repr(msg.content))

    tool = ToolUseBlock(type="tool_use", id="t1", name="f", input={})
    msg = llm._format_output(reply([thought, tool])).generations[0].message
    _check("a tool call keeps its blocks", isinstance(msg.content, list) and msg.tool_calls, repr(msg.content))
    _with_key(None)


def test_picker() -> None:
    print("\n[4] the model picker")
    import biomentis.ollama_utils as ou
    from biomentis.ui_core import list_available_providers

    prev_list, prev_cwd = ou.list_ollama_models, os.getcwd()
    ou.list_ollama_models = lambda *a, **k: []
    os.chdir(tempfile.mkdtemp(prefix="claude-models-"))  # no .env here
    try:
        _with_key(_FAKE_KEY)
        models = list_available_providers().get("Anthropic", [])
        _check("current ids listed", models == ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"], str(models))
        _with_key("your_anthropic_api_key_here")
        _check("placeholder lists no Claude models", "Anthropic" not in list_available_providers())
    finally:
        os.chdir(prev_cwd)
        ou.list_ollama_models = prev_list
        _with_key(None)


def test_defaults() -> None:
    print("\n[5] pinned defaults are current models")
    import inspect

    import biomentis.tool.database as db
    import biomentis.tool.literature as lit

    _check("web search default", lit._DEFAULT_WEB_SEARCH_MODEL == "claude-opus-5-5", lit._DEFAULT_WEB_SEARCH_MODEL)
    _check("SQL helper fallback", '"claude-haiku-5-5"' in inspect.getsource(db), "")


def run_all() -> int:
    print("=" * 60)
    print("Claude model tests")
    print("=" * 60)
    for test in (test_rejects_sampling, test_request_payload, test_reply_is_text, test_picker, test_defaults):
        try:
            test()
        except Exception as e:
            import traceback

            traceback.print_exc()
            _check(f"{test.__name__} ran", False, repr(e))
    print("\n" + "=" * 60)
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    print("=" * 60)
    for name, detail in _FAILED:
        print(f"  ✗ {name}: {detail}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(run_all())
