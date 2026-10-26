"""Tests for Anthropic API key resolution and the degraded web-search path.

The symptom that prompted this:

    [advanced_web_search_claude] Anthropic rejected the request (Error code:
    401 - {'type': 'error', 'error': {'type': 'authentication_error',
    'message': 'invalid x-api-key'}, 'request_id': 'req_011Ceg...'})
    — using keyless web search.

Two separate defects hide behind that one line.

*Resolution.* The key reached three different readers by three different
routes: `.env` (via `load_dotenv(override=False)` in `a1.py`), an exported
shell variable, and a `source ~/.bash_profile` subprocess in `llm.py`. Because
`override=False` means "an already-exported value wins", a stale shell key
silently shadowed a good `.env` key, and nothing in the app said so. Quoted or
newline-padded `.env` values failed the same way, because the raw string went
into the `x-api-key` header verbatim.

*Reporting.* The whole exception — `request_id` and all — was interpolated
into the string handed back to the LLM, so a key problem became several lines
of API noise inside the agent's reasoning trace.

These tests pin down:

  1. Surrounding double quotes are stripped
  2. Surrounding single quotes and stray whitespace/newlines are stripped
  3. `.env` beats a stale exported shell value — the actual shadowing bug
  4. The shell value is still used when `.env` has no entry
  5. A session-applied key (the sidebar) beats both
  6. Blank input is rejected and does NOT clobber a working key
  7. Input with embedded whitespace is rejected and does NOT clobber
  8. `looks_like_anthropic_key` flags a non-`sk-ant-` value without rejecting it
  9. A 401 is classified to a short reason carrying no request_id
 10. The fallback note the LLM sees carries that short reason, not the raw error
 11. A 401 goes keyless immediately, with no retry backoff
 12. The sidebar field prefills from the resolved key
 13. Typing a key in the sidebar applies it
 14. Typing garbage warns in-line and leaves the previous key alone

Tests 12-14 drive the real `render_anthropic_key_field` inside a Streamlit
`AppTest` harness. Note the AppTest quirk called out in CLAUDE.md: an
externally assigned widget value survives only the run immediately after the
assignment, so each case stages its state and its assertion in one `at.run()`.

Run with:
    python -m biomentis.agent.tests.test_anthropic_key
"""

from __future__ import annotations

import os
import sys
import tempfile
import types

from biomentis.config import (
    apply_anthropic_key,
    default_config,
    looks_like_anthropic_key,
    resolve_anthropic_key,
)

# --- Tiny test harness (matches test_ncbi_email.py) ----------------------

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


_GOOD = "sk-ant-api03-" + "A" * 40
_OTHER = "sk-ant-api03-" + "B" * 40


def _reset() -> None:
    """Clear every tier so each case starts from a known state."""
    default_config.anthropic_key = None
    os.environ.pop("ANTHROPIC_API_KEY", None)


def _write_dotenv(body: str) -> str:
    d = tempfile.mkdtemp(prefix="biomni_anthropic_")
    path = os.path.join(d, ".env")
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    return path


# --- 1-2. Normalization --------------------------------------------------


def test_strips_surrounding_double_quotes() -> None:
    print("\n[1] surrounding double quotes are stripped")
    _reset()
    path = _write_dotenv(f'ANTHROPIC_API_KEY="{_GOOD}"\n')
    got = resolve_anthropic_key(env={}, dotenv_path=path)
    _check("quotes gone", got == _GOOD, f"got {got!r}")


def test_strips_single_quotes_and_whitespace() -> None:
    print("\n[2] single quotes and stray whitespace are stripped")
    _reset()
    path = _write_dotenv(f"ANTHROPIC_API_KEY='{_GOOD}'\n")
    _check("dotenv quotes gone", resolve_anthropic_key(env={}, dotenv_path=path) == _GOOD)

    # The shell tier gets the same treatment — a trailing newline from a
    # `source`d profile is the common shape.
    got = resolve_anthropic_key(env={"ANTHROPIC_API_KEY": f"  {_GOOD}\n"}, dotenv_path="/nonexistent")
    _check("env whitespace gone", got == _GOOD, f"got {got!r}")


# --- 3-5. Precedence -----------------------------------------------------


def test_dotenv_beats_stale_shell_value() -> None:
    print("\n[3] .env beats a stale exported shell value (the shadowing bug)")
    _reset()
    path = _write_dotenv(f"ANTHROPIC_API_KEY={_GOOD}\n")
    got = resolve_anthropic_key(env={"ANTHROPIC_API_KEY": "sk-ant-stale-key"}, dotenv_path=path)
    _check("dotenv wins", got == _GOOD, f"got {got!r}")


def test_shell_used_when_dotenv_has_no_entry() -> None:
    print("\n[4] the shell value is used when .env has no entry")
    _reset()
    path = _write_dotenv("OPENAI_API_KEY=sk-other\n")
    got = resolve_anthropic_key(env={"ANTHROPIC_API_KEY": _GOOD}, dotenv_path=path)
    _check("shell used", got == _GOOD, f"got {got!r}")


def test_session_key_beats_both() -> None:
    print("\n[5] a session-applied key beats .env and the shell")
    _reset()
    path = _write_dotenv(f"ANTHROPIC_API_KEY={_OTHER}\n")
    apply_anthropic_key(_GOOD)
    got = resolve_anthropic_key(env={"ANTHROPIC_API_KEY": "sk-ant-stale-key"}, dotenv_path=path)
    _check("session wins", got == _GOOD, f"got {got!r}")
    _reset()


# --- 6-8. Validation -----------------------------------------------------


def test_blank_is_rejected_without_clobbering() -> None:
    print("\n[6] blank input is rejected and the working key survives")
    _reset()
    apply_anthropic_key(_GOOD)
    _check("blank rejected", apply_anthropic_key("   ") is False)
    _check("prior key intact", default_config.anthropic_key == _GOOD, f"got {default_config.anthropic_key!r}")
    _check("env intact", os.environ.get("ANTHROPIC_API_KEY") == _GOOD)
    _reset()


def test_embedded_whitespace_is_rejected_without_clobbering() -> None:
    print("\n[7] a value with embedded whitespace is rejected and does not clobber")
    _reset()
    apply_anthropic_key(_GOOD)
    _check("garbage rejected", apply_anthropic_key("sk-ant not a key") is False)
    _check("prior key intact", default_config.anthropic_key == _GOOD, f"got {default_config.anthropic_key!r}")
    _reset()


def test_looks_like_flags_without_rejecting() -> None:
    print("\n[8] looks_like_anthropic_key flags a non-sk-ant value but does not reject it")
    _reset()
    _check("sk-ant recognised", looks_like_anthropic_key(_GOOD) is True)
    _check("proxy key flagged", looks_like_anthropic_key("gateway-token-" + "C" * 30) is False)
    # ...but a gateway/proxy token is still *appliable*, because ANTHROPIC_BASE_URL
    # users legitimately have keys that don't carry Anthropic's prefix.
    _check("proxy key still applies", apply_anthropic_key("gateway-token-" + "C" * 30) is True)
    _reset()


# --- 9-11. Degraded-path reporting --------------------------------------

_REAL_401 = (
    "Error code: 401 - {'type': 'error', 'error': {'type': "
    "'authentication_error', 'message': 'invalid x-api-key'}, "
    "'request_id': 'req_011CegKLRCxXLS5ZHeckm3Ze'}"
)


def test_401_classified_to_short_reason() -> None:
    print("\n[9] a 401 becomes a short reason with no request_id")
    from biomentis.tool.literature import _classify_anthropic_failure

    reason = _classify_anthropic_failure(Exception(_REAL_401))
    _check("mentions the key", "key" in reason.lower(), f"got {reason!r}")
    _check("no request_id", "req_011" not in reason, f"got {reason!r}")
    _check("stays short", len(reason) < 90, f"len={len(reason)}: {reason!r}")


def test_fallback_note_carries_short_reason_only() -> None:
    print("\n[10] the note the LLM sees carries the short reason, not the raw error")
    import biomentis.tool.literature as lit

    original = lit.keyless_web_search
    lit.keyless_web_search = lambda q, num_results=5, language="en": "Title: x\nURL: http://x"
    try:
        note = lit._advanced_web_search_fallback("q", reason=lit._classify_anthropic_failure(Exception(_REAL_401)))
    finally:
        lit.keyless_web_search = original

    _check("no request_id leaked", "req_011" not in note, note[:200])
    _check("no raw dict leaked", "'type': 'error'" not in note, note[:200])
    _check("still names the tool", "advanced_web_search_claude" in note)
    _check("still carries results", "Title: x" in note)


def test_401_goes_keyless_without_backoff() -> None:
    print("\n[11] a 401 goes keyless immediately, with no retry backoff")
    import biomentis.tool.literature as lit

    class _FakeAuthError(Exception):
        status_code = 401

    class _FakeMessages:
        def create(self, **_kw):
            raise _FakeAuthError(_REAL_401)

    class _FakeClient:
        def __init__(self, **_kw):
            self.messages = _FakeMessages()

    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = _FakeClient

    slept: list[float] = []
    prev_mod = sys.modules.get("anthropic")
    prev_sleep = lit.time.sleep
    prev_search = lit.keyless_web_search
    sys.modules["anthropic"] = fake_anthropic
    lit.time.sleep = lambda s: slept.append(s)
    lit.keyless_web_search = lambda q, num_results=5, language="en": "Title: fallback\nURL: http://f"

    os.environ["ANTHROPIC_API_KEY"] = _GOOD
    os.environ.pop("BIOMENTIS_WEB_SEARCH", None)
    try:
        out = lit.advanced_web_search_claude("some query")
    finally:
        lit.time.sleep = prev_sleep
        lit.keyless_web_search = prev_search
        if prev_mod is not None:
            sys.modules["anthropic"] = prev_mod
        else:
            sys.modules.pop("anthropic", None)
        _reset()

    _check("no backoff slept", slept == [], f"slept {slept}")
    _check("fell back to keyless", "Title: fallback" in out, out[:200])
    _check("no request_id leaked", "req_011" not in out, out[:200])


# --- 12-14. Sidebar field ------------------------------------------------

_HARNESS = '''
import sys
sys.path.insert(0, {repo!r})
import streamlit as st
from biomentis.config import apply_anthropic_key, default_config
from biomentis.ui_settings import ANTHROPIC_KEY_KEY, render_anthropic_key_field

_seed = st.session_state.get("seed_config_key", None)
if _seed is not None:
    default_config.anthropic_key = _seed

with st.sidebar:
    render_anthropic_key_field(verify=False)

st.write("CONFIG=" + repr(default_config.anthropic_key))
st.write("WIDGET=" + repr(st.session_state.get(ANTHROPIC_KEY_KEY)))
'''

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _app_test():
    from streamlit.testing.v1 import AppTest

    path = os.path.join(tempfile.mkdtemp(prefix="biomni_akey_"), "app.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_HARNESS.format(repo=_REPO))
    return AppTest.from_file(path, default_timeout=60)


def _line(at, prefix: str) -> str:
    for m in at.markdown:
        if m.value.startswith(prefix):
            return m.value
    return ""


def test_sidebar_prefills_from_resolved_key() -> None:
    print("\n[12] the sidebar field prefills from the resolved key")
    _reset()
    at = _app_test()
    at.session_state["seed_config_key"] = _GOOD
    at.run()

    _check("no exception", not at.exception, str(at.exception))
    inputs = list(at.sidebar.text_input)
    _check("field rendered", len(inputs) == 1, f"got {len(inputs)} text inputs")
    if inputs:
        _check("prefilled", inputs[0].value == _GOOD, f"got {inputs[0].value!r}")
    _reset()


def test_sidebar_applies_a_typed_key() -> None:
    print("\n[13] typing a key in the sidebar applies it")
    _reset()
    at = _app_test()
    at.run()
    at.sidebar.text_input[0].set_value(_OTHER).run()

    _check("no exception", not at.exception, str(at.exception))
    _check("config updated", _line(at, "CONFIG=") == f"CONFIG={_OTHER!r}", f"got {_line(at, 'CONFIG=')!r}")
    _check("env updated", os.environ.get("ANTHROPIC_API_KEY") == _OTHER)
    _reset()


def test_sidebar_flags_garbage_and_keeps_prior() -> None:
    print("\n[14] garbage is flagged in-line and the prior key survives")
    _reset()
    at = _app_test()
    at.session_state["seed_config_key"] = _GOOD
    at.run()
    at.sidebar.text_input[0].set_value("not a key at all").run()

    _check("no exception", not at.exception, str(at.exception))
    _check("prior key survives", _line(at, "CONFIG=") == f"CONFIG={_GOOD!r}", f"got {_line(at, 'CONFIG=')!r}")
    captions = [c.value for c in at.sidebar.caption]
    _check("warned in-line", any("keeping" in c.lower() for c in captions), f"captions={captions}")
    _reset()


def main() -> int:
    for fn in (
        test_strips_surrounding_double_quotes,
        test_strips_single_quotes_and_whitespace,
        test_dotenv_beats_stale_shell_value,
        test_shell_used_when_dotenv_has_no_entry,
        test_session_key_beats_both,
        test_blank_is_rejected_without_clobbering,
        test_embedded_whitespace_is_rejected_without_clobbering,
        test_looks_like_flags_without_rejecting,
        test_401_classified_to_short_reason,
        test_fallback_note_carries_short_reason_only,
        test_401_goes_keyless_without_backoff,
        test_sidebar_prefills_from_resolved_key,
        test_sidebar_applies_a_typed_key,
        test_sidebar_flags_garbage_and_keeps_prior,
    ):
        fn()

    print(f"\n{len(_PASSED)} passed, {len(_FAILED)} failed")
    for name, detail in _FAILED:
        print(f"  FAIL {name}: {detail}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
