"""Tests for Ollama's hosted models reached with an Ollama API key.

With `OLLAMA_API_KEY` (in .env or the sidebar) the app lists the models on
ollama.com — DeepSeek, Qwen, GLM, ... — as the `OllamaCloud` source and talks
to them there directly, without a local daemon signed in via `ollama signin`.

What these tests pin down:

  1.  The key resolves like the Anthropic key: sidebar, then .env, then the
      shell; the .env.example placeholder counts as not set
  2.  An `OllamaCloud` client points at ollama.com and sends the key as a
      Bearer token; without a key it says how to set one
  3.  Hosted models think (when the API says they can) and get the cloud
      context ceiling, though their names carry no `-cloud` tag
  4.  The hosted model list is cached per key, and a failure is not cached
  5.  The picker lists `OllamaCloud` models only with a key
  6.  verify_ollama_key asks /api/me (listing models works without a key),
      and tells a rejected key from an unreachable API
  7.  The sidebar field prefills, applies a typed key, and names the placeholder

Run with:
    python -m biomentis.agent.tests.test_ollama_cloud
"""

from __future__ import annotations

import os
import sys
import tempfile
from types import SimpleNamespace

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


_KEY = "0123456789abcdef0123456789abcdef.ABCDEFGHIJKLMNOPQRSTUVWX"
_OTHER = "fedcba9876543210fedcba9876543210.XWVUTSRQPONMLKJIHGFEDCBA"


def _reset() -> None:
    from biomentis.config import default_config

    default_config.ollama_key = None
    os.environ.pop("OLLAMA_API_KEY", None)


def _dotenv(body: str) -> str:
    path = os.path.join(tempfile.mkdtemp(prefix="ollama-cloud-"), ".env")
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    return path


class _FakeClient:
    """Stands in for `ollama.Client`; records each construction."""

    made: list[dict] = []
    models = ["qwen3-coder:480b", "deepseek-v3.1:671b"]
    error: Exception | None = None

    def __init__(self, host=None, **kwargs):
        _FakeClient.made.append({"host": host, **kwargs})

    def list(self):
        if _FakeClient.error is not None:
            raise _FakeClient.error
        return SimpleNamespace(models=[SimpleNamespace(model=m) for m in _FakeClient.models])


def _with_fake_client(fn):
    import ollama

    original = ollama.Client
    ollama.Client = _FakeClient
    _FakeClient.made, _FakeClient.error = [], None
    try:
        return fn()
    finally:
        ollama.Client = original


def test_resolution() -> None:
    print("\n[1] key resolution")
    from biomentis.config import apply_ollama_key, default_config, resolve_ollama_key

    _reset()
    path = _dotenv(f"OLLAMA_API_KEY={_KEY}\n")
    _check(".env beats the shell", resolve_ollama_key(env={"OLLAMA_API_KEY": _OTHER}, dotenv_path=path) == _KEY)
    _check("shell used without .env", resolve_ollama_key(env={"OLLAMA_API_KEY": _OTHER}, dotenv_path="/none") == _OTHER)
    placeholder = _dotenv("OLLAMA_API_KEY=your_ollama_api_key_here\n")
    _check("placeholder is not a key", resolve_ollama_key(env={}, dotenv_path=placeholder) is None)
    _check("applied key wins", apply_ollama_key(_OTHER) and resolve_ollama_key(env={}, dotenv_path=path) == _OTHER)
    _check("applied key reaches the ollama package", os.environ.get("OLLAMA_API_KEY") == _OTHER)
    _check("blank is not applied", apply_ollama_key("  ") is False and default_config.ollama_key == _OTHER)
    _reset()


def test_client() -> None:
    print("\n[2-3] the OllamaCloud client")
    from biomentis import llm as llm_mod
    from biomentis.config import apply_ollama_key
    from biomentis.context_budget import OLLAMA_CLOUD_LIMIT, resolve_context_limit

    asked: list = []
    original = llm_mod._ollama_can_think
    llm_mod._ollama_can_think = lambda model, base_url, headers=None: asked.append((model, base_url, headers)) or True
    _reset()
    prev_cwd = os.getcwd()
    os.chdir(tempfile.mkdtemp(prefix="ollama-cloud-"))  # no .env here
    try:
        try:
            llm_mod.get_llm("deepseek-v3.1:671b", source="OllamaCloud")
            _check("no key: error", False, "built a client without a key")
        except ValueError as e:
            _check("no key: says how to set one", "OLLAMA_API_KEY" in str(e), str(e))

        apply_ollama_key(_KEY)
        client = llm_mod.get_llm("deepseek-v3.1:671b", source="OllamaCloud")
        _check("points at ollama.com", client.base_url == "https://ollama.com", client.base_url)
        headers = (client.client_kwargs or {}).get("headers", {})
        _check("sends the key as a Bearer token", headers.get("Authorization") == f"Bearer {_KEY}", str(headers))
        _check("capabilities asked of ollama.com with the key",
               asked == [("deepseek-v3.1:671b", "https://ollama.com", {"Authorization": f"Bearer {_KEY}"})], str(asked))
        _check("thinking on", client.reasoning is True, str(client.reasoning))
        _check("cloud context ceiling", resolve_context_limit(client) == OLLAMA_CLOUD_LIMIT, str(resolve_context_limit(client)))
    finally:
        os.chdir(prev_cwd)
        llm_mod._ollama_can_think = original
        _reset()


def test_listing() -> None:
    print("\n[4-5] hosted model list and the picker")
    import biomentis.ollama_utils as ou
    from biomentis.config import apply_ollama_key
    from biomentis.ui_core import list_available_providers

    ou._cloud_list_cache.clear()

    def run():
        _check("no key, no list", ou.list_ollama_cloud_models(None) == [])
        first = ou.list_ollama_cloud_models(_KEY)
        _check("listed, sorted", first == ["deepseek-v3.1:671b", "qwen3-coder:480b"], str(first))
        made = _FakeClient.made[-1] if _FakeClient.made else {}
        _check("asked ollama.com with the key",
               made.get("host") == "https://ollama.com" and made.get("headers") == {"Authorization": f"Bearer {_KEY}"},
               str(made))
        ou.list_ollama_cloud_models(_KEY)
        _check("second call served from cache", len(_FakeClient.made) == 1, str(len(_FakeClient.made)))

        _FakeClient.error = ConnectionError("offline")
        _check("failure gives []", ou.list_ollama_cloud_models(_OTHER) == [])
        _FakeClient.error = None
        _check("failure not cached", ou.list_ollama_cloud_models(_OTHER) == ["deepseek-v3.1:671b", "qwen3-coder:480b"])

        prev_local, prev_cwd = ou.list_ollama_models, os.getcwd()
        ou.list_ollama_models = lambda *a, **k: ["gemma4:31b-cloud"]
        os.chdir(tempfile.mkdtemp(prefix="ollama-cloud-"))
        try:
            _reset()
            _check("picker: none without a key", "OllamaCloud" not in list_available_providers())
            apply_ollama_key(_KEY)
            providers = list_available_providers()
            _check("picker: listed with a key", providers.get("OllamaCloud") == ["deepseek-v3.1:671b", "qwen3-coder:480b"],
                   str(providers))
            _check("picker: local models kept", providers.get("Ollama") == ["gemma4:31b-cloud"], str(providers))
        finally:
            os.chdir(prev_cwd)
            ou.list_ollama_models = prev_local
            _reset()

    _with_fake_client(run)


def test_verify() -> None:
    print("\n[6] verifying a key")
    import httpx

    from biomentis.config import verify_ollama_key

    calls: list = []
    answer: dict = {}

    def fake_post(url, headers=None, timeout=None):
        calls.append((url, headers))
        if "error" in answer:
            raise answer["error"]
        return SimpleNamespace(status_code=answer["status"])

    original = httpx.post
    httpx.post = fake_post
    try:
        answer.update(status=200)
        ok, message = verify_ollama_key(_KEY)
        _check("accepted", ok and "accepted" in message, message)
        _check("asks /api/me with the key",
               calls[-1] == ("https://ollama.com/api/me", {"Authorization": f"Bearer {_KEY}"}), str(calls[-1]))
        answer.update(status=401)
        ok, message = verify_ollama_key(_KEY)
        _check("401 is a rejection", not ok and "rejected" in message, message)
        answer["error"] = httpx.ConnectError("offline")
        ok, message = verify_ollama_key(_KEY)
        _check("offline is 'couldn't verify'", not ok and "Couldn't verify" in message, message)
        _check("blank", verify_ollama_key("") == (False, "No key set."))
    finally:
        httpx.post = original


_HARNESS = """
import sys
sys.path.insert(0, {repo!r})
import streamlit as st
from biomentis.config import default_config
from biomentis.ui_settings import render_ollama_key_field

_seed = st.session_state.get("seed", None)
if _seed is not None:
    default_config.ollama_key = _seed
with st.sidebar:
    render_ollama_key_field(verify=False)
st.write("CONFIG=" + repr(default_config.ollama_key))
"""


def test_sidebar() -> None:
    print("\n[7] the sidebar field")
    from streamlit.testing.v1 import AppTest

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    path = os.path.join(tempfile.mkdtemp(prefix="ollama-key-"), "app.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_HARNESS.format(repo=repo))

    def line(at):
        return next((m.value for m in at.markdown if m.value.startswith("CONFIG=")), "")

    _reset()
    at = AppTest.from_file(path, default_timeout=60)
    at.session_state["seed"] = _KEY
    at.run()
    _check("no exception", not at.exception, str(at.exception))
    _check("prefilled from the resolved key", at.sidebar.text_input[0].value == _KEY, at.sidebar.text_input[0].value)

    _reset()
    at = AppTest.from_file(path, default_timeout=60)
    at.run()
    at.sidebar.text_input[0].set_value(_OTHER).run()
    _check("typed key applied", line(at) == f"CONFIG={_OTHER!r}", line(at))

    _reset()
    at = AppTest.from_file(path, default_timeout=60)
    at.session_state["seed"] = _KEY
    at.run()
    at.sidebar.text_input[0].set_value("your_ollama_api_key_here").run()
    captions = [c.value for c in at.sidebar.caption]
    _check("placeholder named", any("placeholder" in c for c in captions), str(captions))
    _check("prior key survives", line(at) == f"CONFIG={_KEY!r}", line(at))
    _reset()


def run_all() -> int:
    print("=" * 60)
    print("Ollama Cloud tests")
    print("=" * 60)
    for test in (test_resolution, test_client, test_listing, test_verify, test_sidebar):
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
