"""Tests for the Code Review button and report panel (`ui_code_review`).

Each test drives a small harness app with Streamlit's `AppTest` and a fake
reviewer, so nothing touches the network.

Run with:
    python -m biomentis.agent.tests.test_code_review_ui
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

_HARNESS = '''
import sys
sys.path.insert(0, {repo!r})
import streamlit as st
from biomentis.agent.tests.test_code_review_ui import FACTORY, CODE
from biomentis.code_review import CodeReviewRequest
from biomentis.ui_code_review import render_code_review

st.session_state.setdefault("biomni_run_active", True)
st.session_state.setdefault("tutor_paused", "card-3")
st.write("page body")
render_code_review(
    CodeReviewRequest(code=CODE, language="python", block_index=2),
    key="blk",
    llm_factory=FACTORY[0],
    model="fake-model",
)
'''

CODE = "x = 1\nprint(x)\n"
FACTORY: list = [None]
CALLS: list[str] = []


def _reply(**over) -> str:
    d = {
        "verdict": "run_with_fixes",
        "issues": [{"line": 1, "kind": "bug", "severity": "high", "quote": "x = 1", "problem": "wrong value", "fix": "x = 2"}],
        "corrected_code": "x = 2\nprint(x)",
        "summary": "One value was wrong.",
    }
    d.update(over)
    return json.dumps(d)


class _FakeLLM:
    def __init__(self, reply: str | None, gate: threading.Event | None = None):
        self.reply, self.gate = reply, gate

    def invoke(self, messages, *a, **k):
        CALLS.append("invoke")
        if self.gate is not None:
            self.gate.wait(10)
        if self.reply is None:
            raise RuntimeError("boom")
        from langchain_core.messages import AIMessage

        return AIMessage(content=self.reply)


def _factory(reply: str | None, gate: threading.Event | None = None):
    def make(model: str):
        return _FakeLLM(reply, gate)

    return make


def _start(reply: str | None, gate: threading.Event | None = None):
    from streamlit.testing.v1 import AppTest

    from biomentis import code_review_jobs as jobs

    jobs._clear_for_tests()
    CALLS.clear()
    FACTORY[0] = _factory(reply, gate)
    path = os.path.join(tempfile.mkdtemp(), "harness.py")
    with open(path, "w") as f:
        f.write(_HARNESS.format(repo=_REPO))
    return AppTest.from_file(path, default_timeout=60).run()


def _key():
    from biomentis.code_review import CodeReviewRequest, review_key

    return review_key(CodeReviewRequest(code=CODE, language="python", block_index=2), "fake-model")


def _texts(at) -> str:
    return "\n".join(str(e.value) for e in list(at.markdown) + list(at.caption) + list(at.warning) + list(at.error))


def _with_isolated_dirs():
    os.environ["BIOMENTIS_RUN_DIR"] = tempfile.mkdtemp()


def test_button_submits() -> None:
    from biomentis import code_review_jobs as jobs

    gate = threading.Event()
    at = _start(_reply(), gate)
    _check("a button is shown first", [b.label for b in at.button] == ["🔎 Code Review"], str([b.label for b in at.button]))
    at.button[0].click().run()
    status = jobs.review_status(_key())[0]
    _check("click queues one review", status == "pending", status)
    _check("panel shows Reviewing…", "Reviewing… (fake-model)" in _texts(at), _texts(at))
    _check("button is gone while pending", len(at.button) == 0)
    gate.set()
    jobs._wait_for(_key())
    _check("only one model call", CALLS.count("invoke") == 1, str(CALLS))


def test_pending_review_appears_after_rerun() -> None:
    from biomentis import code_review_jobs as jobs

    at = _start(_reply())
    at.button[0].click().run()
    jobs._wait_for(_key())
    at.run()
    labels = [e.label for e in at.expander]
    _check("expander has the verdict label", labels == ["Code review — ⚠️ Run with fixes"], str(labels))
    _check("corrected code shown", any("x = 2" in c.value for c in at.code), str([c.value for c in at.code]))
    _check("summary shown", "One value was wrong." in _texts(at))
    _check("issue problem shown", "wrong value" in _texts(at))


def test_empty_corrected_code_message() -> None:
    from biomentis import code_review_jobs as jobs

    at = _start(_reply(corrected_code=""))
    at.button[0].click().run()
    jobs._wait_for(_key())
    at.run()
    _check("no-script message", "No corrected script could be produced." in _texts(at), _texts(at))


def test_syntax_warning_shown() -> None:
    from biomentis import code_review_jobs as jobs

    at = _start(_reply(corrected_code="def f(:\n  pass"))
    at.button[0].click().run()
    jobs._wait_for(_key())
    at.run()
    # Streamlit lifts the leading emoji of a warning into its icon.
    warns = [w.value for w in at.warning]
    _check(
        "syntax warning shown",
        len(warns) == 1 and "This corrected script has a syntax error:" in warns[0],
        str(warns),
    )
    _check("script still shown", any("def f(" in c.value for c in at.code))


def test_error_panel_offers_retry() -> None:
    from biomentis import code_review_jobs as jobs

    at = _start(None)
    at.button[0].click().run()
    jobs._wait_for(_key())
    at.run()
    _check("error text shown", len(at.error) == 1, str([e.value for e in at.error]))
    _check("Try again offered", [b.label for b in at.button] == ["Try again"], str([b.label for b in at.button]))
    FACTORY[0] = _factory(_reply())
    at.button[0].click().run()
    status = jobs.review_status(_key())[0]
    _check("Try again resubmits", status in ("pending", "done"), status)
    jobs._wait_for(_key())
    at.run()
    _check("report replaces the error", len(at.expander) == 1 and len(at.error) == 0)


def test_click_does_not_touch_run_state() -> None:
    from biomentis import code_review_jobs as jobs

    gate = threading.Event()
    at = _start(_reply(), gate)
    before = {k: at.session_state[k] for k in ("biomni_run_active", "tutor_paused")}
    keys_before = set(at.session_state.filtered_state)
    at.button[0].click().run()
    after = {k: at.session_state[k] for k in ("biomni_run_active", "tutor_paused")}
    new_keys = set(at.session_state.filtered_state) - keys_before
    _check("run and tutor state unchanged", before == after, f"{before} vs {after}")
    _check("no new session keys beyond the widget", all(k.startswith("blk_") for k in new_keys), str(new_keys))
    gate.set()
    jobs._wait_for(_key())


def run_all() -> int:
    _with_isolated_dirs()
    print("=" * 60)
    print("Code Review UI tests")
    print("=" * 60)
    test_button_submits()
    test_pending_review_appears_after_rerun()
    test_empty_corrected_code_message()
    test_syntax_warning_shown()
    test_error_panel_offers_retry()
    test_click_does_not_touch_run_state()
    print("\n" + "=" * 60)
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    print("=" * 60)
    for name, detail in _FAILED:
        print(f"  ✗ {name}: {detail}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    # The harness app imports this file by its package name; run that copy so
    # both share one FACTORY.
    from biomentis.agent.tests import test_code_review_ui as _self

    sys.exit(_self.run_all())
