"""Tests for the class plot setting ("show" plots vs "students run the code").

What these tests pin down:

  1.  BIOMENTIS_PLOT_MODE is stripped and lowercased; anything unknown is "show"
  2.  The run_start journal record carries plot_mode
  3.  A journal written before this feature restores unchanged
  4.  current_plot_mode() follows the session value and falls back to the default
  5.  The sidebar control logs its first value and every change, once each
  6.  A1 starts with plot_mode "show" and current_run_id None

Run with:
    python -m biomentis.agent.tests.test_plot_mode
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

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


def _setenv(name: str, value) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


LOGGED: list[dict] = []


class _FakeLogger:
    def log(self, record: dict) -> None:
        LOGGED.append(dict(record))


def _harness():
    import streamlit as st

    from biomentis.ui_visuals import current_plot_mode, render_plot_mode_control

    class _L:
        def log(self, record):
            st.session_state.setdefault("_test_logged", []).append(dict(record))

    render_plot_mode_control(logger=_L())
    st.session_state["_seen_mode"] = current_plot_mode()


def test_env_default():
    from biomentis.plots import plot_mode_default

    saved = os.environ.get("BIOMENTIS_PLOT_MODE")
    try:
        for raw, want in (("student", "student"), (" SHOW ", "show"), ("bogus", "show"), (None, "show")):
            _setenv("BIOMENTIS_PLOT_MODE", raw)
            _check(f"{raw!r} -> {want}", plot_mode_default() == want)
    finally:
        _setenv("BIOMENTIS_PLOT_MODE", saved)


def test_constants():
    from biomentis.plots import PLOT_MODE_LABELS, PLOT_MODES

    _check("modes", PLOT_MODES == ("show", "student"))
    _check("labels", PLOT_MODE_LABELS == {"show": "Show plots", "student": "Students run the code"})


def test_run_start_records_mode():
    from biomentis.run_journal import RunJournal, load_run

    with tempfile.TemporaryDirectory() as tmp:
        j = RunJournal("p", run_dir=tmp, plot_mode="student")
        _check("run_start has plot_mode", load_run(j.path)["meta"]["plot_mode"] == "student")


def test_old_journal_restores_unchanged():
    from biomentis.run_journal import entries_for_run, load_run

    with tempfile.TemporaryDirectory() as tmp:
        old = Path(tmp) / "20260101-000000-abc123.jsonl"
        img = str(Path(tmp) / "plot.png")
        records = [
            {"type": "run_start", "prompt": "plot it", "run_id": "r", "ts": "t", "mode": "research"},
            {
                "type": "event",
                "seq": 1,
                "event": {"type": "file", "title": "Image Preview", "file_path": img, "file_kind": "image"},
            },
            {"type": "run_end", "status": "complete", "events": 1},
        ]
        old.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        run = load_run(old)
        entries = entries_for_run(run)
        _check("restores", bool(entries) and entries[-1].get("kind") == "image", str(entries))
        _check("no plot_mode key needed", "plot_mode" not in run["meta"])


def test_current_plot_mode():
    from streamlit.testing.v1 import AppTest

    saved = os.environ.get("BIOMENTIS_PLOT_MODE")

    def app():
        import streamlit as st

        from biomentis.ui_visuals import current_plot_mode

        st.session_state["got"] = current_plot_mode()

    try:
        _setenv("BIOMENTIS_PLOT_MODE", "student")
        at = AppTest.from_function(app).run()
        _check("falls back to env default", at.session_state["got"] == "student")
        at.session_state["biomentis_plot_mode"] = "show"
        at.run()
        _check("session value wins", at.session_state["got"] == "show")
        at.session_state["biomentis_plot_mode"] = "junk"
        at.run()
        _check("invalid session value ignored", at.session_state["got"] == "student")
    finally:
        _setenv("BIOMENTIS_PLOT_MODE", saved)


def test_control_logs_changes():
    from streamlit.testing.v1 import AppTest

    saved = os.environ.get("BIOMENTIS_PLOT_MODE")
    _setenv("BIOMENTIS_PLOT_MODE", None)
    try:
        at = AppTest.from_function(_harness).run()
        _check("defaults to show", at.radio[0].value == "show")
        at.run()  # a plain rerun must not log again
        at.radio[0].set_value("student").run()
        at.run()
        modes = [r["plot_mode"] for r in at.session_state["_test_logged"]]
        _check("logged initial and change", modes == ["show", "student"], str(modes))
        _check("record kind", all(r["kind"] == "plot_mode" for r in at.session_state["_test_logged"]))
        _check("current_plot_mode follows radio", at.session_state["_seen_mode"] == "student")
    finally:
        _setenv("BIOMENTIS_PLOT_MODE", saved)


def test_agent_defaults():
    import inspect

    from biomentis.agent import a1

    src = inspect.getsource(a1.A1.__init__)
    _check("init sets plot_mode", 'self.plot_mode = "show"' in src)
    _check("init sets current_run_id", "self.current_run_id = None" in src)
    launch = inspect.getsource(a1.A1.launch_streamlit_demo)
    _check("launch snapshots mode", "self.plot_mode = current_plot_mode()" in launch)
    _check("launch passes meta", "plot_mode=self.plot_mode" in launch)
    _check("launch sets run id", "self.current_run_id = _journal.run_id" in launch)


def run_all() -> int:
    print("=" * 60)
    print("Plot mode tests")
    print("=" * 60)
    test_constants()
    test_env_default()
    test_run_start_records_mode()
    test_old_journal_restores_unchanged()
    test_current_plot_mode()
    test_control_logs_changes()
    test_agent_defaults()

    print("\n" + "=" * 60)
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    print("=" * 60)
    for name, detail in _FAILED:
        print(f"  ✗ {name}: {detail}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(run_all())
