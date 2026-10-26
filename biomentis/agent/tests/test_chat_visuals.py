"""Tests for plots and structures in the tutor conversation.

  1.  run_plot_code runs in a subprocess and reports errors, timeouts, no figure
  2.  Plot mode "show" draws the tutor's plot_code; "student" keeps code only
  3.  Bad plot code leaves the answer intact and reports the error
  4.  A `structure` field is resolved against the search dirs; unknown is ignored
  5.  The page panel and step Q&A turn dicts carry the visuals keys

Run with:
    python -m biomentis.agent.tests.test_chat_visuals
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

from langchain_core.messages import AIMessage

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
CODE = "import numpy as np, matplotlib.pyplot as plt\nd=np.linspace(2,10,50)\nplt.plot(d, 450+80/d**2)\nplt.xlabel('diameter (nm)')"
_TMP = tempfile.mkdtemp(prefix="chat-visuals-")
os.environ["BIOMENTIS_ARTIFACT_DIR"] = os.path.join(_TMP, "artifacts")
_CLASSIFY = json.dumps({"bloom_level": "Apply", "dok_level": 2, "rubric_hit": [], "confidence": 0.8})


class _LLM:
    def __init__(self, reply: dict):
        self.model = "fake"
        self.reply = json.dumps(reply)
        self.n = 0

    def invoke(self, messages):
        self.n += 1
        return AIMessage(content=self.reply if self.n == 1 else _CLASSIFY)


def _ask(reply: dict, **kw):
    from biomentis.agent.tutor.chat import TutorChat

    return TutorChat(_LLM(reply)).ask("Show me?", mode="chat", **kw)


def test_run_plot_code() -> None:
    print("\n[1] run_plot_code")
    from biomentis.plots import run_plot_code

    out = os.path.join(_TMP, "plots")
    ok = run_plot_code(CODE, out)
    _check("png written", bool(ok.png_path) and open(ok.png_path, "rb").read(4) == b"\x89PNG", repr(ok))
    _check("error reported", (run_plot_code("raise ValueError('boom')", out).error or "").endswith("boom"))
    _check("timeout reported", "longer than" in (run_plot_code("while True: pass", out, timeout=2).error or ""))
    _check("isolated from REPL", run_plot_code("print(_persistent_namespace)", out).error is not None)
    _check("no figure reported", "drew no figure" in (run_plot_code("x = 1", out).error or ""))


def test_show_mode_draws() -> None:
    print("\n[2] show mode draws")
    turn = _ask({"answer": "Here.", "plot_code": CODE}, plot_mode="show")
    _check("code kept", turn.plot_code == CODE)
    _check("figure made", bool(turn.plot_png) and os.path.exists(turn.plot_png), turn.plot_error)


def test_student_mode_code_only() -> None:
    print("\n[3] student mode keeps code only")
    turn = _ask({"answer": "Here.", "plot_code": CODE}, plot_mode="student")
    _check("code kept, no figure", turn.plot_code == CODE and not turn.plot_png and turn.plot_mode == "student")


def test_bad_plot_code_keeps_answer() -> None:
    print("\n[4] bad plot code keeps the answer")
    turn = _ask({"answer": "Here.", "plot_code": "1/0"}, plot_mode="show")
    _check("answer intact", turn.content.startswith("Here.") and not turn.failed, repr(turn.content))
    _check("error shown", "ZeroDivisionError" in turn.plot_error, turn.plot_error)


def test_structure_field() -> None:
    print("\n[5] structure field")
    turn = _ask({"answer": "See it.", "structure": "mini_xray.pdb"}, search_dirs=[FIXTURES])
    _check("resolved from search dirs", turn.structure_path.endswith(".pdb"), turn.structure_path)
    _check("unknown name ignored", _ask({"answer": "x", "structure": "nope.pdb"}).structure_path == "")


def test_ui_dicts() -> None:
    print("\n[6] UI turn dict carries visuals")
    from biomentis import ui_tutor

    turn = _ask({"answer": "Here.", "plot_code": CODE}, plot_mode="student")
    d = ui_tutor._visuals_dict(turn)
    _check("five keys", set(d) == {"plot_code", "plot_png", "plot_error", "structure_path", "plot_mode"}, repr(d))
    _check("values copied", d["plot_code"] == CODE and d["plot_mode"] == "student")


def run_all() -> int:
    print("=" * 60)
    print("Chat visuals tests")
    print("=" * 60)
    for t in (test_run_plot_code, test_show_mode_draws, test_student_mode_code_only,
              test_bad_plot_code_keeps_answer, test_structure_field, test_ui_dicts):
        try:
            t()
        except Exception as exc:
            import traceback

            traceback.print_exc()
            _FAILED.append((t.__name__, repr(exc)))
    print(f"\n{len(_PASSED)} passed, {len(_FAILED)} failed")
    for n, d in _FAILED:
        print(f"  FAILED {n}: {d}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(run_all())
