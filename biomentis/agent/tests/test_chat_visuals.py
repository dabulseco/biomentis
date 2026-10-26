"""Tests for plots and structures in the tutor conversation.

  1.  run_plot_code runs in a subprocess and reports errors, timeouts, no figure
  2.  Plot mode "show" draws the tutor's plot_code; "student" keeps code only
  3.  Bad plot code leaves the answer intact and reports the error
  4.  A `structure` field is resolved against the search dirs; unknown is ignored
  5.  The page panel and step Q&A turn dicts carry the visuals keys
  6.  A reply that shows a structure is checked against that file's header
      and its answer corrected; if the check fails the first answer stays

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

    steer = run_plot_code("import os\nprint('PNG:/etc/hosts')\nos._exit(0)", out)
    _check("child cannot steer the png path", steer.png_path is None, repr(steer))
    os.environ["BIOMENTIS_SENTINEL_KEY"] = "s3cret"
    leak = run_plot_code("import os\nassert 'BIOMENTIS_SENTINEL_KEY' not in os.environ, 'leaked'", out)
    _check("parent env not visible in child", leak.error is None or "leaked" not in leak.error, repr(leak))
    del os.environ["BIOMENTIS_SENTINEL_KEY"]
    import time

    t0 = time.time()
    res = run_plot_code("import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\nwhile True: pass", out, timeout=2)
    _check("timeout with a grandchild returns promptly", time.time() - t0 < 10 and "longer than 2 s" in (res.error or ""), repr(res))


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
    from biomentis.structures import resolve_structure_source

    absolute = os.path.join(FIXTURES, "mini_xray.pdb")
    _check("absolute path ignored on the chat path", resolve_structure_source(absolute, None, []) is None)
    _check("path allowed when asked", bool(resolve_structure_source(absolute, None, [], allow_paths=True)))
    _check("unknown name ignored", _ask({"answer": "x", "structure": "nope.pdb"}).structure_path == "")


def test_ui_dicts() -> None:
    print("\n[6] UI turn dict carries visuals")
    from biomentis import ui_tutor

    turn = _ask({"answer": "Here.", "plot_code": CODE}, plot_mode="student")
    d = ui_tutor._visuals_dict(turn)
    _check("five keys", set(d) == {"plot_code", "plot_png", "plot_error", "structure_path", "plot_mode"}, repr(d))
    _check("values copied", d["plot_code"] == CODE and d["plot_mode"] == "student")


class _ScriptLLM:
    """Replies in order; records each prompt it was sent."""

    def __init__(self, replies: list[str]):
        self.model = "fake"
        self.replies = list(replies)
        self.prompts: list = []

    def invoke(self, messages):
        self.prompts.append(messages)
        return AIMessage(content=self.replies.pop(0) if self.replies else _CLASSIFY)


def test_structure_answer_checked_against_header() -> None:
    print("\n[6] the answer is checked against the structure's header")
    from biomentis.agent.tutor.chat import TutorChat

    wrong = json.dumps({"answer": "This is a cryo-EM structure of a mouse ion channel.",
                        "follow_up": "What does it bind?", "structure": "mini_xray.pdb"})
    right = json.dumps({"answer": "This is an X-ray structure (1.80 Å) of a human test hydrolase with a ligand.",
                        "follow_up": "Which chains form it?", "structure": "mini_xray.pdb"})
    llm = _ScriptLLM([wrong, right])
    turn = TutorChat(llm).ask("Show me this structure", mode="chat", search_dirs=[FIXTURES])
    check = llm.prompts[1][-1].content if len(llm.prompts) > 1 else ""
    _check("the header was shown to the LLM", "MINI TEST PROTEIN" in check and "X-RAY DIFFRACTION" in check, check[:300])
    _check("the corrected answer is kept", "human test hydrolase" in turn.content, turn.content)
    _check("with its follow-up", "Which chains form it?" in turn.content, turn.content)
    _check("the structure still shows", turn.structure_path.endswith("mini_xray.pdb"), turn.structure_path)

    llm = _ScriptLLM([wrong, "not json at all"])
    turn = TutorChat(llm).ask("Show me this structure", mode="chat", search_dirs=[FIXTURES])
    _check("a failed check keeps the first answer", "cryo-EM" in turn.content, turn.content)

    llm = _ScriptLLM([json.dumps({"answer": "No molecule here.", "structure": ""})])
    TutorChat(llm).ask("Explain", mode="chat")
    _check("no structure, no extra call", not any("STRUCTURE_INFO" in str(p[-1].content) for p in llm.prompts),
           str(len(llm.prompts)))


def run_all() -> int:
    print("=" * 60)
    print("Chat visuals tests")
    print("=" * 60)
    for t in (test_run_plot_code, test_show_mode_draws, test_student_mode_code_only,
              test_bad_plot_code_keeps_answer, test_structure_field, test_ui_dicts,
              test_structure_answer_checked_against_header):
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
