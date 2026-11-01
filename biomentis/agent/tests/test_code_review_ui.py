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
st.session_state.setdefault("biomni_tutor_run", {{"step_id": 3, "paused_at": 12.5, "step_qa": {{3: []}}}})
st.session_state.setdefault("biomni_run_cursor", 7)
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
    return "\n".join(str(e.value) for e in list(at.markdown) + list(at.caption) + list(at.warning) + list(at.error)) + "\n" + "\n".join(str(s.label) for s in at.status)


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
    before = {k: at.session_state[k] for k in ("biomni_run_active", "biomni_tutor_run", "biomni_run_cursor")}
    keys_before = set(at.session_state.filtered_state)
    at.button[0].click().run()
    after = {k: at.session_state[k] for k in ("biomni_run_active", "biomni_tutor_run", "biomni_run_cursor")}
    new_keys = set(at.session_state.filtered_state) - keys_before
    _check("run and tutor state unchanged", before == after, f"{before} vs {after}")
    _check("no new session keys beyond the widget", all(k.startswith("blk_") for k in new_keys), str(new_keys))
    gate.set()
    jobs._wait_for(_key())


def test_model_text_shown_literally() -> None:
    from biomentis import code_review_jobs as jobs

    txt = "costs $5 per 500 g and $6 total"
    at = _start(_reply(summary=txt, issues=[{"line": 1, "kind": "units", "severity": "low", "quote": "q", "problem": txt, "fix": "a_b*c"}]))
    at.button[0].click().run()
    jobs._wait_for(_key())
    at.run()
    vals = [m.value for m in at.markdown]
    _check("dollar signs escaped in summary", any(v == r"costs \$5 per 500 g and \$6 total" for v in vals), str(vals))
    _check("dollar signs escaped in problem", any(v.startswith("Problem: ") and r"\$5" in v for v in vals), str(vals))
    _check("underscore and star escaped in fix", any(v == r"Fix: a\_b\*c" for v in vals), str(vals))
    _check("no LaTeX element", len(at.latex) == 0)


def test_empty_error_falls_back() -> None:
    from biomentis import code_review_jobs as jobs
    from biomentis.code_review import CodeReview

    at = _start(_reply())
    with jobs._lock:
        jobs._store[_key()] = {"status": "error", "review": CodeReview(error=""), "run_id": None, "block_index": 2, "seq": 1}
    at.run()
    _check("blank error replaced", [e.value for e in at.error] == ["The review failed."], str([e.value for e in at.error]))


# --- Where the button appears (Task 7) ------------------------------------

# Drives the REAL `A1.launch_streamlit_demo` over a seeded transcript, no run
# in flight. The review model and client are patched in `ui_settings`, where
# `render_code_review` looks them up.
_APP_HARNESS = """
import sys, os
sys.path.insert(0, {repo!r})
os.environ["BIOMENTIS_RUN_DIR"] = {run_dir!r}
import streamlit as st
from types import SimpleNamespace
import biomentis.ui_settings as ui_settings
from biomentis.agent.tests.test_code_review_ui import FACTORY, TRANSCRIPT
ui_settings.code_review_model = lambda: "fake-model"
ui_settings.code_review_llm = FACTORY[0]
from biomentis.agent.a1 import A1

st.session_state.setdefault("biomni_session_key", "apptest-cr")
st.session_state.setdefault("biomni_agent_key", "test")
st.session_state.setdefault("biomni_output_dir", {run_dir!r})
if not st.session_state.get("_seeded"):
    st.session_state["_seeded"] = True
    st.session_state.biomni_transcript = TRANSCRIPT[0]
agent = SimpleNamespace(
    main_history_copy=[],
    path=os.path.join({run_dir!r}, "agent"),
    llm=SimpleNamespace(model_name="test-model"),
)
A1.launch_streamlit_demo(agent)
"""

TRANSCRIPT: list = [[]]
APP_DIR: list = [""]  # the harness's run dir (also its session output folder)
REVIEW_LABEL = "🔎 Code Review"


def _user(text="the task"):
    return {"panel": "main", "role": "user", "content": text}


def _reasoning(text):
    return {"panel": "inner", "role": "assistant", "content": text, "title": "🤔 Reasoning"}


def _code(text, run_id="run-1", block_index=0):
    e = {"panel": "inner", "role": "assistant", "kind": "code", "content": text, "language": "python", "title": "🛠️ Code"}
    if run_id is not None:
        e["journal_run_id"], e["block_index"] = run_id, block_index
    return e


def _obs(text):
    return {"panel": "inner", "role": "assistant", "kind": "observation", "content": text, "collapsible": True}


def _answer(text, title="✅ Answer"):
    return {"panel": "main", "role": "assistant", "content": text, "title": title}


def _start_app(transcript, reply=None):
    from streamlit.testing.v1 import AppTest

    from biomentis import code_review_jobs as jobs

    jobs._clear_for_tests()
    CALLS.clear()
    FACTORY[0] = _factory(reply if reply is not None else _reply())
    TRANSCRIPT[0] = transcript
    tmp = tempfile.mkdtemp(prefix="biomni_cr_ui_")
    APP_DIR[0] = tmp
    path = os.path.join(tmp, "app.py")
    with open(path, "w") as f:
        f.write(_APP_HARNESS.format(repo=_REPO, run_dir=tmp))
    return AppTest.from_file(path, default_timeout=60).run()


def _review_buttons(at):
    return [b for b in at.button if b.label == REVIEW_LABEL]


def test_button_on_agent_code_step() -> None:
    print("\n[9] agent code step: one button, keyed by run and block; click sends output and task")
    from biomentis import code_review_jobs as jobs
    from biomentis.code_review import request_for_entry, review_key

    t = [_user("compute x"), _reasoning("set x"), _code("x = 1\nprint(x)", "run-1", 0), _obs("1")]
    at = _start_app(t)
    _check("no exception", not at.exception, str(at.exception))
    buttons = _review_buttons(at)
    _check("one Code Review button", len(buttons) == 1, str([b.label for b in at.button]))
    _check("keyed by journal run id and block", [b.key for b in buttons] == ["cr-run-1-0_go"], str([b.key for b in buttons]))
    buttons[0].click().run()
    req = request_for_entry(t, t[2], "compute x")
    key = review_key(req, "fake-model")
    _check("click queues the review of this step", jobs.review_status(key)[0] in ("pending", "done"), jobs.review_status(key)[0])
    _check("request has output, purpose and task", (req.output, req.purpose, req.task) == ("1", "set x", "compute x"))
    jobs._wait_for(key)
    at.run()
    _check("report shown under the step", [e.label for e in at.expander if e.label.startswith("Code review")] == ["Code review — ⚠️ Run with fixes"], str([e.label for e in at.expander]))


def test_untagged_and_repeated_code_get_unique_keys() -> None:
    print("\n[10] old untagged entries and repeated code never collide")
    import hashlib

    t = [_user(), _code("print(1)", None), _obs("1"), _code("print(1)", None), _obs("1")]
    at = _start_app(t)
    _check("no exception", not at.exception, str(at.exception))
    h = hashlib.sha1(b"print(1)").hexdigest()[:12]
    keys = [b.key for b in _review_buttons(at)]
    _check("sha1 fallback key, repeat suffixed", keys == [f"cr-{h}_go", f"cr-{h}-2_go"], str(keys))


def test_buttons_under_answer_blocks() -> None:
    print("\n[11] answer with two fenced blocks: two buttons, in the answer's message")
    from biomentis import code_review_jobs as jobs
    from biomentis.code_review import review_key

    answer = "Here is the result.\n\n```python\na = 2\n```\n\nAnd in bash:\n\n```bash\necho hi\n```\n"
    t = [_user("the task"), _code("a = 1", "run-1", 0), _obs("ok"), _answer(answer)]
    at = _start_app(t)
    _check("no exception", not at.exception, str(at.exception))
    keys = [b.key for b in _review_buttons(at)]
    _check("three buttons: one step, two answer blocks", len(keys) == 3, str(keys))
    ans_keys = [k for k in keys if k.startswith("cr-ans-")]
    _check("answer keys numbered 0 and 1", [k.rsplit("-", 1)[-1] for k in ans_keys] == ["0_go", "1_go"], str(ans_keys))
    msg = next((m for m in at.chat_message if any("Here is the result." in md.value for md in m.markdown)), None)
    _check(
        "both answer buttons inside the answer's message",
        msg is not None and len([b for b in msg.button if b.label == REVIEW_LABEL]) == 2,
        "",
    )
    btn = next(b for b in _review_buttons(at) if b.key == ans_keys[1])
    btn.click().run()
    from biomentis.code_review import CodeReviewRequest
    from biomentis.run_journal import format_code_block

    req = CodeReviewRequest(
        code="echo hi", language="bash", task="the task",
        earlier_code=format_code_block(1, "python", "a = 1"), block_index=1, source="answer",
    )
    key = review_key(req, "fake-model")
    _check("bash block reviewed with task and earlier code", jobs.review_status(key)[0] in ("pending", "done"), jobs.review_status(key)[0])
    jobs._wait_for(key)


def test_revised_answers_each_get_buttons() -> None:
    print("\n[12] self-critic: a revised answer, and a repeated one, both get buttons")
    a1 = "```python\nx = 1\n```"
    a2 = "```python\nx = 2\n```"
    t = [_user(), _answer(a1), _answer(a2, "✅ Answer (revised after self-critique #1)"), _answer(a2, "✅ Solution")]
    at = _start_app(t)
    _check("no exception", not at.exception, str(at.exception))
    keys = [b.key for b in _review_buttons(at)]
    _check("one button per answer", len(keys) == 3 and len(set(keys)) == 3, str(keys))


def test_no_button_without_code() -> None:
    print("\n[13] reasoning and observations only: no button")
    t = [_user(), _reasoning("thinking"), _obs("some output"), _answer("No code here.")]
    at = _start_app(t)
    _check("no exception", not at.exception, str(at.exception))
    _check("no Code Review button", len(_review_buttons(at)) == 0, str([b.label for b in at.button]))


_TUTOR_HARNESS = """
import sys
sys.path.insert(0, {repo!r})
import streamlit as st
import biomentis.ui_settings as ui_settings
from biomentis.agent.tests.test_code_review_ui import FACTORY
ui_settings.code_review_model = lambda: "fake-model"
ui_settings.code_review_llm = FACTORY[0]
from biomentis.ui_tutor import _render_visuals

_render_visuals({{"plot_code": "import matplotlib.pyplot as plt\\nplt.plot([1, 2])", "plot_png": None, "task": "plot it"}}, key="chat-panel-0")
_render_visuals({{"plot_code": None, "plot_png": None}}, key="chat-panel-1")
"""


def test_button_on_tutor_plot_code() -> None:
    print("\n[14] tutor plot code: one button, keyed by the turn")
    from streamlit.testing.v1 import AppTest

    from biomentis import code_review_jobs as jobs

    jobs._clear_for_tests()
    FACTORY[0] = _factory(_reply())
    path = os.path.join(tempfile.mkdtemp(), "tutor.py")
    with open(path, "w") as f:
        f.write(_TUTOR_HARNESS.format(repo=_REPO))
    at = AppTest.from_file(path, default_timeout=60).run()
    _check("no exception", not at.exception, str(at.exception))
    keys = [b.key for b in _review_buttons(at)]
    _check("one button for the turn with code", keys == ["cr-tutor-chat-panel-0_go"], str(keys))


_REPL_HARNESS = """
import sys
sys.path.insert(0, {repo!r})
import streamlit as st
import biomentis.ui_settings as ui_settings
from biomentis.agent.tests.test_code_review_ui import FACTORY
ui_settings.code_review_model = lambda: "fake-model"
ui_settings.code_review_llm = FACTORY[0]
from biomentis.ui_repl import render_repl_panel

st.session_state.setdefault("biomni_repl_history", [
    {{"code": "y = 3\\nprint(y)", "output": "3"}},
    {{"code": "# (namespace cleared)", "output": ""}},
    {{"code": "print('hi')", "output": "hi"}},
])
render_repl_panel(None)
"""


def test_button_on_repl_history() -> None:
    print("\n[15] REPL history: a button per code entry, none on the cleared marker")
    from streamlit.testing.v1 import AppTest

    from biomentis import code_review_jobs as jobs

    jobs._clear_for_tests()
    FACTORY[0] = _factory(_reply())
    path = os.path.join(tempfile.mkdtemp(), "repl.py")
    with open(path, "w") as f:
        f.write(_REPL_HARNESS.format(repo=_REPO))
    at = AppTest.from_file(path, default_timeout=60).run()
    _check("no exception", not at.exception, str(at.exception))
    keys = [b.key for b in _review_buttons(at)]
    _check("buttons for #3 and #1 only", keys == ["cr-repl-3_go", "cr-repl-1_go"], str(keys))


def test_answer_review_reloaded_after_restart() -> None:
    print("\n[16] an answer-block review is saved under the run dir and reloaded")
    from biomentis import code_review_jobs as jobs
    from biomentis.code_review import CodeReviewRequest, review_key

    t = [_user("the task"), _answer("```python\nz = 3\n```")]
    at = _start_app(t)
    run_dir = APP_DIR[0]
    _review_buttons(at)[0].click().run()
    key = review_key(CodeReviewRequest(code="z = 3", task="the task", block_index=0, source="answer"), "fake-model")
    jobs._wait_for(key)
    _check("saved in the run dir's no-run sidecar", os.path.exists(os.path.join(run_dir, jobs.NO_RUN_SIDECAR)), str(os.listdir(run_dir)))
    _check("not in the session output folder's file", not os.path.exists(os.path.join(run_dir, jobs.OUTPUT_SIDECAR)))
    jobs._clear_for_tests()
    _check("store emptied (restart)", jobs.review_status(key)[0] == "none")
    loaded = jobs.load_saved_reviews(run_dir)
    _check("reloaded from disk", loaded == 1 and jobs.review_status(key)[0] == "done", f"{loaded} {jobs.review_status(key)[0]}")
    at.run()
    _check("replayed answer shows the stored report", any(e.label.startswith("Code review") for e in at.expander) and not _review_buttons(at), str([e.label for e in at.expander]))


class _FakeChat:
    def ask(self, **kw):
        from types import SimpleNamespace

        return SimpleNamespace(
            content="Here is a plot.", citations=[], also_consider=[], bloom_level=None, dok_level=None,
            rubric_hit=[], confidence=None, failed=False, plot_code="import matplotlib.pyplot as plt\nplt.plot([1])",
            plot_png=None, plot_error="", structure_path=None, plot_mode="student",
        )


TUTOR: list = [None]
RUN: list = [None]

_STEP_QA_HARNESS = """
import sys
sys.path.insert(0, {repo!r})
import streamlit as st
import biomentis.ui_settings as ui_settings
from biomentis.agent.tests.test_code_review_ui import FACTORY, TUTOR, RUN
ui_settings.code_review_model = lambda: "fake-model"
ui_settings.code_review_llm = FACTORY[0]
from biomentis.ui_tutor import _render_step_qa

_render_step_qa(st, 1, RUN[0], TUTOR[0], "r1")
"""


def test_tutor_turn_keeps_its_task() -> None:
    print("\n[17] a tutor turn's review keeps its own task when a new task starts")
    from types import SimpleNamespace

    from streamlit.testing.v1 import AppTest

    from biomentis import code_review_jobs as jobs

    jobs._clear_for_tests()
    FACTORY[0] = _factory(_reply())
    TUTOR[0] = SimpleNamespace(chat=_FakeChat(), current_prompt="task A", kb=None)
    RUN[0] = {"prompt": "task A"}
    path = os.path.join(tempfile.mkdtemp(), "stepqa.py")
    with open(path, "w") as f:
        f.write(_STEP_QA_HARNESS.format(repo=_REPO))
    at = AppTest.from_file(path, default_timeout=60).run()
    at.text_input(key="biomni_tutor_step_ask_input_r1_1").input("plot it?")
    next(b for b in at.button if b.label == "💬 Ask about this step").click().run()
    _check("no exception", not at.exception, str(at.exception))
    turn = RUN[0]["step_qa"][1][0]
    _check("task stored on the turn", turn.get("task") == "task A", str(turn.get("task")))
    buttons = _review_buttons(at)
    _check("button under the turn's plot code", [b.key for b in buttons] == ["cr-tutor-chat-step-r1_1-0_go"], str([b.key for b in buttons]))
    buttons[0].click().run()
    with jobs._lock:
        keys = list(jobs._store)
    _check("one review queued", len(keys) == 1, str(keys))
    jobs._wait_for(keys[0])
    TUTOR[0].current_prompt = "task B"  # the student starts a new task
    at.run()
    _check("review still shown after the new task", any(e.label.startswith("Code review") for e in at.expander), str([e.label for e in at.expander]))
    _check("no button reappears", not _review_buttons(at), str([b.key for b in _review_buttons(at)]))
    with jobs._lock:
        _check("no second review queued", list(jobs._store) == keys, str(list(jobs._store)))


class _FakeLLM_App:
    """Stands in for a chat model; any call would mean the smoke touched a model."""

    def invoke(self, *a, **k):
        raise AssertionError("the app smoke must not call a model")


class _FakeAgent:
    """Just enough of A1 for streamlit_app.py to render one idle page."""

    def __init__(self, *a, **k):
        self.path = "./data"
        self.llm = _FakeLLM_App()
        self.review_llm = None
        self.review_model_name = ""

    def set_model(self, *a, **k):
        pass

    def set_temperatures(self, *a, **k):
        return False

    def configure(self, *a, **k):
        pass

    def launch_streamlit_demo(self, *a, **k):
        import streamlit as st

        st.write("fake agent page")


def test_app_renders_with_code_review_model_setting() -> None:
    """Smoke: streamlit_app.py renders headlessly (agent + Ollama faked) and
    shows the "Code review model" setting."""
    from unittest import mock

    from streamlit.testing.v1 import AppTest

    import biomentis.agent as agent_pkg
    import biomentis.ollama_utils as ollama_utils

    saved_env = {k: os.environ.pop(k, None) for k in ("BIOMNI_LLM", "BIOMNI_SOURCE")}
    try:
        with mock.patch.object(agent_pkg, "A1", _FakeAgent), mock.patch.object(
            ollama_utils, "list_ollama_models", lambda *a, **k: ["fake-review:1b", "fake-other:2b"]
        ), mock.patch("biomentis.code_review_jobs.load_saved_reviews", lambda: 0):
            at = AppTest.from_file(os.path.join(_REPO, "streamlit_app.py"), default_timeout=60)
            at.run()
    finally:
        for k, v in saved_env.items():
            if v is not None:
                os.environ[k] = v
    _check("app smoke: no exception", not at.exception, str([e.value for e in at.exception]))
    labels = [sb.label for sb in at.selectbox]
    _check("app smoke: shows Code review model setting", "Code review model" in labels, str(labels))
    headers = [h.value for h in at.sidebar.header]
    _check(
        "app smoke: picker under its own Code review header, after Ollama Cloud",
        "Code review" in headers and headers.index("Code review") == headers.index("Ollama Cloud") + 1,
        str(headers),
    )
    picker = [sb for sb in at.selectbox if sb.label == "Code review model"]
    _check(
        "app smoke: picker lists the faked Ollama models",
        bool(picker) and "fake-review:1b" in list(picker[0].options),
        str([list(sb.options) for sb in picker]),
    )


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
    test_model_text_shown_literally()
    test_empty_error_falls_back()
    test_button_on_agent_code_step()
    test_untagged_and_repeated_code_get_unique_keys()
    test_buttons_under_answer_blocks()
    test_revised_answers_each_get_buttons()
    test_no_button_without_code()
    test_button_on_tutor_plot_code()
    test_button_on_repl_history()
    test_answer_review_reloaded_after_restart()
    test_tutor_turn_keeps_its_task()
    test_app_renders_with_code_review_model_setting()
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
