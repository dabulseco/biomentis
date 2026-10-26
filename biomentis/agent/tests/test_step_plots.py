"""Tests for plots shown in (or withheld from) the agent's steps.

The class plot setting decides whether a figure the agent makes is shown
("show") or withheld so students run the code themselves ("student"). The
producer bakes the setting into each event's `file_kind`, so a run displays
the way it was recorded — live, after re-attach, and after restore —
whatever the sidebar says now. Under "student" no figure may display.

What these tests pin down:

  1.  "show": one captured PNG yields one `plot` event
  2.  "student": it yields `plot_withheld`, whose entry carries the note
  3.  "student": an image scraped from the observation is withheld too
  4.  A `plt.savefig` path is not shown twice when the figure was captured
  5.  A stale execution entry from an earlier step is not re-used
  6.  A restored student-mode run stays withheld
  7.  save_plot_pngs numbers files from start_index, zero-padded
  8.  The execute-node helper continues numbering across steps
  9.  The HTML export embeds a withheld plot for the instructor, labelled
  10. A live `plot` event exports as an image; withheld keeps its kind
  11. render_entry shows `plot` as an image and `plot_withheld` as a note
  12. The execute node records the current run id for the REPL thread
  13. support_tools captures figures left open after exec

Run with:
    python -m biomentis.agent.tests.test_step_plots
"""

from __future__ import annotations

import base64
import inspect
import os
import sys
import tempfile
from types import SimpleNamespace

from langchain_core.messages import AIMessage

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


# --- Fixtures -------------------------------------------------------------

# A 1x1 PNG.
_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
URI = f"data:image/png;base64,{_PNG_B64}"

_TMP = tempfile.mkdtemp(prefix="step_plots_")


def _write_png(name: str) -> str:
    path = os.path.join(_TMP, name)
    with open(path, "wb") as handle:
        handle.write(base64.b64decode(_PNG_B64))
    return path


png = _write_png("plot-001.png")
other_png = _write_png("saved_by_savefig.png")


def _agent(mode: str, image_paths: list[str], observation: str, append_entry: bool = True, prior=None):
    """A scripted agent: one code message, then (after the execute node's
    entry is appended) one observation message."""
    history: list = []
    agent = SimpleNamespace(
        use_tool_retriever=False,
        path=_TMP,
        user_task="",
        plot_mode=mode,
        _execution_results=list(prior or []),
    )

    def _stream(inputs, stream_mode=None, config=None):
        history.extend(inputs["messages"])
        history.append(AIMessage(content="Plot it.\n<execute>plt.plot([1, 2])</execute>"))
        yield {"messages": list(history)}
        if append_entry:
            agent._execution_results.append({"images": [URI] * len(image_paths), "image_paths": image_paths})
        history.append(AIMessage(content=f"<observation>{observation}</observation>"))
        yield {"messages": list(history)}

    agent.app = SimpleNamespace(stream=_stream)
    return agent


def _events(mode: str, image_paths: list[str], observation: str = "done\n", **kw) -> list:
    from biomentis.ui_core import stream_agent_events

    agent = _agent(mode, image_paths, observation, **kw)
    return list(stream_agent_events(agent, "Plot something.", [], [], thread_id=1))


# --- Tests ----------------------------------------------------------------


def test_show_mode_emits_plot():
    print("\n[1] 'show': a captured PNG is a plot event")
    ev = _events(mode="show", image_paths=[png])
    files = [e for e in ev if e.type == "file" and e.file_path]
    _check("one plot event", [e.file_kind for e in files] == ["plot"], f"{[(e.file_kind, e.file_path) for e in files]}")
    _check("titled as a plot", files and files[0].title == "📈 Plot", f"{files[0].title if files else None}")
    from biomentis.ui_core import transcript_entry_for_event

    entry = transcript_entry_for_event(files[0]) if files else {}
    _check("entry kind is plot", entry.get("kind") == "plot" and entry.get("file_path") == png, f"{entry}")


def test_student_mode_withholds():
    print("\n[2] 'student': a captured PNG is withheld")
    from biomentis.ui_core import transcript_entry_for_event

    ev = _events(mode="student", image_paths=[png])
    kinds = [e.file_kind for e in ev if e.file_path]
    _check("withheld, not shown", kinds == ["plot_withheld"], f"{kinds}")
    entry = transcript_entry_for_event(next(e for e in ev if e.file_path))
    _check(
        "entry carries the note",
        entry["kind"] == "plot_withheld" and "Run this code" in entry["content"],
        f"{entry}",
    )
    _check("entry keeps the path for the export", entry.get("file_path") == png, f"{entry}")


def test_student_mode_withholds_scraped_images():
    print("\n[3] 'student': a scraped image is withheld too (Review Focus 1)")
    ev = _events(mode="student", image_paths=[], observation=f"wrote {png}\n")
    kinds = [e.file_kind for e in ev if e.file_path]
    _check("scraped image withheld", kinds == ["plot_withheld"], f"{kinds}")


def test_show_mode_keeps_scraped_images():
    print("\n[3b] 'show': a scraped image is still an image preview")
    ev = _events(mode="show", image_paths=[], observation=f"wrote {png}\n")
    kinds = [e.file_kind for e in ev if e.file_path]
    _check("scraped image shown", kinds == ["image"], f"{kinds}")


def test_savefig_not_duplicated():
    print("\n[4] A savefig path is not shown twice (Review Focus 2)")
    ev = _events(mode="show", image_paths=[png], observation=f"Plot saved to: {other_png}\n")
    shown = [e for e in ev if e.file_path]
    _check("one figure, not two", len(shown) == 1, f"{[(e.file_kind, e.file_path) for e in shown]}")
    _check("the captured one wins", shown and shown[0].file_path == png, "")
    # With nothing captured, a savefig path is still the only record of the figure.
    ev = _events(mode="show", image_paths=[], observation=f"Plot saved to: {other_png}\n")
    _check(
        "savefig path still shown when nothing was captured",
        [e.file_path for e in ev if e.file_path] == [other_png],
        f"{[e.file_path for e in ev if e.file_path]}",
    )


def test_stale_entry_not_reused():
    print("\n[5] An execution entry from an earlier step is not re-used")
    ev = _events(
        mode="show",
        image_paths=[],
        append_entry=False,
        prior=[{"images": [URI], "image_paths": [png]}],
    )
    _check("no plot from the stale entry", [e for e in ev if e.file_path] == [], f"{[e.file_path for e in ev if e.file_path]}")


def test_recorded_mode_survives_restore():
    print("\n[6] A restored student-mode run stays withheld")
    from biomentis.run_journal import RunJournal, entries_for_run, load_run

    run_dir = tempfile.mkdtemp(prefix="step_plots_runs_")
    journal = RunJournal("Plot something.", run_dir=run_dir, plot_mode="student")
    for event in _events(mode="student", image_paths=[png]):
        journal.append(event)
    journal.finish("complete")
    # The session (and the agent) have since been switched back to "show";
    # the restore must not consult either.
    entries = entries_for_run(load_run(journal.path))
    kinds = [e["kind"] for e in entries if e.get("file_path")]
    _check("restored as withheld", kinds == ["plot_withheld"], f"{kinds}")
    withheld = [e for e in entries if e.get("kind") == "plot_withheld"]
    _check("restored note intact", withheld and "Run this code" in withheld[0]["content"], f"{withheld}")


def test_save_plot_pngs_numbering():
    print("\n[7] save_plot_pngs numbers from start_index")
    from biomentis.plots import save_plot_pngs

    d = tempfile.mkdtemp(prefix="step_plots_save_")
    paths = save_plot_pngs([URI, URI], d, start_index=3)
    _check(
        "numbered from start",
        [os.path.basename(p) for p in paths] == ["plot-003.png", "plot-004.png"],
        f"{paths}",
    )
    _check("absolute paths", all(os.path.isabs(p) for p in paths), f"{paths}")
    _check(
        "bytes decoded",
        all(open(p, "rb").read() == base64.b64decode(_PNG_B64) for p in paths),
        "",
    )
    nested = os.path.join(d, "not", "yet")
    _check("creates the directory", len(save_plot_pngs([URI], nested, start_index=1)) == 1, "")
    _check("skips a non-PNG URI", save_plot_pngs(["not a uri"], d, start_index=9) == [], "")


def test_execute_helper_continues_numbering():
    print("\n[8] The execute node's helper writes to the run's dir and continues numbering")
    from biomentis.agent.a1 import A1

    root = tempfile.mkdtemp(prefix="step_plots_art_")
    old = os.environ.get("BIOMENTIS_ARTIFACT_DIR")
    os.environ["BIOMENTIS_ARTIFACT_DIR"] = root
    try:
        fake = SimpleNamespace(current_run_id="run-xyz")
        first = A1._save_execution_plots(fake, [URI])
        second = A1._save_execution_plots(fake, [URI, URI])
        names = [os.path.basename(p) for p in first + second]
        _check("numbering continues", names == ["plot-001.png", "plot-002.png", "plot-003.png"], f"{names}")
        _check(
            "written under the run's artifact dir",
            all(os.path.dirname(p) == os.path.join(root, "run-xyz") for p in first + second),
            f"{first + second}",
        )
        _check("nothing captured, nothing written", A1._save_execution_plots(fake, []) == [], "")
    finally:
        if old is None:
            os.environ.pop("BIOMENTIS_ARTIFACT_DIR", None)
        else:
            os.environ["BIOMENTIS_ARTIFACT_DIR"] = old

    src = inspect.getsource(A1.configure)
    _check("execution entry records image_paths", '"image_paths"' in src, "")


def test_export_keeps_withheld_plot():
    print("\n[9] The HTML export embeds a withheld plot for the instructor")
    from biomentis.ui_core import ExportEntry, render_transcript_html

    withheld_entry = ExportEntry(
        role="assistant",
        kind="plot_withheld",
        title="📈 Plot",
        content="Run this code to see the plot.",
        file_path=png,
    )
    html = render_transcript_html([withheld_entry], "Executor", "m")
    _check(
        "image embedded for the instructor",
        "data:image/png;base64" in html and "withheld from students" in html,
        html[-600:],
    )


def test_event_export_kinds():
    print("\n[10] Events export as image / plot_withheld")
    from biomentis.ui_core import UIEvent, ui_event_to_export_entry

    shown = ui_event_to_export_entry(UIEvent(type="file", file_kind="plot", title="📈 Plot", file_path=png))
    _check("plot exports as image", shown.kind == "image", shown.kind)
    held = ui_event_to_export_entry(
        UIEvent(
            type="file",
            file_kind="plot_withheld",
            title="📈 Plot",
            content="Run this code to see the plot.",
            file_path=png,
        )
    )
    _check("withheld keeps its kind", held.kind == "plot_withheld", held.kind)

    from biomentis.agent.a1 import A1

    launch = inspect.getsource(A1.launch_streamlit_demo)
    _check("transcript export maps plot to image", '"plot"' in launch and "export_kind" in launch, "")


def test_render_entry_branches():
    print("\n[11] render_entry: plot is an image, plot_withheld is a note")
    from biomentis.agent.a1 import A1

    launch = inspect.getsource(A1.launch_streamlit_demo)
    start = launch.find("def render_entry(")
    body = launch[start : launch.find("class _StepBoxTracker", start)]
    _check("plot branch renders the image", 'in ("image", "plot")' in body, "")
    held = body[body.find('== "plot_withheld"') :]
    held = held[: held.find("elif")]
    _check("withheld branch shows st.info", "st.info(" in held, held)
    _check("withheld branch never shows the image", "st.image" not in held, held)

    gradio = inspect.getsource(A1.launch_gradio_demo)
    _check("gradio withholds too", "plot_withheld" in gradio, "")


def test_execute_sets_current_run_id():
    print("\n[12] The execute node records the current run id for the REPL thread")
    import threading

    from biomentis.agent.a1 import A1
    from biomentis.structures import current_run_id, set_current_run_id

    set_current_run_id("run-abc")
    seen: list = []
    t = threading.Thread(target=lambda: seen.append(current_run_id()))
    t.start()
    t.join()
    _check("visible from another thread", seen == ["run-abc"], f"{seen}")
    set_current_run_id(None)
    _check("cleared", current_run_id() is None, "")

    src = inspect.getsource(A1.configure)
    execute = src[src.find("def execute(state") : src.find("def routing_function")]
    i_set = execute.find("set_current_run_id(self.current_run_id)")
    i_run = execute.find("run_with_timeout(")
    _check("execute sets it before running code", 0 <= i_set < i_run, f"set at {i_set}, run at {i_run}")


def test_repl_captures_open_figures():
    print("\n[13] support_tools captures figures left open after exec")
    from biomentis.tool import support_tools

    src = inspect.getsource(support_tools.run_python_repl)
    _check("capture enabled", "\n            _capture_matplotlib_plots()" in src and "# _capture_matplotlib_plots()" not in src, "")
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        print("  (matplotlib unavailable; skipping live capture)")
        return
    support_tools.clear_captured_plots()
    support_tools.run_python_repl("import matplotlib.pyplot as plt\nplt.figure()\nplt.plot([1, 2, 3])")
    plots = support_tools.get_captured_plots()
    _check("one figure captured", len(plots) == 1 and plots[0].startswith("data:image/png;base64,"), f"{len(plots)}")
    support_tools.clear_captured_plots()


def run_all() -> int:
    print("=" * 60)
    print("Step plot tests")
    print("=" * 60)
    for test in (
        test_show_mode_emits_plot,
        test_student_mode_withholds,
        test_student_mode_withholds_scraped_images,
        test_show_mode_keeps_scraped_images,
        test_savefig_not_duplicated,
        test_stale_entry_not_reused,
        test_recorded_mode_survives_restore,
        test_save_plot_pngs_numbering,
        test_execute_helper_continues_numbering,
        test_export_keeps_withheld_plot,
        test_event_export_kinds,
        test_render_entry_branches,
        test_execute_sets_current_run_id,
        test_repl_captures_open_figures,
    ):
        try:
            test()
        except Exception as exc:  # a crash is a failure, not an abort
            _check(f"{test.__name__} ran", False, f"{type(exc).__name__}: {exc}")

    print("\n" + "=" * 60)
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    print("=" * 60)
    for name, detail in _FAILED:
        print(f"  ✗ {name}: {detail}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(run_all())
