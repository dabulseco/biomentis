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
  14. A failing step owns its figure
  15. A consumer split mid-run re-attaches with each plot/withheld/structure
      event exactly once, and restore matches
  16. The patched plt.show prints "Plot captured." (neutral under "student")
  17. A plot is still saved after Streamlit re-imports support_tools mid-session
  18. plt patched by the old module copy still feeds the new one (runs 3-4,
      2026-10-29); REPL variables survive the reload too
  19. A reported image is shown from a copy taken then; an unchanged image
      is not shown again
  20. The figures the answer cites appear under it in the main panel, latest
      version, withheld under "student"; uncited ones are listed as not found
  21. The export is one folder: two pages, figures/ beside them, nothing embedded
  22. Matching citations to figures: lists, ranges, code blocks, latest wins

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
# Images a step reports are copied into the run folder; keep them out of runs/.
os.environ["BIOMENTIS_ARTIFACT_DIR"] = tempfile.mkdtemp(prefix="step_plots_runs_")


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
    # With nothing captured, a savefig path is still the only record of the
    # figure; it is shown from a copy taken when the step reported it.
    ev = _events(mode="show", image_paths=[], observation=f"Plot saved to: {other_png}\n")
    shown = [e.file_path for e in ev if e.file_path]
    _check(
        "savefig path still shown when nothing was captured, from a copy",
        len(shown) == 1 and shown[0].endswith("image-001-saved_by_savefig.png") and open(shown[0], "rb").read() == open(other_png, "rb").read(),
        f"{shown}",
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
    _check("capture enabled", "_capture_matplotlib_plots()" in src and "# _capture_matplotlib_plots()" not in src, "")
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


def test_failing_step_owns_its_figure():
    print("\n[14] A step that raises owns its partial figure; the next step shows none")
    from biomentis.tool import support_tools

    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        print("  (matplotlib unavailable; skipping)")
        return
    # Step 1: draws, then raises. The execute node clears before each step.
    support_tools.clear_captured_plots()
    out = support_tools.run_python_repl(
        "import matplotlib.pyplot as plt\nplt.figure()\nplt.plot([1, 2])\nraise ValueError('boom')"
    )
    step1 = support_tools.get_captured_plots()
    _check("the failing step reports its error", out.startswith("Error:"), out)
    _check("the failing step's figure is attributed to it", len(step1) == 1, f"{len(step1)}")
    # Step 2: no plotting at all.
    support_tools.clear_captured_plots()
    support_tools.run_python_repl("x = 1 + 1")
    step2 = support_tools.get_captured_plots()
    _check("a non-plotting next step captures nothing", step2 == [], f"{len(step2)}")
    support_tools.clear_captured_plots()


def test_reattach_mid_run_and_restore():
    print("\n[15] Re-attach mid-run and restore: each figure/structure exactly once")
    import shutil
    import threading

    from biomentis.run_journal import RunJournal, entries_for_run, load_run
    from biomentis.run_worker import BackgroundRun
    from biomentis.structures import STRUCTURE_MARKER, default_artifact_dir
    from biomentis.ui_core import stream_agent_events

    old = os.environ.get("BIOMENTIS_ARTIFACT_DIR")
    os.environ["BIOMENTIS_ARTIFACT_DIR"] = tempfile.mkdtemp(prefix="step_plots_art_")
    try:
        pdb = os.path.join(default_artifact_dir("run-reattach"), "mini.pdb")
        os.makedirs(os.path.dirname(pdb), exist_ok=True)
        shutil.copyfile(os.path.join(os.path.dirname(__file__), "fixtures", "mini_xray.pdb"), pdb)
        observation = f"{STRUCTURE_MARKER}{pdb}\ndone\n"

        for mode, figure_kind in (("show", "plot"), ("student", "plot_withheld")):
            gate = threading.Event()
            agent = _agent(mode, [png], observation)

            def produce(agent=agent, gate=gate):
                # Hold the producer after the first file event, so the first
                # consumer is torn down while the run is still going.
                for event in stream_agent_events(agent, "Plot something.", [], [], thread_id=1):
                    yield event
                    if getattr(event, "file_kind", None):
                        gate.wait(5)

            journal = RunJournal("Plot something.", run_dir=tempfile.mkdtemp(prefix="step_plots_runs_"), plot_mode=mode)
            run = BackgroundRun("Plot something.", journal).start(produce)
            seen, cursor = [], 0
            for event in run.events_from(cursor):
                seen.append(event)
                cursor += 1
                if getattr(event, "file_kind", None):
                    break  # consumer torn down mid-run
            _check(f"{mode}: split while the run was alive", not run.done, "run already finished")
            gate.set()
            for event in run.events_from(cursor):  # re-attach at the cursor
                seen.append(event)
                cursor += 1
            kinds = [e.file_kind for e in seen if getattr(e, "file_kind", None)]
            _check(f"{mode}: live {figure_kind} once", kinds.count(figure_kind) == 1, f"{kinds}")
            _check(f"{mode}: live structure once", kinds.count("structure") == 1, f"{kinds}")
            _check(f"{mode}: no gaps or repeats", seen == run.events, f"{len(seen)} vs {len(run.events)}")
            if mode == "student":
                _check("student: no displayed plot live", "plot" not in kinds, f"{kinds}")

            entries = entries_for_run(load_run(journal.path))
            restored = [e["kind"] for e in entries if e.get("file_path")]
            _check(f"{mode}: restored {figure_kind} once", restored.count(figure_kind) == 1, f"{restored}")
            _check(f"{mode}: restored structure once", restored.count("structure") == 1, f"{restored}")
            if mode == "student":
                _check("student: no displayed plot restored", "plot" not in restored, f"{restored}")
    finally:
        if old is None:
            os.environ.pop("BIOMENTIS_ARTIFACT_DIR", None)
        else:
            os.environ["BIOMENTIS_ARTIFACT_DIR"] = old


def test_show_patch_message_is_neutral():
    print("\n[16] The patched plt.show prints a neutral message in both modes")
    src = inspect.getsource(__import__("biomentis.tool.support_tools", fromlist=["x"])._apply_matplotlib_patches)
    _check("says 'Plot captured.'", 'print("Plot captured.")' in src, "")
    _check("never claims the plot was displayed", "displayed" not in src.split("def savefig_with_capture")[0]
           .split("def show_with_capture")[1].replace("captured for the record but never displayed", ""), "")


def test_plot_saved_after_module_reimport():
    print("\n[17] A plot is saved after Streamlit re-imports support_tools")
    # Streamlit drops changed modules from sys.modules between reruns; the
    # agent in session_state keeps its old references. A 2026-10-28 run drew
    # five figures and saved none: the code ran in the old support_tools and
    # the plots were read from the new one.
    import importlib

    from langchain_core.messages import AIMessage

    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        print("  (matplotlib unavailable; skipping)")
        return
    from biomentis.agent.a1 import A1

    class _LLM:
        def __init__(self):
            self.replies = [
                "<execute>import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])\nplt.title('t')\nplt.show()</execute>",
                "<solution>done</solution>",
            ]

        def invoke(self, messages):
            return AIMessage(content=self.replies.pop(0))

    agent = A1(path="./data", use_tool_retriever=False, expected_data_lake_files=[], trace=False)
    agent.llm = _LLM()
    agent.configure()
    old = os.environ.get("BIOMENTIS_ARTIFACT_DIR")
    os.environ["BIOMENTIS_ARTIFACT_DIR"] = tempfile.mkdtemp(prefix="step_plots_reimport_")
    saved = sys.modules.pop("biomentis.tool.support_tools")
    try:
        importlib.import_module("biomentis.tool.support_tools")
        agent.go("Plot something.")
    finally:
        sys.modules["biomentis.tool.support_tools"] = saved
        if old is None:
            os.environ.pop("BIOMENTIS_ARTIFACT_DIR", None)
        else:
            os.environ["BIOMENTIS_ARTIFACT_DIR"] = old
    entries = [e for e in getattr(agent, "_execution_results", []) if e.get("image_paths")]
    _check("the figure was saved as a PNG", len(entries) == 1 and os.path.exists(entries[0]["image_paths"][0]),
           str(getattr(agent, "_execution_results", [])))


def test_patch_from_old_module_copy():
    print("\n[18] plt patched by an old support_tools copy still feeds the new one")
    import importlib

    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        print("  (matplotlib unavailable; skipping)")
        return
    from biomentis.tool import support_tools as first

    first.run_python_repl("import matplotlib.pyplot as plt\nkept = 41")  # patches plt via the first copy
    path = os.path.join(_TMP, "fig1_reloaded.png")
    saved = sys.modules.pop("biomentis.tool.support_tools")
    try:
        second = importlib.import_module("biomentis.tool.support_tools")
        _check("the reload made a new module", second is not first)
        second.clear_captured_plots()
        out = second.run_python_repl(
            f"plt.figure()\nplt.plot([1, 2])\nplt.title('Fig 1 — reloaded')\nplt.savefig({path!r})\nprint(kept + 1)"
        )
        plots = second.get_captured_plots()
        _check("the savefig figure reaches the new copy", len(plots) == 1, f"{len(plots)} / {out}")
        meta = second.get_plot_meta(plots[0]) if plots else {}
        _check("with its title and file name", meta == {"titles": ["Fig 1 — reloaded"], "saved_as": "fig1_reloaded.png"}, str(meta))
        _check("REPL variables survive the reload", "42" in out, out)
        second.clear_captured_plots()
        _check("clearing through either copy clears both", first.get_captured_plots() == [])
    finally:
        sys.modules["biomentis.tool.support_tools"] = saved


def _scripted(steps, answer, mode="show", run_id="run-figures"):
    """An agent whose steps are (observation, image_paths, image_meta), then a solution."""
    history: list = []
    agent = SimpleNamespace(use_tool_retriever=False, path=_TMP, user_task="", plot_mode=mode,
                            _execution_results=[], current_run_id=run_id)

    def _stream(inputs, stream_mode=None, config=None):
        history.extend(inputs["messages"])
        for observation, paths, meta in steps:
            history.append(AIMessage(content="<execute>x = 1</execute>"))
            yield {"messages": list(history)}
            agent._execution_results.append({"images": [URI] * len(paths), "image_paths": paths, "image_meta": meta})
            history.append(AIMessage(content=f"<observation>{observation}</observation>"))
            yield {"messages": list(history)}
        history.append(AIMessage(content=f"<solution>{answer}</solution>"))
        yield {"messages": list(history)}

    agent.app = SimpleNamespace(stream=_stream)
    from biomentis.ui_core import stream_agent_events

    return list(stream_agent_events(agent, "Make figures.", [], [], thread_id=1))


def _png_bytes(seed: int) -> bytes:
    import struct
    import zlib

    raw = b"\x00" + bytes([seed % 256, 0, 0])
    chunk = lambda kind, data: struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def test_reported_image_is_copied():
    print("\n[19] A reported image is shown from a copy; an unchanged one is not repeated")
    from biomentis.structures import default_artifact_dir

    agent_file = os.path.join(_TMP, "fig2_copy.png")
    with open(agent_file, "wb") as f:
        f.write(_png_bytes(1))
    first = f"saved {agent_file}\n"

    def later_delete():
        # The agent deletes its file after the step: the copy must not care.
        os.remove(agent_file)
        return True

    ev = _scripted([(first, [], []), (f"listing: {agent_file}\n", [], [])], "done", run_id="run-copy")
    shown = [e.file_path for e in ev if e.type == "file" and e.file_path]
    _check("shown once, not again for the unchanged re-listing", len(shown) == 1, str(shown))
    _check("from a copy in the run folder",
           shown and shown[0].startswith(default_artifact_dir("run-copy")) and shown[0].endswith("image-001-fig2_copy.png"), str(shown))
    later_delete()
    _check("the copy outlives the agent's file", shown and os.path.isfile(shown[0]))

    lost = _scripted([("Plot saved to: never_written.png\n", [], [])], "done", run_id="run-lost")
    quality = next((e.content for e in lost if e.title == "📊 Run quality"), "")
    _check("a savefig figure that was neither captured nor found counts as lost", "Figures lost" in quality, quality)


def test_answer_figures_in_main_panel():
    print("\n[20] The figures the answer cites appear under it, latest version")
    from biomentis.ui_core import transcript_entry_for_event
    from biomentis.ui_tutor import _is_instruction_bearing, _is_step_visual

    v1, v2 = _write_png("plot-101.png"), _write_png("plot-102.png")
    fig2 = os.path.join(_TMP, "fig2_risk.png")
    with open(fig2, "wb") as f:
        f.write(_png_bytes(2))
    steps = [
        ("Plot saved to: fig1_flow.png\n", [v1], [{"titles": ["Fig 1 — draft"], "saved_as": "fig1_flow.png"}]),
        ("Plot saved to: fig1_flow.png\n", [v2], [{"titles": ["Fig 1 — Workflow"], "saved_as": "fig1_flow.png"}]),
        (f"wrote {fig2}\n", [], []),
    ]
    answer = "See Fig 1 for the plan and fig2_risk.png for the tiers; Fig 3 shows the budget."
    ev = _scripted(steps, answer, run_id="run-answer")
    main_files = [e for e in ev if e.type == "file" and e.channel == "main"]
    _check("two figures under the answer", [e.title for e in main_files] == ["Fig 1", "fig2_risk.png"],
           str([(e.title, e.file_path) for e in main_files]))
    _check("Fig 1 is the latest version", main_files and main_files[0].file_path == v2, str(main_files[:1]))
    _check("captioned with title and file name",
           main_files and main_files[0].caption == "Fig 1 — Workflow — fig1_flow.png", str(main_files[:1] and main_files[0].caption))
    heading = next((e for e in ev if e.title == "🖼️ Figures"), None)
    _check("a heading names the citation with no figure",
           heading is not None and heading.channel == "main" and "Fig 3" in heading.content, str(heading))
    order = [e.title for e in ev if e.channel == "main"]
    _check("they come after the answer and before the run quality",
           order.index("🖼️ Figures") > order.index("✅ Answer") and order.index("🖼️ Figures") < order.index("📊 Run quality"), str(order))
    entry = transcript_entry_for_event(main_files[0]) if main_files else {}
    _check("its transcript entry is in the main panel with the caption",
           entry.get("panel") == "main" and entry.get("caption") == "Fig 1 — Workflow — fig1_flow.png", str(entry))
    _check("the tutor neither pauses on it nor files it under a step",
           main_files and not _is_instruction_bearing(main_files[0]) and not _is_step_visual(main_files[0]))
    student = _scripted(steps, answer, mode="student", run_id="run-answer-student")
    kinds = [e.file_kind for e in student if e.type == "file" and e.channel == "main"]
    _check("withheld under 'student'", kinds == ["plot_withheld", "plot_withheld"], str(kinds))
    # The agent redraws Fig 1 under a new name, then lists the folder, which
    # still holds the old file: the old drawing must not become "latest".
    old_fig, new_fig = os.path.join(_TMP, "fig1_old.png"), os.path.join(_TMP, "fig1_new.png")
    for path, seed in ((old_fig, 4), (new_fig, 5)):
        with open(path, "wb") as f:
            f.write(_png_bytes(seed))
    relisted = _scripted([(f"wrote {old_fig}\n", [], []), (f"wrote {new_fig}\n", [], []),
                          (f"{old_fig}\n{new_fig}\n", [], [])], "See Fig 1.", run_id="run-answer-relist")
    fig1 = [e.file_path for e in relisted if e.type == "file" and e.channel == "main"]
    _check("a re-listed old file does not replace the redrawn figure",
           len(fig1) == 1 and fig1[0].endswith("fig1_new.png"), str(fig1))
    none = _scripted(steps, "No figures cited here.", run_id="run-answer-none")
    _check("no section when the answer cites none", not any(e.title == "🖼️ Figures" for e in none))


def test_export_run_folder():
    print("\n[21] The export is one folder: two pages and figures/, nothing embedded")
    import re

    from biomentis.ui_core import ExportEntry, export_run

    a, b = _write_png("plot-201.png"), os.path.join(_TMP, "sub", "plot-201.png")
    os.makedirs(os.path.dirname(b), exist_ok=True)
    with open(b, "wb") as f:
        f.write(_png_bytes(3))
    main = [ExportEntry(role="assistant", content="Answer"),
            ExportEntry(role="assistant", kind="image", title="Fig 1", file_path=a, caption="Fig 1 — Workflow")]
    inner = [ExportEntry(role="assistant", kind="image", file_path=a),
             ExportEntry(role="assistant", kind="image", file_path=b),
             ExportEntry(role="assistant", kind="image", file_path=os.path.join(_TMP, "gone.png"))]
    out = export_run(main, inner, "m", downloads_dir=tempfile.mkdtemp(prefix="export_"))
    stamp = os.path.basename(out).removeprefix("biomentis_")
    files = sorted(os.listdir(out))
    _check("folder holds both pages and figures/",
           files == sorted([f"biomentis_inner_{stamp}.html", f"biomentis_main_{stamp}.html", "figures"]), str(files))
    pages = {name: open(os.path.join(out, name), encoding="utf-8").read() for name in files if name.endswith(".html")}
    _check("nothing embedded", not any("data:image" in page for page in pages.values()))
    srcs = sorted({src for page in pages.values() for src in re.findall(r'<img src="([^"]+)"', page)})
    _check("images linked from figures/, one copy each, names kept distinct",
           srcs == ["figures/plot-201-2.png", "figures/plot-201.png"] and sorted(os.listdir(os.path.join(out, "figures"))) == ["plot-201-2.png", "plot-201.png"],
           str(srcs))
    main_page = pages[f"biomentis_main_{stamp}.html"]
    _check("caption shown under the figure", "<figcaption" in main_page and "Fig 1 — Workflow" in main_page)
    _check("each page links the other",
           f'href="biomentis_inner_{stamp}.html"' in main_page and f'href="biomentis_main_{stamp}.html"' in pages[f"biomentis_inner_{stamp}.html"])
    _check("a missing image says so", "Image not found" in pages[f"biomentis_inner_{stamp}.html"])
    again = export_run(main, inner, "m", downloads_dir=os.path.dirname(out))
    _check("a second export in the same second gets its own folder", again != out and os.path.isdir(again), again)


def test_cited_figures_matching():
    print("\n[22] Matching citations to figures")
    from biomentis.run_figures import RunFigure, cited_figures, cited_numbers

    _check("lists and ranges", cited_numbers("Figs 2, 3 and 5; Figures 7–9; Fig. 11a") == [2, 3, 5, 7, 8, 9, 11],
           str(cited_numbers("Figs 2, 3 and 5; Figures 7–9; Fig. 11a")))
    _check("code blocks ignored", cited_numbers("```\nplt.title('Fig 4')\n```\nSee Fig 1.") == [1])
    figures = [RunFigure("/a.png", ["Fig 5a — old"]), RunFigure("/b.png", [], "fig5_method.png"),
               RunFigure("/c.png", ["Fig 5a — new", "Fig 5b — new"]), RunFigure("/d.png", [], "fig6_cov.png")]
    chosen, missing = cited_figures("Fig 5 and Fig 6, also Fig 6 again.", figures)
    _check("latest titled figure wins; file name used without a title",
           [(label, f.path) for label, f in chosen] == [("Fig 5", "/c.png"), ("Fig 6", "/d.png")] and missing == [],
           str([(label, f.path) for label, f in chosen]))
    _check("a title beats a file-name number", RunFigure("/e.png", ["Figure 2: x"], "fig9.png").numbers() == {2})


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
        test_failing_step_owns_its_figure,
        test_reattach_mid_run_and_restore,
        test_show_patch_message_is_neutral,
        test_plot_saved_after_module_reimport,
        test_patch_from_old_module_copy,
        test_reported_image_is_copied,
        test_answer_figures_in_main_panel,
        test_export_run_folder,
        test_cited_figures_matching,
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
