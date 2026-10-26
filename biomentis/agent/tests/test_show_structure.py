"""Tests for the agent's plot guidance, `show_structure`, and uploaded structures.

The agent is told how to plot (self-contained blocks a student can paste into
Colab) and how to show a molecule (`show_structure`). `show_structure` prints
a marker line; `stream_agent_events` turns that line into a `structure` file
event, journaled by the producer like a plot, so re-attach and restore work.
A structure file uploaded with the turn is opened the same way, before the
agent starts. The structure viewer is not governed by the plot setting.

What these tests pin down:

  1.  plot_guidance: the self-contained rule, show_structure, per-mode wording
  2.  generate appends plot_guidance per call, from self.plot_mode
  3.  show_structure("6NK5") downloads into the current run's dir and prints the marker
  4.  SMILES is validated and registered; bad input prints no marker
  5.  A SMILES like CCCC is not mistaken for a PDB ID (no network call)
  6.  A .pdb/.sdf/.smi path is imported into the run's dir
  7.  The download tools register into the current run
  8.  A marker line in an observation becomes a structure event
  9.  Under "student" a structure is still shown (not a plot)
  10. An uploaded PDB yields a structure event before the agent runs
  11. transcript entry, export kind, and export HTML (metadata, no coordinates)
  12. render_entry draws a structure with render_structure
  13. The tool description lists show_structure with one required `source`
  14. The execute node clears the run id after each execution
  15. Tutor: an uploaded structure starts no step; the roadmap comes from the plan
  16. Tutor: plots and structures join the current step with no card or pause
  17. Tutor: a step's figure is yielded before its observation, so a paused
      student sees it (stream, tutor and journal order)

Run with:
    python -m biomentis.agent.tests.test_show_structure
"""

from __future__ import annotations

import contextlib
import inspect
import io
import os
import shutil
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

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
XRAY = os.path.join(FIXTURES, "mini_xray.pdb")
ASPIRIN_SMI = os.path.join(FIXTURES, "aspirin.smi")
RUN = "run-show-structure"

_ROOT = tempfile.mkdtemp(prefix="show_structure_art_")
_UPLOADS = tempfile.mkdtemp(prefix="show_structure_up_")
XRAY_COPY = os.path.join(_UPLOADS, "my_upload.pdb")
shutil.copyfile(XRAY, XRAY_COPY)

os.environ["BIOMENTIS_ARTIFACT_DIR"] = _ROOT


class _FakeResponse:
    def __init__(self, content: bytes = b"", payload: dict | None = None, status: int = 200):
        self.content = content
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.text = content.decode("utf-8", "replace")

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


_REQUESTED: list[str] = []


@contextlib.contextmanager
def _fake_network():
    """Serve mini_xray.pdb for any RCSB download; record every URL."""
    from biomentis.tool import database as db

    with open(XRAY, "rb") as handle:
        pdb_bytes = handle.read()

    def handler(url, *a, **k):
        _REQUESTED.append(url)
        if "files.rcsb.org/download/" in url or "alphafold.ebi.ac.uk/files/" in url:
            return _FakeResponse(content=pdb_bytes)
        return _FakeResponse(payload={"struct": {"title": "Test entry"}})

    prev = db.requests
    db.requests = type("_R", (), {"get": staticmethod(handler), "exceptions": prev.exceptions})
    try:
        yield
    finally:
        db.requests = prev


@contextlib.contextmanager
def _in_run(run_id):
    from biomentis.structures import set_current_run_id

    set_current_run_id(run_id)
    try:
        yield
    finally:
        set_current_run_id(None)


def _capture_stdout(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn()
    return buf.getvalue(), result


def _marker_path(out: str) -> str | None:
    from biomentis.structures import STRUCTURE_MARKER

    for line in out.splitlines():
        if line.startswith(STRUCTURE_MARKER):
            return line[len(STRUCTURE_MARKER) :].strip()
    return None


def _events(
    observation: str = "done\n", files: list[str] | None = None, mode: str = "show", image_paths: list[str] | None = None
) -> list:
    """Run a scripted one-step turn through stream_agent_events."""
    from biomentis.ui_core import stream_agent_events

    history: list = []
    agent = SimpleNamespace(
        use_tool_retriever=False,
        path=_UPLOADS,
        user_task="",
        plot_mode=mode,
        current_run_id=RUN,
        _execution_results=[],
    )

    def _stream(inputs, stream_mode=None, config=None):
        history.extend(inputs["messages"])
        history.append(AIMessage(content="Show it.\n<execute>show_structure('6NK5')</execute>"))
        yield {"messages": list(history)}
        paths = list(image_paths or [])
        agent._execution_results.append({"images": ["data:image/png;base64,"] * len(paths), "image_paths": paths})
        history.append(AIMessage(content=f"<observation>{observation}</observation>"))
        yield {"messages": list(history)}

    agent.app = SimpleNamespace(stream=_stream)
    return list(stream_agent_events(agent, "Show a structure.", list(files or []), [], thread_id=1))


# --- Tests ----------------------------------------------------------------


def test_guidance_text():
    print("\n[1] plot_guidance wording")
    from biomentis.plots import plot_guidance

    for m in ("show", "student"):
        g = plot_guidance(m)
        _check(
            f"{m}: self-contained rule",
            "inline" in g and "numpy" in g and "matplotlib" in g and "earlier steps" in g,
            g,
        )
        _check(f"{m}: names show_structure", "show_structure" in g, g)
        _check(f"{m}: plt.show and units", "plt.show()" in g and "units" in g, g)
        _check(f"{m}: finish a figure within one step", "one step" in g, g)
        _check(f"{m}: no biomentis tools in plots", "biomentis tools" in g, g)
    _check("student wording", "will not see the figure" in plot_guidance("student"), plot_guidance("student"))
    _check("student: complete and commented", "commented" in plot_guidance("student"), "")
    _check("show wording", "beside the code" in plot_guidance("show"), plot_guidance("show"))
    _check("show does not use student wording", "will not see the figure" not in plot_guidance("show"), "")


def test_generate_appends_guidance():
    print("\n[2] generate appends plot_guidance per call")
    from biomentis.agent.a1 import A1

    src = inspect.getsource(A1.configure)
    gen = src[src.find("def generate(state") : src.find("def execute(state")]
    _check(
        "generate appends plot_guidance(self.plot_mode)",
        'plot_guidance(getattr(self, "plot_mode", "show"))' in gen,
        "",
    )
    _check("not baked into self.system_prompt", "self.system_prompt += \"\\n\\n\" + plot_guidance" not in src, "")


def test_show_structure_by_id():
    print("\n[3] show_structure('6NK5') downloads into the run dir")
    from biomentis.structures import default_artifact_dir, read_registry
    from biomentis.tool.support_tools import show_structure

    with _fake_network(), _in_run(RUN):
        out, summary = _capture_stdout(lambda: show_structure("6NK5"))
    path = _marker_path(out)
    _check("marker printed", bool(path) and os.path.exists(path), out)
    _check("absolute path", bool(path) and os.path.isabs(path), str(path))
    _check(
        "lands in the run dir",
        bool(path) and os.path.dirname(path) == default_artifact_dir(RUN),
        f"{path} vs {default_artifact_dir(RUN)}",
    )
    _check("registered in the run", any(a.accession == "6NK5" for a in read_registry(default_artifact_dir(RUN))), "")
    _check("summary is one line with the title", "\n" not in summary and "Mini test protein" in summary, summary)


def test_show_structure_smiles_and_bad():
    print("\n[4] SMILES is registered; bad input prints no marker")
    from biomentis.structures import STRUCTURE_MARKER, default_artifact_dir, read_registry
    from biomentis.tool.support_tools import show_structure

    with _in_run(RUN):
        out, summary = _capture_stdout(lambda: show_structure("CC(=O)Oc1ccccc1C(=O)O"))
    path = _marker_path(out) or ""
    _check("SMILES registered", path.endswith(".smi") and os.path.exists(path), out)
    _check("in the run dir", os.path.dirname(path) == default_artifact_dir(RUN), path)
    _check(
        "source_db smiles",
        any(a.source_db == "smiles" for a in read_registry(default_artifact_dir(RUN))),
        str(read_registry(default_artifact_dir(RUN))),
    )
    _check("summary has the formula", "C9H8O4" in summary, summary)

    with _in_run(RUN):
        out, summary = _capture_stdout(lambda: show_structure("not a molecule ((("))
    _check("bad input: no marker", STRUCTURE_MARKER not in out, out)
    _check("bad input: reason returned", summary.startswith("Could not show structure: "), summary)

    out, summary = _capture_stdout(lambda: show_structure("/nonexistent/thing.pdb"))
    _check("missing file: no marker", STRUCTURE_MARKER not in out, out)
    _check("missing file: reason", summary.startswith("Could not show structure: "), summary)


def test_smiles_not_mistaken_for_pdb_id():
    print("\n[5] CCCC is SMILES, not a PDB ID")
    from biomentis.tool.support_tools import show_structure

    _REQUESTED.clear()
    with _fake_network(), _in_run(RUN):
        out, _ = _capture_stdout(lambda: show_structure("CCCC"))
    _check("no network call", _REQUESTED == [], str(_REQUESTED))
    _check("written as .smi", (_marker_path(out) or "").endswith(".smi"), out)


def test_show_structure_from_file():
    print("\n[6] A structure file is imported into the run dir")
    from biomentis.structures import default_artifact_dir, read_registry
    from biomentis.tool.support_tools import show_structure

    with _in_run(RUN):
        out, _ = _capture_stdout(lambda: show_structure(ASPIRIN_SMI))
    path = _marker_path(out) or ""
    _check("imported", os.path.basename(path) == "aspirin.smi" and os.path.exists(path), out)
    _check("in the run dir", os.path.dirname(path) == default_artifact_dir(RUN), path)
    _check(
        "registered as agent",
        any(a.source_db == "agent" and a.accession == "aspirin" for a in read_registry(default_artifact_dir(RUN))),
        "",
    )


def test_download_tools_use_current_run():
    print("\n[7] query_pdb_identifiers / query_alphafold register into the current run")
    from biomentis.structures import default_artifact_dir
    from biomentis.tool import database as db

    run = "run-download-tools"
    with _fake_network(), _in_run(run), contextlib.redirect_stdout(io.StringIO()):
        pdb = db.query_pdb_identifiers(["1ABC"], download=True)
        af = db.query_alphafold("P04637", download=True)
    run_dir = default_artifact_dir(run)
    _check("pdb file in run dir", os.path.exists(os.path.join(run_dir, "1ABC.pdb")), str(pdb))
    _check("pdb artifact run_id", [a.get("run_id") for a in pdb.get("artifacts", [])] == [run], str(pdb.get("artifacts")))
    _check("alphafold artifact run_id", [a.get("run_id") for a in af.get("artifacts", [])] == [run], str(af.get("artifacts")))
    _check(
        "alphafold file in run dir",
        os.path.exists(os.path.join(run_dir, "AF-P04637-F1-model_v4.pdb")),
        str(os.listdir(run_dir)),
    )


def test_marker_becomes_event():
    print("\n[8] A marker line in an observation becomes a structure event")
    from biomentis.structures import STRUCTURE_MARKER, default_artifact_dir

    path = os.path.join(default_artifact_dir(RUN), "6NK5.pdb")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    shutil.copyfile(XRAY, path)
    ev = _events(observation=f"Downloaded structure: {path}\n{STRUCTURE_MARKER}{path}\nShowing 6NK5\n")
    files = [e for e in ev if e.file_path]
    _check("structure event", [e.file_kind for e in files] == ["structure"], f"{[(e.file_kind, e.file_path) for e in files]}")
    _check("titled with the basename", files and files[0].title == "🧬 6NK5.pdb", f"{files[0].title if files else None}")
    _check("absolute path", files and files[0].file_path == os.path.abspath(path), "")
    obs_i = next(i for i, e in enumerate(ev) if e.type == "observation")
    st_i = next(i for i, e in enumerate(ev) if e.file_kind == "structure")
    _check("before its observation (a paused student sees it)", st_i < obs_i, f"{obs_i} {st_i}")

    ev = _events(observation=f"{STRUCTURE_MARKER}{path}\n{STRUCTURE_MARKER}{path}\n")
    _check("same structure twice in one step: one event", len([e for e in ev if e.file_path]) == 1, "")
    ev = _events(observation=f"{STRUCTURE_MARKER}/nonexistent/x.pdb\n")
    _check("missing file: no event", [e for e in ev if e.file_path] == [], "")


def test_student_mode_still_shows_structure():
    print("\n[9] Under 'student' a structure is still shown")
    from biomentis.structures import STRUCTURE_MARKER, default_artifact_dir

    path = os.path.join(default_artifact_dir(RUN), "6NK5.pdb")
    ev = _events(observation=f"{STRUCTURE_MARKER}{path}\n", mode="student")
    _check("structure, not withheld", [e.file_kind for e in ev if e.file_path] == ["structure"], "")


def test_uploaded_pdb_opens():
    print("\n[10] An uploaded PDB opens before the agent runs")
    from biomentis.structures import default_artifact_dir, read_registry

    ev = _events(files=[XRAY_COPY])
    first = next((e for e in ev if e.file_path), None)
    _check("upload yields a structure event first", first is not None and first.file_kind == "structure", f"{first}")
    _check(
        "imported into the run dir",
        first is not None and os.path.dirname(first.file_path) == default_artifact_dir(RUN),
        f"{first.file_path if first else None}",
    )
    code_i = next(i for i, e in enumerate(ev) if e.type == "code")
    _check("before the agent's first step", ev.index(first) < code_i, "")
    _check(
        "registered as upload",
        any(a.source_db == "upload" and a.accession == "my_upload" for a in read_registry(default_artifact_dir(RUN))),
        "",
    )
    png_dir = tempfile.mkdtemp(prefix="show_structure_png_")
    png = os.path.join(png_dir, "fig.png")
    open(png, "wb").close()
    ev = _events(files=[png])
    _check("a non-structure upload yields no structure event", [e for e in ev if e.file_kind == "structure"] == [], "")


def test_entry_and_export():
    print("\n[11] transcript entry, export kind, export HTML")
    from biomentis.structures import default_artifact_dir
    from biomentis.ui_core import (
        ExportEntry,
        UIEvent,
        export_kind,
        render_transcript_html,
        transcript_entry_for_event,
        ui_event_to_export_entry,
    )

    path = os.path.join(default_artifact_dir(RUN), "6NK5.pdb")
    event = UIEvent(type="file", file_kind="structure", title="🧬 6NK5.pdb", file_path=path)
    entry = transcript_entry_for_event(event)
    _check(
        "entry shape",
        entry.get("kind") == "structure" and entry.get("file_path") == path and entry.get("panel") == "inner"
        and entry.get("title") == "🧬 6NK5.pdb",
        f"{entry}",
    )
    _check("export kind kept", export_kind("structure") == "structure", export_kind("structure"))
    _check("event exports as structure", ui_event_to_export_entry(event).kind == "structure", "")

    structure_entry = ExportEntry(role="assistant", kind="structure", title="🧬 6NK5.pdb", file_path=path)
    html = render_transcript_html([structure_entry], "Executor", "m")
    _check("metadata in export", "Resolution" in html and "1.80" in html, html[-800:])
    _check("no ATOM lines in export", "ATOM  " not in html, "")
    _check("no viewer in export", "3Dmol" not in html and "<img" not in html, "")

    missing = ExportEntry(role="assistant", kind="structure", title="🧬 gone.pdb", file_path="/nonexistent/gone.pdb")
    html = render_transcript_html([missing], "Executor", "m")
    _check("missing file exports without crashing", "gone.pdb" in html, "")


def test_render_entry_branch():
    print("\n[12] render_entry draws a structure with render_structure")
    from biomentis.agent.a1 import A1

    launch = inspect.getsource(A1.launch_streamlit_demo)
    start = launch.find("def render_entry(")
    body = launch[start : launch.find("class _StepBoxTracker", start)]
    branch = body[body.find('== "structure"') :]
    branch = branch[: branch.find("elif")]
    _check("structure branch exists", '== "structure"' in body, "")
    _check("calls render_structure", "render_structure(" in branch, branch)

    gradio = inspect.getsource(A1.launch_gradio_demo)
    _check("gradio does not treat a structure as an image", '"structure"' in gradio, "")


def test_tool_description():
    print("\n[13] show_structure is described")
    from biomentis.tool.tool_description.support_tools import description

    entry = next((d for d in description if d["name"] == "show_structure"), None)
    _check("described", entry is not None, "")
    if entry:
        req = entry["required_parameters"]
        _check("one required param 'source'", [p["name"] for p in req] == ["source"], f"{req}")
        _check(
            "param text",
            req[0]["description"] == "a PDB ID (e.g. 6NK5), a path to a .pdb/.sdf/.smi file, or a SMILES string",
            req[0]["description"],
        )
        _check("type str", req[0]["type"] == "str", "")

    from biomentis.tool.tool_imports import tool_name_to_module

    _check("import repair knows it", tool_name_to_module().get("show_structure") == "biomentis.tool.support_tools", "")


def test_execute_resets_run_id():
    print("\n[14] The execute node clears the run id after each execution")
    from biomentis.agent.a1 import A1

    src = inspect.getsource(A1.configure)
    execute = src[src.find("def execute(state") : src.find("def routing_function")]
    i_try = execute.find("try:", execute.find("set_current_run_id(self.current_run_id)"))
    i_finally = execute.find("finally:", i_try)
    i_reset = execute.find("set_current_run_id(None)", i_finally)
    i_run = execute.find("run_with_timeout(", i_try)
    _check("reset in a finally after running code", 0 <= i_try < i_run < i_finally < i_reset, f"{i_try} {i_run} {i_finally} {i_reset}")


# --- Tutor path -------------------------------------------------------------


def _drive_tutor(events: list) -> tuple[list, list, list]:
    """Run `_advance_run_live` to exhaustion over scripted events.

    Returns (yielded events, roadmap source events, card source events). The
    LLM-backed roadmap/card builders are replaced by recorders.
    """
    from biomentis import ui_tutor

    roadmaps: list = []
    cards: list = []
    saved = (
        ui_tutor._generate_roadmap_for_run,
        ui_tutor._generate_or_get_card,
        ui_tutor._log_event_seen,
        ui_tutor._log_advance,
        ui_tutor._record_run_snapshot,
    )
    ui_tutor._generate_roadmap_for_run = lambda ev, run, tutor: roadmaps.append(ev) or SimpleNamespace()
    ui_tutor._generate_or_get_card = lambda ev, run, tutor: cards.append(ev) or None
    ui_tutor._log_event_seen = lambda *a, **k: None
    ui_tutor._log_advance = lambda *a, **k: None
    ui_tutor._record_run_snapshot = lambda *a, **k: None
    saved_roadmap_event = ui_tutor._make_roadmap_event
    ui_tutor._make_roadmap_event = lambda roadmap: SimpleNamespace(type="roadmap")
    run = {"gen": iter(events), "events": [], "run_id": "r1", "step_id": 0, "paused_at": None, "roadmap_shown": False}
    out: list = []
    try:
        while run.get("phase") != "done":
            out.extend(ui_tutor._advance_run_live(run, tutor=SimpleNamespace()))
    finally:
        (
            ui_tutor._generate_roadmap_for_run,
            ui_tutor._generate_or_get_card,
            ui_tutor._log_event_seen,
            ui_tutor._log_advance,
            ui_tutor._record_run_snapshot,
        ) = saved
        ui_tutor._make_roadmap_event = saved_roadmap_event
    return out, roadmaps, cards


def test_tutor_upload_does_not_start_a_step():
    print("\n[15] Tutor: an uploaded structure is not a step; the roadmap comes from the plan")
    from biomentis.ui_core import UIEvent

    upload = UIEvent(type="file", file_kind="structure", title="🧬 my_upload.pdb", file_path=XRAY_COPY)
    plan = UIEvent(type="reasoning", title="🤔 Reasoning", content="1. Fetch the structure\n2. Inspect it")
    out, roadmaps, cards = _drive_tutor([upload, plan])
    _check("roadmap built from the plan", roadmaps == [plan], f"{[getattr(e, 'title', None) for e in roadmaps]}")
    _check("no card for the upload", upload not in cards and cards == [plan], f"{[getattr(e, 'title', None) for e in cards]}")
    pauses = [e for e in out if getattr(e, "type", None) == "paused"]
    _check("one pause, for the plan only", len(pauses) == 1, f"{len(pauses)}")
    _check("upload yielded before the pause", out.index(upload) < out.index(pauses[0]) if pauses else False, "")
    _check("upload is ungrouped (no step)", getattr(upload, "step_id", None) is None, f"{getattr(upload, 'step_id', None)}")
    _check("plan is step 1", getattr(plan, "step_id", None) == 1, f"{getattr(plan, 'step_id', None)}")


def test_tutor_plot_joins_its_step():
    print("\n[16] Tutor: a plot / structure joins the current step with no card or pause")
    from biomentis.ui_core import UIEvent

    code = UIEvent(type="code", content="plt.plot([1, 2])", language="python", title="🛠️ Executing code...")
    plot = UIEvent(type="file", file_kind="plot", title="📈 Plot", file_path="/tmp/plot-001.png")
    held = UIEvent(type="file", file_kind="plot_withheld", title="📈 Plot", file_path="/tmp/plot-002.png")
    mol = UIEvent(type="file", file_kind="structure", title="🧬 6NK5.pdb", file_path=XRAY)
    image = UIEvent(type="file", file_kind="image", title="🖼️ Image Preview", file_path="/tmp/x.png")
    out, roadmaps, cards = _drive_tutor([code, plot, held, mol, image])
    _check("plot shares the code step's step_id", getattr(plot, "step_id", None) == getattr(code, "step_id", "x") == 1, "")
    _check("withheld plot and structure too", held.step_id == 1 and mol.step_id == 1, f"{held.step_id} {mol.step_id}")
    _check("run_id carried", plot.run_id == "r1" and mol.run_id == "r1", "")
    _check("no card for visuals", not any(e in cards for e in (plot, held, mol)), f"{[e.title for e in cards]}")
    _check("a plain image still gets its own step and card", image.step_id == 2 and image in cards, f"{getattr(image, 'step_id', None)}")
    pauses = [e for e in out if getattr(e, "type", None) == "paused"]
    _check("two pauses: code, image", len(pauses) == 2, f"{len(pauses)}")


def test_tutor_pause_on_observation_shows_the_figure():
    print("\n[17] Tutor: pausing on the observation, the step's plot is already shown")
    from biomentis.structures import STRUCTURE_MARKER, default_artifact_dir

    png = os.path.join(_UPLOADS, "plot-001.png")
    open(png, "wb").close()
    mol = os.path.join(default_artifact_dir(RUN), "6NK5.pdb")
    stream = _events(observation=f"{STRUCTURE_MARKER}{mol}\n", image_paths=[png])
    order = [e.type if not e.file_path else e.file_kind for e in stream]
    _check(
        "stream order: code, plot, structure, observation",
        [k for k in order if k in ("code", "plot", "structure", "observation")] == ["code", "plot", "structure", "observation"],
        f"{order}",
    )
    out, _, cards = _drive_tutor(stream)
    code = next(e for e in out if getattr(e, "type", None) == "code")
    plot = next(e for e in out if getattr(e, "file_kind", None) == "plot")
    obs = next(e for e in out if getattr(e, "type", None) == "observation")
    obs_pause = next(
        i for i, e in enumerate(out) if getattr(e, "type", None) == "paused" and i > out.index(obs)
    )
    _check("plot yielded before the observation's pause", out.index(plot) < out.index(obs) < obs_pause, "")
    _check("plot sits in the code step's box", plot.step_id == code.step_id, f"{plot.step_id} {code.step_id}")
    _check("no card for the plot", plot not in cards, "")

    from biomentis.run_journal import RunJournal, entries_for_run, load_run

    journal = RunJournal("Plot.", run_dir=tempfile.mkdtemp(prefix="show_structure_runs_"), plot_mode="show")
    for event in out:
        if hasattr(event, "content") and not isinstance(event, SimpleNamespace):
            journal.append(event)
    journal.finish("complete")
    kinds = [e.get("kind") for e in entries_for_run(load_run(journal.path))]
    _check(
        "journal order: plot and structure before the observation",
        kinds.index("plot") < kinds.index("observation") and kinds.index("structure") < kinds.index("observation"),
        f"{kinds}",
    )


def run_all() -> int:
    print("=" * 60)
    print("show_structure / plot guidance tests")
    print("=" * 60)
    for test in (
        test_guidance_text,
        test_generate_appends_guidance,
        test_show_structure_by_id,
        test_show_structure_smiles_and_bad,
        test_smiles_not_mistaken_for_pdb_id,
        test_show_structure_from_file,
        test_download_tools_use_current_run,
        test_marker_becomes_event,
        test_student_mode_still_shows_structure,
        test_uploaded_pdb_opens,
        test_entry_and_export,
        test_render_entry_branch,
        test_tool_description,
        test_execute_resets_run_id,
        test_tutor_upload_does_not_start_a_step,
        test_tutor_plot_joins_its_step,
        test_tutor_pause_on_observation_shows_the_figure,
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
