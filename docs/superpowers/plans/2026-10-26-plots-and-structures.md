# Plots and Structure Viewing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Plots drawn by the agent and the tutor appear in the app (or are withheld, per a class setting), and PDB/SDF/SMILES structures open in an interactive viewer with a metadata panel, from uploads, the knowledge base, or an agent fetch.

**Architecture:** Structure files become registered artifacts in `runs/<run_id>/` (the September artifact layer). Plots and structures reach the transcript as ordinary journaled `UIEvent(type="file")` events with new `file_kind` values, so re-attach and restore work unchanged and the plot setting is baked into each event by the producer — which is what makes "a run displays the way it was recorded" true by construction. The viewer is 3Dmol.js in `st.components.v1.html`; metadata comes from Biopython and RDKit.

**Tech Stack:** Python 3.11, Streamlit 1.59.2 (built-ins only), matplotlib (Agg), Biopython 1.87, RDKit (pip `rdkit`), 3Dmol.js 2.4.2 from cdnjs.

**Spec:** `docs/superpowers/specs/2026-10-26-visualization-design.md` — and, for Task 1 only, `docs/superpowers/plans/2026-09-03-structure-artifact-layer.md`.

## Global Constraints

- Interpreter: `/opt/anaconda3/envs/biomni/bin/python`, with `PYTHONPATH=/Users/dylanbulseco/Documents/Projects/Biomni`. Base `python3` will not import the repo.
- **No pytest.** Tests are modules run with `python -m biomentis.agent.tests.<name>`, using the `_check(name, cond, detail)` harness (copy it from `biomentis/agent/tests/test_tutor_card_cache.py`). Every new suite goes into the CLAUDE.md test list.
- **No live network in tests.** Monkeypatch `biomentis.tool.database.requests` (pattern: September plan Task 4).
- Formats in scope: **PDB, SDF, SMILES** (`.pdb`, `.sdf`, `.smi`). mmCIF, MOL2, FASTA are out.
- Plot setting values: `"show"` (*Show plots*) and `"student"` (*Students run the code*). Env var `BIOMENTIS_PLOT_MODE=show|student`; anything else means `"show"`.
- Code is always visible, in both settings. Under `"student"` no figure is ever displayed in the app — in a step or in the chat.
- A run displays the way it was recorded; changing the sidebar affects the next run / next question only.
- Figures appear only in the step that produced them, never beside the final answer.
- Coordinate records (`ATOM`/`HETATM` lines, SDF atom/bond blocks) are never shown as text anywhere — metadata panel, transcript, KB snippets, exports.
- No Streamlit community components. No `st.*` calls from code that runs on the REPL thread (it has no script context).
- Run-loop invariants (CLAUDE.md): every new event is produced by `stream_agent_events` and journaled by the producer; a torn-down consumer re-attaches without gaps or repeats. `self_critic=True` may emit several `<solution>` blocks — nothing here touches solutions.
- 3Dmol.js URL: `https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js` (verified 200 on 2026-10-26).
- Package is `biomentis`; a `No module named 'biomni'` error is a stray legacy import.

## Review Focus

1. **The agent saves a figure itself and prints its path under `"student"`.** The existing stdout scraper (`ui_core.py:270-301`) would show that PNG — a leak. Every image in a `"student"` step must be withheld, scraped or captured. Test in Task 5.
2. **`plt.savefig("x.png")` under `"show"`** is both captured (patched savefig) and scraped from "Plot saved to: x.png" — the same figure twice. Test in Task 5.
3. **An NMR or header-poor PDB** (no resolution, no JRNL, several MODELs) must still give a metadata panel, with missing fields omitted, not "None". Test in Task 2.
4. **An invalid SMILES or a malformed/empty file** must give a message and a download button, never an exception or a blank frame. Tests in Tasks 2 and 3.
5. **Journals written before this feature** (no `plot_mode` in `run_start`, no new `file_kind`s) must restore exactly as before. Test in Task 4.

---

### Task 1: Structure artifact layer (September plan, with deltas)

**Files:** as listed in `docs/superpowers/plans/2026-09-03-structure-artifact-layer.md` § File Structure (`biomentis/structures.py`, `biomentis/tool/database.py`, `pyproject.toml`, `biomentis/agent/tests/test_structure_artifacts.py`, `CLAUDE.md`).

**Interfaces:**
- Produces everything the September plan's Tasks 1–5 produce: `default_artifact_dir`, `registry_path`, `UNATTACHED_RUN_DIR`, `StructureArtifact`, `make_artifact_id`, `append_artifact`, `read_registry`, `resolve_artifact_path`, `StructureStats`, `inspect_structure`, `format_for_path`, `record_download`; and the `artifacts` list key on both download tools.
- Plus: `import_file(src_path: str, source_db: str, run_id: str | None = None, artifact_dir: str | None = None) -> StructureArtifact` — copies a file that lives outside the artifact dir into it (keeping its basename; an existing identical file is reused, a different one gets a `-2`, `-3`… suffix), then calls `record_download` with `accession` = basename without extension. Used for `source_db="upload"` and `"kb"`.

- [ ] **Step 1: Execute September plan Tasks 1–5 as written, applying these deltas everywhere they occur**
  - `format_for_path` returns `"pdb"`, `"sdf"`, `"smi"`, or `"unknown"`. Drop `cif` and `mol2` from code and tests.
  - `inspect_structure` on a `.smi` file returns `molecule_class="small_molecule"`, `n_atoms=None`, `chains=[]`, `title=` the name after the SMILES on the first line, if any.
  - The four-class fixture test keeps protein / nucleic_acid / complex / small_molecule; the small-molecule case uses an `.sdf` fixture.
  - Plan file header: the September plan's tasks are checked off in the September file; this task is checked here when all five are done.

- [ ] **Step 2: Write the failing test for `import_file`** in `test_structure_artifacts.py`

```python
def test_import_file_copies_into_artifact_dir() -> None:
    d = tempfile.mkdtemp(); src_dir = tempfile.mkdtemp()
    src = os.path.join(src_dir, "aspirin.smi"); open(src, "w").write("CC(=O)Oc1ccccc1C(=O)O aspirin\n")
    art = import_file(src, source_db="upload", artifact_dir=d)
    _check("copied inside the artifact dir", os.path.exists(os.path.join(d, art.path)) and not os.path.isabs(art.path))
    _check("source and accession recorded", (art.source_db, art.accession, art.fmt) == ("upload", "aspirin", "smi"))
    again = import_file(src, source_db="upload", artifact_dir=d)
    _check("re-importing the same file reuses it", again.path == art.path)
    open(src, "w").write("CCO ethanol\n")
    other = import_file(src, source_db="upload", artifact_dir=d)
    _check("a different file with the same name is not overwritten", other.path != art.path)
```

- [ ] **Step 3: Run it; expect ImportError for `import_file`.** `$P -m biomentis.agent.tests.test_structure_artifacts`
- [ ] **Step 4: Implement `import_file` in `biomentis/structures.py`.** Compare bytes to decide reuse vs suffix.
- [ ] **Step 5: Run the suite; expect all checks to pass.**
- [ ] **Step 6: Commit** — `git add biomentis/structures.py biomentis/tool/database.py pyproject.toml biomentis/agent/tests/test_structure_artifacts.py CLAUDE.md && git commit -m "Add the structure artifact layer; stop writing downloads into the package and CWD"`

---

### Task 2: Structure metadata and viewer input

**Files:**
- Create: `biomentis/structure_info.py`
- Create: `biomentis/agent/tests/fixtures/mini_xray.pdb`, `mini_nmr.pdb`, `two_mols.sdf`
- Create: `biomentis/agent/tests/test_structure_info.py`
- Modify: `requirements.txt:8-16` (add `rdkit>=2024.3` and `biopython>=1.80` to the scientific stack); install into the env: `/opt/anaconda3/envs/biomni/bin/pip install "rdkit>=2024.3"`

**Interfaces:**
- Consumes: `format_for_path`, `inspect_structure` (Task 1).
- Produces:
  - `structure_metadata(path: str) -> list[tuple[str, str]]` — ordered (label, value) pairs; never raises; missing fields omitted; on unreadable input returns `[("Error", "<one-line reason>")]`.
  - `sdf_record_count(path: str) -> int`
  - `viewer_model(path: str, record: int = 0) -> tuple[str, str] | None` — `(model_text, "pdb"|"sdf")` for 3Dmol; SMILES is embedded to 3D with RDKit (`AddHs`, `EmbedMolecule(randomSeed=0xC0FFEE)`, `MMFFOptimizeMolecule`) and returned as a mol block; `None` when it cannot be built.
  - `depiction_png(path: str, record: int = 0) -> bytes | None` — RDKit 2D drawing for `.smi`/`.sdf`; `None` for PDB.
  - `metadata_text(path: str) -> str` — `"label: value"` lines from `structure_metadata`, for KB indexing.

Fixtures are hand-written, ≤ 40 lines each. `mini_xray.pdb` has `HEADER, TITLE, COMPND, SOURCE, KEYWDS, EXPDTA X-RAY DIFFRACTION, AUTHOR, JRNL (AUTH/TITL/REF/DOI), REMARK 2 RESOLUTION. 1.80 ANGSTROMS., HETNAM`, two chains of ATOM records and one HETATM ligand. `mini_nmr.pdb` has `EXPDTA SOLUTION NMR`, no REMARK 2, no JRNL, and two `MODEL`/`ENDMDL` blocks. `two_mols.sdf` has two records, each with `> <PUBCHEM_COMPOUND_CID>` and `> <NAME>` data fields.

PDB labels, in order (omit any absent): Title, Classification, Deposited, Experimental method, Resolution (Å), Organism, Expression system, Molecule(s), Chains, Residues, Atoms, Ligands / heteroatoms (`HETNAM` names, waters excluded), Authors, Reference (journal title, citation, DOI), Keywords. SDF labels: Name, Formula, Molecular weight (g/mol), Atoms, Bonds, then each data field by its tag. SMILES labels: Name (if given), Canonical SMILES, Formula, Molecular weight (g/mol), H-bond donors, H-bond acceptors, Rotatable bonds, LogP, TPSA (Å²).

- [ ] **Step 1: Write the failing tests** in `test_structure_info.py`

```python
def test_pdb_metadata():            # mini_xray.pdb
    labels = dict(structure_metadata(XRAY))
    for k in ("Title", "Experimental method", "Resolution (Å)", "Organism", "Chains", "Ligands / heteroatoms", "Authors", "Reference"):
        _check(f"has {k}", k in labels, list(labels))
    _check("resolution parsed", labels["Resolution (Å)"] == "1.80")
    _check("chains listed", labels["Chains"] == "A, B")

def test_no_coordinates_anywhere():  # every fixture
    for p in (XRAY, NMR, SDF, SMI):
        blob = metadata_text(p)
        _check(f"{p}: no ATOM/HETATM line", not re.search(r"^(ATOM|HETATM)\s", blob, re.M))
        _check(f"{p}: no SDF atom-block line", not re.search(r"^\s*-?\d+\.\d{4}\s+-?\d+\.\d{4}\s+-?\d+\.\d{4}\s+[A-Z]", blob, re.M))

def test_nmr_is_not_none_filled():   # Review Focus 3
    md = structure_metadata(NMR)
    _check("no resolution row", "Resolution (Å)" not in dict(md))
    _check("no 'None' values", all(v and v != "None" for _, v in md))
    _check("method is NMR", "NMR" in dict(md)["Experimental method"])

def test_sdf_records_and_fields():
    _check("two records", sdf_record_count(SDF) == 2)
    _check("data fields shown", "PUBCHEM_COMPOUND_CID" in dict(structure_metadata(SDF)))
    _check("formula computed", dict(structure_metadata(SDF))["Formula"])

def test_smiles():                   # aspirin
    md = dict(structure_metadata(SMI))
    _check("formula", md["Formula"] == "C9H8O4"); _check("MW", md["Molecular weight (g/mol)"].startswith("180.1"))
    model = viewer_model(SMI)
    _check("3D model built as sdf", model is not None and model[1] == "sdf")
    _check("2D depiction", (depiction_png(SMI) or b"")[:4] == b"\x89PNG")

def test_bad_input_degrades():       # Review Focus 4
    for text, ext in (("not_a_smiles(((", ".smi"), ("", ".pdb"), ("garbage\n", ".sdf")):
        p = _tmp(text, ext)
        md = structure_metadata(p)
        _check(f"{ext}: error row, no raise", md and md[0][0] == "Error", md)
        _check(f"{ext}: no model", viewer_model(p) is None)
```

- [ ] **Step 2: Run; expect ImportError.** `$P -m biomentis.agent.tests.test_structure_info`
- [ ] **Step 3: Implement `biomentis/structure_info.py`.** PDB header via `Bio.PDB.parse_pdb_header` plus a line scan for `COMPND`/`SOURCE`/`HETNAM`/`JRNL`; counts from `PDBParser(QUIET=True)` on model 0. RDKit `Chem.SDMolSupplier`, `Chem.MolFromSmiles`, `rdMolDescriptors.CalcMolFormula`, `Descriptors`. Wrap each reader so any exception becomes the `("Error", reason)` row.
- [ ] **Step 4: Run; expect all pass.**
- [ ] **Step 5: Commit** — `git add biomentis/structure_info.py biomentis/agent/tests/test_structure_info.py biomentis/agent/tests/fixtures requirements.txt && git commit -m "Read structure metadata and viewer models from PDB, SDF and SMILES"`

---

### Task 3: The viewer

**Files:**
- Create: `biomentis/ui_visuals.py`
- Create: `biomentis/agent/tests/test_viewer.py`

**Interfaces:**
- Consumes: `structure_metadata`, `viewer_model`, `depiction_png`, `sdf_record_count` (Task 2); `inspect_structure` (Task 1).
- Produces:
  - `THREEDMOL_URL: str` — the pinned URL from Global Constraints.
  - `@dataclass MoleculeView: model_text: str; fmt: str; style: str` — style is `"cartoon"` or `"stick"`. One view per structure, so side-by-side later is a list of two.
  - `default_style(molecule_class: str) -> str` — `protein`/`nucleic_acid`/`complex` → `"cartoon"` (with ligands as sticks), else `"stick"`.
  - `viewer_html(views: list[MoleculeView], height: int = 420) -> str` — self-contained HTML: loads `THREEDMOL_URL`, embeds each model via `JSON.stringify`-safe escaping (`json.dumps`), enables rotate/zoom (3Dmol default), and contains a hidden `<div class="viewer-fallback">` that a `script.onerror` handler and a `try/catch` around model loading reveal with the text "The 3D viewer could not load. Use the download button below."
  - `render_structure(path: str, *, key: str, container=None) -> None` — two columns (viewer left via `st.components.v1.html`, metadata right as a two-column markdown table); for SDF with >1 record a `st.selectbox` record picker; for SMILES/SDF the 2D depiction under the metadata; always a `st.download_button` with `key=f"dl-{key}"`. When `viewer_model` is `None`, shows `st.info("This structure could not be drawn.")` instead of the frame, and the metadata/error rows still render.

- [ ] **Step 1: Write the failing tests** (pure functions only; Streamlit rendering is covered by the manual checklist)

```python
def test_viewer_html_shape():
    html = viewer_html([MoleculeView(model_text=open(XRAY).read(), fmt="pdb", style="cartoon")])
    _check("pinned 3Dmol", THREEDMOL_URL in html and "2.4.2" in html)
    _check("has fallback", "viewer-fallback" in html and "could not load" in html)
    _check("model safely embedded", "</script>" not in html.split(THREEDMOL_URL, 1)[1].split("<script>", 1)[1].rsplit("</script>", 1)[0])

def test_script_injection_is_inert():
    evil = "TITLE    </script><script>alert(1)</script>\nEND\n"
    _check("no raw closing tag from data", "</script><script>alert" not in viewer_html([MoleculeView(evil, "pdb", "stick")]))

def test_default_style():
    _check("protein → cartoon", default_style("protein") == "cartoon")
    _check("small molecule → stick", default_style("small_molecule") == "stick")
    _check("unknown → stick", default_style("unknown") == "stick")
```

`json.dumps` alone does not escape `</`; replace `"</"` with `"<\\/"` in the dumped string.

- [ ] **Step 2: Run; expect ImportError.** `$P -m biomentis.agent.tests.test_viewer`
- [ ] **Step 3: Implement `biomentis/ui_visuals.py`** with the interfaces above.
- [ ] **Step 4: Run; expect pass.**
- [ ] **Step 5: Commit** — `git commit -m "Add the 3Dmol structure viewer with metadata panel and fallback"` (add both files).

---

### Task 4: The class plot setting

**Files:**
- Create: `biomentis/plots.py`
- Modify: `biomentis/ui_visuals.py` (add the control)
- Modify: `biomentis/ui_tutor.py:922-931` (call the control right after the tutor-mode toggle, outside any `if tutor.enabled`)
- Modify: `biomentis/agent/a1.py:4009-4015` (snapshot the setting and run id onto the agent; pass `plot_mode` to `RunJournal`)
- Create: `biomentis/agent/tests/test_plot_mode.py`

**Interfaces:**
- Produces:
  - `PLOT_MODES = ("show", "student")`, `PLOT_MODE_LABELS = {"show": "Show plots", "student": "Students run the code"}` in `biomentis/plots.py`.
  - `plot_mode_default() -> str` — `BIOMENTIS_PLOT_MODE`, stripped and lowercased; not in `PLOT_MODES` → `"show"`.
  - `current_plot_mode() -> str` in `ui_visuals.py` — `st.session_state.get("biomentis_plot_mode")` if valid, else `plot_mode_default()`.
  - `render_plot_mode_control(logger=None) -> str` in `ui_visuals.py` — `st.radio("Plots", PLOT_MODES, format_func=PLOT_MODE_LABELS.get, key="biomentis_plot_mode", horizontal=True, help="Applies to the next run and the next question. Code is always shown.")`, initialised from `plot_mode_default()`; when `logger` is given, logs `{"kind": "plot_mode", "plot_mode": value}` the first time it renders in a session and whenever the value changes (track the last logged value in session state).
  - On the agent: `agent.plot_mode: str` and `agent.current_run_id: str | None`, set in `launch_streamlit_demo` immediately before `RunJournal(...)` is built (`agent.plot_mode = current_plot_mode()`), and `agent.current_run_id = _journal.run_id` right after. `A1.__init__` initialises both (`"show"`, `None`) so non-UI callers work.
  - `run_start` journal record gains `plot_mode` (passed as a `RunJournal` meta kwarg).

- [ ] **Step 1: Write the failing tests**

```python
def test_env_default():
    for raw, want in (("student", "student"), (" SHOW ", "show"), ("bogus", "show"), (None, "show")):
        _setenv("BIOMENTIS_PLOT_MODE", raw)
        _check(f"{raw!r} → {want}", plot_mode_default() == want)

def test_run_start_records_mode():
    j = RunJournal("p", run_dir=tmp, plot_mode="student")
    _check("run_start has plot_mode", load_run(j.path)["meta"]["plot_mode"] == "student")

def test_old_journal_restores_unchanged():   # Review Focus 5
    # write a journal by hand with no plot_mode and one image file event, as before this feature
    run = load_run(old_path)
    entries = entries_for_run(run)
    _check("restores", entries and entries[-1]["kind"] == "image")
    _check("no plot_mode key needed", "plot_mode" not in run["meta"])

def test_control_logs_changes():               # AppTest harness app calling render_plot_mode_control(fake_logger)
    at = AppTest.from_function(_harness).run()
    at.radio[0].set_value("student").run()
    _check("logged initial and change", [r["plot_mode"] for r in LOGGED] == ["show", "student"])
```

- [ ] **Step 2: Run; expect ImportError.** `$P -m biomentis.agent.tests.test_plot_mode`
- [ ] **Step 3: Implement** `biomentis/plots.py` (the two constants and `plot_mode_default`), the control and `current_plot_mode` in `ui_visuals.py`, the call in `render_tutor_sidebar` passing `tutor.logger`, and the a1.py snapshot.
- [ ] **Step 4: Run this suite and `test_background_run`; expect pass.**
- [ ] **Step 5: Commit** — `git commit -m "Add the class plot setting and record it on each run"`

---

### Task 5: Plots in agent steps

**Files:**
- Modify: `biomentis/tool/support_tools.py:35` (enable `_capture_matplotlib_plots()` after `exec`)
- Modify: `biomentis/agent/a1.py:1937-1951` (write captured plots as PNG files)
- Modify: `biomentis/ui_core.py` — `UIEvent.file_kind` literal (L55-73), observation handling (L260-301), `transcript_entry_for_event` file branch (L396-408), `ExportEntryKind` (L522), `ui_event_to_export_entry` (L560), `_entry_body_html` (L727)
- Modify: `biomentis/agent/a1.py:3746-3797` (`render_entry`: new kind)
- Create: `biomentis/agent/tests/test_step_plots.py`

**Interfaces:**
- Consumes: `default_artifact_dir` (Task 1); `agent.plot_mode`, `agent.current_run_id` (Task 4).
- Produces:
  - `save_plot_pngs(data_uris: list[str], out_dir: str, start_index: int) -> list[str]` in `biomentis/plots.py` — decodes each `data:image/png;base64,…` URI to `out_dir/plot-<NNN>.png` (zero-padded, continuing from `start_index`), returns absolute paths.
  - `execution_entry["image_paths"]: list[str]` — set by the execute node next to the existing `"images"` key, written to `default_artifact_dir(self.current_run_id)`. Numbering continues across steps (count existing `plot-*.png` in the dir).
  - New `file_kind` values: `"plot"` (shown) and `"plot_withheld"`. Events: `UIEvent(type="file", file_kind="plot", title="📈 Plot", file_path=abs)` and `UIEvent(type="file", file_kind="plot_withheld", title="📈 Plot", content="Run this code to see the plot.", file_path=abs)`.
  - Transcript entries: `{"panel": "inner", "role": "assistant", "kind": "plot", "file_path", "title"}` and `{"panel": "inner", "role": "assistant", "kind": "plot_withheld", "file_path", "title", "content"}`. `render_entry`: `plot` → `st.image(file_path)`; `plot_withheld` → `st.info(content)` (never the image).
  - Export: `ExportEntryKind` gains `"plot_withheld"`; it exports as an image titled "📈 Plot (withheld from students)". `plot` exports as `image`.

Rules in `stream_agent_events` at an observation, using `agent._execution_results[-1].get("image_paths", [])` (guard: only an entry appended since the previous observation — track the last seen `len(agent._execution_results)`):
1. Each captured path yields one plot event; mode is `getattr(agent, "plot_mode", "show")`.
2. Scraped image paths that appear on a `Plot saved to:` line are skipped when this step captured any plot (Review Focus 2).
3. Under `"student"`, *every* image event in the step — captured or scraped — is `plot_withheld` (Review Focus 1). PDFs are unaffected.

- [ ] **Step 1: Write the failing tests** — use a scripted agent like `_scripted_agent` in `test_tutor_card_cache.py`, giving it `_execution_results`, `plot_mode` and `path`; the scripted graph appends a code AIMessage then an observation AIMessage, and the test appends the execution entry before the observation is yielded.

```python
def test_show_mode_emits_plot():        # one captured png
    ev = _events(mode="show", image_paths=[png])
    _check("one plot event", [e.file_kind for e in ev if e.type == "file" and e.file_path] == ["plot"])

def test_student_mode_withholds():
    ev = _events(mode="student", image_paths=[png])
    kinds = [e.file_kind for e in ev if e.file_path]
    _check("withheld, not shown", kinds == ["plot_withheld"])
    entry = transcript_entry_for_event(next(e for e in ev if e.file_path))
    _check("entry carries the note", entry["kind"] == "plot_withheld" and "Run this code" in entry["content"])

def test_student_mode_withholds_scraped_images():          # Review Focus 1
    ev = _events(mode="student", image_paths=[], observation=f"wrote {png}\n")
    _check("scraped image withheld", [e.file_kind for e in ev if e.file_path] == ["plot_withheld"])

def test_savefig_not_duplicated():                         # Review Focus 2
    ev = _events(mode="show", image_paths=[png], observation=f"Plot saved to: {other_png}\n")
    _check("one figure, not two", len([e for e in ev if e.file_path]) == 1)

def test_recorded_mode_survives_restore():
    # journal the student-mode events, then restore with agent/session mode now "show"
    entries = entries_for_run(load_run(journal_path))
    _check("restored as withheld", [e["kind"] for e in entries if e.get("file_path")] == ["plot_withheld"])

def test_save_plot_pngs_numbering():
    paths = save_plot_pngs([URI, URI], d, start_index=3)
    _check("numbered from start", [os.path.basename(p) for p in paths] == ["plot-003.png", "plot-004.png"])

def test_export_keeps_withheld_plot():
    html = render_transcript_html([withheld_entry], "Executor", "m")
    _check("image embedded for the instructor", "data:image/png;base64" in html and "withheld from students" in html)
```

- [ ] **Step 2: Run; expect failures.** `$P -m biomentis.agent.tests.test_step_plots`
- [ ] **Step 3: Implement** `save_plot_pngs`, the execute-node change, the support_tools line, and the ui_core / render_entry / export changes per the rules above.
- [ ] **Step 4: Run this suite plus `test_background_run`, `test_solution_streaming`, `test_tab_focus`, `test_run_completion`; expect pass.**
- [ ] **Step 5: Commit** — `git commit -m "Show agent plots in their step, or withhold them under Students run the code"`

---

### Task 6: Agent guidance, the `show_structure` tool, and uploaded structures

**Files:**
- Modify: `biomentis/plots.py` (guidance text)
- Modify: `biomentis/agent/a1.py:1704-1712` (append guidance in the `generate` node)
- Modify: `biomentis/tool/support_tools.py` (add `show_structure`)
- Modify: `biomentis/tool/tool_description/support_tools.py` (describe it)
- Modify: `biomentis/ui_core.py` (marker detection at observations; uploaded-structure events before the agent loop; `file_kind` `"structure"`; transcript entry; export)
- Modify: `biomentis/agent/a1.py` `render_entry` (`structure` → `render_structure`)
- Create: `biomentis/agent/tests/test_show_structure.py`

**Interfaces:**
- Consumes: `query_pdb_identifiers` with `download=True` (Task 1), `import_file`, `default_artifact_dir`, `format_for_path`; `render_structure` (Task 3); `agent.current_run_id`, `agent.plot_mode`.
- Produces:
  - `plot_guidance(plot_mode: str) -> str` in `biomentis/plots.py`. Must say: plot data with matplotlib when explaining a quantitative or visual principle; label axes with units; call `plt.show()`; **every plotting block is self-contained — data written inline, only `numpy` and `matplotlib` imported, no variables from earlier steps, no files, no biomentis tools**, so a student can paste it alone into Colab; use `show_structure` to display a molecule. Under `"student"` it adds that students will run the code themselves and will not see the figure in the app, so the code must be complete and commented; under `"show"` that the figure will be shown beside the code.
  - In `generate`: `system_prompt += "\n\n" + plot_guidance(getattr(self, "plot_mode", "show"))` (per call, so the next run picks up a changed setting without `configure()`).
  - `STRUCTURE_MARKER = "🧬 Structure ready: "` in `biomentis/structures.py`.
  - `show_structure(source: str) -> str` in `support_tools.py`: a 4-character alphanumeric PDB ID → `query_pdb_identifiers([id], download=True)` and use its artifact; an existing file path with a supported extension → `import_file(path, "agent")`; otherwise treated as SMILES — validated with RDKit, written to `<artifact dir>/<slug>.smi` and registered with `source_db="smiles"`. Run id comes from module global `support_tools._current_run_id`, which the execute node sets from `self.current_run_id` before each exec. Prints `STRUCTURE_MARKER + <absolute path>` and returns a one-line summary (title / formula). Invalid input returns `"Could not show structure: <reason>"` and prints no marker.
  - Tool description entry: name `show_structure`, one required parameter `source` (str): "a PDB ID (e.g. 6NK5), a path to a .pdb/.sdf/.smi file, or a SMILES string".
  - `file_kind="structure"` events: `UIEvent(type="file", file_kind="structure", title=f"🧬 {basename}", file_path=abs)`. Transcript entry `{"panel": "inner", "role": "assistant", "kind": "structure", "file_path", "title"}`. Export: `ExportEntryKind` gains `"structure"`; `_entry_body_html` renders a metadata table from `structure_metadata`, no viewer.
  - Uploaded files: in `stream_agent_events`, before the agent loop, each file in `files` whose `format_for_path` is `pdb`/`sdf`/`smi` is `import_file(f, "upload", run_id=agent.current_run_id)`'d and yields one structure event. The viewer is not governed by the plot setting.

- [ ] **Step 1: Write the failing tests**

```python
def test_guidance_text():
    for m in ("show", "student"):
        g = plot_guidance(m)
        _check(f"{m}: self-contained rule", "inline" in g and "numpy" in g and "matplotlib" in g and "earlier steps" in g)
        _check(f"{m}: names show_structure", "show_structure" in g)
    _check("student wording", "will not see the figure" in plot_guidance("student"))

def test_show_structure_by_id():        # fake requests serving mini_xray.pdb for 6NK5
    out = _capture_stdout(lambda: show_structure("6NK5"))
    path = _marker_path(out)
    _check("marker printed", path and os.path.exists(path))
    _check("lands in the run dir", os.path.dirname(path) == default_artifact_dir(RUN))

def test_show_structure_smiles_and_bad():
    _check("SMILES registered", _marker_path(_capture_stdout(lambda: show_structure("CC(=O)Oc1ccccc1C(=O)O"))).endswith(".smi"))
    out = _capture_stdout(lambda: show_structure("not a molecule ((("))
    _check("bad input: no marker", STRUCTURE_MARKER not in out)

def test_marker_becomes_event():
    ev = _events(observation=f"{STRUCTURE_MARKER}{path}\n")
    _check("structure event", [e.file_kind for e in ev if e.file_path] == ["structure"])

def test_uploaded_pdb_opens():
    ev = _events(files=[XRAY_COPY])
    _check("upload yields a structure event first", next(e for e in ev if e.file_path).file_kind == "structure")

def test_export_structure_has_no_coordinates():
    html = render_transcript_html([structure_entry], "Executor", "m")
    _check("metadata in export", "Resolution" in html)
    _check("no ATOM lines in export", "ATOM  " not in html)
```

- [ ] **Step 2: Run; expect failures.** `$P -m biomentis.agent.tests.test_show_structure`
- [ ] **Step 3: Implement** per the interfaces.
- [ ] **Step 4: Run this suite plus `test_tool_imports`, `test_stop_sequences`, `test_step_plots`; expect pass.**
- [ ] **Step 5: Commit** — `git commit -m "Teach the agent to plot and to show structures; open uploaded structures"`

---

### Task 7: Plots and structures in the tutor conversation

**Files:**
- Modify: `biomentis/plots.py` (`run_plot_code`)
- Modify: `biomentis/agent/tutor/chat.py` — `ChatTurn` (L64-77), both JSON schemas and rules (L94-100, L135-141 and their RULES lists), `ask` (L302; coercion L435-443; log record L473-488)
- Modify: `biomentis/ui_tutor.py` — `_handle_chat_turn` (L1706; assistant dict L1820-1832), chat rendering (L1628-1679), chat HTML export (L1581-1618), `_handle_step_qa` (L1901-1927) and `_render_step_qa` rendering (L1864-1880)
- Create: `biomentis/agent/tests/test_chat_visuals.py`

**Interfaces:**
- Consumes: `default_artifact_dir`, `UNATTACHED_RUN_DIR` (Task 1); `render_structure` (Task 3); `current_plot_mode` (Task 4); `show_structure`'s source resolution (Task 6 — factor it into `resolve_structure_source(source: str, run_id: str | None, search_dirs: list[str]) -> str | None` in `biomentis/structures.py` and have `show_structure` call it; `search_dirs` lets a bare file name match a KB or upload file).
- Produces:
  - `@dataclass PlotResult: png_path: str | None; error: str | None`
  - `run_plot_code(code: str, out_dir: str, timeout: float = 20.0) -> PlotResult` — runs the code in a **subprocess** (`sys.executable -I -c <runner>`) with `MPLBACKEND=Agg` and cwd a fresh temp dir, so it shares nothing with the agent's REPL; the runner execs the code then saves the last open figure to `out_dir/chat-plot-<uuid8>.png`. Non-zero exit → `error` = last line of stderr; timeout → `error="Plot code took longer than 20 s and was stopped."`; no figure → `error="The code ran but drew no figure."`.
  - `ChatTurn` gains `plot_code: str = ""`, `plot_png: str = ""`, `plot_error: str = ""`, `structure_path: str = ""`, `plot_mode: str = "show"`.
  - Both chat JSON schemas gain `"plot_code": "<self-contained matplotlib code, or empty string>"` and `"structure": "<PDB ID, SMILES, or KB/upload file name, or empty string>"`; rules: use `plot_code` when the student asks to see something or a plot is clearly the better answer, same self-contained rule as `plot_guidance`; use `structure` when the student asks to see a molecule.
  - `TutorChat.ask(..., plot_mode: str = "show", search_dirs: list[str] | None = None)`. After coercion: under `"show"` with non-empty `plot_code`, `run_plot_code(code, default_artifact_dir(None) + "/chat")`; under `"student"` never runs it. `structure` resolved with `resolve_structure_source(..., run_id=None, search_dirs=search_dirs or [])`; unresolvable → ignored. The qa log record gains `plot_mode`, `plot_code`, `plot_png`, `structure_path`.
  - UI turn dicts (page panel and `run["step_qa"]` entries) gain the same five keys. Rendering, after the answer text: `plot_code` → `st.code(plot_code, language="python")` always; then `plot_png` → `st.image`, else `plot_error` → `st.caption("⚠️ " + plot_error)`, else (student mode with code) `st.info("Run this code to see the plot.")`; `structure_path` → `render_structure(path, key=f"chat-{index}")`. Rendering reads only the stored dict, never the current setting. UI callers pass `plot_mode=current_plot_mode()` and `search_dirs=[os.path.join(tutor.kb.path, "raw"), os.path.join(agent_path, "streamlit_uploads")]`.
  - Chat HTML export includes the code block and the PNG (base64) when present.

- [ ] **Step 1: Write the failing tests** (scripted LLM as in `test_tutor_card_cache.py`)

```python
CODE = "import numpy as np, matplotlib.pyplot as plt\nd=np.linspace(2,10,50)\nplt.plot(d, 450+80/d**2)\nplt.xlabel('diameter (nm)')"

def test_run_plot_code():
    ok = run_plot_code(CODE, out)
    _check("png written", ok.png_path and open(ok.png_path, "rb").read(4) == b"\x89PNG")
    _check("error reported", run_plot_code("raise ValueError('boom')", out).error.endswith("boom"))
    _check("timeout reported", "longer than" in run_plot_code("while True: pass", out, timeout=2).error)
    _check("isolated from REPL", run_plot_code("print(_persistent_namespace)", out).error is not None)

def test_show_mode_draws():
    turn = _ask(reply={"answer": "Here.", "plot_code": CODE}, plot_mode="show")
    _check("code kept", turn.plot_code == CODE); _check("figure made", turn.plot_png)

def test_student_mode_code_only():
    turn = _ask(reply={"answer": "Here.", "plot_code": CODE}, plot_mode="student")
    _check("code kept, no figure", turn.plot_code == CODE and not turn.plot_png and turn.plot_mode == "student")

def test_bad_plot_code_keeps_answer():
    turn = _ask(reply={"answer": "Here.", "plot_code": "1/0"}, plot_mode="show")
    _check("answer intact", turn.content.startswith("Here.") and not turn.failed)
    _check("error shown", "ZeroDivisionError" in turn.plot_error)

def test_structure_field():
    turn = _ask(reply={"answer": "See it.", "structure": "mini_xray.pdb"}, search_dirs=[FIXTURES])
    _check("resolved from search dirs", turn.structure_path.endswith(".pdb"))
    _check("unknown name ignored", _ask(reply={"answer": "x", "structure": "nope.pdb"}).structure_path == "")
```

- [ ] **Step 2: Run; expect failures.** `$P -m biomentis.agent.tests.test_chat_visuals`
- [ ] **Step 3: Implement** per the interfaces.
- [ ] **Step 4: Run this suite plus `test_tutor_card_cache`, `biomentis/agent/tutor/tests/test_smoke_e2e.py`, `test_critic.py`; expect pass.**
- [ ] **Step 5: Commit** — `git commit -m "Let the tutor answer with a plot or a structure"`

---

### Task 8: Structure files in the knowledge base

**Files:**
- Modify: `biomentis/agent/tutor/kb.py:357-379` (`_read_file_text` branches)
- Modify: `biomentis/ui_tutor.py:1046` (uploader `type=`), `:1049` (help text), `:1114-1116` (Indexed sources list)
- Create: `biomentis/agent/tests/test_kb_structures.py`

**Interfaces:**
- Consumes: `metadata_text`, `format_for_path` (Tasks 1–2); `import_file` (Task 1); `render_structure` (Task 3).
- Produces:
  - `_read_file_text` returns `metadata_text(path)` for `.pdb`, `.sdf`, `.smi` — the only text indexed for those files.
  - Uploader `type=["pdf", "pptx", "docx", "txt", "pdb", "sdf", "smi"]`.
  - In "Indexed sources", each structure file gets a `st.button("View", key=f"kbview-{name}")` that opens `@st.dialog("Structure", width="large")` calling `render_structure(os.path.join(tutor.kb.path, "raw", name), key=f"kb-{name}")`.

- [ ] **Step 1: Write the failing tests**

```python
def test_kb_indexes_metadata_only():
    kb = KnowledgeBase(session_id="t", path=tmp)      # same construction as test_kb_context.py
    n = kb.add_files([copy_of(XRAY), copy_of(SDF), copy_of(SMI)])
    _check("all three indexed", n == 3)
    docs = kb.retrieve("crystal structure resolution organism", k=4)
    text = "\n".join(d.page_content for d in docs)
    _check("metadata retrievable", "Resolution" in text)
    _check("no coordinates indexed", not re.search(r"^(ATOM|HETATM)\s", text, re.M))
```

- [ ] **Step 2: Run; expect failure (`Unsupported file type: .pdb`).** `$P -m biomentis.agent.tests.test_kb_structures`
- [ ] **Step 3: Implement** the kb branch and the sidebar changes.
- [ ] **Step 4: Run this suite plus `test_kb_context`; expect pass.**
- [ ] **Step 5: Commit** — `git commit -m "Accept PDB, SDF and SMILES files in the knowledge base"`

---

### Task 9: Wrap-up and the evaluation checklist

**Files:** `CLAUDE.md` (test list), `docs/superpowers/specs/2026-10-26-visualization-design.md` (tick checklist items as verified).

- [ ] **Step 1: Add every new suite to the CLAUDE.md test list** — `test_structure_artifacts`, `test_structure_info`, `test_viewer`, `test_plot_mode`, `test_step_plots`, `test_show_structure`, `test_chat_visuals`, `test_kb_structures`, each with a one-line comment in the existing style.
- [ ] **Step 2: Run every suite in the CLAUDE.md list.** Expected: all pass, 0 failed.
- [ ] **Step 3: Walk the spec's Evaluation checklist in the running app** (`streamlit run streamlit_app.py`), with gemma4:31b-cloud and with gemma4:26b-mlx, re-running the quantum-dot task from `runs/20261026-080552-74079b.jsonl` in both plot settings; switch the setting mid-run and after restore; turn the network off for the viewer fallback item. Tick each item in the spec that passes; for any that fails, note it under the checklist and stop.
- [ ] **Step 4: Commit** — `git commit -m "Add plot and structure suites to the test list; record checklist results"`

---

## Self-Review

- **Spec coverage:** Goals 1–2 → Tasks 5–6; Goal 3 → Task 4 (+ 5, 7 consume it); Goal 4 → Task 7; Goals 5–6 → Tasks 2–3; Goal 7 → Tasks 1, 6 (upload, fetch), 8 (KB); Goal 8 → Task 6. Recorded-mode rule → producer-side `file_kind` (Task 5) and stored chat dicts (Task 7). Session-log record → Task 4. Exports → Tasks 5–7. Later work: `MoleculeView` list in Task 3 keeps side-by-side open. Dependencies declared → Tasks 1–2.
- **Types:** `plot_mode` values `"show"|"student"` throughout; `file_kind` values `"plot"`, `"plot_withheld"`, `"structure"`; `import_file`, `resolve_structure_source`, `structure_metadata`, `viewer_model`, `render_structure(path, *, key)` used with the same signatures in every task.
- **Review Focus:** each of the five has a test in its owning task (Tasks 2, 4, 5).
