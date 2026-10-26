# Plots and structure viewing — design

**Date:** 2026-10-26
**Status:** approved design, ready for a build plan
**Supersedes:** `2026-09-03-molecule-visualization-design.md` (and its plan,
`docs/superpowers/plans/2026-09-03-structure-artifact-layer.md`). Neither was
built. Its structure artifact layer is folded in below as the foundation; its
richer capabilities are kept as later work (§ Later work).

## Why

Biomentis teaches biology and biotechnology students by letting them watch a
research agent work, with a tutor layer that explains each step. Many of the
principles it explains are easier to see than to read. The motivating case: a
run comparing InP, ZnTe and ZnSe quantum dots concluded that "ZnTe shifts least
per nanometer of diameter". That is really a picture — emission wavelength
against diameter for three materials, a line at 530 nm, and ZnTe's curve
visibly flattest — but the agent printed numbers and a mock synthesis log
instead. To see anything, a student had to copy the code and run it elsewhere.

That is acceptable in a coding class, where running the code is the point, and
a barrier for a student learning the biology, who loses the thread the moment
they leave the app.

Structures are the same problem in three dimensions. When the material involves
a protein (PDB) or a small molecule (SMILES, SDF), the student should see and
handle the molecule, and read what is known about it, without leaving the app.

## Goals

1. **Plots in the step.** When a principle is visual, the agent writes Python
   (matplotlib) that plots the data it is describing, runs it, and the figure
   appears in that step of the transcript, beneath the code. Figures appear
   only in the step that made them — not beside the final answer in the main
   output tab.
2. **Code always visible.** Every plot is shown together with the code that made
   it. This never changes.
3. **A class-level plot setting**, chosen at the start of each class:
   - **Show plots** — the figure appears in the app.
   - **Students run the code** — the code is shown but the figure is withheld;
     the student copies the code and runs it to see the plot.
4. **Plots in the tutor conversation.** A student in tutor mode can ask for a
   visual ("can you show me that?"), and the tutor's answer can include plotting
   code and — under *Show plots* — the figure.
5. **Structure viewer.** PDB, SDF and SMILES open in an interactive 3D viewer
   (rotate, zoom) with a metadata panel beside it.
6. **Metadata, not coordinates.** The panel shows everything descriptive in the
   file — for PDB: title, classification, organism, expression system,
   experimental method, resolution, deposition/release dates, authors, journal
   reference, compounds, chains, ligands/heteroatoms, residue and atom counts.
   The atom-coordinate records (`ATOM`/`HETATM` lines, the SDF atom and bond
   blocks) are never shown as text.
7. **Three structure sources**, all opening the same viewer:
   - a student uploads a file;
   - the instructor puts it in the knowledge base;
   - the agent fetches it by ID (RCSB PDB, AlphaFold).
8. **Visualization as agent tools.** The agent's toolset gains a structure-display
   tool, described in the tool definitions like every other tool, and its
   system prompt gains guidance on when to plot.

## Non-goals (for now)

- Formats other than **PDB, SDF and SMILES**. mmCIF, MOL2 and FASTA are out.
- Plotting libraries other than matplotlib (no plotly).
- Image generation of any kind. Plots are drawn from data by code.
- Everything in § Later work.

## Anti-goals — failure even if it "works"

- Code hidden or collapsed away from the student, in either setting.
- Under *Students run the code*, a figure leaking into the step or the chat.
- Coordinate lines dumped as text in the metadata panel or the transcript.
- A blank or broken frame where the viewer should be (offline, bad file). There
  must always be a readable fallback.
- A plot or viewer that disappears when the student switches tabs, reloads, or
  re-attaches to a running agent (see Constraints).
- Students having to leave the app to see anything under *Show plots*.

## Constraints

- **Run-loop invariants (CLAUDE.md).** Agent turns run on a background thread and
  every UI event is journaled to `runs/<run_id>.jsonl` by the producer. Plot and
  structure events must be ordinary journaled events: a torn-down consumer must
  re-attach without gaps or repeats, and a restored transcript must show them.
- **Self-critic.** One turn can emit several `<solution>` blocks; nothing here may
  latch onto the first.
- **No community Streamlit components.** The viewer is built from Streamlit's
  built-in HTML embed (`st.components.v1.html`) plus 3Dmol.js. (The component
  registry referenced in the global instructions is not present on this
  machine; this design needs no component, so the check does not bind.)
- **Students install this themselves.** New dependencies must install with pip
  and no compiler. RDKit (pip wheel `rdkit`) qualifies; Biopython is already
  used and must be declared.
- **Environment:** conda env `biomni`; tests are plain `_check()` scripts, no
  pytest, no live network in tests.

## How it works

### 1. Structure artifact layer (from September, unchanged in substance)

Adopt sub-project 1 of the September design as specified there: a
`biomentis/structures.py` module with run-scoped artifact directories
(`runs/<run_id>/`), a JSONL registry of `StructureArtifact` records with
relative paths, molecule classification, and the two download-path fixes
(`query_pdb_identifiers` writing into the package, `query_alphafold` writing
to CWD). The September plan file is a valid starting point for this part.

Changes from September:
- Formats are `pdb`, `sdf`, `smiles` (a `.smi` file holding one SMILES string
  plus an optional name). `cif`/`mol2` are dropped from scope.
- Uploads and knowledge-base files register as artifacts too, with
  `source_db` of `upload` or `kb`.

### 2. Metadata extraction

A function per format returns an ordered list of (label, value) pairs for the
panel. It never returns coordinate data.

- **PDB:** Biopython's header parser plus the `COMPND`, `SOURCE`, `REMARK 2/3`,
  `JRNL` and `HETNAM` records; chain list, residue/atom counts and heteroatom
  names computed from the structure.
- **SDF:** the molecule name line, every `> <tag>` data field, atom/bond counts,
  and formula and molecular weight via RDKit. Multi-record SDF: one panel per
  record, with a record selector.
- **SMILES:** canonical SMILES, formula, molecular weight, and a short set of
  standard descriptors via RDKit.

### 3. The viewer

One render function takes an artifact and draws: 3Dmol.js viewer on the left
(cartoon for proteins/nucleic acids, sticks for small molecules — from
`molecule_class`), metadata panel on the right; stacked on narrow screens.

- SMILES has no coordinates; RDKit embeds 3D coordinates and passes an SDF
  block to the viewer. A 2D depiction is shown as well.
- 3Dmol.js loads from a pinned CDN version. If it fails to load, or the file
  will not parse, the viewer area shows a message and a download button, and
  the metadata panel still renders.
- The viewer is not governed by the plot setting; it always renders.

### 4. Plots in agent steps

The REPL already captures matplotlib figures on `plt.show()`/`savefig()`
(`support_tools.py`), but only the exported report uses them. Change this so
that each captured figure is written as a PNG into the run's artifact directory
and emitted as a journaled image event attached to the step that produced it
(not discovered by scraping stdout). Figures left open when the code finishes
are captured too.

The agent's system prompt gains guidance: when explaining a quantitative or
visual principle, plot the data with matplotlib, label axes with units, and call
`plt.show()`.

**Plotting code is self-contained** (designer's call): every block that draws a
plot writes its data inline and imports only numpy and matplotlib, so a student
can paste that one block into Colab or their own Python and get the plot. It
must not depend on variables from earlier steps, run files, or biomentis tools.
The agent is told which plot setting is active, and the prompt states this rule
in both settings.

**Under *Students run the code*:** the agent still runs its code (it needs the
numbers), but captured figures are not displayed. In their place the step shows
a short note: "Run this code to see the plot." The PNGs are still written and
journaled so an instructor can review them; whether they appear in the
exported report is the builder's call, defaulting to yes.

### 5. Plots and structures in the tutor conversation

The tutor chat reply (JSON) gains two optional fields:
- `plot_code` — self-contained matplotlib code that plots data already in the
  conversation;
- `structure` — a PDB ID, SMILES string, or the name of a structure file in the
  knowledge base or uploads.

The chat prompt tells the model when to use them (the student asks to see
something, or a picture is clearly the better answer). The app shows
`plot_code` as a code block always; under *Show plots* it runs the code in a
fresh, isolated namespace with a timeout and shows the figure beneath it. A
failed run shows the error briefly instead of a figure, and never fails the
answer. `structure` opens the viewer inside the answer. Both are saved in the
chat history so they re-render.

The same fields apply to the per-step "Ask about this step" Q&A, which uses the
same `TutorChat.ask` path.

### 6. Structure sources

- **Uploads:** the existing main-panel uploader already accepts any file type.
  `.pdb`, `.sdf` and `.smi` uploads register as artifacts and open in the viewer
  in the transcript when the run starts.
- **Knowledge base:** the KB uploader accepts `.pdb`, `.sdf` and `.smi`. For
  retrieval, the KB indexes the *metadata text* (§2), never coordinates, so the
  tutor can cite "6NK5 — Crystal structure of …". The sidebar KB file list
  gets a "View" action per structure file.
- **Agent fetch:** the download tools register artifacts (§1); a new agent tool
  `show_structure(source)` — accepting a PDB ID, a file path, or a SMILES
  string — registers if needed and emits a journaled structure event, which the
  transcript renders as the viewer. Fetched structures are *not* auto-displayed;
  the agent shows a structure by calling the tool. The tool goes in the tool
  registry and tool-description files like every other tool.

### 7. The class setting

A sidebar control, **Plots**: *Show plots* / *Students run the code*, near the
tutor-mode toggle. Its default comes from an environment variable
(`BIOMENTIS_PLOT_MODE=show|student`) so an instructor can set it once in `.env`
for a class; the sidebar changes it for the current session. It is a default,
not a lock: a student can change it (decided by the requester). The current
setting is recorded in each run's `run_start` journal record and in the tutor
session log.

**A run always displays the way it was recorded** (decided by the requester).
The renderer reads the setting from the run's `run_start` record, not from the
sidebar, when showing that run — live, on re-attach, and on restore. Changing
the sidebar mid-run affects the next run only. Figures withheld under *Students
run the code* never appear in the app; an instructor reviews them through the
exported report or the PNG files in `runs/<run_id>/`. The same rule applies to
tutor-chat plots: each answer is shown under the setting in force when it was
asked.

## Later work (kept from September, not in this build)

Reserved for a specific application of the agent to be defined later:
- two structures side by side, and superposition;
- an artifact tray listing every structure a run touched;
- highlighting/colouring a selection (selection language);
- snapshot of the viewer into the exported report;
- auto-display of fetched structures;
- mmCIF/MOL2 and other formats.

The viewer's render function should take a view description (structure +
representation) rather than hard-coding one structure, so side-by-side and
selections can be added without rewriting it.

## Left to the builder

- Module layout beyond `biomentis/structures.py`, function names, the event
  type name(s) for plots and structures.
- 3Dmol.js version (pin an exact, established version).
- Which RDKit descriptors appear for SMILES/SDF.
- Exact wording of prompt guidance and the student note.
- Timeout for tutor-chat plot execution (a few seconds is the intent).
- How the exported HTML/PDF report shows a structure (metadata table at
  minimum).

## Evaluation checklist

Each item is checked in the running app unless marked *(test)*.

**Plots**
- [ ] ❌ Re-running the quantum-dot task with *Show plots*, at least one step shows
      a labelled wavelength-vs-diameter figure under its code.
      ↳ 2026-10-26: gemma4:31b-cloud ran the task without plotting (printed numbers). gemma4:26b-mlx plotted when asked explicitly, but looped 120 times on a units bug it could not see (observation only says "Plot captured.").
- [x] The code that made each plot is visible above it, in both settings.
      ↳ verified under Show plots (agent) and both settings (chat).
- [ ] With *Students run the code*, no figure appears in any step; the note
      appears instead; the code still copies and runs elsewhere to produce the
      plot.
      ↳ not walked in the app for an agent run (covered by test_step_plots); verified in the chat.
- [x] Switching tabs, reloading, and re-attaching mid-run keeps every plot in
      place, with none duplicated *(test, journal re-attach)*.
      ↳ test_reattach_mid_run_and_restore.
- [ ] A restored past run shows its plots.
      ↳ not walked in the app (restore covered by tests).
- [x] The run's journal records which plot setting it ran under.
      ↳ run_start plot_mode=show in runs/20261026-111559-3d0e05.jsonl.
- [x] A run made under *Students run the code* still shows no figures after the
      sidebar is switched to *Show plots*, live and after restore *(test)*.
- [x] Each plot's code block, pasted alone into a fresh Python with only numpy
      and matplotlib, produces the plot.
      ↳ agent blocks import only numpy + matplotlib and run in a fresh `python -I`.

**Tutor conversation**
- [x] Asking "can you show me that?" after a quantitative step gets an answer
      with plot code and, under *Show plots*, the figure.
      ↳ chat with gemma4:31b-cloud: code + figure.
- [x] Under *Students run the code*, the same question returns code only.
      ↳ code + "Run this code to see the plot."; the earlier Show-plots answer kept its figure.
- [ ] Plot code that errors shows a short error, and the text answer still
      appears.
      ↳ covered by test_chat_visuals; not forced in the app.
- [x] Asking to see a structure by PDB ID opens the viewer in the answer.
      ↳ 6NK6 opened in the chat answer (the answer text misdescribed it; viewer and metadata correct).

**Structures**
- [x] A PDB file opens from each source: upload, knowledge base, agent fetch by ID.
      ↳ upload and KB verified in the app; agent fetch covered by test_show_structure.
- [x] An SDF file and a SMILES string each open in the viewer.
- [x] The viewer rotates and zooms.
      ↳ headless Chromium with 6NK6 and aspirin; a 2-residue synthetic fixture draws nothing in cartoon style.
- [x] The metadata panel shows title, organism, method, resolution, authors,
      reference, chains and ligands for a PDB entry.
      ↳ 6NK6 and mini_xray.pdb.
- [x] No `ATOM`/`HETATM` line or SDF atom block appears anywhere as text
      *(test)*.
- [x] With the network off, the viewer area shows a message and a download
      button; metadata still shows.
      ↳ CDN blocked and CDN hanging (8 s watchdog) both show the fallback.
- [x] A malformed file shows a message rather than a broken frame *(test)*.
- [x] Knowledge-base retrieval returns structure metadata, not coordinates
      *(test)*.
      ↳ broken.pdb skipped with a sidebar warning.

**Foundation**
- [x] Downloaded structures land in `runs/<run_id>/`, not in the package or the
      repo root *(test)*.
- [x] All existing test suites still pass; new suites are added to CLAUDE.md.
      ↳ 23 suites, 893 checks.
- [x] `rdkit` and `biopython` are declared in the requirements.
