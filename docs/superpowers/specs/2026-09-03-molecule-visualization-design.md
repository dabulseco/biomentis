# Molecule visualization harness — design

**Date:** 2026-09-03
**Status:** approved for sub-project 1
**Scope of this document:** the overall architecture, then a full specification
of **sub-project 1, the structure artifact layer**. Sub-projects 2–4 are
outlined only, and each gets its own spec before implementation.

## Problem

Biomentis can already retrieve structures from RCSB PDB, AlphaFold, PubChem,
ChEMBL and others, but a student never sees them. Three things stand in the
way, all confirmed by reading the code:

1. **Nothing renders 3D.** No `py3Dmol`, `nglview`, `stmol`, RDKit, MDAnalysis,
   biotite or ProDy is installed. Available: Biopython 1.87, matplotlib,
   Pillow, Streamlit 1.59.2.
2. **There is no artifact layer.** Only two functions write structures to disk,
   and both write to the wrong place (§ Motivating defects). Neither returns a
   usable path — it is buried in a nested dict.
3. **The UI cannot see structures.** Files reach the transcript by regex-scraping
   tool stdout for known extensions (`ui_core.py:27` and `ui_core.py:243`, two
   separate hardcoded lists that must agree). `.pdb`, `.cif`, `.sdf` and `.mol2`
   are absent from both, so a downloaded structure is invisible.

The goal is a harness that visualizes small molecules, proteins from PDB files,
and nucleic acids, driven by files the existing database tools download.

## Requirements

Gathered from the requester:

- **Four independent triggers**, all wanted: an agent-called visualize tool;
  auto-render on download; a user-browsable artifact tray; tutor/lesson-pinned
  structures.
- **Four capabilities**, all wanted: highlight/color a selection; per-molecule-class
  representations; two structures shown together; snapshot into the report.
- **Deployment:** local now, hosted later. The data path between "file on disk"
  and "bytes in the viewer" must stay abstracted so hosted mode is a change,
  not a rewrite.
- **Retention:** keep artifacts indefinitely; no automatic cleanup.

## Architecture

Four layers. Each is independently useful and independently testable, and each
is its own spec → plan → implementation cycle.

| # | Layer | Responsibility |
|---|---|---|
| 1 | **Structure artifact layer** | A downloaded file becomes a registered, typed, addressable artifact. No UI. **Specified below.** |
| 2 | **Viewer spec + render surface** | A serializable `MoleculeView` (structure ref + representations + selections + camera) and the 3Dmol.js renderer that turns it into pixels. |
| 3 | **Entry points** | The agent tool, the auto-render policy, the artifact tray, the tutor hook. Thin once 1–2 exist. |
| 4 | **Rich capabilities** | Superposition, snapshot-to-PNG, comparison layout. |

The central decoupling: layer 2 defines *what to render*, layer 3 decides *who
asked for it*. One viewer primitive, four entry points — this is what keeps the
four triggers from becoming four viewers.

### Rendering engine: 3Dmol.js

Chosen over Mol\* and a static Biopython/matplotlib renderer.

- Pure JavaScript, loaded from CDN, injected through `st.components.v1.html`.
  **Zero native dependencies** — decisive, because students install this
  themselves and anything needing a compiler becomes a support burden.
- Reads PDB, mmCIF, SDF and MOL2 natively; no parser work for us.
- Supports cartoon, stick, sphere and surface representations, nucleic-acid
  rendering, and a real selection language (`{chain:'A', resi:'45-60'}`).
- Because the output is HTML + JS, the exported report gets a *live* viewer
  rather than a dead screenshot.

Mol\* is stronger on very large assemblies and on DNA/RNA cartoons, and is worth
revisiting if cryo-EM-scale structures become routine. It is not worth its
heavier API for a teaching tool today.

When the component cannot load, the renderer falls back to a download button —
never a broken frame.

---

# Sub-project 1 — structure artifact layer

## Motivating defects

Two bugs in the current download paths. Both are real independent of this
feature, and both block auto-render, so both are in scope here.

- **`query_pdb_identifiers` writes inside the installed package.**
  `database.py:1009` builds `os.path.join(os.path.dirname(__file__), "data", "pdb")`,
  i.e. `biomentis/tool/data/pdb/`. That directory currently holds ~6.9 MB of
  downloaded structures. The location is not configurable.
- **`query_alphafold` defaults to the process CWD.** `database.py:613` sets
  `output_dir = "."` when none is given. This is why the repository root has
  accumulated loose `6nk5.pdb`, `6nk6.pdb`, `7klv.pdb` and ~30 stray
  `.csv`/`.json`/`.fasta` files.

Neither returns a usable path: `query_pdb_identifiers` buries it at
`detailed_results[i].pdb_file_path`, `query_alphafold` at `download.file_path`.

Existing stray files at the repository root are **left alone**. Deleting a
user's files is not a side effect a code change should have.

## Artifact directory

```
runs/
  ├── <run_id>.jsonl          existing journal, untouched
  └── <run_id>/               new, co-located
        ├── index.jsonl       the registry
        ├── 6NK5.pdb
        └── AF-P04637-F1-model_v4.pdb
```

Artifacts are co-located with the run journal so the two share a lifetime.
`run_journal.entries_for_run` restores a transcript carrying **only file paths**,
never bytes; if an artifact has been cleaned up independently, restore renders a
broken viewer. Co-location closes that hole by construction. (Images have this
latent bug today: `_resolve_file_path`, `ui_core.py:74-85`, checks for existence
at *stream* time, not at restore time.)

There is no name collision: `<run_id>.jsonl` is a file and `<run_id>/` is a
directory.

Root resolution follows the existing `default_run_dir()` idiom
(`run_journal.py:50-52`):

```python
def default_artifact_dir(run_id: str | None = None) -> str:
    """Directory for this run's downloaded artifacts."""
```

Order: `BIOMENTIS_ARTIFACT_DIR`, else `<default_run_dir()>/<run_id>`. Downloads
occurring outside a run — the sidebar REPL, or a programmatic call with no
background run — resolve to `<default_run_dir()>/unattached` rather than
failing. That directory is shared and never swept; artifacts in it carry
`run_id=None`, and the artifact tray shows them under a separate heading so it
is clear they are not tied to any run.

**Paths in the registry are stored relative to the artifact root, never
absolute.** This is the single decision that makes hosted mode a change to one
resolver rather than a migration of every stored record.

## Registry

One record per line in `index.jsonl`, append-only, written by the producer.
This deliberately mirrors `run_journal.py`: the idiom is already proven in this
repo, it is crash-safe, and a partially written final line costs one record
rather than the file.

```python
@dataclass
class StructureArtifact:
    artifact_id: str          # deterministic: f"{source_db}:{accession}:{fmt}",
                              # lowercased. Same structure re-fetched into the
                              # same run yields the same id, which is what makes
                              # registration idempotent rather than duplicating.
    path: str                 # RELATIVE to the artifact root
    source_db: str            # "rcsb" | "alphafold" | "pubchem" | "chembl" | ...
    accession: str            # "6NK5", "P04637", a PubChem CID
    fmt: str                  # "pdb" | "cif" | "sdf" | "mol2"
    molecule_class: str       # see below
    n_atoms: int | None
    chains: list[str]
    title: str | None
    fetched_at: str           # ISO 8601
    run_id: str | None
```

`source_db` + `accession` carry more weight than they appear to: they let the
artifact tray show "6NK5 — RCSB PDB" instead of a filename, and they make
re-fetching the same structure idempotent rather than duplicating it.

Reading a registry with a truncated final line yields every complete record and
drops the partial one, without raising.

## Molecule classification

Classification lives here, not in the viewer, because it is what later selects a
default representation (cartoon for protein, sticks for a ligand, ladder for
DNA).

Using Bio.PDB, by residue composition:

| Composition | `molecule_class` |
|---|---|
| Standard amino acids | `protein` |
| `DA`/`DC`/`DG`/`DT`, or `A`/`C`/`G`/`U` | `nucleic_acid` |
| Both of the above | `complex` |
| HETATM-only, or an SDF/MOL2 file | `small_molecule` |
| Parse failed or Biopython absent | `unknown` |

Degradation is graceful at every step: a parse failure falls back to a cheap
header sniff (`HEADER`/`SEQRES` records), and a missing Biopython yields
`unknown`. 3Dmol renders correctly without a class, so classification never
blocks visualization — it only improves the default.

**Biopython is declared in `pyproject.toml` as part of this work.** Many existing
tools (`molecular_biology.py`, `systems_biology.py:630`, `biochemistry.py`)
already import it; it is currently undeclared in both `requirements.txt` and
`pyproject.toml` and works in the development environment by luck.

## Download tool changes

For `query_pdb_identifiers` and `query_alphafold`:

1. Write to `default_artifact_dir()` — not the package directory, not CWD.
   An explicit caller-supplied `output_dir` is still honored.
2. Register the file, producing a `StructureArtifact`.
3. Add a top-level `artifact` key to the returned dict.
4. **Print the resolved absolute path to stdout.**

Step 4 is what lets sub-project 3's auto-render reuse the existing stdout-scraping
seam at `ui_core.py:243` rather than inventing a parallel channel. Auto-render
therefore cannot ship before this sub-project — a dependency worth stating
plainly.

**Backward compatibility:** the existing nested keys (`detailed_results[i].pdb_file_path`,
`download.file_path`) are retained alongside the new top-level key. Prompts, tool
descriptions, and agent-written code that reads the old shape keep working.

## Out of scope

Deferred to later sub-projects, listed so the boundary is unambiguous:

- Any UI, any Streamlit code, any `UIEvent` or `ExportEntryKind` change.
- The `MoleculeView` spec and the selection language.
- Auto-render, the artifact tray, the agent-facing visualize tool, the tutor hook.
- Superposition and snapshot export.
- Normalizing the ~20 other `query_*` tools, or the `_format_query_results`
  `truncate_strings=100` default (`database.py:430-437`) that mangles long
  sequences and SMILES. Real, but a separate concern.
- RDKit's absence despite being imported at `pharmacology.py:1541`, and
  `env_desc.py:93-115` advertising RDKit/biotite/OpenBabel to the LLM as
  available when they are not. Both are genuine bugs; neither belongs here.

## Testing

Plain scripts with the house `_check()` harness, no pytest, added to the
CLAUDE.md test list as
`biomentis/agent/tests/test_structure_artifacts.py`:

1. Each of the four molecule classes is detected from a real fixture file.
2. Classification returns `unknown` rather than raising when Biopython is absent.
3. A registry round-trips: written records read back identically.
4. Stored paths are relative; resolution against a *different* root still finds
   the file — the hosted-mode guarantee.
5. A registry whose final line is truncated yields all complete records and does
   not raise.
6. `query_pdb_identifiers` writes to the artifact dir, and writes nothing into
   the package directory.
7. `query_alphafold` writes to the artifact dir, and writes nothing to CWD.
8. Both still expose their original nested path keys.
9. Both print a resolvable absolute path to stdout.

Tests 6–9 use a fake HTTP layer; no test performs a live network fetch.

## Decisions and rationale

| Decision | Why | Alternative rejected |
|---|---|---|
| 3Dmol.js | Zero native deps; students self-install | Mol\* (heavier API); static matplotlib (no interactivity) |
| Artifacts co-located with the run journal | Shared lifetime; restore cannot break | A shared content-addressed cache — saves disk, but decouples lifetimes and reintroduces broken restore |
| Keep artifacts indefinitely | Structures are small; restore always works | Age-based sweep — silently breaks old runs, the exact failure mode images have today |
| Relative paths in the registry | Hosted mode becomes one resolver change | Absolute paths — would require migrating every record |
| JSONL registry | Matches `run_journal.py`; crash-safe | SQLite — more machinery than a linear append log needs |
| Additive return-shape change | Old prompts and agent code keep working | Replacing the nested keys — silent breakage |
