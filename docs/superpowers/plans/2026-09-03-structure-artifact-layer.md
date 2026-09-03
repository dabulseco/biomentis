# Structure Artifact Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every structure file downloaded from an external database a registered, typed, addressable artifact stored in a run-scoped directory.

**Architecture:** One new module, `biomentis/structures.py`, owns artifact directory resolution, a JSONL registry, and molecule classification. The two database tools that write structures to disk are rewired to use it. No UI code, no Streamlit, no `UIEvent` changes — this is sub-project 1 of four, and later sub-projects consume `StructureArtifact`.

**Tech Stack:** Python 3.11+, Biopython (`Bio.PDB`), standard library only otherwise. Tests are plain scripts with the repo's hand-rolled `_check()` harness — **no pytest**.

**Spec:** `docs/superpowers/specs/2026-09-03-molecule-visualization-design.md`

## Global Constraints

- Python `requires-python = ">=3.11"`. Use `X | None`, not `Optional[X]`.
- **No pytest.** Tests are modules run with `python -m biomentis.agent.tests.<name>`, using the `_check(name, cond, detail)` harness copied from `biomentis/agent/tests/test_ncbi_email.py`.
- Run everything with the conda env interpreter: `/opt/anaconda3/envs/biomni/bin/python`, with `PYTHONPATH=/Users/dylanbulseco/Documents/Projects/Biomni`.
- **No test performs a live network fetch.** Tasks 4 and 5 monkeypatch `biomentis.tool.database.requests`.
- Registry paths are stored **relative to the artifact root**, never absolute.
- Existing nested return keys (`detailed_results[i].pdb_file_path`, `download.file_path`) are **retained**. Changes are additive only.
- Do not delete or move the stray `.pdb`/`.csv`/`.json` files at the repository root.
- Package name is `biomentis`. If you hit `ModuleNotFoundError: No module named 'biomni'`, that is a stray legacy import — grep for it.

**One refinement to the spec, applied throughout:** the spec says the download tools gain "a top-level `artifact` key". This plan uses **`artifacts`, a list of dicts, in both tools** — `query_pdb_identifiers` can download many structures in one call, and a single key shape means the tool descriptions and any downstream consumer handle one thing rather than two.

## File Structure

| File | Responsibility |
|---|---|
| `biomentis/structures.py` (create) | Everything about "a structure file on disk": directory resolution, `StructureArtifact`, registry read/write, classification. One cohesive module — these change together. |
| `biomentis/tool/database.py` (modify) | Two functions rewired to write into the artifact dir and register. No other function touched. |
| `pyproject.toml` (modify) | Declare `biopython`. |
| `biomentis/agent/tests/test_structure_artifacts.py` (create) | All nine spec tests. |
| `CLAUDE.md` (modify) | Add the new test to the runnable list. |

---

### Task 1: Artifact directory resolution

**Files:**
- Create: `biomentis/structures.py`
- Create: `biomentis/agent/tests/test_structure_artifacts.py`
- Modify: `CLAUDE.md:44` (add a line after the `test_anthropic_key` line)

**Interfaces:**
- Consumes: `biomentis.run_journal.default_run_dir()` — existing, returns `os.getenv("BIOMENTIS_RUN_DIR", "runs")`.
- Produces:
  - `default_artifact_dir(run_id: str | None = None) -> str`
  - `registry_path(artifact_dir: str) -> str`
  - `UNATTACHED_RUN_DIR: str` (value `"unattached"`)

- [ ] **Step 1: Write the failing test**

Create `biomentis/agent/tests/test_structure_artifacts.py`:

```python
"""Tests for the structure artifact layer.

Structures downloaded from RCSB, AlphaFold and friends had nowhere to live.
`query_pdb_identifiers` wrote them *inside the installed package*
(`biomentis/tool/data/pdb/`), and `query_alphafold` defaulted to the process
CWD — which is why loose `6nk5.pdb`-style files accumulate at the repo root.
Neither returned a usable path; it was buried in a nested dict.

This module pins down the replacement: a run-scoped artifact directory, a
crash-safe JSONL registry holding paths *relative* to that directory, and
molecule classification good enough to pick a default representation later.

Run with:
    python -m biomentis.agent.tests.test_structure_artifacts
"""

from __future__ import annotations

import os
import tempfile

from biomentis.structures import UNATTACHED_RUN_DIR, default_artifact_dir, registry_path

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


def test_artifact_dir_is_run_scoped() -> None:
    print("\n[1] the artifact dir is run-scoped and honours the env override")
    root = tempfile.mkdtemp(prefix="biomni_art_")
    prev = os.environ.get("BIOMENTIS_ARTIFACT_DIR")
    os.environ["BIOMENTIS_ARTIFACT_DIR"] = root
    try:
        _check(
            "run id becomes a subdirectory",
            default_artifact_dir("run-123") == os.path.join(root, "run-123"),
            default_artifact_dir("run-123"),
        )
        _check(
            "no run id falls back to the unattached dir",
            default_artifact_dir() == os.path.join(root, UNATTACHED_RUN_DIR),
            default_artifact_dir(),
        )
        _check(
            "registry sits inside the artifact dir",
            registry_path(root).endswith(os.path.join(root, "index.jsonl")),
            registry_path(root),
        )
    finally:
        if prev is None:
            os.environ.pop("BIOMENTIS_ARTIFACT_DIR", None)
        else:
            os.environ["BIOMENTIS_ARTIFACT_DIR"] = prev


def main() -> int:
    for fn in (test_artifact_dir_is_run_scoped,):
        fn()

    print(f"\n{len(_PASSED)} passed, {len(_FAILED)} failed")
    for name, detail in _FAILED:
        print(f"  FAIL {name}: {detail}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run test to verify it fails**

```bash
export PYTHONPATH=/Users/dylanbulseco/Documents/Projects/Biomni
/opt/anaconda3/envs/biomni/bin/python -m biomentis.agent.tests.test_structure_artifacts
```

Expected: `ModuleNotFoundError: No module named 'biomentis.structures'`

- [ ] **Step 3: Write minimal implementation**

Create `biomentis/structures.py`:

```python
"""Registry for structure files downloaded from external databases.

Structures fetched from RCSB, AlphaFold, PubChem and friends used to land in
one of two wrong places — inside the installed package, or in the process CWD —
and the caller got back a path buried in a nested dict, if at all. Nothing
tracked what a file *was*, so nothing downstream could show it.

Artifacts live beside the run journal (`runs/<run_id>.jsonl` and
`runs/<run_id>/`) so that the two share a lifetime. That matters because a
restored transcript carries only file *paths*, never bytes: an artifact swept
independently of its journal would come back as a broken viewer.

Registry paths are stored relative to the artifact root. Moving to a hosted,
per-user layout is then a change to `default_artifact_dir` alone, rather than a
migration of every stored record.
"""

from __future__ import annotations

import os

from biomentis.run_journal import default_run_dir

__all__ = [
    "UNATTACHED_RUN_DIR",
    "default_artifact_dir",
    "registry_path",
]

# Downloads that happen outside a run — the sidebar REPL, or a programmatic
# call with no background run — go here rather than failing. Artifacts in it
# carry run_id=None.
UNATTACHED_RUN_DIR = "unattached"


def default_artifact_dir(run_id: str | None = None) -> str:
    """Directory holding one run's downloaded artifacts.

    `BIOMENTIS_ARTIFACT_DIR` overrides the root; the run id is appended either
    way, so the override relocates the whole tree without flattening it.
    """
    root = os.getenv("BIOMENTIS_ARTIFACT_DIR") or default_run_dir()
    return os.path.join(root, run_id or UNATTACHED_RUN_DIR)


def registry_path(artifact_dir: str) -> str:
    """Path to the JSONL registry inside `artifact_dir`."""
    return os.path.join(artifact_dir, "index.jsonl")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
/opt/anaconda3/envs/biomni/bin/python -m biomentis.agent.tests.test_structure_artifacts
```

Expected: `3 passed, 0 failed`

- [ ] **Step 5: Add the test to the CLAUDE.md list**

In `CLAUDE.md`, immediately after the `test_anthropic_key` line (line 44), add:

```
$P -m biomentis.agent.tests.test_structure_artifacts  # structure artifact registry
```

- [ ] **Step 6: Commit**

```bash
git add biomentis/structures.py biomentis/agent/tests/test_structure_artifacts.py CLAUDE.md
git commit -m "Add run-scoped artifact directory resolution

Downloaded structures had nowhere to live. This gives them a directory
beside the run journal, so an artifact and the journal that references it
share a lifetime — a restored transcript carries only paths, so an
independently swept artifact would render broken."
```

---

### Task 2: The registry

**Files:**
- Modify: `biomentis/structures.py`
- Modify: `biomentis/agent/tests/test_structure_artifacts.py`

**Interfaces:**
- Consumes: `default_artifact_dir`, `registry_path` from Task 1.
- Produces:
  - `StructureArtifact` dataclass with fields `artifact_id, path, source_db, accession, fmt, molecule_class, n_atoms, chains, title, fetched_at, run_id`
  - `make_artifact_id(source_db: str, accession: str, fmt: str) -> str`
  - `append_artifact(artifact: StructureArtifact, artifact_dir: str) -> None`
  - `read_registry(artifact_dir: str) -> list[StructureArtifact]`
  - `resolve_artifact_path(artifact: StructureArtifact, artifact_dir: str) -> str`

- [ ] **Step 1: Write the failing test**

Add to `test_structure_artifacts.py` — new imports at the top, and these two functions before `main()`:

```python
from biomentis.structures import (
    StructureArtifact,
    append_artifact,
    make_artifact_id,
    read_registry,
    resolve_artifact_path,
)


def _artifact(**over) -> StructureArtifact:
    base = {
        "artifact_id": make_artifact_id("rcsb", "6NK5", "pdb"),
        "path": "6NK5.pdb",
        "source_db": "rcsb",
        "accession": "6NK5",
        "fmt": "pdb",
        "molecule_class": "protein",
        "n_atoms": 1234,
        "chains": ["A", "B"],
        "title": "Some structure",
        "fetched_at": "2026-09-03T10:00:00",
        "run_id": "run-123",
    }
    base.update(over)
    return StructureArtifact(**base)


def test_registry_round_trips_and_paths_are_relative() -> None:
    print("\n[2] the registry round-trips, and stored paths are relative")
    d = tempfile.mkdtemp(prefix="biomni_reg_")
    a = _artifact()
    append_artifact(a, d)
    append_artifact(_artifact(accession="7KLV", path="7KLV.pdb",
                              artifact_id=make_artifact_id("rcsb", "7KLV", "pdb")), d)

    back = read_registry(d)
    _check("both records read back", len(back) == 2, f"got {len(back)}")
    _check("round-trips identically", back[0] == a, f"got {back[0]}")
    _check("id is deterministic", make_artifact_id("RCSB", "6nk5", "PDB") == a.artifact_id, a.artifact_id)
    _check("stored path is relative", not os.path.isabs(back[0].path), back[0].path)

    # The hosted-mode guarantee: the same record resolves against a new root.
    moved = os.path.join(d, "moved")
    _check(
        "resolves against a different root",
        resolve_artifact_path(back[0], moved) == os.path.join(moved, "6NK5.pdb"),
        resolve_artifact_path(back[0], moved),
    )


def test_registry_survives_a_truncated_final_line() -> None:
    print("\n[3] a half-written final line costs one record, not the file")
    d = tempfile.mkdtemp(prefix="biomni_trunc_")
    append_artifact(_artifact(), d)
    with open(registry_path(d), "a", encoding="utf-8") as fh:
        fh.write('{"artifact_id": "rcsb:7klv:pd')  # killed mid-write

    back = read_registry(d)
    _check("complete record survives", len(back) == 1, f"got {len(back)}")
    _check("no exception raised", True)
```

Add both to the `main()` tuple:

```python
    for fn in (
        test_artifact_dir_is_run_scoped,
        test_registry_round_trips_and_paths_are_relative,
        test_registry_survives_a_truncated_final_line,
    ):
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/opt/anaconda3/envs/biomni/bin/python -m biomentis.agent.tests.test_structure_artifacts
```

Expected: `ImportError: cannot import name 'StructureArtifact' from 'biomentis.structures'`

- [ ] **Step 3: Write minimal implementation**

Add to `biomentis/structures.py` — extend the imports and `__all__`, then append:

```python
import json
from dataclasses import asdict, dataclass, field
```

```python
@dataclass
class StructureArtifact:
    """One structure file downloaded from an external database.

    `path` is relative to the artifact directory, never absolute — see the
    module docstring for why.
    """

    artifact_id: str
    path: str
    source_db: str
    accession: str
    fmt: str
    molecule_class: str = "unknown"
    n_atoms: int | None = None
    chains: list[str] = field(default_factory=list)
    title: str | None = None
    fetched_at: str = ""
    run_id: str | None = None


def make_artifact_id(source_db: str, accession: str, fmt: str) -> str:
    """A deterministic id, so re-fetching the same structure is idempotent.

    Case-insensitive because the same entry is written `6NK5` by RCSB and
    `6nk5` by half the tools that consume it.
    """
    return f"{source_db}:{accession}:{fmt}".lower()


def append_artifact(artifact: StructureArtifact, artifact_dir: str) -> None:
    """Append one record to the registry.

    One JSON object per line, opened/appended/closed per record so the close is
    the flush — the same idiom as `run_journal._write`, and for the same
    reason: a killed process still leaves a readable registry.
    """
    os.makedirs(artifact_dir, exist_ok=True)
    with open(registry_path(artifact_dir), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(artifact), default=str) + "\n")


def read_registry(artifact_dir: str) -> list[StructureArtifact]:
    """Every complete record in the registry.

    An unparseable line is skipped rather than raising: a registry truncated
    mid-write by a hard kill costs one record, not the file.
    """
    path = registry_path(artifact_dir)
    out: list[StructureArtifact] = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    out.append(StructureArtifact(**record))
                except TypeError:
                    # A record written by a newer version with extra fields.
                    known = {k: v for k, v in record.items() if k in StructureArtifact.__dataclass_fields__}
                    out.append(StructureArtifact(**known))
    except OSError:
        return []
    return out


def resolve_artifact_path(artifact: StructureArtifact, artifact_dir: str) -> str:
    """Absolute path for `artifact` under `artifact_dir`."""
    return os.path.join(artifact_dir, artifact.path)
```

Extend `__all__` to include `StructureArtifact`, `append_artifact`, `make_artifact_id`, `read_registry`, `resolve_artifact_path`.

- [ ] **Step 4: Run test to verify it passes**

```bash
/opt/anaconda3/envs/biomni/bin/python -m biomentis.agent.tests.test_structure_artifacts
```

Expected: `9 passed, 0 failed`

- [ ] **Step 5: Commit**

```bash
git add biomentis/structures.py biomentis/agent/tests/test_structure_artifacts.py
git commit -m "Add the structure artifact registry

One JSON object per line, written open/append/close so the close is the
flush — the run_journal idiom, chosen for the same reason: a killed
process leaves a readable file, and a truncated final line costs one
record rather than all of them.

Paths are stored relative to the artifact root so a hosted deployment
becomes one resolver change instead of a migration of every record."
```

---

### Task 3: Molecule classification

**Files:**
- Modify: `biomentis/structures.py`
- Modify: `biomentis/agent/tests/test_structure_artifacts.py`
- Modify: `pyproject.toml:16-30` (add `biopython` to `dependencies`)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `StructureStats` dataclass with fields `molecule_class: str`, `n_atoms: int | None`, `chains: list[str]`, `title: str | None`
  - `inspect_structure(path: str) -> StructureStats`
  - `format_for_path(path: str) -> str` — returns `"pdb"`, `"cif"`, `"sdf"`, `"mol2"`, or `"unknown"`

- [ ] **Step 1: Write the failing test**

Add to `test_structure_artifacts.py` — new imports plus these functions:

```python
import sys

from biomentis.structures import format_for_path, inspect_structure


def _atom(serial: int, name: str, resname: str, chain: str, resseq: int, record: str = "ATOM  ") -> str:
    """One PDB coordinate line, built by column so Bio.PDB will parse it."""
    return (
        f"{record}"
        f"{serial:>5}"
        f" {name:<4}"
        f" {resname:>3}"
        f" {chain}"
        f"{resseq:>4}"
        f"    "
        f"{11.104:>8.3f}{13.207:>8.3f}{10.000:>8.3f}"
        f"{1.00:>6.2f}{0.00:>6.2f}"
        f"{'':>10}"
        f"{name[0]:>2}"
    )


def _write_pdb(lines: list[str]) -> str:
    d = tempfile.mkdtemp(prefix="biomni_pdb_")
    path = os.path.join(d, "test.pdb")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\nEND\n")
    return path


_PROTEIN = [_atom(1, "N", "ALA", "A", 1), _atom(2, "CA", "GLY", "A", 2)]
_DNA = [_atom(1, "P", "DA", "B", 1), _atom(2, "P", "DT", "B", 2)]
_LIGAND = [_atom(1, "C1", "ATP", "C", 1, record="HETATM")]


def test_classifies_the_four_molecule_classes() -> None:
    print("\n[4] each molecule class is detected")
    _check("protein", inspect_structure(_write_pdb(_PROTEIN)).molecule_class == "protein",
           inspect_structure(_write_pdb(_PROTEIN)).molecule_class)
    _check("nucleic acid", inspect_structure(_write_pdb(_DNA)).molecule_class == "nucleic_acid",
           inspect_structure(_write_pdb(_DNA)).molecule_class)
    _check("complex", inspect_structure(_write_pdb(_PROTEIN + _DNA)).molecule_class == "complex",
           inspect_structure(_write_pdb(_PROTEIN + _DNA)).molecule_class)
    _check("small molecule", inspect_structure(_write_pdb(_LIGAND)).molecule_class == "small_molecule",
           inspect_structure(_write_pdb(_LIGAND)).molecule_class)

    stats = inspect_structure(_write_pdb(_PROTEIN + _DNA))
    _check("counts atoms", stats.n_atoms == 4, f"got {stats.n_atoms}")
    _check("collects chains", sorted(stats.chains) == ["A", "B"], f"got {stats.chains}")


def test_format_detection() -> None:
    print("\n[5] format is taken from the extension")
    _check("pdb", format_for_path("/x/6NK5.pdb") == "pdb")
    _check("cif", format_for_path("/x/6NK5.cif") == "cif")
    _check("sdf", format_for_path("/x/lig.sdf") == "sdf")
    _check("unknown", format_for_path("/x/notes.txt") == "unknown")


def test_classification_degrades_without_biopython() -> None:
    print("\n[6] a missing Biopython yields 'unknown' rather than raising")
    path = _write_pdb(_PROTEIN)
    prev = sys.modules.get("Bio.PDB", "absent")
    sys.modules["Bio.PDB"] = None  # makes `import Bio.PDB` raise ImportError
    try:
        stats = inspect_structure(path)
    finally:
        if prev == "absent":
            sys.modules.pop("Bio.PDB", None)
        else:
            sys.modules["Bio.PDB"] = prev

    _check("class is unknown", stats.molecule_class == "unknown", stats.molecule_class)
    _check("did not raise", True)
```

Add all three to the `main()` tuple.

- [ ] **Step 2: Run test to verify it fails**

```bash
/opt/anaconda3/envs/biomni/bin/python -m biomentis.agent.tests.test_structure_artifacts
```

Expected: `ImportError: cannot import name 'inspect_structure' from 'biomentis.structures'`

- [ ] **Step 3: Write minimal implementation**

Add to `biomentis/structures.py`:

```python
# Residue names that decide the molecule class. MSE (selenomethionine) is here
# because it is a modified residue that appears as HETATM in many crystal
# structures, and treating it as a ligand would misclassify the whole entry.
_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "SEC", "PYL", "MSE",
}
_NUCLEOTIDES = {"DA", "DC", "DG", "DT", "DI", "A", "C", "G", "U", "I"}

_EXTENSIONS = {
    ".pdb": "pdb",
    ".ent": "pdb",
    ".cif": "cif",
    ".mmcif": "cif",
    ".sdf": "sdf",
    ".mol": "sdf",
    ".mol2": "mol2",
}


@dataclass
class StructureStats:
    """What a structure file contains, to the depth a viewer needs."""

    molecule_class: str = "unknown"
    n_atoms: int | None = None
    chains: list[str] = field(default_factory=list)
    title: str | None = None


def format_for_path(path: str) -> str:
    """Structure format implied by the file extension."""
    return _EXTENSIONS.get(os.path.splitext(path)[1].lower(), "unknown")


def _classify(residues: set[str]) -> str:
    has_protein = bool(residues & _AMINO_ACIDS)
    has_nucleic = bool(residues & _NUCLEOTIDES)
    if has_protein and has_nucleic:
        return "complex"
    if has_protein:
        return "protein"
    if has_nucleic:
        return "nucleic_acid"
    if residues:
        return "small_molecule"
    return "unknown"


def inspect_structure(path: str) -> StructureStats:
    """Classify a structure file and count what is in it.

    Degrades rather than raising, at every step: an SDF/MOL2 is a small
    molecule by format alone; a parse failure falls back to a header sniff of
    the raw text; a missing Biopython yields `unknown`. 3Dmol renders correctly
    without a class, so this only ever improves the default representation — it
    never blocks visualization.
    """
    fmt = format_for_path(path)
    if fmt in ("sdf", "mol2"):
        return StructureStats(molecule_class="small_molecule")

    try:
        from Bio.PDB import MMCIFParser, PDBParser
    except Exception:
        return StructureStats()

    import warnings

    parser = MMCIFParser(QUIET=True) if fmt == "cif" else PDBParser(QUIET=True)
    try:
        with warnings.catch_warnings():
            # Real PDB entries routinely trip discontinuity warnings; they are
            # not our problem and they would otherwise flood the console.
            warnings.simplefilter("ignore")
            model = parser.get_structure("s", path)
    except Exception:
        return _sniff_header(path)

    residues: set[str] = set()
    chains: set[str] = set()
    n_atoms = 0
    for chain in model.get_chains():
        chains.add(chain.id.strip() or "?")
    for residue in model.get_residues():
        residues.add(residue.get_resname().strip().upper())
    for _ in model.get_atoms():
        n_atoms += 1

    return StructureStats(
        molecule_class=_classify(residues),
        n_atoms=n_atoms or None,
        chains=sorted(c for c in chains if c),
        title=None,
    )


def _sniff_header(path: str) -> StructureStats:
    """Last-resort classification from raw PDB text.

    Reads residue name and chain straight out of the fixed columns, which is
    enough to pick a representation when the real parser has given up.
    """
    residues: set[str] = set()
    chains: set[str] = set()
    n_atoms = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith(("ATOM  ", "HETATM")):
                    n_atoms += 1
                    residues.add(line[17:20].strip().upper())
                    chain = line[21:22].strip()
                    if chain:
                        chains.add(chain)
    except OSError:
        return StructureStats()

    return StructureStats(
        molecule_class=_classify(residues),
        n_atoms=n_atoms or None,
        chains=sorted(chains),
    )
```

Extend `__all__` with `StructureStats`, `format_for_path`, `inspect_structure`.

- [ ] **Step 4: Declare Biopython**

In `pyproject.toml`, add `"biopython",` to the `dependencies` list (after `"markdown",` on line 29). Many existing tools — `molecular_biology.py`, `systems_biology.py:630`, `biochemistry.py` — already import it; it is currently undeclared and works by luck.

- [ ] **Step 5: Run test to verify it passes**

```bash
/opt/anaconda3/envs/biomni/bin/python -m biomentis.agent.tests.test_structure_artifacts
```

Expected: `21 passed, 0 failed`

- [ ] **Step 6: Commit**

```bash
git add biomentis/structures.py biomentis/agent/tests/test_structure_artifacts.py pyproject.toml
git commit -m "Classify structures by residue composition

Drives the default representation later — cartoon for protein, sticks
for a ligand, ladder for DNA — so it belongs with the artifact rather
than with the viewer.

Degrades at every step: SDF/MOL2 is small-molecule by format, a parse
failure falls back to reading the fixed PDB columns directly, a missing
Biopython yields 'unknown'. 3Dmol renders fine without a class, so this
only improves a default and never blocks a view.

Also declares biopython, which several existing tools already import."
```

---

### Task 4: `record_download` and rewiring `query_alphafold`

**Files:**
- Modify: `biomentis/structures.py`
- Modify: `biomentis/tool/database.py:613-616` and `:640-658`
- Modify: `biomentis/agent/tests/test_structure_artifacts.py`

**Interfaces:**
- Consumes: `default_artifact_dir`, `append_artifact`, `make_artifact_id`, `inspect_structure`, `format_for_path`, `StructureArtifact`.
- Produces: `record_download(path: str, source_db: str, accession: str, run_id: str | None = None, artifact_dir: str | None = None) -> StructureArtifact` — registers a downloaded file and returns its record. This is the single call the database tools make.

- [ ] **Step 1: Write the failing test**

Add to `test_structure_artifacts.py`:

```python
import biomentis.tool.database as db
from biomentis.structures import record_download


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


def _fake_requests(handler):
    """A stand-in for `database.requests` that dispatches on URL."""
    return type("_R", (), {"get": staticmethod(handler), "exceptions": db.requests.exceptions})


def test_record_download_registers_an_artifact() -> None:
    print("\n[7] record_download registers a file and classifies it")
    d = tempfile.mkdtemp(prefix="biomni_rec_")
    path = os.path.join(d, "6NK5.pdb")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_PROTEIN) + "\nEND\n")

    art = record_download(path, source_db="rcsb", accession="6NK5", artifact_dir=d)
    _check("classified", art.molecule_class == "protein", art.molecule_class)
    _check("format detected", art.fmt == "pdb", art.fmt)
    _check("path is relative", art.path == "6NK5.pdb", art.path)
    _check("timestamped", bool(art.fetched_at), art.fetched_at)
    _check("in the registry", len(read_registry(d)) == 1, str(read_registry(d)))


def test_alphafold_writes_to_the_artifact_dir() -> None:
    print("\n[8] query_alphafold writes to the artifact dir, not the CWD")
    root = tempfile.mkdtemp(prefix="biomni_af_")
    pdb_bytes = ("\n".join(_PROTEIN) + "\nEND\n").encode()

    def handler(url, *a, **k):
        if "alphafold.ebi.ac.uk/files/" in url:
            return _FakeResponse(content=pdb_bytes)
        return _FakeResponse(payload=[{"entryId": "AF-P04637-F1"}])

    cwd_before = set(os.listdir("."))
    prev_req, prev_env = db.requests, os.environ.get("BIOMENTIS_ARTIFACT_DIR")
    db.requests = _fake_requests(handler)
    os.environ["BIOMENTIS_ARTIFACT_DIR"] = root
    try:
        out = db.query_alphafold("P04637", download=True)
    finally:
        db.requests = prev_req
        if prev_env is None:
            os.environ.pop("BIOMENTIS_ARTIFACT_DIR", None)
        else:
            os.environ["BIOMENTIS_ARTIFACT_DIR"] = prev_env

    art_dir = os.path.join(root, UNATTACHED_RUN_DIR)
    _check("file landed in the artifact dir",
           os.path.exists(os.path.join(art_dir, "AF-P04637-F1-model_v4.pdb")),
           str(os.listdir(art_dir) if os.path.isdir(art_dir) else "missing"))
    _check("nothing new in CWD", set(os.listdir(".")) == cwd_before,
           str(set(os.listdir(".")) - cwd_before))
    _check("old nested key retained", "file_path" in out.get("download", {}), str(out.get("download")))
    _check("new artifacts key present", isinstance(out.get("artifacts"), list) and len(out["artifacts"]) == 1,
           str(out.get("artifacts")))
    _check("registered", len(read_registry(art_dir)) == 1, str(read_registry(art_dir)))
```

Add both to `main()`.

- [ ] **Step 2: Run test to verify it fails**

```bash
/opt/anaconda3/envs/biomni/bin/python -m biomentis.agent.tests.test_structure_artifacts
```

Expected: `ImportError: cannot import name 'record_download' from 'biomentis.structures'`

- [ ] **Step 3: Add `record_download` to `biomentis/structures.py`**

Add `from datetime import datetime` to the imports, then:

```python
def record_download(
    path: str,
    source_db: str,
    accession: str,
    run_id: str | None = None,
    artifact_dir: str | None = None,
) -> StructureArtifact:
    """Register a freshly downloaded structure file and return its record.

    The single call the database tools make. `path` may be absolute; what gets
    stored is its location relative to `artifact_dir`, so the record survives
    the tree being relocated.
    """
    artifact_dir = artifact_dir or default_artifact_dir(run_id)
    stats = inspect_structure(path)
    fmt = format_for_path(path)

    try:
        relative = os.path.relpath(os.path.abspath(path), os.path.abspath(artifact_dir))
    except ValueError:
        # Different drives on Windows — fall back to the bare filename.
        relative = os.path.basename(path)

    artifact = StructureArtifact(
        artifact_id=make_artifact_id(source_db, accession, fmt),
        path=relative,
        source_db=source_db,
        accession=accession,
        fmt=fmt,
        molecule_class=stats.molecule_class,
        n_atoms=stats.n_atoms,
        chains=stats.chains,
        title=stats.title,
        fetched_at=datetime.now().isoformat(timespec="seconds"),
        run_id=run_id,
    )
    append_artifact(artifact, artifact_dir)
    return artifact
```

Add `record_download` to `__all__`.

- [ ] **Step 4: Rewire `query_alphafold`**

In `biomentis/tool/database.py`, replace lines 613-616:

```python
            # Ensure output directory exists
            if not output_dir:
                output_dir = "."
            os.makedirs(output_dir, exist_ok=True)
```

with:

```python
            # Default to the run-scoped artifact directory. This used to be
            # ".", which is why loose AF-*.pdb files accumulate wherever the
            # app happens to be started from.
            if not output_dir:
                output_dir = default_artifact_dir()
            os.makedirs(output_dir, exist_ok=True)
```

Then in the success branch, after `download_info = {...}` (currently line 632-636), add:

```python
                artifacts.append(
                    asdict(record_download(file_path, source_db="alphafold", accession=uniprot_id))
                )
                # Printed so the UI's file-detection seam can find it.
                print(f"Downloaded structure: {os.path.abspath(file_path)}")
```

Initialise `artifacts: list[dict] = []` alongside `download_info = None`, and after `response_data["download"] = download_info` add:

```python
        if artifacts:
            response_data["artifacts"] = artifacts
```

Add to the imports at the top of `database.py`:

```python
from dataclasses import asdict

from biomentis.structures import default_artifact_dir, record_download
```

- [ ] **Step 5: Run test to verify it passes**

```bash
/opt/anaconda3/envs/biomni/bin/python -m biomentis.agent.tests.test_structure_artifacts
```

Expected: `31 passed, 0 failed`

- [ ] **Step 6: Commit**

```bash
git add biomentis/structures.py biomentis/tool/database.py biomentis/agent/tests/test_structure_artifacts.py
git commit -m "Route query_alphafold downloads into the artifact directory

output_dir defaulted to '.', so every AlphaFold download landed wherever
the app was started from — the reason loose PDB files accumulate at the
repo root. It now defaults to the run-scoped artifact directory, gets
registered, and prints its path so the UI's existing file-detection seam
can find it later.

The nested download.file_path key is retained; the change is additive."
```

---

### Task 5: Rewiring `query_pdb_identifiers`

**Files:**
- Modify: `biomentis/tool/database.py:1006-1025`
- Modify: `biomentis/agent/tests/test_structure_artifacts.py`

**Interfaces:**
- Consumes: `record_download`, `default_artifact_dir` (already imported in Task 4).
- Produces: nothing new.

- [ ] **Step 1: Write the failing test**

Add to `test_structure_artifacts.py`:

```python
def test_pdb_identifiers_never_writes_into_the_package() -> None:
    print("\n[9] query_pdb_identifiers writes to the artifact dir, not the package")
    root = tempfile.mkdtemp(prefix="biomni_pdbid_")
    pdb_bytes = ("\n".join(_PROTEIN) + "\nEND\n").encode()
    package_dir = os.path.join(os.path.dirname(db.__file__), "data", "pdb")
    before = set(os.listdir(package_dir)) if os.path.isdir(package_dir) else set()

    def handler(url, *a, **k):
        if "files.rcsb.org/download/" in url:
            return _FakeResponse(content=pdb_bytes)
        return _FakeResponse(payload={"struct": {"title": "Test entry"}})

    prev_req, prev_env = db.requests, os.environ.get("BIOMENTIS_ARTIFACT_DIR")
    db.requests = _fake_requests(handler)
    os.environ["BIOMENTIS_ARTIFACT_DIR"] = root
    try:
        out = db.query_pdb_identifiers(["6NK5"], download=True)
    finally:
        db.requests = prev_req
        if prev_env is None:
            os.environ.pop("BIOMENTIS_ARTIFACT_DIR", None)
        else:
            os.environ["BIOMENTIS_ARTIFACT_DIR"] = prev_env

    art_dir = os.path.join(root, UNATTACHED_RUN_DIR)
    after = set(os.listdir(package_dir)) if os.path.isdir(package_dir) else set()
    _check("file landed in the artifact dir",
           os.path.exists(os.path.join(art_dir, "6NK5.pdb")),
           str(os.listdir(art_dir) if os.path.isdir(art_dir) else "missing"))
    _check("package directory untouched", after == before, str(after - before))
    _check("old nested key retained",
           any("pdb_file_path" in r for r in out.get("detailed_results", [])),
           str(out.get("detailed_results")))
    _check("new artifacts key present",
           isinstance(out.get("artifacts"), list) and len(out["artifacts"]) == 1,
           str(out.get("artifacts")))
```

Add it to `main()`.

- [ ] **Step 2: Run test to verify it fails**

```bash
/opt/anaconda3/envs/biomni/bin/python -m biomentis.agent.tests.test_structure_artifacts
```

Expected: FAIL on "file landed in the artifact dir" and "new artifacts key present" — the file goes into the package directory instead.

- [ ] **Step 3: Write the implementation**

In `biomentis/tool/database.py`, replace lines 1006-1008:

```python
                    if pdb_response.status_code == 200:
                        # Create data directory if it doesn't exist
                        data_dir = os.path.join(os.path.dirname(__file__), "data", "pdb")
                        os.makedirs(data_dir, exist_ok=True)
```

with:

```python
                    if pdb_response.status_code == 200:
                        # This used to write inside the installed package
                        # (biomentis/tool/data/pdb/), which is not a place a
                        # user's downloads belong and is not configurable.
                        data_dir = default_artifact_dir()
                        os.makedirs(data_dir, exist_ok=True)
```

Then after the `for result in detailed_results:` block that sets `pdb_file_path` (currently ending line 1019), add at the same indentation as that `for`:

```python
                        downloaded.append(
                            asdict(record_download(pdb_file_path, source_db="rcsb", accession=pdb_id))
                        )
                        # Printed so the UI's file-detection seam can find it.
                        print(f"Downloaded structure: {os.path.abspath(pdb_file_path)}")
```

Initialise `downloaded: list[dict] = []` immediately before `if download:` (line 1006 area, before the loop), and change the return at line 1025:

```python
        return {"detailed_results": detailed_results}
```

to:

```python
        payload = {"detailed_results": detailed_results}
        if downloaded:
            payload["artifacts"] = downloaded
        return payload
```

- [ ] **Step 4: Run test to verify it passes**

```bash
/opt/anaconda3/envs/biomni/bin/python -m biomentis.agent.tests.test_structure_artifacts
```

Expected: `35 passed, 0 failed`

- [ ] **Step 5: Run the full suite for regressions**

```bash
export PYTHONPATH=/Users/dylanbulseco/Documents/Projects/Biomni
P=/opt/anaconda3/envs/biomni/bin/python
for t in test_tab_focus test_solution_streaming test_run_completion test_tool_imports \
         test_background_run test_stop_sequences test_context_budget test_ncbi_email \
         test_anthropic_key test_structure_artifacts; do
  echo -n "$t: "; $P -m biomentis.agent.tests.$t 2>&1 | grep -E "passed" | tail -1
done
$P biomentis/agent/tutor/tests/test_smoke_e2e.py 2>&1 | grep -E "passed" | tail -1
$P biomentis/agent/tutor/tests/test_critic.py 2>&1 | grep -E "passed" | tail -1
```

Expected: every line reports `0 failed`.

- [ ] **Step 6: Commit**

```bash
git add biomentis/tool/database.py biomentis/agent/tests/test_structure_artifacts.py
git commit -m "Stop query_pdb_identifiers writing inside the installed package

Downloads went to biomentis/tool/data/pdb/ — inside the package itself,
not configurable, and already holding several MB of structures. They now
go to the run-scoped artifact directory, get registered, and print their
path for the UI's file-detection seam.

detailed_results[i].pdb_file_path is retained; the change is additive."
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: artifact directory → Task 1; registry → Task 2; classification and the Biopython declaration → Task 3; `record_download` and the `query_alphafold` CWD bug → Task 4; the `query_pdb_identifiers` package-directory bug → Task 5. All nine spec tests appear: 1–2 in Task 3, 3–5 in Tasks 1–2, 6–7 in Tasks 4–5, 8–9 in Tasks 4–5. The "out of scope" list is honoured — no task touches the UI, `_format_query_results`, RDKit, or `env_desc.py`.

**Placeholder scan.** No TBD/TODO, no "add error handling", no "similar to Task N". Every code step carries the actual code.

**Type consistency.** `StructureArtifact` field names are identical in Tasks 2 and 4. `inspect_structure` returns `StructureStats` in Task 3 and is consumed as `stats.molecule_class` / `.n_atoms` / `.chains` / `.title` in Task 4. `record_download`'s signature in Task 4's Interfaces matches its definition and both call sites. `default_artifact_dir` takes `run_id` in Tasks 1, 4 and 5 alike.

**One deviation from the spec, deliberate and flagged:** the spec's "top-level `artifact` key" is implemented as `artifacts`, a list, in both tools — `query_pdb_identifiers` downloads many structures per call, and one key shape beats two.
