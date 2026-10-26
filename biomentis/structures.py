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

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime

from biomentis.run_journal import default_run_dir

__all__ = [
    "UNATTACHED_RUN_DIR",
    "STRUCTURE_MARKER",
    "StructureArtifact",
    "StructureStats",
    "append_artifact",
    "current_run_id",
    "default_artifact_dir",
    "format_for_path",
    "import_file",
    "inspect_structure",
    "make_artifact_id",
    "read_registry",
    "record_download",
    "registry_path",
    "resolve_artifact_path",
    "set_current_run_id",
]

# Downloads that happen outside a run — the sidebar REPL, or a programmatic
# call with no background run — go here rather than failing. Artifacts in it
# carry run_id=None.
UNATTACHED_RUN_DIR = "unattached"

# Printed by `show_structure` before a structure's absolute path. The UI turns a
# line starting with it, in a step's observation, into a structure viewer.
STRUCTURE_MARKER = "🧬 Structure ready: "


def default_artifact_dir(run_id: str | None = None) -> str:
    """Directory holding one run's downloaded artifacts.

    `BIOMENTIS_ARTIFACT_DIR` overrides the root; the run id is appended either
    way, so the override relocates the whole tree without flattening it.
    """
    root = os.getenv("BIOMENTIS_ARTIFACT_DIR") or default_run_dir()
    return os.path.join(root, run_id or UNATTACHED_RUN_DIR)


# The run whose code is executing now. The execute node sets it before each
# execution; code in the REPL runs on a worker thread that cannot see the agent,
# so tools that save files (the structure downloads) read it from here. A plain
# module global, not thread-local, precisely so that other thread sees it.
_CURRENT_RUN_ID: str | None = None


def set_current_run_id(run_id: str | None) -> None:
    """Record the run whose code is about to execute (None outside a run)."""
    global _CURRENT_RUN_ID
    _CURRENT_RUN_ID = run_id


def current_run_id() -> str | None:
    """The run id set by the execute node, or None."""
    return _CURRENT_RUN_ID


def registry_path(artifact_dir: str) -> str:
    """Path to the JSONL registry inside `artifact_dir`."""
    return os.path.join(artifact_dir, "index.jsonl")


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
    """Every complete record in the registry, one per artifact id.

    The log is append-only (crash-safe), so a re-fetch appends again; the last
    record for an id wins, which is what makes `make_artifact_id` idempotent.

    An unparseable line is skipped rather than raising: a registry truncated
    mid-write by a hard kill costs one record, not the file.
    """
    path = registry_path(artifact_dir)
    out: dict[str, StructureArtifact] = {}
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
                    art = StructureArtifact(**record)
                except TypeError:
                    # A record written by a newer version with extra fields.
                    known = {k: v for k, v in record.items() if k in StructureArtifact.__dataclass_fields__}
                    art = StructureArtifact(**known)
                out[art.artifact_id] = art
    except OSError:
        return []
    return list(out.values())


def resolve_artifact_path(artifact: StructureArtifact, artifact_dir: str) -> str:
    """Absolute path for `artifact` under `artifact_dir`."""
    return os.path.join(artifact_dir, artifact.path)


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
    ".sdf": "sdf",
    ".mol": "sdf",
    ".smi": "smi",
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


def _smiles_title(path: str) -> str | None:
    """The name after the SMILES string on the first line, if there is one."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            first = handle.readline().strip()
    except OSError:
        return None
    parts = first.split(None, 1)
    return parts[1].strip() if len(parts) == 2 else None


def inspect_structure(path: str) -> StructureStats:
    """Classify a structure file and count what is in it.

    Degrades rather than raising, at every step: an SDF or SMILES file is a
    small molecule by format alone; a parse failure falls back to a header sniff
    of the raw text; a missing Biopython yields `unknown`. 3Dmol renders
    correctly without a class, so this only ever improves the default
    representation — it never blocks visualization.
    """
    fmt = format_for_path(path)
    if fmt == "smi":
        return StructureStats(molecule_class="small_molecule", title=_smiles_title(path))
    if fmt == "sdf":
        return StructureStats(molecule_class="small_molecule")

    try:
        from Bio.PDB import PDBParser
    except Exception:
        return StructureStats()

    import warnings

    parser = PDBParser(QUIET=True)
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


def _same_bytes(a: str, b: str) -> bool:
    with open(a, "rb") as fa, open(b, "rb") as fb:
        return fa.read() == fb.read()


def import_file(
    src_path: str,
    source_db: str,
    run_id: str | None = None,
    artifact_dir: str | None = None,
) -> StructureArtifact:
    """Bring a structure file from outside the artifact dir into it, and register it.

    Used for user uploads and knowledge-base files (`source_db` "upload" /
    "kb"). The basename is kept; an existing identical file is reused, and a
    different file with the same name gets a `-2`, `-3`… suffix rather than
    overwriting. The accession is the basename without its extension.
    """
    artifact_dir = artifact_dir or default_artifact_dir(run_id)
    os.makedirs(artifact_dir, exist_ok=True)
    base, ext = os.path.splitext(os.path.basename(src_path))

    dest = os.path.join(artifact_dir, base + ext)
    n = 1
    while True:
        if not os.path.exists(dest):
            shutil.copyfile(src_path, dest)
            break
        if os.path.abspath(dest) == os.path.abspath(src_path) or _same_bytes(src_path, dest):
            break
        n += 1
        dest = os.path.join(artifact_dir, f"{base}-{n}{ext}")

    return record_download(dest, source_db=source_db, accession=base, run_id=run_id, artifact_dir=artifact_dir)
