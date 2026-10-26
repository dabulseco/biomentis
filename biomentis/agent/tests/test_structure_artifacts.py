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
import sys
import tempfile

import biomentis.tool.database as db
from biomentis.structures import (
    UNATTACHED_RUN_DIR,
    StructureArtifact,
    append_artifact,
    default_artifact_dir,
    format_for_path,
    import_file,
    inspect_structure,
    make_artifact_id,
    read_registry,
    record_download,
    registry_path,
    resolve_artifact_path,
)

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


def _write_text(name: str, text: str) -> str:
    d = tempfile.mkdtemp(prefix="biomni_txt_")
    path = os.path.join(d, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


_PROTEIN = [_atom(1, "N", "ALA", "A", 1), _atom(2, "CA", "GLY", "A", 2)]
_DNA = [_atom(1, "P", "DA", "B", 1), _atom(2, "P", "DT", "B", 2)]
_ETHANOL_SDF = (
    "ethanol\n  test\n\n  3  2  0  0  0  0  0  0  0  0999 V2000\n"
    "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "    1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "    2.0000    1.2000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "  1  2  1  0\n  2  3  1  0\nM  END\n$$$$\n"
)


def test_classifies_the_four_molecule_classes() -> None:
    print("\n[4] each molecule class is detected")
    _check("protein", inspect_structure(_write_pdb(_PROTEIN)).molecule_class == "protein",
           inspect_structure(_write_pdb(_PROTEIN)).molecule_class)
    _check("nucleic acid", inspect_structure(_write_pdb(_DNA)).molecule_class == "nucleic_acid",
           inspect_structure(_write_pdb(_DNA)).molecule_class)
    _check("complex", inspect_structure(_write_pdb(_PROTEIN + _DNA)).molecule_class == "complex",
           inspect_structure(_write_pdb(_PROTEIN + _DNA)).molecule_class)
    sdf = inspect_structure(_write_text("ethanol.sdf", _ETHANOL_SDF))
    _check("small molecule", sdf.molecule_class == "small_molecule", sdf.molecule_class)

    stats = inspect_structure(_write_pdb(_PROTEIN + _DNA))
    _check("counts atoms", stats.n_atoms == 4, f"got {stats.n_atoms}")
    _check("collects chains", sorted(stats.chains) == ["A", "B"], f"got {stats.chains}")


def test_smiles_files_are_small_molecules() -> None:
    print("\n[4b] a .smi file is a small molecule with an optional name as title")
    named = inspect_structure(_write_text("a.smi", "CC(=O)Oc1ccccc1C(=O)O aspirin\n"))
    _check("class", named.molecule_class == "small_molecule", named.molecule_class)
    _check("no atom count", named.n_atoms is None, str(named.n_atoms))
    _check("no chains", named.chains == [], str(named.chains))
    _check("title is the name after the SMILES", named.title == "aspirin", str(named.title))
    bare = inspect_structure(_write_text("b.smi", "CCO\n"))
    _check("unnamed SMILES has no title", bare.title is None and bare.molecule_class == "small_molecule",
           str(bare))


def test_format_detection() -> None:
    print("\n[5] format is taken from the extension")
    _check("pdb", format_for_path("/x/6NK5.pdb") == "pdb")
    _check("sdf", format_for_path("/x/lig.sdf") == "sdf")
    _check("smi", format_for_path("/x/lig.smi") == "smi")
    _check("cif is not supported", format_for_path("/x/6NK5.cif") == "unknown")
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


def test_import_file_copies_into_artifact_dir() -> None:
    print("\n[10] import_file copies outside files into the artifact dir")
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


def main() -> int:
    for fn in (
        test_artifact_dir_is_run_scoped,
        test_registry_round_trips_and_paths_are_relative,
        test_registry_survives_a_truncated_final_line,
        test_classifies_the_four_molecule_classes,
        test_smiles_files_are_small_molecules,
        test_format_detection,
        test_classification_degrades_without_biopython,
        test_record_download_registers_an_artifact,
        test_alphafold_writes_to_the_artifact_dir,
        test_pdb_identifiers_never_writes_into_the_package,
        test_import_file_copies_into_artifact_dir,
    ):
        fn()

    print(f"\n{len(_PASSED)} passed, {len(_FAILED)} failed")
    for name, detail in _FAILED:
        print(f"  FAIL {name}: {detail}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
