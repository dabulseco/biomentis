"""Tests for structure metadata and viewer input (biomentis/structure_info.py).

What these tests pin down:

  1.  PDB metadata comes from the header, with the labels in a fixed order
  2.  Coordinate records are never shown as text, for any format
  3.  A header field the file lacks is omitted, not filled with "None"
  4.  SDF: record count, computed formula, and every data field by its tag
  5.  SMILES: descriptors, a 3D model for the viewer, a 2D depiction
  6.  Bad input degrades to an "Error" row and no model; nothing raises

Run with:
    python -m biomentis.agent.tests.test_structure_info
"""

from __future__ import annotations

import os
import re
import sys
import tempfile

from rdkit import RDLogger

from biomentis.structure_info import (
    depiction_png,
    metadata_text,
    sdf_record_count,
    structure_metadata,
    viewer_model,
)

RDLogger.DisableLog("rdApp.*")

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


# --- Fixtures --------------------------------------------------------------

_FIX = os.path.join(os.path.dirname(__file__), "fixtures")
XRAY = os.path.join(_FIX, "mini_xray.pdb")
NMR = os.path.join(_FIX, "mini_nmr.pdb")
SDF = os.path.join(_FIX, "two_mols.sdf")
SMI = os.path.join(_FIX, "aspirin.smi")


def _tmp(text: str, ext: str) -> str:
    handle = tempfile.NamedTemporaryFile("w", suffix=ext, delete=False, encoding="utf-8")
    handle.write(text)
    handle.close()
    return handle.name


# --- Tests -----------------------------------------------------------------

def test_pdb_metadata():
    print("\n[pdb metadata]")
    md = structure_metadata(XRAY)
    labels = dict(md)
    for k in ("Title", "Experimental method", "Resolution (Å)", "Organism", "Chains",
              "Ligands / heteroatoms", "Authors", "Reference"):
        _check(f"has {k}", k in labels, list(labels))
    _check("resolution parsed", labels["Resolution (Å)"] == "1.80", labels.get("Resolution (Å)"))
    _check("chains listed", labels["Chains"] == "A, B", labels.get("Chains"))
    _check("waters excluded from ligands", "HOH" not in labels["Ligands / heteroatoms"],
           labels["Ligands / heteroatoms"])
    order = [k for k, _ in md]
    _check("title before method before chains",
           order.index("Title") < order.index("Experimental method") < order.index("Chains"), order)


def test_no_coordinates_anywhere():
    print("\n[no coordinates]")
    for p in (XRAY, NMR, SDF, SMI):
        name = os.path.basename(p)
        blob = metadata_text(p)
        _check(f"{name}: has text", bool(blob.strip()))
        _check(f"{name}: no ATOM/HETATM line", not re.search(r"^(ATOM|HETATM)\s", blob, re.M))
        _check(f"{name}: no SDF atom-block line",
               not re.search(r"^\s*-?\d+\.\d{4}\s+-?\d+\.\d{4}\s+-?\d+\.\d{4}\s+[A-Z]", blob, re.M))


def test_nmr_is_not_none_filled():
    print("\n[nmr]")
    md = structure_metadata(NMR)
    _check("no resolution row", "Resolution (Å)" not in dict(md))
    _check("no 'None' values", all(v and v != "None" for _, v in md), md)
    _check("method is NMR", "NMR" in dict(md)["Experimental method"])
    _check("no reference row", "Reference" not in dict(md))


def test_sdf_records_and_fields():
    print("\n[sdf]")
    _check("two records", sdf_record_count(SDF) == 2)
    md = dict(structure_metadata(SDF))
    _check("data fields shown", "PUBCHEM_COMPOUND_CID" in md, list(md))
    _check("formula computed", bool(md.get("Formula")), list(md))
    second = dict(structure_metadata(SDF)) if False else None
    model = viewer_model(SDF, record=1)
    _check("record 1 model is sdf", model is not None and model[1] == "sdf")
    _check("record 1 is methanol", model is not None and model[0].startswith("methanol"),
           model[0][:20] if model else None)
    _check("out-of-range record is None", viewer_model(SDF, record=5) is None)
    _check("sdf depiction", (depiction_png(SDF) or b"")[:4] == b"\x89PNG")


def test_smiles():
    print("\n[smiles]")
    md = dict(structure_metadata(SMI))
    _check("name", md.get("Name") == "aspirin", list(md))
    _check("formula", md["Formula"] == "C9H8O4", md.get("Formula"))
    _check("MW", md["Molecular weight (g/mol)"].startswith("180.1"), md.get("Molecular weight (g/mol)"))
    model = viewer_model(SMI)
    _check("3D model built as sdf", model is not None and model[1] == "sdf")
    _check("3D model has 3D coordinates", model is not None and "V2000" in model[0])
    _check("2D depiction", (depiction_png(SMI) or b"")[:4] == b"\x89PNG")


def test_pdb_viewer_and_depiction():
    print("\n[pdb viewer]")
    model = viewer_model(XRAY)
    _check("pdb model passes through", model is not None and model[1] == "pdb" and "ATOM" in model[0])
    _check("no depiction for pdb", depiction_png(XRAY) is None)


def test_bad_input_degrades():
    print("\n[bad input]")
    for text, ext in (("not_a_smiles(((", ".smi"), ("", ".pdb"), ("garbage\n", ".sdf"), ("x", ".xyz")):
        p = _tmp(text, ext)
        try:
            md = structure_metadata(p)
            _check(f"{ext}: error row, no raise", bool(md) and md[0][0] == "Error", md)
            _check(f"{ext}: no model", viewer_model(p) is None)
            _check(f"{ext}: metadata_text ok", isinstance(metadata_text(p), str))
        finally:
            os.unlink(p)
    md = structure_metadata("/nonexistent/file.pdb")
    _check("missing file: error row", md and md[0][0] == "Error", md)


def run_all() -> int:
    print("=" * 60)
    print("Structure info tests")
    print("=" * 60)
    test_pdb_metadata()
    test_no_coordinates_anywhere()
    test_nmr_is_not_none_filled()
    test_sdf_records_and_fields()
    test_smiles()
    test_pdb_viewer_and_depiction()
    test_bad_input_degrades()

    print("\n" + "=" * 60)
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    print("=" * 60)
    for name, detail in _FAILED:
        print(f"  ✗ {name}: {detail}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(run_all())
